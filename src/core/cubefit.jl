#=
This file contains the CubeFitter struct and its related functions.  The CubeFitter is how
fitting is performed with Loki.
=#

############################## PARAMETER / MODEL STRUCTURES ####################################

abstract type ParamMaps end

"""
    MIRParamMaps

A structure for holding 2D maps of fitting parameters generated after fitting a cube.  Each parameter
that is fit (i.e. stellar continuum temperature, optical depth, line ampltidue, etc.) corresponds to one 
2D map with the value of that parameter, with each spaxel's fit value located at the corresponding location
in the parameter map.

# Fields
- `stellar_continuum::Dict{Symbol, Array{T, 2}}`: The stellar continuum parameters: amplitude and temperature
- `dust_continuum::Dict{Int, Dict{Symbol, Array{Float64, 2}}}`: The dust continuum parameters: amplitude and temperature for each
    dust continuum component
- `power_law::Dict{Int, Dict{Symbol, Array{T, 2}}}`: Power law continuum parameters: amplitude and index for each component.
- `dust_features::Dict{String, Dict{Symbol, Array{Float64, 2}}}`: The dust feature parameters: amplitude, central wavelength, and FWHM
    for each PAH feature
- `extinction::Dict{Symbol, Array{T, 2}}`: Extinction parameters: optical depth at 9.7 μm and mixing ratio
- `hot_dust::Dict{Symbol, Array{T, 2}}`: Hot dust parameters: amplitude, temperature, covering fraction, warm tau, and cold tau
- `templates::Dict{String, Dict{Symbol, Array{T, 2}}}`: Template parameters: amplitude for each template
- `lines::Dict{Symbol, Dict{Symbol, Array{T, 2}}}`: The emission line parameters: amplitude, voff, FWHM, and any additional 
    line profile parameters for each line
- `statistics::Array{T, 2}`: The reduced chi^2 value and degrees of freedom for each fit.

See ['parammaps_empty`](@ref) for a default constructor function.
"""
struct MIRParamMaps{T<:Real} <: ParamMaps

    stellar_continuum::Dict{Symbol, Array{T, 2}}
    dust_continuum::Dict{Int, Dict{Symbol, Array{T, 2}}}
    power_law::Dict{Int, Dict{Symbol, Array{T, 2}}}
    dust_features::Dict{String, Dict{Symbol, Array{T, 2}}}
    abs_features::Dict{String, Dict{Symbol, Array{T, 2}}}
    extinction::Dict{Symbol, Array{T, 2}}
    hot_dust::Dict{Symbol, Array{T, 2}}
    templates::Dict{String, Dict{Symbol, Array{T, 2}}}
    lines::Dict{Symbol, Dict{Symbol, Array{T, 2}}}
    statistics::Dict{Symbol, Array{T, 2}}

end


"""
    OpticalParamMaps(stellar_populations, stellar_kinematics, lines, statistics)

A structure for holding 2D maps of fitting parameters generated after fitting a cube.  Each parameter
that is fit (i.e. stellar age, kinematics, Fe II emission, line amplitudes, etc.) corresponds to one 
2D map with the value of that parameter, with each spaxel's fit value located at the corresponding location
in the parameter map.

# Fields
- `stellar_populations::Dict{Int, Dict{Symbol, Array{T, 2}}}`: The stellar population parameters: mass, age, and metallicity.
- `stellar_kinematics::Dict{Symbol, Array{T, 2}}`: The stellar line-of-sight velocity and velocity dispersion.
- `feii::Dict{Symbol, Array{T, 2}}`: Narrow and/or broad Fe II emission parameters, including amplitude, velocity, and FWHM.
- `power_law::Dict{Int, Dict{Symbol, Array{T, 2}}}`: Power law parameters, including amplitude and index.
- `attenuation::Dict{Symbol, Array{T, 2}}`: Attenuation parameters, including E(B-V), E(B-V) factor, and optionally UV bump slope
and dust covering fraction.
- `lines::Dict{Symbol, Dict{Symbol, Array{T, 2}}}`: Emission line parameters, including amplitude, velocity, and FWHM, and any 
additional line profile parameters for non-Gaussian profiles.
- `statistics::Dict{Symbol, Array{T, 2}}`: Reduced chi^2 and degrees of freedom.

See ['parammaps_empty`](@ref) for a default constructor function.
"""
struct OpticalParamMaps{T<:Real} <: ParamMaps

    stellar_populations::Dict{Int, Dict{Symbol, Array{T, 2}}}
    stellar_kinematics::Dict{Symbol, Array{T, 2}}
    feii::Dict{Symbol, Array{T, 2}}
    power_law::Dict{Int, Dict{Symbol, Array{T, 2}}}
    attenuation::Dict{Symbol, Array{T, 2}}
    lines::Dict{Symbol, Dict{Symbol, Array{T, 2}}}
    statistics::Dict{Symbol, Array{T, 2}}

end


"""
    parammaps_empty(shape, n_dust_cont, n_power_law, cf_dustfeat, ab_names, n_lines, n_comps, cf_lines,
        flexible_wavesol)

A constructor function for making a default empty MIRParamMaps structure with all necessary fields for a given
fit of a DataCube.

# Arguments {S<:Integer}
- `shape::Tuple{S,S,S}`: Tuple specifying the 3D shape of the input data cube.
- `n_dust_cont::S`: The number of dust continuum components in the fit.
- `n_power_law::S`: The number of power law continuum components in the fit.
- `cf_dustfeat::DustFeatures`: A DustFeatures object specifying all of the PAH emission in the fit.
- `ab_names::Vector{String}`: The names of each absorption feature included in the fit.
- `temp_names::Vector{String}`: The names of generic templates in the fit.
- `n_lines::S`: The number of emission lines in the fit.
- `n_comps::S`: The maximum number of profiles that are being fit to a line.
- `cf_lines::TransitionLines`: A TransitionLines object specifying all of the line emission in the fit.
- `flexible_wavesol::Bool`: See the CubeFitter's `flexible_wavesol` parameter.
"""
function parammaps_empty(shape::Tuple{S,S,S}, n_dust_cont::S, n_power_law::S, cf_dustfeat::DustFeatures,
    ab_names::Vector{String}, temp_names::Vector{String}, n_lines::S, n_comps::S, cf_lines::TransitionLines, 
    flexible_wavesol::Bool)::MIRParamMaps where {S<:Integer}

    @debug """\n
    Creating MIRParamMaps struct with shape $shape
    ##############################################
    """

    # Initialize a default array of nans to be used as a placeholder for all the other arrays
    # until the actual fitting parameters are obtained
    nan_arr = ones(shape[1:2]...) .* NaN

    # Add stellar continuum fitting parameters
    stellar_continuum = Dict{Symbol, Array{Float64, 2}}()
    stellar_continuum[:amp] = copy(nan_arr)
    stellar_continuum[:temp] = copy(nan_arr)
    @debug "stellar continuum maps with keys $(keys(stellar_continuum))"

    # Add dust continuum fitting parameters
    dust_continuum = Dict{Int, Dict{Symbol, Array{Float64, 2}}}()
    for i ∈ 1:n_dust_cont
        dust_continuum[i] = Dict{Symbol, Array{Float64, 2}}()
        dust_continuum[i][:amp] = copy(nan_arr)
        dust_continuum[i][:temp] = copy(nan_arr)
        @debug "dust continuum $i maps with keys $(keys(dust_continuum[i]))"
    end

    # Add power law fitting parameters
    power_law = Dict{Int, Dict{Symbol, Array{Float64, 2}}}()
    for p ∈ 1:n_power_law
        power_law[p] = Dict{Symbol, Array{Float64, 2}}()
        power_law[p][:amp] = copy(nan_arr)
        power_law[p][:index] = copy(nan_arr)
        @debug "power law $p maps with keys $(keys(power_law[p]))"
    end

    # Add dust features fitting parameters
    dust_features = Dict{String, Dict{Symbol, Array{Float64, 2}}}()
    for (i, n) ∈ enumerate(cf_dustfeat.names)
        dust_features[n] = Dict{Symbol, Array{Float64, 2}}()
        dust_features[n][:amp] = copy(nan_arr)
        dust_features[n][:mean] = copy(nan_arr)
        dust_features[n][:fwhm] = copy(nan_arr)
        if cf_dustfeat.profiles[i] == :PearsonIV
            dust_features[n][:index] = copy(nan_arr)
            dust_features[n][:cutoff] = copy(nan_arr)
        end
        dust_features[n][:flux] = copy(nan_arr)
        dust_features[n][:eqw] = copy(nan_arr)
        dust_features[n][:SNR] = copy(nan_arr)
        @debug "dust feature $n maps with keys $(keys(dust_features[n]))"
    end

    abs_features = Dict{String, Dict{Symbol, Array{Float64, 2}}}()
    for n ∈ ab_names
        abs_features[n] = Dict{Symbol, Array{Float64, 2}}()
        abs_features[n][:tau] = copy(nan_arr)
        abs_features[n][:mean] = copy(nan_arr)
        abs_features[n][:fwhm] = copy(nan_arr)
        @debug "absorption feature $n maps with keys $(keys(abs_features[n]))"
    end

    # Add extinction fitting parameters
    extinction = Dict{Symbol, Array{Float64, 2}}()
    extinction[:tau_9_7] = copy(nan_arr)
    extinction[:tau_ice] = copy(nan_arr)
    extinction[:tau_ch] = copy(nan_arr)
    extinction[:beta] = copy(nan_arr)
    @debug "extinction maps with keys $(keys(extinction))"

    # Add hot dust fitting parameters
    hot_dust = Dict{Symbol, Array{Float64, 2}}()
    hot_dust[:amp] = copy(nan_arr)
    hot_dust[:temp] = copy(nan_arr)
    hot_dust[:frac] = copy(nan_arr)
    hot_dust[:tau_warm] = copy(nan_arr)
    hot_dust[:tau_cold] = copy(nan_arr)
    hot_dust[:sil_peak] = copy(nan_arr)
    @debug "hot dust maps with keys $(keys(hot_dust))"

    # Add template fitting parameters
    templates = Dict{String, Dict{Symbol, Array{Float64, 2}}}()
    for n ∈ temp_names
        templates[n] = Dict{Symbol, Array{Float64, 2}}()
        templates[n][:amp] = copy(nan_arr)
        @debug "template $n maps with keys $(keys(templates))"
    end

    # Nested dictionary -> first layer keys are line names, second layer keys are parameter names, which contain 2D arrays
    lines = Dict{Symbol, Dict{Symbol, Array{Float64, 2}}}()
    for i ∈ 1:n_lines
        for j ∈ 1:n_comps
            if !isnothing(cf_lines.profiles[i, j])

                line = Symbol(cf_lines.names[i], "_$(j)")
                lines[line] = Dict{Symbol, Array{Float64, 2}}()

                pnames = [:amp, :voff, :fwhm]
                # Need extra voff parameter if using the flexible_wavesol keyword
                if !isnothing(cf_lines.tied_voff[i, j]) && flexible_wavesol && isone(j)
                    pnames = [:amp, :voff, :voff_indiv, :fwhm]
                end
                # Add 3rd and 4th order moments (skewness and kurtosis) for Gauss-Hermite profiles
                if cf_lines.profiles[i, j] == :GaussHermite
                    pnames = [pnames; :h3; :h4]
                # Add mixing parameter for Voigt profiles, but only if NOT tying it
                elseif cf_lines.profiles[i, j] == :Voigt
                    pnames = [pnames; :mixing]
                end
                # Append parameters for flux, equivalent width, and signal-to-noise ratio, which are NOT fitting parameters, but are of interest
                pnames = [pnames; :flux; :eqw; :SNR]
                for pname ∈ pnames
                    lines[line][pname] = copy(nan_arr)
                end
                @debug "line $line maps with keys $pnames"
            end
        end
    end

    statistics = Dict{Symbol, Array{Float64, 2}}()
    # chi^2 statistics of the fits
    statistics[:chi2] = copy(nan_arr)
    @debug "chi^2 map"
    statistics[:dof] = copy(nan_arr)
    @debug "dof map"

    MIRParamMaps{Float64}(stellar_continuum, dust_continuum, power_law, dust_features, abs_features, 
        extinction, hot_dust, templates, lines, statistics)
end


"""
    parammaps_empty(shape, n_ssps, n_power_law, n_lines, n_comps, cf_lines, flexible_wavesol)

A constructor function for making a default empty OpticalParamMaps structure with all necessary fields for a given
fit of a DataCube.

# Arguments {S<:Integer}
- `shape::Tuple{S,S,S}`: Tuple specifying the 3D shape of the input data cube.
- `n_ssps::S`: The number of simple stellar populations in the fit.
- `n_power_law::S`: The number of power law continuum components in the fit.
- `n_lines::S`: The number of emission lines in the fit.
- `n_comps::S`: The maximum number of profiles that are being fit to a line.
- `cf_lines::TransitionLines`: A TransitionLines object specifying all of the line emission in the fit.
- `flexible_wavesol::Bool`: See the CubeFitter's `flexible_wavesol` parameter.
"""
function parammaps_empty(shape::Tuple{S,S,S}, n_ssps::S, n_power_law::S, n_lines::S, n_comps::S, 
    cf_lines::TransitionLines, flexible_wavesol::Bool)::OpticalParamMaps where {S<:Integer}

    @debug """\n
    Creating OpticalParamMaps struct with shape $shape
    ##################################################
    """

    # Initialize a default array of nans to be used as a placeholder for all the other arrays
    # until the actual fitting parameters are obtained
    nan_arr = ones(shape[1:2]...) .* NaN

    # Add stellar population fitting parameters
    stellar_populations = Dict{Int, Dict{Symbol, Array{Float64, 2}}}()
    for i ∈ 1:n_ssps
        stellar_populations[i] = Dict{Symbol, Array{Float64, 2}}()
        stellar_populations[i][:mass] = copy(nan_arr)
        stellar_populations[i][:age] = copy(nan_arr)
        stellar_populations[i][:metallicity] = copy(nan_arr)
        @debug "stellar population $i maps with keys $(keys(stellar_populations[i]))"
    end

    # Add stellar kinematics
    stellar_kinematics = Dict{Symbol, Array{Float64, 2}}()
    stellar_kinematics[:vel] = copy(nan_arr)
    stellar_kinematics[:vdisp] = copy(nan_arr)
    @debug "stellar kinematics maps with keys $(keys(stellar_kinematics))"

    # Add Fe II kinematics
    feii = Dict{Symbol, Array{Float64, 2}}()
    feii[:na_amp] = copy(nan_arr)
    feii[:na_vel] = copy(nan_arr)
    feii[:na_vdisp] = copy(nan_arr)
    feii[:br_amp] = copy(nan_arr)
    feii[:br_vel] = copy(nan_arr)
    feii[:br_vdisp] = copy(nan_arr)
    @debug "Fe II maps with keys $(keys(feii))"

    # Add power laws
    power_law = Dict{Int, Dict{Symbol, Array{Float64, 2}}}()
    for i ∈ 1:n_power_law
        power_law[i] = Dict{Symbol, Array{Float64, 2}}()
        power_law[i][:amp] = copy(nan_arr)
        power_law[i][:index] = copy(nan_arr)
    end

    # Add attenuation parameters
    attenuation = Dict{Symbol, Array{Float64, 2}}()
    attenuation[:E_BV] = copy(nan_arr)
    attenuation[:E_BV_factor] = copy(nan_arr)
    attenuation[:delta_UV] = copy(nan_arr)
    attenuation[:frac] = copy(nan_arr)
    @debug "attenuation maps with keys $(keys(attenuation))"

    # Nested dictionary -> first layer keys are line names, second layer keys are parameter names, which contain 2D arrays
    lines = Dict{Symbol, Dict{Symbol, Array{Float64, 2}}}()
    for i ∈ 1:n_lines
        for j ∈ 1:n_comps
            if !isnothing(cf_lines.profiles[i, j])

                line = Symbol(cf_lines.names[i], "_$(j)")
                lines[line] = Dict{Symbol, Array{Float64, 2}}()

                pnames = [:amp, :voff, :fwhm]
                # Need extra voff parameter if using the flexible_wavesol keyword
                if !isnothing(cf_lines.tied_voff[i, j]) && flexible_wavesol && isone(j)
                    pnames = [:amp, :voff, :voff_indiv, :fwhm]
                end
                # Add 3rd and 4th order moments (skewness and kurtosis) for Gauss-Hermite profiles
                if cf_lines.profiles[i, j] == :GaussHermite
                    pnames = [pnames; :h3; :h4]
                # Add mixing parameter for Voigt profiles, but only if NOT tying it
                elseif cf_lines.profiles[i, j] == :Voigt
                    pnames = [pnames; :mixing]
                end
                # Append parameters for flux, equivalent width, and signal-to-noise ratio, which are NOT fitting parameters, but are of interest
                pnames = [pnames; :flux; :eqw; :SNR]
                for pname ∈ pnames
                    lines[line][pname] = copy(nan_arr)
                end
                @debug "line $line maps with keys $pnames"
            end
        end
    end

    statistics = Dict{Symbol, Array{Float64, 2}}()
    # chi^2 statistics of the fits
    statistics[:chi2] = copy(nan_arr)
    @debug "chi^2 map"
    statistics[:dof] = copy(nan_arr)
    @debug "dof map"

    OpticalParamMaps{Float64}(stellar_populations, stellar_kinematics, feii, power_law, attenuation, lines, statistics)
end


abstract type CubeModel end

"""
    MIRCubeModel(model, stellar, dust_continuum, dust_features, extinction, hot_dust, lines)

A structure for holding 3D models of intensity, split up into model components, generated when fitting a cube.
This will be the same shape as the input data, and preferably the same datatype too (i.e., JWST files have flux
and error in Float32 format, so we should also output in Float32 format).  This is useful as a way to quickly
compare the full model, or model components, to the data.

# Fields {T<:Real}
- `model::Array{T, 3}`: The full 3D model.
- `stellar::Array{T, 3}`: The stellar component of the continuum.
- `dust_continuum::Array{T, 4}`: The dust components of the continuum. The 4th axis runs over each individual dust component.
- `power_law::Array{T, 4}`: The power law components of the continuum. The 4th axis runs over each individual power law.
- `dust_features::Array{T, 4}`: The dust (PAH) feature profiles. The 4th axis runs over each individual dust profile.
- `abs_features::Array{T, 4}`: The absorption feature profiles. The 4th axis runs over each individual absorption profile.
- `extinction::Array{T, 3}`: The extinction profile.
- `abs_ice::Array{T, 3}`: The water-ice absorption feature profile.
- `abs_ch::Array{T, 3}`: The CH absorption feature profile.
- `hot_dust::Array{T, 3}`: The hot dust emission profile.
- `templates::Array{T, 4}`: The generic template profiles.
- `lines::Array{T, 4}`: The line profiles. The 4th axis runs over each individual line.

See [`cubemodel_empty`](@ref) for a default constructor method.
"""
struct MIRCubeModel{T<:Real} <: CubeModel

    model::Array{T, 3}
    stellar::Array{T, 3}
    dust_continuum::Array{T, 4}
    power_law::Array{T, 4}
    dust_features::Array{T, 4}
    abs_features::Array{T, 4}
    extinction::Array{T, 3}
    abs_ice::Array{T, 3}
    abs_ch::Array{T, 3}
    hot_dust::Array{T, 3}
    templates::Array{T, 4}
    lines::Array{T, 4}

end


"""
    OpticalCubeModel(model, stellar, lines)

A structure for holding 3D models of intensity, split up into model components, generated when fitting a cube.
This will be the same shape as the input data, and preferably the same datatype too (i.e., JWST files have flux
and error in Float32 format, so we should also output in Float32 format).  This is useful as a way to quickly
compare the full model, or model components, to the data.

# Fields {T<:Real}
- `model::Array{T, 3}`: The full 3D model.
- `stellar::Array{T, 4}`: The simple stellar population components of the continuum. The 4th axis runs over each individual population.
- `na_feii::Array{T, 3}`: The narrow Fe II emission component.
- `br_feii::Array{T, 3}`: The broad Fe II emission component.
- `power_law::Array{T, 4}`: The power law components of the continuum. The 4th axis runs over each individual power law.
- `attenuation_stars::Array{T, 3}`: The dust attenuation on the stellar population.
- `attenuation_gas::Array{T, 3}`: The dust attenuation on the gas, related to attenuation_stars by E(B-V)_stars = E(B-V)_factor * E(B-V)_gas
- `lines::Array{T, 4}`: The line profiles. The 4th axis runs over each individual line.

See [`cubemodel_empty`](@ref) for a default constructor method.
"""
struct OpticalCubeModel{T<:Real} <: CubeModel

    model::Array{T, 3}
    stellar::Array{T, 4}
    na_feii::Array{T, 3}
    br_feii::Array{T, 3}
    power_law::Array{T, 4}
    attenuation_stars::Array{T, 3}
    attenuation_gas::Array{T, 3}
    lines::Array{T, 4}

end


"""
    cubemodel_empty(shape, n_dust_cont, n_power_law, df_names, ab_names, line_names[, floattype])

A constructor function for making a default empty MIRCubeModel object with all the necessary fields for a given
fit of a DataCube.

# Arguments
- `shape::Tuple`: The dimensions of the DataCube being fit, formatted as a tuple of (nx, ny, nz)
- `n_dust_cont::Integer`: The number of dust continuum components in the fit (usually given by the number of temperatures 
    specified in the dust.toml file)
- `n_power_law::Integer`: The number of power law continuum components in the fit.
- `df_names::Vector{String}`: List of names of PAH features being fit, i.e. "PAH_12.62", ...
- `ab_names::Vector{String}`: List of names of absorption features being fit, i.e. "abs_HCO+_12.1", ...
- `temp_names::Vector{String}`: List of names of generic templates in the fit, i.e. "nuclear", ...
- `line_names::Vector{Symbol}`: List of names of lines being fit, i.e. "NeVI_7652", ...
- `floattype::DataType=Float32`: The type of float to use in the arrays. Should ideally be the same as the input data,
    which for JWST is Float32.
"""
function cubemodel_empty(shape::Tuple, n_dust_cont::Integer, n_power_law::Integer, df_names::Vector{String}, 
    ab_names::Vector{String}, temp_names::Vector{String}, line_names::Vector{Symbol}, floattype::DataType=Float32)::MIRCubeModel

    @debug """\n
    Creating MIRCubeModel struct with shape $shape
    ##############################################
    """

    # Make sure the floattype given is actually a type of float
    @assert floattype <: AbstractFloat "floattype must be a type of AbstractFloat (Float32 or Float64)!"
    # Swap the wavelength axis to be the FIRST axis since it is accessed most often and thus should be continuous in memory
    shape2 = (shape[end], shape[1:end-1]...)

    # Initialize the arrays for each part of the full 3D model
    model = zeros(floattype, shape2...)
    @debug "model cube"
    stellar = zeros(floattype, shape2...)
    @debug "stellar continuum comp cube"
    dust_continuum = zeros(floattype, shape2..., n_dust_cont)
    @debug "dust continuum comp cubes"
    power_law = zeros(floattype, shape2..., n_power_law)
    @debug "power law comp cubes"
    dust_features = zeros(floattype, shape2..., length(df_names))
    @debug "dust features comp cubes"
    abs_features = zeros(floattype, shape2..., length(ab_names))
    @debug "absorption features comp cubes"
    extinction = zeros(floattype, shape2...)
    @debug "extinction comp cube"
    abs_ice = zeros(floattype, shape2...)
    @debug "abs_ice comp cube"
    abs_ch = zeros(floattype, shape2...)
    @debug "abs_ch comp cube"
    hot_dust = zeros(floattype, shape2...)
    @debug "hot dust comp cube"
    templates = zeros(floattype, shape2..., length(temp_names))
    @debug "templates comp cube"
    lines = zeros(floattype, shape2..., length(line_names))
    @debug "lines comp cubes"

    MIRCubeModel(model, stellar, dust_continuum, power_law, dust_features, abs_features, 
        extinction, abs_ice, abs_ch, hot_dust, templates, lines)
end


"""
    cubemodel_empty(shape, n_ssps, n_power_law, line_names[, floattype])

A constructor function for making a default empty OpticalCubeModel object with all the necessary fields for a given
fit of a DataCube.
    
# Arguments
- `shape::Tuple`: The dimensions of the DataCube being fit, formatted as a tuple of (nx, ny, nz)
- `n_ssps::Integer`: The number of simple stellar population continuum components in the fit.
- `n_power_law::Integer`: The number of power law continuum components in the fit.
- `line_names::Vector{Symbol}`: List of names of lines being fit, i.e. "NeVI_7652", ...
- `floattype::DataType=Float32`: The type of float to use in the arrays. Should ideally be the same as the input data,
which for JWST is Float32.
"""
function cubemodel_empty(shape::Tuple, n_ssps::Integer, n_power_law::Integer, line_names::Vector{Symbol}, 
    floattype::DataType=Float32)::OpticalCubeModel

    @debug """\n
    Creating OpticalCubeModel struct with shape $shape
    ##################################################
    """

    @assert floattype <: AbstractFloat "floattype must be a type of AbstractFloat (Float32 or Float64)!"
    # Swap the wavelength axis to be the FIRST axis since it is accessed most often and thus should be continuous in memory
    shape2 = (shape[end], shape[1:end-1]...)

    # Initialize the arrays for each part of the full 3D model
    model = zeros(floattype, shape2...)
    @debug "model cube"
    stellar = zeros(floattype, shape2..., n_ssps)
    @debug "stellar population comp cubes"
    na_feii = zeros(floattype, shape2...)
    @debug "narrow Fe II emission comp cube"
    br_feii = zeros(floattype, shape2...)
    @debug "broad Fe II emission comp cube"
    power_law = zeros(floattype, shape2..., n_power_law)
    @debug "power law comp cubes"
    attenuation_stars = zeros(floattype, shape2...)
    @debug "attenuation_stars comp cube"
    attenuation_gas = zeros(floattype, shape2...)
    @debug "attenuation_gas comp cube"
    lines = zeros(floattype, shape2..., length(line_names))
    @debug "lines comp cubes"

    OpticalCubeModel(model, stellar, na_feii, br_feii, power_law, attenuation_stars, attenuation_gas, lines)

end


"""
    CubeFitter(cube, z, name; <keyword arguments>)

This is the main structure used for fitting IFU cubes, containing all of the necessary data, metadata,
fitting options, and associated functions for generating ParamMaps and CubeModel structures to handle the outputs 
of all the fits.  This is essentially the "configuration object" that tells the rest of the fitting code how
to run. The actual fitting functions (`fit_spaxel` and `fit_cube!`) require an instance of this structure.

Other than `cube`, `z`, and `name`, all fields listed below will be populated by the defaults given in the 
`config.toml`, `dust.toml`, `lines.toml`, and `optical.toml` configuration files, or by quantities derived from
these options. For more detailed explanations on these options, please refer to the options files and the README file.

# Fields {T<:Real, S<:Integer, C<:Complex}

## Data
- `cube::DataCube`: The main DataCube object containing the cube that is being fit
- `z::Real`: The redshift of the target that is being fit
- `name::String`: The name of the fitting run being performed.
- `spectral_region::Symbol`: Either :MIR for mid-infrared, or :OPT for optical.

## Basic configuration options
- `user_mask::Union{Vector{<:Tuple},Nothing}`: An optional mask specifying areas of the spectrum to ignore during fitting.
- `plot_spaxels::Symbol`: A Symbol specifying the plotting backend to be used when plotting individual spaxel fits, can
    be either `:pyplot`, `:plotly`, or `:both`
- `plot_maps::Bool`: Whether or not to plot 2D maps of the best-fit parameters after the fitting is finished
- `plot_range::Union{Vector{<:Tuple},Nothing}`: An optional list of spectral regions to extract zoomed-in plots from (useful for plotting
specific emission lines of interest).
- `parallel::Bool`: Whether or not to fit multiple spaxels in parallel using multiprocessing
- `save_fits::Bool`: Whether or not to save the final best-fit models and parameters as FITS files
- `save_full_model::Bool`: Whether or not to save the full 3D best-fit model as a FITS file.
- `overwrite::Bool`: Whether or not to overwrite old fits of spaxels when rerunning
- `track_memory::Bool`: Whether or not to save diagnostic files showing memory usage of the program
- `track_convergence::Bool`: Whether or not to save diagnostic files showing convergence of line fitting for each spaxel
- `make_movies::Bool`: Whether or not to save mp4 files of the final model

## Basic fitting options
- `extinction_curve::String`: The type of extinction curve being used, i.e. `"kvt"` or `"d+"`
- `extinction_screen::Bool`: Whether or not the extinction is modeled as a screen
- `extinction_map::Union{Matrix{T},Nothing}`: An optional map of estimated extinction values. For MIR spectra, this is interpreted 
as tau_9.7 values, whereas for optical spectra it is interpreted as E(B-V) values. The fits will be locked to the value at the 
corresponding spaxel.
- `fit_stellar_continuum::Bool`: Whether or not to fit MIR stellar continuum
- `fit_sil_emission::Bool`: Whether or not to fit MIR hot silicate dust emission
- `guess_tau::Union{Vector{<:Tuple},Nothing}`: Whether or not to guess the optical depth at 9.7 microns by interpolating between 
    PAH-free parts of the continuum at the given wavelength windows in microns (must have enough spectral coverage). 
    The fitted value will then be constrained to be at least 80% of the inferred value.
- `fit_opt_na_feii::Bool`: Whether or not to fit optical narrow Fe II emission
- `fit_opt_br_feii::Bool`: Whether or not to fit optical broad Fe II emission
- `fit_all_samin::Bool`: Whether or not to fit all spaxels with simulated annealing
- `use_pah_templates::Bool`: Whether or not to fit the continuum in two steps, where the first step uses PAH templates,
and the second step fits the PAH residuals with the PAHFIT Drude model.
- `pah_template_map::BitMatrix`: Map of booleans specifying individual spaxels to fit with PAH templates, if use_pah_templates
is true. By default, if use_pah_templates is true, all spaxels will be fit with PAH templates.
- `fit_joint::Bool`: If true, fit the continuum and lines simultaneously. If false, the lines will first be masked and the
continuum will be fit, then the continuum is subtracted and the lines are fit to the residuals.
- `fit_uv_bump::Bool`: Whether or not to fit the UV bump in the dust attenuation profile. Only applies if the extinction
curve is "calzetti".
- `fit_covering_frac::Bool`: Whether or not to fit a dust covering fraction in the attenuation profile. Only applies if the
extinction curve is "calzetti".

## Continuum parameters
- `continuum::Continuum`: A Continuum structure holding information about various continuum fitting parameters.
This object will obviously be different depending on if the spectral region is :MIR or :OPT.

## MIR continuum parameters
- `n_dust_cont::S`: The number of dust continuum components
- `n_power_law::S`: The number of power law continuum components
- `n_dust_feat::S`: The number of PAH features
- `n_abs_feat::S`: The number of absorption features
- `n_templates::S`: The number of generic templates
- `templates::Array{T,4}`: Each template to be used in the fit. The first 2 axes are spatial, the 3rd axis should be wavelength, 
    and the 4th axis should iterate over each individual template. Each template will get an amplitude parameter in the fit.
- `template_names::Vector{String}`: The names of each generic template in the fit.
- `dust_features::Union{DustFeatures,Nothing}`: All of the fitting parameters for each PAH feature
- `abs_features::Union{DustFeatures,Nothing}`: All of the fitting parameters for each absorption feature
- `abs_taus::Union{Vector{Parameter},Nothing}`: A vector of amplitude parameters for the absorption features

## Optical continuum parameters
- `n_ssps::S`: The number of simple stellar population components
- `ssp_λ::Union{Vector{T},Nothing}`: The wavelength grid for the simple stellar population templates
- `ssp_templates::Union{Vector{Spline2D},Nothing}`: A vector of interpolating functions for the simple stellar populations
at each point in the wavelength grid.
- `feii_templates_fft::Union{Matrix{C},Nothing}`: A matrix of the Fourier transforms of the narrow and broad Fe II templates
- `vres::T`: The constant velocity resolution of the wavelength grid, which assumes logarithmic spacing, in km/s/pixel.
- `vsyst_ssp::T`: The systemic velocity offset between the input wavelength grid and the SSP template wavelength grid.
- `vsyst_feii::T`: The systemic velocity offset between the input wavelength grid and the Fe II template wavelength grid.
- `npad_feii::S`: The length of the Fe II templates (NOT the length of the Fourier transformed templates).

## Line parameters
- `n_lines::S`: The number of lines being fit
- `n_acomps::S`: The summed total number of line profiles fit to all lines in the spectrum, including additional components.
- `n_comps::S`: The maximum number of additional profiles that may be fit to any given line.
- `relative_flags::BitVector`: Flags for whether additional line components should be parametrized relative to the primary component,
    ordered as (amp, voff, fwhm) (global settings for all lines).
- `lines::TransitionLines`: A TransitionLines struct containing parameters and tied information for the emission lines.

## Tied Kinematics
- `tied_kinematics::TiedKinematics`: A TiedKinematics struct containing information about which lines have tied velocities and FWHMs.

## Tied voigt mixing
- `tie_voigt_mixing::Bool`: Whether or not the Voigt mixing parameter is tied between all the lines with Voigt profiles
- `voigt_mix_tied::Parameter`: The actual tied Voigt mixing parameter object, given `tie_voigt_mixing` is true

## Number of parameters
- `n_params_cont::S`: The total number of free fitting parameters for the continuum fit (not including emission lines)
- `n_params_lines::S`: The total number of free fitting parameters for the emission line fit (not including the continuum)
- `n_params_extra::S`: The total number of extra parameters calculated for each fit (includes things like line fluxes and equivalent widths)

## General options
- `cosmology::Cosmology.AbstractCosmology`: The Cosmology, used solely to create physical scale bars on the 2D parameter plots
- `flexible_wavesol::Bool`: Whether or not to allow small variations in the velocity offsets even when tied, to account
    for a bad wavelength solution
- `n_bootstrap::S`: The number of bootstrapping iterations that should be performed for each fit.
- `random_seed::S`: An optional number to use as a seed for the RNG utilized during bootstrapping, to ensure consistency between
attempts.
- `line_test_lines::Vector{Vector{Symbol}}`: A list of lines which should be tested for additional components. They may be grouped
together (hence the vector-of-a-vector structure) such that lines in a group will all be given the maximum number of parameters that
any one line passes the test for.
- `line_test_threshold::T`: A threshold which must be met in the chi^2 ratio between a fit without an additional line profile and
one with the additional profile, in order to include the additional profile in the final fit.
- `plot_line_test::Bool`: Whether or not to plot the line test results.

## Line masking options
- `linemask_Δ::S`: The half-width, in pixels, to use to calculate the second derivative for the line masking algorithm.
- `linemask_n_inc_thresh::S`: The number of times that the flux must increase to the left/right before the line masking window stops.
- `linemask_thresh::T`: The number of sigmas that a peak in the second derivative must reach to be considered a line detection in
    the line masking algorithm.
- `linemask_overrides::Vector{Tuple{T,T}}`: Optional list of tuples specifying (min, max) wavelength ranges that will be forcibly 
    added to the line mask. This is different from the `user_mask` option since it only applies to the continuum fitting step but
    will be ignored during the line fitting step.
- `map_snr_thresh::T`: The SNR threshold below which to mask out spaxels from parameter maps for emission lines.

## Best fit parameters
- `p_init_cont::Vector{T}`: The best-fit continuum parameters for the initial fit to the sum of all spaxels.
- `p_init_line::Vector{T}`: Same as `p_init_cont`, but for the line parameters
- `p_init_pahtemp::Vector{T}`: Same as `p_init_cont`, but for the PAH templates amplitudes

See [`ParamMaps`](@ref), [`parammaps_empta`](@ref), [`CubeModel`](@ref), [`cubemodel_empty`](@ref), 
    [`fit_spaxel`](@ref), [`fit_cube!`](@ref)
"""
struct CubeFitter{T<:Real,S<:Integer,C<:Complex} 

    # See explanations for each field in the docstring!
    
    # Data
    cube::DataCube
    z::T
    name::String
    spectral_region::Symbol

    # Basic fitting options
    user_mask::Union{Vector{<:Tuple},Nothing}
    plot_spaxels::Symbol
    plot_maps::Bool
    plot_range::Union{Vector{<:Tuple},Nothing}
    parallel::Bool
    save_fits::Bool
    save_full_model::Bool
    overwrite::Bool
    track_memory::Bool
    track_convergence::Bool
    make_movies::Bool

    extinction_curve::String
    extinction_screen::Bool
    extinction_map::Union{Matrix{T},Nothing}
    fit_stellar_continuum::Bool
    fit_sil_emission::Bool
    guess_tau::Union{Vector{<:Tuple},Nothing}
    fit_opt_na_feii::Bool
    fit_opt_br_feii::Bool
    fit_all_samin::Bool
    use_pah_templates::Bool
    pah_template_map::BitMatrix
    fit_joint::Bool
    fit_uv_bump::Bool
    fit_covering_frac::Bool

    # Continuum parameters
    continuum::Continuum

    # MIR continuum parameters
    n_dust_cont::S
    n_power_law::S  # (also used for optical power laws)
    n_dust_feat::S
    n_abs_feat::S
    n_templates::S
    # 4D templates: first 3 axes are the template's spatial and spectral axes, while the 4th axis enumerates individual templates
    templates::Array{T, 4}  
    template_names::Vector{String}
    dust_features::Union{DustFeatures,Nothing}
    abs_features::Union{DustFeatures,Nothing}
    abs_taus::Union{Vector{Parameter},Nothing}

    # Optical continuum parameters
    n_ssps::S
    ssp_λ::Union{Vector{T},Nothing}
    ssp_templates::Union{Vector{Spline2D},Nothing}
    feii_templates_fft::Union{Matrix{C},Nothing}
    vres::T
    vsyst_ssp::T
    vsyst_feii::T
    npad_feii::S

    # Line parameters
    n_lines::S
    n_acomps::S
    n_comps::S
    relative_flags::BitVector
    lines::TransitionLines

    # Tied voffs
    tied_kinematics::TiedKinematics

    # Tied voigt mixing
    tie_voigt_mixing::Bool
    voigt_mix_tied::Parameter

    # Number of parameters
    n_params_cont::S
    n_params_lines::S
    n_params_extra::S
    
    # General options
    cosmology::Cosmology.AbstractCosmology
    flexible_wavesol::Bool
    n_bootstrap::S
    random_seed::S
    line_test_lines::Vector{Vector{Symbol}}
    line_test_threshold::T
    plot_line_test::Bool

    # Line masking options
    linemask_Δ::S
    linemask_n_inc_thresh::S
    linemask_thresh::T
    linemask_overrides::Vector{Tuple{T,T}}
    map_snr_thresh::T

    p_init_cont::Vector{T}
    p_init_line::Vector{T}
    p_init_pahtemp::Vector{T}

    p_init_cube_λ::Union{Vector{T},Nothing}
    p_init_cube_cont::Union{Array{T,3},Nothing}
    p_init_cube_lines::Union{Array{T,3},Nothing}
    p_init_cube_wcs::Union{WCSTransform,Nothing}
    p_init_cube_coords::Union{Vector{Vector{T}},Nothing}
    p_init_cube_Ω::Union{T,Nothing}

    #= Constructor function --> the default inputs are all taken from the configuration files, but may be overwritten
    by the kwargs object using the same syntax as any keyword argument. The rest of the fields are generated in the function 
    from these inputs =#
    function CubeFitter(cube::DataCube, z::Real, name::String; kwargs...) 
        
        # Prepare options
        options = parse_options()

        out = copy(options)
        for key in keys(kwargs)
            out[key] = kwargs[key]
        end
        out[:line_test_lines] = [[Symbol(ln) for ln in group] for group in out[:line_test_lines]]
        out[:plot_spaxels] = Symbol(out[:plot_spaxels])
        if !haskey(out, :plot_range)
            out[:plot_range] = nothing
        else
            out[:plot_range] = [tuple(out[:plot_range][i]...) for i in 1:length(out[:plot_range])]
            for  pair in out[:plot_range]
                @assert pair[1] < pair[2] "plot_range pairs must be in ascending order!"
            end
        end
        if !haskey(out, :user_mask)
            out[:user_mask] = nothing
        else
            out[:user_mask] = [tuple(out[:user_mask][i]...) for i in 1:length(out[:user_mask])]
            for  pair in out[:user_mask]
                @assert pair[1] < pair[2] "user_mask pairs must be in ascending order!"
            end
        end

        # Prepare output directories
        @info "Preparing output directories"
        spectral_region = cube.spectral_region
        name = replace(name, #= no spaces! =# " " => "_")
        name = join([name, lowercase(string(spectral_region))], "_")

        # Top-level output directory
        if !isdir("output_$name")
            mkdir("output_$name")
        end
        # Sub-folder for 1D plots of spaxel fits
        if !isdir(joinpath("output_$name", "spaxel_plots"))
            mkdir(joinpath("output_$name", "spaxel_plots"))
        end
        if !isdir(joinpath("output_$name", "zoomed_plots")) && !isnothing(out[:plot_range])
            mkdir(joinpath("output_$name", "zoomed_plots"))
        end
        if !isdir(joinpath("output_$name", "line_tests")) && (length(out[:line_test_lines]) > 0) && out[:plot_line_test]
            mkdir(joinpath("output_$name", "line_tests"))
        end
        # Sub-folder for data files saving the results of individual spaxel fits
        if !isdir(joinpath("output_$name", "spaxel_binaries"))
            mkdir(joinpath("output_$name", "spaxel_binaries"))
        end
        # Sub-folder for 2D parameter maps 
        if !isdir(joinpath("output_$name", "param_maps"))
            mkdir(joinpath("output_$name", "param_maps"))
        end
        # Sub-folder for log files
        if !isdir(joinpath("output_$name", "logs"))
            mkdir(joinpath("output_$name", "logs"))
        end

        # More default options
        if !haskey(out, :templates)
            out[:templates] = Array{Float64, 4}(undef, size(cube.I)..., 0)
        elseif ndims(out[:templates]) == 3
            t4 = Array{Float64, 4}(undef, size(cube.I)..., 1)
            t4[:, :, :, 1] .= out[:templates]
            out[:templates] = t4
        end

        if !haskey(out, :template_names)
            out[:template_names] = String["template_$i" for i in axes(out[:templates], 4)]
        end

        if !haskey(out, :linemask_delta)
            out[:linemask_delta] = spectral_region == :MIR ? 3 : 20
        end
        if !haskey(out, :linemask_n_inc_thresh)
            out[:linemask_n_inc_thresh] = spectral_region == :MIR ? 3 : 7
        end
        if !haskey(out, :linemask_thresh)
            out[:linemask_thresh] = 3.
        end
        if !haskey(out, :linemask_overrides)
            out[:linemask_overrides] = Tuple[]
        end
        if !haskey(out, :map_snr_thresh)
            out[:map_snr_thresh] = 3.
        end
        if !haskey(out, :guess_tau)
            out[:guess_tau] = nothing
        end

        #############################################################

        @debug """\n
        Creating CubeFitter struct for $name at z=$z
        ############################################
        """

        # Alias
        λ = cube.λ
        # Get potential extinction map
        extinction_map = nothing
        if haskey(out, :extinction_map) && !isnothing(out[:extinction_map])
            extinction_map = out[:extinction_map]
            @assert size(extinction_map) == size(cube.I)[1:2] "The extinction map must match the shape of the first two dimensions of the intensity map!"
        end

        if spectral_region == :MIR

            continuum, dust_features_0, abs_features_0, abs_taus_0 = parse_dust()

            # Adjust wavelengths/FWHMs for any local absorption features
            for i in 1:length(abs_features_0.names)
                if abs_features_0._local[i]
                    abs_features_0.mean[i].value /= (1 + z)
                    abs_features_0.mean[i].limits = (abs_features_0.mean[i].limits[1] / (1 + z),
                                                abs_features_0.mean[i].limits[2] / (1 + z))
                    abs_features_0.fwhm[i].value /= (1 + z)
                    abs_features_0.fwhm[i].limits = (abs_features_0.fwhm[i].limits[1] / (1 + z),
                                                abs_features_0.fwhm[i].limits[2] / (1 + z))
                end
            end

            @debug "### Model will include 1 stellar continuum component ###" *
                "\n### at T = $(continuum.T_s.value) K ###"

            #### PREPARE OUTPUTS ####
            n_dust_cont = length(continuum.T_dc)
            msg = "### Model will include $n_dust_cont dust continuum components ###"
            for T_dci ∈ continuum.T_dc
                msg *= "\n### at T = $(T_dci.value) K ###"
            end
            @debug msg 

            n_power_law = length(continuum.α)
            msg = "### Model will include $n_power_law power law components ###"
            for αi ∈ continuum.α
                msg *= "\n### with alpha = $(αi.value) ###"
            end
            @debug msg

            # Only use PAH features within +/-0.5 um of the region being fit (to include wide tails)
            df_filt = [(minimum(λ)-0.5 < dust_features_0.mean[i].value < maximum(λ)+0.5) for i ∈ 1:length(dust_features_0.mean)]
            if !isnothing(out[:user_mask])
                for pair in out[:user_mask]
                    df_filt .&= [~(pair[1] < dust_features_0.mean[i].value < pair[2]) for i ∈ 1:length(dust_features_0.mean)]
                end
            end
            dust_features = DustFeatures(dust_features_0.names[df_filt], 
                                        dust_features_0.profiles[df_filt],
                                        dust_features_0.mean[df_filt],
                                        dust_features_0.fwhm[df_filt],
                                        dust_features_0.index[df_filt],
                                        dust_features_0.cutoff[df_filt],
                                        dust_features_0.complexes[df_filt],
                                        dust_features_0._local[df_filt])
            n_dust_features = length(dust_features.names)
            msg = "### Model will include $n_dust_features dust feature (PAH) components ###"
            for df_mn ∈ dust_features.mean
                msg *= "\n### at lambda = $df_mn um ###"
            end
            @debug msg

            # Only use absorption features within +/-0.5 um of the region being fit
            ab_filt = [(minimum(λ)-0.5 < abs_features_0.mean[i].value < maximum(λ)+0.5) for i ∈ 1:length(abs_features_0.mean)]
            if !isnothing(out[:user_mask])
                for pair in out[:user_mask]
                    ab_filt .&= [~(pair[1] < abs_features_0.mean[i].value < pair[2]) for i ∈ 1:length(abs_features_0.mean)]
                end
            end
            abs_features = DustFeatures(abs_features_0.names[ab_filt],
                                        abs_features_0.profiles[ab_filt],
                                        abs_features_0.mean[ab_filt],
                                        abs_features_0.fwhm[ab_filt],
                                        abs_features_0.index[ab_filt],
                                        abs_features_0.cutoff[ab_filt],
                                        abs_features_0.complexes[ab_filt],
                                        abs_features_0._local[ab_filt])
            abs_taus = abs_taus_0[ab_filt]
            n_abs_features = length(abs_features.names)
            msg = "### Model will include $n_abs_features absorption feature components ###"
            for ab_mn ∈ abs_features.mean
                msg *= "\n### at lambda = $ab_mn um ###"
            end
            @debug msg

            # Set defaults for the optical components that will not be fit
            n_ssps = 0
            n_templates = size(out[:templates], 4)
            ssp_λ = ssp_templates = feii_templates_fft = nothing
            npad_feii = vres = vsyst_ssp = vsyst_feii = 0

            if n_templates == 0
                # Ignore any template amplitude entries in the dust.toml options if there are no templates
                continuum = MIRContinuum(continuum.T_s, continuum.T_dc, continuum.α, continuum.τ_97, continuum.τ_ice,
                                         continuum.τ_ch, continuum.β, continuum.T_hot, continuum.Cf_hot, continuum.τ_warm, 
                                         continuum.τ_cold, continuum.sil_peak, Parameter[])
            end

        elseif spectral_region == :OPT

            continuum = parse_optical()

            # Create the simple stellar population templates with FSPS
            ssp_λ, ages, metals, ssp_templates = generate_stellar_populations(λ, cube.lsf, z, out[:cosmology], name)
            # Create a 2D linear interpolation over the ages/metallicities
            ssp_templates = [Spline2D(ages, metals, ssp_templates[:, :, i], kx=1, ky=1) for i in eachindex(ssp_λ)]

            # Load in the Fe II templates from Veron-Cetty et al. (2004)
            npad_feii, feii_λ, na_feii_fft, br_feii_fft = generate_feii_templates(λ, cube.lsf)
            # Create a matrix containing both templates
            feii_templates_fft = [na_feii_fft br_feii_fft]

            ### PREPARE OUTPUTS ###
            n_ssps = length(continuum.ssp_ages)
            msg = "### Model will include $n_ssps simple stellar population components ###"
            for (age, z) ∈ zip(continuum.ssp_ages, continuum.ssp_metallicities)
                msg *= "\n### at age = $(age.value) Gyr and [M/H] = $(z.value) ###"
            end
            @debug msg

            n_power_law = length(continuum.α)
            msg = "### Model will include $n_power_law power law components ###"
            for pl ∈ continuum.α
                msg *= "\n### with index = $(pl.value) ###"
            end
            @debug msg

            # Calculate velocity scale and systemic velocity offset b/w templates and input
            vres = log(λ[2]/λ[1]) * C_KMS
            vsyst_ssp = log(ssp_λ[1]/λ[1]) * C_KMS
            vsyst_feii = log(feii_λ[1]/λ[1]) * C_KMS

            # Set defaults for the MIR components that will not be fit
            n_dust_cont = n_dust_features = n_abs_features = n_templates = 0
            dust_features = abs_features = abs_taus = nothing

        end

        lines_0, tied_kinematics, flexible_wavesol, tie_voigt_mixing, voigt_mix_tied = parse_lines()

        # Only use lines within the wavelength range being fit
        ln_filt = minimum(λ) .< lines_0.λ₀ .< maximum(λ)
        if !isnothing(out[:user_mask])
            for pair in out[:user_mask]
                ln_filt .&= .~(pair[1] .< lines_0.λ₀ .< pair[2])
            end
        end
        # Convert to a vectorized "TransitionLines" object
        lines = TransitionLines(lines_0.names[ln_filt], lines_0.latex[ln_filt], lines_0.annotate[ln_filt], lines_0.λ₀[ln_filt], 
                                lines_0.profiles[ln_filt, :], lines_0.tied_amp[ln_filt, :], lines_0.tied_voff[ln_filt, :], lines_0.tied_fwhm[ln_filt, :], 
                                lines_0.acomp_amp[ln_filt, :], lines_0.voff[ln_filt, :], lines_0.fwhm[ln_filt, :], lines_0.h3[ln_filt, :], 
                                lines_0.h4[ln_filt, :], lines_0.η[ln_filt, :], lines_0.combined, lines_0.rel_amp, lines_0.rel_voff, lines_0.rel_fwhm)
        n_lines = length(lines.names)
        n_comps = size(lines.profiles, 2)
        n_acomps = sum(.!isnothing.(lines.profiles[:, 2:end]))
        msg = "### Model will include $n_lines emission lines ###"
        for (k, (name, λ0, prof)) ∈ enumerate(zip(lines.names, lines.λ₀, lines.profiles))
            msg *= "\n### $name at lambda = $λ0 um with $prof profile ###"
            for acomp_prof ∈ lines.profiles[k, 2:end]
                if !isnothing(acomp_prof)
                    msg *= "\n###   and $acomp_prof acomp profile   ###"
                end
            end
        end
        @debug msg
        relative_flags = BitVector([lines.rel_amp, lines.rel_voff, lines.rel_fwhm])

        # Remove unnecessary rows/keys from the tied_kinematics object after the lines have been filtered
        @debug "TiedKinematics before filtering: $tied_kinematics"
        for j ∈ 1:n_comps
            keep0 = Int64[]
            for (k, key) ∈ enumerate(tied_kinematics.key_amp[j])
                if any(lines.tied_amp[:, j] .== key)
                    # Remove the unneeded elements
                    append!(keep0, [k])
                end
            end
            tied_kinematics.key_amp[j] = tied_kinematics.key_amp[j][keep0]
            tied_kinematics.amp[j] = tied_kinematics.amp[j][keep0]
            # Remove line ratios for lines that fall outside the fitting region
            for i in eachindex(tied_kinematics.amp[j])
                # Make a copy so we're not modifying the dictionary as we loop through it
                acp = copy(tied_kinematics.amp[j][i])
                for lk in keys(acp)
                    if !(lk in lines.names)
                        delete!(tied_kinematics.amp[j][i], lk)
                    end
                end
                # Renormalize the ratios so the largest is 1.0
                max_ratio = maximum(values(tied_kinematics.amp[j][i]))
                for lk in keys(tied_kinematics.amp[j][i])
                    tied_kinematics.amp[j][i][lk] /= max_ratio
                end
            end

            keep1 = Int64[]
            for (k, key) ∈ enumerate(tied_kinematics.key_voff[j])
                if any(lines.tied_voff[:, j] .== key)
                    # Remove the unneeded elements
                    append!(keep1, [k])
                end
            end
            tied_kinematics.key_voff[j] = tied_kinematics.key_voff[j][keep1]
            tied_kinematics.voff[j] = tied_kinematics.voff[j][keep1]

            keep2 = Int64[]
            for (k, key) ∈ enumerate(tied_kinematics.key_fwhm[j])
                if any(lines.tied_fwhm[:, j] .== key)
                    # Remove the unneeded elements
                    append!(keep2, [k])
                end
            end
            tied_kinematics.key_fwhm[j] = tied_kinematics.key_fwhm[j][keep2]
            tied_kinematics.fwhm[j] = tied_kinematics.fwhm[j][keep2]
        end
        @debug "TiedKinematics after filtering: $tied_kinematics"

        # Also store the "tied" parameter for each line, which will need to be checked against the kin_tied_key
        # during fitting to find the proper location of the tied voff parameter to use
        msg = "### Model will include these tied kinematics parameters for each component ###"
        for lt ∈ tied_kinematics.key_voff
            msg *= "\n### voff for group $lt ###"
        end
        for lt ∈ tied_kinematics.key_fwhm
            msg *= "\n### fwhm for group $lt ###"
        end
        @debug msg

        # One η for all voigt profiles
        if any(lines.profiles .== :Voigt) && tie_voigt_mixing
            @debug "### Model will include 1 tied voigt mixing parameter ###"
        end

        # Total number of parameters for the continuum and line fits
        if spectral_region == :MIR
            n_params_cont = (2+4) + 2n_dust_cont + 2n_power_law + 3n_abs_features + (out[:fit_sil_emission] ? 6 : 0) + n_templates
            n_params_cont += 3 * sum(dust_features.profiles .== :Drude) + 5 * sum(dust_features.profiles .== :PearsonIV)
        elseif spectral_region == :OPT
            n_params_cont = 3n_ssps + 2 + 2 + 2n_power_law
            if out[:fit_opt_na_feii]
                n_params_cont += 3
            end
            if out[:fit_opt_br_feii]
                n_params_cont += 3
            end
            if out[:fit_uv_bump]
                n_params_cont += 1
            end
            if out[:fit_covering_frac]
                n_params_cont += 1
            end
        end

        n_params_lines = 0
        for i ∈ 1:n_lines
            for j ∈ 1:n_comps
                if !isnothing(lines.profiles[i, j])
                    # amplitude, voff, and FWHM parameters
                    n_params_lines += 3
                    if !isnothing(lines.tied_voff[i, j]) && flexible_wavesol && isone(j)
                        # individual voff parameter
                        n_params_lines += 1
                    end
                    if lines.profiles[i, j] == :GaussHermite
                        # extra h3 and h4 parmeters
                        n_params_lines += 2
                    elseif lines.profiles[i, j] == :Voigt
                        # extra mixing parameter
                        n_params_lines += 1
                    end
                end
            end
        end
        n_params_extra = 3 * (n_dust_features + n_lines + n_acomps)
        @debug "### There is a total of $(n_params_cont) continuum parameters ###"
        @debug "### There is a total of $(n_params_lines) emission line parameters ###"
        @debug "### There is a total of $(n_params_extra) extra parameters ###"

        # Prepare initial best fit parameter options
        @debug "Preparing initial best fit parameter vectors with $(n_params_cont) and $(n_params_lines) parameters"
        p_init_cont = zeros(n_params_cont)
        p_init_line = zeros(n_params_lines)
        p_init_pahtemp = zeros(2)

        if !(out[:use_pah_templates])
            pah_template_map = falses(size(cube.I)[1:2])
        else
            if haskey(kwargs, :pah_template_map)
                pah_template_map = kwargs[:pah_template_map]
            else
                pah_template_map = trues(size(cube.I)[1:2])
            end
        end

        # If a fit has been run previously, read in the file containing the rolling best fit parameters
        # to pick up where the fitter left off seamlessly
        if isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv")) && isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"))
            p_init_cont = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv"), ',', Float64, '\n')[:, 1]
            p_init_line = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"), ',', Float64, '\n')[:, 1]
            p_init_pahtemp = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_pahtemp.csv"), ',', Float64, '\n')[:, 1]
        end

        p_init_cube_λ = p_init_cube_cont = p_init_cube_lines = p_init_cube_wcs = p_init_cube_coords = p_init_cube_Ω = nothing
        if haskey(out, :p_init_cube)

            # Use fitting parameters from another run (on a potentially different WCS) as initial conditions
            path = out[:p_init_cube]
            oname = replace(basename(path), "output_" => "")
            hdu = FITS(joinpath(path, "$(oname)_full_model.fits"))
            # Wavelength vector
            p_init_cube_λ = read(hdu["WAVELENGTH"], "wave") ./ (1 + z)
            # WCS
            p_init_cube_wcs = WCS.from_header(read_header(hdu[1], String))[1]
            # Angular resolution
            p_init_cube_Ω = read_header(hdu[1])["PIXAR_SR"]
            # Parameters from the fits
            p_init_cube_cont = ones(size(cube.I)[1:2]..., n_params_cont) .* NaN
            p_init_cube_lines = ones(size(cube.I)[1:2]..., n_params_lines) .* NaN

            # Filter out dust features
            df_filt = [(minimum(p_init_cube_λ)-0.5 < dust_features_0.mean[i].value < maximum(p_init_cube_λ)+0.5) for i ∈ 1:length(dust_features_0.mean)]
            if !isnothing(out[:user_mask])
                for pair in out[:user_mask]
                    df_filt .&= [~(pair[1] < dust_features_0.mean[i].value < pair[2]) for i ∈ 1:length(dust_features_0.mean)]
                end
            end
            initcube_dust_features = DustFeatures(dust_features_0.names[df_filt], 
                                        dust_features_0.profiles[df_filt],
                                        dust_features_0.mean[df_filt],
                                        dust_features_0.fwhm[df_filt],
                                        dust_features_0.index[df_filt],
                                        dust_features_0.cutoff[df_filt],
                                        dust_features_0.complexes[df_filt],
                                        dust_features_0._local[df_filt])
            # Count how many dust features are in the cube template but not in the current fitting region
            n_dfparams_left = n_dfparams_right = 0
            for i in 1:length(initcube_dust_features.names)
                if initcube_dust_features.mean[i].value < (minimum(λ)-0.5)
                    if initcube_dust_features.profiles[i] == :Drude
                        n_dfparams_left += 3
                    elseif initcube_dust_features.profiles[i] == :PearsonIV
                        n_dfparams_left += 5
                    end
                end
                if initcube_dust_features.mean[i].value > (maximum(λ)+0.5)
                    if initcube_dust_features.profiles[i] == :Drude
                        n_dfparams_right += 3
                    elseif initcube_dust_features.profiles[i] == :PearsonIV
                        n_dfparams_right += 5
                    end
                end
            end

            # Repeat for absorption features
            ab_filt = [(minimum(p_init_cube_λ)-0.5 < abs_features_0.mean[i].value < maximum(p_init_cube_λ)+0.5) for i ∈ 1:length(abs_features_0.mean)]
            if !isnothing(out[:user_mask])
                for pair in out[:user_mask]
                    ab_filt .&= [~(pair[1] < abs_features_0.mean[i].value < pair[2]) for i ∈ 1:length(abs_features_0.mean)]
                end
            end
            initcube_abs_features = DustFeatures(abs_features_0.names[ab_filt],
                                        abs_features_0.profiles[ab_filt],
                                        abs_features_0.mean[ab_filt],
                                        abs_features_0.fwhm[ab_filt],
                                        abs_features_0.index[ab_filt],
                                        abs_features_0.cutoff[ab_filt],
                                        abs_features_0.complexes[ab_filt],
                                        abs_features_0._local[ab_filt])
            n_abparams_left = n_abparams_right = 0
            for i in 1:length(initcube_abs_features.names)
                if initcube_abs_features.mean[i].value < (minimum(λ)-0.5)
                    n_abparams_left += 3
                end
                if initcube_abs_features.mean[i].value > (maximum(λ)+0.5)
                    n_abparams_right += 3
                end
            end

            # Repeat for emission lines
            ln_filt = minimum(p_init_cube_λ) .< lines_0.λ₀ .< maximum(p_init_cube_λ)
            if !isnothing(out[:user_mask])
                for pair in out[:user_mask]
                    ln_filt .&= .~(pair[1] .< lines_0.λ₀ .< pair[2])
                end
            end
            # Convert to a vectorized "TransitionLines" object
            initcube_lines = TransitionLines(lines_0.names[ln_filt], lines_0.latex[ln_filt], lines_0.annotate[ln_filt], lines_0.λ₀[ln_filt], 
                                    lines_0.profiles[ln_filt, :], lines_0.tied_amp[ln_filt, :], lines_0.tied_voff[ln_filt, :], lines_0.tied_fwhm[ln_filt, :], 
                                    lines_0.acomp_amp[ln_filt, :], lines_0.voff[ln_filt, :], lines_0.fwhm[ln_filt, :], lines_0.h3[ln_filt, :], 
                                    lines_0.h4[ln_filt, :], lines_0.η[ln_filt, :], lines_0.combined, lines_0.rel_amp, lines_0.rel_voff, lines_0.rel_fwhm)
            n_lineparams_left = n_lineparams_right = 0
            n_initcube_lineparams = 0
            for i in 1:length(initcube_lines.names)
                for j in 1:n_comps
                    if !isnothing(initcube_lines.profiles[i, j])
                        n_initcube_lineparams += 3
                        if !isnothing(initcube_lines.tied_voff[i, j]) && flexible_wavesol && isone(j)
                            n_initcube_lineparams += 1
                        end
                        if initcube_lines.profiles[i, j] == :GaussHermite
                            n_initcube_lineparams += 2
                        elseif initcube_lines.profiles[i, j] == :Voigt
                            n_initcube_lineparams += 1
                        end
                        if initcube_lines.λ₀[i] < minimum(λ)
                            n_lineparams_left += 3
                            if !isnothing(initcube_lines.tied_voff[i, j]) && flexible_wavesol && isone(j)
                                # individual voff parameter
                                n_lineparams_left += 1
                            end
                            if initcube_lines.profiles[i, j] == :GaussHermite
                                # extra h3 and h4 parmeters
                                n_lineparams_left += 2
                            elseif initcube_lines.profiles[i, j] == :Voigt
                                # extra mixing parameter
                                n_lineparams_left += 1
                            end 
                        end
                        if initcube_lines.λ₀[i] > maximum(λ)
                            n_lineparams_right += 3
                            if !isnothing(initcube_lines.tied_voff[i, j]) && flexible_wavesol && isone(j)
                                # individual voff parameter
                                n_lineparams_right += 1
                            end
                            if initcube_lines.profiles[i, j] == :GaussHermite
                                # extra h3 and h4 parmeters
                                n_lineparams_right += 2
                            elseif initcube_lines.profiles[i, j] == :Voigt
                                # extra mixing parameter
                                n_lineparams_right += 1
                            end 
                        end
                    end
                end
            end

            # Now loop through the spaxels and assign the initial fitting parameters based on the saved results
            spaxfiles = [f for f in readdir(joinpath(path, "spaxel_binaries")) if contains(f, "spaxel")]
            for sf in spaxfiles
                params = readdlm(joinpath(path, "spaxel_binaries", sf), ',', Float64, '\n')[:,1]
                c1 = (2+4) + 2n_dust_cont + 2n_power_law
                c2 = c1 + n_abparams_left + 3n_abs_features - n_abparams_right
                c3 = c2 + n_abparams_right + (out[:fit_sil_emission] ? 6 : 0) + n_templates
                c4 = c3 + n_dfparams_left + 3sum(dust_features.profiles .== :Drude) + 5sum(dust_features.profiles .== :PearsonIV)
                c5 = c4 + n_dfparams_right + n_lineparams_left + n_params_lines
                params_cont1 = params[1:c1]
                params_ab = params[(1+c1+n_abparams_left):c2]
                params_cont2 = params[(1+c2+n_abparams_right):c3]
                params_df = params[(1+c3+n_dfparams_left):c4]
                params_lines = params[(1+c4+n_dfparams_right+n_lineparams_left):c5]
                params_cont = [params_cont1; params_ab; params_cont2; params_df]

                @assert length(params_cont) == size(p_init_cube_cont, 3) "Sizes do not match between init cube cont params and current cube params!"
                @assert length(params_lines) == size(p_init_cube_lines, 3) "Sizes do not match between init cube line params and current cube params!"

                spax = split(replace(sf, ".csv" => ""), "_")[end-1:end]
                spax_x = parse(Int, spax[1])
                spax_y = parse(Int, spax[2])
                p_init_cube_cont[spax_x, spax_y, :] .= params_cont
                p_init_cube_lines[spax_x, spax_y, :] .= params_lines
            end

            # Calculate the cube spaxel coordinates in the current WCS frame
            # Get the coordinates of all spaxels that have fit results
            coords0 = [float.(c.I) for c in CartesianIndices(size(p_init_cube_cont)[1:2]) if !all(isnan.(p_init_cube_cont[c,:]))]
            # Transform to coordinates in the WCS of our current frame
            p_init_cube_coords = [world_to_pix(cube.wcs, pix_to_world(p_init_cube_wcs, [coord..., 1.]))[1:2] for coord in coords0]

        end


        ctype = isnothing(feii_templates_fft) ? ComplexF64 : eltype(feii_templates_fft)
        new{typeof(z), typeof(n_lines), ctype}(cube, z, name, spectral_region, out[:user_mask], out[:plot_spaxels], out[:plot_maps], out[:plot_range], 
            out[:parallel], out[:save_fits], out[:save_full_model], out[:overwrite], out[:track_memory], out[:track_convergence], out[:make_movies], 
            out[:extinction_curve], out[:extinction_screen], extinction_map, out[:fit_stellar_continuum], out[:fit_sil_emission], out[:guess_tau], out[:fit_opt_na_feii], 
            out[:fit_opt_br_feii], out[:fit_all_samin], out[:use_pah_templates], pah_template_map, out[:fit_joint], out[:fit_uv_bump], out[:fit_covering_frac], 
            continuum, n_dust_cont, n_power_law, n_dust_features, n_abs_features, n_templates, out[:templates], out[:template_names], dust_features, 
            abs_features, abs_taus, n_ssps, ssp_λ, ssp_templates, feii_templates_fft, vres, vsyst_ssp, vsyst_feii, npad_feii, n_lines, n_acomps, n_comps, relative_flags, 
            lines, tied_kinematics, tie_voigt_mixing, voigt_mix_tied, n_params_cont, n_params_lines, n_params_extra, out[:cosmology], flexible_wavesol, out[:n_bootstrap], 
            out[:random_seed], out[:line_test_lines], out[:line_test_threshold], out[:plot_line_test], out[:linemask_delta], out[:linemask_n_inc_thresh], out[:linemask_thresh], 
            out[:linemask_overrides], out[:map_snr_thresh], p_init_cont, p_init_line, p_init_pahtemp, p_init_cube_λ, p_init_cube_cont, p_init_cube_lines, p_init_cube_wcs, 
            p_init_cube_coords, p_init_cube_Ω)
    end

end


"""
    generate_cubemodel(cube_fitter[, aperture])

Generate a CubeModel object corresponding to the options given by the CubeFitter object
"""
function generate_cubemodel(cube_fitter::CubeFitter, aperture::Bool=false)
    shape = aperture ? (1,1,size(cube_fitter.cube.I, 3)) : size(cube_fitter.cube.I)
    # Full 3D intensity model array
    @debug "Generating full 3D cube models"
    if cube_fitter.spectral_region == :MIR
        arguments = [shape, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.names,
        cube_fitter.abs_features.names, cube_fitter.template_names, cube_fitter.lines.names]
    elseif cube_fitter.spectral_region == :OPT
        arguments = [shape, cube_fitter.n_ssps, cube_fitter.n_power_law, cube_fitter.lines.names]
    end
    cubemodel_empty(arguments...)
end


"""
    generate_parammaps(cube_fitter[, aperture])

Generate three ParamMaps objects (for the values and upper/lower errors) corrresponding to the options given
by the CubeFitter object.
"""
function generate_parammaps(cube_fitter::CubeFitter, aperture::Bool=false)
    shape = aperture ? (1,1,size(cube_fitter.cube.I, 3)) : size(cube_fitter.cube.I)
    # 2D maps of fitting parameters
    @debug "Generating 2D parameter value & error maps"
    if cube_fitter.spectral_region == :MIR
        arguments = [shape, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features, 
            cube_fitter.abs_features.names, cube_fitter.template_names, cube_fitter.n_lines, cube_fitter.n_comps, 
            cube_fitter.lines, cube_fitter.flexible_wavesol]
    elseif cube_fitter.spectral_region == :OPT
        arguments = [shape, cube_fitter.n_ssps, cube_fitter.n_power_law, cube_fitter.n_lines, cube_fitter.n_comps, 
            cube_fitter.lines, cube_fitter.flexible_wavesol]
    end
    param_maps = parammaps_empty(arguments...)
    # 2D maps of fitting parameter +/- 1 sigma errors
    param_errs_lo = parammaps_empty(arguments...)
    param_errs_up = parammaps_empty(arguments...)
    param_errs = [param_errs_lo, param_errs_up]
    param_maps, param_errs
end


"""
    get_continuum_plimits(cube_fitter, spaxel, λ, I, σ, init; split)

Get the continuum limits vector for a given CubeFitter object, possibly split up by the 2 continuum fitting steps.
Also returns a boolean vector for which parameters are allowed to vary.
"""
get_continuum_plimits(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real}, σ::Vector{<:Real},
    init::Bool, templates_spax::Matrix{<:Real}; split::Bool=false) = cube_fitter.spectral_region == :MIR ? 
    get_mir_continuum_plimits(cube_fitter, spaxel, I, σ, init, templates_spax; split=split) : 
    get_opt_continuum_plimits(cube_fitter, λ, I, init)


# MIR implementation of the get_continuum_plimits function
function get_mir_continuum_plimits(cube_fitter::CubeFitter, spaxel::CartesianIndex, I::Vector{<:Real}, σ::Vector{<:Real}, 
    init::Bool, templates_spax::Matrix{<:Real}; split::Bool=false)

    dust_features = cube_fitter.dust_features
    abs_features = cube_fitter.abs_features
    abs_taus = cube_fitter.abs_taus
    continuum = cube_fitter.continuum

    amp_dc_plim = (0., Inf)
    amp_df_plim = (0., clamp(1 / exp(-continuum.τ_97.limits[2]), 1., Inf))

    stellar_plim = [amp_dc_plim, continuum.T_s.limits]
    stellar_lock = [!cube_fitter.fit_stellar_continuum, continuum.T_s.locked]
    dc_plim = vcat([[amp_dc_plim, Ti.limits] for Ti ∈ continuum.T_dc]...)
    dc_lock = vcat([[false, Ti.locked] for Ti ∈ continuum.T_dc]...)
    pl_plim = vcat([[amp_dc_plim, pl.limits] for pl ∈ continuum.α]...)
    pl_lock = vcat([[false, pl.locked] for pl ∈ continuum.α]...)

    df_plim = Tuple{Float64,Float64}[]
    df_lock = Bool[]
    for n in 1:length(dust_features.names)
        append!(df_plim, [amp_df_plim, dust_features.mean[n].limits, dust_features.fwhm[n].limits])
        append!(df_lock, [false, dust_features.mean[n].locked, dust_features.fwhm[n].locked])
        if dust_features.profiles[n] == :PearsonIV
            append!(df_plim, [dust_features.index[n].limits, dust_features.cutoff[n].limits])
            append!(df_lock, [dust_features.index[n].locked, dust_features.cutoff[n].locked])
        end
    end

    ab_plim = vcat([[tau.limits, mi.limits, fi.limits] for (tau, mi, fi) ∈ zip(abs_taus, abs_features.mean, abs_features.fwhm)]...)
    ab_lock = vcat([[tau.locked, mi.locked, fi.locked] for (tau, mi, fi) ∈ zip(abs_taus, abs_features.mean, abs_features.fwhm)]...)
    ext_plim = [continuum.τ_97.limits, continuum.τ_ice.limits, continuum.τ_ch.limits, continuum.β.limits]
    ext_lock = [continuum.τ_97.locked, continuum.τ_ice.locked, continuum.τ_ch.locked, continuum.β.locked]

    # Lock tau_9.7 if an extinction map has been provided
    if !isnothing(cube_fitter.extinction_map) && !init
        ext_lock[1] = true
    end
    # Also lock if the continuum is within 1 std dev of 0
    if nanmedian(I) ≤ nanmedian(σ)
        ext_lock[1:4] .= true
    end
    # if !init
    #     for t in 1:cube_fitter.n_templates
    #         m = minimum(I .- templates_spax[:, t])
    #         if m < nanmedian(σ)
    #             ext_lock[1:4] .= true
    #             ab_lock .= true
    #         end
    #     end
    # end

    hd_plim = cube_fitter.fit_sil_emission ? [amp_dc_plim, continuum.T_hot.limits, continuum.Cf_hot.limits, 
        continuum.τ_warm.limits, continuum.τ_cold.limits, continuum.sil_peak.limits] : []
    hd_lock = cube_fitter.fit_sil_emission ? [false, continuum.T_hot.locked, continuum.Cf_hot.locked,
        continuum.τ_warm.locked, continuum.τ_cold.locked, continuum.sil_peak.locked] : []
    
    temp_plim = [ta.limits for ta in continuum.temp_amp]
    temp_lock = [ta.locked for ta in continuum.temp_amp]

    if !split
        plims = Vector{Tuple}(vcat(stellar_plim, dc_plim, pl_plim, ext_plim, ab_plim, hd_plim, temp_plim, df_plim))
        lock = BitVector(vcat(stellar_lock, dc_lock, pl_lock, ext_lock, ab_lock, hd_lock, temp_lock, df_lock))
        plims, lock
    else
        # Split up for the two different stages of continuum fitting -- with templates and then with the PAHs
        plims_1 = Vector{Tuple}(vcat(stellar_plim, dc_plim, pl_plim, ext_plim, ab_plim, hd_plim, temp_plim, [amp_df_plim, amp_df_plim]))
        lock_1 = BitVector(vcat(stellar_lock, dc_lock, pl_lock, ext_lock, ab_lock, hd_lock, temp_lock, [false, false]))
        plims_2 = Vector{Tuple}(df_plim)
        lock_2 = BitVector(df_lock)
        plims_1, plims_2, lock_1, lock_2
    end

end


# Optical implementation of the get_continuum_plimits function
function get_opt_continuum_plimits(cube_fitter::CubeFitter, λ::Vector{<:Real}, I::Vector{<:Real}, init::Bool)

    continuum = cube_fitter.continuum

    amp_ssp_plim = (0., Inf)
    amp_pl_plim = (0., Inf)
    ssp_plim = vcat([[amp_ssp_plim, ai.limits, zi.limits] for (ai, zi) in zip(continuum.ssp_ages, continuum.ssp_metallicities)]...)
    if !init
        ssp_locked = vcat([[false, true, true] for _ in 1:cube_fitter.n_ssps]...)
    else
        ssp_locked = vcat([[false, ai.locked, zi.locked] for (ai, zi) in zip(continuum.ssp_ages, continuum.ssp_metallicities)]...)
    end

    stel_kin_plim = [continuum.stel_vel.limits, continuum.stel_vdisp.limits]
    stel_kin_locked = [continuum.stel_vel.locked, continuum.stel_vdisp.locked]

    feii_plim = []
    feii_locked = []
    if cube_fitter.fit_opt_na_feii
        append!(feii_plim, [amp_ssp_plim, continuum.na_feii_vel.limits, continuum.na_feii_vdisp.limits])
        append!(feii_locked, [false, continuum.na_feii_vel.locked, continuum.na_feii_vdisp.locked])
    end
    if cube_fitter.fit_opt_br_feii
        append!(feii_plim, [amp_ssp_plim, continuum.br_feii_vel.limits, continuum.br_feii_vdisp.limits])
        append!(feii_locked, [false, continuum.br_feii_vel.locked, continuum.br_feii_vdisp.locked])
    end
    
    pl_plim = vcat([[amp_pl_plim, αi.limits] for αi in continuum.α]...)
    pl_locked = vcat([[false, αi.locked] for αi in continuum.α]...)

    atten_plim = [continuum.E_BV.limits, continuum.E_BV_factor.limits]
    atten_locked = [continuum.E_BV.locked, continuum.E_BV_factor.locked]

    # test the SNR of the H-beta and H-gamma lines
    Hβ_snr = test_line_snr(0.4862691, 0.0080, λ, I)
    Hγ_snr = test_line_snr(0.4341691, 0.0080, λ, I)
    # if the SNR is less than 3, we cannot constrain E(B-V)_gas, so lock it to 0
    if (Hβ_snr < 3) || (Hγ_snr < 2)
        atten_locked = [true, true]
    end
    # Lock E(B-V) if an extinction map has been provided
    if !isnothing(cube_fitter.extinction_map) && !init
        atten_locked = [true, true]
    end

    if cube_fitter.fit_uv_bump && cube_fitter.extinction_curve == "calzetti"
        push!(atten_plim, continuum.δ_uv.limits)
        push!(atten_locked, continuum.δ_uv.locked)
    end
    if cube_fitter.fit_covering_frac && cube_fitter.extinction_curve == "calzetti"
        push!(atten_plim, continuum.frac.limits)
        push!(atten_locked, continuum.frac.locked)
    end

    plims = Vector{Tuple}(vcat(ssp_plim, stel_kin_plim, atten_plim, feii_plim, pl_plim))
    lock = BitVector(vcat(ssp_locked, stel_kin_locked, atten_locked, feii_locked, pl_locked))

    plims, lock

end


"""
    get_continuum_initial_values(cube_fitter, spaxel, λ, I, σ, N, init; split)

Get the vectors of starting values and relative step sizes for the continuum fit for a given CubeFitter object. 
Again, the vector may be split up by the 2 continuum fitting steps in the MIR case.
"""
get_continuum_initial_values(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real},
    σ::Vector{<:Real}, N::Real, init::Bool, templates_spax::Matrix{<:Real}; split::Bool=false) = cube_fitter.spectral_region == :MIR ? 
    get_mir_continuum_initial_values(cube_fitter, spaxel, λ, I, σ, N, init, templates_spax, split=split) :
    get_opt_continuum_initial_values(cube_fitter, spaxel, λ, I, N, init)


# MIR implementation of the get_continuum_initial_values function
function get_mir_continuum_initial_values(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real}, 
    σ::Vector{<:Real}, N::Real, init::Bool, templates_spax::Matrix{<:Real}; split::Bool=false)

    continuum = cube_fitter.continuum

    # guess optical depth from the dip in the continuum level
    if !isnothing(cube_fitter.guess_tau)
        i1 = nanmedian(I[cube_fitter.guess_tau[1][1] .< λ .< cube_fitter.guess_tau[1][2]])
        i2 = nanmedian(I[cube_fitter.guess_tau[2][1] .< λ .< cube_fitter.guess_tau[2][2]])
        m = (i2 - i1) / (mean(cube_fitter.guess_tau[2]) - mean(cube_fitter.guess_tau[1]))
        contin_unextinct = i1 + m * (10.0 - mean(cube_fitter.guess_tau[1]))  # linear extrapolation over the silicate feature
        contin_extinct = clamp(nanmedian(I[9.9 .< λ .< 10.1]), 0., Inf)
        # Optical depth at 10 microns
        r = contin_extinct / contin_unextinct
        tau_10 = r > 0 ? clamp(-log(r), continuum.τ_97.limits...) : 0.
        if !cube_fitter.extinction_screen && r > 0
            # solve nonlinear equation
            f(τ) = r - (1 - exp(-τ[1]))/τ[1]
            try
                soln = nlsolve(f, [tau_10])
                tau_10 = clamp(soln.zero[1], continuum.τ_97.limits...)
            catch
                tau_10 = 0.
            end
        end

        pₑ = 3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law
        # Get the extinction curve
        β = init ? continuum.β.value : cube_fitter.p_init_cont[pₑ+3]
        if cube_fitter.extinction_curve == "d+"
            ext_10 = τ_dp([10.0], β)[1]
        elseif cube_fitter.extinction_curve == "kvt"
            ext_10 = τ_kvt([10.0], β)[1]
        elseif cube_fitter.extinction_curve == "ct"
            ext_10 = τ_ct([10.0])[1]
        elseif cube_fitter.extinction_curve == "ohm"
            ext_10 = τ_ohm([10.0])[1]
        else
            error("Unrecognized extinction curve: $(cube_fitter.extinction_curve)")
        end
        # Convert tau at 10 microns to tau at 9.7 microns
        tau_guess = clamp(tau_10 / ext_10, continuum.τ_97.limits...)
    end

    # Check if cube fitter has initial cube
    if !isnothing(cube_fitter.p_init_cube_λ) && !init

        pcube_cont = cube_fitter.p_init_cube_cont
        # Get the coordinates of all spaxels that have fit results
        coords0 = [float.(c.I) for c in CartesianIndices(size(pcube_cont)[1:2]) if !all(isnan.(pcube_cont[c,:]))]
        coords = cube_fitter.p_init_cube_coords
        # Calculate their distances from the current spaxel
        dist = [hypot(spaxel[1]-c[1], spaxel[2]-c[2]) for c in coords]
        closest = coords0[argmin(dist)]
        @debug "Using initial best fit continuum parameters from coordinates $closest --> $(coords[argmin(dist)])"

        p₀ = pcube_cont[Int.(closest)..., :]
        pᵢ = 3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law + 4 + 3cube_fitter.n_abs_feat + (cube_fitter.fit_sil_emission ? 6 : 0) + cube_fitter.n_templates
        pahtemp = model_pah_residuals(cube_fitter.cube.λ, p₀[pᵢ:end], cube_fitter.dust_features.profiles, ones(length(cube_fitter.cube.λ)))
        pah_frac = repeat([maximum(pahtemp)/2], 2)

    # Check if the cube fitter has initial fit parameters 
    elseif !init

        @debug "Using initial best fit continuum parameters..."

        # Set the parameters to the best parameters
        p₀ = copy(cube_fitter.p_init_cont)
        pah_frac = copy(cube_fitter.p_init_pahtemp)

        # τ_97_0 = cube_fitter.τ_guess[parse(Int, cube_fitter.cube.channel)][spaxel]
        # max_τ = cube_fitter.continuum.τ_97.limits[2]

        # scale all flux amplitudes by the difference in medians between the spaxel and the summed spaxels
        # (should be close to 1 since the sum is already normalized by the number of spaxels included anyways)
        I_init = sumdim(cube_fitter.cube.I, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
        scale = max(nanmedian(I), 1e-10) * N / nanmedian(I_init)
        # max_amp = 1 / exp(-max_τ)

        # Stellar amplitude (rescaled)
        p₀[1] = p₀[1] * scale 
        pᵢ = 3

        # Dust continuum amplitudes (rescaled)
        for _ ∈ 1:cube_fitter.n_dust_cont
            p₀[pᵢ] = p₀[pᵢ] * scale 
            pᵢ += 2
        end

        # Power law amplitudes (NOT rescaled)
        for _ ∈ 1:cube_fitter.n_power_law
            # p₀[pᵢ] = p₀[pᵢ] 
            pᵢ += 2
        end

        # Set optical depth based on the initial guess or the initial fit (whichever is larger)
        p₀[pᵢ] = max(cube_fitter.continuum.τ_97.value, p₀[pᵢ])

        # Set τ_9.7 and τ_CH to 0 if the continuum is within 1 std dev of 0
        lock_abs = false
        if nanmedian(I) ≤ nanmedian(σ)
            p₀[pᵢ] = 0.
            p₀[pᵢ+2] = 0.
            lock_abs = true
        end
        # for t in 1:cube_fitter.n_templates
        #     m = minimum(I .- templates_spax[:, t])
        #     if m < nanmedian(σ)
        #         p₀[pᵢ] = 0.
        #         p₀[pᵢ+2] = 0.
        #         lock_abs = true
        #     end
        # end

        # Set τ_9.7 to the guess if the guess_tau flag is set
        if !isnothing(cube_fitter.guess_tau)
            p₀[pᵢ] = tau_guess
        end

        # Override if an extinction_map was provided
        if !isnothing(cube_fitter.extinction_map)
            @debug "Using the provided τ_9.7 values from the extinction_map and rescaling starting point"
            if !isnothing(cube_fitter.cube.voronoi_bins)
                data_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(spaxel)[1])
                p₀[pᵢ] = mean(cube_fitter.extinction_map[data_indices])
            else
                data_index = spaxel
                p₀[pᵢ] = cube_fitter.extinction_map[data_index]
            end
        end

        # Do not adjust absorption feature amplitudes since they are multiplicative
        pᵢ += 4
        for _ ∈ 1:cube_fitter.n_abs_feat
            if lock_abs
                p₀[pᵢ] = 0.
            end
            pᵢ += 3
        end

        # Hot dust amplitude (rescaled)
        if cube_fitter.fit_sil_emission
            p₀[pᵢ] *= scale
            pᵢ += 6
        end

        # Template amplitudes (not rescaled)
        for n ∈ 1:cube_fitter.n_templates
            p₀[pᵢ] = 1.0
            pᵢ += 1
        end

        # Dust feature amplitudes (not rescaled)
        # for i ∈ 1:cube_fitter.n_dust_feat
        #     pᵢ += 3
        #     if cube_fitter.dust_features.profiles[i] == :PearsonIV
        #         pᵢ += 2
        #     end
        # end

    # Otherwise, we estimate the initial parameters based on the data
    else

        @debug "Calculating initial starting points..."
        cubic_spline = Spline1D(λ, I, k=3)

        temp_pars = [ta.value for ta in continuum.temp_amp]

        # Stellar amplitude
        λ_s = minimum(λ) < 5 ? minimum(λ)+0.1 : 5.1
        A_s = clamp(cubic_spline(λ_s) * N / Blackbody_ν(λ_s, continuum.T_s.value), 0., Inf)
        if !cube_fitter.fit_stellar_continuum
            A_s = 0.
        end
        A_s *= (length(temp_pars) > 0) ? 0.5 : 1.0

        # Dust feature amplitudes
        A_df = repeat([clamp(nanmedian(I)/2, 0., Inf)], cube_fitter.n_dust_feat)
        # PAH templates
        pah_frac = repeat([clamp(nanmedian(I)/2, 0., Inf)], 2)

        # Absorption feature amplitudes
        A_ab = [tau.value for tau ∈ cube_fitter.abs_taus]

        # Dust continuum amplitudes
        λ_dc = clamp.([Wein(Ti.value) for Ti ∈ continuum.T_dc], minimum(λ), maximum(λ))
        A_dc = clamp.([cubic_spline(λ_dci) * N / Blackbody_ν(λ_dci, T_dci.value) for (λ_dci, T_dci) ∈ 
            zip(λ_dc, continuum.T_dc)] .* (λ_dc ./ 9.7).^2 ./ (cube_fitter.n_dust_cont / 2), 0., Inf)
        A_dc .*= (length(temp_pars) > 0) ? 0.5 : 1.0
        
        # Power law amplitudes
        A_pl = [clamp(nanmedian(I), 0., Inf)/exp(-continuum.τ_97.value)/cube_fitter.n_power_law for αi ∈ continuum.α]
        A_pl .*= (length(temp_pars) > 0) ? 0.5 : 1.0
        
        # Hot dust amplitude
        hd = silicate_emission(λ, 1.0, continuum.T_hot.value, continuum.Cf_hot.value, continuum.τ_warm.value, 
            continuum.τ_cold.value, continuum.sil_peak.value)
        mhd = argmax(hd)
        A_hd = clamp(cubic_spline(λ[mhd]) * N / hd[mhd] / 5, 0., Inf)
        A_hd *= (length(temp_pars) > 0) ? 0.5 : 1.0

        stellar_pars = [A_s, continuum.T_s.value]
        dc_pars = vcat([[Ai, Ti.value] for (Ai, Ti) ∈ zip(A_dc, continuum.T_dc)]...)
        pl_pars = vcat([[Ai, αi.value] for (Ai, αi) ∈ zip(A_pl, continuum.α)]...)
        
        df_pars = Float64[]
        for n in 1:length(cube_fitter.dust_features.names)
            append!(df_pars, [A_df[n], cube_fitter.dust_features.mean[n].value, cube_fitter.dust_features.fwhm[n].value])
            if cube_fitter.dust_features.profiles[n] == :PearsonIV
                append!(df_pars, [cube_fitter.dust_features.index[n].value, cube_fitter.dust_features.cutoff[n].value])
            end
        end
        
        ab_pars = vcat([[Ai, mi.value, fi.value] for (Ai, mi, fi) ∈ zip(A_ab, cube_fitter.abs_features.mean, cube_fitter.abs_features.fwhm)]...)
        if cube_fitter.fit_sil_emission
            hd_pars = [A_hd, continuum.T_hot.value, continuum.Cf_hot.value, continuum.τ_warm.value, continuum.τ_cold.value,
                continuum.sil_peak.value]
        else
            hd_pars = []
        end

        extinction_pars = [continuum.τ_97.value, continuum.τ_ice.value, continuum.τ_ch.value, continuum.β.value]
        if !isnothing(cube_fitter.guess_tau)
            extinction_pars[1] = tau_guess
        end

        # Initial parameter vector
        p₀ = Vector{Float64}(vcat(stellar_pars, dc_pars, pl_pars, extinction_pars, ab_pars, hd_pars, temp_pars, df_pars))

    end

    @debug "Continuum Parameter labels: \n [stellar_amp, stellar_temp, " * 
        join(["dust_continuum_amp_$i, dust_continuum_temp_$i" for i ∈ 1:cube_fitter.n_dust_cont], ", ") * 
        join(["power_law_amp_$i, power_law_index_$i" for i ∈ 1:cube_fitter.n_power_law], ", ") *
        ", extinction_tau_97, extinction_tau_ice, extinction_tau_ch, extinction_beta, " *  
        join(["$(ab)_tau, $(ab)_mean, $(ab)_fwhm" for ab ∈ cube_fitter.abs_features.names], ", ") *
        (cube_fitter.fit_sil_emission ? ", hot_dust_amp, hot_dust_temp, hot_dust_covering_frac, hot_dust_tau_warm, hot_dust_tau_cold, hot_dust_sil_peak, " : ", ") *
        join(["$(tp)_amp" for tp ∈ cube_fitter.template_names], ", ") *
        join(["$(df)_amp, $(df)_mean, $(df)_fwhm" * (cube_fitter.dust_features.profiles[n] == :PearsonIV ? ", $(df)_index, $(df)_cutoff" : "") for 
            (n, df) ∈ enumerate(cube_fitter.dust_features.names)], ", ") * "]"
        
    @debug "Continuum Starting Values: \n $p₀"

    # Calculate relative step sizes for finite difference derivatives
    dλ = (λ[end] - λ[1]) / length(λ)
    deps = sqrt(eps())

    stellar_dstep = [deps, 1e-4]
    dc_dstep = vcat([[deps, 1e-4] for _ in continuum.T_dc]...)
    pl_dstep = vcat([[deps, deps] for _ in continuum.α]...)
    df_dstep = Float64[]
    for n in 1:length(cube_fitter.dust_features.names)
        append!(df_dstep, [deps, dλ/10/cube_fitter.dust_features.mean[n].value, dλ/1000/cube_fitter.dust_features.fwhm[n].value])
        if cube_fitter.dust_features.profiles[n] == :PearsonIV
            append!(df_dstep, [deps, deps])
        end
    end
    ab_dstep = vcat([[deps, dλ/10/mi.value, dλ/1000/fi.value] for (mi, fi) in zip(cube_fitter.abs_features.mean, cube_fitter.abs_features.fwhm)]...)
    if cube_fitter.fit_sil_emission
        hd_dstep = [deps, 1e-4, deps, deps, deps, dλ/10/continuum.sil_peak.value]
    else
        hd_dstep = []
    end
    extinction_dstep = [deps, deps, deps, deps]
    temp_dstep = [deps for _ in 1:cube_fitter.n_templates]
    dstep = Vector{Float64}(vcat(stellar_dstep, dc_dstep, pl_dstep, extinction_dstep, ab_dstep, hd_dstep, temp_dstep, df_dstep))

    @debug "Continuum relative step sizes: \n $dstep"

    if !split
        p₀, dstep
    else
        # Step 1: Stellar + Dust blackbodies, 2 new amplitudes for the PAH templates, and the extinction parameters
        pars_1 = vcat(p₀[1:(2+2*cube_fitter.n_dust_cont+2*cube_fitter.n_power_law+4+3*cube_fitter.n_abs_feat+(cube_fitter.fit_sil_emission ? 6 : 0))+cube_fitter.n_templates], pah_frac)
        dstep_1 = vcat(dstep[1:(2+2*cube_fitter.n_dust_cont+2*cube_fitter.n_power_law+4+3*cube_fitter.n_abs_feat+(cube_fitter.fit_sil_emission ? 6 : 0))+cube_fitter.n_templates], [deps, deps])
        # Step 2: The PAH profile amplitudes, centers, and FWHMs
        pars_2 = p₀[(3+2*cube_fitter.n_dust_cont+2*cube_fitter.n_power_law+4+3*cube_fitter.n_abs_feat+(cube_fitter.fit_sil_emission ? 6 : 0)+cube_fitter.n_templates):end]
        dstep_2 = dstep[(3+2*cube_fitter.n_dust_cont+2*cube_fitter.n_power_law+4+3*cube_fitter.n_abs_feat+(cube_fitter.fit_sil_emission ? 6 : 0)+cube_fitter.n_templates):end]

        pars_1, pars_2, dstep_1, dstep_2
    end
end


# Optical implementation of the get_continuum_initial_values function
function get_opt_continuum_initial_values(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real}, 
    N::Real, init::Bool)

    continuum = cube_fitter.continuum

    # Check if the cube fitter has initial fit parameters 
    if !init

        @debug "Using initial best fit continuum parameters..."

        # Set the parameters to the best parameters
        p₀ = copy(cube_fitter.p_init_cont)

        # scale all flux amplitudes by the difference in medians between the spaxel and the summed spaxels
        I_init = sumdim(cube_fitter.cube.I, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
        N0 = Float64(abs(maximum(I_init[isfinite.(I_init)])))
        N0 = N0 ≠ 0. ? N0 : 1.
        # Here we use the normalization from the initial combined intensity because the amplitudes we are rescaling
        # are normalized with respect to N0, not N. This is different from the MIR case where the amplitudes are not
        # normalized to any particular N (at least for the blackbody components).
        scale = max(nanmedian(I), 1e-10) * N0 / nanmedian(I_init)

        if !isnothing(cube_fitter.extinction_map)
            pₑ = 1 + 3cube_fitter.n_ssps + 2
            ebv_orig = p₀[pₑ]
            ebv_factor = p₀[pₑ+1]

            @debug "Using the provided E(B-V) values from the extinction_map and rescaling starting point"
            if !isnothing(cube_fitter.cube.voronoi_bins)
                data_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(spaxel)[1])
                ebv_new = mean(cube_fitter.extinction_map[data_indices])
            else
                data_index = spaxel
                ebv_new = cube_fitter.extinction_map[data_index]
            end
            ebv_factor_new = continuum.E_BV_factor.value

            # Rescale to keep the continuum at a good starting point
            if cube_fitter.extinction_curve == "ccm"
                scale /= median(attenuation_cardelli(λ, ebv_new*ebv_factor_new) ./ attenuation_cardelli(λ, ebv_orig*ebv_factor))
            elseif cube_fitter.extinction_curve == "calzetti"
                scale /= median(attenuation_calzetti(λ, ebv_new*ebv_factor_new) ./ attenuation_calzetti(λ, ebv_orig*ebv_factor))
            else
                error("Unrecognized extinction curve $(cube_fitter.extinction_curve)")
            end

            # Set the new values
            p₀[pₑ] = ebv_new
            p₀[pₑ+1] = ebv_factor_new
        end

        # SSP amplitudes
        pᵢ = 1
        for _ ∈ 1:cube_fitter.n_ssps
            p₀[pᵢ] *= scale
            pᵢ += 3
        end

        # If stellar velocities hit any limits, reset them to sensible starting values
        if (p₀[pᵢ] == continuum.stel_vel.limits[1]) || (p₀[pᵢ] == continuum.stel_vel.limits[2])
            p₀[pᵢ] = 0.
        end
        if (p₀[pᵢ+1] == continuum.stel_vdisp.limits[1]) || (p₀[pᵢ+1] == continuum.stel_vdisp.limits[2])
            p₀[pᵢ+1] = 100.
        end
        pᵢ += 2

        # test the SNR of the H-beta line
        Hβ_snr = test_line_snr(0.4862691, 0.0060, λ, I)
        Hγ_snr = test_line_snr(0.4341691, 0.0080, λ, I)
        # if the SNR is less than 3, we cannot constrain E(B-V), so lock it to 0
        if ((Hβ_snr < 3) || (Hγ_snr < 2)) && isnothing(cube_fitter.extinction_map)
            @debug "Locking E(B-V) to 0 due to insufficient H-beta emission"
            p₀[pᵢ] = 0.
            p₀[pᵢ+1] = continuum.E_BV_factor.value
        end

        if cube_fitter.fit_uv_bump && cube_fitter.extinction_curve == "calzetti"
            pᵢ += 1
        end
        if cube_fitter.fit_covering_frac && cube_fitter.extinction_curve == "calzetti"
            pᵢ += 1
        end
        pᵢ += 2

        # Fe II amplitudes
        if cube_fitter.fit_opt_na_feii
            p₀[pᵢ] *= scale
            pᵢ += 3
        end
        if cube_fitter.fit_opt_br_feii
            p₀[pᵢ] *= scale
            pᵢ += 3
        end

        # Power law amplitudes
        for _ ∈ 1:cube_fitter.n_power_law
            p₀[pᵢ] *= scale
            pᵢ += 2
        end


    else

        @debug "Calculating initial starting points..." 

        if cube_fitter.extinction_curve == "ccm"
            att = attenuation_cardelli([median(λ)], continuum.E_BV.value)[1]
        elseif cube_fitter.extinction_curve == "calzetti"
            att = attenuation_calzetti([median(λ)], continuum.E_BV.value)[1]
        else
            error("Uncrecognized extinction curve $(cube_fitter.extinction_curve)")
        end

        # SSP amplitudes
        m_ssps = zeros(cube_fitter.n_ssps)
        for i in 1:cube_fitter.n_ssps
            m_ssps[i] = nanmedian(I) / att / cube_fitter.n_ssps
        end

        ssp_pars = vcat([[mi, ai.value, zi.value] for (mi, ai, zi) in zip(m_ssps, continuum.ssp_ages, continuum.ssp_metallicities)]...)

        # Stellar kinematics
        stel_kin_pars = [continuum.stel_vel.value, continuum.stel_vdisp.value]

        # Fe II parameters
        a_feii = 0.1 * nanmedian(I) / att
        feii_pars = []
        if cube_fitter.fit_opt_na_feii
            append!(feii_pars, [a_feii, continuum.na_feii_vel.value, continuum.na_feii_vdisp.value])
        end
        if cube_fitter.fit_opt_br_feii
            append!(feii_pars, [a_feii, continuum.br_feii_vel.value, continuum.br_feii_vdisp.value])
        end
        
        # Power laws
        a_pl = 0.5 * nanmedian(I) / att / cube_fitter.n_power_law
        pl_pars = vcat([[a_pl, αi.value] for αi in continuum.α]...)

        # Attenuation
        atten_pars = [continuum.E_BV.value, continuum.E_BV_factor.value]
        if cube_fitter.fit_uv_bump && cube_fitter.extinction_curve == "calzetti"
            push!(atten_pars, continuum.δ_uv.value)
        end
        if cube_fitter.fit_covering_frac && cube_fitter.extinction_curve == "calzetti"
            push!(atten_pars, continuum.frac.value)
        end

        # Initial parameter vector
        p₀ = Vector{Float64}(vcat(ssp_pars, stel_kin_pars, atten_pars, feii_pars, pl_pars))

    end

    # Calculate relative step sizes for finite difference derivatives
    deps = sqrt(eps())

    ssp_dstep = vcat([[deps, deps, deps] for _ in continuum.ssp_ages]...)
    stel_kin_dstep = [1e-4, 1e-4]
    feii_dstep = []
    if cube_fitter.fit_opt_na_feii
        append!(feii_dstep, [deps, 1e-4, 1e-4])
    end
    if cube_fitter.fit_opt_br_feii
        append!(feii_dstep, [deps, 1e-4, 1e-4])
    end
    pl_dstep = vcat([[deps, deps] for _ in continuum.α]...)

    atten_dstep = [deps, deps]
    if cube_fitter.fit_uv_bump && cube_fitter.extinction_curve == "calzetti"
        push!(atten_dstep, deps)
    end
    if cube_fitter.fit_covering_frac && cube_fitter.extinction_curve == "calzetti"
        push!(atten_dstep, deps)
    end

    dstep = Vector{Float64}(vcat(ssp_dstep, stel_kin_dstep, atten_dstep, feii_dstep, pl_dstep))

    @debug "Continuum Parameter labels: \n [" *
        join(["SSP_$(i)_mass, SSP_$(i)_age, SSP_$(i)_metallicity" for i in 1:cube_fitter.n_ssps], ", ") * 
        "stel_vel, stel_vdisp, " * 
        "E_BV, E_BV_factor, " * (cube_fitter.fit_uv_bump ? "delta_uv, " : "") *
        (cube_fitter.fit_covering_frac ? "covering_frac, " : "") * 
        (cube_fitter.fit_opt_na_feii ? "na_feii_amp, na_feii_vel, na_feii_vdisp, " : "") *
        (cube_fitter.fit_opt_br_feii ? "br_feii_amp, br_feii_vel, br_feii_vdisp, " : "") *
        join(["power_law_$(j)_amp, power_law_$(j)_index, " for j in 1:cube_fitter.n_power_law], ", ") * "]"
        
    @debug "Continuum Starting Values: \n $p₀"
    @debug "Continuum relative step sizes: \n $dstep"

    p₀, dstep

end


"""
    get_continuum_parinfo(n_free, lb, ub, dp)

Get the CMPFit parinfo and config objects for a given CubeFitter object, given the vector of initial valuels,
limits, and relative step sizes.
"""
function get_continuum_parinfo(n_free::S, lb::Vector{T}, ub::Vector{T}, dp::Vector{T}) where {S<:Integer,T<:Real}

    parinfo = CMPFit.Parinfo(n_free)

    for pᵢ ∈ 1:n_free
        parinfo[pᵢ].fixed = 0
        parinfo[pᵢ].limited = (1,1)
        parinfo[pᵢ].limits = (lb[pᵢ], ub[pᵢ])
        # Set the relative step size for finite difference derivative calculations
        parinfo[pᵢ].relstep = dp[pᵢ]
    end

    # Create a `config` structure
    config = CMPFit.Config()
    config.maxiter = 500

    parinfo, config

end


# Version for the split fitting if use_pah_templates is enabled
function get_continuum_parinfo(n_free_1::S, n_free_2::S, lb_1::Vector{T}, ub_1::Vector{T}, 
    lb_2::Vector{T}, ub_2::Vector{T}, dp_1::Vector{T}, dp_2::Vector{T}) where {S<:Integer,T<:Real}

    parinfo_1 = CMPFit.Parinfo(n_free_1)
    parinfo_2 = CMPFit.Parinfo(n_free_2)

    for pᵢ ∈ 1:n_free_1
        parinfo_1[pᵢ].fixed = 0
        parinfo_1[pᵢ].limited = (1,1)
        parinfo_1[pᵢ].limits = (lb_1[pᵢ], ub_1[pᵢ])
        parinfo_1[pᵢ].relstep = dp_1[pᵢ]
    end

    for pᵢ ∈ 1:n_free_2
        parinfo_2[pᵢ].fixed = 0
        parinfo_2[pᵢ].limited = (1,1)
        parinfo_2[pᵢ].limits = (lb_2[pᵢ], ub_2[pᵢ])
        parinfo_2[pᵢ].relstep = dp_2[pᵢ]
    end

    # Create a `config` structure
    config = CMPFit.Config()
    config.maxiter = 500

    parinfo_1, parinfo_2, config

end


"""
    pretty_print_continuum_results(cube_fitter, popt, perr, I)

Print out a nicely formatted summary of the continuum fit results for a given CubeFitter object.
"""
pretty_print_continuum_results(cube_fitter::CubeFitter, popt::Vector{<:Real}, perr::Vector{<:Real},
    I::Vector{<:Real}) = cube_fitter.spectral_region == :MIR ? 
        pretty_print_mir_continuum_results(cube_fitter, popt, perr, I) :
        pretty_print_opt_continuum_results(cube_fitter, popt, perr, I)


# MIR implementation of the pretty_print_continuum_results function
function pretty_print_mir_continuum_results(cube_fitter::CubeFitter, popt::Vector{<:Real}, perr::Vector{<:Real},
    I::Vector{<:Real})

    continuum = cube_fitter.continuum

    msg = "######################################################################\n"
    msg *= "################# SPAXEL FIT RESULTS -- CONTINUUM ####################\n"
    msg *= "######################################################################\n"
    msg *= "\n#> STELLAR CONTINUUM <#\n"
    msg *= "Stellar_amp: \t\t\t $(@sprintf "%.3g" popt[1]) +/- $(@sprintf "%.3g" perr[1]) [-] \t Limits: (0, Inf)\n"
    msg *= "Stellar_temp: \t\t\t $(@sprintf "%.0f" popt[2]) +/- $(@sprintf "%.3e" perr[2]) K \t (fixed)\n"
    pᵢ = 3
    msg *= "\n#> DUST CONTINUUM <#\n"
    for i ∈ 1:cube_fitter.n_dust_cont
        msg *= "Dust_continuum_$(i)_amp: \t\t $(@sprintf "%.3g" popt[pᵢ]) +/- $(@sprintf "%.3g" perr[pᵢ]) [-] \t Limits: (0, Inf)\n"
        msg *= "Dust_continuum_$(i)_temp: \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.3e" perr[pᵢ+1]) K \t\t\t (fixed)\n"
        msg *= "\n"
        pᵢ += 2
    end
    msg *= "\n#> POWER LAWS <#\n"
    for k ∈ 1:cube_fitter.n_power_law
        msg *= "Power_law_$(k)_amp: \t\t $(@sprintf "%.3g" popt[pᵢ]) +/- $(@sprintf "%.3g" perr[pᵢ]) [x norm] \t Limits: (0, Inf)\n"
        msg *= "Power_law_$(k)_index: \t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1]) [-] \t Limits: " *
            "($(@sprintf "%.3f" continuum.α[k].limits[1]), $(@sprintf "%.3f" continuum.α[k].limits[2]))" *
            (continuum.α[k].locked ? " (fixed)" : "") * "\n"
        pᵢ += 2
    end
    msg *= "\n#> EXTINCTION <#\n"
    msg *= "τ_9.7: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.τ_97.limits[1]), $(@sprintf "%.2f" continuum.τ_97.limits[2]))" * 
        (continuum.τ_97.locked ? " (fixed)" : "") * "\n"
    msg *= "τ_ice: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+1]) +/- $(@sprintf "%.2f" perr[pᵢ+1]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.τ_ice.limits[1]), $(@sprintf "%.2f" continuum.τ_ice.limits[2]))" *
        (continuum.τ_ice.locked ? " (fixed)" : "") * "\n"
    msg *= "τ_ch: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+2]) +/- $(@sprintf "%.2f" perr[pᵢ+2]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.τ_ch.limits[1]), $(@sprintf "%.2f" continuum.τ_ch.limits[2]))" *
        (continuum.τ_ch.locked ? " (fixed)" : "") * "\n"
    msg *= "β: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+3]) +/- $(@sprintf "%.2f" perr[pᵢ+3]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.β.limits[1]), $(@sprintf "%.2f" continuum.β.limits[2]))" * 
        (continuum.β.locked ? " (fixed)" : "") * "\n"
    msg *= "\n"
    pᵢ += 4
    msg *= "\n#> ABSORPTION FEATURES <#\n"
    for (j, ab) ∈ enumerate(cube_fitter.abs_features.names)
        msg *= "$(ab)_τ:\t\t\t $(@sprintf "%.5f" popt[pᵢ]) +/- $(@sprintf "%.5f" perr[pᵢ]) [x norm] \t Limits: " *
            "($(@sprintf "%.3f" cube_fitter.abs_taus[j].limits[1]), $(@sprintf "%.3f" cube_fitter.abs_taus[j].limits[2]))\n"
        msg *= "$(ab)_mean:  \t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1]) μm \t Limits: " *
            "($(@sprintf "%.3f" cube_fitter.abs_features.mean[j].limits[1]), $(@sprintf "%.3f" cube_fitter.abs_features.mean[j].limits[2]))" * 
            (cube_fitter.abs_features.mean[j].locked ? " (fixed)" : "") * "\n"
        msg *= "$(ab)_fwhm:  \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) μm \t Limits: " *
            "($(@sprintf "%.3f" cube_fitter.abs_features.fwhm[j].limits[1]), $(@sprintf "%.3f" cube_fitter.abs_features.fwhm[j].limits[2]))" * 
            (cube_fitter.abs_features.fwhm[j].locked ? " (fixed)" : "") * "\n"
        msg *= "\n"
        pᵢ += 3
    end 
    if cube_fitter.fit_sil_emission
        msg *= "\n#> HOT DUST <#\n"
        msg *= "Hot_dust_amp: \t\t\t $(@sprintf "%.3g" popt[pᵢ]) +/- $(@sprintf "%.3g" perr[pᵢ]) [-] \t Limits: (0, Inf)\n"
        msg *= "Hot_dust_temp: \t\t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) K \t Limits: " *
            "($(@sprintf "%.0f" continuum.T_hot.limits[1]), $(@sprintf "%.0f" continuum.T_hot.limits[2]))" *
            (continuum.T_hot.locked ? " (fixed)" : "") * "\n"
        msg *= "Hot_dust_frac: \t\t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) [-] \t Limits: " *
            "($(@sprintf "%.3f" continuum.Cf_hot.limits[1]), $(@sprintf "%.3f" continuum.Cf_hot.limits[2]))" *
            (continuum.Cf_hot.locked ? " (fixed)" : "") * "\n"
        msg *= "Hot_dust_τ: \t\t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3]) [-] \t Limits: " *
            "($(@sprintf "%.3f" continuum.τ_warm.limits[1]), $(@sprintf "%.3f" continuum.τ_warm.limits[2]))" *
            (continuum.τ_warm.locked ? " (fixed)" : "") * "\n"
        msg *= "Cold_dust_τ: \t\t\t $(@sprintf "%.3f" popt[pᵢ+4]) +/- $(@sprintf "%.3f" perr[pᵢ+4]) [-] \t Limits: " *
            "($(@sprintf "%.3f" continuum.τ_cold.limits[1]), $(@sprintf "%.3f" continuum.τ_cold.limits[2]))" *
            (continuum.τ_cold.locked ? " (fixed)" : "") * "\n"
        msg *= "Hot_dust_peak: \t\t\t $(@sprintf "%.3f" popt[pᵢ+5]) +/- $(@sprintf "%.3f" perr[pᵢ+5]) [-] \t Limits: " *
            "($(@sprintf "%.3f" continuum.sil_peak.limits[1]), $(@sprintf "%.3f" continuum.sil_peak.limits[2]))" *
            (continuum.sil_peak.locked ? " (fixed)" : "") * "\n"
        pᵢ += 6
    end
    msg *= "\n#> TEMPLATES <#\n"
    for (q, tp) ∈ enumerate(cube_fitter.template_names)
        msg *= "$(tp)_amp:\t\t\t $(@sprintf "%.5f" popt[pᵢ]) +/- $(@sprintf "%.5f" perr[pᵢ]) [x norm] \t Limits: (0, 1)"
        pᵢ += 1
    end
    msg *= "\n#> DUST FEATURES <#\n"
    for (j, df) ∈ enumerate(cube_fitter.dust_features.names)
        msg *= "$(df)_amp:\t\t\t $(@sprintf "%.5f" popt[pᵢ]) +/- $(@sprintf "%.5f" perr[pᵢ]) [x norm] \t Limits: " *
            "(0, $(@sprintf "%.5f" (nanmaximum(I) / exp(-continuum.τ_97.limits[1]))))\n"
        msg *= "$(df)_mean:  \t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1]) μm \t Limits: " *
            "($(@sprintf "%.3f" cube_fitter.dust_features.mean[j].limits[1]), $(@sprintf "%.3f" cube_fitter.dust_features.mean[j].limits[2]))" * 
            (cube_fitter.dust_features.mean[j].locked ? " (fixed)" : "") * "\n"
        msg *= "$(df)_fwhm:  \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) μm \t Limits: " *
            "($(@sprintf "%.3f" cube_fitter.dust_features.fwhm[j].limits[1]), $(@sprintf "%.3f" cube_fitter.dust_features.fwhm[j].limits[2]))" * 
            (cube_fitter.dust_features.fwhm[j].locked ? " (fixed)" : "") * "\n"
        if cube_fitter.dust_features.profiles[j] == :PearsonIV
            msg *= "$(df)_index:  \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3]) μm \t Limits: " *
                "($(@sprintf "%.3f" cube_fitter.dust_features.index[j].limits[1]), $(@sprintf "%.3f" cube_fitter.dust_features.index[j].limits[2]))" * 
                (cube_fitter.dust_features.index[j].locked ? " (fixed)" : "") * "\n"
            msg *= "$(df)_cutoff:  \t\t $(@sprintf "%.3f" popt[pᵢ+4]) +/- $(@sprintf "%.3f" perr[pᵢ+4]) μm \t Limits: " *
                "($(@sprintf "%.3f" cube_fitter.dust_features.cutoff[j].limits[1]), $(@sprintf "%.3f" cube_fitter.dust_features.cutoff[j].limits[2]))" * 
                (cube_fitter.dust_features.cutoff[j].locked ? " (fixed)" : "") * "\n"
            pᵢ += 2
        end
        msg *= "\n"
        pᵢ += 3
    end
    msg *= "######################################################################"
    @debug msg

    msg

end


# Optical implementation of the pretty_print_continuum_results function
function pretty_print_opt_continuum_results(cube_fitter::CubeFitter, popt::Vector{<:Real}, perr::Vector{<:Real},
    I::Vector{<:Real})

    continuum = cube_fitter.continuum

    msg = "######################################################################\n"
    msg *= "################# SPAXEL FIT RESULTS -- CONTINUUM ####################\n"
    msg *= "######################################################################\n"
    msg *= "\n#> STELLAR POPULATIONS <#\n"
    pᵢ = 1
    for i ∈ 1:cube_fitter.n_ssps
        msg *= "SSP_$(i)_amp: \t\t\t $(@sprintf "%.3g" popt[pᵢ]) +/- $(@sprintf "%.3g" perr[pᵢ]) [x norm] \t Limits: (0, Inf)\n"
        msg *= "SSP_$(i)_age: \t\t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1]) Gyr \t Limits: " *
            "($(@sprintf "%.3f" continuum.ssp_ages[i].limits[1]), $(@sprintf "%.3f" continuum.ssp_ages[i].limits[2]))" *
            (continuum.ssp_ages[i].locked ? " (fixed)" : "") * "\n"
        msg *= "SSP_$(i)_metallicity: \t\t $(@sprintf "%.2f" popt[pᵢ+2]) +/- $(@sprintf "%.2f" perr[pᵢ+2]) [M/H] \t Limits: " *
            "($(@sprintf "%.2f" continuum.ssp_metallicities[i].limits[1]), $(@sprintf "%.2f" continuum.ssp_metallicities[i].limits[2]))" *
            (continuum.ssp_metallicities[i].locked ? " (fixed)" : "") * "\n"
        pᵢ += 3
        msg *= "\n"
    end
    msg *= "\n#> STELLAR KINEMATICS <#\n"
    msg *= "stel_vel: \t\t\t\t $(@sprintf "%.0f" popt[pᵢ]) +/- $(@sprintf "%.0f" perr[pᵢ]) km/s \t Limits: " *
        "($(@sprintf "%.0f" continuum.stel_vel.limits[1]), $(@sprintf "%.0f" continuum.stel_vel.limits[2]))" * 
        (continuum.stel_vel.locked ? " (fixed)" : "") * "\n"
    msg *= "stel_vdisp: \t\t\t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) km/s \t Limits: " *
        "($(@sprintf "%.0f" continuum.stel_vdisp.limits[1]), $(@sprintf "%.0f" continuum.stel_vdisp.limits[2]))" * 
        (continuum.stel_vdisp.locked ? " (fixed)" : "") * "\n"
    pᵢ += 2
    msg *= "\n#> ATTENUATION <#\n"
    msg *= "E_BV: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.E_BV.limits[1]), $(@sprintf "%.2f" continuum.E_BV.limits[2]))" * 
        (continuum.E_BV.locked ? " (fixed)" : "") * "\n"
    msg *= "E_BV_factor: \t\t\t $(@sprintf "%.2f" popt[pᵢ+1]) +/- $(@sprintf "%.2f" perr[pᵢ+1]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.E_BV_factor.limits[1]), $(@sprintf "%.2f" continuum.E_BV_factor.limits[2]))" *
        (continuum.E_BV_factor.locked ? " (fixed)" : "") * "\n"
    pᵢ += 2
    if cube_fitter.fit_uv_bump && cube_fitter.extinction_curve == "calzetti"
        msg *= "δ_UV: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t Limits: " *
            "($(@sprintf "%.2f" continuum.δ_uv.limits[1]), $(@sprintf "%.2f" continuum.δ_uv.limits[2]))" * 
            (continuum.δ_uv.locked ? " (fixed)" : "") * "\n"
        pᵢ += 1
    end
    if cube_fitter.fit_covering_frac && cube_fitter.extinction_curve == "calzetti"
        msg *= "frac: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t Limits: " *
            "($(@sprintf "%.2f" continuum.frac.limits[1]), $(@sprintf "%.2f" continuum.frac.limits[2]))" * 
            (continuum.frac.locked ? " (fixed)" : "") * "\n"
        pᵢ += 1
    end
    msg *= "\n#> FE II EMISSION <#\n"
    if cube_fitter.fit_opt_na_feii
        msg *= "na_feii_amp: \t\t\t $(@sprintf "%.3g" popt[pᵢ]) +/- $(@sprintf "%.3g" perr[pᵢ]) [x norm] \t Limits: (0, Inf)\n"
        msg *= "na_feii_vel: \t\t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) km/s \t Limits: " *
            "($(@sprintf "%.0f" continuum.na_feii_vel.limits[1]), $(@sprintf "%.0f" continuum.na_feii_vel.limits[2]))" * 
            (continuum.na_feii_vel.locked ? " (fixed)" : "") * "\n"
        msg *= "na_feii_vdisp: \t\t\t $(@sprintf "%.0f" popt[pᵢ+2]) +/- $(@sprintf "%.0f" perr[pᵢ+2]) km/s \t Limits: " *
            "($(@sprintf "%.0f" continuum.na_feii_vdisp.limits[1]), $(@sprintf "%.0f" continuum.na_feii_vdisp.limits[2]))" * 
            (continuum.na_feii_vel.locked ? " (fixed)" : "") * "\n"
        pᵢ += 3
    end
    if cube_fitter.fit_opt_br_feii
        msg *= "br_feii_amp: \t\t\t $(@sprintf "%.3g" popt[pᵢ]) +/- $(@sprintf "%.3g" perr[pᵢ]) [x norm] \t Limits: (0, Inf)\n"
        msg *= "br_feii_vel: \t\t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) km/s \t Limits: " *
            "($(@sprintf "%.0f" continuum.br_feii_vel.limits[1]), $(@sprintf "%.0f" continuum.br_feii_vel.limits[2]))" * 
            (continuum.br_feii_vel.locked ? " (fixed)" : "") * "\n"
        msg *= "br_feii_vdisp: \t\t\t $(@sprintf "%.0f" popt[pᵢ+2]) +/- $(@sprintf "%.0f" perr[pᵢ+2]) km/s \t Limits: " *
            "($(@sprintf "%.0f" continuum.br_feii_vdisp.limits[1]), $(@sprintf "%.0f" continuum.br_feii_vdisp.limits[2]))" * 
            (continuum.br_feii_vel.locked ? " (fixed)" : "") * "\n"
        pᵢ += 3
    end
    msg *= "\n#> POWER LAWS <#\n"
    for j ∈ 1:cube_fitter.n_power_law
        msg *= "PL_$(j)_amp: \t\t\t\t $(@sprintf "%.3g" popt[pᵢ]) +/- $(@sprintf "%.3g" perr[pᵢ]) [x norm] \t Limits: (0, Inf)\n"
        msg *= "PL_$(j)_index: \t\t\t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1]) [-] \t Limits: " *
            "($(@sprintf "%.3f" continuum.α[j].limits[1]), $(@sprintf "%.3f" continuum.α[j].limits[2]))" *
            (continuum.α[j].locked ? " (fixed)" : "") * "\n"
        pᵢ += 2
    end
    msg *= "\n"
    msg *= "######################################################################"
    @debug msg

    msg 

end


"""
    get_line_plimits(cube_fitter, init[, ext_curve])

Get the line limits vector for a given CubeFitter object. Also returns boolean locked values and
names of each parameter as strings.
"""
function get_line_plimits(cube_fitter::CubeFitter, init::Bool, ext_curve::Union{Vector{<:Real},Nothing}=nothing)

    if !isnothing(ext_curve)
        amp_plim = (0., clamp(1 / minimum(ext_curve), 1., Inf))
    else
        if cube_fitter.spectral_region == :MIR
            max_amp = 1 / exp(-cube_fitter.continuum.τ_97.limits[2])
        elseif cube_fitter.extinction_curve == "ccm"
            max_amp = 1 / attenuation_cardelli([cube_fitter.cube.λ[1]], cube_fitter.continuum.E_BV.limits[2])[1]
        elseif cube_fitter.extinction_curve == "calzetti"
            Cf_dust = cube_fitter.fit_covering_frac ? cube_fitter.continuum.frac : 0.
            if cube_fitter.fit_uv_bump 
                max_amp = 1 / attenuation_calzetti([cube_fitter.cube.λ[1]], cube_fitter.continuum.E_BV.limits[2],
                    cube_fitter.continuum.δ_uv, Cf=Cf_dust)[1]
            else
                max_amp = 1 / attenuation_calzetti([cube_fitter.cube.λ[1]], cube_fitter.continuum.E_BV.limits[2],
                    Cf=Cf_dust)[1]
            end
        end 
        amp_plim = (0., clamp(max_amp, 1., Inf))
    end
    ln_plims = Vector{Tuple}()
    ln_lock = BitVector()
    ln_names = Vector{String}()
    
    amp_tied = []
    amp_ratios = []
    voff_tied = []
    fwhm_tied = []
    for j ∈ 1:cube_fitter.n_comps
        append!(amp_tied, [[[] for _ in cube_fitter.tied_kinematics.key_amp[j]]])
        append!(amp_ratios, [[[] for _ in cube_fitter.tied_kinematics.key_amp[j]]])
        append!(voff_tied, [[[] for _ in cube_fitter.tied_kinematics.key_voff[j]]])
        append!(fwhm_tied, [[[] for _ in cube_fitter.tied_kinematics.key_fwhm[j]]])
    end
    η_tied = []
    
    # Loop through each line and append the new components
    ind = 1
    for i ∈ 1:cube_fitter.n_lines
        for j ∈ 1:cube_fitter.n_comps
            if !isnothing(cube_fitter.lines.profiles[i, j])

                # name
                ln_name = string(cube_fitter.lines.names[i]) * "_$(j)"
                amp_ln_plim = isone(j) ? amp_plim : cube_fitter.lines.acomp_amp[i, j-1].limits
                amp_ln_locked = isone(j) ? false : cube_fitter.lines.acomp_amp[i, j-1].locked

                # get the right amp, voff, and fwhm parameters based on if theyre tied or not
                at = vt = ft = false
                ka = kv = kf = nothing
                if !isnothing(cube_fitter.lines.tied_amp[i, j])
                    key_amp = cube_fitter.lines.tied_amp[i, j]
                    ka = findfirst(cube_fitter.tied_kinematics.key_amp[j] .== key_amp)
                    at = true
                end

                if isnothing(cube_fitter.lines.tied_voff[i, j])
                    voff_ln_plim = cube_fitter.lines.voff[i, j].limits
                    voff_ln_locked = cube_fitter.lines.voff[i, j].locked
                    voff_ln_name = "$(ln_name)_voff"
                else
                    key_voff = cube_fitter.lines.tied_voff[i, j]
                    kv = findfirst(cube_fitter.tied_kinematics.key_voff[j] .== key_voff)
                    voff_ln_plim = cube_fitter.tied_kinematics.voff[j][kv].limits
                    voff_ln_locked = cube_fitter.tied_kinematics.voff[j][kv].locked
                    voff_ln_name = "$(key_voff)_$(j)_voff"
                    vt = true
                end
                if isnothing(cube_fitter.lines.tied_fwhm[i, j])
                    fwhm_ln_plim = cube_fitter.lines.fwhm[i, j].limits
                    fwhm_ln_locked = cube_fitter.lines.fwhm[i, j].locked
                    fwhm_ln_name = "$(ln_name)_fwhm"
                else
                    key_fwhm = cube_fitter.lines.tied_fwhm[i, j]
                    kf = findfirst(cube_fitter.tied_kinematics.key_fwhm[j] .== key_fwhm)
                    fwhm_ln_plim = cube_fitter.tied_kinematics.fwhm[j][kf].limits
                    fwhm_ln_locked = cube_fitter.tied_kinematics.fwhm[j][kf].locked
                    fwhm_ln_name = "$(key_fwhm)_$(j)_fwhm"
                    ft = true
                end

                if at
                    append!(amp_tied[j][ka], [ind])
                    append!(amp_ratios[j][ka], [cube_fitter.tied_kinematics.amp[j][ka][cube_fitter.lines.names[i]]])
                end

                # Depending on flexible_wavesol, we need to add 2 voffs instead of 1 voff
                if !isnothing(cube_fitter.lines.tied_voff[i, j]) && cube_fitter.flexible_wavesol && isone(j)
                    append!(ln_plims, [amp_ln_plim, voff_ln_plim, cube_fitter.lines.voff[i, j].limits, fwhm_ln_plim])
                    append!(ln_lock, [amp_ln_locked, voff_ln_locked, cube_fitter.lines.voff[i, j].locked, fwhm_ln_locked])
                    append!(ln_names, ["$(ln_name)_amp", voff_ln_name, "$(ln_name)_voff_indiv", fwhm_ln_name])
                    append!(voff_tied[j][kv], [ind+1])
                    if ft
                        append!(fwhm_tied[j][kf], [ind+3])
                    end
                    ind += 4
                else
                    append!(ln_plims, [amp_ln_plim, voff_ln_plim, fwhm_ln_plim])
                    append!(ln_lock, [amp_ln_locked, voff_ln_locked, fwhm_ln_locked])
                    append!(ln_names, ["$(ln_name)_amp", voff_ln_name, fwhm_ln_name])
                    if vt
                        append!(voff_tied[j][kv], [ind+1])
                    end
                    if ft
                        append!(fwhm_tied[j][kf], [ind+2])
                    end
                    ind += 3
                end

                # check for additional profile parameters
                if cube_fitter.lines.profiles[i, j] == :GaussHermite
                    # add h3 and h4 moments
                    append!(ln_plims, [cube_fitter.lines.h3[i, j].limits, cube_fitter.lines.h4[i, j].limits])
                    append!(ln_lock, [cube_fitter.lines.h3[i, j].locked, cube_fitter.lines.h4[i, j].locked])
                    append!(ln_names, ["$(ln_name)_h3", "$(ln_name)_h4"])
                    ind += 2
                elseif cube_fitter.lines.profiles[i, j] == :Voigt
                    # add voigt mixing parameter, but only if it's not tied
                    if !cube_fitter.tie_voigt_mixing
                        append!(ln_plims, [cube_fitter.lines.η[i, j].limits])
                        append!(ln_lock, [cube_fitter.lines.η[i, j].locked])
                        append!(ln_names, ["$(ln_name)_eta"])
                    else
                        append!(ln_plims, [cube_fitter.voigt_mix_tied.limits])
                        append!(ln_lock, [cube_fitter.voigt_mix_tied.locked])
                        append!(ln_names, ["eta_tied"])
                        append!(η_tied, [ind])
                    end
                    ind += 1
                end
            end
        end
    end

    # Combine all "tied" vectors
    tied = []
    tied_amp_inds = []
    tied_amp_ratios = []
    for j ∈ 1:cube_fitter.n_comps
        for k ∈ 1:length(cube_fitter.tied_kinematics.key_amp[j])
            append!(tied_amp_inds, [tuple(amp_tied[j][k]...)])
            append!(tied_amp_ratios, [tuple(amp_ratios[j][k]...)])
        end
        for k ∈ 1:length(cube_fitter.tied_kinematics.key_voff[j])
            append!(tied, [tuple(voff_tied[j][k]...)])
        end
        for k ∈ 1:length(cube_fitter.tied_kinematics.key_fwhm[j])
            append!(tied, [tuple(fwhm_tied[j][k]...)])
        end
    end
    append!(tied, [tuple(η_tied...)])

    # Convert the tied vectors into tuples for each pair of parameters
    tied_pairs = []
    for group in tied
        if length(group) > 1
            append!(tied_pairs, [(group[1],group[j],1.0) for j in 2:length(group)])
        end
    end
    for groupind in eachindex(tied_amp_inds)
        group = tied_amp_inds[groupind]
        ratio = tied_amp_ratios[groupind]
        if length(group) > 1
            append!(tied_pairs, [(group[1],group[j],ratio[j]/ratio[1]) for j in 2:length(group)])
        end
    end

    # Convert the paired tuples into indices for each tied parameter
    tied_indices = sort([tp[2] for tp in tied_pairs])

    ln_plims, ln_lock, ln_names, tied_pairs, tied_indices

end


"""
    get_line_initial_values(cube_fitter, spaxel, init)

Get the vector of starting values and relative step sizes for the line fit for a given CubeFitter object.
"""
function get_line_initial_values(cube_fitter::CubeFitter, spaxel::CartesianIndex, init::Bool)

    # Check if cube fitter has initial cube
    if !isnothing(cube_fitter.p_init_cube_λ)

        pcube_lines = cube_fitter.p_init_cube_lines
        # Get the coordinates of all spaxels that have fit results
        coords0 = [float.(c.I) for c in CartesianIndices(size(pcube_lines)[1:2]) if !all(isnan.(pcube_lines[c,:]))]
        # Transform to coordinates in the WCS of our current frame
        coords = cube_fitter.p_init_cube_coords
        # Calculate their distances from the current spaxel
        dist = [hypot(spaxel[1]-c[1], spaxel[2]-c[2]) for c in coords]
        closest = coords0[argmin(dist)]
        @debug "Using initial best fit line parameters from coordinates $closest --> $(coords[argmin(dist)])"

        ln_pars = pcube_lines[Int.(closest)..., :]

    # Check if there are previous best fit parameters
    elseif !init

        @debug "Using initial best fit line parameters..."

        # If so, set the parameters to the previous ones
        ln_pars = copy(cube_fitter.p_init_line)

    else

        @debug "Calculating initial starting points..."
        
        # Start the ampltiudes at 1/2 and 1/4 (in normalized units)
        A_ln = ones(cube_fitter.n_lines) .* 0.5

        # Initial parameter vector
        ln_pars = Float64[]
        for i ∈ 1:cube_fitter.n_lines
            for j ∈ 1:cube_fitter.n_comps
                if !isnothing(cube_fitter.lines.profiles[i, j])

                    amp_ln = isone(j) ? A_ln[i] : cube_fitter.lines.acomp_amp[i, j-1].value
                    if !isnothing(cube_fitter.lines.tied_amp[i, j])
                        key_amp = cube_fitter.lines.tied_amp[i, j]
                        ka = findfirst(cube_fitter.tied_kinematics.key_amp[j] .== key_amp)
                        amp_ln *= cube_fitter.tied_kinematics.amp[j][ka][cube_fitter.lines.names[i]]
                    end
                    if isnothing(cube_fitter.lines.tied_voff[i, j])
                        voff_ln = cube_fitter.lines.voff[i, j].value
                    else
                        key_voff = cube_fitter.lines.tied_voff[i, j]
                        kv = findfirst(cube_fitter.tied_kinematics.key_voff[j] .== key_voff)
                        voff_ln = cube_fitter.tied_kinematics.voff[j][kv].value
                    end
                    if isnothing(cube_fitter.lines.tied_fwhm[i, j])
                        fwhm_ln = cube_fitter.lines.fwhm[i, j].value
                    else
                        key_fwhm = cube_fitter.lines.tied_fwhm[i, j]
                        kf = findfirst(cube_fitter.tied_kinematics.key_fwhm[j] .== key_fwhm)
                        fwhm_ln = cube_fitter.tied_kinematics.fwhm[j][kf].value
                    end

                    # Depending on flexible_wavesol option, we need to add 2 voffs
                    if !isnothing(cube_fitter.lines.tied_voff[i, j]) && cube_fitter.flexible_wavesol && isone(j)
                        append!(ln_pars, [amp_ln, voff_ln, cube_fitter.lines.voff[i, j].value, fwhm_ln])
                    else
                        append!(ln_pars, [amp_ln, voff_ln, fwhm_ln])
                    end

                    if cube_fitter.lines.profiles[i, j] == :GaussHermite
                        # 2 extra parameters: h3 and h4
                        append!(ln_pars, [cube_fitter.lines.h3[i, j].value, cube_fitter.lines.h4[i, j].value])
                    elseif cube_fitter.lines.profiles[i, j] == :Voigt
                        # 1 extra parameter: eta
                        if !cube_fitter.tie_voigt_mixing
                            # Individual eta parameter
                            append!(ln_pars, [cube_fitter.lines.η[i, j].value])
                        else
                            # Tied eta parameter
                            append!(ln_pars, [cube_fitter.voigt_mix_tied.value])
                        end
                    end
                end
            end
        end
    end

    # Relative step size vector (0 tells CMPFit to use a default value)
    ln_dstep = zeros(length(ln_pars))

    # deps = sqrt(eps())
    # ln_dstep = Float64[]
    # for i ∈ 1:cube_fitter.n_lines
    #     for j ∈ 1:cube_fitter.n_comps
    #         if !isnothing(cube_fitter.lines.profiles[i, j])

    #             amp_dstep = deps
    #             voff_dstep = 1e-4
    #             fwhm_dstep = isone(j) ? 1e-4 : deps

    #             # Depending on flexible_wavesol option, we need to add 2 voffs
    #             if !isnothing(cube_fitter.lines.tied_voff[i, j]) && cube_fitter.flexible_wavesol && isone(j)
    #                 append!(ln_dstep, [amp_dstep, voff_dstep, voff_dstep, fwhm_dstep])
    #             else
    #                 append!(ln_dstep, [amp_dstep, voff_dstep, fwhm_dstep])
    #             end

    #             if cube_fitter.lines.profiles[i, j] == :GaussHermite
    #                 # 2 extra parameters: h3 and h4
    #                 append!(ln_dstep, [deps, deps])
    #             elseif cube_fitter.lines.profiles[i, j] == :Voigt
    #                 # 1 extra parameter: eta
    #                 append!(ln_dstep, [deps])
    #             end
    #         end
    #     end
    # end 

    ln_pars, ln_dstep

end


"""
    get_line_parinfo(n_free, lb, ub, dp)

Get the CMPFit parinfo and config objects for a given CubeFitter object, given the vector of initial values,
limits, and relative step sizes.
"""
function get_line_parinfo(n_free, lb, ub, dp)

    # Convert parameter limits into CMPFit object
    parinfo = CMPFit.Parinfo(n_free)
    for pᵢ ∈ 1:n_free
        parinfo[pᵢ].fixed = 0
        parinfo[pᵢ].limited = (1,1)
        parinfo[pᵢ].limits = (lb[pᵢ], ub[pᵢ])
        parinfo[pᵢ].relstep = dp[pᵢ]
    end

    # Create a `config` structure
    config = CMPFit.Config()
    # Lower tolerance level for lines fit
    config.ftol = 1e-14
    config.xtol = 1e-14

    parinfo, config
end


"""
    pretty_print_line_results(cube_fitter, popt, perr)

Print out a nicely formatted summary of the line fit results for a given CubeFitter object.
"""
function pretty_print_line_results(cube_fitter::CubeFitter, popt::Vector{<:Real}, perr::Vector{<:Real})

    rel_amp, rel_voff, rel_fwhm = cube_fitter.relative_flags

    msg = "######################################################################\n"
    msg *= "############### SPAXEL FIT RESULTS -- EMISSION LINES #################\n"
    msg *= "######################################################################\n"
    pᵢ = 1
    msg *= "\n#> EMISSION LINES <#\n"
    for (k, name) ∈ enumerate(cube_fitter.lines.names)
        for j ∈ 1:cube_fitter.n_comps
            if !isnothing(cube_fitter.lines.profiles[k, j])
                nm = string(name) * "_$(j)"
                msg *= "$(nm)_amp:\t\t\t $(@sprintf "%.3f" popt[pᵢ]) +/- $(@sprintf "%.3f" perr[pᵢ]) " * ((isone(j) || !rel_amp) ? "[x norm]" : "[x amp_1]") * "\t " * 
                    "Limits: " * (isone(j) ? "(0, 1)" : "($(@sprintf "%.3f" cube_fitter.lines.acomp_amp[k, j-1].limits[1]), $(@sprintf "%.3f" cube_fitter.lines.acomp_amp[k, j-1].limits[2]))") * "\n"
                msg *= "$(nm)_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) " * ((isone(j) || !rel_voff) ? "km/s" : "[+ voff_1]") * " \t " *
                    "Limits: ($(@sprintf "%.0f" cube_fitter.lines.voff[k, j].limits[1]), $(@sprintf "%.0f" cube_fitter.lines.voff[k, j].limits[2]))\n"
                if !isnothing(cube_fitter.lines.tied_voff[k, j]) && cube_fitter.flexible_wavesol && isone(j)
                    msg *= "$(nm)_voff_indiv:   \t\t $(@sprintf "%.0f" popt[pᵢ+2]) +/- $(@sprintf "%.0f" perr[pᵢ+2]) km/s \t " *
                        "Limits: ($(@sprintf "%.0f" cube_fitter.lines.voff[k, j].limits[1]), $(@sprintf "%.0f" cube_fitter.lines.voff[k, j].limits[2]))\n"
                    msg *= "$(nm)_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+3]) +/- $(@sprintf "%.0f" perr[pᵢ+3]) km/s \t " *
                        "Limits: ($(@sprintf "%.0f" cube_fitter.lines.fwhm[k, j].limits[1]), $(@sprintf "%.0f" cube_fitter.lines.fwhm[k, j].limits[2]))\n"
                    pᵢ += 4
                else
                    if isone(j)
                        msg *= "$(nm)_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+2]) +/- $(@sprintf "%.0f" perr[pᵢ+2]) km/s \t " *
                            "Limits: ($(@sprintf "%.0f" cube_fitter.lines.fwhm[k, j].limits[1]), $(@sprintf "%.0f" cube_fitter.lines.fwhm[k, j].limits[2]))\n"
                    else
                        msg *= "$(nm)_fwhm:   \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) " * ((isone(j) || !rel_fwhm) ? "km/s" : "[x fwhm_1]") * "\t " *
                            "Limits: ($(@sprintf "%.3f" cube_fitter.lines.fwhm[k, j].limits[1]), $(@sprintf "%.3f" cube_fitter.lines.fwhm[k, j].limits[2]))\n"
                    end
                    pᵢ += 3
                end
                if cube_fitter.lines.profiles[k, j] == :GaussHermite
                    msg *= "$(nm)_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ]) +/- $(@sprintf "%.3f" perr[pᵢ])      \t " *
                        "Limits: ($(@sprintf "%.3f" cube_fitter.lines.h3[k, j].limits[1]), $(@sprintf "%.3f" cube_fitter.lines.h3[k, j].limits[2]))\n"
                    msg *= "$(nm)_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1])      \t " *
                        "Limits: ($(@sprintf "%.3f" cube_fitter.lines.h4[k, j].limits[1]), $(@sprintf "%.3f" cube_fitter.lines.h4[k, j].limits[2]))\n"
                    pᵢ += 2
                elseif cube_fitter.lines.profiles[k, j] == :Voigt 
                    msg *= "$(nm)_η:     \t\t $(@sprintf "%.3f" popt[pᵢ]) +/- $(@sprintf "%.3f" perr[pᵢ])      \t " *
                        "Limits: ($(@sprintf "%.3f" cube_fitter.lines.η[k, j].limits[1]), $(@sprintf "%.3f" cube_fitter.lines.η[k, j].limits[2]))\n"
                    pᵢ += 1
                end
            end
        end
        msg *= "\n"
    end 
    msg *= "######################################################################" 
    @debug msg

    msg

end

