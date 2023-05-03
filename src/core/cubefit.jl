#=
This file contains the CubeFitter struct and its related functions.  The CubeFitter is how
fitting is performed with Loki.
=#

############################## PARAMETER / MODEL STRUCTURES ####################################

"""
    ParamMaps(stellar_continuum, dust_continuum, dust_features, lines, tied_voffs, acomp_tied_voffs,
        tied_voigt_mix, extinction, hot_dust, reduced_χ2)

A structure for holding 2D maps of fitting parameters generated after fitting a cube.  Each parameter
that is fit (i.e. stellar continuum temperature, optical depth, line ampltidue, etc.) corresponds to one 
2D map with the value of that parameter, with each spaxel's fit value located at the corresponding location
in the parameter map.

# Fields
- `stellar_continuum::Dict{Symbol, Array{Float64, 2}}`: The stellar continuum parameters: amplitude and temperature
- `dust_continuum::Dict{Int, Dict{Symbol, Array{Float64, 2}}}`: The dust continuum parameters: amplitude and temperature for each
    dust continuum component
- `dust_features::Dict{String, Dict{Symbol, Array{Float64, 2}}}`: The dust feature parameters: amplitude, central wavelength, and FWHM
    for each PAH feature
- `extinction::Dict{Symbol, Array{Float64, 2}}`: Extinction parameters: optical depth at 9.7 μm and mixing ratio
- `hot_dust::Dict{Symbol, Array{Float64, 2}}`: Hot dust parameters: amplitude, temperature, covering fraction, warm tau, and cold tau
- `lines::Dict{Symbol, Dict{Symbol, Array{Float64, 2}}}`: The emission line parameters: amplitude, voff, FWHM, and any additional 
    line profile parameters for each line
- `reduced_χ2::Array{Float64, 2}`: The reduced chi^2 value of each fit

See ['parammaps_empty`](@ref) for a default constructor function.
"""
struct ParamMaps{T<:Real}

    stellar_continuum::Dict{Symbol, Array{T, 2}}
    dust_continuum::Dict{Int, Dict{Symbol, Array{T, 2}}}
    power_law::Dict{Int, Dict{Symbol, Array{T, 2}}}
    dust_features::Dict{String, Dict{Symbol, Array{T, 2}}}
    extinction::Dict{Symbol, Array{T, 2}}
    hot_dust::Dict{Symbol, Array{T, 2}}
    lines::Dict{Symbol, Dict{Symbol, Array{T, 2}}}
    statistics::Dict{Symbol, Array{T, 2}}

end


"""
    parammaps_empty(shape, n_dust_cont, df_names, line_names, line_profiles,
        flexible_wavesol)

A constructor function for making a default empty ParamMaps structure with all necessary fields for a given
fit of a DataCube.
"""
function parammaps_empty(shape::Tuple{S,S,S}, n_dust_cont::Integer, n_power_law::Integer, df_names::Vector{String}, 
    n_lines::S, n_comps::S, cf_lines::TransitionLines, flexible_wavesol::Bool)::ParamMaps where {S<:Integer}

    @debug """\n
    Creating ParamMaps struct with shape $shape
    ###########################################
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
    for n ∈ df_names
        dust_features[n] = Dict{Symbol, Array{Float64, 2}}()
        dust_features[n][:amp] = copy(nan_arr)
        dust_features[n][:mean] = copy(nan_arr)
        dust_features[n][:fwhm] = copy(nan_arr)
        dust_features[n][:flux] = copy(nan_arr)
        dust_features[n][:eqw] = copy(nan_arr)
        dust_features[n][:SNR] = copy(nan_arr)
        @debug "dust feature $n maps with keys $(keys(dust_features[n]))"
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
    @debug "hot dust maps with keys $(keys(hot_dust))"

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

    ParamMaps{Float64}(stellar_continuum, dust_continuum, power_law, dust_features, extinction, hot_dust, lines, statistics)
end


"""
    CubeModel(model, stellar, dust_continuum, dust_features, extinction, hot_dust, lines)

A structure for holding 3D models of intensity, split up into model components, generated when fitting a cube.
This will be the same shape as the input data, and preferably the same datatype too (i.e., JWST files have flux
and error in Float32 format, so we should also output in Float32 format).  This is useful as a way to quickly
compare the full model, or model components, to the data.

# Fields
- `model::Array{T, 3}`: The full 3D model.
- `stellar::Array{T, 3}`: The stellar component of the continuum.
- `dust_continuum::Array{T, 4}`: The dust components of the continuum. The 4th axis runs over each individual dust component.
- `dust_features::Array{T, 4}`: The dust (PAH) feature profiles. The 4th axis runs over each individual dust profile.
- `extinction::Array{T, 3}`: The extinction profile.
- `hot_dust::Array{T, 3}`: The hot dust emission profile
- `lines::Array{T, 4}`: The line profiles. The 4th axis runs over each individual line.

See [`cubemodel_empty`](@ref) for a default constructor method.
"""
struct CubeModel{T<:Real}

    model::Array{T, 3}
    stellar::Array{T, 3}
    dust_continuum::Array{T, 4}
    power_law::Array{T, 4}
    dust_features::Array{T, 4}
    extinction::Array{T, 3}
    abs_ice::Array{T, 3}
    abs_ch::Array{T, 3}
    hot_dust::Array{T, 3}
    lines::Array{T, 4}

end


"""
    cubemodel_empty(shape, n_dust_cont, df_names, line_names; floattype=floattype)

A constructor function for making a default empty CubeModel object with all the necessary fields for a given
fit of a DataCube.

# Arguments
`S<:Integer`
- `shape::Tuple{S,S,S}`: The dimensions of the DataCube being fit, formatted as a tuple of (nx, ny, nz)
- `n_dust_cont::Integer`: The number of dust continuum components in the fit (usually given by the number of temperatures 
    specified in the dust.toml file)
- `df_names::Vector{String}`: List of names of PAH features being fit, i.e. "PAH_12.62", ...
- `line_names::Vector{Symbol}`: List of names of lines being fit, i.e. "NeVI_7652", ...
- `floattype::DataType=Float32`: The type of float to use in the arrays. Should ideally be the same as the input data,
    which for JWST is Float32.
"""
function cubemodel_empty(shape::Tuple, n_dust_cont::Integer, n_power_law::Integer, df_names::Vector{String}, 
    line_names::Vector{Symbol}, floattype::DataType=Float32)::CubeModel

    @debug """\n
    Creating CubeModel struct with shape $shape
    ###########################################
    """

    # Make sure the floattype given is actually a type of float
    @assert floattype <: AbstractFloat "floattype must be a type of AbstractFloat (Float32 or Float64)!"

    # Initialize the arrays for each part of the full 3D model
    model = zeros(floattype, shape...)
    @debug "model cube"
    stellar = zeros(floattype, shape...)
    @debug "stellar continuum comp cube"
    dust_continuum = zeros(floattype, shape..., n_dust_cont)
    @debug "dust continuum comp cubes"
    power_law = zeros(floattype, shape..., n_power_law)
    @debug "power law comp cubes"
    dust_features = zeros(floattype, shape..., length(df_names))
    @debug "dust features comp cubes"
    extinction = zeros(floattype, shape...)
    @debug "extinction comp cube"
    abs_ice = zeros(floattype, shape...)
    @debug "abs_ice comp cube"
    abs_ch = zeros(floattype, shape...)
    @debug "abs_ch comp cube"
    hot_dust = zeros(floattype, shape...)
    @debug "hot dust comp cube"
    lines = zeros(floattype, shape..., length(line_names))
    @debug "lines comp cubes"

    CubeModel(model, stellar, dust_continuum, power_law, dust_features, extinction, abs_ice, abs_ch, hot_dust, lines)
end


"""
    CubeFitter(cube, z, name; plot_spaxels=plot_spaxels, plot_maps=plot_maps, save_fits=save_fits)

This is the main structure used for fitting IFU cubes, containing all of the necessary data, metadata,
fitting options, and associated functions for generating ParamMaps and CubeModel structures to handle the outputs 
of all the fits.  This is essentially the "configuration object" that tells the rest of the fitting code how
to run. The actual fitting functions (`fit_spaxel` and `fit_cube!`) require an instance of this structure.

# Fields
`T<:Real, S<:Integer`
- `cube::DataCube`: The main DataCube object containing the cube that is being fit
- `z::Real`: The redshift of the target that is being fit
- `plot_spaxels::Symbol=:pyplot`: A Symbol specifying the plotting backend to be used when plotting individual spaxel fits, can
    be either `:pyplot` or `:plotly`
- `plot_maps::Bool=true`: Whether or not to plot 2D maps of the best-fit parameters after the fitting is finished
- `parallel::Bool=true`: Whether or not to fit multiple spaxels in parallel using multiprocessing
- `save_fits::Bool=true`: Whether or not to save the final best-fit models and parameters as FITS files
Read from the options files:
- `overwrite::Bool`: Whether or not to overwrite old fits of spaxels when rerunning
- `track_memory::Bool`: Whether or not to save diagnostic files showing memory usage of the program
- `track_convergence::Bool`: Whether or not to save diagnostic files showing convergence of line fitting for each spaxel
- `make_movies::Bool`: Whether or not to save mp4 files of the final model
- `extinction_curve::String`: The type of extinction curve being used, either `"kvt"` or `"d+"`
- `extinction_screen::Bool`: Whether or not the extinction is modeled as a screen
- `T_s::Parameter`: The stellar temperature parameter
- `T_dc::Vector{Parameter}`: The dust continuum temperature parameters
- `τ_97::Parameter`: The dust opacity at 9.7 um parameter
- `τ_ice::Parameter`: The peak opacity from ice absorption (at around 6.9 um)
- `τ_ch::Parameter`: The peak opacity from CH absorption (at around 6.9 um)
- `β::Parameter`: The extinction profile mixing parameter
- `T_hot::Parameter`: The hot dust temperature
- `Cf_hot::Parameter`: The hot dust covering fraction
- `τ_warm::Parameter`: The warm dust optical depth
- `τ_cold::Parameter`: The cold dust optical depth
- `n_dust_cont::Integer`: The number of dust continuum profiles
- `df_names::Vector{String}`: The names of each PAH feature profile
- `dust_features::Vector{Dict}`: All of the fitting parameters for each PAH feature
- `n_lines::Integer`: The number of lines being fit
- `line_names::Vector{Symbol}`: The names of each line being fit
- `line_profiles::Vector{Symbol}`: The profiles of each line being fit
- `line_acomp_profiles::Vector{Union{Nothing,Symbol}}`: Same as `line_profiles`, but for the additional components
- `lines::Vector{TransitionLine}`: All of the fitting parameters for each line
- `n_kin_tied::Integer`: The number of tied velocity offsets
- `line_tied::Vector{Union{String,Nothing}}`: List of line tie keys which specify whether the voff of the given line should be
    tied to other lines. The line tie key itself may be either `nothing` (untied), or a String specifying the group of lines
    being tied, i.e. "H2"
- `kin_tied_key::Vector{String}`: List of only the unique keys in line_tied (not including `nothing`)
- `voff_tied::Vector{Parameter}`: The actual tied voff parameter objects, corresponding to the `kin_tied_key`
- `fwhm_tied::Vector{Parameter}`: The actual tied fwhm parameter objects, corresponding to the `kin_tied_key`
- `n_acomp_kin_tied::Integer`: Same as `n_kin_tied`, but for additional components
- `line_acomp_tied::Vector{Union{String,Nothing}}`: Same as `line_tied`, but for additional components
- `acomp_kin_tied_key::Vector{String}`: Same as `kin_tied_key`, but for additional components
- `acomp_voff_tied::Vector{Parameter}`: Same as `voff_tied`, but for additional components
- `acomp_fwhm_tied::Vector{Parameter}`: Same as `fwhm_tied`, but for additional components
- `tie_voigt_mixing::Bool`: Whether or not the Voigt mixing parameter is tied between all the lines with Voigt profiles
- `voigt_mix_tied::Parameter`: The actual tied Voigt mixing parameter object, given `tie_voigt_mixing` is true
- `n_params_cont::Integer`: The total number of free fitting parameters for the continuum fit (not including emission lines)
- `n_params_lines::Integer`: The total number of free fitting parameters for the emission line fit (not including the continuum)
- `cosmology::Cosmology.AbstractCosmology`: The Cosmology, used solely to create physical scale bars on the 2D parameter plots
- `χ²_thresh::Real`: The threshold for reduced χ² values, below which the best fit parameters for a given
    row will be set
- `flexible_wavesol::Bool`: Whether or not to allow small variations in the velocity offsets even when tied, to account
    for a bad wavelength solution
- `p_best_cont::SharedArray{T}`: A rolling collection of best fit continuum parameters for the best fitting spaxels
    along each row, for fits with a reduced χ² below χ²_thresh, which are used for the starting parameters in the following
    fits for the given row
- `p_best_line::SharedArray{T}`: Same as `p_best_cont`, but for the line parameters
- `χ²_best::SharedVector{T}`: The reduced χ² values associated with the `p_best_cont` and `p_best_line` values
    in each row
- `best_spaxel::SharedVector{Tuple{S,S}}`: The locations of the spaxels associated with the `p_best_cont` and `p_best_line`
    values in each row

See [`ParamMaps`](@ref), [`parammaps_empty`](@ref), [`CubeModel`](@ref), [`cubemodel_empty`](@ref), 
    [`fit_spaxel`](@ref), [`fit_cube!`](@ref)
"""
struct CubeFitter{T<:Real,S<:Integer}

    # See explanations for each field in the docstring!
    
    # Data
    cube::DataCube
    z::T
    name::String

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
    fit_sil_emission::Bool
    fit_all_samin::Bool

    # Continuum parameters
    continuum::Continuum

    # Dust Feature parameters
    n_dust_cont::S
    n_power_law::S
    n_dust_feat::S
    dust_features::DustFeatures

    # Line parameters
    n_lines::S
    n_acomps::S
    n_comps::S
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

    p_init_cont::Vector{T}
    p_init_line::Vector{T}

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
        name = replace(name, #= no spaces! =# " " => "_")

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

        #############################################################

        @debug """\n
        Creating CubeFitter struct for $name at z=$z
        ############################################
        """
        # Alias
        λ = cube.λ

        continuum, dust_features = parse_dust()
        lines, tied_kinematics, flexible_wavesol, tie_voigt_mixing, voigt_mix_tied = parse_lines()

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
        df_filt = [(minimum(λ)-0.5 < dust_features.mean[i].value < maximum(λ)+0.5) for i ∈ 1:length(dust_features.mean)]
        if !isnothing(out[:user_mask])
            for pair in out[:user_mask]
                df_filt .&= [~(pair[1] < dust_features.mean[i].value < pair[2]) for i ∈ 1:length(dust_features.mean)]
            end
        end
        dust_features = DustFeatures(dust_features.names[df_filt], 
                                     dust_features.profiles[df_filt],
                                     dust_features.mean[df_filt],
                                     dust_features.fwhm[df_filt])
        n_dust_features = length(dust_features.names)
        msg = "### Model will include $n_dust_features dust feature (PAH) components ###"
        for df_mn ∈ dust_features.mean
            msg *= "\n### at lambda = $df_mn um ###"
        end
        @debug msg

        # Only use lines within the wavelength range being fit
        ln_filt = minimum(λ) .< lines.λ₀ .< maximum(λ)
        if !isnothing(out[:user_mask])
            for pair in out[:user_mask]
                ln_filt .&= .~(pair[1] .< lines.λ₀ .< pair[2])
            end
        end
        # Convert to a vectorized "TransitionLines" object
        lines = TransitionLines(lines.names[ln_filt], lines.λ₀[ln_filt], lines.profiles[ln_filt, :],
                                lines.tied_voff[ln_filt, :], lines.tied_fwhm[ln_filt, :], lines.voff[ln_filt, :], 
                                lines.fwhm[ln_filt, :], lines.h3[ln_filt, :], lines.h4[ln_filt, :], lines.η[ln_filt, :])
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

        # Remove unnecessary rows/keys from the tied_kinematics object after the lines have been filtered
        @debug "TiedKinematics before filtering: $tied_kinematics"
        for j ∈ 1:n_comps
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
        n_params_cont = (2+4) + 2n_dust_cont + 2n_power_law + 3n_dust_features + (options[:fit_sil_emission] ? 5 : 0)
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
        @debug "Preparing initial best fit parameter vectors with $(n_params_cont+2) and $(n_params_lines) parameters"
        p_init_cont = zeros(n_params_cont+2)
        p_init_line = zeros(n_params_lines)

        # If a fit has been run previously, read in the file containing the rolling best fit parameters
        # to pick up where the fitter left off seamlessly
        if isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv")) && isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"))
            p_init_cont = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv"), ',', Float64, '\n')[:, 1]
            p_init_line = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"), ',', Float64, '\n')[:, 1]
        end

        new{typeof(z), typeof(n_lines)}(cube, z, name, out[:user_mask], out[:plot_spaxels], out[:plot_maps], out[:plot_range], out[:parallel], 
            out[:save_fits], out[:save_full_model], out[:overwrite], out[:track_memory], out[:track_convergence], out[:make_movies], 
            out[:extinction_curve], out[:extinction_screen], out[:fit_sil_emission], out[:fit_all_samin], continuum, n_dust_cont, 
            n_power_law, n_dust_features, dust_features, n_lines, n_acomps, n_comps, lines, tied_kinematics, tie_voigt_mixing, 
            voigt_mix_tied, n_params_cont, n_params_lines, n_params_extra, out[:cosmology], flexible_wavesol, out[:n_bootstrap], out[:random_seed], 
            p_init_cont, p_init_line)
    end

end


"""
    generate_cubemodel(cube_fitter)

Generate a CubeModel object corresponding to the options given by the CubeFitter object
"""
function generate_cubemodel(cube_fitter::CubeFitter, aperture::Bool=false)
    shape = aperture ? (1,1,size(cube_fitter.cube.Iν, 3)) : size(cube_fitter.cube.Iν)
    # Full 3D intensity model array
    @debug "Generating full 3D cube models"
    cubemodel_empty(shape, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.names, cube_fitter.lines.names)
end


"""
    generate_parammaps(cube_fitter)

Generate two ParamMaps objects (for the values and errors) corrresponding to the options given
by the CubeFitter object
"""
function generate_parammaps(cube_fitter::CubeFitter, aperture::Bool=false)
    shape = aperture ? (1,1,size(cube_fitter.cube.Iν, 3)) : size(cube_fitter.cube.Iν)
    # 2D maps of fitting parameters
    @debug "Generating 2D parameter value & error maps"
    param_maps = parammaps_empty(shape, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.names, 
                                 cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, cube_fitter.flexible_wavesol)
    # 2D maps of fitting parameter +/- 1 sigma errors
    param_errs_lo = parammaps_empty(shape, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.names, 
                                 cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, cube_fitter.flexible_wavesol)
    param_errs_up = parammaps_empty(shape, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.names, 
                                 cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, cube_fitter.flexible_wavesol)
    param_errs = [param_errs_lo, param_errs_up]
    param_maps, param_errs
end


"""
    get_continuum_plimits(cube_fitter, I)

Get the continuum limits vector for a given CubeFitter object, split up by the 2 continuum fitting steps.
Also returns a boolean vector for which parameters are allowed to vary.
"""
function get_continuum_plimits(cube_fitter::CubeFitter)

    dust_features = cube_fitter.dust_features
    continuum = cube_fitter.continuum

    amp_dc_plim = (0., Inf)
    amp_df_plim = (0., clamp(1 / exp(-continuum.τ_97.limits[2]), 1., Inf))

    stellar_plim = [amp_dc_plim, continuum.T_s.limits]
    stellar_lock = [false, continuum.T_s.locked]
    dc_plim = vcat([[amp_dc_plim, Ti.limits] for Ti ∈ continuum.T_dc]...)
    dc_lock = vcat([[false, Ti.locked] for Ti ∈ continuum.T_dc]...)
    pl_plim = vcat([[amp_dc_plim, pl.limits] for pl ∈ continuum.α]...)
    pl_lock = vcat([[false, pl.locked] for pl ∈ continuum.α]...)
    df_plim = vcat([[amp_df_plim, mi.limits, fi.limits] for (mi, fi) ∈ zip(dust_features.mean, dust_features.fwhm)]...)
    df_lock = vcat([[false, mi.locked, fi.locked] for (mi, fi) ∈ zip(dust_features.mean, dust_features.fwhm)]...)
    ext_plim = [continuum.τ_97.limits, continuum.τ_ice.limits, continuum.τ_ch.limits, continuum.β.limits]
    ext_lock = [continuum.τ_97.locked, continuum.τ_ice.locked, continuum.τ_ch.locked, continuum.β.locked]
    hd_plim = cube_fitter.fit_sil_emission ? [amp_dc_plim, continuum.T_hot.limits, continuum.Cf_hot.limits, 
        continuum.τ_warm.limits, continuum.τ_cold.limits] : []
    hd_lock = cube_fitter.fit_sil_emission ? [false, continuum.T_hot.locked, continuum.Cf_hot.locked,
        continuum.τ_warm.locked, continuum.τ_cold.locked] : []

    # Split up for the two different stages of continuum fitting -- with templates and then with the PAHs
    plims_1 = Vector{Tuple}(vcat(stellar_plim, dc_plim, pl_plim, ext_plim, hd_plim, [amp_df_plim, amp_df_plim]))
    lock_1 = BitVector(vcat(stellar_lock, dc_lock, pl_lock, ext_lock, hd_lock, [false, false]))
    plims_2 = Vector{Tuple}(df_plim)
    lock_2 = BitVector(df_lock)

    plims_1, plims_2, lock_1, lock_2

end


"""
    get_continuum_initial_values(cube_fitter, I, init)

Get the vector of starting values for the continuum fit for a given CubeFitter object. Again, the
vector is split up by the 2 continuum fitting steps.
"""
function get_continuum_initial_values(cube_fitter::CubeFitter, λ::Vector{<:Real}, I::Vector{<:Real}, N::Real, init::Bool)

    # Check if the cube fitter has initial fit parameters 
    if !init

        @debug "Using initial best fit continuum parameters..."

        # Set the parameters to the best parameters
        p₀ = copy(cube_fitter.p_init_cont)[1:end-2]
        pah_frac = copy(cube_fitter.p_init_cont)[end-1:end]

        # pull out optical depth that was pre-fit
        # τ_97_0 = cube_fitter.τ_guess[parse(Int, cube_fitter.cube.channel)][spaxel]
        max_τ = cube_fitter.continuum.τ_97.limits[2]

        # scale all flux amplitudes by the difference in medians between the spaxel and the summed spaxels
        # (should be close to 1 since the sum is already normalized by the number of spaxels included anyways)
        scale = max(nanmedian(I), 1e-10) * N / nanmedian(sumdim(cube_fitter.cube.Iν, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2)))
        max_amp = 1 / exp(-max_τ)

        # PAH template strengths
        pah_frac .*= scale
        
        # Stellar amplitude
        p₀[1] *= scale
        pᵢ = 3

        # Dust continuum amplitudes
        for _ ∈ 1:cube_fitter.n_dust_cont
            p₀[pᵢ] *= scale
            pᵢ += 2
        end

        # Power law amplitudes
        for _ ∈ 1:cube_fitter.n_power_law
            p₀[pᵢ] = clamp(p₀[pᵢ]*scale, 0., max_amp)
            pᵢ += 2
        end

        # Set optical depth based on the pre-fitting
        # p₀[pᵢ] = τ_97_0
        pᵢ += 4

        if cube_fitter.fit_sil_emission
            # Hot dust amplitude
            p₀[pᵢ] *= scale
            # Warm / cold optical depths
            # p₀[pᵢ+3] = τ_97_0
            # p₀[pᵢ+4] = τ_97_0
            pᵢ += 5
        end

        # Dust feature amplitudes
        for i ∈ 1:cube_fitter.n_dust_feat
            p₀[pᵢ] = clamp(p₀[pᵢ]*scale, 0., max_amp)
            pᵢ += 3
        end

    # Otherwise, we estimate the initial parameters based on the data
    else

        continuum = cube_fitter.continuum

        @debug "Calculating initial starting points..."
        pah_frac = repeat([clamp(nanmedian(I)/2, 0., Inf)], 2)
        cubic_spline = Spline1D(λ, I, k=3)

        # Stellar amplitude
        λ_s = minimum(λ) < 5 ? minimum(λ)+0.1 : 5.1
        A_s = clamp(cubic_spline(λ_s) * N / Blackbody_ν(λ_s, continuum.T_s.value), 0., Inf) 

        # Dust feature amplitudes
        A_df = repeat([clamp(nanmedian(I)/2, 0., Inf)], cube_fitter.n_dust_feat)

        # Dust continuum amplitudes
        λ_dc = clamp.([Wein(Ti.value) for Ti ∈ continuum.T_dc], minimum(λ), maximum(λ))
        A_dc = clamp.([cubic_spline(λ_dci) * N / Blackbody_ν(λ_dci, T_dci.value) for (λ_dci, T_dci) ∈ 
            zip(λ_dc, continuum.T_dc)] .* (λ_dc ./ 9.7).^2 ./ 5., 0., Inf)
        
        # Power law amplitudes
        A_pl = [αi.value > 0 ? clamp(I[end]/2, 0., Inf) : 
                αi.value < 0 ? clamp(I[1]/2, 0., Inf) : 
                clamp(nanmedian(I)/2, 0., Inf) for αi ∈ continuum.α]
        
        # Hot dust amplitude
        λ_hd = clamp(Wein(continuum.T_hot.value), minimum(λ), maximum(λ))
        A_hd = clamp(cubic_spline(λ_hd) * N / Blackbody_ν(λ_hd, continuum.T_hot.value), 0., Inf) / 2

        stellar_pars = [A_s, continuum.T_s.value]
        dc_pars = vcat([[Ai, Ti.value] for (Ai, Ti) ∈ zip(A_dc, continuum.T_dc)]...)
        pl_pars = vcat([[Ai, αi.value] for (Ai, αi) ∈ zip(A_pl, continuum.α)]...)
        df_pars = vcat([[Ai, mi.value, fi.value] for (Ai, mi, fi) ∈ zip(A_df, cube_fitter.dust_features.mean, cube_fitter.dust_features.fwhm)]...)
        if cube_fitter.fit_sil_emission
            hd_pars = [A_hd, continuum.T_hot.value, continuum.Cf_hot.value, continuum.τ_warm.value, continuum.τ_cold.value]
        else
            hd_pars = []
        end

        extinction_pars = [continuum.τ_97.value, continuum.τ_ice.value, continuum.τ_ch.value, continuum.β.value]

        # Initial parameter vector
        p₀ = Vector{Float64}(vcat(stellar_pars, dc_pars, pl_pars, extinction_pars, hd_pars, df_pars))

    end

    @debug "Continuum Parameter labels: \n [stellar_amp, stellar_temp, " * 
        join(["dust_continuum_amp_$i, dust_continuum_temp_$i" for i ∈ 1:cube_fitter.n_dust_cont], ", ") * 
        join(["power_law_amp_$i, power_law_index_$i" for i ∈ 1:cube_fitter.n_power_law], ", ") *
        ", extinction_tau_97, extinction_tau_ice, extinction_tau_ch, extinction_beta, " *  
        (cube_fitter.fit_sil_emission ? "hot_dust_amp, hot_dust_temp, hot_dust_covering_frac, hot_dust_tau, cold_dust_tau, " : "") *
        join(["$(df)_amp, $(df)_mean, $(df)_fwhm" for df ∈ cube_fitter.dust_features.names], ", ") * "]"
        
    @debug "Continuum Starting Values: \n $p₀"

    # Step 1: Stellar + Dust blackbodies, 2 new amplitudes for the PAH templates, and the extinction parameters
    pars_1 = vcat(p₀[1:(2+2*cube_fitter.n_dust_cont+2*cube_fitter.n_power_law+4+(cube_fitter.fit_sil_emission ? 5 : 0))], pah_frac)
    # Step 2: The PAH profile amplitudes, centers, and FWHMs
    pars_2 = p₀[(3+2*cube_fitter.n_dust_cont+2*cube_fitter.n_power_law+4+(cube_fitter.fit_sil_emission ? 5 : 0)):end]

    pars_1, pars_2

end


"""
    get_continuum_parinfo(n_free_1, n_free_2, lb_1, ub_1, lb_2, ub_2)

Get the CMPFit parinfo and config objects for a given CubeFitter object, given the vector of initial valuels,
limits, and boolean locked values.
"""
function get_continuum_parinfo(n_free_1::S, n_free_2::S, lb_1::Vector{T}, ub_1::Vector{T}, 
    lb_2::Vector{T}, ub_2::Vector{T}) where {S<:Integer,T<:Real}

    parinfo_1 = CMPFit.Parinfo(n_free_1)
    parinfo_2 = CMPFit.Parinfo(n_free_2)

    for pᵢ ∈ 1:n_free_1
        parinfo_1[pᵢ].fixed = 0
        parinfo_1[pᵢ].limited = (1,1)
        parinfo_1[pᵢ].limits = (lb_1[pᵢ], ub_1[pᵢ])
    end

    for pᵢ ∈ 1:n_free_2
        parinfo_2[pᵢ].fixed = 0
        parinfo_2[pᵢ].limited = (1,1)
        parinfo_2[pᵢ].limits = (lb_2[pᵢ], ub_2[pᵢ])
    end

    # Create a `config` structure
    config = CMPFit.Config()

    parinfo_1, parinfo_2, config

end


"""
    pretty_print_continuum_results(cube_fitter, popt, perr)

Print out a nicely formatted summary of the continuum fit results for a given CubeFitter object.
"""
function pretty_print_continuum_results(cube_fitter::CubeFitter, popt::Vector{<:Real}, perr::Vector{<:Real},
    I::Vector{<:Real})

    continuum = cube_fitter.continuum

    msg = "######################################################################\n"
    msg *= "################# SPAXEL FIT RESULTS -- CONTINUUM ####################\n"
    msg *= "######################################################################\n"
    msg *= "\n#> STELLAR CONTINUUM <#\n"
    msg *= "Stellar_amp: \t\t\t $(@sprintf "%.3e" popt[1]) +/- $(@sprintf "%.3e" perr[1]) [-] \t Limits: (0, Inf)\n"
    msg *= "Stellar_temp: \t\t\t $(@sprintf "%.0f" popt[2]) +/- $(@sprintf "%.3e" perr[2]) K \t (fixed)\n"
    pᵢ = 3
    msg *= "\n#> DUST CONTINUUM <#\n"
    for i ∈ 1:cube_fitter.n_dust_cont
        msg *= "Dust_continuum_$(i)_amp: \t\t $(@sprintf "%.3e" popt[pᵢ]) +/- $(@sprintf "%.3e" perr[pᵢ]) [-] \t Limits: (0, Inf)\n"
        msg *= "Dust_continuum_$(i)_temp: \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.3e" perr[pᵢ+1]) K \t\t\t (fixed)\n"
        msg *= "\n"
        pᵢ += 2
    end
    msg *= "\n#> POWER LAWS <#\n"
    for k ∈ 1:cube_fitter.n_power_law
        msg *= "Power_law_$(k)_amp: \t\t $(@sprintf "%.3e" popt[pᵢ]) +/- $(@sprintf "%.3e" perr[pᵢ]) [x norm] \t Limits: (0, Inf)\n"
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
    if cube_fitter.fit_sil_emission
        msg *= "\n#> HOT DUST <#\n"
        msg *= "Hot_dust_amp: \t\t\t $(@sprintf "%.3e" popt[pᵢ]) +/- $(@sprintf "%.3e" perr[pᵢ]) [-] \t Limits: (0, Inf)\n"
        msg *= "Hot_dust_temp: \t\t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) K \t Limits: " *
            "($(@sprintf "%.0f" continuum.T_hot.limits[1])), $(@sprintf "%.0f" continuum.T_hot.limits[2]))" *
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
        pᵢ += 5
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
        msg *= "\n"
        pᵢ += 3
    end
    msg *= "######################################################################"
    @debug msg

    msg

end


"""
    get_line_plimits(cube_fitter)

Get the line limits vector for a given CubeFitter object. Also returns boolean locked values and
names of each parameter as strings.
"""
function get_line_plimits(cube_fitter::CubeFitter, ext_curve::Vector{<:Real}, init::Bool)

    # Set up the limits vector
    amp_plim = (0., clamp(maximum(1 ./ ext_curve), 1., Inf))
    amp_acomp_plim = (0., 1.)
    ln_plims = Vector{Tuple}()
    ln_lock = BitVector()
    ln_names = Vector{String}()
    
    voff_tied = []
    fwhm_tied = []
    for j ∈ 1:cube_fitter.n_comps
        append!(voff_tied, [[[] for _ in cube_fitter.tied_kinematics.key_voff[j]]])
        append!(fwhm_tied, [[[] for _ in cube_fitter.tied_kinematics.key_fwhm[j]]])
    end
    η_tied = []
    
    # Loop through each line and append the new components
    ind = 1
    for i ∈ 1:cube_fitter.n_lines
        for j ∈ 1:cube_fitter.n_comps
            
            amp_ln_plim = isone(j) ? amp_plim : amp_acomp_plim
            if !isnothing(cube_fitter.lines.profiles[i, j])

                # name
                ln_name = string(cube_fitter.lines.names[i]) * "_$(j)"

                # get the right voff and fwhm parameters based on if theyre tied or not
                vt = ft = false
                kv = kf = nothing
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

                # Depending on flexible_wavesol, we need to add 2 voffs instead of 1 voff
                if !isnothing(cube_fitter.lines.tied_voff[i, j]) && cube_fitter.flexible_wavesol && isone(j)
                    append!(ln_plims, [amp_ln_plim, voff_ln_plim, cube_fitter.lines.voff[i, j].limits, fwhm_ln_plim])
                    append!(ln_lock, [false, voff_ln_locked, cube_fitter.lines.voff[i, j].locked, fwhm_ln_locked])
                    append!(ln_names, ["$(ln_name)_amp", voff_ln_name, "$(ln_name)_voff_indiv", fwhm_ln_name])
                    append!(voff_tied[j][kv], [ind+1])
                    if ft
                        append!(fwhm_tied[j][kf], [ind+3])
                    end
                    ind += 4
                else
                    append!(ln_plims, [amp_ln_plim, voff_ln_plim, fwhm_ln_plim])
                    append!(ln_lock, [false, voff_ln_locked, fwhm_ln_locked])
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
                        append!(ln_lock, [cube_fitter.lines.η[i, j].locked || !init])
                        append!(ln_names, ["$(ln_name)_eta"])
                    else
                        append!(ln_plims, [cube_fitter.voigt_mix_tied.limits])
                        append!(ln_lock, [cube_fitter.voigt_mix_tied.locked || !init])
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
    for j ∈ 1:cube_fitter.n_comps
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

    # Convert the paired tuples into indices for each tied parameter
    tied_indices = sort([tp[2] for tp in tied_pairs])

    ln_plims, ln_lock, ln_names, tied_pairs, tied_indices

end


"""
    get_line_initial_values(cube_fitter, init)

Get the vector of starting values for the line fit for a given CubeFitter object.
"""
function get_line_initial_values(cube_fitter::CubeFitter, init::Bool)

    # Check if there are previous best fit parameters
    if !init

        @debug "Using initial best fit line parameters..."

        # If so, set the parameters to the previous ones
        ln_pars = copy(cube_fitter.p_init_line)

    else

        @debug "Calculating initial starting points..."
        
        # Start the ampltiudes at 1/2 and 1/4 (in normalized units)
        A_ln = ones(cube_fitter.n_lines) .* 0.5
        A_fl = ones(cube_fitter.n_lines) .* 0.25     # (acomp amp is multiplied with main amp)

        # Initial parameter vector
        ln_pars = Float64[]
        for i ∈ 1:cube_fitter.n_lines
            for j ∈ 1:cube_fitter.n_comps
                if !isnothing(cube_fitter.lines.profiles[i, j])

                    amp_ln = isone(j) ? A_ln[i] : A_fl[i]
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

    ln_pars

end


"""
    get_line_parinfo(n_free, lb, ub)

Get the CMPFit parinfo and config objects for a given CubeFitter object, given the vector of initial values,
limits, and boolean locked values.
"""
function get_line_parinfo(n_free, lb, ub)

    # Convert parameter limits into CMPFit object
    parinfo = CMPFit.Parinfo(n_free)
    for pᵢ ∈ 1:n_free
        parinfo[pᵢ].fixed = 0
        parinfo[pᵢ].limited = (1,1)
        parinfo[pᵢ].limits = (lb[pᵢ], ub[pᵢ])
    end

    # Create a `config` structure
    config = CMPFit.Config()

    parinfo, config
end


"""
    pretty_print_line_results(cube_fitter, popt, perr, prof_ln, acomp_prof_ln)

Print out a nicely formatted summary of the line fit results for a given CubeFitter object.
"""
function pretty_print_line_results(cube_fitter::CubeFitter, popt::Vector{<:Real}, perr::Vector{<:Real})

    msg = "######################################################################\n"
    msg *= "############### SPAXEL FIT RESULTS -- EMISSION LINES #################\n"
    msg *= "######################################################################\n"
    pᵢ = 1
    msg *= "\n#> EMISSION LINES <#\n"
    for (k, name) ∈ enumerate(cube_fitter.lines.names)
        for j ∈ 1:cube_fitter.n_comps
            if !isnothing(cube_fitter.lines.profiles[k, j])
                nm = string(name) * "_$(j)"
                msg *= "$(nm)_amp:\t\t\t $(@sprintf "%.3f" popt[pᵢ]) +/- $(@sprintf "%.3f" perr[pᵢ]) [x norm] \t Limits: (0, 1)\n"
                msg *= "$(nm)_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) " * (isone(j) ? "km/s" : "[+ voff_1]") * " \t " *
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
                        msg *= "$(nm)_fwhm:   \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) [x fwhm_1] \t " *
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

