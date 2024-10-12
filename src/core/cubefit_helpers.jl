


# """
#     MIRCubeModel(model, stellar, dust_continuum, dust_features, extinction, hot_dust, lines)

# A structure for holding 3D models of intensity, split up into model components, generated when fitting a cube.
# This will be the same shape as the input data, and preferably the same datatype too (i.e., JWST files have flux
# and error in Float32 format, so we should also output in Float32 format).  This is useful as a way to quickly
# compare the full model, or model components, to the data.

# # Fields {T<:Real}
# - `model::Array{T, 3}`: The full 3D model.
# - `stellar::Array{T, 3}`: The stellar component of the continuum.
# - `dust_continuum::Array{T, 4}`: The dust components of the continuum. The 4th axis runs over each individual dust component.
# - `power_law::Array{T, 4}`: The power law components of the continuum. The 4th axis runs over each individual power law.
# - `dust_features::Array{T, 4}`: The dust (PAH) feature profiles. The 4th axis runs over each individual dust profile.
# - `abs_features::Array{T, 4}`: The absorption feature profiles. The 4th axis runs over each individual absorption profile.
# - `extinction::Array{T, 3}`: The extinction profile.
# - `abs_ice::Array{T, 3}`: The water-ice absorption feature profile.
# - `abs_ch::Array{T, 3}`: The CH absorption feature profile.
# - `hot_dust::Array{T, 3}`: The hot dust emission profile.
# - `templates::Array{T, 4}`: The generic template profiles.
# - `lines::Array{T, 4}`: The line profiles. The 4th axis runs over each individual line.

# See [`cubemodel_empty`](@ref) for a default constructor method.
# """
# struct CubeModel{T<:Real} <: CubeModel

    # COME BACK TO THIS AFTER SETTING UP THE MODEL 

    # model::Array{T, 3}
    # unobscured_continuum::Array{T, 3}
    # obscured_continuum::Array{T, 3}
    # stellar::Array{T, 3}
    # dust_continuum::Array{T, 4}
    # power_law::Array{T, 4}
    # dust_features::Array{T, 4}
    # abs_features::Array{T, 4}
    # extinction::Array{T, 4}
    # abs_ice::Array{T, 3}
    # abs_ch::Array{T, 3}
    # hot_dust::Array{T, 3}
    # templates::Array{T, 4}
    # lines::Array{T, 4}

# end


# """
#     cubemodel_empty(shape, n_dust_cont, n_power_law, df_names, ab_names, line_names[, floattype])

# A constructor function for making a default empty MIRCubeModel object with all the necessary fields for a given
# fit of a DataCube.

# # Arguments
# - `shape::Tuple`: The dimensions of the DataCube being fit, formatted as a tuple of (nx, ny, nz)
# - `n_dust_cont::Integer`: The number of dust continuum components in the fit (usually given by the number of temperatures 
#     specified in the dust.toml file)
# - `n_power_law::Integer`: The number of power law continuum components in the fit.
# - `df_names::Vector{String}`: List of names of PAH features being fit, i.e. "PAH_12.62", ...
# - `ab_names::Vector{String}`: List of names of absorption features being fit, i.e. "abs_HCO+_12.1", ...
# - `temp_names::Vector{String}`: List of names of generic templates in the fit, i.e. "nuclear", ...
# - `line_names::Vector{Symbol}`: List of names of lines being fit, i.e. "NeVI_7652", ...
# - `floattype::DataType=Float32`: The type of float to use in the arrays. Should ideally be the same as the input data,
#     which for JWST is Float32.
# """
# function cubemodel_empty(shape::Tuple, n_dust_cont::Integer, n_power_law::Integer, df_names::Vector{String}, 
#     ab_names::Vector{String}, temp_names::Vector{String}, line_names::Vector{Symbol}, extinction_curve::String, 
#     floattype::DataType=Float32)::MIRCubeModel

#     @debug """\n
#     Creating MIRCubeModel struct with shape $shape
#     ##############################################
#     """

#     # Make sure the floattype given is actually a type of float
#     @assert floattype <: AbstractFloat "floattype must be a type of AbstractFloat (Float32 or Float64)!"
#     # Swap the wavelength axis to be the FIRST axis since it is accessed most often and thus should be continuous in memory
#     shape2 = (shape[end], shape[1:end-1]...)

#     # Initialize the arrays for each part of the full 3D model
#     model = zeros(floattype, shape2...)
#     @debug "model cube"
#     unobscured_continuum = zeros(floattype, shape2...)
#     @debug "unobscured continuum cube"
#     obscured_continuum = zeros(floattype, shape2...)
#     @debug "obscured continuum cube"
#     stellar = zeros(floattype, shape2...)
#     @debug "stellar continuum comp cube"
#     dust_continuum = zeros(floattype, shape2..., n_dust_cont)
#     @debug "dust continuum comp cubes"
#     power_law = zeros(floattype, shape2..., n_power_law)
#     @debug "power law comp cubes"
#     dust_features = zeros(floattype, shape2..., length(df_names))
#     @debug "dust features comp cubes"
#     abs_features = zeros(floattype, shape2..., length(ab_names))
#     @debug "absorption features comp cubes"
#     extinction = zeros(floattype, shape2..., extinction_curve == "decompose" ? 4 : 1)
#     @debug "extinction comp cube"
#     abs_ice = zeros(floattype, shape2...)
#     @debug "abs_ice comp cube"
#     abs_ch = zeros(floattype, shape2...)
#     @debug "abs_ch comp cube"
#     hot_dust = zeros(floattype, shape2...)
#     @debug "hot dust comp cube"
#     templates = zeros(floattype, shape2..., length(temp_names))
#     @debug "templates comp cube"
#     lines = zeros(floattype, shape2..., length(line_names))
#     @debug "lines comp cubes"

#     MIRCubeModel(model, unobscured_continuum, obscured_continuum, stellar, dust_continuum, power_law, dust_features, abs_features, 
#         extinction, abs_ice, abs_ch, hot_dust, templates, lines)
# end




# Helper function for preparing continuum and dust feature parameters for 
# a CubeFitter object
function cubefitter_prepare_continuum(λ::Vector{<:QWave}, z::Real, out::Dict, λunit::Unitful.Units, 
    Iunit::Unitful.Units, region::SpectralRegion, name::String, cube::DataCube)

    # Construct the ModelParameters object
    model_parameters = construct_model_parameters(out, λunit, Iunit, region, z)
    vres = NaN*u"km/s"   # (vres doesnt make sense if the wavelength vector isnt logarithmic)

    # Count a few different parameters 
    params = get_flattened_fit_parameters(model_parameters)
    pnames = params.names

    # Stellar populations
    ssps = nothing
    n_ssps = div(sum(contains.(pnames, "continuum.stellar_populations.")), 3)  # Each SSP has 3 parameters (mass, age, metallicity)
    if n_ssps > 0
        # Create the simple stellar population templates with FSPS
        ssp_λ, ages, metals, ssp_templates = generate_stellar_populations(λ, Iunit, cube.lsf, z, out[:cosmology], name)
        ssp_unit = unit(ssp_templates[1])
        # Create a 2D linear interpolation over the ages/metallicities
        ssp_templates = [Spline2D(ustrip.(ages), ustrip.(metals), ustrip.(ssp_templates[:, :, i]), kx=1, ky=1) for i in eachindex(ssp_λ)]
        # Systemic velocity offset
        vsyst_ssp = log(ssp_λ[1]/λ[1]) * C_KMS
        # If age and metallicity are both locked for all templates, we can pre-compute the stellar template and save time during fitting
        if all([params["continuum.stellar_populations.$(i).age"].locked && params["continuum.stellar_populations.$(i).metallicity"].locked 
            for i in 1:n_ssps])
            ssp_matrix = zeros(typeof(1.0*ssp_unit), length(ssp_λ), n_ssps)
            for i in 1:n_ssps
                age = params["continuum.stellar_populations.$(i).age"].value
                met = params["continuum.stellar_populations.$(i).metallicity"].value
                ssp_matrix[:, i] .= [ssp_templates[j](age, met) for j in eachindex(ssp_λ)]
            end
            ssp_templates = ssp_matrix
        end
        ssps = StellarPopulations(ssp_λ, ssp_templates, ssp_unit, vsyst_ssp)
    end

    # Fe II templates
    feii = nothing
    if out[:fit_opt_na_feii] || out[:fit_opt_br_feii]
        # Load in the Fe II templates from Veron-Cetty et al. (2004)
        npad_feii, feii_λ, na_feii_fft, br_feii_fft = generate_feii_templates(λ, Iunit, cube.lsf)
        # Make the object
        vsyst_feii = log(feii_λ[1]/λ[1]) * C_KMS
        feii = FeIITemplates(feii_λ, npad_feii, na_feii_fft, br_feii_fft, vsyst_feii)
    end

    # Velocity resolution
    if n_ssps > 0 || out[:fit_opt_na_feii] || out[:fit_opt_br_feii]
        vres = log(λ[2]/λ[1]) * C_KMS
    end

    # Power laws
    n_power_law = div(sum(contains.(pnames, "continuum.power_law.")), 2)   # Each power law has 2 parameters (amp, slope)
    # Dust continua
    n_dust_cont = div(sum(contains.(pnames, "continuum.dust.")), 2)   # Each dust continuum has 2 parameters (amp, temperature)
    # PAH features
    n_dust_feat = total_num_profiles(model_parameters.dust_features)
    # Absorption features
    n_abs_feat = div(sum(contains.(pnames, "abs_features.")), 4)      # Each absorption feature has 4 parameters (drude profile)
    # Templates
    n_templates = size(out[:templates], 4)

    model_parameters, ssps, feii, n_ssps, n_power_law, n_dust_cont, n_dust_feat, n_abs_feat, n_templates, vres
end


# Helper function for preparing emission line parameters for a CubeFitter object
function cubefitter_prepare_lines(out::Dict, λunit::Unitful.Units, Iunit::Unitful.Units,
    cube::DataCube, region::SpectralRegion)

    # Construct the FitFeatures object for the lines
    lines = construct_line_parameters(out, λunit, Iunit, region)

    # Count the parameters
    n_lines = length(lines.names)
    n_acomps = total_num_profiles(lines)

    # Buffer array for the number of fit line profiles in each spaxel
    n_fit_comps = Dict{Symbol, Matrix{Int}}()
    for name ∈ lines.names
        n_fit_comps[name] = ones(Int, size(cube.I)[1:2])
    end

    # Adjust the sort_line_components option based on the relative flags in the lines object
    relative_flags = BitVector([lines.rel_amp, lines.rel_voff, lines.rel_fwhm])
    if !haskey(out, :sort_line_components)
        out[:sort_line_components] = nothing
        if all(.~relative_flags)
            out[:sort_line_components] = :flux
        end
    elseif !isnothing(out[:sort_line_components])
        out[:sort_line_components] = Symbol(out[:sort_line_components])
    end

    lines, n_lines, n_acomps, n_fit_comps
end


# Helper function for counting the total number of MIR continuum parameters
function count_cont_parameters(model::ModelParameters; split::Bool=false)
    # Continuum parameters
    n_params_cont = length(model.continuum)
    # If the "split" option is true, keep the continuum and PAHs separate, otherwise they are combined
    if !split
        n_params_cont += count_dust_parameters(model)
    end
    n_params_cont
end


function count_dust_parameters(model::ModelParameters)
    length(get_flattened_fit_parameters(model.dust_features))
end


# Helper function for calculating an estimate of optical depth based on continuum slopes
function guess_optical_depth(cube_fitter::CubeFitter, λ::Vector{<:QWave}, I::Vector{<:QSIntensity})

    continuum = model(cube_fitter).continuum
    fopt = fit_options(cube_fitter)

    i1 = nanmedian(I[fopt.guess_tau[1][1] .< λ .< fopt.guess_tau[1][2]])
    i2 = nanmedian(I[fopt.guess_tau[2][1] .< λ .< fopt.guess_tau[2][2]])
    m = (i2 - i1) / (mean(fopt.guess_tau[2]) - mean(fopt.guess_tau[1]))
    contin_unextinct = i1 + m * uconvert(unit(λ[1]), 9.7u"μm" - mean(fopt.guess_tau[1]))  # linear extrapolation over the silicate feature
    contin_extinct = clamp(nanmedian(I[9.6u"μm" .< λ .< 9.8u"μm"]), 0., Inf)
    # Optical depth at 10 microns
    r = contin_extinct / contin_unextinct
    tau_10 = r > 0 ? clamp(-log(r), continuum["extinction.tau_97"].limits...) : 0.
    if !fopt.extinction_screen && r > 0
        # solve nonlinear equation
        f(τ) = r - (1 - exp(-τ[1]))/τ[1]
        try
            soln = nlsolve(f, [tau_10])
            tau_10 = clamp(soln.zero[1], continuum["extinction.tau_97"].limits...)
        catch
            tau_10 = 0.
        end
    end

    tau_10
end


# MIR implementation of the pretty_print_continuum_results function
function pretty_print_results(cube_fitter::CubeFitter, pnames::Vector{String}, popt::Vector{T}, perr::Vector{T}, 
    lb::Vector{T}, ub::Vector{T}, plock::BitVector, tied::Vector{Union{Tie,Nothing}}, I::Vector{<:Real}) where {T<:Number}

    # prettify locked and tied vectors
    locked = ifelse.(plock, "yes", "")
    tie_groups = Vector{String}([!isnothing(tie) ? string(tie.group) : "" for tie in tied])

    # make things have appropriate number of sig figs based on the errors
    _popt = copy(popt); _perr = copy(perr); _lb = copy(lb); _ub = copy(ub)

    round_to_digits(x, y) = round(x, digits=-Int(floor(log10(y))))
    _perr = round_to_digits.(_perr, _perr)
    _popt = round_to_digits.(_popt, _perr)
    _lb = round_to_digits.(_lb, _perr)
    _ub = round_to_digits.(_ub, _perr)

    data = DataFrame(name=pnames, best=_popt, error=_perr, lower=_lb, upper=_ub, locked=locked, tied=tie_groups)
    textwidths = [maximum(textwidth.(string.([data[:, i]; names(data)[i]]))) for i in axes(data, 2)]
    msg = ""
    for (i, header) ∈ enumerate(names(data))
        msg *= rpad(header, textwidths[i]) * "\t"
    end
    msg *= "\n"
    for i ∈ axes(data, 1)
        for j ∈ axes(data, 2)
            msg *= rpad(data[i,j], textwidths[j]) * "\t"
        end
        msg *= "\n"
    end
    @debug msg

    msg
end
