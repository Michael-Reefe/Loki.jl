
"""
    parammaps_empty(shape, n_ssps, n_power_law, n_lines, n_comps, cf_lines, flexible_wavesol)

A constructor function for making a default empty OpticalParamMaps structure with all necessary fields for a given
fit of an optical DataCube.

# Arguments {S<:Integer}
- `cube_fitter::CubeFitter`: The CubeFitter object
- `shape::Tuple{S,S,S}`: Tuple specifying the 3D shape of the input data cube.
"""
function parammaps_opt_empty(cube_fitter::CubeFitter, shape::Tuple{S,S,S})::ParamMaps where {S<:Integer}

    @debug """\n
    Creating OpticalParamMaps struct with shape $shape
    ##################################################
    """

    # Add stellar population fitting parameters
    parameter_names = String[]
    parameter_units = String[]
    parameter_labels = String[]
    restframe_tf = Int[]
    log_tf = Int[]
    norm_tf = Int[]
    perfreq_tf = Int[]
    for i ∈ 1:cube_fitter.n_ssps
        si = ["continuum.stellar_populations.$(i).mass", "continuum.stellar_populations.$(i).age", "continuum.stellar_populations.$(i).metallicity"]
        append!(parameter_names, si)
        append!(parameter_units, ["log(Msun)", "Gyr", "[M/H]"])
        append!(parameter_labels, [cube_fitter.cosmology.h ≈ 1.0 ? L"$\log_{10}(Mh^2 / M_{\odot})$" : L"$\log_{10}(M / M_{\odot})$",
            L"$t$ (Gyr)", L"[M$/$H]"])
        append!(restframe_tf, [1, 0, 0])
        append!(log_tf, [1, 0, 0])
        append!(norm_tf, [1, 0, 0])
        append!(perfreq_tf, [0, 0, 0])
    end

    # Add stellar kinematics
    append!(parameter_names, ["continuum.stellar_kinematics.vel", "continuum.stellar_kinematics.vdisp"])
    append!(parameter_units, ["km/s", "km/s"])
    append!(parameter_labels, [L"$v_*$ (km s$^{-1}$)", L"$\sigma_*$ (km s$^{-1}$)"])
    append!(restframe_tf, [0, 0])
    append!(log_tf, [0, 0])
    append!(norm_tf, [0, 0])
    append!(perfreq_tf, [0, 0])

    # Add attenuation parameters
    append!(parameter_names, ["attenuation.E_BV", "attenuation.E_BV_factor"])
    append!(parameter_units, ["mag", "-"])
    append!(parameter_labels, [L"$E(B-V)_{\rm gas}$", L"$E(B-V)_{\rm stars}/E(B-V)_{\rm gas}$"])
    append!(restframe_tf, [0, 0])
    append!(log_tf, [0, 0])
    append!(norm_tf, [0, 0])
    append!(perfreq_tf, [0, 0])
    if cube_fitter.fit_uv_bump
        push!(parameter_names, "attenuation.delta_UV")
        push!(parameter_units, "-")
        push!(parameter_labels, L"$\delta_{\rm UV}$")
        push!(restframe_tf, 0)
        push!(log_tf, 0)
        push!(norm_tf, 0)
        push!(perfreq_tf, 0)
    end
    if cube_fitter.fit_covering_frac
        push!(parameter_names, "attenuation.frac")
        push!(parameter_units, "-")
        push!(parameter_labels, L"$C_f$")
        push!(restframe_tf, 0)
        push!(log_tf, 0)
        push!(norm_tf, 0)
        push!(perfreq_tf, 0)
    end

    # Add Fe II kinematics
    if cube_fitter.fit_opt_na_feii 
        append!(parameter_names, ["continuum.feii.na.amp", "continuum.feii.na.vel", "continuum.feii.na.vdisp"])
        append!(parameter_units, ["log(erg.s-1.cm-2.Hz-1.sr-1)", "km/s", "km/s"])
        append!(parameter_labels, [L"$\log_{10}(I / $ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1})$", L"$v$ (km s$^{-1}$)",
            L"$\sigma$ (km s$^{-1}$)"])
        append!(restframe_tf, [1, 0, 0])
        append!(log_tf, [1, 0, 0])
        append!(norm_tf, [1, 0, 0])
        append!(perfreq_tf, [1, 0, 0])
    end
    if cube_fitter.fit_opt_br_feii
        append!(parameter_names, ["continuum.feii.br.amp", "continuum.feii.br.vel", "continuum.feii.br.vdisp"])
        append!(parameter_units, ["log(erg.s-1.cm-2.Hz-1.sr-1)", "km/s", "km/s"])
        append!(parameter_labels, [L"$\log_{10}(I / $ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1})$", L"$v$ (km s$^{-1}$)",
            L"$\sigma$ (km s$^{-1}$)"])
        append!(restframe_tf, [1, 0, 0])
        append!(log_tf, [1, 0, 0])
        append!(norm_tf, [1, 0, 0])
        append!(perfreq_tf, [1, 0, 0])
    end

    # Add power laws
    for i ∈ 1:cube_fitter.n_power_law
        append!(parameter_names, ["continuum.power_law.$(i).amp", "continuum.power_law.$(i).index"])
        append!(parameter_units, ["log(erg.s-1.cm-2.Hz-1.sr-1)", "-"])
        append!(parameter_labels, [L"$\log_{10}(I / $ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1})$", L"$\alpha$"])
        append!(restframe_tf, [1, 0])
        append!(log_tf, [1, 0])
        append!(norm_tf, [1, 0])
        append!(perfreq_tf, [1, 0])
    end

    # Add template fitting parameters
    for (ni, n) ∈ enumerate(cube_fitter.template_names)
        if !cube_fitter.fit_temp_multexp
            push!(parameter_names, "templates.$(n).amp_1")
            push!(parameter_units, "-")
            push!(parameter_labels, L"$\log_{10}(A_{\rm template})$")
            push!(restframe_tf, 0)
            push!(log_tf, 1)
            push!(norm_tf, 0)
            push!(perfreq_tf, 0)
        else
            if ni > 1
                break
            end 
            for i ∈ 1:4
                append!(parameter_names, ["templates.amp_$i", "templates.index_$i"])
                append!(parameter_units, ["-", "-"])
                append!(parameter_labels, [L"$\log_{10}(A_{\rm template})$", L"$b$"])
                append!(restframe_tf, [0, 0])
                append!(log_tf, [1, 0])
                append!(norm_tf, [0, 0])
                append!(perfreq_tf, [0, 0])
            end
        end
    end

    line_names, line_names_extra, line_units, line_units_extra, line_labels, line_labels_extra, line_restframe, line_extra_restframe, 
        line_log, line_extra_log, line_normalize, line_extra_normalize, line_perfreq, line_extra_perfreq = 
            _get_line_names_and_transforms(cube_fitter.lines, cube_fitter.n_lines, cube_fitter.n_comps, 
                cube_fitter.flexible_wavesol, perfreq=1)

    statistics_names = ["statistics.chi2", "statistics.dof"]
    statistics_units = ["-", "-"]
    statistics_labels = [L"$\chi^2$", "d.o.f."]
    statistics_restframe = [0, 0]
    statistics_log = [0, 0]
    statistics_normalize = [0, 0]
    statistics_perfreq = [0, 0]
    @debug "chi^2 map"
    @debug "dof map"

    parameter_names = [parameter_names; line_names; line_names_extra; statistics_names]
    parameter_units = [parameter_units; line_units; line_units_extra; statistics_units]
    parameter_labels = [parameter_labels; line_labels; line_labels_extra; statistics_labels]
    restframe_tf = [restframe_tf; line_restframe; line_extra_restframe; statistics_restframe]
    log_tf = BitVector([log_tf; line_log; line_extra_log; statistics_log])
    norm_tf = BitVector([norm_tf; line_normalize; line_extra_normalize; statistics_normalize])
    line_tf = BitVector([contains(pname, "lines") for pname in parameter_names])
    perfreq_tf = BitVector([perfreq_tf; line_perfreq; line_extra_perfreq; statistics_perfreq])

    n_params = length(parameter_names)

    @assert n_params == length(parameter_units) == length(parameter_labels) == length(restframe_tf) == length(log_tf) == 
        length(norm_tf) == length(line_tf) == length(perfreq_tf)

    # Initialize a default array of nans to be used as a placeholder for all the other arrays
    # until the actual fitting parameters are obtained
    data = ones(shape[1:2]..., n_params) .* NaN

    ParamMaps(data, copy(data), copy(data), parameter_names, parameter_units, parameter_labels, 
        restframe_tf, log_tf, norm_tf, line_tf, perfreq_tf)
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
- `templates::Array{T, 4}`: The generic template profiles.
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
    templates::Array{T, 4}
    lines::Array{T, 4}

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
- `temp_names::Vector{String}`: List of names of generic templates in the fit, i.e. "nuclear", ...
- `floattype::DataType=Float32`: The type of float to use in the arrays. Should ideally be the same as the input data,
which for JWST is Float32.
"""
function cubemodel_empty(shape::Tuple, n_ssps::Integer, n_power_law::Integer, line_names::Vector{Symbol}, temp_names::Vector{String}, 
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
    templates = zeros(floattype, shape2..., length(temp_names))
    @debug "templates comp cube"
    lines = zeros(floattype, shape2..., length(line_names))
    @debug "lines comp cubes"

    OpticalCubeModel(model, stellar, na_feii, br_feii, power_law, attenuation_stars, attenuation_gas, templates, lines)

end


# Helper function for preparing continuum and SSP parameters for 
# a CubeFitter object
function cubefitter_opt_prepare_continuum(λ::Vector{<:Real}, z::Real, out::Dict, name::String, cube::DataCube)

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

    n_templates = size(out[:templates], 4)

    if n_templates == 0
        # Ignore any template amplitude entries in the dust.toml options if there are no templates
        continuum = OpticalContinuum(continuum.ssp_ages, continuum.ssp_metallicities, continuum.stel_vel,
            continuum.stel_vdisp, continuum.na_feii_vel, continuum.na_feii_vdisp, continuum.br_feii_vel,
            continuum.br_feii_vdisp, continuum.α, continuum.E_BV, continuum.E_BV_factor, continuum.δ_uv,
            continuum.frac, Parameter[])
    end

    # If we are using AGN templates, lock the hot dust component to 0
    if !haskey(out, :lock_hot_dust)
        lock_hot_dust = n_templates > 0
    else
        lock_hot_dust = out[:lock_hot_dust]
    end

    continuum, ssp_λ, ssp_templates, feii_templates_fft, npad_feii, vres, vsyst_ssp, vsyst_feii,
        n_ssps, n_power_law, n_templates
end


# Total number of parameters for the continuum and line fits
function cubefitter_opt_count_cont_parameters(fit_opt_na_feii::Bool, fit_opt_br_feii::Bool, fit_uv_bump::Integer,
    fit_covering_frac::Bool, n_ssps::Integer, n_power_law::Integer, n_templates::Integer)

    n_params_cont = 3n_ssps + 2 + 2 + 2n_power_law + n_templates
    if fit_opt_na_feii
        n_params_cont += 3
    end
    if fit_opt_br_feii
        n_params_cont += 3
    end
    if fit_uv_bump
        n_params_cont += 1
    end
    if fit_covering_frac
        n_params_cont += 1
    end

    n_params_cont
end



# Optical implementation of the get_continuum_plimits function
function get_opt_continuum_plimits(cube_fitter::CubeFitter, λ::Vector{<:Real}, I::Vector{<:Real}, init::Bool,
    nuc_temp_fit::Bool)

    continuum = cube_fitter.continuum

    amp_ssp_plim = (0., Inf)
    amp_pl_plim = (0., Inf)
    age_univ = age(u"Gyr", cube_fitter.cosmology, cube_fitter.z).val
    age_lim = [(ai.limits[1], clamp(ai.limits[2], 0., age_univ)) for ai in continuum.ssp_ages]

    ssp_plim = vcat([[amp_ssp_plim, ai, zi.limits] for (ai, zi) in zip(age_lim, continuum.ssp_metallicities)]...)
    # if !init
    #     ssp_locked = vcat([[false, true, true] for _ in 1:cube_fitter.n_ssps]...)
    # else
    #     ssp_locked = vcat([[false, ai.locked, zi.locked] for (ai, zi) in zip(continuum.ssp_ages, continuum.ssp_metallicities)]...)
    # end
    ssp_locked = vcat([[false, ai.locked, zi.locked] for (ai, zi) in zip(continuum.ssp_ages, continuum.ssp_metallicities)]...)

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

    if cube_fitter.fit_temp_multexp
        temp_plim = repeat([(0.0, Inf)], 8)
        temp_lock = init ? BitVector([0,1,0,1,0,1,0,1]) : falses(8)
    else
        temp_plim = [ta.limits for ta in continuum.temp_amp]
        temp_lock = [ta.locked for ta in continuum.temp_amp]
    end

    plims = Vector{Tuple}(vcat(ssp_plim, stel_kin_plim, atten_plim, feii_plim, pl_plim, temp_plim))
    lock = BitVector(vcat(ssp_locked, stel_kin_locked, atten_locked, feii_locked, pl_locked, temp_lock))
    tied_pairs = Tuple[]
    tied_indices = Int[]

    if nuc_temp_fit 
        # for optical fits there should be no channel splits so just lock the PSF amplitude to 1 during the nuclear fit
        lock[end-cube_fitter.n_templates+1:end] .= 1
    end

    plims, lock, tied_pairs, tied_indices
end


# Helper function for getting optical parameters based on the initial fit
function get_opt_continuum_initial_values_from_previous(cube_fitter::CubeFitter, spaxel::CartesianIndex, 
    continuum::OpticalContinuum, λ::Vector{<:Real}, I::Vector{<:Real})

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

    pₑ = 1 + 3cube_fitter.n_ssps + 2
    ebv_orig = p₀[pₑ]
    ebv_factor = p₀[pₑ+1]

    if !isnothing(cube_fitter.extinction_map)
        @debug "Using the provided E(B-V) values from the extinction_map"
        if !isnothing(cube_fitter.cube.voronoi_bins)
            data_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(spaxel)[1])
            ebv_new = mean(cube_fitter.extinction_map[data_indices, 1])
        else
            data_index = spaxel
            ebv_new = cube_fitter.extinction_map[data_index, 1]
        end
        ebv_factor_new = continuum.E_BV_factor.value
    else
        # Otherwise always start at an E(B-V) of some small value
        ebv_new = 0.01
        ebv_factor_new = continuum.E_BV_factor.value
    end

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

    # Template amplitudes (not rescaled)
    if cube_fitter.fit_temp_multexp
        tamp = sum(p₀[[pᵢ,pᵢ+2,pᵢ+4,pᵢ+6]]) / 4
        for _ ∈ 1:4
            p₀[pᵢ] = tamp
            pᵢ += 2
        end
    else
        for _ ∈ 1:(cube_fitter.n_templates)
            p₀[pᵢ] = 1/cube_fitter.n_templates
            pᵢ += 1
        end
    end

    p₀
end


# Helper function for getting optical continuum parameters based on estimates from the data
function get_opt_continuum_initial_values_from_estimation(cube_fitter::CubeFitter, continuum::OpticalContinuum,
    λ::Vector{<:Real}, I::Vector{<:Real})

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

    # SSP ages
    a_ssps = copy([ai.value for ai in continuum.ssp_ages])
    for i in eachindex(a_ssps)
        if iszero(a_ssps[i])
            # take 0 to mean that we should guess the age based on the redshift
            a_ssps[i] = age(u"Gyr", cube_fitter.cosmology, cube_fitter.z).val - 0.1
        end
    end

    ssp_pars = vcat([[mi, ai, zi.value] for (mi, ai, zi) in zip(m_ssps, a_ssps, continuum.ssp_metallicities)]...)

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

    # Templates
    if cube_fitter.fit_temp_multexp
        temp_pars = [0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0]
    else
        temp_pars = [ta.value for ta in continuum.temp_amp]
    end
    # apply the nuclear template amplitudes for the initial fit
    if cube_fitter.nuc_fit_flag[1]
        temp_pars ./= cube_fitter.nuc_temp_amps
    end

    # Initial parameter vector
    p₀ = Vector{Float64}(vcat(ssp_pars, stel_kin_pars, atten_pars, feii_pars, pl_pars, temp_pars))

    p₀
end


# Helper function for getting optical parameter step sizes 
function get_opt_continuum_step_sizes(cube_fitter::CubeFitter, continuum::OpticalContinuum)

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

    temp_dstep = [deps for _ in 1:(cube_fitter.fit_temp_multexp ? 8 : cube_fitter.n_templates)]

    dstep = Vector{Float64}(vcat(ssp_dstep, stel_kin_dstep, atten_dstep, feii_dstep, pl_dstep, temp_dstep))

    dstep
end


# Optical implementation of the get_continuum_initial_values function
function get_opt_continuum_initial_values(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real}, 
    N::Real, init::Bool, nuc_temp_fit::Bool)

    continuum = cube_fitter.continuum

    # Check if the cube fitter has initial fit parameters 
    if !init
        @debug "Using initial best fit continuum parameters..."
        p₀ = get_opt_continuum_initial_values_from_previous(cube_fitter, spaxel, continuum, λ, I)
    else
        @debug "Calculating initial starting points..." 
        p₀ = get_opt_continuum_initial_values_from_estimation(cube_fitter, continuum, λ, I)
    end

    if nuc_temp_fit
        p₀[end-cube_fitter.n_templates+1:end] .= 1.0
    end

    @debug "Continuum Parameter labels: \n [" *
        join(["SSP_$(i)_mass, SSP_$(i)_age, SSP_$(i)_metallicity" for i in 1:cube_fitter.n_ssps], ", ") * 
        "stel_vel, stel_vdisp, " * 
        "E_BV, E_BV_factor, " * (cube_fitter.fit_uv_bump ? "delta_uv, " : "") *
        (cube_fitter.fit_covering_frac ? "covering_frac, " : "") * 
        (cube_fitter.fit_opt_na_feii ? "na_feii_amp, na_feii_vel, na_feii_vdisp, " : "") *
        (cube_fitter.fit_opt_br_feii ? "br_feii_amp, br_feii_vel, br_feii_vdisp, " : "") *
        join(["power_law_$(j)_amp, power_law_$(j)_index, " for j in 1:cube_fitter.n_power_law], ", ") * 
        (cube_fitter.fit_temp_multexp ? "temp_multexp_amp1, temp_multexp_ind1, temp_multexp_amp2, " * 
        "temp_multexp_ind2, temp_multexp_amp3, temp_multexp_ind3, temp_multexp_amp4, temp_multexp_ind4, " : 
        join(["$(tp)_amp_1" for tp ∈ cube_fitter.template_names], ", ")) "]"

    dstep = get_opt_continuum_step_sizes(cube_fitter, continuum) 

    @debug "Continuum Starting Values: \n $p₀"
    @debug "Continuum relative step sizes: \n $dstep"

    p₀, dstep
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
    msg *= "\n#> TEMPLATES <#\n"
    if !cube_fitter.fit_temp_multexp
        for (q, tp) ∈ enumerate(cube_fitter.template_names)
            msg *= "$(tp)_amp:\t\t\t $(@sprintf "%.5f" popt[pᵢ]) +/- $(@sprintf "%.5f" perr[pᵢ]) [x norm] \t Limits: (0, 1)\n"
            pᵢ += 1
        end
    else
        for q ∈ 1:4
            msg *= "temp_multexp_amp$q:\t\t\t $(@sprintf "%.5f" popt[pᵢ]) +/- $(@sprintf "%.5f" perr[pᵢ]) [x norm] \t Limits: (0, 1)\n"
            msg *= "temp_multexp_ind$q:\t\t\t $(@sprintf "%.5f" popt[pᵢ+1]) +/- $(@sprintf "%.5f" perr[pᵢ+1]) [-] \t Limits: (0, 1)\n"
            pᵢ += 2
        end
    end
    msg *= "\n"
    msg *= "######################################################################"
    @debug msg

    msg 

end
