
# Helper function for getting dust feature names and transformation vector for making ParamMaps objects
function _get_dust_feature_names_and_transforms(cf_dustfeat::DustFeatures)

    # Add dust features fitting parameters
    dust_feature_names = String[]
    dust_feature_units = String[]
    dust_feature_labels = String[]
    df_restframe = Int[]
    df_log = Int[]
    df_normalize = Int[]
    dust_feature_names_extra = String[]
    dust_feature_units_extra = String[]
    dust_feature_labels_extra = String[]
    df_extra_restframe = Int[]
    df_extra_log = Int[]
    df_extra_normalize = Int[]
    for (i, n) ∈ enumerate(cf_dustfeat.names)
        dfi = ["dust_features.$(n).amp", "dust_features.$(n).mean", "dust_features.$(n).fwhm"]
        dfu = ["log(erg.s-1.cm-2.Hz-1.sr-1)", "um", "um"]
        dfl = [L"$\log_{10}(I / $ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1})$", L"$\mu$ ($\mu$m)", L"FWHM ($\mu$m)"]
        dfi_restframe = [1, 2, 2]
        dfi_log = [1, 0, 0]
        dfi_normalize = [1, 0, 0]
        if cf_dustfeat.profiles[i] == :PearsonIV
            append!(dfi, ["dust_features.$(n).index", "dust_features.$(n).cutoff"])
            append!(dfu, ["-", "-"])
            append!(dfl, [L"$m$", L"$\nu$"])
            append!(dfi_restframe, [0, 0])
            append!(dfi_log, [0, 0])
            append!(dfi_normalize, [0, 0])
        else
            push!(dfi, "dust_features.$(n).asym")
            push!(dfu, "-")
            push!(dfl, L"$a$")
            push!(dfi_restframe, 0)
            push!(dfi_log, 0)
            push!(dfi_normalize, 0)
        end
        append!(dust_feature_names, dfi)
        append!(dust_feature_units, dfu)
        append!(dust_feature_labels, dfl)
        append!(df_restframe, dfi_restframe)
        append!(df_log, dfi_log)
        append!(df_normalize, dfi_normalize)

        append!(dust_feature_names_extra, ["dust_features.$(n).flux", "dust_features.$(n).eqw", "dust_features.$(n).SNR"])
        append!(dust_feature_units_extra, ["log(erg.s-1.cm-2)", "um", "-"])
        append!(dust_feature_labels_extra, [L"$\log_{10}(F /$ erg s$^{-1}$ cm$^{-2}$)", L"$W_{\rm eq}$ ($\mu$m)", L"$S/N$"])
        append!(df_extra_restframe, [0, 2, 0])
        append!(df_extra_log, [1, 0, 0])
        append!(df_extra_normalize, [0, 0, 0])
    end
    @debug "dust feature maps with keys $dust_feature_names"
    @debug "extra dust feature maps with keys $dust_feature_names_extra"

    dust_feature_names, dust_feature_units, dust_feature_labels, df_restframe, df_log, df_normalize,
        dust_feature_names_extra, dust_feature_units_extra, dust_feature_labels_extra, df_extra_restframe, 
        df_extra_log, df_extra_normalize 
end


"""
    parammaps_empty(shape, n_dust_cont, n_power_law, cf_dustfeat, ab_names, n_lines, n_comps, cf_lines,
        flexible_wavesol)

A constructor function for making a default empty ParamMaps structure with all necessary fields for a given
fit of a MIR DataCube.

# Arguments {S<:Integer}
- `cube_fitter::CubeFitter`: The CubeFitter object
- `shape::Tuple{S,S,S}`: Tuple specifying the 3D shape of the input data cube.
"""
function parammaps_mir_empty(cube_fitter::CubeFitter, shape::Tuple{S,S,S})::ParamMaps where {S<:Integer}

    @debug """\n
    Creating MIRParamMaps struct with shape $shape
    ##############################################
    """

    # Add stellar continuum fitting parameters
    parameter_names = ["continuum.stellar.amp", "continuum.stellar.temp"]
    parameter_units = ["-", "K"]
    parameter_labels = [L"$\log_{10}(A_{*})$",  L"$T$ (K)"]
    restframe_tf = [1, 0]
    log_tf = [1, 0]
    norm_tf = [0, 0]

    # Add dust continuum fitting parameters
    for i ∈ 1:cube_fitter.n_dust_cont
        append!(parameter_names, ["continuum.dust.$(i).amp", "continuum.dust.$(i).temp"])
        append!(parameter_units, ["-", "K"])
        append!(parameter_labels, [L"$\log_{10}(A_{\rm dust})$", L"$T$ (K)"])
        append!(restframe_tf, [1, 0])
        append!(log_tf, [1, 0])
        append!(norm_tf, [0, 0])
    end

    # Add power law fitting parameters
    for p ∈ 1:cube_fitter.n_power_law
        append!(parameter_names, ["continuum.power_law.$(p).amp", "continuum.power_law.$(p).index"])
        append!(parameter_units, ["log(erg.s-1.cm-2.Hz-1.sr-1)", "-"])
        append!(parameter_labels, [L"$\log_{10}(A_{\rm pl} / $ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$)",  L"$\alpha$"])
        append!(restframe_tf, [1, 0])
        append!(log_tf, [1, 0])
        append!(norm_tf, [1, 0])
    end

    # Add extinction fitting parameters
    if cube_fitter.extinction_curve == "decompose"
        append!(parameter_names, ["extinction.N_oli", "extinction.N_pyr", "extinction.N_for"])
        append!(parameter_units, ["log(g.cm-2)", "log(N_oli)", "log(N_oli)"])
        append!(parameter_labels, [L"$\log_{10}(N_{\rm oli} / $ g cm$^{-2}$)", L"$\log_{10}(N_{\rm pyr} / N_{\rm oli}$)",
            L"$\log_{10}(N_{\rm for} / N_{\rm oli}$)"])
        append!(restframe_tf, [0, 0, 0])
        append!(log_tf, [1, 1, 1])
        append!(norm_tf, [0, 0, 0])
    else
        push!(parameter_names, "extinction.tau_9_7")
        push!(parameter_units, "-")
        push!(parameter_labels, L"$\tau_{9.7}$")
        push!(restframe_tf, 0)
        push!(log_tf, 0)
        push!(norm_tf, 0)
    end
    append!(parameter_names, ["extinction.tau_ice", "extinction.tau_ch", "extinction.beta", "extinction.frac"])
    append!(parameter_units, ["-", "-", "-", "-"])
    append!(parameter_labels, [L"$\tau_{\rm ice}$", L"$\tau_{\rm CH}$", L"$\beta$", L"$C_f$"])
    append!(restframe_tf, [0, 0, 0, 0])
    append!(log_tf, [0, 0, 0, 0])
    append!(norm_tf, [0, 0, 0, 0])

    # Add absorption feature fitting parameters
    for n ∈ cube_fitter.abs_features.names
        append!(parameter_names, ["abs_features.$(n).tau", "abs_features.$(n).mean", "abs_features.$(n).fwhm", "abs_features.$(n).asym"])
        append!(parameter_units, ["-", "um", "um", "-"])
        append!(parameter_labels, [L"$\tau$", L"$\mu$ ($\mu$m)", L"FWHM ($\mu$m)", L"$a$"])
        append!(restframe_tf, [0, 2, 2, 0])
        append!(log_tf, [0, 0, 0, 0])
        append!(norm_tf, [0, 0, 0, 0])
    end

    # Add hot dust fitting parameters
    if cube_fitter.fit_sil_emission
        append!(parameter_names, ["continuum.hot_dust.amp", "continuum.hot_dust.temp", "continuum.hot_dust.frac", 
            "continuum.hot_dust.tau_warm", "continuum.hot_dust.tau_cold", "continuum.hot_dust.sil_peak"])
        append!(parameter_units, ["-", "K", "-", "-", "-", "um"])
        append!(parameter_labels, [L"$\log_{10}(A_{\rm sil})$", L"$T$ (K)", L"$C_f$", L"$\tau_{\rm warm}$", 
            L"$\tau_{\rm cold}$", L"$\mu$ ($\mu$m)"])
        append!(restframe_tf, [1, 0, 0, 0, 0, 0])
        append!(log_tf, [1, 0, 0, 0, 0, 0])
        append!(norm_tf, [0, 0, 0, 0, 0, 0])
    end

    # Add template fitting parameters
    for (ni, n) ∈ enumerate(cube_fitter.template_names)
        # template amplitudes are not rest-frame transformed because they are interpreted as multiplying a 
        # spectrum which is already in the frame of interest
        if !cube_fitter.fit_temp_multexp
            for i ∈ 1:cube_fitter.n_channels
                push!(parameter_names, "templates.$(n).amp_$i")
                push!(parameter_units, "-")
                push!(parameter_labels, L"$\log_{10}(A_{\rm template})$")
                push!(restframe_tf, 0)
                push!(log_tf, 1)
                push!(norm_tf, 0)
            end
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
            end
        end
    end

    # Add dust feature fitting parameters
    dust_feature_names, dust_feature_units, dust_feature_labels, df_restframe, df_log, df_normalize, 
        dust_feature_names_extra, dust_feature_units_extra, dust_feature_labels_extra, df_extra_restframe, 
        df_extra_log, df_extra_normalize = _get_dust_feature_names_and_transforms(cube_fitter.dust_features)

    # Add emission line fitting parameters
    line_names, line_names_extra, line_units, line_units_extra, line_labels, line_labels_extra, line_restframe, line_extra_restframe, 
        line_log, line_extra_log, line_normalize, line_extra_normalize, _, _ = _get_line_names_and_transforms(
            cube_fitter.lines, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.flexible_wavesol, cube_fitter.lines_allow_negative, perfreq=0)

    # Add statistics parameters
    statistics_names = ["statistics.chi2", "statistics.dof"]
    statistics_units = ["-", "-"]
    statistics_labels = [L"$\chi^2$", "d.o.f."]
    statistics_restframe = [0, 0]
    statistics_log = [0, 0]
    statistics_normalize = [0, 0]
    @debug "chi^2 map"
    @debug "dof map"

    # Combine everything into big vectors
    parameter_names = [parameter_names; dust_feature_names; line_names; dust_feature_names_extra; line_names_extra; statistics_names]
    parameter_units = [parameter_units; dust_feature_units; line_units; dust_feature_units_extra; line_units_extra; statistics_units]
    parameter_labels = [parameter_labels; dust_feature_labels; line_labels; dust_feature_labels_extra; line_labels_extra; 
        statistics_labels]
    restframe_tf = [restframe_tf; df_restframe; line_restframe; df_extra_restframe; line_extra_restframe; statistics_restframe]
    log_tf = BitVector([log_tf; df_log; line_log; df_extra_log; line_extra_log; statistics_log])
    norm_tf = BitVector([norm_tf; df_normalize; line_normalize; df_extra_normalize; line_extra_normalize; statistics_normalize])
    line_tf = BitVector([contains(pname, "lines") for pname in parameter_names])

    n_params = length(parameter_names)

    # Do not perform any per-wavelength to per-frequency transformations because the MIR is already given per-frequency
    perfreq_tf = falses(n_params)

    @assert n_params == length(parameter_units) == length(parameter_labels) == length(restframe_tf) == length(log_tf) == 
        length(norm_tf) == length(line_tf) == length(perfreq_tf)

    # Initialize a default array of nans to be used as a placeholder for all the other arrays
    # until the actual fitting parameters are obtained
    data = ones(shape[1:2]..., n_params) .* NaN

    ParamMaps{Float64}(data, copy(data), copy(data), parameter_names, parameter_units, parameter_labels, 
        restframe_tf, log_tf, norm_tf, line_tf, perfreq_tf)
end


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
    unobscured_continuum::Array{T, 3}
    obscured_continuum::Array{T, 3}
    stellar::Array{T, 3}
    dust_continuum::Array{T, 4}
    power_law::Array{T, 4}
    dust_features::Array{T, 4}
    abs_features::Array{T, 4}
    extinction::Array{T, 4}
    abs_ice::Array{T, 3}
    abs_ch::Array{T, 3}
    hot_dust::Array{T, 3}
    templates::Array{T, 4}
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
    ab_names::Vector{String}, temp_names::Vector{String}, line_names::Vector{Symbol}, extinction_curve::String, 
    floattype::DataType=Float32)::MIRCubeModel

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
    unobscured_continuum = zeros(floattype, shape2...)
    @debug "unobscured continuum cube"
    obscured_continuum = zeros(floattype, shape2...)
    @debug "obscured continuum cube"
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
    extinction = zeros(floattype, shape2..., extinction_curve == "decompose" ? 4 : 1)
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

    MIRCubeModel(model, unobscured_continuum, obscured_continuum, stellar, dust_continuum, power_law, dust_features, abs_features, 
        extinction, abs_ice, abs_ch, hot_dust, templates, lines)
end


# Helper function for calculating the number of subchannels covered by MIRI observations
function cubefitter_mir_get_n_channels(λ::Vector{<:Real}, z::Real)
    # NOTE: do not use n_channels to count the ACTUAL number of channels/bands in an observation,
    #  as n_channels counts the overlapping regions between channels as separate channels altogether
    #  to allow them to have different normalizations
    n_channels = 0
    channel_masks = []
    ch_edge_sort = sort(channel_edges)
    for i in 2:(length(ch_edge_sort))
        left = ch_edge_sort[i-1]
        right = ch_edge_sort[i]
        ch_mask = left .< (λ .* (1 .+ z)) .< right
        n_region = sum(ch_mask)

        if n_region > 0
            n_channels += 1
            push!(channel_masks, ch_mask)
        end
    end
    # filter out small beginning/end sections
    if sum(channel_masks[1]) < 200
        channel_masks[2] .|= channel_masks[1]
        popfirst!(channel_masks)
        n_channels -= 1
    end
    if sum(channel_masks[end]) < 200
        channel_masks[end-1] .|= channel_masks[end]
        pop!(channel_masks)
        n_channels -= 1
    end

    n_channels, channel_masks
end


# Helper function for preparing continuum and dust feature parameters for 
# a CubeFitter object
function cubefitter_mir_prepare_continuum(λ::Vector{<:Real}, z::Real, out::Dict, n_channels::Integer)

    # Get dust options from the configuration file
    λlim = extrema(λ)
    continuum, dust_features_0, abs_features_0, abs_taus_0 = construct_parameters_mir(out, λlim, n_channels)

    #### PREPARE OUTPUTS ####
    @debug "### Model will include 1 stellar continuum component ###" *
        "\n### at T = $(continuum.T_s.value) K ###"

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
    df_filt = [((minimum(λ)-0.1) < dust_features_0.mean[i].value < (maximum(λ)+0.1)) for i ∈ 1:length(dust_features_0.mean)]
    if !isnothing(out[:user_mask])
        for pair in out[:user_mask]
            df_filt .&= [~(pair[1] < dust_features_0.mean[i].value < pair[2]) for i ∈ 1:length(dust_features_0.mean)]
        end
    end
    dust_features = DustFeatures(dust_features_0.names[df_filt], 
                                dust_features_0.profiles[df_filt],
                                dust_features_0.mean[df_filt],
                                dust_features_0.fwhm[df_filt],
                                dust_features_0.asym[df_filt],
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
    ab_filt = [((minimum(λ)-0.1) < abs_features_0.mean[i].value < (maximum(λ)+0.1)) for i ∈ 1:length(abs_features_0.mean)]
    if !isnothing(out[:user_mask])
        for pair in out[:user_mask]
            ab_filt .&= [~(pair[1] < abs_features_0.mean[i].value < pair[2]) for i ∈ 1:length(abs_features_0.mean)]
        end
    end
    abs_features = DustFeatures(abs_features_0.names[ab_filt],
                                abs_features_0.profiles[ab_filt],
                                abs_features_0.mean[ab_filt],
                                abs_features_0.fwhm[ab_filt],
                                abs_features_0.asym[ab_filt],
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


    if n_templates == 0
        # Ignore any template amplitude entries in the dust.toml options if there are no templates
        continuum = MIRContinuum(continuum.T_s, continuum.T_dc, continuum.α, continuum.τ_97, 
                                    continuum.N_oli, continuum.N_pyr, continuum.N_for, continuum.τ_ice,
                                    continuum.τ_ch, continuum.β, continuum.Cf, continuum.T_hot, continuum.Cf_hot, continuum.τ_warm, 
                                    continuum.τ_cold, continuum.sil_peak, Parameter[])
    end

    # Check for locked tau_CH parameter
    if haskey(out, :fit_ch_abs)
        if !out[:fit_ch_abs]
            continuum.τ_ch.value = 0.
            continuum.τ_ch.locked = true
        else
            continuum.τ_ch.locked = false
        end
    end

    # check for F test for extinction
    if !haskey(out, :F_test_ext)
        F_test_ext = false
    else
        F_test_ext = out[:F_test_ext]
    end

    continuum, dust_features_0, dust_features, abs_features_0, abs_features, abs_taus, 
        n_dust_cont, n_power_law, n_dust_features, n_abs_features,
        F_test_ext
end


# Helper function for counting the total number of MIR continuum parameters
function cubefitter_mir_count_cont_parameters(extinction_curve::String, fit_sil_emission::Bool, fit_temp_multexp::Bool, 
    n_dust_cont::Integer, n_power_law::Integer, n_abs_features::Integer, n_templates::Integer, n_channels::Integer, 
    dust_features::DustFeatures; split::Bool=false)

    n_params_cont = (2+4) + (extinction_curve == "decompose" ? 3 : 1) + 2n_dust_cont + 2n_power_law + 
                    4n_abs_features + (fit_sil_emission ? 6 : 0) + (fit_temp_multexp ? 8 : n_templates*n_channels)
    if !split
        # If split=true, return the index at which the parameters should be split (before the PAHs)
        n_params_cont += 4 * sum(dust_features.profiles .== :Drude) + 5 * sum(dust_features.profiles .== :PearsonIV)
    end

    n_params_cont
end


# MIR implementation of the get_continuum_plimits function
function get_mir_continuum_plimits(cube_fitter::CubeFitter; init::Bool=false, split::Bool=false)

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
    if cube_fitter.lock_hot_dust[1] || cube_fitter.nuc_fit_flag[1]
        dc_lock[1:2] .= true
    end

    df_plim = Tuple{Float64,Float64}[]
    df_lock = Bool[]
    for n in 1:length(dust_features.names)
        append!(df_plim, [amp_df_plim, dust_features.mean[n].limits, dust_features.fwhm[n].limits])
        append!(df_lock, [false, dust_features.mean[n].locked, dust_features.fwhm[n].locked])
        if dust_features.profiles[n] == :PearsonIV
            append!(df_plim, [dust_features.index[n].limits, dust_features.cutoff[n].limits])
            append!(df_lock, [dust_features.index[n].locked, dust_features.cutoff[n].locked])
        else
            push!(df_plim, dust_features.asym[n].limits)
            push!(df_lock, dust_features.asym[n].locked)
        end
    end

    ab_plim = vcat([[tau.limits, mi.limits, fi.limits, ai.locked] for (tau, mi, fi, ai) ∈ 
        zip(abs_taus, abs_features.mean, abs_features.fwhm, abs_features.asym)]...)
    ab_lock = vcat([[tau.locked, mi.locked, fi.locked, ai.locked] for (tau, mi, fi, ai) ∈ 
        zip(abs_taus, abs_features.mean, abs_features.fwhm, abs_features.asym)]...)

    if cube_fitter.extinction_curve != "decompose"
        ext_plim = [continuum.τ_97.limits, continuum.τ_ice.limits, continuum.τ_ch.limits, continuum.β.limits, continuum.Cf.limits]
        ext_lock = [continuum.τ_97.locked, continuum.τ_ice.locked, continuum.τ_ch.locked, continuum.β.locked, continuum.Cf.locked]
    else
        ext_plim = [continuum.N_oli.limits, continuum.N_pyr.limits, continuum.N_for.limits, 
                    continuum.τ_ice.limits, continuum.τ_ch.limits, continuum.β.limits, continuum.Cf.limits]
        ext_lock = [continuum.N_oli.locked, continuum.N_pyr.locked, continuum.N_for.locked, 
                    continuum.τ_ice.locked, continuum.τ_ch.locked, continuum.β.locked, continuum.Cf.locked]
    end

    # Lock tau_9.7 if an extinction map has been provided
    if !isnothing(cube_fitter.extinction_map) && !init
        if cube_fitter.extinction_curve != "decompose"
            ext_lock[1] = true
        else
            ext_lock[1:3] .= true
        end
    end
    # Also lock if force_noext
    # if force_noext
    #     ext_lock[1] = true
    # end
    # # Also lock if the continuum is within 1 std dev of 0
    # if nanmedian(I) ≤ 2nanmedian(σ)
    #     ext_lock[:] .= true
    # end
    # Lock N_pyr and N_for if the option is given
    if cube_fitter.extinction_curve == "decompose" && cube_fitter.decompose_lock_column_densities && !init
        ext_lock[2:3] .= true
    end

    hd_plim = cube_fitter.fit_sil_emission ? [amp_dc_plim, continuum.T_hot.limits, continuum.Cf_hot.limits, 
        continuum.τ_warm.limits, continuum.τ_cold.limits, continuum.sil_peak.limits] : []
    hd_lock = cube_fitter.fit_sil_emission ? [false, continuum.T_hot.locked, continuum.Cf_hot.locked,
        continuum.τ_warm.locked, continuum.τ_cold.locked, continuum.sil_peak.locked] : []
   
    if cube_fitter.fit_temp_multexp
        temp_plim = repeat([(0.0, Inf)], 8)
        temp_lock = init ? BitVector([0,1,0,1,0,1,0,1]) : falses(8)
    else
        temp_plim = [ta.limits for ta in continuum.temp_amp]
        temp_lock = [ta.locked for ta in continuum.temp_amp]
    end
    temp_ind_0 = 1 + length(stellar_lock) + length(dc_lock) + length(pl_lock) + length(ext_lock) + length(ab_lock) + length(hd_lock)
    temp_ind_1 = temp_ind_0 + length(temp_lock) - 1 
    if cube_fitter.tie_template_amps
        tied_pairs = Tuple[]
        for i in (temp_ind_0+1):temp_ind_1
            push!(tied_pairs, (temp_ind_0, i, 1.0))
        end
        tied_indices = Vector{Int}(sort([tp[2] for tp in tied_pairs]))
    else
        tied_pairs = Tuple[]
        tied_indices = Int[]
    end

    if !split
        plims = Vector{Tuple}(vcat(stellar_plim, dc_plim, pl_plim, ext_plim, ab_plim, hd_plim, temp_plim, df_plim))
        lock = BitVector(vcat(stellar_lock, dc_lock, pl_lock, ext_lock, ab_lock, hd_lock, temp_lock, df_lock))
        plims, lock, tied_pairs, tied_indices
    else
        # Split up for the two different stages of continuum fitting -- with templates and then with the PAHs
        plims_1 = Vector{Tuple}(vcat(stellar_plim, dc_plim, pl_plim, ext_plim, ab_plim, hd_plim, temp_plim, [amp_df_plim, amp_df_plim]))
        lock_1 = BitVector(vcat(stellar_lock, dc_lock, pl_lock, ext_lock, ab_lock, hd_lock, temp_lock, [false, false]))
        plims_2 = Vector{Tuple}(df_plim)
        lock_2 = BitVector(df_lock)
        plims_1, plims_2, lock_1, lock_2, tied_pairs, tied_indices
    end
end


# Helper function for calculating an estimate of optical depth based on continuum slopes
function guess_optical_depth(cube_fitter::CubeFitter, λ::Vector{<:Real}, init::Bool)

    continuum = cube_fitter.continuum
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

    tau_guess
end


# Helper function for getting initial MIR parameters based on an input previous parameter cube
function get_mir_continuum_initial_values_from_pcube(cube_fitter::CubeFitter, spaxel::CartesianIndex, n_split::Integer)

    pcube_cont = cube_fitter.p_init_cube_cont
    # Get the coordinates of all spaxels that have fit results
    coords0 = [float.(c.I) for c in CartesianIndices(size(pcube_cont)[1:2]) if !all(isnan.(pcube_cont[c,:]))]
    coords = cube_fitter.p_init_cube_coords
    # Calculate their distances from the current spaxel
    dist = [hypot(spaxel[1]-c[1], spaxel[2]-c[2]) for c in coords]
    closest = coords0[argmin(dist)]
    @debug "Using initial best fit continuum parameters from coordinates $closest --> $(coords[argmin(dist)])"

    p₀ = pcube_cont[Int.(closest)..., :]
    pahtemp = model_pah_residuals(cube_fitter.cube.λ, p₀[(n_split+1):end], cube_fitter.dust_features.profiles, ones(length(cube_fitter.cube.λ)),
        nothing, false)
    pah_frac = repeat([maximum(pahtemp)/2], 2)

    p₀, pah_frac
end


# Helper function for getting initial MIR parameters based on the initial fit
function get_mir_continuum_initial_values_from_previous(cube_fitter::CubeFitter, spaxel::CartesianIndex,
    I::Vector{<:Real}, N::Real, tau_guess::Real)

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
    for di ∈ 1:cube_fitter.n_dust_cont
        p₀[pᵢ] = p₀[pᵢ] * scale 
        if (cube_fitter.lock_hot_dust[1] || cube_fitter.nuc_fit_flag[1]) && isone(di)
            p₀[pᵢ] = 0.
        end
        pᵢ += 2
    end

    # Power law amplitudes (NOT rescaled)
    for _ ∈ 1:cube_fitter.n_power_law
        # p₀[pᵢ] = p₀[pᵢ] 
        pᵢ += 2
    end

    # Set optical depth based on the initial guess or the initial fit (whichever is larger)
    if cube_fitter.extinction_curve != "decompose"
        p₀[pᵢ] = max(cube_fitter.continuum.τ_97.value, p₀[pᵢ])
    end

    # Set τ_9.7 and τ_CH to 0 if the continuum is within 1 std dev of 0
    # lock_abs = false
    # if nanmedian(I) ≤ 2nanmedian(σ)
    #     lock_abs = true
    #     if cube_fitter.extinction_curve != "decompose"
    #         p₀[pᵢ] = 0.
    #         p₀[pᵢ+2] = 0.
    #     else
    #         p₀[pᵢ:pᵢ+2] .= 0.
    #         p₀[pᵢ+4] = 0.
    #     end
    # end

    # Set τ_9.7 to the guess if the guess_tau flag is set
    if !isnothing(cube_fitter.guess_tau) && (cube_fitter.extinction_curve != "decompose")
        p₀[pᵢ] = tau_guess
    end

    # Override if an extinction_map was provided
    if !isnothing(cube_fitter.extinction_map)
        @debug "Using the provided τ_9.7 values from the extinction_map"
        pₑ = [pᵢ]
        if cube_fitter.extinction_curve == "decompose"
            append!(pₑ, [pᵢ+1, pᵢ+2])
        end
        if !isnothing(cube_fitter.cube.voronoi_bins)
            data_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(spaxel)[1])
            for i in eachindex(pₑ)
                p₀[pₑ[i]] = mean(cube_fitter.extinction_map[data_indices, i])
            end
        else
            data_index = spaxel
            for i in eachindex(pₑ)
                p₀[pₑ[i]] = cube_fitter.extinction_map[data_index, i]
            end
        end
    end

    # Do not adjust absorption feature amplitudes since they are multiplicative
    pᵢ += 4 + (cube_fitter.extinction_curve == "decompose" ? 3 : 1)
    for _ ∈ 1:cube_fitter.n_abs_feat
        # if lock_abs
        #     p₀[pᵢ] = 0.
        # end
        pᵢ += 4
    end

    # Hot dust amplitude (rescaled)
    if cube_fitter.fit_sil_emission
        p₀[pᵢ] *= scale
        pᵢ += 6
    end

    # Template amplitudes (not rescaled)
    if cube_fitter.fit_temp_multexp
        tamp = sum(p₀[[pᵢ,pᵢ+2,pᵢ+4,pᵢ+6]]) / 4
        for _ ∈ 1:4
            p₀[pᵢ] = tamp
            pᵢ += 2
        end
    else
        for _ ∈ 1:(cube_fitter.n_templates*cube_fitter.n_channels)
            p₀[pᵢ] = 1/cube_fitter.n_templates
            pᵢ += 1
        end
    end

    # Dust feature amplitudes (not rescaled)
    # for i ∈ 1:cube_fitter.n_dust_feat
    #     pᵢ += 3
    #     if cube_fitter.dust_features.profiles[i] == :PearsonIV
    #         pᵢ += 2
    #     end
    # end

    p₀, pah_frac
end


# Helper function for getting initial MIR parameters based on estimates from the data
function get_mir_continuum_initial_values_from_estimation(cube_fitter::CubeFitter, λ::Vector{<:Real}, 
    I::Vector{<:Real}, N::Real, tau_guess::Real)

    cubic_spline = Spline1D(λ, I, k=3)
    continuum = cube_fitter.continuum

    # Stellar amplitude
    λ_s = minimum(λ) < 5 ? minimum(λ)+0.1 : 5.1
    A_s = clamp(cubic_spline(λ_s) * N / Blackbody_ν(λ_s, continuum.T_s.value), 0., Inf)
    if !cube_fitter.fit_stellar_continuum
        A_s = 0.
    end

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
    if cube_fitter.lock_hot_dust[1] || cube_fitter.nuc_fit_flag[1]
        A_dc[1] = 0.
    end
    
    # Power law amplitudes
    A_pl = [clamp(nanmedian(I), 0., Inf)/exp(-continuum.τ_97.value)/cube_fitter.n_power_law for αi ∈ continuum.α]
    
    # Hot dust amplitude
    hd = silicate_emission(λ, 1.0, continuum.T_hot.value, continuum.Cf_hot.value, continuum.τ_warm.value, 
        continuum.τ_cold.value, continuum.sil_peak.value)
    mhd = argmax(hd)
    A_hd = clamp(cubic_spline(λ[mhd]) * N / hd[mhd] / 5, 0., Inf)

    stellar_pars = [A_s, continuum.T_s.value]
    dc_pars = vcat([[Ai, Ti.value] for (Ai, Ti) ∈ zip(A_dc, continuum.T_dc)]...)
    pl_pars = vcat([[Ai, αi.value] for (Ai, αi) ∈ zip(A_pl, continuum.α)]...)
    
    df_pars = Float64[]
    for n in 1:length(cube_fitter.dust_features.names)
        append!(df_pars, [A_df[n], cube_fitter.dust_features.mean[n].value, cube_fitter.dust_features.fwhm[n].value])
        if cube_fitter.dust_features.profiles[n] == :PearsonIV
            append!(df_pars, [cube_fitter.dust_features.index[n].value, cube_fitter.dust_features.cutoff[n].value])
        else
            push!(df_pars, cube_fitter.dust_features.asym[n].value)
        end
    end
    
    ab_pars = vcat([[Ai, mi.value, fi.value, ai.value] for (Ai, mi, fi, ai) ∈ 
        zip(A_ab, cube_fitter.abs_features.mean, cube_fitter.abs_features.fwhm, cube_fitter.abs_features.asym)]...)
    if cube_fitter.fit_sil_emission
        hd_pars = [A_hd, continuum.T_hot.value, continuum.Cf_hot.value, continuum.τ_warm.value, continuum.τ_cold.value,
            continuum.sil_peak.value]
    else
        hd_pars = []
    end

    if cube_fitter.extinction_curve != "decompose"
        extinction_pars = [continuum.τ_97.value, continuum.τ_ice.value, continuum.τ_ch.value, continuum.β.value, continuum.Cf.value]
    else
        extinction_pars = [continuum.N_oli.value, continuum.N_pyr.value, continuum.N_for.value,
                        continuum.τ_ice.value, continuum.τ_ch.value, continuum.β.value, continuum.Cf.value]
    end
    if !isnothing(cube_fitter.guess_tau) && (cube_fitter.extinction_curve != "decompose")
        extinction_pars[1] = tau_guess
    end

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
    p₀ = Vector{Float64}(vcat(stellar_pars, dc_pars, pl_pars, extinction_pars, ab_pars, hd_pars, temp_pars, df_pars))

    p₀, pah_frac
end


# Helper function for getting MIR parameter step sizes 
function get_mir_continuum_step_sizes(cube_fitter::CubeFitter, λ::Vector{<:Real})

    # Calculate relative step sizes for finite difference derivatives
    dλ = (λ[end] - λ[1]) / length(λ)
    deps = sqrt(eps())
    continuum = cube_fitter.continuum

    stellar_dstep = [deps, 1e-4]
    dc_dstep = vcat([[deps, 1e-4] for _ in continuum.T_dc]...)
    pl_dstep = vcat([[deps, deps] for _ in continuum.α]...)
    df_dstep = Float64[]
    for n in 1:length(cube_fitter.dust_features.names)
        append!(df_dstep, [deps, dλ/10/cube_fitter.dust_features.mean[n].value, dλ/1000/cube_fitter.dust_features.fwhm[n].value])
        if cube_fitter.dust_features.profiles[n] == :PearsonIV
            append!(df_dstep, [deps, deps])
        else
            push!(df_dstep, deps)
        end
    end
    ab_dstep = vcat([[deps, dλ/10/mi.value, dλ/1000/fi.value, deps] for (mi, fi) in zip(cube_fitter.abs_features.mean, cube_fitter.abs_features.fwhm)]...)
    if cube_fitter.fit_sil_emission
        hd_dstep = [deps, 1e-4, deps, deps, deps, dλ/10/continuum.sil_peak.value]
    else
        hd_dstep = []
    end
    extinction_dstep = repeat([deps], cube_fitter.extinction_curve == "decompose" ? 7 : 5)
    temp_dstep = [deps for _ in 1:(cube_fitter.fit_temp_multexp ? 8 : cube_fitter.n_templates*cube_fitter.n_channels)]
    dstep = Vector{Float64}(vcat(stellar_dstep, dc_dstep, pl_dstep, extinction_dstep, ab_dstep, hd_dstep, temp_dstep, df_dstep))

    deps, dstep
end


# MIR implementation of the get_continuum_initial_values function
function get_mir_continuum_initial_values(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real}, 
    N::Real; init::Bool=false, split::Bool=false, force_noext::Bool=false)

    continuum = cube_fitter.continuum
    n_split = cubefitter_mir_count_cont_parameters(cube_fitter.extinction_curve, cube_fitter.fit_sil_emission, 
        cube_fitter.fit_temp_multexp, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_abs_feat, 
        cube_fitter.n_templates, cube_fitter.n_channels, cube_fitter.dust_features; split=true)

    # guess optical depth from the dip in the continuum level
    tau_guess = 0.
    if !isnothing(cube_fitter.guess_tau) && (cube_fitter.extinction_curve != "decompose")
        tau_guess = guess_optical_depth(cube_fitter, λ, init)
    end

    # Check if cube fitter has initial cube
    if !isnothing(cube_fitter.p_init_cube_λ) && !init
        @debug "Using parameter cube best fit continuum parameters..."
        p₀, pah_frac = get_mir_continuum_initial_values_from_pcube(cube_fitter, spaxel, n_split)
    # Check if the cube fitter has initial fit parameters 
    elseif !init
        @debug "Using initial best fit continuum parameters..."
        p₀, pah_frac = get_mir_continuum_initial_values_from_previous(cube_fitter, spaxel, I, N, tau_guess)
    # Otherwise, we estimate the initial parameters based on the data
    else
        @debug "Calculating initial starting points..."
        p₀, pah_frac = get_mir_continuum_initial_values_from_estimation(cube_fitter, λ, I, N, tau_guess)
    end
    if force_noext
        pₑ = [3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law]
        p₀[pₑ] .= 0.
    end

    @debug "Continuum Parameter labels: \n [stellar_amp, stellar_temp, " * 
        join(["dust_continuum_amp_$i, dust_continuum_temp_$i" for i ∈ 1:cube_fitter.n_dust_cont], ", ") * 
        join(["power_law_amp_$i, power_law_index_$i" for i ∈ 1:cube_fitter.n_power_law], ", ") *
        (cube_fitter.extinction_curve == "decompose" ? ", extinction_N_oli, extinction_N_pyr, extinction_N_for" : ", extinction_tau_97") *
        ", extinction_tau_ice, extinction_tau_ch, extinction_beta, extinction_Cf, " *  
        join(["$(ab)_tau, $(ab)_mean, $(ab)_fwhm" for ab ∈ cube_fitter.abs_features.names], ", ") *
        (cube_fitter.fit_sil_emission ? ", hot_dust_amp, hot_dust_temp, hot_dust_covering_frac, hot_dust_tau_warm, hot_dust_tau_cold, hot_dust_sil_peak, " : ", ") *
        (cube_fitter.fit_temp_multexp ? "temp_multexp_amp1, temp_multexp_ind1, temp_multexp_amp2, temp_multexp_ind2, temp_multexp_amp3, temp_multexp_ind3, " * 
        "temp_multexp_amp4, temp_multexp_ind4, " : join(["$(tp)_amp_$i" for i in 1:cube_fitter.n_channels for tp ∈ cube_fitter.template_names], ", ")) *
        join(["$(df)_amp, $(df)_mean, $(df)_fwhm" * (cube_fitter.dust_features.profiles[n] == :PearsonIV ? ", $(df)_index, $(df)_cutoff" : 
            "$(df)_asym") for (n, df) ∈ enumerate(cube_fitter.dust_features.names)], ", ") * "]"
    @debug "Continuum Starting Values: \n $p₀"

    deps, dstep = get_mir_continuum_step_sizes(cube_fitter, λ)
    @debug "Continuum relative step sizes: \n $dstep"

    if !split
        p₀, dstep
    else
        # Step 1: Stellar + Dust blackbodies, 2 new amplitudes for the PAH templates, and the extinction parameters
        pars_1 = vcat(p₀[1:n_split], pah_frac)
        dstep_1 = vcat(dstep[1:n_split], [deps, deps])
        # Step 2: The PAH profile amplitudes, centers, and FWHMs
        pars_2 = p₀[(n_split+1):end]
        dstep_2 = dstep[(n_split+1):end]

        pars_1, pars_2, dstep_1, dstep_2
    end
end


"""
    nuc_temp_fit_minimize_psftemp_amp!(cube_fitter, popt_0)

If fitting the nuclear spectrum with individual PSF template amplitudes, this function cleans the posteriors
by minimizing their mean distance from 1. It does so by moving the amplitude in the PSF template amplitudes to
the amplitudes of the other parts of the continuum. The resulting model should be identical.
"""
function nuc_temp_fit_minimize_psftemp_amp!(cube_fitter::CubeFitter, popt_0::Vector{<:Real})
    p_temp_0 = 1 + 2 + 2*cube_fitter.n_dust_cont + 2*cube_fitter.n_power_law + 4 + (cube_fitter.extinction_curve == "decompose" ? 3 : 1) +
        3*cube_fitter.n_abs_feat + (cube_fitter.fit_sil_emission ? 6 : 0)
    p_temp_1 = p_temp_0 + (cube_fitter.fit_temp_multexp ? 8 : cube_fitter.n_templates*cube_fitter.n_channels) - 1

    resid_amp = nanmean(popt_0[p_temp_0:p_temp_1])
    # normalize the residual amplitude from the PSF normalizations
    popt_0[p_temp_0:p_temp_1] ./= resid_amp

    # add the same amount of amplitude back into the continuum
    popt_0[1] *= resid_amp
    pᵢ = 3
    for _ in 1:cube_fitter.n_dust_cont
        popt_0[pᵢ] *= resid_amp
        pᵢ += 2
    end
    for _ in 1:cube_fitter.n_power_law
        popt_0[pᵢ] *= resid_amp
        pᵢ += 2
    end
    pᵢ += cube_fitter.extinction_curve == "decompose" ? 3 : 1
    pᵢ += 4
    pᵢ += 3*cube_fitter.n_abs_feat
    if cube_fitter.fit_sil_emission
        popt_0[pᵢ] *= resid_amp
        pᵢ += 6
    end
    pᵢ += cube_fitter.fit_temp_multexp ? 8 : cube_fitter.n_channels*cube_fitter.n_templates
    for prof in enumerate(cube_fitter.dust_features.profiles)
        popt_0[pᵢ] *= resid_amp
        if prof == :Drude
            pᵢ += 4
        elseif prof == :PearsonIV
            pᵢ += 5
        end
    end
    return popt_0
end


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
    if cube_fitter.extinction_curve != "decompose"
        msg *= "τ_9.7: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t Limits: " *
            "($(@sprintf "%.2f" continuum.τ_97.limits[1]), $(@sprintf "%.2f" continuum.τ_97.limits[2]))" * 
            (continuum.τ_97.locked ? " (fixed)" : "") * "\n"
        pᵢ += 1
    else
        msg *= "N_oli: \t\t\t\t $(@sprintf "%.2g" popt[pᵢ]) +/- $(@sprintf "%.2g" perr[pᵢ]) [-] \t Limits: " *
            "($(@sprintf "%.2g" continuum.N_oli.limits[1]), $(@sprintf "%.2g" continuum.N_oli.limits[2]))" * 
            (continuum.N_oli.locked ? " (fixed)" : "") * "\n"
        msg *= "N_pyr: \t\t\t\t $(@sprintf "%.2g" popt[pᵢ+1]) +/- $(@sprintf "%.2g" perr[pᵢ+1]) [-] \t Limits: " *
            "($(@sprintf "%.2g" continuum.N_pyr.limits[1]), $(@sprintf "%.2g" continuum.N_pyr.limits[2]))" * 
            (continuum.N_pyr.locked ? " (fixed)" : "") * "\n"
        msg *= "N_for: \t\t\t\t $(@sprintf "%.2g" popt[pᵢ+2]) +/- $(@sprintf "%.2g" perr[pᵢ+2]) [-] \t Limits: " *
            "($(@sprintf "%.2g" continuum.N_for.limits[1]), $(@sprintf "%.2g" continuum.N_for.limits[2]))" * 
            (continuum.N_for.locked ? " (fixed)" : "") * "\n"
        pᵢ += 3
    end
    msg *= "τ_ice: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.τ_ice.limits[1]), $(@sprintf "%.2f" continuum.τ_ice.limits[2]))" *
        (continuum.τ_ice.locked ? " (fixed)" : "") * "\n"
    msg *= "τ_ch: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+1]) +/- $(@sprintf "%.2f" perr[pᵢ+1]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.τ_ch.limits[1]), $(@sprintf "%.2f" continuum.τ_ch.limits[2]))" *
        (continuum.τ_ch.locked ? " (fixed)" : "") * "\n"
    msg *= "β: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+2]) +/- $(@sprintf "%.2f" perr[pᵢ+2]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.β.limits[1]), $(@sprintf "%.2f" continuum.β.limits[2]))" * 
        (continuum.β.locked ? " (fixed)" : "") * "\n"
    msg *= "Cf: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+3]) +/- $(@sprintf "%.2f" perr[pᵢ+3]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.Cf.limits[1]), $(@sprintf "%.2f" continuum.Cf.limits[2]))" * 
        (continuum.Cf.locked ? " (fixed)" : "") * "\n"
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
        msg *= "$(ab)_asym: \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3]) [-] \t Limits: " *
            "($(@sprintf "%.3f" cube_fitter.abs_features.asym[j].limits[1]), $(@sprintf "%.3f" cube_fitter.abs_features.asym[j].limits[2]))" *
            (cube_fitter.abs_features.asym[j].locked ? " (fixed)" : "") * "\n"
        msg *= "\n"
        pᵢ += 4
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
    if !cube_fitter.fit_temp_multexp
        for (q, tp) ∈ enumerate(cube_fitter.template_names)
            for qi ∈ 1:cube_fitter.n_channels
                msg *= "$(tp)_amp_$qi:\t\t\t $(@sprintf "%.5f" popt[pᵢ]) +/- $(@sprintf "%.5f" perr[pᵢ]) [x norm] \t Limits: (0, 1)\n"
                pᵢ += 1
            end
        end
    else
        for q ∈ 1:4
            msg *= "temp_multexp_amp$q:\t\t\t $(@sprintf "%.5f" popt[pᵢ]) +/- $(@sprintf "%.5f" perr[pᵢ]) [x norm] \t Limits: (0, 1)\n"
            msg *= "temp_multexp_ind$q:\t\t\t $(@sprintf "%.5f" popt[pᵢ+1]) +/- $(@sprintf "%.5f" perr[pᵢ+1]) [-] \t Limits: (0, 1)\n"
            pᵢ += 2
        end
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
        else
            msg *= "$(df)_asym:  \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3]) [-] \t Limits: " *
                "($(@sprintf "%.3f" cube_fitter.dust_features.asym[j].limits[1]), $(@sprintf "%.3f" cube_fitter.dust_features.asym[j].limits[2]))" * 
                (cube_fitter.dust_features.asym[j].locked ? " (fixed)" : "") * "\n"
            pᵢ += 1
        end
        msg *= "\n"
        pᵢ += 3
    end
    msg *= "######################################################################"
    @debug msg

    msg

end
