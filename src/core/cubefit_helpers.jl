
# Helper function for preparing continuum and dust feature parameters for 
# a CubeFitter object
function cubefitter_prepare_continuum(λ::Vector{<:QWave}, z::Real, out::Dict, λunit::Unitful.Units, 
    Iunit::Unitful.Units, region::SpectralRegion, name::String, cube::DataCube, 
    custom_stellar_template_wave::Union{Nothing,Vector{<:Real}}, custom_stellar_templates::Union{Nothing,Array{<:Real,2}})

    # Construct the ModelParameters object
    model_parameters = construct_model_parameters(out, λunit, Iunit, region, z)
    vres = NaN*u"km/s"   # (vres doesnt make sense if the wavelength vector isnt logarithmic)

    # Count a few different parameters 
    params = get_flattened_fit_parameters(model_parameters)
    pnames = params.names

    # Stellar populations
    ssps = nothing
    n_ssps = 0
    if out[:fit_stellar_continuum]
        # Create the simple stellar population templates with FSPS
        ssp_λ, ages, metals, ssp_templates = generate_stellar_populations(λ, Iunit, cube.lsf, z, out[:stellar_template_type], 
            out[:cosmology], out[:ssps], out[:stars], name, out[:user_mask], custom_stellar_template_wave, custom_stellar_templates)
        # flatten template array
        ssp_temp_flat = reshape(ssp_templates, size(ssp_templates,1), :)
        # 2nd axis ordering: (age1_z1, age2_z1, age3_z1, ..., age1_z2, age2_z2, age3_z2, ...)
        n_ssps = size(ssp_temp_flat, 2)
        # Systemic velocity offset
        vsyst_ssp = [log(ssp_λ[1]/λ[1]) * C_KMS]
        # add velocity offsets for each separate region
        for gap in region.gaps
            push!(vsyst_ssp, log(ssp_λ[1]/λ[λ .> gap[2]][1]) * C_KMS)
        end
        # Build a StellarPopulations object
        ssps = StellarPopulations(ssp_λ, ages, collect(metals), ssp_temp_flat, vsyst_ssp)
    end

    # Fe II templates
    feii = nothing
    if out[:fit_opt_na_feii] || out[:fit_opt_br_feii]
        # Load in the Fe II templates from Veron-Cetty et al. (2004)
        npad_feii, feii_λ, na_feii_fft, br_feii_fft = generate_feii_templates(λ, Iunit, cube.lsf)
        # Make the object
        vsyst_feii = [log(feii_λ[1]/λ[1]) * C_KMS]
        for gap in region.gaps
            push!(vsyst_feii, log(feii_λ[1]/λ[λ .> gap[2]][1]) * C_KMS)
        end
        feii = FeIITemplates(feii_λ, npad_feii, na_feii_fft, br_feii_fft, vsyst_feii)
    end

    # Velocity resolution
    if n_ssps > 0 || out[:fit_opt_na_feii] || out[:fit_opt_br_feii]
        vres = log(λ[2]/λ[1]) * C_KMS
        @assert isapprox(vres, (log(λ[end]/λ[end-1]) * C_KMS), rtol=1e-3)
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
    if !haskey(out, :sort_line_components)
        relative_flags = BitVector([lines.config.rel_amp, lines.config.rel_voff, lines.config.rel_fwhm])
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
function guess_optical_depth(cube_fitter::CubeFitter, λ::Vector{<:QWave}, I::Vector{<:Real})

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


"""
    get_continuum_parameter_limits(cube_fitter, I, σ; init, force_noext)

Get the continuum limits and locked vectors for a given CubeFitter object.
"""
function get_continuum_parameter_limits(cube_fitter::CubeFitter, I::Vector{<:Real}, σ::Vector{<:Real}; 
    init::Bool=false, force_noext::Bool=false, split::Bool=false)

    continuum = model(cube_fitter).continuum
    dust_features = model(cube_fitter).dust_features
    fopt = fit_options(cube_fitter)

    # Get the default lower/upper limits from the model object
    pnames = continuum.names
    plims = continuum.limits
    plock = continuum.locked
    df_params = get_flattened_fit_parameters(dust_features)
    df_names = df_params.names
    df_plims = df_params.limits
    df_plock = df_params.locked

    # Make modifications at the individual spaxel level (this doesnt change the model object since these are copies)

    # Lock E(B-V) if an extinction map has been provided
    ext_criterion = force_noext || nanmedian(I) ≤ nanmedian(σ)
    if !isnothing(fopt.ebv_map) || ext_criterion
        prefix = "extinction."
        pwhere = prefix .* ["E_BV", "E_BV_factor"]
        inds = fast_indexin(pwhere, pnames)
        plock[inds] .= true
    end
    if (!isnothing(fopt.sil_abs_map) && !init) || ext_criterion
        if fopt.silicate_absorption == "decompose"
            pwhere = ["extinction.N_oli"]
        else
            pwhere = ["extinction.tau_97"]
        end
        inds = fast_indexin(pwhere, pnames)
        plock[inds] .= true
    end
    # Lock the hottest dust component if some flags have been set
    if fopt.lock_hot_dust
        pwhere = "continuum.dust.1." .* ["amp", "temp"]
        inds = fast_indexin(pwhere, pnames)
        plock[inds] .= true
    end
    # Lock N_pyr and N_for if the option is given
    if fopt.silicate_absorption == "decompose" && fopt.decompose_lock_column_densities && !init
        pwhere = "extinction." .* ["N_pyr", "N_for"]
        inds = fast_indexin(pwhere, pnames)
        plock[inds] .= true
    end
    # Lock template amplitudes during the initial fit
    # if !fopt.fit_temp_multexp && init
    #     for tname in cube_fitter.template_names
    #         for n in 1:nchannels(cube_fitter.spectral_region)
    #             ind = fast_indexin("templates.$(tname).amp_$n", pnames)
    #             plock[ind] = true
    #         end
    #     end
    # end

    # Get the tied pairs/indices
    tied_pairs, tied_indices = get_tied_pairs(continuum)
    tie_vec = continuum.ties
    df_tie_vec = df_params.ties

    if split
        # add the 2 amplitudes for the PAH templates to the continuum, and return the PAH parameters separately
        pnames = [pnames; "dust_features.templates.1.amp"; "dust_features.templates.2.amp"]
        plims = [plims; (0., Inf); (0., Inf)]
        plock = BitVector([plock; 0; 0])
        tie_vec = [tie_vec; nothing; nothing]
        pnames, df_names, plims, df_plims, plock, df_plock, tied_pairs, tied_indices, tie_vec, df_tie_vec
    else
        # add all of the PAH parameters
        pnames = [pnames; df_names]
        plims = [plims; df_plims]
        plock = [plock; df_plock]
        tie_vec = [tie_vec; df_tie_vec]
        pnames, plims, plock, tied_pairs, tied_indices, tie_vec
    end
end


# Helper function for getting initial MIR parameters based on estimates from the data
# --- remember I is already normalized at this point!
function get_continuum_initial_values_from_estimation(cube_fitter::CubeFitter, λ::Vector{<:QWave}, 
    I::Vector{<:Real}, N::QSIntensity; force_noext::Bool=false)

    # Get the initial parameter vector read from the config files
    fopt = fit_options(cube_fitter)
    continuum = model(cube_fitter).continuum
    p₀ = continuum.values
    pnames = continuum.names

    dust_features = model(cube_fitter).dust_features
    df_params = get_flattened_fit_parameters(dust_features)
    df_p₀ = df_params.values
    df_names = df_params.names
    df_feature_names = dust_features.config.all_feature_names

    # Extinction maps
    if !isnothing(fopt.ebv_map)
        @debug "Using the provided E(B-V) values from the extinction_map"
        # use the intensity-weighted EBV value from the cube
        ebv_avg = sum(fopt.ebv_map[:, :, 1] .* sumdim(ustrip.(cube_fitter.cube.I), 3)) ./ nansum(ustrip.(cube_fitter.cube.I))
        ind = fast_indexin("extinction.E_BV", pnames)
        p₀[ind] = ebv_avg
    end

    # We need to fill in some missing values - particularly the amplitudes of different components
    att_gas_med, att_star_med, _ = extinction_profiles([median(λ)], ustrip.(p₀), 1, fopt.fit_uv_bump, fopt.extinction_curve)
    inv_atten_star_med = 1 / att_star_med[1]
    inv_atten_gas_med = 1 / att_gas_med[1]
    I_med = clamp(nanmedian(I), 0., Inf)
    λlim = extrema(λ)

    # Extinction
    if !isnothing(fopt.guess_tau) && fopt.silicate_absorption != "decompose"
        ind = fast_indexin("extinction.tau_97", pnames)
        p₀[ind] = guess_optical_depth(cube_fitter, λ, I)
    end
    # Force no extinction
    if force_noext
        inds = [fast_indexin("extinction.E_BV", pnames)]
        if fopt.silicate_absorption != "decompose"
            push!(inds, fast_indexin("extinction.tau_97", pnames))
        else
            push!(inds, fast_indexin("extinction.N_oli", pnames))
        end
        p₀[inds] .= 0.
    end

    # Fe II amplitudes
    if fopt.fit_opt_na_feii
        ind = fast_indexin("continuum.feii.na.amp", pnames)
        p₀[ind] = clamp(0.1 * I_med * inv_atten_gas_med, 0., Inf)
    end
    if fopt.fit_opt_br_feii
        ind = fast_indexin("continuum.feii.br.amp", pnames)
        p₀[ind] = clamp(0.1 * I_med * inv_atten_gas_med, 0., Inf)
    end
    
    # Power laws
    for j ∈ 1:cube_fitter.n_power_law
        inds = fast_indexin("continuum.power_law.$(j)." .* ["amp", "index"], pnames)
        if p₀[inds[2]] > 0   # index > 0 means it gets brighter at longer wavelengths
            atten_j, _, _ = extinction_profiles([λlim[2]], ustrip.(p₀), 1, fopt.fit_uv_bump, fopt.extinction_curve)
            A_i = 0.5 * nanmedian(I[end-10:end]) / atten_j[1] / cube_fitter.n_power_law
        else                 # index < 0 means it gets brighter at shorter wavelengths
            atten_j, _, _ = extinction_profiles([λlim[1]], ustrip.(p₀), 1, fopt.fit_uv_bump, fopt.extinction_curve)
            A_i = 0.5 * nanmedian(I[1:10]) / atten_j[1] / cube_fitter.n_power_law
        end
        if cube_fitter.n_templates > 0
            A_i *= 0.5
        end
        p₀[inds[1]] = clamp(A_i, 0., Inf)
    end

    # Dust continuum amplitudes
    for k ∈ 1:cube_fitter.n_dust_cont
        inds = fast_indexin("continuum.dust.$(k)." .* ["amp", "temp"], pnames)
        if fopt.lock_hot_dust && isone(k)
            p₀[inds[1]] = 0.
            continue
        end
        T_dc = p₀[inds[2]]
        λ_dc = clamp(Wein(T_dc), λlim...)
        i_dc = argmin(abs.(λ .- λ_dc))
        A_i = nanmedian(I[max(i_dc-5,1):min(i_dc+5,length(I))]) / cube_fitter.n_dust_cont
        if cube_fitter.n_templates > 0 && T_dc > 100u"K"
            A_i *= 0.5
        end
        if fopt.fit_sil_emission && T_dc > 500u"K"
            A_i *= 0.8
        end
        p₀[inds[1]] = clamp(A_i, 0., Inf)
    end

    # Hot dust amplitude
    if fopt.fit_sil_emission
        inds = fast_indexin("continuum.hot_dust." .* ["amp", "temp", "frac", "tau_warm", "tau_cold", "sil_peak"], pnames)
        T_hd, Cf, τw, τc, peak = p₀[inds[2:end]]
        hd = silicate_emission(λ, T_hd, Cf, τw, τc, peak, N, cube_fitter)
        mhd = argmax(hd)
        A_hd = 0.2 * nanmedian(I[max(mhd-5,1):min(mhd+5,length(I))])
        p₀[inds[1]] = clamp(A_hd, 0., Inf)
    end

    # Templates
    if fopt.fit_temp_multexp
        inds = fast_indexin("templates.amp_" .* string.(1:4), pnames)
        p₀[inds] .= 0.25
    end

    # Dust features
    # do a quick and dirty check around 7.7um or 11.3um to see if the PAHs are strong
    checks = [7.7, 7.7, 11.3, 11.3, 12.7, 12.7] .* u"μm"
    refs = [7.0, 10.0, 10.0, 12.0, 12.0, 13.2] .* u"μm"
    diffs = Float64[]
    for (ref, check) in zip(refs, checks)
        if (λlim[1] < check < λlim[2]) && (λlim[1] < ref < λlim[2])
            icheck = argmin(abs.(λ .- check))
            iref = argmin(abs.(λ .- ref))
            Icheck = nanmedian(I[max(icheck-5,1):min(icheck+5,length(I))])
            Iref = nanmedian(I[max(iref-5,1):min(iref+5,length(I))])
            push!(diffs, clamp(Icheck - Iref, 0., Inf))
        end
    end
    df_amp = length(diffs) > 0 ? nanmedian(diffs) : I_med/4

    for df ∈ df_feature_names
        ind = fast_indexin("dust_features.$(df).amp", df_names)
        df_p₀[ind] = clamp(df_amp, 0., Inf)
    end

    # For the PAH templates
    pah_frac = repeat([clamp(df_amp, 0., Inf)], 2)
    append!(p₀, df_p₀)

    p₀, pah_frac
end


# Helper function for getting initial MIR parameters based on the initial fit
function get_continuum_initial_values_from_previous(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:QWave}, 
    I::Vector{<:Real}, σ::Vector{<:Real}; force_noext::Bool=false)

    fopt = fit_options(cube_fitter)
    continuum = model(cube_fitter).continuum
    pnames = continuum.names
    ab_feature_names = model(cube_fitter).abs_features.config.all_feature_names

    # Set the parameters to the best parameters
    p₀ = copy(cube_fitter.p_init_cont)
    pah_frac = copy(cube_fitter.p_init_pahtemp)

    # scale all flux amplitudes by the difference in medians between the spaxel and the summed spaxels
    Iunit = unit(cube_fitter.cube.I[1])
    I_init = sumdim(ustrip.(cube_fitter.cube.I), (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2)) .* Iunit
    N0 = Float64(abs(maximum(I_init[isfinite.(I_init)])))
    N0 = N0 ≠ 0.0Iunit ? N0 : 1.0Iunit
    scale = max(nanmedian(I), 1e-10) * N0 / nanmedian(I_init)   # (should be close to 1)
    @assert unit(scale) == NoUnits 

    atten_gas_0, atten_star_0, _  = extinction_profiles([median(λ)], ustrip.(p₀), 1, fopt.fit_uv_bump, fopt.extinction_curve)
    atten_star_0, atten_gas_0 = atten_star_0[1], atten_gas_0[1]

    # Force no extinction
    if force_noext
        inds = [fast_indexin("extinction.E_BV", pnames)]
        if fopt.silicate_absorption != "decompose"
            push!(inds, fast_indexin("extinction.tau_97", pnames))
        else
            push!(inds, fast_indexin("extinction.N_oli", pnames))
        end
        p₀[inds] .= 0.
    end
    # Extinction maps
    if !isnothing(fopt.ebv_map)
        @debug "Using the provided E(B-V) values from the extinction_map"
        if !isnothing(cube_fitter.cube.voronoi_bins)
            data_indices = findall(vbin -> vbin == Tuple(spaxel)[1], cube_fitter.cube.voronoi_bins)
            ebv_new = mean(fopt.ebv_map[data_indices, 1])
        else
            ebv_new = fopt.ebv_map[spaxel, 1]
        end
        ind = fast_indexin("extinction.E_BV", pnames)
        p₀[ind] = ebv_new
    end
    if !isnothing(fopt.sil_abs_map)
        @debug "Using the provided τ_9.7 values from the extinction_map"
        if fopt.silicate_absorption == "decompose"
            inds = fast_indexin("extinction." .* ["N_oli", "N_pyr", "N_for"], pnames)
        else
            inds = fast_indexin(["extinction.tau_97"], pnames)
        end
        if !isnothing(cube_fitter.cube.voronoi_bins)
            data_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(spaxel)[1])
            for i in eachindex(inds)
                p₀[inds[i]] = mean(fopt.sil_abs_map[data_indices, i])
            end
        else
            data_index = spaxel
            for i in eachindex(inds)
                p₀[inds[i]] = fopt.sil_abs_map[data_index, i]
            end
        end
    end

    if fopt.silicate_absorption != "decompose"
        ind = fast_indexin("extinction.tau_97", pnames)
        p₀[ind] = max(model(cube_fitter).continuum["extinction.tau_97"].value, p₀[ind])
    end

    # Set τ_9.7 and τ_CH to 0 if the continuum is within 1 std dev of 0
    lock_abs = false
    if nanmedian(I) ≤ nanmedian(σ)
        lock_abs = true
        if fopt.silicate_absorption == "decompose"
            pwhere = ["extinction.N_oli"]
        else
            pwhere = ["extinction.tau_97"]
        end
        inds = fast_indexin(pwhere, pnames)
        p₀[inds] .= 0.
    end
    # Set τ_9.7 to the guess if the guess_tau flag is set
    if !isnothing(fopt.guess_tau) && (fopt.silicate_absorption != "decompose")
        ind = fast_indexin("extinction.tau_97", pnames)
        p₀[ind] = guess_optical_depth(cube_fitter, λ, I)
    end
    # Do not adjust absorption feature amplitudes since they are multiplicative
    for ab ∈ ab_feature_names
        ind = fast_indexin("abs_features.$(ab).tau", pnames)
        if lock_abs
            p₀[ind] = 0.
        end
    end

    # Recompute the new extinction and adjust the rescaling factor
    atten_gas_new, atten_star_new, _ = extinction_profiles([median(λ)], ustrip.(p₀), 1, fopt.fit_uv_bump, fopt.extinction_curve)
    atten_star_new, atten_gas_new = atten_star_new[1], atten_gas_new[1]

    scale_star = scale * atten_star_0 / atten_star_new    # may no longer be close to 1
    scale_gas = scale * atten_gas_0 / atten_gas_new

    # Fe II amplitudes
    if fopt.fit_opt_na_feii
        ind = fast_indexin("continuum.feii.na.amp", pnames)
        p₀[ind] *= scale_gas
    end
    if fopt.fit_opt_br_feii
        ind = fast_indexin("continuum.feii.br.amp", pnames)
        p₀[ind] *= scale_gas
    end

    # Power law amplitudes
    for j ∈ 1:cube_fitter.n_power_law
        ind = fast_indexin("continuum.power_law.$(j).amp", pnames)
        p₀[ind] *= scale_gas
    end

    # Dust continuum amplitudes (rescaled)
    for di ∈ 1:cube_fitter.n_dust_cont
        ind = fast_indexin("continuum.dust.$(di).amp", pnames)
        p₀[ind] *= scale_gas 
        if fopt.lock_hot_dust && isone(di)
            p₀[ind] = 0.
        end
    end

    # Hot dust amplitude (rescaled)
    if fopt.fit_sil_emission
        ind = fast_indexin("continuum.hot_dust.amp", pnames)
        p₀[ind] *= scale_gas
    end

    # PAH template amplitudes
    pah_frac *= scale_gas

    p₀, pah_frac
end


"""
    get_continuum_initial_values(cube_fitter, spaxel, λ, I, σ, N, init; split)

Get the vectors of starting values and relative step sizes for the continuum fit for a given CubeFitter object. 
Again, the vector may be split up by the 2 continuum fitting steps in the MIR case.
"""
function get_continuum_initial_values(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:QWave}, I::Vector{<:Real},
    σ::Vector{<:Real}, N::QSIntensity; init::Bool=false, split::Bool=true, force_noext::Bool=false)

    # Check if the cube fitter has initial fit parameters 
    if !init
        @debug "Using initial best fit continuum parameters..."
        p₀, pah_frac = get_continuum_initial_values_from_previous(cube_fitter, spaxel, λ, I, σ; force_noext=force_noext)
    else
        @debug "Calculating initial starting points..." 
        p₀, pah_frac = get_continuum_initial_values_from_estimation(cube_fitter, λ, I, N; force_noext=force_noext)
    end
    dstep, dstep_pahfrac = get_continuum_step_sizes(cube_fitter, λ) 

    cf_model = model(cube_fitter)
    continuum = cf_model.continuum
    pnames = continuum.names

    @debug "Continuum parameters: \n $pnames"
    @debug "Continuum starting values: \n $p₀"
    @debug "Continuum relative step sizes: \n $dstep"

    if !split
        p₀, dstep
    else
        n_split = count_cont_parameters(cf_model; split=true)
        # Step 1: Stellar + Dust blackbodies, 2 new amplitudes for the PAH templates, and the extinction parameters
        pars_1 = vcat(p₀[1:n_split], pah_frac)
        dstep_1 = vcat(dstep[1:n_split], dstep_pahfrac)
        # Step 2: The PAH profile amplitudes, centers, and FWHMs
        pars_2 = p₀[n_split+1:end]
        dstep_2 = dstep[n_split+1:end]

        pars_1, pars_2, dstep_1, dstep_2
    end
end


# Helper function for getting MIR parameter step sizes 
function get_continuum_step_sizes(cube_fitter::CubeFitter, λ::Vector{<:QWave})

    # Calculate relative step sizes for finite difference derivatives
    dλ = (λ[end] - λ[1]) / (length(λ)-1)
    deps = sqrt(eps())
    fopt = fit_options(cube_fitter)
    ab_feature_names = model(cube_fitter).abs_features.config.all_feature_names
    df_feature_names = model(cube_fitter).dust_features.config.all_feature_names

    params = get_flattened_fit_parameters(model(cube_fitter))
    pnames = params.names
    p₀ = params.values

    # Start out with all "deps" steps
    dstep = repeat([deps], cube_fitter.n_params_cont)

    # Replace absorption feature means/fwhms
    for ab in ab_feature_names
        inds = fast_indexin("abs_features.$(ab)." .* ["mean", "fwhm"], pnames)
        dstep[inds[1]] = dλ/10/p₀[inds[1]]
        dstep[inds[2]] = dλ/1000/p₀[inds[2]]
    end
    # Stellar kinematics
    if fopt.fit_stellar_continuum
        inds = fast_indexin("continuum.stellar_kinematics." .* ["vel", "vdisp"], pnames)
        dstep[inds] .= 1e-4
    end
    # Fe II kinematics
    if fopt.fit_opt_na_feii
        inds = fast_indexin("continuum.feii.na." .* ["vel", "vdisp"], pnames)
        dstep[inds] .= 1e-4
    end
    if fopt.fit_opt_br_feii
        inds = fast_indexin("continuum.feii.br." .* ["vel", "vdisp"], pnames)
        dstep[inds] .= 1e-4
    end
    # Dust continua temperatures
    for j ∈ 1:cube_fitter.n_dust_cont
        ind = fast_indexin("continuum.dust.$(j).temp", pnames)
        dstep[ind] = 1e-4
    end
    # Hot dust temperature/peak
    if fopt.fit_sil_emission
        inds = fast_indexin("continuum.hot_dust." .* ["temp", "sil_peak"], pnames)
        dstep[inds[1]] = 1e-4
        dstep[inds[2]] = dλ/10/p₀[inds[2]]
    end
    # PAH feature means/fwhms
    for df ∈ df_feature_names
        inds = fast_indexin("dust_features.$(df)." .* ["mean", "fwhm"], pnames)
        dstep[inds[1]] = dλ/10/p₀[inds[1]]
        dstep[inds[2]] = dλ/1000/p₀[inds[2]]
    end

    dstep_pahtemp = [deps, deps]

    dstep, dstep_pahtemp
end


"""
    get_continuum_parinfo(n_free, lb, ub, dp)

Get the CMPFit parinfo and config objects for a given CubeFitter object, given the vector of initial valuels,
limits, and relative step sizes.
"""
function get_continuum_parinfo(n_free::S, lb::Vector{T}, ub::Vector{T}, dp::Vector{<:Real}) where {S<:Integer,T<:Number}

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
    lb_2::Vector{T}, ub_2::Vector{T}, dp_1::Vector{<:Real}, dp_2::Vector{<:Real}) where {S<:Integer,T<:Number}

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


function get_line_initial_values_limits_locks_from_estimation(cube_fitter::CubeFitter, λ::Vector{<:QWave}, 
    I::Vector{<:Real}, ext_curve::Union{Nothing,Vector{<:Real}}=nothing; init::Bool=false)

    fopt = fit_options(cube_fitter)

    continuum = model(cube_fitter).continuum
    p0_contin = continuum.values
    pnames_contin = continuum.names

    # Get the default lower/upper limits from the model object
    lines = model(cube_fitter).lines
    params = get_flattened_fit_parameters(lines)
    pnames = params.names
    plims = params.limits
    plock = params.locked
    p₀ = init ? params.values : copy(cube_fitter.p_init_line)

    if !isnothing(ext_curve)
        max_amp = clamp(1 / minimum(ext_curve), 1., 1e100)
    else
        # find the maximum extinction factor between the extinction curve and the silicate absorption
        atten, _, _ = extinction_profiles(λ, ustrip.(p0_contin), 1, fopt.fit_uv_bump, fopt.extinction_curve)
        inv_atten = 1 / minimum(atten)
        pstart = fast_indexin(fopt.silicate_absorption == "decompose" ? "extinction.N_oli" : "extinction.tau_97", pnames_contin)
        silab, = silicate_absorption(λ, ustrip.(p0_contin), pstart, cube_fitter)
        inv_abs = 1 / minimum(silab)
        max_amp = max(inv_atten, inv_abs)
    end
    if fopt.lines_allow_negative
        amp_plim = (-max_amp, max_amp)
    else
        amp_plim = (0., max_amp)
    end

    # A_ln = 0.33 * max_amp
    for (k, line) in enumerate(lines.profiles)   # <- iterates over emission lines
        ln_name = lines.names[k]
        λ₀ = lines.λ₀[k]
        for (j, component) in enumerate(line)    # <- iterates over individual velocity components
            ind_amp = fast_indexin("lines.$(ln_name).$(j).amp", pnames)
            # only set the amplitude limits for the FIRST component
            # or, set it for the other components, but only if rel_amp is not set
            if isone(j) || !lines.config.rel_amp
                plims[ind_amp] = amp_plim
                plock[ind_amp] = false
            end
            # If this is the initial fit, we want to make some educated guesses on the initial values for the parameters,
            # just like for the continuum fit
            if init
                # make a velocity vector
                vel = Doppler_width_v.(λ .- λ₀, λ₀)
                # choose a window size equal to the linemask width
                region = abs.(vel) .< cube_fitter.linemask_width
                I_max, mx = findmax(I[region])
                I_min = minimum(I[region])
                # amplitude is ~the max - the min within the window
                if isone(j)
                    p₀[ind_amp] = clamp(I_max-I_min, amp_plim...)
                elseif !lines.config.rel_amp
                    # make a somewhat random guess that the secondary component is ~20% of the total amplitude
                    p₀[ind_amp] = clamp(0.2*(I_max-I_min), amp_plim...)
                    # readjust the first component down a bit
                    ind_amp1 = fast_indexin("lines.$(ln_name).1.amp", pnames)
                    p₀[ind_amp1] = clamp(0.8p₀[ind_amp1], amp_plim...)
                else
                    p₀[ind_amp] = 0.2
                end
                # velocity is ~the velocity at the max ind
                ind_vel = fast_indexin("lines.$(ln_name).$(j).voff", pnames)
                if isone(j) || !lines.config.rel_voff
                    p₀[ind_vel] = clamp(vel[region][mx], plims[ind_vel]...)
                else
                    p₀[ind_vel] = clamp(0.0u"km/s", plims[ind_vel]...)
                end
                # fwhm is ~the intensity-weighted average of the velocities
                ind_fwhm = fast_indexin("lines.$(ln_name).$(j).fwhm", pnames)
                if sum(I[region]) > 0.
                    fwhm_full = 2.355*sqrt(clamp(sum(I[region].*vel[region].^2) / sum(I[region]), 0.0u"km^2/s^2", Inf*u"km^2/s^2"))
                else
                    fwhm_full = 0.0*u"km/s"
                end
                fwhm_intr2 = clamp(fwhm_full^2 - cube_fitter.lsf(λ[region][mx])^2, 0.0u"km^2/s^2", Inf*u"km^2/s^2")
                if isone(j) || !lines.config.rel_fwhm
                    p₀[ind_fwhm] = clamp(sqrt(fwhm_intr2), plims[ind_fwhm]...)
                else
                    p₀[ind_fwhm] = clamp(1.0, plims[ind_fwhm]...)
                end
            end
        end
    end

    # We need to make sure the tied relationships still hold -- if we're doing the initial estimation,
    # the tied velocities and fwhms should be a median of the estimations we did for each individual line
    tie_vec = params.ties
    if init
        tie_vec_groups = [isnothing(tv) ? nothing : tv.group for tv in tie_vec]
        tvu = unique(tie_vec[.~isnothing.(tie_vec)])  # all of the tied groups
        for tvui in tvu
            if contains(string(tvui), "voff") || contains(string(tvui), "fwhm")
                wh = tie_vec_groups .== tvui.group
                p₀[wh] .= median(p₀[wh])
            end
            # amplitudes are more tricky...but they should work themselves out in the next step below
        end
    end

    tied_pairs, tied_indices = get_tied_pairs(params)
    for tp in tied_pairs
        p₀[tp[2]] = p₀[tp[1]] * tp[3]
        plims[tp[2]] = plims[tp[1]] .* tp[3]
        plock[tp[2]] = plock[tp[1]]
    end

    p₀, plims, plock, pnames, tied_pairs, tied_indices, tie_vec
end


# Helper function for getting line parameter step sizes
function get_line_step_sizes(cube_fitter::CubeFitter, ln_pars::Vector{<:Number}, init::Bool)

    lines = model(cube_fitter).lines
    pnames = get_flattened_fit_parameters(lines).names

    # Absolute step size vector (0 tells CMPFit to use a default value)
    ln_astep = [0.0*unit(lnp) for lnp in ln_pars]

    if !init
        for (k, line) in enumerate(lines.profiles)   # <- iterates over emission lines
            ln_name = lines.names[k]
            for (j, component) in enumerate(line)    # <- iterates over individual velocity components 
                inds = fast_indexin("lines.$(ln_name).$(j)." .* ["voff", "fwhm"], pnames)
                ln_astep[inds[1]] = 1e-5u"km/s"
                ln_astep[inds[2]] = j > 1 && lines.config.rel_fwhm ? 0. : 1e-5u"km/s"
            end
        end
    end

    ln_astep
end


"""
    get_line_initial_values(cube_fitter, spaxel, init)

Get the vector of starting values and relative step sizes for the line fit for a given CubeFitter object.
"""
function get_line_initial_values_limits_locked(cube_fitter::CubeFitter, λ::Vector{<:QWave}, 
    I::Vector{<:Real}, ext_curve::Union{Nothing,Vector{<:Real}}=nothing; init::Bool=false)

    # Get initial values, locks, etc.
    ln_pars, plims, plock, pnames, tied_pairs, tied_indices, tie_vec = 
        get_line_initial_values_limits_locks_from_estimation(cube_fitter, λ, I, ext_curve; init=init)

    # Get step sizes for each parameter
    ln_astep = get_line_step_sizes(cube_fitter, ln_pars, init)

    ln_pars, plims, plock, pnames, ln_astep, tied_pairs, tied_indices, tie_vec
end


"""
    get_line_parinfo(n_free, lb, ub, dp)

Get the CMPFit parinfo and config objects for a given CubeFitter object, given the vector of initial values,
limits, and absolute step sizes.
"""
function get_line_parinfo(n_free, lb, ub, dp)

    # Convert parameter limits into CMPFit object
    parinfo = CMPFit.Parinfo(n_free)
    for pᵢ ∈ 1:n_free
        parinfo[pᵢ].fixed = 0
        parinfo[pᵢ].limited = (1,1)
        parinfo[pᵢ].limits = (lb[pᵢ], ub[pᵢ])
        parinfo[pᵢ].step = dp[pᵢ]
    end

    # Create a `config` structure
    config = CMPFit.Config()
    # Lower tolerance level for lines fit
    config.ftol = 1e-16
    config.xtol = 1e-16
    config.maxiter = 500

    parinfo, config
end


"""
    clean_line_parameters(cube_fitter, popt, lower_bounds, upper_bounds)

Takes the results of an initial global line fit and prepares the parameters for individual spaxel
fits by sorting the line components, adjusting the voffs/FWHMs for lines that are not detected, and 
various other small adjustments.
"""
function clean_line_parameters(cube_fitter::CubeFitter, popt::Vector{<:Number}, lower_bounds::Vector{<:Real}, upper_bounds::Vector{<:Real})

    lines = model(cube_fitter).lines
    params = get_flattened_fit_parameters(lines)

    pᵢ = 1
    for (k, line) in enumerate(lines.profiles)   # <- iterates over emission lines
        pstart = Int[]
        pfwhm = Int[]
        pend = Int[]
        amp_main = popt[pᵢ]
        voff_main = popt[pᵢ+1]
        fwhm_main = popt[pᵢ+2]

        for (j, component) in enumerate(line)    # <- iterates over individual velocity components
            n_prof = length(line)
            push!(pstart, pᵢ)

            # If additional components arent detected, set them to a small nonzero value
            replace_line = iszero(popt[pᵢ])
            if replace_line
                if j > 1
                    popt[pᵢ] = lines.config.rel_amp ? 0.1 * 1/(n_prof-1) : 0.1 * 1/(n_prof-1) * amp_main
                    popt[pᵢ+1] = lines.config.rel_voff ? 0.0u"km/s" : voff_main
                    popt[pᵢ+2] = lines.config.rel_fwhm ? 1.0 : fwhm_main
                else
                    if isnothing(params[pᵢ+1].tie)
                        popt[pᵢ+1] = voff_main = 0.0u"km/s" # voff
                    end
                    if isnothing(params[pᵢ+2].tie)
                        popt[pᵢ+2] = fwhm_main = (lower_bounds[pᵢ+2]+upper_bounds[pᵢ+2])/2*u"km/s" # fwhm
                    end
                end
            end
            # Velocity offsets for the integrated spectrum shouldnt be too large
            # if abs(popt[pᵢ+1]) > 500.
            # if !cube_fitter.fit_all_global
            #     popt[pᵢ+1] = 0.
            # end
            pc = 3
            push!(pfwhm, pᵢ+2)

            if component.profile == :GaussHermite
                pc += 2
            elseif component.profile == :Voigt
                # Set the Voigt mixing ratios back to 0.5 since a summed fit may lose the structure of the line-spread function
                if !params[pᵢ+pc].locked
                    popt[pᵢ+pc] = 0.5
                end
                pc += 1
            end

            pᵢ += pc
            push!(pend, pᵢ-1)
        end

        # resort line components by decreasing flux
        if !lines.config.rel_amp && !lines.config.rel_voff && !lines.config.rel_fwhm
            pnew = copy(popt)
            # pstart gives the amplitude indices
            ss = sortperm(popt[pstart].*popt[pfwhm], rev=true)
            for k in eachindex(ss)
                pnew[pstart[k]:pend[k]] .= popt[pstart[ss[k]]:pend[ss[k]]]
            end
            popt = pnew
        end
    end
    return popt
end
