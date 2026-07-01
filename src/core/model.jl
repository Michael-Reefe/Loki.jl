
############################################## FITTING FUNCTIONS #############################################

function unit_check(u1::Unitful.Units, u2::Unitful.Units)
    @assert u1 == u2 "Uh oh...I think something's wrong with your units: $(u1) ≠ $(u2)"
end


# Helper function for getting the normalized templates for a given fit
function get_normalized_templates(λ::Vector{<:QWave}, params::Vector{<:Real}, templates::Matrix{<:Real}, 
    channel_masks::Vector{BitVector}, fit_temp_multexp::Bool, pstart::Integer)
    temp_norm = zeros(size(templates)...)
    # Add generic templates with a normalization parameter
    if fit_temp_multexp
        ex = multiplicative_exponentials(λ, params[pstart:pstart+7])
        for i in axes(templates, 2)
            temp_norm[:,i] .+= sum([ex[:,j] .* templates[:,i] for j in axes(ex,2)])
        end
        dp = 8
    else
        dp = 0
        for i in axes(templates, 2)
            for ch_mask in channel_masks
                temp_norm[ch_mask,i] .+= params[pstart+dp] .* templates[ch_mask, i]
                dp += 1
            end
        end
    end
    temp_norm, dp
end
# the in-place version
function get_normalized_templates!(contin::Vector{<:Real}, λ::Vector{<:QWave}, params::Vector{<:Real}, 
    templates::Matrix{<:Real}, channel_masks::Vector{BitVector}, fit_temp_multexp::Bool, mpoly::Vector{<:Real}, 
    pstart::Integer)
    # Add generic templates with a normalization parameter
    if fit_temp_multexp
        ex = multiplicative_exponentials(λ, params[pstart:pstart+7])
        for i in axes(templates, 2)
            contin .+= sum([ex[:,j] .* templates[:,i] for j in axes(ex,2)]) .* mpoly
        end
        dp = 8
    else
        dp = 0
        for i in axes(templates, 2)
            for ch_mask in channel_masks
                c = @view contin[ch_mask]
                c .+= params[pstart+dp] .* templates[ch_mask, i] .* mpoly[ch_mask]
                dp += 1
            end
        end
    end
    dp
end


"""
    model_continuum(λ, params, N, cube_fitter, nuc_temp_fit, return_components)

Create a model of the continuum (including stellar+dust continuum, PAH features, and extinction, excluding emission lines)
at the given wavelengths `λ`, given the parameter vector `params`.

Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments
- `s`: The spaxel object containing the wavelength data, normalization, templates, and channel masks.
- `params`: Parameter vector. 
- `cube_fitter`: The CubeFitter object with all of the fitting options.
- `use_pah_templates`: Whether or not to fit the PAHs with templates instead of Drude profiles 
- `return_components`: Whether or not to return the individual components of the fit as a dictionary, in 
    addition to the overall fit
"""
function model_continuum(s::Spaxel, N::QSIntensity, params::Vector{<:Real}, punits::Vector{<:Unitful.Units}, 
    cube_fitter::CubeFitter, use_pah_templates::Bool, do_gaps::Bool, mask_lines::Bool, return_components::Bool)

    # Get options from the cube fitter
    fopt = fit_options(cube_fitter)

    # Get the wavelength data 
    λ = s.λ
    λunit = unit(λ[1])
    nλ = length(λ)

    # Prepare outputs
    out_type = typeof(ustrip(λ[1]))
    comps = Dict{String, Union{Vector{out_type}, Vector{typeof(N)}}}()
    norms = Dict{String, Union{Number, Matrix{out_type}}}()
    contin = zeros(out_type, nλ)
    pᵢ::Int64 = 1

    ##### 1. EXTINCTION AND ABSORPTION FEATURES #####

    # --> General dust extinction curve 
    comps["extinction_gas"], comps["extinction_stars"], dp = extinction_profiles(λ, params, pᵢ, fopt.fit_uv_bump, 
        fopt.extinction_curve)
    pᵢ += dp
    # --> Silicate absorption features
    comps["absorption_silicates"], dp, abs_oli, abs_pyr, abs_for = silicate_absorption(λ, params, pᵢ, cube_fitter)
    pᵢ += dp
    if fopt.silicate_absorption == "decompose"
        comps["absorption_oli"] = abs_oli
        comps["absorption_pyr"] = abs_pyr
        comps["absorption_for"] = abs_for
    end
    abs_sil = comps["absorption_silicates"]
    abs_tot = ones(out_type, length(λ))
    # --> Ice+CH absorption features
    if fopt.fit_ch_abs
        comps["absorption_ice"] = extinction_factor.(τ_ice(λ, cube_fitter), params[pᵢ]*params[pᵢ+1], screen=true)
        comps["absorption_ch"] = extinction_factor.(τ_ch(λ, cube_fitter), params[pᵢ+1], screen=true) 
        abs_tot .*= comps["absorption_ice"] .* comps["absorption_ch"]
        pᵢ += 2
    end
    # --> Other absorption features
    for k ∈ 1:cube_fitter.n_abs_feat
        # this needs a type hint because it can't be inferred if we're not fitting any absorption profiles,
        # and it affects the type inference of abs_tot and hence ext_gas and ext_stars 
        unit_check(punits[pᵢ+1], λunit)
        unit_check(punits[pᵢ+2], λunit)
        unit_check(punits[pᵢ+3], λunit^-1)
        comps["absorption_feat_$k"] = extinction_factor.(Drude.(λ, 1.0, params[pᵢ+1]*λunit, params[pᵢ+2]*λunit, 
            params[pᵢ+3]/λunit), params[pᵢ], screen=true)
        abs_tot .*= comps["absorption_feat_$k"]
        pᵢ += 4
    end
    # --> combine everything together with optional covering fraction
    Cf = 1.0
    if fopt.fit_covering_frac
        Cf = params[pᵢ]
        pᵢ += 1
    end
    comps["total_extinction_stars"] = @. Cf * comps["extinction_stars"] * abs_sil * abs_tot + (1 - Cf)
    comps["total_extinction_gas"] = @. Cf * comps["extinction_gas"] * abs_sil * abs_tot + (1 - Cf)

    ##### 2. POLYNOMIALS    #####

    # Additive polynomials 
    if cube_fitter.apoly_degree ≥ 0
        coeffs = params[pᵢ:pᵢ+cube_fitter.apoly_degree]
        # rescale wavelengths to range from -1 to +1
        x = ustrip.(-1.0 .+ 2.0 ./ (s.λ[end].-s.λ[1]) .* (s.λ.-s.λ[1]))
        if cube_fitter.apoly_type == :Legendre
            comps["apoly"] = Legendre(coeffs).(x)
        elseif cube_fitter.apoly_type == :Chebyshev
            comps["apoly"] = Chebyshev(coeffs).(x)
        else 
            error("Unrecognized polynomial type: $(cube_fitter.apoly_type)")
        end
        # Don't allow the polynomial to be negative
        comps["apoly"] .= clamp.(comps["apoly"], 0., Inf)
        # Additive polynomials are not affected by any dust extinction or 
        # multiplicative polynomials, since they're meant to model background/
        # foreground emission or other systematics.
        contin .+= comps["apoly"]
        pᵢ += cube_fitter.apoly_degree + 1
    end

    # Multiplicative polynomials
    comps["mpoly"] = ones(out_type, length(λ))
    if cube_fitter.mpoly_degree ≥ 1
        coeffs = [1.0; params[pᵢ:pᵢ+cube_fitter.mpoly_degree-1]]
        # rescale wavelengths to range from -1 to +1
        x = ustrip.(-1.0 .+ 2.0 ./ (s.λ[end].-s.λ[1]) .* (s.λ.-s.λ[1]))
        if cube_fitter.mpoly_type == :Legendre 
            comps["mpoly"] .= Legendre(coeffs).(x)
        elseif cube_fitter.mpoly_type == :Chebyshev 
            comps["mpoly"] .= Chebyshev(coeffs).(x)
        else 
            error("Unrecognized polynomial type: $(cube_fitter.apoly_type)")
        end
        # clamp extreme values
        comps["mpoly"] .= clamp.(comps["mpoly"], 0.1, 10.)
        pᵢ += cube_fitter.mpoly_degree
    end

    #############################

    # Collect the stellar velocities
    stel_vel = stel_sig = 0.0u"km/s"
    if fopt.fit_stellar_continuum
        unit_check(punits[pᵢ], u"km/s")
        unit_check(punits[pᵢ+1], u"km/s")
        stel_vel = params[pᵢ]*u"km/s"
        stel_sig = params[pᵢ+1]*u"km/s"
        pᵢ += 2
    end

    #############################
    
    ##### 3. FE II EMISSION #####

    gap_masks = get_gap_masks(s.λ, cube_fitter.spectral_region.gaps)
    if fopt.fit_opt_na_feii
        unit_check(punits[pᵢ+1], u"km/s")
        unit_check(punits[pᵢ+2], u"km/s")
        conv_na_feii = zeros(nλ, 1)
        for (gi, gap_mask) in enumerate(gap_masks)
            # split if the spectrum has a few separated regions
            conv_na_feii[gap_mask, 1] .= convolve_losvd(cube_fitter.feii.na_fft, cube_fitter.feii.vsysts[gi], 
                params[pᵢ+1]*u"km/s", params[pᵢ+2]*u"km/s", s.vres, sum(gap_mask), temp_fft=true, 
                npad_in=cube_fitter.feii.npad)
        end
        # These templates also need the solid angle divided out
        conv_na_feii[:, 1] ./= s.area_sr
        conv_na_feii[.~isfinite.(conv_na_feii[:, 1]), 1] .= 0.0   # (area_sr may be 0 at some points)
        norms["continuum.feii.na.amp"] = maximum(conv_na_feii[:, 1])
        comps["na_feii"] = params[pᵢ] .* conv_na_feii[:, 1] ./ norms["continuum.feii.na.amp"]
        contin .+= comps["na_feii"] .* comps["total_extinction_gas"]
        pᵢ += 3
    end
    if fopt.fit_opt_br_feii
        unit_check(punits[pᵢ+1], u"km/s")
        unit_check(punits[pᵢ+2], u"km/s")
        conv_br_feii = zeros(nλ, 1)
        for (gi, gap_mask) in enumerate(gap_masks)
            conv_br_feii[gap_mask, 1] .= convolve_losvd(cube_fitter.feii.br_fft, cube_fitter.feii.vsysts[gi], 
                params[pᵢ+1]*u"km/s", params[pᵢ+2]*u"km/s", s.vres, sum(gap_mask), temp_fft=true, 
                npad_in=cube_fitter.feii.npad)
        end
        conv_br_feii[:, 1] ./= s.area_sr
        conv_br_feii[.~isfinite.(conv_br_feii[:, 1]), 1] .= 0.0   # (area_sr may be 0 at some points)
        norms["continuum.feii.br.amp"] = maximum(conv_br_feii[:, 1])
        comps["br_feii"] = params[pᵢ] .* conv_br_feii[:, 1] ./ norms["continuum.feii.br.amp"]
        contin .+= comps["br_feii"] .* comps["total_extinction_gas"]
        pᵢ += 3
    end

    ##### 4. POWER LAWS #####

    for j ∈ 1:cube_fitter.n_power_law
        # Reference wavelength at the median wavelength of the input spectrum
        pl = power_law(λ, params[pᵢ+1], median(λ))
        norms["continuum.power_law.$(j).amp"] = maximum(pl)
        comps["power_law_$j"] = params[pᵢ] .* pl./norms["continuum.power_law.$(j).amp"]
        contin .+= comps["power_law_$j"] .* comps["total_extinction_gas"] .* comps["mpoly"]
        pᵢ += 2
    end

    ##### 5. THERMAL DUST CONTINUA #####

    for i ∈ 1:cube_fitter.n_dust_cont
        unit_check(punits[pᵢ+1], u"K")
        bb = Blackbody_modified.(λ, params[pᵢ+1]*u"K", N)
        norms["continuum.dust.$(i).amp"] = maximum(bb)
        comps["dust_cont_$i"] = params[pᵢ] .* bb./norms["continuum.dust.$(i).amp"]
        contin .+= comps["dust_cont_$i"] .* comps["total_extinction_gas"] .* comps["mpoly"]
        pᵢ += 2
    end

    ##### 6. SILICATE EMISSION #####

    if fopt.fit_sil_emission
        # Add Silicate emission from hot dust (amplitude, temperature, covering fraction, warm tau, cold tau)
        # Ref: Gallimore et al. 2010
        unit_check(punits[pᵢ+1], u"K")
        unit_check(punits[pᵢ+5], λunit)
        sil_emission = silicate_emission(λ, params[pᵢ+1]*u"K", params[pᵢ+2], params[pᵢ+3], params[pᵢ+4], 
            params[pᵢ+5]*λunit, N, cube_fitter)
        norms["continuum.hot_dust.amp"] = maximum(sil_emission)
        comps["hot_dust"] = params[pᵢ] .* sil_emission ./ norms["continuum.hot_dust.amp"]
        contin .+= comps["hot_dust"] .* abs_tot .* comps["mpoly"]
        pᵢ += 6
    end

    ##### 7. PSF TEMPLATES #####

    if haskey(s.aux, "templates") && haskey(s.aux, "channel_masks") && size(s.aux["templates"], 2) > 0
        temp_norm, dp = get_normalized_templates(λ, params, s.aux["templates"], s.aux["channel_masks"], 
            fopt.fit_temp_multexp, pᵢ)
        for i in axes(temp_norm, 2)
            comps["templates_$i"] = temp_norm[:,i]
        end
        contin .+= sumdim(temp_norm, 2; nan=false) .* comps["mpoly"]
        pᵢ += dp
    end

    ##### 8. PAH EMISSION FEATURES #####

    pah_contin = zeros(length(λ))
    if use_pah_templates
        pah_contin .+= params[pᵢ] .* cube_fitter.dust_interpolators["smith3"](ustrip.(λ)) .* comps["total_extinction_gas"] .* comps["mpoly"]
        pah_contin .+= params[pᵢ+1] .* cube_fitter.dust_interpolators["smith4"](ustrip.(λ)) .* comps["total_extinction_gas"] .* comps["mpoly"]
        pᵢ += 2
    else
        pah_contin, pah_comps = model_pah_residuals(s, params[pᵢ:end], punits[pᵢ:end], comps["total_extinction_gas"], 
            cube_fitter, true)
        comps = merge(comps, pah_comps)
    end
    contin .+= pah_contin

    ##### 9. STELLAR POPULATIONS #####

    if fopt.fit_stellar_continuum
        ssp_contin, norms["continuum.stellar_norm"], norms["continuum.stellar_weights"] = stellar_populations_nnls(
            s, contin, comps["total_extinction_stars"].*comps["mpoly"], stel_vel, stel_sig, cube_fitter; 
            do_gaps=do_gaps, mask_lines=mask_lines)
        comps["SSPs"] = ssp_contin ./ (comps["total_extinction_stars"].*comps["mpoly"])   # (save the unmodified stellar continuum)
        contin .+= ssp_contin
    end

    ##### ------------------------ #####

    # Save the current state of the continuum with everything except the PAH features #
    comps["continuum"] = contin .- pah_contin
    # Save the state of the continuum with everything including the PAH features # 
    comps["continuum_and_pahs"] = contin

    ##### ------------------------ #####

    # Return components if necessary
    if return_components
        return contin, comps, norms
    end
    contin
end


# Multiple dispatch for more efficiency --> not allocating the dictionary improves performance DRAMATICALLY
# --> the N argument NEEDS to be separate from the spaxel object for optimal type stability
function model_continuum(s::Spaxel, N::QSIntensity, params::Vector{<:Real}, punits::Vector{<:Unitful.Units}, 
    cube_fitter::CubeFitter, use_pah_templates::Bool, do_gaps::Bool, mask_lines::Bool; return_extinction::Bool=false) 

    # Get options from the cube fitter
    fopt = fit_options(cube_fitter)

    # Get the wavelength data 
    λ = s.λ
    nλ = length(λ)

    # Prepare outputs
    out_type = typeof(ustrip(λ[1]))
    λunit = unit(λ[1])
    contin = zeros(out_type, length(λ))
    pᵢ::Int64 = 1

    ##### 1. EXTINCTION AND ABSORPTION FEATURES #####

    # --> General dust extinction curve 
    ext_gas, ext_stars, dp = extinction_profiles(λ, params, pᵢ, fopt.fit_uv_bump, fopt.extinction_curve)
    pᵢ += dp
    # --> Silicate absorption features
    abs_sil, dp, _, _, _ = silicate_absorption(λ, params, pᵢ, cube_fitter)
    pᵢ += dp
    abs_tot = ones(out_type, length(λ))
    # --> Ice+CH absorption features
    if fopt.fit_ch_abs
        abs_tot .*= extinction_factor.(τ_ice(λ, cube_fitter), params[pᵢ]*params[pᵢ+1], screen=true)
        abs_tot .*= extinction_factor.(τ_ch(λ, cube_fitter), params[pᵢ+1], screen=true) 
        pᵢ += 2
    end
    # --> Other absorption features
    for k ∈ 1:cube_fitter.n_abs_feat
        # this needs a type hint because it can't be inferred if we're not fitting any absorption profiles,
        # and it affects the type inference of abs_tot and hence ext_gas and ext_stars 
        unit_check(punits[pᵢ+1], λunit)
        unit_check(punits[pᵢ+2], λunit)
        unit_check(punits[pᵢ+3], λunit^-1)
        abs_tot .*= extinction_factor.(Drude.(λ, 1.0, params[pᵢ+1]*λunit, params[pᵢ+2]*λunit, params[pᵢ+3]/λunit), 
            params[pᵢ], screen=true)
        pᵢ += 4
    end
    # --> combine everything together with optional covering fraction
    Cf = 1.0
    if fopt.fit_covering_frac
        Cf = params[pᵢ]
        pᵢ += 1
    end
    @. ext_stars = Cf * ext_stars * abs_sil * abs_tot + (1.0 - Cf)
    @. ext_gas = Cf * ext_gas * abs_sil * abs_tot + (1.0 - Cf)

    ##### 2. POLYNOMIALS    #####

    # Additive polynomials 
    if cube_fitter.apoly_degree ≥ 0
        coeffs = params[pᵢ:pᵢ+cube_fitter.apoly_degree]
        # rescale wavelengths to range from -1 to +1
        x = ustrip.(-1.0 .+ 2.0 ./ (s.λ[end].-s.λ[1]) .* (s.λ.-s.λ[1]))
        if cube_fitter.apoly_type == :Legendre
            contin .+= clamp.(Legendre(coeffs).(x), 0., Inf)
        elseif cube_fitter.apoly_type == :Chebyshev
            contin .+= clamp.(Chebyshev(coeffs).(x), 0., Inf)
        else 
            error("Unrecognized polynomial type: $(cube_fitter.apoly_type)")
        end
        # Additive polynomials are not affected by any dust extinction or 
        # multiplicative polynomials, since they're meant to model background/
        # foreground emission or other systematics.
        pᵢ += cube_fitter.apoly_degree + 1
    end

    # Multiplicative polynomials
    mpoly = ones(out_type, length(λ))
    if cube_fitter.mpoly_degree ≥ 1
        coeffs = [1.0; params[pᵢ:pᵢ+cube_fitter.mpoly_degree-1]]
        # rescale wavelengths to range from -1 to +1
        x = ustrip.(-1.0 .+ 2.0 ./ (s.λ[end].-s.λ[1]) .* (s.λ.-s.λ[1]))
        if cube_fitter.mpoly_type == :Legendre 
            mpoly .= Legendre(coeffs).(x)
        elseif cube_fitter.mpoly_type == :Chebyshev 
            mpoly .= Chebyshev(coeffs).(x)
        else 
            error("Unrecognized polynomial type: $(cube_fitter.apoly_type)")
        end
        # clamp extreme values
        mpoly .= clamp.(mpoly, 0.1, 10.)
        pᵢ += cube_fitter.mpoly_degree
    end

    #############################

    # Collect the stellar velocities
    stel_vel = stel_sig = 0.0u"km/s"
    if fopt.fit_stellar_continuum
        unit_check(punits[pᵢ], u"km/s")
        unit_check(punits[pᵢ+1], u"km/s")
        stel_vel = params[pᵢ]*u"km/s"
        stel_sig = params[pᵢ+1]*u"km/s"
        pᵢ += 2
    end

    #############################
    
    ##### 3. FE II EMISSION #####

    gap_masks = get_gap_masks(s.λ, cube_fitter.spectral_region.gaps)
    if fopt.fit_opt_na_feii
        unit_check(punits[pᵢ+1], u"km/s")
        unit_check(punits[pᵢ+2], u"km/s")
        conv_na_feii = zeros(nλ, 1)
        for (gi, gap_mask) in enumerate(gap_masks)
            # split if the spectrum has a few separated regions
            conv_na_feii[gap_mask, 1] .= convolve_losvd(cube_fitter.feii.na_fft, cube_fitter.feii.vsysts[gi], 
                params[pᵢ+1]*u"km/s", params[pᵢ+2]*u"km/s", s.vres, sum(gap_mask), temp_fft=true, 
                npad_in=cube_fitter.feii.npad)
        end
        # These templates also need the solid angle divided out
        conv_na_feii[:, 1] ./= s.area_sr
        conv_na_feii[.~isfinite.(conv_na_feii[:, 1]), 1] .= 0.0   # (area_sr may be 0 at some points)
        contin .+= params[pᵢ] .* safenorm(conv_na_feii[:, 1]) .* ext_gas
        pᵢ += 3
    end
    if fopt.fit_opt_br_feii
        unit_check(punits[pᵢ+1], u"km/s")
        unit_check(punits[pᵢ+2], u"km/s")
        conv_br_feii = zeros(nλ, 1)
        for (gi, gap_mask) in enumerate(gap_masks)
            conv_br_feii[gap_mask, 1] .= convolve_losvd(cube_fitter.feii.br_fft, cube_fitter.feii.vsysts[gi], 
                params[pᵢ+1]*u"km/s", params[pᵢ+2]*u"km/s", s.vres, sum(gap_mask), temp_fft=true, 
                npad_in=cube_fitter.feii.npad)
        end
        conv_br_feii[:, 1] ./= s.area_sr
        conv_br_feii[.~isfinite.(conv_br_feii[:, 1]), 1] .= 0.0   # (area_sr may be 0 at some points)
        contin .+= params[pᵢ] .* safenorm(conv_br_feii[:, 1]) .* ext_gas
        pᵢ += 3
    end

    ##### 4. POWER LAWS #####

    for j ∈ 1:cube_fitter.n_power_law
        # Reference wavelength at the median wavelength of the input spectrum
        pl = power_law(λ, params[pᵢ+1], median(λ))
        contin .+= params[pᵢ] .* safenorm(pl) .* ext_gas .* mpoly
        pᵢ += 2
    end

    ##### 5. THERMAL DUST CONTINUA #####

    for i ∈ 1:cube_fitter.n_dust_cont
        unit_check(punits[pᵢ+1], u"K")
        bb = Blackbody_modified.(λ, params[pᵢ+1]*u"K", N)
        contin .+= params[pᵢ] .* safenorm(bb) .* ext_gas .* mpoly
        pᵢ += 2
    end

    ##### 6. SILICATE EMISSION #####

    if fopt.fit_sil_emission
        # Add Silicate emission from hot dust (amplitude, temperature, covering fraction, warm tau, cold tau)
        # Ref: Gallimore et al. 2010
        unit_check(punits[pᵢ+1], u"K")
        unit_check(punits[pᵢ+5], λunit)
        sil_emission = silicate_emission(λ, params[pᵢ+1]*u"K", params[pᵢ+2], params[pᵢ+3], params[pᵢ+4], 
            params[pᵢ+5]*λunit, N, cube_fitter)
        contin .+= params[pᵢ] .* safenorm(sil_emission) .* abs_tot .* mpoly
        pᵢ += 6
    end

    ##### 7. PSF TEMPLATES #####

    if haskey(s.aux, "templates") && haskey(s.aux, "channel_masks") && size(s.aux["templates"], 2) > 0
        # use the in-place version for F A S T  (its not actually that much faster lmao)
        dp = get_normalized_templates!(contin, λ, params, s.aux["templates"], s.aux["channel_masks"], 
            fopt.fit_temp_multexp, mpoly, pᵢ)
        pᵢ += dp
    end

    ##### 8. PAH EMISSION FEATURES #####

    if use_pah_templates
        contin .+= params[pᵢ] .* cube_fitter.dust_interpolators["smith3"](ustrip.(λ)) .* ext_gas .* mpoly
        contin .+= params[pᵢ+1] .* cube_fitter.dust_interpolators["smith4"](ustrip.(λ)) .* ext_gas .* mpoly
        pᵢ += 2
    else
        for dcomplex in model(cube_fitter).dust_features.profiles   # <- iterates over PAH complexes
            for component in dcomplex                               # <- iterates over each individual component
                profile = component.profile
                if profile == :Drude
                    unit_check(punits[pᵢ+1], λunit)
                    unit_check(punits[pᵢ+2], λunit)
                    unit_check(punits[pᵢ+3], λunit^-1)
                    contin .+= Drude.(λ, params[pᵢ], params[pᵢ+1]*λunit, params[pᵢ+2]*λunit,
                        params[pᵢ+3]/λunit) .* ext_gas
                    pᵢ += 4
                elseif profile == :PearsonIV
                    unit_check(punits[pᵢ+1], λunit)
                    unit_check(punits[pᵢ+2], λunit)
                    contin .+= PearsonIV.(λ, params[pᵢ], params[pᵢ+1]*λunit, params[pᵢ+2]*λunit, 
                        params[pᵢ+3], params[pᵢ+4]) .* ext_gas
                    pᵢ += 5
                end
            end
        end
    end

    ##### 9. STELLAR POPULATIONS #####

    if fopt.fit_stellar_continuum
        contin .+= stellar_populations_nnls(s, contin, ext_stars.*mpoly, stel_vel, stel_sig, cube_fitter; do_gaps=do_gaps, 
            mask_lines=mask_lines)[1]
    end

    if return_extinction
        return contin, ext_gas
    end

    contin
end


"""
    model_pah_residuals(λ, params, cube_fitter, ext_curve, return_components)

Create a model of the PAH features at the given wavelengths `λ`, given the parameter vector `params`.
Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments
- `s`: The spaxel object containing the wavelength data, normalization, templates, and channel masks.
- `params`: Parameter vector. Parameters should be ordered as: `(amp, center, fwhm) for each PAH profile`
- `cube_fitter`: The CubeFitter object containing the fitting options
- `template_norm`: The normalization PSF template that was fit using model_continuum
- `nuc_temp_fit`: Whether or not to apply the PSF normalization template
- `return_components`: Whether or not to return the individual components of the fit as a dictionary, in
    addition to the overall fit
"""
function model_pah_residuals(s::Spaxel, params::Vector{<:Real}, punits::Vector{<:Unitful.Units}, 
    ext_gas::Vector{<:Real}, cube_fitter::CubeFitter, return_components::Bool) 

    λ = s.λ
    λunit = unit(λ[1])

    # Prepare outputs
    out_type = typeof(ustrip(λ[1]))
    comps = Dict{String, Union{Vector{out_type}, Vector{typeof(s.N)}}}()
    contin = zeros(out_type, length(λ))

    pᵢ = 1
    for (k, dcomplex) in enumerate(model(cube_fitter).dust_features.profiles)   # <- iterates over PAH complexes
        for (j, component) in enumerate(dcomplex)                               # <- iterates over each individual component
            profile = component.profile
            if profile == :Drude
                unit_check(punits[pᵢ+1], λunit)
                unit_check(punits[pᵢ+2], λunit)
                unit_check(punits[pᵢ+3], λunit^-1)
                comps["dust_feat_$(k)_$(j)"] = Drude.(λ, params[pᵢ], params[pᵢ+1]*λunit, params[pᵢ+2]*λunit,
                    params[pᵢ+3]/λunit)
                pᵢ += 4
            elseif profile == :PearsonIV
                unit_check(punits[pᵢ+1], λunit)
                unit_check(punits[pᵢ+2], λunit)
                comps["dust_feat_$(k)_$(j)"] = PearsonIV.(λ, params[pᵢ], params[pᵢ+1]*λunit, params[pᵢ+2]*λunit, 
                    params[pᵢ+3], params[pᵢ+4]) 
                pᵢ += 5
            end
            contin .+= comps["dust_feat_$(k)_$(j)"] .* ext_gas
        end
    end

    # Return components, if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency
function model_pah_residuals(s::Spaxel, params::Vector{<:Real}, punits::Vector{<:Unitful.Units}, 
    ext_gas::Vector{<:Real}, cube_fitter::CubeFitter) 

    λ = s.λ
    λunit = unit(λ[1])

    # Prepare outputs
    out_type = typeof(ustrip(λ[1]))
    contin = zeros(out_type, length(λ))

    pᵢ = 1
    for dcomplex in model(cube_fitter).dust_features.profiles   # <- iterates over PAH complexes
        for component in dcomplex                               # <- iterates over each individual component
            profile = component.profile
            if profile == :Drude
                unit_check(punits[pᵢ+1], λunit)
                unit_check(punits[pᵢ+2], λunit)
                unit_check(punits[pᵢ+3], λunit^-1)
                contin .+= Drude.(λ, params[pᵢ], params[pᵢ+1]*λunit, params[pᵢ+2]*λunit,
                    params[pᵢ+3]/λunit) .* ext_gas
                pᵢ += 4
            elseif profile == :PearsonIV
                unit_check(punits[pᵢ+1], λunit)
                unit_check(punits[pᵢ+2], λunit)
                contin .+= PearsonIV.(λ, params[pᵢ], params[pᵢ+1]*λunit, params[pᵢ+2]*λunit, 
                    params[pᵢ+3], params[pᵢ+4]) .* ext_gas
                pᵢ += 5
            end
        end
    end

    contin
end


"""
    model_line_residuals(s, params, cube_fitter, template_norm, nuc_fit_norm, return_components) 

Create a model of the emission lines at the given wavelengths `λ`, given the parameter vector `params`.

Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments {S<:Integer}
- `s`: The spaxel object containing the wavelength data, normalization, templates, and channel masks.
- `params`: Parameter vector.
- `cube_fitter`: The CubeFitter object.
- `return_components`: Whether or not to return the individual components of the fit as a dictionary, in 
addition to the overall fit
"""
function model_line_residuals(s::Spaxel, params::Vector{<:Real}, punits::Vector{<:Unitful.Units}, 
    lines::FitFeatures, lsf::Function, ext_gas::Vector{<:Real}, mask::BitVector, return_components::Bool) 

    λ = s.λ[mask]

    # Prepare outputs
    out_type = typeof(ustrip(λ[1]))
    comps = Dict{String, Union{Vector{out_type}, Vector{typeof(s.N)}}}()
    contin = zeros(out_type, length(λ))

    pᵢ = 1
    for (k, line) in enumerate(lines.profiles)   # <- iterates over emission lines
        amp_1 = voff_1 = fwhm_1 = nothing        #    (initializing some variables)
        for (j, component) in enumerate(line)    # <- iterates over each velocity component in one line (for multi-component fits)
            profile = component.profile
            λ₀ = lines.λ₀[k]
            # Unpack the components of the line
            unit_check(punits[pᵢ], NoUnits)
            unit_check(punits[pᵢ+1], u"km/s")
            if (j == 1) || !lines.config.rel_fwhm
                unit_check(punits[pᵢ+2], u"km/s")
            else 
                unit_check(punits[pᵢ+2], NoUnits)
            end
            amp = params[pᵢ]
            voff = params[pᵢ+1]*u"km/s"
            fwhm = params[pᵢ+2]*((lines.config.rel_fwhm && j > 1) ? 1.0 : u"km/s")
            pᵢ += 3
            if profile == :GaussHermite
                # Get additional h3, h4 components
                unit_check(punits[pᵢ], NoUnits)
                unit_check(punits[pᵢ+1], NoUnits)
                h3 = params[pᵢ]
                h4 = params[pᵢ+1]
                pᵢ += 2
            elseif profile == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                unit_check(punits[pᵢ], NoUnits)
                η = params[pᵢ]
                pᵢ += 1
            end
            # Save the j = 1 (primary component) parameters for reference 
            if isone(j)
                amp_1 = amp
                voff_1 = voff
                fwhm_1 = fwhm
            # For the additional components, we (optionally) parametrize them this way to essentially give them soft constraints
            # relative to the primary component
            else
                if lines.config.rel_amp
                    amp *= amp_1
                end
                if lines.config.rel_voff
                    voff += voff_1
                end
                if lines.config.rel_fwhm
                    fwhm *= fwhm_1
                end
            end

            # Broaden the FWHM by the instrumental FWHM at the location of the line
            fwhm_inst = lsf(λ₀)
            fwhm = hypot(fwhm, fwhm_inst)

            # Convert voff in km/s to mean wavelength
            mean_wave = λ₀ + Doppler_width_λ(voff, λ₀)
            # Convert FWHM from km/s to wavelength units
            fwhm_wave = Doppler_width_λ(fwhm, λ₀)

            # Evaluate line profile
            if profile == :Gaussian
                comps["line_$(k)_$(j)"] = Gaussian.(λ, amp, mean_wave, fwhm_wave)
            elseif profile == :Lorentzian
                comps["line_$(k)_$(j)"] = Lorentzian.(λ, amp, mean_wave, fwhm_wave)
            elseif profile == :GaussHermite
                comps["line_$(k)_$(j)"] = GaussHermite.(λ, amp, mean_wave, fwhm_wave, h3, h4)
            elseif profile == :Voigt
                comps["line_$(k)_$(j)"] = Voigt.(λ, amp, mean_wave, fwhm_wave, η)
            else
                error("Unrecognized line profile $(profile)!")
            end

            # Add the line profile into the overall model
            contin .+= comps["line_$(k)_$(j)"]
        end
    end

    # Apply extinction
    contin .*= ext_gas[mask]

    # Return components if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency --> not allocating the dictionary improves performance DRAMATICALLY
function model_line_residuals(s::Spaxel, params::Vector{<:Real}, punits::Vector{<:Unitful.Units}, 
    lines::FitFeatures, lsf::Function, ext_gas::Vector{<:Real}, mask::BitVector) 

    λ = s.λ[mask]

    # Prepare outputs
    out_type = eltype(params)
    contin = zeros(out_type, length(λ))

    pᵢ = 1
    for (k, line) in enumerate(lines.profiles)   # <- iterates over emission lines
        amp_1 = voff_1 = fwhm_1 = nothing        #    (initializing some variables)
        for (j, component) in enumerate(line)    # <- iterates over each velocity component in one line (for multi-component fits)
            profile = component.profile
            λ₀ = lines.λ₀[k]
            # Unpack the components of the line
            unit_check(punits[pᵢ], NoUnits)
            unit_check(punits[pᵢ+1], u"km/s")
            if (j == 1) || !lines.config.rel_fwhm
                unit_check(punits[pᵢ+2], u"km/s")
            else 
                unit_check(punits[pᵢ+2], NoUnits)
            end
            amp = params[pᵢ]
            voff = params[pᵢ+1]*u"km/s"
            fwhm = params[pᵢ+2]*((lines.config.rel_fwhm && j > 1) ? 1.0 : u"km/s")
            pᵢ += 3
            if profile == :GaussHermite
                # Get additional h3, h4 components
                unit_check(punits[pᵢ], NoUnits)
                unit_check(punits[pᵢ+1], NoUnits)
                h3 = params[pᵢ]
                h4 = params[pᵢ+1]
                pᵢ += 2
            elseif profile == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                unit_check(punits[pᵢ], NoUnits)
                η = params[pᵢ]
                pᵢ += 1
            end
            # Save the j = 1 (primary component) parameters for reference 
            if isone(j)
                amp_1 = amp
                voff_1 = voff
                fwhm_1 = fwhm
            # For the additional components, we (optionally) parametrize them this way to essentially give them soft constraints
            # relative to the primary component
            else
                if lines.config.rel_amp
                    amp *= amp_1
                end
                if lines.config.rel_voff
                    voff += voff_1
                end
                if lines.config.rel_fwhm
                    fwhm *= fwhm_1
                end
            end

            # Broaden the FWHM by the instrumental FWHM at the location of the line
            fwhm_inst = lsf(λ₀)
            fwhm = hypot(fwhm, fwhm_inst)

            # Convert voff in km/s to mean wavelength
            mean_wave = λ₀ + Doppler_width_λ(voff, λ₀)
            # Convert FWHM from km/s to wavelength units
            fwhm_wave = Doppler_width_λ(fwhm, λ₀)

            # Evaluate line profile
            if profile == :Gaussian
                contin .+= Gaussian.(λ, amp, mean_wave, fwhm_wave)
            elseif profile == :Lorentzian
                contin .+= Lorentzian.(λ, amp, mean_wave, fwhm_wave)
            elseif profile == :GaussHermite
                contin .+= GaussHermite.(λ, amp, mean_wave, fwhm_wave, h3, h4)
            elseif profile == :Voigt
                contin .+= Voigt.(λ, amp, mean_wave, fwhm_wave, η)
            else
                error("Unrecognized line profile $(profile)!")
            end
        end
    end

    # Apply extinction
    contin .*= ext_gas[mask]

    contin
end


"""
    calculate_flux(profile, amp, amp_err, peak, peak_err, fwhm, fwhm_err; <keyword_args>)

Calculate the integrated flux of a spectral feature, i.e. a PAH or emission line. Calculates the integral
of the feature profile, using an analytic form if available, otherwise integrating numerically with QuadGK.
"""
function calculate_flux(profile::Symbol, λ::AbstractVector{<:QWave}, amp::T, amp_err::T, peak::S, peak_err::S, fwhm::S, fwhm_err::S;
    asym=nothing, asym_err=nothing, m=nothing, m_err=nothing, ν=nothing, ν_err=nothing, h3=nothing,
    h3_err=nothing, h4=nothing, h4_err=nothing, η=nothing, η_err=nothing, propagate_err::Bool=true) where {T<:Number,S<:Number}

    @debug "calculate_flux: profile=$profile, amp=$(ustrip(amp)), peak=$(ustrip(peak)), fwhm=$(ustrip(fwhm)), propagate_err=$propagate_err"
    # Evaluate the line profiles according to whether there is a simple analytic form
    # otherwise, integrate numerically with quadgk
    if profile == :Drude
        if isnothing(asym) || iszero(asym) 
            # (integral = π/2 * A * fwhm)
            flux, f_err = propagate_err ? ∫Drude(amp, amp_err, fwhm, fwhm_err) : (∫Drude(amp, fwhm), 0.0*amp*fwhm)
        else
            flux = NumericalIntegration.integrate(λ, Drude.(λ, amp, peak, fwhm, asym), Trapezoidal())
            if propagate_err
                err_l = abs(NumericalIntegration.integrate(λ, Drude.(λ, max(amp-amp_err, 0.0*amp), peak, 
                    max(fwhm-fwhm_err, eps()*unit(fwhm)), asym-asym_err), Trapezoidal()))
                err_u = abs(NumericalIntegration.integrate(λ, Drude.(λ, amp+amp_err, peak, fwhm+fwhm_err, asym+asym_err), 
                    Trapezoidal()))
                f_err = (err_l + err_u)/2
            else
                f_err = 0.0*amp*fwhm
            end
        end
    elseif profile == :PearsonIV
        flux = ∫PearsonIV(amp, fwhm, m, ν)
        if propagate_err
            e_upp = ∫PearsonIV(amp+amp_err, fwhm+fwhm_err, m+m_err, ν+ν_err) - flux
            e_low = flux - ∫PearsonIV(max(amp-amp_err, 0.0*amp), max(fwhm-fwhm_err, eps()*fwhm), max(m-m_err, 0.51), ν-ν_err)
            f_err = (e_upp + e_low) / 2
        else
            f_err = 0.0*amp*fwhm
        end
    elseif profile == :Gaussian
        # (integral = √(π / (4log(2))) * A * fwhm)
        flux, f_err = propagate_err ? ∫Gaussian(amp, amp_err, fwhm, fwhm_err) : (∫Gaussian(amp, fwhm), 0.0*amp*fwhm)
    elseif profile == :Lorentzian
        # (integral is the same as a Drude profile)
        flux, f_err = propagate_err ? ∫Lorentzian(amp, amp_err, fwhm, fwhm_err) : (∫Lorentzian(amp, fwhm), 0.0*amp*fwhm)
    elseif profile == :Voigt
        # (integral is an interpolation between Gaussian and Lorentzian)
        flux, f_err = propagate_err ? ∫Voigt(amp, amp_err, fwhm, fwhm_err, η, η_err) : (∫Voigt(amp, fwhm, η), 0.0*amp*fwhm)
    elseif profile == :GaussHermite
        # there is no general analytical solution (that I know of) for integrated Gauss-Hermite functions
        # (one probably exists but I'm too lazy to find it)
        # so we just do numerical integration for this case (trapezoid rule)
        flux = NumericalIntegration.integrate(λ, GaussHermite.(λ, amp, peak, fwhm, h3, h4), Trapezoidal())
        # estimate error by evaluating the integral at +/- 1 sigma
        if propagate_err
            err_l = abs(NumericalIntegration.integrate(λ, GaussHermite.(λ, max(amp-amp_err, 0.0*amp), peak, 
                max(fwhm-fwhm_err, eps()*fwhm), h3-h3_err, h4-h4_err), Trapezoidal()))
            err_u = abs(NumericalIntegration.integrate(λ, GaussHermite.(λ, amp+amp_err, peak, fwhm+fwhm_err, 
                        h3+h3_err, h4+h4_err), Trapezoidal()))
            f_err = (err_l + err_u)/2
        else
            f_err = 0.0*amp*fwhm
        end
    else
        error("Unrecognized line profile $profile")
    end

    flux, f_err
end


"""
    calculate_composite_params(λ, flux, λ0, fwhm_inst)

Calculate the w80 (width containing 80% of the flux) and Δv (asymmetry parameter), both in km/s,
for a line profile.
"""
function calculate_composite_params(λ::AbstractVector{T}, flux::AbstractVector{<:QSIntensity}, λ0::T,
    fwhm_inst::QVelocity) where {T<:QWave}

    n_pos = count(f -> f > 0.0*unit(f), flux)
    @debug "calculate_composite_params: λ0=$(ustrip(λ0)) $(unit(λ0)), nλ=$(length(λ)), n_positive_flux=$n_pos, fwhm_inst=$(ustrip(fwhm_inst)) $(unit(fwhm_inst))"
    # Get the cumulative distribution function
    m = flux .> 0.0*unit(flux[1])
    if sum(m) < 2
        @debug "calculate_composite_params: fewer than 2 positive flux points — returning zeros"
        return (0., 0., 0., 0.) .* u"km/s"
    end

    line_cdf = cumsum(flux[m] ./ sum(flux[m]))
    # this has to use the actual accurate doppler shift formula since large wavelength differences
    # can get close to c
    velocity = Doppler_shift_v.(λ[m], λ0) 

    # Cut below a threshold, otherwise Spline1D produces NaNs for some reason
    w = (line_cdf .> 0.001) .& (line_cdf .< 0.999)
    line_cdf = line_cdf[w]
    velocity = velocity[w]
    if length(line_cdf) < 4
        return (0., 0., 0., 0.) .* u"km/s"
    end
    # Cut any pixels that are not increasing the CDF (otherwise may cause the spline fit to fail)
    wd = BitVector([1; diff(line_cdf) .> 0.])
    line_cdf = line_cdf[wd]
    velocity = velocity[wd]
    if length(line_cdf) < 4
        return (0., 0., 0., 0) .* u"km/s"
    end

    # Interpolate to find where velocity is at 5, 10 and 90, and 95%
    vinterp = Spline1D(line_cdf, ustrip.(velocity), k=3, bc="extrapolate")
    v5 = vinterp(0.05)*u"km/s"
    v10 = vinterp(0.10)*u"km/s"
    vmed = vinterp(0.50)*u"km/s"
    v90 = vinterp(0.90)*u"km/s"
    v95 = vinterp(0.95)*u"km/s"

    # Calculate W80
    w80 = v90 - v10
    # Correct for intrumental line spread function (w80 = 1.09FWHM for a Gaussian)
    w80_inst = 1.09 * fwhm_inst
    w80 = sqrt(clamp(w80^2 - w80_inst^2, 0.0*u"km/s"^2, Inf*u"km/s"^2))

    # Calculate peak velocity
    finterp = Spline1D(ustrip.(velocity), ustrip.(flux[m][w][wd]), k=3, bc="extrapolate")
    guess = ustrip(velocity[nanargmax(ustrip.(flux)[m][w][wd])])
    res = Optim.optimize(v -> -finterp(v), guess-100.0, guess+100.0)
    vpeak = res.minimizer[1]*u"km/s"

    # Calculate Δv (see Harrison et al. 2014: https://ui.adsabs.harvard.edu/abs/2014MNRAS.441.3306H/abstract)
    Δv = (v5 + v95)/2 

    w80, Δv, vmed, vpeak
end


"""
    calculate_extra_parameters(λ, I, N, comps, n_channels, n_dust_cont, n_power_law, n_dust_feat, dust_profiles,
        n_abs_feat, fit_sil_emission, n_lines, n_acomps, n_comps, lines, flexible_wavesol, popt_c,
        popt_l, perr_c, perr_l, extinction, mask_lines, continuum, area_sr[, propagate_err])

Calculate extra parameters that are not fit, but are nevertheless important to know, for a given spaxel.
Currently this includes the integrated intensity, equivalent width, and signal to noise ratios of dust features and emission lines.
"""
function calculate_extra_parameters(s::Spaxel, s_model::Spaxel, cube_fitter::CubeFitter, comps::Dict, result_c::SpaxelFitResult, 
    result_l::SpaxelFitResult, continuum::Vector{<:Real}, propagate_err::Bool=true)

    @debug "calculate_extra_parameters: coords=$(s.coords), n_dust_feat=$(cube_fitter.n_dust_feat), n_lines=$(cube_fitter.n_lines), propagate_err=$propagate_err"

    popt_c = result_c.popt
    perr_c = result_c.perr
    popt_l = result_l.popt
    perr_l = result_l.perr

    # Normalization
    N = s.N
    @debug "calculate_extra_parameters: normalization N=$N"

    n_extra_df = length(get_flattened_nonfit_parameters(model(cube_fitter).dust_features))
    p_dust = Vector{Quantity{eltype(s.I)}}(undef, n_extra_df)
    p_dust_err = Vector{Quantity{eltype(s.I)}}(undef, n_extra_df)
    p_dust_err .= 0.
    pₒ = 1

    # Initial parameter vector index where dust profiles start
    i_split = count_cont_parameters(model(cube_fitter); split=true)
    pᵢ = i_split + 1

    if typeof(N) <: QPerAng
        @assert unit(s.λ[1]) == u"angstrom" "Wavelength and intensity units are not consistent!"
    end
    if typeof(N) <: QPerum
        @assert unit(s.λ[1]) == u"μm" "Wavelength and intensity units are not consistent!"
    end
    perwave_unit = typeof(N) <: QPerWave ? unit(N) : unit(N)*u"Hz"/unit(s.λ[1])

    # Loop through dust features
    # for ii ∈ 1:cube_fitter.n_dust_feat
    for dcomplex in model(cube_fitter).dust_features.profiles   # <- iterates over PAH complexes
        
        # initialize values
        total_flux = 0.0*u"erg/s/cm^2"
        total_flux_err = 0.0*u"erg/s/cm^2"
        total_eqw = 0.0*unit(s.λ[1])
        total_eqw_err = 0.0*unit(s.λ[1])
        total_snr = 0.0

        for component in dcomplex                               # <- iterates over each individual component 

            # unpack the parameters
            A, μ, fwhm = popt_c[pᵢ:pᵢ+2]
            A_err, μ_err, fwhm_err = perr_c[pᵢ:pᵢ+2]

            # Convert peak intensity to per-wavelength units if it isnt already (erg s^-1 cm^-2 μm^-1 sr^-1)
            A_cgs = match_fluxunits(A*N, 1.0perwave_unit, μ)
            if iszero(A) 
                A_cgs_err = propagate_err ? match_fluxunits(A_err*N, 1.0perwave_unit, μ) : 0.0perwave_unit
            else
                A_cgs_err = propagate_err ? hypot(A_err/A, 2(μ_err/μ))*A_cgs : 0.0perwave_unit
            end

            # Get the index of the central wavelength
            cent_ind = argmin(abs.(s.λ .- μ))
            
            # Integrate over the solid angle
            A_cgs = A_cgs * s.area_sr[cent_ind]
            A_cgs_err = A_cgs_err * s.area_sr[cent_ind]

            # Get the extinction profile at the center
            # ext = ext_curve[cent_ind] 
            # tn = tnorm[cent_ind]

            prof = component.profile
            if prof == :PearsonIV
                m, m_err = popt_c[pᵢ+3], perr_c[pᵢ+3]
                ν, ν_err = popt_c[pᵢ+4], perr_c[pᵢ+4]
                asym, asym_err = 0., 0.
            else
                m, m_err = 0., 0.
                ν, ν_err = 0., 0.
                asym, asym_err = popt_c[pᵢ+3], perr_c[pᵢ+3]
            end

            # Recreate the profiles 
            # (why do we need to do this if it's already stored in comps? short answer is that we need the +/-1 sigma
            #  versions so that we can calculate the appropriate error on the equivalent width)
            feature_err = nothing
            if prof == :Drude
                feature = Drude.(s_model.λ, A*N, μ, fwhm, asym)
                if propagate_err
                    feature_err = hcat(Drude.(s_model.λ, max(A*N-A_err*N, 0.0*N), μ, max(fwhm-fwhm_err, eps()*unit(fwhm)), asym-asym_err),
                                    Drude.(s_model.λ, A*N+A_err*N, μ, fwhm+fwhm_err, asym+asym_err))
                end
            elseif prof == :PearsonIV
                feature = PearsonIV.(s_model.λ, A*N, μ, fwhm, m, ν)
                if propagate_err
                    feature_err = hcat(PearsonIV.(s_model.λ, max(A*N-A_err*N, 0.0*N), μ, max(fwhm-fwhm_err, eps()*unit(fwhm)), m-m_err, ν-ν_err),
                                    PearsonIV.(s_model.λ, A*N+A_err*N, μ, fwhm+fwhm_err, m+m_err, ν+ν_err))
                end
            else
                error("Unrecognized PAH profile: $prof")
            end
            feature .*= comps["total_extinction_gas"]
            if propagate_err
                feature_err[:,1] .*= comps["total_extinction_gas"]
                feature_err[:,2] .*= comps["total_extinction_gas"]
            end

            # Calculate the flux using the utility function
            flux, f_err = calculate_flux(prof, cube_fitter.cube.λ, A_cgs, A_cgs_err, μ, μ_err, fwhm, fwhm_err, 
                asym=asym, asym_err=asym_err, m=m, m_err=m_err, ν=ν, ν_err=ν_err, propagate_err=propagate_err)
            unit_check(unit(flux), u"erg/s/cm^2")
            
            # Calculate the equivalent width using the utility function
            eqw, e_err = calculate_eqw(s_model.λ, feature, comps, false, feature_err=feature_err, propagate_err=propagate_err)
            unit_check(unit(eqw), unit(s.λ[1]))
            
            # snr = A*N*ext*tn / std(I[.!mask_lines .& (abs.(λ .- μ) .< 2fwhm)] .- continuum[.!mask_lines .& (abs.(λ .- μ) .< 2fwhm)])
            window = abs.(s.λ .- μ) .< 3fwhm
            feature_d = feature
            if s != s_model
                feature_d = Spline1D(ustrip.(s_model.λ), ustrip.(feature), k=1)(ustrip.(s.λ)) .* unit(feature[1])
            end
            snr = sum(feature_d[window]) / sqrt(sum((s.σ.*N)[window].^2))
            snr = round(snr, digits=2)

            @debug "PAH feature ($prof) with ($A_cgs, $μ, $fwhm, $asym, $m, $ν) and errors " * 
                "($A_cgs_err, $μ_err, $fwhm_err, $asym_err, $m_err, $ν_err)"
            @debug "Flux=$flux +/- $f_err, EQW=$eqw +/- $e_err, SNR=$snr"

            # increment the parameter index
            pᵢ += 4
            if prof == :PearsonIV
                pᵢ += 1
            end

            # flux units: erg s^-1 cm^-2 sr^-1 (integrated over μm)
            p_dust[pₒ] = flux
            p_dust_err[pₒ] = f_err

            # eqw units: μm
            p_dust[pₒ+1] = eqw
            p_dust_err[pₒ+1] = e_err

            # SNR, calculated as (peak amplitude) / (RMS intensity of the surrounding spectrum)
            # include the extinction factor when calculating the SNR
            p_dust[pₒ+2] = snr

            total_flux += flux
            total_flux_err = sqrt(total_flux_err^2 + f_err^2)
            total_eqw += eqw
            total_eqw_err = sqrt(total_eqw_err^2 + e_err^2)
            total_snr += snr

            pₒ += 3
        end

        # Values for the whole PAH complex
        p_dust[pₒ] = total_flux
        p_dust_err[pₒ] = total_flux_err
        p_dust[pₒ+1] = total_eqw
        p_dust_err[pₒ+1] = total_eqw_err
        p_dust[pₒ+2] = total_snr
        pₒ += 3

    end

    # Loop through lines
    n_extra_line = length(get_flattened_nonfit_parameters(model(cube_fitter).lines))
    p_lines = Vector{Quantity{eltype(s.I)}}(undef, n_extra_line)
    p_lines_err = Vector{Quantity{eltype(s.I)}}(undef, n_extra_line)
    p_lines_err .= 0.

    # Unpack the relative flags
    lines = model(cube_fitter).lines

    pₒ = pᵢ = 1
    for (k, line) in enumerate(lines.profiles)   # <- iterates over emission lines
        
        # initialize composite line profile
        amp_1 = amp_1_err = voff_1 = voff_1_err = fwhm_1 = fwhm_1_err = nothing
        total_profile = zeros(typeof(N), length(s_model.λ))
        profile_err_lo = profile_err_hi = nothing
        if propagate_err
            profile_err_lo = copy(total_profile)
            profile_err_hi = copy(total_profile)
        end

        # initialize values
        total_flux = 0.0*u"erg/s/cm^2"
        total_flux_err = 0.0*u"erg/s/cm^2"
        total_eqw = 0.0*unit(s.λ[1])
        total_eqw_err = 0.0*unit(s.λ[1])
        total_snr = 0.0

        λ0 = lines.λ₀[k]

        for (j, component) in enumerate(line)    # <- iterates over each velocity component in one line (for multi-component fits)

            # (\/ pretty much the same as the model_line_residuals function, but calculating the integrated intensities)
            amp = popt_l[pᵢ]
            amp_err = propagate_err ? perr_l[pᵢ] : 0.0

            voff = popt_l[pᵢ+1]
            voff_err = propagate_err ? perr_l[pᵢ+1] : 0.0*perr_l[pᵢ+1]

            fwhm = popt_l[pᵢ+2]
            fwhm_err = propagate_err ? perr_l[pᵢ+2] : 0.0*perr_l[pᵢ+2]
            pᵢ += 3

            # fill values with nothings for profiles that may / may not have them
            h3 = h3_err = h4 = h4_err = η = η_err = nothing

            if component.profile == :GaussHermite
                # Get additional h3, h4 components
                h3 = popt_l[pᵢ]
                h3_err = propagate_err ? perr_l[pᵢ] : 0.0
                h4 = popt_l[pᵢ+1]
                h4_err = propagate_err ? perr_l[pᵢ+1] : 0.0
                pᵢ += 2
            elseif component.profile == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                η = popt_l[pᵢ]
                η_err = propagate_err ? perr_l[pᵢ] : 0.0
                pᵢ += 1
            end

            # Save the j = 1 parameters for reference 
            if isone(j)
                amp_1 = amp
                amp_1_err = amp_err
                voff_1 = voff
                voff_1_err = voff_err
                fwhm_1 = fwhm
                fwhm_1_err = fwhm_err
            # For the additional components, we parametrize them this way to essentially give them soft constraints
            # relative to the primary component
            else
                if lines.config.rel_amp
                    amp_err = propagate_err ? hypot(amp_1_err*amp, amp_err*amp_1) : 0.
                    amp *= amp_1
                end
                if lines.config.rel_voff 
                    voff_err = propagate_err ? hypot(voff_err, voff_1_err) : 0.0*voff_err
                    voff += voff_1
                end
                if lines.config.rel_fwhm
                    fwhm_err = propagate_err ? hypot(fwhm_1_err*fwhm, fwhm_err*fwhm_1) : 0.0*fwhm_err
                    fwhm *= fwhm_1
                end
            end

            # Broaden the FWHM by the instrumental FWHM at the location of the line
            fwhm_inst = cube_fitter.lsf(λ0)
            fwhm_err = propagate_err ? fwhm / hypot(fwhm, fwhm_inst) * fwhm_err : 0.0*fwhm_err
            fwhm = hypot(fwhm, fwhm_inst)

            # Convert voff in km/s to mean wavelength in μm
            mean_wave = λ0 + Doppler_width_λ(voff, λ0)
            mean_wave_err = propagate_err ? λ0 / C_KMS * voff_err : 0.0*unit(mean_wave)

            # Convert FWHM from km/s to μm
            fwhm_wave = Doppler_width_λ(fwhm, λ0)
            fwhm_wave_err = propagate_err ? λ0 / C_KMS * fwhm_err : 0.0*unit(fwhm_wave)

            # Convert amplitude to per-wavelength units, put back in the normalization
            amp_cgs = match_fluxunits(amp*N, 1.0perwave_unit, mean_wave)
            if iszero(amp) 
                amp_cgs_err = propagate_err ? match_fluxunits(amp_err*N, 1.0perwave_unit, mean_wave) : 0.0perwave_unit
            else
                if typeof(N) <: QPerFreq
                    amp_cgs_err = propagate_err ? hypot(amp_err/amp, 2(mean_wave_err/mean_wave))*amp_cgs : 0.0perwave_unit
                else
                    amp_cgs_err = propagate_err ? abs(amp_err/amp*amp_cgs) : 0.0perwave_unit
                end
            end

            # Get the index of the central wavelength
            cent_ind = argmin(abs.(s.λ .- mean_wave))

            # Integrate over the solid angle
            amp_cgs *= s.area_sr[cent_ind]
            amp_cgs_err *= s.area_sr[cent_ind]

            # Get the extinction factor at the line center
            ext = comps["total_extinction_gas"][cent_ind]
            
            # Create the extincted line profile in units matching the continuum
            feature_err = nothing
            if component.profile == :Gaussian
                feature = Gaussian.(s_model.λ, amp*N, mean_wave, fwhm_wave)
                if propagate_err
                    feature_err = hcat(Gaussian.(s_model.λ, max(amp*N-amp_err*N, 0.0*N), mean_wave, max(fwhm_wave-fwhm_wave_err, eps()*unit(fwhm_wave))),
                                    Gaussian.(s_model.λ, amp*N+amp_err*N, mean_wave, fwhm_wave+fwhm_wave_err))
                end
            elseif component.profile == :Lorentzian
                feature = Lorentzian.(s_model.λ, amp*N, mean_wave, fwhm_wave)
                if propagate_err
                    feature_err = hcat(Lorentzian.(s_model.λ, max(amp*N-amp_err*N, 0.0*N), mean_wave, max(fwhm_wave-fwhm_wave_err, eps()*unit(fwhm_wave))),
                                    Lorentzian.(s_model.λ, amp*N+amp_err*N, mean_wave, fwhm_wave+fwhm_wave_err))
                end
            elseif component.profile == :GaussHermite
                feature = GaussHermite.(s_model.λ, amp*N, mean_wave, fwhm_wave, h3, h4)
                if propagate_err
                    feature_err = hcat(GaussHermite.(s_model.λ, max(amp*N-amp_err*N, 0.0*N), mean_wave, max(fwhm_wave-fwhm_wave_err, eps()*unit(fwhm_wave)), 
                        h3-h3_err, h4-h4_err), GaussHermite.(s_model.λ, amp*N+amp_err*N, mean_wave, fwhm_wave+fwhm_wave_err, h3+h3_err, h4+h4_err))
                end
            elseif component.profile == :Voigt
                feature = Voigt.(s_model.λ, amp*N, mean_wave, fwhm_wave, η)
                if propagate_err
                    feature_err = hcat(Voigt.(s_model.λ, max(amp*N-amp_err*N, 0.0*N), mean_wave, max(fwhm_wave-fwhm_wave_err, eps()*unit(fwhm_wave)), 
                        max(η-η_err, 0.)), Voigt.(s_model.λ, amp*N+amp_err*N, mean_wave, fwhm_wave+fwhm_wave_err, min(η+η_err, 1.)))
                end
            else
                error("Unrecognized line profile $(component.profile)")
            end
            feature .*= comps["total_extinction_gas"]
            if propagate_err
                feature_err[:,1] .*= comps["total_extinction_gas"]
                feature_err[:,2] .*= comps["total_extinction_gas"]
            end

            # Calculate line flux using the helper function
            p_lines[pₒ], p_lines_err[pₒ] = calculate_flux(component.profile, cube_fitter.cube.λ, amp_cgs, amp_cgs_err, 
                mean_wave, mean_wave_err, fwhm_wave, fwhm_wave_err, h3=h3, h3_err=h3_err, h4=h4, h4_err=h4_err, η=η, η_err=η_err, 
                propagate_err=propagate_err)
            
            # Calculate equivalent width using the helper function
            p_lines[pₒ+1], p_lines_err[pₒ+1] = calculate_eqw(s_model.λ, feature, comps, true, feature_err=feature_err, 
                propagate_err=propagate_err)

            # SNR
            region = .~s.mask_lines .& ((abs.(s.λ .- mean_wave)./mean_wave.*C_KMS) .< 5000.0u"km/s")
            continuum_d = continuum
            if s != s_model
                continuum_d = Spline1D(ustrip.(s_model.λ), continuum, k=1)(ustrip.(s.λ))
            end
            snr = amp*ext / std(s.I[region] .- continuum_d[region])
            p_lines[pₒ+2] = round(snr, digits=2)
            # rms = std(I[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)] .- continuum[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)]) 
            # p_lines[pₒ+2] = sum(feature) / (sqrt(sum(snr_filter)) * rms)
            # window = abs.(λ .- mean_wave) .< 3fwhm_wave
            # p_lines[pₒ+2] = sum(feature[window]) / sqrt(sum((σ.*N)[window].^2))

            # Add to the total line profile
            total_profile .+= feature
            if propagate_err
                profile_err_lo .+= feature_err[:,1]
                profile_err_hi .+= feature_err[:,2]
            end
            
            @debug "Line with ($amp_cgs, $mean_wave, $fwhm_wave) and errors ($amp_cgs_err, $mean_wave_err, $fwhm_wave_err)"
            @debug "Flux=$(p_lines[pₒ]) +/- $(p_lines_err[pₒ]), EQW=$(p_lines[pₒ+1]) +/- $(p_lines_err[pₒ+1]), SNR=$(p_lines[pₒ+2])"

            total_flux += p_lines[pₒ]
            total_flux_err = sqrt(total_flux_err^2 + p_lines_err[pₒ]^2)
            total_eqw += p_lines[pₒ+1]
            total_eqw_err = sqrt(total_eqw_err^2 + p_lines_err[pₒ+1]^2)
            total_snr += p_lines[pₒ+2]

            # Advance the output vector index by 3
            pₒ += 3
        end

        # total integrated fluxes/eqws/SNRs
        p_lines[pₒ] = total_flux
        p_lines_err[pₒ] = total_flux_err
        p_lines[pₒ+1] = total_eqw
        p_lines_err[pₒ+1] = total_eqw_err
        p_lines[pₒ+2] = total_snr
        pₒ += 3

        # Number of velocity components
        p_lines[pₒ] = cube_fitter.n_fit_comps[lines.names[k]][s.coords]

        # W80 and Δv parameters
        fwhm_inst = cube_fitter.lsf(λ0)
        w80, Δv, vmed, vpeak = calculate_composite_params(s_model.λ, total_profile, λ0, fwhm_inst)
        p_lines[pₒ+1] = w80
        p_lines[pₒ+2] = Δv
        p_lines[pₒ+3] = vmed
        p_lines[pₒ+4] = vpeak
        if propagate_err
            w80_lo, Δv_lo, vmed_lo, vpeak_lo = calculate_composite_params(s_model.λ, profile_err_lo, λ0, fwhm_inst)
            w80_hi, Δv_hi, vmed_hi, vpeak_hi = calculate_composite_params(s_model.λ, profile_err_hi, λ0, fwhm_inst)
            w80_err = abs(mean([w80 - w80_lo, w80_hi - w80]))
            Δv_err = abs(mean([Δv - Δv_lo, Δv_hi - Δv]))
            vmed_err = abs(mean([vmed - vmed_lo, vmed_hi - vmed]))
            vpeak_err = abs(mean([vpeak - vpeak_lo, vpeak_hi - vpeak]))
            p_lines_err[pₒ+1] = w80_err
            p_lines_err[pₒ+2] = Δv_err
            p_lines_err[pₒ+3] = vmed_err
            p_lines_err[pₒ+4] = vpeak_err
        end
        @debug "W80=$(w80), DELTA_V=$(Δv), VMED=$(vmed), VPEAK=$(vpeak)"

        pₒ += 5
    end

    # Make a fit result object
    params_extra = get_flattened_nonfit_parameters(model(cube_fitter))
    popt_extra = [p_dust; p_lines]
    perr_extra = [p_dust_err; p_lines_err]
    punit_extra = unit.(popt_extra)
    result = SpaxelFitResult(params_extra.names[1:end-2], popt_extra, [perr_extra perr_extra],
        [repeat([-Inf], length(popt_extra)).*punit_extra repeat([Inf], length(popt_extra)).*punit_extra],
        falses(length(popt_extra)), Vector{Union{Tie,Nothing}}([nothing for _ in 1:length(popt_extra)]))

    @debug "calculate_extra_parameters: done — $(length(popt_extra)) extra parameters computed ($(length(p_dust)) dust + $(length(p_lines)) line)"
    result
end


"""
    calculate_eqw(λ, profile, comps, line, n_dust_cont, n_power_law, n_abs_feat, n_dust_feat,
        fit_sil_emission, amp, amp_err, peak, peak_err, fwhm, fwhm_err; <keyword arguments>)

Calculate the equivalent width (in microns) of a spectral feature, i.e. a PAH or emission line. Calculates the
integral of the ratio of the feature profile to the underlying continuum.
"""
function calculate_eqw(λ::Vector{<:QWave}, feature::Vector{T}, comps::Dict, line::Bool;
    feature_err::Union{Matrix{T},Nothing}=nothing, propagate_err::Bool=true) where {T<:QSIntensity}

    @debug "calculate_eqw: line=$line, nλ=$(length(λ)), propagate_err=$propagate_err"
    contin = line ? comps["continuum_and_pahs"] : comps["continuum"]

    # The integrand feature/contin blows up where the continuum crosses or approaches zero; zero out any
    # non-finite points so the equivalent width stays finite instead of silently becoming Inf/NaN.
    safe_ratio(f) = (r = f ./ contin; r[.~isfinite.(r)] .= zero(eltype(r)); r)
    eqw = NumericalIntegration.integrate(λ, safe_ratio(feature), Trapezoidal())
    err = 0. * unit(λ[1])
    if propagate_err
        err_lo = eqw - NumericalIntegration.integrate(λ, safe_ratio(feature_err[:,1]), Trapezoidal())
        err_up = NumericalIntegration.integrate(λ, safe_ratio(feature_err[:,2]), Trapezoidal()) - eqw
        err = (err_up + err_lo) / 2
    end

    eqw, err
end

