
############################################## FITTING FUNCTIONS #############################################

function unit_check(u1::Unitful.Units, u2::Unitful.Units)
    @assert u1 == u2 "Uh oh...I think something's wrong with your units: $(u1) ≠ $(u2)"
end


# Helper function for getting the normalized templates for a given fit
function get_normalized_templates(λ::Vector{<:Number}, params::Vector{<:Number}, templates::Matrix{T}, 
    N::T, channel_masks::Vector{BitVector}, fit_temp_multexp::Bool, pstart::Integer) where {T<:Number}

    temp_norm = zeros(eltype(params), size(templates)...)
    # Add generic templates with a normalization parameter
    if fit_temp_multexp
        ex = multiplicative_exponentials(λ, params[pstart:pstart+7])
        for i in axes(templates, 2)
            temp_norm[:,i] .+= sum([ex[:,j] .* templates[:,i] ./ N for j in axes(ex,2)])
        end
        dp = 8
    else
        dp = 0
        for i in axes(templates, 2)
            for ch_mask in channel_masks
                temp_norm[ch_mask,i] .+= params[pstart+dp] .* templates[ch_mask, i] ./ N
                dp += 1
            end
        end
    end

    temp_norm, dp
end


# Helper function to calculate the normalized nuclear template amplitudes for a given fit
function get_nuctempfit_templates(params::Vector{<:Number}, templates::Matrix{<:Number}, 
    channel_masks::Vector{BitVector}, pstart::Integer)

    nuc_temp_norm = zeros(eltype(params), size(templates, 1))
    dp = 0
    for i in axes(templates, 2)
        for ch_mask in channel_masks
            nuc_temp_norm[ch_mask] .= params[pstart+dp] .* templates[ch_mask, i] 
            dp += 1
        end
    end

    nuc_temp_norm, dp
end


"""
    model_continuum(λ, params, N, cube_fitter, nuc_temp_fit, return_components)

Create a model of the continuum (including stellar+dust continuum, PAH features, and extinction, excluding emission lines)
at the given wavelengths `λ`, given the parameter vector `params`.

Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments
- `λ`: Wavelength vector of the spectrum to be fit
- `params`: Parameter vector. 
- `N`: The normalization.
- `cube_fitter`: The CubeFitter object with all of the fitting options.
- `nuc_temp_fit::Bool`: If true, the nuclear template is being fit, meaning we are to normalize the model by the templates (which should
    be the PSF templates here).
- `return_components::Bool`: Whether or not to return the individual components of the fit as a dictionary, in 
    addition to the overall fit
"""
function model_continuum(λ::Vector{<:QWave}, params::Vector{Quantity{<:Real}}, N::QSIntensity, cube_fitter::CubeFitter, 
    nuc_temp_fit::Bool, return_components::Bool)

    # Get options from the cube fitter
    fopt = fit_options(cube_fitter)

    # Prepare outputs
    out_type = typeof(ustrip(λ[1]))
    comps = Dict{String, Vector{out_type}}()
    contin = zeros(out_type, length(λ))
    pᵢ = 1

    ##### 1. EXTINCTION AND ABSORPTION FEATURES #####

    # --> General dust extinction curve 
    comps["extinction_gas"], comps["extinction_stars"], dp = extinction_profiles(λ, params, pᵢ, fopt.fit_uv_bump, 
        fopt.fit_covering_frac, fopt.extinction_curve)
    pᵢ += dp
    # --> Silicate absorption features
    comps["absorption_silicates"], dp, comps["absorption_oli"], comps["absorption_pyr"], comps["absorption_for"] = 
        silicate_absorption(λ, params, pᵢ, fopt.κ_abs, fopt.silicate_absorption, fopt.extinction_screen)
    pᵢ += dp
    abs_sil = comps["absorption_silicates"]
    abs_tot = one(out_type)
    # --> Ice+CH absorption features
    if fopt.fit_ch_abs
        comps["abs_ice"] = extinction_factor(τ_ice(λ), params[pᵢ] * params[pᵢ+1], screen=true)
        comps["abs_ch"] = extinction_factor(τ_ch(λ), params[pᵢ+1], screen=true) 
        abs_tot .*= comps["abs_ice"] .* comps["abs_ch"]
        pᵢ += 2
    end
    # --> Other absorption features
    for k ∈ 1:cube_fitter.n_abs_feat
        prof = Drude(λ, 1.0, params[pᵢ+1:pᵢ+3]...)
        comps["abs_feat_$k"] = extinction_factor(prof, params[pᵢ], screen=true)
        abs_tot .*= comps["abs_feat_$k"]
        pᵢ += 4
    end
    # --> combine everything together with optional covering fraction
    if fopt.fit_covering_frac
        Cf = params[pᵢ]
        comps["total_extinction_stars"] = @. Cf * comps["extinction_stars"] * abs_sil * abs_tot + (1 - Cf)
        comps["total_extinction_gas"] = @. Cf * comps["extinction_gas"] * abs_sil * abs_tot + (1 - Cf)
        pᵢ += 1
    else
        comps["total_extinction_stars"] = @. comps["extinction_stars"] * abs_sil * abs_tot
        comps["total_extinction_gas"] = @. comps["extinction_gas"] * abs_sil * abs_tot
    end
    
    ##### 2. STELLAR POPULATIONS #####

    if fopt.fit_stellar_continuum
        ssps = zeros(out_type, length(cube_fitter.ssps.λ), cube_fitter.n_ssps)
        # Interpolate the SSPs to the right ages/metallicities (this is slow)
        for i in 1:cube_fitter.n_ssps
            # normalize the templates by their median so that the amplitude is properly separated from the age and metallicity during fitting
            if cube_fitter.ssps.templates isa Vector{Spline2D}
                temp = [cube_fitter.ssps.templates[j](params[pᵢ+1], params[pᵢ+2]) for j in eachindex(cube_fitter.ssps.λ)]
            else
                temp = @view cube_fitter.ssps.templates[:,i]
            end
            # Sanity check on units
            unit_check(unit(temp[1]), unit(N)*u"sr/Msun")
            ssps[:,i] = @. params[pᵢ] * temp/maximum(temp)
            pᵢ += 3
        end
        # Convolve with a line-of-sight velocity distribution (LOSVD) according to the stellar velocity and dispersion
        conv_ssps = convolve_losvd(ssps, fopt.vsyst_ssp, params[pᵢ], params[pᵢ+1], cube_fitter.vres, length(λ))
        pᵢ += 2
        # Combine the convolved stellar templates together with the weights
        for i in 1:cube_fitter.n_ssps
            comps["SSP_$i"] = conv_ssps[:, i]
            contin .+= comps["SSP_$i"] .* comps["total_extinction_stars"]
        end
    end

    ##### 3. FE II EMISSION #####

    if fopt.fit_opt_na_feii
        conv_na_feii = convolve_losvd(cube_fitter.feii.na_fft, cube_fitter.feii.vsyst, params[pᵢ+1], params[pᵢ+2], 
            cube_fitter.vres, length(λ), temp_fft=true, npad_in=cube_fitter.feii.npad)
        comps["na_feii"] = params[pᵢ] .* conv_na_feii[:, 1] ./ maximum(conv_na_feii[:, 1])
        unit_check(comps["na_feii"][1], NoUnits)
        contin .+= comps["na_feii"] .* comps["total_extinction_gas"]
        pᵢ += 3
    end
    if fopt.fit_opt_br_feii
        conv_br_feii = convolve_losvd(cube_fitter.feii.br_fft, cube_fitter.feii.vsyst, params[pᵢ+1], params[pᵢ+2], 
            cube_fitter.vres, length(λ), temp_fft=true, npad_in=cube_fitter.feii.npad)
        comps["br_feii"] = params[pᵢ] .* conv_br_feii[:, 1] ./ maximum(conv_br_feii[:, 1])
        unit_check(comps["br_feii"][1], NoUnits)
        contin .+= comps["br_feii"] .* comps["total_extinction_gas"]
        pᵢ += 3
    end

    ##### 4. POWER LAWS #####

    for j ∈ 1:cube_fitter.n_power_law
        # Reference wavelength at the median wavelength of the input spectrum
        pl = power_law(λ, params[pᵢ+1], median(λ))
        comps["power_law_$j"] = params[pᵢ] .* pl ./ maximum(pl)
        unit_check(comps["power_law_$j"][1], NoUnits)
        contin .+= comps["power_law_$j"] .* comps["total_extinction_gas"]
        pᵢ += 2
    end

    ##### 5. THERMAL DUST CONTINUA #####

    for i ∈ 1:cube_fitter.n_dust_cont
        bb = Blackbody_modified(λ, params[pᵢ+1], unit(N))
        comps["dust_cont_$i"] = params[pᵢ] .* bb ./ maximum(bb)
        unit_check(comps["dust_cont_$i"][1], NoUnits)
        contin .+= comps["dust_cont_$i"] .* comps["total_extinction_gas"] 
        pᵢ += 2
    end

    ##### 6. SILICATE EMISSION #####

    if fopt.fit_sil_emission
        # Add Silicate emission from hot dust (amplitude, temperature, covering fraction, warm tau, cold tau)
        # Ref: Gallimore et al. 2010
        sil_emission = silicate_emission(λ, params[pᵢ+1:pᵢ+5]..., unit(N))
        comps["hot_dust"] = params[pᵢ] .* sil_emission ./ maximum(sil_emission)
        unit_check(comps["hot_dust"][1], NoUnits)
        # Do NOT apply silicate absorption to this component (duh)
        contin .+= comps["hot_dust"] .* abs_tot 
        pᵢ += 6
    end

    ##### 7. PSF TEMPLATES #####

    nuc_temp_norm = nothing
    if size(cube_fitter.templates, 2) > 0
        if !nuc_temp_fit
            temp_norm, dp = get_normalized_templates(λ, params, templates, N, cube_fitter.cube.spectral_region.channel_masks, 
                fopt.fit_temp_multexp, pᵢ)
            for i in axes(temp_norm, 2)
                comps["templates_$i"] = temp_norm[:,i]
            end
            contin .+= sumdim(temp_norm, 2)
            pᵢ += dp
        else
            # Fitting the nuclear spectrum - here the templates are just the PSF model, which we normalize by to make the spectrum
            # continuous.
            nuc_temp_norm, dp = get_nuctempfit_templates(params, templates, cube_fitter.cube.spectral_region.channel_masks, pᵢ)
            comps["templates_1"] = nuc_temp_norm
            pᵢ += dp
        end
    end

    ##### ------------------------ #####

    # Save the state of the continuum with everything except the PAH features #
    if nuc_temp_fit
        comps["continuum"] = contin .* comps["templates_1"]
    else
        comps["continuum"] = contin
    end

    ##### ------------------------ #####

    ##### 8. PAH EMISSION FEATURES #####

    if fopt.use_pah_templates
        pah3 = dust_interpolators["smith3"](λ)
        @. contin += params[pᵢ] * pah3/maximum(pah3) * comps["total_extinction_gas"]
        pah4 = dust_interpolators["smith4"](λ)
        @. contin += params[pᵢ+1] * pah4/maximum(pah4) * comps["total_extinction_gas"]
    else
        for (k, dcomplex) in enumerate(model(cube_fitter).dust_features.profiles)   # <- iterates over PAH complexes
            for (j, component) in enumerate(dcomplex)                               # <- iterates over each individual component
                profile = component.profile
                if profile == :Drude
                    comps["dust_feat_$(k)_$(j)"] = Drude(λ, params[pᵢ:pᵢ+3]...)
                    pᵢ += 4
                elseif profile == :PearsonIV
                    comps["dust_feat_$(k)_$(j)"] = PearsonIV(λ, params[pᵢ:pᵢ+4]...)
                    pᵢ += 5
                end
                @. contin += comps["dust_feat_$(k)_$(j)"] * comps["total_extinction_gas"] 
            end
        end
    end

    if nuc_temp_fit
        contin .*= comps["templates_1"]
    end

    ##### ------------------------ #####

    # Save the state of the continuum with everything including the PAH features # 
    comps["continuum_and_pahs"] = contin

    ##### ------------------------ #####

    # Return components if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency --> not allocating the dictionary improves performance DRAMATICALLY
function model_continuum(λ::Vector{<:QWave}, params::Vector{Quantity{<:Real}}, N::QSIntensity, cube_fitter::CubeFitter, 
    nuc_temp_fit::Bool)

    # Get options from the cube fitter
    fopt = fit_options(cube_fitter)

    # Prepare outputs
    out_type = typeof(ustrip(λ[1]))
    contin = zeros(out_type, length(λ))
    pᵢ = 1

    ##### 1. EXTINCTION AND ABSORPTION FEATURES #####

    # --> General dust extinction curve 
    ext_gas, ext_stars, dp = extinction_profiles(λ, params, pᵢ, fopt.fit_uv_bump, fopt.fit_covering_frac, fopt.extinction_curve)
    pᵢ += dp
    # --> Silicate absorption features
    abs_sil, dp, _, _, _ = silicate_absorption(λ, params, pᵢ, fopt.κ_abs, fopt.silicate_absorption, fopt.extinction_screen)
    pᵢ += dp
    abs_tot = one(out_type)
    # --> Ice+CH absorption features
    if fopt.fit_ch_abs
        abs_tot = abs_tot .* extinction_factor(τ_ice(λ), params[pᵢ] * params[pᵢ+1], screen=true)
        abs_tot = abs_tot .* extinction_factor(τ_ch(λ), params[pᵢ+1], screen=true) 
        pᵢ += 2
    end
    # --> Other absorption features
    for k ∈ 1:cube_fitter.n_abs_feat
        prof = Drude(λ, 1.0, params[pᵢ+1:pᵢ+3]...)
        abs_tot = abs_tot .* extinction_factor(prof, params[pᵢ], screen=true)
        pᵢ += 4
    end
    # --> combine everything together with optional covering fraction
    if fopt.fit_covering_frac
        Cf = params[pᵢ]
        @. ext_stars = Cf * ext_stars * abs_sil * abs_tot + (1 - Cf)
        @. ext_gas = Cf * ext_gas * abs_sil * abs_tot + (1 - Cf)
        pᵢ += 1
    else
        @. ext_stars = ext_stars * abs_sil * abs_tot
        @. ext_gas = ext_gas * abs_sil * abs_tot
    end
    
    ##### 2. STELLAR POPULATIONS #####

    if fopt.fit_stellar_continuum
        ssps = zeros(out_type, length(cube_fitter.ssps.λ), cube_fitter.n_ssps)
        # Interpolate the SSPs to the right ages/metallicities (this is slow)
        for i in 1:cube_fitter.n_ssps
            # normalize the templates by their median so that the amplitude is properly separated from the age and metallicity during fitting
            if cube_fitter.ssps.templates isa Vector{Spline2D}
                temp = [cube_fitter.ssps.templates[j](params[pᵢ+1], params[pᵢ+2]) for j in eachindex(cube_fitter.ssps.λ)]
            else
                temp = @view cube_fitter.ssps.templates[:,i]
            end
            # Sanity check on units
            unit_check(unit(temp[1]), unit(N)*u"sr/Msun")
            ssps[:,i] = params[pᵢ] .* temp ./ maximum(temp)
            pᵢ += 3
        end
        # Convolve with a line-of-sight velocity distribution (LOSVD) according to the stellar velocity and dispersion
        conv_ssps = convolve_losvd(ssps, fopt.vsyst_ssp, params[pᵢ], params[pᵢ+1], cube_fitter.vres, length(λ))
        pᵢ += 2
        @views for i in 1:cube_fitter.n_ssps
            contin .+= conv_ssps[:, i] .* ext_stars
        end
    end

    ##### 3. FE II EMISSION #####

    if fopt.fit_opt_na_feii
        conv_na_feii = convolve_losvd(cube_fitter.feii.na_fft, cube_fitter.feii.vsyst, params[pᵢ+1], params[pᵢ+2], 
            cube_fitter.vres, length(λ), temp_fft=true, npad_in=cube_fitter.feii.npad)
        @views contin .+= params[pᵢ] .* conv_na_feii[:, 1]./maximum(conv_na_feii[:, 1]) .* ext_gas
        pᵢ += 3
    end
    if fopt.fit_opt_br_feii
        conv_br_feii = convolve_losvd(cube_fitter.feii.br_fft, cube_fitter.feii.vsyst, params[pᵢ+1], params[pᵢ+2], 
            cube_fitter.vres, length(λ), temp_fft=true, npad_in=cube_fitter.feii.npad)
        @views contin .+= params[pᵢ] .* conv_br_feii[:, 1]./maximum(conv_br_feii[:, 1]) .* ext_gas
        pᵢ += 3
    end

    ##### 4. POWER LAWS #####

    for j ∈ 1:cube_fitter.n_power_law
        # Reference wavelength at the median wavelength of the input spectrum
        pl = power_law(λ, params[pᵢ+1], median(λ))
        contin .+= params[pᵢ] .* pl./maximum(pl) .* ext_gas
        pᵢ += 2
    end

    ##### 5. THERMAL DUST CONTINUA #####

    for i ∈ 1:cube_fitter.n_dust_cont
        bb = Blackbody_modified(λ, params[pᵢ+1], unit(N))
        contin .+= params[pᵢ] .* bb./maximum(bb) .* ext_gas
        pᵢ += 2
    end

    ##### 6. SILICATE EMISSION #####

    if fopt.fit_sil_emission
        # Add Silicate emission from hot dust (amplitude, temperature, covering fraction, warm tau, cold tau)
        # Ref: Gallimore et al. 2010
        sil_emission = silicate_emission(λ, params[pᵢ+1:pᵢ+5]..., unit(N))
        contin .+= params[pᵢ] .* sil_emission./maximum(sil_emission) .* abs_tot
        pᵢ += 6
    end

    ##### 7. PSF TEMPLATES #####

    nuc_temp_norm = nothing
    if size(cube_fitter.templates, 2) > 0
        if !nuc_temp_fit
            temp_norm, dp = get_normalized_templates(λ, params, templates, N, cube_fitter.cube.spectral_region.channel_masks, 
                fopt.fit_temp_multexp, pᵢ)
            contin .+= sumdim(temp_norm, 2)
            pᵢ += dp
        else
            # Fitting the nuclear spectrum - here the templates are just the PSF model, which we normalize by to make the spectrum
            # continuous.
            nuc_temp_norm, dp = get_nuctempfit_templates(params, templates, cube_fitter.cube.spectral_region.channel_masks, pᵢ)
            pᵢ += dp
        end
    end

    ##### 8. PAH EMISSION FEATURES #####

    if fopt.use_pah_templates
        pah3 = dust_interpolators["smith3"](λ)
        @. contin += params[pᵢ] * pah3/maximum(pah3) * ext_gas
        pah4 = dust_interpolators["smith4"](λ)
        @. contin += params[pᵢ+1] * pah4/maximum(pah4) * ext_gas
    else
        for dcomplex in model(cube_fitter).dust_features.profiles   # <- iterates over PAH complexes
            for component in dcomplex                               # <- iterates over each individual component
                profile = component.profile
                if profile == :Drude
                    contin .+= Drude(λ, params[pᵢ:pᵢ+3]...) .* ext_gas
                    pᵢ += 4
                elseif profile == :PearsonIV
                    contin .+= PearsonIV(λ, params[pᵢ:pᵢ+4]...) .* ext_gas
                    pᵢ += 5
                end
            end
        end
    end

    if nuc_temp_fit
        contin .*= nuc_temp_norm
    end

    contin
end


"""
    model_pah_residuals(λ, params, cube_fitter, ext_curve, return_components)

Create a model of the PAH features at the given wavelengths `λ`, given the parameter vector `params`.
Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments
- `λ`: Wavelength vector of the spectrum to be fit
- `params`: Parameter vector. Parameters should be ordered as: `(amp, center, fwhm) for each PAH profile`
- `cube_fitter`: The CubeFitter object containing the fitting options
- `ext_curve`: The extinction curve that was fit using model_continuum
- `template_norm`: The normalization PSF template that was fit using model_continuum
- `nuc_temp_fit`: Whether or not to apply the PSF normalization template
- `return_components`: Whether or not to return the individual components of the fit as a dictionary, in
    addition to the overall fit
"""
function model_pah_residuals(λ::Vector{<:QWave}, params::Vector{Quantity{<:Real}}, cube_fitter::CubeFitter, 
    ext_curve::Vector{<:Real}, template_norm::Union{Nothing,Vector{<:Real}}, nuc_temp_fit::Bool, return_components::Bool) 

    # Prepare outputs
    out_type = typeof(ustrip(λ[1]))
    comps = Dict{String, Vector{out_type}}()
    contin = zeros(out_type, length(λ))

    pᵢ = 1
    for (k, dcomplex) in enumerate(model(cube_fitter).dust_features.profiles)   # <- iterates over PAH complexes
        for (j, component) in enumerate(dcomplex)                               # <- iterates over each individual component
            profile = component.profile
            if profile == :Drude
                comps["dust_feat_$(k)_$(j)"] = Drude(λ, params[pᵢ:pᵢ+3]...)
                pᵢ += 4
            elseif profile == :PearsonIV
                comps["dust_feat_$(k)_$(j)"] = PearsonIV(λ, params[pᵢ:pᵢ+4]...)
                pᵢ += 5
            end
            contin .+= comps["dust_feat_$(k)_$(j)"] .* ext_curve
        end
    end

    # Apply PSF normalization
    if nuc_temp_fit
        contin .*= template_norm
    end

    # Return components, if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency
function model_pah_residuals(λ::Vector{<:QWave}, params::Vector{Quantity{<:Real}}, cube_fitter::CubeFitter, 
    ext_curve::Vector{<:Real}, template_norm::Union{Nothing,Vector{<:Real}}, nuc_temp_fit::Bool) 

    # Prepare outputs
    out_type = typeof(ustrip(λ[1]))
    contin = zeros(out_type, length(λ))

    pᵢ = 1
    for dcomplex in model(cube_fitter).dust_features.profiles   # <- iterates over PAH complexes
        for component in dcomplex                               # <- iterates over each individual component
            profile = component.profile
            if profile == :Drude
                contin .+= Drude(λ, params[pᵢ:pᵢ+3]...) .* ext_curve
                pᵢ += 4
            elseif profile == :PearsonIV
                contin .+= PearsonIV(λ, params[pᵢ:pᵢ+4]...) .* ext_curve
                pᵢ += 5
            end
        end
    end

    # Apply PSF normalization
    if nuc_temp_fit
        contin .*= template_norm
    end

    contin
end


"""
    model_line_residuals(λ, params, n_lines, n_comps, lines, flexible_wavesol, ext_curve, lsf,
        relative_flags, return_components) 

Create a model of the emission lines at the given wavelengths `λ`, given the parameter vector `params`.

Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments {S<:Integer}
- `λ::Vector{<:Real}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:Real}`: Parameter vector.
- `n_lines::S`: Number of lines being fit
- `n_comps::S`: Maximum number of profiles to be fit to any given line
- `lines::TransitionLines`: Object containing information about each transition line being fit.
- `flexible_wavesol::Bool`: Whether or not to allow small variations in tied velocity offsets, to account for a poor
wavelength solution in the data
- `ext_curve::Vector{<:Real}`: The extinction curve fit with model_{mir|opt}_continuum
- `lsf::Function`: A function giving the FWHM of the line-spread function in km/s as a function of rest-frame wavelength in microns.
- `relative_flags::BitVector`: BitVector giving flags for whether the amp, voff, and fwhm of additional line profiles should be
    parametrized relative to the main profile or not.
- `template_norm::Union{Nothing,Vector{<:Real}}`: The normalization PSF template that was fit using model_continuum
- `nuc_temp_fit::Bool`: Whether or not to apply the PSF normalization template
- `return_components::Bool=false`: Whether or not to return the individual components of the fit as a dictionary, in 
addition to the overall fit
"""
function model_line_residuals(λ::AbstractVector{<:QWave}, params::AbstractVector{<:Number}, cube_fitter::CubeFitter,
    ext_curve::Vector{<:Real}, lsf::Function, template_norm::Union{Nothing,Vector{<:Real}}, nuc_temp_fit::Bool, 
    return_components::Bool) 

    # Prepare outputs
    out_type = typeof(ustrip(λ[1]))
    comps = Dict{String, Vector{out_type}}()
    contin = zeros(out_type, length(λ))

    lines = model(cube_fitter).lines

    pᵢ = 1
    for (k, line) in enumerate(lines.profiles)   # <- iterates over emission lines
        amp_1 = voff_1 = fwhm_1 = nothing        #    (initializing some variables)
        for (j, component) in enumerate(line)    # <- iterates over each velocity component in one line (for multi-component fits)
            profile = component.profile
            λ₀ = lines.λ₀[k]
            # Unpack the components of the line
            amp = params[pᵢ]
            voff = params[pᵢ+1]
            fwhm = params[pᵢ+2]
            pᵢ += 3
            if profile == :GaussHermite
                # Get additional h3, h4 components
                h3 = params[pᵢ]
                h4 = params[pᵢ+1]
                pᵢ += 2
            elseif profile == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
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
                comps["line_$(k)_$(j)"] = Gaussian(λ, amp, mean_wave, fwhm_wave)
            elseif profile == :Lorentzian
                comps["line_$(k)_$(j)"] = Lorentzian(λ, amp, mean_wave, fwhm_wave)
            elseif profile == :GaussHermite
                comps["line_$(k)_$(j)"] = GaussHermite(λ, amp, mean_wave, fwhm_wave, h3, h4)
            elseif profile == :Voigt
                comps["line_$(k)_$(j)"] = Voigt(λ, amp, mean_wave, fwhm_wave, η)
            else
                error("Unrecognized line profile $(profile)!")
            end

            # Add the line profile into the overall model
            contin .+= comps["line_$(k)_$(j)"]
        end
    end

    # Apply extinction
    contin .*= ext_curve
    # Apply PSF normalization
    if nuc_temp_fit
        contin .*= template_norm
    end

    # Return components if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency --> not allocating the dictionary improves performance DRAMATICALLY
function model_line_residuals(λ::AbstractVector{<:QWave}, params::AbstractVector{<:Number}, cube_fitter::CubeFitter,
    ext_curve::Vector{<:Real}, lsf::Function, template_norm::Union{Nothing,Vector{<:Real}}, nuc_temp_fit::Bool) 

    # Prepare outputs
    out_type = eltype(params)
    contin = zeros(out_type, length(λ))

    lines = model(cube_fitter).lines

    pᵢ = 1
    for (k, line) in enumerate(lines.profiles)   # <- iterates over emission lines
        amp_1 = voff_1 = fwhm_1 = nothing        #    (initializing some variables)
        for (j, component) in enumerate(line)    # <- iterates over each velocity component in one line (for multi-component fits)
            profile = component.profile
            λ₀ = lines.λ₀[k]
            # Unpack the components of the line
            amp = params[pᵢ]
            voff = params[pᵢ+1]
            fwhm = params[pᵢ+2]
            pᵢ += 3
            if profile == :GaussHermite
                # Get additional h3, h4 components
                h3 = params[pᵢ]
                h4 = params[pᵢ+1]
                pᵢ += 2
            elseif profile == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
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
                contin .+= Gaussian(λ, amp, mean_wave, fwhm_wave)
            elseif profile == :Lorentzian
                contin .+= Lorentzian(λ, amp, mean_wave, fwhm_wave)
            elseif profile == :GaussHermite
                contin .+= GaussHermite(λ, amp, mean_wave, fwhm_wave, h3, h4)
            elseif profile == :Voigt
                contin .+= Voigt(λ, amp, mean_wave, fwhm_wave, η)
            else
                error("Unrecognized line profile $(profile)!")
            end
        end
    end

    # Apply extinction
    contin .*= ext_curve
    # Apply PSF normalization
    if nuc_temp_fit
        contin .*= template_norm
    end

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

    # Evaluate the line profiles according to whether there is a simple analytic form
    # otherwise, integrate numerically with quadgk
    if profile == :Drude
        if isnothing(asym) || iszero(asym) 
            # (integral = π/2 * A * fwhm)
            flux, f_err = propagate_err ? ∫Drude(amp, amp_err, fwhm, fwhm_err) : (∫Drude(amp, fwhm), 0.)
        else
            flux = NumericalIntegration.integrate(λ, Drude(λ, amp, peak, fwhm, asym), Trapezoidal())
            if propagate_err
                err_l = abs(NumericalIntegration.integrate(λ, Drude(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), 
                    asym-asym_err), Trapezoidal()))
                err_u = abs(NumericalIntegration.integrate(λ, Drude(λ, amp+amp_err, peak, fwhm+fwhm_err, asym+asym_err), 
                    Trapezoidal()))
                f_err = (err_l + err_u)/2
            else
                f_err = 0.
            end
        end
    elseif profile == :PearsonIV
        flux = ∫PearsonIV(amp, fwhm, m, ν)
        if propagate_err
            e_upp = ∫PearsonIV(amp+amp_err, fwhm+fwhm_err, m+m_err, ν+ν_err) - flux
            e_low = flux - ∫PearsonIV(max(amp-amp_err, 0.), max(fwhm-fwhm_err, eps()), max(m-m_err, 0.5), ν-ν_err)
            f_err = (e_upp + e_low) / 2
        else
            f_err = 0.
        end
    elseif profile == :Gaussian
        # (integral = √(π / (4log(2))) * A * fwhm)
        flux, f_err = propagate_err ? ∫Gaussian(amp, amp_err, fwhm, fwhm_err) : (∫Gaussian(amp, fwhm), 0.)
    elseif profile == :Lorentzian
        # (integral is the same as a Drude profile)
        flux, f_err = propagate_err ? ∫Lorentzian(amp, amp_err, fwhm, fwhm_err) : (∫Lorentzian(amp, fwhm), 0.)
    elseif profile == :Voigt
        # (integral is an interpolation between Gaussian and Lorentzian)
        flux, f_err = propagate_err ? ∫Voigt(amp, amp_err, fwhm, fwhm_err, η, η_err) : (∫Voigt(amp, fwhm, η), 0.)
    elseif profile == :GaussHermite
        # there is no general analytical solution (that I know of) for integrated Gauss-Hermite functions
        # (one probably exists but I'm too lazy to find it)
        # so we just do numerical integration for this case (trapezoid rule)
        flux = NumericalIntegration.integrate(λ, GaussHermite(λ, amp, peak, fwhm, h3, h4), Trapezoidal())
        # estimate error by evaluating the integral at +/- 1 sigma
        if propagate_err
            err_l = abs(NumericalIntegration.integrate(λ, GaussHermite(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), 
                        h3-h3_err, h4-h4_err), Trapezoidal()))
            err_u = abs(NumericalIntegration.integrate(λ, GaussHermite(λ, amp+amp_err, peak, fwhm+fwhm_err, 
                        h3+h3_err, h4+h4_err), Trapezoidal()))
            f_err = (err_l + err_u)/2
        else
            f_err = 0.
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

    # Get the cumulative distribution function
    m = flux .> 0.0*unit(flux[1])
    if sum(m) < 2
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
    guess = ustrip(velocity[nanargmax(flux[m][w][wd])])
    res = Optim.optimize(v -> -finterp(v), guess-100.0, guess+100.0)
    vpeak = res.minimizer[1]

    # Calculate Δv (see Harrison et al. 2014: https://ui.adsabs.harvard.edu/abs/2014MNRAS.441.3306H/abstract)
    Δv = (v5 + v95)/2 

    w80, Δv, vmed, vpeak
end


"""
    calculate_extra_parameters(λ, I, N, comps, n_channels, n_dust_cont, n_power_law, n_dust_feat, dust_profiles,
        n_abs_feat, fit_sil_emission, n_lines, n_acomps, n_comps, lines, flexible_wavesol, lsf, popt_c,
        popt_l, perr_c, perr_l, extinction, mask_lines, continuum, area_sr[, propagate_err])

Calculate extra parameters that are not fit, but are nevertheless important to know, for a given spaxel.
Currently this includes the integrated intensity, equivalent width, and signal to noise ratios of dust features and emission lines.
"""
function calculate_extra_parameters(cube_fitter::CubeFitter, λ::AbstractVector{<:QWave}, I::Vector{T}, σ::Vector{T}, N::QSIntensity, 
    comps::Dict, nuc_temp_fit::Bool, lsf::Function, popt_c::Vector{T}, popt_l::Vector{T}, perr_c::Vector{T}, perr_l::Vector{T}, 
    ext_curve::Vector{T}, extinction::Vector{T}, templates_psfnuc::Union{Nothing,Vector{<:Real}}, mask_lines::BitVector, 
    continuum::Vector{T}, area_sr::Vector{<:typeof(1.0u"sr")}, spaxel::CartesianIndex, propagate_err::Bool=true) where {T<:Real}

    @debug "Calculating extra parameters"

    # Normalization
    @debug "Normalization: $N"

    n_extra_df = length(get_flattened_nonfit_parameters(model(cube_fitter).dust_features))
    p_dust = Vector{Quantity{eltype(I)}}(undef, n_extra_df)
    p_dust_err = Vector{Quantity{eltype(I)}}(undef, n_extra_df)
    pₒ = 1

    # Initial parameter vector index where dust profiles start
    i_split = count_cont_parameters(model(cube_fitter); split=true)
    pᵢ = i_split + 1

    if typeof(N) <: QPerAng
        @assert unit(λ[1]) == u"angstrom" "Wavelength and intensity units are not consistent!"
    end
    if typeof(N) <: QPerum
        @assert unit(λ[1]) == u"μm" "Wavelength and intensity units are not consistent!"
    end
    perwave_unit = typeof(N) <: QPerWave ? unit(N) : unit(N)*u"Hz"/unit(λ[1])

    # Loop through dust features
    # for ii ∈ 1:cube_fitter.n_dust_feat
    for dcomplex in model(cube_fitter).dust_features.profiles   # <- iterates over PAH complexes
        
        # initialize values
        total_flux = 0.0*u"erg/s/cm^2"
        total_flux_err = 0.0*u"erg/s/cm^2"
        total_eqw = 0.0*unit(λ[1])
        total_eqw_err = 0.0*unit(λ[1])
        total_snr = 0.0
        total_snr_err = 0.0

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
            cent_ind = argmin(abs.(λ .- μ))
            
            # Integrate over the solid angle
            A_cgs = A_cgs * area_sr[cent_ind]
            if propagate_err
                A_cgs_err = A_cgs_err * area_sr[cent_ind]
            end

            # Get the extinction profile at the center
            # ext = ext_curve[cent_ind] 
            tnorm = nuc_temp_fit ? templates_psfnuc : ones(length(λ))
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
                feature = Drude(λ, A*N, μ, fwhm, asym)
                if propagate_err
                    feature_err = hcat(Drude(λ, max(A*N-A_err*N, 0.), μ, max(fwhm-fwhm_err, eps()), asym-asym_err),
                                    Drude(λ, A*N+A_err*N, μ, fwhm+fwhm_err, asym+asym_err))
                end
            elseif prof == :PearsonIV
                feature = PearsonIV(λ, A*N, μ, fwhm, m, ν)
                if propagate_err
                    feature_err = hcat(PearsonIV(λ, max(A*N-A_err*N, 0.), μ, max(fwhm-fwhm_err, eps()), m-m_err, ν-ν_err),
                                    PearsonIV(λ, A*N+A_err*N, μ, fwhm+fwhm_err, m+m_err, ν+ν_err))
                end
            else
                error("Unrecognized PAH profile: $prof")
            end
            feature .*= ext_curve .* tnorm
            if propagate_err
                feature_err[:,1] .*= ext_curve .* tnorm
                feature_err[:,2] .*= ext_curve .* tnorm
            end

            # Calculate the flux using the utility function
            flux, f_err = calculate_flux(prof, cube_fitter.cube.λ, A_cgs, A_cgs_err, μ, μ_err, fwhm, fwhm_err, 
                asym=asym, asym_err=asym_err, m=m, m_err=m_err, ν=ν, ν_err=ν_err, propagate_err=propagate_err)
            unit_check(unit(flux), u"erg/s/cm^2")
            
            # Calculate the equivalent width using the utility function
            eqw, e_err = calculate_eqw(λ, feature, comps, false, feature_err=feature_err, propagate_err=propagate_err)
            unit_check(unit(eqw), unit(λ[1]))
            
            # snr = A*N*ext*tn / std(I[.!mask_lines .& (abs.(λ .- μ) .< 2fwhm)] .- continuum[.!mask_lines .& (abs.(λ .- μ) .< 2fwhm)])
            window = abs.(λ .- μ) .< 3fwhm
            snr = sum(feature[window]) / sqrt(sum(σ[window].^2))

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
    p_lines = Vector{Quantity{eltype(I)}}(undef, n_extra_line)
    p_lines_err = Vector{Quantity{eltype(I)}}(undef, n_extra_line)

    # Unpack the relative flags
    lines = model(cube_fitter).lines

    pₒ = pᵢ = 1
    for (k, line) in enumerate(lines.profiles)   # <- iterates over emission lines
        
        # initialize composite line profile
        amp_1 = amp_1_err = voff_1 = voff_1_err = fwhm_1 = fwhm_1_err = nothing
        total_profile = zeros(typeof(N), length(λ))
        profile_err_lo = profile_err_hi = nothing
        if propagate_err
            profile_err_lo = copy(total_profile)
            profile_err_hi = copy(total_profile)
        end

        # initialize values
        total_flux = 0.0*u"erg/s/cm^2"
        total_flux_err = 0.0*u"erg/s/cm^2"
        total_eqw = 0.0*unit(λ[1])
        total_eqw_err = 0.0*unit(λ[1])
        total_snr = 0.0
        total_snr_err = 0.0

        λ0 = lines.λ₀[k]

        for (j, component) in enumerate(line)    # <- iterates over each velocity component in one line (for multi-component fits)

            # (\/ pretty much the same as the model_line_residuals function, but calculating the integrated intensities)
            amp = popt_l[pᵢ]
            amp_err = propagate_err ? perr_l[pᵢ] : 0.

            voff = popt_l[pᵢ+1]
            voff_err = propagate_err ? perr_l[pᵢ+1] : 0.

            fwhm = popt_l[pᵢ+2]
            fwhm_err = propagate_err ? perr_l[pᵢ+2] : 0.
            pᵢ += 3

            # fill values with nothings for profiles that may / may not have them
            h3 = h3_err = h4 = h4_err = η = η_err = nothing

            if component.profile == :GaussHermite
                # Get additional h3, h4 components
                h3 = popt_l[pᵢ]
                h3_err = propagate_err ? perr_l[pᵢ] : 0.
                h4 = popt_l[pᵢ+1]
                h4_err = propagate_err ? perr_l[pᵢ+1] : 0.
                pᵢ += 2
            elseif component.profile == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                η = popt_l[pᵢ]
                η_err = propagate_err ? perr_l[pᵢ] : 0.
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
                    voff_err = propagate_err ? hypot(voff_err, voff_1_err) : 0.
                    voff += voff_1
                end
                if lines.config.rel_fwhm
                    fwhm_err = propagate_err ? hypot(fwhm_1_err*fwhm, fwhm_err*fwhm_1) : 0.
                    fwhm *= fwhm_1
                end
            end

            # Broaden the FWHM by the instrumental FWHM at the location of the line
            fwhm_inst = lsf(λ0)
            fwhm_err = propagate_err ? fwhm / hypot(fwhm, fwhm_inst) * fwhm_err : 0.
            fwhm = hypot(fwhm, fwhm_inst)

            # Convert voff in km/s to mean wavelength in μm
            mean_wave = λ0 + Doppler_width_λ(voff, λ0)
            mean_wave_err = propagate_err ? λ0 / C_KMS * voff_err : 0.

            # Convert FWHM from km/s to μm
            fwhm_wave = Doppler_width_λ(fwhm, λ0)
            fwhm_wave_err = propagate_err ? λ0 / C_KMS * fwhm_err : 0.

            # Convert amplitude to per-wavelength units, put back in the normalization
            amp_cgs = match_fluxunits(amp*N, 1.0perwave_unit, mean_wave)
            if iszero(amp) 
                amp_cgs_err = propagate_err ? match_fluxunits(amp_err*N, 1.0perwave_unit, mean_wave) : 0.0perwave_unit
            else
                amp_cgs_err = propagate_err ? hypot(amp_err/amp, 2(mean_wave_err/mean_wave))*amp_cgs : 0.0perwave_unit
            end

            # Get the index of the central wavelength
            cent_ind = argmin(abs.(λ .- mean_wave))

            # Integrate over the solid angle
            amp_cgs *= area_sr[cent_ind]
            if propagate_err
                amp_cgs_err *= area_sr[cent_ind]
            end

            # Get the extinction factor at the line center
            # ext = extinction[cent_ind]
            tnorm = nuc_temp_fit ? templates_psfnuc : ones(length(λ))
            # tn = tnorm[cent_ind]        
            
            # Create the extincted line profile in units matching the continuum
            feature_err = nothing
            if component.profile == :Gaussian
                feature = Gaussian(λ, amp*N, mean_wave, fwhm_wave)
                if propagate_err
                    feature_err = hcat(Gaussian(λ, max(amp*N-amp_err*N, 0.), mean_wave, max(fwhm_wave-fwhm_wave_err, eps())),
                                    Gaussian(λ, amp*N+amp_err*N, mean_wave, fwhm_wave+fwhm_wave_err))
                end
            elseif component.profile == :Lorentzian
                feature = Lorentzian(λ, amp*N, mean_wave, fwhm_wave)
                if propagate_err
                    feature_err = hcat(Lorentzian(λ, max(amp*N-amp_err*N, 0.), mean_wave, max(fwhm_wave-fwhm_wave_err, eps())),
                                    Lorentzian(λ, amp*N+amp_err*N, mean_wave, fwhm_wave+fwhm_wave_err))
                end
            elseif component.profile == :GaussHermite
                feature = GaussHermite(λ, amp*N, mean_wave, fwhm_wave, h3, h4)
                if propagate_err
                    feature_err = hcat(GaussHermite(λ, max(amp*N-amp_err*N, 0.), mean_wave, max(fwhm_wave-fwhm_wave_err, eps()), h3-h3_err, h4-h4_err),
                                    GaussHermite(λ, amp*N+amp_err*N, mean_wave, fwhm_wave+fwhm_wave_err, h3+h3_err, h4+h4_err))
                end
            elseif component.profile == :Voigt
                feature = Voigt(λ, amp*N, mean_wave, fwhm_wave, η)
                if propagate_err
                    feature_err = hcat(Voigt(λ, max(amp*N-amp_err*N, 0.), mean_wave, max(fwhm_wave-fwhm_wave_err, eps()), max(η-η_err, 0.)),
                                    Voigt(λ, amp*N+amp_err*N, mean_wave, fwhm_wave+fwhm_wave_err, min(η+η_err, 1.)))
                end
            else
                error("Unrecognized line profile $(component.profile)")
            end
            feature .*= extinction .* tnorm
            if propagate_err
                feature_err[:,1] .*= extinction .* tnorm
                feature_err[:,2] .*= extinction .* tnorm
            end

            # Calculate line flux using the helper function
            p_lines[pₒ], p_lines_err[pₒ] = calculate_flux(component.profile, cube_fitter.cube.λ, amp_cgs, amp_cgs_err, 
                mean_wave, mean_wave_err, fwhm_wave, fwhm_wave_err, h3=h3, h3_err=h3_err, h4=h4, h4_err=h4_err, η=η, η_err=η_err, 
                propagate_err=propagate_err)
            
            # Calculate equivalent width using the helper function
            p_lines[pₒ+1], p_lines_err[pₒ+1] = calculate_eqw(λ, feature, comps, true, feature_err=feature_err, 
                propagate_err=propagate_err)

            # SNR
            # p_lines[pₒ+2] = amp*N*ext*tn / std(I[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)] .- continuum[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)])
            # rms = std(I[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)] .- continuum[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)]) 
            # p_lines[pₒ+2] = sum(feature) / (sqrt(sum(snr_filter)) * rms)
            window = abs.(λ .- mean_wave) .< 3fwhm_wave
            p_lines[pₒ+2] = sum(feature[window]) / sqrt(sum(σ[window].^2))

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
        p_lines[pₒ] = cube_fitter.n_fit_comps[lines.names[k]][spaxel]

        # W80 and Δv parameters
        fwhm_inst = lsf(λ0)
        w80, Δv, vmed, vpeak = calculate_composite_params(λ, total_profile, λ0, fwhm_inst)
        p_lines[pₒ+1] = w80
        p_lines[pₒ+2] = Δv
        p_lines[pₒ+3] = vmed
        p_lines[pₒ+4] = vpeak
        if propagate_err
            w80_lo, Δv_lo, vmed_lo, vpeak_lo = calculate_composite_params(λ, profile_err_lo, λ0, fwhm_inst)
            w80_hi, Δv_hi, vmed_hi, vpeak_hi = calculate_composite_params(λ, profile_err_hi, λ0, fwhm_inst)
            w80_err = mean([w80 - w80_lo, w80_hi - w80])
            Δv_err = mean([Δv - Δv_lo, Δv_hi - Δv])
            vmed_err = mean([vmed - vmed_lo, vmed_hi - vmed])
            vpeak_err = mean([vpeak - vpeak_lo, vpeak_hi - vpeak])
            p_lines_err[pₒ+1] = w80_err
            p_lines_err[pₒ+2] = Δv_err
            p_lines_err[pₒ+3] = vmed_err
            p_lines_err[pₒ+4] = vpeak_err
        end
        @debug "W80=$(w80), DELTA_V=$(Δv), VMED=$(vmed), VPEAK=$(vpeak)"

        pₒ += 5
    end

    p_dust, p_lines, p_dust_err, p_lines_err
end


"""
    calculate_eqw(λ, profile, comps, line, n_dust_cont, n_power_law, n_abs_feat, n_dust_feat,
        fit_sil_emission, amp, amp_err, peak, peak_err, fwhm, fwhm_err; <keyword arguments>)

Calculate the equivalent width (in microns) of a spectral feature, i.e. a PAH or emission line. Calculates the
integral of the ratio of the feature profile to the underlying continuum.
"""
function calculate_eqw(λ::Vector{T}, feature::Vector{T}, comps::Dict, line::Bool; 
    feature_err::Union{Matrix{T},Nothing}=nothing, propagate_err::Bool=true) where {T<:Real}

    contin = line ? comps["continuum_and_pahs"] : comps["continuum"]

    # May blow up for spaxels where the continuum is close to 0
    eqw = NumericalIntegration.integrate(λ, feature ./ contin, Trapezoidal())
    err = 0.
    if propagate_err
        err_lo = eqw - NumericalIntegration.integrate(λ, feature_err[:,1] ./ contin, Trapezoidal())
        err_up = NumericalIntegration.integrate(λ, feature_err[:,2] ./ contin, Trapezoidal()) - eqw
        err = (err_up + err_lo) / 2
    end

    eqw, err
end

