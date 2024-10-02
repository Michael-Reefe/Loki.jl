
############################################## FITTING FUNCTIONS #############################################

# Helper function for getting the extinction profile for a given fit
function get_extinction_profile(λ::Vector{<:Real}, params::Vector{<:Real}, extinction_curve::String,
    fit_uv_bump::Bool, fit_covering_frac::Bool, n_ssps::Integer)

    pₑ = 1 + 3n_ssps + 2

    # Apply attenuation law
    E_BV = params[pₑ]
    E_BV_factor = params[pₑ+1]
    δ = nothing
    Cf_dust = 0.
    dp = 0
    if fit_uv_bump && fit_covering_frac
        δ = params[pₑ+2]
        Cf_dust = params[pₑ+3]
        dp = 2
    elseif fit_uv_bump && extinction_curve == "calzetti"
        δ = params[pₑ+2]
        dp = 1
    elseif fit_covering_frac && extinction_curve == "calzetti"
        Cf_dust = params[pₑ+2]
        dp = 1
    end
    dp += 2
    if extinction_curve == "ccm"
        att_stars = attenuation_cardelli(λ, E_BV * E_BV_factor)
        att_gas = attenuation_cardelli(λ, E_BV)
    elseif extinction_curve == "calzetti"
        if isnothing(δ)
            att_stars = attenuation_calzetti(λ, E_BV * E_BV_factor, Cf=Cf_dust)
            att_gas = attenuation_calzetti(λ, E_BV, Cf=Cf_dust)
        else
            att_stars = attenuation_calzetti(λ, E_BV * E_BV_factor, δ, Cf=Cf_dust)
            att_gas = attenuation_calzetti(λ, E_BV, δ, Cf=Cf_dust)
        end
    else
        error("Unrecognized extinctino curve $extinction_curve")
    end

    att_gas, att_stars, dp
end


# Helper function for getting the normalized templates for a given fit
function get_normalized_templates(λ::Vector{<:Real}, params::Vector{<:Real}, templates::Matrix{<:Real}, 
    N::Real, fit_temp_multexp::Bool, pstart::Integer)

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
            temp_norm[:,i] .+= params[pstart+dp] .* templates[:, i] ./ N
            dp += 1
        end
    end

    temp_norm, dp
end


# Helper function to calculate the normalized nuclear template amplitudes for a given fit
function get_nuctempfit_templates(params::Vector{<:Real}, templates::Matrix{<:Real}, 
    pstart::Integer)

    nuc_temp_norm = nothing
    dp = 0
    for i in axes(templates, 2)
        nuc_temp_norm = params[pstart+dp] .* templates[:, i] 
        dp += 1
    end

    nuc_temp_norm, dp
end


"""
    model_continuum(λ, params, N, vres, vsyst_ssp, vsyst_feii, npad_feii, n_ssps, ssp_λ, ssp_templates,
        feii_templates_fft, n_power_law, fit_uv_bump, fit_covering_frac, fit_opt_na_feii, fit_opt_br_feii,
        extinction_curve, return_components)

Create a model of the continuum (including stellar+dust continuum, PAH features, and extinction, excluding emission lines)
at the given wavelengths `λ`, given the parameter vector `params`.

# Arguments
- `λ::Vector{<:AbstractFloat}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:AbstractFloat}`: Parameter vector. 
- `N::Real`: The normalization.
- `vres::Real`: The constant velocity resolution between pixels, assuming the wavelength vector is logarithmically binned, in km/s/pix.
- `vsyst_ssp::Real`: The systemic velocity offset between the input wavelength grid and the SSP template wavelength grid
- `vsyst_feii::Real`: The systemic velocity offset between the input wavelength grid and the Fe II template wavelength grid
- `npad_feii::Integer`: The length of the Fe II wavelength grid (NOT the length of the Fourier transformed templates)
- `n_ssps::Integer`: The number of simple stellar populations to be fit
- `ssp_λ::Vector{<:Real}`: The SSP template wavelength grid
- `ssp_templates::Union{Vector{Spline2D},Matrix{<:Real}}`: The SSP templates
- `feii_templates_fft::Matrix{<:Complex}`: The Fourier transform of the Fe II templates
- `n_power_law::Integer`: The number of power laws to be fit
- `fit_uv_bump::Bool`: Whether or not to fit the UV bump in the attenuation model (only applies for "calzetti")
- `fit_covering_frac::Bool`: Whether or not to fit a covering fraction in the attenuation model (only applies for "calzetti")
- `fit_opt_na_feii::Bool`: Whether or not to fit narrow Fe II emission
- `fit_opt_br_feii::Bool`: Whether or not to fit broad Fe II emission
- `extinction_curve::String`: The name of the extinction curve to use, either "ccm" or "calzetti"
- `templates::Matrix{<:Real}`: The generic templates to be used in the fit
- `fit_temp_multexp::Bool`: Whether or not to apply and fit multiplicative exponentials to the provided templates
- `nuc_temp_fit::Bool`: If true, the nuclear template is being fit, meaning we are to normalize the model by the templates (which should
    be the PSF templates here).
- `return_components::Bool`: Whether or not to return the individual components of the fit as a dictionary, in 
    addition to the overall fit
"""
function model_continuum(λ::Vector{<:Real}, params::Vector{<:Real}, N::Real, vres::Real, vsyst_ssp::Real, vsyst_feii::Real, 
    npad_feii::Integer, n_ssps::Integer, ssp_λ::Vector{<:Real}, ssp_templates::Union{Vector{Interpolations.Extrapolation},Matrix{<:Real}}, 
    feii_templates_fft::Matrix{<:Complex}, n_power_law::Integer, fit_uv_bump::Bool, fit_covering_frac::Bool, fit_opt_na_feii::Bool, 
    fit_opt_br_feii::Bool, extinction_curve::String, templates::Matrix{<:Real}, fit_temp_multexp::Bool, nuc_temp_fit::Bool, 
    return_components::Bool)   

    # Prepare outputs
    out_type = eltype(params)
    comps = Dict{String, Vector{out_type}}()
    contin = zeros(out_type, length(λ))
    pᵢ = 1

    ssps = zeros(out_type, length(ssp_λ), n_ssps)
    # Interpolate the SSPs to the right ages/metallicities (this is slow)
    for i in 1:n_ssps
        # normalize the templates by their median so that the amplitude is properly separated from the age and metallicity during fitting
        if ssp_templates isa Vector{Interpolations.Extrapolation}
            temp = [ssp_templates[j](params[pᵢ+1], params[pᵢ+2]) for j in eachindex(ssp_λ)]
        else
            temp = @view ssp_templates[:,i]
        end
        ssps[:, i] = params[pᵢ] .* temp ./ median(temp)
        pᵢ += 3
    end

    # Convolve with a line-of-sight velocity distribution (LOSVD) according to the stellar velocity and dispersion
    conv_ssps = convolve_losvd(ssps, vsyst_ssp, params[pᵢ], params[pᵢ+1], vres, length(λ))
    pᵢ += 2

    # Combine the convolved stellar templates together with the weights
    for i in 1:n_ssps
        comps["SSP_$i"] = conv_ssps[:, i]
        contin .+= comps["SSP_$i"]
    end

    # Apply attenuation law
    att_gas, att_stars, dp = get_extinction_profile(λ, params, extinction_curve, fit_uv_bump, fit_covering_frac, n_ssps)
    comps["attenuation_gas"] = att_gas
    comps["attenuation_stars"] = att_stars
    pᵢ += dp
    contin .*= comps["attenuation_stars"]

    # Fe II emission
    if fit_opt_na_feii
        conv_na_feii = convolve_losvd(feii_templates_fft[:, 1], vsyst_feii, params[pᵢ+1], params[pᵢ+2], vres, length(λ), 
            temp_fft=true, npad_in=npad_feii)
        comps["na_feii"] = params[pᵢ] .* conv_na_feii[:, 1]
        contin .+= comps["na_feii"] .* comps["attenuation_gas"]
        pᵢ += 3
    end
    if fit_opt_br_feii
        conv_br_feii = convolve_losvd(feii_templates_fft[:, 2], vsyst_feii, params[pᵢ+1], params[pᵢ+2], vres, length(λ),
            temp_fft=true, npad_in=npad_feii)
        comps["br_feii"] = params[pᵢ] .* conv_br_feii[:, 1]
        contin .+= comps["br_feii"] .* comps["attenuation_gas"]
        pᵢ += 3
    end

    # Power laws
    for j ∈ 1:n_power_law
        # Reference wavelength at 5100 angstroms for the amplitude
        comps["power_law_$j"] = params[pᵢ] .* power_law.(λ, params[pᵢ+1], 0.5100)
        contin .+= comps["power_law_$j"]
        pᵢ += 2
    end

    if size(templates, 2) > 0
        if !nuc_temp_fit
            temp_norm, dp = get_normalized_templates(λ, params, templates, N, fit_temp_multexp, pᵢ)
            for i in axes(temp_norm, 2)
                comps["templates_$i"] = temp_norm[:,i]
            end
            contin .+= sumdim(temp_norm, 2)
            pᵢ += dp
        else
            # Fitting the nuclear spectrum - here the templates are just the PSF model, which we normalize by to make the spectrum
            # continuous.
            nuc_temp_norm, dp = get_nuctempfit_templates(params, templates, pᵢ)
            comps["templates_1"] = nuc_temp_norm
            contin .*= comps["templates_1"]
            pᵢ += dp
        end
    end

    if return_components
        return contin, comps
    end
    contin

end


# Multiple versions for more efficiency
function model_continuum(λ::Vector{<:Real}, params::Vector{<:Real}, N::Real, vres::Real, vsyst_ssp::Real, vsyst_feii::Real, 
    npad_feii::Integer, n_ssps::Integer, ssp_λ::Vector{<:Real}, ssp_templates::Union{Vector{Interpolations.Extrapolation},Matrix{<:Real}}, 
    feii_templates_fft::Matrix{<:Complex}, n_power_law::Integer, fit_uv_bump::Bool, fit_covering_frac::Bool, fit_opt_na_feii::Bool, 
    fit_opt_br_feii::Bool, extinction_curve::String, templates::Matrix{<:Real}, fit_temp_multexp::Bool, nuc_temp_fit::Bool)   

    # Prepare outputs
    out_type = eltype(params)
    contin = zeros(out_type, length(λ))
    pᵢ = 1

    ssps = zeros(out_type, length(ssp_λ), n_ssps)
    # Interpolate the SSPs to the right ages/metallicities (this is slow)
    for i in 1:n_ssps
        # normalize the templates by their median so that the amplitude is properly separated from the age and metallicity during fitting
        if ssp_templates isa Vector{Interpolations.Extrapolation}
            temp = [ssp_templates[j](params[pᵢ+1], params[pᵢ+2]) for j in eachindex(ssp_λ)]
        else
            temp = @view ssp_templates[:,i]
        end
        ssps[:, i] = params[pᵢ] .* temp ./ median(temp)
        pᵢ += 3
    end

    # Convolve with a line-of-sight velocity distribution (LOSVD) according to the stellar velocity and dispersion
    conv_ssps = convolve_losvd(ssps, vsyst_ssp, params[pᵢ], params[pᵢ+1], vres, length(λ))
    pᵢ += 2

    # Combine the convolved stellar templates together with the weights
    @views for i in 1:n_ssps
        contin .+= conv_ssps[:, i]
    end

    # Apply attenuation law
    att_gas, att_stars, dp = get_extinction_profile(λ, params, extinction_curve, fit_uv_bump, fit_covering_frac, n_ssps)
    pᵢ += dp
    contin .*= att_stars

    # Fe II emission
    if fit_opt_na_feii
        conv_na_feii = convolve_losvd(feii_templates_fft[:, 1], vsyst_feii, params[pᵢ+1], params[pᵢ+2], vres, length(λ), 
            temp_fft=true, npad_in=npad_feii)
        @views contin .+= params[pᵢ] .* conv_na_feii[:, 1] .* att_gas
        pᵢ += 3
    end
    if fit_opt_br_feii
        conv_br_feii = convolve_losvd(feii_templates_fft[:, 2], vsyst_feii, params[pᵢ+1], params[pᵢ+2], vres, length(λ),
            temp_fft=true, npad_in=npad_feii)
        @views contin .+= params[pᵢ] .* conv_br_feii[:, 1] .* att_gas
        pᵢ += 3
    end

    # Power laws
    for _ ∈ 1:n_power_law
        # Reference wavelength at 5100 angstroms for the amplitude
        contin .+= params[pᵢ] .* power_law.(λ, params[pᵢ+1], 0.5100)
        pᵢ += 2
    end

    if size(templates, 2) > 0
        if !nuc_temp_fit
            temp_norm, dp = get_normalized_templates(λ, params, templates, N, fit_temp_multexp, pᵢ)
            contin .+= sumdim(temp_norm, 2)
            pᵢ += dp
        else
            # Fitting the nuclear spectrum - here the templates are just the PSF model, which we normalize by to make the spectrum
            # continuous.
            nuc_temp_norm, dp = get_nuctempfit_templates(params, templates, pᵢ)
            contin .*= nuc_temp_norm
            pᵢ += dp
        end
    end

    contin
end


"""
    calculate_extra_parameters(λ, I, N, comps, comps, n_ssps, n_power_law, fit_opt_na_feii,
        fit_opt_br_feii, n_lines, n_acomps, n_comps, lines, flexible_wavesol, lsf, popt_l, perr_l,
        extinction, mask_lines, continuum, area_sr[, propagate_err])

Calculate extra parameters that are not fit, but are nevertheless important to know, for a given spaxel.
Currently this includes the integrated intensity, equivalent width, and signal to noise ratios of dust features and emission lines.
"""
function calculate_extra_parameters(cube_fitter::CubeFitter, λ::Vector{<:Real}, I::Vector{<:Real}, N::Real, comps::Dict, 
    nuc_temp_fit::Bool, lsf::Function, popt_l::Vector{T}, perr_l::Vector{T}, extinction::Vector{T}, mask_lines::BitVector, 
    continuum::Vector{T}, area_sr::Vector{T}, spaxel::CartesianIndex, propagate_err::Bool=true) where {T<:Real}

    @debug "Calculating extra parameters"

    # Normalization
    @debug "Normalization: $N"

    # Max extinction factor
    # max_ext = 1 / minimum(extinction)

    # Loop through lines
    p_lines = zeros(3cube_fitter.n_lines+3cube_fitter.n_acomps+5cube_fitter.n_lines)
    p_lines_err = zeros(3cube_fitter.n_lines+3cube_fitter.n_acomps+5cube_fitter.n_lines)
    rel_amp, rel_voff, rel_fwhm = cube_fitter.relative_flags
    pₒ = pᵢ = 1
    for (k, λ0) ∈ enumerate(cube_fitter.lines.λ₀)
        amp_1 = amp_1_err = voff_1 = voff_1_err = fwhm_1 = fwhm_1_err = nothing
        total_profile = zeros(eltype(λ), length(λ))
        profile_err_lo = profile_err_hi = nothing
        if propagate_err
            profile_err_lo = copy(total_profile)
            profile_err_hi = copy(total_profile)
        end

        for j ∈ 1:cube_fitter.n_comps
            if !isnothing(cube_fitter.lines.profiles[k, j])

                # (\/ pretty much the same as the model_line_residuals function, but calculating the integrated intensities)
                amp = popt_l[pᵢ]
                amp_err = propagate_err ? perr_l[pᵢ] : 0.
                voff = popt_l[pᵢ+1]
                voff_err = propagate_err ? perr_l[pᵢ+1] : 0.
                # fill values with nothings for profiles that may / may not have them
                h3 = h3_err = h4 = h4_err = η = η_err = nothing

                if !isnothing(cube_fitter.lines.tied_voff[k, j]) && cube_fitter.flexible_wavesol && isone(j)
                    voff += popt_l[pᵢ+2]
                    voff_err = propagate_err ? hypot(voff_err, perr_l[pᵢ+2]) : 0.
                    fwhm = popt_l[pᵢ+3]
                    fwhm_err = propagate_err ? perr_l[pᵢ+3] : 0.
                    pᵢ += 4
                else
                    fwhm = popt_l[pᵢ+2]
                    fwhm_err = propagate_err ? perr_l[pᵢ+2] : 0.
                    pᵢ += 3
                end

                if cube_fitter.lines.profiles[k, j] == :GaussHermite
                    # Get additional h3, h4 components
                    h3 = popt_l[pᵢ]
                    h3_err = propagate_err ? perr_l[pᵢ] : 0.
                    h4 = popt_l[pᵢ+1]
                    h4_err = propagate_err ? perr_l[pᵢ+1] : 0.
                    pᵢ += 2
                elseif cube_fitter.lines.profiles[k, j] == :Voigt
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
                    if rel_amp
                        amp_err = propagate_err ? hypot(amp_1_err*amp, amp_err*amp_1) : 0.
                        amp *= amp_1
                    end
                    if rel_voff 
                        voff_err = propagate_err ? hypot(voff_err, voff_1_err) : 0.
                        voff += voff_1
                    end
                    if rel_fwhm
                        fwhm_err = propagate_err ? hypot(fwhm_1_err*fwhm, fwhm_err*fwhm_1) : 0.
                        fwhm *= fwhm_1
                    end
                end

                # Broaden the FWHM by the instrumental FWHM at the location of the line
                fwhm_inst = lsf(λ0)
                fwhm_err = propagate_err ? fwhm / hypot(fwhm, fwhm_inst) * fwhm_err : 0.
                fwhm = hypot(fwhm, fwhm_inst)

                # Convert voff in km/s to mean wavelength in μm
                mean_μm = Doppler_shift_λ(λ0, voff)
                mean_μm_err = propagate_err ? λ0 / C_KMS * voff_err : 0.
                # WARNING:
                # Probably should set to 0 if using flexible tied voffs since they are highly degenerate and result in massive errors
                # if !isnothing(cube_fitter.line_tied[k]) && cube_fitter.flexible_wavesol
                #     mean_μm_err = 0.
                # end

                # Convert FWHM from km/s to μm
                fwhm_μm = Doppler_shift_λ(λ0, fwhm/2) - Doppler_shift_λ(λ0, -fwhm/2)
                fwhm_μm_err = propagate_err ? λ0 / C_KMS * fwhm_err : 0.

                # Undo the normalization from the extinction
                # amp *= max_ext
                # amp_err *= max_ext

                # Convert from erg s^-1 cm^-2 Ang^-1 sr^-1 to erg s^-1 cm^-2 μm^-1 sr^-1, putting back in the normalization
                amp_cgs = amp * N * 1e4
                amp_cgs_err = propagate_err ? amp_err * N * 1e4 : 0.

                # Get the index of the central wavelength
                cent_ind = argmin(abs.(λ .- mean_μm))

                # Integrate over the solid angle
                amp_cgs *= area_sr[cent_ind]
                if propagate_err
                    amp_cgs_err *= area_sr[cent_ind]
                end

                # Get the extinction factor at the line center
                ext = extinction[cent_ind]

                # Create the extincted line profile in units matching the continuum
                feature_err = nothing
                if cube_fitter.lines.profiles[k, j] == :Gaussian
                    feature = Gaussian.(λ, 1., mean_μm, fwhm_μm)
                    if propagate_err
                        feature_err = hcat(Gaussian.(λ, max(amp*N-amp_err*N, 0.), mean_μm, max(fwhm_μm-fwhm_μm_err, eps())),
                                       Gaussian.(λ, amp*N+amp_err*N, mean_μm, fwhm_μm+fwhm_μm_err))
                    end
                elseif cube_fitter.lines.profiles[k, j] == :Lorentzian
                    feature = Lorentzian.(λ, 1., mean_μm, fwhm_μm)
                    if propagate_err
                        feature_err = hcat(Lorentzian.(λ, max(amp*N-amp_err*N, 0.), mean_μm, max(fwhm_μm-fwhm_μm_err, eps())),
                                       Lorentzian.(λ, amp*N+amp_err*N, mean_μm, fwhm_μm+fwhm_μm_err))
                    end
                elseif cube_fitter.lines.profiles[k, j] == :GaussHermite
                    feature = GaussHermite.(λ, 1., mean_μm, fwhm_μm, h3, h4)
                    if propagate_err
                        feature_err = hcat(GaussHermite.(λ, max(amp*N-amp_err*N, 0.), mean_μm, max(fwhm_μm-fwhm_μm_err, eps()), h3-h3_err, h4-h4_err),
                                       GaussHermite.(λ, amp*N+amp_err*N, mean_μm, fwhm_μm+fwhm_μm_err, h3+h3_err, h4+h4_err))
                    end
                elseif cube_fitter.lines.profiles[k, j] == :Voigt
                    feature = Voigt.(λ, 1., mean_μm, fwhm_μm, η)
                    if propagate_err
                        feature_err = hcat(Voigt.(λ, max(amp*N-amp_err*N, 0.), mean_μm, max(fwhm_μm-fwhm_μm_err, eps()), max(η-η_err, 0.)),
                                       Voigt.(λ, amp*N+amp_err*N, mean_μm, fwhm_μm+fwhm_μm_err, min(η+η_err, 1.)))
                    end
                else
                    error("Unrecognized line profile $(cube_fitter.lines.profiles[k, j])")
                end
                snr_filter = feature .> 1e-10
                feature .*= amp .* N .* extinction
                if propagate_err
                    feature_err[:,1] .*= extinction
                    feature_err[:,2] .*= extinction
                end

                # Calculate line flux using the helper function
                p_lines[pₒ], p_lines_err[pₒ] = calculate_flux(cube_fitter.lines.profiles[k, j], cube_fitter.cube.λ, amp_cgs, amp_cgs_err, 
                    mean_μm, mean_μm_err, fwhm_μm, fwhm_μm_err, h3=h3, h3_err=h3_err, h4=h4, h4_err=h4_err, η=η, η_err=η_err, 
                    propagate_err=propagate_err)
                
                # Calculate equivalent width using the helper function
                p_lines[pₒ+1], p_lines_err[pₒ+1] = calculate_eqw(cube_fitter, λ, feature, comps, nuc_temp_fit, 
                    feature_err=feature_err, propagate_err=propagate_err)

                # SNR
                p_lines[pₒ+2] = amp*N*ext / std(I[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)] .- continuum[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)])
                # rms = std(I[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)] .- continuum[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)]) 
                # p_lines[pₒ+2] = sum(feature) / (sqrt(sum(snr_filter)) * rms)

                total_profile .+= feature
                if propagate_err
                    profile_err_lo .+= feature_err[:,1]
                    profile_err_hi .+= feature_err[:,2]
                end
                
                @debug "Line with ($amp_cgs, $mean_μm, $fwhm_μm) and errors ($amp_cgs_err, $mean_μm_err, $fwhm_μm_err)"
                @debug "Flux=$(p_lines[pₒ]) +/- $(p_lines_err[pₒ]), EQW=$(p_lines[pₒ+1]) +/- $(p_lines_err[pₒ+1]), SNR=$(p_lines[pₒ+2])"

                # Advance the output vector index by 3
                pₒ += 3
            end
        end

        # Number of velocity components
        p_lines[pₒ] = cube_fitter.n_fit_comps[cube_fitter.lines.names[k]][spaxel]

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

    p_lines, p_lines_err
end


"""
    calculate_eqw(cube_fitter, λ, profile, comps, amp, amp_err, peak, peak_err, 
        fwhm, fwhm_err; <keyword arguments>)

Calculate the equivalent width (in microns) of a spectral feature, i.e. a PAH or emission line. Calculates the
integral of the ratio of the feature profile to the underlying continuum.
"""
function calculate_eqw(cube_fitter::CubeFitter, λ::Vector{T}, feature::Vector{T}, comps::Dict,
    nuc_temp_fit::Bool; feature_err::Union{Matrix{T},Nothing}=nothing, propagate_err::Bool=true) where {T<:Real}

    contin = zeros(length(λ))
    for i ∈ 1:cube_fitter.n_ssps
        contin .+= comps["SSP_$i"]
    end
    contin .*= comps["attenuation_stars"]
    if cube_fitter.fit_opt_na_feii
        contin .+= comps["na_feii"] .* comps["attenuation_gas"]
    end
    if cube_fitter.fit_opt_br_feii
        contin .+= comps["br_feii"] .* comps["attenuation_gas"]
    end
    for j ∈ 1:cube_fitter.n_power_law
        contin .+= comps["power_law_$j"]
    end
    for q ∈ 1:cube_fitter.n_templates
        if !nuc_temp_fit
            contin .+= comps["templates_$q"]
        else
            contin .*= comps["templates_$q"]
        end
    end

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
