
############################################## FITTING FUNCTIONS #############################################

# Helper function for getting the extinction profile for a given fit
function get_extinction_profile(λ::Vector{<:Real}, params::Vector{<:Real}, 
    extinction_curve::String, extinction_screen::Bool, κ_abs::Vector{Spline1D},
    custom_ext::Union{Spline1D,Nothing}, n_dust_cont::Integer, n_power_law::Integer)

    pₑ = 3 + 2n_dust_cont + 2n_power_law
    τ_oli = τ_pyr = τ_for = nothing
    if extinction_curve == "d+"
        ext_curve = τ_dp(λ, params[pₑ+3])
        ext = extinction.(ext_curve, params[pₑ], screen=extinction_screen)
        dp = 1
    elseif extinction_curve == "kvt"
        ext_curve = τ_kvt(λ, params[pₑ+3])
        ext = extinction.(ext_curve, params[pₑ], screen=extinction_screen)
        dp = 1
    elseif extinction_curve == "ct"
        ext_curve = τ_ct(λ)
        ext = extinction.(ext_curve, params[pₑ], screen=extinction_screen)
        dp = 1
    elseif extinction_curve == "ohm"
        ext_curve = τ_ohm(λ)
        ext = extinction.(ext_curve, params[pₑ], screen=extinction_screen)
        dp = 1
    elseif extinction_curve == "custom"
        ext_curve = custom_ext(λ)
        ext = extinction.(ext_curve, params[pₑ], screen=extinction_screen)
        dp = 1
    elseif extinction_curve == "decompose"
        τ_oli = params[pₑ] .* κ_abs[1](λ)
        τ_pyr = params[pₑ] .* params[pₑ+1] .* κ_abs[2](λ)
        τ_for = params[pₑ] .* params[pₑ+2] .* κ_abs[3](λ)
        τ_97 = params[pₑ] * κ_abs[1](9.7) + params[pₑ] * params[pₑ+1] * κ_abs[2](9.7) + params[pₑ] * params[pₑ+2] * κ_abs[3](9.7)
        ext = extinction.((1 .- params[pₑ+5]) .* (τ_oli .+ τ_pyr .+ τ_for) .+ params[pₑ+5] .* τ_97 .* (9.7./λ).^1.7, 1., screen=extinction_screen)
        dp = 3
    else
        error("Unrecognized extinction curve: $extinction_curve")
    end

    ext, dp, τ_oli, τ_pyr, τ_for
end


# Helper function for getting the normalized templates for a given fit
function get_normalized_templates(λ::Vector{<:Real}, params::Vector{<:Real}, templates::Matrix{<:Real}, 
    N::Real, channel_masks::Vector{BitVector}, fit_temp_multexp::Bool, pstart::Integer)

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
function get_nuctempfit_templates(params::Vector{<:Real}, templates::Matrix{<:Real}, 
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
    model_continuum(λ, params, N, n_dust_cont, n_power_law, dust_prof, n_abs_feat, extinction_curve, extinction_screen, 
        κ_abs, custom_ext, fit_sil_emission, use_pah_templates, templates, channel_masks, return_components)

Create a model of the continuum (including stellar+dust continuum, PAH features, and extinction, excluding emission lines)
at the given wavelengths `λ`, given the parameter vector `params`.

Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments
- `λ::Vector{<:AbstractFloat}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:AbstractFloat}`: Parameter vector. 
- `N::Real`: The normalization.
- `n_dust_cont::Integer`: Number of dust continuum profiles to be fit
- `n_power_law::Integer`: Number of power laws to be fit
- `dust_prof::Vector{Symbol}`: Vector giving the profiles to fit for each dust feature (either :Drude or :PearsonIV)
- `n_abs_feat::Integer`: Number of absorption features to be fit
- `extinction_curve::String`: The type of extinction curve to use, "kvt" or "d+"
- `extinction_screen::Bool`: Whether or not to use a screen model for the extinction curve
- `κ_abs::Vector{Spline1D}`: A series of interpolating functions over wavelength giving the mass absorption coefficients for
    amorphous olivine, amorphous pyroxene, and crystalline forsterite
- `custom_ext::Union{Spline1D,Nothing}`: An optional custom extinction template.
- `fit_sil_emission::Bool`: Whether or not to fit silicate emission with a hot dust continuum component
- `fit_temp_multexp::Bool`: Whether or not to apply and fit multiplicative exponentials to the provided templates
- `use_pah_templates::Bool`: Whether or not to use PAH templates to model the PAH emission
- `templates::Matrix{<:Real}`: The PSF templates to be used in the fit
- `channel_masks::Vector{BitVector}`: Masks that pick out each subchannel in the data
- `nuc_temp_fit::Bool`: If true, the nuclear template is being fit, meaning we are to normalize the model by the templates (which should
    be the PSF templates here).
- `return_components::Bool`: Whether or not to return the individual components of the fit as a dictionary, in 
    addition to the overall fit
"""
function model_continuum(λ::Vector{<:Real}, params::Vector{<:Real}, N::Real, n_dust_cont::Integer, n_power_law::Integer, 
    dust_prof::Vector{Symbol}, n_abs_feat::Integer, extinction_curve::String, extinction_screen::Bool, κ_abs::Vector{Spline1D}, custom_ext::Union{Spline1D,Nothing},
    fit_sil_emission::Bool, fit_temp_multexp::Bool, use_pah_templates::Bool, templates::Matrix{<:Real}, channel_masks::Vector{BitVector}, 
    nuc_temp_fit::Bool, return_components::Bool)

    # Prepare outputs
    out_type = eltype(params)
    comps = Dict{String, Vector{out_type}}()
    contin = zeros(out_type, length(λ))

    # Stellar blackbody continuum (usually at 5000 K)
    comps["stellar"] = params[1] .* Blackbody_ν.(λ, params[2]) ./ N
    contin .+= comps["stellar"]
    pᵢ = 3

    # Add dust continua at various temperatures
    for i ∈ 1:n_dust_cont
        comps["dust_cont_$i"] = params[pᵢ] .* (9.7 ./ λ).^2 .* Blackbody_ν.(λ, params[pᵢ+1]) ./ N
        contin .+= comps["dust_cont_$i"] 
        pᵢ += 2
    end

    # Add power laws with various indices
    for j ∈ 1:n_power_law
        comps["power_law_$j"] = params[pᵢ] .* power_law.(λ, params[pᵢ+1])
        contin .+= comps["power_law_$j"]
        pᵢ += 2
    end

    # Extinction 
    ext, dp, τ_oli, τ_pyr, τ_for = get_extinction_profile(λ, params, extinction_curve, extinction_screen, 
        κ_abs, custom_ext, n_dust_cont, n_power_law)
    comps["extinction"] = ext
    if extinction_curve == "decompose"
        comps["abs_oli"] = extinction.(τ_oli, 1., screen=extinction_screen)
        comps["abs_pyr"] = extinction.(τ_pyr, 1., screen=extinction_screen)
        comps["abs_for"] = extinction.(τ_for, 1., screen=extinction_screen)
    end
    pᵢ += dp

    # Ice+CH Absorption
    ext_ice = τ_ice(λ)
    comps["abs_ice"] = extinction.(ext_ice, params[pᵢ] * params[pᵢ+1], screen=true)
    ext_ch = τ_ch(λ)
    comps["abs_ch"] = extinction.(ext_ch, params[pᵢ+1], screen=true)
    Cf = params[pᵢ+3]
    pᵢ += 4

    # Other absorption features
    abs_tot = one(out_type)
    for k ∈ 1:n_abs_feat
        prof = Gaussian.(λ, 1.0, params[pᵢ+1:pᵢ+2]...)
        comps["abs_feat_$k"] = extinction.(prof, params[pᵢ], screen=true)
        abs_tot = abs_tot .* comps["abs_feat_$k"]
        pᵢ += 3
    end

    # only obscure the part of the continuum that is defined by the covering fraction Cf
    comps["obscured_continuum"] = @. Cf * contin * comps["extinction"] * comps["abs_ice"] * comps["abs_ch"] * abs_tot
    comps["unobscured_continuum"] = @. (1 - Cf) * contin
    contin = comps["obscured_continuum"] .+ comps["unobscured_continuum"]

    if fit_sil_emission
        # Add Silicate emission from hot dust (amplitude, temperature, covering fraction, warm tau, cold tau)
        # Ref: Gallimore et al. 2010
        comps["hot_dust"] = silicate_emission(λ, params[pᵢ:pᵢ+5]...) ./ N
        contin .+= comps["hot_dust"] .* comps["abs_ice"] .* comps["abs_ch"] .* abs_tot
        pᵢ += 6
    end

    nuc_temp_norm = nothing
    if size(templates, 2) > 0
        if !nuc_temp_fit
            temp_norm, dp = get_normalized_templates(λ, params, templates, N, channel_masks, fit_temp_multexp, pᵢ)
            for i in axes(temp_norm, 2)
                comps["templates_$i"] = temp_norm[:,i]
            end
            contin .+= sumdim(temp_norm, 2)
            pᵢ += dp
        else
            # Fitting the nuclear spectrum - here the templates are just the PSF model, which we normalize by to make the spectrum
            # continuous.
            nuc_temp_norm, dp = get_nuctempfit_templates(params, templates, channel_masks, pᵢ)
            comps["templates_1"] = nuc_temp_norm
            pᵢ += dp
        end
    end

    if use_pah_templates
        pah3 = Smith3_interp(λ)
        contin .+= params[pᵢ] .* pah3  ./ maximum(pah3) .* comps["extinction"]
        pah4 = Smith4_interp(λ)
        contin .+= params[pᵢ+1] .* pah4 ./ maximum(pah4) .* comps["extinction"]
    else
        for (j, prof) ∈ enumerate(dust_prof)
            if prof == :Drude
                comps["dust_feat_$j"] = Drude.(λ, params[pᵢ:pᵢ+2]...)
                pᵢ += 3
            elseif prof == :PearsonIV
                comps["dust_feat_$j"] = PearsonIV.(λ, params[pᵢ:pᵢ+4]...)
                pᵢ += 5
            end
            contin .+= comps["dust_feat_$j"] .* comps["extinction"] 
        end
    end

    if nuc_temp_fit
        contin .*= comps["templates_1"]
    end

    # Return components if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency --> not allocating the dictionary improves performance DRAMATICALLY

function model_continuum(λ::Vector{<:Real}, params::Vector{<:Real}, N::Real, n_dust_cont::Integer, n_power_law::Integer, dust_prof::Vector{Symbol},
    n_abs_feat::Integer, extinction_curve::String, extinction_screen::Bool, κ_abs::Vector{Spline1D}, custom_ext::Union{Spline1D,Nothing}, 
    fit_sil_emission::Bool, fit_temp_multexp::Bool, use_pah_templates::Bool, templates::Matrix{<:Real}, channel_masks::Vector{BitVector},
    nuc_temp_fit::Bool)

    # Prepare outputs
    out_type = eltype(params)
    contin = zeros(out_type, length(λ))

    # Stellar blackbody continuum (usually at 5000 K)
    contin .+= params[1] .* Blackbody_ν.(λ, params[2]) ./ N
    pᵢ = 3

    # Add dust continua at various temperatures
    for i ∈ 1:n_dust_cont
        contin .+= params[pᵢ] .* (9.7 ./ λ).^2 .* Blackbody_ν.(λ, params[pᵢ+1]) ./ N
        pᵢ += 2
    end

    # Add power laws with various indices
    for j ∈ 1:n_power_law
        contin .+= params[pᵢ] .* power_law.(λ, params[pᵢ+1])
        pᵢ += 2
    end

    # Extinction 
    ext, dp, τ_oli, τ_pyr, τ_for = get_extinction_profile(λ, params, extinction_curve, extinction_screen, 
        κ_abs, custom_ext, n_dust_cont, n_power_law)
    pᵢ += dp

    # Ice+CH absorption
    ext_ice = τ_ice(λ)
    abs_ice = extinction.(ext_ice, params[pᵢ] * params[pᵢ+1], screen=true)
    ext_ch = τ_ch(λ)
    abs_ch = extinction.(ext_ch, params[pᵢ+1], screen=true)
    Cf = params[pᵢ+3]
    pᵢ += 4

    # Other absorption features
    abs_tot = one(out_type)
    for k ∈ 1:n_abs_feat
        prof = Gaussian.(λ, 1.0, params[pᵢ+1:pᵢ+2]...)
        abs_tot = abs_tot .* extinction.(prof, params[pᵢ], screen=true)
        pᵢ += 3
    end
    
    contin = @. Cf * contin * ext * abs_ice * abs_ch * abs_tot + (1 - Cf) * contin

    if fit_sil_emission
        # Add Silicate emission from hot dust (amplitude, temperature, covering fraction, warm tau, cold tau)
        # Ref: Gallimore et al. 2010
        contin .+= silicate_emission(λ, params[pᵢ:pᵢ+5]...) ./ N .* abs_ice .* abs_ch .* abs_tot
        pᵢ += 6
    end

    nuc_temp_norm = nothing
    if size(templates, 2) > 0
        if !nuc_temp_fit
            temp_norm, dp = get_normalized_templates(λ, params, templates, N, channel_masks, fit_temp_multexp, pᵢ)
            contin .+= sumdim(temp_norm, 2)
            pᵢ += dp
        else
            # Fitting the nuclear spectrum - here the templates are just the PSF model, which we normalize by to make the spectrum
            # continuous.
            nuc_temp_norm, dp = get_nuctempfit_templates(params, templates, channel_masks, pᵢ)
            pᵢ += dp
        end
    end

    if use_pah_templates
        pah3 = Smith3_interp(λ)
        contin .+= params[pᵢ] .* pah3 ./ maximum(pah3) .* ext
        pah4 = Smith4_interp(λ)
        contin .+= params[pᵢ+1] .* pah4 ./ maximum(pah4) .* ext
    else
        if all(dust_prof .== :Drude)
            for j ∈ 1:length(dust_prof) 
                contin .+= Drude.(λ, params[pᵢ:pᵢ+2]...) .* ext
                pᵢ += 3
            end
        else
            for (j, prof) ∈ enumerate(dust_prof)
                if prof == :Drude
                    df = Drude.(λ, params[pᵢ:pᵢ+2]...)
                    pᵢ += 3
                elseif prof == :PearsonIV
                    df = PearsonIV.(λ, params[pᵢ:pᵢ+4]...)
                    pᵢ += 5
                end
                contin .+= df .* ext
            end
        end
    end

    if nuc_temp_fit
        contin .*= nuc_temp_norm
    end

    contin

end


"""
    model_pah_residuals(λ, params, dust_prof, ext_curve, return_components)

Create a model of the PAH features at the given wavelengths `λ`, given the parameter vector `params`.
Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments
- `λ::Vector{<:Real}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:Real}`: Parameter vector. Parameters should be ordered as: `(amp, center, fwhm) for each PAH profile`
- `dust_prof::Vector{Symbol}`: The profiles of each PAH feature being fit (either :Drude or :PearsonIV)
- `ext_curve::Vector{<:Real}`: The extinction curve that was fit using model_continuum
- `template_norm::Union{Nothing,Vector{<:Real}}`: The normalization PSF template that was fit using model_continuum
- `nuc_temp_fit::Bool`: Whether or not to apply the PSF normalization template
- `return_components::Bool`: Whether or not to return the individual components of the fit as a dictionary, in
    addition to the overall fit
"""
function model_pah_residuals(λ::Vector{<:Real}, params::Vector{<:Real}, dust_prof::Vector{Symbol}, ext_curve::Vector{<:Real}, 
    template_norm::Union{Nothing,Vector{<:Real}}, nuc_temp_fit::Bool, return_components::Bool) 

    # Prepare outputs
    out_type = eltype(params)
    comps = Dict{String, Vector{out_type}}()
    contin = zeros(out_type, length(λ))

    # Add dust features with drude profiles
    pᵢ = 1
    for (j, prof) ∈ enumerate(dust_prof)
        if prof == :Drude
            df = Drude.(λ, params[pᵢ:pᵢ+2]...)
            pᵢ += 3
        elseif prof == :PearsonIV
            df = PearsonIV.(λ, params[pᵢ:pᵢ+4]...)
            pᵢ += 5
        end
        contin .+= df
    end

    # Apply extinction
    contin .*= ext_curve
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
function model_pah_residuals(λ::Vector{<:Real}, params::Vector{<:Real}, dust_prof::Vector{Symbol}, ext_curve::Vector{<:Real},
    template_norm::Union{Nothing,Vector{<:Real}}, nuc_temp_fit::Bool)

    # Prepare outputs
    out_type = eltype(params)
    contin = zeros(out_type, length(λ))

    # Add dust features with drude profiles
    pᵢ = 1
    if all(dust_prof .== :Drude)
        for j ∈ 1:length(dust_prof) 
            contin .+= Drude.(λ, params[pᵢ:pᵢ+2]...)
            pᵢ += 3
        end
    else
        for (j, prof) ∈ enumerate(dust_prof)
            if prof == :Drude
                df = Drude.(λ, params[pᵢ:pᵢ+2]...)
                pᵢ += 3
            elseif prof == :PearsonIV
                df = PearsonIV.(λ, params[pᵢ:pᵢ+4]...)
                pᵢ += 5
            end
            contin .+= df
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
    calculate_extra_parameters(λ, I, N, comps, n_channels, n_dust_cont, n_power_law, n_dust_feat, dust_profiles,
        n_abs_feat, fit_sil_emission, n_lines, n_acomps, n_comps, lines, flexible_wavesol, lsf, popt_c,
        popt_l, perr_c, perr_l, extinction, mask_lines, continuum, area_sr[, propagate_err])

Calculate extra parameters that are not fit, but are nevertheless important to know, for a given spaxel.
Currently this includes the integrated intensity, equivalent width, and signal to noise ratios of dust features and emission lines.
"""
function calculate_extra_parameters(cube_fitter::CubeFitter, λ::Vector{<:Real}, I::Vector{<:Real}, N::Real, comps::Dict, 
    nuc_temp_fit::Bool, lsf::Function, popt_c::Vector{T}, popt_l::Vector{T}, perr_c::Vector{T}, perr_l::Vector{T}, 
    extinction_pah::Vector{T}, extinction::Vector{T}, templates_psfnuc::Union{Nothing,Vector{<:Real}}, 
    mask_lines::BitVector, continuum::Vector{T}, area_sr::Vector{T}, spaxel::CartesianIndex, propagate_err::Bool=true) where {T<:Real}

    @debug "Calculating extra parameters"

    # Normalization
    @debug "Normalization: $N"

    # Loop through dust features
    p_dust = zeros(3cube_fitter.n_dust_feat)
    p_dust_err = zeros(3cube_fitter.n_dust_feat)
    pₒ = 1
    # Initial parameter vector index where dust profiles start
    i_split = cubefitter_mir_count_cont_parameters(cube_fitter.extinction_curve, cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp,
        cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_abs_feat, cube_fitter.n_templates, cube_fitter.n_channels,
        cube_fitter.dust_features; split=true)
    pᵢ = i_split + 1
    # Extinction normalization factor
    # max_ext = 1 / minimum(extinction)

    for ii ∈ 1:cube_fitter.n_dust_feat

        # unpack the parameters
        A, μ, fwhm = popt_c[pᵢ:pᵢ+2]
        A_err, μ_err, fwhm_err = perr_c[pᵢ:pᵢ+2]
        # Undo the normalization due to the extinction
        # A *= max_ext
        # A_err *= max_ext
        # Convert peak intensity to CGS units (erg s^-1 cm^-2 μm^-1 sr^-1)
        A_cgs = MJysr_to_cgs(A*N, μ)
        # Convert the error in the intensity to CGS units
        A_cgs_err = propagate_err ? MJysr_to_cgs_err(A*N, A_err*N, μ, μ_err) : 0.

        # Get the index of the central wavelength
        cent_ind = argmin(abs.(λ .- μ))
        
        # Integrate over the solid angle
        A_cgs *= area_sr[cent_ind]
        if propagate_err
            A_cgs_err *= area_sr[cent_ind]
        end

        # Get the extinction profile at the center
        ext = extinction_pah[cent_ind] 
        tnorm = nuc_temp_fit ? templates_psfnuc : ones(length(λ))
        tn = tnorm[cent_ind]

        prof = cube_fitter.dust_features.profiles[ii]
        if prof == :PearsonIV
            m, m_err = popt_c[pᵢ+3], perr_c[pᵢ+3]
            ν, ν_err = popt_c[pᵢ+4], perr_c[pᵢ+4]
        else
            m, m_err = 0., 0.
            ν, ν_err = 0., 0.
        end

        # Create the profile
        feature_err = nothing
        if prof == :Drude
            feature = Drude.(λ, 1., μ, fwhm)
            if propagate_err
                feature_err = hcat(Drude.(λ, max(A*N-A_err*N, 0.), μ, max(fwhm-fwhm_err, eps())),
                                Drude.(λ, A*N+A_err*N, μ, fwhm+fwhm_err))
            end
        elseif prof == :PearsonIV
            feature = PearsonIV.(λ, 1., μ, fwhm, m, ν)
            if propagate_err
                feature_err = hcat(PearsonIV.(λ, max(A*N-A_err*N, 0.), μ, max(fwhm-fwhm_err, eps()), m-m_err, ν-ν_err),
                                PearsonIV.(λ, A*N+A_err*N, μ, fwhm+fwhm_err, m+m_err, ν+ν_err))
            end
        else
            error("Unrecognized PAH profile: $prof")
        end
        snr_filter = feature .> 1e-10
        feature .*= A .* N .* extinction_pah .* tnorm
        if propagate_err
            feature_err[:,1] .*= extinction_pah .* tnorm
            feature_err[:,2] .*= extinction_pah .* tnorm
        end

        # Calculate the flux using the utility function
        flux, f_err = calculate_flux(prof, A_cgs, A_cgs_err, μ, μ_err, fwhm, fwhm_err, 
            m=m, m_err=m_err, ν=ν, ν_err=ν_err, propagate_err=propagate_err)
        
        # Calculate the equivalent width using the utility function
        eqw, e_err = calculate_eqw(cube_fitter, λ, feature, comps, false, nuc_temp_fit, 
            feature_err=feature_err, propagate_err=propagate_err)
        
        # snr = A*N*ext*tn / std(I[.!mask_lines .& (abs.(λ .- μ) .< 2fwhm)] .- continuum[.!mask_lines .& (abs.(λ .- μ) .< 2fwhm)])
        rms = std(I[.!mask_lines .& (abs.(λ .- μ) .< 2fwhm)] .- continuum[.!mask_lines .& (abs.(λ .- μ) .< 2fwhm)])
        snr = sum(feature) / (sqrt(sum(snr_filter)) * rms)

        @debug "PAH feature ($prof) with ($A_cgs, $μ, $fwhm, $m, $ν) and errors ($A_cgs_err, $μ_err, $fwhm_err, $m_err, $ν_err)"
        @debug "Flux=$flux +/- $f_err, EQW=$eqw +/- $e_err, SNR=$snr"

        # increment the parameter index
        pᵢ += 3
        if prof == :PearsonIV
            pᵢ += 2
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

        pₒ += 3
    end

    # Loop through lines
    p_lines = zeros(3cube_fitter.n_lines+3cube_fitter.n_acomps+5cube_fitter.n_lines)
    p_lines_err = zeros(3cube_fitter.n_lines+3cube_fitter.n_acomps+5cube_fitter.n_lines)

    # Unpack the relative flags
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

                # Convert amplitude to erg s^-1 cm^-2 μm^-1 sr^-1, put back in the normalization
                amp_cgs = MJysr_to_cgs(amp*N, mean_μm)
                amp_cgs_err = propagate_err ? MJysr_to_cgs_err(amp*N, amp_err*N, mean_μm, mean_μm_err) : 0.

                # Get the index of the central wavelength
                cent_ind = argmin(abs.(λ .- mean_μm))

                # Integrate over the solid angle
                amp_cgs *= area_sr[cent_ind]
                if propagate_err
                    amp_cgs_err *= area_sr[cent_ind]
                end

                # Get the extinction factor at the line center
                ext = extinction[cent_ind]
                tnorm = nuc_temp_fit ? templates_psfnuc : ones(length(λ))
                tn = tnorm[cent_ind]        
                
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
                feature .*= amp .* N .* extinction .* tnorm
                if propagate_err
                    feature_err[:,1] .*= extinction .* tnorm
                    feature_err[:,2] .*= extinction .* tnorm
                end

                # Calculate line flux using the helper function
                p_lines[pₒ], p_lines_err[pₒ] = calculate_flux(cube_fitter.lines.profiles[k, j], amp_cgs, amp_cgs_err, 
                    mean_μm, mean_μm_err, fwhm_μm, fwhm_μm_err, h3=h3, h3_err=h3_err, h4=h4, h4_err=h4_err, η=η, η_err=η_err, 
                    propagate_err=propagate_err)
                
                # Calculate equivalent width using the helper function
                p_lines[pₒ+1], p_lines_err[pₒ+1] = calculate_eqw(cube_fitter, λ, feature, comps, true, nuc_temp_fit, 
                    feature_err=feature_err, propagate_err=propagate_err)

                # SNR
                # p_lines[pₒ+2] = amp*N*ext*tn / std(I[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)] .- continuum[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)])
                rms = std(I[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)] .- continuum[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)]) 
                p_lines[pₒ+2] = sum(feature) / (sqrt(sum(snr_filter)) * rms)

                # Add to the total line profile
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

    p_dust, p_lines, p_dust_err, p_lines_err
end


"""
    calculate_eqw(λ, profile, comps, line, n_dust_cont, n_power_law, n_abs_feat, n_dust_feat,
        fit_sil_emission, amp, amp_err, peak, peak_err, fwhm, fwhm_err; <keyword arguments>)

Calculate the equivalent width (in microns) of a spectral feature, i.e. a PAH or emission line. Calculates the
integral of the ratio of the feature profile to the underlying continuum.
"""
function calculate_eqw(cube_fitter::CubeFitter, λ::Vector{T}, feature::Vector{T}, comps::Dict, line::Bool,
    nuc_temp_fit::Bool; feature_err::Union{Matrix{T},Nothing}=nothing, propagate_err::Bool=true) where {T<:Real}

    contin = zeros(length(λ))
    contin .+= comps["obscured_continuum"] .+ comps["unobscured_continuum"]
    if cube_fitter.fit_sil_emission
        abs_tot = ones(length(λ))
        for k ∈ 1:cube_fitter.n_abs_feat
            abs_tot .*= comps["abs_feat_$k"]
        end
        contin .+= comps["hot_dust"] .* comps["abs_ice"] .* comps["abs_ch"] .* abs_tot
    end
    templates_norm_nuc = nothing
    for q ∈ 1:cube_fitter.n_templates
        if !nuc_temp_fit
            contin .+= comps["templates_$q"]
        else
            templates_norm_nuc = comps["templates_$q"]
        end
    end
    # For line EQWs, we consider PAHs as part of the "continuum"
    if line
        for l ∈ 1:cube_fitter.n_dust_feat
            contin .+= comps["dust_feat_$l"] .* comps["extinction"]
        end
    end
    if nuc_temp_fit
        contin .*= templates_norm_nuc
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

