#=
This is the main file for actually fitting IFU cubes.  It contains functions for actually performing the
fitting across a single spaxel and an entire cube. The main calling sequence one will want to 
perform is first loading in the data, presumably from some FITS files, with the cubedata functions,
then creating a CubeFitter struct from the DataCube struct, followed by calling fit_cube! on the
CubeFitter. An example of this is provided in the test driver files in the test directory.
=#

############################## FITTING FUNCTIONS AND HELPERS ####################################


"""
    mask_emission_lines(λ, I; [Δ, n_inc_thresh, thresh])

Mask out emission lines in a given spectrum using a numerical second derivative and flagging 
negative(positive) spikes, indicating strong concave-downness(upness) up to some tolerance threshold (i.e. 3-sigma).
The widths of the lines are estimated using the number of pixels for which the numerical first derivative
is above some (potentially different) tolerance threshold. Returns the mask as a BitVector.

# Arguments
- `λ::Vector{<:Real}`: The wavelength vector of the spectrum
- `I::Vector{<:Real}`: The intensity vector of the spectrum
- `spectral_region::Symbol`: Either :MIR for mid-infrared or :OPT for optical
- `Δ::Union{Integer,Nothing}=nothing`: The resolution (width) of the numerical derivative calculations, in pixels
- `n_inc_thresh::Union{Integer,Nothing}=nothing`
- `thresh::Real=3.`: The sensitivity threshold for flagging lines with the second derivative test, in units of the RMS.

See also [`continuum_cubic_spline`](@ref)
"""
function mask_emission_lines(λ::Vector{<:Real}, I::Vector{<:Real}, Δ::Integer, n_inc_thresh::Integer, thresh::Real,
    overrides::Vector{Tuple{T,T}}=Vector{Tuple{Real,Real}}()) where {T<:Real}


    mask = falses(length(λ))
    # manual regions that wish to be masked out
    if length(overrides) > 0
        for override in overrides
            mask[override[1] .< λ .< override[2]] .= 1
        end
        return mask
    end

    # Wavelength difference vector
    diffs = diff(λ)
    diffs = [diffs; diffs[end]]

    # Calculate the numerical second derivative
    d2f = zeros(length(λ))
    @simd for i ∈ 1:length(λ)
        d2f[i] = (I[min(length(λ), i+Δ)] - 2I[i] + I[max(1, i-Δ)]) / (Δ * diffs[i])^2
    end
    W = (10 * Δ, 1000)

    # Find where the second derivative is significantly concave-down 
    for j ∈ 7:(length(λ)-7)
        # Only consider the spectrum within +/- W pixels from the point in question
        if any([abs(d2f[j]) > thresh * nanstd(d2f[max(1, j-Wi):min(length(λ), j+Wi)]) for Wi ∈ W])

            n_pix_l = 0
            n_pix_r = 0

            # Travel left/right from the center until the flux goes up 3 times
            n_inc = 0
            while n_inc < n_inc_thresh
                dI = I[j-n_pix_l-1] - I[j-n_pix_l]
                if dI > 0
                    n_inc += 1
                end
                if j-n_pix_l-1 == 1
                    break
                end
                n_pix_l += 1
            end
            n_inc = 0
            while n_inc < n_inc_thresh
                dI = I[j+n_pix_r+1] - I[j+n_pix_r]
                if dI > 0
                    n_inc += 1
                end
                if j+n_pix_r+1 == length(I)
                    break
                end
                n_pix_r += 1
            end

            mask[max(j-n_pix_l,1):min(j+n_pix_r,length(mask))] .= 1

        end
    end

    # Don't mask out these regions that tends to trick this method sometimes
    mask[11.10 .< λ .< 11.15] .= 0
    mask[11.17 .< λ .< 11.24] .= 0
    mask[11.26 .< λ .< 11.355] .= 0
    # mask[12.5 .< λ .< 12.79] .= 0

    # Force the beginning/end few pixels to be unmasked to prevent runaway splines at the edges
    mask[1:7] .= 0
    mask[end-7:end] .= 0

    # Return the line locations and the mask
    mask

end


"""
    mask_vectors!(mask_bad, user_mask, λ, I, σ, templates, channel_masks[, I_spline])

Apply two masks (mask_bad and user_mask) to a set of vectors (λ, I, σ, templates, channel_masks) to prepare
them for fitting. The mask_bad is a pixel-by-pixel mask flagging bad data, while the user_mask is a set of pairs
of wavelengths specifying entire regions to mask out.  The vectors are modified in-place.
"""
function mask_vectors!(mask_bad::BitVector, user_mask::Union{Nothing,Vector{<:Tuple}}, λ::Vector{<:Real}, I::Vector{<:Real}, 
    σ::Vector{<:Real}, templates::Matrix{<:Real}, channel_masks::Vector{BitVector}, I_spline::Union{Nothing,Vector{<:Real}}=nothing)

    do_spline = !isnothing(I_spline)

    λ = λ[.~mask_bad]
    I = I[.~mask_bad]
    σ = σ[.~mask_bad]
    templates = templates[.~mask_bad, :]
    channel_masks = [ch_mask[.~mask_bad] for ch_mask in channel_masks]
    if do_spline 
        I_spline = I_spline[.~mask_bad]
    end

    if !isnothing(user_mask)
        for pair in user_mask
            region = pair[1] .< λ .< pair[2]
            λ = λ[.~region]
            I = I[.~region]
            σ = σ[.~region]
            templates = templates[.~region, :]
            channel_masks = [ch_mask[.~region] for ch_mask in channel_masks]
            if do_spline
                I_spline = I_spline[.~region]
            end
        end
    end
    
end


"""
    continuum_cubic_spline(λ, I, σ, spectral_region)

Mask out the emission lines in a given spectrum using `mask_emission_lines` and replace them with
a coarse cubic spline fit to the continuum, using wide knots to avoid reinterpolating the lines or
noise. Returns the line mask, spline-interpolated I, and spline-interpolated σ.

# Arguments
- `λ::Vector{<:Real}`: The wavelength vector of the spectrum
- `I::Vector{<:Real}`: The flux vector of the spectrum
- `σ::Vector{<:Real}`: The uncertainty vector of the spectrum 
- `spectral_region::Symbol`: Either :MIR for mid-infrared or :OPT for optical

See also [`mask_emission_lines`](@ref)
"""
function continuum_cubic_spline(λ::Vector{<:Real}, I::Vector{<:Real}, σ::Vector{<:Real}, linemask_Δ::Integer, 
    linemask_n_inc_thresh::Integer, linemask_thresh::Real, overrides::Vector{Tuple{T,T}}=Vector{Tuple{Real,Real}}()) where {T<:Real}

    # Mask out emission lines so that they aren't included in the continuum fit
    mask_lines = mask_emission_lines(λ, I, linemask_Δ, linemask_n_inc_thresh, linemask_thresh, overrides)

    # Interpolate the NaNs
    # Break up cubic spline interpolation into knots 7 pixels long
    # (longer than a narrow emission line but not too long)
    scale = 7

    # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
    λknots = λ[1+scale:scale:end-scale]
    # Remove any knots that happen to fall within a masked pixel
    good = []
    for i ∈ eachindex(λknots)
        _, ind = findmin(abs.(λ .- λknots[i]))
        if ~mask_lines[ind]
            append!(good, [i])
        end
    end
    λknots = λknots[good]
    @debug "Performing cubic spline continuum fit with $(length(λknots)) knots"

    # Do a full cubic spline interpolation of the data
    I_spline = Spline1D(λ[.~mask_lines], I[.~mask_lines], λknots, k=3, bc="extrapolate").(λ)
    σ_spline = Spline1D(λ[.~mask_lines], σ[.~mask_lines], λknots, k=3, bc="extrapolate").(λ)
    # Linear interpolation over the lines
    I_spline[mask_lines] .= Spline1D(λ[.~mask_lines], I[.~mask_lines], λknots, k=1, bc="extrapolate").(λ[mask_lines])
    σ_spline[mask_lines] .= Spline1D(λ[.~mask_lines], σ[.~mask_lines], λknots, k=1, bc="extrapolate").(λ[mask_lines])

    mask_lines, I_spline, σ_spline
end


"""
    repeat_fit_jitter(λ_spax, I_spax, σ_spax, fit_func, params, lb, ub, parinfo, config, res0, fit_type; check_cont_snr)

Checks the status of a fit to see if it has gotten stuck at the initial position (i.e. if number of iterations is < 5),
and if so it jitters the starting parameters and tries to fit again.  It will try up to 10 times before giving up.
"""
function repeat_fit_jitter(λ_spax::Vector{<:Real}, I_spax::Vector{<:Real}, σ_spax::Vector{<:Real}, fit_func::Function,
    params::Vector{<:Real}, lb::Vector{<:Real}, ub::Vector{<:Real}, parinfo::Vector{CMPFit.Parinfo}, 
    config::CMPFit.Config, res0::CMPFit.Result, fit_type::String, spaxel::CartesianIndex; check_cont_snr::Bool=false)

    res = res0
    cont_snr = 10.
    if check_cont_snr
        cont_snr = nanmedian(I_spax ./ σ_spax)
    end
    n = 1
    while (res.niter < 5) && (cont_snr > 3)
        @warn "LM Solver is stuck on the initial state for the $fit_type fit of spaxel $spaxel. Jittering starting params..."
        # Jitter the starting parameters a bit
        jit_lo = (lb .- params) ./ 20  # defined to be negative
        jit_hi = (ub .- params) ./ 20  # defined to be positive
        # handle infinite upper bounds
        jit_hi[.~isfinite.(jit_hi)] .= .-jit_lo[.~isfinite.(jit_hi)]
        jit = dropdims(minimum(abs, [jit_lo jit_hi], dims=2), dims=2)
        # sample from a uniform distribution
        jitter = [j > 0 ? rand(Uniform(-j, j)) : 0.0 for j in jit]
        # redo the fit with the slightly jittered starting parameters
        @debug "Jittered starting parameters: $(params .+ jitter)"
        res = cmpfit(λ_spax, I_spax, σ_spax, fit_func, params .+ jitter, parinfo=parinfo, config=config)
        n += 1
        if n > 10
            @warn "LM solver has exceeded 10 tries on the continuum fit of spaxel $spaxel. Aborting."
            break
        end   
    end

    return res
end


"""
    continuum_fit_spaxel(cube_fitter, spaxel, λ, I, σ, mask_lines, mask_bad, N; 
        [init, use_ap, bootstrap_iter, p1_boots])

Fit the continuum of a given spaxel in the DataCube, masking out the emission lines, using the 
Levenberg-Marquardt least squares fitting method with the `CMPFit` package.  

This procedure has been adapted from PAHFIT (with some heavy adjustments -> masking out lines, allowing
PAH parameters to vary, and tying certain parameters together). See Smith, Draine, et al. 2007; 
http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::CartesianIndex`: The 2D index of the spaxel being fit.
- `λ::Vector{<:Real}`: The 1D wavelength vector 
- `I::Vector{<:Real}`: The 1D intensity vector
- `σ::Vector{<:Real}`: The 1D error vector
- `templates::Matrix{<:Real}`: The 2D templates array (1st dimension = wavelength, 2nd dimension = each template)
- `mask_lines::BitVector`: The mask giving the locations of emission lines
- `mask_bad::BitVector`: The mask giving the locations of bad pixels
- `N::Real`: The normalization
- `init::Bool=false`: Flag for the initial fit which fits the sum of all spaxels, to get an estimation for
    the initial parameter vector for individual spaxel fits
- `use_ap::Bool=false`: Flag for fitting an integrated spectrum within an aperture
- `nuc_temp_fit::Bool=false`: Flag for fitting a nuclear template
- `bootstrap_iter::Bool=false`: Flag for fitting multiple iterations of the same spectrum with bootstrapping
- `p1_boots::Union{Vector{<:Real},Nothing}=nothing`: If `bootstrap_iter` is true, this should give the best-fit parameter vector
for the initial non-bootstrapped fit of the spectrum.
"""
function continuum_fit_spaxel(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real}, 
    σ::Vector{<:Real}, templates::Matrix{<:Real}, mask_lines::BitVector, mask_bad::BitVector, N::Real; init::Bool=false, 
    use_ap::Bool=false, nuc_temp_fit::Bool=false, bootstrap_iter::Bool=false, p1_boots::Union{Vector{<:Real},Nothing}=nothing, 
    force_noext::Bool=false) 

    @debug """\n
    #########################################################
    ###   Beginning continuum fit for spaxel $spaxel...   ###
    #########################################################
    """
    # Normalize
    λ_spax = copy(λ)
    I_spax = I ./ N
    σ_spax = σ ./ N
    templates_spax = copy(templates)
    channel_masks = copy(cube_fitter.channel_masks)

    scale = 7
    # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
    λknots = λ[.~mask_lines][(1+scale):scale:(length(λ[.~mask_lines])-scale)]
    # Replace the masked lines with a linear interpolation
    I_spax[mask_lines] .= Spline1D(λ_spax[.~mask_lines], I_spax[.~mask_lines], λknots, k=1, bc="extrapolate").(λ[mask_lines]) 
    σ_spax[mask_lines] .= Spline1D(λ_spax[.~mask_lines], σ_spax[.~mask_lines], λknots, k=1, bc="extrapolate").(λ[mask_lines])
    for s in axes(templates_spax, 2)
        m = .~isfinite.(templates_spax[:, s])
        templates_spax[mask_lines .| m, s] .= Spline1D(λ[.~mask_lines .& .~m], templates_spax[.~mask_lines .& .~m, s], λknots, 
            k=1, bc="extrapolate")(λ[mask_lines .| m])
    end

    # Masked spectrum
    mask_vectors!(mask_bad, cube_fitter.user_mask, λ_spax, I_spax, σ_spax, templates_spax, channel_masks)

    # Get the priors and "locked" booleans for each parameter, split up by the 2 steps for the continuum fit
    plims, plock, tied_pairs, tied_indices = get_continuum_plimits(cube_fitter, spaxel, λ_spax, I_spax, σ_spax, init || use_ap, 
        templates_spax./N)

    # Split up the initial parameter vector into the components that we need for each fitting step
    pars_0, dstep_0 = get_continuum_initial_values(cube_fitter, spaxel, λ_spax, I_spax, σ_spax, N, init || use_ap, templates_spax./N)
    if bootstrap_iter
        pars_0 = p1_boots
    end
    if nuc_temp_fit && cube_fitter.spectral_region == :OPT
        # for optical fits there should be no channel splits so just lock the PSF amplitude to 1 during the nuclear fit
        plock[end-cube_fitter.n_templates+1:end] .= 1
        pars_0[end-cube_fitter.n_templates+1:end] .= 1.0
    end

    # Constrain optical depth to be at least 80% of the guess
    if !isnothing(cube_fitter.guess_tau) && (cube_fitter.extinction_curve != "decompose")
        pₑ = 3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law
        plims[pₑ] = (0.8 * pars_0[pₑ], plims[pₑ][2]) 
    end
    if force_noext
        pₑ = [3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law]
        pars_0[pₑ] .= 0.
        plock[pₑ] .= 1
    end

    @debug """\n
    ##########################################################################################################
    ########################################## FITTING THE CONTINUUM #########################################
    ##########################################################################################################
    """

    @debug "Continuum Parameters:"
    pfree_tied, pfix_tied, dfree_tied, plock_tied, lbfree_tied, ubfree_tied, n_free, n_tied = split_parameters(
        pars_0, dstep_0, plims, plock, tied_indices)

    # Convert parameter limits into CMPFit object
    parinfo, config = get_continuum_parinfo(n_free, lbfree_tied, ubfree_tied, dfree_tied)

    # Pre-compute the stellar templates, if all the ages and metallicities are locked
    # (to speed up fitting)
    if cube_fitter.spectral_region == :OPT
        lock_ssps = true
        ages = []
        logzs = []
        pᵢ = 1
        for _ in 1:cube_fitter.n_ssps
            lock_ssps &= plock[pᵢ+1] & plock[pᵢ+2]
            push!(ages, pars_0[pᵢ+1])
            push!(logzs, pars_0[pᵢ+2])
            pᵢ += 3
        end
        if lock_ssps
            @debug "Pre-calculating SSPs since all ages and metallicities are locked."
            stellar_templates = zeros(length(cube_fitter.ssp_λ), cube_fitter.n_ssps)
            for i in 1:cube_fitter.n_ssps
                stellar_templates[:, i] .= [cube_fitter.ssp_templates[j](ages[i], logzs[i]) for j in eachindex(cube_fitter.ssp_λ)]
            end
        else
            @debug "SSPs will be interpolated during the fit since ages and/or metallicities are free."
            stellar_templates = cube_fitter.ssp_templates
        end
    end

    # Fitting functions for either optical or MIR continuum
    function fit_cont_mir(x, pfree_tied)
        ptot = rebuild_full_parameters(pfree_tied, pfix_tied, plock_tied, tied_pairs, tied_indices, n_tied)
        model_continuum(x, ptot, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.profiles,
            cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.κ_abs, 
            cube_fitter.custom_ext_template, cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, false, templates_spax,
            channel_masks, nuc_temp_fit)
    end
    function fit_cont_opt(x, pfree_tied)
        ptot = rebuild_full_parameters(pfree_tied, pfix_tied, plock_tied, tied_pairs, tied_indices, n_tied)
        model_continuum(x, ptot, N, cube_fitter.vres, cube_fitter.vsyst_ssp, cube_fitter.vsyst_feii, cube_fitter.npad_feii,
            cube_fitter.n_ssps, cube_fitter.ssp_λ, stellar_templates, cube_fitter.feii_templates_fft, cube_fitter.n_power_law, 
            cube_fitter.fit_uv_bump, cube_fitter.fit_covering_frac, cube_fitter.fit_opt_na_feii, cube_fitter.fit_opt_br_feii, 
            cube_fitter.extinction_curve, templates_spax, cube_fitter.fit_temp_multexp, nuc_temp_fit)
    end
    fit_cont = cube_fitter.spectral_region == :MIR ? fit_cont_mir : fit_cont_opt

    # Do initial fit
    res = cmpfit(λ_spax, I_spax, σ_spax, fit_cont, pfree_tied, parinfo=parinfo, config=config)
    # This function checks how many iterations the first fit took, and if its less than 5 it repeats it with slightly different
    # starting parameters to try to get it to converge better. This is because rarely the first fit can get stuck in the starting
    # position.
    res = repeat_fit_jitter(λ_spax, I_spax, σ_spax, fit_cont, pfree_tied, lbfree_tied, ubfree_tied, parinfo, config, res, "continuum",
        spaxel; check_cont_snr=true)
    
    popt_0 = rebuild_full_parameters(res.param, pfix_tied, plock_tied, tied_pairs, tied_indices, n_tied)

    # if fitting the nuclear template, minimize the distance from 1.0 for the PSF template amplitudes and move
    # the residual amplitude to the continuum
    if nuc_temp_fit && cube_fitter.spectral_region == :MIR
        nuc_temp_fit_minimize_psftemp_amp!(cube_fitter, popt_0)
    end

    @debug "Continuum CMPFit Status: $(res.status)"

    popt = popt_0
    perr = zeros(Float64, length(popt))
    if !bootstrap_iter
        perr = rebuild_full_parameters(res.perror, zeros(length(pfix_tied)), plock_tied, tied_pairs, tied_indices, n_tied)
    end

    @debug "Best fit continuum parameters: \n $popt"
    @debug "Continuum parameter errors: \n $perr"
    # @debug "Continuum covariance matrix: \n $covar"

    # Create the full model, again only if not bootstrapping
    if cube_fitter.spectral_region == :MIR
        I_model, comps = model_continuum(λ, popt, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.profiles,
            cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.κ_abs, cube_fitter.custom_ext_template, 
            cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, false, templates, cube_fitter.channel_masks, nuc_temp_fit, true)
        # set optical depth to 0 if the template fits all of the spectrum
        for s in 1:cube_fitter.n_templates
            pₑ = [3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law]
            sil_abs_region = 9.0 .< λ .< 11.0
            if any(.~plock[pₑ]) && !init && !force_noext && any(popt[pₑ] .≠ 0) && sum(sil_abs_region) > 100
                if sum((abs.(I_model .- comps["templates_$s"])[sil_abs_region]) .< (σ[sil_abs_region]./N)) > (sum(sil_abs_region)/2)
                    @debug "Redoing the fit with optical depth locked to 0 due to template amplitudes"
                    return continuum_fit_spaxel(cube_fitter, spaxel, λ, I, σ, templates, mask_lines, mask_bad, N, false, init=init,
                        use_ap=use_ap, bootstrap_iter=bootstrap_iter, p1_boots=p1_boots, force_noext=true)
                end
            end
        end
    else
        I_model, comps = model_continuum(λ, popt, N, cube_fitter.vres, cube_fitter.vsyst_ssp, cube_fitter.vsyst_feii, cube_fitter.npad_feii,
            cube_fitter.n_ssps, cube_fitter.ssp_λ, stellar_templates, cube_fitter.feii_templates_fft, cube_fitter.n_power_law, cube_fitter.fit_uv_bump, 
            cube_fitter.fit_covering_frac, cube_fitter.fit_opt_na_feii, cube_fitter.fit_opt_br_feii, cube_fitter.extinction_curve, templates,
            cube_fitter.fit_temp_multexp, nuc_temp_fit, true)
    end
    
    # Estimate PAH template amplitude
    if cube_fitter.spectral_region == :MIR
        pahtemp = zeros(length(λ))
        for i in 1:cube_fitter.n_dust_feat
            pahtemp .+= comps["dust_feat_$i"]
        end
        pah_amp = repeat([maximum(pahtemp)/2], 2)
    else
        pah_amp = zeros(2)
    end

    if init
        cube_fitter.p_init_cont[:] .= popt
        # Save the results to a file 
        # save running best fit parameters in case the fitting is interrupted
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_cont.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_cont, ',')
        end
        cube_fitter.p_init_pahtemp[:] .= pah_amp
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_pahtemp.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_pahtemp, ',')
        end
    end

    # Print the results (to the logger)
    pretty_print_continuum_results(cube_fitter, popt, perr, I_spax)

    popt, I_model, comps, n_free, perr, pah_amp
end


function continuum_fit_spaxel(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real}, 
    σ::Vector{<:Real}, templates::Matrix{<:Real}, mask_lines::BitVector, mask_bad::BitVector, N::Real, split_flag::Bool; 
    init::Bool=false, use_ap::Bool=false, nuc_temp_fit::Bool=false, bootstrap_iter::Bool=false, p1_boots::Union{Vector{<:Real},Nothing}=nothing,
    force_noext::Bool=false) 

    if (!split_flag) || (cube_fitter.spectral_region != :MIR)
        return continuum_fit_spaxel(cube_fitter, spaxel, λ, I, σ, templates, mask_lines, mask_bad, N; init=init, use_ap=use_ap,
            bootstrap_iter=bootstrap_iter, p1_boots=p1_boots)
    end

    # This version of the function should only ever be called for MIR fitting since "split_flag" doesn't apply for optical fitting
    @assert !cube_fitter.fit_joint "The fit_joint and use_pah_templates options are mutually exclusive!"

    @debug """\n
    #########################################################
    ###   Beginning continuum fit for spaxel $spaxel...   ###
    #########################################################
    """
    # Normalize
    λ_spax = copy(λ)
    I_spax = I ./ N
    σ_spax = σ ./ N
    templates_spax = copy(templates)
    channel_masks = copy(cube_fitter.channel_masks)

    scale = 7
    # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
    λknots = λ[.~mask_lines][(1+scale):scale:(length(λ[.~mask_lines])-scale)]
    # Replace the masked lines with a linear interpolation
    I_spax[mask_lines] .= Spline1D(λ_spax[.~mask_lines], I_spax[.~mask_lines], λknots, k=1, bc="extrapolate").(λ[mask_lines]) 
    σ_spax[mask_lines] .= Spline1D(λ_spax[.~mask_lines], σ_spax[.~mask_lines], λknots, k=1, bc="extrapolate").(λ[mask_lines])
    for s in axes(templates_spax, 2)
        m = .~isfinite.(templates_spax[:, s])
        templates_spax[mask_lines .| m, s] .= Spline1D(λ[.~mask_lines .& .~m], templates_spax[.~mask_lines .& .~m, s], λknots, 
            k=1, bc="extrapolate")(λ[mask_lines .| m])
    end

    # Masked spectrum
    mask_vectors!(mask_bad, cube_fitter.user_mask, λ_spax, I_spax, σ_spax, templates_spax, channel_masks)

    # Get the priors and "locked" booleans for each parameter, split up by the 2 steps for the continuum fit
    plims_1, plims_2, lock_1, lock_2, tied_pairs, tied_indices = get_continuum_plimits(cube_fitter, spaxel, λ_spax, I_spax, σ_spax, init || use_ap,
        templates_spax./N, split=true)

    # Split up the initial parameter vector into the components that we need for each fitting step
    pars_1, pars_2, dstep_1, dstep_2 = get_continuum_initial_values(cube_fitter, spaxel, λ_spax, I_spax, σ_spax, N, init || use_ap, 
        templates_spax./N, split=true)
    if bootstrap_iter
        pars_1 = vcat(p1_boots[1:(2+2*cube_fitter.n_dust_cont+2*cube_fitter.n_power_law+4+(cube_fitter.extinction_curve == "decompose" ? 3 : 1)+
            3*cube_fitter.n_abs_feat+(cube_fitter.fit_sil_emission ? 6 : 0)+(cube_fitter.fit_temp_multexp ? 8 : cube_fitter.n_templates*cube_fitter.n_channels))], 
            p1_boots[end-1:end])
        pars_2 = p1_boots[(3+2*cube_fitter.n_dust_cont+2*cube_fitter.n_power_law+4+(cube_fitter.extinction_curve == "decompose" ? 3 : 1)+
            3*cube_fitter.n_abs_feat+(cube_fitter.fit_sil_emission ? 6 : 0)+(cube_fitter.fit_temp_multexp ? 8 : cube_fitter.n_templates*cube_fitter.n_channels)):end-2]
    end

    # Constrain optical depth to be at least 80% of the guess
    if !isnothing(cube_fitter.guess_tau) && (cube_fitter.extinction_curve != "decompose")
        pₑ = 3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law
        plims_1[pₑ] = (0.8 * pars_1[pₑ], plims_1[pₑ][2])
    end
    if force_noext
        pₑ = [3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law]
        pars_1[pₑ] .= 0.
        lock_1[pₑ] .= 1
    end

    @debug """\n
    ##########################################################################################################
    ########################## STEP 1 - FIT THE BLACKBODY CONTINUUM WITH PAH TEMPLATES #######################
    ##########################################################################################################
    """

    @debug "Continuum Step 1 Parameters:"
    p1free_tied, p1fix_tied, d1free_tied, lock_1_tied, lb_1_free_tied, ub_1_free_tied, n_free_1, n_tied_1 = split_parameters(
        pars_1, dstep_1, plims_1, lock_1, tied_indices)

    # None of the step 2 parameters can be tied
    p2fix = pars_2[lock_2]
    p2free = pars_2[.~lock_2]
    d2free = dstep_2[.~lock_2]

    # Count free parameters
    n_free_2 = sum(.~lock_2)

    # Lower/upper limits
    lb_2 = [pl[1] for pl in plims_2[.~lock_2]]
    ub_2 = [pl[2] for pl in plims_2[.~lock_2]]

    # Convert parameter limits into CMPFit object
    parinfo_1, parinfo_2, config = get_continuum_parinfo(n_free_1, n_free_2, lb_1_free_tied, ub_1_free_tied, lb_2, ub_2, d1free_tied, d2free)

    @debug "Beginning continuum fitting with Levenberg-Marquardt least squares (CMPFit):"

    function fit_step1(x, pfree_tied, return_comps=false)
        ptot = rebuild_full_parameters(pfree_tied, p1fix_tied, lock_1_tied, tied_pairs, tied_indices, n_tied_1)
        if !return_comps
            model_continuum(x, ptot, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.profiles,
                cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.κ_abs, 
                cube_fitter.custom_ext_template, cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, true, templates_spax,
                channel_masks, nuc_temp_fit)
        else
            model_continuum(x, ptot, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.profiles,
                cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.κ_abs, 
                cube_fitter.custom_ext_template, cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, true, templates_spax,
                channel_masks, nuc_temp_fit, true)
        end
    end
    res_1 = cmpfit(λ_spax, I_spax, σ_spax, fit_step1, p1free_tied, parinfo=parinfo_1, config=config)
    res_1 = repeat_fit_jitter(λ_spax, I_spax, σ_spax, fit_step1, p1free_tied, lb_1_free_tied, ub_1_free_tied, parinfo_1, config,
        res_1, "continuum (step 1)", spaxel; check_cont_snr=true)

    @debug "Continuum CMPFit Status: $(res_1.status)"

    popt_0 = rebuild_full_parameters(res_1.param, p1fix_tied, lock_1_tied, tied_pairs, tied_indices, n_tied_1)

    # if fitting the nuclear template, minimize the distance from 1.0 for the PSF template amplitudes and move
    # the residual amplitude to the continuum
    if nuc_temp_fit
        nuc_temp_fit_minimize_psftemp_amp!(cube_fitter, popt_0)
    end
    popt_0_tied = copy(popt_0)
    deleteat!(popt_0_tied, tied_indices)
    popt_0_free_tied = popt_0_tied[.~lock_1_tied]

    # Create continuum without the PAH features
    _, ccomps = fit_step1(λ_spax, popt_0_free_tied, true)

    I_cont = ccomps["obscured_continuum"] .+ ccomps["unobscured_continuum"]
    if cube_fitter.fit_sil_emission
        abs_tot = ones(length(λ))
        for k ∈ 1:cube_fitter.n_abs_feat
            abs_tot .*= ccomps["abs_feat_$k"]
        end
        I_cont .+= ccomps["hot_dust"] .* abs_tot
    end
    template_norm = nothing
    for l ∈ 1:cube_fitter.n_templates
        if !nuc_temp_fit
            I_cont .+= ccomps["templates_$l"]
        else
            template_norm = ccomps["templates_$l"]
        end
    end
    if nuc_temp_fit
        I_cont .*= template_norm
    end

    @debug """\n
    ##########################################################################################################
    ################################# STEP 2 - FIT THE PAHs AS DRUDE PROFILES ################################
    ##########################################################################################################
    """
    @debug "Continuum Step 2 Parameters: \n $pars_2"
    @debug "Continuum Parameters locked? \n $lock_2"
    @debug "Continuum Lower limits: \n $([pl[1] for pl in plims_2])"
    @debug "Continuum Upper limits: \n $([pl[2] for pl in plims_2])"

    @debug "Beginning continuum fitting with Levenberg-Marquardt least squares (CMPFit):"

    # Wrapper function
    function fit_step2(x, pfree, return_comps=false)
        ptot = zeros(Float64, length(pars_2))
        ptot[.~lock_2] .= pfree
        ptot[lock_2] .= p2fix
        if !return_comps
            model_pah_residuals(x, ptot, cube_fitter.dust_features.profiles, ccomps["extinction"], template_norm, nuc_temp_fit)
        else
            model_pah_residuals(x, ptot, cube_fitter.dust_features.profiles, ccomps["extinction"], template_norm, nuc_temp_fit, true)
        end
    end
    res_2 = cmpfit(λ_spax, I_spax.-I_cont, σ_spax, fit_step2, p2free, parinfo=parinfo_2, config=config)
    res_2 = repeat_fit_jitter(λ_spax, I_spax.-I_cont, σ_spax, fit_step2, p2free, lb_2, ub_2, parinfo_2, config, res_2, 
        "continuum (step 2)", spaxel; check_cont_snr=true)

    @debug "Continuum CMPFit Step 2 status: $(res_2.status)"

    popt_1 = rebuild_full_parameters(popt_0_free_tied, p1fix_tied, lock_1_tied, tied_pairs, tied_indices, n_tied_1)
    # remove the PAH template amplitudes
    popt_1 = popt_1[1:end-2]
    perr_1 = zeros(length(popt_1))
    if !bootstrap_iter
        perr_1 = rebuild_full_parameters(res_1.perror, zeros(length(p1fix_tied)), lock_1_tied, tied_pairs, tied_indices, n_tied_1)
        perr_1 = perr_1[1:end-2]
    end

    popt_2 = zeros(length(pars_2))
    popt_2[.~lock_2] .= res_2.param
    popt_2[lock_2] .= p2fix
    perr_2 = zeros(length(pars_2))
    if !bootstrap_iter
        perr_2[.~lock_2] .= res_2.perror
    end

    popt = [popt_1; popt_2]
    perr = [perr_1; perr_2]

    n_free = n_free_1 + n_free_2 - 2
    pahtemp = cube_fitter.spectral_region == :MIR ? popt_0[end-1:end] : zeros(2)

    @debug "Best fit continuum parameters: \n $popt"
    @debug "Continuum parameter errors: \n $perr"
    # @debug "Continuum covariance matrix: \n $covar"

    # Create the full model, again only if not bootstrapping
    I_model, comps = model_continuum(λ, popt, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.profiles,
        cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.κ_abs, cube_fitter.custom_ext_template, 
        cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, false, templates, cube_fitter.channel_masks, nuc_temp_fit, true)

    # set optical depth to 0 if the template fits all of the spectrum
    for s in 1:cube_fitter.n_templates
        pₑ = [3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law]
        sil_abs_region = 9.0 .< λ .< 11.0
        if any(.~lock_1[pₑ]) && !init && !force_noext && any(popt[pₑ] .≠ 0) && sum(sil_abs_region) > 100
            if sum((abs.(I_model .- comps["templates_$s"])[sil_abs_region]) .< (σ[sil_abs_region]./N)) > (sum(sil_abs_region)/2)
                @debug "Redoing the fit with optical depth locked to 0 due to template amplitudes"
                return continuum_fit_spaxel(cube_fitter, spaxel, λ, I, σ, templates, mask_lines, mask_bad, N, split_flag, init=init,
                    use_ap=use_ap, bootstrap_iter=bootstrap_iter, p1_boots=p1_boots, force_noext=true)
            end
        end
    end

    if init
        cube_fitter.p_init_cont[:] .= popt
        cube_fitter.p_init_pahtemp[:] .= res_1.param[end-1:end]
        # Save the results to a file 
        # save running best fit parameters in case the fitting is interrupted
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_cont.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_cont, ',')
        end
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_pahtemp.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_pahtemp, ',')
        end
    end

    # Print the results (to the logger)
    pretty_print_continuum_results(cube_fitter, popt, perr, I_spax)

    popt, I_model, comps, n_free, perr, pahtemp
end


"""
    perform_line_component_test!(cube_fitter, spaxel, p₀, param_lock, lower_bounds, upper_bounds,
        λnorm, Inorm, σnorm, lsf_interp_func)

Calculates the significance of additional line components and determines whether or not to include them in the fit.
Modifies the p₀ and param_lock vectors in-place to reflect these changes (by locking the amplitudes and other parameters
for non-necessary line components to 0).

# Arguments 
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::CartesianIndex`: The 2D index of the spaxel being fit.
- `p₀::Vector{<:Real}`: The initial parameter vector 
- `param_lock::BitVector`: A vector specifying which parameters are locked
- `lower_bounds::Vector{<:Real}`: A vector specifying lower limits on the parameters
- `upper_bounds::Vector{<:Real}`: A vector specifying upper limits on the parameters
- `λnorm::Vector{<:Real}`: The 1D wavelength vector
- `Inorm::Vector{<:Real}`: The 1D normalized intensity vector
- `σnorm::Vector{<:Real}`: The 1D normalized error vector
- `lsf_interp_func::Function`: Interpolating function for the line-spread function (LSF) in km/s as a function of wavelength
- `all_fail::Bool=false`: Flag to force all tests to automatically fail, fitting each line with one profile.
- `bootstrap_iter::Bool=false`: If true, disables plotting 
"""
function perform_line_component_test!(cube_fitter::CubeFitter, spaxel::CartesianIndex, p₀::Vector{<:Real}, 
    param_lock::BitVector, lower_bounds::Vector{<:Real}, upper_bounds::Vector{<:Real}, λnorm::Vector{<:Real},
    Inorm::Vector{<:Real}, σnorm::Vector{<:Real}, lsf_interp_func::Function; all_fail::Bool=false, bootstrap_iter::Bool=false)
    @debug "Performing line component testing..."

    # Perform a test to see if each line with > 1 component really needs multiple components to be fit
    line_names = Symbol[]
    profiles_to_fit_list = Int[]
    pᵢ = 1
    for i in 1:cube_fitter.n_lines
        n_prof = 0
        pstart = pᵢ
        pcomps = Int[]
        for j in 1:cube_fitter.n_comps
            if !isnothing(cube_fitter.lines.profiles[i, j])
                n_prof += 1
                # Check if using a flexible_wavesol tied voff -> if so there is an extra voff parameter
                if !isnothing(cube_fitter.lines.tied_voff[i, j]) && cube_fitter.flexible_wavesol && isone(j)
                    pc = 4
                else
                    pc = 3
                end
                if cube_fitter.lines.profiles[i, j] == :GaussHermite
                    pc += 2
                elseif cube_fitter.lines.profiles[i, j] == :Voigt
                    pc += 1
                end
                push!(pcomps, pc)
                pᵢ += pc
            end
        end
        # Skip lines with only 1 profile
        if n_prof == 1
            continue
        end
        # Only test the lines that are specified to be tested
        if !(cube_fitter.lines.names[i] ∈ vcat(cube_fitter.line_test_lines...))
            continue
        end

        # Constrain the fitting region
        voff_max = max(abs(lower_bounds[pstart+1]), abs(upper_bounds[pstart+1]))
        fwhm_max = (!isnothing(cube_fitter.lines.tied_voff[i, 1]) && cube_fitter.flexible_wavesol) ? upper_bounds[pstart+3] : upper_bounds[pstart+2]
        wbounds = cube_fitter.lines.λ₀[i] .* (1-(voff_max+fwhm_max)/C_KMS, 1+(voff_max+fwhm_max)/C_KMS)
        region = wbounds[1] .< λnorm .< wbounds[2]

        line_object = TransitionLines(
            [cube_fitter.lines.names[i]], 
            [cube_fitter.lines.latex[i]], 
            [cube_fitter.lines.annotate[i]],
            [cube_fitter.lines.λ₀[i]], 
            [cube_fitter.lines.sort_order[i]],
            reshape(cube_fitter.lines.profiles[i, :], (1, cube_fitter.n_comps)), 
            reshape(cube_fitter.lines.tied_amp[i, :], (1, cube_fitter.n_comps)),
            reshape(cube_fitter.lines.tied_voff[i, :], (1, cube_fitter.n_comps)),
            reshape(cube_fitter.lines.tied_fwhm[i, :], (1, cube_fitter.n_comps)), 
            reshape(cube_fitter.lines.acomp_amp[i, :], (1, cube_fitter.n_comps-1)), 
            reshape(cube_fitter.lines.voff[i, :], (1, cube_fitter.n_comps)), 
            reshape(cube_fitter.lines.fwhm[i, :], (1, cube_fitter.n_comps)), 
            reshape(cube_fitter.lines.h3[i, :], (1, cube_fitter.n_comps)),
            reshape(cube_fitter.lines.h4[i, :], (1, cube_fitter.n_comps)), 
            reshape(cube_fitter.lines.η[i, :], (1, cube_fitter.n_comps)),
            cube_fitter.lines.combined, 
            cube_fitter.lines.rel_amp,
            cube_fitter.lines.rel_voff,
            cube_fitter.lines.rel_fwhm
        )

        if cube_fitter.plot_line_test && !bootstrap_iter
            fig, ax = plt.subplots()
            ax.plot(λnorm[region], Inorm[region], "k-", label="Data")
        end

        # Perform fits for all possible numbers of components
        last_chi2 = test_chi2 = 0.
        test_p = last_p = 0
        test_n = length(λnorm[region]) 
        critical_val = 0.
        F_data = 0.
        chi2_A = chi2_B = 0.
        profiles_to_fit = 0
        for np in 1:n_prof
            @debug "Testing $(cube_fitter.lines.names[i]) with $np components:"
            if all_fail
                @debug "all_fail flag has been set -- fitting with 1 profile"
                profiles_to_fit = 1
                break
            end

            # Stop index
            pstop = pstart + sum(pcomps[1:np]) - 1

            # Parameters
            parameters = p₀[pstart:pstop]
            plock = param_lock[pstart:pstop]
            pfree = parameters[.~plock]
            pfree = clamp.(pfree, lower_bounds[pstart:pstop][.~plock], upper_bounds[pstart:pstop][.~plock])

            function fit_func_test(x, pfree) 
                p = zeros(length(parameters))
                p[.~plock] .= pfree
                p[plock] .= parameters[plock]
                model_line_residuals(x, p, 1, np, line_object, cube_fitter.flexible_wavesol,
                    ones(sum(region)), lsf_interp_func, cube_fitter.relative_flags, nothing, false, false)
            end

            fit_func_lnL = p -> -ln_likelihood(Inorm[region], fit_func_test(λnorm[region], p), σnorm[region])
            x_tol = 1e-5
            f_tol = abs(fit_func_lnL(pfree) - fit_func_lnL(pfree .* (1 .- x_tol)))
            f_tol = 10^floor(log10(f_tol))
            
            # Parameter info
            # parinfo_test = CMPFit.Parinfo(length(pfree))
            # for i in eachindex(pfree)
            #     # parinfo_test[i].fixed = param_lock[pstart+i-1]
            #     parinfo_test[i].limited = (1,1)
            #     parinfo_test[i].limits = (lower_bounds[pstart:pstop][.~plock][i], upper_bounds[pstart:pstop][.~plock][i])
            # end
            # config_test = CMPFit.Config()

            # res_test = cmpfit(λnorm[region], Inorm[region], σnorm[region], fit_func_test, pfree, 
            #                   parinfo=parinfo_test, config=config_test)

            # Fit with simulated annealing
            res_test = Optim.optimize(fit_func_lnL, lower_bounds[pstart:pstop][.~plock], upper_bounds[pstart:pstop][.~plock], pfree,
                SAMIN(;rt=0.9, nt=5, ns=5, neps=5, x_tol=x_tol, f_tol=f_tol, verbosity=0), Optim.Options(iterations=10^6))

            # Save the reduced chi2 values
            test_model = fit_func_test(λnorm[region], res_test.minimizer)
            χ² = sum((Inorm[region] .- test_model).^2 ./ σnorm[region].^2)
            test_p = sum(.~plock)
            dof = test_n - test_p
            test_chi2 = χ²

            chi2_A = round(last_chi2/dof, digits=3)
            chi2_B = round(test_chi2/dof, digits=3)
            # test_stat = 1.0 - test_chi2/last_chi2

            ##### Perform an F-test #####
            test_passed, F_data, critical_val = F_test(test_n, last_p, test_p, last_chi2, test_chi2, cube_fitter.line_test_threshold)

            @debug "(A) Previous chi^2 = $last_chi2"
            @debug "(B) New chi^2 = $test_chi2"
            @debug "F-value = $F_data | Critical value = $critical_val"
            @debug "TEST $(test_passed ? "SUCCESS" : "FAILURE")"

            if cube_fitter.plot_line_test && !bootstrap_iter
                ax.plot(λnorm[region], test_model, linestyle="-", label="$np-component model")
            end

            if !test_passed && (np > 1)
                break
            end
            last_chi2 = test_chi2
            last_p = test_p
            profiles_to_fit += 1
        end
        F_final = round(F_data, sigdigits=3)
        crit_final = round(critical_val, sigdigits=3)
        @debug "$(cube_fitter.lines.names[i]) will have $profiles_to_fit components"

        push!(profiles_to_fit_list, profiles_to_fit)
        push!(line_names, cube_fitter.lines.names[i])

        if cube_fitter.plot_line_test && !bootstrap_iter
            ax.set_xlabel(L"$\lambda_{\rm rest}$ ($\mu$m)")
            ax.set_ylabel("Normalized Intensity")
            ax.set_title(cube_fitter.lines.latex[i])
            ax.legend(loc="upper right")
            ax.set_xlim(wbounds[1], wbounds[2])
            ax.annotate("Result: $profiles_to_fit profile(s)\n" * L"$\tilde{\chi}^2_A = %$chi2_A$" * "\n" *
                L"$\tilde{\chi}^2_B = %$chi2_B$" * "\n" * L"$F = %$F_final$" * "\n" * L"$F_{\rm crit} = %$crit_final$", 
                (0.05, 0.95), xycoords="axes fraction", ha="left", va="top")
            ax.axvline(cube_fitter.lines.λ₀[i], linestyle="--", alpha=0.5, color="k", lw=0.5)
            folder = joinpath("output_$(cube_fitter.name)", "line_tests", "$(cube_fitter.lines.names[i])")
            if !isdir(folder)
                mkpath(folder)
            end
            fname = isone(length(spaxel)) ? "voronoi_bin_$(spaxel[1])" : "spaxel_$(spaxel[1])_$(spaxel[2])"
            plt.savefig(joinpath(folder, "$fname.pdf"), dpi=300, bbox_inches="tight")
            plt.close()
        end
    end

    for grp in cube_fitter.line_test_lines
        # Use "*" as a special character in the line name to indicate that a line should not be tested, but should still
        # be included in the group such that it is fit with the same number of components as the other group members.
        group = [Symbol(replace(string(g), "*" => "")) for g in grp]
        # Get the group member indices
        inds = [findfirst(ln .== line_names) for ln in group]
        inds = inds[.~isnothing.(inds)]  # may get nothings if the "*" flag is used, but this is ok since we only need the inds
                                         # to find the maximum number of profiles in the group, so the "*"s are irrelevant here
        # Fit the maximum number of components in the group
        profiles_group = maximum(profiles_to_fit_list[inds])

        # Save the number of components fit for this spaxel
        if spaxel != CartesianIndex(0,0)
            for ln in group
                cube_fitter.n_fit_comps[ln][spaxel] = profiles_group
            end
        end

        # Lock the amplitudes to 0 for any profiles that will not be fit
        pᵢ = 1
        for i in 1:cube_fitter.n_lines
            for j in 1:cube_fitter.n_comps
                if !isnothing(cube_fitter.lines.profiles[i, j])
                    amp_ind = pᵢ
                    # Check if using a flexible_wavesol tied voff -> if so there is an extra voff parameter
                    if !isnothing(cube_fitter.lines.tied_voff[i, j]) && cube_fitter.flexible_wavesol && isone(j)
                        pc = 4
                    else
                        pc = 3
                    end
                    if cube_fitter.lines.profiles[i, j] == :GaussHermite
                        pc += 2
                    elseif cube_fitter.lines.profiles[i, j] == :Voigt
                        pc += 1
                    end
                    pᵢ += pc
                    # Check if the line is a member of the group, and if so, lock any profiles > profiles_group
                    if (cube_fitter.lines.names[i] in group) && (profiles_group < j)
                        # Amplitude
                        p₀[amp_ind] = 0.
                        # Voff (flexible_wavesol only applies to the first component which is always fit, So
                        # here we can safely assume voff is always the 2nd and fwhm is always the 3rd comp)
                        # FWHM (lock at a small nonzero value to prevent infinities)
                        param_lock[amp_ind:amp_ind+2] .= 1
                        if cube_fitter.lines.profiles[i, j] == :GaussHermite
                            param_lock[amp_ind+3:amp_ind+4] .= 1
                        end
                        if cube_fitter.lines.profiles[i, j] == :Voigt
                            param_lock[amp_ind+3] = 1
                        end
                    end
                end
            end
        end
    end
end


"""
    line_fit_spaxel(cube_fitter, spaxel, λ, I, σ, mask_lines, mask_bad, continuum, ext_curve, 
        lsf_interp_func, N; [init, use_ap, bootstrap_iter, p1_boots])

Fit the emission lines of a given spaxel in the DataCube, subtracting the continuum, using the 
Simulated Annealing fitting method with the `Optim` package and the Levenberg-Marquardt method with `CMPFit`.

This procedure has been adapted from PAHFIT (with some heavy adjustments). 
See Smith, Draine, et al. 2007; http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::CartesianIndex`: The 2D index of the spaxel being fit.
- `λ::Vector{<:Real}`: The 1D wavelength vector 
- `I::Vector{<:Real}`: The 1D intensity vector
- `σ::Vector{<:Real}`: The 1D error vector
- `mask_lines::BitVector`: The mask giving the locations of emission Lines
- `mask_bad::BitVector`: The mask giving the locations of bad pixels
- `continuum::Vector{<:Real}`: The fitted continuum level of the spaxel being fit (which will be subtracted
    before the lines are fit)
- `ext_curve::Vector{<:Real}`: The extinction curve of the spaxel being fit (which will be used to calculate
    extinction-corrected line amplitudes and fluxes)
- `lsf_interp_func::Function`: Interpolating function for the line-spread function (LSF) in km/s as a function of wavelength
- `template_psfnuc::Union{Vector{<:Real},Nothing}`: The PSF template at the nuclear spectrum to be used when fitting the nuclear template
- `N::Real`: The normalization
- `init::Bool=false`: Flag for the initial fit which fits the sum of all spaxels, to get an estimation for
    the initial parameter vector for individual spaxel fits
- `use_ap::Bool=false`: Flag for fitting an integrated spectrum within an aperture
- `nuc_temp_fit::Bool=false`: Flag for fitting the nuclear template
- `bootstrap_iter::Bool=false`: Flag for fitting multiple iterations of the same spectrum with bootstrapping
- `p1_boots::Union{Vector{<:Real},Nothing}=nothing`: If `bootstrap_iter` is true, this should give the best-fit parameter vector
for the initial non-bootstrapped fit of the spectrum.
"""
function line_fit_spaxel(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real},
    σ::Vector{<:Real}, mask_bad::BitVector, continuum::Vector{<:Real}, ext_curve::Vector{<:Real}, 
    lsf_interp_func::Function, template_psfnuc::Union{Vector{<:Real},Nothing}, N::Real; init::Bool=false, use_ap::Bool=false, nuc_temp_fit::Bool=false, 
    bootstrap_iter::Bool=false, p1_boots::Union{Vector{<:Real},Nothing}=nothing)

    @debug """\n
    #########################################################
    ###      Beginning line fit for spaxel $spaxel...     ###
    #########################################################
    """

    @debug "Using normalization N=$N"

    # Normalized flux and uncertainty by subtracting the continuum fit and dividing by the maximum
    λnorm = copy(λ)
    Inorm = I ./ N .- continuum
    σnorm = σ ./ N
    ext_curve_norm = copy(ext_curve)
    template_norm = nuc_temp_fit ? copy(template_psfnuc) : nothing

    # This ensures that any lines that fall within the masked regions will go to 0
    Inorm[mask_bad] .= 0.

    # if !isnothing(cube_fitter.user_mask)
    #     # Mask out additional regions
    #     for pair in cube_fitter.user_mask
    #         region = pair[1] .< λnorm .< pair[2]
    #         Inorm[region] .= 0.
    #     end
    # end

    plimits, param_lock, param_names, tied_pairs, tied_indices = get_line_plimits(cube_fitter, init || use_ap, nuc_temp_fit, ext_curve_norm)
    p₀, step = get_line_initial_values(cube_fitter, spaxel, init || use_ap)
    lower_bounds = [pl[1] for pl in plimits]
    upper_bounds = [pl[2] for pl in plimits]

    # Perform line component tests to determine which line components are actually necessary to include in the fit
    if (length(cube_fitter.line_test_lines) > 0) && !init
        perform_line_component_test!(cube_fitter, spaxel, p₀, param_lock, lower_bounds, upper_bounds, λnorm, Inorm, σnorm,
            lsf_interp_func, bootstrap_iter=bootstrap_iter)
    end

    @debug "Line Parameters:"
    pfree_tied, pfix_tied, dfree_tied, param_lock_tied, lbfree_tied, ubfree_tied, pnames_tied, n_free, n_tied = split_parameters(
        p₀, step, plimits, param_lock, tied_indices; param_names=param_names)

    # Get CMPFit parinfo object from bounds
    parinfo, config = get_line_parinfo(n_free, lbfree_tied, ubfree_tied, dfree_tied)

    # Place initial values within the lower/upper limits (amplitudes may be too large if the extinction levels differ between spaxels)
    pfree_tied = clamp.(pfree_tied, lbfree_tied, ubfree_tied)

    # Wrapper function for fitting only the free, tied parameters
    function fit_step3(x, pfree_tied, func; n=0)
        ptot = rebuild_full_parameters(pfree_tied, pfix_tied, param_lock_tied, tied_pairs, tied_indices, n_tied)
        func(x, ptot, n)
    end

    _fit_global = false
    if (init || use_ap || cube_fitter.fit_all_global) && !bootstrap_iter
    # if false
        @debug "Beginning Line fitting with Simulated Annealing:"

        fit_func = (x, p, n) -> -ln_likelihood(
                                Inorm, 
                                model_line_residuals(x, p, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, 
                                    cube_fitter.flexible_wavesol, ext_curve_norm, lsf_interp_func, cube_fitter.relative_flags,
                                    template_norm, nuc_temp_fit), 
                                σnorm)
        x_tol = 1e-5
        f_tol = abs(fit_func(λnorm, p₀, 0) - fit_func(λnorm, clamp.(p₀ .* (1 .- x_tol), lower_bounds, upper_bounds), 0))
        f_tol = 10^floor(log10(f_tol))
        lb_global = lbfree_tied
        ub_global = [isfinite(ub) ? ub : 1e10 for ub in ubfree_tied]

        # First, perform a bounded Simulated Annealing search for the optimal parameters with a generous max iterations and temperature rate (rt)
        res = Optim.optimize(p -> fit_step3(λnorm, p, fit_func), lb_global, ub_global, pfree_tied, 
            SAMIN(;rt=0.9, nt=5, ns=5, neps=5, f_tol=f_tol, x_tol=x_tol, verbosity=0), Optim.Options(iterations=10^6))
        p₁ = res.minimizer 

        # Write convergence results to file, if specified
        if cube_fitter.track_convergence
            global file_lock
            # use the ReentrantLock to prevent multiple processes from trying to write to the same file at once
            lock(file_lock) do 
                open(joinpath("output_$(cube_fitter.name)", "loki.convergence.log"), "a") do conv
                    redirect_stdout(conv) do
                        label = isone(length(spaxel)) ? "Voronoi bin $(spaxel[1])" : "Spaxel ($(spaxel[1]),$(spaxel[2]))"
                        println("$label on worker $(myid()):")
                        println(res)
                        println("-------------------------------------------------------")
                    end
                end
            end
        end

        _fit_global = true

    elseif bootstrap_iter
        p₁ = copy(p1_boots)
        deleteat!(p₁, tied_indices)
        p₁ = p₁[.~param_lock_tied]

    else
        p₁ = pfree_tied

    end
    
    @debug "Beginning Line fitting with Levenberg-Marquardt:"

    ############################################# FIT WITH LEVMAR ###################################################

    fit_func_2 = (x, p, n) -> model_line_residuals(x, p, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, 
        cube_fitter.flexible_wavesol, ext_curve_norm[1+n:end-n], lsf_interp_func, cube_fitter.relative_flags,
        nuc_temp_fit ? template_norm[1+n:end-n] : nothing, nuc_temp_fit)
    
    res = cmpfit(λnorm, Inorm, σnorm, (x, p) -> fit_step3(x, p, fit_func_2), p₁, parinfo=parinfo, config=config)
    if !_fit_global
        res = repeat_fit_jitter(λnorm, Inorm, σnorm, (x, p) -> fit_step3(x, p, fit_func_2), p₁, lbfree_tied, ubfree_tied,
            parinfo, config, res, "line", spaxel)
    end

    @debug "Line CMPFit status: $(res.status)"

    # Get the results and errors
    popt = rebuild_full_parameters(res.param, pfix_tied, param_lock_tied, tied_pairs, tied_indices, n_tied)

    # Dont both with uncertainties if bootstrapping
    perr = zeros(Float64, length(popt))
    if !bootstrap_iter
        perr = rebuild_full_parameters(res.perror, zeros(length(pfix_tied)), param_lock_tied, tied_pairs, tied_indices, n_tied)
    end

    ######################################################################################################################

    @debug "Best fit line parameters: \n $popt"
    @debug "Line parameter errors: \n $perr"
    # @debug "Line covariance matrix: \n $covar"

    # Final optimized fit
    I_model, comps = model_line_residuals(λ, popt, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, 
        cube_fitter.flexible_wavesol, ext_curve, lsf_interp_func, cube_fitter.relative_flags, template_norm, nuc_temp_fit, true)
    
    if init
        # Clean up the initial parameters to be more amenable to individual spaxel fits
        popt = clean_line_parameters(cube_fitter, popt, lower_bounds, upper_bounds)
        # Save results to file
        cube_fitter.p_init_line[:] .= copy(popt)
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_line.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_line, ',')
        end
    end

    # Log the fit results
    pretty_print_line_results(cube_fitter, popt, perr)

    popt, I_model, comps, n_free, perr
end


"""
    all_fit_spaxel(cube_fitter, spaxel, λ, I, σ, templates, mask_lines, mask_bad, I_spline, N, area_sr, lsf_interp_func;
        [init, use_ap, bootstrap_iter, p1_boots_cont, p1_boots_line])

Fit the continuum and emission lines in a spaxel simultaneously with a combination of the Simulated Annealing 
and Levenberg-Marquardt algorithms.

This procedure has been adapted from PAHFIT (with some heavy adjustments). 
See Smith, Draine, et al. 2007; http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::CartesianIndex`: The 2D index of the spaxel being fit.
- `λ::Vector{<:Real}`: The 1D wavelength vector 
- `I::Vector{<:Real}`: The 1D intensity vector
- `σ::Vector{<:Real}`: The 1D error vector
- `templates::Matrix{<:Real}`: The 2D templates array (1st dimension = wavelength, 2nd dimension = each template)
- `mask_lines::BitVector`: The mask giving the locations of emission Lines
- `mask_bad::BitVector`: The mask giving the locations of bad pixels
- `I_spline::Vector{<:Real}`: The cubic-spline fit to the continuum 
- `N::Real`: The normalization
- `area_sr::Vector{<:Real}`: The 1D solid angle vector
- `lsf_interp_func::Function`: Interpolating function for the line-spread function (LSF) in km/s as a function of wavelength
- `init::Bool=false`: Flag for the initial fit which fits the sum of all spaxels, to get an estimation for
the initial parameter vector for individual spaxel fits
- `use_ap::Bool=false`: Flag for fitting an integrated spectrum within an aperture
- `nuc_temp_fit::Bool=false`: Flag for fitting a nuclear template
- `bootstrap_iter::Bool=false`: Flag for fitting multiple iterations of the same spectrum with bootstrapping
- `p1_boots_cont::Union{Vector{<:Real},Nothing}=nothing`: If `bootstrap_iter` is true, this should give the best-fit parameter vector
for the initial non-bootstrapped fit of the continuum.
- `p1_boots_line::Union{Vector{<:Real},Nothing}=nothing`: Same as `p1_boots_cont`, but for the line parameters.
"""
function all_fit_spaxel(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real}, 
    σ::Vector{<:Real}, templates::Matrix{<:Real}, mask_bad::BitVector, I_spline::Vector{<:Real}, N::Real, area_sr::Vector{<:Real}, 
    lsf_interp_func::Function; init::Bool=false, use_ap::Bool=false, nuc_temp_fit::Bool=false, bootstrap_iter::Bool=false, 
    p1_boots_cont::Union{Vector{<:Real},Nothing}=nothing, p1_boots_line::Union{Vector{<:Real},Nothing}=nothing) 

    @assert !cube_fitter.use_pah_templates "The fit_joint and use_pah_templates options are mutually exclusive!"

    @debug """\n
    ################################################################
    ###   Beginning continuum & line fit for spaxel $spaxel...   ###
    ################################################################
    """

    # Normalize
    λ_spax = copy(λ)
    I_spax = I ./ N
    I_spline_spax = I_spline ./ N
    σ_spax = σ ./ N
    templates_spax = copy(templates)
    channel_masks = copy(cube_fitter.channel_masks)

    scale = 7
    λknots = λ[(1+scale):scale:(length(λ)-scale)]
    for s in axes(templates_spax, 2)
        m = .~isfinite.(templates_spax[:, s])
        templates_spax[m, s] .= Spline1D(λ[.~m], templates_spax[.~m], λknots, k=1, bc="extrapolate")(λ[m])
    end

    # Masked spectrum
    mask_vectors!(mask_bad, cube_fitter.user_mask, λ_spax, I_spax, σ_spax, templates_spax, channel_masks, I_spline_spax)

    @debug """\n
    ##################################################################################################################
    ########################################## FITTING THE CONTINUUM & LINES #########################################
    ##################################################################################################################
    """

    # Get the priors and "locked" booleans for each parameter, split up by the 2 steps for the continuum fit
    plims_cont, lock_cont, tied_pairs_cont, tied_indices_cont = get_continuum_plimits(cube_fitter, spaxel, λ_spax, I_spax, σ_spax, 
        init || use_ap, templates_spax./N)

    # Split up the initial parameter vector into the components that we need for each fitting step
    pars_0_cont, dstep_0_cont = get_continuum_initial_values(cube_fitter, spaxel, λ_spax, I_spax, σ_spax, N, init || use_ap,
        templates_spax./N)
    if bootstrap_iter
        pars_0_cont = p1_boots_cont
    end
    if nuc_temp_fit && cube_fitter.spectral_region == :OPT
        # for optical fits there should be no channel splits so just lock the PSF amplitude to 1 during the nuclear fit
        lock_cont[end-cube_fitter.n_templates+1:end] .= 1
        pars_0_cont[end-cube_fitter.n_templates+1:end] .= 1.0
    end

    @debug "Continuum Parameters:"
    s_params = split_parameters(pars_0_cont, dstep_0_cont, plims_cont, lock_cont, tied_indices_cont)
    pfree_cont_tied, pfix_cont_tied, dfree_cont_tied, lock_cont_tied, lbfree_cont_tied, ubfree_cont_tied, n_free_cont, n_tied_cont = s_params

    plims_lines, lock_lines, names_lines, tied_pairs_lines, tied_indices_lines = get_line_plimits(cube_fitter, init || use_ap, nuc_temp_fit)
    pars_0_lines, dstep_0_lines = get_line_initial_values(cube_fitter, spaxel, init || use_ap)
    if bootstrap_iter
        pars_0_lines = p1_boots_line
    end
    lower_bounds_lines = [pl[1] for pl in plims_lines]
    upper_bounds_lines = [pl[2] for pl in plims_lines]

    # Perform line component tests to determine which line components are actually necessary to include in the fit
    if (length(cube_fitter.line_test_lines) > 0) && !init
        perform_line_component_test!(cube_fitter, spaxel, pars_0_lines, lock_lines, lower_bounds_lines, upper_bounds_lines, 
            λ_spax, I_spax .- I_spline_spax, σ_spax, lsf_interp_func, bootstrap_iter=bootstrap_iter)
    end

    @debug "Line Parameters:"
    s_params = split_parameters(pars_0_lines, dstep_0_lines, plims_lines, lock_lines, tied_indices_lines; param_names=names_lines)
    pfree_lines_tied, pfix_lines_tied, dfree_lines_tied, lock_lines_tied, lbfree_lines_tied, ubfree_lines_tied, names_lines_tied, n_free_lines, n_tied_lines = s_params

    # Place initial values within the lower/upper limits (amplitudes may be too large if the extinction levels differ between spaxels)
    pfree_lines_tied = clamp.(pfree_lines_tied, lbfree_lines_tied, ubfree_lines_tied)

    # Pre-compute the stellar templates, if all the ages and metallicities are locked
    if cube_fitter.spectral_region == :OPT
        lock_ssps = true
        ages = []
        logzs = []
        pᵢ = 1
        for _ in 1:cube_fitter.n_ssps
            lock_ssps &= lock_cont[pᵢ+1] & lock_cont[pᵢ+2]
            push!(ages, pars_0_cont[pᵢ+1])
            push!(logzs, pars_0_cont[pᵢ+2])
            pᵢ += 3
        end
        if lock_ssps
            @debug "Pre-calculating SSPs since all ages and metallicities are locked."
            stellar_templates = zeros(length(cube_fitter.ssp_λ), cube_fitter.n_ssps)
            for i in 1:cube_fitter.n_ssps
                stellar_templates[:, i] .= [cube_fitter.ssp_templates[j](ages[i], logzs[i]) for j in eachindex(cube_fitter.ssp_λ)]
            end
        else
            @debug "SSPs will be interpolated during the fit since ages and/or metallicities are free."
            stellar_templates = cube_fitter.ssp_templates
        end
    end

    # Fitting functions for either optical or MIR continuum
    function fit_joint_mir(x, pfree_tied_all; n=0)
        out_type = eltype(pfree_tied_all)

        # Split into continuum and lines parameters
        pfree_cont_tied = pfree_tied_all[1:n_free_cont]
        pfree_lines_tied = pfree_tied_all[n_free_cont+1:end]

        # Organize parameters
        ptot_cont = rebuild_full_parameters(pfree_cont_tied, pfix_cont_tied, lock_cont_tied, tied_pairs_cont, tied_indices_cont, n_tied_cont)
        ptot_lines = rebuild_full_parameters(pfree_lines_tied, pfix_lines_tied, lock_lines_tied, tied_pairs_lines, tied_indices_lines, n_tied_lines)

        # Generate the extinction curve beforehand so it can be used for the lines
        pₑ = 3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law
        if cube_fitter.extinction_curve == "d+"
            ext_curve = τ_dp(λ, ptot_cont[pₑ+3])
            ext_curve = extinction.(ext_curve, ptot_cont[pₑ], screen=cube_fitter.extinction_screen)
            pₑ += 1
        elseif cube_fitter.extinction_curve == "kvt"
            ext_curve = τ_kvt(λ, ptot_cont[pₑ+3])
            ext_curve = extinction.(ext_curve, ptot_cont[pₑ], screen=cube_fitter.extinction_screen)
            pₑ += 1
        elseif cube_fitter.extinction_curve == "ct"
            ext_curve = τ_ct(λ)
            ext_curve = extinction.(ext_curve, ptot_cont[pₑ], screen=cube_fitter.extinction_screen)
            pₑ += 1
        elseif cube_fitter.extinction_curve == "ohm"
            ext_curve = τ_ohm(λ)
            ext_curve = extinction.(ext_curve, ptot_cont[pₑ], screen=cube_fitter.extinction_screen)
            pₑ += 1
        elseif cube_fitter.extinction_curve == "custom"
            ext_curve = cube_fitter.custom_ext_template(λ)
            ext_curve = extinction.(ext_curve, ptot_cont[pₑ], screen=cube_fitter.extinction_screen)
            pₑ += 1
        elseif cube_fitter.extinction_curve == "decompose"
            τ_oli = ptot_cont[pₑ] .* cube_fitter.κ_abs[1](λ)
            τ_pyr = ptot_cont[pₑ] .* ptot_cont[pₑ+1] .* cube_fitter.κ_abs[2](λ)
            τ_for = ptot_cont[pₑ] .* ptot_cont[pₑ+2] .* cube_fitter.κ_abs[3](λ)
            τ_97 = ptot_cont[pₑ] * cube_fitter.κ_abs[1](9.7) + ptot_cont[pₑ] * ptot_cont[pₑ+1] * cube_fitter.κ_abs[2](9.7) + ptot_cont[pₑ] * ptot_cont[pₑ+2] * cube_fitter.κ_abs[3](9.7)
            ext_curve = extinction.((1 .- ptot_cont[pₑ+5]) .* (τ_oli .+ τ_pyr .+ τ_for) .+ ptot_cont[pₑ+5] .* τ_97 .* (9.7./λ).^1.7, 1., screen=cube_fitter.extinction_screen)
            pₑ += 3
        else
            error("Unrecognized extinction curve: $(cube_fitter.extinction_curve)")
        end
        pₑ += 4 + 3cube_fitter.n_abs_feat + (cube_fitter.fit_sil_emission ? 6 : 0)

        template_norm = nothing
        if nuc_temp_fit
            template_norm = zeros(eltype(ptot_cont), length(x))
            for ch_mask in channel_masks
                template_norm[ch_mask] .= ptot_cont[pₑ] .* templates_spax[ch_mask, 1]
            end
        end

        # Generate the models
        Icont = model_continuum(x, ptot_cont, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.profiles,
            cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.κ_abs, cube_fitter.custom_ext_template,
            cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, false, templates_spax, channel_masks, nuc_temp_fit)
        Ilines = model_line_residuals(x, ptot_lines, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, cube_fitter.flexible_wavesol,
            ext_curve, lsf_interp_func, cube_fitter.relative_flags, template_norm, nuc_temp_fit)
        
        # Return the sum of the models
        Icont .+ Ilines
    end

    function fit_joint_opt(x, pfree_tied_all; n=0)
        out_type = eltype(pfree_tied_all)

        # Split into continuum and lines parameters
        pfree_cont_tied = pfree_tied_all[1:n_free_cont]
        pfree_lines_tied = pfree_tied_all[n_free_cont+1:end]

        # Organize parameters
        ptot_cont = rebuild_full_parameters(pfree_cont_tied, pfix_cont_tied, lock_cont_tied, tied_pairs_cont, tied_indices_cont, n_tied_cont)
        ptot_lines = rebuild_full_parameters(pfree_lines_tied, pfix_lines_tied, lock_lines_tied, tied_pairs_lines, tied_indices_lines, n_tied_lines)

        # Generate the attenuation curve beforehand so it can be used for the lines
        pₑ = 1 + 3cube_fitter.n_ssps + 2
        E_BV = ptot_cont[pₑ]
        E_BV_factor = ptot_cont[pₑ+1]
        δ_UV = nothing
        Cf_dust = 0.
        pₑ += 2
        if cube_fitter.fit_uv_bump && cube_fitter.extinction_curve == "calzetti"
            δ_UV = ptot_cont[pₑ]
            pₑ += 1
        end
        if cube_fitter.fit_covering_frac && cube_fitter.extinction_curve == "calzetti"
            Cf_dust = ptot_cont[pₑ]
            pₑ += 1
        end
        # E(B-V)_stars = 0.44E(B-V)_gas
        if cube_fitter.extinction_curve == "ccm"
            ext_curve_gas = attenuation_cardelli(x, E_BV)
        elseif cube_fitter.extinction_curve == "calzetti"
            if isnothing(δ_UV)
                ext_curve_gas = attenuation_calzetti(x, E_BV, Cf=Cf_dust)
            else
                ext_curve_gas = attenuation_calzetti(x, E_BV, δ_UV, Cf=Cf_dust)
            end
        else
            error("Unrecognized extinction curve $(cube_fitter.extinction_curve)")
        end

        template_norm = nothing
        if nuc_temp_fit
            template_norm = ptot_cont[pₑ] .* templates_spax[:, 1]
        end
        
        # Generate the models
        Icont = model_continuum(x, ptot_cont, N, cube_fitter.vres, cube_fitter.vsyst_ssp, cube_fitter.vsyst_feii, cube_fitter.npad_feii,
            cube_fitter.n_ssps, cube_fitter.ssp_λ, stellar_templates, cube_fitter.feii_templates_fft, cube_fitter.n_power_law, cube_fitter.fit_uv_bump, 
            cube_fitter.fit_covering_frac, cube_fitter.fit_opt_na_feii, cube_fitter.fit_opt_br_feii, cube_fitter.extinction_curve,
            templates_spax, cube_fitter.fit_temp_multexp, nuc_temp_fit)
        Ilines = model_line_residuals(x, ptot_lines, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, cube_fitter.flexible_wavesol,
            ext_curve_gas, lsf_interp_func, cube_fitter.relative_flags, template_norm, nuc_temp_fit)

        # Return the sum of the models
        Icont .+ Ilines
    end
    fit_joint = cube_fitter.spectral_region == :MIR ? fit_joint_mir : fit_joint_opt
       
    # Combine parameters
    p₀ = [pfree_cont_tied; pfree_lines_tied]
    lower_bounds = [lbfree_cont_tied; lbfree_lines_tied]
    upper_bounds = [ubfree_cont_tied; ubfree_lines_tied]
    dstep = [dfree_cont_tied; dfree_lines_tied]

    _fit_global = false
    if (init || use_ap || cube_fitter.fit_all_global) && !bootstrap_iter
    # if false
        @debug "Beginning joint continuum+line fitting with Simulated Annealing:"

        fit_func = p -> -ln_likelihood(I_spax, fit_joint(λ_spax, p, n=0), σ_spax)
        x_tol = 1e-5
        f_tol = abs(fit_func(p₀) - fit_func(clamp.(p₀ .* (1 .- x_tol), lower_bounds, upper_bounds)))
        f_tol = 10^floor(log10(f_tol))
        lb_global = lower_bounds
        ub_global = [isfinite(ub) ? ub : 1e10 for ub in upper_bounds]

        # First, perform a bounded Simulated Annealing search for the optimal parameters with a large population (10n)
        res = Optim.optimize(fit_func, lb_global, ub_global, p₀, 
            SAMIN(;rt=0.9, nt=5, ns=5, neps=5, f_tol=f_tol, x_tol=x_tol, verbosity=0), Optim.Options(iterations=10^6))
        p₁ = res.minimizer

        # Write convergence results to file, if specified
        if cube_fitter.track_convergence
            global file_lock
            # use the ReentrantLock to prevent multiple processes from trying to write to the same file at once
            lock(file_lock) do 
                open(joinpath("output_$(cube_fitter.name)", "loki.convergence.log"), "a") do conv
                    redirect_stdout(conv) do
                        label = isone(length(spaxel)) ? "Voronoi bin $(spaxel[1])" : "Spaxel ($(spaxel[1]),$(spaxel[2]))"
                        println("$label on worker $(myid()):")
                        println(res)
                        println("-------------------------------------------------------")
                    end
                end
            end
        end

        _fit_global = true

    else
        p₁ = p₀

    end

    # Parinfo and config objects
    parinfo, config = get_continuum_parinfo(n_free_cont+n_free_lines, lower_bounds, upper_bounds, dstep)

    res = cmpfit(λ_spax, I_spax, σ_spax, fit_joint, p₁, parinfo=parinfo, config=config)
    if !_fit_global
        res = repeat_fit_jitter(λ_spax, I_spax, σ_spax, fit_joint, p₁, lower_bounds, upper_bounds, parinfo, config, res,
            "continuum+lines", spaxel; check_cont_snr=true)
    end

    @debug "Continuum+Lines CMPFit Status: $(res.status)"

    # Split back up into continuum and lines parameters
    popt_cont = rebuild_full_parameters(res.param[1:n_free_cont], pfix_cont_tied, lock_cont_tied, tied_pairs_cont, tied_indices_cont, n_tied_cont)
    perr_cont = zeros(Float64, length(popt_cont))
    if !bootstrap_iter
        perr_cont = rebuild_full_parameters(res.perror[1:n_free_cont], zeros(length(pfix_cont_tied)), lock_cont_tied, tied_pairs_cont, 
            tied_indices_cont, n_tied_cont)
    end

    popt_lines = rebuild_full_parameters(res.param[n_free_cont+1:end], pfix_lines_tied, lock_lines_tied, tied_pairs_lines, tied_indices_lines, n_tied_lines)
    perr_lines = zeros(Float64, length(popt_lines))
    if !bootstrap_iter
        perr_lines = rebuild_full_parameters(res.perror[n_free_cont+1:end], zeros(length(pfix_lines_tied)), lock_lines_tied, tied_pairs_lines, 
            tied_indices_lines, n_tied_lines)
    end

    @debug "Best fit continuum parameters: \n $popt_cont"
    @debug "Continuum parameter errors: \n $perr_cont"
    @debug "Best fit line parameters: \n $popt_lines"
    @debug "Line parameter errors: \n $perr_lines"

    # Create the full model
    if cube_fitter.spectral_region == :MIR
        Icont, comps_cont = model_continuum(λ, popt_cont, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.profiles,
            cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.κ_abs, cube_fitter.custom_ext_template, 
            cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, false, templates, cube_fitter.channel_masks, nuc_temp_fit, true)
        ext_key = "extinction"
    else
        Icont, comps_cont = model_continuum(λ, popt_cont, N, cube_fitter.vres, cube_fitter.vsyst_ssp, cube_fitter.vsyst_feii, cube_fitter.npad_feii,
            cube_fitter.n_ssps, cube_fitter.ssp_λ, stellar_templates, cube_fitter.feii_templates_fft, cube_fitter.n_power_law, cube_fitter.fit_uv_bump, 
            cube_fitter.fit_covering_frac, cube_fitter.fit_opt_na_feii, cube_fitter.fit_opt_br_feii, cube_fitter.extinction_curve, templates,
            cube_fitter.fit_temp_multexp, nuc_temp_fit, true)
        ext_key = "attenuation_gas"
    end
    Ilines, comps_lines = model_line_residuals(λ, popt_lines, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, cube_fitter.flexible_wavesol,
        comps_cont[ext_key], lsf_interp_func, cube_fitter.relative_flags, nuc_temp_fit ? comps_cont["templates_1"] : nothing, nuc_temp_fit, true)

    # Estimate PAH template amplitude
    if cube_fitter.spectral_region == :MIR
        pahtemp = zeros(length(λ))
        for i in 1:cube_fitter.n_dust_feat
            pahtemp .+= comps_cont["dust_feat_$i"]
        end
        pah_amp = maximum(pahtemp)/2
    else
        pah_amp = zeros(2)
    end

    if init
        cube_fitter.p_init_cont[:] .= popt_cont
        cube_fitter.p_init_line[:] .= popt_lines
        # Save the results to a file 
        # save running best fit parameters in case the fitting is interrupted
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_cont.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_cont, ',')
        end
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_line.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_line, ',')
        end
        cube_fitter.p_init_pahtemp[:] .= pah_amp
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_pahtemp.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_pahtemp, ',')
        end
    end

    # Print the results (to the logger)
    pretty_print_continuum_results(cube_fitter, popt_cont, perr_cont, I_spax)
    pretty_print_line_results(cube_fitter, popt_lines, perr_lines)

    popt_cont, popt_lines, Icont, Ilines, comps_cont, comps_lines, n_free_cont, n_free_lines, perr_cont, perr_lines, pah_amp
end


"""
    plot_spaxel_fit(spectral_region, λ_um, I, I_model, σ, mask_bad, mask_lines, comps, n_dust_cont, n_power_law,
        n_dust_features, n_abs_features, n_ssps, n_comps, line_wave_um, line_names, line_annotate, line_latex, 
        screen, z, χ2red, name, label; [backend, I_boot_min, I_boot_max, range_um, spline])

Plot the best fit for an individual spaxel using the given backend (`:pyplot` or `:plotly`).

# Arguments {T<:Real}
- `spectral_region::Symbol`: Either :MIR for mid-infrared or :OPT for optical
- `λ_um::Vector{<:Real}`: The wavelength vector of the spaxel to be plotted, in microns
- `I::Vector{<:Real}`: The intensity data vector of the spaxel to be plotted
- `I_model::Vector{<:Real}`: The intensity model vector of the spaxel to be plotted
- `σ::Vector{<:Real}`: The uncertainty vector of the spaxel to be plotted
- `mask_bad::BitVector`: The mask giving the locations of bad pixels
- `mask_lines::BitVector`: The mask giving the locations of emission lines
- `user_mask::Vector{<:Tuple}`: The mask defined by the user
- `comps::Dict{String, Vector{T}}`: The dictionary of individual components of the model intensity
- `n_dust_cont::Integer`: The number of dust continuum components in the fit
- `n_power_law::Integer`: The number of power law continuum components in the fit
- `n_dust_features::Integer`: The number of PAH features in the fit
- `n_abs_features::Integer`: The number of absorption features in the fit
- `n_templates::Integer`: The number of generic templates in the fit
- `n_ssps::Integer`: The number of simple stellar populations in the fit
- `nuc_temp_fit::Bool`: Whether or not one is fitting the nuclear template
- `n_comps::Integer`: The maximum number of line profiles per line
- `line_wave_um::Vector{<:Real}`: List of nominal central wavelengths for each line in the fit, in microns
- `line_names::Vector{Symbol}`: List of names for each line in the fit
- `line_annotate::BitVector`: List of booleans determining whether or not to add an annotation for the given line
- `line_latex::Vector{String}`: List of LaTeX-formatted line names to be used for the annotations
- `screen::Bool`: The type of model used for extinction screening
- `Cf::Real`: The dust covering fraction parameter
- `z::Real`: The redshift of the object being fit
- `χ2red::Real`: The reduced χ^2 value of the fit
- `name::String`: The name of the object being fit
- `label::String`: A label for the individual spaxel being plotted, to be put in the file name
- `backend::Symbol=:pyplot`: The backend to use to plot, may be `:pyplot`, `:plotly`, or `:both`
- `I_boot_min::Union{Vector{<:Real},Nothing}=nothing`: Optional vector giving the minimum model out of all bootstrap iterations
- `I_boot_max::Union{Vector{<:Real},Nothing}=nothing`: Optional vector giving the maximum model out of all bootstrap iterations
- `range_um::Union{Tuple,Nothing}=nothing`: Optional tuple specifying min/max wavelength values to truncate the x-axis of the plot to
- `spline::Union{Vector{<:Real},Nothing}=nothing`: Optional vector giving the cubic spline interpolation of the continuum to plot
"""
function plot_spaxel_fit(spectral_region::Symbol, λ_um::Vector{<:Real}, I::Vector{<:Real}, I_model::Vector{<:Real}, σ::Vector{<:Real}, mask_bad::BitVector, 
    mask_lines::BitVector, user_mask::Vector{<:Tuple}, comps::Dict{String, Vector{T}}, n_dust_cont::Integer, n_power_law::Integer, n_dust_features::Integer, 
    n_abs_features::Integer, n_templates::Integer, n_ssps::Integer, nuc_temp_fit::Bool, n_comps::Integer, line_wave_um::Vector{<:Real}, line_names::Vector{Symbol}, 
    line_annotate::BitVector, line_latex::Vector{String}, screen::Bool, Cf::Real, z::Real, χ2red::Real, name::String, label::String; backend::Symbol=:pyplot, 
    I_boot_min::Union{Vector{<:Real},Nothing}=nothing, I_boot_max::Union{Vector{<:Real},Nothing}=nothing, 
    range_um::Union{Tuple,Nothing}=nothing, spline::Union{Vector{<:Real},Nothing}=nothing) where {T<:Real}

    range = nothing
    split_ext = false
    if spectral_region == :MIR
        fit_sil_emission = haskey(comps, "hot_dust")
        fit_opt_na_feii = fit_opt_br_feii = false
        abs_feat = n_abs_features ≥ 1 ? reduce(.*, [comps["abs_feat_$i"] for i ∈ 1:n_abs_features]) : ones(length(λ_um))
        abs_full = comps["abs_ice"] .* comps["abs_ch"] .* abs_feat
        ext_full = abs_full .* comps["extinction"]
        split_ext = haskey(comps, "abs_oli")
        # Plot in microns for MIR data
        λ = λ_um
        line_wave = line_wave_um
        if !isnothing(range_um)
            range = range_um
        end
    else
        fit_sil_emission = false
        fit_opt_na_feii = haskey(comps, "na_feii")
        fit_opt_br_feii = haskey(comps, "br_feii")
        abs_feat = ones(length(λ_um))
        abs_full = ones(length(λ_um))
        att_stars = comps["attenuation_stars"]
        att_gas = comps["attenuation_gas"]
        # Plot in angstroms for optical data
        λ = λ_um .* 1e4
        line_wave = line_wave_um .* 1e4
        if !isnothing(range_um)
            range = range_um .* 1e4
        end
    end
    nuc_temp_norm = ones(length(λ))
    if nuc_temp_fit
        nuc_temp_norm = comps["templates_1"]
    end

    # Plotly ---> useful interactive plots for visually inspecting data, but not publication-quality
    if (backend == :plotly || backend == :both) && isnothing(range)
        # Plot the overall data / model
        trace1 = PlotlyJS.scatter(x=λ, y=I, mode="lines", line=Dict(:color => "black", :width => 1), name="Data", showlegend=true)
        trace2 = PlotlyJS.scatter(x=λ, y=I_model, mode="lines", line=Dict(:color => "red", :width => 1), name="Model", showlegend=true)
        traces = [trace1, trace2]
        # Loop over and plot individual model components
        for comp ∈ keys(comps)
            if (comp == "extinction") || (comp == "attenuation_stars")
                append!(traces, [PlotlyJS.scatter(x=λ, y=(spectral_region == :MIR ? ext_full : att_gas ./ median(att_gas)) .* maximum(I_model) .* 1.1, 
                    mode="lines", line=Dict(:color => "black", :width => 1, :dash => "dash"), name="Extinction")])
            elseif comp == "abs_oli"
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* maximum(I_model) .* 1.1, mode="lines", line=Dict(:color => "blue", :width => 1, :dash => "dash"),
                    name="Olivine Absorption")])
            elseif comp == "abs_pyr"
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* maximum(I_model) .* 1.1, mode="lines", line=Dict(:color => "red", :width => 1, :dash => "dash"),
                    name="Pyroxene Absorption")])
            elseif comp == "abs_for"
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* maximum(I_model) .* 1.1, mode="lines", line=Dict(:color => "orange", :width => 1, :dash => "dash"),
                    name="Forsterite Absorption")])
            elseif occursin("hot_dust", comp)
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* abs_full .* nuc_temp_norm, mode="lines", line=Dict(:color => "yellow", :width => 1),
                    name="Hot Dust")])
            elseif occursin("unobscured_continuum", comp)
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* nuc_temp_norm, mode="lines", line=Dict(:color => "black", :width => 0.5), 
                    name="Unobscured Continuum")])
            elseif occursin("na_feii", comp)
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* att_gas .* nuc_temp_norm, mode="lines", line=Dict(:color => "yellow", :width => 1),
                    name="Narrow Fe II")])
            elseif occursin("br_feii", comp)
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* att_gas .* nuc_temp_norm, mode="lines", line=Dict(:color => "yellow", :width => 2),
                    name="Broad Fe II")])
            elseif occursin("line", comp)
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* (spectral_region == :MIR ? comps["extinction"] : att_gas) .* nuc_temp_norm, 
                    mode="lines", line=Dict(:color => "rebeccapurple", :width => 1), name="Lines")])
            end
        end
        # Add vertical dashed lines for emission line rest wavelengths
        for (lw, ln) ∈ zip(line_wave, line_names)
            append!(traces, [PlotlyJS.scatter(x=[lw, lw], y=[0., nanmaximum(I)*1.1], mode="lines", 
                line=Dict(:color => occursin("H2", String(ln)) ? "red" : 
                              (any(occursin.(["alpha", "beta", "gamma", "delta"], String(ln))) ? "#ff7f0e" : "rebeccapurple"), 
                          :width => 0.5, :dash => "dash"))])
        end
        if spectral_region == :MIR
            # Add the summed up continuum
            append!(traces, [PlotlyJS.scatter(x=λ, y=(abs_full .* (fit_sil_emission ? comps["hot_dust"] : zeros(length(λ))) .+ comps["unobscured_continuum"] .+
                comps["obscured_continuum"] .+ ((n_templates > 0) && !nuc_temp_fit ? sum([comps["templates_$k"] for k ∈ 1:n_templates], dims=1)[1] : zeros(length(λ)))) .*
                nuc_temp_norm, mode="lines", line=Dict(:color => "green", :width => 1), name="Total Continuum")])
            # Summed up PAH features
            append!(traces, [PlotlyJS.scatter(x=λ, y=sum([comps["dust_feat_$i"] for i ∈ 1:n_dust_features], dims=1)[1] .* comps["extinction"] .* 
                nuc_temp_norm, mode="lines", line=Dict(:color => "blue", :width => 1), name="PAHs")])
            # Individual PAH features
            for i in 1:n_dust_features
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps["dust_feat_$i"] .* comps["extinction"] .* nuc_temp_norm, 
                    mode="lines", line=Dict(:color => "blue", :width => 1), name="PAHs")])
            end
            # Individual templates
            if !nuc_temp_fit
                for j in 1:n_templates
                    append!(traces, [PlotlyJS.scatter(x=λ, y=comps["templates_$j"], mode="lines", line=Dict(:color => "green", :width => 1), name="Template $j")])
                end
            end
        else
            # Add the summed up continuum
            append!(traces, [PlotlyJS.scatter(x=λ, y=nuc_temp_norm .* (att_stars .* sum([comps["SSP_$i"] for i ∈ 1:n_ssps], dims=1)[1] .+
                (fit_opt_na_feii ? comps["na_feii"] .* att_gas : zeros(length(λ))) .+ (fit_opt_br_feii ? comps["br_feii"] .* att_gas : zeros(length(λ))) .+
                (n_power_law > 0 ? sum([comps["power_law_$j"] for j in 1:n_power_law], dims=1)[1] : zeros(length(λ)))), mode="lines",
                line=Dict(:color => "green", :width => 1), name="Continuum")])
            for i in 1:n_ssps
                append!(traces, [PlotlyJS.scatter(x=λ, y=att_stars .* comps["SSP_$i"] .* nuc_temp_norm, mode="lines", line=Dict(:color => "green", :width => 0.5))])
            end
            # Individual templates
            if !nuc_temp_fit
                for j in 1:n_templates
                    append!(traces, [PlotlyJS.scatter(x=λ, y=comps["templates_$j"], mode="lines", line=Dict(:color => "green", :width => 1), name="Template $j")])
                end
            end
        end

        if !isnothing(spline)
            append!(traces, [PlotlyJS.scatter(x=λ, y=spline, mode="lines", line=Dict(:color => "red", :width => 1, :dash => "dash"), name="Cubic Spline")])
        end
        # axes labels / titles / fonts
        layout = PlotlyJS.Layout(
            xaxis_title=spectral_region == :MIR ? L"$\lambda\ (\mu{\rm m})$" : L"$\lambda\ (\mathring{A})$",
            yaxis_title=spectral_region == :MIR ? L"$I_{\nu}\ ({\rm MJy}\,{\rm sr}^{-1})$" : 
                                                  L"I_{\lambda}\ ({\rm erg}\,{\rm s}^{-1}\,{\rm cm}^{-2}\,{\mathring{A}}^{-1}\,{\rm sr}^{-1})",
            title=L"$\tilde{\chi}^2 = %$χ2red$",
            xaxis_constrain="domain",
            font_family="Georgia, Times New Roman, Serif",
            template="plotly_white",
            # annotations=[
            #     attr(x=lw, y=nanmaximum(I)*.75, text=ll) for (lw, ll) ∈ zip(line_wave, line_latex)
            # ]
        )
        # save as HTML file
        p = PlotlyJS.plot(traces, layout)
        PlotlyJS.savefig(p, isnothing(label) ? joinpath("output_$name", "spaxel_plots", "levmar_fit_spaxel.html") : 
            joinpath("output_$name", "spaxel_plots", "$label.html"))

    end

    # Pyplot --> actually publication-quality plots finely tuned to be the most useful and visually appealing that I could make them
    if backend == :pyplot || backend == :both

        # If max is above 10^4, normalize so the y axis labels aren't super wide
        factor = spectral_region == :MIR ? 1 ./ λ : ones(length(λ))
        power = floor(Int, log10(maximum(I .* factor)))
        if (power ≥ 4) || (power ≤ -4)
            norm = 10.0^power
        else
            norm = 1.0
        end

        min_inten = ((sum((I ./ norm .* factor) .< -0.01) > (length(λ)/10)) && (spectral_region == :OPT)) ? -2nanstd(I ./ norm .* factor) : -0.01
        max_inten = isnothing(range) ? 
                    1.3nanmaximum(I[.~mask_lines .& .~mask_bad] ./ norm .* factor[.~mask_lines .& .~mask_bad]) : 
                    1.1nanmaximum((I ./ norm .* factor)[range[1] .< λ .< range[2]])
        
        if (max_inten < 1) && (norm ≠ 1.0)
            norm /= 10
            max_inten *= 10
            power -= 1
        end

        # make color schemes: https://paletton.com/#uid=3000z0kDlkVsFuswHp7LfgmSRaH
        # https://paletton.com/#uid=73u1F0k++++qKZWAF+V+VAEZXqK

        # https://stats.stackexchange.com/questions/118033/best-series-of-colors-to-use-for-differentiating-series-in-publication-quality 
        # XKCD colors: https://xkcd.com/color/rgb/

        # Set up subplots with gridspec
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(nrows=4, ncols=1, hspace=0.)

        # ax1 is the main plot
        ax1 = fig.add_subplot(py"$(gs)[:-1, :]")
        # ax2 is the residuals plot
        ax2 = fig.add_subplot(py"$(gs)[-1, :]")

        ax1.plot(λ, I ./ norm .* factor, "k-", label="Data")

        # plot cubic spline
        if !isnothing(spline)
            ax1.plot(λ, spline ./ norm .* factor, color="#2ca02c", linestyle="--", label="Cubic Spline")
        end

        ax1.plot(λ, I_model ./ norm .* factor, "-", color="#ff5d00", label="Model")
        if !isnothing(I_boot_min) && !isnothing(I_boot_max)
            ax1.fill_between(λ, I_boot_min ./ norm .* factor, I_boot_max ./ norm .* factor, color="#ff5d00", alpha=0.5, zorder=10)
        end

        ax2.plot(λ, (I.-I_model) ./ norm .* factor, "k-")

        χ2_str = @sprintf "%.3f" χ2red
        ax2.plot(λ, zeros(length(λ)), "-", color="#ff5d00", label=L"$\tilde{\chi}^2 = %$χ2_str$")
        if !isnothing(I_boot_min) && !isnothing(I_boot_max)
            ax2.fill_between(λ, (I_boot_min .- I_model) ./ norm .* factor, (I_boot_max .- I_model) ./ norm .* factor, color="#ff5d00", alpha=0.5,
                zorder=10)
        end
        # ax2.fill_between(λ, (I.-I_cont.+σ)./norm./λ, (I.-I_cont.-σ)./norm./λ, color="k", alpha=0.5)

        # twin axes with different labels --> extinction for ax3 and observed wavelength for ax4
        ax3 = ax1.twinx()
        # ax4 = ax1.twiny()

        # full continuum
        if spectral_region == :MIR
            ax1.plot(λ, (((n_templates > 0) && !nuc_temp_fit ? sum([comps["templates_$k"] for k ∈ 1:n_templates], dims=1)[1] : zeros(length(λ))) .+
                abs_full .* (fit_sil_emission ? comps["hot_dust"] : zeros(length(λ))) .+
                comps["obscured_continuum"] .+ comps["unobscured_continuum"]) .* nuc_temp_norm ./ norm .* factor, "k-", lw=2, alpha=0.5, label="Continuum")
            # individual continuum components
            ax1.plot(λ, comps["stellar"] .* (Cf .* ext_full .+ (1 .- Cf)) .* nuc_temp_norm ./ norm .* factor, "m--", alpha=0.5, label="Stellar continuum")
            for i in 1:n_dust_cont
                ax1.plot(λ, comps["dust_cont_$i"] .* (Cf .* ext_full .+ (1 .- Cf)) .* nuc_temp_norm ./ norm .* factor, "k-", alpha=0.5, label="Dust continuum")
            end
            for i in 1:n_power_law
                ax1.plot(λ, comps["power_law_$i"] .* (Cf .* ext_full .+ (1 .- Cf)) .* nuc_temp_norm ./ norm .* factor, "k-", alpha=0.5, label="Power Law")
            end
            # ax1.plot(λ, comps["obscured_continuum"] ./ norm .* factor, "-", color="#0b450a", alpha=0.8, label="Obscured continuum")
            # ax1.plot(λ, comps["unobscured_continuum"] ./ norm .* factor, "--", color="#0b450a", alpha=0.8, label="Unobscured continuum")
            # full PAH profile
            ax1.plot(λ, sum([comps["dust_feat_$i"] for i ∈ 1:n_dust_features], dims=1)[1] .* comps["extinction"] .* nuc_temp_norm ./ norm .* factor, "-", 
                color="#0065ff", label="PAHs")
            # plot hot dust
            if haskey(comps, "hot_dust")
                ax1.plot(λ, comps["hot_dust"] .* abs_full .* nuc_temp_norm ./ norm .* factor, "-", color="#8ac800", alpha=0.8, label="Hot Dust")
            end
            # templates
            if !nuc_temp_fit
                for k ∈ 1:n_templates
                    ax1.plot(λ, comps["templates_$k"] ./ norm .* factor, "-", color="#50630d", label="Template $k")
                end
            end
        else
            ax1.plot(λ, (((n_templates > 0) && !nuc_temp_fit ? sum([comps["templates_$k"] for k ∈ 1:n_templates], dims=1)[1] : zeros(length(λ))) .+
                att_stars .* sum([comps["SSP_$i"] for i ∈ 1:n_ssps], dims=1)[1] .+
                (fit_opt_na_feii ? comps["na_feii"] .* att_gas : zeros(length(λ))) .+ 
                (fit_opt_br_feii ? comps["br_feii"] .* att_gas : zeros(length(λ))) .+
                (n_power_law > 0 ? sum([comps["power_law_$j"] for j in 1:n_power_law], dims=1)[1] : zeros(length(λ)))) ./ 
                norm .* nuc_temp_norm .* factor, "k-", lw=2, alpha=0.5, label="Continuum")
            # individual continuum components
            for i in 1:n_ssps
                ax1.plot(λ, comps["SSP_$i"] .* att_stars ./ norm .* nuc_temp_norm .* factor, "g-", alpha=0.75, label="SSPs")
            end
            for i in 1:n_power_law
                ax1.plot(λ, comps["power_law_$i"] ./ norm .* nuc_temp_norm .* factor, "k-", alpha=0.5, label="Power Law")
            end
            if haskey(comps, "na_feii")
                ax1.plot(λ, comps["na_feii"] .* att_gas ./ norm .* nuc_temp_norm .* factor, "-", color="goldenrod", alpha=0.8, label="Narrow Fe II")
            end
            if haskey(comps, "br_feii")
                ax1.plot(λ, comps["br_feii"] .* att_gas ./ norm .* nuc_temp_norm .* factor, "--", color="goldenrod", alpha=0.8, label="Broad Fe II")
            end
        end
        # full line profile
        if isnothing(range)
            ax1.plot(λ, sum([haskey(comps, "line_$(i)_$(j)") ? comps["line_$(i)_$(j)"] : zeros(length(λ)) 
                for i ∈ 1:length(line_wave), j ∈ 1:n_comps], dims=(1,2))[1] .* 
                (spectral_region == :MIR ? comps["extinction"] : att_gas) .* nuc_temp_norm ./ norm .* factor, "-", 
                color="rebeccapurple", alpha=0.6, label="Lines")
        else
            for i ∈ 1:length(line_wave), j ∈ 1:n_comps
                if haskey(comps, "line_$(i)_$(j)")
                    ax1.plot(λ, comps["line_$(i)_$(j)"] .* 
                        (spectral_region == :MIR ? comps["extinction"] : att_gas) .* nuc_temp_norm ./ norm .* factor, "-", color="rebeccapurple",
                        alpha=0.6, label="Lines")
                end
            end
        end
        # plot extinction
        if spectral_region == :MIR && split_ext
            ax3.plot(λ, comps["abs_oli"], "k", linestyle=(0, (3, 1, 1, 1, 1, 1)), alpha=0.5, label="Olivine Absorption")
            ax3.plot(λ, comps["abs_pyr"], "k", linestyle="dashdot", alpha=0.5, label="Pyroxene Absorption")
            ax3.plot(λ, comps["abs_for"], "k", linestyle="dashed", alpha=0.5, label="Forsterite Absorption")
            ax3.plot(λ, comps["extinction"], "k", linestyle="dotted", label="Full Extinction")
        else
            ax3.plot(λ, spectral_region == :MIR ? ext_full : att_gas, "k:", alpha=0.5, label="Extinction")
        end
        if nuc_temp_fit
            ax3.plot(λ, nuc_temp_norm, "k", linestyle="--", alpha=0.5, label="PSF")
        end

        # plot vertical dashed lines for emission line wavelengths
        for (lw, ln) ∈ zip(line_wave, line_names)
            ax1.axvline(lw, linestyle="--", color="k", lw=0.5, alpha=0.5)
            ax2.axvline(lw, linestyle="--", color="k", lw=0.5, alpha=0.5)
        end

        # Shade in masked regions
        user_mask_bits = falses(length(λ))
        for region in user_mask
            user_mask_bits .|= region[1] .< λ .< region[2]
        end
        total_mask = mask_bad .| user_mask_bits
        l_edges = findall(diff(total_mask) .== 1) .+ 1
        r_edges = findall(diff(total_mask) .== -1)
        # Edge cases
        if total_mask[1] == 1
            l_edges = [1; l_edges]
        end
        if total_mask[end] == 1
            r_edges = [r_edges; length(λ)]
        end
        for (le, re) in zip(l_edges, r_edges)
            ax1.axvspan(λ[le], λ[re], alpha=0.5, color="k")
            ax2.axvspan(λ[le], λ[re], alpha=0.5, color="k")
        end

        # mark channel boundaries 
        if spectral_region == :MIR
            ax1.plot(channel_boundaries ./ (1 .+ z), ones(length(channel_boundaries)) .* max_inten .* 0.99, "v", 
                color="#0065ff", markersize=3.0)
        end

        # set axes limits and labels
        if isnothing(range)
            λmin, λmax = minimum(λ), maximum(λ)
            ax1.set_xlim(λmin, λmax)
            ax2.set_xlim(λmin, λmax)
            # ax4.set_xlim(λmin * (1 + z), λmax * (1 + z))
            ax1.set_ylim(min_inten, max_inten)
        else
            ax1.set_xlim(range[1], range[2])
            ax2.set_xlim(range[1], range[2])
            # ax4.set_xlim(range[1] * (1 + z), range[2] * (1 + z))
            ax1.set_ylim(min_inten, max_inten)
        end
        ax2.set_ylim(-1.1maximum(((I.-I_model) ./ norm .* factor)[.~mask_lines .& .~mask_bad]), 1.1maximum(((I.-I_model) ./ norm .* factor)[.~mask_lines .& .~mask_bad]))
        ax3.set_yscale("log") # logarithmic extinction axis
        ax3.set_ylim(spectral_region == :MIR ? 1e-3 : 1e-5, 1.)
        if spectral_region == :MIR
            if screen
                ax3.set_ylabel(L"$e^{-\tau_{\lambda}}$")
            else
                ax3.set_ylabel(L"$(1-e^{-\tau_{\lambda}}) / \tau_{\lambda}$")
            end
        else
            ax3.set_ylabel(L"$10^{-0.4E(B-V)_{\rm gas}k'(\lambda)}$")
        end
        if (power ≥ 4) || (power ≤ -4)
            prefix = L"$10^{%$power}$ "
        else
            prefix = ""
        end
        if spectral_region == :MIR
            ax1.set_ylabel(L"$I_{\nu}/\lambda$ (%$(prefix)MJy sr$^{-1}$ $\mu$m$^{-1}$)")
        else
            ax1.set_ylabel(L"$I_{\lambda}$ (%$(prefix)erg s$^{-1}$ cm$^{-2}$ ${\rm \mathring{A}}^{-1}$ sr$^{-1}$)")
        end
        ax2.set_ylabel(L"$O-C$")  # ---> residuals, (O)bserved - (C)alculated
        if spectral_region == :MIR
            ax2.set_xlabel(L"$\lambda_{\rm rest}$ ($\mu$m)")
        else
            ax2.set_xlabel(L"$\lambda_{\rm rest}$ (${\rm \mathring{A}}$)")
        end
        # ax4.set_xlabel(L"$\lambda_{\rm obs}$ ($\mu$m)")
        ax2.legend(loc="upper left")

        # Set minor ticks as multiples of 0.1 μm for x axis and automatic for y axis
        ax1.xaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax1.yaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax2.xaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax2.yaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        # ax4.xaxis.set_minor_locator(py_ticker.AutoMinorLocator())

        # Set major ticks and formats
        ax1.set_xticklabels([]) # ---> will be covered up by the residuals plot
        ax2.set_yticks([-round(maximum(((I.-I_model) ./ norm .* factor)[.~mask_lines .& .~mask_bad]) / 2, sigdigits=1), 0.0, 
                        round(maximum(((I.-I_model) ./ norm .* factor)[.~mask_lines .& .~mask_bad]) / 2, sigdigits=1)])
        # ax1.tick_params(which="both", axis="both", direction="in")
        ax1.tick_params(which="both", axis="both", direction="in", top=true)
        ax2.tick_params(which="both", axis="both", direction="in", labelright=true, right=true, top=true)
        ax3.tick_params(which="both", axis="both", direction="in")
        # ax4.tick_params(which="both", axis="both", direction="in")

        # Annotate emission lines 
        ak = py_lineidplot.initial_annotate_kwargs()
        ak["verticalalignment"] = "bottom"
        ak["horizontalalignment"] = "center"
        pk = py_lineidplot.initial_plot_kwargs()
        pk["lw"] = 0.5
        pk["alpha"] = 0.5
        fig, ax1 = py_lineidplot.plot_line_ids(copy(λ), copy(I ./ norm .* factor), line_wave[line_annotate], line_latex[line_annotate], ax=ax1,
            extend=false, label1_size=12, plot_kwargs=pk, annotate_kwargs=ak)

        # Output file path creation
        out_folder = joinpath("output_$name", isnothing(range) ? "spaxel_plots" : joinpath("zoomed_plots", split(label, "_")[end]))
        if !isdir(out_folder)
            mkdir(out_folder)
        end
        # Save figure as PDF, yay for vector graphics!
        plt.savefig(joinpath(out_folder, isnothing(label) ? "levmar_fit_spaxel.pdf" : "$label.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    end
end


# Helper function for fitting one iteration (i.e. for bootstrapping)
function _fit_spaxel_iterfunc(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real}, 
    σ::Vector{<:Real}, templates::Matrix{<:Real}, norm::Real, area_sr::Vector{<:Real}, mask_lines::BitVector, mask_bad::BitVector, 
    mask_chi2::BitVector, I_spline::Vector{<:Real}; bootstrap_iter::Bool=false, p1_boots_c::Union{Vector{<:Real},Nothing}=nothing, 
    p1_boots_l::Union{Vector{<:Real},Nothing}=nothing, p1_boots_pah::Union{Vector{<:Real},Nothing}=nothing, use_ap::Bool=false, 
    init::Bool=false, nuc_temp_fit::Bool=false)

    # Interpolate the LSF
    lsf_interp = Spline1D(λ, cube_fitter.cube.lsf, k=1)
    lsf_interp_func = x -> lsf_interp(x)    # Interpolate the LSF

    if use_ap || init
        pah_template_spaxel = true
    else
        pah_template_spaxel = cube_fitter.pah_template_map[spaxel]
    end
    p1_boots_cont = p1_boots_c
    if cube_fitter.use_pah_templates && pah_template_spaxel && bootstrap_iter
        p1_boots_cont = [p1_boots_cont; p1_boots_pah]
    end

    # Fit the spaxel
    ext_key = cube_fitter.spectral_region == :MIR ? "extinction" : "attenuation_gas"
    if !cube_fitter.fit_joint
        popt_c, I_cont, comps_cont, n_free_c, perr_c, pahtemp = continuum_fit_spaxel(cube_fitter, spaxel, λ, I, σ, templates, mask_lines, mask_bad, norm, 
            cube_fitter.use_pah_templates && pah_template_spaxel, use_ap=use_ap, init=init, nuc_temp_fit=nuc_temp_fit, bootstrap_iter=bootstrap_iter, 
            p1_boots=p1_boots_cont)

        # Use the real continuum fit or a cubic spline continuum fit based on the settings
        line_cont = copy(I_cont)
        if cube_fitter.subtract_cubic_spline
            full_cont = I ./ norm
            temp_cont = zeros(eltype(line_cont), length(line_cont))
            # subtract any templates first
            if !nuc_temp_fit
                for comp ∈ keys(comps_cont)
                    if contains(comp, "templates_")
                        temp_cont .+= comps_cont[comp]
                    end
                end
            end
            # do cubic spline fit
            _, notemp_cont, _ = continuum_cubic_spline(λ, full_cont .- temp_cont, zeros(eltype(line_cont), length(line_cont)), 
                cube_fitter.linemask_Δ, cube_fitter.linemask_n_inc_thresh, cube_fitter.linemask_thresh, cube_fitter.linemask_overrides)
            line_cont = temp_cont .+ notemp_cont
        end

        templates_psfnuc = nuc_temp_fit ? comps_cont["templates_1"] : nothing
        popt_l, I_line, comps_line, n_free_l, perr_l = line_fit_spaxel(cube_fitter, spaxel, λ, I, σ, mask_bad, line_cont, comps_cont[ext_key], 
            lsf_interp_func, templates_psfnuc, norm, use_ap=use_ap, init=init, nuc_temp_fit=nuc_temp_fit, bootstrap_iter=bootstrap_iter, p1_boots=p1_boots_l)
    else
        popt_c, popt_l, I_cont, I_line, comps_cont, comps_line, n_free_c, n_free_l, perr_c, perr_l, pahtemp = all_fit_spaxel(cube_fitter,
            spaxel, λ, I, σ, templates, mask_bad, I_spline, norm, area_sr, lsf_interp_func, use_ap=use_ap, init=init, nuc_temp_fit=nuc_temp_fit, 
            bootstrap_iter=bootstrap_iter, p1_boots_cont=p1_boots_cont, p1_boots_line=p1_boots_l)
    end

    # Combine the continuum and line models
    I_model = I_cont .+ I_line
    comps = merge(comps_cont, comps_line)

    # Renormalize
    I_model .*= norm
    for comp ∈ keys(comps)
        if (comp == "extinction") || contains(comp, "ext_") || contains(comp, "abs_") || contains(comp, "attenuation")
            continue
        end
        if nuc_temp_fit && contains(comp, "template")
            continue
        end
        comps[comp] .*= norm
    end
    templates_psfnuc = nothing
    if nuc_temp_fit
        templates_psfnuc = comps["templates_1"]
    end

    # Total free parameters
    n_free = n_free_c + n_free_l
    n_data = length(I)

    # Degrees of freedom
    dof = n_data - n_free

    # chi^2 and reduced chi^2 of the model
    χ2 = sum(@. (I[.~mask_chi2] - I_model[.~mask_chi2])^2 / σ[.~mask_chi2]^2)

    # Add dust feature and line parameters (intensity and SNR)
    if !init
        if cube_fitter.spectral_region == :MIR
            p_dust, p_lines, p_dust_err, p_lines_err = calculate_extra_parameters(λ, I, norm, comps, cube_fitter.n_channels, cube_fitter.n_dust_cont,
                cube_fitter.n_power_law, cube_fitter.n_dust_feat, cube_fitter.dust_features.profiles, cube_fitter.n_abs_feat, 
                cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, cube_fitter.n_templates, nuc_temp_fit, cube_fitter.n_lines, cube_fitter.n_acomps, 
                cube_fitter.n_comps, cube_fitter.lines, cube_fitter.flexible_wavesol, lsf_interp_func, cube_fitter.relative_flags, popt_c, 
                popt_l, perr_c, perr_l, comps[ext_key], comps[ext_key], templates_psfnuc, cube_fitter.extinction_curve, mask_lines, I_spline, area_sr, 
                cube_fitter.n_fit_comps, spaxel, !bootstrap_iter)
            p_out = [popt_c; popt_l; p_dust; p_lines; χ2; dof]
            p_err = [perr_c; perr_l; p_dust_err; p_lines_err; 0.; 0.]
        else
            p_lines, p_lines_err = calculate_extra_parameters(λ, I, norm, comps, cube_fitter.n_ssps, cube_fitter.n_power_law,
                cube_fitter.fit_opt_na_feii, cube_fitter.fit_opt_br_feii, cube_fitter.n_lines, cube_fitter.n_acomps, cube_fitter.n_comps, 
                cube_fitter.lines, cube_fitter.flexible_wavesol, lsf_interp_func, cube_fitter.relative_flags, popt_l, perr_l, comps[ext_key], 
                mask_lines, I_spline, area_sr, cube_fitter.n_fit_comps, cube_fitter.n_templates, nuc_temp_fit, spaxel, !bootstrap_iter)
            p_out = [popt_c; popt_l; p_lines; χ2; dof]
            p_err = [perr_c; perr_l; p_lines_err; 0.; 0.]
        end
        
        return p_out, p_err, popt_c, popt_l, perr_c, perr_l, I_model, comps, χ2, dof, pahtemp
    end
    return [popt_c; popt_l], [perr_c; perr_l], I_model, comps, χ2, dof
end


"""
    fit_spaxel(cube_fitter, cube_data, spaxel; [use_ap])

Wrapper function to perform a full fit of a single spaxel, calling `continuum_fit_spaxel` and `line_fit_spaxel` and
concatenating the best-fit parameters. The outputs are also saved to files so the fit need not be repeated in the case
of a crash.

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `cube_data::NamedTuple`: Contains the wavelength, intensity, and error vectors that will be fit, as well as the solid angle vector
- `spaxel::CartesianIndex`: The coordinates of the spaxel to be fit
- `use_ap::Bool=false`: Flag determining whether or not one is fitting an integrated spectrum within an aperture
- `use_vorbins::Bool=false`: Flag for using voronoi binning 
- `nuc_temp_fit::Bool=false`: Flag for fitting a nuclear template
- `σ_min::Real=0.`: Optional minimum uncertainty to add in quadrature with the statistical uncertainty. Only applies if
    use_ap or use_vorbins is true.
"""
function fit_spaxel(cube_fitter::CubeFitter, cube_data::NamedTuple, spaxel::CartesianIndex; use_ap::Bool=false,
    use_vorbins::Bool=false, nuc_temp_fit::Bool=false, σ_min::Real=0.)

    λ = cube_data.λ
    I = cube_data.I[spaxel, :]
    σ = cube_data.σ[spaxel, :]
    area_sr = cube_data.area_sr[spaxel, :]
    templates = cube_data.templates[spaxel, :, :]

    # if there are any NaNs, skip over the spaxel
    if any(.!isfinite.(I))
        @debug "Non-finite values found in the intensity! Not fitting spaxel $spaxel"
        return nothing, nothing
    end

    for t in 1:cube_fitter.n_templates
        if any(.~isfinite.(templates[:, t]))
            # If a template is fully NaN/Inf, we can't fit 
            @debug "Non-finite values found in the templates! Not fitting spaxel $spaxel"
            return nothing, nothing
        end
    end

    # Perform a cubic spline fit, also obtaining the line mask
    mask_lines, I_spline, σ_spline = continuum_cubic_spline(λ, I, σ, cube_fitter.linemask_Δ, cube_fitter.linemask_n_inc_thresh,
        cube_fitter.linemask_thresh, cube_fitter.linemask_overrides)
    mask_bad = (use_ap || use_vorbins) ? iszero.(I) .| iszero.(σ) : cube_fitter.cube.mask[spaxel, :]
    mask = mask_lines .| mask_bad

    # Check the line mask against the expected line locations from the fitted line list
    line_reference = falses(length(λ))
    for i ∈ 1:cube_fitter.n_lines
        max_voff = maximum(abs, cube_fitter.lines.voff[i, 1].limits)
        max_fwhm = maximum(cube_fitter.lines.fwhm[i, 1].limits)
        center = cube_fitter.lines.λ₀[i]
        region = center .* (1 - (max_voff+2max_fwhm)/C_KMS, 1 + (max_voff+2max_fwhm)/C_KMS)
        line_reference[region[1] .< λ .< region[2]] .= 1
    end
    # If any fall outside of this region, do not include these pixels in the chi^2 calculations
    mask_chi2 = mask_bad .| (mask_lines .& .~line_reference)

    if use_ap || use_vorbins
        l_mask = sum(.~mask)
        # Statistical uncertainties based on the local RMS of the residuals with a cubic spline fit
        σ_stat = zeros(l_mask)
        for i in 1:l_mask
            indices = sortperm(abs.((1:l_mask) .- i))[1:60]
            σ_stat[i] = std(I[.~mask][indices] .- I_spline[.~mask][indices])
        end
        # We insert at the locations of the lines since the cubic spline does not include them
        l_all = length(λ)
        line_inds = (1:l_all)[mask]
        for line_ind ∈ line_inds
            insert!(σ_stat, line_ind, σ_stat[max(line_ind-1, 1)])
        end
        @debug "Statistical uncertainties: ($(σ_stat[1]) - $(σ_stat[end]))"
        σ .= σ_stat
        # Add in quadrature with minimum uncertainty
        σ = sqrt.(σ.^2 .+ σ_min.^2)
    end

    # Add systematic error in quadrature
    σ = sqrt.(σ.^2 .+ (cube_fitter.sys_err .* I).^2)

    # Check if the fit has already been performed
    fname = use_vorbins ? "voronoi_bin_$(spaxel[1])" : "spaxel_$(spaxel[1])_$(spaxel[2])"
    if nuc_temp_fit
        fname *= "_nuc"
    end
    if !isfile(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.csv")) || cube_fitter.overwrite
        
        # Create a local logger for this individual spaxel
        timestamp_logger(logger) = TransformerLogger(logger) do log
            merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
        end
        # This log should be entirely handled by 1 process, since each spaxel is entirely handled by 1 process
        # so there should be no problems with I/O race conditions
        spaxel_logger = TeeLogger(ConsoleLogger(stdout, Logging.Info), timestamp_logger(MinLevelLogger(FileLogger(
                             joinpath("output_$(cube_fitter.name)", "logs", "loki.$fname.log"); 
                             always_flush=true), Logging.Debug)))

        p_out, p_err = with_logger(spaxel_logger) do

            # Use a fixed normalization for the line fits so that the bootstrapped amplitudes are consistent with each other
            norm = Float64(abs(nanmaximum(I)))
            norm = norm ≠ 0. ? norm : 1.

            # Perform the regular fit
            p_out, p_err, popt_c, popt_l, perr_c, perr_l, I_model, comps, χ2, dof, pahtemp = _fit_spaxel_iterfunc(
                cube_fitter, spaxel, λ, I, σ, templates, norm, area_sr, mask_lines, mask_bad, mask_chi2, I_spline; 
                bootstrap_iter=false, use_ap=use_ap, nuc_temp_fit=nuc_temp_fit)
            # Convert p_err into 2 columns for the lower/upper errorbars
            p_err = [p_err p_err]

            # Perform the bootstrapping iterations, if n_bootstrap > 0
            I_boot_min = I_boot_max = nothing
            if cube_fitter.n_bootstrap > 0
                # Set the random seed
                @debug "Setting the random seed to: $(cube_fitter.random_seed)"
                Random.seed!(cube_fitter.random_seed)
                # Resample the data using normal distributions with the statistical uncertainties
                I_boots = [rand.(Normal.(cube_data.I[spaxel, :], σ)) for _ in 1:cube_fitter.n_bootstrap]
                # Initialize 2D parameter array
                p_boot = zeros(length(p_out), cube_fitter.n_bootstrap)
                # Initialize bootstrap model array
                I_model_boot = zeros(length(I_model), cube_fitter.n_bootstrap)

                # Do bootstrapping multi-threaded to save time
                @debug "Performing multi-threaded bootstrapping iterations for spaxel $spaxel..."
                Threads.@threads for nboot ∈ 1:cube_fitter.n_bootstrap
                    lock(file_lock) do
                        @debug "Bootstrap iteration: $nboot"
                    end
                    # Get the bootstrapped data vector for this iteration
                    # N.b.: Don't use rand() inside a Threads loop! It is not thread-safe, i.e. the RNG will not be consistent
                    #  even after setting the seed
                    I_boot = I_boots[nboot]
                    # Redo the cubic spline fit
                    mask_lines_boot, I_spline_boot, σ_spline_boot = with_logger(NullLogger()) do 
                        continuum_cubic_spline(λ, I, σ, cube_fitter.linemask_Δ, cube_fitter.linemask_n_inc_thresh, 
                            cube_fitter.linemask_thresh, cube_fitter.linemask_overrides)
                    end

                    # Re-perform the fitting on the resampled data
                    pb_i, _, _, _, _, _, Ib_i, _, _, _, _ = with_logger(NullLogger()) do
                        _fit_spaxel_iterfunc(cube_fitter, spaxel, λ, I_boot, σ, templates, norm, area_sr, mask_lines_boot, mask_bad, mask_chi2, 
                            I_spline_boot; bootstrap_iter=true, p1_boots_c=popt_c, p1_boots_l=popt_l, p1_boots_pah=pahtemp, use_ap=use_ap)
                    end
                    # Sort the emission line components (if not using relative parameters) to avoid multi-modal distirbutions
                    # only mask additional line components after the first bootstrap so that they have a minimum of 1 finite sample
                    sort_line_components!(cube_fitter, pb_i, mask_zeros=nboot > 1)

                    p_boot[:, nboot] .= pb_i
                    I_model_boot[:, nboot] .= Ib_i
                end

                # RESULTS: Values are the 50th percentile, and errors are the (15.9th, 84.1st) percentiles
                open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$(fname)_p_boots.csv"), "w") do f
                    writedlm(f, p_boot) 
                end

                # Filter out any large (>5 sigma) outliers
                p_med = dropdims(nanquantile(p_boot, 0.50, dims=2), dims=2) 
                p_mad = nanmad(p_boot, dims=2) .* 1.4826   # factor of 1.4826 to normalize the MAD it so it is interpretable as a standard deviation
                p_mask = (p_boot .< (p_med .- 5 .* p_mad)) .| (p_boot .> (p_med .+ 5 .* p_mad))
                p_boot[p_mask] .= NaN

                if cube_fitter.bootstrap_use == :med
                    p_out = p_med
                end
                # (if set to :best, p_out is already the best-fit values from earlier)
                p_err_lo = p_med .- dropdims(nanquantile(p_boot, 0.159, dims=2), dims=2)
                p_err_up = dropdims(nanquantile(p_boot, 0.841, dims=2), dims=2) .- p_med
                p_err = [p_err_lo p_err_up]

                # Get the minimum/maximum pointwise bootstrapped models
                I_boot_min = dropdims(nanminimum(I_model_boot, dims=2), dims=2)
                I_boot_max = dropdims(nanmaximum(I_model_boot, dims=2), dims=2)

                split1 = length(popt_c)
                split2 = length(popt_c) + length(popt_l)
                lsf_interp = Spline1D(λ, cube_fitter.cube.lsf, k=1)
                lsf_interp_func = x -> lsf_interp(x)

                # Replace the best-fit model with the 50th percentile model to be consistent with p_out
                if cube_fitter.spectral_region == :MIR
                    I_boot_cont, comps_boot_cont = model_continuum(λ, p_out[1:split1], norm, cube_fitter.n_dust_cont, cube_fitter.n_power_law, 
                        cube_fitter.dust_features.profiles, cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, 
                        cube_fitter.κ_abs, cube_fitter.custom_ext_template, cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, false, 
                        templates, cube_fitter.channel_masks, nuc_temp_fit, true)
                    ext_key = "extinction"
                else
                    I_boot_cont, comps_boot_cont = model_continuum(λ, p_out[1:split1], norm, cube_fitter.vres, cube_fitter.vsyst_ssp, 
                        cube_fitter.vsyst_feii, cube_fitter.npad_feii, cube_fitter.n_ssps, cube_fitter.ssp_λ, cube_fitter.ssp_templates,
                         cube_fitter.feii_templates_fft, cube_fitter.n_power_law, cube_fitter.fit_uv_bump, cube_fitter.fit_covering_frac, 
                         cube_fitter.fit_opt_na_feii, cube_fitter.fit_opt_br_feii, cube_fitter.extinction_curve, templates, cube_fitter.fit_temp_multexp,
                         nuc_temp_fit, true)
                    ext_key = "attenuation_gas"
                end
                I_boot_line, comps_boot_line = model_line_residuals(λ, p_out[split1+1:split2], cube_fitter.n_lines, cube_fitter.n_comps,
                    cube_fitter.lines, cube_fitter.flexible_wavesol, comps_boot_cont[ext_key], lsf_interp_func, cube_fitter.relative_flags,
                    nuc_temp_fit ? comps_boot_cont["templates_1"] : nothing, nuc_temp_fit, true)

                # Reconstruct the full model
                I_model = I_boot_cont .+ I_boot_line
                comps = merge(comps_boot_cont, comps_boot_line)

                # Renormalize
                I_model .*= norm
                for comp ∈ keys(comps)
                    if (comp == "extinction") || contains(comp, "ext_") || contains(comp, "abs_") || contains(comp, "attenuation")
                        continue
                    end
                    if nuc_temp_fit && contains(comp, "template")
                        continue
                    end
                    comps[comp] .*= norm
                end

                # Recalculate chi^2 based on the median model
                p_out[end-1] = sum(@. (I - I_model)^2/σ^2)
                χ2 = p_out[end-1]

            end

            # Plot the fit
            if cube_fitter.plot_spaxels != :none
                @debug "Plotting spaxel $spaxel best fit" 
                p_cf = p_out[3+2cube_fitter.n_dust_cont+2cube_fitter.n_power_law+(cube_fitter.extinction_curve=="decompose" ? 3 : 1)+3]
                plot_spaxel_fit(cube_fitter.spectral_region, λ, I, I_model, σ, mask_bad, mask_lines, cube_fitter.user_mask, comps, 
                    cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_dust_feat, cube_fitter.n_abs_feat, cube_fitter.n_templates, cube_fitter.n_ssps, 
                    nuc_temp_fit, cube_fitter.n_comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, cube_fitter.lines.annotate, cube_fitter.lines.latex, 
                    cube_fitter.extinction_screen, p_cf, cube_fitter.z, χ2/dof, cube_fitter.name, fname, backend=cube_fitter.plot_spaxels, I_boot_min=I_boot_min, 
                    I_boot_max=I_boot_max)
                if !isnothing(cube_fitter.plot_range)
                    for (i, plot_range) ∈ enumerate(cube_fitter.plot_range)
                        fname2 = use_vorbins ? "lines_bin_$(spaxel[1])_$i" : "lines_$(spaxel[1])_$(spaxel[2])_$i"
                        plot_spaxel_fit(cube_fitter.spectral_region, λ, I, I_model, σ, mask_bad, mask_lines, cube_fitter.user_mask, comps,
                            cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_dust_feat, cube_fitter.n_abs_feat, cube_fitter.n_templates, cube_fitter.n_ssps, 
                            nuc_temp_fit, cube_fitter.n_comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, cube_fitter.lines.annotate, cube_fitter.lines.latex, 
                            cube_fitter.extinction_screen, p_cf, cube_fitter.z, χ2/dof, cube_fitter.name, fname2, backend=cube_fitter.plot_spaxels, I_boot_min=I_boot_min, 
                            I_boot_max=I_boot_max, range_um=plot_range)
                    end
                end
            end

            @debug "Saving results to csv for spaxel $spaxel"
            # serialize(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).LOKI"), (p_out=p_out, p_err=p_err))
            # save output as csv file
            open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.csv"), "w") do f 
                writedlm(f, [p_out p_err], ',')
            end
 
            # save memory allocations & other logistic data to a separate log file
            if cube_fitter.track_memory
                open(joinpath("output_$(cube_fitter.name)", "logs", "mem.$fname.log"), "w") do f

                    print(f, """
                    ### PROCESS ID: $(getpid()) ###
                    Memory usage stats:
                    CubeFitter - $(Base.summarysize(cube_fitter) ÷ 10^6) MB
                        Cube - $(Base.summarysize(cube_fitter.cube) ÷ 10^6) MB 
                    """)

                    print(f, """
                    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
                    """)
                end
            end

            # Overwrite the errors with the statistical errors
            cube_data.σ[spaxel, :] .= σ

            p_out, p_err
        end

        return p_out, p_err

    end

    # Otherwise, just grab the results from before
    results = readdlm(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.csv"), ',', Float64, '\n')
    p_out = results[:, 1]
    p_err = results[:, 2:3]

    # Still need to overwrite the raw errors with the statistical errors
    cube_data.σ[spaxel, :] .= σ

    p_out, p_err

end


"""
    fill_bad_pixels(cube_fitter, I, σ, template)

Helper function to fill in NaNs/Infs in a 1D intensity/error vector and 2D templates vector.
"""
function fill_bad_pixels(cube_fitter::CubeFitter, I::Vector{<:Real}, σ::Vector{<:Real}, templates::Array{<:Real,2})

    # Edge cases
    if !isfinite(I[1])
        I[1] = nanmedian(I)
    end
    if !isfinite(I[end])
        I[end] = nanmedian(I)
    end
    if !isfinite(σ[1])
        σ[1] = nanmedian(σ)
    end
    if !isfinite(σ[end])
        σ[end] = nanmedian(σ)
    end
    for s in 1:cube_fitter.n_templates
        if !isfinite(templates[1,s])
            templates[1,s] = nanmedian(templates[:,s])
        end
        if !isfinite(templates[end,s])
            templates[end,s] = nanmedian(templates[:,s])
        end
    end

    bad = findall(.~isfinite.(I) .| .~isfinite.(σ))
    # Replace with the average of the points to the left and right
    l = length(I)
    for badi in bad
        lind = findfirst(isfinite, I[max(badi-1,1):-1:1])
        rind = findfirst(isfinite, I[min(badi+1,l):end])
        I[badi] = (I[max(badi-1,1):-1:1][lind] + I[min(badi+1,l):end][rind]) / 2
        σ[badi] = (σ[max(badi-1,1):-1:1][lind] + σ[min(badi+1,l):end][rind]) / 2
    end
    @assert all(isfinite.(I) .& isfinite.(σ)) "Error: Non-finite values found in the summed intensity/error arrays!"
    for s in 1:cube_fitter.n_templates
        bad = findall(.~isfinite.(templates[:, s]))
        l = length(templates[:, s])
        for badi in bad
            lind = findfirst(isfinite, templates[max(badi-1,1):-1:1, s])
            rind = findfirst(isfinite, templates[min(badi+1,l):end, s])
            templates[badi, s] = (templates[max(badi-1,1):-1:1, s][lind] + templates[min(badi+1,l):end, s][rind]) / 2
        end
    end
    @assert all(isfinite.(templates)) "Error: Non-finite values found in the summed template arrays!"

    return I, σ, templates
end


"""
    fit_stack!(cube_fitter)

Perform an initial fit to the sum of all spaxels (the stack) to get an estimate for the initial parameter
vector to use with all of the individual spaxel fits.  The only input is the CubeFitter object, which is
modified with the resultant fit parameters.  There is no output.
"""
function fit_stack!(cube_fitter::CubeFitter)

    @info "===> Performing initial fit to the sum of all spaxels... <==="
    # Collect the data
    λ_init = cube_fitter.cube.λ
    I_sum_init = sumdim(cube_fitter.cube.I, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    σ_sum_init = sqrt.(sumdim(cube_fitter.cube.σ.^2, (1,2))) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    area_sr_init = cube_fitter.cube.Ω .* sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    templates_init = zeros(eltype(I_sum_init), size(I_sum_init)..., cube_fitter.n_templates)
    for s in 1:cube_fitter.n_templates
        templates_init[:, s] .= sumdim(cube_fitter.templates[:,:,:,s], (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    end
    I_sum_init, σ_sum_init, templates_init = fill_bad_pixels(cube_fitter, I_sum_init, σ_sum_init, templates_init)

    # Perform a cubic spline fit, also obtaining the line mask
    mask_lines_init, I_spline_init, σ_spline_init = continuum_cubic_spline(λ_init, I_sum_init, σ_sum_init, cube_fitter.linemask_Δ,
        cube_fitter.linemask_n_inc_thresh, cube_fitter.linemask_thresh, cube_fitter.linemask_overrides)
    mask_bad_init = iszero.(I_sum_init) .| iszero.(σ_sum_init)
    mask_init = mask_lines_init .| mask_bad_init

    l_mask = sum(.~mask_init)

    # Statistical uncertainties based on the local RMS of the residuals with a cubic spline fit
    σ_stat_init = zeros(l_mask)
    for i in 1:l_mask
        indices = sortperm(abs.((1:l_mask) .- i))[1:60]
        σ_stat_init[i] = std(I_sum_init[.~mask_init][indices] .- I_spline_init[.~mask_init][indices])
    end
    # We insert at the locations of the lines since the cubic spline does not include them
    l_all = length(λ_init)
    line_inds = (1:l_all)[mask_init]
    for line_ind ∈ line_inds
        insert!(σ_stat_init, line_ind, σ_stat_init[max(line_ind-1, 1)])
    end
    @debug "Statistical uncertainties: ($(σ_stat_init[1]) - $(σ_stat_init[end]))"
    mask_chi2_init = mask_bad_init

    σ_sum_init .= σ_stat_init

    # Get the normalization
    norm = abs(nanmaximum(I_sum_init))
    norm = norm ≠ 0. ? norm : 1.

    p_out, p_err, I_model_init, comps_init, χ2_init, dof_init = _fit_spaxel_iterfunc(
        cube_fitter, CartesianIndex(0,0), λ_init, I_sum_init, σ_sum_init, templates_init, norm, area_sr_init, mask_lines_init, 
        mask_bad_init, mask_chi2_init, I_spline_init; bootstrap_iter=false, use_ap=false, init=true)

    χ2red_init = χ2_init / dof_init

    # Plot the fit
    if cube_fitter.plot_spaxels != :none
        @debug "Plotting spaxel sum initial fit"
        p_cf = p_out[3+2cube_fitter.n_dust_cont+2cube_fitter.n_power_law+(cube_fitter.extinction_curve=="decompose" ? 3 : 1)+3]
        plot_spaxel_fit(cube_fitter.spectral_region, λ_init, I_sum_init, I_model_init, σ_sum_init, mask_bad_init, mask_lines_init, cube_fitter.user_mask, comps_init,
            cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_dust_feat, cube_fitter.n_abs_feat, cube_fitter.n_templates, cube_fitter.n_ssps, 
            false, cube_fitter.n_comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, cube_fitter.lines.annotate, cube_fitter.lines.latex, cube_fitter.extinction_screen,
            p_cf, cube_fitter.z, χ2red_init, cube_fitter.name, "initial_sum_fit", backend=:both)
        if !isnothing(cube_fitter.plot_range)
            for (i, plot_range) ∈ enumerate(cube_fitter.plot_range)
                plot_spaxel_fit(cube_fitter.spectral_region, λ_init, I_sum_init, I_model_init, σ_sum_init, mask_bad_init, mask_lines_init, cube_fitter.user_mask, comps_init,
                    cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_dust_feat, cube_fitter.n_abs_feat, cube_fitter.n_templates, 
                    cube_fitter.n_ssps, false, cube_fitter.n_comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, cube_fitter.lines.annotate, cube_fitter.lines.latex, 
                    cube_fitter.extinction_screen, p_cf, cube_fitter.z, χ2red_init, cube_fitter.name, "initial_sum_line_$i", backend=:both; range_um=plot_range)
            end
        end

    end

end


"""
    fit_cube!(cube_fitter)

This is the main cube fitting function!! It's essentially a wrapper function to perform a full fit of an 
entire IFU cube, calling `fit_spaxel` for each spaxel in a parallel or serial loop depending on the cube_fitter options.  
Results are then concatenated into `ParamMaps` and `CubeModel` structs and plotted/saved, also based on the 
cube_fitter options.

The default behavior is to perform spaxel-by-spaxel fits. However, if an aperture is specified, then fitting
will only be performed on the integrated spectrum within the aperture.
"""
function fit_cube!(cube_fitter::CubeFitter)

    @info """\n
    #############################################################################
    ######## BEGINNING FULL CUBE FITTING ROUTINE FOR $(cube_fitter.name) ########
    #############################################################################

    ------------------------
    Worker Processes:     $(nworkers())
    Threads per process:  $(Threads.nthreads())
    ------------------------
    """
    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    shape = size(cube_fitter.cube.I)

    # Prepare output array
    @info "===> Preparing output data structures... <==="
    out_params = ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2) .* NaN
    out_errs = ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2, 2) .* NaN

    # "cube_data" object holds the primary wavelength, intensity, and errors
    # this is just a convenience object since these may be different when fitting an integrated spectrum
    # within an aperture, or when using voronoi bins
    cube_data = (λ=cube_fitter.cube.λ, I=cube_fitter.cube.I, σ=cube_fitter.cube.σ, 
        templates=cube_fitter.n_templates > 0 ? cube_fitter.templates : Array{Float64}(undef, shape..., 0),
        area_sr=cube_fitter.cube.Ω .* ones(shape))

    vorbin = !isnothing(cube_fitter.cube.voronoi_bins)
    if vorbin
        # Reformat cube data as a 2D array with the first axis slicing each voronoi bin
        n_bins = maximum(cube_fitter.cube.voronoi_bins)
        I_vorbin = zeros(n_bins, shape[3])
        σ_vorbin = zeros(n_bins, shape[3])
        area_vorbin = zeros(n_bins, shape[3])
        template_vorbin = zeros(n_bins, shape[3], cube_fitter.n_templates)
        for n in 1:n_bins
            w = cube_fitter.cube.voronoi_bins .== n
            I_vorbin[n, :] .= sumdim(cube_fitter.cube.I[w, :], 1) ./ sum(w)
            σ_vorbin[n, :] .= sqrt.(sumdim(cube_fitter.cube.σ[w, :].^2, 1)) ./ sum(w)
            area_vorbin[n, :] .= sum(w) .* cube_fitter.cube.Ω
            for s in 1:cube_fitter.n_templates
                template_vorbin[n, :, s] .= sumdim(cube_fitter.templates[w, :, s], 1) ./ sum(w)
            end
        end
        cube_data = (λ=cube_fitter.cube.λ, I=I_vorbin, σ=σ_vorbin, area_sr=area_vorbin, templates=template_vorbin)
    end

    ######################### DO AN INITIAL FIT WITH THE SUM OF ALL SPAXELS ###################

    @debug """
    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
    """

    # Don't repeat if it's already been done, and also dont do the initial fit if we're just fitting in an aperture
    if all(iszero.(cube_fitter.p_init_cont)) && isnothing(cube_fitter.p_init_cube_λ)
        fit_stack!(cube_fitter)
    else
        @info "===> Initial fit to the sum of all spaxels is being skipped, either because it has already " *
            "been performed, an aperture was specified, or an initial condition was input <==="
    end

    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    ##############################################################################################

    # Get the indices of all spaxels
    spaxels = CartesianIndices(selectdim(cube_fitter.cube.I, 3, 1))
    # The indices are different for voronoi-binned cubes
    if vorbin
        spaxels = CartesianIndices((n_bins,))
    end

    # Only loop over spaxels with data
    _m = [any(.~isfinite.(cube_data.I[spaxel, :])) || any(.~isfinite.(cube_data.templates[spaxel, :, :])) for spaxel in spaxels]
    spaxels = spaxels[.~_m]

    # First (non-parallel) loop over already-completed spaxels
    _m2 = falses(length(spaxels))
    if !cube_fitter.overwrite
        @info "Loading in previous fit results (if any...)"
        @showprogress for (ii, index) ∈ enumerate(spaxels)
            fname = vorbin ? "voronoi_bin_$(index[1])" : "spaxel_$(index[1])_$(index[2])"
            fpath = joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.csv")
            if isfile(fpath)
                # just loads in the previous fit results since they already exist
                p_out, p_err = fit_spaxel(cube_fitter, cube_data, index; use_vorbins=vorbin)
                if vorbin
                    out_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(index)[1])
                    for out_index in out_indices
                        out_params[out_index, :] .= p_out
                        out_errs[out_index, :, :] .= p_err
                    end
                else
                    out_params[index, :] .= p_out
                    out_errs[index, :, :] .= p_err
                end
                # remove this spaxel from the list of spaxels that still need to be fit
                _m2[ii] = 1
            end
        end
    end
    spaxels = spaxels[.~_m2]

    # Use multiprocessing (not threading) to iterate over multiple spaxels at once using multiple CPUs
    if cube_fitter.parallel
        @info "===> Beginning individual spaxel fitting... <==="
        # Parallel loops require these to be SharedArrays so that each process accesses the same array
        out_params = SharedArray(out_params)
        out_errs = SharedArray(out_errs)

        if cube_fitter.parallel_strategy == "pmap"
            prog = Progress(length(spaxels))
            progress_pmap(CachingPool(workers()), spaxels, progress=prog) do index
                p_out, p_err = fit_spaxel(cube_fitter, cube_data, index; use_vorbins=vorbin)
                if !isnothing(p_out)
                    if vorbin
                        out_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(index)[1])
                        for out_index in out_indices
                            out_params[out_index, :] .= p_out
                            out_errs[out_index, :, :] .= p_err
                        end
                    else
                        out_params[index, :] .= p_out
                        out_errs[index, :, :] .= p_err
                    end
                end
                nothing
            end
        elseif cube_fitter.parallel_strategy == "distributed"
            @showprogress @distributed for index ∈ spaxels
                p_out, p_err = fit_spaxel(cube_fitter, cube_data, index; use_vorbins=vorbin)
                if !isnothing(p_out)
                    if vorbin
                        out_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(index)[1])
                        for out_index in out_indices
                            out_params[out_index, :] .= p_out
                            out_errs[out_index, :, :] .= p_err
                        end
                    else
                        out_params[index, :] .= p_out
                        out_errs[index, :, :] .= p_err
                    end
                end
            end
        else
            error("Unrecognized parallel strategy: $(cube_fitter.parallel_strategy). May be \"pmap\" or \"distributed\".")
        end
    else
        @info "===> Beginning individual spaxel fitting... <==="
        prog = Progress(length(spaxels))
        for index ∈ spaxels
            p_out, p_err = fit_spaxel(cube_fitter, cube_data, index; use_vorbins=vorbin)
            if !isnothing(p_out)
                if vorbin
                    out_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(index)[1])
                    for out_index in out_indices
                        out_params[out_index, :] .= p_out
                        out_errs[out_index, :, :] .= p_err
                    end
                else
                    out_params[index, :] .= p_out
                    out_errs[index, :, :] .= p_err
                end
            end
            next!(prog)
        end
    end

    @info "===> Generating parameter maps and model cubes... <==="

    # Create the ParamMaps and CubeModel structs containing the outputs
    param_maps, cube_model = assign_outputs(out_params, out_errs, cube_fitter, cube_data, cube_fitter.z, false)

    if cube_fitter.plot_maps
        @info "===> Plotting parameter maps... <==="
        plot_parameter_maps(cube_fitter, param_maps, snr_thresh=cube_fitter.map_snr_thresh)
    end

    if cube_fitter.save_fits
        @info "===> Writing FITS outputs... <==="
        write_fits(cube_fitter, cube_data, cube_model, param_maps)
    end

    # Save a copy of the options file used to run the code, so the settings can be referenced/reused
    # (for example, if you need to recall which lines you tied, what your limits were, etc.)
    cp(joinpath(@__DIR__, "..", "options", "options.toml"), joinpath("output_$(cube_fitter.name)", "general_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "dust.toml"), joinpath("output_$(cube_fitter.name)", "dust_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "lines.toml"), joinpath("output_$(cube_fitter.name)", "lines_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "optical.toml"), joinpath("output_$(cube_fitter.name)", "optical_options.archive.toml"), force=true)

    if cube_fitter.make_movies
        @info "===> Writing MP4 movies... (this may take a while) <==="
        make_movie(cube_fitter, cube_model)
    end

    # copy the main log file again
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    @info """\n
    #############################################################################
    ################################### Done!! ##################################
    #############################################################################
    """

    # Return the final cube_fitter object, along with the param maps/errs and cube model
    cube_fitter, param_maps, cube_model

end


function fit_cube!(cube_fitter::CubeFitter, aperture::Aperture.AbstractAperture)
    # Extend the single aperture into an array of apertures and call the corresponding method of fit_cube!
    apertures = repeat([aperture], length(cube_fitter.cube.λ))
    fit_cube!(cube_fitter, apertures)
end


function fit_cube!(cube_fitter::CubeFitter, aperture::Union{Vector{<:Aperture.AbstractAperture},String})

    @info """\n
    #############################################################################
    ######## BEGINNING FULL CUBE FITTING ROUTINE FOR $(cube_fitter.name) ########
    #############################################################################

    ------------------------
    Worker Processes:     $(nworkers())
    Threads per process:  $(Threads.nthreads())
    ------------------------
    """
    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    shape = (1,1,size(cube_fitter.cube.I, 3))

    # Prepare output array
    @info "===> Preparing output data structures... <==="
    out_params = ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2) .* NaN
    out_errs = ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2, 2) .* NaN

    # Loop through each wavelength pixel and perform the aperture photometry
    if aperture isa String
        @assert lowercase(aperture) == "all" "The only accepted string input for 'aperture' is 'all' to signify the entire cube."
        
        @info "Integrating spectrum across the whole cube..."
        I = sumdim(cube_fitter.cube.I, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
        I = reshape(I, shape)
        σ = sqrt.(sumdim(cube_fitter.cube.σ.^2, (1,2))) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
        σ = reshape(σ, shape)
        area_sr = cube_fitter.cube.Ω .* sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
        area_sr = reshape(area_sr, shape)
        templates = zeros(shape..., cube_fitter.n_templates)
        for s in 1:cube_fitter.n_templates
            templates[1,1,:,s] .= sumdim(cube_fitter.templates, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
            templates[1,1,.~isfinite.(templates[1,1,:,s]),s] .= Spline1D(cube_fitter.cube.λ[isfinite.(templates[1,1,:,s])], 
                templates[1,1,:,s][isfinite.(templates[1,1,:,s])], k=1, bc="extrapolate")(cube_fitter.cube.λ[.~isfinite.(templates[1,1,:,s])])
        end
        I[.~isfinite.(I)] .= Spline1D(cube_fitter.cube.λ[isfinite.(I[1,1,:])], I[isfinite.(I)], k=1, bc="extrapolate")(cube_fitter.cube.λ[.~isfinite.(I[1,1,:])])
        σ[.~isfinite.(σ)] .= Spline1D(cube_fitter.cube.λ[isfinite.(σ[1,1,:])], σ[isfinite.(σ)], k=1, bc="extrapolate")(cube_fitter.cube.λ[.~isfinite.(σ[1,1,:])])

    else
        # If using an aperture, overwrite the cube_data object with the quantities within
        # the aperture, which are calculated here.
        # Prepare the 1D arrays
        I = zeros(Float32, shape)
        σ = zeros(Float32, shape)
        templates = zeros(Float32, shape..., cube_fitter.n_templates)
        area_sr = zeros(shape)

        @info "Performing aperture photometry to get an integrated spectrum..."
        for z ∈ 1:shape[3]

            # Sum up the FLUX within the aperture
            Fz = cube_fitter.cube.I[:, :, z] .* cube_fitter.cube.Ω
            e_Fz = cube_fitter.cube.σ[:, :, z] .* cube_fitter.cube.Ω
            # Zero out the masked spaxels
            Fz[cube_fitter.cube.mask[:, :, z]] .= 0.
            e_Fz[cube_fitter.cube.mask[:, :, z]] .= 0.
            # Perform the aperture photometry
            (_, _, F_ap, eF_ap) = photometry(aperture[z], Fz, e_Fz)

            # Convert back to intensity by dividing out the aperture area
            area_sr[1,1,z] = get_area(aperture[z]) * cube_fitter.cube.Ω
            I[1,1,z] = F_ap / area_sr[1,1,z]
            σ[1,1,z] = eF_ap / area_sr[1,1,z]

            # repeat for templates
            for s in 1:cube_fitter.n_templates
                Ftz = cube_fitter.templates[:, :, z, s] .* cube_fitter.cube.Ω
                Ftz[.~isfinite.(Ftz) .| cube_fitter.cube.mask[:, :, z]] .= 0.
                Ft_ap = photometry(aperture[z], Ftz).aperture_sum
                templates[1,1,z,s] = Ft_ap / area_sr[1,1,z]
            end

        end
    end
    cube_data = (λ=cube_fitter.cube.λ, I=I, σ=σ, templates=templates, area_sr=area_sr)

    ######################### DO AN INITIAL FIT WITH THE SUM OF ALL SPAXELS ###################

    @debug """
    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
    """

    # If using an aperture, plot the aperture
    if !(aperture isa String)
        plot_2d(cube_fitter.cube, joinpath("output_$(cube_fitter.name)", "aperture_plot_beg.pdf"); err=false, aperture=aperture[10],
            z=cube_fitter.z, cosmo=cube_fitter.cosmology, slice=1)
        plot_2d(cube_fitter.cube, joinpath("output_$(cube_fitter.name)", "aperture_plot_mid.pdf"); err=false, aperture=aperture[end÷2],
            z=cube_fitter.z, cosmo=cube_fitter.cosmology, slice=shape[3]÷2)
        plot_2d(cube_fitter.cube, joinpath("output_$(cube_fitter.name)", "aperture_plot_end.pdf"); err=false, aperture=aperture[end-9],
            z=cube_fitter.z, cosmo=cube_fitter.cosmology, slice=shape[3])
    end

    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    ##############################################################################################

    # Get the indices of all spaxels
    spaxels = CartesianIndices((1,1))

    @info "===> Beginning integrated spectrum fitting... <==="
    p_out, p_err = fit_spaxel(cube_fitter, cube_data, spaxels[1]; use_ap=true)
    if !isnothing(p_out)
        out_params[spaxels[1], :] .= p_out
        out_errs[spaxels[1], :, :] .= p_err
    end

    @info "===> Generating parameter maps and model cubes... <==="

    # Create the ParamMaps and CubeModel structs containing the outputs
    param_maps, cube_model = assign_outputs(out_params, out_errs, cube_fitter, cube_data, cube_fitter.z, true)

    if cube_fitter.save_fits
        @info "===> Writing FITS outputs... <==="
        write_fits(cube_fitter, cube_data, cube_model, param_maps, aperture=aperture)
    end

    # Save a copy of the options file used to run the code, so the settings can be referenced/reused
    # (for example, if you need to recall which lines you tied, what your limits were, etc.)
    cp(joinpath(@__DIR__, "..", "options", "options.toml"), joinpath("output_$(cube_fitter.name)", "general_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "dust.toml"), joinpath("output_$(cube_fitter.name)", "dust_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "lines.toml"), joinpath("output_$(cube_fitter.name)", "lines_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "optical.toml"), joinpath("output_$(cube_fitter.name)", "optical_options.archive.toml"), force=true)

    if cube_fitter.make_movies
        @info "===> Writing MP4 movies... (this may take a while) <==="
        make_movie(cube_fitter, cube_model)
    end

    # copy the main log file again
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    @info """\n
    #############################################################################
    ################################### Done!! ##################################
    #############################################################################
    """

    # Return the final cube_fitter object, along with the param maps/errs and cube model
    cube_fitter, param_maps, cube_model

end


"""
    fit_nuclear_template!(cube_fitter)

A helper function to fit a model to the nuclear template spectrum when decomposing a bright QSO into the QSO portion
dispersed by the PSF and the host galaxy portion. This routine produces a model for the nuclear spectrum normalized by the
nuclear PSF model, which should be continuous after the fitting procedure. 

The templates across the full field of view can then be produced by doing (nuclear model) x (PSF model) for the PSF model
in each spaxel. These should be fed into the full cube fitting routine after running this function to fully decompose the
host and the QSO.
"""
function fit_nuclear_template!(cube_fitter::CubeFitter)

    @info """\n
    BEGINNING NUCELAR TEMPLATE FITTING ROUTINE FOR $(cube_fitter.name)
    """
    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    shape = (1,1,size(cube_fitter.cube.I, 3))
    # Unlock the hot dust component for the nuclear template fits
    cube_fitter.lock_hot_dust[1] = false

    # Prepare output array
    @info "===> Preparing output data structures... <==="
    out_params = ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2) .* NaN
    out_errs = ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2, 2) .* NaN

    cube = cube_fitter.cube

    data2d = dropdims(nansum(cube.I, dims=3), dims=3)
    data2d[.~isfinite.(data2d)] .= 0.
    _, mx = findmax(data2d)

    I = zeros(Float32, shape)
    σ = zeros(Float32, shape)
    templates = zeros(Float32, shape..., cube_fitter.n_templates)
    area_sr = zeros(shape)

    # Take the brightest spaxel
    I[1,1,:] .= cube.I[mx,:]
    σ[1,1,:] .= cube.σ[mx,:]
    for s in 1:cube_fitter.n_templates
        templates[1,1,:,s] .= cube_fitter.templates[mx, :, s]
    end
    area_sr[1,1,:] .= cube_fitter.cube.Ω

    cube_data = (λ=cube_fitter.cube.λ, I=I, σ=σ, templates=templates, area_sr=area_sr)

    @debug """
    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
    """
    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    @info "===> Beginning nuclear spectrum fitting... <==="
    p_out, p_err = fit_spaxel(cube_fitter, cube_data, CartesianIndex(1,1); use_ap=true, nuc_temp_fit=true)
    if !isnothing(p_out)
        out_params[1, 1, :] .= p_out
        out_errs[1, 1, :, :] .= p_err
    end

    @info "===> Generating parameter maps and model cubes... <==="

    # Create the ParamMaps and CubeModel structs containing the outputs
    param_maps, cube_model = assign_outputs(out_params, out_errs, cube_fitter, cube_data, cube_fitter.z, true,
        nuc_temp_fit=true)

    if cube_fitter.save_fits
        @info "===> Writing FITS outputs... <==="
        write_fits(cube_fitter, cube_data, cube_model, param_maps, nuc_temp_fit=true, nuc_spax=mx)
    end

    # copy the main log file again
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    # Update the templates in perparation for the full fitting procedure
    for s in 1:cube_fitter.n_templates
        cube_fitter.templates[:, :, :, s] .= [
            cube_model.model[k,1,1] / cube_fitter.templates[mx,k,s] * cube_fitter.templates[i,j,k,s] for
                i in axes(cube_fitter.templates, 1),
                j in axes(cube_fitter.templates, 2),
                k in axes(cube_fitter.templates, 3)
            ]
        if cube_fitter.spectral_region == :MIR
            cube_fitter.templates ./= (1 .+ cube_fitter.z)
        else
            cube_fitter.templates .*= (1 .+ cube_fitter.z)
        end
    end
    cube_fitter.nuc_fit_flag[1] = true
    # Save the template amplitudes 
    for nch in 1:cube_fitter.n_channels
        tp = cube_fitter.template_names[1]
        cube_fitter.nuc_temp_amps[nch] = exp10(get(param_maps, CartesianIndex(1,1), "templates.$tp.amp_$nch"))
    end
    # lock the hot dust component for the rest of the fits
    cube_fitter.lock_hot_dust[1] = true

    @info "Done!!"

    # Return the final cube_fitter object, along with the param maps/errs and cube model
    cube_fitter, param_maps, cube_model

end


"""
    post_fit_nuclear_template!(cube_fitter, agn_templates)

A helper function to fit a model to the nuclear template spectrum when decomposing a bright QSO into the QSO portion
dispersed by the PSF and the host galaxy portion. This function is to be used AFTER fitting on the full cube has been
performed and will fit a model to the integrated QSO spectrum over the full FOV of the cube. This has the advantage of
including the normalizations of the QSO spectrum in each individual spaxel so that the overall model is consistent.
"""
function post_fit_nuclear_template!(cube_fitter::CubeFitter, agn_templates::Array{<:Real,3})

    @info """\n
    BEGINNING NUCELAR TEMPLATE FITTING ROUTINE FOR $(cube_fitter.name)
    """
    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    shape = (1,1,size(cube_fitter.cube.I, 3))
    # Unlock the hot dust component for the nuclear template fits
    cube_fitter.lock_hot_dust[1] = false

    # Prepare output array
    @info "===> Preparing output data structures... <==="
    out_params = ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2) .* NaN
    out_errs = ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2, 2) .* NaN

    cube = cube_fitter.cube

    # Get the AGN model over the whole cube
    I_agn = nansum(agn_templates, dims=(1,2)) ./ nansum(.~cube.mask, dims=(1,2))
    σ_agn = nanmedian(I_agn) .* ones(Float64, size(I_agn))  # errors will be overwritten by statistical errors
    templates = Array{Float64,4}(undef, size(I_agn)..., 0)
    area_sr = cube_fitter.cube.Ω .* nansum(cube.mask, dims=(1,2))
    I, σ, temps = fill_bad_pixels(cube_fitter, I_agn[1,1,:], σ_agn[1,1,:], templates[1,1,:,:])

    I_agn[1,1,:] .= I
    σ_agn[1,1,:] .= σ
    templates[1,1,:,:] .= temps

    cube_data = (λ=cube_fitter.cube.λ, I=I_agn, σ=σ_agn, templates=templates, area_sr=area_sr)

    @debug """
    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
    """
    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    @info "===> Beginning nuclear spectrum fitting... <==="
    # Add a minimum uncertainty to prevent the statistical uncertainties on the model from being close to 0 due to it being
    # a smooth model rather than actual data
    p_out, p_err = fit_spaxel(cube_fitter, cube_data, CartesianIndex(1,1); use_ap=true, σ_min=nanminimum(cube_fitter.cube.σ))
    if !isnothing(p_out)
        out_params[1, 1, :] .= p_out
        out_errs[1, 1, :, :] .= p_err
    end

    @info "===> Generating parameter maps and model cubes... <==="

    # Create the ParamMaps and CubeModel structs containing the outputs
    param_maps, cube_model = assign_outputs(out_params, out_errs, cube_fitter, cube_data, cube_fitter.z, true,
        nuc_temp_fit=false)

    if cube_fitter.save_fits
        @info "===> Writing FITS outputs... <==="
        write_fits(cube_fitter, cube_data, cube_model, param_maps, nuc_temp_fit=false, aperture="all")
    end
    # lock the hot dust component for the rest of the fits
    cube_fitter.lock_hot_dust[1] = true

    # copy the main log file again
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)
    @info "Done!!"

    # Return the final cube_fitter object, along with the param maps/errs and cube model
    cube_fitter, param_maps, cube_model

end

# Different methods for reading from FITS file or cube_model directly:
function post_fit_nuclear_template!(cube_fitter::CubeFitter, full_cube_model::String, template_name::String)
    hdu = FITS(full_cube_model)
    post_fit_nuclear_template!(cube_fitter, hdu, template_name)
end

function post_fit_nuclear_template!(cube_fitter::CubeFitter, full_cube_model::FITS, template_name::String)
    agn_templates = read(full_cube_model["TEMPLATES.$(uppercase(template_name))"])
    # shift back into the rest frame
    if cube_fitter.spectral_region == :MIR
        agn_templates ./= (1 .+ cube_fitter.z)
    elseif cube_fitter.spectral_region == :OPT
        agn_templates .*= (1 .+ cube_fitter.z)
    end
    post_fit_nuclear_template!(cube_fitter, agn_templates)
end

function post_fit_nuclear_template!(cube_fitter::CubeFitter, full_cube_model::CubeModel, template_index::Int=1)
    agn_templates = full_cube_model.templates[:,:,:,template_index]
    # shift back into the rest frame
    if cube_fitter.spectral_region == :MIR
        agn_templates ./= (1 .+ cube_fitter.z)
    elseif cube_fitter.spectral_region == :OPT
        agn_templates .*= (1 .+ cube_fitter.z)
    end
    post_fit_nuclear_template!(cube_fitter, agn_templates)
end

