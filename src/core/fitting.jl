#=
This is the main file for actually fitting IFU cubes.  It contains functions for actually performing the
fitting across a single spaxel and an entire cube. The main calling sequence one will want to 
perform is first loading in the data, presumably from some FITS files, with the cubedata functions,
then creating a CubeFitter struct from the DataCube struct, followed by calling fit_cube! on the
CubeFitter. An example of this is provided in the test driver files in the test directory.
=#


############################## FITTING FUNCTIONS AND HELPERS ####################################


"""
    mask_emission_lines(λ, I)

Mask out emission lines in a given spectrum using a numerical second derivative and flagging 
negative spikes (indicating strong concave-downness) up to some tolerance threshold (i.e. 3-sigma)

# Arguments
- `λ::Vector{<:Real}`: The wavelength vector of the spectrum
- `I::Vector{<:Real}`: The flux vector of the spectrum
- `Δ::Integer=3`: The half-width of the numerical second derivative approximation, in pixels
- `W::Real=0.5`: The half-width of the window with which to calculate the standard deviation of the derivative of the spectrum
    in comparison to the point in question
- `thresh::Real=3`: The threshold by which to count a spike as a line that should be masked, in units
    of sigma.
- `n_iter::Integer=2`: How many iterations to perform the derivative test after masking the previous results

See also [`continuum_cubic_spline`](@ref)
"""
function mask_emission_lines(λ::Vector{<:Real}, I::Vector{<:Real}; Δ::Integer=3, thresh::Real=3.)

    diffs = diff(λ)
    # Numerical derivative width in microns
    h = Δ * median(diffs)

    # Calculate the numerical first and second derivative
    df = zeros(length(λ))
    @simd for i ∈ 1:length(λ)
        df[i] = (I[min(length(λ), i+fld(Δ, 2))] - I[max(1, i-fld(Δ, 2))]) / h
    end
    d2f = zeros(length(λ))
    @simd for i ∈ 1:length(λ)
        d2f[i] = (I[min(length(λ), i+Δ)] - 2I[i] + I[max(1, i-Δ)]) / h^2
    end
    mask = falses(length(λ))
    W = (30, 1000)

    # Find where the second derivative is significantly concave-down 
    for j ∈ 1:length(λ)
        # Only consider the spectrum within +/- W pixels from the point in question
        if any([abs(d2f[j]) > thresh * nanstd(d2f[max(1, j-Wi):min(length(λ), j+Wi)]) for Wi ∈ W])

            # the width of the mask is based on the peaks in the numerical first derivative
            w = 10
            p₁ = nanargmax(df[max(j-w,1):min(j+w,length(df))])
            p₂ = nanargmin(df[max(j-w,1):min(j+w,length(df))])
            n_pix = 4 * (p₂ - p₁)
            n_pix = n_pix > 0 ? n_pix : 1
            # If n_pix is too wide it may not be a line, so we dont mask it
            if abs(λ[p₂] - λ[p₁])/λ[(p₁+p₂)÷2] * C_KMS > 1000
                n_pix = 0
            end
            mask[max(j-n_pix,1):min(j+n_pix,length(mask))] .= 1

        end
    end

    # Don't mask out this region that tends to trick this method sometimes
    mask[11.10 .< λ .< 11.15] .= 0
    mask[11.17 .< λ .< 11.24] .= 0
    mask[11.26 .< λ .< 11.355] .= 0
    mask[12.5 .< λ .< 12.8] .= 0

    # Clip outliers in flux -- sensitive to both positive and negative spikes
    I_med = [nanmedian(I[max(i-10Δ,1):min(i+10Δ,length(I))]) for i in 1:length(I)]
    outliers = [abs(I[i] - I_med[i]) > 2 * nanstd((I - I_med)[max(i-10Δ,1):min(i+10Δ,length(I))]) for i in 1:length(I)]
    mask .|= outliers

    # Force the beginning/end few pixels to be unmasked to prevent runaway splines at the edges
    mask[1:7] .= 0
    mask[end-7:end] .= 0

    # Return the line locations and the mask
    mask

end


"""
    continuum_cubic_spline(λ, I, σ)

Mask out the emission lines in a given spectrum using `mask_emission_lines` and replace them with
a coarse cubic spline fit to the continuum, using wide knots to avoid reinterpolating the lines or
noise.

# Arguments
- `λ::Vector{<:Real}`: The wavelength vector of the spectrum
- `I::Vector{<:Real}`: The flux vector of the spectrum
- `σ::Vector{<:Real}`: The uncertainty vector of the spectrum 
- `z::Real`: The redshift.

See also [`mask_emission_lines`](@ref)
"""
function continuum_cubic_spline(λ::Vector{<:Real}, I::Vector{<:Real}, σ::Vector{<:Real}, Δ::Integer=3, thresh::Real=3.)

    # Mask out emission lines so that they aren't included in the continuum fit
    mask_lines = mask_emission_lines(λ, I; Δ=Δ, thresh=thresh)

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
    @debug "Performing cubic spline continuum fit with knots at $λknots"

    # Do a full cubic spline interpolation of the data
    I_spline = Spline1D(λ[.~mask_lines], I[.~mask_lines], λknots, k=3, bc="extrapolate").(λ)
    σ_spline = Spline1D(λ[.~mask_lines], σ[.~mask_lines], λknots, k=3, bc="extrapolate").(λ)
    # Linear interpolation over the lines
    I_spline[mask_lines] .= Spline1D(λ[.~mask_lines], I[.~mask_lines], λknots, k=1, bc="extrapolate").(λ[mask_lines])
    σ_spline[mask_lines] .= Spline1D(λ[.~mask_lines], σ[.~mask_lines], λknots, k=1, bc="extrapolate").(λ[mask_lines])

    mask_lines, I_spline, σ_spline
end


"""
    continuum_fit_spaxel(cube_fitter, λ, I, σ, mask_lines, I_spline, σ_spline; init=init, use_ap=use_ap)

Fit the continuum of a given spaxel in the DataCube, masking out the emission lines, using the 
Levenberg-Marquardt least squares fitting method with the `CMPFit` package.  

This function has been adapted from PAHFIT (with some heavy adjustments -> masking out lines, allowing
PAH parameters to vary, and tying certain parameters together). See Smith, Draine, et al. 2007; 
http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::CartesianIndex`: The 2D index of the spaxel being fit.
- `λ::Vector{<:Real}`: The 1D wavelength vector 
- `I::Vector{<:Real}`: The 1D intensity vector
- `σ::Vector{<:Real}`: The 1D error vector
- `init::Bool=false`: Flag for the initial fit which fits the sum of all spaxels, to get an estimation for
    the initial parameter vector for individual spaxel fits
"""
function continuum_fit_spaxel(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real}, 
    σ::Vector{<:Real}, mask_lines::BitVector, N::Real; init::Bool=false, use_ap::Bool=false, bootstrap_iter::Bool=false, 
    p1_boots::Union{Vector{<:Real},Nothing}=nothing) 

    @debug """\n
    #########################################################
    ###   Beginning continuum fit for spaxel $spaxel...   ###
    #########################################################
    """

    # Mask out the spectrum with the line mask, and normalize
    λ_spax = λ[.~mask_lines]
    I_spax = I[.~mask_lines] ./ N
    σ_spax = σ[.~mask_lines] ./ N

    if !isnothing(cube_fitter.user_mask)
        # Mask out additional regions
        for pair in cube_fitter.user_mask
            region = pair[1] .< λ_spax .< pair[2]
            λ_spax = λ_spax[.~region]
            I_spax = I_spax[.~region]
            σ_spax = σ_spax[.~region]
        end
    end

    # Get the priors and "locked" booleans for each parameter, split up by the 2 steps for the continuum fit
    plims_1, plims_2, lock_1, lock_2 = get_continuum_plimits(cube_fitter)

    # Split up the initial parameter vector into the components that we need for each fitting step
    if !bootstrap_iter
        pars_1, pars_2 = get_continuum_initial_values(cube_fitter, λ_spax, I_spax, N, init || use_ap)
    else
        pars_1 = vcat(p1_boots[1:(2+2*cube_fitter.n_dust_cont+2*cube_fitter.n_power_law+4+(cube_fitter.fit_sil_emission ? 5 : 0))], 
            p1_boots[end-1:end])
        pars_2 = p1_boots[(3+2*cube_fitter.n_dust_cont+2*cube_fitter.n_power_law+4+(cube_fitter.fit_sil_emission ? 5 : 0)):end-2]
    end

    # Sort parameters by those that are locked and those that are unlocked
    p1fix = pars_1[lock_1]
    p1free = pars_1[.~lock_1]
    p2fix = pars_2[lock_2]
    p2free = pars_2[.~lock_2]

    # Count free parameters
    n_free_1 = sum(.~lock_1)
    n_free_2 = sum(.~lock_2)

    # Lower/upper bounds
    lb_1 = [pl[1] for pl in plims_1[.~lock_1]]
    ub_1 = [pl[2] for pl in plims_1[.~lock_1]]
    lb_2 = [pl[1] for pl in plims_2[.~lock_2]]
    ub_2 = [pl[2] for pl in plims_2[.~lock_2]]

    # Convert parameter limits into CMPFit object
    parinfo_1, parinfo_2, config = get_continuum_parinfo(n_free_1, n_free_2, lb_1, ub_1, lb_2, ub_2)

    @debug """\n
    ##########################################################################################################
    ########################## STEP 1 - FIT THE BLACKBODY CONTINUUM WITH PAH TEMPLATES #######################
    ##########################################################################################################
    """
    @debug "Continuum Step 1 Parameters: \n $pars_1"
    @debug "Continuum Parameters locked? \n $lock_1"
    @debug "Continuum Lower limits: \n $([pl[1] for pl in plims_1])"
    @debug "Continuum Upper limits: \n $([pl[2] for pl in plims_1])"

    @debug "Beginning continuum fitting with Levenberg-Marquardt least squares (CMPFit):"

    # Wrapper fitting function separating the free and fixed parameters
    function fit_step1(x, pfree, return_comps=false)
        ptot = zeros(Float64, length(pars_1))
        ptot[.~lock_1] .= pfree
        ptot[lock_1] .= p1fix
        if !return_comps
            model_continuum(x, ptot, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.extinction_curve, 
                cube_fitter.extinction_screen, cube_fitter.fit_sil_emission)
        else
            model_continuum(x, ptot, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.extinction_curve, 
                cube_fitter.extinction_screen, cube_fitter.fit_sil_emission, true)
        end            
    end
    res_1 = cmpfit(λ_spax, I_spax, σ_spax, fit_step1, p1free, parinfo=parinfo_1, config=config)

    @debug "Continuum CMPFit Step 1 status: $(res_1.status)"

    # Create continuum without the PAH features
    _, ccomps = fit_step1(λ_spax, res_1.param, true)

    I_cont = ccomps["stellar"]
    for i ∈ 1:cube_fitter.n_dust_cont
        I_cont .+= ccomps["dust_cont_$i"]
    end
    for j ∈ 1:cube_fitter.n_power_law
        I_cont .+= ccomps["power_law_$j"]
    end
    I_cont .*= ccomps["extinction"] .* ccomps["abs_ice"] .* ccomps["abs_ch"]
    if cube_fitter.fit_sil_emission
        I_cont .+= ccomps["hot_dust"]
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
            model_pah_residuals(x, ptot, cube_fitter.n_dust_feat, ccomps["extinction"])
        else
            model_pah_residuals(x, ptot, cube_fitter.n_dust_feat, ccomps["extinction"], true)
        end
    end
    res_2 = cmpfit(λ_spax, I_spax.-I_cont, σ_spax, fit_step2, p2free, parinfo=parinfo_2, config=config)

    @debug "Continuum CMPFit Step 2 status: $(res_2.status)"

    # Get combined best fit results
    lock = vcat(lock_1[1:end-2], lock_2)

    # Combined Best fit parameters
    popt = zeros(length(pars_1)+length(pars_2)-2)
    popt[.~lock] .= vcat(res_1.param[1:end-2], res_2.param)
    popt[lock] .= vcat(p1fix, p2fix)

    # Only bother with the uncertainties if not bootstrapping
    if !bootstrap_iter
        # Combined 1-sigma uncertainties
        perr = zeros(length(popt))
        perr[.~lock] .= vcat(res_1.perror[1:end-2], res_2.perror)
    else
        perr = zeros(length(popt))
    end

    n_free = n_free_1 + n_free_2 - 2

    @debug "Best fit continuum parameters: \n $popt"
    @debug "Continuum parameter errors: \n $perr"
    # @debug "Continuum covariance matrix: \n $covar"

    # Create the full model, again only if not bootstrapping
    I_model, comps = model_continuum_and_pah(λ, popt, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_dust_feat,
        cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.fit_sil_emission)

    if init
        cube_fitter.p_init_cont[:] .= vcat(popt, res_1.param[end-1:end])
        # Save the results to a file 
        # save running best fit parameters in case the fitting is interrupted
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_cont.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_cont, ',')
        end
    end

    # Print the results (to the logger)
    pretty_print_continuum_results(cube_fitter, popt, perr, I_spax)

    vcat(popt, res_1.param[end-1:end]), I_model, comps, n_free, vcat(perr, res_1.perror[end-1:end])
end



"""
    line_fit_spaxel(cube_fitter, spaxel, λ, I, σ, continuum, ext_curve, mask_lines, lsf_interp_func; init=init, use_ap=use_ap)

Fit the emission lines of a given spaxel in the DataCube, subtracting the continuum, using the 
Simulated Annealing and L-BFGS fitting methods with the `Optim` package.

This function has been adapted from PAHFIT (with some heavy adjustments -> lines are now fit in a 
separate second step, and no longer using the Levenberg-Marquardt method; line width and voff limits are also
adjusted to compensate for the better spectral resolution of JWST compared to Spitzer). 
See Smith, Draine, et al. 2007; http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php

# Arguments
`S<:Integer`
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::CartesianIndex`: The 2D index of the spaxel being fit.
- `λ::Vector{<:Real}`: The 1D wavelength vector 
- `I::Vector{<:Real}`: The 1D intensity vector
- `σ::Vector{<:Real}`: The 1D error vector
- `continuum::Vector{<:Real}`: The fitted continuum level of the spaxel being fit (which will be subtracted
    before the lines are fit)
- `ext_curve::Vector{<:Real}`: The extinction curve of the spaxel being fit (which will be used to calculate
    extinction-corrected line amplitudes and fluxes)
- `init::Bool=false`: Flag for the initial fit which fits the sum of all spaxels, to get an estimation for
    the initial parameter vector for individual spaxel fits
"""
function line_fit_spaxel(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real},
    σ::Vector{<:Real}, continuum::Vector{<:Real}, ext_curve::Vector{<:Real}, lsf_interp_func::Function, N::Real; 
    init::Bool=false, use_ap::Bool=false, bootstrap_iter::Bool=false, p1_boots::Union{Vector{<:Real},Nothing}=nothing)

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

    if !isnothing(cube_fitter.user_mask)
        # Mask out additional regions
        for pair in cube_fitter.user_mask
            region = pair[1] .< λnorm .< pair[2]
            λnorm = λnorm[.~region]
            Inorm = Inorm[.~region]
            σnorm = σnorm[.~region]
            ext_curve_norm = ext_curve[.~region]
        end
    end

    plimits, param_lock, param_names, tied_pairs, tied_indices = get_line_plimits(cube_fitter, ext_curve_norm, init || use_ap)
    p₀ = get_line_initial_values(cube_fitter, init || use_ap)

    # Combine all of the tied parameters
    p_tied = copy(p₀)
    plims_tied = copy(plimits)
    param_lock_tied = copy(param_lock)
    param_names_tied = copy(param_names)
    deleteat!(p_tied, tied_indices)
    deleteat!(plims_tied, tied_indices)
    deleteat!(param_lock_tied, tied_indices)
    deleteat!(param_names_tied, tied_indices)

    # Split up into free and locked parameters
    pfree_tied = p_tied[.~param_lock_tied]
    pfix_tied = p_tied[param_lock_tied]

    # Count free parameters
    n_free = sum(.~param_lock_tied)

    # Lower and upper limits on each parameter
    lower_bounds = [pl[1] for pl in plimits]
    upper_bounds = [pl[2] for pl in plimits]
    lower_bounds_tied = [pl[1] for pl in plims_tied]
    upper_bounds_tied = [pl[2] for pl in plims_tied]
    lbfree_tied = lower_bounds_tied[.~param_lock_tied]
    ubfree_tied = upper_bounds_tied[.~param_lock_tied]

    @debug "Line Parameter labels: \n $param_names_tied"
    @debug "Line starting values: \n $p_tied"

    @debug "Line Lower limits: \n $(lower_bounds_tied)"
    @debug "Line Upper Limits: \n $(upper_bounds_tied))"

    # Get CMPFit parinfo object from bounds
    parinfo, config = get_line_parinfo(n_free, lbfree_tied, ubfree_tied)

    # Wrapper function for fitting only the free, tied parameters
    function fit_step3(x, pfree_tied, func)
        ptot = zeros(Float64, length(p_tied))
        ptot[.~param_lock_tied] .= pfree_tied
        ptot[param_lock_tied] .= pfix_tied

        for tind in tied_indices
            insert!(ptot, tind, 0.)
        end
        for tie in tied_pairs
            ptot[tie[2]] = ptot[tie[1]] * tie[3]
        end
        func(x, ptot)
    end

    # if (init || use_ap || cube_fitter.fit_all_samin) && !bootstrap_iter
    if false
        @debug "Beginning Line fitting with Simulated Annealing:"

        # Parameter and function tolerance levels for convergence with SAMIN,
        # these are a bit loose since we're mainly just looking to get into the right global minimum region with SAMIN
        # before refining the fit later with a LevMar local minimum routine
        fit_func = (x, p) -> -ln_likelihood(
                                Inorm, 
                                model_line_residuals(x, p, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, 
                                    cube_fitter.flexible_wavesol, ext_curve_norm, lsf_interp_func), 
                                σnorm)
        x_tol = 1e-5
        f_tol = abs(fit_func(λnorm, p₀) - fit_func(λnorm, clamp.(p₀ .- x_tol, lower_bounds, upper_bounds)))

        # First, perform a bounded Simulated Annealing search for the optimal parameters with a generous max iterations and temperature rate (rt)
        res = Optim.optimize(p -> fit_step3(λnorm, p, fit_func), lbfree_tied, ubfree_tied, pfree_tied, 
            SAMIN(;rt=0.9, nt=5, ns=5, neps=5, f_tol=f_tol, x_tol=x_tol, verbosity=0), Optim.Options(iterations=10^6))
        
        p₁ = res.minimizer

        # Write convergence results to file, if specified
        if cube_fitter.track_convergence
            global file_lock
            # use the ReentrantLock to prevent multiple processes from trying to write to the same file at once
            lock(file_lock) do 
                open(joinpath("output_$(cube_fitter.name)", "loki.convergence.log"), "a") do conv
                    redirect_stdout(conv) do
                        println("Spaxel ($(spaxel[1]),$(spaxel[2])) on worker $(myid()):")
                        println(res)
                        println("-------------------------------------------------------")
                    end
                end
            end
        end

    elseif bootstrap_iter
        p₁ = copy(p1_boots)
        deleteat!(p₁, tied_indices)
        p₁ = p₁[.~param_lock_tied]

    else
        p₁ = pfree_tied

    end
    
    @debug "Beginning Line fitting with Levenberg-Marquardt:"

    ############################################# FIT WITH LEVMAR ###################################################

    fit_func_2 = (x, p) -> model_line_residuals(x, p, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, 
        cube_fitter.flexible_wavesol, ext_curve_norm, lsf_interp_func)
    
    χ2_before = sum((Inorm .- fit_step3(λnorm, p₁, fit_func_2)).^2 ./ σnorm.^2)
    res = cmpfit(λnorm, Inorm, σnorm, (x, p) -> fit_step3(x, p, fit_func_2), p₁, parinfo=parinfo, config=config)
    χ2_after = res.bestnorm

    # Check if CMPFit improved the reduced chi^2 at all.  If not, redo simulated annealing.
    if χ2_after ≥ χ2_before && !init && !use_ap
        @warn "The Levenberg-Marquardt solver for spaxel $spaxel did not improve the chi-squared. Re-running with simulated annealing."

        fit_func = (x, p) -> -ln_likelihood(
                                Inorm, 
                                model_line_residuals(x, p, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, 
                                    cube_fitter.flexible_wavesol, ext_curve_norm, lsf_interp_func), 
                                σnorm)
        x_tol = 1e-5
        f_tol = abs(fit_func(λnorm, p₀) - fit_func(λnorm, clamp.(p₀ .- x_tol, lower_bounds, upper_bounds)))

        # First, perform a bounded Simulated Annealing search for the optimal parameters with a generous max iterations and temperature rate (rt)
        res = Optim.optimize(p -> fit_step3(λnorm, p, fit_func), lbfree_tied, ubfree_tied, pfree_tied, 
            SAMIN(;rt=0.9, nt=5, ns=5, neps=5, f_tol=f_tol, x_tol=x_tol, verbosity=0), Optim.Options(iterations=10^6))
        p₁ = res.minimizer

        res = cmpfit(λnorm, Inorm, σnorm, (x, p) -> fit_step3(x, p, fit_func_2), p₁, parinfo=parinfo, config=config)
    end 

    @debug "Line CMPFit status: $(res.status)"

    # Get the results and errors
    popt = zeros(Float64, length(p_tied))
    popt[.~param_lock_tied] .= res.param
    popt[param_lock_tied] .= pfix_tied
    for tind in tied_indices
        insert!(popt, tind, 0.)
    end
    for tie in tied_pairs
        popt[tie[2]] = popt[tie[1]] * tie[3]
    end

    # Dont both with uncertainties if bootstrapping
    if !bootstrap_iter
        perr = zeros(Float64, length(p_tied))
        perr[.~param_lock_tied] .= res.perror
        for tind in tied_indices
            insert!(perr, tind, 0.)
        end
        for tie in tied_pairs
            perr[tie[2]] = perr[tie[1]] * tie[3]
        end
    else
        perr = zeros(Float64, length(popt))
    end

    ######################################################################################################################

    @debug "Best fit line parameters: \n $popt"
    @debug "Line parameter errors: \n $perr"
    # @debug "Line covariance matrix: \n $covar"

    # Final optimized fit
    I_model, comps = model_line_residuals(λ, popt, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, 
        cube_fitter.flexible_wavesol, ext_curve, lsf_interp_func, true)

    if init
        cube_fitter.p_init_line[:] .= copy(popt)
        # Save results to file
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_line.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_line, ',')
        end
    end

    # Log the fit results
    pretty_print_line_results(cube_fitter, popt, perr)

    popt, I_model, comps, n_free, perr
end


"""
    plot_spaxel_fit(λ, I, I_cont, σ, comps, n_dust_cont, n_dust_features, line_wave, line_names, screen, z, χ2red, 
        name, label; backend=backend)

Plot the fit for an individual spaxel, `I_cont`, and its individual components `comps`, using the given 
backend (`:pyplot` or `:plotly`).

# Arguments
`T<:Real,S<:Integer`
- `λ::Vector{<:Real}`: The wavelength vector of the spaxel to be plotted
- `I::Vector{<:Real}`: The intensity data vector of the spaxel to be plotted
- `I_cont::Vector{<:Real}`: The intensity model vector of the spaxel to be plotted
- `σ::Vector{<:Real}`: The uncertainty vector of the spaxel to be plotted
- `comps::Dict{String, Vector{T}}`: The dictionary of individual components of the model intensity
- `n_dust_cont::Integer`: The number of dust continuum components in the fit
- `n_dust_features::Integer`: The number of PAH features in the fit
- `line_wave::Vector{<:Real}`: List of nominal central wavelengths for each line in the fit
- `line_names::Vector{Symbol}`: List of names for each line in the fit
- `screen::Bool`: The type of model used for extinction screening
- `z::Real`: The redshift of the object being fit
- `χ2red::Real`: The reduced χ^2 value of the fit
- `name::String`: The name of the object being fit
- `label::String`: A label for the individual spaxel being plotted, to be put in the file name
- `backend::Symbol`: The backend to use to plot, either `:pyplot` or `:plotly`
"""
function plot_spaxel_fit(λ::Vector{<:Real}, I::Vector{<:Real}, I_cont::Vector{<:Real}, σ::Vector{<:Real}, comps::Dict{String, Vector{T}}, 
    n_dust_cont::Integer, n_power_law::Integer, n_dust_features::Integer, n_comps::Integer, line_wave::Vector{<:Real}, line_names::Vector{Symbol}, screen::Bool, 
    z::Real, χ2red::Real, name::String, label::String; backend::Symbol=:pyplot, I_boot_min::Union{Vector{<:Real},Nothing}=nothing, 
    I_boot_max::Union{Vector{<:Real},Nothing}=nothing, range::Union{Tuple,Nothing}=nothing, spline::Union{Vector{<:Real},Nothing}=nothing) where {T<:Real}

    fit_sil_emission = haskey(comps, "hot_dust")

    # Plotly ---> useful interactive plots for visually inspecting data, but not publication-quality
    if (backend == :plotly || backend == :both) && isnothing(range)
        # Plot the overall data / model
        trace1 = PlotlyJS.scatter(x=λ, y=I, mode="lines", line=Dict(:color => "black", :width => 1), name="Data", showlegend=true)
        trace2 = PlotlyJS.scatter(x=λ, y=I_cont, mode="lines", line=Dict(:color => "red", :width => 1), name="Continuum Fit", showlegend=true)
        traces = [trace1, trace2]
        # Loop over and plot individual model components
        for comp ∈ keys(comps)
            if comp == "extinction"
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["abs_ice"] .* comps["abs_ch"] .* maximum(I_cont) .* 1.1, mode="lines", 
                    line=Dict(:color => "black", :width => 1, :dash => "dash"), name="Extinction")])
            elseif occursin("hot_dust", comp)
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp], mode="lines", line=Dict(:color => "yellow", :width => 1),
                    name="Hot Dust")])
            elseif occursin("line", comp)
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"], mode="lines",
                    line=Dict(:color => "rebeccapurple", :width => 1), name="Lines")])
            end
        end
        # Add vertical dashed lines for emission line rest wavelengths
        for (lw, ln) ∈ zip(line_wave, line_names)
            append!(traces, [PlotlyJS.scatter(x=[lw, lw], y=[0., nanmaximum(I)*1.1], mode="lines", 
                line=Dict(:color => occursin("H2", String(ln)) ? "red" : 
                              (any(occursin.(["alpha", "beta", "gamma", "delta"], String(ln))) ? "#ff7f0e" : "rebeccapurple"), 
                          :width => 0.5, :dash => "dash"))])
        end
        # Add the summed up continuum
        append!(traces, [PlotlyJS.scatter(x=λ, y=comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"] .* (
            (n_dust_cont > 0 ? sum([comps["dust_cont_$i"] for i ∈ 1:n_dust_cont], dims=1)[1] : zeros(length(λ))) .+ 
            (n_power_law > 0 ? sum([comps["power_law_$j"] for j ∈ 1:n_power_law], dims=1)[1] : zeros(length(λ))) .+ comps["stellar"]),
            mode="lines", line=Dict(:color => "green", :width => 1), name="Dust+Stellar Continuum")])
        # Summed up PAH features
        append!(traces, [PlotlyJS.scatter(x=λ, y=sum([comps["dust_feat_$i"] for i ∈ 1:n_dust_features], dims=1)[1] .* comps["extinction"],
            mode="lines", line=Dict(:color => "blue", :width => 1), name="PAHs")])
        # Individual PAH features
        for i in 1:n_dust_features
            append!(traces, [PlotlyJS.scatter(x=λ, y=comps["dust_feat_$i"] .* comps["extinction"], mode="lines", line=Dict(:color => "blue", :width => 1), name="PAHs")])
        end
        if !isnothing(spline)
            append!(traces, [PlotlyJS.scatter(x=λ, y=spline, mode="lines", line=Dict(:color => "red", :width => 1, :dash => "dash"), name="Cubic Spline")])
        end
        # axes labels / titles / fonts
        layout = PlotlyJS.Layout(
            xaxis_title=L"$\lambda\ (\mu{\rm m})$",
            yaxis_title=L"$I_{\nu}\ ({\rm MJy}\,{\rm sr}^{-1})$",
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
        power = floor(Int, log10(maximum(I ./ λ)))
        if power ≥ 4
            norm = 10^power
        else
            norm = 1
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

        ax1.plot(λ, I ./ norm ./ λ, "k-", label="Data")

        # plot cubic spline
        if !isnothing(spline)
            ax1.plot(λ, spline ./ norm ./ λ, color="#2ca02c", linestyle="--", label="Cubic Spline")
        end

        ax1.plot(λ, I_cont ./ norm ./ λ, "-", color="#ff5d00", label="Model")
        if !isnothing(I_boot_min) && !isnothing(I_boot_max)
            ax1.fill_between(λ, I_boot_min ./ norm ./ λ, I_boot_max ./ norm ./ λ, color="#ff5d00", alpha=0.5, zorder=10)
        end

        ax2.plot(λ, (I.-I_cont) ./ norm ./ λ, "k-")

        χ2_str = @sprintf "%.3f" χ2red
        ax2.plot(λ, zeros(length(λ)), "-", color="#ff5d00", label=L"$\tilde{\chi}^2 = %$χ2_str$")
        if !isnothing(I_boot_min) && !isnothing(I_boot_max)
            ax2.fill_between(λ, (I_boot_min .- I_cont) ./ norm ./ λ, (I_boot_max .- I_cont) ./ norm ./ λ, color="#ff5d00", alpha=0.5,
                zorder=10)
        end
        # ax2.fill_between(λ, (I.-I_cont.+σ)./norm./λ, (I.-I_cont.-σ)./norm./λ, color="k", alpha=0.5)

        # twin axes with different labels --> extinction for ax3 and observed wavelength for ax4
        ax3 = ax1.twinx()
        ax4 = ax1.twiny()

        # full continuum
        ext_full = comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"] 
        ax1.plot(λ, ext_full .* (
            (n_dust_cont > 0 ? sum([comps["dust_cont_$i"] for i ∈ 1:n_dust_cont], dims=1)[1] : zeros(length(λ))) .+ 
            (n_power_law > 0 ? sum([comps["power_law_$j"] for j ∈ 1:n_power_law], dims=1)[1] : zeros(length(λ))) .+ 
            comps["stellar"] .+ (fit_sil_emission ? comps["hot_dust"] : zeros(length(λ)))
            ) ./ norm ./ λ, "k-", lw=2, alpha=0.5, label="Continuum")
        # individual continuum components
        ax1.plot(λ, comps["stellar"] .* ext_full ./ norm ./ λ, "m--", alpha=0.5, label="Stellar continuum")
        for i in 1:n_dust_cont
            ax1.plot(λ, comps["dust_cont_$i"] .* ext_full ./ norm ./ λ, "k-", alpha=0.5, label="Dust continuum")
        end
        # full PAH profile
        ax1.plot(λ, sum([comps["dust_feat_$i"] for i ∈ 1:n_dust_features], dims=1)[1] .* comps["extinction"] ./ norm ./ λ, "-", 
            color="#0065ff", label="PAHs")
        # full line profile
        ax1.plot(λ, sum([haskey(comps, "line_$(i)_$(j)") ? comps["line_$(i)_$(j)"] : zeros(length(λ)) 
            for i ∈ 1:length(line_wave), j ∈ 1:n_comps], dims=(1,2))[1] .* comps["extinction"] ./ norm ./ λ, "-", 
            color="rebeccapurple", alpha=0.6, label="Lines")
        # plot extinction
        ax3.plot(λ, comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"], "k:", alpha=0.5, label="Extinction")
        # plot hot dust
        if haskey(comps, "hot_dust")
            ax1.plot(λ, comps["hot_dust"] ./ norm ./ λ, "-", color="#8ac800", alpha=0.6, label="Hot Dust")
        end

        # plot vertical dashed lines for emission line wavelengths
        for (lw, ln) ∈ zip(line_wave, line_names)
            ax1.axvline(lw, linestyle="--", color="k", lw=0.5, alpha=0.5)
            ax2.axvline(lw, linestyle="--", color="k", lw=0.5, alpha=0.5)
        end

        line_mask = falses(length(λ))
        for ln ∈ line_wave
            window_size = 3000. / C_KMS * ln
            window = (ln - window_size) .< λ .< (ln + window_size)
            line_mask .|= window
        end
        # set axes limits and labels
        if isnothing(range)
            λmin, λmax = minimum(λ), maximum(λ)
            ax1.set_xlim(λmin, λmax)
            ax2.set_xlim(λmin, λmax)
            ax4.set_xlim(λmin * (1 + z), λmax * (1 + z))
            ax1.set_ylim(-0.01, 1.3nanmaximum(I[.!line_mask] ./ norm ./ λ[.!line_mask]))
        else
            ax1.set_xlim(range[1], range[2])
            ax2.set_xlim(range[1], range[2])
            ax4.set_xlim(range[1] * (1 + z), range[2] * (1 + z))
            ax1.set_ylim(-0.01, 1.1nanmaximum((I ./ norm ./ λ)[range[1] .< λ .< range[2]]))
        end
        ax2.set_ylim(-1.1maximum(((I.-I_cont) ./ norm ./ λ)[.!line_mask]), 1.1maximum(((I.-I_cont) ./ norm ./ λ)[.!line_mask]))
        ax3.set_yscale("log") # logarithmic extinction axis
        ax3.set_ylim(1e-3, 1.)
        if screen
            ax3.set_ylabel(L"$e^{-\tau_{\lambda}}$")
        else
            ax3.set_ylabel(L"$(1-e^{-\tau_{\lambda}}) / \tau_{\lambda}$")
        end
        if power ≥ 4
            ax1.set_ylabel(L"$I_{\nu}/\lambda$ ($10^{%$power}$ MJy sr$^{-1}$ $\mu$m$^{-1}$)")
        else
            ax1.set_ylabel(L"$I_{\nu}/\lambda$ (MJy sr$^{-1}$ $\mu$m$^{-1}$)")
        end
        ax2.set_ylabel(L"$O-C$")  # ---> residuals, (O)bserved - (C)alculated
        ax2.set_xlabel(L"$\lambda_{\rm rest}$ ($\mu$m)")
        ax4.set_xlabel(L"$\lambda_{\rm obs}$ ($\mu$m)")
        ax2.legend(loc="upper left")

        # Set minor ticks as multiples of 0.1 μm for x axis and automatic for y axis
        ax1.xaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax1.yaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax2.xaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax2.yaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax4.xaxis.set_minor_locator(py_ticker.AutoMinorLocator())

        # Set major ticks and formats
        ax1.set_xticklabels([]) # ---> will be covered up by the residuals plot
        ax2.set_yticks([-round(maximum(((I.-I_cont) ./ norm ./ λ)[.!line_mask]) / 2, sigdigits=1), 0.0, round(maximum(((I.-I_cont) ./ norm ./ λ)[.!line_mask]) / 2, sigdigits=1)])
        ax1.tick_params(which="both", axis="both", direction="in")
        ax2.tick_params(which="both", axis="both", direction="in", labelright=true, right=true, top=true)
        ax3.tick_params(which="both", axis="both", direction="in")
        ax4.tick_params(which="both", axis="both", direction="in")
        
        # Save figure as PDF, yay for vector graphics!
        plt.savefig(isnothing(label) ? joinpath("output_$name", "spaxel_plots", "levmar_fit_spaxel.pdf") : 
            joinpath("output_$name", isnothing(range) ? "spaxel_plots" : "zoomed_plots", "$label.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    end
end


# Helper function for fitting one iteration (i.e. for bootstrapping)
function _fit_spaxel_iterfunc(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real}, 
    σ::Vector{<:Real}, norm::Real, area_sr::Vector{<:Real}, mask_lines::BitVector, I_spline::Vector{<:Real}; bootstrap_iter::Bool=false, 
    p1_boots_c::Union{Vector{<:Real},Nothing}=nothing, p1_boots_l::Union{Vector{<:Real},Nothing}=nothing, 
    use_ap::Bool=false, init::Bool=false)

    # Interpolate the LSF
    lsf_interp = Spline1D(λ, cube_fitter.cube.lsf, k=1)
    lsf_interp_func = x -> lsf_interp(x)    # Interpolate the LSF
    lsf_interp = Spline1D(λ, cube_fitter.cube.lsf, k=1)
    lsf_interp_func = x -> lsf_interp(x)

    # Fit the spaxel
    popt_c, I_cont, comps_cont, n_free_c, perr_c = 
        @timeit timer_output "continuum_fit_spaxel" continuum_fit_spaxel(cube_fitter, spaxel, λ, I, σ, mask_lines, norm, use_ap=use_ap,
        init=init, bootstrap_iter=bootstrap_iter, p1_boots=p1_boots_c)
    popt_l, I_line, comps_line, n_free_l, perr_l = 
        @timeit timer_output "line_fit_spaxel" line_fit_spaxel(cube_fitter, spaxel, λ, I, σ, I_cont, comps_cont["extinction"], 
        lsf_interp_func, norm, use_ap=use_ap, init=init, bootstrap_iter=bootstrap_iter, p1_boots=p1_boots_l)

    # Combine the continuum and line models
    I_model = I_cont .+ I_line
    comps = merge(comps_cont, comps_line)

    # Renormalize
    I_model .*= norm
    for comp ∈ keys(comps)
        if !(comp ∈ ["extinction", "abs_ice", "abs_ch"])
            comps[comp] .*= norm
        end
    end

    # Total free parameters
    n_free = n_free_c + n_free_l
    n_data = length(I)
    n_masked = 0
    if !isnothing(cube_fitter.user_mask)
        for pair in cube_fitter.user_mask
            region = pair[1] .< λ .< pair[2]
            n_masked += sum(region)
        end
    end
    n_data -= n_masked

    # Degrees of freedom
    dof = n_data - n_free

    # chi^2 and reduced chi^2 of the model
    χ2 = sum(@. (I - I_model)^2 / σ^2)

    # Add dust feature and line parameters (intensity and SNR)
    if !init
        p_dust, p_lines, p_dust_err, p_lines_err = 
            @timeit timer_output "calculate_extra_parameters" calculate_extra_parameters(λ, I, norm, cube_fitter.n_dust_cont,
                cube_fitter.n_power_law, cube_fitter.n_dust_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, 
                cube_fitter.fit_sil_emission, cube_fitter.n_lines, cube_fitter.n_acomps, cube_fitter.n_comps, cube_fitter.lines, 
                cube_fitter.flexible_wavesol, lsf_interp_func, popt_c[1:end-2], popt_l, perr_c[1:end-2], perr_l, comps["extinction"], 
                mask_lines, I_spline, area_sr, !bootstrap_iter)
        p_out = [popt_c[1:end-2]; popt_l; p_dust; p_lines; χ2; dof]
        p_err = [perr_c[1:end-2]; perr_l; p_dust_err; p_lines_err; 0.; 0.]
        
        return p_out, p_err, popt_c, popt_l, perr_c, perr_l, I_model, comps, χ2, dof
    end
    return I_model, comps, χ2, dof
end


"""
    fit_spaxel(cube_fitter, spaxel; aperture=nothing, plot_spline=false)

Wrapper function to perform a full fit of a single spaxel, calling `continuum_fit_spaxel` and `line_fit_spaxel` and
concatenating the best-fit parameters. The outputs are also saved to files so the fit need not be repeated in the case
of a crash.

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::CartesianIndex`: The coordinates of the spaxel to be fit
- `aperture=nothing`: If specified, perform a fit to the integrated spectrum within the aperture. Must be an `Aperture` object
from the `photutils` python package (using PyCall).
"""
function fit_spaxel(cube_fitter::CubeFitter, cube_data::NamedTuple, spaxel::CartesianIndex; use_ap::Bool=false)

    λ = cube_data.λ
    I = cube_data.I[spaxel, :]
    σ = cube_data.σ[spaxel, :]
    area_sr = cube_data.area_sr

    # if there are any NaNs, skip over the spaxel
    if any(.!isfinite.(I))
        return nothing, nothing
    end

    # Perform a cubic spline fit, also obtaining the line mask
    mask_lines, I_spline, σ_spline = continuum_cubic_spline(λ, I, σ)

    # l_mask = sum(.!mask_lines)
    # # Statistical uncertainties based on the local RMS of the residuals with a cubic spline fit
    # σ_stat = [std(I[.!mask_lines][max(i-30,1):min(i+30,l_mask)] .- I_spline[.!mask_lines][max(i-30,1):min(i+30,l_mask)]) for i ∈ 1:l_mask]
    # # We insert at the locations of the lines since the cubic spline does not include them
    # l_all = length(λ)
    # line_inds = (1:l_all)[mask_lines]
    # for line_ind ∈ line_inds
    #     insert!(σ_stat, line_ind, σ_stat[max(line_ind-1, 1)])
    # end
    # @debug "Statistical uncertainties: ($(σ_stat[1]) - $(σ_stat[end]))"
    # σ = hypot.(σ, σ_stat)

    resid = I[.!mask_lines] .- I_spline[.!mask_lines]
    σ_stat = std(resid[resid .< 3std(resid)])
    σ .= σ_stat

    # Check if the fit has already been performed
    if !isfile(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).csv")) || cube_fitter.overwrite
        
        # Create a local logger for this individual spaxel
        timestamp_logger(logger) = TransformerLogger(logger) do log
            merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
        end
        # This log should be entirely handled by 1 process, since each spaxel is entirely handled by 1 process
        # so there should be no problems with I/O race conditions
        spaxel_logger = TeeLogger(ConsoleLogger(stdout, Logging.Info), timestamp_logger(MinLevelLogger(FileLogger(
                             joinpath("output_$(cube_fitter.name)", "logs", "loki.spaxel_$(spaxel[1])_$(spaxel[2]).log"); 
                             always_flush=true), Logging.Debug)))

        p_out, p_err = with_logger(spaxel_logger) do

            # Use a fixed normalization for the line fits so that the bootstrapped amplitudes are consistent with each other
            norm = Float64(abs(nanmaximum(I)))
            norm = norm ≠ 0. ? norm : 1.

            # Perform the regular fit
            p_out, p_err, popt_c, popt_l, perr_c, perr_l, I_model, comps, χ2, dof = _fit_spaxel_iterfunc(
                cube_fitter, spaxel, λ, I, σ, norm, area_sr, mask_lines, I_spline; bootstrap_iter=false, use_ap=use_ap)
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
                p_boot = SharedArray(zeros(length(p_out), cube_fitter.n_bootstrap))
                # Initialize bootstrap model array
                I_model_boot = SharedArray(zeros(length(I_model), cube_fitter.n_bootstrap))

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
                        continuum_cubic_spline(λ, I, σ)
                    end
                    # (do not recalculate sigma_stat since it would be increased by a factor of sqrt(2), but in reality
                    # we would have to divide out the sqrt(2) because we now have 2 "measurements")

                    # Re-perform the fitting on the resampled data
                    pb_i, _, _, _, _, _, Ib_i, _, _, _ = with_logger(NullLogger()) do
                        _fit_spaxel_iterfunc(cube_fitter, spaxel, λ, I_boot, σ, norm, area_sr, mask_lines_boot, I_spline_boot; 
                            bootstrap_iter=true, p1_boots_c=popt_c, p1_boots_l=popt_l, use_ap=use_ap)
                    end
                    p_boot[:, nboot] .= pb_i
                    I_model_boot[:, nboot] .= Ib_i
                end

                # RESULTS: Values are the 50th percentile, and errors are the (15.9th, 84.1st) percentiles
                p_out = dropdims(nanquantile(p_boot, 0.50, dims=2), dims=2)
                p_err_lo = p_out .- dropdims(nanquantile(p_boot, 0.159, dims=2), dims=2)
                p_err_up = dropdims(nanquantile(p_boot, 0.841, dims=2), dims=2) .- p_out
                p_err = [p_err_lo p_err_up]

                # Get the minimum/maximum pointwise bootstrapped models
                I_boot_min = dropdims(nanminimum(I_model_boot, dims=2), dims=2)
                I_boot_max = dropdims(nanmaximum(I_model_boot, dims=2), dims=2)

                split1 = length(popt_c) - 2
                split2 = length(popt_c) - 2 + length(popt_l)
                lsf_interp = Spline1D(λ, cube_fitter.cube.lsf, k=1)
                lsf_interp_func = x -> lsf_interp(x)

                # Replace the best-fit model with the 50th percentile model to be consistent with p_out
                I_boot_cont, comps_boot_cont = model_continuum_and_pah(λ, p_out[1:split1], norm, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, 
                    cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.fit_sil_emission, true)
                I_boot_line, comps_boot_line = model_line_residuals(λ, p_out[split1+1:split2], cube_fitter.n_lines, cube_fitter.n_comps,
                    cube_fitter.lines, cube_fitter.flexible_wavesol, comps_boot_cont["extinction"], lsf_interp_func, true)

                # Reconstruct the full model
                I_model = I_boot_cont .+ I_boot_line
                comps = merge(comps_boot_cont, comps_boot_line)

                # Renormalize
                I_model .*= norm
                for comp ∈ keys(comps)
                    if !(comp ∈ ["extinction", "abs_ice", "abs_ch"])
                        comps[comp] .*= norm
                    end
                end

                # Recalculate chi^2 based on the median model
                p_out[end-1] = sum(@. (I - I_model)^2/σ^2)
                χ2 = p_out[end-1]

            end

            # Plot the fit
            if cube_fitter.plot_spaxels != :none
                @debug "Plotting spaxel $spaxel best fit"
                @timeit timer_output "plot_spaxel_fit" plot_spaxel_fit(λ, I, I_model, σ, comps, 
                    cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_dust_feat, cube_fitter.n_comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, 
                    cube_fitter.extinction_screen, cube_fitter.z, χ2/dof, cube_fitter.name, "spaxel_$(spaxel[1])_$(spaxel[2])", backend=cube_fitter.plot_spaxels,
                    I_boot_min=I_boot_min, I_boot_max=I_boot_max)
                if !isnothing(cube_fitter.plot_range)
                    for (i, plot_range) ∈ enumerate(cube_fitter.plot_range)
                        @timeit timer_output "plot_line_fit" plot_spaxel_fit(λ, I, I_model, σ, comps,
                            cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_dust_feat, cube_fitter.n_comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, 
                            cube_fitter.extinction_screen, cube_fitter.z, χ2/dof, cube_fitter.name, "lines_$(spaxel[1])_$(spaxel[2])_$i", backend=cube_fitter.plot_spaxels,
                            I_boot_min=I_boot_min, I_boot_max=I_boot_max, range=plot_range)
                    end
                end
            end

            @debug "Saving results to csv for spaxel $spaxel"
            # serialize(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).LOKI"), (p_out=p_out, p_err=p_err))
            # save output as csv file
            open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).csv"), "w") do f 
                @timeit timer_output "writedlm" writedlm(f, [p_out p_err], ',')
            end
 
            # save memory allocations & other logistic data to a separate log file
            if cube_fitter.track_memory
                open(joinpath("output_$(cube_fitter.name)", "logs", "mem.spaxel_$(spaxel[1])_$(spaxel[2]).log"), "w") do f

                    print(f, """
                    ### PROCESS ID: $(getpid()) ###
                    Memory usage stats:
                    CubeFitter - $(Base.summarysize(cube_fitter) ÷ 10^6) MB
                        Cube - $(Base.summarysize(cube_fitter.cube) ÷ 10^6) MB 
                    """)

                    print(f, """
                    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
                    """)

                    print_timer(f, timer_output, sortby=:name)
                end
            end

            # Overwrite the errors with the statistical errors
            cube_data.σ[spaxel, :] .= σ

            p_out, p_err
        end

        return p_out, p_err

    end

    # Otherwise, just grab the results from before
    results = readdlm(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).csv"), ',', Float64, '\n')
    p_out = results[:, 1]
    p_err = results[:, 2:3]

    # Still need to overwrite the raw errors with the statistical errors
    cube_data.σ[spaxel, :] .= σ

    p_out, p_err

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
    I_sum_init = sumdim(cube_fitter.cube.Iν, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    σ_sum_init = sqrt.(sumdim(cube_fitter.cube.σI.^2, (1,2))) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    area_sr_init = cube_fitter.cube.Ω .* sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))

    bad = findall(.~isfinite.(I_sum_init) .| .~isfinite.(σ_sum_init))
    # Replace with the average of the points to the left and right
    l = length(I_sum_init)
    for badi in bad
        lind = findfirst(x -> isfinite(x), I_sum_init[max(badi-1,1):-1:1])
        rind = findfirst(x -> isfinite(x), I_sum_init[min(badi+1,l):end])
        I_sum_init[badi] = (I_sum_init[max(badi-1,1):-1:1][lind] + I_sum_init[min(badi+1,l):end][rind]) / 2
        σ_sum_init[badi] = (σ_sum_init[max(badi-1,1):-1:1][lind] + σ_sum_init[min(badi+1,l):end][rind]) / 2
    end
    @assert all(isfinite.(I_sum_init) .& isfinite.(σ_sum_init)) "Error: Non-finite values found in the summed intensity/error arrays!"

    # Perform a cubic spline fit, also obtaining the line mask
    mask_lines_init, I_spline_init, σ_spline_init = continuum_cubic_spline(λ_init, I_sum_init, σ_sum_init)
    # l_mask = sum(.!mask_lines_init)

    # # Statistical uncertainties based on the local RMS of the residuals with a cubic spline fit
    # σ_stat_init = [std(I_sum_init[.!mask_lines_init][max(i-100,1):min(i+100,l_mask)] .- 
    #     I_spline_init[.!mask_lines_init][max(i-100,1):min(i+100,l_mask)]) for i ∈ 1:l_mask]
    # # We insert at the locations of the lines since the cubic spline does not include them
    # l_all = length(λ_init)
    # line_inds = (1:l_all)[mask_lines_init]
    # for line_ind ∈ line_inds
    #     insert!(σ_stat_init, line_ind, σ_stat_init[max(line_ind-1, 1)])
    # end
    # @debug "Statistical uncertainties: ($(σ_stat_init[1]) - $(σ_stat_init[end]))"
    # σ_sum_init = hypot.(σ_sum_init, σ_stat_init)
    resid = I_sum_init[.!mask_lines_init] .- I_spline_init[.!mask_lines_init]
    σ_stat_init = std(resid[resid .< 3std(resid)])
    σ_sum_init .= σ_stat_init
    
    # Get the normalization
    norm = abs(nanmaximum(I_sum_init))
    norm = norm ≠ 0. ? norm : 1.

    I_model_init, comps_init, χ2_init, dof_init = _fit_spaxel_iterfunc(
        cube_fitter, CartesianIndex(0,0), λ_init, I_sum_init, σ_sum_init, norm, area_sr_init, mask_lines_init, I_spline_init; 
        bootstrap_iter=false, use_ap=false, init=true)

    χ2red_init = χ2_init / dof_init

    # Plot the fit
    if cube_fitter.plot_spaxels != :none
        @debug "Plotting spaxel sum initial fit"
        plot_spaxel_fit(λ_init, I_sum_init, I_model_init, σ_sum_init, comps_init,
            cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_dust_feat, cube_fitter.n_comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, 
            cube_fitter.extinction_screen, cube_fitter.z, χ2red_init, cube_fitter.name, "initial_sum_fit", backend=cube_fitter.plot_spaxels)
        if !isnothing(cube_fitter.plot_range)
            for (i, plot_range) ∈ enumerate(cube_fitter.plot_range)
                plot_spaxel_fit(λ_init, I_sum_init, I_model_init, σ_sum_init, comps_init,
                    cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_dust_feat, cube_fitter.n_comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, 
                    cube_fitter.extinction_screen, cube_fitter.z, χ2red_init, cube_fitter.name, "initial_sum_line_$i", backend=cube_fitter.plot_spaxels;
                    range=plot_range)
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

    shape = size(cube_fitter.cube.Iν)

    # Prepare output array
    @info "===> Preparing output data structures... <==="
    out_params = SharedArray(ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2) .* NaN)
    out_errs = SharedArray(ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2, 2) .* NaN)
    # "cube_data" object holds the primary wavelength, intensity, and errors
    # this is just a convenience object since these may be different when fitting an integrated spectrum
    # within an aperture
    cube_data = (λ=cube_fitter.cube.λ, I=cube_fitter.cube.Iν, σ=cube_fitter.cube.σI, area_sr=cube_fitter.cube.Ω .* ones(shape[3]))

    ######################### DO AN INITIAL FIT WITH THE SUM OF ALL SPAXELS ###################

    @debug """
    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
    """

    # Don't repeat if it's already been done, and also dont do the initial fit if we're just fitting in an aperture
    if all(iszero.(cube_fitter.p_init_cont))
        fit_stack!(cube_fitter)
    else
        @info "===> Initial fit to the sum of all spaxels is being skipped, either because it has already " *
            "been performed, or an aperture was specified <==="
    end

    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    ##############################################################################################

    # Get the indices of all spaxels
    spaxels = CartesianIndices(selectdim(cube_fitter.cube.Iν, 3, 1))

    # Wrapper function 
    function fit_spax_i(index::CartesianIndex)

        p_out, p_err = fit_spaxel(cube_fitter, cube_data, index)
        if !isnothing(p_out)
            out_params[index, :] .= p_out
            out_errs[index, :, :] .= p_err
        end

        return
    end

    # Use multiprocessing (not threading) to iterate over multiple spaxels at once using multiple CPUs
    if cube_fitter.parallel
        @info "===> Beginning individual spaxel fitting... <==="
        prog = Progress(length(spaxels); showspeed=true)
        progress_pmap(spaxels, progress=prog) do index
            fit_spax_i(index)
        end
    else
        @info "===> Beginning individual spaxel fitting... <==="
        prog = Progress(length(spaxels); showspeed=true)
        for index ∈ spaxels
            fit_spax_i(index)
            next!(prog)
        end
    end

    @info "===> Generating parameter maps and model cubes... <==="

    # Create the ParamMaps and CubeModel structs containing the outputs
    param_maps, param_errs, cube_model = assign_outputs(out_params, out_errs, cube_fitter, cube_data, spaxels, cube_fitter.z, false)

    if cube_fitter.plot_maps
        @info "===> Plotting parameter maps... <==="
        plot_parameter_maps(cube_fitter, param_maps)
    end

    if cube_fitter.save_fits
        @info "===> Writing FITS outputs... <==="
        write_fits(cube_fitter, cube_data, cube_model, param_maps, param_errs)
    end

    # Save a copy of the options file used to run the code, so the settings can be referenced/reused
    # (for example, if you need to recall which lines you tied, what your limits were, etc.)
    cp(joinpath(@__DIR__, "..", "options", "options.toml"), joinpath("output_$(cube_fitter.name)", "general_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "dust.toml"), joinpath("output_$(cube_fitter.name)", "dust_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "lines.toml"), joinpath("output_$(cube_fitter.name)", "lines_options.archive.toml"), force=true)

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
    cube_fitter, param_maps, param_errs, cube_model

end


function fit_cube!(cube_fitter::CubeFitter, aperture::PyObject)
    # Extend the single aperture into an array of apertures and call the corresponding method of fit_cube!
    apertures = repeat([aperture], length(cube_fitter.cube.λ))
    fit_cube!(cube_fitter, apertures)
end


function fit_cube!(cube_fitter::CubeFitter, aperture::Vector{PyObject})

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

    shape = (1,1,size(cube_fitter.cube.Iν, 3))

    # Prepare output array
    @info "===> Preparing output data structures... <==="
    out_params = SharedArray(ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2) .* NaN)
    out_errs = SharedArray(ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2, 2) .* NaN)

    # If using an aperture, overwrite the cube_data object with the quantities within
    # the aperture, which are calculated here.
    # Prepare the 1D arrays
    I = zeros(Float32, shape)
    σ = zeros(Float32, shape)
    area_sr = zeros(shape[3])
    @debug "Performing aperture photometry to get the integrated spectrum"

    # Loop through each wavelength pixel and perform the aperture photometry
    for z ∈ 1:shape[3]
        # Sum up the FLUX within the aperture
        Fz = cube_fitter.cube.Iν[:, :, z] .* cube_fitter.cube.Ω
        e_Fz = cube_fitter.cube.σI[:, :, z] .* cube_fitter.cube.Ω
        _, _, _, F_ap, eF_ap = py_photutils.aperture.aperture_photometry(Fz', aperture[z], 
            error=e_Fz', mask=cube_fitter.cube.mask[:, :, z]', method="exact")[1]

        # Convert back to intensity by dividing out the aperture area
        area_sr[z] = aperture[z].area * cube_fitter.cube.Ω
        I[1,1,z] = F_ap / area_sr[z]
        σ[1,1,z] = eF_ap / area_sr[z]
    end
    cube_data = (λ=cube_fitter.cube.λ, I=I, σ=σ, area_sr=area_sr)

    ######################### DO AN INITIAL FIT WITH THE SUM OF ALL SPAXELS ###################

    @debug """
    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
    """

    # If using an aperture, plot the aperture
    plot_2d(cube_fitter.cube, joinpath("output_$(cube_fitter.name)", "aperture_plot_beg.pdf"); err=false, aperture=aperture[1],
        z=cube_fitter.z, cosmo=cube_fitter.cosmology, slice=1)
    plot_2d(cube_fitter.cube, joinpath("output_$(cube_fitter.name)", "aperture_plot_mid.pdf"); err=false, aperture=aperture[end÷2],
        z=cube_fitter.z, cosmo=cube_fitter.cosmology, slice=shape[3]÷2)
    plot_2d(cube_fitter.cube, joinpath("output_$(cube_fitter.name)", "aperture_plot_end.pdf"); err=false, aperture=aperture[end],
        z=cube_fitter.z, cosmo=cube_fitter.cosmology, slice=shape[3])

    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    ##############################################################################################

    # Get the indices of all spaxels
    spaxels = CartesianIndices((1,1))

    # Wrapper function 
    function fit_spax_i(index::CartesianIndex)

        p_out, p_err = fit_spaxel(cube_fitter, cube_data, index; use_ap=true)
        if !isnothing(p_out)
            out_params[index, :] .= p_out
            out_errs[index, :, :] .= p_err
        end

        return
    end

    @info "===> Beginninng integrated spectrum fitting... <==="
    fit_spax_i(spaxels[1])

    @info "===> Generating parameter maps and model cubes... <==="

    # Create the ParamMaps and CubeModel structs containing the outputs
    param_maps, param_errs, cube_model = assign_outputs(out_params, out_errs, cube_fitter, cube_data, spaxels, cube_fitter.z, true)

    if cube_fitter.save_fits
        @info "===> Writing FITS outputs... <==="
        write_fits(cube_fitter, cube_data, cube_model, param_maps, param_errs, aperture=aperture)
    end

    # Save a copy of the options file used to run the code, so the settings can be referenced/reused
    # (for example, if you need to recall which lines you tied, what your limits were, etc.)
    cp(joinpath(@__DIR__, "..", "options", "options.toml"), joinpath("output_$(cube_fitter.name)", "general_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "dust.toml"), joinpath("output_$(cube_fitter.name)", "dust_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "lines.toml"), joinpath("output_$(cube_fitter.name)", "lines_options.archive.toml"), force=true)

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
    cube_fitter, param_maps, param_errs, cube_model

end
