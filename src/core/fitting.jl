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
function mask_emission_lines(λ::Vector{<:Real}, I::Vector{<:Real}, z::Real; Δ::Integer=3, W::Real=0.5, 
    thresh::Real=3., n_iter::Integer=1)

    diffs = diff(λ)
    # Numerical derivative width in microns
    h = Δ * median(diffs)

    # Calculate numerical first and second derivatives
    df = zeros(length(λ))
    @inbounds @simd for i ∈ 1:length(λ)
        df[i] = (I[min(length(λ), i+fld(Δ, 2))] - I[max(1, i-fld(Δ, 2))]) / h
    end
    d2f = zeros(length(λ))
    @inbounds @simd for i ∈ 1:length(λ)
        d2f[i] = (I[min(length(λ), i+Δ)] - 2I[i] + I[max(1, i-Δ)]) / h^2
    end
    d2f_i = copy(d2f)
    mask = falses(length(λ))

    # Sigma-clip to find the lines based on the *local* noise level
    @inbounds for j ∈ 1:length(λ)
        # Skip if the pixel has already been masked
        if mask[j] == 1
            continue
        end
        # Only consider the spectrum within +/- W microns from the point in question
        wi = Int(W ÷ diffs[min(length(λ)-1, j)])
        wl = Int(0.1 ÷ diffs[min(length(λ)-1, j)])
        if d2f[j] < -thresh * nanstd(d2f[max(1, j-wi):min(length(λ), j+wi)])

            # First, mask out +/-3*delta pixels by default
            mask[j-3Δ:j+3Δ] .= 1

            # Count how many pixels are above the local RMS level using a cubic spline interpolation
            λ_noline = [λ[max(1, j-wl):max(1, j-fld(wl, 2))]; λ[min(length(λ), j+fld(wl, 2)):min(length(λ), j+wl)]]
            s_noline = [I[max(1, j-wl):max(1, j-fld(wl, 2))]; I[min(length(λ), j+fld(wl, 2)):min(length(λ), j+wl)]]

            scale = 0.025 / (1 + z)
            offset = findfirst(λ[.~mask] .> (λ[.~mask][1] + scale))
            λknots = λ[.~mask][offset+1]:scale:λ[.~mask][end-offset-1]
            good = []
            for i ∈ eachindex(λknots)
                _, ind = findmin(abs.(λ .- λknots[i]))
                if !isnan(I[ind])
                    append!(good, [i])
                end
            end
            λknots = λknots[good]
            try
                cont_cub = Spline1D(λ[.~mask], I[.~mask], λknots, k=3)
                rms = nanstd(s_noline .- cont_cub.(λ_noline))
                n_pix = length(findall((I[max(1, j-wl):min(length(λ), j+wl)] .- cont_cub.(λ[max(1, j-wl):min(length(λ), j+wl)])) .> 3*rms))
                # Mask out this many pixels to the left and right
                mask[j-n_pix:j+n_pix] .= 1            
            catch
                n_pix = 3Δ
                # Mask out this many pixels to the left and right
                mask[j-n_pix:j+n_pix] .= 1
            end

        end
    end

    # Don't mask out this region that tends to trick this method sometimes
    mask[11.10 .< λ .< 11.15] .= 0
    mask[11.17 .< λ .< 11.355] .= 0

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
function continuum_cubic_spline(λ::Vector{<:Real}, I::Vector{<:Real}, σ::Vector{<:Real}, z::Real)

    # Copy arrays
    I_out = copy(I)
    σ_out = copy(σ)

    # Mask out emission lines so that they aren't included in the continuum fit
    mask_lines = mask_emission_lines(λ, I, z)
    I_out[mask_lines] .= NaN
    σ_out[mask_lines] .= NaN 

    # Interpolate the NaNs
    diffs = diff(λ)
    Δλ = mean(diffs)
    # Break up cubic spline interpolation into knots 0.05 um long
    # (longer than a narrow emission line but not too long)
    scale = 0.025 / (1 + z)
    finite = isfinite.(I_out)
    offset = findfirst(λ[finite] .> (scale + λ[finite][1]))

    # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
    λknots = λ[finite][offset+1]:scale:λ[finite][end-offset-1]
    # Remove any knots that happen to fall within a masked pixel
    good = []
    for i ∈ eachindex(λknots)
        _, ind = findmin(abs.(λ .- λknots[i]))
        if !isnan(I_out[ind])
            append!(good, [i])
        end
    end
    λknots = λknots[good]
    @debug "Performing cubic spline continuum fit with knots at $λknots"

    # Do a full cubic spline remapping of the data
    I_out = Spline1D(λ[isfinite.(I_out)], I_out[isfinite.(I_out)], λknots, k=3, bc="extrapolate").(λ)
    σ_out = Spline1D(λ[isfinite.(σ_out)], σ_out[isfinite.(σ_out)], λknots, k=3, bc="extrapolate").(λ)

    mask_lines, I_out, σ_out
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
    σ::Vector{<:Real}, mask_lines::BitVector, I_spline::Vector{<:Real}, σ_spline::Vector{<:Real}; init::Bool=false,
    use_ap::Bool=false) 

    @debug """\n
    #########################################################
    ###   Beginning continuum fit for spaxel $spaxel...   ###
    #########################################################
    """

    # Copy I and σ so that the inputs are not accidentally overwritten
    I_spax = Vector{Float64}(I)
    σ_spax = Vector{Float64}(σ)

    # Fill in the data where the lines are with the cubic spline interpolation
    I_spax[mask_lines] .= I_spline[mask_lines]
    σ_spax[mask_lines] .= σ_spline[mask_lines]

    # Add statistical uncertainties to the systematic uncertainties in quadrature
    σ_stat = std(I_spax[.!mask_lines] .- I_spline[.!mask_lines])
    σ_spax .= .√(σ_spax.^2 .+ σ_stat.^2)

    @debug "Adding statistical error of $σ_stat in quadrature"

    # Get the priors and "locked" booleans for each parameter, split up by the 2 steps for the continuum fit
    plims_1, plims_2, lock_1, lock_2 = get_continuum_plimits(cube_fitter, I_spax)

    # Split up the initial parameter vector into the components that we need for each fitting step
    pars_1, pars_2 = get_continuum_initial_values(cube_fitter, λ, I_spax, init || use_ap)

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
            model_continuum(x, ptot, cube_fitter.n_dust_cont, cube_fitter.extinction_curve, cube_fitter.extinction_screen,
                cube_fitter.fit_sil_emission)
        else
            model_continuum(x, ptot, cube_fitter.n_dust_cont, cube_fitter.extinction_curve, cube_fitter.extinction_screen,
                cube_fitter.fit_sil_emission, true)
        end            
    end
    res_1 = cmpfit(λ, I_spax, σ_spax, fit_step1, p1free, parinfo=parinfo_1, config=config)

    @debug "Continuum CMPFit Step 1 status: $(res_1.status)"

    # Create continuum without the PAH features
    _, ccomps = fit_step1(λ, res_1.param, true)

    I_cont = ccomps["stellar"]
    for i ∈ 1:cube_fitter.n_dust_cont
        I_cont .+= ccomps["dust_cont_$i"]
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
    res_2 = cmpfit(λ, I_spax.-I_cont, σ_spax, fit_step2, p2free, parinfo=parinfo_2, config=config)

    @debug "Continuum CMPFit Step 2 status: $(res_2.status)"

    # Get combined best fit results
    lock = vcat(lock_1[1:end-2], lock_2)

    # Combined Best fit parameters
    popt = zeros(length(pars_1)+length(pars_2)-2)
    popt[.~lock] .= vcat(res_1.param[1:end-2], res_2.param)
    popt[lock] .= vcat(p1fix, p2fix)

    # Combined 1-sigma uncertainties
    perr = zeros(length(popt))
    perr[.~lock] .= vcat(res_1.perror[1:end-2], res_2.perror)

    # Individual covariance matrices
    covar_1 = zeros(length(pars_1)-2, length(pars_1)-2)
    covar_1[.~lock_1[1:end-2], .~lock_1[1:end-2]] .= res_1.covar[1:end-2, 1:end-2]
    covar_2 = zeros(length(pars_2), length(pars_2))
    covar_2[.~lock_2, .~lock_2] .= res_2.covar

    # Combined covariance matrix
    covar = zeros(length(popt), length(popt))
    covar[1:length(pars_1)-2, 1:length(pars_1)-2] .= covar_1
    covar[length(pars_1)-1:end, length(pars_1)-1:end] .= covar_2

    n_free = n_free_1 + n_free_2 - 2

    @debug "Best fit continuum parameters: \n $popt"
    @debug "Continuum parameter errors: \n $perr"
    # @debug "Continuum covariance matrix: \n $covar"

    # Create the full model
    I_model, comps = model_continuum_and_pah(λ, popt, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat,
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

    σ_spax, popt, I_model, comps, n_free, perr, covar
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
    σ::Vector{<:Real}, continuum::Vector{<:Real}, ext_curve::Vector{<:Real}, mask_lines::BitVector, lsf_interp_func::Function; 
    init::Bool=false, use_ap::Bool=false)

    @debug """\n
    #########################################################
    ###      Beginning line fit for spaxel $spaxel...     ###
    #########################################################
    """

    # Get the normalization
    N = Float64(abs(nanmaximum(I)))
    N = N ≠ 0. ? N : 1.

    @debug "Using normalization N=$N"

    # Add statistical uncertainties to the systematic uncertainties in quadrature
    σ_stat = std(I[.!mask_lines] .- continuum[.!mask_lines])
    σ_spax = .√(σ.^2 .+ σ_stat.^2)

    @debug "Adding statistical error of $σ_stat in quadrature"

    # Normalized flux and uncertainty by subtracting the cubic spline fit and dividing by the maximum
    Inorm = (I .- continuum) ./ N
    σnorm = σ_spax ./ N

    plimits, param_lock, param_names, tied_pairs, tied_indices = get_line_plimits(cube_fitter, init || use_ap)
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

    if init || use_ap || cube_fitter.fit_all_samin
        @debug "Beginning Line fitting with Simulated Annealing:"

        # Parameter and function tolerance levels for convergence with SAMIN,
        # these are a bit loose since we're mainly just looking to get into the right global minimum region with SAMIN
        # before refining the fit later with a LevMar local minimum routine
        fit_func = (x, p) -> -ln_likelihood(
                                Inorm, 
                                model_line_residuals(x, p, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, 
                                    cube_fitter.flexible_wavesol, ext_curve, lsf_interp_func), 
                                σnorm)
        x_tol = 1e-5
        f_tol = abs(fit_func(λ, p₀) - fit_func(λ, clamp.(p₀ .- x_tol, lower_bounds, upper_bounds)))

        # First, perform a bounded Simulated Annealing search for the optimal parameters with a generous max iterations and temperature rate (rt)
        res = Optim.optimize(p -> fit_step3(λ, p, fit_func), lbfree_tied, ubfree_tied, pfree_tied, 
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
    else
        p₁ = pfree_tied
    end    
    
    @debug "Beginning Line fitting with Levenberg-Marquardt:"

    ############################################# FIT WITH LEVMAR ###################################################

    fit_func_2 = (x, p) -> model_line_residuals(x, p, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, 
        cube_fitter.flexible_wavesol, ext_curve, lsf_interp_func)
    
    χ2_before = sum((Inorm .- fit_step3(λ, p₁, fit_func_2)).^2 ./ σnorm.^2)
    res = cmpfit(λ, Inorm, σnorm, (x, p) -> fit_step3(x, p, fit_func_2), p₁, parinfo=parinfo, config=config)
    χ2_after = res.bestnorm

    # Check if CMPFit improved the reduced chi^2 at all.  If not, redo simulated annealing.
    if χ2_after ≥ χ2_before && !init && !use_ap
        @warn "The Levenberg-Marquardt solver for spaxel $spaxel did not improve the chi-squared. Re-running with simulated annealing."

        fit_func = (x, p) -> -ln_likelihood(
                                Inorm, 
                                model_line_residuals(x, p, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, 
                                    cube_fitter.flexible_wavesol, ext_curve, lsf_interp_func), 
                                σnorm)
        x_tol = 1e-5
        f_tol = abs(fit_func(λ, p₀) - fit_func(λ, clamp.(p₀ .- x_tol, lower_bounds, upper_bounds)))

        # First, perform a bounded Simulated Annealing search for the optimal parameters with a generous max iterations and temperature rate (rt)
        res = Optim.optimize(p -> fit_step3(λ, p, fit_func), lbfree_tied, ubfree_tied, pfree_tied, 
            SAMIN(;rt=0.9, nt=5, ns=5, neps=5, f_tol=f_tol, x_tol=x_tol, verbosity=0), Optim.Options(iterations=10^6))
        p₁ = res.minimizer

        res = cmpfit(λ, Inorm, σnorm, (x, p) -> fit_step3(x, p, fit_func_2), p₁, parinfo=parinfo, config=config)
    end 

    @debug "Line CMPFit status: $(res.status)"

    # Get the results and errors
    popt = zeros(Float64, length(p_tied))
    perr = zeros(Float64, length(p_tied))
    popt[.~param_lock_tied] .= res.param
    perr[.~param_lock_tied] .= res.perror
    popt[param_lock_tied] .= pfix_tied
    for tind in tied_indices
        insert!(popt, tind, 0.)
        insert!(perr, tind, 0.)
    end
    for tie in tied_pairs
        popt[tie[2]] = popt[tie[1]] * tie[3]
        perr[tie[2]] = perr[tie[1]] * tie[3]
    end

    # Covariance matrix
    covar = zeros(Float64, length(p_tied), length(p_tied))
    covar[.~param_lock_tied, .~param_lock_tied] .= res.covar
    for tind in tied_indices
        covar = cat(covar[1:tind-1, :], zeros(size(covar, 2))', covar[tind:end, :], dims=1)
        covar = cat(covar[:, 1:tind-1], zeros(size(covar, 1)), covar[:, tind:end], dims=2)
    end
    for tie in tied_pairs
        covar[tie[2], :] .= covar[tie[1], :] .* tie[3]
        covar[:, tie[2]] .= covar[:, tie[1]] .* tie[3]
        covar[tie[2], tie[2]] /= tie[3]
    end

    ######################################################################################################################

    @debug "Best fit line parameters: \n $popt"
    @debug "Line parameter errors: \n $perr"
    # @debug "Line covariance matrix: \n $covar"

    # Final optimized fit
    I_model, comps = model_line_residuals(λ, popt, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, 
        cube_fitter.flexible_wavesol, ext_curve, lsf_interp_func, true)
    
    # Renormalize
    I_model = I_model .* N
    for comp ∈ keys(comps)
        comps[comp] = comps[comp] .* N
    end

    if init
        cube_fitter.p_init_line[:] .= copy(popt)
        # Save results to file
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_line.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_line, ',')
        end
    end

    # Log the fit results
    pretty_print_line_results(cube_fitter, popt, perr)

    σ_spax, popt, I_model, comps, n_free, perr, covar
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
    n_dust_cont::Integer, n_dust_features::Integer, n_comps::Integer, line_wave::Vector{<:Real}, line_names::Vector{Symbol}, screen::Bool, 
    z::Real, χ2red::Real, name::String, label::String; backend::Symbol=:pyplot, range::Union{Tuple,Nothing}=nothing,
    spline::Union{Vector{<:Real},Nothing}=nothing) where {T<:Real}

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
        append!(traces, [PlotlyJS.scatter(x=λ, y=("hot_dust" ∈ keys(comps) ? comps["hot_dust"] : zeros(length(λ))) .+ 
            comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"] .* (sum([comps["dust_cont_$i"] for i ∈ 1:n_dust_cont], dims=1)[1] .+ comps["stellar"]),
            mode="lines", line=Dict(:color => "green", :width => 1), name="Dust+Stellar Continuum")])
        append!(traces, [PlotlyJS.scatter(x=λ, y=sum([comps["dust_feat_$i"] for i ∈ 1:n_dust_features], dims=1)[1] .* comps["extinction"],
            mode="lines", line=Dict(:color => "blue", :width => 1), name="PAHs")])
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
        ax2.plot(λ, (I.-I_cont) ./ norm ./ λ, "k-")
        χ2_str = @sprintf "%.3f" χ2red
        ax2.plot(λ, zeros(length(λ)), "-", color="#ff5d00", label=L"$\tilde{\chi}^2 = %$χ2_str$")
        ax2.fill_between(λ, (I.-I_cont.+σ)./norm./λ, (I.-I_cont.-σ)./norm./λ, color="k", alpha=0.5)
        # twin axes with different labels --> extinction for ax3 and observed wavelength for ax4
        ax3 = ax1.twinx()
        ax4 = ax1.twiny()

        # full continuum
        ax1.plot(λ, (comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"] .* 
            (sum([comps["dust_cont_$i"] for i ∈ 1:n_dust_cont], dims=1)[1] .+ comps["stellar"])) ./ norm ./ λ, "k--", alpha=0.5, 
            label="Continuum")
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
function fit_spaxel(cube_fitter::CubeFitter, cube_data::NamedTuple, spaxel::CartesianIndex; apertures=nothing, plot_spline=false)

    local p_out
    local p_err

    λ = cube_data.λ
    I = cube_data.I[spaxel, :]
    σ = cube_data.σ[spaxel, :]
    area_sr = cube_data.area_sr

    # if there are any NaNs, skip over the spaxel
    if any(.!isfinite.(I))
        return nothing, nothing
    end

    # Check if the fit has already been performed
    if !isfile(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).csv")) || cube_fitter.overwrite
        
        # Create a local logger for this individual spaxel
        timestamp_logger(logger) = TransformerLogger(logger) do log
            merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
        end
        # This log should be entirely handled by 1 process, since each spaxel is entirely handled by 1 process
        # so there should be no problems with I/O race conditions
        logger = TeeLogger(ConsoleLogger(stdout, Logging.Info), timestamp_logger(MinLevelLogger(FileLogger(
                             joinpath("output_$(cube_fitter.name)", "logs", "loki.spaxel_$(spaxel[1])_$(spaxel[2]).log"); 
                             always_flush=true), Logging.Debug)))

        with_logger(logger) do

            mask_lines, I_spline, σ_spline = continuum_cubic_spline(λ, I, σ, cube_fitter.z)
            
            # Interpolate the LSF
            lsf_interp = Spline1D(λ, cube_fitter.cube.lsf, k=1)
            lsf_interp_func = x -> lsf_interp(x)

            # If there is an aperture, set the "init" keyword since we're only fitting once
            use_ap = !isnothing(apertures)

            # Fit the spaxel
            σ, popt_c, I_cont, comps_cont, n_free_c, perr_c, covar_c = 
                @timeit timer_output "continuum_fit_spaxel" continuum_fit_spaxel(cube_fitter, spaxel, λ, I, σ, mask_lines, I_spline, σ_spline, use_ap=use_ap)
            _, popt_l, I_line, comps_line, n_free_l, perr_l, covar_l = 
                @timeit timer_output "line_fit_spaxel" line_fit_spaxel(cube_fitter, spaxel, λ, I, σ, cube_fitter.subtract_cubic ? I_spline : I_cont,
                comps_cont["extinction"], mask_lines, lsf_interp_func, use_ap=use_ap)

            # Combine the continuum and line models
            I_model = I_cont .+ I_line
            comps = merge(comps_cont, comps_line)

            # Total free parameters
            n_free = n_free_c + n_free_l
            n_data = length(I)

            # Degrees of freedom
            dof = n_data - n_free

            # chi^2 and reduced chi^2 of the model
            χ2 = sum(@. (I - I_model)^2 / σ^2)
            χ2red = χ2 / dof

            # Add dust feature and line parameters (intensity and SNR)
            p_dust, p_lines, p_dust_err, p_lines_err = 
                @timeit timer_output "calculate_extra_parameters" calculate_extra_parameters(λ, I, σ, cube_fitter.n_dust_cont,
                    cube_fitter.n_dust_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.fit_sil_emission,
                    cube_fitter.n_lines, cube_fitter.n_acomps, cube_fitter.n_comps, cube_fitter.lines, cube_fitter.flexible_wavesol, 
                    lsf_interp_func, popt_c, popt_l, perr_c, perr_l, comps["extinction"], mask_lines, I_spline, area_sr)
            p_out = [popt_c; popt_l; p_dust; p_lines; χ2; dof]
            p_err = [perr_c; perr_l; p_dust_err; p_lines_err; 0.; 0.]

            # Plot the fit
            if cube_fitter.plot_spaxels != :none
                @debug "Plotting spaxel $spaxel best fit"
                @timeit timer_output "plot_spaxel_fit" plot_spaxel_fit(λ, I, I_model, σ, comps, 
                    cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, cube_fitter.n_comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, 
                    cube_fitter.extinction_screen, cube_fitter.z, χ2red, cube_fitter.name, "spaxel_$(spaxel[1])_$(spaxel[2])", backend=cube_fitter.plot_spaxels,
                    spline=plot_spline ? I_spline : nothing)
                if !isnothing(cube_fitter.plot_range)
                    for (i, plot_range) ∈ enumerate(cube_fitter.plot_range)
                        @timeit timer_output "plot_line_fit" plot_spaxel_fit(λ, I, I_model, σ, comps,
                            cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, cube_fitter.n_comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, 
                            cube_fitter.extinction_screen, cube_fitter.z, χ2red, cube_fitter.name, "lines_$(spaxel[1])_$(spaxel[2])_$i", backend=cube_fitter.plot_spaxels,
                            range=plot_range, spline=plot_spline ? I_spline : nothing)
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

        end

        return p_out, p_err

    end

    # Otherwise, just grab the results from before
    results = readdlm(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).csv"), ',', Float64, '\n')
    p_out = results[:, 1]
    p_err = results[:, 2]

    p_out, p_err

end


"""
    fit_stack!(cube_fitter; plot_spline=false)

Perform an initial fit to the sum of all spaxels (the stack) to get an estimate for the initial parameter
vector to use with all of the individual spaxel fits.  The only input is the CubeFitter object, which is
modified with the resultant fit parameters.  There is no output.
"""
function fit_stack!(cube_fitter::CubeFitter; plot_spline=false)

    @info "===> Performing initial fit to the sum of all spaxels... <==="
    # Collect the data
    λ_init = cube_fitter.cube.λ
    I_sum_init = sumdim(cube_fitter.cube.Iν, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    σ_sum_init = sqrt.(sumdim(cube_fitter.cube.σI.^2, (1,2))) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))

    mask_lines, I_spline_init, σ_spline_init = continuum_cubic_spline(λ_init, I_sum_init, σ_sum_init, cube_fitter.z)

    # Interpolate the LSF
    lsf_interp = Spline1D(cube_fitter.cube.λ, cube_fitter.cube.lsf, k=1)
    lsf_interp_func = x -> lsf_interp(x)
    
    # Continuum and line fits
    σ_init, popt_c_init, I_c_init, comps_c_init, n_free_c_init, _, _ = continuum_fit_spaxel(cube_fitter, CartesianIndex(0,0),
        λ_init, I_sum_init, σ_sum_init, mask_lines, I_spline_init, σ_spline_init; init=true)
    _, popt_l_init, I_l_init, comps_l_init, n_free_l_init, _, _ = line_fit_spaxel(cube_fitter, CartesianIndex(0,0), 
        λ_init, I_sum_init, σ_sum_init, cube_fitter.subtract_cubic ? I_spline_init : I_c_init, comps_c_init["extinction"], mask_lines, 
        lsf_interp_func; init=true)

    # Get the overall models
    I_model_init = I_c_init .+ I_l_init
    comps_init = merge(comps_c_init, comps_l_init)

    n_free_init = n_free_c_init + n_free_l_init
    n_data_init = length(I_sum_init)

    # Calculate reduce chi^2
    χ2red_init = 1 / (n_data_init - n_free_init) * sum((I_sum_init .- I_model_init).^2 ./ σ_init.^2)

    # Plot the fit
    if cube_fitter.plot_spaxels != :none
        @debug "Plotting spaxel sum initial fit"
        plot_spaxel_fit(λ_init, I_sum_init, I_model_init, σ_init, comps_init,
            cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, cube_fitter.n_comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, 
            cube_fitter.extinction_screen, cube_fitter.z, χ2red_init, cube_fitter.name, "initial_sum_fit", backend=cube_fitter.plot_spaxels, 
            spline=plot_spline ? I_spline_init : nothing)
        if !isnothing(cube_fitter.plot_range)
            for (i, plot_range) ∈ enumerate(cube_fitter.plot_range)
                plot_spaxel_fit(λ_init, I_sum_init, I_model_init, σ_init, comps_init,
                    cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, cube_fitter.n_comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, 
                    cube_fitter.extinction_screen, cube_fitter.z, χ2red_init, cube_fitter.name, "initial_sum_line_$i", backend=cube_fitter.plot_spaxels;
                    range=plot_range, spline=plot_spline ? I_spline_init : nothing)
            end
        end
            
    end

end


"""
    fit_cube!(cube_fitter, aperture=nothing)

This is the main cube fitting function!! It's essentially a wrapper function to perform a full fit of an 
entire IFU cube, calling `fit_spaxel` for each spaxel in a parallel or serial loop depending on the cube_fitter options.  
Results are then concatenated into `ParamMaps` and `CubeModel` structs and plotted/saved, also based on the 
cube_fitter options.

The default behavior is to perform spaxel-by-spaxel fits. However, if an aperture is specified, then fitting
will only be performed on the integrated spectrum within the aperture.
"""
function fit_cube!(cube_fitter::CubeFitter; aperture::Union{PyObject,Vector{PyObject},Nothing}=nothing)

    @info """\n
    #############################################################################
    ######## BEGINNING FULL CUBE FITTING ROUTINE FOR $(cube_fitter.name) ########
    #############################################################################
    """
    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)
    use_ap = !isnothing(aperture)

    shape = !use_ap ? size(cube_fitter.cube.Iν) : (1,1,size(cube_fitter.cube.Iν, 3))

    # Prepare output array
    @info "===> Preparing output data structures... <==="
    out_params = SharedArray(ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2) .* NaN)
    out_errs = SharedArray(ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 2) .* NaN)
    # "cube_data" object holds the primary wavelength, intensity, and errors
    # this is just a convenience object since these may be different when fitting an integrated spectrum
    # within an aperture
    cube_data = (λ=cube_fitter.cube.λ, I=cube_fitter.cube.Iν, σ=cube_fitter.cube.σI, area_sr=cube_fitter.cube.Ω .* ones(shape[3]))

    ######################### DO AN INITIAL FIT WITH THE SUM OF ALL SPAXELS ###################

    @debug """
    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
    """

    # If using a single aperture, extend to length of wavelength axis
    if !(typeof(aperture) <: Vector) && use_ap
        aperture = repeat(aperture, shape[3])
    end

    # If using an aperture, plot the aperture
    if use_ap
        plot_2d(cube_fitter.cube, joinpath("output_$(cube_fitter.name)", "aperture_plot_beg.pdf"); err=false, aperture=aperture[1],
            z=cube_fitter.z, cosmo=cube_fitter.cosmology, slice=1)
        plot_2d(cube_fitter.cube, joinpath("output_$(cube_fitter.name)", "aperture_plot_mid.pdf"); err=false, aperture=aperture[end÷2],
            z=cube_fitter.z, cosmo=cube_fitter.cosmology, slice=shape[3]÷2)
        plot_2d(cube_fitter.cube, joinpath("output_$(cube_fitter.name)", "aperture_plot_end.pdf"); err=false, aperture=aperture[end],
            z=cube_fitter.z, cosmo=cube_fitter.cosmology, slice=shape[3])
    end

    # Don't repeat if it's already been done, and also dont do the initial fit if we're just fitting in an aperture
    if all(iszero.(cube_fitter.p_init_cont)) && !use_ap
        fit_stack!(cube_fitter)
    else
        @info "===> Initial fit to the sum of all spaxels is being skipped, either because it has already " *
            "been performed, or an aperture was specified <==="
    end

    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    ##############################################################################################

    # Get the indices of all spaxels
    spaxels = !use_ap ? CartesianIndices(selectdim(cube_fitter.cube.Iν, 3, 1)) : CartesianIndices((1,1))

    # If using an aperture, overwrite the cube_data object with the quantities within
    # the aperture, which are calculated here.
    if use_ap
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
        cube_data.I = I
        cube_data.σ = σ
        cube_data.area_sr = area_sr
    end

    # Wrapper function 
    function fit_spax_i(index::CartesianIndex)

        p_out, p_err = fit_spaxel(cube_fitter, cube_data, index; apertures=aperture)
        if !isnothing(p_out)
            out_params[index, :] .= p_out
            out_errs[index, :] .= p_err
        end

        return
    end

    # Use multiprocessing (not threading) to iterate over multiple spaxels at once using multiple CPUs
    if cube_fitter.parallel && !use_ap
        @info "===> Beginning individual spaxel fitting... <==="
        prog = Progress(length(spaxels); showspeed=true)
        progress_pmap(spaxels, progress=prog) do index
            fit_spax_i(index)
        end
    elseif !use_ap
        @info "===> Beginning individual spaxel fitting... <==="
        prog = Progress(length(spaxels); showspeed=true)
        for index ∈ spaxels
            fit_spax_i(index)
            next!(prog)
        end
    else
        @info "===> Beginninng integrated spectrum fitting... <==="
        fit_spax_i(spaxels[1])
    end

    @info "===> Generating parameter maps and model cubes... <==="

    # Create the ParamMaps and CubeModel structs containing the outputs
    param_maps, param_errs, cube_model = assign_outputs(out_params, out_errs, cube_fitter, cube_data, spaxels, cube_fitter.z, use_ap)

    if cube_fitter.plot_maps && !use_ap
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
