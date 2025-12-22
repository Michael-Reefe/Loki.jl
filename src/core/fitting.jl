#=
This is the main file for actually fitting IFU cubes.  It contains functions for actually performing the
fitting across a single spaxel and an entire cube. The main calling sequence one will want to 
perform is first loading in the data, presumably from some FITS files, with the cubedata functions,
then creating a CubeFitter struct from the DataCube struct, followed by calling fit_cube! on the
CubeFitter. An example of this is provided in the test driver files in the test directory.
=#

############################## FITTING FUNCTIONS AND HELPERS ####################################


"""
    repeat_fit_jitter(λ_spax, I_spax, σ_spax, fit_func, params, lb, ub, parinfo, config, res0, fit_type; check_cont_snr)

Checks the status of a fit to see if it has gotten stuck at the initial position (i.e. if number of iterations is < 5),
and if so it jitters the starting parameters and tries to fit again.  It will try up to 10 times before giving up.
"""
function repeat_fit_jitter(λ_spax::Vector{<:QWave}, I_spax::Vector{<:Real}, σ_spax::Vector{<:Real}, fit_func::Function,
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
    continuum_fit_spaxel(spaxel, cube_fitter[, init, use_ap, nuc_temp_fit, bootstrap_iter, p1_boots,
        force_noext])

Fit the continuum of a given spaxel in the DataCube, masking out the emission lines, using the 
Levenberg-Marquardt least squares fitting method with the `CMPFit` package.  

# Arguments
- `spaxel`: The Spaxel object containing the data
- `cube_fitter`: The CubeFitter object containing the fitting parameters and options 
- `init`: Flag for the initial fit which fits the sum of all spaxels, to get an estimation for
    the initial parameter vector for individual spaxel fits
- `use_ap`: Flag for fitting an integrated spectrum within an aperture
- `bootstrap_iter`: Flag for fitting multiple iterations of the same spectrum with bootstrapping
- `p1_boots`: If `bootstrap_iter` is true, this should give the best-fit parameter vector
for the initial non-bootstrapped fit of the spectrum.
"""
function continuum_fit_spaxel(spaxel::Spaxel, cube_fitter::CubeFitter; init::Bool=false, use_ap::Bool=false, 
    bootstrap_iter::Bool=false, p1_boots::Union{Vector{<:Real},Nothing}=nothing, force_noext::Bool=false)

    @debug """\n
    #########################################################
    ###   Beginning continuum fit for spaxel $(spaxel.coords)...   ###
    #########################################################
    """

    @assert spaxel.normalized
    # Create a copy of the original intensity vector so that we can replace it at the end after doing the line fitting
    s = copy(spaxel)
    # Interpolate over the MIR emission lines
    interpolate_over_lines!(s, 7)
    # Get a mask that excludes bad pixels, optical emission lines, and the user mask
    spaxel_mask = .~get_vector_mask(s; lines=true, user_mask=cube_fitter.cube.spectral_region.mask)

    # Get the priors and "locked" booleans for each parameter, split up by the 2 steps for the continuum fit
    pnames, plims, plock, tied_pairs, tied_indices, tie_vec = get_continuum_parameter_limits(cube_fitter, 
        s.I[spaxel_mask], s.σ[spaxel_mask]; init=init||use_ap, force_noext=force_noext, split=false)
    # Get the initial values and step sizes
    pars_0, dstep_0 = get_continuum_initial_values(cube_fitter, s.coords, s.λ[spaxel_mask], s.I[spaxel_mask], 
        s.σ[spaxel_mask], s.N; init=init||use_ap, split=false, force_noext=force_noext)
    # Get the units of the parameters and strip them off of the actual parameter vector
    pars_0, plims, punits = strip_units(pars_0, plims)
    if bootstrap_iter
        pars_0 = p1_boots
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

    # The actual fitting function
    function fit_cont(x, pfree_tied::AbstractVector)
        ptot = rebuild_full_parameters(pfree_tied, pfix_tied, plock_tied, tied_pairs, tied_indices, n_tied)
        # important! evaluate over the whole wavelength vector in s, THEN apply the mask at the very end
        # (why? if fitting any stellar or fe ii templates, they depend on the wavelength vector being 
        #  logarithmic without any weird gaps due to the FFTs that are performed)
        model_continuum(s, s.N, ptot, punits, cube_fitter, cube_fitter.fitting.use_pah_templates, true, true)[spaxel_mask]
    end

    # Do initial fit
    res = cmpfit(s.λ[spaxel_mask], s.I[spaxel_mask], s.σ[spaxel_mask], fit_cont, pfree_tied, parinfo=parinfo, config=config)

    # This function checks how many iterations the first fit took, and if its less than 5 it repeats it with slightly different
    # starting parameters to try to get it to converge better. This is because rarely the first fit can get stuck in the starting
    # position.
    res = repeat_fit_jitter(s.λ[spaxel_mask], s.I[spaxel_mask], s.σ[spaxel_mask], fit_cont, pfree_tied, lbfree_tied, ubfree_tied, 
        parinfo, config, res, "continuum", s.coords; check_cont_snr=true)
    chi2 = res.bestnorm
    n_free = length(pfree_tied)
    @debug "Continuum CMPFit Status: $(res.status)"

    # Get the results of the fit as a parameter vector, error vector, model intensity vector, and comps dict
    popt, perr, I_model, comps, norms, results_stellar, spaxel_model = collect_cont_fit_results(res, pfix_tied, plock_tied, 
        punits, tied_pairs, tied_indices, n_tied, spaxel, cube_fitter; bootstrap_iter=bootstrap_iter)

    fopt = fit_options(cube_fitter)
    if determine_fit_redo_with0extinction(cube_fitter, s.λ, s.σ, pnames, ustrip.(popt), plock, 
        I_model, comps, init, force_noext)

        # set optical depth to 0 if the template fits all of the spectrum
        @debug "Redoing the fit with optical depth locked to 0 due to template amplitudes"
        result0, I_model0, comps0, norms0, pah_amp0, results_stellar0, spaxel_model0, chi20, nfree0 = continuum_fit_spaxel(spaxel, 
            cube_fitter; init=init, use_ap=use_ap, bootstrap_iter=bootstrap_iter, p1_boots=p1_boots, force_noext=true)

        # Does an F-test to determine whether or not the extinction is actually statistically significant
        if fopt.F_test_ext
            test_passed, F_data, F_crit = F_test(length(s.I[spaxel_mask]), nfree0, n_free, chi20, chi2, 
                fopt.line_test_threshold)
            @debug "Extinction F-test results: $(test_passed ? "SUCCESS" : "FAILURE")"
            @debug "Extinction F-test value = $F_data | Critical value = $F_crit | Threshold = $(fopt.line_test_threshold)"
            if !test_passed
                return result0, I_model0, comps0, norms0, pah_amp0, results_stellar0, spaxel_model0, chi20, nfree0
            end
        else
            return result0, I_model0, comps0, norms0, pah_amp0, results_stellar0, spaxel_model0, chi20, nfree0
        end
    end

    # Get an estimate for PAH template amplitudes since we didnt actually use them in the fit
    pah_amp = estimate_pah_template_amplitude(cube_fitter, s.λ, comps)

    # Create a fit result object
    bounds = [[plim[1] for plim in plims].*punits [plim[2] for plim in plims].*punits]
    result = SpaxelFitResult(pnames, popt, [perr perr], bounds, plock, tie_vec)
    pretty_print_results(result)

    # Save the results if doing an initial integrated fit
    if init || (use_ap && !bootstrap_iter && fopt.n_bootstrap > 0) 
        # save_init_fit_outputs!(cube_fitter, ustrip.(popt), ustrip.(pah_amp))
        result_pahamp = SpaxelFitResult(["dust_features.PAH_template1.amp", "dust_features.PAH_template2.amp"],
            pah_amp, [0. 0.; 0. 0.], [-Inf Inf; -Inf Inf], falses(2), Union{Tie,Nothing}[nothing, nothing])
        write_fit_results_csv(cube_fitter, "init_fit_cont", result)
        write_fit_results_csv(cube_fitter, "init_fit_pahtemp", result_pahamp)
        cube_fitter.p_init_cont[:] .= popt
        cube_fitter.p_init_pahtemp[:] .= pah_amp
    end

    result, I_model, comps, norms, pah_amp, results_stellar, spaxel_model, chi2, n_free
end


function continuum_fit_spaxel(spaxel::Spaxel, cube_fitter::CubeFitter, split_flag::Bool; init::Bool=false, 
    use_ap::Bool=false, bootstrap_iter::Bool=false, p1_boots::Union{Vector{<:Real},Nothing}=nothing,
    force_noext::Bool=false) 

    if !split_flag
        return continuum_fit_spaxel(spaxel, cube_fitter; init=init, use_ap=use_ap, bootstrap_iter=bootstrap_iter, 
            p1_boots=p1_boots, force_noext=force_noext)
    end

    # This version of the function should only ever be called for MIR fitting since "split_flag" doesn't apply for optical fitting
    fopt = fit_options(cube_fitter)
    @assert !fopt.fit_joint "The fit_joint and use_pah_templates options are mutually exclusive!"

    @debug """\n
    #########################################################
    ###   Beginning continuum fit for spaxel $(spaxel.coords)...   ###
    #########################################################
    """

    @assert spaxel.normalized
    # Create a copy of the original intensity vector so that we can replace it at the end after doing the line fitting
    s = copy(spaxel)
    # Interpolate over the MIR emission lines
    interpolate_over_lines!(s, 7)
    # Get a mask that excludes bad pixels, optical emission lines, and the user mask
    spaxel_mask = .~get_vector_mask(s; lines=true, user_mask=cube_fitter.cube.spectral_region.mask)

    # Split up the initial parameter vector into the components that we need for each fitting step
    pnames_1, pnames_2, plims_1, plims_2, lock_1, lock_2, tied_pairs, tied_indices, tie_1, tie_2 = 
        get_continuum_parameter_limits(cube_fitter, s.I[spaxel_mask], s.σ[spaxel_mask]; init=init||use_ap, split=true, 
            force_noext=force_noext)
    pars_1, pars_2, dstep_1, dstep_2 = get_continuum_initial_values(cube_fitter, s.coords, s.λ[spaxel_mask], s.I[spaxel_mask], 
        s.σ[spaxel_mask], s.N; init=init||use_ap, split=true, force_noext=force_noext)
    # Get the units of the parameters and strip them off of the actual parameter vector
    pars_1, plims_1, punits_1, pars_2, plims_2, punits_2 = strip_units(pars_1, pars_2, plims_1, plims_2)
    if bootstrap_iter
        n_split = count_cont_parameters(model(cube_fitter); split=true)
        pars_1 = vcat(p1_boots[1:n_split], p1_boots[end-1:end])
        pars_2 = p1_boots[n_split+1:end-2]
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
    p2fix, p2free, d2free, lb_2, ub_2, n_free_2 = split_parameters(pars_2, dstep_2, plims_2, lock_2)

    # Convert parameter limits into CMPFit object
    parinfo_1, parinfo_2, config = get_continuum_parinfo(n_free_1, n_free_2, lb_1_free_tied, ub_1_free_tied, lb_2, ub_2, 
        d1free_tied, d2free)

    @debug "Beginning continuum fitting with Levenberg-Marquardt least squares (CMPFit):"

    function fit_step1(x, p1free_tied::AbstractVector)
        ptot = rebuild_full_parameters(p1free_tied, p1fix_tied, lock_1_tied, tied_pairs, tied_indices, n_tied_1)
        model_continuum(s, s.N, ptot, punits_1, cube_fitter, cube_fitter.fitting.use_pah_templates, true, true)[spaxel_mask]
    end

    res_1 = cmpfit(s.λ[spaxel_mask], s.I[spaxel_mask], s.σ[spaxel_mask], fit_step1, p1free_tied, parinfo=parinfo_1, config=config)
    res_1 = repeat_fit_jitter(s.λ[spaxel_mask], s.I[spaxel_mask], s.σ[spaxel_mask], fit_step1, p1free_tied, lb_1_free_tied, ub_1_free_tied, parinfo_1, config,
        res_1, "continuum (step 1)", s.coords; check_cont_snr=true)

    @debug "Continuum CMPFit Status: $(res_1.status)"

    popt, perr, I_model, comps, norms, _, _ = collect_cont_fit_results(res_1, p1fix_tied, lock_1_tied, punits_1, tied_pairs, tied_indices, 
        n_tied_1, spaxel, cube_fitter; bootstrap_iter=bootstrap_iter, step1=true)
    # Get the continuum component without the PAHs so that we can subtract it
    I_cont = comps["continuum"]

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

    # Fitting functions for MIR continuum step 2: just the PAHs with no templates (includes automatic differentiation)
    # fit_step2 = cmpfit_function(cube_fitter, pars_2, p2fix, lock_2, ccomps, template_norm, nuc_temp_fit)
    function fit_step2(x, pfree::AbstractVector)
        ptot = zeros(Float64, length(pars_2))
        ptot[.~lock_2] .= pfree
        ptot[lock_2] .= p2fix
        model_pah_residuals(s, ptot, punits_2, comps["total_extinction_gas"], cube_fitter)[spaxel_mask]
    end

    res_2 = cmpfit(s.λ[spaxel_mask], (s.I.-I_cont)[spaxel_mask], s.σ[spaxel_mask], fit_step2, p2free, parinfo=parinfo_2, config=config)
    res_2 = repeat_fit_jitter(s.λ[spaxel_mask], (s.I.-I_cont)[spaxel_mask], s.σ[spaxel_mask], fit_step2, p2free, lb_2, ub_2, parinfo_2, config, res_2, 
        "continuum (step 2)", s.coords; check_cont_snr=true)
    chi2 = res_2.bestnorm
    n_free = length(p1free_tied) + length(p2free)

    @debug "Continuum CMPFit Step 2 status: $(res_2.status)"

    popt, perr, n_free, pahtemp, I_model, comps, norms, results_stellar, spaxel_model = collect_cont_fit_results(res_1, p1fix_tied, 
        lock_1_tied, punits_1, tied_pairs, tied_indices, n_tied_1, res_2, p2fix, lock_2, punits_2, n_free_1, n_free_2, cube_fitter, 
        spaxel; bootstrap_iter=bootstrap_iter)

    # set optical depth to 0 if the template fits all of the spectrum
    if determine_fit_redo_with0extinction(cube_fitter, s.λ, s.σ, pnames_1, ustrip.(popt), lock_1, 
        I_model, comps, init, force_noext)

        @debug "Redoing the fit with optical depth locked to 0 due to template amplitudes"
        result0, I_model0, comps0, norms0, pah_amp0, results_stellar0, spaxel_model0, chi20, nfree0 = continuum_fit_spaxel(spaxel, 
            cube_fitter, split_flag; init=init, use_ap=use_ap, bootstrap_iter=bootstrap_iter, p1_boots=p1_boots, force_noext=true)

        # Does an F-test to determine whether or not the extinction is actually statistically significant
        if fopt.F_test_ext
            test_passed, F_data, F_crit = F_test(length(s.I[spaxel_mask]), nfree0, n_free, chi20, chi2, fopt.line_test_threshold)
            @debug "Extinction F-test results: $(test_passed ? "SUCCESS" : "FAILURE")"
            @debug "Extinction F-test value = $F_data | Critical value = $F_crit | Threshold = $(fopt.line_test_threshold)"
            if !test_passed
                return result0, I_model0, comps0, norms0, pah_amp0, results_stellar0, spaxel_model0, chi20, nfree0
            end
        else
            return result0, I_model0, comps0, norms0, pah_amp0, results_stellar0, spaxel_model0, chi20, nfree0
        end

    end

    plims = [plims_1[1:end-2]; plims_2]
    plock = [lock_1[1:end-2]; lock_2]
    tie_vec = [tie_1[1:end-2]; tie_2]
    pnames = [pnames_1[1:end-2]; pnames_2]
    punits = [punits_1[1:end-2]; punits_2]
    bounds = [[plim[1] for plim in plims].*punits [plim[2] for plim in plims].*punits]

    # Create a fit result object
    result = SpaxelFitResult(pnames, popt, [perr perr], bounds, plock, tie_vec)
    pretty_print_results(result)

    if init || (use_ap && !bootstrap_iter && fopt.n_bootstrap > 0)
        # save_init_fit_outputs!(cube_fitter, ustrip.(popt), ustrip.(pahtemp))
        result_pahamp = SpaxelFitResult(["dust_features.PAH_template1.amp", "dust_features.PAH_template2.amp"],
            pahtemp, [0. 0.; 0. 0.], [-Inf Inf; -Inf Inf], falses(2), Union{Tie,Nothing}[nothing, nothing])
        write_fit_results_csv(cube_fitter, "init_fit_cont", result)
        write_fit_results_csv(cube_fitter, "init_fit_pahtemp", result_pahamp)
        cube_fitter.p_init_cont[:] .= popt
        cube_fitter.p_init_pahtemp[:] .= pahtemp
    end

    result, I_model, comps, norms, pahtemp, results_stellar, spaxel_model, chi2, n_free
end


# Helper function that takes the results of the line tests and locks the appropriate parameters in the parameter vector.
# (i.e. all amplitudes are set to 0 for the component profiles that are deemed not significant)
function lock_nonfit_component_amps!(cube_fitter::CubeFitter, coords::CartesianIndex, profiles_to_fit_list::Vector{<:Integer},
    line_names::Vector{Symbol}, p₀::Vector{<:Real}, param_lock::BitVector)

    fopt = fit_options(cube_fitter)
    lines = model(cube_fitter).lines

    for grp in fopt.line_test_lines
        # if the line being tested is a part of a kinematic group, we should change the components of all the tied lines 
        # as well 
        group = copy(grp)
        @assert group isa Vector{Symbol}
        @debug "Line test group -- lines that were actually tested: $grp"

        line_params = get_flattened_fit_parameters(lines)
        for line in grp
            pinds = fast_indexin(["lines.$(line).1.amp", "lines.$(line).1.voff", "lines.$(line).1.fwhm"], line_params.names)
            pties = [!isnothing(ptie) ? ptie.group : nothing for ptie in line_params.ties[pinds]]
            if any(.!isnothing.(pties))
                pties_all = [!isnothing(ptie) ? ptie.group : nothing for ptie in line_params.ties]
                pties_uni = unique(pties[.!isnothing.(pties)])
                for tie in pties_uni
                    wh = tie .== pties_all
                    for lineparname in line_params.names[wh]
                        linename = Symbol(split(lineparname, ".")[2])
                        if !(linename in group)
                            push!(group, linename)
                        end
                    end
                end
            end
        end
        @debug "Line test group -- all lines that will be affected, including tied lines: $group"

        # Get the group member indices
        inds = [findfirst(ln .== line_names) for ln in group]
        inds = inds[.~isnothing.(inds)]  # may get nothings if the "*" flag is used, but this is ok since we only need the inds
                                        # to find the maximum number of profiles in the group, so the "*"s are irrelevant here
        # Fit the maximum number of components in the group
        profiles_group = maximum(profiles_to_fit_list[inds]; init=1)

        # Save the number of components fit for this spaxel
        if coords != CartesianIndex(0,0)
            for ln in group
                cube_fitter.n_fit_comps[ln][coords] = profiles_group
            end
        end

        # Lock the amplitudes to 0 for any profiles that will not be fit
        pᵢ = 1
        for (i, line) in enumerate(lines.profiles)   # <- iterates over emission lines
            for (j, component) in enumerate(line) 

                amp_ind = pᵢ
                # Check if using a flexible_wavesol tied voff -> if so there is an extra voff parameter
                pc = 3
                if component.profile == :GaussHermite
                    pc += 2
                elseif component.profile == :Voigt
                    pc += 1
                end
                pᵢ += pc
                # Check if the line is a member of the group, and if so, lock any profiles > profiles_group
                if (lines.names[i] in group) && (profiles_group < j)
                    # Amplitude is set to 0
                    p₀[amp_ind] = 0.
                    # Amplitude, Voff, and FWHM become locked 
                    param_lock[amp_ind:amp_ind+2] .= 1
                    # Also lock additional parameters
                    if component.profile == :GaussHermite
                        param_lock[amp_ind+3:amp_ind+4] .= 1
                    end
                    if component.profile == :Voigt
                        param_lock[amp_ind+3] = 1
                    end
                end
            end
        end
    end

end


"""
    perform_line_component_test!(cube_fitter, spaxel, p₀, param_lock, lower_bounds, upper_bounds[, all_fail, bootstrap_iter])

Calculates the significance of additional line components and determines whether or not to include them in the fit.
Modifies the p₀ and param_lock vectors in-place to reflect these changes (by locking the amplitudes and other parameters
for non-necessary line components to 0).
"""
function perform_line_component_test!(cube_fitter::CubeFitter, spaxel::Spaxel, extinction_curve::Vector{<:Real}, 
    p₀::Vector{<:Real}, punits::Vector{<:Unitful.Units}, param_lock::BitVector, lower_bounds::Vector{<:Real}, 
    upper_bounds::Vector{<:Real}; all_fail::Bool=false, bootstrap_iter::Bool=false)

    @debug "Performing line component testing..."

    # Perform a test to see if each line with > 1 component really needs multiple components to be fit
    fopt = fit_options(cube_fitter)
    lines = model(cube_fitter).lines

    line_names = Symbol[]
    profiles_to_fit_list = Int[]
    pᵢ = 1
    for i in 1:cube_fitter.n_lines
        
        pstart = pᵢ
        # Get number of line profiles and number of parameters for each profile
        n_prof, pcomps = get_line_nprof_ncomp(cube_fitter, i)
        pᵢ += sum(pcomps)

        # - Skip lines with only 1 profile
        # - Only test the lines that are specified to be tested
        if (n_prof == 1) || !(lines.names[i] ∈ vcat(fopt.line_test_lines...))
            continue
        end

        # Constrain the fitting region and create a single TransitionLine object
        voff_max = max(abs(lower_bounds[pstart+1]), abs(upper_bounds[pstart+1])) * u"km/s"
        fwhm_max = upper_bounds[pstart+2] * u"km/s"
        wbounds = lines.λ₀[i] .* (1-(voff_max+fwhm_max)/C_KMS, 1+(voff_max+fwhm_max)/C_KMS)
        region = wbounds[1] .< spaxel.λ .< wbounds[2]

        if fopt.plot_line_test && !bootstrap_iter
            fig, ax = plt.subplots()
            ax.plot(ustrip.(spaxel.λ[region]), spaxel.I[region], "k-", label="Data")
        end

        # Perform fits for all possible numbers of components
        last_chi2 = test_chi2 = 0.
        test_p = last_p = 0
        test_n = length(spaxel.λ[region]) 
        critical_val = 0.
        F_data = 0.
        chi2_A = chi2_B = 0.
        profiles_to_fit = 0
        for np in 1:n_prof
            @debug "Testing $(lines.names[i]) with $np components:"
            if all_fail
                @debug "all_fail flag has been set -- fitting with 1 profile"
                profiles_to_fit = 1
                break
            end

            # Stop index
            pstop = pstart + sum(pcomps[1:np]) - 1

            # Parameters
            parameters = p₀[pstart:pstop]
            punits_i = punits[pstart:pstop]
            plock = param_lock[pstart:pstop]
            pfree = parameters[.~plock]
            pfree = clamp.(pfree, lower_bounds[pstart:pstop][.~plock], upper_bounds[pstart:pstop][.~plock])

            # make a subset emission line object with only this one line and only np profiles
            line_object = make_single_line_object(lines, i, np)

            function fit_func_test(x, pfree) 
                p = zeros(length(parameters))
                p[.~plock] .= pfree
                p[plock] .= parameters[plock]
                model_line_residuals(spaxel, p, punits_i, line_object, cube_fitter.lsf, extinction_curve, region)
            end

            p₁ = perform_global_SAMIN_fit(spaxel, cube_fitter, pfree, lower_bounds[pstart:pstop][.~plock],
                upper_bounds[pstart:pstop][.~plock], fit_func_test; update_log=false, spaxel_mask=region)

            # Save the reduced chi2 values
            test_model = fit_func_test(nothing, p₁)
            χ² = sum((spaxel.I[region] .- test_model).^2 ./ spaxel.σ[region].^2)
            test_p = sum(.~plock)
            dof = test_n - test_p
            test_chi2 = χ²

            chi2_A = round(last_chi2/dof, digits=3)
            chi2_B = round(test_chi2/dof, digits=3)

            ##### Perform an F-test #####
            test_passed, F_data, critical_val = F_test(test_n, last_p, test_p, last_chi2, test_chi2, 
                fopt.line_test_threshold)

            @debug "(A) Previous chi^2 = $last_chi2"
            @debug "(B) New chi^2 = $test_chi2"
            @debug "F-value = $F_data | Critical value = $critical_val"
            @debug "TEST $(test_passed ? "SUCCESS" : "FAILURE")"

            if fopt.plot_line_test && !bootstrap_iter
                ax.plot(ustrip.(spaxel.λ[region]), test_model, linestyle="-", label="$np-component model")
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
        @debug "$(lines.names[i]) will have $profiles_to_fit components"

        push!(profiles_to_fit_list, profiles_to_fit)
        push!(line_names, lines.names[i])

        if fopt.plot_line_test && !bootstrap_iter
            ax.set_xlabel(L"$\lambda_{\rm rest}$ (%$(latex(unit(spaxel.λ[1]))))")
            ax.set_ylabel("Normalized Intensity")
            ax.set_title(lines.labels[i])
            ax.legend(loc="upper right")
            ax.set_xlim(ustrip(wbounds[1]), ustrip(wbounds[2]))
            ax.annotate("Result: $profiles_to_fit profile(s)\n" * L"$\tilde{\chi}^2_A = %$chi2_A$" * "\n" *
                L"$\tilde{\chi}^2_B = %$chi2_B$" * "\n" * L"$F = %$F_final$" * "\n" * L"$F_{\rm crit} = %$crit_final$", 
                (0.05, 0.95), xycoords="axes fraction", ha="left", va="top")
            ax.axvline(ustrip(lines.λ₀[i]), linestyle="--", alpha=0.5, color="k", lw=0.5)
            folder = joinpath("output_$(cube_fitter.name)", "line_tests", "$(lines.names[i])")
            if !isdir(folder)
                mkpath(folder)
            end
            fname = isone(length(spaxel.coords)) ? "voronoi_bin_$(spaxel.coords[1])" : "spaxel_$(spaxel.coords[1])_$(spaxel.coords[2])"
            plt.savefig(joinpath(folder, "$fname.pdf"), dpi=300, bbox_inches="tight")
            plt.close()
        end
    end

    # Lock the amplitudes for all of the line profiles that did not pass the line test
    lock_nonfit_component_amps!(cube_fitter, spaxel.coords, profiles_to_fit_list, line_names, p₀, param_lock)
end


# Helper function that does a global fit to the emission lines with Simulated Annealing
function perform_global_SAMIN_fit(spaxel::Spaxel, cube_fitter::CubeFitter, pfree_tied::Vector{<:Real}, 
    lbfree_tied::Vector{<:Real}, ubfree_tied::Vector{<:Real}, fit_func_inner::Function; update_log::Bool=true,
    spaxel_mask::BitVector=trues(length(spaxel.λ)))

    @debug "Beginning global fitting with Simulated Annealing:"

    fit_func = p -> -ln_likelihood(spaxel.I[spaxel_mask], fit_func_inner(nothing, p), spaxel.σ[spaxel_mask])
    x_tol = 1e-5
    f_tol = abs(fit_func(pfree_tied) - fit_func(clamp.(pfree_tied .* (1 .- x_tol), lbfree_tied, ubfree_tied)))
    f_tol = 10^floor(log10(f_tol))
    lb_global = lbfree_tied
    ub_global = [isfinite(ub) ? ub : 1e10 for ub in ubfree_tied]

    # First, perform a bounded Simulated Annealing search for the optimal parameters with a generous max iterations and temperature rate (rt)
    res = Optim.optimize(fit_func, lb_global, ub_global, pfree_tied, 
        SAMIN(;rt=0.9, nt=5, ns=5, neps=5, f_tol=f_tol, x_tol=x_tol, verbosity=0), Optim.Options(iterations=10^6))
    p₁ = res.minimizer 

    # Write convergence results to file, if specified
    oopt = out_options(cube_fitter)
    if oopt.track_convergence && update_log
        update_global_convergence_log(cube_fitter, spaxel.coords, res)
    end

    p₁
end


"""
    line_fit_spaxel(s, cube_fitter, continuum, extinction_curve; init, use_ap, bootstrap_iter, p1_boots)

Fit the emission lines of a given spaxel in the DataCube, subtracting the continuum, using the 
Simulated Annealing fitting method with the `Optim` package and the Levenberg-Marquardt method with `CMPFit`.

This procedure has been adapted from PAHFIT (with some heavy adjustments). 
See Smith, Draine, et al. 2007; http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php

# Arguments
- `s`: The spaxel being fit
- `cube_fitter`: The CubeFitter object containing the parameters and options for the fit
- `continuum`: The fitted continuum level of the spaxel being fit (which will be subtracted
    before the lines are fit)
- `extinction_curve`: The gas extinction to be applied to the lines
- `init`: Flag for the initial fit which fits the sum of all spaxels, to get an estimation for
    the initial parameter vector for individual spaxel fits
- `use_ap`: Flag for fitting an integrated spectrum within an aperture
- `bootstrap_iter`: Flag for fitting multiple iterations of the same spectrum with bootstrapping
- `p1_boots`: If `bootstrap_iter` is true, this should give the best-fit parameter vector
for the initial non-bootstrapped fit of the spectrum.
"""
function line_fit_spaxel(s::Spaxel, s_model::Spaxel, cube_fitter::CubeFitter, continuum::Vector{<:Real}, 
    extinction_gas::Vector{<:Real}; init::Bool=false, use_ap::Bool=false, bootstrap_iter::Bool=false, 
    p1_boots::Union{Vector{<:Real},Nothing}=nothing)

    @debug """\n
    #########################################################
    ###      Beginning line fit for spaxel $(s.coords)...     ###
    #########################################################
    """

    @assert s.normalized
    # Create a copy of the original intensity vector so that we can replace it at the end after doing the line fitting
    _orig_I = copy(s.I)

    # Subtract the continuum
    subtract_continuum!(s, s_model, continuum)
    # This ensures that any lines that fall within the masked regions will go to 0
    spaxel_mask = get_vector_mask(s; lines=false, user_mask=cube_fitter.cube.spectral_region.mask)
    s.I[spaxel_mask] .= 0.0*unit(s.I[1])
    @debug "Using normalization N=$(s.N)"

    # interpolate extinction curve
    extinction_curve = extinction_gas
    if s != s_model
        extinction_curve = Spline1D(ustrip.(s_model.λ), extinction_gas, k=1)(ustrip.(s.λ))
    end

    # Get the line parameter initial values, limits, locks, and tied indices
    p₀, plimits, param_lock, param_names, step, tied_pairs, tied_indices, tie_vec = get_line_initial_values_limits_locked(
        cube_fitter, s.λ, s.I, extinction_curve; init=init||use_ap)
    p₀, plimits, punits = strip_units(p₀, plimits)
    step = ustrip.(step)
    lower_bounds = [pl[1] for pl in plimits]
    upper_bounds = [pl[2] for pl in plimits]

    # Perform line component tests to determine which line components are actually necessary to include in the fit
    fopt = fit_options(cube_fitter)
    if (length(fopt.line_test_lines) > 0) && !init
        perform_line_component_test!(cube_fitter, s, extinction_curve, p₀, punits, param_lock, lower_bounds, upper_bounds; 
            bootstrap_iter=bootstrap_iter)
    end

    @debug "Line Parameters:"
    # Split up parameters into tied and untied
    pfree_tied, pfix_tied, dfree_tied, param_lock_tied, lbfree_tied, ubfree_tied, pnames_tied, n_free, n_tied = 
        split_parameters(p₀, step, plimits, param_lock, tied_indices; param_names=param_names)
    # Place initial values within the lower/upper limits (amplitudes may be too large if the extinction levels differ between spaxels)
    pfree_tied = clamp.(pfree_tied, lbfree_tied, ubfree_tied)

    # Get CMPFit parinfo object from bounds
    parinfo, config = get_line_parinfo(n_free, lbfree_tied, ubfree_tied, dfree_tied)

    function fit_step3(x, pfree_tied::AbstractVector)
        ptot = rebuild_full_parameters(pfree_tied, pfix_tied, param_lock_tied, tied_pairs, tied_indices, n_tied)
        model_line_residuals(s, ptot, punits, model(cube_fitter).lines, cube_fitter.lsf, extinction_curve, trues(length(s.λ)))
    end

    # Do an optional global fit with simulated annealing
    _fit_global = false
    # We dont really need a global fit unless lines have multiple components
    # max_ncomps = maximum([length(prof) for prof in model(cube_fitter).lines.profiles])
    # if (init || use_ap || fopt.fit_all_global) && (max_ncomps > 1) && !bootstrap_iter
    if fopt.fit_all_global && !bootstrap_iter
        # @info "Detected that you are fitting multiple velocity components for emission lines. We will perform a global minimization (this may take a while)."
        p₁ = perform_global_SAMIN_fit(s, cube_fitter, pfree_tied, lbfree_tied, ubfree_tied, fit_step3)
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

    res = cmpfit(s.λ, s.I, s.σ, fit_step3, p₁, parinfo=parinfo, config=config)
    res = repeat_fit_jitter(s.λ, s.I, s.σ, fit_step3, p₁, lbfree_tied, ubfree_tied,
        parinfo, config, res, "line", s.coords)

    @debug "Line CMPFit status: $(res.status)"

    # Collect the results of the fit
    popt, perr, I_model, comps = collect_line_fit_results(res, pfix_tied, param_lock_tied, punits, tied_pairs,
        tied_indices, n_tied, s_model, extinction_gas, cube_fitter; bootstrap_iter=bootstrap_iter)

    ######################################################################################################################
    
    # Create a fit result object
    bounds = [[plim[1] for plim in plimits].*punits [plim[2] for plim in plimits].*punits]
    result = SpaxelFitResult(param_names, popt, [perr perr], bounds, param_lock, tie_vec)
    pretty_print_results(result)

    if init || (use_ap && !bootstrap_iter && fopt.n_bootstrap > 0)
        # Clean up the initial parameters to be more amenable to individual spaxel fits
        popt = clean_line_parameters(cube_fitter, popt, lower_bounds, upper_bounds)
        # Save results to the cube_fitter and to files
        # save_init_fit_outputs!(cube_fitter, ustrip.(popt))
        write_fit_results_csv(cube_fitter, "init_fit_line", result)
        cube_fitter.p_init_line[:] .= popt
    end

    # Now we can replace back the original intensity
    s.I .= _orig_I

    result, I_model, comps, n_free
end


"""
    all_fit_spaxel(cube_fitter, spaxel, λ, I, σ, templates, mask_lines, mask_bad, I_spline, N, area_sr, lsf_interp_func;
        [init, use_ap, bootstrap_iter, p1_boots_cont, p1_boots_line])

Fit the continuum and emission lines in a spaxel simultaneously with a combination of the Simulated Annealing 
and Levenberg-Marquardt algorithms (SLOW).

# Arguments
- `spaxel`: The spaxel object
- `cube_fitter`: The CubeFitter object containing the parameters and options for the fit
- `init`: Flag for the initial fit which fits the sum of all spaxels, to get an estimation for
the initial parameter vector for individual spaxel fits
- `use_ap`: Flag for fitting an integrated spectrum within an aperture
- `bootstrap_iter`: Flag for fitting multiple iterations of the same spectrum with bootstrapping
- `p1_boots_cont`: If `bootstrap_iter` is true, this should give the best-fit parameter vector
for the initial non-bootstrapped fit of the continuum.
- `p1_boots_line`: Same as `p1_boots_cont`, but for the line parameters.
"""
function all_fit_spaxel(s::Spaxel, cube_fitter::CubeFitter; init::Bool=false, use_ap::Bool=false, 
    bootstrap_iter::Bool=false, p1_boots_cont::Union{Vector{<:Real},Nothing}=nothing, 
    p1_boots_line::Union{Vector{<:Real},Nothing}=nothing) 

    fopt = fit_options(cube_fitter)
    @assert !fopt.use_pah_templates "The fit_joint and use_pah_templates options are mutually exclusive!"

    @debug """\n
    ################################################################
    ###   Beginning continuum & line fit for spaxel $s...   ###
    ################################################################
    """

    @assert s.normalized
    spaxel_mask = .~get_vector_mask(s; lines=false, user_mask=cube_fitter.cube.spectral_region.mask)

    @debug """\n
    ##################################################################################################################
    ########################################## FITTING THE CONTINUUM & LINES #########################################
    ##################################################################################################################
    """

    # Get the priors and "locked" booleans for each parameter
    pnames_cont, plims_cont, lock_cont, tied_pairs_cont, tied_indices_cont, tie_vec_cont = get_continuum_parameter_limits(
        cube_fitter, s.I[spaxel_mask], s.σ[spaxel_mask]; init=init||use_ap)
    # Get the initial values and step sizes
    pars_0_cont, dstep_0_cont = get_continuum_initial_values(cube_fitter, s.coords, s.λ[spaxel_mask], s.I[spaxel_mask], 
        s.σ[spaxel_mask], s.N; init=init||use_ap, split=false)
    # Get the units of the parameters and strip them off of the actual parameter vector
    pars_0_cont, plims_cont, punits_cont = strip_units(pars_0_cont, plims_cont)
    dstep_0_cont = ustrip.(dstep_0_cont)
    if bootstrap_iter
        pars_0_cont = p1_boots_cont
    end

    @debug "Continuum Parameters:"
    pfree_cont_tied, pfix_cont_tied, dfree_cont_tied, lock_cont_tied, lbfree_cont_tied, ubfree_cont_tied, 
        n_free_cont, n_tied_cont = split_parameters(pars_0_cont, dstep_0_cont, plims_cont, lock_cont, tied_indices_cont)

    pars_0_lines, plims_lines, lock_lines, names_lines, dstep_0_lines, tied_pairs_lines, tied_indices_lines, tie_vec_lines = 
        get_line_initial_values_limits_locked(cube_fitter, s.λ[spaxel_mask], s.I[spaxel_mask]; init=init||use_ap)
    pars_0_lines, plims_lines, punits_lines = strip_units(pars_0_lines, plims_lines)
    dstep_0_lines = ustrip.(dstep_0_lines)
    if bootstrap_iter
        pars_0_lines = p1_boots_line
    end
    lower_bounds_lines = [pl[1] for pl in plims_lines]
    upper_bounds_lines = [pl[2] for pl in plims_lines]

    # Perform line component tests to determine which line components are actually necessary to include in the fit
    if (length(fopt.line_test_lines) > 0) && !init
        s.I .-= s.I_spline
        # (set the extinction curve to all 1s for this -- it doesnt really matter)
        perform_line_component_test!(cube_fitter, s, ones(length(s.λ)), pars_0_lines, punits_lines, lock_lines, lower_bounds_lines, 
            upper_bounds_lines, bootstrap_iter=bootstrap_iter)
        s.I .+= s.I_spline 
    end
    @debug "Line Parameters:"
    pfree_lines_tied, pfix_lines_tied, dfree_lines_tied, lock_lines_tied, lbfree_lines_tied, 
        ubfree_lines_tied, names_lines_tied, n_free_lines, n_tied_lines = split_parameters(
        pars_0_lines, dstep_0_lines, plims_lines, lock_lines, tied_indices_lines; param_names=names_lines)

    # Place initial values within the lower/upper limits (amplitudes may be too large if the extinction levels differ between spaxels)
    pfree_lines_tied = clamp.(pfree_lines_tied, lbfree_lines_tied, ubfree_lines_tied)

    function fit_joint(x, pfree_tied_all::AbstractVector; n=0)
        # Split into continuum and lines parameters
        pfree_cont_tied = pfree_tied_all[1:n_free_cont]
        pfree_lines_tied = pfree_tied_all[n_free_cont+1:end]

        # Organize parameters
        ptot_cont = rebuild_full_parameters(pfree_cont_tied, pfix_cont_tied, lock_cont_tied, tied_pairs_cont, tied_indices_cont, n_tied_cont)
        ptot_lines = rebuild_full_parameters(pfree_lines_tied, pfix_lines_tied, lock_lines_tied, tied_pairs_lines, tied_indices_lines, n_tied_lines)
    
        # First, get the extinction profile
        fopt = fit_options(cube_fitter)
        ext_gas, _, _ = extinction_profiles(s.λ, ptot_cont, 1, fopt.fit_uv_bump, fopt.extinction_curve)
        # Generate the lines 
        Ilines = model_line_residuals(s, ptot_lines, punits_lines, model(cube_fitter).lines, cube_fitter.lsf, ext_gas, trues(length(s.λ)))
        # Subtract the lines
        s.I .-= Ilines
        # Generate the continuum model (with the mask_lines option set to false)
        Icont, ext_gas = model_continuum(s, s.N, ptot_cont, punits_cont, cube_fitter, fopt.use_pah_templates, true, false; return_extinction=true)
        # Add the lines back in
        s.I .+= Ilines

        # Return the sum of the models
        (Icont .+ Ilines)[spaxel_mask]
    end

    # Combine parameters
    pnames = [pnames_cont; names_lines]
    p₀ = [pfree_cont_tied; pfree_lines_tied]
    lower_bounds = [lbfree_cont_tied; lbfree_lines_tied]
    upper_bounds = [ubfree_cont_tied; ubfree_lines_tied]
    punits = [punits_cont; punits_lines]
    dstep = [dfree_cont_tied; dfree_lines_tied]

    _fit_global = false
    # max_ncomps = maximum([length(prof) for prof in model(cube_fitter).lines.profiles])
    # if (init || use_ap || fopt.fit_all_global) && (max_ncomps > 1) && !bootstrap_iter
    if fopt.fit_all_global && !bootstrap_iter
        # @info "Detected that you are fitting multiple velocity components for emission lines. We will perform a global minimization (this may take ~20 minutes or so)."
        p₁ = perform_global_SAMIN_fit(s, cube_fitter, p₀, lower_bounds, upper_bounds, fit_joint; spaxel_mask=spaxel_mask)
        _fit_global = true
    else
        p₁ = p₀
    end

    # Parinfo and config objects
    parinfo, config = get_continuum_parinfo(n_free_cont+n_free_lines, lower_bounds, upper_bounds, dstep)

    res = cmpfit(s.λ[spaxel_mask], s.I[spaxel_mask], s.σ[spaxel_mask], fit_joint, p₁, parinfo=parinfo, config=config)
    res = repeat_fit_jitter(s.λ[spaxel_mask], s.I[spaxel_mask], s.σ[spaxel_mask], fit_joint, p₁, lower_bounds, upper_bounds, 
        parinfo, config, res, "continuum+lines", s.coords; check_cont_snr=true)

    @debug "Continuum+Lines CMPFit Status: $(res.status)"
    popt_cont, perr_cont, I_cont, comps_cont, norms, popt_lines, perr_lines, I_lines, comps_lines, result_stellar, s_model = 
        collect_fit_results(res, pfix_cont_tied, lock_cont_tied, punits_cont, tied_pairs_cont, tied_indices_cont,
        n_tied_cont, n_free_cont, pfix_lines_tied, lock_lines_tied, punits_lines, tied_pairs_lines, tied_indices_lines,
        n_tied_lines, n_free_lines, s, cube_fitter; bootstrap_iter=bootstrap_iter)

    # Estimate PAH template amplitude
    pah_amp = estimate_pah_template_amplitude(cube_fitter, s.λ, comps_cont)

    # Create a fit result object
    bounds_cont = [[plim[1] for plim in plims_cont].*punits_cont [plim[2] for plim in plims_cont].*punits_cont]
    result_cont = SpaxelFitResult(pnames_cont, popt_cont, [perr_cont perr_cont], bounds_cont, lock_cont, tie_vec_cont)
    pretty_print_results(result_cont)

    bounds_lines = [[plim[1] for plim in plims_lines].*punits_lines [plim[2] for plim in plims_lines].*punits_lines]
    result_lines = SpaxelFitResult(names_lines, popt_lines, [perr_lines perr_lines], bounds_lines, lock_lines, tie_vec_lines)
    pretty_print_results(result_lines)

    if init
        # save_init_fit_outputs!(cube_fitter, ustrip.(popt_cont), ustrip.(pah_amp))
        # save_init_fit_outputs!(cube_fitter, ustrip.(popt_lines))
        result_pahamp = SpaxelFitResult(["dust_features.PAH_template1.amp", "dust_features.PAH_template2.amp"],
            pah_amp, [0. 0.; 0. 0.], [-Inf Inf; -Inf Inf], falses(2), Union{Tie,Nothing}[nothing, nothing])
        write_fit_results_csv(cube_fitter, "init_fit_cont", result_cont)
        write_fit_results_csv(cube_fitter, "init_fit_pahtemp", result_pahamp)
        write_fit_results_csv(cube_fitter, "init_fit_line", result_lines)

        # Save results to the cube_fitter and to files
        # Clean up the initial parameters to be more amenable to individual spaxel fits
        popt_lines = clean_line_parameters(cube_fitter, popt_lines, ustrip.(bounds_lines[:,1]), ustrip.(bounds_lines[:,2]))
        cube_fitter.p_init_cont[:] .= popt_cont
        cube_fitter.p_init_line[:] .= popt_lines
    end

    result_cont, result_lines, I_cont, I_lines, comps_cont, norms, comps_lines, n_free_cont, n_free_lines, pah_amp, result_stellar, s_model
end


# Helper function for fitting one iteration (i.e. for bootstrapping)
function _fit_spaxel_iterfunc(spaxel::Spaxel, cube_fitter::CubeFitter; bootstrap_iter::Bool=false, 
    p1_boots_c::Union{Vector{<:Real},Nothing}=nothing, p1_boots_l::Union{Vector{<:Real},Nothing}=nothing, 
    p1_boots_pah::Union{Vector{<:Real},Nothing}=nothing, use_ap::Bool=false, init::Bool=false)

    fopt = fit_options(cube_fitter)
    # organize bootstrap fit components depending on if PAHs are being jointly fit
    p1_boots_cont = p1_boots_c
    if fopt.use_pah_templates && bootstrap_iter
        p1_boots_cont = [p1_boots_cont; p1_boots_pah]
    end

    if !fopt.fit_joint
        # First, do the continuum fit 
        result_c, I_cont, comps_cont, norms, pahtemp, result_stellar, spaxel_model, _, n_free_c = continuum_fit_spaxel(
            spaxel, cube_fitter, fopt.use_pah_templates; init=init, use_ap=use_ap, bootstrap_iter=bootstrap_iter, 
            p1_boots=p1_boots_cont)
        # Decide whether to subtract the real continuum fit or a cubic spline continuum fit based on the settings 
        line_cont = get_continuum_for_line_fit(spaxel, cube_fitter, I_cont, comps_cont)
        # Then do the line fit
        result_l, I_line, comps_line, n_free_l = line_fit_spaxel(spaxel, spaxel_model, cube_fitter, line_cont, 
            comps_cont["total_extinction_gas"]; init=init, use_ap=use_ap, bootstrap_iter=bootstrap_iter, p1_boots=p1_boots_l)
    else
        # Fit the continuum and lines together
        result_c, result_l, I_cont, I_line, comps_cont, norms, comps_line, n_free_c, n_free_l, pahtemp, result_stellar, spaxel_model = 
            all_fit_spaxel(spaxel, cube_fitter; init=init, use_ap=use_ap, bootstrap_iter=bootstrap_iter, p1_boots_cont=p1_boots_cont, 
            p1_boots_line=p1_boots_l)
        line_cont = get_continuum_for_line_fit(spaxel, cube_fitter, I_cont, comps_cont)
    end
    # Get the total fit results
    I_model, comps, χ2, dof = collect_total_fit_results(spaxel, spaxel_model, cube_fitter, I_cont, I_line, comps_cont, comps_line, 
        n_free_c, n_free_l)

    result = copy(result_c)
    combine!(result, result_l)

    if !init
        # The extra parameters (fluxes, equivalent widths, SNRs, etc.)
        result_e = calculate_extra_parameters(spaxel, spaxel_model, cube_fitter, comps, result_c, result_l, line_cont, !bootstrap_iter)
        # The chi2 and dof 
        result_s = SpaxelFitResult(["statistics.chi2", "statistics.dof"], [χ2, dof], [0. 0.; 0. 0.], [-Inf Inf; -Inf Inf],
            falses(2), Vector{Union{Nothing,Tie}}([nothing, nothing]))
        
        # Add the extra parameters to the final result object
        combine!(result, result_e)
        combine!(result, result_s)

    end

    return result, result_c, result_l, result_stellar, pahtemp, I_model, comps, spaxel_model, χ2, dof
end


function _fit_bootstraps(spax::Spaxel, spax_model::Spaxel, cube_fitter::CubeFitter, I_model::Vector{<:QSIntensity}, 
    result::SpaxelFitResult, result_c::SpaxelFitResult, result_l::SpaxelFitResult, pahtemp::Vector{<:Real}, 
    fname::String; use_ap::Bool=false)

    # Set the random seed
    fopt = fit_options(cube_fitter)
    @debug "Setting the random seed to: $(fopt.random_seed)"
    Random.seed!(fopt.random_seed)
    # Resample the data using normal distributions with the statistical uncertainties
    I_boots = [rand.(Normal.(spax.I, spax.σ)) for _ in 1:fopt.n_bootstrap]
    # Create a 2D array buffer for the bootstrapped parameters
    p_boot = Matrix{Quantity{Float64}}(undef, length(result.popt), fopt.n_bootstrap)
    # Create a buffer for the bootstrapped models
    I_model_boot = Matrix{eltype(I_model)}(undef, length(I_model), fopt.n_bootstrap)
    # Save the original spaxel intensity
    I0 = copy(spax.I)

    # Bootstrapping 
    @debug "Performing bootstrapping iterations for spaxel $(spax.coords)..."

    # ---> I'm disabling threads here for now because it causes something fucky in the results
    #      and I have no idea what the root cause is...
    # Threads.@threads for nboot ∈ 1:fopt.n_bootstrap
    for nboot ∈ 1:fopt.n_bootstrap
        lock(file_lock) do
            @debug "Bootstrap iteration: $nboot"
        end
        # Get the bootstrapped data vector for this iteration
        # N.b.: Don't use rand() inside a Threads loop! It is not thread-safe, i.e. the RNG will not be consistent
        #  even after setting the seed
        spax.I .= I_boots[nboot]

        # Re-perform the fitting on the resampled data
        result_boot, _, _, _, _, Ib_i, = with_logger(NullLogger()) do
            _fit_spaxel_iterfunc(spax, cube_fitter; bootstrap_iter=true, p1_boots_c=ustrip.(result_c.popt), 
                p1_boots_l=ustrip.(result_l.popt), p1_boots_pah=ustrip.(pahtemp), use_ap=use_ap)
        end
        # Sort the emission line components (if not using relative parameters) to avoid multi-modal distirbutions
        # only mask additional line components after the first bootstrap so that they have a minimum of 1 finite sample
        sort_line_components!(cube_fitter, result_boot, mask_zeros=nboot>1)

        p_boot[:, nboot] .= result_boot.popt
        I_model_boot[:, nboot] .= Ib_i
    end
    # reset the spaxel intensity
    spax.I = I0

    # RESULTS: Values are the 50th percentile, and errors are the (15.9th, 84.1st) percentiles
    open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$(fname)_p_boots.csv"), "w") do f
        writedlm(f, p_boot) 
    end
    split1 = length(result_c.popt)
    split2 = length(result_c.popt) + length(result_l.popt)
    
    p_out, p_err, I_boot_min, I_boot_max, I_model, comps, norms, χ2 = collect_bootstrapped_results(spax, spax_model, cube_fitter, 
        p_boot, I_model_boot, split1, split2)
    result.popt .= p_out
    result.perr .= p_err
    result_stellar = nothing
    if fopt.fit_stellar_continuum
        result_stellar = calculate_stellar_parameters(cube_fitter, norms, spax.N)
    end

    result, result_stellar, I_boot_min, I_boot_max, I_model, comps, χ2
end


"""
    fit_spaxel(cube_fitter, cube_data, spaxel; [use_ap])

Wrapper function to perform a full fit of a single spaxel, calling `continuum_fit_spaxel` and `line_fit_spaxel` and
concatenating the best-fit parameters. The outputs are also saved to files so the fit need not be repeated in the case
of a crash.

# Arguments
- `cube_fitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `cube_data`: Contains the wavelength, intensity, and error vectors that will be fit, as well as the solid angle vector
- `coords`: The coordinates of the spaxel to be fit
- `use_ap`: Flag determining whether or not one is fitting an integrated spectrum within an aperture
- `use_vorbins`: Flag for using voronoi binning 
- `σ_min`: Optional minimum uncertainty to add in quadrature with the statistical uncertainty. Only applies if
    use_ap or use_vorbins is true.
"""
function fit_spaxel(cube_fitter::CubeFitter, cube_data::NamedTuple, coords::CartesianIndex; use_ap::Bool=false,
    use_vorbins::Bool=false, σ_min::Vector=zeros(eltype(cube_data.σ), size(cube_data.σ)[end]))

    # Check if the fit has already been performed
    fopt = fit_options(cube_fitter)
    oopt = out_options(cube_fitter)
    fname = use_vorbins ? "voronoi_bin_$(coords[1])" : "spaxel_$(coords[1])_$(coords[2])"
    if isfile(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.csv")) && 
        (!fopt.fit_stellar_continuum || isfile(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.ssp"))) && 
        !oopt.overwrite
        # Otherwise, just grab the results from before
        p_out, p_err = read_fit_results_csv(cube_fitter.name, fname)
        np_ssp = 0
        if fopt.fit_stellar_continuum
            rs = deserialize(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.ssp"))
            np_ssp = length(rs.ages)
        end
        return p_out, p_err, np_ssp
    end

    # if there are any NaNs, skip over the spaxel
    if any(.!isfinite.(cube_data.I[coords, :]))
        @debug "Non-finite values found in the intensity! Not fitting spaxel $coords"
        return nothing, nothing, nothing
    end
    for t in 1:cube_fitter.n_templates
        if any(.~isfinite.(cube_data.templates[coords, :, t]))
            # If a template is fully NaN/Inf, we can't fit 
            @debug "Non-finite values found in the templates! Not fitting spaxel $coords"
            return nothing, nothing, nothing
        end
    end

    # Make a spaxel object
    spax = make_normalized_spaxel(cube_data, coords, cube_fitter; use_ap=use_ap, use_vorbins=use_vorbins, σ_min=σ_min)
    # Overwrite the raw errors with the updated errors
    # cube_data.σ[coords, :] .= spax.σ .* spax.N

    # Check that there is enough data within the masked region
    spaxel_mask = .~get_vector_mask(spax; lines=true, user_mask=cube_fitter.cube.spectral_region.mask)
    if length(spax.I[spaxel_mask]) < (cube_fitter.n_params_cont + cube_fitter.n_params_lines)
        @debug "The masked region is too small to contain enough data to fit! Not fitting spaxel $coords"
        return nothing, nothing, nothing
    end

    # This log should be entirely handled by 1 process, since each spaxel is entirely handled by 1 process
    # so there should be no problems w.ith I/O race conditions
    spaxel_logger = make_spaxel_logger(cube_fitter.name, fname)

    result, np_ssp = with_logger(spaxel_logger) do

        # Perform the regular fit
        result, result_c, result_l, result_stellar, pahtemp, I_model, comps, spax_model, χ2, dof = _fit_spaxel_iterfunc(
            spax, cube_fitter; bootstrap_iter=false, use_ap=use_ap)

        # Perform the bootstrapping iterations, if n_bootstrap > 0
        I_boot_min = I_boot_max = nothing
        if fopt.n_bootstrap > 0
            result, result_stellar, I_boot_min, I_boot_max, I_model, comps, χ2 = _fit_bootstraps(
                spax, spax_model, cube_fitter, I_model, result, result_c, result_l, pahtemp, fname; use_ap=use_ap)
        end

        # Plot the fit
        if oopt.plot_spaxels != :none
            @debug "Plotting spaxel $coords best fit" 
            plot_spaxel_fit(cube_fitter, spax, spax_model, I_model, comps, χ2/dof, fname; backend=oopt.plot_spaxels, I_boot_min=I_boot_min,
                I_boot_max=I_boot_max, logy=false)
            if !isnothing(oopt.plot_range)
                for (i, plot_range) ∈ enumerate(oopt.plot_range)
                    fname2 = use_vorbins ? "lines_bin_$(spax.coords[1])_$i" : "lines_$(spax.coords[1])_$(spax.coords[2])_$i"
                    plot_spaxel_fit(cube_fitter, spax, spax_model, I_model, comps, χ2/dof, fname2; backend=oopt.plot_spaxels, 
                        I_boot_min=I_boot_min, I_boot_max=I_boot_max, range=plot_range, logy=false)
                end
            end
        end

        @debug "Saving results to csv for spaxel $(spax.coords)"
        # serialize(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).LOKI"), (p_out=p_out, p_err=p_err))
        # save output as csv file
        write_fit_results_csv(cube_fitter, fname, result)

        # Plot and save the stellar populations, if relevant
        np_ssp = 0
        if fopt.fit_stellar_continuum && (oopt.plot_spaxels != :none)
            plot_stellar_grids(result_stellar, cube_fitter, fname)
            serialize(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.ssp"), result_stellar)
            np_ssp = length(result_stellar.ages)
        end
        
        # save memory allocations & other logistic data to a separate log file
        if oopt.track_memory
            write_memory_log(cube_fitter, fname)
        end

        result, np_ssp
    end

    return result.popt, result.perr, np_ssp
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
    I_sum_init, σ_sum_init, templates_init, area_sr_init = get_total_integrated_intensities(cube_fitter)

    # Perform a cubic spline fit, also obtaining the line mask
    mask_lines_init, I_spline_init, _ = continuum_cubic_spline(λ_init, I_sum_init, σ_sum_init, cube_fitter.linemask_overrides)
    mask_bad_init = iszero.(I_sum_init) .| iszero.(σ_sum_init)
    mask_init = mask_lines_init .| mask_bad_init
    mask_chi2_init = mask_bad_init
    σ_sum_init .= calculate_statistical_errors(I_sum_init, I_spline_init, mask_init)

    # Mask out the chip gaps in NIRSPEC observations 
    # (they would technically work considering we are measuring surface brightness and not flux, but in practice it doesn't because 
    # we are covering up regions that may drastically differ in brightness across different spatial regions in the cube, so it leads 
    # to aritifical dips in the continuum)
    if fit_options(cube_fitter).nirspec_mask_chip_gaps
        λobs = λ_init .* (1 .+ cube_fitter.z)
        for chip_gap in chip_gaps_nir
            mask_bad_init .|= (chip_gap[1] .< λobs .< chip_gap[2])
            σ_sum_init[mask_bad_init] .= maximum(σ_sum_init) .* 100.   # Inflate errors over the chip gap
        end
    end

    # Get the normalization
    norm = abs(nanmaximum(ustrip.(I_sum_init))) .* unit(I_sum_init[1])
    norm = norm ≠ 0. ? norm : 1.
    vres_init = isapprox((λ_init[2]/λ_init[1]), (λ_init[end]/λ_init[end-1]), rtol=1e-6) ? 
        log(λ_init[2]/λ_init[1]) * C_KMS : nothing

    # Create a spaxel object and normalize it
    aux = Dict("templates" => templates_init, "channel_masks" => cube_fitter.spectral_region.channel_masks)
    s_init = Spaxel(CartesianIndex(0,0), λ_init, I_sum_init, σ_sum_init, area_sr_init, mask_lines_init, mask_bad_init,
        vres_init, I_spline_init; aux=aux)
    normalize!(s_init, norm)
    fill_bad_pixels!(s_init)

    # No bootstrapping on the first integrated fit
    _, _, _, result_stellar, pahtemp, I_model_init, comps_init, s_init_model, χ2_init, dof_init = _fit_spaxel_iterfunc(
        s_init, cube_fitter; bootstrap_iter=false, use_ap=false, init=true)
    χ2red_init = χ2_init / dof_init

    # Plot the fit
    fopt = fit_options(cube_fitter)
    oopt = out_options(cube_fitter)
    if oopt.plot_spaxels != :none
        @debug "Plotting spaxel sum initial fit"
        plot_spaxel_fit(cube_fitter, s_init, s_init_model, I_model_init, comps_init, χ2red_init, "initial_sum_fit"; backend=:both, logy=false)
        if !isnothing(oopt.plot_range)
            for (i, plot_range) ∈ enumerate(oopt.plot_range)
                plot_spaxel_fit(cube_fitter, s_init, s_init_model, I_model_init, comps_init, χ2red_init, "initial_sum_line_$i"; 
                    backend=:both, range=plot_range)
            end
        end
    end

    # Plot the stellar matrix
    if fopt.fit_stellar_continuum && (oopt.plot_spaxels != :none)
        plot_stellar_grids(result_stellar, cube_fitter, "initial_sum_fit")
    end

end


# Helper function that loops over previously fit spaxels and adds their results into
# the out_params and out_errs
function spaxel_loop_previous(cube_fitter::CubeFitter, cube_data::NamedTuple, spaxels::Vector, 
    vorbin::Bool, out_params::AbstractArray{<:Real,3}, out_errs::AbstractArray{<:Real,4}, out_np_ssp::AbstractArray{Int,2})

    _m2 = falses(length(spaxels))
    if !(out_options(cube_fitter).overwrite)
        @info "Loading in previous fit results (if any...)"
        @showprogress for (ii, index) ∈ enumerate(spaxels)
            fname = vorbin ? "voronoi_bin_$(index[1])" : "spaxel_$(index[1])_$(index[2])"
            fpath1 = joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.csv")
            fpath2 = joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.ssp")
            if isfile(fpath1) && (!fit_options(cube_fitter).fit_stellar_continuum || isfile(fpath2))
                # just loads in the previous fit results since they already exist
                p_out, p_err, np_ssp = fit_spaxel(cube_fitter, cube_data, index; use_vorbins=vorbin)
                if vorbin
                    out_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(index)[1])
                    for out_index in out_indices
                        out_params[out_index, :] .= ustrip.(p_out)
                        out_errs[out_index, :, :] .= ustrip.(p_err)
                        out_np_ssp[out_index] = np_ssp
                    end
                else
                    out_params[index, :] .= ustrip.(p_out)
                    out_errs[index, :, :] .= ustrip.(p_err)
                    out_np_ssp[index] = np_ssp
                end
                # remove this spaxel from the list of spaxels that still need to be fit
                _m2[ii] = 1
            end
        end
    end

    _m2
end


# Helper function that loops over spaxels using pmap to fit them in parallel
function spaxel_loop_pmap!(cube_fitter::CubeFitter, cube_data::NamedTuple, spaxels::Vector,
    vorbin::Bool, out_params::AbstractArray{<:Real,3}, out_errs::AbstractArray{<:Real,4}, 
    out_np_ssp::AbstractArray{Int,2})

    @info "Number of spaxels to fit: $(length(spaxels))"

    prog = Progress(length(spaxels))
    # => create a closure where the only actual function argument is the spaxel index; 
    #    I think this, in combination with the default Caching Pool, means the large cube_fitter 
    #    and cube_data structures dont have to all be passed to the individual workers
    pmap_result = progress_pmap(spaxels, progress=prog, retry_delays=ones(1)) do index
        p_out, p_err, np_ssp = fit_spaxel(cube_fitter, cube_data, index; use_vorbins=vorbin) 
        return index, p_out, p_err, np_ssp
    end

    # Loop over the results and insert them into the output arrays at the correct indices
    for (index, p_out, p_err, np_ssp) in pmap_result
        if !isnothing(p_out)
            if vorbin
                out_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(index)[1])
                for out_index in out_indices
                    out_params[out_index, :] .= ustrip.(p_out)
                    out_errs[out_index, :, :] .= ustrip.(p_err)
                    out_np_ssp[out_index] = np_ssp
                end
            else
                out_params[index, :] .= ustrip.(p_out)
                out_errs[index, :, :] .= ustrip.(p_err)
                out_np_ssp[index] = np_ssp
            end
        end
    end

    # We no longer need the extra worker processes after this point
    rmprocs(workers())

end


# Helper function that loops over spaxels using distributed to fit them in parallel
function spaxel_loop_distributed!(cube_fitter::CubeFitter, cube_data::NamedTuple,
    spaxels::Vector, vorbin::Bool, out_params::AbstractArray{<:Real,3}, 
    out_errs::AbstractArray{<:Real,4}, out_np_ssp::AbstractArray{Int,2})

    error("Sorry, the \"distributed\" parallelization option is not currently implemented!")
end


# Helper function that loops over spaxels, fitting them in sequence (not parallel)
function spaxel_loop_serial!(cube_fitter::CubeFitter, cube_data::NamedTuple, spaxels::Vector,
    vorbin::Bool, out_params::AbstractArray{<:Real,3}, out_errs::AbstractArray{<:Real,4}, out_np_ssp::AbstractArray{Int,2})

    @info "Number of spaxels to fit: $(length(spaxels))"

    prog = Progress(length(spaxels))
    for index ∈ spaxels
        p_out, p_err, np_ssp = fit_spaxel(cube_fitter, cube_data, index; use_vorbins=vorbin)
        if !isnothing(p_out)
            if vorbin
                out_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(index)[1])
                for out_index in out_indices
                    out_params[out_index, :] .= ustrip.(p_out)
                    out_errs[out_index, :, :] .= ustrip.(p_err)
                    out_np_ssp[out_index] = np_ssp
                end
            else
                out_params[index, :] .= ustrip.(p_out)
                out_errs[index, :, :] .= ustrip.(p_err)
                out_np_ssp[index] = np_ssp
            end
        end
        next!(prog)
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

    # if the "cube" is a size 1x1xN, call the aperture method with the "all" keyword
    shape = size(cube_fitter.cube.I)
    if shape[1:2] == (1,1)
        return fit_cube!(cube_fitter, "all")
    end

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

    # Prepare output array
    @info "===> Preparing output data structures... <==="
    out_params = Array{Float64}(undef, shape[1:2]..., cube_fitter.n_params_total)
    out_params .= NaN
    out_errs = Array{Float64}(undef, shape[1:2]..., cube_fitter.n_params_total, 2)
    out_errs .= NaN
    out_np_ssp = zeros(Int, shape[1:2]...)

    # "cube_data" object holds the primary wavelength, intensity, and errors
    # this is just a convenience object since these may be different when fitting an integrated spectrum
    # within an aperture, or when using voronoi bins
    cube_data, vorbin, n_bins = create_cube_data(cube_fitter, shape)

    ######################### DO AN INITIAL FIT WITH THE SUM OF ALL SPAXELS ###################

    @debug """
    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
    """

    # Don't repeat if it's already been done, and also dont do the initial fit if we're just fitting in an aperture
    if all(iszero.(cube_fitter.p_init_cont)) || all(iszero.(cube_fitter.p_init_line))
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
    if vorbin spaxels = CartesianIndices((n_bins,)) end
    # Only loop over spaxels with data
    _m = [any(.~isfinite.(cube_data.I[spaxel, :])) || any(.~isfinite.(cube_data.templates[spaxel, :, :])) for spaxel in spaxels]
    spaxels = spaxels[.~_m]

    # First (non-parallel) loop over already-completed spaxels
    _m2 = spaxel_loop_previous(cube_fitter, cube_data, spaxels, vorbin, out_params, out_errs, out_np_ssp)
    spaxels = spaxels[.~_m2]

    fopt = fit_options(cube_fitter)
    oopt = out_options(cube_fitter)

    # Use multiprocessing (not threading) to iterate over multiple spaxels at once using multiple CPUs
    if oopt.parallel && nworkers() > 1
        @info "===> Beginning individual spaxel fitting... <==="
        if oopt.parallel_strategy == "pmap"
            spaxel_loop_pmap!(cube_fitter, cube_data, spaxels, vorbin, out_params, out_errs, out_np_ssp)
        elseif oopt.parallel_strategy == "distributed"
            spaxel_loop_distributed!(cube_fitter, cube_data, spaxels, vorbin, out_params, out_errs, out_np_ssp)
        else
            error("Unrecognized parallel strategy: $(oopt.parallel_strategy). May be \"pmap\" or \"distributed\".")
        end
    else
        @info "===> Beginning individual spaxel fitting... <==="
        spaxel_loop_serial!(cube_fitter, cube_data, spaxels, vorbin, out_params, out_errs, out_np_ssp)
    end

    @info "===> Generating parameter maps and model cubes... <==="
    # apply the units
    out_units = get_units(get_flattened_parameters(model(cube_fitter)))
    out_params = out_params .* repeat(reshape(out_units, (1,1,length(out_units))), outer=[shape[1],shape[2],1])
    out_errs = out_errs .* repeat(reshape(out_units, (1,1,length(out_units),1)), outer=[shape[1],shape[2],1,2])

    # Create the ParamMaps and CubeModel structs containing the outputs
    param_maps, param_units, cube_model = assign_outputs(out_params, out_errs, out_np_ssp, cube_fitter, cube_data, false)

    if oopt.plot_maps
        @info "===> Plotting parameter maps... <==="
        plot_parameter_maps(cube_fitter, param_maps, snr_thresh=oopt.map_snr_thresh)
    end

    if oopt.save_fits
        @info "===> Writing FITS outputs... <==="
        write_fits(cube_fitter, cube_data, cube_model, param_maps, param_units)
    end
    if oopt.save_tables
        @info "===> Writing CSV tables... <=== "
        write_table(cube_fitter, param_maps, param_units)
    end

    # Save a copy of the options file used to run the code, so the settings can be referenced/reused
    # (for example, if you need to recall which lines you tied, what your limits were, etc.)
    cp(joinpath(@__DIR__, "..", "options", "options.toml"), joinpath("output_$(cube_fitter.name)", "options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "optical.toml"), joinpath("output_$(cube_fitter.name)", "optical.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "infrared.toml"), joinpath("output_$(cube_fitter.name)", "infrared.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "lines.toml"), joinpath("output_$(cube_fitter.name)", "lines.archive.toml"), force=true)

    if oopt.make_movies
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


# Fit one integrated spectrum within an aperture
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
    out_params = Array{Float64}(undef, shape[1:2]..., cube_fitter.n_params_total)
    out_params .= NaN
    out_errs = Array{Float64}(undef, shape[1:2]..., cube_fitter.n_params_total, 2)
    out_errs .= NaN
    out_np_ssp = zeros(Int, shape[1:2]...)

    # Loop through each wavelength pixel and perform the aperture photometry
    if aperture isa String
        @assert lowercase(aperture) == "all" "The only accepted string input for 'aperture' is 'all' to signify the entire cube."
        I, σ, templates, area_sr = get_total_integrated_intensities(cube_fitter; shape=shape)
    else
        I, σ, templates, area_sr = get_aperture_integrated_intensities(cube_fitter, shape, aperture)
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
    p_out, p_err, np_ssp = fit_spaxel(cube_fitter, cube_data, spaxels[1]; use_ap=true)
    if !isnothing(p_out)
        out_params[spaxels[1], :] .= ustrip.(p_out)
        out_errs[spaxels[1], :, :] .= ustrip.(p_err)
        out_np_ssp[spaxels[1]] = np_ssp
    end

    @info "===> Generating parameter maps and model cubes... <==="
    # apply the units
    out_units = get_units(get_flattened_parameters(model(cube_fitter)))
    out_params = out_params .* repeat(reshape(out_units, (1,1,length(out_units))), outer=[shape[1],shape[2],1])
    out_errs = out_errs .* repeat(reshape(out_units, (1,1,length(out_units),1)), outer=[shape[1],shape[2],1,2])

    # Create the ParamMaps and CubeModel structs containing the outputs
    param_maps, param_units, cube_model = assign_outputs(out_params, out_errs, out_np_ssp, cube_fitter, cube_data, true)

    oopt = out_options(cube_fitter)
    if oopt.save_fits
        @info "===> Writing FITS outputs... <==="
        write_fits(cube_fitter, cube_data, cube_model, param_maps, param_units, aperture=aperture)
    end
    if oopt.save_tables
        @info "===> Writing CSV tables... <=== "
        write_table(cube_fitter, param_maps, param_units)
    end

    # Save a copy of the options file used to run the code, so the settings can be referenced/reused
    # (for example, if you need to recall which lines you tied, what your limits were, etc.)
    cp(joinpath(@__DIR__, "..", "options", "options.toml"), joinpath("output_$(cube_fitter.name)", "general_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "infrared.toml"), joinpath("output_$(cube_fitter.name)", "dust_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "lines.toml"), joinpath("output_$(cube_fitter.name)", "lines_options.archive.toml"), force=true)
    cp(joinpath(@__DIR__, "..", "options", "optical.toml"), joinpath("output_$(cube_fitter.name)", "optical_options.archive.toml"), force=true)

    if oopt.make_movies
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
    post_fit_nuclear_template!(cube_fitter, agn_templates, psf_norm)

A helper function to fit a model to the nuclear template spectrum when decomposing a bright QSO into the QSO portion
dispersed by the PSF and the host galaxy portion. This function is to be used AFTER fitting on the full cube has been
performed and will fit a model to the integrated QSO spectrum over the full FOV of the cube. This has the advantage of
including the normalizations of the QSO spectrum in each individual spaxel so that the overall model is consistent.
"""
function post_fit_nuclear_template!(cube_fitter::CubeFitter, agn_templates::Array{<:QSIntensity,3}, psf_norm::Array{<:Real,3})

    @info """\n
    BEGINNING NUCELAR TEMPLATE FITTING ROUTINE FOR $(cube_fitter.name)
    """
    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    shape = (1,1,size(cube_fitter.cube.I, 3))

    fopt = fit_options(cube_fitter)
    oopt = out_options(cube_fitter)
    # Unlock the hot dust component for the nuclear template fits
    fopt.lock_hot_dust = false

    # Prepare output array
    @info "===> Preparing output data structures... <==="
    out_params = Array{Float64}(undef, shape[1:2]..., cube_fitter.n_params_total)
    out_params .= NaN
    out_errs = Array{Float64}(undef, shape[1:2]..., cube_fitter.n_params_total, 2)
    out_errs .= NaN
    out_np_ssp = zeros(Int, shape[1:2]...)
    cube_data = create_cube_data_postnuctemp(cube_fitter, agn_templates)

    @debug """
    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
    """
    # copy the main log file
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)

    λ_init = cube_fitter.cube.λ
    I_sum_init, σ_sum_init, templates_init, area_sr_init = get_total_integrated_intensities(cube_fitter)

    # Perform a cubic spline fit, also obtaining the line mask
    mask_lines_init, I_spline_init, _ = continuum_cubic_spline(λ_init, I_sum_init, σ_sum_init, cube_fitter.linemask_overrides)
    mask_bad_init = iszero.(I_sum_init) .| iszero.(σ_sum_init)
    mask_init = mask_lines_init .| mask_bad_init
    mask_chi2_init = mask_bad_init
    σ_sum_init  .= calculate_statistical_errors(I_sum_init, I_spline_init, mask_init)
    σ_sum_init .*= nanmedian(cube_data.I./I_sum_init)

    @info "===> Beginning nuclear spectrum fitting... <==="
    # Add a minimum uncertainty to prevent the statistical uncertainties on the model from being close to 0 due to it being
    # a smooth model rather than actual data
    p_out, p_err, np_ssp = fit_spaxel(cube_fitter, cube_data, CartesianIndex(1,1); use_ap=true, σ_min=σ_sum_init)
    if !isnothing(p_out)
        out_params[1, 1, :] .= ustrip.(p_out)
        out_errs[1, 1, :, :] .= ustrip.(p_err)
        out_np_ssp[1,1] = np_ssp
    end

    @info "===> Generating parameter maps and model cubes... <==="
    # apply the units
    out_units = get_units(get_flattened_parameters(model(cube_fitter)))
    out_params = out_params .* repeat(reshape(out_units, (1,1,length(out_units))), outer=[shape[1],shape[2],1])
    out_errs = out_errs .* repeat(reshape(out_units, (1,1,length(out_units),1)), outer=[shape[1],shape[2],1,2])

    # Create the ParamMaps and CubeModel structs containing the outputs
    param_maps, param_units, cube_model = assign_outputs(out_params, out_errs, out_np_ssp, cube_fitter, cube_data, true)

    # Create another set of ParamMaps and CubeModels--this time 3D--containing the features fluxes dispersed by the PSF model
    param_maps_3d, cube_model_3d = assign_qso3d_outputs(param_maps, cube_model, cube_fitter, psf_norm)
    if oopt.plot_maps
        @info "===> Plotting QSO parameter maps... <==="
        plot_parameter_maps(cube_fitter, param_maps_3d, snr_thresh=oopt.map_snr_thresh, qso3d=true)
    end

    if oopt.save_fits
        @info "===> Writing 1D FITS outputs... <==="
        write_fits(cube_fitter, cube_data, cube_model, param_maps, param_units, nuc_temp_fit=false, aperture="all")

        @info "===> Writing 3D FITS outputs... <==="
        write_fits(cube_fitter, cube_data, cube_model_3d, param_maps_3d, param_units, nuc_temp_fit=false, qso3d=true)
    end
    if oopt.save_tables
        @info "===> Writing CSV tables... <=== "
        write_table(cube_fitter, param_maps, param_units)
    end

    # lock the hot dust component for the rest of the fits
    fopt.lock_hot_dust = true

    # copy the main log file again
    cp(joinpath(@__DIR__, "..", "loki.main.log"), joinpath("output_$(cube_fitter.name)", "loki.main.log"), force=true)
    @info "Done!!"

    # Return the final cube_fitter object, along with the param maps/errs and cube model
    cube_fitter, param_maps, cube_model

end

# Different methods for reading from FITS file or cube_model directly:
function post_fit_nuclear_template!(cube_fitter::CubeFitter, params::String, full_cube_model::String, template_name::String)
    hdu_param = FITS(params)
    hdu_model = FITS(full_cube_model)
    post_fit_nuclear_template!(cube_fitter, hdu_param, hdu_model, template_name)
end

function post_fit_nuclear_template!(cube_fitter::CubeFitter, hdu_param::FITS, hdu_model::FITS, template_name::String)

    # get the AGN templates
    ftype = typeof(1.0)
    agn_templates = ftype.(read(hdu_model["TEMPLATES.$(uppercase(template_name))"]))
    # shift back into the rest frame
    if eltype(cube_fitter.cube.I) <: QPerFreq
        agn_templates ./= (1 .+ cube_fitter.z)
    elseif eltype(cube_fitter.cube.I) <: QPerWave
        agn_templates .*= (1 .+ cube_fitter.z)
    end
    agn_templates = agn_templates .* unit(cube_fitter.cube.I[1])

    # get the PSF normalization
    psf_norm = copy(cube_fitter.cube.psf_model)                              # PSF model (integrates to 1)
    psf_norm[.~isfinite.(psf_norm)] .= 0.
    for (i, ch_mask) ∈ enumerate(cube_fitter.cube.spectral_region.channel_masks)
        fit_norm = 10 .^ read(hdu_param["TEMPLATES.$(uppercase(template_name)).AMP_$i"])         # Fit amplitudes
        fit_norm[.~isfinite.(fit_norm)] .= 1.
        for k ∈ findall(ch_mask)
            s = nansum(psf_norm[:, :, k] .* fit_norm) 
            psf_norm[:, :, k] ./= s                                     # Dividing by the sum of (PSF) x (fit amp)
        end                                                                  
    end 

    post_fit_nuclear_template!(cube_fitter, agn_templates, psf_norm)
end

function post_fit_nuclear_template!(cube_fitter::CubeFitter, param_maps::ParamMaps, full_cube_model::CubeModel, template_index::Int=1)

    # get the AGN templates
    agn_templates = full_cube_model.templates[:,:,:,template_index]
    # shift back into the rest frame
    if eltype(cube_fitter.cube.I) <: QPerFreq
        agn_templates ./= (1 .+ cube_fitter.z)
    elseif eltype(cube_fitter.cube.I) <: QPerWave
        agn_templates .*= (1 .+ cube_fitter.z)
    end
    agn_templates = agn_templates .* unit(cube_fitter.cube.I[1])

    # get the PSF normalization
    psf_norm = copy(cube_fitter.cube.psf_model)                              # PSF model (integrates to 1)
    psf_norm[.~isfinite.(psf_norm)] .= 0.
    for (i, ch_mask) ∈ enumerate(cube_fitter.cube.spectral_region.channel_masks)
        fit_norm = 10 .^ get(param_maps, "templates.nuclear.amp_$i")         # Fit amplitudes
        fit_norm[.~isfinite.(fit_norm)] .= 1.
        for k ∈ findall(ch_mask)
            s = nansum(psf_norm[:, :, k] .* fit_norm) 
            psf_norm[:, :, k] ./= s                                     # Dividing by the sum of (PSF) x (fit amp)
        end                                                                  
    end 

    post_fit_nuclear_template!(cube_fitter, agn_templates, psf_norm)
end


"""
    evaluate_model(λ, spaxel, cube_fitter[, cube_fitter_nuc])

A helper function that can be used AFTER fitting a cube, to evaluate the cube model at any arbitrary 
wavelength for any spaxel. This can be used to interpolate models onto a higher or lower resolution, or
to extrapolate models to wavelengths outside the fit region (use with caution).
"""
function evaluate_model(λ::Vector{<:QWave}, spaxel::CartesianIndex, cube_fitter::CubeFitter, 
    output_path::String, cube_fitter_nuc::Union{CubeFitter,Nothing}=nothing, output_path_nuc::Union{String,Nothing}=nothing)

    # Get the optimized fitting parameters for the given spaxel
    cube_data, use_vorbins, n_bins = create_cube_data(cube_fitter, size(cube_fitter.cube.I))
    fname = use_vorbins ? "voronoi_bin_$(spaxel[1])" : "spaxel_$(spaxel[1])_$(spaxel[2])"
    p_out, p_err = read_fit_results_csv(cube_fitter.name, fname; output_path=output_path)
    
    # Split up into continuum & line parameters
    split1 = cube_fitter.n_params_cont
    split2 = cube_fitter.n_params_cont + cube_fitter.n_params_lines
    p_cont = p_out[1:split1]
    p_lines = p_out[split1+1:split2]

    # Get the normalization
    N = Float64(nanmaximum(abs.(ustrip.(cube_data.I[spaxel, :])))) .* unit(cube_data.I[1])

    # The nuclear template must be able to be evaluated at any point
    templates_spax = cube_fitter.templates[spaxel, :, :] ./ N
    channel_masks = cube_fitter.cube.spectral_region.channel_masks

    if !isnothing(cube_fitter_nuc)
        # First, get the nuclear template over the whole FOV
        fname = joinpath(output_path, "$(cube_fitter.name)_full_model.fits")
        hdu_model = FITS(fname)
        agn_templates = read(hdu_model["TEMPLATES.NUCLEAR"])

        I_nuc = evaluate_post_fit_nuclear_template_model(λ, cube_fitter_nuc, agn_templates, output_path_nuc)
        I_nuc .*= nanmaximum(nansum(.~cube_fitter_nuc.cube.mask, dims=(1,2)))   # solid angle factor

        # Read in the parameter maps
        fname = joinpath(output_path, "$(cube_fitter.name)_parameter_maps.fits")
        hdu_param = FITS(fname)

        # Now we need to renormalize to get the value in an individual spaxel
        psf_norm = copy(cube_fitter.cube.psf_model)                              # PSF model (integrates to 1)
        psf_norm[.~isfinite.(psf_norm)] .= 0.
        for (i, ch_mask) ∈ enumerate(cube_fitter.cube.spectral_region.channel_masks)
            fit_norm = 10 .^ read(hdu_param["TEMPLATES.NUCLEAR.AMP_$i"])         # Fit amplitudes
            fit_norm[.~isfinite.(fit_norm)] .= 1.
            for k ∈ findall(ch_mask)
                s = nansum(psf_norm[:, :, k] .* fit_norm) 
                psf_norm[:, :, k] ./= s                                     # Dividing by the sum of (PSF) x (fit amp)
            end                                                                  
        end

        # In pseudo-math terminology, what we've done here is as follows:
        # We started with a template I_nuc given by:
        #        I_nuc = (brightest_spaxel) * Sum[(PSF) * (fit_amp)]
        # where the sum is over (x,y).
        # What we actually want is the individual (x,y) components, i.e.
        #        T(x,y) = (brightest_spaxel) * PSF(x,y) * (fit_amp)(x,y)
        # We can back out T(x,y) from I_nuc, PSF, and fit_amp as follows:
        #        T(x,y) = I_nuc * PSF(x,y) * (fit_amp)(x,y) / Sum[(PSF) * (fit_amp)]
        # During the model generation function, the templates are automatically multiplied by (fit_amp)(x,y),
        # so the inputs must have this factor removed.  Thus the final value we are looking for is
        #        T'(x,y) = I_nuc * PSF(x,y) / Sum[(PSF) * (fit_amp)]
        # In the code, psf_norm represents the term that we multiply with I_nuc to obtain T'(x,y).
        # This must be performed separately at every z index (wavelength).

        # Now we must interpolate psf_norm onto the proper wavelength grid
        # Assume that the PSF normalization stays consistent past the boundaries
        psf_norm_spax = Spline1D(ustrip.(cube_fitter.cube.λ), psf_norm[spaxel, :]; s=0., bc="nearest")
        I_nuc_spax = I_nuc .* psf_norm_spax(λ)

        templates_spax = Matrix{Float64}(undef, length(λ), 1)
        templates_spax[:, 1] .= I_nuc_spax ./ N

        # Create new channel masks carefully -- dont add new channel masks if the extrapolation extends into other 
        # wavebands that could've been covered, just add them into the existing channel masks
        channel_masks = Vector{BitVector}()
        for i ∈ eachindex(cube_fitter.cube.spectral_region.channel_masks)
            bounds = extrema(cube_fitter.cube.λ[cube_fitter.cube.spectral_region.channel_masks[i]])
            dλ = mean(diff(cube_fitter.cube.λ[cube_fitter.cube.spectral_region.channel_masks[i]]))
            if i == 1 && length(cube_fitter.cube.spectral_region.channel_masks) == 1
                mask_i = trues(length(λ))
            elseif i == 1
                mask_i = λ .< (bounds[2]+dλ/2)
            elseif i == length(cube_fitter.cube.spectral_region.channel_masks)
                mask_i = λ .> (bounds[1]-dλ/2)
            else
                mask_i = (bounds[1]-dλ/2) .< λ .< (bounds[2]+dλ/2)
            end
            push!(channel_masks, mask_i)
        end
    end

    _evaluate_model(λ, cube_fitter, ustrip.(p_cont), ustrip.(p_lines), unit.(p_cont), unit.(p_lines), N, 
        templates_spax, channel_masks)
end


# Method for using apertures
function evaluate_model(λ::Vector{<:QWave}, aperture::Aperture.AbstractAperture, cube_fitter::CubeFitter, 
    output_path::String, cube_fitter_nuc::Union{CubeFitter,Nothing}=nothing, output_path_nuc::Union{String,Nothing}=nothing)
    apertures = repeat([aperture], length(cube_fitter.cube.λ))
    evaluate_model(λ, apertures, cube_fitter, output_path, cube_fitter_nuc, output_path_nuc)
end


# Method for using apertures
function evaluate_model(λ::Vector{<:QWave}, aperture::Union{Vector{<:Aperture.AbstractAperture},String}, cube_fitter::CubeFitter, 
    output_path::String, cube_fitter_nuc::Union{CubeFitter,Nothing}=nothing, output_path_nuc::Union{String,Nothing}=nothing)

    # Loop through each wavelength pixel and perform the aperture photometry
    shape = (1,1,size(cube_fitter.cube.I, 3))
    if aperture isa String
        @assert lowercase(aperture) == "all" "The only accepted string input for 'aperture' is 'all' to signify the entire cube."
        I, σ, templates, area_sr = get_total_integrated_intensities(cube_fitter; shape=shape)
    else
        I, σ, templates, area_sr = get_aperture_integrated_intensities(cube_fitter, shape, aperture)
    end
    cube_data = (λ=cube_fitter.cube.λ, I=I, σ=σ, templates=templates, area_sr=area_sr)

    p_out, p_err = read_fit_results_csv(cube_fitter.name, "spaxel_1_1.csv"; output_path=output_path)
    
    # Split up into continuum & line parameters
    split1 = cube_fitter.n_params_cont
    split2 = cube_fitter.n_params_cont + cube_fitter.n_params_lines
    p_cont = p_out[1:split1]
    p_lines = p_out[split1+1:split2]

    # Get the normalization
    N = Float64(nanmaximum(abs.(cube_data.I[1, 1, :])))

    # The nuclear template must be able to be evaluated at any point
    templates_spax = cube_fitter.templates[1, 1, :, :] ./ N
    channel_masks = cube_fitter.cube.spectral_region.channel_masks

    if !isnothing(cube_fitter_nuc)
        # First, get the nuclear template over the whole FOV
        fname = joinpath(output_path, "$(cube_fitter.name)_full_model.fits")
        hdu_model = FITS(fname)
        agn_templates = read(hdu_model["TEMPLATES.NUCLEAR"]) .* nanmaximum(nansum(.~cube_fitter_nuc.cube.mask, dims=(1,2)))

        I_nuc = evaluate_post_fit_nuclear_template_model(λ, cube_fitter_nuc, agn_templates, output_path_nuc)

        # Read in the parameter maps
        fname = joinpath(output_path, "$(cube_fitter.name)_parameter_maps.fits")
        hdu_param = FITS(fname)

        # first combine into 1d array, then renormalize
        if aperture isa String
            @assert lowercase(aperture) == "all" "The only accepted string input for 'aperture' is 'all' to signify the entire cube."
            I_nuc_spax = I_nuc
        else
            psf_norm = copy(cube_fitter.cube.psf_model)
            psf_norm[.~isfinite.(psf_norm)] .= 0.
            psf_norm_1d = zeros(size(psf_norm, 3))
            for (i, ch_mask) ∈ enumerate(cube_fitter.cube.spectral_region.channel_masks)
                fit_norm = 10 .^ read(hdu_param["TEMPLATES.NUCLEAR.AMP_$i"])[1,1]      # Fit amplitudes
                for k ∈ findall(ch_mask)
                    # s = nansum(psf_norm[:, :, k] .* fit_norm) 
                    # psf_norm[:, :, k] ./= s                                     
                    psf_norm_1d[k] = photometry(aperture[k], psf_norm[:, :, k]).aperture_sum  # Fraction of the total PSF

                    # Need to convert I_nuc from an average over the whole FOV to an average over just the aperture
                    #  --> multiply by area of all spaxels and divide by area of aperture
                    psf_norm_1d[k] *= nansum(.~cube_fitter.cube.mask[:,:,k])
                    psf_norm_1d[k] /= get_area(aperture[k])
                end                                                                  
            end
            psf_norm_spax = Spline1D(cube_fitter.cube.λ, psf_norm_1d; s=0., bc="nearest")
            I_nuc_spax = I_nuc .* psf_norm_spax(λ)
        end

        templates_spax = Matrix{Float64}(undef, length(λ), 1)
        templates_spax[:, 1] .= I_nuc_spax ./ N

        # Create new channel masks carefully -- dont add new channel masks if the extrapolation extends into other 
        # wavebands that could've been covered, just add them into the existing channel masks
        channel_masks = Vector{BitVector}()
        for i ∈ eachindex(cube_fitter.cube.spectral_region.channel_masks)
            bounds = extrema(cube_fitter.cube.λ[cube_fitter.cube.spectral_region.channel_masks[i]])
            dλ = mean(diff(cube_fitter.cube.λ[cube_fitter.cube.spectral_region.channel_masks[i]]))
            if i == 1 && length(cube_fitter.cube.spectral_region.channel_masks) == 1
                mask_i = trues(length(λ))
            elseif i == 1
                mask_i = λ .< (bounds[2]+dλ/2)
            elseif i == length(cube_fitter.cube.spectral_region.channel_masks)
                mask_i = λ .> (bounds[1]-dλ/2)
            else
                mask_i = (bounds[1]-dλ/2) .< λ .< (bounds[2]+dλ/2)
            end
            push!(channel_masks, mask_i)
        end
    end

    _evaluate_model(λ, cube_fitter, ustrip.(p_cont), ustrip.(p_lines), unit.(p_cont), unit.(p_lines), 
        N, templates_spax, channel_masks)
end



# Internal helper function used by all iterations of evaluate_model
function _evaluate_model(λ::Vector{<:QWave}, cube_fitter::CubeFitter, p_cont::Vector{<:Real}, p_lines::Vector{<:Real}, 
    p_units_cont::Vector{<:Unitful.Units}, p_units_lines::Vector{<:Unitful.Units}, N::QSIntensity, 
    templates_spax::Union{Matrix{<:Quantity},Nothing}=nothing, channel_masks::Union{Vector{BitVector},Nothing}=nothing)

    # update systemic velocity offsets for the new input wavelength vector
    if !isnothing(cube_fitter.ssps)
        cube_fitter.ssps.vsyst = log(cube_fitter.ssps.λ[1]/λ[1]) * C_KMS
    end
    if !isnothing(cube_fitter.feii)
        cube_fitter.feii.vsyst = log(cube_fitter.feii.λ[1]/λ[1]) * C_KMS
    end

    # input the templates if given
    aux = Dict()
    if !isnothing(templates_spax) && !isnothing(channel_masks)
        aux["templates"] = templates_spax
        aux["channel_masks"] = channel_masks
    end
    # also update the velocity resolution for the new wavelength vector
    vres = isapprox((λ[2]/λ[1]), (λ[end]/λ[end-1]), rtol=1e-6) ? log(λ[2]/λ[1]) * C_KMS : nothing
    # create the spaxel object with dummy intensity/error data
    spax = Spaxel(CartesianIndex(0,0), λ, ones(length(λ)), ones(length(λ)), ones(length(λ)), falses(length(λ)),
        falses(length(λ)), vres; N=N, aux=aux)

    # evaluate the model
    I_cont, comps_cont, = model_continuum(spax, spax.N, p_cont, p_units_cont, cube_fitter, false, true, true, true)
    I_lines, comps_lines = model_line_residuals(spax, p_lines, p_units_lines, model(cube_fitter).lines, cube_fitter.lsf,
        comps_cont["total_extinction_gas"], trues(length(spax.λ)), true)
    
    # combine and renormalize everything
    I_model, comps, = collect_total_fit_results(spax, cube_fitter, I_cont, I_lines, comps_cont, comps_lines, 0, 0)

    I_model, comps
end


function evaluate_post_fit_nuclear_template_model(λ::Vector{<:QWave}, cube_fitter::CubeFitter,
    agn_templates::Array{<:Real,3}, output_path::String)

    # Repeat the process of reading in the fit results, this time for the nuclear fit
    # This assumes you've used the post_fit_nuclear_template method
    cube_data = create_cube_data_postnuctemp(cube_fitter, agn_templates)
    use_vorbins = !isnothing(cube_fitter.cube.voronoi_bins)
    fname = use_vorbins ? "voronoi_bin_1" : "spaxel_1_1"
    p_out, p_err = read_fit_results_csv(cube_fitter.name, fname; output_path=output_path)

    # Split into continuum and line parameters
    split1 = cube_fitter.n_params_cont
    split2 = cube_fitter.n_params_cont + cube_fitter.n_params_lines
    p_cont = p_out[1:split1]
    p_lines = p_out[split1+1:split2]

    # Get normalization
    N = Float64(nanmaximum(abs.(cube_data.I[1,1,:])))

    # Should be empty for the post-fit nuclear model
    templates_spax = cube_data.templates[1,1,:,:]

    I_model, _ = _evaluate_model(λ, cube_fitter, ustrip.(p_cont), ustrip.(p_lines), unit.(p_cont), unit.(p_lines), 
        N, templates_spax, cube_fitter.cube.spectral_region.channel_masks)

    I_model
end