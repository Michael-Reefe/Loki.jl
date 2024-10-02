### DEPRECATED FUNCTIONS THAT ARE NO LONGER USED ###

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

    cube_data, mx = create_cube_data_nuctemp(cube_fitter, shape)

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
    fit_optical_depth(obs)

Gets an estimated value for tau_9.7, the optical depth at 9.7 um, for each spaxel in an observation.
This requires rebinning the cube to get full spectral coverage across the spatial dimension, then linearly
interpolating the continuum from 6.7-13 um, and taking the ratio with the observed value at 9.7 um.  This
is done four times, each time rebinning the spaxels spatially to match the grid of one of the MIRI channels.
This is necessary to preserve the highest spatail resolution in each of the four channels.

# Arguments
- `obs::Observation`: The Observation object containing the MIRI data in all four channels to be fit.
"""
function fit_optical_depth(obs::Observation)

    @info "Fitting optical depth at 9.7 um with linear interpolation..."
    name = replace(obs.name, " " => "_")

    # Create dictionary to hold outputs
    τ_97 = Dict{Int, Matrix{Float64}}()
    τ_97[0] = zeros(length(keys(obs.channels)),1)
    if !isdir("output_$(name)_optical_depth")
        mkdir("output_$(name)_optical_depth")
    end
    
    # Check for outputs that already have been saved
    c1 = false
    if isfile(joinpath("output_$(name)_optical_depth", "optical_depth_$(name)_sum.csv"))
        @debug "Optical depth sum file found for $(name)"
        τ_97[0] = readdlm(joinpath("output_$(name)_optical_depth", "optical_depth_$(name)_sum.csv"), 
            ',', Float64, '\n')
        c1 = true 
    end

    # Loop through each channel
    for channel ∈ keys(obs.channels)

        # Check if file already exists
        c2 = false
        if isfile(joinpath("output_$(name)_optical_depth", "optical_depth_$(name)_ch$channel.csv"))
            @debug "Optical depth files found for $(name) in channel $channel"
            τ_97[channel] = readdlm(joinpath("output_$(name)_optical_depth", "optical_depth_$(name)_ch$channel.csv"), 
                ',', Float64, '\n')
            c2 = true
        end

        if c1 && c2
            continue
        end

        # Rebin the other channels onto this channel's grid
        cube_combine!(obs, out_grid=channel, out_id=0)
        τ_97[channel] = zeros(size(obs.channels[0].Iν)[1:2])

        # Find specific wavelength points in the vectors
        _, p1 = findmin(x -> abs(x - 6.7), obs.channels[0].λ)
        λ1 = obs.channels[0].λ[p1]
        _, p2 = findmin(x -> abs(x - 13), obs.channels[0].λ)
        λ2 = obs.channels[0].λ[p2]
        _, p3 = findmin(x -> abs(x - 9.7), obs.channels[0].λ)

        # Loop through each spaxel
        @debug "Calculating optical depth for each spaxel in channel $channel"
        for spax ∈ CartesianIndices(size(obs.channels[0].Iν)[1:2])
            # Linear interpolation from 6.7 um to 13 um
            i1 = mean(obs.channels[0].Iν[spax, p1-5:p1+5])
            i2 = mean(obs.channels[0].Iν[spax, p2-5:p2+5])
            slope = (i2 - i1) / (λ2 - λ1)
            # extrapolate to get intrinsic flux value at 9.7 um
            i_97 = i1 + slope * (9.7 - λ1)
            # get the observed flux at 9.7 um
            o_97 = mean(obs.channels[0].Iν[spax, p3-5:p3+5])
            # take the ratio of the observed to intrinsic flux to get the optical depth
            # REFERENCE: Donnan et al. 2023, MNRAS 519, 3691-3705 https://doi.org/10.1093/mnras/stac3729
            ratio = (o_97 / i_97) > 0 ? (o_97 / i_97) : 1.
            ratio == 1. && @debug "Spaxel $spax, ratio <= 0 with obs. $o_97 and intrin. $i_97"
            τ_97[channel][spax] = max(0., -log(ratio) / 0.9)
        end

        # Calculate for the sum of spaxels
        i1 = mean(sumdim(obs.channels[0].Iν, (1,2))[p1-5:p1+5])
        i2 = mean(sumdim(obs.channels[0].Iν, (1,2))[p2-5:p2+5])
        slope = (i2 - i1) / (λ2 - λ1)
        i_97 = i1 + slope * (9.7 - λ1)
        o_97 = mean(sumdim(obs.channels[0].Iν, (1,2))[p3-5:p3+5])
        ratio = (o_97 / i_97) > 0 ? (o_97 / i_97) : 1.
        ratio == 1. && @debug "Sum, ratio <= 0 with obs. $o_97 and intrin. $i_97"
        τ_97[0][channel,1] = max(0., -log(ratio) / 0.9)

        # save outputs to CSV files
        @debug "Writing outputs to file optical_depth_$(name)_ch$channel.csv"
        open(joinpath("output_$(name)_optical_depth", "optical_depth_$(name)_ch$channel.csv"), "w") do file
            writedlm(file, τ_97[channel], ',')
        end

    end

    open(joinpath("output_$(name)_optical_depth", "optical_depth_$(name)_sum.csv"), "w") do file
        writedlm(file, τ_97[0], ',')
    end

    τ_97
end




"""
    calculate_SNR(resolution, continuum, prof, amp, peak, fwhm; <keyword_args>)

Calculate the signal to noise ratio of a spectral feature, i.e. a PAH or emission line. Calculates the ratio
of the peak intensity of the feature over the root-mean-square (RMS) deviation of the surrounding spectrum.
"""
function calculate_SNR(resolution::T, continuum::Vector{T}, prof::Symbol, amp::T, peak::T, 
    fwhm::T; h3::Union{T,Nothing}=nothing, h4::Union{T,Nothing}=nothing, 
    η::Union{T,Nothing}=nothing, acomp_prof::Union{Symbol,Nothing}=nothing, 
    acomp_amp::Union{T,Nothing}=nothing, acomp_peak::Union{T,Nothing}=nothing, 
    acomp_fwhm::Union{T,Nothing}=nothing, acomp_h3::Union{T,Nothing}=nothing, 
    acomp_h4::Union{T,Nothing}=nothing, acomp_η::Union{T,Nothing}=nothing) where {T<:Real}

    # PAH / Drude profiles do not have extra components, so it's a simple A/RMS
    if prof == :Drude
        return amp / std(continuum)
    end

    # Prepare an anonymous function for calculating the line profile
    if prof == :Gaussian
        profile = x -> Gaussian(x, amp, peak, fwhm)
    elseif prof == :Lorentzian
        profile = x -> Lorentzian(x, amp, peak, fwhm)
    elseif prof == :GaussHermite
        profile = x -> GaussHermite(x, amp, peak, fwhm, h3, h4)
    elseif prof == :Voigt
        profile = x -> Voigt(x, amp, peak, fwhm, η)
    else
        error("Unrecognized line profile $(cube_fitter.line_profiles[k])!")
    end

    # Add in additional component profiles if necessary
    if !isnothing(acomp_prof)
        if acomp_prof == :Gaussian
            profile = let profile = profile
                x -> profile(x) + Gaussian(x, acomp_amp, acomp_peak, acomp_fwhm)
            end
        elseif acomp_prof == :Lorentzian
            profile = let profile = profile
                x -> profile(x) + Lorentzian(x, acomp_amp, acomp_peak, acomp_fwhm)
            end
        elseif acomp_prof == :GaussHermite
            profile = let profile = profile
                x -> profile(x) + GaussHermite(x, acomp_amp, acomp_peak, acomp_fwhm, acomp_h3, acomp_h4)
            end
        elseif acomp_prof == :Voigt
            profile = let profile = profile
                x -> profile(x) + Voigt(x, acomp_amp, acomp_peak, acomp_fwhm, acomp_η)
            end
        else
            error("Unrecognized acomp line profile $(cube_fitter.line_acomp_profiles[k])!")
        end
    end

    λ_arr = (peak-10fwhm):resolution:(peak+10fwhm)
    i_max, _ = findmax(profile.(λ_arr))
    # Factor in extinction

    i_max / std(continuum)
end


    # # if requested, recalculate the extra parameters obtained with the calculate_extra_parameters function
    # if recalculate_params
    #     # Get the line mask and cubic spline continuum
    #     mask_lines, I_spline = continuum_cubic_spline(λ, I, σ)

    #     # Separate the fit parameters
    #     popt_c = p_out[1:cube_fitter.n_params_cont]
    #     perr_c = p_err[1:cube_fitter.n_params_cont]
    #     popt_l = p_out[(cube_fitter.n_params_cont+1):(cube_fitter.n_params_cont+cube_fitter.n_params_lines)]
    #     perr_l = p_err[(cube_fitter.n_params_cont+1):(cube_fitter.n_params_cont+cube_fitter.n_params_lines)]

    #     # Get extinction parameters
    #     τ = popt_c[3+2cube_fitter.n_dust_cont]
    #     β = popt_c[3+2cube_fitter.n_dust_cont+3]

    #     # Get the extinction curve
    #     if cube_fitter.extinction_curve == "d+"
    #         ext_curve = τ_dp.(λ, β)
    #     elseif cube_fitter.extinction_curve == "kvt"
    #         ext_curve = τ_kvt.(λ, β)
    #     elseif cube_fitter.extinction_curve == "ct"
    #         ext_curve = τ_ct.(λ)
    #     else
    #         error("Unrecognized extinction curve: $extinction_curve")
    #     end
    #     extinction = Extinction.(ext_curve, τ, screen=cube_fitter.extinction_screen)

    #     p_dust, p_lines, p_dust_err, p_lines_err = 
    #         @timeit timer_output "calculate_extra_parameters" calculate_extra_parameters(λ, I, σ, cube_fitter.n_dust_cont,
    #         cube_fitter.n_dust_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.fit_sil_emission,
    #         cube_fitter.n_lines, cube_fitter.n_acomps, cube_fitter.lines, cube_fitter.n_kin_tied, cube_fitter.tied_kinematics, 
    #         cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing, popt_c, popt_l, perr_c, perr_l, extinction,
    #         mask_lines, I_spline)

    #     # Reconstruct the output and error vectors
    #     p_out = [popt_c; popt_l; p_dust; p_lines; p_out[end]]
    #     p_err = [perr_c; perr_l; p_dust_err; p_lines_err; p_err[end]]

    #     # Rewrite outputs
    #     open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).csv"), "w") do f 
    #         @timeit timer_output "writedlm" writedlm(f, [p_out p_err], ',')
    #     end
    # end


    
    # Use a flat, featureless part of the spectrum to determine the centroid
    # contin = Dict(
    #     1 => (6.75, 6.90),
    #     2 => (9.7, 10.45),
    #     3 => (14.4, 15.5),
    #     4 => (19.5, 22.9),
    #     :A1 => (4.9, 5.01),
    #     :B1 => (5.7, 5.90), # unavoidable PAH features here
    #     :C1 => (6.75, 6.89),
    #     :A2 => (8.07, 8.51),  # unavoidable PAH features here
    #     :B2 => (9.71, 10.1),
    #     :C2 => (10.05, 10.46),
    #     :A3 => (11.7, 12.25),
    #     :B3 => (13.55, 14.2),
    #     :C3 => (15.6, 16.2),
    #     :A4 => (18.0, 18.6),
    #     :B4 => (20.7, 22.9),
    #     :C4 => (24.35, 25.85)
    # )
    # use_filter = true
    # for channel ∈ channels
    #     filter = contin[channel][1] .< obs.channels[channel].λ .< contin[channel][2]
    #     if sum(filter) ≤ 20
    #         @warn "Channel $channel does not cover enough continuum in the predefined region...using the entire subchannel to align"
    #         use_filter = false
    #     end
    # end
