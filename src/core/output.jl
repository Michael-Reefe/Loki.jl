############################## OUTPUT / SAVING FUNCTIONS ####################################


"""
    assign_outputs(out_params, out_errs, cube_fitter, spaxels, z)

Create ParamMaps objects for the parameter values and errors, and a CubeModel object for the full model, and
fill them with the maximum likelihood values and errors given by out_params and out_errs over each spaxel in
spaxels.
"""
function assign_outputs(out_params::SharedArray{<:Real}, out_errs::SharedArray{<:Real}, cube_fitter::CubeFitter,
    spaxels::CartesianIndices, z::Real)

    # Create the CubeModel and ParamMaps structs to be filled in
    cube_model = generate_cubemodel(cube_fitter)
    param_maps, param_errs = generate_parammaps(cube_fitter)

    # Loop over each spaxel and fill in the associated fitting parameters into the ParamMaps and CubeModel
    # I know this is long and ugly and looks stupid but it works for now and I'll make it pretty later
    prog = Progress(length(spaxels); showspeed=true)
    @inbounds @simd for index ∈ spaxels

        # Set the 2D parameter map outputs

        # Conversion factor from MJy sr^-1 to erg s^-1 cm^-2 Hz^-1 sr^-1 = 10^6 * 10^-23 = 10^-17
        # So, log10(A * 1e-17) = log10(A) - 17

        # Stellar continuum amplitude, temp
        # Convert back to observed-frame amplitudes by multiplying by 1+z
        param_maps.stellar_continuum[:amp][index] = out_params[index, 1] > 0. ? log10(out_params[index, 1]*(1+z))-17 : -Inf 
        param_errs.stellar_continuum[:amp][index] = out_params[index, 1] > 0. ? out_errs[index, 1] / (log(10) * out_params[index, 1]) : NaN
        param_maps.stellar_continuum[:temp][index] = out_params[index, 2]
        param_errs.stellar_continuum[:temp][index] = out_errs[index, 2]
        pᵢ = 3

        # Dust continuum amplitude, temp
        for i ∈ 1:cube_fitter.n_dust_cont
            param_maps.dust_continuum[i][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]*(1+z))-17 : -Inf
            param_errs.dust_continuum[i][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.dust_continuum[i][:temp][index] = out_params[index, pᵢ+1]
            param_errs.dust_continuum[i][:temp][index] = out_errs[index, pᵢ+1]
            pᵢ += 2
        end

        # Extinction parameters
        param_maps.extinction[:tau_9_7][index] = out_params[index, pᵢ]
        param_errs.extinction[:tau_9_7][index] = out_errs[index, pᵢ]
        param_maps.extinction[:tau_ice][index] = out_params[index, pᵢ+1]
        param_errs.extinction[:tau_ice][index] = out_errs[index, pᵢ+1]
        param_maps.extinction[:tau_ch][index] = out_params[index, pᵢ+2]
        param_errs.extinction[:tau_ch][index] = out_errs[index, pᵢ+2]
        param_maps.extinction[:beta][index] = out_params[index, pᵢ+3]
        param_errs.extinction[:beta][index] = out_errs[index, pᵢ+3]
        pᵢ += 4

        if cube_fitter.fit_sil_emission
            # Hot dust parameters
            param_maps.hot_dust[:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]*(1+z))-17 : -Inf
            param_errs.hot_dust[:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.hot_dust[:temp][index] = out_params[index, pᵢ+1]
            param_errs.hot_dust[:temp][index] = out_errs[index, pᵢ+1]
            param_maps.hot_dust[:frac][index] = out_params[index, pᵢ+2]
            param_errs.hot_dust[:frac][index] = out_errs[index, pᵢ+2]
            param_maps.hot_dust[:tau_warm][index] = out_params[index, pᵢ+3]
            param_errs.hot_dust[:tau_warm][index] = out_errs[index, pᵢ+3]
            param_maps.hot_dust[:tau_cold][index] = out_params[index, pᵢ+4]
            param_errs.hot_dust[:tau_cold][index] = out_errs[index, pᵢ+4]
            pᵢ += 5
        end

        # Dust feature log(amplitude), mean, FWHM
        for df ∈ cube_fitter.dust_features.names
            param_maps.dust_features[df][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]*(1+z))-17 : -Inf
            param_errs.dust_features[df][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.dust_features[df][:mean][index] = out_params[index, pᵢ+1]/(1+z)
            param_errs.dust_features[df][:mean][index] = out_errs[index, pᵢ+1]/(1+z)
            param_maps.dust_features[df][:fwhm][index] = out_params[index, pᵢ+2]/(1+z)
            param_errs.dust_features[df][:fwhm][index] = out_errs[index, pᵢ+2]/(1+z)
            pᵢ += 3
        end

        if cube_fitter.save_full_model
            # End of continuum parameters: recreate the continuum model
            I_cont, comps_c = fit_full_continuum(cube_fitter.cube.λ, out_params[index, 1:pᵢ-1], cube_fitter.n_dust_cont, cube_fitter.n_dust_feat,
                cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.fit_sil_emission)
        end

        # Tied line kinematics
        vᵢ = pᵢ
        for j ∈ 1:cube_fitter.n_comps
            for vk ∈ cube_fitter.tied_kinematics.key[j]
                vkj = Symbol(vk, "_$j")
                param_maps.tied_voffs[vkj][index] = out_params[index, pᵢ]
                param_errs.tied_voffs[vkj][index] = out_errs[index, pᵢ]
                pᵢ += 1
            end
            for vk ∈ cube_fitter.tied_kinematics.key[j]
                vkj = Symbol(vk, "_$j")
                param_maps.tied_fwhms[vkj][index] = out_params[index, pᵢ]
                param_errs.tied_fwhms[vkj][index] = out_errs[index, pᵢ]
                pᵢ += 1
            end
        end

        # Tied voigt mixing
        if cube_fitter.tie_voigt_mixing
            param_maps.tied_voigt_mix[index] = out_params[index, pᵢ]
            param_errs.tied_voigt_mix[index] = out_errs[index, pᵢ]
            pᵢ += 1
        end

        for k ∈ 1:cube_fitter.n_lines

            ln = cube_fitter.lines.names[k]
            amp = out_params[index, pᵢ]
            amp_err = out_errs[index, pᵢ]
            param_maps.lines[ln][:amp][index] = amp
            param_errs.lines[ln][:amp][index] = amp_err
            fwhm_res = C_KMS / cube_fitter.interp_R(cube_fitter.lines.λ₀[k] * (1+z))

            if isnothing(cube_fitter.lines.tied[k, 1])

                # Individual shift
                voff = out_params[index, pᵢ+1]
                voff_err = out_errs[index, pᵢ+1]
                param_maps.lines[ln][:voff][index] = voff
                param_errs.lines[ln][:voff][index] = voff_err

                # FWHM -> subtract instrumental resolution in quadrature
                fwhm = out_params[index, pᵢ+2]
                fwhm_err = out_errs[index, pᵢ+2]
                if fwhm_res > out_params[index, pᵢ+2]
                    param_maps.lines[ln][:fwhm][index] = 0.
                    param_errs.lines[ln][:fwhm][index] = fwhm_err
                else
                    if fwhm_res ≥ fwhm
                        param_maps.lines[ln][:fwhm][index] = 0.
                        param_errs.lines[ln][:fwhm][index] = fwhm_err
                    else
                        param_maps.lines[ln][:fwhm][index] = √(fwhm^2 - fwhm_res^2)
                        param_errs.lines[ln][:fwhm][index] = fwhm / √(fwhm^2 - fwhm_res^2) * fwhm_err
                    end
                end
                pᵢ += 3

            elseif cube_fitter.flexible_wavesol
                # If velocity is tied while flexible, show only the small shift from the tied velocity offset in the
                # individual parameter maps for each line
                voff = out_params[index, pᵢ+1]
                voff_err = out_errs[index, pᵢ+1]
                param_maps.lines[ln][:voff][index] = voff
                param_errs.lines[ln][:voff][index] = voff_err
                # Convert to the overall voff for later with the acomps
                vwhere = findfirst(x -> x == cube_fitter.lines.tied[k, 1], cube_fitter.tied_kinematics.key[1])
                voff += out_params[index, vᵢ+vwhere-1]
                voff_err = √(voff_err^2 + out_errs[index, vᵢ+vwhere-1]^2)
                # FWHM (tied)
                fwhm = out_params[index, vᵢ+cube_fitter.n_kin_tied[1]+vwhere-1]
                fwhm_err = out_errs[index, vᵢ+cube_fitter.n_kin_tied[1]+vwhere-1]
                pᵢ += 2

            else
                # Tied shift only
                vwhere = findfirst(x -> x == cube_fitter.lines.tied[k, 1], cube_fitter.tied_kinematics.key[1])
                voff = out_params[index, vᵢ+vwhere-1]
                voff_err = out_errs[index, vᵢ+vwhere-1]
                # FWHM (tied)
                fwhm = out_params[index, vᵢ+cube_fitter.n_kin_tied[1]+vwhere-1]
                fwhm_err = out_errs[index, vᵢ+cube_fitter.n_kin_tied[1]+vwhere-1]
                pᵢ += 1

            end
            # Get Gauss-Hermite 3rd and 4th order moments
            if cube_fitter.lines.profiles[k, 1] == :GaussHermite
                param_maps.lines[ln][:h3][index] = out_params[index, pᵢ]
                param_errs.lines[ln][:h3][index] = out_errs[index, pᵢ]
                param_maps.lines[ln][:h4][index] = out_params[index, pᵢ+1]
                param_errs.lines[ln][:h4][index] = out_errs[index, pᵢ+1]
                pᵢ += 2
            elseif cube_fitter.lines.profiles[k, 1] == :Voigt && !cube_fitter.tie_voigt_mixing
                param_maps.lines[k][:mixing][index] = out_params[index, pᵢ]
                param_errs.lines[k][:mixing][index] = out_errs[index, pᵢ]
                pᵢ += 1
            end

            for j ∈ 2:cube_fitter.n_comps
                if !isnothing(cube_fitter.lines.profiles[k, j])

                    param_maps.lines[ln][Symbol(:acomp_amp, "_$j")][index] = amp * out_params[index, pᵢ]
                    param_errs.lines[ln][Symbol(:acomp_amp, "_$j")][index] = 
                        √((amp * out_params[index, pᵢ])^2 * ((amp_err / amp)^2 + (out_errs[index, pᵢ] / out_params[index, pᵢ])^2))

                    if isnothing(cube_fitter.lines.tied[k, j])
                        # note of caution: the output/saved/plotted values of acomp voffs are ALWAYS given relative to
                        # the rest wavelength of the line; same goes for the errors (which add in quadrature)
                        param_maps.lines[ln][Symbol(:acomp_voff, "_$j")][index] = out_params[index, pᵢ+1] + voff
                        param_errs.lines[ln][Symbol(:acomp_voff, "_$j")][index] = √(out_errs[index, pᵢ+1]^2 + voff_err^2)

                        # FWHM -> subtract instrumental resolution in quadrature
                        acomp_fwhm = fwhm * out_params[index, pᵢ+2]
                        acomp_fwhm_err = √(acomp_fwhm^2 * ((fwhm_err / fwhm)^2 + (out_errs[index, pᵢ+2] / out_params[index, pᵢ+2])^2))
                        if fwhm_res > acomp_fwhm
                            param_maps.lines[ln][Symbol(:acomp_fwhm, "_$j")][index] = 0.
                            param_errs.lines[ln][Symbol(:acomp_fwhm, "_$j")][index] = acomp_fwhm_err
                        else
                            param_maps.lines[ln][Symbol(:acomp_fwhm, "_$j")][index] = √(acomp_fwhm^2 - fwhm_res^2)
                            param_errs.lines[ln][Symbol(:acomp_fwhm, "_$j")][index] = acomp_fwhm / √(acomp_fwhm^2 - fwhm_res^2) * acomp_fwhm_err
                        end
                        pᵢ += 3
                    else
                        pᵢ += 1
                    end
                    # Get Gauss-Hermite 3rd and 4th order moments
                    if cube_fitter.lines.profiles[k, j] == :GaussHermite
                        param_maps.lines[ln][Symbol(:acomp_h3, "_$j")][index] = out_params[index, pᵢ]
                        param_errs.lines[ln][Symbol(:acomp_h3, "_$j")][index] = out_errs[index, pᵢ]
                        param_maps.lines[ln][Symbol(:acomp_h4, "_$j")][index] = out_params[index, pᵢ+1]
                        param_errs.lines[ln][Symbol(:acomp_h4, "_$j")][index] = out_errs[index, pᵢ+1]
                        pᵢ += 2
                    elseif cube_fitter.lines.profiles[k, j] == :Voigt && !cube_fitter.tie_voigt_mixing
                        param_maps.lines[ln][Symbol(:acomp_mixing, "_$j")][index] = out_params[index, pᵢ]
                        param_errs.lines[ln][Symbol(:acomp_mixing, "_$j")][index] = out_errs[index, pᵢ]
                        pᵢ += 1
                    end
                end
            end

        end

        N = Float64(abs(nanmaximum(cube_fitter.cube.Iν[index, :])))
        N = N ≠ 0. ? N : 1.
        if cube_fitter.save_full_model
            # End of line parameters: recreate the un-extincted (intrinsic) line model
            I_line, comps_l = fit_line_residuals(cube_fitter.cube.λ, out_params[index, vᵢ:pᵢ-1], cube_fitter.n_lines, cube_fitter.n_comps,
                cube_fitter.lines, cube_fitter.n_kin_tied, cube_fitter.tied_kinematics, cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing, 
                comps_c["extinction"], true)

            # Renormalize
            for comp ∈ keys(comps_l)
                # (dont include extinction correction here since it's already included in the fitted line amplitudes)
                comps_l[comp] .*= N
            end
            I_line .*= N
            
            # Combine the continuum and line models, which both have the extinction profile already applied to them 
            I_model = I_cont .+ I_line
            comps = merge(comps_c, comps_l)
        end

        # Dust feature intensity, EQW, and SNR, from calculate_extra_parameters
        for df ∈ cube_fitter.dust_features.names
            param_maps.dust_features[df][:flux][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]) : -Inf
            param_errs.dust_features[df][:flux][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.dust_features[df][:eqw][index] = out_params[index, pᵢ+1]/(1+z)
            param_errs.dust_features[df][:eqw][index] = out_errs[index, pᵢ+1]/(1+z)
            param_maps.dust_features[df][:SNR][index] = out_params[index, pᵢ+2]
            pᵢ += 3
        end

        for k ∈ 1:cube_fitter.n_lines

            ln = cube_fitter.lines.names[k]
            # Convert amplitudes to the correct units, then take the log
            amp_norm = param_maps.lines[ln][:amp][index]
            amp_norm_err = param_errs.lines[ln][:amp][index]
            param_maps.lines[ln][:amp][index] = amp_norm > 0 ? log10(amp_norm * N * (1+z))-17 : -Inf
            param_errs.lines[ln][:amp][index] = amp_norm > 0 ? amp_norm_err / (log(10) * amp_norm) : NaN

            # Line intensity, EQW, and SNR, from calculate_extra_parameters
            param_maps.lines[ln][:flux][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]) : -Inf
            param_errs.lines[ln][:flux][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.lines[ln][:eqw][index] = out_params[index, pᵢ+1]/(1+z)
            param_errs.lines[ln][:eqw][index] = out_errs[index, pᵢ+1]/(1+z)
            param_maps.lines[ln][:SNR][index] = out_params[index, pᵢ+2]

            for j ∈ 2:cube_fitter.n_comps
                if !isnothing(cube_fitter.lines.profiles[k, j])
                    acomp_amp_norm = param_maps.lines[ln][Symbol(:acomp_amp, "_$j")][index]
                    acomp_amp_norm_err = param_errs.lines[ln][Symbol(:acomp_amp, "_$j")][index]
                    param_maps.lines[ln][Symbol(:acomp_amp, "_$j")][index] = acomp_amp_norm > 0 ? log10(acomp_amp_norm * N * (1+z))-17 : -Inf
                    param_errs.lines[ln][Symbol(:acomp_amp, "_$j")][index] = acomp_amp_norm > 0 ? acomp_amp_norm_err / (log(10) * acomp_amp_norm) : NaN

                    # Line intensity, EQW, and SNR from calculate_extra_parameters
                    param_maps.lines[ln][Symbol(:acomp_flux, "_$j")][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]) : -Inf
                    param_errs.lines[ln][Symbol(:acomp_flux, "_$j")][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / (log(10) * out_params[index, pᵢ]) : NaN
                    param_maps.lines[ln][Symbol(:acomp_eqw, "_$j")][index] = out_params[index, pᵢ+1]/(1+z)
                    param_errs.lines[ln][Symbol(:acomp_eqw, "_$j")][index] = out_params[index, pᵢ+1]/(1+z)
                    param_maps.lines[ln][Symbol(:acomp_SNR, "_$j")][index] = out_params[index, pᵢ+2]

                    pᵢ += 3
                end
            end

            pᵢ += 3
        end

        # Reduced χ^2
        param_maps.reduced_χ2[index] = out_params[index, pᵢ]

        if cube_fitter.save_full_model
            # Set 3D model cube outputs, shifted back to the observed frame
            cube_model.model[index, :] .= I_model .* (1 .+ z)
            cube_model.stellar[index, :] .= comps["stellar"] .* (1 .+ z)
            for i ∈ 1:cube_fitter.n_dust_cont
                cube_model.dust_continuum[index, :, i] .= comps["dust_cont_$i"] .* (1 .+ z)
            end
            for j ∈ 1:cube_fitter.n_dust_feat
                cube_model.dust_features[index, :, j] .= comps["dust_feat_$j"] .* (1 .+ z)
            end
            if cube_fitter.fit_sil_emission
                cube_model.hot_dust[index, :] .= comps["hot_dust"] .* (1 .+ z)
            end
            for j ∈ 1:cube_fitter.n_comps
                for k ∈ 1:cube_fitter.n_lines
                    cube_model.lines[index, :, k] .+= comps["line_$(k)_$(j)"] .* (1 .+ z)
                end
            end
            cube_model.extinction[index, :] .= comps["extinction"]
            cube_model.abs_ice[index, :] .= comps["abs_ice"]
            cube_model.abs_ch[index, :] .= comps["abs_ch"]
        end

        next!(prog)

    end

    # Subtract the average of the individual voffs from the tied voffs, based on the SNR, for each group
    if cube_fitter.flexible_wavesol
        @debug "Adjusting individual voffs due to the flexible_wavesol option"
        for vk ∈ cube_fitter.tied_kinematics.key[1]
            indiv_voffs = nothing
            snrs = nothing
            # Loop through and create 3D arrays of the voffs and SNRs of each line in the tied kinematic group
            for k ∈ 1:cube_fitter.n_lines
                name = cube_fitter.lines.names[k]
                if cube_fitter.lines.tied[k, 1] == vk
                    if isnothing(indiv_voffs)
                        indiv_voffs = param_maps.lines[name][:voff]
                        snrs = param_maps.lines[name][:SNR]
                        continue
                    end
                    indiv_voffs = cat(indiv_voffs, param_maps.lines[name][:voff], dims=3)
                    snrs = cat(snrs, param_maps.lines[name][:SNR], dims=3)
                end
            end
            # Collapse the voff array into an average along the 3rd dimension, ignoring any with an SNR < 3
            if !isnothing(indiv_voffs) && !isnothing(snrs)
                indiv_voffs[snrs .< 3] .= NaN
                avg_offset = dropdims(nanmean(indiv_voffs, dims=3), dims=3)
                # Subtract the average offset from the individual voffs
                # (the goal is to have the average offset of the individual voffs be 0, relative to the tied voff)
                for k ∈ 1:cube_fitter.n_lines
                    name = cube_fitter.lines.names[k]
                    if cube_fitter.lines.tied[k, 1] == vk
                        param_maps.lines[name][:voff] .-= avg_offset
                    end
                end
                # and add it to the tied voff
                param_maps.tied_voffs[Symbol(vk, "_1")] .+= avg_offset
            end
        end
    end

    param_maps, param_errs, cube_model

end


"""
    plot_parameter_map(data, name, name_i, Ω, z, cosmo; snr_filter=snr_filter, snr_thresh=snr_thresh,
        cmap=cmap)

Plotting function for 2D parameter maps which are output by `fit_cube!`

# Arguments
- `data::Matrix{Float64}`: The 2D array of data to be plotted
- `name::String`: The name of the object whose fitting parameter is being plotted, i.e. "NGC_7469"
- `name_i::String`: The name of the individual parameter being plotted, i.e. "dust_features_PAH_5.24_amp"
- `Ω::Float64`: The solid angle subtended by each pixel, in steradians (used for angular scalebar)
- `z::Float64`: The redshift of the object (used for physical scalebar)
- `psf_fwhm::Float64`: The FWHM of the point-spread function in arcseconds (used to add a circular patch with this size)
- `cosmo::Cosmology.AbstractCosmology`: The cosmology to use to calculate distance for the physical scalebar
- `python_wcs::PyObject`: The astropy WCS object used to project the maps onto RA/Dec space
- `snr_filter::Matrix{Float64}=Matrix{Float64}(undef,0,0)`: A 2D array of S/N values to
    be used to filter out certain spaxels from being plotted - must be the same size as `data` to filter
- `snr_thresh::Float64=3.`: The S/N threshold below which to cut out any spaxels using the values in snr_filter
- `cmap::Symbol=:cubehelix`: The colormap used in the plot, defaults to the cubehelix map
"""
function plot_parameter_map(data::Matrix{Float64}, name::String, name_i::String, Ω::Float64, z::Float64, psf_fwhm::Float64,
    cosmo::Cosmology.AbstractCosmology, python_wcs::PyObject; snr_filter::Union{Nothing,Matrix{Float64}}=nothing, 
    snr_thresh::Float64=3., cmap::Symbol=:cubehelix)

    # I know this is ugly but I couldn't figure out a better way to do it lmao
    if occursin("amp", String(name_i))
        bunit = L"$\log_{10}(I / $ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1})$"
    elseif occursin("temp", String(name_i))
        bunit = L"$T$ (K)"
    elseif occursin("fwhm", String(name_i)) && occursin("PAH", String(name_i))
        bunit = L"FWHM ($\mu$m)"
    elseif occursin("fwhm", String(name_i)) && !occursin("PAH", String(name_i))
        bunit = L"FWHM (km s$^{-1}$)"
    elseif occursin("mean", String(name_i))
        bunit = L"$\mu$ ($\mu$m)"
    elseif occursin("voff", String(name_i))
        bunit = L"$v_{\rm off}$ (km s$^{-1}$)"
    elseif occursin("SNR", String(name_i))
        bunit = L"$S/N$"
    elseif occursin("tau", String(name_i))
        if occursin("warm", String(name_i))
            bunit = L"$\tau_{\rm warm}$"
        elseif occursin("cold", String(name_i))
            bunit = L"$\tau_{\rm cold}$"
        elseif occursin("ice", String(name_i))
            bunit = L"$\tau_{\rm ice}$"
        elseif occursin("ch", String(name_i))
            bunit = L"$\tau_{\rm CH}$"
        else
            bunit = L"$\tau_{9.7}$"
        end
    elseif occursin("flux", String(name_i))
        bunit = L"$\log_{10}(F /$ erg s$^{-1}$ cm$^{-2}$)"
    elseif occursin("eqw", String(name_i))
        bunit = L"$W_{\rm eq}$ ($\mu$m)"
    elseif occursin("chi2", String(name_i))
        bunit = L"$\tilde{\chi}^2$"
    elseif occursin("h3", String(name_i))
        bunit = L"$h_3$"
    elseif occursin("h4", String(name_i))
        bunit = L"$h_4$"
    elseif occursin("mixing", String(name_i))
        bunit = L"$\eta$"
    elseif occursin("beta", String(name_i))
        bunit = L"$\beta$"
    elseif occursin("frac", String(name_i))
        bunit = L"$C_f$"
    end

    @debug "Plotting 2D map of $name_i with units $bunit"

    filtered = copy(data)
    # Convert Infs into NaNs
    filtered[.!isfinite.(filtered)] .= NaN
    # Filter out low SNR points
    if !isnothing(snr_filter)
        filtered[snr_filter .≤ snr_thresh] .= NaN
        @debug "Performing SNR filtering, $(sum(isfinite.(filtered)))/$(length(filtered)) passed"
    end
    # filter out insane/unphysical equivalent widths (due to ~0 continuum level)
    if occursin("eqw", String(name_i))
        filtered[filtered .> 100] .= NaN
    end
    if occursin("voff", String(name_i))
        # Perform a 5-sigma clip to remove outliers
        f_avg = nanmean(filtered)
        f_std = nanstd(filtered)
        filtered[abs.(filtered .- f_avg) .> 5f_std] .= NaN
    end

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=python_wcs[0])  # slice the WCS to remove the wavelength axis
    # Need to filter out any NaNs in order to use quantile
    vmin = nanminimum(filtered)
    vmax = nanmaximum(filtered)
    # override vmin/vmax for mixing parameter
    if occursin("mixing", String(name_i))
        vmin = 0.
        vmax = 1.
    end
    # if taking a voff, make sure vmin/vmax are symmetric and change the colormap to coolwarm
    if occursin("voff", String(name_i))
        vabs = max(abs(vmin), abs(vmax))
        vmin = -vabs
        vmax = vabs
        if cmap == :cubehelix
            cmap = :coolwarm
        end
    end
    # default cmap is magma for FWHMs and equivalent widths
    if (occursin("fwhm", String(name_i)) || occursin("eqw", String(name_i))) && cmap == :cubehelix
        cmap = :magma
    end
    cdata = ax.imshow(filtered', origin=:lower, cmap=cmap, vmin=vmin, vmax=vmax)
    # ax.axis(:off)
    ax.tick_params(which="both", axis="both", direction="in")
    ax.set_xlabel("R.A.")
    ax.set_ylabel("Dec.")

    # Angular and physical scalebars
    pix_as = sqrt(Ω) * 180/π * 3600
    n_pix = 1/pix_as
    @debug "Using angular diameter distance $(angular_diameter_dist(cosmo, z))"
    # Calculate in Mpc
    dA = angular_diameter_dist(u"pc", cosmo, z)
    # Remove units
    dA = uconvert(NoUnits, dA/u"pc")
    l = dA * π/180 / 3600  # l = d * theta (1")
    # Round to a nice even number
    l = Int(round(l, sigdigits=1))
     # new angular size for this scale
    θ = l / dA
    n_pix = 1/sqrt(Ω) * θ   # number of pixels = (pixels per radian) * radians
    if cosmo.h ≈ 1.0
        scalebar = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, L"%$l$h^{-1}$ pc", "lower left", pad=1, color=:black, 
            frameon=false, size_vertical=0.4, label_top=false)
    else
        scalebar = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, L"%$l pc", "lower left", pad=1, color=:black,
            frameon=false, size_vertical=0.4, label_top=false)
    end
    ax.add_artist(scalebar)

    # Add circle for the PSF FWHM
    psf = plt.Circle(size(data) .* 0.9, psf_fwhm / pix_as / 2, color="k")
    ax.add_patch(psf)
    ax.annotate("PSF", size(data) .* 0.9 .- (0., psf_fwhm / pix_as / 2 + 1.75), ha=:center, va=:center)

    fig.colorbar(cdata, ax=ax, label=bunit)
    plt.savefig(joinpath("output_$(name)", "param_maps", "$(name_i).pdf"), dpi=300, bbox_inches=:tight)
    plt.close()

end


"""
    plot_parameter_maps(param_maps; snr_thresh=snr_thresh)

Wrapper function for `plot_parameter_map`, iterating through all the parameters in a `CubeFitter`'s `ParamMaps` object
and creating 2D maps of them.

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the fitting options
- `param_maps::ParamMaps`: The ParamMaps object containing the parameter values
- `snr_thresh::Real`: The S/N threshold to be used when filtering the parameter maps by S/N, for those applicable
"""
function plot_parameter_maps(cube_fitter::CubeFitter, param_maps::ParamMaps; snr_thresh::Real=3.)

    # Iterate over model parameters and make 2D maps
    @debug "Using solid angle $(cube_fitter.cube.Ω), redshift $(cube_fitter.z), cosmology $(cube_fitter.cosmology)"

    # Stellar continuum parameters
    for parameter ∈ keys(param_maps.stellar_continuum)
        data = param_maps.stellar_continuum[parameter]
        name_i = join(["stellar_continuum", parameter], "_")
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf_fwhm), 
            cube_fitter.cosmology, cube_fitter.cube.wcs)
    end

    # Dust continuum parameters
    for i ∈ keys(param_maps.dust_continuum)
        for parameter ∈ keys(param_maps.dust_continuum[i])
            data = param_maps.dust_continuum[i][parameter]
            name_i = join(["dust_continuum", i, parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf_fwhm), 
                cube_fitter.cosmology, cube_fitter.cube.wcs)
        end
    end

    # Dust feature (PAH) parameters
    for df ∈ keys(param_maps.dust_features)
        snr = param_maps.dust_features[df][:SNR]
        # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
        wave_i = nanmedian(param_maps.dust_features[df][:mean])
        _, ind_i = findmin(abs.(cube_fitter.cube.λ .- wave_i))

        for parameter ∈ keys(param_maps.dust_features[df])
            data = param_maps.dust_features[df][parameter]
            name_i = join(["dust_features", df, parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cube.psf_fwhm[ind_i],
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=parameter !== :SNR ? snr : nothing, snr_thresh=snr_thresh)
        end
    end

    # Extinction parameters
    for parameter ∈ keys(param_maps.extinction)
        data = param_maps.extinction[parameter]
        name_i = join(["extinction", parameter], "_")
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf_fwhm), 
            cube_fitter.cosmology, cube_fitter.cube.wcs)
    end

    if cube_fitter.fit_sil_emission
        # Hot dust parameters
        for parameter ∈ keys(param_maps.hot_dust)
            data = param_maps.hot_dust[parameter]
            name_i = join(["hot_dust", parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf_fwhm), 
                cube_fitter.cosmology, cube_fitter.cube.wcs)
        end
    end

    # Tied Voigt mixing parameter
    if cube_fitter.tie_voigt_mixing
        data = param_maps.tied_voigt_mix
        name_i = "tied_voigt_mixing"
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf_fwhm), 
            cube_fitter.cosmology, cube_fitter.cube.wcs)
    end

    # Tied kinematics
    for j ∈ 1:cube_fitter.n_comps
        for vk ∈ cube_fitter.tied_kinematics.key[j]
            data = param_maps.tied_voffs[Symbol(vk, "_$j")]
            name_i = join(["tied_voffs", vk, "$j"], "_")
            # Get the SNRs of each line in the tied kinematic group
            snr = nothing
            for k ∈ 1:cube_fitter.n_lines
                name = cube_fitter.lines.names[k]
                if cube_fitter.lines.tied[k, j] == vk
                    if isnothing(snr)
                        snr = param_maps.lines[name][:SNR]
                        continue
                    end
                    snr = cat(snr, param_maps.lines[name][:SNR], dims=3)
                end
            end
            # Take the maximum SNR along the 3rd axis
            if ndims(snr) == 3
                snr = dropdims(nanmaximum(snr, dims=3), dims=3)
            end
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf_fwhm), 
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr, snr_thresh=snr_thresh)
        end
        for vk ∈ cube_fitter.tied_kinematics.key[j]
            data = param_maps.tied_fwhms[Symbol(vk, "_$j")]
            name_i = join(["tied_fwhms", vk, "$j"], "_")
            # Get the SNRs of each line in the tied kinematic group
            snr = nothing
            for k ∈ 1:cube_fitter.n_lines
                name = cube_fitter.lines.names[k]
                if cube_fitter.lines.tied[k, j] == vk
                    if isnothing(snr)
                        snr = param_maps.lines[name][:SNR]
                        continue
                    end
                    snr = cat(snr, param_maps.lines[name][:SNR], dims=3)
                end
            end
            # Take the maximum SNR along the 3rd axis
            if ndims(snr) == 3
                snr = dropdims(nanmaximum(snr, dims=3), dims=3)
            end
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf_fwhm), 
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr, snr_thresh=snr_thresh)
        end
    end

    # Line parameters
    for line ∈ keys(param_maps.lines)
        # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
        line_i = findfirst(cube_fitter.lines.names .== line)
        wave_i = cube_fitter.lines.λ₀[line_i]
        _, ind_i = findmin(abs.(cube_fitter.cube.λ .- wave_i))

        for parameter ∈ keys(param_maps.lines[line])
            data = param_maps.lines[line][parameter]
            name_i = join(["lines", line, parameter], "_")
            snr_filter = param_maps.lines[line][:SNR]
            if contains(String(parameter), "SNR")
                snr_filter = nothing
            elseif contains(String(parameter), "acomp")
                ind = split(String(parameter), "_")[end]
                snr_filter = param_maps.lines[line][Symbol(:acomp_SNR, "_$ind")]
            end
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cube.psf_fwhm[ind_i], 
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh)
        end
    end

    # Reduced chi^2 
    data = param_maps.reduced_χ2
    name_i = "reduced_chi2"
    plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf_fwhm), 
        cube_fitter.cosmology, cube_fitter.cube.wcs)

    return

end


"""
    make_movie(cube_fitter, cube_model; cmap=cmap)

A function for making mp4 movie files of the cube data and model! This is mostly just for fun
and not really useful scientifically...but they're pretty to watch!

N.B. You must have FFMPeg (https://ffmpeg.org/) and the FFMPeg-python and Astropy Python Packages 
installed on your system in order to use this function, as it uses Matplotlib's animation library
with the FFMPeg backend to generate the mp4 files.  Astropy is required to use the WCS module
to annotate the plots with right ascension and declination.

N.B. This function takes a while to run so be prepared to wait a good few minutes...

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the fitting options
- `cube_model::CubeModel`: The CubeModel object containing the full 3D models of the IFU cube
- `cmap::Symbol=:cubehelix`: The colormap to be used to plot the intensity of the data over time,
    defaults to cubehelix.
"""
function make_movie(cube_fitter::CubeFitter, cube_model::CubeModel; cmap::Symbol=:cubehelix)

    for (full_data, title) ∈ zip([cube_fitter.cube.Iν, cube_model.model], ["DATA", "MODEL"])

        # Writer using FFMpeg to create an mp4 file
        metadata = Dict(:title => title, :artist => "LOKI", :fps => 60)
        writer = py_animation.FFMpegWriter(fps=60, metadata=metadata)

        # Set up plots with gridspec
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=20,  nrows=10)
        ax1 = fig.add_subplot(py"$(gs)[0:8, 0:18]", projection=cube_fitter.cube.wcs[0])
        ax2 = fig.add_subplot(py"$(gs)[9:10, :]")
        ax3 = fig.add_subplot(py"$(gs)[0:8, 18:19]")

        # First wavelength slice of the model
        wave_rest = cube_fitter.cube.λ
        data = full_data[:, :, 1]

        # Get average along the wavelength dimension
        datasum = sumdim(full_data, 3)
        dataavg = datasum ./ size(full_data, 3)
        flatavg = dataavg[isfinite.(dataavg)]

        # Plot the first slice
        image = ax1.imshow(data', origin=:lower, cmap=cmap, vmin=quantile(flatavg, 0.01), vmax=quantile(flatavg, 0.99))
        ax1.set_xlabel(" ")
        ax1.set_ylabel(" ")
        # Add colorbar on the side
        plt.colorbar(image, cax=ax3, label=L"$I_{\nu}$ (MJy sr$^{-1}$)")
        # Prepare the bottom 1D plot that slides along the wavelength 
        ax2.hlines(10, wave_rest[1], wave_rest[end], color="k")
        ln, = ax2.plot(wave_rest[1], 24, "|", ms=20, color="y")
        ax2.axis(:off)
        ax2.set_ylim(-10, 24)
        ax2.text(wave_rest[length(wave_rest) ÷ 2], -8, L"$\lambda_{\rm rest}$ ($\AA$)", ha="center", va="center")
        # Annotate with wavelength value at the current time
        time_text = ax2.text(wave_rest[length(wave_rest) ÷ 2], 20, (@sprintf "%.3f" wave_rest[1]), ha="center", va="center")
        ax2.text(wave_rest[1], -8, (@sprintf "%.3f" wave_rest[1]), ha="center", va="center")
        ax2.text(wave_rest[end], -8, (@sprintf "%.3f" wave_rest[end]), ha="center", va="center")
        # plt.tight_layout()  ---> doesn't work for some reason

        # Loop over the wavelength axis and set the image data to the new slice for each frame
        output_file = joinpath("output_$(cube_fitter.name)", "$title.mp4")
        writer.setup(fig, output_file, dpi=300)
        for i ∈ 1:size(full_data, 3)
            data_i = full_data[:, :, i] 
            image.set_array(data_i')
            ln.set_data(wave_rest[i], 24)
            time_text.set_text(@sprintf "%.3f" wave_rest[i])
            writer.grab_frame()
        end
        writer.finish()
        plt.close()

    end

end


"""
    write_fits(cube_fitter, cube_model, param_maps, param_errs)

Save the best fit results for the cube into two FITS files: one for the full 3D intensity model of the cube, split up by
individual model components, and one for 2D parameter maps of the best-fit parameters for each spaxel in the cube.
"""
function write_fits(cube_fitter::CubeFitter, cube_model::CubeModel, param_maps::ParamMaps, param_errs::ParamMaps)

    # Header information
    hdr = FITSHeader(
        ["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "PIXAR_SR", "RA", "DEC", "WCSAXES",
            "CDELT1", "CDELT2", "CDELT3", "CTYPE1", "CTYPE2", "CTYPE3", "CRPIX1", "CRPIX2", "CRPIX3",
            "CRVAL1", "CRVAL2", "CRVAL3", "CUNIT1", "CUNIT2", "CUNIT3", "PC1_1", "PC1_2", "PC1_3", 
            "PC2_1", "PC2_2", "PC2_3", "PC3_1", "PC3_2", "PC3_3"],

        # Check if the redshift correction is right for the third WCS axis?
        [cube_fitter.name, cube_fitter.z, cube_fitter.cube.channel, cube_fitter.cube.band, cube_fitter.cube.Ω, cube_fitter.cube.α, cube_fitter.cube.δ, 
         cube_fitter.cube.wcs.wcs.naxis, cube_fitter.cube.wcs.wcs.cdelt[1], cube_fitter.cube.wcs.wcs.cdelt[2], cube_fitter.cube.wcs.wcs.cdelt[3], 
         cube_fitter.cube.wcs.wcs.ctype[1], cube_fitter.cube.wcs.wcs.ctype[2], cube_fitter.cube.wcs.wcs.ctype[3], cube_fitter.cube.wcs.wcs.crpix[1], 
         cube_fitter.cube.wcs.wcs.crpix[2], cube_fitter.cube.wcs.wcs.crpix[3], cube_fitter.cube.wcs.wcs.crval[1], cube_fitter.cube.wcs.wcs.crval[2], 
         cube_fitter.cube.wcs.wcs.crval[3], cube_fitter.cube.wcs.wcs.cunit[1].name, cube_fitter.cube.wcs.wcs.cunit[2].name, cube_fitter.cube.wcs.wcs.cunit[3].name, 
         cube_fitter.cube.wcs.wcs.pc[1,1], cube_fitter.cube.wcs.wcs.pc[1,2], cube_fitter.cube.wcs.wcs.pc[1,3], cube_fitter.cube.wcs.wcs.pc[2,1], cube_fitter.cube.wcs.wcs.pc[2,2], 
         cube_fitter.cube.wcs.wcs.pc[2,3], cube_fitter.cube.wcs.wcs.pc[3,1], cube_fitter.cube.wcs.wcs.pc[3,2], cube_fitter.cube.wcs.wcs.pc[3,3]],

        ["Target name", "Target redshift", "MIRI channel", "MIRI band",
        "Solid angle per pixel (rad.)", "Right ascension of target (deg.)", "Declination of target (deg.)",
        "number of World Coordinate System axes", 
        "first axis increment per pixel", "second axis increment per pixel", "third axis increment per pixel", 
        "first axis coordinate type", "second axis coordinate type", "third axis coordinate type", 
        "axis 1 coordinate of the reference pixel", "axis 2 coordinate of the reference pixel", "axis 3 coordinate of the reference pixel",
        "first axis value at the reference pixel", "second axis value at the reference pixel", "third axis value at the reference pixel",
        "first axis units", "second axis units", "third axis units",
        "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element",
        "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element",
        "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element"]
    )

    if cube_fitter.save_full_model
        # Create the 3D intensity model FITS file
        FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_3D_model.fits"), "w") do f

            @debug "Writing 3D model FITS HDUs"

            write(f, Vector{Int}())                                                                     # Primary HDU (empty)
            write(f, cube_fitter.cube.Iν; header=hdr, name="DATA")                                      # Raw data with nans inserted
            write(f, cube_model.model; header=hdr, name="MODEL")                                        # Full intensity model
            write(f, cube_fitter.cube.Iν .- cube_model.model; header=hdr, name="RESIDUALS")             # Residuals (data - model)
            write(f, cube_model.stellar; header=hdr, name="STELLAR_CONTINUUM")                          # Stellar continuum model
            for i ∈ 1:size(cube_model.dust_continuum, 4)
                write(f, cube_model.dust_continuum[:, :, :, i]; header=hdr, name="DUST_CONTINUUM_$i")   # Dust continuum models
            end
            for (j, df) ∈ enumerate(cube_fitter.df_names)
                write(f, cube_model.dust_features[:, :, :, j]; header=hdr, name="$df")                  # Dust feature profiles
            end
            for (k, line) ∈ enumerate(cube_fitter.line_names)
                write(f, cube_model.lines[:, :, :, k]; header=hdr, name="$line")                        # Emission line profiles
            end
            write(f, cube_model.extinction; header=hdr, name="EXTINCTION")                              # Extinction model
            write(f, cube_model.abs_ice; header=hdr, name="ABS_ICE")                                    # Ice Absorption model
            write(f, cube_model.abs_ch; header=hdr, name="ABS_CH")                                      # CH Absorption model
            if cube_fitter.fit_sil_emission
                write(f, cube_model.hot_dust; header=hdr, name="HOT_DUST")                              # Hot dust model
            end
            
            write(f, ["wave_rest", "wave_obs"],                                                                     # 1D Rest frame and observed frame
                    [cube_fitter.cube.λ, observed_frame(cube_fitter.cube.λ, cube_fitter.z)],                  # wavelength vectors
                hdutype=TableHDU, name="WAVELENGTH", units=Dict(:wave_rest => "um", :wave_obs => "um"))

            # Insert physical units into the headers of each HDU -> MegaJansky per steradian for all except
            # the extinction profile, which is a multiplicative constant
            write_key(f["DATA"], "BUNIT", "MJy/sr")
            write_key(f["MODEL"], "BUNIT", "MJy/sr")
            write_key(f["RESIDUALS"], "BUNIT", "MJy/sr")
            write_key(f["STELLAR_CONTINUUM"], "BUNIT", "MJy/sr")
            for i ∈ 1:size(cube_model.dust_continuum, 4)
                write_key(f["DUST_CONTINUUM_$i"], "BUNIT", "MJy/sr")
            end
            for df ∈ cube_fitter.df_names
                write_key(f["$df"], "BUNIT", "MJy/sr")
            end
            for line ∈ cube_fitter.line_names
                write_key(f["$line"], "BUNIT", "MJy/sr")
            end
            write_key(f["EXTINCTION"], "BUNIT", "unitless")
            write_key(f["ABS_ICE"], "BUNIT", "unitless")
            write_key(f["ABS_CH"], "BUNIT", "unitless")
            if cube_fitter.fit_sil_emission
                write_key(f["HOT_DUST"], "BUNIT", "MJy/sr")
            end
        end
    end

    # Create the 2D parameter map FITS file
    FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_parameter_maps.fits"), "w") do f

        @debug "Writing 2D parameter map FITS HDUs"

        write(f, Vector{Int}())  # Primary HDU (empty)

        # Iterate over model parameters and make 2D maps

        # Stellar continuum parameters
        for parameter ∈ keys(param_maps.stellar_continuum)
            data = param_maps.stellar_continuum[parameter]
            name_i = join(["stellar_continuum", parameter], "_")
            if occursin("amp", String(name_i))
                bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
            elseif occursin("temp", String(name_i))
                bunit = "Kelvin"
            end
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end
        for parameter ∈ keys(param_errs.stellar_continuum)
            data = param_errs.stellar_continuum[parameter]
            name_i = join(["stellar_continuum", parameter, "err"], "_")
            if occursin("amp", String(name_i))
                bunit = "dex"
            elseif occursin("temp", String(name_i))
                bunit = "Kelvin"
            end
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end

        # Dust continuum parameters
        for i ∈ keys(param_maps.dust_continuum)
            for parameter ∈ keys(param_maps.dust_continuum[i])
                data = param_maps.dust_continuum[i][parameter]
                name_i = join(["dust_continuum", i, parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                elseif occursin("temp", String(name_i))
                    bunit = "Kelvin"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)  
            end
        end
        for i ∈ keys(param_errs.dust_continuum)
            for parameter ∈ keys(param_errs.dust_continuum[i])
                data = param_errs.dust_continuum[i][parameter]
                name_i = join(["dust_continuum", i, parameter, "err"], "_")
                if occursin("amp", String(name_i))
                    bunit = "dex"
                elseif occursin("temp", String(name_i))
                    bunit = "Kelvin"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)  
            end
        end

        if cube_fitter.fit_sil_emission
            # Hot dust parameters
            for parameter ∈ keys(param_maps.hot_dust)
                data = param_maps.hot_dust[parameter]
                name_i = join(["hot_dust", parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                elseif occursin("temp", String(name_i))
                    bunit = "Kelvin"
                elseif occursin("frac", String(name_i)) || occursin("tau", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)
            end
            for parameter ∈ keys(param_errs.hot_dust)
                data = param_errs.hot_dust[parameter]
                name_i = join(["hot_dust", parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                elseif occursin("temp", String(name_i))
                    bunit = "Kelvin"
                elseif occursin("frac", String(name_i)) || occursin("tau", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)
            end
        end

        # Dust feature (PAH) parameters
        for df ∈ keys(param_maps.dust_features)
            for parameter ∈ keys(param_maps.dust_features[df])
                data = param_maps.dust_features[df][parameter]
                name_i = join(["dust_features", df, parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                elseif occursin("fwhm", String(name_i)) || occursin("mean", String(name_i)) || occursin("eqw", String(name_i))
                    bunit = "um"
                elseif occursin("flux", String(name_i))
                    bunit = "log10(F / erg s^-1 cm^-2)"
                elseif occursin("SNR", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)      
            end
        end
        for df ∈ keys(param_errs.dust_features)
            for parameter ∈ keys(param_errs.dust_features[df])
                data = param_errs.dust_features[df][parameter]
                name_i = join(["dust_features", df, parameter, "err"], "_")
                if occursin("amp", String(name_i))
                    bunit = "dex"
                elseif occursin("fwhm", String(name_i)) || occursin("mean", String(name_i)) || occursin("eqw", String(name_i))
                    bunit = "um"
                elseif occursin("flux", String(name_i))
                    bunit = "dex"
                elseif occursin("SNR", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)      
            end
        end

        # Line parameters
        for line ∈ keys(param_maps.lines)
            for parameter ∈ keys(param_maps.lines[line])
                data = param_maps.lines[line][parameter]
                name_i = join(["lines", line, parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                elseif occursin("fwhm", String(name_i)) || occursin("voff", String(name_i))
                    bunit = "km/s"
                elseif occursin("flux", String(name_i))
                    bunit = "log10(I / erg s^-1 cm^-2)"
                elseif occursin("eqw", String(name_i))
                    bunit = "um"
                elseif occursin("SNR", String(name_i)) || occursin("h3", String(name_i)) || 
                    occursin("h4", String(name_i)) || occursin("mixing", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)   
            end
        end
        for line ∈ keys(param_errs.lines)
            for parameter ∈ keys(param_errs.lines[line])
                data = param_errs.lines[line][parameter]
                name_i = join(["lines", line, parameter, "err"], "_")
                if occursin("amp", String(name_i))
                    bunit = "dex"
                elseif occursin("fwhm", String(name_i)) || occursin("voff", String(name_i))
                    bunit = "km/s"
                elseif occursin("flux", String(name_i))
                    bunit = "dex"
                elseif occursin("eqw", String(name_i))
                    bunit = "um"
                elseif occursin("SNR", String(name_i)) || occursin("h3", String(name_i)) || 
                    occursin("h4", String(name_i)) || occursin("mixing", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)   
            end
        end

        # Extinction parameters
        for parameter ∈ keys(param_maps.extinction)
            data = param_maps.extinction[parameter]
            name_i = join(["extinction", parameter], "_")
            bunit = "unitless"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)  
        end
        for parameter ∈ keys(param_errs.extinction)
            data = param_errs.extinction[parameter]
            name_i = join(["extinction", parameter, "err"], "_")
            bunit = "unitless"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)  
        end

        # Tied Voigt mixing parameter
        if cube_fitter.tie_voigt_mixing
            data = param_maps.tied_voigt_mix
            name_i = "tied_voigt_mixing"
            bunit = "unitless"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end
        if cube_fitter.tie_voigt_mixing
            data = param_errs.tied_voigt_mix
            name_i = "tied_voigt_mixing_err"
            bunit = "unitless"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end

        # Tied kinematics
        for j ∈ 1:cube_fitter.n_comps
            for vk ∈ cube_fitter.tied_kinematics.key[j]
                data = param_maps.tied_voffs[Symbol(vk, "_$j")]
                name_i = join(["tied_voffs", vk, "$j"], "_")
                bunit = "km/s"
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)
            end
            for vk ∈ cube_fitter.tied_kinematics.key[j]
                data = param_errs.tied_voffs[Symbol(vk, "_$j")]
                name_i = join(["tied_voffs", vk, "$j", "err"], "_")
                bunit = "km/s"
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)
            end
        end

        # Reduced chi^2
        data = param_maps.reduced_χ2
        name_i = "reduced_chi2"
        bunit = "unitless"
        write(f, data; header=hdr, name=name_i)
        write_key(f[name_i], "BUNIT", bunit)

    end
end

