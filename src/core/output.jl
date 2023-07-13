############################## OUTPUT / SAVING FUNCTIONS ####################################

const ascii_lowercase = "abcdefghijklmnopqrstuvwxyz"


"""
    assign_outputs(out_params, out_errs, cube_fitter, cube_data, spaxels, z)

Create ParamMaps objects for the parameter values and errors, and a CubeModel object for the full model, and
fill them with the maximum likelihood values and errors given by out_params and out_errs over each spaxel in
spaxels.
"""
assign_outputs(out_params::Array{<:Real}, out_errs::Array{<:Real}, cube_fitter::CubeFitter, 
    cube_data::NamedTuple, spaxels::CartesianIndices, z::Real, aperture::Bool=false) = 
    cube_fitter.spectral_region == :MIR ? 
        assign_outputs_mir(out_params, out_errs, cube_fitter, cube_data, spaxels, z, aperture) : 
        assign_outputs_opt(out_params, out_errs, cube_fitter, cube_data, spaxels, z, aperture)


function assign_outputs_mir(out_params::Array{<:Real}, out_errs::Array{<:Real}, cube_fitter::CubeFitter,
    cube_data::NamedTuple, spaxels::CartesianIndices, z::Real, aperture::Bool=false)

    # Create the CubeModel and ParamMaps structs to be filled in
    cube_model = generate_cubemodel(cube_fitter, aperture)
    param_maps, param_errs = generate_parammaps(cube_fitter, aperture)

    # Loop over each spaxel and fill in the associated fitting parameters into the ParamMaps and CubeModel
    # I know this is long and ugly and looks stupid but it works for now and I'll make it pretty later
    prog = Progress(length(spaxels); showspeed=true)
    @simd for index ∈ spaxels

        # Get the normalization to un-normalized the fitted parameters
        N = Float64(abs(nanmaximum(cube_data.I[index, :])))
        N = N ≠ 0. ? N : 1.

        # Set the 2D parameter map outputs

        # Conversion factor from MJy sr^-1 to erg s^-1 cm^-2 Hz^-1 sr^-1 = 10^6 * 10^-23 = 10^-17
        # So, log10(A * 1e-17) = log10(A) - 17

        # Stellar continuum amplitude, temp
        # Convert back to observed-frame amplitudes by multiplying by 1+z
        param_maps.stellar_continuum[:amp][index] = out_params[index, 1] > 0. ? log10(out_params[index, 1]*(1+z)) : -Inf 
        param_errs[1].stellar_continuum[:amp][index] = out_params[index, 1] > 0. ? out_errs[index, 1, 1] / (log(10) * out_params[index, 1]) : NaN
        param_errs[2].stellar_continuum[:amp][index] = out_params[index, 1] > 0. ? out_errs[index, 1, 2] / (log(10) * out_params[index, 1]) : NaN
        param_maps.stellar_continuum[:temp][index] = out_params[index, 2]
        param_errs[1].stellar_continuum[:temp][index] = out_errs[index, 2, 1]
        param_errs[2].stellar_continuum[:temp][index] = out_errs[index, 2, 2]
        pᵢ = 3

        # Dust continuum amplitude, temp
        for i ∈ 1:cube_fitter.n_dust_cont
            param_maps.dust_continuum[i][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]*(1+z)) : -Inf
            param_errs[1].dust_continuum[i][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 1] / (log(10) * out_params[index, pᵢ]) : NaN
            param_errs[2].dust_continuum[i][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 2] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.dust_continuum[i][:temp][index] = out_params[index, pᵢ+1]
            param_errs[1].dust_continuum[i][:temp][index] = out_errs[index, pᵢ+1, 1]
            param_errs[2].dust_continuum[i][:temp][index] = out_errs[index, pᵢ+1, 2]
            pᵢ += 2
        end

        for j ∈ 1:cube_fitter.n_power_law
            param_maps.power_law[j][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]*(1+z)) : -Inf
            param_errs[1].power_law[j][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 1] / (log(10) * out_params[index, pᵢ]) : NaN 
            param_errs[2].power_law[j][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 2] / (log(10) * out_params[index, pᵢ]) : NaN 
            param_maps.power_law[j][:index][index] = out_params[index, pᵢ+1]
            param_errs[1].power_law[j][:index][index] = out_errs[index, pᵢ+1, 1]
            param_errs[2].power_law[j][:index][index] = out_errs[index, pᵢ+1, 2]
            pᵢ += 2
        end

        # Extinction parameters
        param_maps.extinction[:tau_9_7][index] = out_params[index, pᵢ]
        param_errs[1].extinction[:tau_9_7][index] = out_errs[index, pᵢ, 1]
        param_errs[2].extinction[:tau_9_7][index] = out_errs[index, pᵢ, 2]
        param_maps.extinction[:tau_ice][index] = out_params[index, pᵢ+1]
        param_errs[1].extinction[:tau_ice][index] = out_errs[index, pᵢ+1, 1]
        param_errs[2].extinction[:tau_ice][index] = out_errs[index, pᵢ+1, 2]
        param_maps.extinction[:tau_ch][index] = out_params[index, pᵢ+2]
        param_errs[1].extinction[:tau_ch][index] = out_errs[index, pᵢ+2, 1]
        param_errs[2].extinction[:tau_ch][index] = out_errs[index, pᵢ+2, 2]
        param_maps.extinction[:beta][index] = out_params[index, pᵢ+3]
        param_errs[1].extinction[:beta][index] = out_errs[index, pᵢ+3, 1]
        param_errs[2].extinction[:beta][index] = out_errs[index, pᵢ+3, 2]
        pᵢ += 4

        for ab ∈ cube_fitter.abs_features.names
            param_maps.abs_features[ab][:tau][index] = out_params[index, pᵢ]
            param_errs[1].abs_features[ab][:tau][index] = out_errs[index, pᵢ, 1]
            param_errs[2].abs_features[ab][:tau][index] = out_errs[index, pᵢ, 2]
            param_maps.abs_features[ab][:mean][index] = out_params[index, pᵢ+1] * (1+z)
            param_errs[1].abs_features[ab][:mean][index] = out_errs[index, pᵢ+1, 1] * (1+z)
            param_errs[2].abs_features[ab][:mean][index] = out_errs[index, pᵢ+1, 2] * (1+z)
            param_maps.abs_features[ab][:fwhm][index] = out_params[index, pᵢ+2] * (1+z)
            param_errs[1].abs_features[ab][:fwhm][index] = out_errs[index, pᵢ+2, 1] * (1+z)
            param_errs[2].abs_features[ab][:fwhm][index] = out_errs[index, pᵢ+2, 2] * (1+z)
            pᵢ += 3
        end

        if cube_fitter.fit_sil_emission
            # Hot dust parameters
            param_maps.hot_dust[:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]*(1+z)) : -Inf
            param_errs[1].hot_dust[:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 1] / (log(10) * out_params[index, pᵢ]) : NaN
            param_errs[2].hot_dust[:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 2] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.hot_dust[:temp][index] = out_params[index, pᵢ+1]
            param_errs[1].hot_dust[:temp][index] = out_errs[index, pᵢ+1, 1]
            param_errs[2].hot_dust[:temp][index] = out_errs[index, pᵢ+1, 2]
            param_maps.hot_dust[:frac][index] = out_params[index, pᵢ+2]
            param_errs[1].hot_dust[:frac][index] = out_errs[index, pᵢ+2, 1]
            param_errs[2].hot_dust[:frac][index] = out_errs[index, pᵢ+2, 2]
            param_maps.hot_dust[:tau_warm][index] = out_params[index, pᵢ+3]
            param_errs[1].hot_dust[:tau_warm][index] = out_errs[index, pᵢ+3, 1]
            param_errs[2].hot_dust[:tau_warm][index] = out_errs[index, pᵢ+3, 2]
            param_maps.hot_dust[:tau_cold][index] = out_params[index, pᵢ+4]
            param_errs[1].hot_dust[:tau_cold][index] = out_errs[index, pᵢ+4, 1]
            param_errs[2].hot_dust[:tau_cold][index] = out_errs[index, pᵢ+4, 2]
            param_maps.hot_dust[:sil_peak][index] = out_params[index, pᵢ+5]
            param_errs[1].hot_dust[:sil_peak][index] = out_errs[index, pᵢ+5, 1]
            param_errs[2].hot_dust[:sil_peak][index] = out_errs[index, pᵢ+5, 2]
            pᵢ += 6
        end

        # Extinction normalization factor for the PAH/line amplitudes
        max_ext = out_params[index, end]

        # Dust feature log(amplitude), mean, FWHM
        for (k, df) ∈ enumerate(cube_fitter.dust_features.names)
            param_maps.dust_features[df][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]*(1+z)*max_ext*N)-17 : -Inf
            param_errs[1].dust_features[df][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 1] / (log(10) * out_params[index, pᵢ]) : NaN
            param_errs[2].dust_features[df][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 2] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.dust_features[df][:mean][index] = out_params[index, pᵢ+1] * (1+z)
            param_errs[1].dust_features[df][:mean][index] = out_errs[index, pᵢ+1, 1] * (1+z)
            param_errs[2].dust_features[df][:mean][index] = out_errs[index, pᵢ+1, 2] * (1+z)
            param_maps.dust_features[df][:fwhm][index] = out_params[index, pᵢ+2] * (1+z)
            param_errs[1].dust_features[df][:fwhm][index] = out_errs[index, pᵢ+2, 1] * (1+z)
            param_errs[2].dust_features[df][:fwhm][index] = out_errs[index, pᵢ+2, 2] * (1+z)
            if cube_fitter.dust_features.profiles[k] == :PearsonIV
                param_maps.dust_features[df][:index][index] = out_params[index, pᵢ+3]
                param_errs[1].dust_features[df][:index][index] = out_errs[index, pᵢ+3, 1] 
                param_errs[2].dust_features[df][:index][index] = out_errs[index, pᵢ+3, 2]
                param_maps.dust_features[df][:cutoff][index] = out_params[index, pᵢ+4]
                param_errs[1].dust_features[df][:cutoff][index] = out_errs[index, pᵢ+4, 1] 
                param_errs[2].dust_features[df][:cutoff][index] = out_errs[index, pᵢ+4, 2]
                pᵢ += 2
            end
            pᵢ += 3
        end

        if cube_fitter.save_full_model
            # End of continuum parameters: recreate the continuum model
            I_cont, comps_c = model_continuum(cube_fitter.cube.λ, out_params[index, 1:pᵢ-1], N, cube_fitter.n_dust_cont, cube_fitter.n_power_law,
                cube_fitter.dust_features.profiles, cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, 
                cube_fitter.fit_sil_emission, false, true)
        end

        # Save marker of the point where the continuum parameters end and the line parameters begin
        vᵢ = pᵢ

        for k ∈ 1:cube_fitter.n_lines
            amp_1 = amp_1_err = voff_1 = voff_1_err = fwhm_1 = fwhm_1_err = nothing
            for j ∈ 1:cube_fitter.n_comps
                if !isnothing(cube_fitter.lines.profiles[k, j])

                    ln = Symbol(cube_fitter.lines.names[k], "_$(j)")
                    amp = out_params[index, pᵢ]
                    amp_err = out_errs[index, pᵢ, :]
                    if isone(j)
                        param_maps.lines[ln][:amp][index] = amp
                        param_errs[1].lines[ln][:amp][index] = amp_err[1]
                        param_errs[2].lines[ln][:amp][index] = amp_err[2]
                        amp_1 = amp
                        amp_1_err = amp_err
                    else
                        param_maps.lines[ln][:amp][index] = amp * amp_1
                        param_errs[1].lines[ln][:amp][index] = √((amp_1_err[1] * amp)^2 + (amp_err[1] * amp_1)^2)
                        param_errs[2].lines[ln][:amp][index] = √((amp_1_err[2] * amp)^2 + (amp_err[2] * amp_1)^2)
                    end
                    
                    # Voff parameter
                    voff = out_params[index, pᵢ+1]
                    voff_err = out_errs[index, pᵢ+1, :]
                    if isone(j)
                        param_maps.lines[ln][:voff][index] = voff
                        param_errs[1].lines[ln][:voff][index] = voff_err[1]
                        param_errs[2].lines[ln][:voff][index] = voff_err[2]
                        voff_1 = voff
                        voff_1_err = voff_err
                    else
                        param_maps.lines[ln][:voff][index] = voff + voff_1
                        param_errs[1].lines[ln][:voff][index] = √(voff_err[1]^2 + voff_1_err[1]^2)
                        param_errs[2].lines[ln][:voff][index] = √(voff_err[2]^2 + voff_1_err[2]^2)
                    end

                    # Individual voff parameter
                    if !isnothing(cube_fitter.lines.tied_voff[k, j]) && cube_fitter.flexible_wavesol && isone(j)
                        voff_indiv = out_params[index, pᵢ+2]
                        voff_indiv_err = out_errs[index, pᵢ+2, :]
                        param_maps.lines[ln][:voff_indiv][index] = voff_indiv
                        param_errs[1].lines[ln][:voff_indiv][index] = voff_indiv_err[1]
                        param_errs[2].lines[ln][:voff_indiv][index] = voff_indiv_err[2]
                        fwhm = out_params[index, pᵢ+3]
                        fwhm_err = out_errs[index, pᵢ+3, :]
                        pᵢ += 4
                    else
                        fwhm = out_params[index, pᵢ+2]
                        fwhm_err = out_errs[index, pᵢ+2, :]                        
                        pᵢ += 3
                    end

                    # FWHM parameter
                    if isone(j)
                        param_maps.lines[ln][:fwhm][index] = fwhm
                        param_errs[1].lines[ln][:fwhm][index] = fwhm_err[1]
                        param_errs[2].lines[ln][:fwhm][index] = fwhm_err[2]
                        fwhm_1 = fwhm
                        fwhm_1_err = fwhm_err
                    else
                        param_maps.lines[ln][:fwhm][index] = fwhm * fwhm_1
                        param_errs[1].lines[ln][:fwhm][index] = √((fwhm_1_err[1] * fwhm)^2 + (fwhm_err[1] * fwhm_1)^2)
                        param_errs[2].lines[ln][:fwhm][index] = √((fwhm_1_err[2] * fwhm)^2 + (fwhm_err[2] * fwhm_1)^2)
                    end

                    # Get Gauss-Hermite 3rd and 4th order moments
                    if cube_fitter.lines.profiles[k, j] == :GaussHermite
                        param_maps.lines[ln][:h3][index] = out_params[index, pᵢ]
                        param_errs[1].lines[ln][:h3][index] = out_errs[index, pᵢ, 1]
                        param_errs[2].lines[ln][:h3][index] = out_errs[index, pᵢ, 2]
                        param_maps.lines[ln][:h4][index] = out_params[index, pᵢ+1]
                        param_errs[1].lines[ln][:h4][index] = out_errs[index, pᵢ+1, 1]
                        param_errs[2].lines[ln][:h4][index] = out_errs[index, pᵢ+1, 2]
                        pᵢ += 2
                    elseif cube_fitter.lines.profiles[k, j] == :Voigt
                        param_maps.lines[ln][:mixing][index] = out_params[index, pᵢ]
                        param_errs[1].lines[ln][:mixing][index] = out_errs[index, pᵢ, 1]
                        param_errs[2].lines[ln][:mixing][index] = out_errs[index, pᵢ, 2]
                        pᵢ += 1
                    end
                end
            end
        end

        if cube_fitter.save_full_model

            # Interpolate the LSF
            lsf_interp = Spline1D(cube_fitter.cube.λ, cube_fitter.cube.lsf, k=1)
            lsf_interp_func = x -> lsf_interp(x)

            # End of line parameters: recreate the un-extincted (intrinsic) line model
            I_line, comps_l = model_line_residuals(cube_fitter.cube.λ, out_params[index, vᵢ:pᵢ-1], cube_fitter.n_lines, cube_fitter.n_comps,
                cube_fitter.lines, cube_fitter.flexible_wavesol, comps_c["extinction"], lsf_interp_func, true)

            # Combine the continuum and line models
            I_model = I_cont .+ I_line
            comps = merge(comps_c, comps_l)

            # Renormalize
            I_model .*= N
            for comp ∈ keys(comps)
                if !(comp ∈ ["extinction", "abs_ice", "abs_ch"])
                    comps[comp] .*= N
                end
            end
            
        end

        # Dust feature intensity, EQW, and SNR, from calculate_extra_parameters
        for df ∈ cube_fitter.dust_features.names
            param_maps.dust_features[df][:flux][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]) : -Inf
            param_errs[1].dust_features[df][:flux][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 1] / (log(10) * out_params[index, pᵢ]) : NaN
            param_errs[2].dust_features[df][:flux][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 2] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.dust_features[df][:eqw][index] = out_params[index, pᵢ+1] * (1+z)
            param_errs[1].dust_features[df][:eqw][index] = out_errs[index, pᵢ+1, 1] * (1+z)
            param_errs[2].dust_features[df][:eqw][index] = out_errs[index, pᵢ+1, 2] * (1+z)
            param_maps.dust_features[df][:SNR][index] = out_params[index, pᵢ+2]
            param_errs[1].dust_features[df][:SNR][index] = out_errs[index, pᵢ+2, 1]
            param_errs[2].dust_features[df][:SNR][index] = out_errs[index, pᵢ+2, 2]
            pᵢ += 3
        end

        for k ∈ 1:cube_fitter.n_lines
            for j ∈ 1:cube_fitter.n_comps
                if !isnothing(cube_fitter.lines.profiles[k, j])

                    ln = Symbol(cube_fitter.lines.names[k], "_$(j)")

                    # Convert amplitudes to the correct units, then take the log
                    amp_norm = param_maps.lines[ln][:amp][index]
                    amp_norm_err = [param_errs[1].lines[ln][:amp][index], param_errs[2].lines[ln][:amp][index]]
                    param_maps.lines[ln][:amp][index] = amp_norm > 0 ? log10(amp_norm * max_ext * N * (1+z))-17 : -Inf
                    param_errs[1].lines[ln][:amp][index] = amp_norm > 0 ? amp_norm_err[1] / (log(10) * amp_norm) : NaN
                    param_errs[2].lines[ln][:amp][index] = amp_norm > 0 ? amp_norm_err[2] / (log(10) * amp_norm) : NaN

                    # Line intensity, EQW, and SNR, from calculate_extra_parameters
                    param_maps.lines[ln][:flux][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]) : -Inf
                    param_errs[1].lines[ln][:flux][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 1] / (log(10) * out_params[index, pᵢ]) : NaN
                    param_errs[2].lines[ln][:flux][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 2] / (log(10) * out_params[index, pᵢ]) : NaN
                    param_maps.lines[ln][:eqw][index] = out_params[index, pᵢ+1] * (1+z)
                    param_errs[1].lines[ln][:eqw][index] = out_errs[index, pᵢ+1, 1] * (1+z)
                    param_errs[2].lines[ln][:eqw][index] = out_errs[index, pᵢ+1, 2] * (1+z)
                    param_maps.lines[ln][:SNR][index] = out_params[index, pᵢ+2]
                    param_errs[1].lines[ln][:SNR][index] = out_errs[index, pᵢ+2, 1]
                    param_errs[2].lines[ln][:SNR][index] = out_errs[index, pᵢ+2, 2]
                    pᵢ += 3
                end
            end
        end

        # Reduced χ^2 = χ^2 / dof
        param_maps.statistics[:chi2][index] = out_params[index, pᵢ]
        param_maps.statistics[:dof][index] = out_params[index, pᵢ+1]

        if cube_fitter.save_full_model
            # Set 3D model cube outputs, shifted back to the observed frame
            cube_model.model[index, :] .= I_model .* (1 .+ z)
            cube_model.stellar[index, :] .= comps["stellar"] .* (1 .+ z)
            for i ∈ 1:cube_fitter.n_dust_cont
                cube_model.dust_continuum[index, :, i] .= comps["dust_cont_$i"] .* (1 .+ z)
            end
            for l ∈ 1:cube_fitter.n_power_law
                cube_model.power_law[index, :, l] .= comps["power_law_$l"] .* (1 .+ z)
            end
            for j ∈ 1:cube_fitter.n_dust_feat
                cube_model.dust_features[index, :, j] .= comps["dust_feat_$j"] .* (1 .+ z)
            end
            for m ∈ 1:cube_fitter.n_abs_feat
                cube_model.abs_features[index, :, m] .= comps["abs_feat_$m"]
            end
            if cube_fitter.fit_sil_emission
                cube_model.hot_dust[index, :] .= comps["hot_dust"] .* (1 .+ z)
            end
            for j ∈ 1:cube_fitter.n_comps
                for k ∈ 1:cube_fitter.n_lines
                    if !isnothing(cube_fitter.lines.profiles[k, j])
                        cube_model.lines[index, :, k] .+= comps["line_$(k)_$(j)"] .* (1 .+ z)
                    end
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
        for vk ∈ cube_fitter.tied_kinematics.key_voff[1]
            indiv_voffs = nothing
            snrs = nothing
            # Loop through and create 3D arrays of the voffs and SNRs of each line in the tied kinematic group
            for k ∈ 1:cube_fitter.n_lines
                name = Symbol(cube_fitter.lines.names[k], "_1")
                if cube_fitter.lines.tied_voff[k, 1] == vk
                    if isnothing(indiv_voffs)
                        indiv_voffs = param_maps.lines[name][:voff_indiv]
                        snrs = param_maps.lines[name][:SNR]
                        continue
                    end
                    indiv_voffs = cat(indiv_voffs, param_maps.lines[name][:voff_indiv], dims=3)
                    snrs = cat(snrs, param_maps.lines[name][:SNR], dims=3)
                end
            end
            # Collapse the voff array into an average along the 3rd dimension, ignoring any with an SNR < 3
            if !isnothing(indiv_voffs) && !isnothing(snrs)
                indiv_voffs[snrs .< 3] .= NaN
                avg_offset = dropdims(nanmean(indiv_voffs, dims=3), dims=3)
                # (the goal is to have the average offset of the individual voffs be 0, relative to the tied voff)
                for k ∈ 1:cube_fitter.n_lines
                    name = Symbol(cube_fitter.lines.names[k], "_1")
                    if cube_fitter.lines.tied_voff[k, 1] == vk
                        # Subtract the average offset from the individual voffs
                        param_maps.lines[name][:voff_indiv] .-= avg_offset
                        # and add it to the tied voffs
                        param_maps.lines[name][:voff] .+= avg_offset
                    end
                end
            end
        end
    end

    param_maps, param_errs, cube_model

end


function assign_outputs_opt(out_params::Array{<:Real}, out_errs::Array{<:Real}, cube_fitter::CubeFitter,
    cube_data::NamedTuple, spaxels::CartesianIndices, z::Real, aperture::Bool=false)

    # Create the CubeModel and ParamMaps structs to be filled in
    cube_model = generate_cubemodel(cube_fitter, aperture)
    param_maps, param_errs = generate_parammaps(cube_fitter, aperture)

    # Loop over each spaxel and fill in the associated fitting parameters into the ParamMaps and CubeModel
    # I know this is long and ugly and looks stupid but it works for now and I'll make it pretty later
    prog = Progress(length(spaxels); showspeed=true)
    @simd for index ∈ spaxels

        # Get the normalization to un-normalized the fitted parameters
        N = Float64(abs(nanmaximum(cube_data.I[index, :])))
        N = N ≠ 0. ? N : 1.

        # Set the 2D parameter map outputs

        # Conversion factor from MJy sr^-1 to erg s^-1 cm^-2 Hz^-1 sr^-1 = 10^6 * 10^-23 = 10^-17
        # So, log10(A * 1e-17) = log10(A) - 17

        # Stellar continuum amplitude, temp
        # Convert back to observed-frame amplitudes by multiplying by 1+z
        pᵢ = 1
        for i ∈ 1:cube_fitter.n_ssps
            # Un-normalize the amplitudes by applying the normalization factors used in the fitting routines
            # (median of the SSP template, followed by normalization N)
            ssp_med = median([cube_fitter.ssp_templates[j](out_params[index, pᵢ+1], out_params[index, pᵢ+2]) for j in eachindex(cube_fitter.ssp_λ)])
            param_maps.stellar_populations[i][:mass][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ] / ssp_med * N / (1+z)) : -Inf
            param_errs[1].stellar_populations[i][:mass][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 1] / (log(10) * out_params[index, pᵢ]) : NaN
            param_errs[2].stellar_populations[i][:mass][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 2] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.stellar_populations[i][:age][index] = out_params[index, pᵢ+1]
            param_errs[1].stellar_populations[i][:age][index] = out_errs[index, pᵢ+1, 1] 
            param_errs[2].stellar_populations[i][:age][index] = out_errs[index, pᵢ+1, 2] 
            param_maps.stellar_populations[i][:metallicity][index] = out_params[index, pᵢ+2]
            param_errs[1].stellar_populations[i][:metallicity][index] = out_errs[index, pᵢ+2, 1] 
            param_errs[2].stellar_populations[i][:metallicity][index] = out_errs[index, pᵢ+2, 2] 
            pᵢ += 3
        end

        param_maps.stellar_kinematics[:vel][index] = out_params[index, pᵢ]
        param_errs[1].stellar_kinematics[:vel][index] = out_errs[index, pᵢ, 1]
        param_errs[2].stellar_kinematics[:vel][index] = out_errs[index, pᵢ, 2]
        param_maps.stellar_kinematics[:vdisp][index] = out_params[index, pᵢ+1]
        param_errs[1].stellar_kinematics[:vdisp][index] = out_errs[index, pᵢ+1, 1]
        param_errs[2].stellar_kinematics[:vdisp][index] = out_errs[index, pᵢ+1, 2]
        pᵢ += 2

        # Report the E(B-V) for gas, rather than stars
        param_maps.attenuation[:E_BV][index] = out_params[index, pᵢ] / 0.44
        param_errs[1].attenuation[:E_BV][index] = out_errs[index, pᵢ, 1] / 0.44
        param_errs[2].attenuation[:E_BV][index] = out_errs[index, pᵢ, 2] / 0.44
        pᵢ += 1
        if cube_fitter.fit_uv_bump && cube_fitter.extinction_curve == "calzetti"
            param_maps.attenuation[:delta_UV][index] = out_params[index, pᵢ]
            param_errs[1].attenuation[:delta_UV][index] = out_errs[index, pᵢ, 1]
            param_errs[2].attenuation[:delta_UV][index] = out_errs[index, pᵢ, 2]
            pᵢ += 1
        end
        if cube_fitter.fit_covering_frac && cube_fitter.extinction_curve == "calzetti"
            param_maps.attenuation[:frac][index] = out_params[index, pᵢ]
            param_errs[1].attenuation[:frac][index] = out_errs[index, pᵢ, 1]
            param_errs[2].attenuation[:frac][index] = out_errs[index, pᵢ, 2]
            pᵢ += 1
        end

        if cube_fitter.save_full_model
            # End of continuum parameters: recreate the continuum model
            I_cont, comps_c = model_continuum(cube_fitter.cube.λ, out_params[index, 1:pᵢ-1], N, cube_fitter.velscale, cube_fitter.vsyst,
                cube_fitter.n_ssps, cube_fitter.ssp_λ, cube_fitter.ssp_templates, cube_fitter.fit_uv_bump, cube_fitter.fit_covering_frac,
                cube_data.area_sr, cube_fitter.extinction_curve, true)
        end

        # Save marker of the point where the continuum parameters end and the line parameters begin
        vᵢ = pᵢ

        for k ∈ 1:cube_fitter.n_lines
            amp_1 = amp_1_err = voff_1 = voff_1_err = fwhm_1 = fwhm_1_err = nothing
            for j ∈ 1:cube_fitter.n_comps
                if !isnothing(cube_fitter.lines.profiles[k, j])

                    ln = Symbol(cube_fitter.lines.names[k], "_$(j)")
                    amp = out_params[index, pᵢ]
                    amp_err = out_errs[index, pᵢ, :]
                    if isone(j)
                        param_maps.lines[ln][:amp][index] = amp
                        param_errs[1].lines[ln][:amp][index] = amp_err[1]
                        param_errs[2].lines[ln][:amp][index] = amp_err[2]
                        amp_1 = amp
                        amp_1_err = amp_err
                    else
                        param_maps.lines[ln][:amp][index] = amp * amp_1
                        param_errs[1].lines[ln][:amp][index] = √((amp_1_err[1] * amp)^2 + (amp_err[1] * amp_1)^2)
                        param_errs[2].lines[ln][:amp][index] = √((amp_1_err[2] * amp)^2 + (amp_err[2] * amp_1)^2)
                    end
                    
                    # Voff parameter
                    voff = out_params[index, pᵢ+1]
                    voff_err = out_errs[index, pᵢ+1, :]
                    if isone(j)
                        param_maps.lines[ln][:voff][index] = voff
                        param_errs[1].lines[ln][:voff][index] = voff_err[1]
                        param_errs[2].lines[ln][:voff][index] = voff_err[2]
                        voff_1 = voff
                        voff_1_err = voff_err
                    else
                        param_maps.lines[ln][:voff][index] = voff + voff_1
                        param_errs[1].lines[ln][:voff][index] = √(voff_err[1]^2 + voff_1_err[1]^2)
                        param_errs[2].lines[ln][:voff][index] = √(voff_err[2]^2 + voff_1_err[2]^2)
                    end

                    # Individual voff parameter
                    if !isnothing(cube_fitter.lines.tied_voff[k, j]) && cube_fitter.flexible_wavesol && isone(j)
                        voff_indiv = out_params[index, pᵢ+2]
                        voff_indiv_err = out_errs[index, pᵢ+2, :]
                        param_maps.lines[ln][:voff_indiv][index] = voff_indiv
                        param_errs[1].lines[ln][:voff_indiv][index] = voff_indiv_err[1]
                        param_errs[2].lines[ln][:voff_indiv][index] = voff_indiv_err[2]
                        fwhm = out_params[index, pᵢ+3]
                        fwhm_err = out_errs[index, pᵢ+3, :]
                        pᵢ += 4
                    else
                        fwhm = out_params[index, pᵢ+2]
                        fwhm_err = out_errs[index, pᵢ+2, :]                        
                        pᵢ += 3
                    end

                    # FWHM parameter
                    if isone(j)
                        param_maps.lines[ln][:fwhm][index] = fwhm
                        param_errs[1].lines[ln][:fwhm][index] = fwhm_err[1]
                        param_errs[2].lines[ln][:fwhm][index] = fwhm_err[2]
                        fwhm_1 = fwhm
                        fwhm_1_err = fwhm_err
                    else
                        param_maps.lines[ln][:fwhm][index] = fwhm * fwhm_1
                        param_errs[1].lines[ln][:fwhm][index] = √((fwhm_1_err[1] * fwhm)^2 + (fwhm_err[1] * fwhm_1)^2)
                        param_errs[2].lines[ln][:fwhm][index] = √((fwhm_1_err[2] * fwhm)^2 + (fwhm_err[2] * fwhm_1)^2)
                    end

                    # Get Gauss-Hermite 3rd and 4th order moments
                    if cube_fitter.lines.profiles[k, j] == :GaussHermite
                        param_maps.lines[ln][:h3][index] = out_params[index, pᵢ]
                        param_errs[1].lines[ln][:h3][index] = out_errs[index, pᵢ, 1]
                        param_errs[2].lines[ln][:h3][index] = out_errs[index, pᵢ, 2]
                        param_maps.lines[ln][:h4][index] = out_params[index, pᵢ+1]
                        param_errs[1].lines[ln][:h4][index] = out_errs[index, pᵢ+1, 1]
                        param_errs[2].lines[ln][:h4][index] = out_errs[index, pᵢ+1, 2]
                        pᵢ += 2
                    elseif cube_fitter.lines.profiles[k, j] == :Voigt
                        param_maps.lines[ln][:mixing][index] = out_params[index, pᵢ]
                        param_errs[1].lines[ln][:mixing][index] = out_errs[index, pᵢ, 1]
                        param_errs[2].lines[ln][:mixing][index] = out_errs[index, pᵢ, 2]
                        pᵢ += 1
                    end
                end
            end
        end

        if cube_fitter.save_full_model

            # Interpolate the LSF
            lsf_interp = Spline1D(cube_fitter.cube.λ, cube_fitter.cube.lsf, k=1)
            lsf_interp_func = x -> lsf_interp(x)

            # End of line parameters: recreate the un-extincted (intrinsic) line model
            I_line, comps_l = model_line_residuals(cube_fitter.cube.λ, out_params[index, vᵢ:pᵢ-1], cube_fitter.n_lines, cube_fitter.n_comps,
                cube_fitter.lines, cube_fitter.flexible_wavesol, comps_c["attenuation_gas"], lsf_interp_func, true)

            # Combine the continuum and line models
            I_model = I_cont .+ I_line
            comps = merge(comps_c, comps_l)

            # Renormalize
            I_model .*= N
            for comp ∈ keys(comps)
                if !(comp ∈ ["attenuation_stars", "attenuation_gas"])
                    comps[comp] .*= N
                end
            end
            
        end

        # Extinction normalization factor for line amplitudes
        max_ext = out_params[index, end]

        for k ∈ 1:cube_fitter.n_lines
            for j ∈ 1:cube_fitter.n_comps
                if !isnothing(cube_fitter.lines.profiles[k, j])

                    ln = Symbol(cube_fitter.lines.names[k], "_$(j)")
                    λ0 = cube_fitter.lines.λ₀[k] * 1e4

                    # Convert amplitudes to the correct units, then take the log
                    amp_norm = param_maps.lines[ln][:amp][index]
                    amp_norm_err = [param_errs[1].lines[ln][:amp][index], param_errs[2].lines[ln][:amp][index]]
                    # Convert amplitude to erg/s/cm^2/Hz/sr to match with the MIR 
                    param_maps.lines[ln][:amp][index] = amp_norm > 0 ? log10(amp_norm * max_ext * N * λ0^2/(C_KMS * 1e13) / (1+z)) : -Inf
                    param_errs[1].lines[ln][:amp][index] = amp_norm > 0 ? amp_norm_err[1] / (log(10) * amp_norm) : NaN
                    param_errs[2].lines[ln][:amp][index] = amp_norm > 0 ? amp_norm_err[2] / (log(10) * amp_norm) : NaN

                    # Line intensity, EQW, and SNR, from calculate_extra_parameters
                    param_maps.lines[ln][:flux][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]) : -Inf
                    param_errs[1].lines[ln][:flux][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 1] / (log(10) * out_params[index, pᵢ]) : NaN
                    param_errs[2].lines[ln][:flux][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 2] / (log(10) * out_params[index, pᵢ]) : NaN
                    param_maps.lines[ln][:eqw][index] = out_params[index, pᵢ+1] * (1+z)
                    param_errs[1].lines[ln][:eqw][index] = out_errs[index, pᵢ+1, 1] * (1+z)
                    param_errs[2].lines[ln][:eqw][index] = out_errs[index, pᵢ+1, 2] * (1+z)
                    param_maps.lines[ln][:SNR][index] = out_params[index, pᵢ+2]
                    param_errs[1].lines[ln][:SNR][index] = out_errs[index, pᵢ+2, 1]
                    param_errs[2].lines[ln][:SNR][index] = out_errs[index, pᵢ+2, 2]
                    pᵢ += 3
                end
            end
        end

        # Reduced χ^2 = χ^2 / dof
        param_maps.statistics[:chi2][index] = out_params[index, pᵢ]
        param_maps.statistics[:dof][index] = out_params[index, pᵢ+1]

        if cube_fitter.save_full_model
            # Set 3D model cube outputs, shifted back to the observed frame
            cube_model.model[index, :] .= I_model ./ (1 .+ z)
            for i ∈ 1:cube_fitter.n_ssps
                cube_model.stellar[index, :, i] .= comps["SSP_$i"] ./ (1 .+ z)
            end
            cube_model.attenuation_stars[index, :] .= comps["attenuation_stars"]
            cube_model.attenuation_gas[index, :] .= comps["attenuation_gas"]

            for j ∈ 1:cube_fitter.n_comps
                for k ∈ 1:cube_fitter.n_lines
                    if !isnothing(cube_fitter.lines.profiles[k, j])
                        cube_model.lines[index, :, k] .+= comps["line_$(k)_$(j)"] ./ (1 .+ z)
                    end
                end
            end
        end

        next!(prog)

    end

    # Subtract the average of the individual voffs from the tied voffs, based on the SNR, for each group
    if cube_fitter.flexible_wavesol
        @debug "Adjusting individual voffs due to the flexible_wavesol option"
        for vk ∈ cube_fitter.tied_kinematics.key_voff[1]
            indiv_voffs = nothing
            snrs = nothing
            # Loop through and create 3D arrays of the voffs and SNRs of each line in the tied kinematic group
            for k ∈ 1:cube_fitter.n_lines
                name = Symbol(cube_fitter.lines.names[k], "_1")
                if cube_fitter.lines.tied_voff[k, 1] == vk
                    if isnothing(indiv_voffs)
                        indiv_voffs = param_maps.lines[name][:voff_indiv]
                        snrs = param_maps.lines[name][:SNR]
                        continue
                    end
                    indiv_voffs = cat(indiv_voffs, param_maps.lines[name][:voff_indiv], dims=3)
                    snrs = cat(snrs, param_maps.lines[name][:SNR], dims=3)
                end
            end
            # Collapse the voff array into an average along the 3rd dimension, ignoring any with an SNR < 3
            if !isnothing(indiv_voffs) && !isnothing(snrs)
                indiv_voffs[snrs .< 3] .= NaN
                avg_offset = dropdims(nanmean(indiv_voffs, dims=3), dims=3)
                # (the goal is to have the average offset of the individual voffs be 0, relative to the tied voff)
                for k ∈ 1:cube_fitter.n_lines
                    name = Symbol(cube_fitter.lines.names[k], "_1")
                    if cube_fitter.lines.tied_voff[k, 1] == vk
                        # Subtract the average offset from the individual voffs
                        param_maps.lines[name][:voff_indiv] .-= avg_offset
                        # and add it to the tied voffs
                        param_maps.lines[name][:voff] .+= avg_offset
                    end
                end
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
- `name_i::String`: The name of the individual parameter being plotted, i.e. "dust_features_PAH_5.24_amp"
- `save_path::String`: The file path to save the plot to.
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
function plot_parameter_map(data::Matrix{Float64}, name_i::String, save_path::String, Ω::Float64, z::Float64, psf_fwhm::Float64,
    cosmo::Cosmology.AbstractCosmology, python_wcs::Union{PyObject,Nothing}; snr_filter::Union{Nothing,Matrix{Float64}}=nothing, 
    snr_thresh::Float64=3., cmap::PyObject=py_colormap.cubehelix, line_latex::Union{String,Nothing}=nothing)

    # I know this is ugly but I couldn't figure out a better way to do it lmao
    if occursin("amp", String(name_i))
        if occursin("stellar_continuum", String(name_i))
            bunit = L"$\log_{10}(A_{*})$" # normalized
        elseif occursin("dust_continuum", String(name_i))
            bunit = L"$\log_{10}(A_{\rm dust})$" # normalized
        elseif occursin("power_law", String(name_i))
            bunit = L"$\log_{10}(A_{\rm pl})$" # normalized
        elseif occursin("hot_dust", String(name_i))
            bunit = L"$\log_{10}(A_{\rm sil})$" # normalized
        else
            bunit = L"$\log_{10}(I / $ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1})$"
        end
    elseif occursin("temp", String(name_i))
        bunit = L"$T$ (K)"
    elseif occursin("fwhm", String(name_i)) && (occursin("PAH", String(name_i)) || occursin("abs", String(name_i)))
        bunit = L"FWHM ($\mu$m)"
    elseif occursin("fwhm", String(name_i)) && !occursin("PAH", String(name_i)) && !occursin("abs", String(name_i))
        bunit = L"FWHM (km s$^{-1}$)"
    elseif occursin("mean", String(name_i)) || occursin("peak", String(name_i))
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
        elseif occursin("9_7", String(name_i))
            bunit = L"$\tau_{9.7}$"
        else
            bunit = L"$\tau$"
        end
    elseif occursin("flux", String(name_i))
        bunit = L"$\log_{10}(F /$ erg s$^{-1}$ cm$^{-2}$)"
    elseif occursin("eqw", String(name_i))
        bunit = L"$W_{\rm eq}$ ($\mu$m)"
    elseif occursin("chi2", String(name_i))
        bunit = L"$\tilde{\chi}^2$"
    elseif occursin("dof", String(name_i))
        bunit = "d.o.f."
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
    elseif occursin("index", String(name_i)) && occursin("PAH", String(name_i))
        bunit = L"$m$"
    elseif occursin("index", String(name_i)) && !occursin("PAH", String(name_i))
        bunit = L"$\alpha$"
    elseif occursin("cutoff", String(name_i))
        bunit = L"$\nu$"
    elseif occursin("mass", String(name_i))
        bunit = L"$\log_{10}(M / M_{\odot})$"
    elseif occursin("age", String(name_i))
        bunit = L"$t$ (Gyr)"
    elseif occursin("metallicity", String(name_i))
        bunit = L"[M$/$H]"
    elseif occursin("vel", String(name_i))
        bunit = L"$v_*$ (km s$^{-1}$)"
    elseif occursin("vdisp", String(name_i))
        bunit = L"$\sigma_*$ (km s$^{-1}$)"
    elseif occursin("E_BV", String(name_i))
        bunit = L"$E(B-V)_{\rm gas}$"
    elseif occursin("delta_UV", String(name_i))
        bunit = L"$\delta$"
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
    ax = fig.add_subplot(111, projection=python_wcs) 
    # Need to filter out any NaNs in order to use quantile
    vmin = nanminimum(filtered)
    vmax = nanmaximum(filtered)
    # override vmin/vmax for mixing parameter
    if occursin("mixing", String(name_i))
        vmin = 0.
        vmax = 1.
    end
    nan_color = "k"
    text_color = "w"
    # if taking a voff, make sure vmin/vmax are symmetric and change the colormap to coolwarm
    if occursin("voff", String(name_i)) || occursin("index", String(name_i)) || occursin("vel", String(name_i))
        vabs = max(abs(vmin), abs(vmax))
        vmin = -vabs
        vmax = vabs
        if cmap == py_colormap.cubehelix
            cmap = py_colormap.RdBu_r
            nan_color = "w"
            text_color = "k"
        end
    end
    if occursin("chi2", String(name_i))
        vmin = 0
        # Hard upper limit on the reduced chi^2 map to show the structure
        vmax = min(nanmaximum(filtered), 30)
    end
    # default cmap is magma for FWHMs and equivalent widths
    if (occursin("fwhm", String(name_i)) || occursin("eqw", String(name_i))) || occursin("vdisp", String(name_i)) && cmap == py_colormap.cubehelix
        cmap = py_colormap.magma
    end

    # Add small value to vmax to prevent the maximum color value from being the same as the background
    small = 0.
    if cmap == py_colormap.cubehelix
        small = (vmax - vmin) / 1e3
    end

    # Set NaN color to either black or white
    cmap.set_bad(color=nan_color)

    cdata = ax.imshow(filtered', origin=:lower, cmap=cmap, vmin=vmin, vmax=vmax+small)
    # ax.axis(:off)
    ax.tick_params(which="both", axis="both", direction="in")
    ax.set_xlabel(isnothing(python_wcs) ? L"$x$ (spaxels)" : "R.A.")
    ax.set_ylabel(isnothing(python_wcs) ? L"$y$ (spaxels)" : "Dec.")

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
    unit = "pc"
    # convert to kpc if l is more than 1000 pc
    if l ≥ 10^3
        l = Int(l / 10^3)
        unit = "kpc"
    elseif l ≥ 10^6
        l = Int(l / 10^6)
        unit = "Mpc"
    elseif l ≥ 10^9
        l = Int(l / 10^9)
        unit = "Gpc"
    end
    if cosmo.h ≈ 1.0
        scalebar = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, L"%$l$h^{-1}$ %$unit", "lower left", pad=1, color=text_color, 
            frameon=false, size_vertical=0.4, label_top=false)
    else
        scalebar = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, L"%$l %$unit", "lower left", pad=1, color=text_color,
            frameon=false, size_vertical=0.4, label_top=false)
    end
    ax.add_artist(scalebar)

    # Add circle for the PSF FWHM
    psf = plt.Circle(size(data) .* (0.9, 0.1), psf_fwhm / pix_as / 2, color=text_color)
    ax.add_patch(psf)
    ax.annotate("PSF", size(data) .* (0.9, 0.1) .+ (0., psf_fwhm / pix_as / 2 * 1.5 + 1.75), ha=:center, va=:center, color=text_color)

    # Add line label, if applicable
    if !isnothing(line_latex)
        ax.annotate(line_latex, size(data) .* 0.95, ha=:right, va=:top, fontsize=16, color=text_color)
    end

    fig.colorbar(cdata, ax=ax, label=bunit)

    # Make directories
    if !isdir(dirname(save_path))
        mkpath(dirname(save_path))
    end
    plt.savefig(save_path, dpi=300, bbox_inches=:tight)
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
plot_parameter_maps(cube_fitter::CubeFitter, param_maps::ParamMaps; snr_thresh::Real=3.) = 
    cube_fitter.spectral_region == :MIR ?
    plot_mir_parameter_maps(cube_fitter, param_maps; snr_thresh=snr_thresh) :
    plot_opt_parameter_maps(cube_fitter, param_maps; snr_thresh=snr_thresh)


function plot_mir_parameter_maps(cube_fitter::CubeFitter, param_maps::ParamMaps; snr_thresh::Real=3.)

    # Iterate over model parameters and make 2D maps
    @debug "Using solid angle $(cube_fitter.cube.Ω), redshift $(cube_fitter.z), cosmology $(cube_fitter.cosmology)"

    # Ineterpolate the PSF
    psf_interp = Spline1D(cube_fitter.cube.λ, cube_fitter.cube.psf, k=1)

    # Stellar continuum parameters
    for parameter ∈ keys(param_maps.stellar_continuum)
        data = param_maps.stellar_continuum[parameter]
        name_i = join(["stellar_continuum", parameter], "_")
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")
        plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf), 
            cube_fitter.cosmology, cube_fitter.cube.wcs)
    end

    # Dust continuum parameters
    for i ∈ keys(param_maps.dust_continuum)
        for parameter ∈ keys(param_maps.dust_continuum[i])
            data = param_maps.dust_continuum[i][parameter]
            name_i = join(["dust_continuum", i, parameter], "_") 
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf), 
                cube_fitter.cosmology, cube_fitter.cube.wcs)
        end
    end

    # Power law parameters
    for j ∈ keys(param_maps.power_law)
        for parameter ∈ keys(param_maps.power_law[j])
            data = param_maps.power_law[j][parameter]
            name_i = join(["power_law", j, parameter], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
                cube_fitter.cosmology, cube_fitter.cube.wcs)
        end
    end

    # Dust feature (PAH) parameters
    for df ∈ keys(param_maps.dust_features)
        snr = param_maps.dust_features[df][:SNR]
        # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
        wave_i = nanmedian(param_maps.dust_features[df][:mean]) / (1 + cube_fitter.z)
        
        # Create the name to annotate on the plot
        ind = findfirst(cube_fitter.dust_features.names .== df)
        comp = cube_fitter.dust_features.complexes[ind]
        latex_i = replace(df, "_" => " ") * L" $\mu$m"
        if !isnothing(comp)
            comp_name = L"PAH %$comp $\mu$m"
            indiv_inds = findall(cube_fitter.dust_features.complexes .== comp)
            if length(indiv_inds) > 1
                # Will already be sorted
                indivs = [cube_fitter.dust_features.names[i] for i ∈ indiv_inds]
                out_ind = findfirst(indivs .== df)
                latex_i = comp_name * " " * ascii_lowercase[out_ind]
            end
        end

        for parameter ∈ keys(param_maps.dust_features[df])
            data = param_maps.dust_features[df][parameter]
            name_i = join([df, parameter], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "dust_features", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=parameter !== :SNR ? snr : nothing, snr_thresh=snr_thresh,
                line_latex=latex_i)
        end
    end

    # Total parameters for PAH complexes
    dust_complexes = unique(cube_fitter.dust_features.complexes[.!isnothing.(cube_fitter.dust_features.complexes)])
    for dust_complex in dust_complexes
        # Get the components that make up the complex
        indiv_inds = findall(cube_fitter.dust_features.complexes .== dust_complex)
        indivs = [cube_fitter.dust_features.names[i] for i ∈ indiv_inds]
        # Sum up individual component fluxes
        total_flux = log10.(sum([exp10.(param_maps.dust_features[df][:flux]) for df ∈ indivs]))
        # Wavelength and name
        wave_i = parse(Float64, dust_complex)
        name_i = "complex_$(dust_complex)_total_flux"
        comp_name = "PAH $dust_complex " * L"$\mu$m"
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "dust_features", "$(name_i).pdf")
        plot_parameter_map(total_flux, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
            cube_fitter.cosmology, cube_fitter.cube.wcs, line_latex=comp_name)
    end

    # Absorption feature parameters
    for ab ∈ keys(param_maps.abs_features)
        wave_i = nanmedian(param_maps.abs_features[ab][:mean]) / (1 + cube_fitter.z)
        for parameter ∈ keys(param_maps.abs_features[ab])
            data = param_maps.abs_features[ab][parameter]
            name_i = join([ab, parameter], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "absorption_features", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs)
        end
    end

    # Extinction parameters
    for parameter ∈ keys(param_maps.extinction)
        data = param_maps.extinction[parameter]
        name_i = join(["extinction", parameter], "_")
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "extinction", "$(name_i).pdf")
        plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf), 
            cube_fitter.cosmology, cube_fitter.cube.wcs)
    end

    if cube_fitter.fit_sil_emission
        # Hot dust parameters
        for parameter ∈ keys(param_maps.hot_dust)
            data = param_maps.hot_dust[parameter]
            name_i = join(["hot_dust", parameter], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "hot_dust", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf), 
                cube_fitter.cosmology, cube_fitter.cube.wcs)
        end
    end

    # Line parameters
    for line ∈ keys(param_maps.lines)
        # Remove the component index from the line name
        line_key = Symbol(join(split(string(line), "_")[1:end-1], "_"))
        # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
        line_i = findfirst(cube_fitter.lines.names .== line_key)
        wave_i = cube_fitter.lines.λ₀[line_i]
        latex_i = cube_fitter.lines.latex[line_i]

        for parameter ∈ keys(param_maps.lines[line])
            data = param_maps.lines[line][parameter]
            name_i = join([line, parameter], "_")
            snr_filter = param_maps.lines[line][:SNR]
            if contains(string(parameter), "SNR")
                snr_filter = nothing
            end
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(line_key)", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i), 
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=latex_i)
        end
    end

    # Total parameters for lines with multiple components
    for (k, name) ∈ enumerate(cube_fitter.lines.names)
        n_line_comps = sum(.!isnothing.(cube_fitter.lines.profiles[k, :]))
        wave_i = cube_fitter.lines.λ₀[k]
        snr_filter = param_maps.lines[Symbol(name, "_1")][:SNR]
        if n_line_comps > 1
            component_keys = [Symbol(name, "_$(j)") for j in 1:n_line_comps]

            # Total flux
            total_flux = log10.(sum([exp10.(param_maps.lines[comp][:flux]) for comp in component_keys]))
            name_i = join([name, "total_flux"], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(name)", "$(name_i).pdf")
            plot_parameter_map(total_flux, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=cube_fitter.lines.latex[k])

            # Total equivalent width
            total_eqw = sum([param_maps.lines[comp][:eqw] for comp in component_keys])
            name_i = join([name, "total_eqw"], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(name)", "$(name_i).pdf")
            plot_parameter_map(total_eqw, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=cube_fitter.lines.latex[k])
            
        end
    end

    # Reduced chi^2 
    data = param_maps.statistics[:chi2] ./ param_maps.statistics[:dof]
    name_i = "reduced_chi2"
    save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "$(name_i).pdf")
    plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf), 
        cube_fitter.cosmology, cube_fitter.cube.wcs)

    return

end


function plot_opt_parameter_maps(cube_fitter::CubeFitter, param_maps::ParamMaps; snr_thresh::Real=3.)

    # Iterate over model parameters and make 2D maps
    @debug "Using solid angle $(cube_fitter.cube.Ω), redshift $(cube_fitter.z), cosmology $(cube_fitter.cosmology)"

    # Ineterpolate the PSF
    psf_interp = Spline1D(cube_fitter.cube.λ, cube_fitter.cube.psf, k=1)

    # Stellar population parameters
    for i ∈ keys(param_maps.stellar_populations)
        for parameter ∈ keys(param_maps.stellar_populations[i])
            data = param_maps.stellar_populations[i][parameter]
            name_i = join(["stellar_populations", i, parameter], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
                cube_fitter.cosmology, cube_fitter.cube.wcs)
        end
    end

    # Stellar kinematics parameters
    for parameter ∈ keys(param_maps.stellar_kinematics)
        data = param_maps.stellar_kinematics[parameter]
        name_i = join(["stellar_kinematics", parameter], "_")
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")
        plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
            cube_fitter.cosmology, cube_fitter.cube.wcs)
    end

    # Attenuation parameters
    for parameter ∈ keys(param_maps.attenuation)
        data = param_maps.attenuation[parameter]
        name_i = join(["attenuation", parameter], "_")
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "extinction", "$(name_i).pdf")
        plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
            cube_fitter.cosmology, cube_fitter.cube.wcs)
    end

    # Line parameters
    for line ∈ keys(param_maps.lines)
        # Remove the component index from the line name
        line_key = Symbol(join(split(string(line), "_")[1:end-1], "_"))
        # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
        line_i = findfirst(cube_fitter.lines.names .== line_key)
        wave_i = cube_fitter.lines.λ₀[line_i]
        latex_i = cube_fitter.lines.latex[line_i]

        for parameter ∈ keys(param_maps.lines[line])
            data = param_maps.lines[line][parameter]
            name_i = join([line, parameter], "_")
            snr_filter = param_maps.lines[line][:SNR]
            if contains(string(parameter), "SNR")
                snr_filter = nothing
            end
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(line_key)", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i), 
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=latex_i)
        end
    end

    # Total parameters for lines with multiple components
    for (k, name) ∈ enumerate(cube_fitter.lines.names)
        n_line_comps = sum(.!isnothing.(cube_fitter.lines.profiles[k, :]))
        wave_i = cube_fitter.lines.λ₀[k]
        snr_filter = param_maps.lines[Symbol(name, "_1")][:SNR]
        if n_line_comps > 1
            component_keys = [Symbol(name, "_$(j)") for j in 1:n_line_comps]

            # Total flux
            total_flux = log10.(sum([exp10.(param_maps.lines[comp][:flux]) for comp in component_keys]))
            name_i = join([name, "total_flux"], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(name)", "$(name_i).pdf")
            plot_parameter_map(total_flux, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=cube_fitter.lines.latex[k])

            # Total equivalent width
            total_eqw = sum([param_maps.lines[comp][:eqw] for comp in component_keys])
            name_i = join([name, "total_eqw"], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(name)", "$(name_i).pdf")
            plot_parameter_map(total_eqw, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=cube_fitter.lines.latex[k])
            
        end
    end

    # Reduced chi^2 
    data = param_maps.statistics[:chi2] ./ param_maps.statistics[:dof]
    name_i = "reduced_chi2"
    save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "$(name_i).pdf")
    plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf), 
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

    for (full_data, title) ∈ zip([cube_fitter.cube.I .* (1 .+ cube_fitter.z), cube_model.model], ["DATA", "MODEL"])

        # Writer using FFMpeg to create an mp4 file
        metadata = Dict(:title => title, :artist => "LOKI", :fps => 60)
        writer = py_animation.FFMpegWriter(fps=60, metadata=metadata)

        # Set up plots with gridspec
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=20,  nrows=10)
        ax1 = fig.add_subplot(py"$(gs)[0:8, 0:18]", projection=cube_fitter.cube.wcs)
        ax2 = fig.add_subplot(py"$(gs)[9:10, :]")
        ax3 = fig.add_subplot(py"$(gs)[0:8, 18:19]")

        # First wavelength slice of the model
        wave_rest = cube_fitter.cube.λ
        data = full_data[:, :, 1]

        # Get average along the wavelength dimension
        datasum = sumdim(full_data, 3)
        dataavg = datasum ./ size(full_data, 3)

        # Plot the first slice
        image = ax1.imshow(data', origin=:lower, cmap=cmap, vmin=nanquantile(dataavg, 0.01), vmax=nanquantile(dataavg, 0.99))
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
        for i ∈ axes(full_data, 3)
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
write_fits(cube_fitter::CubeFitter, cube_data::NamedTuple, cube_model::CubeModel, param_maps::ParamMaps, 
    param_errs::Vector{<:ParamMaps}; aperture::Union{Vector{PyObject},Nothing}=nothing) =
    cube_fitter.spectral_region == :MIR ?
    write_fits_mir(cube_fitter, cube_data, cube_model, param_maps, param_errs; aperture=aperture) :
    write_fits_opt(cube_fitter, cube_data, cube_model, param_maps, param_errs; aperture=aperture)


function write_fits_mir(cube_fitter::CubeFitter, cube_data::NamedTuple, cube_model::CubeModel, param_maps::ParamMaps, 
    param_errs::Vector{<:ParamMaps}; aperture::Union{Vector{PyObject},Nothing}=nothing)

    aperture_keys = []
    aperture_vals = []
    aperture_comments = []
    # If using an aperture, extract its properties 
    if !isnothing(aperture)
        # Get the RA and Dec of the centroid
        sky_aperture = aperture[1].to_sky(cube_fitter.cube.wcs)
        sky_cent = sky_aperture.positions
        ra_cent = format_angle(ha2hms(sky_cent.ra[1]/15); delim=["h","m","s"])
        dec_cent = format_angle(deg2dms(sky_cent.dec[1]); delim=["d","m","s"])

        # Get the name (giving the shape of the aperture: circular, elliptical, or rectangular)
        ap_shape = aperture[1].__class__.__name__

        aperture_keys = ["AP_SHAPE", "AP_RA", "AP_DEC"]
        aperture_vals = Any[ap_shape, ra_cent, dec_cent]
        aperture_comments = ["The shape of the spectrum extraction aperture", "The RA of the aperture",
            "The dec of the aperture"]

        # Get the properties, i.e. radius for circular 
        if ap_shape == "CircularAperture"
            append!(aperture_keys, ["AP_RADIUS"])
            append!(aperture_vals, sky_aperture.r[1])
            append!(aperture_comments, ["Radius of aperture in arcsec"])
        elseif ap_shape == "EllipticalAperture"
            append!(aperture_keys, ["AP_A", "AP_B", "AP_PA"])
            append!(aperture_vals, [sky_aperture.a[1], sky_aperture.b[1], sky_aperture.theta[1]])
            append!(aperture_comments, ["Semimajor axis of aperture in arcsec", 
                "Semiminor axis of aperture in arcsec", "Aperture position angle in rad."])
        elseif ap_shape == "RectangularAperture"
            append!(aperture_keys, ["AP_W", "AP_H", "AP_PA"])
            append!(aperture_vals, [sky_aperture.w[1], sky_aperture.h[1], sky_aperture.theta[1]])
            append!(aperture_comments, ["Width of aperture in arcsec", 
                "Height of aperture in arcsec", "Aperture position angle in rad."])
        end

        # Also append the aperture area
        append!(aperture_keys, ["AP_AR_SR"])
        append!(aperture_vals, [cube_data.area_sr[1]])
        append!(aperture_comments, ["Area of aperture in steradians"])
    end

    # Header information
    hdr = FITSHeader(
        Vector{String}(cat(["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "PIXAR_SR", "RA", "DEC", "WCSAXES",
            "CDELT1", "CDELT2", "CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CUNIT1", "CUNIT2", 
            "PC1_1", "PC1_2", "PC2_1", "PC2_2"], aperture_keys, dims=1)),

        cat([cube_fitter.name, cube_fitter.z, cube_fitter.cube.channel, cube_fitter.cube.band, cube_fitter.cube.Ω, 
         cube_fitter.cube.α, cube_fitter.cube.δ, cube_fitter.cube.wcs.wcs.naxis, 
         cube_fitter.cube.wcs.wcs.cdelt[1], cube_fitter.cube.wcs.wcs.cdelt[2], 
         cube_fitter.cube.wcs.wcs.ctype[1], cube_fitter.cube.wcs.wcs.ctype[2], 
         cube_fitter.cube.wcs.wcs.crpix[1], cube_fitter.cube.wcs.wcs.crpix[2], 
         cube_fitter.cube.wcs.wcs.crval[1], cube_fitter.cube.wcs.wcs.crval[2], 
         cube_fitter.cube.wcs.wcs.cunit[1].name, cube_fitter.cube.wcs.wcs.cunit[2].name, 
         cube_fitter.cube.wcs.wcs.pc[1,1], cube_fitter.cube.wcs.wcs.pc[1,2], 
         cube_fitter.cube.wcs.wcs.pc[2,1], cube_fitter.cube.wcs.wcs.pc[2,2]], aperture_vals, dims=1),

        Vector{String}(cat(["Target name", "Target redshift", "MIRI channel", "MIRI band",
        "Solid angle per pixel (rad.)", "Right ascension of target (deg.)", "Declination of target (deg.)",
        "number of World Coordinate System axes", 
        "first axis increment per pixel", "second axis increment per pixel",
        "first axis coordinate type", "second axis coordinate type",
        "axis 1 coordinate of the reference pixel", "axis 2 coordinate of the reference pixel",
        "first axis value at the reference pixel", "second axis value at the reference pixel",
        "first axis units", "second axis units",
        "linear transformation matrix element", "linear transformation matrix element",
        "linear transformation matrix element", "linear transformation matrix element"], aperture_comments, dims=1))
    )

    if cube_fitter.save_full_model
        # Create the 3D intensity model FITS file
        FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_full_model.fits"), "w") do f

            @debug "Writing 3D model FITS HDUs"

            write(f, Vector{Int}(); header=hdr)                                                         # Primary HDU (empty)
            write(f, Float32.(cube_data.I .* (1 .+ cube_fitter.z)); name="DATA")                        # Raw data 
            write(f, Float32.(cube_data.σ .* (1 .+ cube_fitter.z)); name="ERROR")                       # Error in the raw data
            write(f, cube_model.model; name="MODEL")                                                    # Full intensity model
            write(f, cube_model.stellar; name="STELLAR_CONTINUUM")                                      # Stellar continuum model
            for i ∈ 1:size(cube_model.dust_continuum, 4)
                write(f, cube_model.dust_continuum[:, :, :, i]; name="DUST_CONTINUUM_$i")               # Dust continua
            end
            for l ∈ 1:size(cube_model.power_law, 4)                                                     # Power laws
                write(f, cube_model.power_law[:, :, :, l]; name="POWER_LAW_$l")
            end
            for (j, df) ∈ enumerate(cube_fitter.dust_features.names)
                write(f, cube_model.dust_features[:, :, :, j]; name="$df")                              # Dust feature profiles
            end
            for (m, ab) ∈ enumerate(cube_fitter.abs_features.names)                                     
                write(f, cube_model.abs_features[:, :, :, m]; name="$ab")                               # Absorption feature profiles
            end
            for (k, line) ∈ enumerate(cube_fitter.lines.names)
                write(f, cube_model.lines[:, :, :, k]; name="$line")                                    # Emission line profiles
            end
            write(f, cube_model.extinction; name="EXTINCTION")                                          # Extinction model
            write(f, cube_model.abs_ice; name="ABS_ICE")                                                # Ice Absorption model
            write(f, cube_model.abs_ch; name="ABS_CH")                                                  # CH Absorption model
            if cube_fitter.fit_sil_emission
                write(f, cube_model.hot_dust; name="HOT_DUST")                                          # Hot dust model
            end
            
            write(f, ["wave"], [cube_data.λ .* (1 .+ cube_fitter.z)],                                   # wavelength vector
                hdutype=TableHDU, name="WAVELENGTH", units=Dict(:wave => "um"))

            # Insert physical units into the headers of each HDU -> MegaJansky per steradian for all except
            # the extinction profile, which is a multiplicative constant
            write_key(f["DATA"], "BUNIT", "MJy/sr")
            write_key(f["ERROR"], "BUNIT", "MJy/sr")
            write_key(f["MODEL"], "BUNIT", "MJy/sr")
            write_key(f["STELLAR_CONTINUUM"], "BUNIT", "MJy/sr")
            for i ∈ 1:size(cube_model.dust_continuum, 4)
                write_key(f["DUST_CONTINUUM_$i"], "BUNIT", "MJy/sr")
            end
            for l ∈ 1:size(cube_model.power_law, 4)
                write_key(f["POWER_LAW_$l"], "BUNIT", "MJy/sr")
            end
            for df ∈ cube_fitter.dust_features.names
                write_key(f["$df"], "BUNIT", "MJy/sr")
            end
            for ab ∈ cube_fitter.abs_features.names
                write_key(f["$ab"], "BUNIT", "unitless")
            end
            for line ∈ cube_fitter.lines.names
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

    # Create the 2D parameter map FITS file for the parameters and the errors
    for (index, param_data) ∈ enumerate([param_maps, param_errs[1], param_errs[2]])

        FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_parameter_" * 
            ("maps", "errs_low", "errs_upp")[index] * ".fits"), "w") do f

            @debug "Writing 2D parameter map FITS HDUs"

            write(f, Vector{Int}(), header=hdr)  # Primary HDU (empty)

            # Stellar continuum parameters
            for parameter ∈ keys(param_data.stellar_continuum)
                data = param_data.stellar_continuum[parameter]
                name_i = join(["stellar_continuum", parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                elseif occursin("temp", String(name_i))
                    bunit = "Kelvin"
                end
                write(f, data; name=name_i)
                write_key(f[name_i], "BUNIT", bunit)
            end

            # Dust continuum parameters
            for i ∈ keys(param_data.dust_continuum)
                for parameter ∈ keys(param_data.dust_continuum[i])
                    data = param_data.dust_continuum[i][parameter]
                    name_i = join(["dust_continuum", i, parameter], "_")
                    if occursin("amp", String(name_i))
                        bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                    elseif occursin("temp", String(name_i))
                        bunit = "Kelvin"
                    end
                    write(f, data; name=name_i)
                    write_key(f[name_i], "BUNIT", bunit)  
                end
            end

            # Power law parameters
            for l ∈ keys(param_data.power_law)
                for parameter ∈ keys(param_data.power_law[l])
                    data = param_data.power_law[l][parameter]
                    name_i = join(["power_law", l, parameter], "_")
                    if occursin("amp", String(name_i))
                        bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                    elseif occursin("index", String(name_i))
                        bunit = "unitless"
                    end
                    write(f, data; name=name_i)
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            if cube_fitter.fit_sil_emission
                # Hot dust parameters
                for parameter ∈ keys(param_data.hot_dust)
                    data = param_data.hot_dust[parameter]
                    name_i = join(["hot_dust", parameter], "_")
                    if occursin("amp", String(name_i))
                        bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                    elseif occursin("temp", String(name_i))
                        bunit = "Kelvin"
                    elseif occursin("frac", String(name_i)) || occursin("tau", String(name_i))
                        bunit = "unitless"
                    elseif occursin("peak", String(name_i))
                        bunit = "um"
                    end
                    write(f, data; name=name_i)
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            # Dust feature (PAH) parameters
            for df ∈ keys(param_data.dust_features)
                for parameter ∈ keys(param_data.dust_features[df])
                    data = param_data.dust_features[df][parameter]
                    name_i = join(["dust_features", df, parameter], "_")
                    if occursin("amp", String(name_i))
                        bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                    elseif occursin("fwhm", String(name_i)) || occursin("mean", String(name_i)) || occursin("eqw", String(name_i))
                        bunit = "um"
                    elseif occursin("flux", String(name_i))
                        bunit = "log10(F / erg s^-1 cm^-2)"
                    elseif occursin("SNR", String(name_i)) || occursin("index", String(name_i)) || occursin("cutoff", String(name_i))
                        bunit = "unitless"
                    end
                    write(f, data; name=name_i)
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            # Absorption feature parameters
            for ab ∈ keys(param_data.abs_features)
                for parameter ∈ keys(param_data.abs_features[ab])
                    data = param_data.abs_features[ab][parameter]
                    name_i = join(["abs_features", ab, parameter], "_")
                    if occursin("tau", String(name_i))
                        bunit = "unitless"
                    elseif occursin("fwhm", String(name_i)) || occursin("mean", String(name_i)) || occursin("eqw", String(name_i))
                        bunit = "um"
                    end
                    write(f, data; name=name_i)
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end
            
            # Extinction parameters
            for parameter ∈ keys(param_data.extinction)
                data = param_data.extinction[parameter]
                name_i = join(["extinction", parameter], "_")
                bunit = "unitless"
                write(f, data; name=name_i)
                write_key(f[name_i], "BUNIT", bunit)  
            end

            # Line parameters
            for line ∈ keys(param_data.lines)
                for parameter ∈ keys(param_data.lines[line])
                    data = param_data.lines[line][parameter]
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
                    write(f, data; name=name_i)
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            if isone(index)
                # chi^2 statistics
                for parameter ∈ keys(param_data.statistics)
                    data = param_maps.statistics[parameter]
                    name_i = join(["statistics", parameter], "_")
                    bunit = "unitless"
                    write(f, data; name=name_i)
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end
        end
    end
end


function write_fits_opt(cube_fitter::CubeFitter, cube_data::NamedTuple, cube_model::CubeModel, param_maps::ParamMaps, 
    param_errs::Vector{<:ParamMaps}; aperture::Union{Vector{PyObject},Nothing}=nothing)

    aperture_keys = []
    aperture_vals = []
    aperture_comments = []
    # If using an aperture, extract its properties 
    if !isnothing(aperture)
        # Get the RA and Dec of the centroid
        sky_aperture = aperture[1].to_sky(cube_fitter.cube.wcs)
        sky_cent = sky_aperture.positions
        ra_cent = format_angle(ha2hms(sky_cent.ra[1]/15); delim=["h","m","s"])
        dec_cent = format_angle(deg2dms(sky_cent.dec[1]); delim=["d","m","s"])

        # Get the name (giving the shape of the aperture: circular, elliptical, or rectangular)
        ap_shape = aperture[1].__class__.__name__

        aperture_keys = ["AP_SHAPE", "AP_RA", "AP_DEC"]
        aperture_vals = Any[ap_shape, ra_cent, dec_cent]
        aperture_comments = ["The shape of the spectrum extraction aperture", "The RA of the aperture",
            "The dec of the aperture"]

        # Get the properties, i.e. radius for circular 
        if ap_shape == "CircularAperture"
            append!(aperture_keys, ["AP_RADIUS"])
            append!(aperture_vals, sky_aperture.r[1])
            append!(aperture_comments, ["Radius of aperture in arcsec"])
        elseif ap_shape == "EllipticalAperture"
            append!(aperture_keys, ["AP_A", "AP_B", "AP_PA"])
            append!(aperture_vals, [sky_aperture.a[1], sky_aperture.b[1], sky_aperture.theta[1]])
            append!(aperture_comments, ["Semimajor axis of aperture in arcsec", 
                "Semiminor axis of aperture in arcsec", "Aperture position angle in rad."])
        elseif ap_shape == "RectangularAperture"
            append!(aperture_keys, ["AP_W", "AP_H", "AP_PA"])
            append!(aperture_vals, [sky_aperture.w[1], sky_aperture.h[1], sky_aperture.theta[1]])
            append!(aperture_comments, ["Width of aperture in arcsec", 
                "Height of aperture in arcsec", "Aperture position angle in rad."])
        end

        # Also append the aperture area
        append!(aperture_keys, ["AP_AR_SR"])
        append!(aperture_vals, [cube_data.area_sr[1]])
        append!(aperture_comments, ["Area of aperture in steradians"])
    end

    # Header information
    if !isnothing(cube_fitter.cube.wcs)
        hdr = FITSHeader(
            Vector{String}(cat(["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "PIXAR_SR", "RA", "DEC", "WCSAXES",
                "CDELT1", "CDELT2", "CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CUNIT1", "CUNIT2", 
                "PC1_1", "PC1_2", "PC2_1", "PC2_2"], aperture_keys, dims=1)),

            cat([cube_fitter.name, cube_fitter.z, cube_fitter.cube.channel, cube_fitter.cube.band, cube_fitter.cube.Ω, 
            cube_fitter.cube.α, cube_fitter.cube.δ, cube_fitter.cube.wcs.wcs.naxis, 
            cube_fitter.cube.wcs.wcs.cdelt[1], cube_fitter.cube.wcs.wcs.cdelt[2], 
            cube_fitter.cube.wcs.wcs.ctype[1], cube_fitter.cube.wcs.wcs.ctype[2], 
            cube_fitter.cube.wcs.wcs.crpix[1], cube_fitter.cube.wcs.wcs.crpix[2], 
            cube_fitter.cube.wcs.wcs.crval[1], cube_fitter.cube.wcs.wcs.crval[2], 
            cube_fitter.cube.wcs.wcs.cunit[1].name, cube_fitter.cube.wcs.wcs.cunit[2].name, 
            cube_fitter.cube.wcs.wcs.pc[1,1], cube_fitter.cube.wcs.wcs.pc[1,2], 
            cube_fitter.cube.wcs.wcs.pc[2,1], cube_fitter.cube.wcs.wcs.pc[2,2]], aperture_vals, dims=1),

            Vector{String}(cat(["Target name", "Target redshift", "MIRI channel", "MIRI band",
            "Solid angle per pixel (rad.)", "Right ascension of target (deg.)", "Declination of target (deg.)",
            "number of World Coordinate System axes", 
            "first axis increment per pixel", "second axis increment per pixel",
            "first axis coordinate type", "second axis coordinate type",
            "axis 1 coordinate of the reference pixel", "axis 2 coordinate of the reference pixel",
            "first axis value at the reference pixel", "second axis value at the reference pixel",
            "first axis units", "second axis units",
            "linear transformation matrix element", "linear transformation matrix element",
            "linear transformation matrix element", "linear transformation matrix element"], aperture_comments, dims=1))
        )
    else
        hdr = FITSHeader(
            Vector{String}(cat(["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "PIXAR_SR", "RA", "DEC"], aperture_keys, dims=1)),

            cat([cube_fitter.name, cube_fitter.z, cube_fitter.cube.channel, cube_fitter.cube.band, cube_fitter.cube.Ω, 
            cube_fitter.cube.α, cube_fitter.cube.δ], aperture_vals, dims=1),

            Vector{String}(cat(["Target name", "Target redshift", "MIRI channel", "MIRI band",
            "Solid angle per pixel (rad.)", "Right ascension of target (deg.)", "Declination of target (deg.)"], 
            aperture_comments, dims=1))
        )
    end

    if cube_fitter.save_full_model
        # Create the 3D intensity model FITS file
        FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_full_model.fits"), "w") do f

            @debug "Writing 3D model FITS HDUs"

            write(f, Vector{Int}(); header=hdr)                                                         # Primary HDU (empty)
            write(f, Float32.(cube_data.I ./ (1 .+ cube_fitter.z)); name="DATA")                        # Raw data 
            write(f, Float32.(cube_data.σ ./ (1 .+ cube_fitter.z)); name="ERROR")                       # Error in the raw data
            write(f, cube_model.model; name="MODEL")                                                    # Full intensity model
            for i ∈ 1:size(cube_model.stellar, 4)
                write(f, cube_model.stellar[:, :, :, i]; name="STELLAR_POPULATION_$i")                  # Stellar population models
            end
            for (k, line) ∈ enumerate(cube_fitter.lines.names)
                write(f, cube_model.lines[:, :, :, k]; name="$line")                                    # Emission line profiles
            end
            write(f, cube_model.attenuation_stars; name="ATTENUATION_STARS")                            # Starlight attenuation model
            write(f, cube_model.attenuation_gas; name="ATTENUATION_GAS")                                # Gas attenuation model
            write(f, ["wave"], [cube_data.λ .* (1 .+ cube_fitter.z)],                                   # wavelength vector
                hdutype=TableHDU, name="WAVELENGTH", units=Dict(:wave => "um"))

            # Insert physical units into the headers of each HDU -> MegaJansky per steradian for all except
            # the extinction profile, which is a multiplicative constant
            write_key(f["DATA"], "BUNIT", "erg/s/cm^2/ang/sr")
            write_key(f["ERROR"], "BUNIT", "erg/s/cm^2/ang/sr")
            write_key(f["MODEL"], "BUNIT", "erg/s/cm^2/ang/sr")
            for i ∈ 1:size(cube_model.stellar, 4)
                write_key(f["STELLAR_POPULATION_$i"], "BUNIT", "erg/s/cm^2/ang/sr")
            end
            write_key(f["ATTENUATION"], "BUNIT", "unitless")
        end
    end

    # Create the 2D parameter map FITS file for the parameters and the errors
    for (index, param_data) ∈ enumerate([param_maps, param_errs[1], param_errs[2]])

        FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_parameter_" * 
            ("maps", "errs_low", "errs_upp")[index] * ".fits"), "w") do f

            @debug "Writing 2D parameter map FITS HDUs"

            write(f, Vector{Int}(), header=hdr)  # Primary HDU (empty)

            # Stellar population parameters
            for i ∈ keys(param_data.stellar_populations)
                for parameter ∈ keys(param_data.stellar_populations[i])
                    data = param_data.stellar_populations[i][parameter]
                    name_i = join(["stellar_populations", i, parameter], "_")
                    if occursin("mass", String(name_i))
                        bunit = "log10(M/Msun)"
                    elseif occursin("age", String(name_i))
                        bunit = "Gyr"
                    elseif occursin("metallicity", String(name_i))
                        bunit = "[M/H]"
                    end
                    write(f, data; name=name_i)
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            # Stellar kinematics parameters
            for parameter ∈ keys(param_data.stellar_kinematics)
                data = param_data.stellar_kinematics[parameter]
                name_i = join(["stellar_kinematics", parameter], "_")
                bunit = "km/s"
                write(f, data; name=name_i)
                write_key(f[name_i], "BUNIT", bunit)
            end

            # Attenuation parameters
            for parameter ∈ keys(param_data.attenuation)
                data = param_data.attenuation[parameter]
                name_i = join(["attenuation", parameter], "_")
                if occursin("E_BV", String(name_i))
                    bunit = "mag"
                else
                    bunit = "unitless"
                end
                write(f, data; name=name_i)
                write_key(f[name_i], "BUNIT", bunit)
            end

            # Line parameters
            for line ∈ keys(param_data.lines)
                for parameter ∈ keys(param_data.lines[line])
                    data = param_data.lines[line][parameter]
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
                    write(f, data; name=name_i)
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            if isone(index)
                # chi^2 statistics
                for parameter ∈ keys(param_data.statistics)
                    data = param_maps.statistics[parameter]
                    name_i = join(["statistics", parameter], "_")
                    bunit = "unitless"
                    write(f, data; name=name_i)
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end
        end
    end
end

