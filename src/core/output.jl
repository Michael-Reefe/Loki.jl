############################## OUTPUT / SAVING FUNCTIONS ####################################

const ascii_lowercase = "abcdefghijklmnopqrstuvwxyz"

function sort_line_components!(cube_fitter::CubeFitter, param_maps::ParamMaps, param_errs::Vector{<:ParamMaps}, 
    index::CartesianIndex, cube_data::NamedTuple)

    # Reorder line parameters sorted by the FWHM velocity
    if !isnothing(cube_fitter.sort_line_components)
        for k ∈ 1:cube_fitter.n_lines
            # n_prof = sum([!isnothing(cube_fitter.lines.profiles[k, j]) for j ∈ 1:cube_fitter.n_comps])
            n_prof = param_maps.lines_comp[cube_fitter.lines.names[k]][:n_comps][index]
            n_prof = isfinite(n_prof) ? Int(n_prof) : n_prof

            if n_prof > 1

                ln = cube_fitter.lines.names[k]
                # Collect parameters for each line component
                amps = zeros(n_prof)
                amp_errs_lo = zeros(n_prof)
                amp_errs_hi = zeros(n_prof)
                voffs = zeros(n_prof)
                voff_errs_lo = zeros(n_prof)
                voff_errs_hi = zeros(n_prof)
                voff_indivs = zeros(n_prof)
                voff_indiv_errs_lo = zeros(n_prof)
                voff_indiv_errs_hi = zeros(n_prof)
                fwhms = zeros(n_prof)
                fwhm_errs_lo = zeros(n_prof)
                fwhm_errs_hi = zeros(n_prof)
                h3s = zeros(n_prof)
                h3_errs_lo = zeros(n_prof)
                h3_errs_hi = zeros(n_prof)
                h4s = zeros(n_prof)
                h4_errs_lo = zeros(n_prof)
                h4_errs_hi = zeros(n_prof)
                etas = zeros(n_prof)
                eta_errs_lo = zeros(n_prof)
                eta_errs_hi = zeros(n_prof)
                fluxes = zeros(n_prof)
                flux_errs_lo = zeros(n_prof)
                flux_errs_hi = zeros(n_prof)
                snrs = zeros(n_prof)

                for j ∈ 1:n_prof
                    lnj = Symbol(ln, "_$j")

                    amps[j] = param_maps.lines[lnj][:amp][index]
                    amp_errs_lo[j] = param_errs[1].lines[lnj][:amp][index]
                    amp_errs_hi[j] = param_errs[2].lines[lnj][:amp][index]
                    voffs[j] = param_maps.lines[lnj][:voff][index]
                    voff_errs_lo[j] = param_errs[1].lines[lnj][:voff][index]
                    voff_errs_hi[j] = param_errs[2].lines[lnj][:voff][index]
                    fwhms[j] = param_maps.lines[lnj][:fwhm][index]
                    fwhm_errs_lo[j] = param_errs[1].lines[lnj][:fwhm][index]
                    fwhm_errs_hi[j] = param_errs[2].lines[lnj][:fwhm][index]
                    if !isnothing(cube_fitter.lines.tied_voff[k, j]) && cube_fitter.flexible_wavesol && isone(j)
                        voff_indivs[j] = param_maps.lines[lnj][:voff_indiv][index]
                        voff_indiv_errs_lo[j] = param_errs[1].lines[lnj][:voff_indiv][index]
                        voff_indiv_errs_hi[j] = param_errs[2].lines[lnj][:voff_indiv][index]
                    end
                    if cube_fitter.lines.profiles[k, j] == :GaussHermite
                        h3s[j] = param_maps.lines[lnj][:h3][index]
                        h3_errs_lo[j] = param_errs[1].lines[lnj][:h3][index]
                        h3_errs_hi[j] = param_errs[2].lines[lnj][:h3][index]
                        h4s[j] = param_maps.lines[lnj][:h4][index]
                        h4_errs_lo[j] = param_errs[1].lines[lnj][:h4][index]
                        h4_errs_hi[j] = param_errs[2].lines[lnj][:h4][index]
                    end
                    if cube_fitter.lines.profiles[k, j] == :Voigt
                        etas[j] = param_maps.lines[lnj][:mixing][index]
                        eta_errs_lo[j] = param_errs[1].lines[lnj][:mixing][index]
                        eta_errs_hi[j] = param_errs[2].lines[lnj][:mixing][index]
                    end
                    fluxes[j] = param_maps.lines[lnj][:flux][index]
                    flux_errs_lo[j] = param_errs[1].lines[lnj][:flux][index]
                    flux_errs_hi[j] = param_errs[2].lines[lnj][:flux][index]
                    snrs[j] = param_maps.lines[lnj][:SNR][index]
                end     

                # Check the SNRs of each component
                bad = findall(snrs .< cube_fitter.map_snr_thresh)

                if cube_fitter.sort_line_components == :flux
                    sort_quantity = copy(fluxes)
                elseif cube_fitter.sort_line_components == :amp
                    sort_quantity = copy(amps)
                elseif cube_fitter.sort_line_components == :fwhm
                    sort_quantity = copy(fwhms)
                elseif cube_fitter.sort_line_components == :voff
                    sort_quantity = abs.(voffs)
                else
                    error("Unrecognized sorting quantity: $(cube_fitter.sort_line_components)")
                end
                sort_quantity[bad] .= NaN

                # Sort by the relevant sorting quantity (NaNs are always placed at the end)
                #  1 = sort in increasing order
                # -1 = sort in decreasing order
                if cube_fitter.lines.sort_order[k] == 1
                    ss = sortperm(sort_quantity)
                elseif cube_fitter.lines.sort_order[k] == -1
                    # Cant just reverse because then NaNs would be placed at the beginning
                    n_inf = sum(.~isfinite.(sort_quantity))
                    ss = [sortperm(sort_quantity, rev=true)[n_inf+1:end]; findall(.~isfinite.(sort_quantity))]
                else
                    error("Unrecognized sort order: $(cube_fitter.lines.sort_order[k]) (must be 1 or -1)")
                end

                # Reassign the parameters in this order
                for i ∈ eachindex(ss)
                    lni = Symbol(ln, "_$i")
                    nan_arr = ones(size(cube_data.I)[1:2]) .* NaN

                    # We need to add keys if the components have diferent types of profiles (not recommended)
                    if !isnothing(cube_fitter.lines.tied_voff[k, 1]) && cube_fitter.flexible_wavesol && !haskey(param_maps.lines[lni], :voff_indiv)
                        param_maps.lines[lni][:voff_indiv] = copy(nan_arr)
                        param_errs[1].lines[lni][:voff_indiv] = copy(nan_arr)
                        param_errs[2].lines[lni][:voff_indiv] = copy(nan_arr)
                    end
                    if cube_fitter.lines.profiles[k, i] == :GaussHermite && !haskey(param_maps.lines[lni], :h3)
                        param_maps.lines[lni][:h3] = copy(nan_arr)
                        param_errs[1].lines[lni][:h3] = copy(nan_arr)
                        param_errs[2].lines[lni][:h3] = copy(nan_arr)
                        param_maps.lines[lni][:h4] = copy(nan_arr)
                        param_errs[1].lines[lni][:h4] = copy(nan_arr)
                        param_errs[2].lines[lni][:h4] = copy(nan_arr)
                    end
                    if cube_fitter.lines.profiles[k, i] == :Voigt && !haskey(param_maps.lines[lni], :mixing)
                        param_maps.lines[lni][:mixing] = copy(nan_arr)
                        param_errs[1].lines[lni][:mixing] = copy(nan_arr)
                        param_errs[2].lines[lni][:mixing] = copy(nan_arr)
                    end

                    param_maps.lines[lni][:amp][index] = amps[ss[i]]
                    param_errs[1].lines[lni][:amp][index] = amp_errs_lo[ss[i]]
                    param_errs[2].lines[lni][:amp][index] = amp_errs_hi[ss[i]]
                    param_maps.lines[lni][:voff][index] = voffs[ss[i]]
                    param_errs[1].lines[lni][:voff][index] = voff_errs_lo[ss[i]]
                    param_errs[2].lines[lni][:voff][index] = voff_errs_hi[ss[i]]
                    param_maps.lines[lni][:fwhm][index] = fwhms[ss[i]]
                    param_errs[1].lines[lni][:fwhm][index] = fwhm_errs_lo[ss[i]]
                    param_errs[2].lines[lni][:fwhm][index] = fwhm_errs_hi[ss[i]]
                    if !isnothing(cube_fitter.lines.tied_voff[k, 1]) && cube_fitter.flexible_wavesol
                        param_maps.lines[lni][:voff_indiv][index] = voff_indivs[ss[i]]
                        param_errs[1].lines[lni][:voff_indiv][index] = voff_indiv_errs_lo[ss[i]]
                        param_errs[2].lines[lni][:voff_indiv][index] = voff_indiv_errs_hi[ss[i]]
                    end
                    if cube_fitter.lines.profiles[k, i] == :GaussHermite
                        param_maps.lines[lni][:h3][index] = h3s[ss[i]]
                        param_errs[1].lines[lni][:h3][index] = h3_errs_lo[ss[i]]
                        param_errs[2].lines[lni][:h3][index] = h3_errs_hi[ss[i]]
                        param_maps.lines[lni][:h4][index] = h4s[ss[i]]
                        param_errs[1].lines[lni][:h4][index] = h4_errs_lo[ss[i]]
                        param_errs[2].lines[lni][:h4][index] = h4_errs_hi[ss[i]]
                    end
                    if cube_fitter.lines.profiles[k, i] == :Voigt
                        param_maps.lines[lni][:mixing][index] = etas[ss[i]]
                        param_errs[1].lines[lni][:mixing][index] = eta_errs_lo[ss[i]]
                        param_errs[2].lines[lni][:mixing][index] = eta_errs_hi[ss[i]]
                    end
                    param_maps.lines[lni][:flux][index] = fluxes[ss[i]]
                    param_errs[1].lines[lni][:flux][index] = flux_errs_lo[ss[i]]
                    param_errs[2].lines[lni][:flux][index] = flux_errs_hi[ss[i]]
                    param_maps.lines[lni][:SNR][index] = snrs[ss[i]]
                end
            end
        end
    end

end


"""
    assign_outputs(out_params, out_errs, cube_fitter, cube_data, spaxels, z[, aperture])

Create ParamMaps objects for the parameter values and errors, and a CubeModel object for the full model, and
fill them with the maximum likelihood values and errors given by out_params and out_errs over each spaxel in
spaxels.
"""
assign_outputs(out_params::Array{<:Real}, out_errs::Array{<:Real}, cube_fitter::CubeFitter, 
    cube_data::NamedTuple, z::Real, aperture::Bool=false) = 
    cube_fitter.spectral_region == :MIR ? 
        assign_outputs_mir(out_params, out_errs, cube_fitter, cube_data, z, aperture) : 
        assign_outputs_opt(out_params, out_errs, cube_fitter, cube_data, z, aperture)


# MIR implementation of the assign_outputs function
function assign_outputs_mir(out_params::Array{<:Real}, out_errs::Array{<:Real}, cube_fitter::CubeFitter,
    cube_data::NamedTuple, z::Real, aperture::Bool=false)

    # Create the CubeModel and ParamMaps structs to be filled in
    cube_model = generate_cubemodel(cube_fitter, aperture)
    param_maps, param_errs = generate_parammaps(cube_fitter, aperture)

    # Unpack relative flags for lines
    rel_amp, rel_voff, rel_fwhm = cube_fitter.relative_flags

    # Loop over each spaxel and fill in the associated fitting parameters into the ParamMaps and CubeModel
    # I know this is long and ugly and looks stupid but it works for now and I'll make it pretty later
    spaxels = CartesianIndices(size(out_params)[1:2])
    prog = Progress(length(spaxels); showspeed=true)
    @simd for index ∈ spaxels

        # Get the normalization to un-normalized the fitted parameters
        data_index = !isnothing(cube_fitter.cube.voronoi_bins) ? cube_fitter.cube.voronoi_bins[index] : index
        if Tuple(data_index)[1] > 0
            N = Float64(abs(nanmaximum(cube_data.I[data_index, :])))
            N = N ≠ 0. ? N : 1.
        else
            N = 1.
        end

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
            param_maps.power_law[j][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]*N*(1+z))-17 : -Inf
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

        # Template amplitudes
        for (q, tp) ∈ enumerate(cube_fitter.template_names)
            param_maps.templates[tp][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]) : -Inf
            param_errs[1].templates[tp][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 1] / (log(10) * out_params[index, pᵢ]) : NaN
            param_errs[2].templates[tp][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 2] / (log(10) * out_params[index, pᵢ]) : NaN
            pᵢ += 1
        end

        # Dust feature log(amplitude), mean, FWHM
        for (k, df) ∈ enumerate(cube_fitter.dust_features.names)
            param_maps.dust_features[df][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]*(1+z)*N)-17 : -Inf
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
                cube_fitter.fit_sil_emission, false, cube_data.templates[index, :, :], true)
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
                    if isone(j) || !rel_amp
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
                    if isone(j) || !rel_voff
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
                    if isone(j) || !rel_fwhm
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
                cube_fitter.lines, cube_fitter.flexible_wavesol, comps_c["extinction"], lsf_interp_func, cube_fitter.relative_flags, true)

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
                    param_maps.lines[ln][:amp][index] = amp_norm > 0 ? log10(amp_norm * N * (1+z))-17 : -Inf
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

            # Get composite line parameters
            ln = cube_fitter.lines.names[k]
            param_maps.lines_comp[ln][:n_comps][index] = out_params[index, pᵢ]
            param_errs[1].lines_comp[ln][:n_comps][index] = out_errs[index, pᵢ, 1]
            param_errs[2].lines_comp[ln][:n_comps][index] = out_errs[index, pᵢ, 2]
            pᵢ += 1
        end

        # Sort the parameters for multicomponent lines
        sort_line_components!(cube_fitter, param_maps, param_errs, index, cube_data)

        # Reduced χ^2 = χ^2 / dof
        param_maps.statistics[:chi2][index] = out_params[index, pᵢ]
        param_maps.statistics[:dof][index] = out_params[index, pᵢ+1]

        if cube_fitter.save_full_model
            # Set 3D model cube outputs, shifted back to the observed frame
            # Remember the wavelength axis is the first axis here to increase efficiency
            cube_model.model[:, index] .= I_model .* (1 .+ z)
            cube_model.stellar[:, index] .= comps["stellar"] .* (1 .+ z)
            for i ∈ 1:cube_fitter.n_dust_cont
                cube_model.dust_continuum[:, index, i] .= comps["dust_cont_$i"] .* (1 .+ z)
            end
            for l ∈ 1:cube_fitter.n_power_law
                cube_model.power_law[:, index, l] .= comps["power_law_$l"] .* (1 .+ z)
            end
            for j ∈ 1:cube_fitter.n_dust_feat
                cube_model.dust_features[:, index, j] .= comps["dust_feat_$j"] .* (1 .+ z)
            end
            for m ∈ 1:cube_fitter.n_abs_feat
                cube_model.abs_features[:, index, m] .= comps["abs_feat_$m"]
            end
            if cube_fitter.fit_sil_emission
                cube_model.hot_dust[:, index] .= comps["hot_dust"] .* (1 .+ z)
            end
            for q ∈ 1:cube_fitter.n_templates
                cube_model.templates[:, index, q] .= comps["templates_$q"] .* (1 .+ z)
            end
            for j ∈ 1:cube_fitter.n_comps
                for k ∈ 1:cube_fitter.n_lines
                    if !isnothing(cube_fitter.lines.profiles[k, j])
                        cube_model.lines[:, index, k] .+= comps["line_$(k)_$(j)"] .* (1 .+ z)
                    end
                end
            end
            cube_model.extinction[:, index] .= comps["extinction"]
            cube_model.abs_ice[:, index] .= comps["abs_ice"]
            cube_model.abs_ch[:, index] .= comps["abs_ch"]
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


# Optical implementation of the assign_outputs function
function assign_outputs_opt(out_params::Array{<:Real}, out_errs::Array{<:Real}, cube_fitter::CubeFitter,
    cube_data::NamedTuple, z::Real, aperture::Bool=false)

    # Create the CubeModel and ParamMaps structs to be filled in
    cube_model = generate_cubemodel(cube_fitter, aperture)
    param_maps, param_errs = generate_parammaps(cube_fitter, aperture)

    # Unpack relative flags for lines
    rel_amp, rel_voff, rel_fwhm = cube_fitter.relative_flags

    # Loop over each spaxel and fill in the associated fitting parameters into the ParamMaps and CubeModel
    # I know this is long and ugly and looks stupid but it works for now and I'll make it pretty later
    spaxels = CartesianIndices(size(out_params)[1:2])
    prog = Progress(length(spaxels); showspeed=true)
    @simd for index ∈ spaxels

        # Get the normalization to un-normalized the fitted parameters
        data_index = !isnothing(cube_fitter.cube.voronoi_bins) ? cube_fitter.cube.voronoi_bins[index] : index
        if Tuple(data_index)[1] > 0
            N = Float64(abs(nanmaximum(cube_data.I[data_index, :])))
            N = N ≠ 0. ? N : 1.
        else
            N = 1.
        end

        # Set the 2D parameter map outputs

        # Conversion factor from MJy sr^-1 to erg s^-1 cm^-2 Hz^-1 sr^-1 = 10^6 * 10^-23 = 10^-17
        # So, log10(A * 1e-17) = log10(A) - 17

        # Stellar continuum amplitude, temp
        # Convert back to observed-frame amplitudes by multiplying by 1+z
        Ω_med = median(cube_fitter.cube.Ω)
        pᵢ = 1
        for i ∈ 1:cube_fitter.n_ssps
            # Un-normalize the amplitudes by applying the normalization factors used in the fitting routines
            # (median of the SSP template, followed by normalization N)
            ssp_med = median([cube_fitter.ssp_templates[j](out_params[index, pᵢ+1], out_params[index, pᵢ+2]) for j in eachindex(cube_fitter.ssp_λ)])
            param_maps.stellar_populations[i][:mass][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ] * Ω_med / ssp_med * N / (1+z)) : -Inf
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

        # Attenuation 
        param_maps.attenuation[:E_BV][index] = out_params[index, pᵢ]
        param_errs[1].attenuation[:E_BV][index] = out_errs[index, pᵢ, 1] 
        param_errs[2].attenuation[:E_BV][index] = out_errs[index, pᵢ, 2] 
        param_maps.attenuation[:E_BV_factor][index] = out_params[index, pᵢ+1]
        param_errs[1].attenuation[:E_BV_factor][index] = out_errs[index, pᵢ+1, 1] 
        param_errs[2].attenuation[:E_BV_factor][index] = out_errs[index, pᵢ+1, 2] 
        pᵢ += 2
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

        # Fe II emission
        if cube_fitter.fit_opt_na_feii
            λ0_na_feii = 1e4 * (1+z) * cube_data.λ[argmax(convolve_losvd(cube_fitter.feii_templates_fft[:, 1], cube_fitter.vsyst_feii, 
                out_params[index, pᵢ+1], out_params[index, pᵢ+2], cube_fitter.vres, length(cube_data.λ), temp_fft=true, npad_in=cube_fitter.npad_feii))]
            param_maps.feii[:na_amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ] * N * λ0_na_feii^2/(C_KMS * 1e13) / (1+z)) : -Inf
            param_errs[1].feii[:na_amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 1] / (log(10) * out_params[index, pᵢ]) : NaN
            param_errs[2].feii[:na_amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 2] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.feii[:na_vel][index] = out_params[index, pᵢ+1]
            param_errs[1].feii[:na_vel][index] = out_errs[index, pᵢ+1, 1]
            param_errs[2].feii[:na_vel][index] = out_errs[index, pᵢ+1, 2]
            param_maps.feii[:na_vdisp][index] = out_params[index, pᵢ+2]
            param_errs[1].feii[:na_vdisp][index] = out_errs[index, pᵢ+2, 1]
            param_errs[2].feii[:na_vdisp][index] = out_errs[index, pᵢ+2, 2]
            pᵢ += 3
        end
        if cube_fitter.fit_opt_br_feii
            λ0_br_feii = 1e4 * (1+z) * cube_data.λ[argmax(convolve_losvd(cube_fitter.feii_templates_fft[:, 2], cube_fitter.vsyst_feii,
                out_params[index, pᵢ+1], out_params[index, pᵢ+2], cube_fitter.vres, length(cube_data.λ), temp_fft=true, npad_in=cube_fitter.npad_feii))]
            param_maps.feii[:br_amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ] * N * λ0_br_feii^2/(C_KMS * 1e13) / (1+z)) : -Inf
            param_errs[1].feii[:br_amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 1] / (log(10) * out_params[index, pᵢ]) : NaN
            param_errs[2].feii[:br_amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 2] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.feii[:br_vel][index] = out_params[index, pᵢ+1]
            param_errs[1].feii[:br_vel][index] = out_errs[index, pᵢ+1, 1]
            param_errs[2].feii[:br_vel][index] = out_errs[index, pᵢ+1, 2]
            param_maps.feii[:br_vdisp][index] = out_params[index, pᵢ+2]
            param_errs[1].feii[:br_vdisp][index] = out_errs[index, pᵢ+2, 1]
            param_errs[2].feii[:br_vdisp][index] = out_errs[index, pᵢ+2, 2]
            pᵢ += 3
        end

        # Power laws
        for j ∈ 1:cube_fitter.n_power_law
            λ0_pl = 5100.0 * (1+z)
            param_maps.power_law[j][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ] * N * λ0_pl^2/(C_KMS * 1e13) / (1+z)) : -Inf
            param_errs[1].power_law[j][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 1] / (log(10) * out_params[index, pᵢ]) : NaN
            param_errs[2].power_law[j][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ, 2] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.power_law[:index][index] = out_params[index, pᵢ+1]
            param_errs[1].power_law[:index][index] = out_errs[index, pᵢ+1, 1]
            param_errs[2].power_law[:index][index] = out_errs[index, pᵢ+1, 2]
            pᵢ += 2
        end

        if cube_fitter.save_full_model
            # End of continuum parameters: recreate the continuum model
            I_cont, comps_c = model_continuum(cube_fitter.cube.λ, out_params[index, 1:pᵢ-1], N, cube_fitter.vres, cube_fitter.vsyst_ssp,
                cube_fitter.vsyst_feii, cube_fitter.npad_feii, cube_fitter.n_ssps, cube_fitter.ssp_λ, cube_fitter.ssp_templates, 
                cube_fitter.feii_templates_fft, cube_fitter.n_power_law, cube_fitter.fit_uv_bump, cube_fitter.fit_covering_frac, 
                cube_fitter.fit_opt_na_feii, cube_fitter.fit_opt_br_feii, cube_fitter.extinction_curve, true)
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
                    if isone(j) || !rel_amp
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
                    if isone(j) || !rel_voff
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
                    if isone(j) || !rel_fwhm
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
                cube_fitter.lines, cube_fitter.flexible_wavesol, comps_c["attenuation_gas"], lsf_interp_func, cube_fitter.relative_flags, true)

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

        for k ∈ 1:cube_fitter.n_lines
            for j ∈ 1:cube_fitter.n_comps
                if !isnothing(cube_fitter.lines.profiles[k, j])

                    ln = Symbol(cube_fitter.lines.names[k], "_$(j)")
                    λ0 = cube_fitter.lines.λ₀[k] * (1 + param_maps.lines[ln][:voff][index]/C_KMS) * 1e4 * (1+z)

                    # Convert amplitudes to the correct units, then take the log
                    amp_norm = param_maps.lines[ln][:amp][index]
                    amp_norm_err = [param_errs[1].lines[ln][:amp][index], param_errs[2].lines[ln][:amp][index]]
                    # Convert amplitude to erg/s/cm^2/Hz/sr to match with the MIR 
                    param_maps.lines[ln][:amp][index] = amp_norm > 0 ? log10(amp_norm * N * λ0^2/(C_KMS * 1e13) / (1+z)) : -Inf
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

            # Get composite line parameters
            ln = cube_fitter.lines.names[k]
            param_maps.lines_comp[ln][:n_comps][index] = isfinite(out_params[index, pᵢ]) ? out_params[index, pᵢ] : 0.
            param_errs[1].lines_comp[ln][:n_comps][index] = out_errs[index, pᵢ, 1]
            param_errs[2].lines_comp[ln][:n_comps][index] = out_errs[index, pᵢ, 2]
            pᵢ += 1
        end

        # Sort the parameters for multicomponent lines
        sort_line_components!(cube_fitter, param_maps, param_errs, index, cube_data)

        # Reduced χ^2 = χ^2 / dof
        param_maps.statistics[:chi2][index] = out_params[index, pᵢ]
        param_maps.statistics[:dof][index] = out_params[index, pᵢ+1]

        if cube_fitter.save_full_model
            # Set 3D model cube outputs, shifted back to the observed frame
            # Remember the wavelength axis is the first axis here to increase efficiency
            cube_model.model[:, index] .= I_model ./ (1 .+ z)
            for i ∈ 1:cube_fitter.n_ssps
                cube_model.stellar[:, index, i] .= comps["SSP_$i"] ./ (1 .+ z)
            end
            cube_model.attenuation_stars[:, index] .= comps["attenuation_stars"]
            cube_model.attenuation_gas[:, index] .= comps["attenuation_gas"]
            if cube_fitter.fit_opt_na_feii
                cube_model.na_feii[:, index] .= comps["na_feii"]
            end
            if cube_fitter.fit_opt_br_feii
                cube_model.br_feii[:, index] .= comps["br_feii"]
            end
            for l ∈ 1:cube_fitter.n_power_law
                cube_model.power_law[:, index] .= comps["power_law_$l"]
            end

            for j ∈ 1:cube_fitter.n_comps
                for k ∈ 1:cube_fitter.n_lines
                    if !isnothing(cube_fitter.lines.profiles[k, j])
                        cube_model.lines[:, index, k] .+= comps["line_$(k)_$(j)"] ./ (1 .+ z)
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


# I know this is ugly but I couldn't figure out a better way to do it lmao
function paramstring_to_latex(name_i, cosmo)
    if occursin("amp", name_i)
        if occursin("stellar_continuum", name_i)
            bunit = L"$\log_{10}(A_{*})$" # normalized
        elseif occursin("dust_continuum", name_i)
            bunit = L"$\log_{10}(A_{\rm dust})$" # normalized
        elseif occursin("power_law", name_i)
            bunit = L"$\log_{10}(A_{\rm pl} / $ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$)"
        elseif occursin("hot_dust", name_i)
            bunit = L"$\log_{10}(A_{\rm sil})$" # normalized
        elseif occursin("template", name_i)
            bunit = L"$\log_{10}(A_{\rm template})$"
        else
            bunit = L"$\log_{10}(I / $ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1})$"
        end
    elseif occursin("temp", name_i)
        bunit = L"$T$ (K)"
    elseif occursin("fwhm", name_i) && (occursin("PAH", name_i) || occursin("abs", name_i))
        bunit = L"FWHM ($\mu$m)"
    elseif occursin("fwhm", name_i) && !occursin("PAH", name_i) && !occursin("abs", name_i)
        bunit = L"FWHM (km s$^{-1}$)"
    elseif occursin("mean", name_i) || occursin("peak", name_i)
        bunit = L"$\mu$ ($\mu$m)"
    elseif occursin("voff", name_i)
        bunit = L"$v_{\rm off}$ (km s$^{-1}$)"
    elseif occursin("SNR", name_i)
        bunit = L"$S/N$"
    elseif occursin("n_comps", name_i)
        bunit = L"$n_{\rm comp}$"
    elseif occursin("tau", name_i)
        if occursin("warm", name_i)
            bunit = L"$\tau_{\rm warm}$"
        elseif occursin("cold", name_i)
            bunit = L"$\tau_{\rm cold}$"
        elseif occursin("ice", name_i)
            bunit = L"$\tau_{\rm ice}$"
        elseif occursin("ch", name_i)
            bunit = L"$\tau_{\rm CH}$"
        elseif occursin("9_7", name_i)
            bunit = L"$\tau_{9.7}$"
        else
            bunit = L"$\tau$"
        end
    elseif occursin("flux", name_i)
        bunit = L"$\log_{10}(F /$ erg s$^{-1}$ cm$^{-2}$)"
    elseif occursin("eqw", name_i)
        bunit = L"$W_{\rm eq}$ ($\mu$m)"
    elseif occursin("chi2", name_i)
        bunit = L"$\tilde{\chi}^2$"
    elseif occursin("dof", name_i)
        bunit = "d.o.f."
    elseif occursin("h3", name_i)
        bunit = L"$h_3$"
    elseif occursin("h4", name_i)
        bunit = L"$h_4$"
    elseif occursin("mixing", name_i)
        bunit = L"$\eta$"
    elseif occursin("beta", name_i)
        bunit = L"$\beta$"
    elseif occursin("frac", name_i)
        bunit = L"$C_f$"
    elseif occursin("index", name_i) && occursin("PAH", name_i)
        bunit = L"$m$"
    elseif occursin("index", name_i)
        bunit = L"$\alpha$"
    elseif occursin("cutoff", name_i)
        bunit = L"$\nu$"
    elseif occursin("mass", name_i)
        bunit = cosmo.h ≈ 1.0 ? L"$\log_{10}(Mh^2 / M_{\odot})$" : L"$\log_{10}(M / M_{\odot})$"
    elseif occursin("age", name_i)
        bunit = L"$t$ (Gyr)"
    elseif occursin("metallicity", name_i)
        bunit = L"[M$/$H]"
    elseif occursin("vel", name_i) && occursin("stellar", name_i)
        bunit = L"$v_*$ (km s$^{-1}$)"
    elseif occursin("vel", name_i)
        bunit = L"$v$ (km s$^{-1}$)"
    elseif occursin("vdisp", name_i) && occursin("stellar", name_i)
        bunit = L"$\sigma_*$ (km s$^{-1}$)"
    elseif occursin("vdisp", name_i)
        bunit = L"$\sigma$ (km s$^{-1}$)"
    elseif occursin("E_BV_factor", name_i)
        bunit = L"$E(B-V)_{\rm stars}/E(B-V)_{\rm gas}$"
    elseif occursin("E_BV", name_i)
        bunit = L"$E(B-V)_{\rm gas}$"
    elseif occursin("delta_UV", name_i)
        bunit = L"$\delta$"
    end
    return bunit
end


"""
    plot_parameter_map(data, name_i, save_path, Ω, z, psf_fwhm, cosmo, python_wcs; 
        [snr_filter, snr_thresh, cmap, line_latex])

Plotting function for 2D parameter maps which are output by `fit_cube!`

# Arguments
- `data::Matrix{Float64}`: The 2D array of data to be plotted
- `name_i::String`: The name of the individual parameter being plotted, i.e. "dust_features_PAH_5.24_amp"
- `save_path::String`: The file path to save the plot to.
- `Ω::Float64`: The solid angle subtended by each pixel, in steradians (used for angular scalebar)
- `z::Float64`: The redshift of the object (used for physical scalebar)
- `psf_fwhm::Float64`: The FWHM of the point-spread function in arcseconds (used to add a circular patch with this size)
- `cosmo::Cosmology.AbstractCosmology`: The cosmology to use to calculate distance for the physical scalebar
- `wcs::WCSTransform`: The WCS object used to project the maps onto RA/Dec space
- `snr_filter::Union{Nothing,Matrix{Float64}}=nothing`: A 2D array of S/N values to
    be used to filter out certain spaxels from being plotted - must be the same size as `data` to filter
- `snr_thresh::Float64=3.`: The S/N threshold below which to cut out any spaxels using the values in snr_filter
- `cmap::Symbol=:cubehelix`: The colormap used in the plot, defaults to the cubehelix map
- `line_latex::Union{String,Nothing}=nothing`: LaTeX-formatted label for the emission line to be added to the top-right corner.
- `disable_axes::Bool=true`: If true, the x/y or RA/Dec axes are turned off, and instead an angular label is added to the scale bar.
- `disable_colorbar::Bool=false`: If true, turns off the color scale.
- `modify_ax::Union{Tuple{PyObject,PyObject},Nothing}=nothing`: If one wishes to apply this plotting routine on a pre-existing axis object,
input the figure and axis objects here as a tuple, and the modified axis object will be returned.
- `colorscale_limits::Union{Tuple{<:Real,<:Real},Nothing}=nothing`: If specified, gives lower and upper limits for the color scale. Otherwise,
they will be determined automatically from the data.
- `custom_bunit::Union{LaTeXString,Nothing}=nothing`: If provided, overwrites the colorbar axis label. Otherwise the label is determined
automtically using the `name_i` parameter.
"""
function plot_parameter_map(data::Matrix{Float64}, name_i::String, save_path::String, Ω::Float64, z::Float64, psf_fwhm::Float64,
    cosmo::Cosmology.AbstractCosmology, wcs::Union{WCSTransform,Nothing}; snr_filter::Union{Nothing,Matrix{Float64}}=nothing, 
    snr_thresh::Float64=3., abs_thresh::Union{Float64,Nothing}=nothing, cmap=py_colormap.cubehelix, line_latex::Union{String,Nothing}=nothing, 
    marker::Union{Vector{<:Real},Nothing}=nothing, disable_axes::Bool=true, disable_colorbar::Bool=false, modify_ax=nothing, colorscale_limits=nothing, 
    custom_bunit::Union{LaTeXString,Nothing}=nothing)

    bunit = paramstring_to_latex(name_i, cosmo)
    # Overwrite with input if provided
    if !isnothing(custom_bunit)
        bunit = custom_bunit
    end
    @debug "Plotting 2D map of $name_i with units $bunit"

    filtered = copy(data)
    # Convert Infs into NaNs
    filtered[.!isfinite.(filtered)] .= NaN
    # Filter out low SNR points
    filt = falses(size(filtered))
    if !isnothing(snr_filter) && !isnothing(abs_thresh)
        filt = (snr_filter .≤ snr_thresh) .& (filtered .< abs_thresh)
    elseif !isnothing(abs_thresh)
        filt = filtered .< abs_thresh
    elseif !isnothing(snr_filter)
        filt = snr_filter .≤ snr_thresh
    end
    filtered[filt] .= NaN
    @debug "Performing SNR filtering, $(sum(isfinite.(filtered)))/$(length(filtered)) passed"
    # filter out insane/unphysical equivalent widths (due to ~0 continuum level)
    if occursin("eqw", name_i)
        filtered[filtered .> 100] .= NaN
    end
    if occursin("voff", name_i)
        # Perform a 5-sigma clip to remove outliers
        f_avg = nanmean(filtered)
        f_std = nanstd(filtered)
        filtered[abs.(filtered .- f_avg) .> 5f_std] .= NaN
    end

    if isnothing(modify_ax)
        fig = plt.figure()
        ax = fig.add_subplot(111) 
    else
        fig, ax = modify_ax
    end
    # Need to filter out any NaNs in order to use quantile
    vmin = nanquantile(filtered, 0.01)
    vmax = nanquantile(filtered, 0.99)
    # override vmin/vmax for mixing parameter
    if occursin("mixing", name_i)
        vmin = 0.
        vmax = 1.
    end
    nan_color = "k"
    text_color = "w"
    # if taking a voff, make sure vmin/vmax are symmetric and change the colormap to coolwarm
    if occursin("voff", name_i) || occursin("index", name_i) || occursin("vel", name_i)
        vabs = max(abs(vmin), abs(vmax))
        vmin = -vabs
        vmax = vabs
        if cmap == py_colormap.cubehelix
            cmap = py_colormap.RdBu_r
            # nan_color = "w"
            # text_color = "k"
        end
    end
    if occursin("chi2", name_i)
        vmin = 0
        # Hard upper limit on the reduced chi^2 map to show the structure
        vmax = min(nanmaximum(filtered), 30)
    end
    # default cmap is magma for FWHMs and equivalent widths
    if (occursin("fwhm", name_i) || occursin("eqw", name_i)) || occursin("vdisp", name_i) && cmap == py_colormap.cubehelix
        cmap = py_colormap.magma
    end
    # get discrete colormap for number of line components
    if occursin("n_comps", name_i) && cmap == py_colormap.cubehelix
        n_comps = nanmaximum(filtered)
        if !isfinite(n_comps)
            n_comps = 1
        else
            n_comps = Int(n_comps)
        end
        @assert n_comps < 20 "More than 20 components are unsupported!"
        # cmap_colors = py_colormap.rainbow(collect(range(0, 1, 21)))
        # cmap = py_colors.ListedColormap(cmap_colors[1:n_comps+1])
        cmap = plt.get_cmap("rainbow", n_comps+1)
        vmin = 0
        vmax = n_comps
    end

    # Add small value to vmax to prevent the maximum color value from being the same as the background
    small = 0.
    if cmap == py_colormap.cubehelix
        small = (vmax - vmin) / 1e3
    end

    # Set NaN color to either black or white
    cmap.set_bad(color=nan_color)

    if isnothing(colorscale_limits)
        cdata = ax.imshow(filtered', origin=:lower, cmap=cmap, vmin=vmin, vmax=vmax+small)
    else
        cdata = ax.imshow(filtered', origin=:lower, cmap=cmap, vmin=colorscale_limits[1], vmax=colorscale_limits[2])
    end
    ax.tick_params(which="both", axis="both", direction="in", color=text_color)
    ax.set_xlabel(L"$x$ (spaxels)")
    ax.set_ylabel(L"$y$ (spaxels)")
    if disable_axes
        ax.axis(:off)
    end

    # Angular and physical scalebars
    pix_as = sqrt(Ω) * 180/π * 3600
    n_pix = 1/pix_as
    @debug "Using angular diameter distance $(angular_diameter_dist(cosmo, z))"
    # Calculate in Mpc
    dA = angular_diameter_dist(u"pc", cosmo, z)
    # Remove units
    dA = uconvert(NoUnits, dA/u"pc")
    # l = d * theta (") where theta is chosen as 1/4 the horizontal extent of the image
    l = dA * (size(data, 1) * pix_as / 4) * π/180 / 3600  
    # Round to a nice even number
    l = Int(round(l, sigdigits=1))
     # new angular size for this scale
    θ = l / dA
    θ_as = round(θ * 180/π * 3600, digits=1)  # should be close to the original theta, by definition
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
    scalebar_text = cosmo.h ≈ 1.0 ? L"%$l$h^{-1}$ %$unit" : L"%$l %$unit"
    scalebar_1 = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, scalebar_text, "upper center", pad=0, borderpad=0, 
        color=text_color, frameon=false, size_vertical=0.1, label_top=false, bbox_to_anchor=(0.17, 0.1), bbox_transform=ax.transAxes)
    ax.add_artist(scalebar_1)
    if disable_axes
        scalebar_text = L"$\ang[angle-symbol-over-decimal]{;;%$θ_as}$"
        if θ_as > 60
            θ_as = round(θ_as/60, digits=1)  # convert to arcminutes
            scalebar_text = L"$\ang[angle-symbol-over-decimal]{;%$θ_as;}$"
        end
        scalebar_2 = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, scalebar_text, "lower center", pad=0, borderpad=0, 
            color=text_color, frameon=false, size_vertical=0.1, label_top=true, bbox_to_anchor=(0.17, 0.1), bbox_transform=ax.transAxes)
        ax.add_artist(scalebar_2)
    end

    # Add circle for the PSF FWHM
    r = psf_fwhm / pix_as / 2
    psf = plt.Circle(size(data) .* (0.93, 0.05) .+ (-r, r), r, color=text_color)
    ax.add_patch(psf)
    ax.annotate("PSF", size(data) .* (0.93, 0.05) .+ (-r, 2.5r + 1.75), ha=:center, va=:center, color=text_color)

    # Add line label, if applicable
    if !isnothing(line_latex)
        ax.annotate(line_latex, size(data) .* 0.95, ha=:right, va=:top, fontsize=16, color=text_color)
    end
    if !disable_colorbar
        if occursin("n_comps", name_i)
            n_comps = nanmaximum(filtered)
            if !isfinite(n_comps)
                n_comps = 1
            else
                n_comps = Int(n_comps)
            end
            ticks = collect(0:n_comps)
            cbar = fig.colorbar(cdata, ax=ax, label=bunit, ticks=ticks)
            cbar.ax.set_yticklabels(string.(ticks))
        else
            fig.colorbar(cdata, ax=ax, label=bunit)
        end
    end

    # Add a marker, if given
    if !isnothing(marker)
        ax.plot(marker[1]-1, marker[2]-1, "rx", ms=5)
    end

    # Make directories
    if (save_path ≠ "") && isnothing(modify_ax)
        if !isdir(dirname(save_path))
            mkpath(dirname(save_path))
        end
        plt.savefig(save_path, dpi=300, bbox_inches=:tight)
    end
    if !isnothing(modify_ax)
        return fig, ax, cdata
    end
    plt.close()

end


# Line parameters
"""
    plot_multiline_parameters(cube_fitter, param_maps, psf_interp[, snr_thresh, marker])

Helper function to plot line parameters on a grid of plots with a consistent color scale. This is useful
in particular for lines with multiple components. For example, for the flux, if a line has two components, 
this will make a grid of 3 plots containing the total combined flux, and the flux of each individual component,
all on the same color scale. Kinematic components such as the velocity don't have an equivalent "total" that is well
defined, so they would just get 2 plots containing the kinematics of each component.
"""
function plot_multiline_parameters(cube_fitter::CubeFitter, param_maps::ParamMaps, psf_interp::Spline1D, 
    snr_thresh::Real=3., marker::Union{Vector{<:Real},Nothing}=nothing)

    plotted_lines = []
    for line ∈ keys(param_maps.lines)
        # Remove the component index from the line name
        line_key = Symbol(join(split(string(line), "_")[1:end-1], "_"))
        # Dont repeat for lines that have multiple components since that's already handled within the body of this loop
        if line_key in plotted_lines
            continue
        end

        # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
        line_i = findfirst(cube_fitter.lines.names .== line_key)
        wave_i = cube_fitter.lines.λ₀[line_i]
        latex_i = cube_fitter.lines.latex[line_i]
        n_line_comps = sum(.!isnothing.(cube_fitter.lines.profiles[line_i, :]))
        component_keys = [Symbol(line_key, "_$(j)") for j in 1:n_line_comps]
        if n_line_comps < 2
            # Dont bother for lines with only 1 component
            continue
        end

        snr_filter = dropdims(nanmaximum(cat([param_maps.lines[comp][:SNR] for comp in component_keys]..., dims=3), dims=3), dims=3)

        # Get the total flux and eqw for lines with multiple components
        plot_total = false
        if n_line_comps > 1
            plot_total = true
            total_flux = log10.(sum([exp10.(param_maps.lines[comp][:flux]) for comp in component_keys])) 
            total_eqw = sum([param_maps.lines[comp][:eqw] for comp in component_keys])
        end

        for parameter ∈ keys(param_maps.lines[line])

            # Make a combined plot for all of the line components
            n_subplots = n_line_comps + 1 + (parameter ∈ [:flux, :eqw] && plot_total ? 1 : 0)
            width_ratios = [[20 for _ in 1:n_subplots-1]; 1]

            # Use gridspec to make the axes proper ratios
            fig = plt.figure(figsize=(6 * (n_subplots-1), 5.4))
            gs = fig.add_gridspec(1, n_subplots, width_ratios=width_ratios, height_ratios=(1,), wspace=0.05, hspace=0.05)
            ax = []
            for i in 1:n_subplots-1
                push!(ax, fig.add_subplot(gs[1,i]))
            end
            cax = fig.add_subplot(gs[1,n_subplots])

            # Get the minimum/maximum for the color scale based on the combined dataset
            vmin, vmax = 0., 0.
            if parameter == :flux && plot_total
                vmin = quantile(total_flux[isfinite.(total_flux) .& (snr_filter .> 3)], 0.01)
                vmax = quantile(total_flux[isfinite.(total_flux) .& (snr_filter .> 3)], 0.99)
            elseif parameter == :eqw && plot_total
                vmin = quantile(total_eqw[isfinite.(total_eqw) .& (snr_filter .> 3)], 0.01)
                vmax = quantile(total_eqw[isfinite.(total_eqw) .& (snr_filter .> 3)], 0.99)
            else
                # Each element of 'minmax' is a tuple with the minimum and maximum for that spaxel
                minmax = dropdims(nanextrema(cat([param_maps.lines[comp][parameter] for comp in component_keys]..., dims=3), dims=3), dims=3)
                mindata = [m[1] for m in minmax]
                maxdata = [m[2] for m in minmax]
                vmin = quantile(mindata[isfinite.(mindata) .& (snr_filter .> 3)], 0.01)
                vmax = quantile(maxdata[isfinite.(maxdata) .& (snr_filter .> 3)], 0.99)
                if parameter in (:voff, :voff_indiv)
                    vlim = max(abs(vmin), abs(vmax))
                    vmin = -vlim
                    vmax = vlim
                end
            end

            cdata = nothing
            ci = 1
            if parameter == :flux && plot_total
                name_i = join([line, "total_flux"], "_")
                save_path = ""
                _, _, cdata = plot_parameter_map(total_flux, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                    line_latex=latex_i, modify_ax=(fig, ax[ci]), disable_colorbar=true, colorscale_limits=(vmin, vmax), marker=marker)
                ci += 1
            end
            if parameter == :eqw && plot_total
                name_i = join([line, "total_eqw"], "_")
                save_path = ""
                _, _, cdata = plot_parameter_map(total_eqw, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                    cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                    line_latex=latex_i, modify_ax=(fig, ax[ci]), disable_colorbar=true, colorscale_limits=(vmin, vmax), marker=marker)
                ci += 1
            end
            for i in 1:n_line_comps
                data = param_maps.lines[component_keys[i]][parameter]
                name_i = join([line, parameter], "_")
                snr_filt = param_maps.lines[component_keys[i]][:SNR]
                if contains(string(parameter), "SNR")
                    snr_filt = nothing
                end
                save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(line_key)", "$(name_i).pdf")
                _, _, cdata = plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i), 
                    cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filt, snr_thresh=snr_thresh, 
                    line_latex=latex_i, modify_ax=(fig, ax[ci]), disable_colorbar=true, colorscale_limits=(vmin, vmax),
                    marker=marker)
                ci += 1
            end
            # Save the final figure
            name_final = join([line_key, parameter], "_")
            # Add the colorbar to cax
            fig.colorbar(cdata, cax=cax, label=paramstring_to_latex(name_final, cube_fitter.cosmology))
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$line_key", "$(name_final).pdf")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        end

        push!(plotted_lines, line_key)
    end
end



"""
    plot_parameter_maps(cube_fitter, param_maps; [snr_thresh])

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


# MIR implementation of plot_parameter_maps
function plot_mir_parameter_maps(cube_fitter::CubeFitter, param_maps::ParamMaps; snr_thresh::Real=3.)

    # Iterate over model parameters and make 2D maps
    @debug "Using solid angle $(cube_fitter.cube.Ω), redshift $(cube_fitter.z), cosmology $(cube_fitter.cosmology)"

    # Ineterpolate the PSF FWHM
    psf_interp = Spline1D(cube_fitter.cube.λ, cube_fitter.cube.psf, k=1)

    # Calculate the centroid
    data2d = sumdim(cube_fitter.cube.I, 3)
    _, mx = findmax(data2d)
    centroid = centroid_com(data2d[mx[1]-5:mx[1]+5, mx[2]-5:mx[2]+5]) .+ (mx.I .- 5) .- 1

    # Stellar continuum parameters
    for parameter ∈ keys(param_maps.stellar_continuum)
        data = param_maps.stellar_continuum[parameter]
        name_i = join(["stellar_continuum", parameter], "_")
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")
        plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf), 
            cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
    end

    # Dust continuum parameters
    for i ∈ keys(param_maps.dust_continuum)
        for parameter ∈ keys(param_maps.dust_continuum[i])
            data = param_maps.dust_continuum[i][parameter]
            name_i = join(["dust_continuum", i, parameter], "_") 
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf), 
                cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
        end
    end

    # Power law parameters
    for j ∈ keys(param_maps.power_law)
        for parameter ∈ keys(param_maps.power_law[j])
            data = param_maps.power_law[j][parameter]
            name_i = join(["power_law", j, parameter], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
                cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
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
                line_latex=latex_i, marker=centroid)
        end
    end

    # Total parameters for PAH complexes
    dust_complexes = unique(cube_fitter.dust_features.complexes[.!isnothing.(cube_fitter.dust_features.complexes)])
    for dust_complex in dust_complexes
        # Get the components that make up the complex
        indiv_inds = findall(cube_fitter.dust_features.complexes .== dust_complex)
        indivs = [cube_fitter.dust_features.names[i] for i ∈ indiv_inds]
        snr = dropdims(nanmaximum(cat([param_maps.dust_features[df][:SNR] for df ∈ indivs]..., dims=3), dims=3), dims=3)

        # Sum up individual component fluxes
        total_flux = log10.(sum([exp10.(param_maps.dust_features[df][:flux]) for df ∈ indivs]))
        # Wavelength and name
        wave_i = parse(Float64, dust_complex)
        name_i = "complex_$(dust_complex)_total_flux"
        comp_name = "PAH $dust_complex " * L"$\mu$m"
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "dust_features", "$(name_i).pdf")
        plot_parameter_map(total_flux, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
            cube_fitter.cosmology, cube_fitter.cube.wcs, line_latex=comp_name, snr_filter=snr, snr_thresh=snr_thresh,
            marker=centroid)

        # Repeat for equivalent width
        total_eqw = sum([param_maps.dust_features[df][:eqw] for df ∈ indivs])
        name_i = "complex_$(dust_complex)_total_eqw"
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "dust_features", "$(name_i).pdf")
        plot_parameter_map(total_eqw, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
            cube_fitter.cosmology, cube_fitter.cube.wcs, line_latex=comp_name, snr_filter=snr, snr_thresh=snr_thresh,
            marker=centroid)
    end

    # Absorption feature parameters
    for ab ∈ keys(param_maps.abs_features)
        wave_i = nanmedian(param_maps.abs_features[ab][:mean]) / (1 + cube_fitter.z)
        for parameter ∈ keys(param_maps.abs_features[ab])
            data = param_maps.abs_features[ab][parameter]
            name_i = join([ab, parameter], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "absorption_features", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
        end
    end

    # Extinction parameters
    for parameter ∈ keys(param_maps.extinction)
        data = param_maps.extinction[parameter]
        name_i = join(["extinction", parameter], "_")
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "extinction", "$(name_i).pdf")
        plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf), 
            cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
    end

    if cube_fitter.fit_sil_emission
        # Hot dust parameters
        for parameter ∈ keys(param_maps.hot_dust)
            data = param_maps.hot_dust[parameter]
            name_i = join(["hot_dust", parameter], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "hot_dust", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf), 
                cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
        end
    end

    # Template parameters
    for temp ∈ keys(param_maps.templates)
        for parameter ∈ keys(param_maps.templates[temp])
            data = param_maps.templates[temp][parameter]
            name_i = join(["template", temp, parameter], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
                cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
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
        # n_line_comps = sum(.!isnothing.(cube_fitter.lines.profiles[line_i, :]))
        # snr_filter = dropdims(
        #     nanmaximum(cat([param_maps.lines[Symbol(line_key, "_$i")][:SNR] for i in 1:n_line_comps]..., dims=3), dims=3), dims=3)

        for parameter ∈ keys(param_maps.lines[line])
            data = param_maps.lines[line][parameter]
            name_i = join([line, parameter], "_")
            snr_filt = param_maps.lines[line][:SNR]
            if contains(string(parameter), "SNR")
                snr_filt = nothing
            end
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(line_key)", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i), 
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filt, snr_thresh=snr_thresh,
                line_latex=latex_i, marker=centroid)
        end
    end
    # Composite line parameters
    for line ∈ keys(param_maps.lines_comp)
        # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
        line_i = findfirst(cube_fitter.lines.names .== line)
        wave_i = cube_fitter.lines.λ₀[line_i]
        latex_i = cube_fitter.lines.latex[line_i] 
        for parameter ∈ keys(param_maps.lines_comp[line])
            data = param_maps.lines_comp[line][parameter]
            name_i = join([line, parameter], "_")
            line_ind = findfirst(cube_fitter.lines.names .== line)
            component_keys = [Symbol(line, "_$(j)") for j in 1:sum(.~isnothing.(cube_fitter.lines.profiles[line_ind, :]))]
            snr_filt = dropdims(maximum(cat([param_maps.lines[comp][:SNR] for comp in component_keys]..., dims=3), dims=3), dims=3)

            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$line", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filt, snr_thresh=snr_thresh,
                line_latex=latex_i, marker=centroid)
        end
    end

    # Make combined plots for lines with multiple components
    plot_multiline_parameters(cube_fitter, param_maps, psf_interp, snr_thresh, centroid)

    # Total parameters for combined lines
    for comb_lines in cube_fitter.lines.combined
        # Check to make sure the lines were actually fit
        if !all([ln in cube_fitter.lines.names for ln in comb_lines])
            continue
        end
        # Get all of the line names + additional components
        component_keys = Symbol[]
        line_inds = [findfirst(name .== cube_fitter.lines.names) for name in comb_lines]
        for (ind, name) in zip(line_inds, comb_lines)
            n_line_comps = sum(.!isnothing.(cube_fitter.lines.profiles[ind, :]))
            append!(component_keys, [Symbol(name, "_$(j)") for j in 1:n_line_comps])
        end
        # Generate a group name based on the lines in the group
        species = unique([replace(string(ln), match(r"(_[0-9]+)", string(ln))[1] => "") for ln in comb_lines])
        species = String[]
        for line in comb_lines
            ln = string(line)
            m = match(r"(_[0-9]+)", ln)
            if !isnothing(m)
                ln = replace(ln, m[1] => "")
            end
            m = match(r"(HI_)", ln)
            if !isnothing(m)
                ln = replace(ln, m[1] => "")
            end
            push!(species, ln)
        end
        species = unique(species)
        group_name = join(species, "+")

        # Get the SNR filter and wavelength
        snr_filter = dropdims(maximum(cat([param_maps.lines[comp][:SNR] for comp in component_keys]..., dims=3), dims=3), dims=3)
        wave_i = median([cube_fitter.lines.λ₀[ind] for ind in line_inds])

        # Make a latex group name similar to the other group name
        species_ltx = unique([cube_fitter.lines.latex[ind] for ind in line_inds])
        group_name_ltx = join(species_ltx, L"$+$")

        # Total Flux
        total_flux = log10.(sum([exp10.(param_maps.lines[comp][:flux]) for comp in component_keys]))
        name_i = join([group_name, "total_flux"], "_")
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf")
        plot_parameter_map(total_flux, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
            cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
            line_latex=group_name_ltx, marker=centroid)
        
        # Total equivalent width
        total_eqw = sum([param_maps.lines[comp][:eqw] for comp in component_keys])
        name_i = join([group_name, "total_eqw"], "_")
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf")
        plot_parameter_map(total_eqw, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
            cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
            line_latex=group_name_ltx, marker=centroid)

        # Voff and FWHM
        if all(.!isnothing.(cube_fitter.lines.tied_voff[line_inds, 1]))
            voff = param_maps.lines[component_keys[1]][:voff]
            name_i = join([group_name, "voff"], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf") 
            plot_parameter_map(voff, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=group_name_ltx, marker=centroid) 
        end
        if all(.!isnothing.(cube_fitter.lines.tied_fwhm[line_inds, 1]))
            fwhm = param_maps.lines[component_keys[1]][:fwhm]
            name_i = join([group_name, "fwhm"], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf") 
            plot_parameter_map(fwhm, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=group_name_ltx, marker=centroid) 
        end
        
    end


    # Reduced chi^2 
    data = param_maps.statistics[:chi2] ./ param_maps.statistics[:dof]
    name_i = "reduced_chi2"
    save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "$(name_i).pdf")
    plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf), 
        cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)

    return

end


# Optical implementation of plot_parameter_maps
function plot_opt_parameter_maps(cube_fitter::CubeFitter, param_maps::ParamMaps; snr_thresh::Real=3.)

    # Iterate over model parameters and make 2D maps
    @debug "Using solid angle $(cube_fitter.cube.Ω), redshift $(cube_fitter.z), cosmology $(cube_fitter.cosmology)"

    # Ineterpolate the PSF
    psf_interp = Spline1D(cube_fitter.cube.λ, cube_fitter.cube.psf, k=1)

    data2d = sumdim(cube_fitter.cube.I, 3)
    _, mx = findmax(data2d)
    centroid = centroid_com(data2d[mx[1]-5:mx[1]+5, mx[2]-5:mx[2]+5]) .+ (mx.I .- 5) .- 1

    # Stellar population parameters
    for i ∈ keys(param_maps.stellar_populations)
        for parameter ∈ keys(param_maps.stellar_populations[i])
            data = param_maps.stellar_populations[i][parameter]
            name_i = join(["stellar_populations", i, parameter], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
                cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
        end
    end

    # Stellar kinematics parameters
    for parameter ∈ keys(param_maps.stellar_kinematics)
        data = param_maps.stellar_kinematics[parameter]
        name_i = join(["stellar_kinematics", parameter], "_")
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")
        plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
            cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
    end

    # Attenuation parameters
    for parameter ∈ keys(param_maps.attenuation)
        data = param_maps.attenuation[parameter]
        name_i = join(["attenuation", parameter], "_")
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "extinction", "$(name_i).pdf")
        plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
            cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
    end

    # Fe II parameters
    if cube_fitter.fit_opt_na_feii || cube_fitter.fit_opt_br_feii
        for parameter ∈ keys(param_maps.feii)
            data = param_maps.feii[parameter]
            name_i = join(["feii", parameter], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")    
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
                cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
        end
    end

    # Power law parameters
    for j in keys(param_maps.power_law)
        for parameter in keys(param_maps.power_law[j])
            data = param_maps.power_law[j][parameter]
            name_i = join(["power_law", j, parameter], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "continuum", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
                cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
        end
    end

    # Line parameters

    # Line parameters
    for line ∈ keys(param_maps.lines)
        # Remove the component index from the line name
        line_key = Symbol(join(split(string(line), "_")[1:end-1], "_"))
        # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
        line_i = findfirst(cube_fitter.lines.names .== line_key)
        wave_i = cube_fitter.lines.λ₀[line_i]
        latex_i = cube_fitter.lines.latex[line_i]
        # n_line_comps = sum(.!isnothing.(cube_fitter.lines.profiles[line_i, :]))
        # snr_filter = dropdims(
        #     nanmaximum(cat([param_maps.lines[Symbol(line_key, "_$i")][:SNR] for i in 1:n_line_comps]..., dims=3), dims=3), dims=3)

        for parameter ∈ keys(param_maps.lines[line])
            data = param_maps.lines[line][parameter]
            name_i = join([line, parameter], "_")
            snr_filt = param_maps.lines[line][:SNR]
            if contains(string(parameter), "SNR")
                snr_filt = nothing
            end
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(line_key)", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i), 
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filt, snr_thresh=snr_thresh,
                line_latex=latex_i, marker=centroid)
        end
    end
    # Composite line parameters
    for line ∈ keys(param_maps.lines_comp)
        # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
        line_i = findfirst(cube_fitter.lines.names .== line)
        wave_i = cube_fitter.lines.λ₀[line_i]
        latex_i = cube_fitter.lines.latex[line_i] 
        for parameter ∈ keys(param_maps.lines_comp[line])
            data = param_maps.lines_comp[line][parameter]
            name_i = join([line, parameter], "_")
            line_ind = findfirst(cube_fitter.lines.names .== line)
            component_keys = [Symbol(line, "_$(j)") for j in 1:sum(.~isnothing.(cube_fitter.lines.profiles[line_ind, :]))]
            snr_filt = dropdims(maximum(cat([param_maps.lines[comp][:SNR] for comp in component_keys]..., dims=3), dims=3), dims=3)

            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$line", "$(name_i).pdf")
            plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filt, snr_thresh=snr_thresh,
                line_latex=latex_i, marker=centroid)
        end
    end

    # Make combined plots for lines with multiple components
    plot_multiline_parameters(cube_fitter, param_maps, psf_interp, snr_thresh, centroid)

    # Total parameters for lines with multiple components
    for (k, name) ∈ enumerate(cube_fitter.lines.names)
        n_line_comps = sum(.!isnothing.(cube_fitter.lines.profiles[k, :]))
        wave_i = cube_fitter.lines.λ₀[k]
        if n_line_comps > 1
            component_keys = [Symbol(name, "_$(j)") for j in 1:n_line_comps]
            snr_filter = dropdims(maximum(cat([param_maps.lines[comp][:SNR] for comp in component_keys]..., dims=3), dims=3), dims=3)

            # Total flux
            total_flux = log10.(sum([exp10.(param_maps.lines[comp][:flux]) for comp in component_keys]))
            name_i = join([name, "total_flux"], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(name)", "$(name_i).pdf")
            plot_parameter_map(total_flux, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=cube_fitter.lines.latex[k], marker=centroid)

            # Total equivalent width
            total_eqw = sum([param_maps.lines[comp][:eqw] for comp in component_keys])
            name_i = join([name, "total_eqw"], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(name)", "$(name_i).pdf")
            plot_parameter_map(total_eqw, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=cube_fitter.lines.latex[k], marker=centroid)
            
        end
    end

    # Total parameters for combined lines
    for comb_lines in cube_fitter.lines.combined
        # Check to make sure the lines were actually fit
        if !all([ln in cube_fitter.lines.names for ln in comb_lines])
            continue
        end
        # Get all of the line names + additional components
        component_keys = Symbol[]
        line_inds = [findfirst(name .== cube_fitter.lines.names) for name in comb_lines]
        for (ind, name) in zip(line_inds, comb_lines)
            n_line_comps = sum(.!isnothing.(cube_fitter.lines.profiles[ind, :]))
            append!(component_keys, [Symbol(name, "_$(j)") for j in 1:n_line_comps])
        end
        # Generate a group name based on the lines in the group
        species = unique([replace(string(ln), match(r"(_[0-9]+)", string(ln))[1] => "") for ln in comb_lines])
        species = String[]
        for line in comb_lines
            ln = string(line)
            m = match(r"(_[0-9]+)", ln)
            if !isnothing(m)
                ln = replace(ln, m[1] => "")
            end
            m = match(r"(HI_)", ln)
            if !isnothing(m)
                ln = replace(ln, m[1] => "")
            end
            push!(species, ln)
        end
        species = unique(species)
        group_name = join(species, "+")

        # Get the SNR filter and wavelength
        snr_filter = dropdims(maximum(cat([param_maps.lines[comp][:SNR] for comp in component_keys]..., dims=3), dims=3), dims=3)
        wave_i = median([cube_fitter.lines.λ₀[ind] for ind in line_inds])

        # Make a latex group name similar to the other group name
        species_ltx = unique([cube_fitter.lines.latex[ind] for ind in line_inds])
        group_name_ltx = join(species_ltx, L"$+$")

        # Total Flux
        total_flux = log10.(sum([exp10.(param_maps.lines[comp][:flux]) for comp in component_keys]))
        name_i = join([group_name, "total_flux"], "_")
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf")
        plot_parameter_map(total_flux, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
            cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
            line_latex=group_name_ltx, marker=centroid)
        
        # Total equivalent width
        total_eqw = sum([param_maps.lines[comp][:eqw] for comp in component_keys])
        name_i = join([group_name, "total_eqw"], "_")
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf")
        plot_parameter_map(total_eqw, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
            cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
            line_latex=group_name_ltx, marker=centroid)
         
        # Voff and FWHM
        if all(.!isnothing.(cube_fitter.lines.tied_voff[line_inds, 1]))
            voff = param_maps.lines[component_keys[1]][:voff]
            name_i = join([group_name, "voff"], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf") 
            plot_parameter_map(voff, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=group_name_ltx, marker=centroid) 
        end
        if all(.!isnothing.(cube_fitter.lines.tied_fwhm[line_inds, 1]))
            fwhm = param_maps.lines[component_keys[1]][:fwhm]
            name_i = join([group_name, "fwhm"], "_")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf") 
            plot_parameter_map(fwhm, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=group_name_ltx, marker=centroid) 
        end
        
    end

    # Reduced chi^2 
    data = param_maps.statistics[:chi2] ./ param_maps.statistics[:dof]
    name_i = "reduced_chi2"
    save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "$(name_i).pdf")
    plot_parameter_map(data, name_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf), 
        cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)

    return

end


"""
    make_movie(cube_fitter, cube_model; [cmap])

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
    write_fits(cube_fitter, cube_data, cube_model, param_maps, param_errs; [aperture])

Save the best fit results for the cube into two FITS files: one for the full 3D intensity model of the cube, split up by
individual model components, and one for 2D parameter maps of the best-fit parameters for each spaxel in the cube.
"""
write_fits(cube_fitter::CubeFitter, cube_data::NamedTuple, cube_model::CubeModel, param_maps::ParamMaps, 
    param_errs::Vector{<:ParamMaps}; aperture::Union{Vector{<:Aperture.AbstractAperture},String,Nothing}=nothing) =
    cube_fitter.spectral_region == :MIR ?
    write_fits_mir(cube_fitter, cube_data, cube_model, param_maps, param_errs; aperture=aperture) :
    write_fits_opt(cube_fitter, cube_data, cube_model, param_maps, param_errs; aperture=aperture)


# MIR implementation of the write_fits function
function write_fits_mir(cube_fitter::CubeFitter, cube_data::NamedTuple, cube_model::CubeModel, param_maps::ParamMaps, 
    param_errs::Vector{<:ParamMaps}; aperture::Union{Vector{<:Aperture.AbstractAperture},String,Nothing}=nothing)

    aperture_keys = []
    aperture_vals = []
    aperture_comments = []
    # If using an aperture, extract its properties 
    if typeof(aperture) <: Aperture.AbstractAperture

        # Get the name (giving the shape of the aperture: circular, elliptical, or rectangular)
        ap_shape = string(typeof(aperture))

        aperture_keys = ["AP_SHAPE", "AP_X", "AP_Y"]
        aperture_vals = Any[ap_shape, aperture.x, aperture.y]
        aperture_comments = ["The shape of the spectrum extraction aperture", "The x coordinate of the aperture",
            "The y coordinate of the aperture"]

        # Get the properties, i.e. radius for circular 
        if contains(ap_shape, "CircularAperture")
            push!(aperture_keys, "AP_RADIUS")
            push!(aperture_vals, aperture.r)
            push!(aperture_comments, "Radius of aperture (pixels)")
        elseif contains(ap_shape, "EllipticalAperture")
            append!(aperture_keys, ["AP_A", "AP_B", "AP_ANGLE"])
            append!(aperture_vals, [aperture.a, aperture.b, aperture.θ])
            append!(aperture_comments, ["Semimajor axis of aperture (pixels)", 
                "Semiminor axis of aperture (pixels)", "Aperture angle in deg."])
        elseif contains(ap_shape, "RectangularAperture")
            append!(aperture_keys, ["AP_W", "AP_H", "AP_ANGLE"])
            append!(aperture_vals, [aperture.w, aperture.h, aperture.θ])
            append!(aperture_comments, ["Width of aperture (pixels)", 
                "Height of aperture (pixels)", "Aperture angle in deg."])
        elseif contains(ap_shape, "CircularAnnulus")
            append!(aperture_keys, ["AP_R_IN", "AP_R_OUT"])
            append!(aperture_vals, [aperture.r_in, aperture.r_out])
            append!(aperture_comments, ["Inner radius of annulus (pixels)", "Outer radius of annulus (pixels)"])
        elseif contains(ap_shape, "EllipticalAnnulus")
            append!(aperture_keys, ["AP_A_IN", "AP_A_OUT", "AP_B_IN", "AP_B_OUT", "AP_ANGLE"])
            append!(aperture_vals, [aperture.a_in, aperture.a_out, aperture.b_in, aperture.b_out, aperture.θ])
            append!(aperture_comments, ["Inner semimajor axis of annulus (pixels)", "Outer semimajor axis of annulus (pixels)",
                "Inner semiminor axis of annulus (pixels)", "Outer semiminor axis of annulus (pixels)", "Annulus angle in deg."])
        elseif contains(ap_shape, "RectangularAnnulus")
            append!(aperture_keys, ["AP_W_IN", "AP_W_OUT", "AP_H_IN", "AP_H_OUT", "AP_ANGLE"])
            append!(aperture_vals, [aperture.w_in, aperture.w_out, aperture.h_in, aperture.h_out, aperture.θ])
            append!(aperture_comments, ["Inner width of annulus (pixels)", "Outer width of annulus (pixels)",
                "Inner height of annulus (pixels)", "Outer height of annulus (pixels)", "Aperture angle in deg."])    
        end

        # Also append the aperture area
        push!(aperture_keys, "AP_AREA")
        push!(aperture_vals, get_area(aperture))
        push!(aperture_comments, "Area of aperture in pixels")
    
    elseif aperture isa String

        n_pix = [sum(.~cube_fitter.cube.mask[:, :, i]) for i in axes(cube_fitter.cube.mask, 3)]
        aperture_keys = ["AP_SHAPE", "AP_AREA"]
        aperture_vals = Any["full_cube", median(n_pix[isfinite.(n_pix)])]

    end

    # Header information
    hdr = FITSHeader(
        Vector{String}(cat(["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "PIXAR_SR", "RA", "DEC", "WCSAXES",
            "CDELT1", "CDELT2", "CDELT3", "CTYPE1", "CTYPE2", "CTYPE3", "CRPIX1", "CRPIX2", "CRPIX3", 
            "CRVAL1", "CRVAL2", "CRVAL3", "CUNIT1", "CUNIT2", "CUNIT3", "PC1_1", "PC1_2", "PC1_3", 
            "PC2_1", "PC2_2", "PC2_3", "PC3_1", "PC3_2", "PC3_3"], aperture_keys, dims=1)),

        cat([cube_fitter.name, cube_fitter.z, cube_fitter.cube.channel, cube_fitter.cube.band, cube_fitter.cube.Ω, 
         cube_fitter.cube.α, cube_fitter.cube.δ, cube_fitter.cube.wcs.naxis],
         cube_fitter.cube.wcs.cdelt, cube_fitter.cube.wcs.ctype, cube_fitter.cube.wcs.crpix,
         cube_fitter.cube.wcs.crval, cube_fitter.cube.wcs.cunit, reshape(cube_fitter.cube.wcs.pc, (9,)), aperture_vals, dims=1),

        Vector{String}(cat(["Target name", "Target redshift", "MIRI channel", "MIRI band",
        "Solid angle per pixel (rad.)", "Right ascension of target (deg.)", "Declination of target (deg.)",
        "number of World Coordinate System axes", 
        "first axis increment per pixel", "second axis increment per pixel", "third axis increment per pixel",
        "first axis coordinate type", "second axis coordinate type", "third axis coordinate type",
        "axis 1 coordinate of the reference pixel", "axis 2 coordinate of the reference pixel", "axis 3 coordinate of the reference pixel",
        "first axis value at the reference pixel", "second axis value at the reference pixel", "third axis value at the reference pixel",
        "first axis units", "second axis units", "third axis units",
        "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element",
        "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element",
        "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element"], 
        aperture_comments, dims=1))
    )

    if cube_fitter.save_full_model
        # Create the 3D intensity model FITS file
        FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_full_model.fits"), "w") do f

            @debug "Writing 3D model FITS HDUs"
            # Permute the wavelength axis here back to the third axis to be consistent with conventions

            write(f, Vector{Int}())                                                                     # Primary HDU (empty)
            write(f, Float32.(cube_data.I .* (1 .+ cube_fitter.z)); name="DATA", header=hdr)            # Raw data 
            write(f, Float32.(cube_data.σ .* (1 .+ cube_fitter.z)); name="ERROR")                       # Error in the raw data
            write(f, permutedims(cube_model.model, (2,3,1)); name="MODEL")                              # Full intensity model
            write(f, permutedims(cube_model.stellar, (2,3,1)); name="STELLAR_CONTINUUM")                # Stellar continuum model
            for i ∈ 1:size(cube_model.dust_continuum, 4)
                write(f, permutedims(cube_model.dust_continuum[:, :, :, i], (2,3,1)); name="DUST_CONTINUUM_$i")   # Dust continua
            end
            for l ∈ 1:size(cube_model.power_law, 4)                                                     # Power laws
                write(f, permutedims(cube_model.power_law[:, :, :, l], (2,3,1)); name="POWER_LAW_$l")
            end
            for (j, df) ∈ enumerate(cube_fitter.dust_features.names)
                write(f, permutedims(cube_model.dust_features[:, :, :, j], (2,3,1)); name=uppercase("$df"))        # Dust feature profiles
            end
            for (m, ab) ∈ enumerate(cube_fitter.abs_features.names)                                     
                write(f, permutedims(cube_model.abs_features[:, :, :, m], (2,3,1)); name=uppercase("$ab"))         # Absorption feature profiles
            end
            for (q, tp) ∈ enumerate(cube_fitter.template_names)
                write(f, permutedims(cube_model.templates[:, :, :, q], (2,3,1)); name=uppercase("TEMPLATE_$tp"))   # Template profiles
            end
            for (k, line) ∈ enumerate(cube_fitter.lines.names)
                write(f, permutedims(cube_model.lines[:, :, :, k], (2,3,1)); name=uppercase("$line"))              # Emission line profiles
            end
            write(f, permutedims(cube_model.extinction, (2,3,1)); name="EXTINCTION")                    # Extinction model
            write(f, permutedims(cube_model.abs_ice, (2,3,1)); name="ABS_ICE")                          # Ice Absorption model
            write(f, permutedims(cube_model.abs_ch, (2,3,1)); name="ABS_CH")                            # CH Absorption model
            if cube_fitter.fit_sil_emission
                write(f, permutedims(cube_model.hot_dust, (2,3,1)); name="HOT_DUST")                    # Hot dust model
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
                write_key(f["$ab"], "BUNIT", "-")
            end
            for tp ∈ cube_fitter.template_names
                write_key(f["TEMPLATE_$tp"], "BUNIT", "MJy/sr")
            end
            for line ∈ cube_fitter.lines.names
                write_key(f["$line"], "BUNIT", "MJy/sr")
            end
            write_key(f["EXTINCTION"], "BUNIT", "-")
            write_key(f["ABS_ICE"], "BUNIT", "-")
            write_key(f["ABS_CH"], "BUNIT", "-")
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

            write(f, Vector{Int}())  # Primary HDU (empty)

            # Stellar continuum parameters
            for (i, parameter) ∈ enumerate(keys(param_data.stellar_continuum))
                data = param_data.stellar_continuum[parameter]
                name_i = join(["stellar_continuum", parameter], "_")
                if occursin("amp", name_i)
                    bunit = "log(erg.s-1.cm-2.Hz-1.sr-1)"
                elseif occursin("temp", name_i)
                    bunit = "K"
                end
                write(f, data; name=uppercase(name_i), header=i==1 ? hdr : nothing)
                write_key(f[name_i], "BUNIT", bunit)
            end

            # Dust continuum parameters
            for i ∈ keys(param_data.dust_continuum)
                for parameter ∈ keys(param_data.dust_continuum[i])
                    data = param_data.dust_continuum[i][parameter]
                    name_i = join(["dust_continuum", i, parameter], "_")
                    if occursin("amp", name_i)
                        bunit = "log(erg.s-1.cm-2.Hz-1.sr-1)"
                    elseif occursin("temp", name_i)
                        bunit = "K"
                    end
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)  
                end
            end

            # Power law parameters
            for l ∈ keys(param_data.power_law)
                for parameter ∈ keys(param_data.power_law[l])
                    data = param_data.power_law[l][parameter]
                    name_i = join(["power_law", l, parameter], "_")
                    if occursin("amp", name_i)
                        bunit = "log(erg.s-1.cm-2.Hz-1.sr-1)"
                    elseif occursin("index", name_i)
                        bunit = "-"
                    end
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            if cube_fitter.fit_sil_emission
                # Hot dust parameters
                for parameter ∈ keys(param_data.hot_dust)
                    data = param_data.hot_dust[parameter]
                    name_i = join(["hot_dust", parameter], "_")
                    if occursin("amp", name_i)
                        bunit = "log(erg.s-1.cm-2.Hz-1.sr-1)"
                    elseif occursin("temp", name_i)
                        bunit = "K"
                    elseif occursin("frac", name_i) || occursin("tau", name_i)
                        bunit = "-"
                    elseif occursin("peak", name_i)
                        bunit = "um"
                    end
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            # Dust feature (PAH) parameters
            for df ∈ keys(param_data.dust_features)
                for parameter ∈ keys(param_data.dust_features[df])
                    data = param_data.dust_features[df][parameter]
                    name_i = join(["dust_features", df, parameter], "_")
                    if occursin("amp", name_i)
                        bunit = "log(erg.s-1.cm-2.Hz-1.sr-1)"
                    elseif occursin("fwhm", name_i) || occursin("mean", name_i) || occursin("eqw", name_i)
                        bunit = "um"
                    elseif occursin("flux", name_i)
                        bunit = "log(erg.s-1.cm-2)"
                    elseif occursin("SNR", name_i) || occursin("index", name_i) || occursin("cutoff", name_i)
                        bunit = "-"
                    end
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            # Absorption feature parameters
            for ab ∈ keys(param_data.abs_features)
                for parameter ∈ keys(param_data.abs_features[ab])
                    data = param_data.abs_features[ab][parameter]
                    name_i = join(["abs_features", ab, parameter], "_")
                    if occursin("tau", name_i)
                        bunit = "-"
                    elseif occursin("fwhm", name_i) || occursin("mean", name_i) || occursin("eqw", name_i)
                        bunit = "um"
                    end
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end
            
            # Extinction parameters
            for parameter ∈ keys(param_data.extinction)
                data = param_data.extinction[parameter]
                name_i = join(["extinction", parameter], "_")
                bunit = "-"
                write(f, data; name=uppercase(name_i))
                write_key(f[name_i], "BUNIT", bunit)  
            end

            # Template parameters
            for temp ∈ keys(param_data.templates)
                for parameter ∈ keys(param_data.templates[temp])
                    data = param_data.templates[temp][parameter]
                    name_i = join(["templates", temp, parameter], "_")
                    bunit = "-"
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            # Line parameters
            for line ∈ keys(param_data.lines)
                for parameter ∈ keys(param_data.lines[line])
                    data = param_data.lines[line][parameter]
                    name_i = join(["lines", line, parameter], "_")
                    if occursin("amp", name_i)
                        bunit = "log(erg.s-1.cm-2.Hz-1.sr-1)"
                    elseif occursin("fwhm", name_i) || occursin("voff", name_i)
                        bunit = "km/s"
                    elseif occursin("flux", name_i)
                        bunit = "log(erg.s-1.cm-2)"
                    elseif occursin("eqw", name_i)
                        bunit = "um"
                    elseif occursin("SNR", name_i) || occursin("h3", name_i) || 
                        occursin("h4", name_i) || occursin("mixing", name_i)
                        bunit = "-"
                    end
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end
            # Composite line parameters
            for line ∈ keys(param_data.lines_comp)
                for parameter ∈ keys(param_data.lines_comp[line])
                    data = param_data.lines_comp[line][parameter]
                    name_i = join(["lines", line, parameter], "_")
                    if occursin("n_comps", name_i)
                        bunit = "-"
                    end
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            if isone(index)
                # chi^2 statistics
                for parameter ∈ keys(param_data.statistics)
                    data = param_maps.statistics[parameter]
                    name_i = join(["statistics", parameter], "_")
                    bunit = "-"
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end
              
            # Add another HDU for the voronoi bin map, if applicable
            if !isnothing(cube_fitter.cube.voronoi_bins)
                write(f, cube_fitter.cube.voronoi_bins; name="VORONOI_BINS")
            end
        end
    end
end


# Optical implementation of the write_fits function
function write_fits_opt(cube_fitter::CubeFitter, cube_data::NamedTuple, cube_model::CubeModel, param_maps::ParamMaps, 
    param_errs::Vector{<:ParamMaps}; aperture::Union{Vector{Aperture.AbstractAperture},Nothing}=nothing)

    aperture_keys = []
    aperture_vals = []
    aperture_comments = []
    # If using an aperture, extract its properties 
    if typeof(aperture) <: Aperture.AbstractAperture

        # Get the name (giving the shape of the aperture: circular, elliptical, or rectangular)
        ap_shape = string(typeof(aperture))

        aperture_keys = ["AP_SHAPE", "AP_X", "AP_Y"]
        aperture_vals = Any[ap_shape, aperture.x, aperture.y]
        aperture_comments = ["The shape of the spectrum extraction aperture", "The x coordinate of the aperture",
            "The y coordinate of the aperture"]

        # Get the properties, i.e. radius for circular 
        if contains(ap_shape, "CircularAperture")
            push!(aperture_keys, "AP_RADIUS")
            push!(aperture_vals, aperture.r)
            push!(aperture_comments, "Radius of aperture (pixels)")
        elseif contains(ap_shape, "EllipticalAperture")
            append!(aperture_keys, ["AP_A", "AP_B", "AP_ANGLE"])
            append!(aperture_vals, [aperture.a, aperture.b, aperture.θ])
            append!(aperture_comments, ["Semimajor axis of aperture (pixels)", 
                "Semiminor axis of aperture (pixels)", "Aperture angle in deg."])
        elseif contains(ap_shape, "RectangularAperture")
            append!(aperture_keys, ["AP_W", "AP_H", "AP_ANGLE"])
            append!(aperture_vals, [aperture.w, aperture.h, aperture.θ])
            append!(aperture_comments, ["Width of aperture (pixels)", 
                "Height of aperture (pixels)", "Aperture angle in deg."])
        elseif contains(ap_shape, "CircularAnnulus")
            append!(aperture_keys, ["AP_R_IN", "AP_R_OUT"])
            append!(aperture_vals, [aperture.r_in, aperture.r_out])
            append!(aperture_comments, ["Inner radius of annulus (pixels)", "Outer radius of annulus (pixels)"])
        elseif contains(ap_shape, "EllipticalAnnulus")
            append!(aperture_keys, ["AP_A_IN", "AP_A_OUT", "AP_B_IN", "AP_B_OUT", "AP_ANGLE"])
            append!(aperture_vals, [aperture.a_in, aperture.a_out, aperture.b_in, aperture.b_out, aperture.θ])
            append!(aperture_comments, ["Inner semimajor axis of annulus (pixels)", "Outer semimajor axis of annulus (pixels)",
                "Inner semiminor axis of annulus (pixels)", "Outer semiminor axis of annulus (pixels)", "Annulus angle in deg."])
        elseif contains(ap_shape, "RectangularAnnulus")
            append!(aperture_keys, ["AP_W_IN", "AP_W_OUT", "AP_H_IN", "AP_H_OUT", "AP_ANGLE"])
            append!(aperture_vals, [aperture.w_in, aperture.w_out, aperture.h_in, aperture.h_out, aperture.θ])
            append!(aperture_comments, ["Inner width of annulus (pixels)", "Outer width of annulus (pixels)",
                "Inner height of annulus (pixels)", "Outer height of annulus (pixels)", "Aperture angle in deg."])   
        end

        # Also append the aperture area
        push!(aperture_keys, "AP_AREA")
        push!(aperture_vals, get_area(aperture))
        push!(aperture_comments, "Area of aperture in pixels")
    
    elseif aperture isa String

        n_pix = [sum(.~cube_fitter.cube.mask[:, :, i]) for i in axes(cube_fitter.cube.mask, 3)]
        aperture_keys = ["AP_SHAPE", "AP_AREA"]
        aperture_vals = Any["full_cube", median(n_pix[isfinite.(n_pix)])]

    end
    
    # Header information
    if !isnothing(cube_fitter.cube.wcs)

        hdr = FITSHeader(
            Vector{String}(cat(["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "PIXAR_SR", "RA", "DEC", "WCSAXES",
                "CDELT1", "CDELT2", "CDELT3", "CTYPE1", "CTYPE2", "CTYPE3", "CRPIX1", "CRPIX2", "CRPIX3", 
                "CRVAL1", "CRVAL2", "CRVAL3", "CUNIT1", "CUNIT2", "CUNIT3", "PC1_1", "PC1_2", "PC1_3", 
                "PC2_1", "PC2_2", "PC2_3", "PC3_1", "PC3_2", "PC3_3"], aperture_keys, dims=1)),

            cat([cube_fitter.name, cube_fitter.z, cube_fitter.cube.channel, cube_fitter.cube.band, cube_fitter.cube.Ω, 
            cube_fitter.cube.α, cube_fitter.cube.δ, cube_fitter.cube.wcs.naxis],
            cube_fitter.cube.wcs.cdelt, cube_fitter.cube.wcs.ctype, cube_fitter.cube.wcs.crpix,
            cube_fitter.cube.wcs.crval, cube_fitter.cube.wcs.cunit, reshape(cube_fitter.cube.wcs.pc, (9,)), aperture_vals, dims=1),

            Vector{String}(cat(["Target name", "Target redshift", "MIRI channel", "MIRI band",
            "Solid angle per pixel (rad.)", "Right ascension of target (deg.)", "Declination of target (deg.)",
            "number of World Coordinate System axes", 
            "first axis increment per pixel", "second axis increment per pixel", "third axis increment per pixel",
            "first axis coordinate type", "second axis coordinate type", "third axis coordinate type",
            "axis 1 coordinate of the reference pixel", "axis 2 coordinate of the reference pixel", "axis 3 coordinate of the reference pixel",
            "first axis value at the reference pixel", "second axis value at the reference pixel", "third axis value at the reference pixel",
            "first axis units", "second axis units", "third axis units",
            "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element",
            "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element",
            "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element"], 
            aperture_comments, dims=1))
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
            # Permute the wavelength axis here back to the third axis to be consistent with conventions

            write(f, Vector{Int}())                                                                     # Primary HDU (empty)
            write(f, Float32.(cube_data.I ./ (1 .+ cube_fitter.z)); name="DATA", header=hdr)           # Raw data 
            write(f, Float32.(cube_data.σ ./ (1 .+ cube_fitter.z)); name="ERROR")                       # Error in the raw data
            write(f, permutedims(cube_model.model, (2,3,1)); name="MODEL")                              # Full intensity model
            for i ∈ 1:size(cube_model.stellar, 4)
                write(f, permutedims(cube_model.stellar[:, :, :, i], (2,3,1)); name="STELLAR_POPULATION_$i")   # Stellar population models
            end
            if cube_fitter.fit_opt_na_feii
                write(f, permutedims(cube_model.na_feii, (2,3,1)); name="NA_FEII")                      # Narrow Fe II emission
            end
            if cube_fitter.fit_opt_br_feii
                write(f, permutedims(cube_model.br_feii, (2,3,1)); name="BR_FEII")                      # Broad Fe II emission
            end
            for j in 1:size(cube_model.power_law, 4)
                write(f, permutedims(cube_model.power_law[:, :, :, j], (2,3,1)); name="POWER_LAW_$j")   # Power laws
            end
            for (k, line) ∈ enumerate(cube_fitter.lines.names)
                write(f, permutedims(cube_model.lines[:, :, :, k], (2,3,1)); name=uppercase("$line"))   # Emission line profiles
            end
            write(f, permutedims(cube_model.attenuation_stars, (2,3,1)); name="ATTENUATION_STARS")      # Starlight attenuation model
            write(f, permutedims(cube_model.attenuation_gas, (2,3,1)); name="ATTENUATION_GAS")          # Gas attenuation model
            write(f, ["wave"], [cube_data.λ .* (1 .+ cube_fitter.z)],                                   # wavelength vector
                hdutype=TableHDU, name="WAVELENGTH", units=Dict(:wave => "um"))

            # Insert physical units into the headers of each HDU -> MegaJansky per steradian for all except
            # the extinction profile, which is a multiplicative constant
            write_key(f["DATA"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
            write_key(f["ERROR"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
            write_key(f["MODEL"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
            for i ∈ 1:size(cube_model.stellar, 4)
                write_key(f["STELLAR_POPULATION_$i"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
            end
            if cube_fitter.fit_opt_na_feii
                write_key(f["NA_FEII"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
            end
            if cube_fitter.fit_opt_br_feii
                write_key(f["BR_FEII"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
            end
            for j ∈ 1:size(cube_model.power_law, 4)
                write_key(f["POWER_LAW_$j"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
            end
            write_key(f["ATTENUATION_STARS"], "BUNIT", "-")
            write_key(f["ATTENUATION_GAS"], "BUNIT", "-")
        end
    end

    # Create the 2D parameter map FITS file for the parameters and the errors
    for (index, param_data) ∈ enumerate([param_maps, param_errs[1], param_errs[2]])

        FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_parameter_" * 
            ("maps", "errs_low", "errs_upp")[index] * ".fits"), "w") do f

            @debug "Writing 2D parameter map FITS HDUs"

            write(f, Vector{Int}())  # Primary HDU (empty)

            # Stellar population parameters
            for i ∈ keys(param_data.stellar_populations)
                for (j, parameter) ∈ enumerate(keys(param_data.stellar_populations[i]))
                    data = param_data.stellar_populations[i][parameter]
                    name_i = join(["stellar_populations", i, parameter], "_")
                    if occursin("mass", name_i)
                        bunit = "log(solMass)"
                    elseif occursin("age", name_i)
                        bunit = "Gyr"
                    elseif occursin("metallicity", name_i)
                        bunit = "-"
                    end
                    write(f, data; name=uppercase(name_i), header=j==1 ? hdr : nothing)
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            # Stellar kinematics parameters
            for parameter ∈ keys(param_data.stellar_kinematics)
                data = param_data.stellar_kinematics[parameter]
                name_i = join(["stellar_kinematics", parameter], "_")
                bunit = "km/s"
                write(f, data; name=uppercase(name_i))
                write_key(f[name_i], "BUNIT", bunit)
            end

            # Attenuation parameters
            for parameter ∈ keys(param_data.attenuation)
                data = param_data.attenuation[parameter]
                name_i = join(["attenuation", parameter], "_")
                if occursin("E_BV", name_i)
                    bunit = "mag"
                else
                    bunit = "-"
                end
                write(f, data; name=uppercase(name_i))
                write_key(f[name_i], "BUNIT", bunit)
            end

            # Fe II emission
            if cube_fitter.fit_opt_na_feii || cube_fitter.fit_opt_br_feii
                for parameter ∈ keys(param_data.feii)
                    data = param_data.feii[parameter]
                    name_i = join(["feii", parameter], "_")
                    if occursin("amp", name_i)
                        bunit = "log(erg.s-1.cm-2.Hz-1.sr-1)"
                    else
                        bunit = "km/s"
                    end
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            # Power laws
            for j ∈ keys(param_data.power_law)
                for parameter ∈ keys(param_data.power_law[j])
                    data = param_data.power_law[j][parameter]
                    name_i = join(["power_law", j, parameter], "_")
                    if occursin("amp", name_i)
                        bunit = "log(erg.s-1.cm-2.Hz-1.sr-1)"
                    else
                        bunit = "-"
                    end
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            # Line parameters
            for line ∈ keys(param_data.lines)
                for parameter ∈ keys(param_data.lines[line])
                    data = param_data.lines[line][parameter]
                    name_i = join(["lines", line, parameter], "_")
                    if occursin("amp", name_i)
                        bunit = "log(erg.s-1.cm-2.Hz-1.sr-1)"
                    elseif occursin("fwhm", name_i) || occursin("voff", name_i)
                        bunit = "km/s"
                    elseif occursin("flux", name_i)
                        bunit = "log(erg.s-1.cm-2)"
                    elseif occursin("eqw", name_i)
                        bunit = "um"
                    elseif occursin("SNR", name_i) || occursin("h3", name_i) || 
                        occursin("h4", name_i) || occursin("mixing", name_i)
                        bunit = "-"
                    end
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end
            # Composite line parameters
            for line ∈ keys(param_data.lines_comp)
                for parameter ∈ keys(param_data.lines_comp[line])
                    data = param_data.lines_comp[line][parameter]
                    name_i = join(["lines", line, parameter], "_")
                    if occursin("n_comps", name_i)
                        bunit = "-"
                    end
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            if isone(index)
                # chi^2 statistics
                for parameter ∈ keys(param_data.statistics)
                    data = param_maps.statistics[parameter]
                    name_i = join(["statistics", parameter], "_")
                    bunit = "-"
                    write(f, data; name=uppercase(name_i))
                    write_key(f[name_i], "BUNIT", bunit)
                end
            end

            # Add another HDU for the voronoi bin map, if applicable
            if !isnothing(cube_fitter.cube.voronoi_bins)
                write(f, cube_fitter.cube.voronoi_bins; name="VORONOI_BINS")
            end
        end
    end
end

