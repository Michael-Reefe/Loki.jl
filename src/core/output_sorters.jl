

"""
    sort_line_components!(cube_fitter, params)

A helper function that sorts line parameters based on a sorting criterion (flux, FWHM, voff, etc.) so that
the bootstrapped values are correct. This method sorts the parameters before they have been sorted into cubes.
This is necessary before taking the 50th, 16th, and 84th percentiles of the parameters since they may fit 
different line components during different bootstrap iterations, forming bimodal distributions. This function
sorts the parameters first to prevent this from happening.
"""
function sort_line_components!(cube_fitter::CubeFitter, result::SpaxelFitResult; mask_zeros::Bool=true)

    params = result.popt
    oopt = out_options(cube_fitter)
    lines = model(cube_fitter).lines
    df = model(cube_fitter).dust_features
    # Do not sort
    if isnothing(oopt.sort_line_components)
        return
    end
    if lines.config.rel_amp || lines.config.rel_voff || lines.config.rel_fwhm
        return
    end

    # pᵢ = cube_fitter.n_params_cont + 1
    # pⱼ = pᵢ + cube_fitter.n_params_lines + length(get_flattened_nonfit_parameters(df))

    for (k, line) in enumerate(lines.profiles)   # <- iterates over emission lines

        profiles = [comp.profile for comp in line]
        line_name = lines.names[k]
        @assert all(profiles .== profiles[1]) "All line profiles must be the same to use sorted bootstrapping"

        # Dont need to sort if there is only 1 profile
        n_prof = length(profiles)
        if n_prof < 2
            continue
        end

        i0 = zeros(Int, n_prof)
        amps = i0
        voffs = copy(i0)
        fwhms = copy(i0)
        h3s = copy(i0)
        h4s = copy(i0)
        etas = copy(i0)
        fluxes = copy(i0)
        eqws = copy(i0)
        snrs = copy(i0)

        for (j, component) in enumerate(line)    # <- iterates over each velocity component in one line (for multi-component fits)

            n_prof += 1
            # set parameters to nan so they are ignored when calculating the percentiles
            amps[j], voffs[j], fwhms[j] = fast_indexin(["lines.$line_name.$j.amp", "lines.$line_name.$j.voff", "lines.$line_name.$j.fwhm"], result.pnames)
            mask_line = (params[amps[j]] == 0.) && (j > 1) && mask_zeros
            if mask_line
                # amplitude is not overwritten so that the zeros downweight the final amplitude (same for flux, eqw, snr)
                params[voffs[j]] *= NaN
                params[fwhms[j]] *= NaN
            end

            if component.profile == :GaussHermite
                h3s[j], h4s[j] = fast_indexin(["lines.$line_name.$j.h3", "lines.$line_name.$j.h4"], result.pnames)
                if mask_line
                    params[h3s[j]] *= NaN
                    params[h4s[j]] *= NaN
                end
            elseif component.profile == :Voigt
                etas[j] = fast_indexin("lines.$line_name.$j.mixing", result.pnames)
                if mask_line
                    params[etas[j]] *= NaN
                end
            end
            fluxes[j], eqws[j], snrs[j] = fast_indexin(["lines.$line_name.$j.flux", "lines.$line_name.$j.eqw", "lines.$line_name.$j.SNR"], result.pnames)
        end

        # always sort by voff for bootstrap iterations!
        if oopt.sort_line_components == :flux
            sort_quantity = params[fluxes]
        elseif oopt.sort_line_components == :amp
            sort_quantity = params[amps]
        elseif oopt.sort_line_components == :fwhm
            sort_quantity = params[fwhms]
        elseif oopt.sort_line_components == :voff
            sort_quantity = params[voffs]
        else
            error("Unrecognized sorting quantity: $(oopt.sort_line_components)")
        end

        # Sort by the relevant sorting quantity (NaNs are always placed at the end)
        #  1 = sort in increasing order
        # -1 = sort in decreasing order
        if lines.config.sort_order[k] == 1
            ss = sortperm(ustrip.(sort_quantity))
        elseif lines.config.sort_order[k] == -1
            # Cant just reverse because then NaNs would be placed at the beginning
            n_inf = sum(.~isfinite.(sort_quantity))
            ss = [sortperm(ustrip.(sort_quantity), rev=true)[n_inf+1:end]; findall(.~isfinite.(sort_quantity))]
        else
            error("Unrecognized sort order: $(lines.config.sort_order[k]) (must be 1 or -1)")
        end

        # Reassign the parameters in the new order
        params[amps] .= params[amps][ss]
        params[voffs] .= params[voffs][ss]
        params[fwhms] .= params[fwhms][ss]
        if any(profiles .== :GaussHermite)
            params[h3s] .= params[h3s][ss]
            params[h4s] .= params[h4s][ss]
        end
        if any(profiles .== :Voigt)
            params[etas] .= params[etas][ss]
        end
        params[fluxes] .= params[fluxes][ss]
        params[eqws] .= params[eqws][ss]
        params[snrs] .= params[snrs][ss]
    end

end


"""
    sort_line_components!(cube_fitter, param_maps, index, cube_data)

A helper function that sorts line parameters based on a sorting criterion (flux, FWHM, voff, etc.) so that
the parameter maps look continuous. This method sorts the parameters after the full fitting procedure.
"""
function sort_line_components!(cube_fitter::CubeFitter, param_maps::ParamMaps, index::CartesianIndex, cube_data::NamedTuple)

    oopt = out_options(cube_fitter)
    lines = model(cube_fitter).lines
    if isnothing(oopt.sort_line_components)
        return
    end
    if lines.config.rel_amp || lines.config.rel_fwhm || lines.config.rel_voff
        @warn "Skipping line component sorting due to a relative line parameter flag being set!"
        return
    end

    for k ∈ 1:cube_fitter.n_lines
        prefix = "lines.$(lines.names[k])"
        n_prof = get_val(param_maps, "$(prefix).n_comps")[index]
        n_prof = isfinite(n_prof) ? floor(Int, n_prof) : n_prof
        if (n_prof <= 1) || !isfinite(n_prof)
            @debug "Skipping line component sorting for $prefix due to 1 or fewer profiles in this spaxel"
            continue
        end
        prof_1 = lines.profiles[k][1].profile
        prof_all = [p.profile for p in lines.profiles[k][1:n_prof]]
        if !all(prof_all .== prof_1)
            @warn "Skipping line component sorting for $prefix because it is not supported for different line profiles!"
            continue
        end
        snrs = get_val(param_maps, index, ["$(prefix).$(j).SNR" for j in 1:n_prof])
        if oopt.sort_line_components == :flux
            sort_inds = ["$(prefix).$(j).flux" for j in 1:n_prof]
        elseif oopt.sort_line_components == :amp
            sort_inds = ["$(prefix).$(j).amp" for j in 1:n_prof]
        elseif oopt.sort_line_components == :fwhm
            sort_inds = ["$(prefix).$(j).fwhm" for j in 1:n_prof]
        elseif oopt.sort_line_components == :voff
            sort_inds = ["$(prefix).$(j).voff" for j in 1:n_prof]
        else
            error("Unrecognized sorting quantity: $(oopt.sort_line_components)")
        end
        sort_quantity = get_val(param_maps, index, sort_inds)

        # Check the SNRs of each component
        bad = findall(snrs .< oopt.map_snr_thresh)
        sort_quantity[bad] .*= NaN

        # Sort by the relevant sorting quantity (NaNs are always placed at the end)
        #  1 = sort in increasing order
        # -1 = sort in decreasing order
        if lines.config.sort_order[k] == 1
            ss = sortperm(ustrip.(sort_quantity))
        elseif lines.config.sort_order[k] == -1
            # Cant just reverse because then NaNs would be placed at the beginning
            n_inf = sum(.~isfinite.(sort_quantity))
            ss = [sortperm(ustrip.(sort_quantity), rev=true)[n_inf+1:end]; findall(.~isfinite.(sort_quantity))]
        else
            error("Unrecognized sort order: $(lines.config.sort_order[k]) (must be 1 or -1)")
        end

        # Reassign the parameters in this order
        params_to_sort = ["amp", "voff", "fwhm"]
        if lines.profiles[k][1] == :GaussHermite
            append!(params_to_sort, ["h3", "h4"])
        end
        if lines.profiles[k][1] == :Voigt
            append!(params_to_sort, ["mixing"])
        end
        append!(params_to_sort, ["flux", "eqw", "SNR"])
        for ps in params_to_sort
            for d in (param_maps.data, param_maps.err_upp, param_maps.err_low)
                param_inds = [findfirst(param_maps.parameters.names .== "$(prefix).$(j).$(ps)") for j in 1:n_prof]
                d[index, param_inds] .= d[index, param_inds[ss]]
            end
        end
    end
end


"""
    sort_temperatures!(cube_fitter, param_maps, index)

A helper function that sorts dust continua parameters based on the temperature parameters so that the 
parameter maps look continuous.
"""
function sort_temperatures!(cube_fitter::CubeFitter, param_maps::ParamMaps, index::CartesianIndex)

    # Collect the relevant dust continuum parameters
    temp_inds = [findfirst(param_maps.parameters.names .== "continuum.dust.$(i).temp") for i in 1:cube_fitter.n_dust_cont]
    amp_inds = [findfirst(param_maps.parameters.names .== "continuum.dust.$(i).amp") for i in 1:cube_fitter.n_dust_cont]

    temps = param_maps.data[index, temp_inds]
    ss = sortperm(ustrip.(temps), rev=true)

    # Sort the dust parameters
    for d in (param_maps.data, param_maps.err_upp, param_maps.err_low)
        d[index, temp_inds] .= d[index, temp_inds[ss]]
        d[index, amp_inds] .= d[index, amp_inds[ss]]
    end

end

