

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

    pᵢ = cube_fitter.n_params_cont + 1
    pⱼ = pᵢ + cube_fitter.n_params_lines + length(get_flattened_nonfit_parameters(df))

    for (k, line) in enumerate(lines.profiles)   # <- iterates over emission lines

        amps = Int64[]
        voffs = Int64[]
        fwhms = Int64[]
        h3s = Int64[]
        h4s = Int64[]
        etas = Int64[]
        fluxes = Int64[]
        eqws = Int64[]
        snrs = Int64[]
        n_prof = 0

        profiles = [comp.profile for comp in line]
        @assert all(profiles .== profiles[1]) "All line profiles must be the same to use sorted bootstrapping"

        for (j, component) in enumerate(line)    # <- iterates over each velocity component in one line (for multi-component fits)

            n_prof += 1
            # set parameters to nan so they are ignored when calculating the percentiles
            mask_line = (params[pᵢ] == 0.) && (j > 1) && mask_zeros
            if mask_line
                # amplitude is not overwritten so that the zeros downweight the final amplitude (same for flux, eqw, snr)
                params[pᵢ+1] *= NaN
                params[pᵢ+2] *= NaN
            end
            push!(amps, pᵢ)
            push!(voffs, pᵢ+1)
            push!(fwhms, pᵢ+2)
            pᵢ += 3

            if component.profile == :GaussHermite
                push!(h3s, pᵢ)
                push!(h4s, pᵢ+1)
                if mask_line
                    params[pᵢ:pᵢ+1] .*= NaN
                end
                pᵢ += 2
            elseif component.profile == :Voigt
                push!(etas, pᵢ)
                if mask_line
                    params[pᵢ] *= NaN
                end
                pᵢ += 1
            end
            push!(fluxes, pⱼ)
            push!(eqws, pⱼ+1)
            push!(snrs, pⱼ+2)
            pⱼ += 3
        end
        # Dont forget to count the composite line parameters
        pⱼ += 5

        # Dont need to sort if there is only 1 profile
        if n_prof < 2
            continue
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
        if lines.sort_order[k] == 1
            ss = sortperm(sort_quantity)
        elseif lines.sort_order[k] == -1
            # Cant just reverse because then NaNs would be placed at the beginning
            n_inf = sum(.~isfinite.(sort_quantity))
            ss = [sortperm(sort_quantity, rev=true)[n_inf+1:end]; findall(.~isfinite.(sort_quantity))]
        else
            error("Unrecognized sort order: $(lines.sort_order[k]) (must be 1 or -1)")
        end

        # Reassign the parameters in the new order
        params[amps] .= params[amps][ss]
        params[voffs] .= params[voffs][ss]
        params[fwhms] .= params[fwhms][ss]
        if length(h3s) > 0
            params[h3s] .= params[h3s][ss]
            params[h4s] .= params[h4s][ss]
        end
        if length(etas) > 0
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
        if !all(lines.profiles[k][1:n_prof] .== lines.profiles[k][1])
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
            ss = sortperm(sort_quantity)
        elseif lines.config.sort_order[k] == -1
            # Cant just reverse because then NaNs would be placed at the beginning
            n_inf = sum(.~isfinite.(sort_quantity))
            ss = [sortperm(sort_quantity, rev=true)[n_inf+1:end]; findall(.~isfinite.(sort_quantity))]
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
    ss = sortperm(temps, rev=true)

    # Sort the dust parameters
    for d in (param_maps.data, param_maps.err_upp, param_maps.err_low)
        d[index, temp_inds] .= d[index, temp_inds[ss]]
        d[index, amp_inds] .= d[index, amp_inds[ss]]
    end

end
