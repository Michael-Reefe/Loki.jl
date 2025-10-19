############################## OUTPUT / SAVING FUNCTIONS ####################################

const ascii_lowercase = "abcdefghijklmnopqrstuvwxyz"


function assign_outputs(out_params::AbstractArray{<:Number}, out_errs::AbstractArray{<:Number}, 
    out_np_ssp::AbstractArray{Int}, cube_fitter::CubeFitter, cube_data::NamedTuple, aperture::Bool=false)

    # Create the CubeModel and ParamMaps structs to be filled in
    np_ssp = maximum(out_np_ssp)
    spaxels = CartesianIndices(size(out_params)[1:2])

    firsti = findfirst(index->any(isfinite.(out_params[index, :])), spaxels)
    if !isnothing(cube_fitter.cube.voronoi_bins)
        firsti = CartesianIndex(cube_fitter.cube.voronoi_bins[firsti])
    end
    spax = make_normalized_spaxel(cube_data, firsti, cube_fitter; use_ap=aperture, 
        use_vorbins=!isnothing(cube_fitter.cube.voronoi_bins))
    spax_model = spax
    if length(cube_fitter.spectral_region.gaps) > 0
        spax_model = get_model_spaxel(cube_fitter, spax, nothing)
    end
    size3 = length(spax_model.λ)

    cube_model = generate_cubemodel(cube_fitter, (size(cube_fitter.cube.I)[1:2]..., size3); do_1d=aperture)
    param_maps = generate_parammaps(cube_fitter; do_1d=aperture, stellar_params=np_ssp)

    fopt = fit_options(cube_fitter)
    oopt = out_options(cube_fitter)

    line_config = model(cube_fitter).lines.config
    if eltype(cube_data.I) <: QPerFreq
        restframe_factor = 1 + cube_fitter.z
    else
        restframe_factor = 1 / (1 + cube_fitter.z)
    end

    # prepare an array that will hold unit strings for the output    
    param_units = Vector{String}(undef, size(param_maps.data, 3))
    param_units .= ""

    # Loop over each spaxel and fill in the associated fitting parameters into the ParamMaps and CubeModel
    # I know this is long and ugly and looks stupid but it works for now and I'll make it pretty later
    prog = Progress(length(spaxels); showspeed=true)
    for index ∈ spaxels

        # skip spaxels that weren't actually fit
        if all(.~isfinite.(out_params[index, :]))
            next!(prog)
            continue
        end

        # Get the normalization to un-normalized the fitted parameters
        data_index = !isnothing(cube_fitter.cube.voronoi_bins) ? cube_fitter.cube.voronoi_bins[index] : index
        @debug "Spaxel $index"

        # Grab the stellar results
        stellar_vals = nothing
        results_stellar = nothing
        if fopt.fit_stellar_continuum
            fname = !isnothing(cube_fitter.cube.voronoi_bins) ? "voronoi_bin_$(index[1])" : "spaxel_$(index[1])_$(index[2])"
            results_stellar = deserialize(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.ssp"))
            stellar_vals = Array{Quantity{Float64}}(undef, 1+2np_ssp)
            stellar_vals[:] .= NaN
            stellar_vals[1] = results_stellar.mtot
            stellar_vals[2:1+length(results_stellar.ages)] .= results_stellar.ages
            stellar_vals[2+np_ssp:1+np_ssp+length(results_stellar.logzs)] .= results_stellar.logzs
        end

        # Set the 2D parameter map outputs
        # First re-evaluate the model and grab the normalizations we need
        pc = cube_fitter.n_params_cont + 1
        iindex = index
        if !isnothing(cube_fitter.cube.voronoi_bins)
            iindex = CartesianIndex(cube_fitter.cube.voronoi_bins[index])
        end
        spax = make_normalized_spaxel(cube_data, iindex, cube_fitter; use_ap=aperture, 
            use_vorbins=!isnothing(cube_fitter.cube.voronoi_bins))
        spax_model = copy(spax)
        if length(cube_fitter.spectral_region.gaps) > 0
            spax_model = get_model_spaxel(cube_fitter, spax, results_stellar)
        else
            add_stellar_weights!(spax_model, results_stellar)
        end

        I_cont, comps_c, norms = model_continuum(spax_model, spax_model.N, ustrip.(out_params[index, 1:pc-1]), unit.(out_params[index, 1:pc-1]),
            cube_fitter, false, spax_model == spax, true, true)
        
        # Loop through parameters and save them in the parammaps data structure 
        for (pᵢ, pname) in enumerate(param_maps.parameters.names)

            # Get the values
            if pᵢ ≤ size(out_params, 3)
                val = out_params[index, pᵢ]
                err_upp, err_low = out_errs[index, pᵢ, 1:2]
            else
                @assert fopt.fit_stellar_continuum 
                val = stellar_vals[pᵢ-size(out_params, 3)]
                err_upp = NaN * unit(val)
                err_low = NaN * unit(val)
            end
            unit_check(unit(val), unit(err_upp))
            unit_check(unit(err_upp), unit(err_low))

            # Handle line parameters
            # if param_maps.line_transform[pᵢ]
            if contains(pname, "lines")
                line_comp = split(pname, ".")[end-1]
                if isdigit(line_comp[1])
                    line_comp = parse(Int, line_comp)
                    if !isone(line_comp) && ((contains(pname, "amp") && line_config.rel_amp) || (contains(pname, "fwhm") && line_config.rel_fwhm))
                        # get the amp/fwhm of the first line component
                        pstr = contains(pname, "amp") ? "amp" : "fwhm"
                        ind_1 = findfirst(param_maps.parameters.names .== replace(pname, "$line_comp.$pstr" => "1.$pstr"))
                        # apply multiplicative transformation
                        err_upp = √((out_errs[index, ind_1, 1] * val)^2 + (err_upp * out_params[index, ind_1])^2) 
                        err_low = √((out_errs[index, ind_1, 2] * val)^2 + (err_low * out_params[index, ind_1])^2) 
                        val *= out_params[index, ind_1]
                    elseif !isone(line_comp) && contains(pname, "voff") && line_config.rel_voff
                        # get the voff of the first line component
                        voff_1_ind = findfirst(param_maps.parameters.names .== replace(pname, "$line_comp.voff" => "1.voff"))
                        err_upp = √(err_upp^2 + out_errs[index, voff_1_ind, 1]^2) 
                        err_low = √(err_low^2 + out_errs[index, voff_1_ind, 2]^2) 
                        # apply additive transformation
                        val += out_params[index, voff_1_ind]
                    end
                end
            end

            # Apply other transformations
            # Shift back to observed frame
            do_log = false
            for transform in param_maps.parameters.transformations[pᵢ]
                if transform == RestframeTransform 
                    if typeof(val) <: QWave
                        # if it's some kind of wavelength, ALWAYS do the wavelength transformation
                        val *= 1 + cube_fitter.z
                        err_upp *= 1 + cube_fitter.z
                        err_low *= 1 + cube_fitter.z
                    elseif typeof(val) <: QInvWave
                        # Drude asymmetry profiles (units of inverse wavelength)
                        val /= 1 + cube_fitter.z
                        err_upp /= 1 + cube_fitter.z
                        err_low /= 1 + cube_fitter.z
                    else
                        # otherwise, assume it's some kind of amplitude with specific intensity units
                        val *= restframe_factor
                        err_upp *= restframe_factor
                        err_low *= restframe_factor
                    end
                end
                # Normalize units
                if transform == NormalizeTransform
                    # grab a potential extra normalization constant from the fitting function
                    if pname in keys(norms)
                        norm_i = norms[pname]
                    else
                        norm_i = 1.0
                    end
                    val *= spax_model.N / norm_i
                    err_upp *= spax_model.N / norm_i
                    err_low *= spax_model.N / norm_i
                end
                # Take the log10 --> do it this way to ensure that this is always done LAST
                if transform == LogTransform
                    do_log = true
                end
            end
            @debug "Output parameter: $pname | Value: $(ustrip(val)) + $(ustrip(err_upp)) - $(ustrip(err_low)) | Units: $(unit(val))"
            punit = replace(string(unit(val)), "μ" => "u", "Å" => "angstrom")
            if punit == "NoUnits"
                punit = ""
            end

            if do_log
                # edit the unit string to indicate logarithmic units
                punit = "[" * punit * "]"
                err_upp = err_upp / (log(10) * val)
                err_low = err_low / (log(10) * val) 
                val = log10(ustrip(val))
            end

            # a few special cases
            if split(pname, ".")[end] == "metallicity"
                punit = "[Fe/H]"
            end
            if split(pname, ".")[end] == "E_BV"
                punit = "mag"
            end

            # Set the values
            param_maps.data[index, pᵢ] = val
            param_maps.err_upp[index, pᵢ] = err_upp
            param_maps.err_low[index, pᵢ] = err_low
            param_units[pᵢ] = punit
        end

        # Save marker of the point where the continuum parameters end and the line parameters begin
        vᵢ = pc
        pᵢ = pc + cube_fitter.n_params_lines

        I_line, comps_l = model_line_residuals(spax_model, ustrip.(out_params[index, vᵢ:pᵢ-1]), unit.(out_params[index, vᵢ:pᵢ-1]), 
            model(cube_fitter).lines, cube_fitter.lsf, comps_c["total_extinction_gas"], trues(length(spax_model.λ)), true)

        I_model, comps, = collect_total_fit_results(spax, spax_model, cube_fitter, I_cont, I_line, comps_c, comps_l, 0, 0)

        # Sort the dust continuum parameters based on the temperature
        sort_temperatures!(cube_fitter, param_maps, index)

        # Sort the parameters for multicomponent lines
        sort_line_components!(cube_fitter, param_maps, index, cube_data)

        if oopt.save_full_model
            # Set 3D model cube outputs, shifted back to the observed frame
            # Remember the wavelength axis is the first axis here to increase efficiency
            cube_model.model[:, index] .= I_model .* restframe_factor
            cube_model.extinction_stars[:, index] .= comps["total_extinction_stars"]
            cube_model.extinction_gas[:, index] .= comps["total_extinction_gas"]
            if fopt.extinction_curve == "decompose"
                cube_model.absorption_silicates[:, index, 1] .= comps["absorption_oli"]
                cube_model.absorption_silicates[:, index, 2] .= comps["absorption_pyr"]
                cube_model.absorption_silicates[:, index, 3] .= comps["absorption_for"]
            else
                cube_model.absorption_silicates[:, index, 1] .= comps["absorption_silicates"]
            end
            if fopt.fit_ch_abs
                cube_model.abs_ice[:, index] .= comps["absorption_ice"]
                cube_model.abs_ch[:, index] .= comps["absorption_ch"]
            end
            if fopt.fit_stellar_continuum
                cube_model.stellar[:, index] .= comps["SSPs"] .* restframe_factor
            end
            if fopt.fit_opt_na_feii
                cube_model.na_feii[:, index] .= comps["na_feii"] .* restframe_factor
            end
            if fopt.fit_opt_br_feii
                cube_model.br_feii[:, index] .= comps["br_feii"] .* restframe_factor
            end
            for l ∈ 1:cube_fitter.n_power_law
                cube_model.power_law[:, index, l] .= comps["power_law_$l"] .* restframe_factor
            end
            for i ∈ 1:cube_fitter.n_dust_cont
                cube_model.dust_continuum[:, index, i] .= comps["dust_cont_$i"] .* restframe_factor
            end
            if fopt.fit_sil_emission
                cube_model.hot_dust[:, index] .= comps["hot_dust"] .* restframe_factor
            end
            for q ∈ 1:cube_fitter.n_templates
                cube_model.templates[:, index, q] .= comps["templates_$q"] .* restframe_factor
            end
            for (k, dcomplex) in enumerate(model(cube_fitter).dust_features.profiles)
                for (j, component) in enumerate(dcomplex)
                    cube_model.dust_features[:, index, k] .+= comps["dust_feat_$(k)_$(j)"] .* restframe_factor
                end
            end
            for m ∈ 1:cube_fitter.n_abs_feat
                cube_model.abs_features[:, index, m] .= comps["absorption_feat_$m"]
            end
            for (k, line) in enumerate(model(cube_fitter).lines.profiles)
                for (j, component) in enumerate(line) 
                    cube_model.lines[:, index, k] .+= comps["line_$(k)_$(j)"] .* restframe_factor
                end
            end

        next!(prog)
        end
    end

    param_maps, param_units, cube_model
end


# MIR implementation of the assign_qso3d_outputs function
function assign_qso3d_outputs(param_maps::ParamMaps, cube_model::CubeModel, cube_fitter::CubeFitter,
    psf_norm::Array{<:Real,3})

    param_maps_3d = generate_parammaps(cube_fitter; do_1d=false)
    cube_model_3d = generate_cubemodel(cube_fitter, size(cube_fitter.cube.I); do_1d=false)
    fopt = fit_options(cube_fitter)
    lines = model(cube_fitter).lines
    dust_features = model(cube_fitter).dust_features

    n_pix = nansum(.~cube_fitter.cube.mask, dims=(1,2))
    for k in axes(param_maps_3d.data, 3)
        name = param_maps_3d.parameters.names[k]
        if !contains(name, "flux") && !contains(name, "amp")
            param_maps_3d.data[:,:,k] .= param_maps.data[1,1,k]
            continue
        end
        # get the central wavelength of the feature in question
        namelist = split(name, ".")
        if namelist[1] == "dust_features"
            λi = pah_name_to_float(namelist[2]) * u"μm"
        elseif namelist[1] == "lines"
            ln_ind = findfirst(lines.names .== Symbol(namelist[2]))
            λi = lines.λ₀[ln_ind]
        else
            # continuum amplitudes
            λi = 10^nanmean(log10.(ustrip.(cube_fitter.cube.λ))) .* unit(cube_fitter.cube.λ[1])
        end
        # convert to an index
        w = argmin(abs.(cube_fitter.cube.λ .- λi))
        dont_log = (namelist[1] == "lines") && fopt.lines_allow_negative
        amp_factor = contains(name, "amp") ? n_pix[k] : 1.
        # intensities need an extra conversion factor
        # (because the intensities recorded in param_maps are over the whole FOV
        #  whereas these intensities are over single spaxels: area_FOV / area_spaxel = number of spaxels)
        if dont_log
            param_maps_3d.data[:,:,k] .= param_maps.data[1,1,k] .* psf_norm[:,:,w] .* amp_factor
        else
            new = 10 .^ param_maps.data[1,1,k] .* psf_norm[:,:,w] .* amp_factor
            new[new .< 0] .= 0.0
            param_maps_3d.data[:,:,k] .= log10.(new)
        end
    end

    psf_norm_p = permutedims(psf_norm .* n_pix, (3,1,2))
    _shape = size(psf_norm)[1:2]

    # Remember the wavelength axis is the first axis here to increase efficiency
    cube_model_3d.model .= extendp(cube_model.model[:,1,1], _shape) .* psf_norm_p
    cube_model_3d.extinction_stars .= cube_model.extinction_stars[:,1,1] 
    cube_model_3d.extinction_gas .= cube_model.extinction_gas[:,1,1]
    if fopt.extinction_curve == "decompose"
        cube_model_3d.absorption_silicates[:,:,:,1] .= cube_model.absorption_silicates[:,1,1,1]
        cube_model_3d.absorption_silicates[:,:,:,2] .= cube_model.absorption_silicates[:,1,1,2]
        cube_model_3d.absorption_silicates[:,:,:,3] .= cube_model.absorption_silicates[:,1,1,3]
    else
        cube_model_3d.absorption_silicates[:,:,:,1] .= cube_model.absorption_silicates[:,1,1,1]
    end
    if fopt.fit_ch_abs
        cube_model_3d.abs_ice .= cube_model.abs_ice[:,1,1]
        cube_model_3d.abs_ch .= cube_model.abs_ch[:,1,1]
    end
    cube_model_3d.stellar .= extendp(cube_model.stellar[:,1,1], _shape) .* psf_norm_p
    if fopt.fit_opt_na_feii
        cube_model_3d.na_feii .= extendp(cube_model.na_feii[:,1,1], _shape) .* psf_norm_p
    end
    if fopt.fit_opt_br_feii
        cube_model_3d.br_feii .= extendp(cube_model.br_feii[:,1,1], _shape) .* psf_norm_p
    end
    for l ∈ 1:cube_fitter.n_power_law
        cube_model_3d.power_law[:,:,:,l] .= extendp(cube_model.power_law[:,1,1,l], _shape) .* psf_norm_p
    end
    for i ∈ 1:cube_fitter.n_dust_cont
        cube_model_3d.dust_continuum[:,:,:,i] .= extendp(cube_model.dust_continuum[:,1,1,i], _shape) .* psf_norm_p
    end
    if fopt.fit_sil_emission
        cube_model_3d.hot_dust .= extendp(cube_model.hot_dust[:,1,1], _shape) .* psf_norm_p
    end
    for q ∈ 1:cube_fitter.n_templates
        cube_model_3d.templates[:,:,:,q] .= extendp(cube_model.templates[:,1,1,q], _shape) .* psf_norm_p
    end
    for (k, dcomplex) in enumerate(model(cube_fitter).dust_features.profiles)
        cube_model_3d.dust_features[:,:,:,k] .= extendp(cube_model.dust_features[:,1,1,k], _shape) .* psf_norm_p
    end
    for m ∈ 1:cube_fitter.n_abs_feat
        cube_model_3d.abs_features[:,:,:,m] .= cube_model.abs_features[:,1,1,m]
    end
    for (k, line) in enumerate(model(cube_fitter).lines.profiles)
        cube_model_3d.lines[:,:,:,k] .= extendp(cube_model.lines[:,1,1,k], _shape) .* psf_norm_p
    end 

    param_maps_3d, cube_model_3d
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

    fopt = fit_options(cube_fitter)
    lines = model(cube_fitter).lines

    for (i, line) ∈ enumerate(lines.names)

        # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
        wave_i = uconvert(unit(cube_fitter.cube.λ[1]), lines.λ₀[i])
        latex_i = lines.labels[i]
        n_line_comps = length(lines.profiles[i])
        component_keys = [string(line) * ".$(j)" for j in 1:n_line_comps]
        if n_line_comps < 2
            # Dont bother for lines with only 1 component
            continue
        end
        snr_filter = get_val(param_maps, "lines.$(string(line)).total_snr")

        # Get the total flux and eqw for lines with multiple components
        plot_total = false
        if n_line_comps > 1
            plot_total = true
            total_flux = ustrip.(get_val(param_maps, "lines.$(string(line)).total_flux"))
            total_eqw = ustrip.(get_val(param_maps, "lines.$(string(line)).total_eqw"))
        end

        parameters = unique([split(pname, ".")[end] for pname in param_maps.parameters.names if contains(pname, "lines.$line.1")])       
       
        for parameter ∈ parameters

            # Make a combined plot for all of the line components
            n_subplots = n_line_comps + 1 + (parameter ∈ ["flux", "eqw"] && plot_total ? 1 : 0)
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
            if parameter == "flux" && plot_total
                vmin = quantile([0.; total_flux[isfinite.(total_flux) .& (snr_filter .> 3)]], 0.01)
                vmax = quantile([0.; total_flux[isfinite.(total_flux) .& (snr_filter .> 3)]], 0.99)
            elseif parameter == "eqw" && plot_total
                vmin = quantile([0.; total_eqw[isfinite.(total_eqw) .& (snr_filter .> 3)]], 0.01)
                vmax = quantile([0.; total_eqw[isfinite.(total_eqw) .& (snr_filter .> 3)]], 0.99)
            else
                # Each element of 'minmax' is a tuple with the minimum and maximum for that spaxel
                minmax = dropdims(nanextrema(ustrip.(get_val(param_maps, ["lines.$comp.$parameter" for comp in component_keys])), 
                    dims=3), dims=3)
                mindata = [m[1] for m in minmax]
                maxdata = [m[2] for m in minmax]
                mask1 = isfinite.(mindata) .& (snr_filter .> 3)
                mask2 = isfinite.(maxdata) .& (snr_filter .> 3)
                if sum(mask1) > 0 && sum(mask2) > 0
                    vmin = quantile(mindata[mask1], 0.01)
                    vmax = quantile(maxdata[mask2], 0.99)
                else
                    vmin = 0.
                    vmax = 1.
                end
                if parameter == "voff"
                    vlim = max(abs(vmin), abs(vmax))
                    vmin = -vlim
                    vmax = vlim
                end
            end

            cdata = nothing
            ci = 1
            if parameter == "flux" && plot_total
                name_i = join([line, "total_flux"], ".")
                bunit = get_label(param_maps, "lines.$(component_keys[1]).flux")
                save_path = ""
                _, _, cdata = plot_parameter_map(total_flux, name_i, bunit, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(ustrip(wave_i)),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=ustrip.(snr_filter), snr_thresh=snr_thresh,
                    line_latex=latex_i, modify_ax=(fig, ax[ci]), disable_colorbar=true, colorscale_limits=(vmin, vmax), marker=marker,
                    wave_unit=unit(cube_fitter.cube.λ[1]))
                ci += 1
            end
            if parameter == "eqw" && plot_total
                name_i = join([line, "total_eqw"], ".")
                bunit = get_label(param_maps, "lines.$(component_keys[1]).eqw")
                save_path = ""
                _, _, cdata = plot_parameter_map(total_eqw, name_i, bunit, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(ustrip(wave_i)),
                    cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=ustrip.(snr_filter), snr_thresh=snr_thresh,
                    line_latex=latex_i, modify_ax=(fig, ax[ci]), disable_colorbar=true, colorscale_limits=(vmin, vmax), marker=marker,
                    wave_unit=unit(cube_fitter.cube.λ[1]))
                ci += 1
            end

            for i in 1:n_line_comps
                data = ustrip.(get_val(param_maps, "lines.$(component_keys[i]).$parameter"))
                name_i = join([line, parameter], ".")
                bunit = get_label(param_maps, "lines.$(component_keys[i]).$parameter")
                snr_filt = ustrip.(get_val(param_maps, "lines.$(component_keys[i]).SNR"))
                if contains(parameter, "SNR")
                    snr_filt = nothing
                end
                save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$line", "$(name_i).pdf")
                _, _, cdata = plot_parameter_map(data, name_i, bunit, save_path, cube_fitter.cube.Ω, cube_fitter.z, 
                    psf_interp(ustrip(wave_i)), cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filt, 
                    snr_thresh=snr_thresh, line_latex=latex_i, modify_ax=(fig, ax[ci]), disable_colorbar=true, 
                    colorscale_limits=(vmin, vmax), marker=marker, wave_unit=unit(cube_fitter.cube.λ[1]))
                ci += 1
            end
            # Save the final figure
            name_final = join([line, parameter], ".")
            bunit_final = get_label(param_maps, "lines.$line.1.$parameter")
            # Add the colorbar to cax
            fig.colorbar(cdata, cax=cax, label=bunit_final)
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$line", "$(name_final).pdf")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        end

    end
end



"""
    plot_parameter_maps(cube_fitter, param_maps; [snr_thresh, qso3d])

Wrapper function for `plot_parameter_map`, iterating through all the parameters in a `CubeFitter`'s `ParamMaps` object
and creating 2D maps of them.

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the fitting options
- `param_maps::ParamMaps`: The ParamMaps object containing the parameter values
- `snr_thresh::Real`: The S/N threshold to be used when filtering the parameter maps by S/N, for those applicable
- `qso3d::Bool`: If set to true, only flux maps will be created (as when mapping QSO template parameters, these are
    the only parameters that it makes sense to map)
"""
function plot_parameter_maps(cube_fitter::CubeFitter, param_maps::ParamMaps; snr_thresh::Real=3.,
    qso3d::Bool=false)

    # Iterate over model parameters and make 2D maps
    @debug "Using solid angle $(cube_fitter.cube.Ω), redshift $(cube_fitter.z), cosmology $(cube_fitter.cosmology)"

    dust_features = model(cube_fitter).dust_features
    lines = model(cube_fitter).lines
    fopt = fit_options(cube_fitter)

    # Ineterpolate the PSF FWHM
    λunit = unit(cube_fitter.cube.λ[1])
    psf_interp = Spline1D(ustrip.(cube_fitter.cube.λ), ustrip.(cube_fitter.cube.psf), k=1)
    psf_med = median(ustrip.(cube_fitter.cube.psf))

    # Calculate the centroid
    data2d = sumdim(ustrip.(cube_fitter.cube.I), 3)
    _, mx = findmax(data2d)
    centroid = centroid_com(data2d[mx[1]-5:mx[1]+5, mx[2]-5:mx[2]+5]) .+ (mx.I .- 5) .- 1

    # Plot individual parameter maps
    for (i, parameter) ∈ enumerate(param_maps.parameters.names)

        data = ustrip.(param_maps.data[:, :, i])
        category = split(parameter, ".")[1]
        name_i = join(split(parameter, ".")[2:end], ".")
        bunit = param_maps.parameters.labels[i]

        if category == "continuum"
            subcategory = split(parameter, ".")[2]
            if subcategory ∈ ("stellar_populations", "stellar_kinematics")
                snr_filt = dropdims(nanmedian(cube_fitter.cube.I ./ cube_fitter.cube.σ, dims=3), dims=3)
            else
                snr_filt = nothing
            end
            psf = psf_med
            latex_i = nothing
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", category, "$(name_i).pdf")

        # Dust feature (PAH) parameters
        elseif category == "dust_features"
            df = split(parameter, ".")[2]  # get the name of the dust feature
            if "dust_features.$df.SNR" in param_maps.parameters.names
                snr_filt = ustrip.(get_val(param_maps, "dust_features.$df.SNR"))
            else
                snr_filt = ustrip.(get_val(param_maps, "dust_features.$df.total_snr"))
            end
            # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
            if "dust_features.$df.mean" in param_maps.parameters.names
                wave_i = nanmedian(ustrip.(get_val(param_maps, "dust_features.$df.mean"))) / (1 + cube_fitter.z) * λunit
            else
                wave_i = pah_name_to_float(df)
            end
            psf = psf_interp(ustrip(wave_i))
            # Create the name to annotate on the plot
            if df in string.(dust_features.config.all_feature_names)
                ind = findfirst(dust_features.config.all_feature_names .== df)
                latex_i = dust_features.config.all_feature_labels[ind]
            else
                ind = findfirst(string.(dust_features.names) .== df)
                latex_i = dust_features.labels[ind]
            end
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", category, "$(name_i).pdf")

        # Absorption features
        elseif category == "abs_features"
            ab = split(parameter, ".")[2]   # get the name of the absorption feature
            snr_filt = nothing
            # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
            wave_i = nanmedian(ustrip.(get_val(param_maps, "abs_features.$ab.mean"))) / (1 + cube_fitter.z) * λunit
            psf = psf_interp(ustrip(wave_i))
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", category, "$(name_i).pdf")

        # Lines
        elseif category == "lines"
            # Remove the component index from the line name
            line_key, line_comp = split(parameter, ".")[2:3]
            line_i = findfirst(lines.names .== Symbol(line_key))
            # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
            wave_i = uconvert(λunit, lines.λ₀[line_i])
            psf = psf_interp(ustrip(wave_i))
            latex_i = lines.labels[line_i]
            if isdigit(line_comp[1])
                # individual line components
                snr_filt = ustrip.(get_val(param_maps, "lines.$line_key.$line_comp.SNR"))
                if contains(parameter, "SNR")
                    snr_filt = nothing
                end
            else
                # composite line components
                snr_filt = ustrip.(get_val(param_maps, "lines.$line_key.total_snr"))
            end
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", category, "$line_key", "$(name_i).pdf")

        # Generic (continuum, extinction, templates, etc.) parameters
        else
            snr_filt = nothing
            psf = psf_med
            latex_i = nothing
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", category, "$(name_i).pdf")

        end

        # Skip all non-flux/amp maps if doing qso3d plotting
        if qso3d && !contains(parameter, "flux") && !contains(parameter, "amp")
            continue
        end

        plot_parameter_map(data, name_i, bunit, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf,
            cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=split(parameter, ".")[end] != "SNR" ? snr_filt : nothing, 
            snr_thresh=snr_thresh, line_latex=latex_i, marker=centroid, wave_unit=unit(cube_fitter.cube.λ[1]))
    end

    # Calculate a tau_9.7 map if using the "decompose" method
    if fopt.extinction_curve == "decompose"
        N_oli = exp10.(get_val(param_maps, "extinction.N_oli"))
        N_oli[.~isfinite.(N_oli)] .= 0.
        N_pyr = exp10.(get_val(param_maps, "extinction.N_pyr"))
        N_pyr[.~isfinite.(N_pyr)] .= 0.
        N_for = exp10.(get_val(param_maps, "extinction.N_for"))
        N_for[.~isfinite.(N_for)] .= 0.
        data = N_oli .* fopt.κ_abs[1](9.7) .+ N_oli .* N_pyr .* fopt.κ_abs[2](9.7) .+ N_oli .* N_for .* fopt.κ_abs[3](9.7)
        name_i = "tau_9_7"
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "extinction", "$(name_i).pdf")
        plot_parameter_map(data, name_i, L"$\tau_{9.7}$", save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
            cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid, wave_unit=unit(cube_fitter.cube.λ))
    end

    # Make combined plots for lines with multiple components
    plot_multiline_parameters(cube_fitter, param_maps, psf_interp, snr_thresh, centroid)

    # Total parameters for combined lines
    for comb_lines in lines.config.combined
        # Check to make sure the lines were actually fit
        if !all([ln in lines.names for ln in comb_lines])
            @warn "You have requested combined plots for the lines $comb_lines, but at least one of these lines was not fit!"
            continue
        end
        # Get all of the line names + additional components
        line_inds = [findfirst(name .== lines.names) for name in comb_lines]
        component_keys = [string(name) for name in comb_lines]
        tied_voff = tied_fwhm = true
        for (line_ind, line_name) in zip(line_inds, component_keys)
            for (i, profile) in enumerate(lines.profiles[line_ind])
                ties = profile.fit_parameters.ties
                voff_i = fast_indexin("lines.$line_name.$i.voff", profile.fit_parameters.names)
                fwhm_i = fast_indexin("lines.$line_name.$i.fwhm", profile.fit_parameters.names)
                if isnothing(ties[voff_i]) 
                    tied_voff = false
                end
                if isnothing(ties[fwhm_i])
                    tied_fwhm = false
                end
            end
        end

        # Generate a group name based on the lines in the group
        species = String[]
        for cln in comb_lines
            ln = string(cln)
            m = match(r"(_[0-9]+[Am])", ln)
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
        snr_filter = dropdims(nanmaximum(ustrip.(get_val(param_maps, ["lines.$comp.total_snr" for comp in component_keys])), dims=3), dims=3)
        wave_i = median([lines.λ₀[ind] for ind in line_inds])

        # Make a latex group name similar to the other group name
        species_ltx = unique([lines.labels[ind] for ind in line_inds])
        group_name_ltx = join(species_ltx, L"$+$")

        # Total Flux+EQW
        if fopt.lines_allow_negative
            total_flux = ustrip.(sum([get_val(param_maps, "lines.$comp.total_flux") for comp in component_keys]))
        else
            total_flux = log10.(sum([exp10.(get_val(param_maps, "lines.$comp.total_flux")) for comp in component_keys]))
        end
        total_eqw = ustrip.(sum([get_val(param_maps, "lines.$comp.total_eqw") for comp in component_keys]))
        for (nm_i, bu_i, total_i) in zip(["total_flux", "total_eqw"], 
                get_label(param_maps, ["lines.$(component_keys[1]).total_flux", "lines.$(component_keys[1]).total_eqw"]),
                    [total_flux, total_eqw])
            name_i = join([group_name, nm_i], ".")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf")
            plot_parameter_map(total_i, name_i, bu_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(ustrip(wave_i)),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=group_name_ltx, marker=centroid, wave_unit=unit(cube_fitter.cube.λ[1]))
        end

        # Voff and FWHM
        if tied_voff
            vpeak = get_val(param_maps, "lines.$(component_keys[1]).vpeak")
            name_i = join([group_name, "vpeak"], ".")
            bunit = get_label(param_maps, "lines.$(component_keys[1]).vpeak")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf") 
            plot_parameter_map(ustrip.(vpeak), name_i, bunit, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(ustrip(wave_i)),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=group_name_ltx, marker=centroid, wave_unit=unit(cube_fitter.cube.λ[1])) 
        end
        if tied_fwhm
            w80 = get_val(param_maps, "lines.$(component_keys[1]).w80")
            name_i = join([group_name, "w80"], ".")
            bunit = get_label(param_maps, "lines.$(component_keys[1]).w80")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf") 
            plot_parameter_map(ustrip.(w80), name_i, bunit, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(ustrip(wave_i)),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=group_name_ltx, marker=centroid, wave_unit=unit(cube_fitter.cube.λ[1])) 
        end
        
    end

    # Reduced chi^2 
    data = get_val(param_maps, "statistics.chi2") ./ get_val(param_maps, "statistics.dof")
    name_i = "reduced_chi2"
    save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "$(name_i).pdf")
    plot_parameter_map(data, name_i, L"$\tilde{\chi}^2$", save_path, cube_fitter.cube.Ω, cube_fitter.z, 
        ustrip(median(cube_fitter.cube.psf)), cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid,
        wave_unit=unit(cube_fitter.cube.λ[1]))

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

    if eltype(cube_fitter.cube.I) <: QPerFreq
        restframe_factor = 1 + cube_fitter.z
    else
        restframe_factor = 1 / (1 + cube_fitter.z)
    end

    for (full_data, title) ∈ zip([cube_fitter.cube.I .* restframe_factor, cube_model.model], ["DATA", "MODEL"])

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
        wave_rest = ustrip.(cube_fitter.cube.λ)
        data = full_data[:, :, 1]

        # Get average along the wavelength dimension
        datasum = sumdim(ustrip.(full_data), 3)
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
        ax2.text(wave_rest[length(wave_rest) ÷ 2], -8, L"$\lambda_{\rm rest}$", ha="center", va="center")
        # Annotate with wavelength value at the current time
        time_text = ax2.text(wave_rest[length(wave_rest) ÷ 2], 20, (@sprintf "%.3f" wave_rest[1]), ha="center", va="center")
        ax2.text(wave_rest[1], -8, (@sprintf "%.3f" wave_rest[1]), ha="center", va="center")
        ax2.text(wave_rest[end], -8, (@sprintf "%.3f" wave_rest[end]), ha="center", va="center")
        # plt.tight_layout()  ---> doesn't work for some reason

        # Loop over the wavelength axis and set the image data to the new slice for each frame
        output_file = joinpath("output_$(cube_fitter.name)", "$title.mp4")
        writer.setup(fig, output_file, dpi=300)
        for i ∈ axes(full_data, 3)
            data_i = ustrip.(full_data)[:, :, i] 
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
function write_fits(cube_fitter::CubeFitter, cube_data::NamedTuple, cube_model::CubeModel, param_maps::ParamMaps,
    param_units::Vector{String}; aperture::Union{Vector{<:Aperture.AbstractAperture},String,Nothing}=nothing, 
    nuc_temp_fit::Bool=false, nuc_spax::Union{Nothing,CartesianIndex}=nothing, qso3d::Bool=false)

    aperture_keys = []
    aperture_vals = []
    aperture_comments = []
    # If using an aperture, extract its properties 
    if eltype(aperture) <: Aperture.AbstractAperture

        # Get the name (giving the shape of the aperture: circular, elliptical, or rectangular)
        ap_shape = string(eltype(aperture))
  
        aperture_keys = String["AP_SHAPE", "AP_X", "AP_Y"]
        aperture_vals = Any[ap_shape, aperture[1].x, aperture[1].y]
        aperture_comments = String["The shape of the spectrum extraction aperture", "The x coordinate of the aperture",
            "The y coordinate of the aperture"]

        # Get the properties, i.e. radius for circular 
        if contains(ap_shape, "CircularAperture")
            append!(aperture_keys, ["AP_RADIUS", "AP_SCALE"])
            append!(aperture_vals, [aperture[1].r, aperture[end].r/aperture[1].r])
            append!(aperture_comments, ["Radius of aperture (pixels)", "Fractional aperture size increase over wavelength"])
        elseif contains(ap_shape, "EllipticalAperture")
            append!(aperture_keys, ["AP_A", "AP_B", "AP_ANGLE", "AP_SCALE"])
            append!(aperture_vals, [aperture[1].a, aperture[1].b, aperture[1].theta, aperture[end].a/aperture[1].a])
            append!(aperture_comments, ["Semimajor axis of aperture (pixels)", 
                "Semiminor axis of aperture (pixels)", "Aperture angle in deg.", "Fractional aperture size increase over wavelength"])
        elseif contains(ap_shape, "RectangularAperture")
            append!(aperture_keys, ["AP_W", "AP_H", "AP_ANGLE", "AP_SCALE"])
            append!(aperture_vals, [aperture[1].w, aperture[1].h, aperture[1].theta, aperture[end].w/aperture[1].w])
            append!(aperture_comments, ["Width of aperture (pixels)", 
                "Height of aperture (pixels)", "Aperture angle in deg.", "Fractional aperture size increase over wavelength"])
        elseif contains(ap_shape, "CircularAnnulus")
            append!(aperture_keys, ["AP_R_IN", "AP_R_OUT", "AP_SCALE"])
            append!(aperture_vals, [aperture[1].r_in, aperture[1].r_out, aperture[end].r_in/aperture[1].r_in])
            append!(aperture_comments, ["Inner radius of annulus (pixels)", "Outer radius of annulus (pixels)", "Fractional aperture size increase over wavelength"])
        elseif contains(ap_shape, "EllipticalAnnulus")
            append!(aperture_keys, ["AP_A_IN", "AP_A_OUT", "AP_B_IN", "AP_B_OUT", "AP_ANGLE", "AP_SCALE"])
            append!(aperture_vals, [aperture[1].a_in, aperture[1].a_out, aperture[1].b_in, aperture[1].b_out, aperture[1].theta, aperture[end].a_in/aperture[1].a_in])
            append!(aperture_comments, ["Inner semimajor axis of annulus (pixels)", "Outer semimajor axis of annulus (pixels)",
                "Inner semiminor axis of annulus (pixels)", "Outer semiminor axis of annulus (pixels)", "Annulus angle in deg.", 
                "Fractional aperture size increase over wavelength"])
        elseif contains(ap_shape, "RectangularAnnulus")
            append!(aperture_keys, ["AP_W_IN", "AP_W_OUT", "AP_H_IN", "AP_H_OUT", "AP_ANGLE", "AP_SCALE"])
            append!(aperture_vals, [aperture[1].w_in, aperture[1].w_out, aperture[1].h_in, aperture[1].h_out, aperture[1].theta, aperture[end].w_in/aperture[1].w_in])
            append!(aperture_comments, ["Inner width of annulus (pixels)", "Outer width of annulus (pixels)",
                "Inner height of annulus (pixels)", "Outer height of annulus (pixels)", "Aperture angle in deg.",
                "Fractional aperture size increase over wavelength"])    
        end

        # Also append the aperture area
        push!(aperture_keys, "AP_AREA")
        push!(aperture_vals, get_area(aperture[1]))
        push!(aperture_comments, "Area of aperture in pixels") 
    
    elseif aperture isa String

        n_pix = [sum(.~cube_fitter.cube.mask[:, :, i]) for i in axes(cube_fitter.cube.mask, 3)]
        aperture_keys = ["AP_SHAPE", "AP_AREA"]
        aperture_vals = Any["full_cube", median(n_pix[isfinite.(n_pix)])]
        aperture_comments = ["The shape of the spectrum extraction aperture", "Area of aperture in pixels"]

    elseif nuc_temp_fit

        aperture_keys = ["SPAXEL_X", "SPAXEL_Y"]
        aperture_vals = [nuc_spax.I[1], nuc_spax.I[2]] 
        aperture_comments = ["x coordinate of nuclear spaxel", "y coordinate of nuclear spaxel"]

    end

    # Header information
    hdr0_keys = Vector{String}(cat(["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "TARG_RA", "TARG_DEC",
                        "ROTANGLE", "NCHAN", "RESTFRAM", "MASKED", "VACWAVE", "LOGBIN", "DERED", "DATAMODL", "NAXIS"],
                        aperture_keys, dims=1))
    hdr1_keys_3d =  String["PIXAR_SR", "PIXAR_A2"]
    hdr1_keys_2d = copy(hdr1_keys_3d)
        
    hdr0_vals = cat(Any[cube_fitter.name, cube_fitter.z, string(cube_fitter.cube.channel), string(cube_fitter.cube.band), 
                    ustrip(cube_fitter.cube.α), ustrip(cube_fitter.cube.δ), ustrip(cube_fitter.cube.θ_sky), 
                    nchannels(cube_fitter.cube.spectral_region), cube_fitter.cube.rest_frame, cube_fitter.cube.masked, 
                    cube_fitter.cube.vacuum_wave, cube_fitter.cube.log_binned, cube_fitter.cube.dereddened, "IFUCubeModel", 0],
                    aperture_vals, dims=1)
    hdr1_vals_3d = Any[ustrip(cube_fitter.cube.Ω), ustrip(uconvert(u"arcsecond^2", cube_fitter.cube.Ω))]
    hdr1_vals_2d = copy(hdr1_vals_3d)
            
    hdr0_comments = Vector{String}(cat(["Target name", "Target redshift", "Channel", "Band",
                               "Right ascension of target (deg.)", "Declination of target (deg.)",
                               "rotation angle to sky axes", "number of individual wavelength channels/bands in the data", 
                               "data in rest frame?", "data masked?", "vacuum wavelengths?", "log binned?", "dereddened?", "data model", 
                               "Number of data axes"], aperture_comments, dims=1))
    hdr1_comments_3d = String["Nominal pixel area in steradians", "Nominal pixel area in arcsec^2"]
    hdr1_comments_2d = copy(hdr1_comments_3d)
    
    wave_unit_str = string(unit(cube_fitter.cube.λ[1]))
    wave_unit_str = replace(wave_unit_str, "Å" => "angstrom", 'μ' => 'u', ' ' => '.', '*' => '.')

    # WCS information to add to header -- do separately for 3D and 2D cases
    wcsstr_3d = WCS.to_header(cube_fitter.cube.wcs)
    for m in eachmatch(r"(.{8})\=(.+?)\ \/\ (.{45})", wcsstr_3d)
        key = strip(m[1])
        push!(hdr1_keys_3d, key)
        val = strip(m[2])
        if occursin("\'", val)
            val = replace(val, "\'" => "")
        else
            val = parse(Float64, val)
        end
        if key == "CTYPE3"
            val = "WAVE-TAB"   # overwrite to make sure we output as a tabular wavelength
        end
        if key == "CUNIT3"
            val = wave_unit_str
        end
        if key in ("CRPIX3", "CRVAL3", "CDELT3")
            val = 1.
        end
        push!(hdr1_vals_3d, val)
        comment = strip(m[3])
        push!(hdr1_comments_3d, comment)
    end
    wcsstr_2d = WCS.to_header(shrink_wcs_dimensions(cube_fitter.cube.wcs))
    for m in eachmatch(r"(.{8})\=(.+?)\ \/\ (.{45})", wcsstr_2d)
        key = strip(m[1])
        push!(hdr1_keys_2d, key)
        val = strip(m[2])
        if occursin("\'", val)
            val = replace(val, "\'" => "")
        else
            val = parse(Float64, val)
        end
        push!(hdr1_vals_2d, val)
        comment = strip(m[3])
        push!(hdr1_comments_2d, comment)
    end

    # Fill in the rest of the PC elements with defaults if they weren't included in the
    # to_header formulation of the WCS
    if any(cube_fitter.cube.wcs.pc .> 0)
        for i in 1:3
            for j in 1:3
                if !("PC$(i)_$(j)" in hdr1_keys_3d)
                    pcij = i == j ? 1. : 0.
                    push!(hdr1_keys_3d, "PC$(i)_$(j)")
                    push!(hdr1_vals_3d, pcij)
                    push!(hdr1_comments_3d, "Coordinate transformation matrix element")
                end
            end
        end
        for i in 1:2
            for j in 1:2
                if !("PC$(i)_$(j)" in hdr1_keys_2d)
                    pcij = i == j ? 1. : 0.
                    push!(hdr1_keys_2d, "PC$(i)_$(j)")
                    push!(hdr1_vals_2d, pcij)
                    push!(hdr1_comments_2d, "Coordinate transformation matrix element")
                end
            end
        end
    end

    # Add MJD-OBS = MJD-AVG because for some reason the JWST cubes dont have it, and it is necessary
    # for astropy's wcs module to perform pixel/world coordinate transforms with tabular wavelength data
    if ("MJD-OBS" ∉ hdr1_keys_3d)
        for alternate in ("MJD-AVG", "MJD-BEG", "MJD-END")
            if alternate in hdr1_keys_3d
                push!(hdr1_vals_3d, hdr1_vals_3d[findfirst(hdr1_keys_3d .== alternate)]) 
                push!(hdr1_keys_3d, "MJD-OBS")
                push!(hdr1_comments_3d, "[d] MJD of observation")
                break
            end
        end
    end
    if ("MJD-OBS" ∉ hdr1_keys_2d)
        for alternate in ("MJD-AVG", "MJD-BEG", "MJD-END")
            if alternate in hdr1_keys_2d
                push!(hdr1_vals_2d, hdr1_vals_2d[findfirst(hdr1_keys_2d .== alternate)]) 
                push!(hdr1_keys_2d, "MJD-OBS")
                push!(hdr1_comments_2d, "[d] MJD of observation")
                break
            end
        end
    end

    # Since we are using a WAVE-TAB header, extra values need to be added
    append!(hdr1_keys_3d, ["PS3_0", "PS3_1"])
    append!(hdr1_vals_3d, ["WAVELENGTH", "wave"])
    append!(hdr1_comments_3d, ["Coordinate table extension name", "Coordinate table column name"])

    hdr0 = FITSHeader(hdr0_keys, hdr0_vals, hdr0_comments)
    hdr1_3d = FITSHeader(hdr1_keys_3d, hdr1_vals_3d, hdr1_comments_3d)
    hdr1_2d = FITSHeader(hdr1_keys_2d, hdr1_vals_2d, hdr1_comments_2d)

    oopt = out_options(cube_fitter)
    if oopt.save_full_model
        write_fits_full_model(cube_fitter, cube_data, cube_model, hdr0, hdr1_3d, nuc_temp_fit; qso3d=qso3d)
    end

    # Create the 2D parameter map FITS file for the parameters and the errors
    for (index, param_data) ∈ enumerate([param_maps.data, param_maps.err_upp, param_maps.err_low])

        FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_$(nuc_temp_fit ? "nuc_" : "")parameter_" * 
            ("maps", "errs_low", "errs_upp")[index] * "$(qso3d ? "_3d" : "").fits"), "w") do f

            @debug "Writing 2D parameter map FITS HDUs"

            write(f, Vector{Int}(), header=hdr0)  # Primary HDU (empty)

            # Loop through parameters and write them to the fits file along with the header and units
            for (i, parameter) ∈ enumerate(param_maps.parameters.names)
                # Skip chi2 and dof for the error cubes
                if ((split(parameter, ".")[1] == "statistics") && (index != 1)) || split(parameter, ".")[end] == "SNR"
                    continue
                end
                good = isfinite.(param_data[:, :, i])
                data = ustrip.(param_data[:, :, i])
                name_i = uppercase(parameter)
                write(f, data; name=name_i, header=hdr1_2d)
                write_key(f[name_i], "BUNIT", param_units[i])
            end
              
            # Add another HDU for the voronoi bin map, if applicable
            if !isnothing(cube_fitter.cube.voronoi_bins)
                write(f, cube_fitter.cube.voronoi_bins; name="VORONOI_BINS", header=hdr1_2d)
            end
        end
    end
end


# Helper function for writing the output for a MIR cube model
function write_fits_full_model(cube_fitter::CubeFitter, cube_data::NamedTuple, cube_model::CubeModel,
    hdr0::FITSHeader, hdr1_3d::FITSHeader, nuc_temp_fit::Bool; qso3d::Bool=false)

    fopt = fit_options(cube_fitter)
    if eltype(cube_data.I) <: QPerFreq
        restframe_factor = 1 + cube_fitter.z
    else
        restframe_factor = 1 / (1 + cube_fitter.z)
    end
    λunit = replace(string(unit(cube_data.λ[1])), 'μ' => 'u')
    Iunit = replace(string(unit(cube_data.I[1])), 'μ' => 'u', "Å" => "angstrom")

    if !isnothing(cube_fitter.cube.voronoi_bins)
        out_data = ones(eltype(cube_data.I), size(cube_fitter.cube.I)).*NaN
        out_err  = ones(eltype(cube_data.σ), size(cube_fitter.cube.I)).*NaN
        for spaxel in CartesianIndices(size(out_data)[1:2])
            vbin = cube_fitter.cube.voronoi_bins[spaxel]
            if vbin > 0
                out_data[spaxel, :] .= cube_data.I[vbin, :]   # no division by npix is necessary since these are intensities
                out_err[spaxel, :]  .= cube_data.σ[vbin, :]
            end
        end
    else
        out_data = cube_data.I
        out_err  = cube_data.σ
    end

    # Create the 3D intensity model FITS file
    FITS(joinpath("output_$(cube_fitter.name)", 
                  "$(cube_fitter.name)_$(nuc_temp_fit ? "nuc_model" : "full_model")$(qso3d ? "_3d" : "").fits"), 
                  "w") do f

        @debug "Writing 3D model FITS HDUs"
        # Permute the wavelength axis here back to the third axis to be consistent with conventions

        write(f, Vector{Int}(), header=hdr0)                                                        # Primary HDU (empty)
        write(f, Float32.(ustrip.(out_data) .* restframe_factor); name="DATA", header=hdr1_3d)      # Raw data 
        write(f, Float32.(ustrip.(out_err)  .* restframe_factor); name="ERROR")                     # Error in the raw data
        write(f, Float32.(ustrip.(permutedims(cube_model.model, (2,3,1)))); name="MODEL")           # Full intensity model
        write_key(f["DATA"], "BUNIT", Iunit)
        write_key(f["ERROR"], "BUNIT", Iunit)
        write_key(f["MODEL"], "BUNIT", Iunit) 

        write(f, Float32.(permutedims(cube_model.extinction_stars, (2,3,1))); name="EXTINCTION.STARS") # Stellar extinction
        write(f, Float32.(permutedims(cube_model.extinction_gas, (2,3,1))); name="EXTINCTION.GAS")     # Gas extinction
        ext_names = fopt.extinction_curve == "decompose" ? 
            ["EXTINCTION.ABS_OLIVINE", "EXTINCTION.ABS_PYROXENE", "EXTINCTION.ABS_FORSTERITE"] : 
            ["EXTINCTION.ABS_SILICATES"]
        for r ∈ axes(cube_model.absorption_silicates, 4)                                               # Silicate absorption
            write(f, Float32.(permutedims(cube_model.absorption_silicates[:, :, :, r], (2,3,1))); name=ext_names[r])       
        end
        if fopt.fit_ch_abs
            write(f, Float32.(permutedims(cube_model.abs_ice, (2,3,1))); name="EXTINCTION.ABS_ICE")    # Ice Absorption model
            write(f, Float32.(permutedims(cube_model.abs_ch, (2,3,1))); name="EXTINCTION.ABS_CH")      # CH Absorption model
        end
        if fopt.fit_stellar_continuum
            write(f, Float32.(permutedims(ustrip.(cube_model.stellar), (2,3,1)));      # Stellar population models
                name="CONTINUUM.STELLAR_POPULATIONS")   
            write_key(f["CONTINUUM.STELLAR_POPULATIONS"], "BUNIT", Iunit)
        end
        if fopt.fit_opt_na_feii
            write(f, Float32.(permutedims(ustrip.(cube_model.na_feii), (2,3,1))); name="CONTINUUM.FEII.NA")  # Narrow Fe II emission
            write_key(f["CONTINUUM.FEII.NA"], "BUNIT", Iunit)
        end
        if fopt.fit_opt_br_feii
            write(f, Float32.(permutedims(ustrip.(cube_model.br_feii), (2,3,1))); name="CONTINUUM.FEII.BR")  # Broad Fe II emission
            write_key(f["CONTINUUM.FEII.BR"], "BUNIT", Iunit)
        end
        for j in 1:size(cube_model.power_law, 4)
            write(f, Float32.(permutedims(ustrip.(cube_model.power_law[:, :, :, j]), (2,3,1))); name="CONTINUUM.POWER_LAW.$j") # Power laws
            write_key(f["CONTINUUM.POWER_LAW.$j"], "BUNIT", Iunit)
        end
        for i ∈ 1:size(cube_model.dust_continuum, 4)
            write(f, Float32.(permutedims(ustrip.(cube_model.dust_continuum[:, :, :, i]), (2,3,1))); name="CONTINUUM.DUST.$i")   # Dust continua
            write_key(f["CONTINUUM.DUST.$i"], "BUNIT", Iunit)
        end
        if fopt.fit_sil_emission
            write(f, Float32.(permutedims(ustrip.(cube_model.hot_dust), (2,3,1))); name="CONTINUUM.HOT_DUST") # Hot dust model
            write_key(f["CONTINUUM.HOT_DUST"], "BUNIT", Iunit)
        end
        for (q, tp) ∈ enumerate(cube_fitter.template_names)
            tpu = uppercase("$tp")
            write(f, Float32.(permutedims(ustrip.(cube_model.templates[:, :, :, q]), (2,3,1))); name="TEMPLATES.$tpu") # Template profiles
            write_key(f["TEMPLATES.$tpu"], "BUNIT", Iunit)
        end

        for (j, df) ∈ enumerate(model(cube_fitter).dust_features.names)
            dfu = uppercase("$df")
            write(f, Float32.(permutedims(ustrip.(cube_model.dust_features[:, :, :, j]), (2,3,1))); name="DUST_FEATURES.$dfu") # Dust feature profiles
            write_key(f["DUST_FEATURES.$dfu"], "BUNIT", Iunit)
        end
        for (m, ab) ∈ enumerate(model(cube_fitter).abs_features.names)                                     
            abu = uppercase("$ab")
            write(f, Float32.(permutedims(ustrip.(cube_model.abs_features[:, :, :, m]), (2,3,1))); name="ABS_FEATURES.$abu") # Absorption feature profiles
            write_key(f["ABS_FEATURES.$abu"], "BUNIT", Iunit)
        end
        for (k, line) ∈ enumerate(model(cube_fitter).lines.names)
            lnu = uppercase("$line")
            write(f, Float32.(permutedims(ustrip.(cube_model.lines[:, :, :, k]), (2,3,1))); name="LINES.$lnu")  # Emission line profiles
            write_key(f["LINES.$lnu"], "BUNIT", Iunit)
        end

        if !isnothing(cube_fitter.cube.voronoi_bins)
            write(f, cube_fitter.cube.voronoi_bins; name="VORONOI_BINS")  # Voronoi bin indices
        end
        
        write(f, ["wave"], [reshape(ustrip.(cube_data.λ) .* (1 .+ cube_fitter.z), (1, length(cube_data.λ), 1))],  # wavelength vector
            hdutype=TableHDU, name="WAVELENGTH", units=Dict(:wave => λunit))
        write_key(f["WAVELENGTH"], "TDIM2", "(1,$(length(cube_data.λ)))", "Wavetable dimension")
    end
end


"""
    write_table(cube_fitter, cube_data, param_maps)

Save the best fit results for the cube into a simple CSV table file for each spaxel, much like the files 
that are saved in the "spaxel_binaries" folder to save the progress of the fitting results.  But these tables 
will instead hold the "canonical" outputs of the code, with all quantities in physical units and transformed 
to the observed frame.
"""
function write_table(cube_fitter::CubeFitter, param_maps::ParamMaps, param_units::Vector{String})

    param_names = param_maps.parameters.names

    tblpath = joinpath("output_$(cube_fitter.name)", "output_tables")
    if !isdir(tblpath)
        mkdir(tblpath)
    end

    # loop over spaxels
    for spaxel in CartesianIndices(size(param_maps.data)[1:2])

        # skip spaxels that aren't fit
        if all(.~isfinite.(param_maps.data[spaxel,:]))
            continue
        end

        # we can immediately get 1d vectors of the parameter values, errors, and units
        best  = ustrip.(param_maps.data[spaxel,:])
        err_l = ustrip.(param_maps.err_low[spaxel,:])
        err_u = ustrip.(param_maps.err_upp[spaxel,:])
        units = param_units

        # to get the lower/upper limits and locked booleans, we read back in the binary file
        if !isnothing(cube_fitter.cube.voronoi_bins)
            bin_index = cube_fitter.cube.voronoi_bins[spaxel]
            fname = "voronoi_bin_$(bin_index)"
        else
            fname = "spaxel_$(spaxel[1])_$(spaxel[2])"
        end
        folder = "output_$(cube_fitter.name)"
        df = CSV.read(joinpath(folder, "spaxel_binaries", "$fname.csv"), DataFrame, delim='\t', stripwhitespace=true)
        rename!(df, strip.(names(df)))

        # read in the lower/upper bounds, locked vector, and tied vector
        # bound_l = [strip(b) == "" ? NaN : parse(Float64, b) for b in df[!, "bound_lower"]]
        # bound_u = [strip(b) == "" ? NaN : parse(Float64, b) for b in df[!, "bound_upper"]]
        locked_fit  = df[!, "locked"]
        tied_fit    = df[!, "tied"]

        # pad for the extra non-fit parameters
        locked = cat(locked_fit, repeat([""], length(best)-length(locked_fit)), dims=1)
        tied   = cat(tied_fit,   repeat([""], length(best)-length(locked_fit)), dims=1)

        # make a dataframe
        data = DataFrame(name=param_names, best=best, error_lower=err_l, error_upper=err_u, unit=units, 
            locked=locked, tied=tied)
        textwidths = [maximum(textwidth.(string.([data[:, i]; names(data)[i]]))) for i in axes(data, 2)]
        # convert to a string with nicely padded column widths
        msg = ""
        for (i, header) ∈ enumerate(names(data))
            msg *= rpad(header, textwidths[i]) * "\t"
        end
        msg *= "\n"
        for i ∈ axes(data, 1)
            for j ∈ axes(data, 2)
                msg *= rpad(data[i,j], textwidths[j]) * "\t"
            end
            msg *= "\n"
        end
        @debug msg

        # write the output table
        open(joinpath(tblpath, "$fname.final.csv"), "w") do f
            write(f, msg)
        end

    end

end

