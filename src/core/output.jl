############################## OUTPUT / SAVING FUNCTIONS ####################################

const ascii_lowercase = "abcdefghijklmnopqrstuvwxyz"


"""
    sort_line_components!(cube_fitter, params)

A helper function that sorts line parameters based on a sorting criterion (flux, FWHM, voff, etc.) so that
the bootstrapped values are correct. This method sorts the parameters before they have been sorted into cubes.
This is necessary before taking the 50th, 16th, and 84th percentiles of the parameters since they may fit 
different line components during different bootstrap iterations, forming bimodal distributions. This function
sorts the parameters first to prevent this from happening.
"""
function sort_line_components!(cube_fitter::CubeFitter, params::Vector{<:Real}; mask_zeros::Bool=true)

    # Do not sort
    if isnothing(cube_fitter.sort_line_components)
        return
    end
    if any(cube_fitter.relative_flags)
        return
    end

    pᵢ = cube_fitter.n_params_cont + 1
    pⱼ = pᵢ + cube_fitter.n_params_lines + (cube_fitter.spectral_region == :MIR ? 3cube_fitter.n_dust_feat : 0)

    for k ∈ 1:cube_fitter.n_lines
        amps = Int64[]
        voffs = Int64[]
        voff_indivs = Int64[]
        fwhms = Int64[]
        h3s = Int64[]
        h4s = Int64[]
        etas = Int64[]
        fluxes = Int64[]
        eqws = Int64[]
        snrs = Int64[]
        n_prof = 0
        @assert all(cube_fitter.lines.profiles[k,:][.!isnothing.(cube_fitter.lines.profiles[k,:])] .== 
            cube_fitter.lines.profiles[k,1]) "All line profiles must be the same to use sorted bootstrapping"

        for j ∈ 1:cube_fitter.n_comps
            if isnothing(cube_fitter.lines.profiles[k,j])
                continue
            end
            n_prof += 1

            # set parameters to nan so they are ignored when calculating the percentiles
            mask_line = (params[pᵢ] == 0.) && (j > 1) && mask_zeros
            if mask_line
                # amplitude is not overwritten so that the zeros downweight the final amplitude (same for flux, eqw, snr)
                params[pᵢ+1] = NaN
            end

            push!(amps, pᵢ)
            push!(voffs, pᵢ+1)
            if !isnothing(cube_fitter.lines.tied_voff[k, j]) && cube_fitter.flexible_wavesol && isone(j)
                error("Cannot use the flexible_wavesol option with sorted bootstrapping!")
                # push!(voff_indivs, pᵢ+2)
                # push!(fwhms, pᵢ+3)
                # pᵢ += 4
            else
                push!(fwhms, pᵢ+2)
                if mask_line
                    params[pᵢ+2] = NaN
                end
                pᵢ += 3
            end
            if cube_fitter.lines.profiles[k,j] == :GaussHermite
                push!(h3s, pᵢ)
                push!(h4s, pᵢ+1)
                if mask_line
                    params[pᵢ:pᵢ+1] .= NaN
                end
                pᵢ += 2
            elseif cube_fitter.lines.profiles[k,j] == :Voigt
                push!(etas, pᵢ)
                if mask_line
                    params[pᵢ] = NaN
                end
                pᵢ += 1
            end
            push!(fluxes, pⱼ)
            push!(eqws, pⱼ+1)
            push!(snrs, pⱼ+2)
            pⱼ += 3
        end
        # Dont forget to count the composite line parameters
        pⱼ += 4

        # Dont need to sort if there is only 1 profile
        if n_prof < 2
            continue
        end

        # always sort by voff for bootstrap iterations!
        if cube_fitter.sort_line_components == :flux
            sort_quantity = params[fluxes]
        elseif cube_fitter.sort_line_components == :amp
            sort_quantity = params[amps]
        elseif cube_fitter.sort_line_components == :fwhm
            sort_quantity = params[fwhms]
        elseif cube_fitter.sort_line_components == :voff
            sort_quantity = params[voffs]
        else
            error("Unrecognized sorting quantity: $(cube_fitter.sort_line_components)")
        end

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

    if isnothing(cube_fitter.sort_line_components)
        return
    end
    if any(cube_fitter.relative_flags)
        @warn "Skipping line component sorting due to a relative line parameter flag being set!"
        return
    end

    for k ∈ 1:cube_fitter.n_lines
        prefix = "lines.$(cube_fitter.lines.names[k])"
        n_prof = get(param_maps, "$(prefix).n_comps")[index]
        n_prof = isfinite(n_prof) ? floor(Int, n_prof) : n_prof
        if (n_prof <= 1) || !isfinite(n_prof)
            @debug "Skipping line component sorting for $prefix due to 1 or fewer profiles in this spaxel"
            continue
        end
        if !all(cube_fitter.lines.profiles[k,1:n_prof] .== cube_fitter.lines.profiles[k,1])
            @warn "Skipping line component sorting for $prefix because it is not supported for different line profiles!"
            continue
        end
        if cube_fitter.flexible_wavesol
            @warn "Skipping line component sorting for $prefix because it is not supported if the flexible_wavesol option is enabled!"
            continue
        end
        snrs = param_maps.data[index, [findfirst(param_maps.names .== "$(prefix).$(j).SNR") for j in 1:n_prof]]
        if cube_fitter.sort_line_components == :flux
            sort_inds = ["$(prefix).$(j).flux" for j in 1:n_prof]
        elseif cube_fitter.sort_line_components == :amp
            sort_inds = ["$(prefix).$(j).amp" for j in 1:n_prof]
        elseif cube_fitter.sort_line_components == :fwhm
            sort_inds = ["$(prefix).$(j).fwhm" for j in 1:n_prof]
        elseif cube_fitter.sort_line_components == :voff
            sort_inds = ["$(prefix).$(j).voff" for j in 1:n_prof]
        else
            error("Unrecognized sorting quantity: $(cube_fitter.sort_line_components)")
        end
        sort_quantity = get(param_maps, index, sort_inds)

        # Check the SNRs of each component
        bad = findall(snrs .< cube_fitter.map_snr_thresh)
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
        params_to_sort = ["amp", "voff", "fwhm"]
        if cube_fitter.lines.profiles[k,1] == :GaussHermite
            append!(params_to_sort, ["h3", "h4"])
        end
        if cube_fitter.lines.profiles[k,1] == :Voigt
            append!(params_to_sort, ["mixing"])
        end
        append!(params_to_sort, ["flux", "eqw", "SNR"])
        for ps in params_to_sort
            for d in (param_maps.data, param_maps.err_upp, param_maps.err_low)
                param_inds = [findfirst(param_maps.names .== "$(prefix).$(j).$(ps)") for j in 1:n_prof]
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
    temp_inds = [findfirst(param_maps.names .== "continuum.dust.$(i).temp") for i in 1:cube_fitter.n_dust_cont]
    amp_inds = [findfirst(param_maps.names .== "continuum.dust.$(i).amp") for i in 1:cube_fitter.n_dust_cont]

    temps = param_maps.data[index, temp_inds]
    ss = sortperm(temps, rev=true)

    # Sort the dust parameters
    for d in (param_maps.data, param_maps.err_upp, param_maps.err_low)
        d[index, temp_inds] .= d[index, temp_inds[ss]]
        d[index, amp_inds] .= d[index, amp_inds[ss]]
    end

end


"""
    assign_outputs(out_params, out_errs, cube_fitter, cube_data, spaxels, z[, aperture])

Create ParamMaps objects for the parameter values and errors, and a CubeModel object for the full model, and
fill them with the maximum likelihood values and errors given by out_params and out_errs over each spaxel in
spaxels.
"""
assign_outputs(out_params::AbstractArray{<:Real}, out_errs::AbstractArray{<:Real}, cube_fitter::CubeFitter, 
    cube_data::NamedTuple, z::Real, aperture::Bool=false; kwargs...) = 
    cube_fitter.spectral_region == :MIR ? 
        assign_outputs_mir(out_params, out_errs, cube_fitter, cube_data, z, aperture; kwargs...) : 
        assign_outputs_opt(out_params, out_errs, cube_fitter, cube_data, z, aperture; kwargs...)


"""
    plot_parameter_map(data, name_i, save_path, Ω, z, psf_fwhm, cosmo, python_wcs; 
        [snr_filter, snr_thresh, cmap, line_latex])

Plotting function for 2D parameter maps which are output by `fit_cube!`

# Arguments
- `data::Matrix{Float64}`: The 2D array of data to be plotted
- `name_i::String`: The name of the individual parameter being plotted, i.e. "dust_features_PAH_5.24_amp"
- `bunit::AbstractString`: A label to apply to the colorbar
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
function plot_parameter_map(data::Matrix{Float64}, name_i::String, bunit::AbstractString, save_path::String, Ω::Float64, z::Float64, 
    psf_fwhm::Float64, cosmo::Cosmology.AbstractCosmology, wcs::Union{WCSTransform,Nothing}; snr_filter::Union{Nothing,Matrix{Float64}}=nothing, 
    snr_thresh::Float64=3., abs_thresh::Union{Float64,Nothing}=nothing, cmap=py_colormap.cubehelix, line_latex::Union{String,Nothing}=nothing, 
    marker::Union{Vector{<:Real},Nothing}=nothing, disable_axes::Bool=true, disable_colorbar::Bool=false, modify_ax=nothing, colorscale_limits=nothing, 
    custom_bunit::Union{AbstractString,Nothing}=nothing)

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
    if occursin("voff", name_i) || occursin("index", name_i) || occursin("vel", name_i) || occursin("delta_v", name_i) || occursin("vmed", name_i)
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
    if (occursin("fwhm", name_i) || occursin("eqw", name_i) || occursin("vdisp", name_i) || occursin("w80", name_i)) && 
        cmap == py_colormap.cubehelix

        cmap = py_colormap.magma
    end
    # get discrete colormap for number of line components
    if occursin("n_comps", name_i) && cmap == py_colormap.cubehelix
        n_comps = nanmaximum(filtered)
        if !isfinite(n_comps)
            n_comps = 1
        else
            n_comps = floor(Int, n_comps)
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
    # l = d * theta (") where theta is chosen as 1/5 the horizontal extent of the image
    l = dA * (size(data, 1) * pix_as / 5) * π/180 / 3600  
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
    scalebar_text = cosmo.h ≈ 1.0 ? L"%$l$h^{-1}$ %$unit" : L"$%$l$ %$unit"
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
                n_comps = floor(Int, n_comps)
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

    for (i, line) ∈ enumerate(cube_fitter.lines.names)

        # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
        wave_i = cube_fitter.lines.λ₀[i]
        latex_i = cube_fitter.lines.latex[i]
        n_line_comps = sum(.!isnothing.(cube_fitter.lines.profiles[i, :]))
        component_keys = [string(line) * ".$(j)" for j in 1:n_line_comps]
        if n_line_comps < 2
            # Dont bother for lines with only 1 component
            continue
        end
        snr_filter = dropdims(nanmaximum(get(param_maps, ["lines.$comp.SNR" for comp in component_keys]), dims=3), dims=3)

        # Get the total flux and eqw for lines with multiple components
        plot_total = false
        if n_line_comps > 1
            plot_total = true
            total_flux = log10.(sum([exp10.(get(param_maps, "lines.$comp.flux")) for comp in component_keys])) 
            total_eqw = sum([get(param_maps, "lines.$comp.eqw") for comp in component_keys])
        end

        parameters = unique([split(pname, ".")[end] for pname in param_maps.names if contains(pname, "lines.$line.1")])       
       
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
                minmax = dropdims(nanextrema(get(param_maps, ["lines.$comp.$parameter" for comp in component_keys]), dims=3), dims=3)
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
                if parameter in ("voff", "voff_indiv")
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
                _, _, cdata = plot_parameter_map(total_flux, name_i, bunit, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                    line_latex=latex_i, modify_ax=(fig, ax[ci]), disable_colorbar=true, colorscale_limits=(vmin, vmax), marker=marker)
                ci += 1
            end
            if parameter == "eqw" && plot_total
                name_i = join([line, "total_eqw"], ".")
                bunit = get_label(param_maps, "lines.$(component_keys[1]).eqw")
                save_path = ""
                _, _, cdata = plot_parameter_map(total_eqw, name_i, bunit, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                    cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                    line_latex=latex_i, modify_ax=(fig, ax[ci]), disable_colorbar=true, colorscale_limits=(vmin, vmax), marker=marker)
                ci += 1
            end

            for i in 1:n_line_comps
                data = get(param_maps, "lines.$(component_keys[i]).$parameter")
                name_i = join([line, parameter], ".")
                bunit = get_label(param_maps, "lines.$(component_keys[i]).$parameter")
                snr_filt = get(param_maps, "lines.$(component_keys[i]).SNR")
                if contains(parameter, "SNR")
                    snr_filt = nothing
                end
                save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$line", "$(name_i).pdf")
                _, _, cdata = plot_parameter_map(data, name_i, bunit, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i), 
                    cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filt, snr_thresh=snr_thresh, 
                    line_latex=latex_i, modify_ax=(fig, ax[ci]), disable_colorbar=true, colorscale_limits=(vmin, vmax),
                    marker=marker)
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
    plot_parameter_maps(cube_fitter, param_maps; [snr_thresh])

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

    # Ineterpolate the PSF FWHM
    psf_interp = Spline1D(cube_fitter.cube.λ, cube_fitter.cube.psf, k=1)
    psf_med = median(cube_fitter.cube.psf)

    # Calculate the centroid
    data2d = sumdim(cube_fitter.cube.I, 3)
    _, mx = findmax(data2d)
    centroid = centroid_com(data2d[mx[1]-5:mx[1]+5, mx[2]-5:mx[2]+5]) .+ (mx.I .- 5) .- 1

    # Plot individual parameter maps
    for (i, parameter) ∈ enumerate(param_maps.names)

        data = param_maps.data[:, :, i]
        category = split(parameter, ".")[1]
        name_i = join(split(parameter, ".")[2:end], ".")
        bunit = param_maps.labels[i]

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
            snr_filt = get(param_maps, "dust_features.$df.SNR")
            # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
            wave_i = nanmedian(get(param_maps, "dust_features.$df.mean")) / (1 + cube_fitter.z)
            psf = psf_interp(wave_i)
            # Create the name to annotate on the plot
            ind = findfirst(string.(cube_fitter.dust_features.names) .== df)
            comp = cube_fitter.dust_features.complexes[ind]
            # always match by assuming the last two digits are after the decimal and everything else is before the decimal
            latex_i = replace(df, r"([0-9]+)([0-9][0-9])" => s"\1.\2")
            latex_i = replace(latex_i, "_" => " ") * L" $\mu$m"
            if !isnothing(comp)
                comp_name = L"PAH %$comp $\mu$m"
                indiv_inds = findall(cube_fitter.dust_features.complexes .== comp)
                if length(indiv_inds) > 1
                    # will already be sorted
                    indivs = [cube_fitter.dust_features.names[i] for i ∈ indiv_inds]
                    out_ind = findfirst(indivs .== df)
                    latex_i = comp_name * " " * ascii_lowercase[out_ind]
                end
            end
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", category, "$(name_i).pdf")

        # Absorption features
        elseif category == "abs_features"
            ab = split(parameter, ".")[2]   # get the name of the absorption feature
            snr_filt = nothing
            # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
            wave_i = nanmedian(get(param_maps, "abs_features.$ab.mean")) / (1 + cube_fitter.z)
            psf = psf_interp(wave_i)
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", category, "$(name_i).pdf")

        # Lines
        elseif category == "lines"
            # Remove the component index from the line name
            line_key, line_comp = split(parameter, ".")[2:3]
            line_i = findfirst(cube_fitter.lines.names .== Symbol(line_key))
            # Find the wavelength/index at which to get the PSF FWHM for the circle in the plot
            wave_i = cube_fitter.lines.λ₀[line_i]
            psf = psf_interp(wave_i)
            latex_i = cube_fitter.lines.latex[line_i]
            if isdigit(line_comp[1])
                # individual line components
                snr_filt = get(param_maps, "lines.$line_key.$line_comp.SNR")
                if contains(parameter, "SNR")
                    snr_filt = nothing
                end
            else
                # composite line components
                line_ind = findfirst(cube_fitter.lines.names .== Symbol(line_key))
                component_keys = [line_key * ".$(j)" for j in 1:sum(.~isnothing.(cube_fitter.lines.profiles[line_ind, :]))]
                snr_filt = dropdims(nanmaximum(get(param_maps, ["lines.$comp.SNR" for comp in component_keys]), dims=3), dims=3)
            end
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", category, "$line_key", "$(name_i).pdf")

        # Generic (continuum, extinction, templates, etc.) parameters
        else
            snr_filt = nothing
            psf = psf_med
            latex_i = nothing
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", category, "$(name_i).pdf")

        end

        plot_parameter_map(data, name_i, bunit, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf,
            cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=split(parameter, ".")[end] != "SNR" ? snr_filt : nothing, 
            snr_thresh=snr_thresh, line_latex=latex_i, marker=centroid)
    end

    # Total parameters for PAH complexes
    dust_complexes = []
    if cube_fitter.spectral_region == :MIR
        dust_complexes = unique(cube_fitter.dust_features.complexes[.!isnothing.(cube_fitter.dust_features.complexes)])
    end
    for dust_complex in dust_complexes
        # Get the components that make up the complex
        indiv_inds = findall(cube_fitter.dust_features.complexes .== dust_complex)
        indivs = [cube_fitter.dust_features.names[i] for i ∈ indiv_inds]
        snr = dropdims(nanmaximum(get(param_maps, ["dust_features.$df.SNR" for df ∈ indivs]), dims=3), dims=3)

        # Sum up individual component fluxes and equivalent widths
        total_flux = log10.(sum([exp10.(get(param_maps, "dust_features.$df.flux")) for df ∈ indivs]))
        total_eqw = sum([get(param_maps, "dust_features.$df.eqw") for df ∈ indivs])

        # Wavelength and name
        wave_i = parse(Float64, dust_complex)
        comp_name = "PAH $dust_complex " * L"$\mu$m"

        for (name_i, bunit_i, total_i) ∈ zip(["complex_$(dust_complex).total_flux", "complex_$(dust_complex).total_eqw"],
                                    get_label(param_maps, ["dust_features.$(indivs[1]).flux", "dust_features.$(indivs[1]).eqw"]),
                                    [total_flux, total_eqw])
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "dust_features", "$(name_i).pdf")
            plot_parameter_map(total_i, name_i, bunit_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, line_latex=comp_name, snr_filter=snr, snr_thresh=snr_thresh,
                marker=centroid)
        end
    end

    # Calculate a tau_9.7 map if using the "decompose" method
    if cube_fitter.extinction_curve == "decompose"
        N_oli = exp10.(get(param_maps, "extinction.N_oli"))
        N_oli[.~isfinite.(N_oli)] .= 0.
        N_pyr = exp10.(get(param_maps, "extinction.N_pyr"))
        N_pyr[.~isfinite.(N_pyr)] .= 0.
        N_for = exp10.(get(param_maps, "extinction.N_for"))
        N_for[.~isfinite.(N_for)] .= 0.
        data = N_oli .* cube_fitter.κ_abs[1](9.7) .+ N_oli .* N_pyr .* cube_fitter.κ_abs[2](9.7) .+ N_oli .* N_for .* cube_fitter.κ_abs[3](9.7)
        name_i = "tau_9_7"
        save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "extinction", "$(name_i).pdf")
        plot_parameter_map(data, name_i, L"$\tau_{9.7}$", save_path, cube_fitter.cube.Ω, cube_fitter.z, median(cube_fitter.cube.psf),
            cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)
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
        component_keys = String[]
        line_inds = [findfirst(name .== cube_fitter.lines.names) for name in comb_lines]
        for (ind, name) in zip(line_inds, comb_lines)
            n_line_comps = sum(.!isnothing.(cube_fitter.lines.profiles[ind, :]))
            append!(component_keys, [string(name) * ".$(j)" for j in 1:n_line_comps])
        end
        # Generate a group name based on the lines in the group
        species = String[]
        for cln in comb_lines
            ln = string(cln)
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
        snr_filter = dropdims(nanmaximum(get(param_maps, ["lines.$comp.SNR" for comp in component_keys]), dims=3), dims=3)
        wave_i = median([cube_fitter.lines.λ₀[ind] for ind in line_inds])

        # Make a latex group name similar to the other group name
        species_ltx = unique([cube_fitter.lines.latex[ind] for ind in line_inds])
        group_name_ltx = join(species_ltx, L"$+$")

        # Total Flux+EQW
        total_flux = log10.(sum([exp10.(get(param_maps, "lines.$comp.flux")) for comp in component_keys]))
        total_eqw = sum([get(param_maps, "lines.$comp.eqw") for comp in component_keys])
        for (nm_i, bu_i, total_i) in zip(["total_flux", "total_eqw"], 
                get_label(param_maps, ["lines.$(component_keys[1]).flux", "lines.$(component_keys[1]).eqw"]),
                [total_flux, total_eqw])
            name_i = join([group_name, nm_i], ".")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf")
            plot_parameter_map(total_i, name_i, bu_i, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=group_name_ltx, marker=centroid)
        end

        # Voff and FWHM
        if all(.!isnothing.(cube_fitter.lines.tied_voff[line_inds, 1]))
            voff = get(param_maps, "lines.$(component_keys[1]).voff")
            name_i = join([group_name, "voff"], ".")
            bunit = get_label(param_maps, "lines.$(component_keys[1]).voff")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf") 
            plot_parameter_map(voff, name_i, bunit, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=group_name_ltx, marker=centroid) 
        end
        if all(.!isnothing.(cube_fitter.lines.tied_fwhm[line_inds, 1]))
            fwhm = get(param_maps, "lines.$(component_keys[1]).fwhm")
            name_i = join([group_name, "fwhm"], ".")
            bunit = get_label(param_maps, "lines.$(component_keys[1]).fwhm")
            save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "lines", "$(group_name)", "$(name_i).pdf") 
            plot_parameter_map(fwhm, name_i, bunit, save_path, cube_fitter.cube.Ω, cube_fitter.z, psf_interp(wave_i),
                cube_fitter.cosmology, cube_fitter.cube.wcs, snr_filter=snr_filter, snr_thresh=snr_thresh,
                line_latex=group_name_ltx, marker=centroid) 
        end
        
    end

    # Reduced chi^2 
    data = get(param_maps, "statistics.chi2") ./ get(param_maps, "statistics.dof")
    name_i = "reduced_chi2"
    save_path = joinpath("output_$(cube_fitter.name)", "param_maps", "$(name_i).pdf")
    plot_parameter_map(data, name_i, L"$\tilde{\chi}^2$", save_path, cube_fitter.cube.Ω, cube_fitter.z, 
        median(cube_fitter.cube.psf), cube_fitter.cosmology, cube_fitter.cube.wcs, marker=centroid)

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
function write_fits(cube_fitter::CubeFitter, cube_data::NamedTuple, cube_model::CubeModel, param_maps::ParamMaps;
    aperture::Union{Vector{<:Aperture.AbstractAperture},String,Nothing}=nothing, nuc_temp_fit::Bool=false,
    nuc_spax::Union{Nothing,CartesianIndex}=nothing)

    aperture_keys = []
    aperture_vals = []
    aperture_comments = []
    # If using an aperture, extract its properties 
    if eltype(aperture) <: Aperture.AbstractAperture

        # Get the name (giving the shape of the aperture: circular, elliptical, or rectangular)
        ap_shape = string(eltype(aperture))
  
        aperture_keys = ["AP_SHAPE", "AP_X", "AP_Y"]
        aperture_vals = Any[ap_shape, aperture[1].x, aperture[1].y]
        aperture_comments = ["The shape of the spectrum extraction aperture", "The x coordinate of the aperture",
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
        if cube_fitter.spectral_region == :MIR
            write_fits_full_model_mir(cube_fitter, cube_data, cube_model, hdr, nuc_temp_fit)
        else
            write_fits_full_model_opt(cube_fitter, cube_data, cube_model, hdr, nuc_temp_fit)
        end
    end

    # Create the 2D parameter map FITS file for the parameters and the errors
    for (index, param_data) ∈ enumerate([param_maps.data, param_maps.err_upp, param_maps.err_low])

        FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_$(nuc_temp_fit ? "nuc_" : "")parameter_" * 
            ("maps", "errs_low", "errs_upp")[index] * ".fits"), "w") do f

            @debug "Writing 2D parameter map FITS HDUs"

            write(f, Vector{Int}())  # Primary HDU (empty)

            # Loop through parameters and write them to the fits file along with the header and units
            for (i, parameter) ∈ enumerate(param_maps.names)
                # Skip chi2 and dof for the error cubes
                if (split(parameter, ".")[1] == "statistics") && (index != 1)
                    continue
                end
                data = param_data[:, :, i]
                name_i = uppercase(parameter)
                write(f, data; name=name_i, header=hdr)
                write_key(f[name_i], "BUNIT", param_maps.units[i])
            end
              
            # Add another HDU for the voronoi bin map, if applicable
            if !isnothing(cube_fitter.cube.voronoi_bins)
                write(f, cube_fitter.cube.voronoi_bins; name="VORONOI_BINS", header=hdr)
            end
        end
    end
end

