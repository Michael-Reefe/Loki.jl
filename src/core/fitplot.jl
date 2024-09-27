
# Helper function for creating and formatting the figure and plotting overall data/models
function plot_full_models_plotly(λ::Vector{<:Real}, I::Vector{<:Real}, I_model::Vector{<:Real}, comps::Dict,
    line_wave::Vector{<:Real}, line_names::Vector{Symbol}, xlabel::AbstractString, ylabel::AbstractString, 
    nuc_temp_fit::Bool, n_templates::Integer, χ2red::Real; spline::Union{Vector{<:Real},Nothing}=nothing)

    # Plot the overall data / model
    trace1 = PlotlyJS.scatter(x=λ, y=I, mode="lines", line=Dict(:color => "black", :width => 1), name="Data", showlegend=true)
    trace2 = PlotlyJS.scatter(x=λ, y=I_model, mode="lines", line=Dict(:color => "red", :width => 1), name="Model", showlegend=true)
    traces = [trace1, trace2]
    if !isnothing(spline)
        append!(traces, [PlotlyJS.scatter(x=λ, y=spline, mode="lines", line=Dict(:color => "red", :width => 1, :dash => "dash"), name="Cubic Spline")])
    end
    # Individual templates
    if !nuc_temp_fit
        for j in 1:n_templates
            append!(traces, [PlotlyJS.scatter(x=λ, y=comps["templates_$j"], mode="lines", line=Dict(:color => "green", :width => 1), name="Template $j")])
        end
    end

    # Add vertical dashed lines for emission line rest wavelengths
    for (lw, ln) ∈ zip(line_wave, line_names)
        append!(traces, [PlotlyJS.scatter(x=[lw, lw], y=[0., nanmaximum(I)*1.1], mode="lines", 
            line=Dict(:color => occursin("H2", String(ln)) ? "red" : 
                        (any(occursin.(["alpha", "beta", "gamma", "delta"], String(ln))) ? "#ff7f0e" : "rebeccapurple"), 
                    :width => 0.5, :dash => "dash"))])
    end

    # axes labels / titles / fonts
    layout = PlotlyJS.Layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        title=L"$\tilde{\chi}^2 = %$χ2red$",
        xaxis_constrain="domain",
        font_family="Georgia, Times New Roman, Serif",
        template="plotly_white",
        # annotations=[
        #     attr(x=lw, y=nanmaximum(I)*.75, text=ll) for (lw, ll) ∈ zip(line_wave, line_latex)
        # ]
    )

    traces, layout
end

# Helper function to fit a plotly HTML plot of a spaxel fit
function plot_mir_spaxel_fit_plotly(cube_fitter::CubeFitter, λ::Vector{<:Real}, I::Vector{<:Real}, I_model::Vector{<:Real}, 
    comps::Dict, nuc_temp_norm::Vector{<:Real}, nuc_temp_fit::Bool, χ2red::Real, label::String; 
    spline::Union{Vector{<:Real},Nothing}=nothing)

    xlabel = L"$\lambda\ (\mu{\rm m})$"
    ylabel = L"$I_{\nu}\ ({\rm MJy}\,{\rm sr}^{-1})$" 
    traces, layout = plot_full_models_plotly(λ, I, I_model, comps, cube_fitter.lines.λ₀, cube_fitter.lines.names, 
        xlabel, ylabel, nuc_temp_fit, cube_fitter.n_templates, χ2red; spline=spline)
    
    abs_feat = cube_fitter.n_abs_feat ≥ 1 ? reduce(.*, [comps["abs_feat_$i"] for i ∈ 1:cube_fitter.n_abs_feat]) : ones(length(λ))
    abs_full = comps["abs_ice"] .* comps["abs_ch"] .* abs_feat
    ext_full = abs_full .* comps["extinction"]

    # Loop over and plot individual model components
    for comp ∈ keys(comps)
        if comp == "extinction"
            append!(traces, [PlotlyJS.scatter(x=λ, y=ext_full .* maximum(I_model) .* 1.1, 
                mode="lines", line=Dict(:color => "black", :width => 1, :dash => "dash"), name="Extinction")])
        elseif comp == "abs_oli"
            append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* maximum(I_model) .* 1.1, mode="lines", line=Dict(:color => "blue", :width => 1, :dash => "dash"),
                name="Olivine Absorption")])
        elseif comp == "abs_pyr"
            append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* maximum(I_model) .* 1.1, mode="lines", line=Dict(:color => "red", :width => 1, :dash => "dash"),
                name="Pyroxene Absorption")])
        elseif comp == "abs_for"
            append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* maximum(I_model) .* 1.1, mode="lines", line=Dict(:color => "orange", :width => 1, :dash => "dash"),
                name="Forsterite Absorption")])
        elseif occursin("hot_dust", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* abs_full .* nuc_temp_norm, mode="lines", line=Dict(:color => "yellow", :width => 1),
                name="Hot Dust")])
        elseif occursin("unobscured_continuum", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* nuc_temp_norm, mode="lines", line=Dict(:color => "black", :width => 0.5), 
                name="Unobscured Continuum")])
        elseif occursin("line", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"] .* nuc_temp_norm, 
                mode="lines", line=Dict(:color => "rebeccapurple", :width => 1), name="Lines")])
        end
    end

    # Add the summed up continuum
    append!(traces, [PlotlyJS.scatter(x=λ, y=(abs_full .* (cube_fitter.fit_sil_emission ? comps["hot_dust"] : zeros(length(λ))) .+ 
        comps["unobscured_continuum"] .+ comps["obscured_continuum"] .+ ((cube_fitter.n_templates > 0) && !nuc_temp_fit ? 
        sum([comps["templates_$k"] for k ∈ 1:cube_fitter.n_templates], dims=1)[1] : zeros(length(λ)))) .* nuc_temp_norm, 
        mode="lines", line=Dict(:color => "green", :width => 1), name="Total Continuum")])
    # Summed up PAH features
    append!(traces, [PlotlyJS.scatter(x=λ, y=sum([comps["dust_feat_$i"] for i ∈ 1:cube_fitter.n_dust_feat], dims=1)[1] .* 
        comps["extinction"] .* nuc_temp_norm, mode="lines", line=Dict(:color => "blue", :width => 1), name="PAHs")])
    # Individual PAH features
    for i in 1:cube_fitter.n_dust_feat
        append!(traces, [PlotlyJS.scatter(x=λ, y=comps["dust_feat_$i"] .* comps["extinction"] .* nuc_temp_norm, 
            mode="lines", line=Dict(:color => "blue", :width => 1), name="PAHs")])
    end

    # save as HTML file
    p = PlotlyJS.plot(traces, layout)
    PlotlyJS.savefig(p, isnothing(label) ? joinpath("output_$(cube_fitter.name)", "spaxel_plots", "levmar_fit_spaxel.html") : 
        joinpath("output_$(cube_fitter.name)", "spaxel_plots", "$label.html"))
end


function plot_opt_spaxel_fit_plotly(cube_fitter::CubeFitter, λ::Vector{<:Real}, I::Vector{<:Real}, I_model::Vector{<:Real}, 
    comps::Dict, nuc_temp_norm::Vector{<:Real}, nuc_temp_fit::Bool, χ2red::Real, label::String; 
    spline::Union{Vector{<:Real},Nothing}=nothing)

    # Plot the overall data / model
    xlabel = L"$\lambda\ (\mathring{A})$"
    ylabel = L"I_{\lambda}\ ({\rm erg}\,{\rm s}^{-1}\,{\rm cm}^{-2}\,{\mathring{A}}^{-1}\,{\rm sr}^{-1})"
    traces, layout = plot_full_models_plotly(λ, I, I_model, comps, cube_fitter.lines.λ₀.*1e4, cube_fitter.lines.names, 
        xlabel, ylabel, nuc_temp_fit, cube_fitter.n_templates, χ2red; spline=spline)
    
    att_stars = comps["attenuation_stars"]
    att_gas = comps["attenuation_gas"]

    # Loop over and plot individual model components
    for comp ∈ keys(comps)
        if comp == "attenuation_stars"
            append!(traces, [PlotlyJS.scatter(x=λ, y=att_gas ./ median(att_gas) .* maximum(I_model) .* 1.1, 
                mode="lines", line=Dict(:color => "black", :width => 1, :dash => "dash"), name="Extinction")])
        elseif occursin("na_feii", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* att_gas .* nuc_temp_norm, mode="lines", line=Dict(:color => "yellow", :width => 1),
                name="Narrow Fe II")])
        elseif occursin("br_feii", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* att_gas .* nuc_temp_norm, mode="lines", line=Dict(:color => "yellow", :width => 2),
                name="Broad Fe II")])
        elseif occursin("line", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* att_gas .* nuc_temp_norm, 
                mode="lines", line=Dict(:color => "rebeccapurple", :width => 1), name="Lines")])
        end
    end

    # Add the summed up continuum
    append!(traces, [PlotlyJS.scatter(x=λ, y=nuc_temp_norm .* (att_stars .* sum([comps["SSP_$i"] for i ∈ 1:cube_fitter.n_ssps], dims=1)[1] .+
        (cube_fitter.fit_opt_na_feii ? comps["na_feii"] .* att_gas : zeros(length(λ))) .+ 
        (cube_fitter.fit_opt_br_feii ? comps["br_feii"] .* att_gas : zeros(length(λ))) .+
        (cube_fitter.n_power_law > 0 ? sum([comps["power_law_$j"] for j in 1:cube_fitter.n_power_law], dims=1)[1] : zeros(length(λ)))), 
        mode="lines", line=Dict(:color => "green", :width => 1), name="Continuum")])
    for i in 1:cube_fitter.n_ssps
        append!(traces, [PlotlyJS.scatter(x=λ, y=att_stars .* comps["SSP_$i"] .* nuc_temp_norm, mode="lines", line=Dict(:color => "green", :width => 0.5))])
    end

    # save as HTML file
    p = PlotlyJS.plot(traces, layout)
    PlotlyJS.savefig(p, isnothing(label) ? joinpath("output_$(cube_fitter.name)", "spaxel_plots", "levmar_fit_spaxel.html") : 
        joinpath("output_$(cube_fitter.name)", "spaxel_plots", "$label.html"))
end


# Helper function for creating and formatting the figure and plotting overall data/models
function plot_full_models_pyplot(λ::Vector{<:Real}, I::Vector{<:Real}, I_model::Vector{<:Real},
    mask_lines::BitVector, mask_bad::BitVector, user_mask::Vector{<:Tuple}, range::Union{Tuple,Nothing}, 
    factor::Vector{<:Real}, extscale_limits::Tuple, ext_label::AbstractString, I_label::Function, 
    χ2red::Real; min_inten::Union{Real,Nothing}=nothing, spline::Union{Vector{<:Real},Nothing}=nothing, 
    I_boot_min::Union{Vector{<:Real},Nothing}=nothing, I_boot_max::Union{Vector{<:Real},Nothing}=nothing,
    major_tick_space::Real=2.5, minor_tick_space::Real=0.5, logy::Bool=false)

    # If max is above 10^4, normalize so the y axis labels aren't super wide
    power = floor(Int, log10(maximum(I .* factor)))
    if (power ≥ 4) || (power ≤ -4)
        norm = 10.0^power
    else
        norm = 1.0
    end

    if isnothing(min_inten)
        if !logy
            min_inten = (sum((I ./ norm .* factor) .< -0.01) > (length(λ)/10)) ? -2nanstd(I ./ norm .* factor) : -0.01
        else
            min_inten = 0.1nanminimum(I[I .> 0.] ./ norm .* factor)
        end
    end
    if !logy
        max_inten = isnothing(range) ? 
                    1.3nanmaximum(I[.~mask_lines .& .~mask_bad] ./ norm .* factor[.~mask_lines .& .~mask_bad]) : 
                    1.1nanmaximum((I ./ norm .* factor)[range[1] .< λ .< range[2]])
    else
        max_inten = isnothing(range) ? 
                    1.1nanmaximum(I[.~mask_bad] ./ norm .* factor[.~mask_bad]) : 
                    1.1nanmaximum((I ./ norm .* factor)[range[1] .< λ .< range[2]])
    end
    max_resid = 1.1maximum(((I.-I_model) ./ norm .* factor)[.~mask_lines .& .~mask_bad])
    min_resid = -max_resid
    
    if (max_inten < 1) && (norm ≠ 1.0)
        norm /= 10
        max_inten *= 10
        power -= 1
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

    ax1.plot(λ, I ./ norm .* factor, "k-", label="Data")

    # plot cubic spline
    if !isnothing(spline)
        ax1.plot(λ, spline ./ norm .* factor, color="#2ca02c", linestyle="--", label="Cubic Spline")
    end

    ax1.plot(λ, I_model ./ norm .* factor, "-", color="#ff5d00", label="Model")
    if !isnothing(I_boot_min) && !isnothing(I_boot_max)
        ax1.fill_between(λ, I_boot_min ./ norm .* factor, I_boot_max ./ norm .* factor, color="#ff5d00", alpha=0.5, zorder=10)
    end

    ax2.plot(λ, (I.-I_model) ./ norm .* factor, "k-")

    χ2_str = @sprintf "%.3f" χ2red
    ax2.plot(λ, zeros(length(λ)), "-", color="#ff5d00", label=L"$\tilde{\chi}^2 = %$χ2_str$")
    if !isnothing(I_boot_min) && !isnothing(I_boot_max)
        ax2.fill_between(λ, (I_boot_min .- I_model) ./ norm .* factor, (I_boot_max .- I_model) ./ norm .* factor, color="#ff5d00", alpha=0.5,
            zorder=10)
    end
    # ax2.fill_between(λ, (I.-I_cont.+σ)./norm./λ, (I.-I_cont.-σ)./norm./λ, color="k", alpha=0.5)

    # twin axes with different labels --> extinction for ax3 and observed wavelength for ax4
    ax3 = ax1.twinx()
    # ax4 = ax1.twiny()

    # Shade in masked regions
    user_mask_bits = falses(length(λ))
    for region in user_mask
        user_mask_bits .|= region[1] .< λ .< region[2]
    end
    total_mask = mask_bad .| user_mask_bits
    l_edges = findall(diff(total_mask) .== 1) .+ 1
    r_edges = findall(diff(total_mask) .== -1)
    # Edge cases
    if total_mask[1] == 1
        l_edges = [1; l_edges]
    end
    if total_mask[end] == 1
        r_edges = [r_edges; length(λ)]
    end
    for (le, re) in zip(l_edges, r_edges)
        ax1.axvspan(λ[le], λ[re], alpha=0.5, color="k")
        ax2.axvspan(λ[le], λ[re], alpha=0.5, color="k")
    end

    if isnothing(range)
        λmin, λmax = minimum(λ), maximum(λ)
        ax1.set_xlim(λmin, λmax)
        ax2.set_xlim(λmin, λmax)
        # ax4.set_xlim(λmin * (1 + z), λmax * (1 + z))
        ax1.set_ylim(min_inten, max_inten)
    else
        ax1.set_xlim(range[1], range[2])
        ax2.set_xlim(range[1], range[2])
        # ax4.set_xlim(range[1] * (1 + z), range[2] * (1 + z))
        ax1.set_ylim(min_inten, max_inten)
    end
    ax2.set_ylim(min_resid, max_resid)
    ax3.set_ylim(extscale_limits...)
    ax3.set_ylabel(ext_label)
    if (power ≥ 4) || (power ≤ -4)
        prefix = L"$10^{%$power}$ "
    else
        prefix = ""
    end
    ax1.set_ylabel(I_label(prefix))
    ax2.set_ylabel(L"$O-C$")  # ---> residuals, (O)bserved - (C)alculated
    ax2.set_xlabel(L"$\lambda_{\rm rest}$ ($\mu$m)")
    # ax4.set_xlabel(L"$\lambda_{\rm obs}$ ($\mu$m)")
    ax2.legend(loc="upper left")

    # axis scaling
    ax1.set_xscale("log") # logarithmic wavelength axis
    ax2.set_xscale("log") # logarithmic wavelength axis
    ax3.set_yscale("log") # logarithmic extinction axis

    # Set minor ticks as multiples of 0.1 μm for x axis and automatic for y axis
    ax1.xaxis.set_minor_locator(py_ticker.MultipleLocator(minor_tick_space))
    ax1.xaxis.set_major_locator(py_ticker.MultipleLocator(major_tick_space))
    ax1.yaxis.set_minor_locator(py_ticker.AutoMinorLocator())
    ax2.xaxis.set_minor_locator(py_ticker.MultipleLocator(minor_tick_space))
    ax2.xaxis.set_major_locator(py_ticker.MultipleLocator(major_tick_space))
    ax2.yaxis.set_minor_locator(py_ticker.AutoMinorLocator())
    # ax4.xaxis.set_minor_locator(py_ticker.AutoMinorLocator())

    # Set tick formats so that we dont use scientific notation where it isnt needed
    ax1.xaxis.set_major_formatter(py_ticker.ScalarFormatter())
    ax1.xaxis.set_minor_formatter(py_ticker.NullFormatter())
    ax2.xaxis.set_major_formatter(py_ticker.ScalarFormatter())
    ax2.xaxis.set_minor_formatter(py_ticker.NullFormatter())

    if logy
        ax1.set_yscale("log")   # logarithmic intensity axis
    end

    # Set major ticks and formats
    ax1.set_xticklabels([]) # ---> will be covered up by the residuals plot
    ax2.set_yticks([-round(maximum(((I.-I_model) ./ norm .* factor)[.~mask_lines .& .~mask_bad]) / 2, sigdigits=1), 0.0, 
                    round(maximum(((I.-I_model) ./ norm .* factor)[.~mask_lines .& .~mask_bad]) / 2, sigdigits=1)])
    # ax1.tick_params(which="both", axis="both", direction="in")
    ax1.tick_params(which="both", axis="both", direction="in", top=true)
    ax2.tick_params(which="both", axis="both", direction="in", labelright=true, right=true, top=true)
    ax3.tick_params(which="both", axis="both", direction="in")
    # ax4.tick_params(which="both", axis="both", direction="in")

    norm, max_inten, fig, ax1, ax2, ax3
end


# Helper function for plotting and annotating the emission lines
function pyplot_annotate_emission_lines!(λ::Vector{<:Real}, comps::Dict, n_comps::Integer,
    extinction::Vector{<:Real}, normalization::Vector{<:Real}, line_wave::Vector{<:Real}, line_annotate::BitVector, 
    line_latex::Vector{String}, line_f::Vector{<:Real}, ax1, ax2, range::Union{Nothing,Tuple})

    # full line profile
    if isnothing(range)
        ax1.plot(λ, sum([haskey(comps, "line_$(i)_$(j)") ? comps["line_$(i)_$(j)"] : zeros(length(λ)) 
            for i ∈ 1:length(line_wave), j ∈ 1:n_comps], dims=(1,2))[1] .* extinction .* normalization, 
            "-", color="rebeccapurple", alpha=0.6, label="Lines")
    else
        for i ∈ 1:length(line_wave), j ∈ 1:n_comps
            if haskey(comps, "line_$(i)_$(j)")
                ax1.plot(λ, comps["line_$(i)_$(j)"] .* extinction .* normalization, 
                    "-", color="rebeccapurple", alpha=0.6, label="Lines")
            end
        end
    end

    # plot vertical dashed lines for emission line wavelengths
    for lw ∈ line_wave
        ax1.axvline(lw, linestyle="--", color="k", lw=0.5, alpha=0.5)
        ax2.axvline(lw, linestyle="--", color="k", lw=0.5, alpha=0.5)
    end

    # Annotate emission lines 
    line_λ = copy(λ)
    ak = py_lineidplot.initial_annotate_kwargs()
    ak["verticalalignment"] = "bottom"
    ak["horizontalalignment"] = "center"
    pk = py_lineidplot.initial_plot_kwargs()
    pk["lw"] = 0.5
    pk["alpha"] = 0.5
    fig, ax1 = py_lineidplot.plot_line_ids(line_λ, line_f, line_wave[line_annotate], line_latex[line_annotate], ax=ax1,
        extend=false, label1_size=12, plot_kwargs=pk, annotate_kwargs=ak)

end


function plot_mir_spaxel_fit_pyplot(cube_fitter::CubeFitter, λ::Vector{<:Real}, I::Vector{<:Real}, I_model::Vector{<:Real}, 
    mask_lines::BitVector, mask_bad::BitVector, range::Union{Tuple,Nothing}, comps::Dict, nuc_temp_norm::Vector{<:Real}, 
    nuc_temp_fit::Bool, Cf::Real, χ2red::Real, label::String; spline::Union{Vector{<:Real},Nothing}=nothing, 
    I_boot_min::Union{Vector{<:Real},Nothing}=nothing, I_boot_max::Union{Vector{<:Real},Nothing}=nothing, logy::Bool=false)

    # MIR-specific normalizations and axis labels/limits
    factor = 1 ./ λ 
    min_inten = logy ? nothing : -0.01
    extscale_limits = (1e-3, 1.)
    ext_label = cube_fitter.extinction_screen ? L"$e^{-\tau_{\lambda}}$" : L"$(1-e^{-\tau_{\lambda}}) / \tau_{\lambda}$"
    I_label = prefix -> L"$I_{\nu}/\lambda$ (%$(prefix)MJy sr$^{-1}$ $\mu$m$^{-1}$)"

    norm, max_inten, fig, ax1, ax2, ax3 = plot_full_models_pyplot(λ, I, I_model,
        mask_lines, mask_bad, cube_fitter.user_mask, range, factor, extscale_limits, ext_label, I_label, χ2red; 
        min_inten=min_inten, spline=spline, I_boot_min=I_boot_min, I_boot_max=I_boot_max, minor_tick_space=0.5,
        major_tick_space=2.5, logy=logy)

    # MIR-specific quantities to plot
    abs_feat = cube_fitter.n_abs_feat ≥ 1 ? reduce(.*, [comps["abs_feat_$i"] for i ∈ 1:cube_fitter.n_abs_feat]) : ones(length(λ))
    abs_full = comps["abs_ice"] .* comps["abs_ch"] .* abs_feat
    ext_full = abs_full .* comps["extinction"]
    split_ext = haskey(comps, "abs_oli")

    # full continuum
    ax1.plot(λ, (((cube_fitter.n_templates > 0) && !nuc_temp_fit ? sum([comps["templates_$k"] for k ∈ 1:cube_fitter.n_templates], 
        dims=1)[1] : zeros(length(λ))) .+ abs_full .* (cube_fitter.fit_sil_emission ? comps["hot_dust"] : zeros(length(λ))) .+
        comps["obscured_continuum"] .+ comps["unobscured_continuum"]) .* nuc_temp_norm ./ norm .* factor, "k-", lw=2, alpha=0.5, label="Continuum")
    # individual continuum components
    ax1.plot(λ, comps["stellar"] .* (Cf .* ext_full .+ (1 .- Cf)) .* nuc_temp_norm ./ norm .* factor, "m--", alpha=0.5, label="Stellar continuum")
    for i in 1:cube_fitter.n_dust_cont
        ax1.plot(λ, comps["dust_cont_$i"] .* (Cf .* ext_full .+ (1 .- Cf)) .* nuc_temp_norm ./ norm .* factor, "k-", alpha=0.5, label="Dust continuum")
    end
    for i in 1:cube_fitter.n_power_law
        ax1.plot(λ, comps["power_law_$i"] .* (Cf .* ext_full .+ (1 .- Cf)) .* nuc_temp_norm ./ norm .* factor, "k-", alpha=0.5, label="Power Law")
    end
    # ax1.plot(λ, comps["obscured_continuum"] ./ norm .* factor, "-", color="#0b450a", alpha=0.8, label="Obscured continuum")
    # ax1.plot(λ, comps["unobscured_continuum"] ./ norm .* factor, "--", color="#0b450a", alpha=0.8, label="Unobscured continuum")
    # full PAH profile
    ax1.plot(λ, sum([comps["dust_feat_$i"] for i ∈ 1:cube_fitter.n_dust_feat], dims=1)[1] .* comps["extinction"] .* nuc_temp_norm ./ norm .* factor, "-", 
        color="#0065ff", label="PAHs")
    # plot hot dust
    if haskey(comps, "hot_dust")
        ax1.plot(λ, comps["hot_dust"] .* abs_full .* nuc_temp_norm ./ norm .* factor, "-", color="#8ac800", alpha=0.8, label="Hot Dust")
    end
    # templates
    if !nuc_temp_fit
        for k ∈ 1:cube_fitter.n_templates
            ax1.plot(λ, comps["templates_$k"] ./ norm .* factor, "-", color="#50630d", label="Template $k")
        end
    end

    # plot extinction
    if split_ext
        ax3.plot(λ, comps["abs_oli"], "k", linestyle=(0, (3, 1, 1, 1, 1, 1)), alpha=0.5, label="Olivine Absorption")
        ax3.plot(λ, comps["abs_pyr"], "k", linestyle="dashdot", alpha=0.5, label="Pyroxene Absorption")
        ax3.plot(λ, comps["abs_for"], "k", linestyle="dashed", alpha=0.5, label="Forsterite Absorption")
        ax3.plot(λ, comps["extinction"], "k", linestyle="dotted", label="Full Extinction")
    else
        ax3.plot(λ, ext_full, "k:", alpha=0.5, label="Extinction")
    end
    if nuc_temp_fit
        ax3.plot(λ, nuc_temp_norm, "k", linestyle="--", alpha=0.5, label="PSF")
    end

    # mark channel boundaries 
    ax1.plot(channel_boundaries ./ (1 .+ cube_fitter.z), ones(length(channel_boundaries)) .* max_inten .* 0.99, "v", 
        color="#0065ff", markersize=3.0)

    # annotate emission lines
    pyplot_annotate_emission_lines!(λ, comps, cube_fitter.n_comps, comps["extinction"], nuc_temp_norm ./ norm .* factor,
        cube_fitter.lines.λ₀, cube_fitter.lines.annotate, cube_fitter.lines.latex, copy(I ./ norm .* factor), ax1, ax2,
        range)

    # Output file path creation
    out_folder = joinpath("output_$(cube_fitter.name)", isnothing(range) ? "spaxel_plots" : joinpath("zoomed_plots", split(label, "_")[end]))
    if !isdir(out_folder)
        mkdir(out_folder)
    end
    # Save figure as PDF, yay for vector graphics!
    plt.savefig(joinpath(out_folder, isnothing(label) ? "levmar_fit_spaxel.pdf" : "$label.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
end


function plot_opt_spaxel_fit_pyplot(cube_fitter::CubeFitter, λ::Vector{<:Real}, I::Vector{<:Real}, I_model::Vector{<:Real},  
    mask_lines::BitVector, mask_bad::BitVector, range::Union{Tuple,Nothing},
    comps::Dict, nuc_temp_norm::Vector{<:Real}, nuc_temp_fit::Bool, χ2red::Real, label::String; 
    spline::Union{Vector{<:Real},Nothing}=nothing, I_boot_min::Union{Vector{<:Real},Nothing}=nothing, 
    I_boot_max::Union{Vector{<:Real},Nothing}=nothing, logy::Bool=false)

    # Optical-specific normalizations and axis labels/limits
    factor = ones(length(λ))
    extscale_limits = (1e-5, 1.)
    ext_label = L"$10^{-0.4E(B-V)_{\rm gas}k'(\lambda)}$"
    I_label = prefix -> L"$I_{\lambda}$ (%$(prefix)erg s$^{-1}$ cm$^{-2}$ ${\rm \mathring{A}}^{-1}$ sr$^{-1}$)"

    norm, max_inten, fig, ax1, ax2, ax3 = plot_full_models_pyplot(λ, I, I_model,
        mask_lines, mask_bad, cube_fitter.user_mask, range, factor, extscale_limits, ext_label, I_label, χ2red; 
        spline=spline, I_boot_min=I_boot_min, I_boot_max=I_boot_max, minor_tick_space=100, major_tick_space=500,
        logy=logy)

    # Optical-specific quantities to plot
    att_stars = comps["attenuation_stars"]
    att_gas = comps["attenuation_gas"]

    # full continuum
    ax1.plot(λ, (((cube_fitter.n_templates > 0) && !nuc_temp_fit ? sum([comps["templates_$k"] for k ∈ 1:cube_fitter.n_templates], 
        dims=1)[1] : zeros(length(λ))) .+ att_stars .* sum([comps["SSP_$i"] for i ∈ 1:cube_fitter.n_ssps], dims=1)[1] .+
        (cube_fitter.fit_opt_na_feii ? comps["na_feii"] .* att_gas : zeros(length(λ))) .+ 
        (cube_fitter.fit_opt_br_feii ? comps["br_feii"] .* att_gas : zeros(length(λ))) .+
        (cube_fitter.n_power_law > 0 ? sum([comps["power_law_$j"] for j in 1:cube_fitter.n_power_law], dims=1)[1] : 
        zeros(length(λ)))) ./ norm .* nuc_temp_norm .* factor, "k-", lw=2, alpha=0.5, label="Continuum")
    # individual continuum components
    for i in 1:cube_fitter.n_ssps
        ax1.plot(λ, comps["SSP_$i"] .* att_stars ./ norm .* nuc_temp_norm .* factor, "g-", alpha=0.75, label="SSPs")
    end
    for i in 1:cube_fitter.n_power_law
        ax1.plot(λ, comps["power_law_$i"] ./ norm .* nuc_temp_norm .* factor, "k-", alpha=0.5, label="Power Law")
    end
    if cube_fitter.fit_opt_na_feii
        ax1.plot(λ, comps["na_feii"] .* att_gas ./ norm .* nuc_temp_norm .* factor, "-", color="goldenrod", alpha=0.8, label="Narrow Fe II")
    end
    if cube_fitter.fit_opt_br_feii
        ax1.plot(λ, comps["br_feii"] .* att_gas ./ norm .* nuc_temp_norm .* factor, "--", color="goldenrod", alpha=0.8, label="Broad Fe II")
    end

    # plot extinction
    ax3.plot(λ, att_gas, "k:", alpha=0.5, label="Extinction")
    if nuc_temp_fit
        ax3.plot(λ, nuc_temp_norm, "k", linestyle="--", alpha=0.5, label="PSF")
    end

    # Annotate emission lines
    pyplot_annotate_emission_lines!(λ, comps, cube_fitter.n_comps, att_gas, nuc_temp_norm ./ norm .* factor,
        cube_fitter.lines.λ₀.*1e4, cube_fitter.lines.annotate, cube_fitter.lines.latex, copy(I ./ norm .* factor), ax1, ax2,
        range)

    # Output file path creation
    out_folder = joinpath("output_$(cube_fitter.name)", isnothing(range) ? "spaxel_plots" : joinpath("zoomed_plots", split(label, "_")[end]))
    if !isdir(out_folder)
        mkdir(out_folder)
    end
    # Save figure as PDF, yay for vector graphics!
    plt.savefig(joinpath(out_folder, isnothing(label) ? "levmar_fit_spaxel.pdf" : "$label.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
end


"""
    plot_spaxel_fit(spectral_region, λ_um, I, I_model, σ, mask_bad, mask_lines, comps, n_dust_cont, n_power_law,
        n_dust_features, n_abs_features, n_ssps, n_comps, line_wave_um, line_names, line_annotate, line_latex, 
        screen, z, χ2red, name, label; [backend, I_boot_min, I_boot_max, range_um, spline])

Plot the best fit for an individual spaxel using the given backend (`:pyplot` or `:plotly`).

# Arguments {T<:Real}
- `cube_fitter::CubeFitter`: The CubeFitter object
- `λ_um::Vector{<:Real}`: The wavelength vector of the spaxel to be plotted, in microns
- `I::Vector{<:Real}`: The intensity data vector of the spaxel to be plotted
- `I_model::Vector{<:Real}`: The intensity model vector of the spaxel to be plotted
- `mask_bad::BitVector`: The mask giving the locations of bad pixels
- `mask_lines::BitVector`: The mask giving the locations of emission lines
- `comps::Dict{String, Vector{T}}`: The dictionary of individual components of the model intensity
- `nuc_temp_fit::Bool`: Whether or not one is fitting the nuclear template
- `Cf::Real`: The dust covering fraction parameter
- `χ2red::Real`: The reduced χ^2 value of the fit
- `label::String`: A label for the individual spaxel being plotted, to be put in the file name
- `backend::Symbol=:pyplot`: The backend to use to plot, may be `:pyplot`, `:plotly`, or `:both`
- `I_boot_min::Union{Vector{<:Real},Nothing}=nothing`: Optional vector giving the minimum model out of all bootstrap iterations
- `I_boot_max::Union{Vector{<:Real},Nothing}=nothing`: Optional vector giving the maximum model out of all bootstrap iterations
- `range_um::Union{Tuple,Nothing}=nothing`: Optional tuple specifying min/max wavelength values to truncate the x-axis of the plot to
- `spline::Union{Vector{<:Real},Nothing}=nothing`: Optional vector giving the cubic spline interpolation of the continuum to plot
- `logy::Bool=false`: If true, make the y-axis logarithmic (only for pyplot backend)
"""
function plot_spaxel_fit(cube_fitter::CubeFitter, λ_um::Vector{<:Real}, I::Vector{<:Real}, I_model::Vector{<:Real}, mask_bad::BitVector, 
    mask_lines::BitVector, comps::Dict{String, Vector{T}}, nuc_temp_fit::Bool, Cf::Real, χ2red::Real, label::String; 
    backend::Symbol=:pyplot, I_boot_min::Union{Vector{<:Real},Nothing}=nothing, I_boot_max::Union{Vector{<:Real},Nothing}=nothing, 
    range_um::Union{Tuple,Nothing}=nothing, spline::Union{Vector{<:Real},Nothing}=nothing, logy::Bool=false) where {T<:Real}

    # Set up variables
    range = nothing
    if cube_fitter.spectral_region == :MIR
        # Plot in microns for MIR data
        λ = λ_um
        if !isnothing(range_um)
            range = range_um
        end
    else
        # Plot in angstroms for optical data
        λ = λ_um .* 1e4
        if !isnothing(range_um)
            range = range_um .* 1e4
        end
    end
    nuc_temp_norm = ones(length(λ))
    if nuc_temp_fit
        nuc_temp_norm = comps["templates_1"]
    end

    # Plotly ---> useful interactive plots for visually inspecting data, but not publication-quality
    if (backend == :plotly || backend == :both) && isnothing(range)
        # plot_spaxel_fit_plotly(spectral_region, λ, I, I_model, σ, comps, ext_full, abs_full, 
        #     att_gas, att_stars, nuc_temp_norm, n_dust_features, n_power_law, fit_sil_emission, n_ssps, fit_opt_na_feii,
        #     fit_opt_br_feii, nuc_temp_fit, n_templates, line_wave, line_names, χ2red, label, name; 
        #     spline=spline)
        if cube_fitter.spectral_region == :MIR
            plot_mir_spaxel_fit_plotly(cube_fitter, λ, I, I_model, comps, nuc_temp_norm, nuc_temp_fit, χ2red, label; 
                spline=spline)
        else
            plot_opt_spaxel_fit_plotly(cube_fitter, λ, I, I_model, comps, nuc_temp_norm, nuc_temp_fit, χ2red, label; 
                spline=spline)
        end
    end

    # Pyplot --> actually publication-quality plots finely tuned to be the most useful and visually appealing that I could make them
    if backend == :pyplot || backend == :both
        # plot_spaxel_fit_pyplot(spectral_region, λ, I, I_model, σ, mask_lines, mask_bad, range,
        #     user_mask, comps, split_ext, ext_full, abs_full, att_gas, att_stars, nuc_temp_norm, 
        #     n_dust_cont, n_dust_features, n_power_law, fit_sil_emission, n_ssps, fit_opt_na_feii,
        #     fit_opt_br_feii, nuc_temp_fit, n_templates, n_comps, line_wave, line_names, line_annotate, 
        #     line_latex, screen, Cf, z, χ2red, label, name; spline=spline, I_boot_min=I_boot_min, 
        #     I_boot_max=I_boot_max)
        if cube_fitter.spectral_region == :MIR
            plot_mir_spaxel_fit_pyplot(cube_fitter, λ, I, I_model, mask_lines, mask_bad, range, comps, nuc_temp_norm, 
                nuc_temp_fit, Cf, χ2red, label; spline=spline, I_boot_min=I_boot_min, I_boot_max=I_boot_max, logy=logy)
        else
            plot_opt_spaxel_fit_pyplot(cube_fitter, λ, I, I_model, mask_lines, mask_bad, range, 
                comps, nuc_temp_norm, nuc_temp_fit, χ2red, label; spline=spline, I_boot_min=I_boot_min, 
                I_boot_max=I_boot_max, logy=logy)
        end
    end

end


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
    if occursin("voff", name_i) || occursin("index", name_i) || occursin("vel", name_i) || occursin("delta_v", name_i) || 
        occursin("vmed", name_i) || occursin("vpeak", name_i)
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
