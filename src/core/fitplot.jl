
# Helper function to fit a plotly HTML plot of a spaxel fit
function plot_spaxel_fit_plotly(cube_fitter::CubeFitter, spaxel::Spaxel, I_model_u::Vector{<:QSIntensity}, 
    comps::Dict, label::String, χ2red::Real; spline::Union{Vector{<:QSIntensity},Nothing}=nothing)

    # Get units
    fopt = fit_options(cube_fitter)
    λunit = replace(latex(unit(spaxel.λ[1])), '$' => "")  # plotly cannot parse more than one latex math expression per axis label
    Iunit = replace(latex(unit(spaxel.N)), '$' => "") 

    # plotly doesnt like \mathrm for some reason
    perfreq = typeof(spaxel.N) <: QPerFreq
    sub = perfreq ? "\\nu" : "\\lambda"

    # Remove units
    λ = ustrip.(spaxel.λ)
    I = ustrip.(spaxel.I .* spaxel.N)
    I_model = ustrip.(I_model_u)

    xlabel = L"$\lambda\,\, (%$λunit)$"
    ylabel = L"$I_{%$sub}\,\, (%$Iunit)$" 

    # Plot the overall data / model
    trace1 = PlotlyJS.scatter(x=λ, y=I, mode="lines", line=Dict(:color => "black", :width => 1), name="Data", yaxis="y1", showlegend=true)
    trace2 = PlotlyJS.scatter(x=λ, y=I_model, mode="lines", line=Dict(:color => "red", :width => 1), name="Model", yaxis="y1", showlegend=true)
    traces = [trace1, trace2]
    if !isnothing(spline)
        append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(spline), mode="lines", line=Dict(:color => "red", :width => 1, :dash => "dash"), 
            name="Cubic Spline", yaxis="y1")])
    end
    # Individual templates
    for j in 1:cube_fitter.n_templates
        append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps["templates_$j"]), mode="lines", line=Dict(:color => "green", :width => 1), 
            name="Template $j", yaxis="y1")])
    end

    # Add vertical dashed lines for emission line rest wavelengths
    lines = model(cube_fitter).lines
    for (lw, ln) ∈ zip(lines.λ₀, lines.names)
        lwi = ustrip(uconvert(unit(spaxel.λ[1]), lw))
        append!(traces, [PlotlyJS.scatter(x=[lwi, lwi], y=[0., nanmaximum(I)*1.1], mode="lines", 
            line=Dict(:color => occursin("H2", String(ln)) ? "red" : 
                        (any(occursin.(["alpha", "beta", "gamma", "delta"], String(ln))) ? "#ff7f0e" : "rebeccapurple"), 
                    :width => 0.5, :dash => "dash"), showlegend=false)])
    end

    # axes labels / titles / fonts
    ext_label = "Extinction"

    layout = PlotlyJS.Layout(
        xaxis_title=xlabel,
        title=L"$\tilde{\chi}^2 = %$χ2red$",
        xaxis_constrain="domain",
        font_family="Georgia, Times New Roman, Serif",
        template="plotly_white",
        yaxis_title=ylabel,
        yaxis_side="left",
        yaxis_showexponent="all",
        yaxis_exponentformat="power",
        yaxis2_title=ext_label,
        yaxis2_overlaying="y",
        yaxis2_range=[-5, 0.],        # for logarithmic axis, the ranges are the LOG of the values
        yaxis2_showexponent="all",
        yaxis2_exponentformat="power",
        yaxis2_type="log",
        yaxis2_side="right"
    )
    
    abs_full = cube_fitter.n_abs_feat ≥ 1 ? reduce(.*, [comps["absorption_feat_$i"] for i ∈ 1:cube_fitter.n_abs_feat]) : ones(length(λ))
    if fopt.fit_ch_abs
        abs_full .*= comps["absorption_ice"] .* comps["absorption_ch"]
    end
    ext_gas = comps["total_extinction_gas"]
    ext_stars = comps["total_extinction_stars"]

    # Loop over and plot individual model components
    for comp ∈ keys(comps)
        if comp == "total_extinction_gas"
            append!(traces, [PlotlyJS.scatter(x=λ, y=ext_gas, mode="lines", 
                line=Dict(:color => "black", :width => 0.5, :dash => "dash"), name="Extinction", yaxis="y2")])
        elseif comp == "absorption_oli"
            append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps[comp]), mode="lines", 
                line=Dict(:color => "black", :width => 0.5, :dash => "dash"), name="Olivine Absorption", yaxis="y2")])
        elseif comp == "absorption_pyr"
            append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps[comp]), mode="lines", 
                line=Dict(:color => "black", :width => 0.5, :dash => "dash"), name="Pyroxene Absorption", yaxis="y2")])
        elseif comp == "absorption_for"
            append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps[comp]), mode="lines", 
                line=Dict(:color => "black", :width => 0.5, :dash => "dash"), name="Forsterite Absorption", yaixs="y2")])
        elseif occursin("SSP", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps[comp]) .* ext_stars, mode="lines", 
                line=Dict(:color => "#FF00FF", :width => 0.5), name="SSPs", yaxis="y1")])
        elseif occursin("na_feii", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps[comp]) .* ext_gas, mode="lines", 
                line=Dict(:color => "yellow", :width => 0.5), name="Narrow Fe II", yaxis="y1")])
        elseif occursin("br_feii", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps[comp]) .* ext_gas, mode="lines", 
                line=Dict(:color => "yellow", :width => 1), name="Broad Fe II", yaxis="y1")])
        elseif occursin("power_law", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps[comp]) .* ext_gas, mode="lines", 
                line=Dict(:color => "orange", :width => 0.5), name="Power laws", yaxis="y1")])
        elseif occursin("dust_cont", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps[comp]) .* ext_gas, mode="lines", 
                line=Dict(:color => "orange", :width => 0.5), name="Dust continuum", yaxis="y1")])
        elseif occursin("hot_dust", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps[comp]) .* abs_full, mode="lines", 
                line=Dict(:color => "yellow", :width => 0.5), name="Hot Dust", yaxis="y1")])
        elseif occursin("line", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps[comp]) .* ext_gas,
                mode="lines", line=Dict(:color => "rebeccapurple", :width => 0.5), name="Lines", yaxis="y1")])
        elseif occursin("dust_feat", comp)
            append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps[comp]) .* ext_gas,
                mode="lines", line=Dict(:color => "blue", :width => 0.5), name="PAHs", yaxis="y1")])
        end
    end

    # Add the summed up continuum
    append!(traces, [PlotlyJS.scatter(x=λ, y=ustrip.(comps["continuum"]), mode="lines",
        line=Dict(:color => "gray", :width => 1), name="Total continuum", yaxis="y1")])
    # Summed up PAH features
    sdust = zeros(length(λ))
    for (k, dcomplex) in enumerate(model(cube_fitter).dust_features.profiles)
        for (j, component) in enumerate(dcomplex)
            sdust .+= ustrip.(comps["dust_feat_$(k)_$(j)"]) .* ext_gas 
        end
    end
    append!(traces, [PlotlyJS.scatter(x=λ, y=sdust, mode="lines", line=Dict(:color => "blue", :width => 1), name="Total PAHs")])

    # save as HTML file
    p = PlotlyJS.plot(traces, layout)
    PlotlyJS.savefig(p, isnothing(label) ? joinpath("output_$(cube_fitter.name)", "spaxel_plots", "levmar_fit_spaxel.html") : 
        joinpath("output_$(cube_fitter.name)", "spaxel_plots", "$label.html"))
end


# Helper function for plotting and annotating the emission lines
function pyplot_annotate_emission_lines!(λ::Vector{<:QWave}, comps::Dict, lines::FitFeatures,
    extinction::Vector{<:Real}, normalization::Vector{<:Quantity}, cube_fitter::CubeFitter,
    ax1, ax2, range::Union{Nothing,Tuple})

    # full line profile
    if isnothing(range)
        s = zeros(length(λ))
        for (k, line) in enumerate(lines.profiles)   
            for (j, component) in enumerate(line)    
                s .+= comps["line_$(k)_$(j)"] .* extinction .* normalization
            end
        end
        ax1.plot(ustrip.(λ), s, "-", color="rebeccapurple", alpha=0.6, label="Lines")
    else
        for (k, line) in enumerate(lines.profiles)   
            for (j, component) in enumerate(line)    
                ax1.plot(ustrip.(λ), comps["line_$(k)_$(j)"] .* extinction .* normalization, 
                    "-", color="rebeccapurple", alpha=0.6, label="Lines")
            end
        end
    end

    # plot vertical dashed lines for emission line wavelengths
    line_wave = ustrip.(uconvert.(unit(λ[1]), lines.λ₀))
    for lw ∈ line_wave
        ax1.axvline(lw, linestyle="--", color="k", lw=0.5, alpha=0.5)
        ax2.axvline(lw, linestyle="--", color="k", lw=0.5, alpha=0.5)
    end

    # Annotate emission lines 
    line_λ = ustrip.(λ)
    arrow_tip = ax1.get_ybound()[2]
    box_loc = arrow_tip*1.06
    # First time this is being plot - set up the line annotation positions with the python lineid_plot package
    if all(iszero, cube_fitter.output.plot_line_annotation_positions) || !isnothing(range)
        ak = py_lineidplot.initial_annotate_kwargs()
        ak["verticalalignment"] = "bottom"
        ak["horizontalalignment"] = "center"
        pk = py_lineidplot.initial_plot_kwargs()
        pk["lw"] = 0.5
        pk["alpha"] = 0.5
        # Set y values manually because sometimes it f*cks up if left to the defaults...
        fig, ax1 = py_lineidplot.plot_line_ids(line_λ, ones(length(line_λ)),
            line_wave[lines.config.annotate], lines.labels[lines.config.annotate], ax=ax1, extend=false, label1_size=12, 
            plot_kwargs=pk, annotate_kwargs=ak, box_loc=box_loc, add_label_to_artists=true)
        # Save the line annotation positions so they can be re-used later
        if isnothing(range)
            for i in 1:sum(lines.config.annotate)
                box = ax1.texts[i]
                cube_fitter.output.plot_line_annotation_positions[i] = box.xyann[1]
            end
            # Save these to a file 
            open(joinpath("output_$(cube_fitter.name)", "line_annotation_positions.csv"), "w") do f
                writedlm(f, cube_fitter.output.plot_line_annotation_positions, ',')
            end
        end
    else
        # read in the already-set annotation positions and re-use them
        for i in 1:sum(lines.config.annotate)
            # re-add the annotations at the same positions 
            ax1.annotate(lines.labels[lines.config.annotate][i], 
                         xy=(line_wave[lines.config.annotate][i], arrow_tip),
                         xytext=(cube_fitter.output.plot_line_annotation_positions[i], box_loc),
                         fontsize=12, va="bottom", ha="center", xycoords="data", textcoords="data",
                         rotation=90., arrowprops=Dict("arrowstyle" => "-", "relpos" => (0.5, 0.0)))
        end
    end
end


function plot_spaxel_fit_pyplot(cube_fitter::CubeFitter, spaxel::Spaxel, I_model::Vector{<:QSIntensity}, 
    comps::Dict, range::Union{Tuple,Nothing}, χ2red::Real, label::String; spline::Union{Vector{<:QSIntensity},Nothing}=nothing, 
    I_boot_min::Union{Vector{<:QSIntensity},Nothing}=nothing, I_boot_max::Union{Vector{<:QSIntensity},Nothing}=nothing, 
    logy::Bool=false)

    fopt = fit_options(cube_fitter)
    λunit = unit(spaxel.λ[1])
    Nunit = unit(spaxel.N)
    mask_lines = spaxel.mask_lines
    mask_bad = spaxel.mask_bad
    λ = spaxel.λ
    I = spaxel.I .* spaxel.N

    perfreq = typeof(spaxel.N) <: QPerFreq
    if perfreq
        factor = uconvert.(u"Hz", C_KMS ./ spaxel.λ)
        Iunit = Nunit*u"Hz"
        sub = "\\nu"
    else
        factor = spaxel.λ
        Iunit = Nunit*λunit
        sub = "\\lambda"
    end

    min_inten = logy ? nothing : -0.01
    extscale_limits = (1e-5, 1.)
    ext_label = "Extinction"
    I_label = prefix -> L"$%$sub I_{%$sub}$ (%$prefix%$(latex(Iunit)))"

    # If max is above 10^4, normalize so the y axis labels aren't super wide
    power = floor(Int, log10(ustrip(maximum(I .* factor))))
    if (power ≥ 4) || (power ≤ -4)
        norm = 10.0^power * Iunit
    else
        norm = 1.0 * Iunit
    end

    if isnothing(min_inten)
        if !logy || !isnothing(range)
            min_inten = (sum((I ./ norm .* factor) .< -0.01) > (length(λ)/10)) ? -2nanstd(I ./ norm .* factor) : -0.01
        else
            min_inten = 0.1nanminimum(I[I .> 0.0*Nunit] ./ norm .* factor)
        end
    end
    if !logy || !isnothing(range)
        max_inten = isnothing(range) ? 
                    1.3nanmaximum((I ./ norm .* factor)[.~mask_lines .& .~mask_bad]) : 
                    1.1nanmaximum((I ./ norm .* factor)[range[1] .< λ .< range[2]])
    else
        max_inten = isnothing(range) ? 
                    1.1nanmaximum(I[.~mask_bad] ./ norm .* factor[.~mask_bad]) : 
                    1.1nanmaximum((I ./ norm .* factor)[range[1] .< λ .< range[2]])
    end
    # max_resid = 1.1maximum(((I.-I_model) ./ norm .* factor)[.~mask_lines .& .~mask_bad][2:end-1])
    mask_resid = mask_lines .| mask_bad
    for pair in cube_fitter.spectral_region.mask
        mask_resid .|= pair[1] .< spaxel.λ .< pair[2]
    end
    max_resid = 5nanstd(((I.-I_model) ./ norm .* factor)[.~mask_resid][2:end-1])
    min_resid = -max_resid
    @assert unit(max_inten) == unit(min_inten) == unit(max_resid) == unit(min_resid) == NoUnits

    if (max_inten < 1) && (norm ≠ 1.0Iunit)
        norm /= 10
        max_inten *= 10
        max_resid *= 10
        min_resid *= 10
        power -= 1
    end
    if norm ≠ 1.0Iunit
        prefix = L"$10^{%$power}$ "
    else
        prefix = ""
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

    ax1.plot(ustrip.(λ), I ./ norm .* factor, "k-", label="Data")

    # plot cubic spline
    if !isnothing(spline)
        ax1.plot(ustrip.(λ), spline ./ norm .* factor, color="#2ca02c", linestyle="--", label="Cubic Spline")
    end

    ax1.plot(ustrip.(λ), I_model ./ norm .* factor, "-", color="#ff5d00", label="Model")
    if !isnothing(I_boot_min) && !isnothing(I_boot_max)
        ax1.fill_between(ustrip.(λ), I_boot_min ./ norm .* factor, I_boot_max ./ norm .* factor, color="#ff5d00", 
            alpha=0.5, zorder=10)
    end

    ax2.plot(ustrip.(λ), (I.-I_model) ./ norm .* factor, "k-")

    χ2_str = @sprintf "%.3f" χ2red
    ax2.plot(ustrip.(λ), zeros(length(λ)), "-", color="#ff5d00", label=L"$\tilde{\chi}^2 = %$χ2_str$")
    if !isnothing(I_boot_min) && !isnothing(I_boot_max)
        ax2.fill_between(ustrip.(λ), (I_boot_min .- I_model) ./ norm .* factor, (I_boot_max .- I_model) ./ norm .* factor, 
            color="#ff5d00", alpha=0.5, zorder=10)
    end
    # ax2.fill_between(λ, (I.-I_cont.+σ)./norm./λ, (I.-I_cont.-σ)./norm./λ, color="k", alpha=0.5)

    # twin axes with different labels --> extinction for ax3 and observed wavelength for ax4
    ax3 = ax1.twinx()
    # ax4 = ax1.twiny()

    # Shade in masked regions
    user_mask_bits = falses(length(λ))
    user_mask = cube_fitter.spectral_region.mask
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
        ax1.axvspan(ustrip.(λ)[le], ustrip.(λ)[re], alpha=0.5, color="k")
        ax2.axvspan(ustrip.(λ)[le], ustrip.(λ)[re], alpha=0.5, color="k")
    end

    if isnothing(range)
        λmin, λmax = ustrip.(extrema(λ))
        ax1.set_xlim(λmin, λmax)
        ax2.set_xlim(λmin, λmax)
        # ax4.set_xlim(λmin * (1 + z), λmax * (1 + z))
        ax1.set_ylim(min_inten, max_inten)
    else
        ax1.set_xlim(ustrip(range[1]), ustrip(range[2]))
        ax2.set_xlim(ustrip(range[1]), ustrip(range[2]))
        ax1.set_ylim(min_inten, max_inten)
    end
    ax2.set_ylim(min_resid, max_resid)
    ax3.set_ylim(extscale_limits...)
    ax3.set_ylabel(ext_label)
    ax1.set_ylabel(I_label(prefix))
    ax2.set_ylabel(L"$O-C$")  # ---> residuals, (O)bserved - (C)alculated
    ax2.set_xlabel(L"$\lambda_{\rm rest}$ (%$(latex(λunit)))")
    # ax4.set_xlabel(L"$\lambda_{\rm obs}$ ($\mu$m)")
    ax2.legend(loc="upper left")

    # axis scaling
    if isnothing(range)
        ax1.set_xscale("log") # logarithmic wavelength axis
        ax2.set_xscale("log") # logarithmic wavelength axis
    end
    ax3.set_yscale("log") # logarithmic extinction axis

    # Set minor ticks as multiples of 0.1 μm for x axis and automatic for y axis
    if isnothing(range)
        # make sensible tick marks
        major_tick_space = round(ustrip(maximum(λ)-minimum(λ))/9, sigdigits=1)
        minor_tick_space = round(ustrip(maximum(λ)-minimum(λ))/45, sigdigits=1)
        ax1.xaxis.set_minor_locator(py_ticker.MultipleLocator(minor_tick_space))
        ax1.xaxis.set_major_locator(py_ticker.MultipleLocator(major_tick_space))
        ax2.xaxis.set_minor_locator(py_ticker.MultipleLocator(minor_tick_space))
        ax2.xaxis.set_major_locator(py_ticker.MultipleLocator(major_tick_space))
    end
    ax1.yaxis.set_minor_locator(py_ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(py_ticker.AutoMinorLocator())
    # ax4.xaxis.set_minor_locator(py_ticker.AutoMinorLocator())

    # Set tick formats so that we dont use scientific notation where it isnt needed
    if isnothing(range)
        ax1.xaxis.set_major_formatter(py_ticker.ScalarFormatter())
        ax1.xaxis.set_minor_formatter(py_ticker.NullFormatter())
        ax2.xaxis.set_major_formatter(py_ticker.ScalarFormatter())
        ax2.xaxis.set_minor_formatter(py_ticker.NullFormatter())
    end

    if logy && isnothing(range)
        ax1.set_yscale("log")   # logarithmic intensity axis
    end

    # Set major ticks and formats
    ax1.set_xticklabels([]) # ---> will be covered up by the residuals plot
    ax2.set_yticks([-round(max_resid/2, sigdigits=1), 0.0, round(max_resid/2, sigdigits=1)])
    # ax1.tick_params(which="both", axis="both", direction="in")
    ax1.tick_params(which="both", axis="both", direction="in", top=true)
    ax2.tick_params(which="both", axis="both", direction="in", labelright=true, right=true, top=true)
    ax3.tick_params(which="both", axis="both", direction="in")
    # ax4.tick_params(which="both", axis="both", direction="in")

    ### Plot individual components of the model ###

    abs_full = cube_fitter.n_abs_feat ≥ 1 ? reduce(.*, [comps["absorption_feat_$i"] for i ∈ 1:cube_fitter.n_abs_feat]) : ones(length(λ))
    if fopt.fit_ch_abs
        abs_full .*= comps["absorption_ice"] .* comps["absorption_ch"]
    end
    ext_gas = comps["total_extinction_gas"]
    ext_stars = comps["total_extinction_stars"]
    split_ext = haskey(comps, "absorption_oli")

    # full continuum (no PAHs)
    ax1.plot(ustrip.(λ), comps["continuum"] ./ norm .* factor, "k-", lw=2, alpha=0.5, label="Continuum")
    # individual continuum components
    if fopt.fit_stellar_continuum
        ax1.plot(ustrip.(λ), comps["SSPs"] .* ext_stars ./ norm .* factor, "-", color="fuchsia", alpha=0.75, label="SSPs")
    end
    if fopt.fit_opt_na_feii
        ax1.plot(ustrip.(λ), comps["na_feii"] .* ext_gas ./ norm .* factor, "-", color="goldenrod", alpha=0.8, label="Narrow Fe II")
    end
    if fopt.fit_opt_br_feii
        ax1.plot(ustrip.(λ), comps["br_feii"] .* ext_gas ./ norm .* factor, "--", color="goldenrod", alpha=0.8, label="Broad Fe II")
    end
    for i in 1:cube_fitter.n_power_law
        ax1.plot(ustrip.(λ), comps["power_law_$i"] .* ext_gas ./ norm .* factor, "k-", alpha=0.5, label="Power Law")
    end
    for i in 1:cube_fitter.n_dust_cont
        ax1.plot(ustrip.(λ), comps["dust_cont_$i"] .* ext_gas ./ norm .* factor, "k-", alpha=0.5, label="Dust continuum")
    end
    if fopt.fit_sil_emission
        ax1.plot(ustrip.(λ), comps["hot_dust"] .* abs_full ./ norm .* factor, "-", color="#8ac800", alpha=0.8, label="Hot Dust")
    end
    # templates
    for k ∈ 1:cube_fitter.n_templates
        ax1.plot(ustrip.(λ), comps["templates_$k"] ./ norm .* factor, "-", color="#50630d", label="Template $k")
    end
    # full PAH profile
    sdust = zeros(length(λ))
    for (k, dcomplex) in enumerate(model(cube_fitter).dust_features.profiles)
        for (j, component) in enumerate(dcomplex)
            sdust .+= comps["dust_feat_$(k)_$(j)"] .* ext_gas ./ norm .* factor 
        end
    end
    ax1.plot(ustrip.(λ), sdust, "-", color="#0065ff", label="PAHs")
    # extinction
    if split_ext
        ax3.plot(ustrip.(λ), comps["absorption_oli"], "k", linestyle=(0, (3, 1, 1, 1, 1, 1)), alpha=0.5, label="Olivine Absorption")
        ax3.plot(ustrip.(λ), comps["absorption_pyr"], "k", linestyle="dashdot", alpha=0.5, label="Pyroxene Absorption")
        ax3.plot(ustrip.(λ), comps["absorption_for"], "k", linestyle="dashed", alpha=0.5, label="Forsterite Absorption")
        ax3.plot(ustrip.(λ), ext_gas, "k", linestyle="dotted", alpha=0.5, label="Full Extinction")
    else
        ax3.plot(ustrip.(λ), ext_gas, "k:", alpha=0.5, label="Extinction")
    end

    # mark channel boundaries 
    if length(cube_fitter.cube.spectral_region.channel_bounds) > 0
        ax1.plot(ustrip.(cube_fitter.cube.spectral_region.channel_bounds) ./ (1 .+ cube_fitter.z), 
            ones(length(cube_fitter.cube.spectral_region.channel_bounds)) .* max_inten .* 0.99, "v", 
            color="#0065ff", markersize=4.0)
    end

    # annotate emission lines
    pyplot_annotate_emission_lines!(λ, comps, model(cube_fitter).lines, ext_gas, factor ./ norm, cube_fitter, ax1, ax2, range)

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
- `cube_fitter`: The CubeFitter object
- `spaxel`: The spaxel object
- `I_model`: The intensity model vector of the spaxel to be plotted
- `comps`: The dictionary of individual components of the model intensity
- `χ2red`: The reduced χ^2 value of the fit
- `label`: A label for the individual spaxel being plotted, to be put in the file name
- `backend`: The backend to use to plot, may be `:pyplot`, `:plotly`, or `:both`
- `I_boot_min`: Optional vector giving the minimum model out of all bootstrap iterations
- `I_boot_max`: Optional vector giving the maximum model out of all bootstrap iterations
- `range`: Optional tuple specifying min/max wavelength values to truncate the x-axis of the plot to
- `spline`: Optional vector giving the cubic spline interpolation of the continuum to plot
- `logy`: If true, make the y-axis logarithmic (only for pyplot backend)
"""
function plot_spaxel_fit(cube_fitter::CubeFitter, spaxel::Spaxel, I_model::Vector{<:QSIntensity},
    comps::Dict, χ2red::Real, label::String; backend::Symbol=:pyplot, I_boot_min::Union{Vector{<:QSIntensity},Nothing}=nothing, 
    I_boot_max::Union{Vector{<:QSIntensity},Nothing}=nothing, range::Union{Tuple,Nothing}=nothing, 
    spline::Union{Vector{<:QSIntensity},Nothing}=nothing, logy::Bool=false) 

    # Plotly ---> useful interactive plots for visually inspecting data, but not publication-quality
    if (backend == :plotly || backend == :both) && isnothing(range)
        plot_spaxel_fit_plotly(cube_fitter, spaxel, I_model, comps, label, χ2red; spline=spline)
    end

    # Pyplot --> actually publication-quality plots finely tuned to be the most useful and visually appealing that I could make them
    if backend == :pyplot || backend == :both
        plot_spaxel_fit_pyplot(cube_fitter, spaxel, I_model, comps, range, χ2red, label; spline=spline, I_boot_min=I_boot_min,
            I_boot_max=I_boot_max, logy=logy)
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
function plot_parameter_map(data::Matrix{Float64}, name_i::String, bunit::AbstractString, save_path::String, Ω::typeof(1.0u"sr"), z::Float64, 
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
        # vmax = min(nanmaximum(filtered), 30)
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

    pix_as, n_pix, scalebar_text_dist, scalebar_text_ang = get_physical_scales(size(data), Ω, cosmo, z)
    scalebar_1 = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, scalebar_text_dist, "upper center", pad=0, borderpad=0, 
        color=text_color, frameon=false, size_vertical=0.1, label_top=false, bbox_to_anchor=(0.17, 0.1), bbox_transform=ax.transAxes)
    ax.add_artist(scalebar_1)
    if disable_axes
        scalebar_2 = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, scalebar_text_ang, "lower center", pad=0, borderpad=0, 
            color=text_color, frameon=false, size_vertical=0.1, label_top=true, bbox_to_anchor=(0.17, 0.1), bbox_transform=ax.transAxes)
        ax.add_artist(scalebar_2)
    end

    # Add circle for the PSF FWHM
    r = psf_fwhm / ustrip(pix_as) / 2
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


function plot_stellar_grids(result::StellarResult, cube_fitter::CubeFitter, label::String)

    logm = log10.(result.masses)
    all_ages = log10.(ustrip.(uconvert.(u"yr", cube_fitter.ssps.ages)))
    all_logzs = cube_fitter.ssps.logzs

    # Create a 2D colormap of the masses
    fig, ax = plt.subplots()
    cdata = ax.imshow(logm', origin=:lower, cmap=:cubehelix, vmin=minimum(logm), vmax=maximum(logm),
        extent=[minimum(all_ages), maximum(all_ages), minimum(all_logzs), maximum(all_logzs)])
    # add colorbar
    fig.colorbar(cdata, ax=ax, label=L"$\log_{10}(M/M_\odot)$")
    # axis labels
    ax.set_xlabel(L"$\log_{10}$(Age / yr)")
    ax.set_ylabel(L"$\log_{10}(Z/Z_\odot)$")

    # plot points at the located maxima of the distribution
    for (ai, zi) in zip(result.ages, result.logzs)
        ax.plot(log10(ustrip(uconvert(u"yr", ai))), zi, "rx", ms=5)
    end

    # Output file path creation
    out_folder = joinpath("output_$(cube_fitter.name)", "stellar_grids")
    if !isdir(out_folder)
        mkdir(out_folder)
    end
    # Save figure as PDF, yay for vector graphics!
    plt.savefig(joinpath(out_folder, "$label.stellar_grid.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

end
