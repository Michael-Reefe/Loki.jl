module CubeFit

export fit_cube

# Import packages
using Distributions
using Interpolations
using NaNStatistics
using Optim
# using BlackBoxOptim
using CMPFit
# using LsqFit
using TOML
using NumericalIntegration
using ProgressMeter
using Reexport
using PyCall
using PyPlot
using PlotlyJS

include("parameters.jl")
@reexport using .Param

include("cubedata.jl")
@reexport using .CubeData

sourcepath = dirname(Base.source_path())

function parse_options()

    options = TOML.parsefile(joinpath(sourcepath, "options.toml"))
    options_out = Dict()
    keylist1 = ["stellar_continuum_temp", "dust_continuum_temps", "dust_features", "extinction"]
    keylist2 = ["wave", "fwhm"]
    keylist3 = ["tau_9_7", "beta"]
    keylist4 = ["val", "prior", "pval", "locked"]
    for key ∈ keylist1
        if !(key ∈ keys(options))
            error("Missing option $key in options file!")
        end
    end
    for key ∈ keylist4
        if !(key ∈ keys(options["stellar_continuum_temp"]))
            error("Missing option $key in stellar continuum temp options!")
        end
        for dc ∈ options["dust_continuum_temps"]
            if !(key ∈ keys(dc))
                error("Missing option $key in dust continuum temp options!")
            end
        end
        for df_key ∈ keys(options["dust_features"])
            for df_key2 ∈ keylist2
                if !(df_key2 ∈ keys(options["dust_features"][df_key]))
                    error("Missing option $df_key2 in dust feature $df_key options!")
                end
                if !(key ∈ keys(options["dust_features"][df_key][df_key2]))
                    error("Missing option $key in dust features $df_key, $df_key2 options!")
                end
            end
        end
        for ex_key ∈ keylist3
            if !(ex_key ∈ keys(options["extinction"]))
                error("Missing option $ex_key in extinction options!")
            end
            if !(key ∈ keys(options["extinction"][ex_key]))
                error("Missing option $key in $ex_key options!")
            end
        end
    end
    
    options_out[:stellar_continuum_temp] = Param.from_dict(options["stellar_continuum_temp"])
    options_out[:dust_continuum_temps] = [Param.from_dict(options["dust_continuum_temps"][i]) for i ∈ 1:length(options["dust_continuum_temps"])]
    options_out[:dust_features] = Dict()
    for df ∈ keys(options["dust_features"])
        options_out[:dust_features][df] = Param.ParamDict()
        options_out[:dust_features][df][:wave] = Param.from_dict_wave(options["dust_features"][df]["wave"])
        options_out[:dust_features][df][:fwhm] = Param.from_dict_fwhm(options["dust_features"][df]["fwhm"])
    end
    options_out[:extinction] = Param.ParamDict()
    options_out[:extinction][:tau_9_7] = Param.from_dict(options["extinction"]["tau_9_7"])
    options_out[:extinction][:beta] = Param.from_dict(options["extinction"]["beta"])

    return options_out
end

function parse_lines()
    lines = TOML.parsefile(joinpath(sourcepath, "lines.toml"))
    lines_out = Param.LineDict()
    keylist1 = ["wave", "profile", "voff", "fwhm"]
    keylist2 = ["val", "prior", "pval", "locked"]
    for line ∈ keys(lines)
        for key ∈ keylist1
            if !(key ∈ keys(lines[line]))
                error("$line missing $key option!")
            end
        end
        for key ∈ ["voff", "fwhm"]
            for key2 ∈ keylist2
                if !(key2 ∈ keys(lines[line][key]))
                    error("$line's $key parameter is missing $key2 option!")
                end
            end
        end
        voff = Param.from_dict(lines[line]["voff"])
        fwhm = Param.from_dict(lines[line]["fwhm"])
        params = Param.ParamDict(:voff => voff, :fwhm => fwhm)
        lines_out[Symbol(line)] = Param.TransitionLine(lines[line]["wave"], Symbol(lines[line]["profile"]), params)
    end
    return lines_out
end

options = parse_options()
lines = parse_lines()


function continuum_fit_spaxel(λ::Vector{Float64}, I::Vector{Float64}, σ::Vector{Float64}; 
    verbose::Bool=true, plot::Symbol=:none, label::Union{String,Nothing}=nothing)

    # Read options file
    T_s = options[:stellar_continuum_temp]
    T_dc = options[:dust_continuum_temps]
    τ_97 = options[:extinction][:tau_9_7]
    β = options[:extinction][:beta]

    # Remove features outside fitting range
    for df ∈ keys(options[:dust_features])
        if (options[:dust_features][df][:wave].value > maximum(λ)+0.5) || (options[:dust_features][df][:wave].value < minimum(λ)-0.5)
            delete!(options[:dust_features], df)
        end
    end

    n_dust_cont = length(T_dc)
    n_dust_features = length(options[:dust_features])

    Δλ = diff(λ)[1]
    λ_min, λ_max = minimum(λ), maximum(λ)
    interp_func = cubic_spline_interpolation(λ_min:Δλ:λ_max, I, extrapolation_bc=Line())

    # Stellar amplitude
    A_s = clamp(interp_func(5.5) / Util.Blackbody_ν(5.5, T_s.value), 0., Inf)
    # Dust feature amplitudes
    A_df = repeat([clamp(nanmedian(I)/2, 0., Inf)], n_dust_features)
    # Dust continuum amplitudes
    λ_dc = clamp.(2898 ./ [Ti.value for Ti ∈ T_dc], minimum(λ), maximum(λ))
    A_dc = clamp.(interp_func.(λ_dc) ./ [Util.Blackbody_ν(λ_dci, T_dci.value) for (λ_dci, T_dci) ∈ zip(λ_dc, T_dc)] .* (λ_dc ./ 9.7).^2 ./ 5., 0., Inf)
    amp_dc_prior = Uniform(0., 1e20)
    amp_df_prior = Uniform(0., maximum(I))
    
    mean_df = [options[:dust_features][df][:wave] for df ∈ keys(options[:dust_features])]
    fwhm_df = [options[:dust_features][df][:fwhm] for df ∈ keys(options[:dust_features])]

    stellar_pars = [A_s, T_s.value]
    stellar_pnames = ["A_s", "T_s"]
    stellar_priors = [amp_dc_prior, T_s.prior]

    dc_pars = vcat([[Ai, Ti.value] for (Ai, Ti) ∈ zip(A_dc, T_dc)]...)
    dc_pnames = vcat([["A_$i", "T_$i"] for i ∈ 1:n_dust_cont]...)
    dc_priors = vcat([[amp_dc_prior, Ti.prior] for Ti ∈ T_dc]...)

    df_pars = vcat([[Ai, mi.value, fi.value] for (Ai, mi, fi) ∈ zip(A_df, mean_df, fwhm_df)]...)
    df_pnames = vcat([["A_$i", "μ_$i", "FWHM_$i"] for i ∈ 1:n_dust_features]...)
    df_priors = vcat([[amp_df_prior, mi.prior, fi.prior] for (mi, fi) ∈ zip(mean_df, fwhm_df)]...)

    # Initial parameter vector
    p₀ = vcat(stellar_pars, dc_pars, df_pars, [τ_97.value], [β.value])
    pnames = vcat(stellar_pnames, dc_pnames, df_pnames, ["τ_9.7"], ["β"])
    priors = vcat(stellar_priors, dc_priors, df_priors, [τ_97.prior], [β.prior])

    # Convert parameter limits into CMPFit object
    parinfo = CMPFit.Parinfo(length(p₀))

    # Stellar amplitude
    parinfo[1].limited = (1,0)
    parinfo[1].limits = (0., Inf)
    # Stellar temp
    parinfo[2].fixed = T_s.locked
    if !(T_s.locked)
        @assert T_s.prior isa Uniform
        parinfo[2].limited = (1,1)
        parinfo[2].limits = params(T_s.prior)
    end

    # Dust cont amplitudes and temps
    pᵢ = 3
    for i ∈ 1:n_dust_cont
        parinfo[pᵢ].limited = (1,0)
        parinfo[pᵢ].limits = (0., Inf)
        parinfo[pᵢ+1].fixed = T_dc[i].locked
        if !(T_dc[i].locked)
            @assert T_dc[i].prior isa Uniform
            parinfo[pᵢ+1].limited = (1,1)
            parinfo[pᵢ+1].limits = params(T_dc[i].prior)
        end
        pᵢ += 2
    end

    # Dust feature amplitude, mean, fwhm
    for i ∈ 1:n_dust_features
        parinfo[pᵢ].limited = (1,0)
        parinfo[pᵢ].limits = (0., Inf)
        parinfo[pᵢ+1].fixed = mean_df[i].locked
        if !(mean_df[i].locked)
            @assert mean_df[i].prior isa Uniform
            parinfo[pᵢ+1].limited = (1,1)
            parinfo[pᵢ+1].limits = params(mean_df[i].prior)
        end
        parinfo[pᵢ+2].fixed = fwhm_df[i].locked
        if !(fwhm_df[i].locked)
            @assert fwhm_df[i].prior isa Uniform
            parinfo[pᵢ+2].limited = (1,1)
            parinfo[pᵢ+2].limits = params(fwhm_df[i].prior)
        end
        pᵢ += 3
    end

    # Extinction
    parinfo[pᵢ].fixed = τ_97.locked
    if !(τ_97.locked)
        @assert τ_97.prior isa Uniform
        parinfo[pᵢ].limited = (1,1)
        parinfo[pᵢ].limits = params(τ_97.prior)
    end
    parinfo[pᵢ+1].fixed = β.locked
    if !(β.locked)
        @assert β.prior isa Uniform
        parinfo[pᵢ+1].limited = (1,1)
        parinfo[pᵢ+1].limits = params(β.prior)
    end

    # Create a `config` structure
    config = CMPFit.Config()
    config.ftol = 1e-10

    res = cmpfit(λ, I, σ, (x, p) -> Util.Continuum(x, p, n_dust_cont, n_dust_features), p₀, parinfo=parinfo, config=config)
    popt = res.param
    χ2 = res.bestnorm
    χ2red = res.bestnorm / res.dof

    # res = curve_fit((x, p) -> Util.Continuum(x, p, n_dust_cont, n_dust_features), λ, I, p₀, lower=minimum.(priors), upper=maximum.(priors),
    #     autodiff=:forward)
    # popt = res.param
    # χ2 = χ2red = sum(res.resid.^2) / length(p₀)

    # function ln_prior(p)
    #     logpdfs = [logpdf(priors[i], p[i]) for i ∈ 1:length(p)]
    #     return sum(logpdfs)
    # end

    # function nln_probability(p)
    #     model = Util.Continuum(λ, p, n_dust_cont, n_dust_features)
    #     return -Util.ln_likelihood(I, model, σ)
    # end

    # df = TwiceDifferentiable(nln_probability, p₀; autodiff=:forward)
    # dfc = TwiceDifferentiableConstraints(minimum.(priors), maximum.(priors))
    # res = optimize(df, dfc, p₀, IPNewton())
    # popt = res.minimizer
    # χ2 = χ2red = -res.minimum

    # Final optimized continuum fit
    I_cont, comps = Util.Continuum(λ, popt, n_dust_cont, n_dust_features, return_components=true)

    if verbose
        println("Fit Results")
        println("===========")
        println("χ^2       = $χ2")
        println("χ^2 / dof = $χ2red")
    end

    # Update values
    opt_out = deepcopy(options)
    opt_out[:stellar_continuum_temp].value = popt[1]
    pᵢ = 3
    for i ∈ 1:n_dust_cont
        opt_out[:dust_continuum_temps][i].value = popt[pᵢ+1]
        pᵢ += 2
    end
    for (i, ki) ∈ enumerate(keys(options[:dust_features]))
        opt_out[:dust_features][ki][:wave].value = popt[pᵢ+1]
        opt_out[:dust_features][ki][:fwhm].value = popt[pᵢ+2]
        pᵢ += 3
    end
    opt_out[:extinction][:tau_9_7].value = popt[pᵢ]
    opt_out[:extinction][:beta].value = popt[pᵢ+1]

    if plot == :plotly
        if !isdir("continuum_plots")
            mkdir("continuum_plots")
        end
        trace1 = PlotlyJS.scatter(x=λ, y=I, mode="lines", line=Dict(:color => "black", :width => 1), name="Data", showlegend=true)
        trace2 = PlotlyJS.scatter(x=λ, y=I_cont, mode="lines", line=Dict(:color => "red", :width => 1), name="Continuum Fit", showlegend=true)
        traces = [trace1, trace2]
        for comp ∈ keys(comps)
            if comp == "extinction"
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* maximum(I_cont) .* 1.1, mode="lines", 
                    line=Dict(:color => "black", :width => 1, :dash => "dash"), name="Extinction")])
            elseif comp == "stellar"
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"], mode="lines",
                    line=Dict(:color => "red", :width => 1, :dash => "dash"), name="Stellar Continuum")])
            elseif occursin("dust_cont", comp)
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"], mode="lines",
                    line=Dict(:color => "green", :width => 1, :dash => "dash"), name="Dust Continuum")])
            elseif occursin("dust_feat", comp)
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"], mode="lines",
                    line=Dict(:color => "blue", :width => 1), name="Dust Features")])
            end
        end
        # append!(traces, [PlotlyJS.scatter(x=λ, y=Util.Continuum(λ, p₀, n_dust_cont, n_dust_features), mode="lines",
            # line=Dict(:color => "red", :width => 1, :dash => "dash"), name="Initial Guess")])
        append!(traces, [PlotlyJS.scatter(x=λ, y=comps["extinction"] .* (sum([comps["dust_cont_$i"] for i ∈ 1:n_dust_cont], dims=1)[1] .+ comps["stellar"]), mode="lines",
            line=Dict(:color => "green", :width => 1), name="Dust+Stellar Continuum")])
        layout = PlotlyJS.Layout(
            xaxis_title="\$\\lambda\\ (\\mu{\\rm m})\$",
            yaxis_title="\$I_{\\nu}\\ ({\\rm MJy}\\,{\\rm sr}^{-1})\$",
            title="\$\\tilde{\\chi}^2 = $χ2red\$",
            xaxis_constrain="domain",
            font_family="Georgia, Times New Roman, Serif",
            template="plotly_white"
        )
        p = PlotlyJS.plot(traces, layout)
        PlotlyJS.savefig(p, isnothing(label) ? "continuum_plots/continuum_fit_spaxel.html" : "continuum_plots/$label.html")

    elseif plot == :pyplot
        if !isdir("continuum_plots")
            mkdir("continuum_plots")
        end
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(nrows=4, ncols=1, hspace=0.)
        ax1 = fig.add_subplot(py"$(gs)[:-1, :]")
        ax2 = fig.add_subplot(py"$(gs)[-1, :]")
        ax1.plot(λ, I, "k-", label="Data")
        ax1.plot(λ, I_cont, "r-", label="Continuum Fit")
        ax2.plot(λ, I.-I_cont, "k-")
        ax3 = ax1.twinx()
        for comp ∈ keys(comps)
            if comp == "extinction"
                ax3.plot(λ, comps[comp], "k--", alpha=0.5)
            elseif comp == "stellar"
                ax1.plot(λ, comps[comp] .* comps["extinction"], "r--", alpha=0.5)
            elseif occursin("dust_cont", comp)
                ax1.plot(λ, comps[comp] .* comps["extinction"], "g--", alpha=0.5)
            elseif occursin("dust_feat", comp)
                ax1.plot(λ, comps[comp] .* comps["extinction"], "b-", alpha=0.5)
            end
        end
        ax1.plot(λ, comps["extinction"] .* (sum([comps["dust_cont_$i"] for i ∈ 1:n_dust_cont], dims=1)[1] .+ comps["stellar"]), "g-")
        ax1.set_xlim(minimum(λ), maximum(λ))
        ax2.set_xlim(minimum(λ), maximum(λ))
        ax3.set_ylim(0., 1.1)
        ax3.set_ylabel("Extinction")
        ax1.set_xlabel("\$ \\lambda \$ (\$\\mu\$m)")
        ax1.set_ylabel("\$ I_{\\nu} \$ (MJy sr\$^{-1}\$)")
        ax2.set_ylabel("Residuals")
        ax1.set_title("\$\\tilde{\\chi}^2 = $χ2red\$")
        plt.savefig(isnothing(label) ? "continuum_plots/continuum_fit_spaxel.pdf" : "continuum_plots/$label.pdf", dpi=300, bbox_inches="tight")
        plt.close()
    end

    return opt_out, I_cont

end


function line_fit_spaxel(λ::Vector{Float64}, I::Vector{Float64}, σ::Vector{Float64}, I_cont::Vector{Float64}, fitline::Symbol; 
    window_size::Float64=.025, plot::Symbol=:none, label::Union{String,Nothing}=nothing)

    line = lines[fitline]

    # Store wavelength difference
    Δλ = diff(λ)[1]

    # Define the fitting region
    window = (λ .> (line.λ₀ - window_size)) .& (λ .< (line.λ₀ + window_size))
    norm_window = ((λ .> (line.λ₀ - 2window_size)) .& (λ .< (line.λ₀ - window_size))) .| ((λ .> (line.λ₀ + window_size)) .& (λ .< (line.λ₀ + 2window_size)))

    # Pixel vector
    pix = 1:length(λ[window])
    pix_interp = linear_interpolation(λ[window], pix)
    λ_interp = linear_interpolation(pix, λ[window])

    # Subtract continuum flux
    Inorm = I[window] .- I_cont[window]
    σnorm = σ[window]

    # Normalization
    N = nanmaximum(Inorm)
    Inorm ./= N
    σnorm ./= N

    # Add amplitude parameter
    line.parameters[:amp] = Param.Parameter(N, false, Uniform(0., N*1.1))

    # Initial parameter vector
    A = 1.
    μ_pix = pix_interp(Util.Doppler_shift_λ(line.λ₀, line.parameters[:voff].value))
    fwhm_pix = (Util.Doppler_shift_λ(line.λ₀, line.parameters[:fwhm].value) - line.λ₀) / Δλ

    p₀ = [A, μ_pix, fwhm_pix]
    param_names = [:amp, :voff, :fwhm]
    p_func = eval(Meta.parse("Util." * String(line.profile)))

    p_to_phys(p) = [p[1] * N, Util.Doppler_shift_v(λ_interp(p[2]), line.λ₀), Util.Doppler_shift_v(p[3] * Δλ + line.λ₀, line.λ₀)]

    # Sum of the ln(prior) distributions
    function ln_prior(p)
        # Convert parameters into physical values
        p_phys = p_to_phys(p)
        return sum([logpdf(line.parameters[key].prior, pᵢ) for (key, pᵢ) ∈ zip(param_names, p_phys)])
    end

    # Overall probability function
    function ln_probability(p, x, y, err)
        model = p_func(x, p)
        return Util.ln_likelihood(y, model, err) + ln_prior(p)
    end

    # Optimize with Nelder-Mead
    res = optimize(p -> -ln_probability(p, pix, Inorm, σnorm), p₀, NelderMead())
    popt = Optim.minimizer(res)
    lnP = -Optim.minimum(res)

    # Construct the optimized fit
    line_fit = zeros(size(I)...)
    line_fit[window] = p_func(pix, popt) .* N

    p_phys = p_to_phys(popt)
    p_out = deepcopy(line.parameters)
    p_out[:amp].value = p_phys[1]
    p_out[:voff].value = p_phys[2]
    p_out[:fwhm].value = p_phys[3]

    # Integrated flux
    ∫I = NumericalIntegration.integrate(λ[window], line_fit[window], SimpsonEven())

    # Signal to Noise ratio
    R = 3000
    SNR = ∫I / Util.∫Gaussian(std(I[norm_window]), (Util.Doppler_shift_λ(line.λ₀, p_phys[3]) - line.λ₀)/R)

    if plot == :pyplot
        if !isdir("line_plots")
            mkdir("line_plots")
        end

        fig, ax = plt.subplots()
        ax.plot(λ[window], I[window], "k-")
        ax.plot(λ[window], I_cont[window] .+ line_fit[window], "r-")
        plt.savefig(isnothing(label) ? "line_plots/line_fit_spaxel.pdf" : "line_plots/$label.pdf", bbox_inches=:tight)
        plt.close()
    end

    return p_out, ∫I, SNR, line_fit

end

function fit_cube(cube::CubeData.DataCube; window_size::Float64=.025, progress::Bool=false, 
    plot_lines::Symbol=:none, plot_continua::Symbol=:none)

    # Get shape
    shape = size(cube.Iλ)

    # Prepare outputs
    # full 3D flux model array
    I_model = zeros(shape...)

    # Nested dictionary -> first layer keys are line names, second layer keys are parameter names, which contain 2D arrays
    line_maps = Dict()
    for line ∈ keys(lines)
        line_maps[line] = Dict()
        for pname in [:amp, :voff, :fwhm, :∫I, :SNR]
            line_maps[line][pname] = ones(shape[1:2]...) .* NaN
        end
    end

    # Alias
    λ = cube.λ
    # Make sure the wavelength vector is linear, since it is assumed later in the function
    diffs = diff(λ)
    @assert diffs[1] ≈ diffs[end]

    # Prepare progress bar
    if progress
        prog = Progress(shape[1] * shape[2], dt=1e-5, desc="Fitting cube...", showspeed=true)
    end

    for (xᵢ, yᵢ) ∈ Iterators.product(1:shape[1], 1:shape[2])

        # Filter NaNs
        Iᵢ = cube.Iλ[xᵢ, yᵢ, :]
        σᵢ = cube.σI[xᵢ, yᵢ, :]
        if sum(.!isfinite.(Iᵢ) .| .!isfinite.(σᵢ)) > (shape[3] / 10)
            if progress
                next!(prog)
            end
            continue
        end
        filt = .!isfinite.(Iᵢ) .& .!isfinite.(σᵢ)
        Iᵢ[filt] .= nanmedian(Iᵢ)
        σᵢ[filt] .= nanmedian(σᵢ)
        
        # Fit the continuum
        p_cont, I_cont = continuum_fit_spaxel(λ, Iᵢ, σᵢ; verbose=false, plot=plot_continua, label="spaxel_$(xᵢ)_$(yᵢ)")
        I_model[xᵢ, yᵢ, :] .+= I_cont

        # Fit each line in line_list 
        for line ∈ keys(lines)

            # Fit the line
            p_out, ∫I, SNR, line_fit = line_fit_spaxel(λ, Iᵢ, σᵢ, I_cont, line, window_size=window_size, plot=plot_lines, label="spaxel_$(xᵢ)_$(yᵢ)")
            I_model[xᵢ, yᵢ, :] .+= line_fit

            line_maps[line][:amp][xᵢ, yᵢ] = p_out[:amp].value
            line_maps[line][:voff][xᵢ, yᵢ] = p_out[:voff].value
            line_maps[line][:fwhm][xᵢ, yᵢ] = p_out[:fwhm].value
            line_maps[line][:SNR][xᵢ, yᵢ] = SNR
            line_maps[line][:∫I][xᵢ, yᵢ] = ∫I
        end

        if progress
            next!(prog)
        end

    end

    return line_maps, I_model

end

end