module CubeFit

export fit_cube

# Import packages
using Distributions
using Interpolations
using NaNStatistics
using Optim
using NumericalIntegration
using ProgressMeter
using Reexport
using PyCall
using PyPlot

astropy_units = pyimport("astropy.units")
helpers = pyimport("pahfit.helpers")

include("parameters.jl")
@reexport using .Param

include("cubedata.jl")
@reexport using .CubeData


function continuum_fit_spaxel(λ::Vector{Float64}, F::Vector{Float64}, σ::Vector{Float64}, label::String; method::Symbol=:pahfit, 
    loc::Union{Float64,Nothing}=nothing, window_size::Union{Float64,Nothing}=nothing, nostart::Bool=false, maxiter=10_000)

    if method == :linear

        if isnothing(loc)
            loc = nanmedian(λ)
        end
        if isnothing(window_size)
            window_size = 250.
        end
        # Fitting window around the line
        fit_window = (λ .> (loc - 2window_size)) .& (λ .< (loc - window_size))  .| (λ .> (loc + window_size)) .& (λ .< (loc + 2window_size))

        # Get endpoints
        x₀ = λ[fit_window][1]
        x₁ = λ[fit_window][end]
        y₀ = F[fit_window][1]
        y₁ = F[fit_window][end]

        # Construct linear fit with point-slope form
        slope = (y₁ - y₀) / (x₁ - x₀)
        F_cont = y₀ .+ slope .* (λ .- x₀)

        model = Dict(:x₀ => x₀, :y₀ => y₀, :m => slope)

    elseif method == :pahfit

        # Add astropy units
        λ_q = λ << astropy_units.Angstrom
        F_q = Fᵢ << astropy_units.Unit("erg / (s cm2 AA)")
        σ_q = σᵢ << astropy_units.Unit("erg / (s cm2 AA)")

        # Create the data object
        pah_obs = Dict(
            "x" => λ_q.to("um") << astropy_units.um,
            "y" => F_q.to("Jy", equivalencies=astropy_units.spectral_density(λ_q)) << astropy_units.Jy,
            "unc" => σ_q.to("Jy", equivalencies=astropy_units.spectral_density(λ_q)) << astropy_units.Jy
            )

        # using DataFrames
        # using CSV
        # CSV.write("$label.csv", DataFrame(λ=λ_q.to("um"), 
        #                                   F=F_q.to("Jy", equivalencies=astropy_units.spectral_density(λ_q)),
        #                                   σ=σ_q.to("Jy", equivalencies=astropy_units.spectral_density(λ_q))))
        
        # Create the model
        pah_model = helpers.initialize_model("scipack_ExGal_SpitzerIRSSLLL.ipac", pah_obs, !nostart)

        # Fit the spectrum with LevMarLSQFitter
        pah_fit = helpers.fit_spectrum(pah_obs, pah_model, maxiter=maxiter)

        pah_model.save(pah_fit, label, "ipac")
        pah_model = helpers.initialize_model("$(label)_output.ipac", pah_obs, false)

        compounds = helpers.calculate_compounds(pah_obs, pah_model)
        cont_flux_Jy = (compounds["tot_cont"] .+ compounds["dust_features"]) .* compounds["extinction_model"]

        # Convert to erg/s/cm2/Ang
        F_cont = cont_flux_Jy .* 1e-23 .* (C_MS .* 1e10) ./ (pah_obs["x"].value .* 1e4).^2

        model = pah_model
    end

    return model, F_cont
end

function line_fit_spaxel(λ::Vector{Float64}, F::Vector{Float64}, σ::Vector{Float64}, F_cont::Vector{Float64}, 
    line::Param.TransitionLine; window_size::Float64=250., plot_lines::Bool=false, label::Union{String,Nothing}=nothing)

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
    Fnorm = F[window] .- F_cont[window]
    σnorm = σ[window]

    # Normalization
    N = nanmaximum(Fnorm)
    Fnorm ./= N
    σnorm ./= N

    # Normalized parameter vector
    p₀ = Vector{Float64}()

    if line.profile == :Gaussian
        # Amplitude
        @assert :A ∈ keys(line.parameters)
        append!(p₀, [line.parameters[:A].value / N])

        # Central wavelength
        @assert :voff ∈ keys(line.parameters)
        append!(p₀, [pix_interp(Util.Doppler_shift_λ(line.λ₀, line.parameters[:voff].value))])

        # FWHM
        @assert :FWHM ∈ keys(line.parameters)
        append!(p₀, [(Util.Doppler_shift_λ(line.λ₀, line.parameters[:FWHM].value) - line.λ₀) / Δλ])

        # Function
        p_func = Util.Gaussian
        param_names = [:A, :voff, :FWHM]
    end

    p_to_phys(p) = [p[1] * N, Util.Doppler_shift_v(λ_interp(p[2]), line.λ₀), Util.Doppler_shift_v(p[3] * Δλ + line.λ₀, line.λ₀)]

    # Sum of the ln(prior) distributions
    function ln_prior(p)
        # Convert parameters into physical values
        p_phys = p_to_phys(p)
        return sum([logpdf(line.parameters[key].prior, pᵢ) for (key, pᵢ) ∈ zip(param_names, p_phys)])
    end

    # Overall probability function
    function neg_ln_probability(p, x, y, err)
        model = p_func(x, p)
        return -Util.ln_likelihood(y, model, err) - ln_prior(p)
    end

    # Optimize with Nelder-Mead
    res = optimize(p -> neg_ln_probability(p, pix, Fnorm, σnorm), p₀, NelderMead())
    popt = Optim.minimizer(res)

    # Construct the optimized fit
    line_fit = zeros(size(F)...)
    line_fit[window] = p_func(pix, popt) .* N
    p_phys = p_to_phys(popt)

    # Integrated flux
    ∫F = integrate(λ[window], line_fit[window], SimpsonEven())

    # Signal to Noise ratio
    R = 3000
    SNR = ∫F / Util.∫Gaussian(std(F[norm_window]), (Util.Doppler_shift_λ(line.λ₀, p_phys[3]) - line.λ₀)/R)
    # SNR = p_phys[1] / std(F[norm_window])

    append!(p_phys, [∫F, SNR])

    if plot_lines
        if !isdir("spaxels")
            mkdir("spaxels")
        end

        fig, ax = plt.subplots()
        ax.plot(λ[window], F[window], "k-")
        ax.plot(λ[window], F_cont[window] .+ line_fit[window], "r-")
        plt.savefig("spaxels/$label.pdf", bbox_inches=:tight)
        plt.close()
    end

    return p_phys, line_fit

end

function fit_cube(cube::CubeData.DataCube, line_list::Vector{Param.TransitionLine};
    window_size::Float64=250., continuum_method::Symbol=:pahfit, loc::Union{Float64,Nothing}=nothing, 
    nostart::Bool=false, maxiter::Int=10_000, progress::Bool=false, plot_lines::Bool=false)

    # Get shape
    shape = size(cube.Iλ)

    # Prepare outputs
    # full 3D flux model array
    F_model = zeros(shape...)

    # Nested dictionary -> first layer keys are line names, second layer keys are parameter names, which contain 2D arrays
    line_maps = Dict()
    for line ∈ line_list
        line_maps[line.name] = Dict()
        for pname in [:A, :voff, :FWHM, :∫F, :SNR]
            line_maps[line.name][pname] = ones(shape[1:2]...) .* NaN
        end
    end

    # Alias
    λ = cube.λ
    # Make sure the wavelength vector is linear, since it is assumed later in the function
    diffs = diff(λ)
    @assert diffs[1] ≈ diffs[end]

    # Prepare progress bar
    if progress
        prog = Progress(shape[1] * shape[2], dt=0.01, desc="Fitting cube...", showspeed=true)
    end

    for (xᵢ, yᵢ) ∈ Iterators.product(1:shape[1], 1:shape[2])

        # Filter NaNs
        Fᵢ = cube.Iλ[xᵢ, yᵢ, :]
        σᵢ = cube.σI[xᵢ, yᵢ, :]
        if sum(.!isfinite.(Fᵢ) .| .!isfinite.(σᵢ)) > (shape[3] / 10)
            if progress
                next!(prog)
            end
            continue
        end
        filt = .!isfinite.(Fᵢ) .& .!isfinite.(σᵢ)
        Fᵢ[filt] .= nanmedian(Fᵢ)
        σᵢ[filt] .= nanmedian(σᵢ)
        
        # Fit the continuum with the method given by continuum_method 
        model_cont, F_cont = continuum_fit_spaxel(λ, Fᵢ, σᵢ, "spaxel_$(xᵢ)_$(yᵢ)", method=continuum_method,
            loc=loc, window_size=window_size, nostart=nostart, maxiter=maxiter)
        F_model[xᵢ, yᵢ, :] .+= F_cont

        # Fit each line in line_list 
        for line ∈ line_list

            # Fit the line
            popt, line_fit = line_fit_spaxel(λ, Fᵢ, σᵢ, F_cont, line, window_size=window_size, plot_lines=plot_lines, label="spaxel_$(xᵢ)_$(yᵢ)")
            F_model[xᵢ, yᵢ, :] .+= line_fit

            line_maps[line.name][:A][xᵢ, yᵢ], line_maps[line.name][:voff][xᵢ, yᵢ], line_maps[line.name][:FWHM][xᵢ, yᵢ], 
                line_maps[line.name][:∫F][xᵢ, yᵢ], line_maps[line.name][:SNR][xᵢ, yᵢ] = popt
        end

        if progress
            next!(prog)
        end

    end

    return line_maps, F_model

end

end