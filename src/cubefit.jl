module CubeFit

export fit_cube

# Import packages
using Distributions
using Interpolations
using Dierckx
using NaNStatistics
using Optim
# using LsqFit
using CMPFit
using Distributed
using SharedArrays
using TOML
using CSV
using DataFrames
using NumericalIntegration
using ProgressMeter
using Reexport
using FITSIO
using Cosmology
using Printf
using PyCall
using PyPlot
using PlotlyJS

const py_anchored_artists = PyNULL()
function __init__()
    copy!(py_anchored_artists, pyimport_conda("mpl_toolkits.axes_grid1.anchored_artists", "matplotlib"))
    plt.switch_backend("Agg")
end

include("parameters.jl")
@reexport using .Param

include("cubedata.jl")
@reexport using .CubeData

sourcepath = dirname(Base.source_path())

function parse_options()

    options = TOML.parsefile(joinpath(sourcepath, "options.toml"))
    options_out = Dict()
    keylist1 = ["cosmology", "stellar_continuum_temp", "dust_continuum_temps", "dust_features", "extinction"]
    keylist2 = ["wave", "fwhm"]
    keylist3 = ["tau_9_7", "beta"]
    keylist4 = ["val", "plim", "locked"]
    keylist5 = ["h", "omega_m", "omega_K", "omega_r"]
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
    for key ∈ keylist5
        if !(key ∈ keys(options["cosmology"]))
            error("Missing option $key in cosmology options!")
        end
    end
    
    options_out[:cosmology] = cosmology(h=options["cosmology"]["h"], 
                                        OmegaM=options["cosmology"]["omega_m"],
                                        OmegaK=options["cosmology"]["omega_K"],
                                        OmegaR=options["cosmology"]["omega_r"])
    options_out[:stellar_continuum_temp] = Param.from_dict(options["stellar_continuum_temp"])
    options_out[:dust_continuum_temps] = [Param.from_dict(options["dust_continuum_temps"][i]) for i ∈ 1:length(options["dust_continuum_temps"])]
    options_out[:dust_features] = Dict()
    for df ∈ keys(options["dust_features"])
        options_out[:dust_features][df] = Dict()
        options_out[:dust_features][df][:complex] = "complex" ∈ keys(options["dust_features"][df]) ? options["dust_features"][df]["complex"] : split(df, "_")[2]
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
    keylist1 = ["profile", "latex", "wave", "fwhm"]
    keylist2 = ["val", "plim", "locked"]
    for line ∈ keys(lines)
        for key ∈ keylist1
            if !(key ∈ keys(lines[line]))
                error("$line missing $key option!")
            end
        end
        for key ∈ ["wave", "fwhm"]
            for key2 ∈ keylist2
                if !(key2 ∈ keys(lines[line][key]))
                    error("$line's $key parameter is missing $key2 option!")
                end
            end
        end
        wave = Param.from_dict_wave(lines[line]["wave"])
        fwhm = Param.from_dict_fwhm(lines[line]["fwhm"])
        params = Param.ParamDict(:wave => wave, :fwhm => fwhm)
        lines_out[Symbol(line)] = Param.TransitionLine(Symbol(lines[line]["profile"]), lines[line]["latex"], params)
    end
    return lines_out
end

const options = parse_options() 
const line_list = parse_lines()

struct ParamMaps
    """
    A structure for holding 2D maps of fitting parameters generated when fitting a cube.
    """

    stellar_continuum::Dict{Symbol, Array{Float64, 2}}
    dust_continuum::Dict{Int, Dict{Symbol, Array{Float64, 2}}}
    dust_features::Dict{String, Dict{Symbol, Array{Float64, 2}}}
    lines::Dict{Symbol, Dict{Symbol, Array{Float64, 2}}}
    extinction::Dict{Symbol, Array{Float64, 2}}
    dust_complexes::Dict{String, Dict{Symbol, Array{Float64, 2}}}

end

function parammaps_empty(shape::Tuple{Int,Int,Int}, n_dust_cont::Int, df_names::Vector{String}, 
    complexes::Vector{String}, line_names::Vector{Symbol})

    nan_arr = ones(shape[1:2]...) .* NaN

    # Add stellar continuum fitting parameters
    stellar_continuum = Dict{Symbol, Array{Float64, 2}}()
    stellar_continuum[:amp] = copy(nan_arr)
    stellar_continuum[:temp] = copy(nan_arr)

    # Add dust continuum fitting parameters
    dust_continuum = Dict{Int, Dict{Symbol, Array{Float64, 2}}}()
    for i ∈ 1:n_dust_cont
        dust_continuum[i] = Dict{Symbol, Array{Float64, 2}}()
        dust_continuum[i][:amp] = copy(nan_arr)
        dust_continuum[i][:temp] = copy(nan_arr)
    end

    # Add dust features fitting parameters
    dust_features = Dict{String, Dict{Symbol, Array{Float64, 2}}}()
    for n ∈ df_names
        dust_features[n] = Dict{Symbol, Array{Float64, 2}}()
        dust_features[n][:amp] = copy(nan_arr)
        dust_features[n][:mean] = copy(nan_arr)
        dust_features[n][:fwhm] = copy(nan_arr)
    end

    # Nested dictionary -> first layer keys are line names, second layer keys are parameter names, which contain 2D arrays
    lines = Dict{Symbol, Dict{Symbol, Array{Float64, 2}}}()
    for line ∈ line_names
        lines[line] = Dict{Symbol, Array{Float64, 2}}()
        for pname ∈ [:amp, :voff, :fwhm, :intI, :SNR]
            lines[line][pname] = copy(nan_arr)
        end
    end

    # Add extinction fitting parameters
    extinction = Dict{Symbol, Array{Float64, 2}}()
    extinction[:tau_9_7] = copy(nan_arr)
    extinction[:beta] = copy(nan_arr)

    # Add dusts complexes with extrapolated parameters
    dust_complexes = Dict{String, Dict{Symbol, Array{Float64, 2}}}()
    for c ∈ complexes
        dust_complexes[c] = Dict{Symbol, Array{Float64, 2}}()
        dust_complexes[c][:intI] = copy(nan_arr)
        dust_complexes[c][:SNR] = copy(nan_arr)
    end

    return ParamMaps(stellar_continuum, dust_continuum, dust_features, lines, extinction, dust_complexes)
end


struct CubeModel
    """
    A structure for holding 3D models of intensity, split up into model components, generated when fitting a cube
    """

    model::SharedArray{Float64, 3}
    stellar::SharedArray{Float64, 3}
    dust_continuum::SharedArray{Float64, 4}
    dust_features::SharedArray{Float64, 4}
    extinction::SharedArray{Float64, 3}
    lines::SharedArray{Float64, 4}

    df_names::Vector{String}
    line_names::Vector{Symbol}

end

function cubemodel_empty(shape::Tuple{Int,Int,Int}, n_dust_cont::Int, df_names::Vector{String}, line_names::Vector{Symbol})

    model = SharedArray(zeros(shape...))
    stellar = SharedArray(zeros(shape...))
    dust_continuum = SharedArray(zeros(shape..., n_dust_cont))
    dust_features = SharedArray(zeros(shape..., length(df_names)))
    extinction = SharedArray(zeros(shape...))
    lines = SharedArray(zeros(shape..., length(line_names)))

    return CubeModel(model, stellar, dust_continuum, dust_features, extinction, lines, df_names, line_names)
end

# TODO: make CubeFit into a struct


function levmar_fit_spaxel(λ::Vector{Float64}, I::Vector{Float64}, σ::Vector{Float64}; 
    n_dust_cont::Union{Int,Nothing}=nothing, n_dust_features::Union{Int,Nothing}=nothing, 
    n_lines::Union{Int,Nothing}=nothing, n_complexes::Union{Int,Nothing}=nothing, 
    df_names::Union{Nothing,Vector{String}}=nothing, complexes::Union{Nothing,Vector{String}}=nothing,
    line_names::Union{Nothing,Vector{Symbol}}=nothing, line_profiles::Union{Nothing,Vector{Symbol}}=nothing, 
    line_latex::Union{Nothing,Vector{String}}=nothing, window_size::Float64=.025, plot::Symbol=:none, name::Union{String,Nothing}=nothing, 
    label::Union{String,Nothing}=nothing) where {T<:UnivariateDistribution}

    if plot != :none
        @assert !isnothing(name) && !isnothing(label)
    end

    T_s = options[:stellar_continuum_temp]
    T_dc = options[:dust_continuum_temps]
    τ_97 = options[:extinction][:tau_9_7]
    β = options[:extinction][:beta]
    
    #### PREPARE OUTPUTS ####
    if isnothing(n_dust_cont)
        n_dust_cont = length(options[:dust_continuum_temps])
    end
    if isnothing(df_names)
        df_filt = [(minimum(λ)-0.5 < options[:dust_features][df][:wave].value < maximum(λ)+0.5) for df ∈ keys(options[:dust_features])]
        df_names = Vector{String}(collect(keys(options[:dust_features]))[df_filt])
        df_mean = [parse(Float64, split(df, "_")[2]) for df ∈ df_names]
        ss = sortperm(df_mean)
        df_names = df_names[ss]
    end
    if isnothing(n_dust_features)
        n_dust_features = length(df_names)
    end
    if isnothing(complexes)
        complexes = Vector{String}(unique([options[:dust_features][n][:complex] for n ∈ df_names]))
        ss = sortperm([parse(Float64, c) for c ∈ complexes])
        complexes = complexes[ss]
    end
    if isnothing(n_complexes)
        n_complexes = length(complexes)
    end
    if isnothing(line_names) || isnothing(line_profiles) || isnothing(line_latex)
        line_wave = [line_list[line].parameters[:wave].value for line ∈ keys(line_list)]
        ln_filt = [minimum(λ) < lw < maximum(λ) for lw ∈ line_wave]
        line_names = Vector{Symbol}(collect(keys(line_list))[ln_filt])
        line_profiles = [line_list[line].profile for line ∈ line_names]
        line_latex = [line_list[line].latex for line ∈ line_names]
        ss = sortperm(line_wave[ln_filt])
        line_names = line_names[ss]
        line_profiles = line_profiles[ss]
        line_latex = line_latex[ss]
    end
    if isnothing(n_lines)
        n_lines = length(line_names)
    end

    # Function to interpolate the data with least squares quadratic fitting
    function interp_func(x)
        ind = findmin(abs.(λ .- x))[2]
        lo = ind - 1
        hi = ind + 2
        while lo ≤ 0
            lo += 1
            hi += 1
        end
        while hi > length(λ)
            hi -= 1
            lo -= 1
        end
        A = [λ[lo:hi].^2 λ[lo:hi] ones(4)]
        y = I[lo:hi]
        param = A \ y
        return param[1].*x.^2 .+ param[2].*x .+ param[3]
    end

    # Stellar amplitude
    A_s = clamp(interp_func(5.5) / Util.Blackbody_ν(5.5, T_s.value), 0., Inf)
    # Dust feature amplitudes
    A_df = repeat([clamp(nanmedian(I)/2, 0., Inf)], n_dust_features)
    A_ln = repeat([clamp(nanmedian(I)/2, 0., Inf)], n_lines)
    # Dust continuum amplitudes
    λ_dc = clamp.(2898 ./ [Ti.value for Ti ∈ T_dc], minimum(λ), maximum(λ))
    A_dc = clamp.(interp_func.(λ_dc) ./ [Util.Blackbody_ν(λ_dci, T_dci.value) for (λ_dci, T_dci) ∈ zip(λ_dc, T_dc)] .* (λ_dc ./ 9.7).^2 ./ 5., 0., Inf)
    
    amp_dc_prior = Uniform(0., 1e100)
    amp_df_prior = Uniform(0., nanmaximum(I) > 0. ? nanmaximum(I) : 1e100)
    amp_ln_prior = Uniform(0., nanmaximum(I) > 0. ? nanmaximum(I) : 1e100)
    
    mean_df = [options[:dust_features][df][:wave] for df ∈ df_names]
    fwhm_df = [options[:dust_features][df][:fwhm] for df ∈ df_names]

    wave_ln = [line_list[ln].parameters[:wave] for ln ∈ line_names]
    fwhm_ln = [line_list[ln].parameters[:fwhm] for ln ∈ line_names]

    stellar_pars = [A_s, T_s.value]
    # stellar_pnames = ["A_s", "T_s"]
    stellar_priors = [amp_dc_prior, T_s.prior]

    dc_pars = vcat([[Ai, Ti.value] for (Ai, Ti) ∈ zip(A_dc, T_dc)]...)
    # dc_pnames = vcat([["A_$i", "T_$i"] for i ∈ 1:n_dust_cont]...)
    dc_priors = vcat([[amp_dc_prior, Ti.prior] for Ti ∈ T_dc]...)

    df_pars = vcat([[Ai, mi.value, fi.value] for (Ai, mi, fi) ∈ zip(A_df, mean_df, fwhm_df)]...)
    # df_pnames = vcat([["A_$i", "μ_$i", "FWHM_$i"] for i ∈ 1:n_dust_features]...)
    df_priors = vcat([[amp_df_prior, mi.prior, fi.prior] for (mi, fi) ∈ zip(mean_df, fwhm_df)]...)

    ln_pars = vcat([[Ai, wi.value, fi.value] for (Ai, wi, fi) ∈ zip(A_ln, wave_ln, fwhm_ln)]...)
    ln_priors = vcat([[amp_ln_prior, wi.value, fi.value] for (wi, fi) ∈ zip(wave_ln, fwhm_ln)]...)

    # Initial parameter vector
    p₀ = vcat(stellar_pars, dc_pars, df_pars, ln_pars, [τ_97.value], [β.value])
    # pnames = vcat(stellar_pnames, dc_pnames, df_pnames, ln_pnames, ["τ_9.7"], ["β"])
    priors = vcat(stellar_priors, dc_priors, df_priors, ln_priors, [τ_97.prior], [β.prior])

    # Convert parameter limits into CMPFit object
    parinfo = CMPFit.Parinfo(length(p₀))

    # Stellar amplitude
    parinfo[1].limited = (1,0)
    parinfo[1].limits = (0., 0.)
    # Stellar temp
    parinfo[2].fixed = T_s.locked
    if !(T_s.locked)
        parinfo[2].limited = (1,1)
        parinfo[2].limits = (minimum(T_s.prior), maximum(T_s.prior))
    end

    # Dust cont amplitudes and temps
    pᵢ = 3
    for i ∈ 1:n_dust_cont
        parinfo[pᵢ].limited = (1,0)
        parinfo[pᵢ].limits = (0., 0.)
        parinfo[pᵢ+1].fixed = T_dc[i].locked
        if !(T_dc[i].locked)
            parinfo[pᵢ+1].limited = (1,1)
            parinfo[pᵢ+1].limits = (minimum(T_dc[i].prior), maximum(T_dc[i].prior))
        end
        pᵢ += 2
    end

    # Dust feature amplitude, mean, fwhm
    for i ∈ 1:n_dust_features
        parinfo[pᵢ].limited = (1,0)
        parinfo[pᵢ].limits = (0., 0.)
        parinfo[pᵢ+1].fixed = mean_df[i].locked
        if !(mean_df[i].locked)
            parinfo[pᵢ+1].limited = (1,1)
            parinfo[pᵢ+1].limits = (minimum(mean_df[i].prior), maximum(mean_df[i].prior))
        end
        parinfo[pᵢ+2].fixed = fwhm_df[i].locked
        if !(fwhm_df[i].locked)
            parinfo[pᵢ+2].limited = (1,1)
            parinfo[pᵢ+2].limits = (minimum(fwhm_df[i].prior), maximum(mean_df[i].prior))
        end
        pᵢ += 3
    end

    # Emission line amplitude, mean, fwhm
    for i ∈ 1:n_lines
        parinfo[pᵢ].limited = (1,0)
        parinfo[pᵢ].limits = (0., 0.)
        parinfo[pᵢ+1].fixed = wave_ln[i].locked
        if !(wave_ln[i].locked)
            parinfo[pᵢ+1].limited = (1,1)
            parinfo[pᵢ+1].limits = (minimum(wave_ln[i].prior), maximum(wave_ln[i].prior))
        end
        parinfo[pᵢ+2].fixed = fwhm_ln[i].locked
        if !(fwhm_ln[i].locked)
            parinfo[pᵢ+2].limited = (1,1)
            parinfo[pᵢ+2].limits = (minimum(fwhm_ln[i].prior), maximum(fwhm_ln[i].prior))
        end
        pᵢ += 3
    end

    # Extinction
    parinfo[pᵢ].fixed = τ_97.locked
    if !(τ_97.locked)
        parinfo[pᵢ].limited = (1,1)
        parinfo[pᵢ].limits = (minimum(τ_97.prior), maximum(τ_97.prior))
    end
    parinfo[pᵢ+1].fixed = β.locked
    if !(β.locked)
        parinfo[pᵢ+1].limited = (1,1)
        parinfo[pᵢ+1].limits = (minimum(β.prior), maximum(β.prior))
    end

    # Create a `config` structure
    config = CMPFit.Config()

    res = cmpfit(λ, I, σ, (x, p) -> Util.fit_spectrum(x, p, n_dust_cont, n_dust_features, n_lines, line_profiles), 
        p₀, parinfo=parinfo, config=config)
    popt = res.param
    χ2 = res.bestnorm
    χ2red = res.bestnorm / res.dof

    # Final optimized fit
    I_model, comps = Util.fit_spectrum(λ, popt, n_dust_cont, n_dust_features, n_lines, line_profiles, return_components=true)

    # Prepare outputs
    p_out = zeros(4 + 2n_dust_cont + 3n_dust_features + 5n_lines + 2n_complexes)
    p_out[1] = popt[1] > 0. ? log10(popt[1]) : -Inf    # log stellar amp
    p_out[2] = popt[2]    # stellar temp
    p_i = 3
    for i ∈ 1:n_dust_cont
        p_out[p_i] = popt[p_i] > 0. ? log10(popt[p_i]) : -Inf  # log amp
        p_out[p_i+1] = popt[p_i+1]  # temp
        p_i += 2
    end
    for j ∈ 1:n_dust_features
        p_out[p_i] = popt[p_i] > 0. ? log10(popt[p_i]) : -Inf   # log amp
        p_out[p_i+1] = popt[p_i+1]  # mean
        p_out[p_i+2] = popt[p_i+2]  # fwhm
        p_i += 3
    end
    p_o = p_i
    for (k, ln) ∈ enumerate(line_names)
        p_out[p_o] = popt[p_i] > 0. ? log10(popt[p_i]) : -Inf   # log amp
        # Convert mean wavelength into velocity shift (km/s)
        p_out[p_o+1] = Util.Doppler_shift_v(popt[p_i+1], line_list[ln].parameters[:wave].value)
        # Convert fwhm to km/s
        p_out[p_o+2] = Util.Doppler_shift_v(popt[p_i+2] + line_list[ln].parameters[:wave].value, line_list[ln].parameters[:wave].value)

        # Line intensity
        profile = comps["line_$(k)"]
        window = line_list[ln].parameters[:wave].value
        p_out[p_o+3] = NumericalIntegration.integrate(λ, profile, SimpsonEven())
        p_out[p_o+3] = p_out[p_o+3] > 0. ? log10(p_out[p_o+3]) : -Inf

        # SNR
        p_out[p_o+4] = p_out[p_o+3] /
            std(I[(window-2window_size .< λ .< window-window_size) .| (window+window_size .< λ .< window+2window_size)])
        
        p_i += 3
        p_o += 5
    end
    p_out[p_o] = popt[p_i]       # tau_97
    p_out[p_o+1] = popt[p_i+1]   # beta
    p_o += 2
    for c ∈ complexes
        Iᵢ = zeros(length(λ))
        # Add up the dust feature profiles that belong to this complex
        for (ii, df) ∈ enumerate(df_names)
            if options[:dust_features][df][:complex] == c
                Iᵢ .+= comps["dust_feat_$ii"]
            end
        end
        # Integrate the intensity of the combined profile
        window = parse(Float64, c)
        p_out[p_o] = NumericalIntegration.integrate(λ, Iᵢ, SimpsonEven())
        p_out[p_o] = p_out[p_o] > 0. ? log10(p_out[p_o]) : -Inf
        p_out[p_o+1] = maximum(Iᵢ) / 
            std(I[(window-2window_size .< λ .< window-window_size) .| (window+window_size .< λ .< window+2window_size)])
        p_o += 2
    end 

    if plot != :none
        plot_spaxel_fit(λ, I, I_model, comps, n_dust_cont, n_dust_features, [wl.value for wl ∈ wave_ln], 
            line_latex, χ2red, name, label, backend=plot)
    end

    return p_out, I_model, comps

end

function continuum_pahfit_spaxel(λ::Vector{Float64}, I::Vector{Float64}, σ::Vector{Float64}, x::Int, y::Int)

    folder = joinpath(@__DIR__, "idl_data")
    if !isdir(folder)
        mkdir(folder)
    end

    if !isfile("$folder/spaxel_$(x)_$(y)_params.csv") || !isfile("$folder/spaxel_$(x)_$(y)_fit.csv")    
        CSV.write("$folder/spaxel_$(x)_$(y).csv", DataFrame(wave=λ, intensity=I, err=σ))
        IDL_DIR = ENV["IDL_DIR"]
        run(`$IDL_DIR/bin/idl $(@__DIR__)/fit_spaxel.pro -args $(@__DIR__)/idl_data/spaxel_$(x)_$(y).csv`);
    end

    popt = CSV.read("$folder/spaxel_$(x)_$(y)_params.csv", DataFrame)[!, :params]
    I_cont = CSV.read("$folder/spaxel_$(x)_$(y)_fit.csv", DataFrame)[!, :intensity]

    return popt, I_cont

end


function fit_cube(cube::CubeData.DataCube; window_size::Float64=.025, plot_spaxels::Symbol=:none,
    plot_maps::Bool=true, parallel::Bool=true, save_fits::Bool=true, name::Union{String,Nothing}=nothing, 
    Ω::Union{Float64,Nothing}=nothing, z::Union{Float64,Nothing}=nothing)

    if (plot_maps || save_fits) && isnothing(name)
        error("Please provide a name to plot and/or save the FITS files with!")
    end

    # Get shape
    shape = size(cube.Iλ)
    # Alias
    λ = cube.λ
    # Make sure the wavelength vector is linear, since it is assumed later in the function
    diffs = diff(λ)
    @assert diffs[1] ≈ diffs[end]
    Δλ = diffs[1]

    #### PREPARE OUTPUTS ####
    n_dust_cont = length(options[:dust_continuum_temps])

    df_filt = [(minimum(λ)-0.5 < options[:dust_features][df][:wave].value < maximum(λ)+0.5) for df ∈ keys(options[:dust_features])]
    df_names = Vector{String}(collect(keys(options[:dust_features]))[df_filt])
    df_mean = [parse(Float64, split(df, "_")[2]) for df ∈ df_names]
    ss = sortperm(df_mean)
    df_names = df_names[ss]
    n_dust_features = length(df_names)

    complexes = Vector{String}(unique([options[:dust_features][n][:complex] for n ∈ df_names]))
    ss = sortperm([parse(Float64, c) for c ∈ complexes])
    complexes = complexes[ss]
    n_complexes = length(complexes)

    line_wave = [line_list[line].parameters[:wave].value for line ∈ keys(line_list)]
    ln_filt = [minimum(λ) < lw < maximum(λ) for lw ∈ line_wave]
    line_names = Vector{Symbol}(collect(keys(line_list))[ln_filt])
    line_profiles = [line_list[line].profile for line ∈ line_names]
    line_latex = [line_list[line].latex for line ∈ line_names]
    ss = sortperm(line_wave[ln_filt])
    line_wave = line_wave[ss]
    line_names = line_names[ss]
    line_profiles = line_profiles[ss]
    line_latex = line_latex[ss]
    n_lines = length(line_names)

    # FULL 3D flux model array
    cube_model = cubemodel_empty(shape, n_dust_cont, df_names, line_names)
    # 2D maps of fitting parameters
    param_maps = parammaps_empty(shape, n_dust_cont, df_names, complexes, line_names)

    # Shared output parameter array
    n_params = 4 + 2n_dust_cont + 3n_dust_features + 5n_lines + 2n_complexes
    out_params = SharedArray(ones(shape[1:2]..., n_params) .* NaN)

    #########################

    # Utility function for fitting a single spaxel
    function fit_spaxel(xᵢ::Int, yᵢ::Int)

        # Filter NaNs
        Iᵢ = cube.Iλ[xᵢ, yᵢ, :]
        σᵢ = cube.σI[xᵢ, yᵢ, :]
        if sum(.!isfinite.(Iᵢ) .| .!isfinite.(σᵢ)) > (shape[3] / 10)
            return
        end
        filt = .!isfinite.(Iᵢ) .& .!isfinite.(σᵢ)

        # Interpolate the NaNs
        if sum(filt) > 0
            # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
            λknots = λ[length(λ) ÷ 50]:Δλ*100:λ[end-(length(λ) ÷ 50)]
            # ONLY replace NaN values, keep the rest of the data as-is
            Iᵢ[filt] .= Spline1D(λ[isfinite.(Iᵢ)], Iᵢ[isfinite.(Iᵢ)], λknots, k=3, bc="extrapolate").(λ[filt])
            σᵢ[filt] .= Spline1D(λ[isfinite.(σᵢ)], σᵢ[isfinite.(σᵢ)], λknots, k=3, bc="extrapolate").(λ[filt])
        end
        
        # Fit the continuum
        p_out, I_model, comps = levmar_fit_spaxel(λ, Iᵢ, σᵢ, n_dust_cont=n_dust_cont, n_dust_features=n_dust_features, 
            n_lines=n_lines, n_complexes=n_complexes, df_names=df_names, complexes=complexes, line_names=line_names, 
            line_profiles=line_profiles, line_latex=line_latex, plot=plot_spaxels, name=name, label="spaxel_$(xᵢ)_$(yᵢ)")
        # p_cont, I_cont = continuum_pahfit_spaxel(λ, Iᵢ, σᵢ, xᵢ, yᵢ)

        # Set the 3D intensity maps
        cube_model.model[xᵢ, yᵢ, :] .= I_model
        cube_model.stellar[xᵢ, yᵢ, :] .= comps["stellar"]
        for i ∈ 1:n_dust_cont
            cube_model.dust_continuum[xᵢ, yᵢ, :, i] .= comps["dust_cont_$i"]
        end
        for j ∈ 1:n_dust_features
            cube_model.dust_features[xᵢ, yᵢ, :, j] .= comps["dust_feat_$j"]
        end
        for k ∈ 1:n_lines
            cube_model.lines[xᵢ, yᵢ, :, k] .= comps["line_$k"]
        end
        cube_model.extinction[xᵢ, yᵢ, :] .= comps["extinction"]

        # Set the 2D raw parameter outputs
        out_params[xᵢ, yᵢ, :] .= p_out
        
        return
    end

    # Use multiprocessing (not threading) to iterate over multiple spaxels at once using multiple CPUs
    if parallel
        @showprogress pmap(Iterators.product(1:shape[1], 1:shape[2])) do (xᵢ, yᵢ)
            fit_spaxel(xᵢ, yᵢ)
        end
    else
        @showprogress for (xᵢ, yᵢ) ∈ Iterators.product(1:shape[1], 1:shape[2])
            fit_spaxel(xᵢ, yᵢ)
        end
    end

    # Set the 2D parameter map outputs
    param_maps.stellar_continuum[:amp] .= out_params[:, :, 1]
    param_maps.stellar_continuum[:temp] .= out_params[:, :, 2]
    pᵢ = 3
    for i ∈ 1:n_dust_cont
        param_maps.dust_continuum[i][:amp] .= out_params[:, :, pᵢ]
        param_maps.dust_continuum[i][:temp] .= out_params[:, :, pᵢ+1]
        pᵢ += 2
    end
    for df ∈ df_names
        param_maps.dust_features[df][:amp] .= out_params[:, :, pᵢ]
        param_maps.dust_features[df][:mean] .= out_params[:, :, pᵢ+1]
        param_maps.dust_features[df][:fwhm] .= out_params[:, :, pᵢ+2]
        pᵢ += 3
    end
    for ln ∈ line_names
        param_maps.lines[ln][:amp] .= out_params[:, :, pᵢ]
        param_maps.lines[ln][:voff] .= out_params[:, :, pᵢ+1]
        param_maps.lines[ln][:fwhm] .= out_params[:, :, pᵢ+2]
        param_maps.lines[ln][:intI] .= out_params[:, :, pᵢ+3]
        param_maps.lines[ln][:SNR] .= out_params[:, :, pᵢ+4]
        pᵢ += 5
    end
    param_maps.extinction[:tau_9_7] .= out_params[:, :, pᵢ]
    param_maps.extinction[:beta] .= out_params[:, :, pᵢ+1]
    pᵢ += 2
    for c ∈ complexes
        param_maps.dust_complexes[c][:intI] .= out_params[:, :, pᵢ]
        param_maps.dust_complexes[c][:SNR] .= out_params[:, :, pᵢ+1]
        pᵢ += 2
    end

    if plot_maps
        plot_parameter_maps(param_maps, name, cube.Ω, z)
    end

    if save_fits
        write_fits(cube, param_maps, cube_model, name, z)
    end

    return param_maps, cube_model

end

# Plotting utility function
function plot_parameter_map(data::Union{Matrix{Float64},SharedMatrix{Float64}}, name::String, name_i::String;
    Ω::Union{Float64,Nothing}=nothing, z::Union{Float64,Nothing}=nothing, 
    snr_filter::Union{Matrix{Float64},SharedMatrix{Float64},Nothing}=nothing, snr_thresh::Float64=3.)
    """
    Plotting function for 2D parameter maps which are output by fit_cube
    :param data: Matrix{Float64}
        The 2D parameter map
    :param name: String
        The name of the target, i.e. "NGC_7469"
    :param name_i: String
        The name of the parameter, i.e. "dust_features_PAH_5.24_amp"
    :param z: Float64
        The redshift of the target (for scalebar in pc)
    :param snr_filter: Matrix{Float64}
        A 2D SNR map that can be used to filter the "data" map to only show spaxels above snr_thresh
    :param snr_thresh: Float64
        The SNR threshold used to filter the data map, if snr_filter is given
    """

    # 😬 I know this is ugly but I couldn't figure out a better way to do it lmao
    if occursin("amp", String(name_i))
        bunit = "\$\\log_{10}(I /\$MJy sr\$^{-1})\$"
    elseif occursin("temp", String(name_i))
        bunit = "\$T\$ (K)"
    elseif occursin("fwhm", String(name_i)) && occursin("PAH", String(name_i))
        bunit = "FWHM (\$\\mu\$m)"
    elseif occursin("fwhm", String(name_i)) && !occursin("PAH", String(name_i))
        bunit = "FWHM (km s\$^{-1}\$)"
    elseif occursin("mean", String(name_i))
        bunit = "\$\\mu\$ (\$\\mu\$m)"
    elseif occursin("voff", String(name_i))
        bunit = "\$v_{\\rm off}\$ (km s\$^{-1}\$)"
    elseif occursin("SNR", String(name_i))
        bunit = "\$S/N\$"
    elseif occursin("beta", String(name_i))
        bunit = "\$\\beta\$"
    elseif occursin("tau", String(name_i))
        bunit = "\$\\tau_{9.7}\$"
    elseif occursin("intI", String(name_i))
        bunit = "\$\\log_{10}(I /\$MJy sr\$^{-1}\$ \$\\mu\$m)"
    end

    filtered = copy(data)
    if !isnothing(snr_filter)
        filtered[snr_filter .≤ snr_thresh] .= NaN
    end

    fig = plt.figure()
    ax = plt.subplot()
    flatdata = filtered[isfinite.(filtered)]
    vmin = length(flatdata) > 0 ? quantile(flatdata, 0.01) : 0.0
    vmax = length(flatdata) > 0 ? quantile(flatdata, 0.99) : 0.0
    cdata = ax.imshow(filtered', origin=:lower, cmap=:magma, vmin=vmin, vmax=vmax)
    ax.axis(:off)

    if !isnothing(Ω) && !isnothing(z)
        n_pix = 1/(sqrt(Ω) * 180/π * 3600)
        dL = luminosity_dist(options[:cosmology], z).val * 1e6 / (180/π * 3600)  # l = d * theta (1")
        dL = @sprintf "%.0f" dL
        scalebar = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, "1\$\'\'\$ / $dL pc", "lower left", pad=1, color=:black, 
            frameon=false, size_vertical=0.2, label_top=true)
        ax.add_artist(scalebar)
    end

    fig.colorbar(cdata, ax=ax, label=bunit)
    plt.savefig("output_$(name)/$(name_i).pdf", dpi=300, bbox_inches=:tight)
    plt.close()

end

function plot_parameter_maps(param_maps::ParamMaps, name::String, Ω::Union{Float64,Nothing}=nothing,
    z::Union{Float64,Nothing}=nothing; snr_thresh=3.)

    # Remove any spaces in name
    name = replace(name, " " => "_")
    # Make subdirectory
    if !isdir("output_$name")
        mkdir("output_$name")
    end

    # Iterate over model parameters and make 2D maps
    for parameter ∈ keys(param_maps.stellar_continuum)
        data = param_maps.stellar_continuum[parameter]
        name_i = join(["stellar_continuum", parameter], "_")
        plot_parameter_map(data, name, name_i, Ω=Ω, z=z)
    end
    for i ∈ keys(param_maps.dust_continuum)
        for parameter ∈ keys(param_maps.dust_continuum[i])
            data = param_maps.dust_continuum[i][parameter]
            name_i = join(["dust_continuum", i, parameter], "_")
            plot_parameter_map(data, name, name_i, Ω=Ω, z=z)
        end
    end
    for df ∈ keys(param_maps.dust_features)
        for parameter ∈ keys(param_maps.dust_features[df])
            data = param_maps.dust_features[df][parameter]
            name_i = join(["dust_features", df, parameter], "_")
            plot_parameter_map(data, name, name_i, Ω=Ω, z=z)
        end
    end
    for line ∈ keys(param_maps.lines)
        snr = param_maps.lines[line][:SNR]
        for parameter ∈ keys(param_maps.lines[line])
            data = param_maps.lines[line][parameter]
            name_i = join(["lines", line, parameter], "_")
            plot_parameter_map(data, name, name_i, Ω=Ω, z=z, snr_filter=parameter ≠ :SNR ? snr : nothing, snr_thresh=snr_thresh)
        end
    end
    for parameter ∈ keys(param_maps.extinction)
        data = param_maps.extinction[parameter]
        name_i = join(["extinction", parameter], "_")
        plot_parameter_map(data, name, name_i, Ω=Ω, z=z)
    end
    for c ∈ keys(param_maps.dust_complexes)
        snr = param_maps.dust_complexes[c][:SNR]
        for parameter ∈ keys(param_maps.dust_complexes[c])
            data = param_maps.dust_complexes[c][parameter]
            name_i = join(["dust_complexes", c, parameter], "_")
            plot_parameter_map(data, name, name_i, Ω=Ω, z=z, snr_filter=parameter ≠ :SNR ? snr : nothing, snr_thresh=snr_thresh)
        end
    end

end

function write_fits(cube::CubeData.DataCube, param_maps::ParamMaps, cube_model::CubeModel, name::String, z::Float64)

    # Remove any spaces in name
    name = replace(name, " " => "_")
    # Make subdirectory
    if !isdir("output_$name")
        mkdir("output_$name")
    end

    # Header information
    hdr = FITSHeader(
        ["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "PIXAR_SR", "RA", "DEC", "WCSAXES",
            "CDELT1", "CDELT2", "CDELT3", "CTYPE1", "CTYPE2", "CTYPE3", "CRPIX1", "CRPIX2", "CRPIX3",
            "CRVAL1", "CRVAL2", "CRVAL3", "CUNIT1", "CUNIT2", "CUNIT3", "PC1_1", "PC1_2", "PC1_3", 
            "PC2_1", "PC2_2", "PC2_3", "PC3_1", "PC3_2", "PC3_3"],

        # Check if the redshift correction is right for the third WCS axis?
        [name, z, cube.channel, cube.band, cube.Ω, cube.α, cube.δ, cube.wcs.naxis,
            cube.wcs.cdelt[1], cube.wcs.cdelt[2], cube.wcs.cdelt[3]/(1+z), cube.wcs.ctype[1], cube.wcs.ctype[2], cube.wcs.ctype[3], 
            cube.wcs.crpix[1], cube.wcs.crpix[2], cube.wcs.crpix[3], cube.wcs.crval[1], cube.wcs.crval[2], cube.wcs.crval[3]/(1+z),
            cube.wcs.cunit[1], cube.wcs.cunit[2], cube.wcs.cunit[3], cube.wcs.pc[1,1], cube.wcs.pc[1,2], cube.wcs.pc[1,3],
            cube.wcs.pc[2,1], cube.wcs.pc[2,2], cube.wcs.pc[2,3], cube.wcs.pc[3,1], cube.wcs.pc[3,2], cube.wcs.pc[3,3]],

        ["Target name", "Target redshift", "MIRI channel", "MIRI band",
        "Solid angle per pixel (rad.)", "Right ascension of target (deg.)", "Declination of target (deg.)",
        "number of World Coordinate System axes", 
        "first axis increment per pixel", "second axis increment per pixel", "third axis increment per pixel", 
        "first axis coordinate type", "second axis coordinate type", "third axis coordinate type", 
        "axis 1 coordinate of the reference pixel", "axis 2 coordinate of the reference pixel", "axis 3 coordinate of the reference pixel",
        "first axis value at the reference pixel", "second axis value at the reference pixel", "third axis value at the reference pixel",
        "first axis units", "second axis units", "third axis units",
        "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element",
        "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element",
        "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element"]
    )

    FITS("output_$(name)/$(name)_3D_model.fits", "w") do f
        write(f, Vector{Int}())                                                                     # Primary HDU (empty)
        write(f, cube_model.model; header=hdr, name="MODEL")                                        # Full intensity model
        write(f, cube_model.stellar; header=hdr, name="STELLAR_CONTINUUM")                          # Stellar continuum model
        for i ∈ 1:size(cube_model.dust_continuum, 4)
            write(f, cube_model.dust_continuum[:, :, :, i]; header=hdr, name="DUST_CONTINUUM_$i")   # Dust continuum models
        end
        for (j, df) ∈ enumerate(cube_model.df_names)
            write(f, cube_model.dust_features[:, :, :, j]; header=hdr, name="$df")                  # Dust feature profiles
        end
        for (k, line) ∈ enumerate(cube_model.line_names)
            write(f, cube_model.lines[:, :, :, k]; header=hdr, name="$line")                        # Emission line profiles
        end
        write(f, cube_model.extinction; header=hdr, name="EXTINCTION")                              # Extinction model
        write_key(f["MODEL"], "BUNIT", "MJY/sr")
    end

    FITS("output_$(name)/$(name)_parameter_maps.fits", "w") do f

        write(f, Vector{Int}())  # Primary HDU (empty)

        # Iterate over model parameters and make 2D maps
        for parameter ∈ keys(param_maps.stellar_continuum)
            data = param_maps.stellar_continuum[parameter]
            name_i = join(["stellar_continuum", parameter], "_")
            if occursin("amp", String(name_i))
                bunit = "log10(I / MJy sr^-1)"
            elseif occursin("temp", String(name_i))
                bunit = "Kelvin"
            end
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end
        for i ∈ keys(param_maps.dust_continuum)
            for parameter ∈ keys(param_maps.dust_continuum[i])
                data = param_maps.dust_continuum[i][parameter]
                name_i = join(["dust_continuum", i, parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / MJy sr^-1)"
                elseif occursin("temp", String(name_i))
                    bunit = "Kelvin"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)  
            end
        end
        for df ∈ keys(param_maps.dust_features)
            for parameter ∈ keys(param_maps.dust_features[df])
                data = param_maps.dust_features[df][parameter]
                name_i = join(["dust_features", df, parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / MJy sr^-1)"
                elseif occursin("fwhm", String(name_i)) || occursin("mean", String(name_i))
                    bunit = "um"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)      
            end
        end
        for c ∈ keys(param_maps.dust_complexes)
            snr = param_maps.dust_complexes[c][:SNR]
            for parameter ∈ keys(param_maps.dust_complexes[c])
                data = param_maps.dust_complexes[c][parameter]
                name_i = join(["dust_complexes", c, parameter], "_")
                if occursin("intI", String(name_i))
                    bunit = "log10(I / MJy sr^-1 um)"
                elseif occursin("SNR", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)
            end
        end
        for line ∈ keys(param_maps.lines)
            for parameter ∈ keys(param_maps.lines[line])
                data = param_maps.lines[line][parameter]
                name_i = join(["lines", line, parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / MJy sr^-1)"
                elseif occursin("fwhm", String(name_i)) || occursin("voff", String(name_i))
                    bunit = "km/s"
                elseif occursin("intI", String(name_i))
                    bunit = "log10(I / MJy sr^-1 um)"
                elseif occursin("SNR", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)   
            end
        end
        for parameter ∈ keys(param_maps.extinction)
            data = param_maps.extinction[parameter]
            name_i = join(["extinction", parameter], "_")
            bunit = "unitless"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)  
        end

    end
end

function plot_spaxel_fit(λ::Vector{Float64}, I::Vector{Float64}, I_cont::Vector{Float64}, comps::Dict{String, Vector{Float64}}, 
    n_dust_cont::Int, n_dust_features::Int, line_wave::Vector{Float64}, line_latex::Vector{String}, χ2red::Float64, name::String, 
    label::String; backend::Symbol=:pyplot)

    # Remove any spaces in name
    name = replace(name, " " => "_")
    # Make subdirectory
    if !isdir("output_$name")
        mkdir("output_$name")
    end

    if !isdir("output_$name/spaxel_fits")
        mkdir("output_$name/spaxel_fits")
    end

    if backend == :plotly
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
            elseif occursin("line", comp)
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"], mode="lines",
                    line=Dict(:color => "rebeccapurple", :width => 1), name="Lines")])
            end
        end
        for (lw, ll) ∈ zip(line_wave, line_latex)
            append!(traces, [PlotlyJS.scatter(x=[lw, lw], y=[0., nanmaximum(I)*1.1], mode="lines", line=Dict(:color => "rebeccapurple", :width => 0.5, :dash => "dash"))])
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
            template="plotly_white",
            annotations=[
                attr(x=lw, y=nanmaximum(I)*.75, text=ll) for (lw, ll) ∈ zip(line_wave, line_latex)
            ]
        )
        p = PlotlyJS.plot(traces, layout)
        PlotlyJS.savefig(p, isnothing(label) ? "output_$name/spaxel_fits/levmar_fit_spaxel.html" : "output_$name/spaxel_fits/$label.html")

    elseif backend == :pyplot
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
            elseif occursin("line", comp)
                ax1.plot(λ, comps[comp] .* comps["extinction"], "-", color=:rebeccapurple, alpha=0.5)
            end
        end
        for (lw, ll) ∈ zip(line_wave, line_latex)
            ax1.axvline(lw, linestyle="--", color=:rebeccapurple, lw=0.5, alpha=0.5)
            ax1.text(lw+.05, nanmaximum(I)/2, ll)
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
        plt.savefig(isnothing(label) ? "output_$name/spaxel_fits/levmar_fit_spaxel.pdf" : "output_$name/spaxel_fits/$label.pdf", dpi=300, bbox_inches="tight")
        plt.close()
    end
end

end