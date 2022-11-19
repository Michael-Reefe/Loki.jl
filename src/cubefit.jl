module CubeFit

export CubeFitter, fit_cube, continuum_fit_spaxel, line_fit_spaxel, fit_spaxel, plot_parameter_maps, write_fits

# Import packages
using Distributions
using Interpolations
using Dierckx
using NaNStatistics
using Optim
# using LsqFit
using CMPFit
# using NLopt
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
const scipy_opt = PyNULL()
function __init__()
    copy!(py_anchored_artists, pyimport_conda("mpl_toolkits.axes_grid1.anchored_artists", "matplotlib"))
    copy!(scipy_opt, pyimport_conda("scipy.optimize", "scipy"))
    plt.switch_backend("Agg")
end

include("parameters.jl")
@reexport using .Param

include("cubedata.jl")
@reexport using .CubeData

const sourcepath = dirname(Base.source_path())

# MATPLOTLIB SETTINGS TO MAKE PLOTS LOOK PRETTY :)
const SMALL = 12
const MED = 14
const BIG = 16

plt.rc("font", size=MED)          # controls default text sizes
plt.rc("axes", titlesize=MED)     # fontsize of the axes title
plt.rc("axes", labelsize=MED)     # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL)  # fontsize of the tick labels
plt.rc("legend", fontsize=MED)    # legend fontsize
plt.rc("figure", titlesize=BIG)   # fontsize of the figure title
plt.rc("text", usetex=true)
plt.rc("font", family="Times New Roman")


function parse_options()

    options = TOML.parsefile(joinpath(sourcepath, "options.toml"))
    options_out = Dict()
    keylist1 = ["chi2_threshold", "cosmology"]
    keylist2 = ["h", "omega_m", "omega_K", "omega_r"]
    for key ∈ keylist1 
        if !(key ∈ keys(options))
            error("Missing option $key in options file!")
        end
    end
    for key ∈ keylist2
        if !(key ∈ keys(options["cosmology"]))
            error("Missing option $key in cosmology options!")
        end
    end
    
    options_out[:chi2_threshold] = options["chi2_threshold"]
    options_out[:cosmology] = cosmology(h=options["cosmology"]["h"], 
                                        OmegaM=options["cosmology"]["omega_m"],
                                        OmegaK=options["cosmology"]["omega_K"],
                                        OmegaR=options["cosmology"]["omega_r"])
    
    return options_out
end

function parse_dust()

    dust = TOML.parsefile(joinpath(sourcepath, "dust.toml"))
    dust_out = Dict()
    keylist1 = ["stellar_continuum_temp", "dust_continuum_temps", "dust_features", "extinction"]
    keylist2 = ["wave", "fwhm"]
    keylist3 = ["tau_9_7", "beta"]
    keylist4 = ["val", "plim", "locked"]
    for key ∈ keylist1
        if !(key ∈ keys(dust))
            error("Missing option $key in dust file!")
        end
    end
    for key ∈ keylist4
        if !(key ∈ keys(dust["stellar_continuum_temp"]))
            error("Missing option $key in stellar continuum temp options!")
        end
        for dc ∈ dust["dust_continuum_temps"]
            if !(key ∈ keys(dc))
                error("Missing option $key in dust continuum temp options!")
            end
        end
        for df_key ∈ keys(dust["dust_features"])
            for df_key2 ∈ keylist2
                if !(df_key2 ∈ keys(dust["dust_features"][df_key]))
                    error("Missing option $df_key2 in dust feature $df_key options!")
                end
                if !(key ∈ keys(dust["dust_features"][df_key][df_key2]))
                    error("Missing option $key in dust features $df_key, $df_key2 options!")
                end
            end
        end
        for ex_key ∈ keylist3
            if !(ex_key ∈ keys(dust["extinction"]))
                error("Missing option $ex_key in extinction options!")
            end
            if !(key ∈ keys(dust["extinction"][ex_key]))
                error("Missing option $key in $ex_key options!")
            end
        end
    end

    dust_out[:stellar_continuum_temp] = Param.from_dict(dust["stellar_continuum_temp"])
    dust_out[:dust_continuum_temps] = [Param.from_dict(dust["dust_continuum_temps"][i]) for i ∈ 1:length(dust["dust_continuum_temps"])]
    dust_out[:dust_features] = Dict()
    for df ∈ keys(dust["dust_features"])
        dust_out[:dust_features][df] = Dict()
        dust_out[:dust_features][df][:complex] = "complex" ∈ keys(dust["dust_features"][df]) ? dust["dust_features"][df]["complex"] : split(df, "_")[2]
        dust_out[:dust_features][df][:wave] = Param.from_dict_wave(dust["dust_features"][df]["wave"])
        dust_out[:dust_features][df][:fwhm] = Param.from_dict_fwhm(dust["dust_features"][df]["fwhm"])
    end
    dust_out[:extinction] = Param.ParamDict()
    dust_out[:extinction][:tau_9_7] = Param.from_dict(dust["extinction"]["tau_9_7"])
    dust_out[:extinction][:beta] = Param.from_dict(dust["extinction"]["beta"])

    return dust_out
end

function parse_lines(channel::Int)
    lines = TOML.parsefile(joinpath(sourcepath, "lines.toml"))
    lines_out = Param.LineDict()

    keylist1 = ["tie_H2_voff", "tie_IP_voff", "tie_H2_flow_voff", "tie_IP_flow_voff", 
        "voff_plim", "fwhm_pmax", "h3_plim", "h4_plim", "R", 
        "flexible_wavesol", "wavesol_unc", "channels", "lines", "profiles", "flows"]
    for key ∈ keylist1
        if !haskey(lines, key)
            error("$key not found in line options!")
        end
    end
    if !haskey(lines["profiles"], "default")
        error("default not found in line profile options!")
    end
    profiles = Dict(ln => lines["profiles"]["default"] for ln ∈ keys(lines["lines"]))
    if haskey(lines, "profiles")
        for line ∈ keys(lines["lines"])
            if haskey(lines["profiles"], line)
                profiles[line] = lines["profiles"][line]
            end
        end
    end
    flow_profiles = Dict{String, Union{String,Nothing}}(ln => nothing for ln ∈ keys(lines["lines"]))
    for line ∈ keys(lines["lines"])
        if haskey(lines, "flows")
            if haskey(lines["flows"], line)
                flow_profiles[line] = lines["flows"][line]
            end
        end
    end

    fwhm_pmin = Util.C_KMS / lines["R"][channel]

    # Initial values
    fwhm_init = "fwhm_init" ∈ keys(lines) ? lines["fwhm_init"] : fwhm_pmin + 1
    voff_init = "voff_init" ∈ keys(lines) ? lines["voff_init"] : 0.0
    h3_init = "h3_init" ∈ keys(lines) ? lines["h3_init"] : 0.0
    h4_init = "h4_init" ∈ keys(lines) ? lines["h4_init"] : 0.0
    η_init = "η_init" ∈ keys(lines) ? lines["eta_init"] : 1.0  # start fully gaussian

    for line ∈ keys(lines["lines"])

        voff_prior = Uniform(lines["voff_plim"]...)
        voff_locked = false
        fwhm_prior = Uniform(fwhm_pmin, profiles[line] == "GaussHermite" ? lines["fwhm_pmax"] * 2 : lines["fwhm_pmax"])
        fwhm_locked = false
        if profiles[line] == "GaussHermite"
            h3_prior = truncated(Normal(0.0, 0.1), lines["h3_plim"]...)
            h3_locked = false
            h4_prior = truncated(Normal(0.0, 0.1), lines["h4_plim"]...)
            h4_locked = false
        elseif profiles[line] == "Voigt"
            η_prior = Uniform(0.0, 1.0)
            η_locked = false
        end

        flow_voff_prior = Uniform(lines["flow_voff_plim"]...)
        flow_voff_locked = false
        flow_fwhm_prior = Uniform(fwhm_pmin, lines["flow_fwhm_pmax"])
        flow_fwhm_locked = false
        if flow_profiles[line] == "GaussHermite"
            flow_h3_prior = truncated(Normal(0.0, 0.1), lines["h3_plim"]...)
            flow_h3_locked = false
            flow_h4_prior = truncated(Normal(0.0, 0.1), lines["h4_plim"]...)
            flow_h4_locked = false
        elseif flow_profiles[line] == "Voigt"
            flow_η_prior = Uniform(0.0, 1.0)
            flow_η_locked = false
        end

        if haskey(lines, "priors")
            if haskey(lines["priors"], line)
                if haskey(lines["priors"][line], "voff")
                    voff_prior = eval(Meta.parse(lines["priors"][line]["voff"]["pstr"]))
                    voff_locked = lines["priors"][line]["voff"]["locked"]
                end
                if haskey(lines["priors"][line], "fwhm")
                    fwhm_prior = eval(Meta.parse(lines["priors"][line]["fwhm"]["pstr"]))
                    fwhm_locked = lines["priors"][line]["fwhm"]["locked"]
                end
                if haskey(lines["priors"][line], "h3")
                    h3_prior = eval(Meta.parse(lines["priors"][line]["h3"]["pstr"]))
                    h3_locked = lines["priors"][line]["h3"]["locked"]
                end
                if haskey(lines["priors"][line], "h4")
                    h4_prior = eval(Meta.parse(lines["priors"][line]["h4"]["pstr"]))
                    h4_locked = lines["priors"][line]["h4"]["locked"]
                end
                if haskey(lines["priors"][line], "eta")
                    η_prior = eval(Meta.parse(lines["priors"][line]["eta"]["pstr"]))
                    η_locked = lines["priors"][line]["eta"]["locked"]
                end

                if haskey(lines["priors"][line], "flow_voff")
                    flow_voff_prior = eval(Meta.parse(lines["priors"][line]["flow_voff"]["pstr"]))
                    flow_voff_locked = lines["priors"][line]["flow_voff"]["locked"]
                end
                if haskey(lines["priors"][line], "flow_fwhm")
                    flow_fwhm_prior = eval(Meta.parse(lines["priors"][line]["flow_fwhm"]["pstr"]))
                    flow_fwhm_locked = lines["priors"][line]["flow_fwhm"]["locked"]
                end
                if haskey(lines["priors"][line], "flow_h3")
                    flow_h3_prior = eval(Meta.parse(lines["priors"][line]["flow_h3"]["pstr"]))
                    flow_h3_locked = lines["priors"][line]["flow_h3"]["locked"]
                end
                if haskey(lines["priors"][line], "flow_h4")
                    flow_h4_prior = eval(Meta.parse(lines["priors"][line]["flow_h4"]["pstr"]))
                    flow_h4_locked = lines["priors"][line]["flow_h4"]["locked"]
                end
                if haskey(lines["priors"][line], "flow_eta")
                    flow_η_prior = eval(Meta.parse(lines["priors"][line]["flow_eta"]["pstr"]))
                    flow_η_locked = lines["priors"][line]["flow_eta"]["locked"]
                end
            end
        end

        tied = nothing
        if lines["tie_H2_voff"] && occursin("H2", line)
            tied = "H2"
            if lines["flexible_wavesol"]
                δv = lines["wavesol_unc"][channel]
                voff_prior = Uniform(-δv, δv)
            end
        end
        flow_tied = nothing
        if lines["tie_H2_flow_voff"] && occursin("H2", line)
            flow_tied = "H2"
        end

        voff = Param.Parameter(voff_init, voff_locked, voff_prior)
        fwhm = Param.Parameter(fwhm_init, fwhm_locked, fwhm_prior)
        if profiles[line] == "Gaussian"
            params = Param.ParamDict(:voff => voff, :fwhm => fwhm)
        elseif profiles[line] == "GaussHermite"
            h3 = Param.Parameter(h3_init, h3_locked, h3_prior)
            h4 = Param.Parameter(h4_init, h4_locked, h4_prior)
            params = Param.ParamDict(:voff => voff, :fwhm => fwhm, :h3 => h3, :h4 => h4)
        elseif profiles[line] == "Voigt"
            η = Param.Parameter(η_init, η_locked, η_prior)
            params = Param.ParamDict(:voff => voff, :fwhm => fwhm, :mixing => η)
        end
        if !isnothing(flow_profiles[line])
            flow_voff = Param.Parameter(voff_init, flow_voff_locked, flow_voff_prior)
            flow_fwhm = Param.Parameter(fwhm_init, flow_fwhm_locked, flow_fwhm_prior)
            if flow_profiles[line] == "Gaussian"
                flow_params = Param.ParamDict(:flow_voff => flow_voff, :flow_fwhm => flow_fwhm)
            elseif flow_profiles[line] == "GaussHermite"
                flow_h3 = Param.Parameter(h3_init, flow_h3_locked, flow_h3_prior)
                flow_h4 = Param.Parameter(h4_init, flow_h4_locked, flow_h4_prior)
                flow_params = Param.ParamDict(:flow_voff => flow_voff, :flow_fwhm => flow_fwhm, :flow_h3 => flow_h3, :flow_h4 => flow_h4)
            elseif flow_profiles[line] == "Voigt"
                flow_η = Param.Parameter(η_init, flow_η_locked, flow_η_prior)
                flow_params = Param.ParamDict(:flow_voff => flow_voff, :flow_fwhm => flow_fwhm, :flow_mixing => flow_η)
            end
            params = merge(params, flow_params)
        end
        lines_out[Symbol(line)] = Param.TransitionLine(lines["lines"][line], 
            Symbol(profiles[line]), !isnothing(flow_profiles[line]) ? Symbol(flow_profiles[line]) : nothing, params, tied, flow_tied)

    end

    voff_tied_key = unique([lines_out[line].tied for line ∈ keys(lines_out)])
    voff_tied_key = voff_tied_key[.!isnothing.(voff_tied_key)]
    voff_tied = Dict{String, Param.Parameter}()
    for voff_tie ∈ voff_tied_key
        prior = Uniform(lines["voff_plim"]...)
        locked = false
        if haskey(lines, "priors")
            if haskey(lines["priors"], voff_tie)
                prior = lines["priors"][voff_tie]["pstr"]
                locked = lines["priors"][voff_tie]["locked"]
            end
        end
        voff_tied[voff_tie] = Param.Parameter(voff_init, locked, prior)
    end

    flow_voff_tied_key = unique([lines_out[line].flow_tied for line ∈ keys(lines_out)])
    flow_voff_tied_key = flow_voff_tied_key[.!isnothing.(flow_voff_tied_key)]
    flow_voff_tied = Dict{String, Param.Parameter}()
    for flow_voff_tie ∈ flow_voff_tied_key
        flow_prior = Uniform(lines["flow_voff_plim"]...)
        flow_locked = false
        if haskey(lines, "priors")
            if haskey(lines["priors"], flow_voff_tie)
                flow_prior = lines["priors"][flow_voff_tie]["pstr"]
                flow_locked = lines["priors"][flow_voff_tie]["locked"]
            end
        end
        flow_voff_tied[flow_voff_tie] = Param.Parameter(voff_init, flow_locked, flow_prior)
    end

    return lines_out, voff_tied, flow_voff_tied, lines["R"], lines["flexible_wavesol"]
end

struct ParamMaps
    """
    A structure for holding 2D maps of fitting parameters generated when fitting a cube.
    """

    stellar_continuum::Dict{Symbol, Array{Float64, 2}}
    dust_continuum::Dict{Int, Dict{Symbol, Array{Float64, 2}}}
    dust_features::Dict{String, Dict{Symbol, Array{Float64, 2}}}
    lines::Dict{Symbol, Dict{Symbol, Array{Float64, 2}}}
    tied_voffs::Dict{String, Array{Float64, 2}}
    flow_tied_voffs::Dict{String, Array{Float64, 2}}
    extinction::Dict{Symbol, Array{Float64, 2}}
    dust_complexes::Dict{String, Dict{Symbol, Array{Float64, 2}}}
    reduced_χ2::Array{Float64, 2}

end

function parammaps_empty(shape::Tuple{Int,Int,Int}, n_dust_cont::Int, df_names::Vector{String}, 
    complexes::Vector{String}, line_names::Vector{Symbol}, line_tied::Vector{Union{String,Nothing}},
    line_profiles::Vector{Symbol}, line_flow_tied::Vector{Union{String,Nothing}}, line_flow_profiles::Vector{Union{Symbol,Nothing}},
    voff_tied_key::Vector{String}, flow_voff_tied_key::Vector{String}, flexible_wavesol::Bool)

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
    for (line, tie, prof, flowtie, flowprof) ∈ zip(line_names, line_tied, line_profiles, line_flow_tied, line_flow_profiles)
        lines[line] = Dict{Symbol, Array{Float64, 2}}()
        pnames = isnothing(tie) || flexible_wavesol ? [:amp, :voff, :fwhm] : [:amp, :fwhm]
        if prof == :GaussHermite
            pnames = [pnames; :h3; :h4]
        elseif prof == :Voigt
            pnames = [pnames; :mixing]
        end
        if !isnothing(flowprof)
            pnames = isnothing(flowtie) ? [pnames; :flow_amp; :flow_voff; :flow_fwhm] : [pnames; :flow_amp; :flow_fwhm]
            if flowprof == :GaussHermite
                pnames = [pnames; :flow_h3; :flow_h4]
            elseif flowprof == :Voigt
                pnames = [pnames; :flow_mixing]
            end
        end
        pnames = [pnames; :intI; :SNR]
        for pname ∈ pnames
            lines[line][pname] = copy(nan_arr)
        end
    end

    # Tied voff parameters
    tied_voffs = Dict{String, Array{Float64, 2}}()
    for vk ∈ voff_tied_key
        tied_voffs[vk] = copy(nan_arr)
    end

    flow_tied_voffs = Dict{String, Array{Float64, 2}}()
    for fvk ∈ flow_voff_tied_key
        flow_tied_voffs[fvk] = copy(nan_arr)
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

    reduced_χ2 = copy(nan_arr)

    return ParamMaps(stellar_continuum, dust_continuum, dust_features, lines, tied_voffs, flow_tied_voffs,
        extinction, dust_complexes, reduced_χ2)
end


struct CubeModel
    """
    A structure for holding 3D models of intensity, split up into model components, generated when fitting a cube
    """

    model::Array{Float32, 3}
    stellar::Array{Float32, 3}
    dust_continuum::Array{Float32, 4}
    dust_features::Array{Float32, 4}
    extinction::Array{Float32, 3}
    lines::Array{Float32, 4}

end

function cubemodel_empty(shape::Tuple{Int,Int,Int}, n_dust_cont::Int, df_names::Vector{String}, line_names::Vector{Symbol})

    model = zeros(Float32, shape...)
    stellar = zeros(Float32, shape...)
    dust_continuum = zeros(Float32, shape..., n_dust_cont)
    dust_features = zeros(Float32, shape..., length(df_names))
    extinction = zeros(Float32, shape...)
    lines = zeros(Float32, shape..., length(line_names))

    return CubeModel(model, stellar, dust_continuum, dust_features, extinction, lines)
end


mutable struct CubeFitter
    
    cube::CubeData.DataCube
    z::Float64
    name::String
    cube_model::CubeModel
    param_maps::ParamMaps
    window_size::Float64
    plot_spaxels::Symbol
    plot_maps::Bool
    parallel::Bool
    save_fits::Bool

    T_s::Param.Parameter
    T_dc::Vector{Param.Parameter}
    τ_97::Param.Parameter
    β::Param.Parameter

    n_dust_cont::Int

    n_dust_feat::Int
    df_names::Vector{String}
    dust_features::Vector{Dict}

    n_lines::Int
    line_names::Vector{Symbol}
    line_profiles::Vector{Symbol}
    line_flow_profiles::Vector{Union{Nothing,Symbol}}
    lines::Vector{Param.TransitionLine}

    n_voff_tied::Int
    line_tied::Vector{Union{String,Nothing}}
    voff_tied_key::Vector{String}
    voff_tied::Vector{Param.Parameter}

    n_flow_voff_tied::Int
    line_flow_tied::Vector{Union{String,Nothing}}
    flow_voff_tied_key::Vector{String}
    flow_voff_tied::Vector{Param.Parameter}

    n_complexes::Int
    complexes::Vector{String}
    n_params_cont::Int
    n_params_lines::Int
    
    cosmology::Cosmology.AbstractCosmology
    χ²_thresh::Float64
    R::Int64
    flexible_wavesol::Bool

    p_best_cont::Union{Nothing,Vector{Float64}}
    p_best_line::Union{Nothing,Vector{Float64}}
    χ²_best::Union{Nothing,Float64}
    best_spaxel::Union{Nothing,Tuple{Int,Int}}

    function CubeFitter(cube::CubeData.DataCube, z::Float64, name::String; window_size::Float64=.025, 
        plot_spaxels::Symbol=:pyplot, plot_maps::Bool=true, parallel::Bool=true, save_fits::Bool=true)

        dust = parse_dust() 
        options = parse_options()
        line_list, voff_tied, flow_voff_tied, R, flexible_wavesol = parse_lines(parse(Int, cube.channel))

        # Get shape
        shape = size(cube.Iλ)
        # Alias
        λ = cube.λ

        T_s = dust[:stellar_continuum_temp]
        T_dc = dust[:dust_continuum_temps]
        τ_97 = dust[:extinction][:tau_9_7]
        β = dust[:extinction][:beta]

        #### PREPARE OUTPUTS ####
        n_dust_cont = length(T_dc)

        df_filt = [(minimum(λ)-0.5 < dust[:dust_features][df][:wave].value < maximum(λ)+0.5) for df ∈ keys(dust[:dust_features])]
        df_names = Vector{String}(collect(keys(dust[:dust_features]))[df_filt])
        df_mean = [parse(Float64, split(df, "_")[2]) for df ∈ df_names]
        ss = sortperm(df_mean)
        df_names = df_names[ss]
        dust_features = [dust[:dust_features][df] for df ∈ df_names]
        n_dust_features = length(df_names)

        complexes = Vector{String}(unique([dust[:dust_features][n][:complex] for n ∈ df_names]))
        ss = sortperm([parse(Float64, c) for c ∈ complexes])
        complexes = complexes[ss]
        n_complexes = length(complexes)

        line_wave = [line_list[line].λ₀ for line ∈ keys(line_list)]
        ln_filt = [minimum(λ) < lw < maximum(λ) for lw ∈ line_wave]
        line_names = Vector{Symbol}(collect(keys(line_list))[ln_filt])
        ss = sortperm(line_wave[ln_filt])
        line_names = line_names[ss]
        lines = [line_list[line] for line ∈ line_names]
        line_profiles = [line_list[line].profile for line ∈ line_names]
        line_flow_profiles = Vector{Union{Symbol,Nothing}}([line_list[line].flow_profile for line ∈ line_names])
        n_lines = length(line_names)

        # Unpack the voff_tied dictionary
        voff_tied_key = collect(keys(voff_tied))
        voff_tied = [voff_tied[voff] for voff ∈ voff_tied_key]
        n_voff_tied = length(voff_tied)
        # Also store the "tied" parameter for each line, which will need to be checked against the voff_tied_key
        # during fitting to find the proper location of the tied voff parameter to use
        line_tied = Vector{Union{Nothing,String}}([line.tied for line ∈ lines])

        # Repeat for in/outflow velocity offsets
        flow_voff_tied_key = collect(keys(flow_voff_tied))
        flow_voff_tied = [flow_voff_tied[flow_voff] for flow_voff ∈ flow_voff_tied_key]
        n_flow_voff_tied = length(flow_voff_tied)
        line_flow_tied = Vector{Union{Nothing,String}}([line.flow_tied for line ∈ lines])

        # Total number of parameters
        n_params_cont = (2+2) + 2n_dust_cont + 3n_dust_features + 2n_complexes
        n_params_lines = n_voff_tied + n_flow_voff_tied
        for i ∈ 1:n_lines
            if isnothing(line_tied[i]) || flexible_wavesol
                n_params_lines += 3
            else
                n_params_lines += 2
            end
            if line_profiles[i] == :GaussHermite
                n_params_lines += 2
            elseif line_profiles[i] == :Voigt
                n_params_lines += 1
            end
            if !isnothing(line_flow_profiles[i])
                if isnothing(line_flow_tied[i])
                    n_params_lines += 3
                else
                    n_params_lines += 2
                end
                if line_flow_profiles[i] == :GaussHermite
                    n_params_lines += 2
                elseif line_flow_profiles[i] == :Voigt
                    n_params_lines += 1
                end
            end
            n_params_lines += 2
        end 

        # Full 3D intensity model array
        cube_model = cubemodel_empty(shape, n_dust_cont, df_names, line_names)
        # 2D maps of fitting parameters
        param_maps = parammaps_empty(shape, n_dust_cont, df_names, complexes, line_names, line_tied,
            line_profiles, line_flow_tied, line_flow_profiles, voff_tied_key, flow_voff_tied_key, flexible_wavesol)

        # Prepare output directories
        name = replace(name, " " => "_")
        if !isdir("output_$name")
            mkdir("output_$name")
        end

        # Prepare options
        χ²_thresh = options[:chi2_threshold]
        cosmo = options[:cosmology]

        return new(cube, z, name, cube_model, param_maps, window_size, plot_spaxels, plot_maps, parallel, save_fits,
            T_s, T_dc, τ_97, β, n_dust_cont, n_dust_features, df_names, dust_features, n_lines, line_names, line_profiles, 
            line_flow_profiles, lines, n_voff_tied, line_tied, voff_tied_key, voff_tied, n_flow_voff_tied, line_flow_tied,
            flow_voff_tied_key, flow_voff_tied, n_complexes, complexes, n_params_cont, n_params_lines, 
            cosmo, χ²_thresh, R[parse(Int, cube.channel)], flexible_wavesol, nothing, nothing, nothing, nothing)
    end

end

function mask_emission_lines(λ::Vector{Float64}, I::Vector{Float32}, σ::Vector{Float32})

    # Series of window sizes to perform median filtering
    window_sizes = [2, 5, 10, 50, 100, 250, 500]
    med_spec = zeros(length(λ), length(window_sizes))
    mask = falses(length(λ))

    # cubic spline interpolation 
    Δλ = diff(λ)[1]
    λknots = λ[51]:Δλ*50:λ[end-51]
    I_cub = Spline1D(λ, I, λknots, k=3, bc="extrapolate")

    # For each window size, do a sliding median filter
    for i ∈ 1:length(window_sizes)
        pix = 1:length(λ)
        for p ∈ pix
            i_sort = sortperm(abs.(p .- pix))
            idx = pix[i_sort][1:window_sizes[i]]
            med_spec[p, i] = nanmedian(I[idx] .- I_cub.(λ[idx]))
        end
    end
    # Check if the std between the window medians is larger than the noise -> if so, there is a line
    for j ∈ 1:length(λ)
        mask[j] = dropdims(std(med_spec, dims=2), dims=2)[j] > std(I[λ[j]-0.1 .< λ .< λ[j]+0.1] .- I_cub.(λ[λ[j]-0.1 .< λ .< λ[j]+0.1]))
    end
    # Extend mask edges by a few pixels
    mask_edges = findall(x -> x == 1, diff(mask))
    for me ∈ mask_edges
        mask[maximum([1, me-5]):minimum([length(mask), me+5])] .= 1
    end

    return mask
end

function continuum_cubic_spline(λ::Vector{Float64}, I::Vector{Float32}, σ::Vector{Float32})

    # Copy arrays
    I_out = copy(I)
    σ_out = copy(σ)

    # Mask out emission lines so that they aren't included in the continuum fit
    mask_lines = mask_emission_lines(λ, I, σ)
    I_out[mask_lines] .= NaN
    σ_out[mask_lines] .= NaN 

    # Interpolate the NaNs
    if sum(mask_lines) > 0
        # Make sure the wavelength vector is linear, since it is assumed later in the function
        diffs = diff(λ)
        @assert diffs[1] ≈ diffs[end]
        Δλ = diffs[1]

        # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
        λknots = λ[51]:Δλ*50:λ[end-51]

        # Do a full cubic spline remapping of the data
        I_out = Spline1D(λ[isfinite.(I_out)], I_out[isfinite.(I_out)], λknots, k=3, bc="extrapolate").(λ)
        σ_out = Spline1D(λ[isfinite.(σ_out)], σ_out[isfinite.(σ_out)], λknots, k=3, bc="extrapolate").(λ)
    end  

    return mask_lines, I_out, σ_out
end

function continuum_fit_spaxel(cube_fitter::CubeFitter, spaxel::Tuple{Int, Int}; verbose::Bool=false)

    # Extract spaxel to be fit
    λ = cube_fitter.cube.λ
    I = cube_fitter.cube.Iλ[spaxel..., :]
    σ = cube_fitter.cube.σI[spaxel..., :]

    # Mask out emission lines so that they aren't included in the continuum fit
    mask_lines, I_cubic, σ_cubic = continuum_cubic_spline(λ, I, σ)
    # Fill in the data where the lines are with the cubic spline interpolation
    I[mask_lines] .= I_cubic[mask_lines]
    σ[mask_lines] .= σ_cubic[mask_lines]
    # Add statistical uncertainties to the systematic uncertainties in quadrature
    σ_stat = std(I .- I_cubic)
    σ .= .√(σ.^2 .+ σ_stat.^2)
 
    mean_df = [cdf[:wave] for cdf ∈ cube_fitter.dust_features]
    fwhm_df = [cdf[:fwhm] for cdf ∈ cube_fitter.dust_features]

    # Check if the cube fitter has best fit parameters from a previous fit
    if !isnothing(cube_fitter.p_best_cont)

        # Set the parameters to the best parameters
        p₀ = cube_fitter.p_best_cont

        # Scale all of the amplitudes by the ratio of the median of the datas in the two spaxels
        scale = nanmedian(cube_fitter.cube.Iλ[spaxel..., :]) / 
            nanmedian(cube_fitter.cube.Iλ[cube_fitter.best_spaxel..., :])
        
        # Stellar amplitude
        p₀[1] *= scale
        pᵢ = 3

        # Dust continuum amplitudes
        for i ∈ 1:cube_fitter.n_dust_cont
            p₀[pᵢ] *= scale
            pᵢ += 2
        end

        # Dust feature amplitudes
        for i ∈ 1:cube_fitter.n_dust_feat
            p₀[pᵢ] *= scale
            pᵢ += 3
        end

    # Otherwise, we estimate the initial parameters based on the data
    else

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
        A_s = clamp(interp_func(5.5) / Util.Blackbody_ν(5.5, cube_fitter.T_s.value), 0., Inf)

        # Dust feature amplitudes
        A_df = repeat([clamp(nanmedian(I)/2, 0., Inf)], cube_fitter.n_dust_feat)

        # Dust continuum amplitudes
        λ_dc = clamp.(2898 ./ [Ti.value for Ti ∈ cube_fitter.T_dc], minimum(λ), maximum(λ))
        A_dc = clamp.(interp_func.(λ_dc) ./ [Util.Blackbody_ν(λ_dci, T_dci.value) for (λ_dci, T_dci) ∈ zip(λ_dc, cube_fitter.T_dc)] .* 
            (λ_dc ./ 9.7).^2 ./ 5., 0., Inf)
            
        # amp_dc_prior = amp_agn_prior = Uniform(0., 1e100)  # just set it arbitrarily large, otherwise infinity gives bad logpdfs
        # amp_df_prior = Uniform(0., maximum(I) > 0. ? maximum(I) : 1e100)

        stellar_pars = [A_s, cube_fitter.T_s.value]
        # stellar_priors = [amp_dc_prior, cube_fitter.T_s.prior]

        dc_pars = vcat([[Ai, Ti.value] for (Ai, Ti) ∈ zip(A_dc, cube_fitter.T_dc)]...)
        # dc_priors = vcat([[amp_dc_prior, Ti.prior] for Ti ∈ cube_fitter.T_dc]...)

        df_pars = vcat([[Ai, mi.value, fi.value] for (Ai, mi, fi) ∈ zip(A_df, mean_df, fwhm_df)]...)
        # df_priors = vcat([[amp_df_prior, mi.prior, fi.prior] for (mi, fi) ∈ zip(mean_df, fwhm_df)]...)

        # Initial parameter vector
        p₀ = vcat(stellar_pars, dc_pars, df_pars, [cube_fitter.τ_97.value, cube_fitter.β.value])
        # priors = vcat(stellar_priors, dc_priors, df_priors, [cube_fitter.τ_97.prior, cube_fitter.β.prior])

    end

    # Convert parameter limits into CMPFit object
    parinfo = CMPFit.Parinfo(length(p₀))

    # Stellar amplitude
    parinfo[1].limited = (1,0)
    parinfo[1].limits = (0., 0.)

    # Stellar temp
    parinfo[2].fixed = cube_fitter.T_s.locked
    if !(cube_fitter.T_s.locked)
        parinfo[2].limited = (1,1)
        parinfo[2].limits = (minimum(cube_fitter.T_s.prior), maximum(cube_fitter.T_s.prior))
    end

    # Dust cont amplitudes and temps
    pᵢ = 3
    for i ∈ 1:cube_fitter.n_dust_cont
        parinfo[pᵢ].limited = (1,0)
        parinfo[pᵢ].limits = (0., 0.)
        parinfo[pᵢ+1].fixed = cube_fitter.T_dc[i].locked
        if !(cube_fitter.T_dc[i].locked)
            parinfo[pᵢ+1].limited = (1,1)
            parinfo[pᵢ+1].limits = (minimum(cube_fitter.T_dc[i].prior), maximum(cube_fitter.T_dc[i].prior))
        end
        pᵢ += 2
    end

    # Dust feature amplitude, mean, fwhm
    for i ∈ 1:cube_fitter.n_dust_feat
        parinfo[pᵢ].limited = (1,1)
        parinfo[pᵢ].limits = (0., nanmaximum(I))
        parinfo[pᵢ+1].fixed = mean_df[i].locked
        if !(mean_df[i].locked)
            parinfo[pᵢ+1].limited = (1,1)
            parinfo[pᵢ+1].limits = (minimum(mean_df[i].prior), maximum(mean_df[i].prior))
        end
        parinfo[pᵢ+2].fixed = fwhm_df[i].locked
        if !(fwhm_df[i].locked)
            parinfo[pᵢ+2].limited = (1,1)
            parinfo[pᵢ+2].limits = (minimum(fwhm_df[i].prior), maximum(fwhm_df[i].prior))
        end
        pᵢ += 3
    end

    # Extinction
    parinfo[pᵢ].fixed = cube_fitter.τ_97.locked
    if !(cube_fitter.τ_97.locked)
        parinfo[pᵢ].limited = (1,1)
        parinfo[pᵢ].limits = (minimum(cube_fitter.τ_97.prior), maximum(cube_fitter.τ_97.prior))
    end
    parinfo[pᵢ+1].fixed = cube_fitter.β.locked
    if !(cube_fitter.β.locked)
        parinfo[pᵢ+1].limited = (1,1)
        parinfo[pᵢ+1].limits = (minimum(cube_fitter.β.prior), maximum(cube_fitter.β.prior))
    end
    pᵢ += 2

    # Create a `config` structure
    config = CMPFit.Config()

    res = cmpfit(λ, I, σ, (x, p) -> Util.fit_spectrum(x, p, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat), 
        p₀, parinfo=parinfo, config=config)

    # Get best fit results
    popt = res.param
    # Count free parameters
    n_free = 0
    for pᵢ ∈ 1:length(popt)
        if iszero(parinfo[pᵢ].fixed)
            n_free += 1
        end
    end

    # function ln_prior(p)
    #     logpdfs = [logpdf(priors[i], p[i]) for i ∈ 1:length(p)]
    #     return sum(logpdfs)
    # end

    # function nln_probability(p)
    #     model = Util.fit_spectrum(λ, p, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat)
    #     return -Util.ln_likelihood(I, model, σ) - ln_prior(p)
    # end

    # res = optimize(nln_probability, minimum.(priors), maximum.(priors), p₀, NelderMead())
    # popt = res.minimizer
    # χ2 = χ2red = -res.minimum

    # Final optimized fit
    I_model, comps = Util.fit_spectrum(λ, popt, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, return_components=true)

    if verbose
        println("######################################################################")
        println("################# SPAXEL FIT RESULTS -- CONTINUUM ####################")
        println("######################################################################")
        println()
        println()
        println("#> STELLAR CONTINUUM <#")
        println()
        println("Stellar_amp: \t\t\t $(@sprintf "%.3e" popt[1]) MJy/sr \t Limits: (0, Inf)")
        println("Stellar_temp: \t\t\t $(@sprintf "%.0f" popt[2]) K \t (fixed)")
        pᵢ = 3
        println()
        println("#> DUST CONTINUUM <#")
        println()
        for i ∈ 1:cube_fitter.n_dust_cont
            println("Dust_continuum_$(i)_amp: \t\t $(@sprintf "%.3e" popt[pᵢ]) MJy/sr \t Limits: (0, Inf)")
            println("Dust_continuum_$(i)_temp: \t\t $(@sprintf "%.0f" popt[pᵢ+1]) K \t\t\t (fixed)")
            println()
            pᵢ += 2
        end
        println()
        println("#> DUST FEATURES <#")
        println()
        for (j, df) ∈ enumerate(cube_fitter.df_names)
            println("$(df)_amp:\t\t\t $(@sprintf "%.1f" popt[pᵢ]) MJy/sr \t Limits: " *
                "(0, $(@sprintf "%.1f" nanmaximum(I)))")
            println("$(df)_mean:  \t\t $(@sprintf "%.3f" popt[pᵢ+1]) μm \t Limits: " *
                "($(@sprintf "%.3f" minimum(mean_df[j].prior)), $(@sprintf "%.3f" maximum(mean_df[j].prior)))" * 
                (mean_df[j].locked ? " (fixed)" : ""))
            println("$(df)_fwhm:  \t\t $(@sprintf "%.3f" popt[pᵢ+2]) μm \t Limits: " *
                "($(@sprintf "%.3f" minimum(fwhm_df[j].prior)), $(@sprintf "%.3f" maximum(fwhm_df[j].prior)))" * 
                (fwhm_df[j].locked ? " (fixed)" : ""))
            println()
            pᵢ += 3
        end
        println()
        println("#> EXTINCTION <#")
        println()
        println("τ_9.7: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) [-] \t Limits: " *
            "($(@sprintf "%.2f" minimum(cube_fitter.τ_97.prior)), $(@sprintf "%.2f" maximum(cube_fitter.τ_97.prior)))" * 
            (cube_fitter.τ_97.locked ? " (fixed)" : ""))
        println("β: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+1]) [-] \t Limits: " *
            "($(@sprintf "%.2f" minimum(cube_fitter.β.prior)), $(@sprintf "%.2f" maximum(cube_fitter.β.prior)))" * 
            (cube_fitter.β.locked ? " (fixed)" : ""))
        println()
        println("######################################################################")
    end

    return σ, popt, I_model, comps, n_free

end

function line_fit_spaxel(cube_fitter::CubeFitter, spaxel::Tuple{Int, Int}, continuum::Vector{Float64}; verbose::Bool=false)

    # Extract spaxel to be fit
    λ = cube_fitter.cube.λ
    I = cube_fitter.cube.Iλ[spaxel..., :]
    σ = cube_fitter.cube.σI[spaxel..., :]

    _, continuum, _ = continuum_cubic_spline(λ, I, σ)
    N = Float64(abs(nanmaximum(I)))
    N = N ≠ 0. ? N : 1.

    # Add statistical uncertainties to the systematic uncertainties in quadrature
    σ_stat = std(I .- continuum)
    σ .= .√(σ.^2 .+ σ_stat.^2)

    # Normalized flux and uncertainty
    Inorm = (I .- continuum) ./ N
    σnorm = σ ./ N

    voff_ln = [ln.parameters[:voff] for ln ∈ cube_fitter.lines]
    fwhm_ln = [ln.parameters[:fwhm] for ln ∈ cube_fitter.lines]
    h3_ln = [ln.profile == :GaussHermite ? ln.parameters[:h3] : nothing for ln ∈ cube_fitter.lines]
    h4_ln = [ln.profile == :GaussHermite ? ln.parameters[:h4] : nothing for ln ∈ cube_fitter.lines]
    η_ln = [ln.profile == :Voigt ? ln.parameters[:mixing] : nothing for ln ∈ cube_fitter.lines]

    flow_voff_ln = [isnothing(ln.flow_profile) ? nothing : ln.parameters[:flow_voff] for ln ∈ cube_fitter.lines]
    flow_fwhm_ln = [isnothing(ln.flow_profile) ? nothing : ln.parameters[:flow_fwhm] for ln ∈ cube_fitter.lines]
    flow_h3_ln = [ln.flow_profile == :GaussHermite ? ln.parameters[:flow_h3] : nothing for ln ∈ cube_fitter.lines]
    flow_h4_ln = [ln.flow_profile == :GaussHermite ? ln.parameters[:flow_h4] : nothing for ln ∈ cube_fitter.lines]
    flow_η_ln = [ln.flow_profile == :Voigt ? ln.parameters[:flow_mixing] : nothing for ln ∈ cube_fitter.lines]

    # Set up the prior vector
    amp_ln_prior = Uniform(0., 1.)
    amp_flow_prior = Uniform(0., 1.)
    λ0_ln = Vector{Float64}()
    prof_ln = Vector{Symbol}()
    flow_prof_ln = Vector{Union{Symbol,Nothing}}()
    ln_priors = Vector{Any}()
    for (i, ln) ∈ enumerate(cube_fitter.lines)
        if isnothing(ln.tied) || cube_fitter.flexible_wavesol
            append!(ln_priors, [amp_ln_prior, voff_ln[i].prior, fwhm_ln[i].prior])
        else
            append!(ln_priors, [amp_ln_prior, fwhm_ln[i].prior])
        end
        if ln.profile == :GaussHermite
            append!(ln_priors, [h3_ln[i].prior, h4_ln[i].prior])
        elseif ln.profile == :Voigt
            append!(ln_priors, [η_ln[i].prior])
        end
        if !isnothing(ln.flow_profile)
            if isnothing(ln.flow_tied)
                append!(ln_priors, [amp_flow_prior, flow_voff_ln[i].prior, flow_fwhm_ln[i].prior])
            else
                append!(ln_priors, [amp_flow_prior, flow_fwhm_ln[i].prior])
            end
            if ln.flow_profile == :GaussHermite
                append!(ln_priors, [flow_h3_ln[i].prior, flow_h4_ln[i].prior])
            elseif ln.flow_profile == :Voigt
                append!(ln_prior, [flow_η_ln[i].prior])
            end
        end
        append!(λ0_ln, [ln.λ₀])
        append!(prof_ln, [ln.profile])
        append!(flow_prof_ln, [ln.flow_profile])
    end
    voff_tied_priors = [cube_fitter.voff_tied[i].prior for i ∈ 1:cube_fitter.n_voff_tied]
    flow_voff_tied_priors = [cube_fitter.flow_voff_tied[i].prior for i ∈ 1:cube_fitter.n_flow_voff_tied]

    # Initial prior vector
    priors = vcat(voff_tied_priors, flow_voff_tied_priors, ln_priors)

    # Check if there are previous best fit parameters
    if !isnothing(cube_fitter.p_best_line)

        # If so, set the parameters to the best fit ones
        p₀ = cube_fitter.p_best_line
        # Make sure all amplitudes are nonzero
        pᵢ = 1 + cube_fitter.n_voff_tied + cube_fitter.n_flow_voff_tied
        for i ∈ 1:cube_fitter.n_lines
            p₀[pᵢ] = 0.5
            if isnothing(cube_fitter.line_tied[i]) || cube_fitter.flexible_wavesol
                pᵢ += 3
            else
                pᵢ += 2
            end
            if prof_ln[i] == :GaussHermite
                pᵢ += 2
            elseif prof_ln[i] == :Voigt
                pᵢ += 1
            end
            if !isnothing(flow_prof_ln[i])
                p₀[pᵢ] = 0.25
                if isnothing(cube_fitter.line_flow_tied[i])
                    pᵢ += 3
                else
                    pᵢ += 2
                end
                if flow_prof_ln[i] == :GaussHermite
                    pᵢ += 2
                elseif flow_prof_ln[i] == :Voigt
                    pᵢ += 1
                end
            end
        end

    else

        A_ln = ones(cube_fitter.n_lines) .* 0.5
        A_fl = ones(cube_fitter.n_lines) .* 0.25

        ln_pars = Vector{Float64}()
        for (i, ln) ∈ enumerate(cube_fitter.lines)
            if isnothing(ln.tied) || cube_fitter.flexible_wavesol
                append!(ln_pars, [A_ln[i], voff_ln[i].value, fwhm_ln[i].value])
            else
                append!(ln_pars, [A_ln[i], fwhm_ln[i].value])
            end
            if ln.profile == :GaussHermite
                append!(ln_pars, [h3_ln[i].value, h4_ln[i].value])
            elseif ln.profile == :Voigt
                append!(ln_pars, [η_ln[i].value])
            end
            if !isnothing(ln.flow_profile)
                if isnothing(ln.flow_tied)
                    append!(ln_pars, [A_fl[i], flow_voff_ln[i].value, flow_fwhm_ln[i].value])
                else
                    append!(ln_pars, [A_fl[i], flow_fwhm_ln[i].value])
                end
                if ln.flow_profile == :GaussHermite
                    append!(ln_pars, [flow_h3_ln[i].value, flow_h4_ln[i].value])
                elseif ln.flow_profile == :Voigt
                    append!(ln_pars, [flow_η_ln[i].value])
                end
            end
        end
        voff_tied_pars = [cube_fitter.voff_tied[i].value for i ∈ 1:cube_fitter.n_voff_tied]
        flow_voff_tied_pars = [cube_fitter.flow_voff_tied[i].value for i ∈ 1:cube_fitter.n_flow_voff_tied]

        # Initial parameter vector
        p₀ = vcat(voff_tied_pars, flow_voff_tied_pars, ln_pars)

    end

    function ln_prior(p)
        # sum the log prior distribution of each parameter
        lnpdf = sum([logpdf(priors[i], p[i]) for i ∈ 1:length(p)])

        # penalize the likelihood if any in/outflow FWHMs are smaller than the corresponding narrow lines
        # or if the voffs are too small
        pᵢ = 1 + cube_fitter.n_voff_tied + cube_fitter.n_flow_voff_tied
        for i ∈ 1:cube_fitter.n_lines
            na_amp = p[pᵢ]
            if isnothing(cube_fitter.line_tied[i]) || cube_fitter.flexible_wavesol
                na_fwhm = p[pᵢ+2]
                pᵢ += 3
            else
                na_fwhm = p[pᵢ+1]
                pᵢ += 2
            end
            if prof_ln[i] == :GaussHermite
                pᵢ += 2
            elseif prof_ln[i] == :Voigt
                pᵢ += 1
            end
            fwhm_res = Util.C_KMS / cube_fitter.R
            if !isnothing(flow_prof_ln[i])
                flow_amp = p[pᵢ]
                if isnothing(cube_fitter.line_flow_tied[i])
                    flow_fwhm = p[pᵢ+2]
                    pᵢ += 3
                else
                    flow_fwhm = p[pᵢ+1]
                    pᵢ += 2
                end
                if flow_prof_ln[i] == :GaussHermite
                    pᵢ += 2
                elseif flow_prof_ln[i] == :Voigt
                    pᵢ += 1
                end
                if flow_fwhm ≤ na_fwhm || flow_amp ≥ na_amp
                    lnpdf += -Inf
                end
            end
        end 

        return lnpdf
    end

    function negln_probability(p)
        model = Util.fit_line_residuals(λ, p, cube_fitter.n_lines, cube_fitter.n_voff_tied, 
            cube_fitter.voff_tied_key, cube_fitter.line_tied, prof_ln, cube_fitter.n_flow_voff_tied,
            cube_fitter.flow_voff_tied_key, cube_fitter.line_flow_tied, flow_prof_ln, λ0_ln, 
            cube_fitter.flexible_wavesol)
        lnP = Util.ln_likelihood(Inorm, model, σnorm) + ln_prior(p)
        return -lnP 
    end

    lower_bounds = minimum.(priors)
    upper_bounds = maximum.(priors)
    # First, perform a bounded Simulated Annealing search for the optimal parameters with a generous rt and max iterations
    res = optimize(negln_probability, lower_bounds, upper_bounds, p₀, SAMIN(;rt=0.9, verbosity=0), Optim.Options(iterations=10^6))
    # Then, refine the solution with a bounded local minimum search with L-BFGS
    res = optimize(negln_probability, lower_bounds, upper_bounds, res.minimizer, Fminbox(LBFGS()))
    # Get the results
    popt = res.minimizer
    lnP = -res.minimum

    n_free = length(p₀)

    # Final optimized fit
    I_model, comps = Util.fit_line_residuals(λ, popt, cube_fitter.n_lines, cube_fitter.n_voff_tied, 
        cube_fitter.voff_tied_key, cube_fitter.line_tied, prof_ln, cube_fitter.n_flow_voff_tied,
        cube_fitter.flow_voff_tied_key, cube_fitter.line_flow_tied, flow_prof_ln, λ0_ln, 
        cube_fitter.flexible_wavesol, return_components=true)
    
    # Renormalize
    I_model = I_model .* N
    for comp ∈ keys(comps)
        comps[comp] = comps[comp] .* N
    end

    if verbose
        println("######################################################################")
        println("############### SPAXEL FIT RESULTS -- EMISSION LINES #################")
        println("######################################################################")
        println()
        pᵢ = 1
        println("#> TIED VELOCITY OFFSETS <#")
        println()
        for (i, vk) ∈ enumerate(cube_fitter.voff_tied_key)
            println("$(vk)_tied_voff: \t\t\t $(@sprintf "%.0f" popt[pᵢ]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.voff_tied[i].prior)), $(@sprintf "%.0f" maximum(cube_fitter.voff_tied[i].prior)))")
            pᵢ += 1
        end
        for (j, fvk) ∈ enumerate(cube_fitter.flow_voff_tied_key)
            println("$(fvk)_flow_tied_voff:\t\t\t $(@sprintf "%.0f" popt[pᵢ]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.flow_voff_tied[j].prior)), $(@sprintf "%.0f" maximum(cube_fitter.flow_voff_tied[j].prior)))")
            pᵢ += 1
        end
        println()
        println("#> EMISSION LINES <#")
        println()
        for (k, (ln, nm)) ∈ enumerate(zip(cube_fitter.lines, cube_fitter.line_names))
            println("$(nm)_amp:\t\t\t $(@sprintf "%.0f" popt[pᵢ]*N) MJy/sr \t Limits: (0, $(@sprintf "%.0f" nanmaximum(I)))")
            if isnothing(cube_fitter.line_tied[k]) || cube_fitter.flexible_wavesol
                println("$(nm)_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) km/s \t " *
                    "Limits: ($(@sprintf "%.0f" minimum(voff_ln[k].prior)), $(@sprintf "%.0f" maximum(voff_ln[k].prior)))")
                println("$(nm)_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+2]) km/s \t " *
                    "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))")
                if prof_ln[k] == :GaussHermite
                    println("$(nm)_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+3])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))")
                    println("$(nm)_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+4])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))")
                    pᵢ += 2
                elseif prof_ln[k] == :Voigt
                    println("$(nm)_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+3])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(η_ln[k].prior)), $(@sprintf "%.3f" maximum(η_ln[k].prior)))")
                    pᵢ += 1
                end
                pᵢ += 3
            else
                println("$(nm)_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) km/s \t " *
                    "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))")
                if prof_ln[k] == :GaussHermite
                    println("$(nm)_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+2])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))")
                    println("$(nm)_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+3])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))")
                    pᵢ += 2
                elseif prof_ln[k] == :Voigt
                    println("$(nm)_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+2])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(η_ln[k].prior)), $(@sprintf "%.3f" maximum(η_ln[k].prior)))")
                    pᵢ += 1
                end
                pᵢ += 2
            end
            if !isnothing(flow_prof_ln[k])
                println()
                println("$(nm)_flow_amp:\t\t\t $(@sprintf "%.0f" popt[pᵢ]*N) MJy/sr \t Limits: (0, $(@sprintf "%.0f" nanmaximum(I)))")
                if isnothing(cube_fitter.line_flow_tied[k])
                    println("$(nm)_flow_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) km/s \t " *
                        "Limits: ($(@sprintf "%.0f" minimum(voff_ln[k].prior)), $(@sprintf "%.0f" maximum(voff_ln[k].prior)))")
                    println("$(nm)_flow_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+2]) km/s \t " *
                        "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))")
                    if flow_prof_ln[k] == :GaussHermite
                        println("$(nm)_flow_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+3])      \t " *
                            "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))")
                        println("$(nm)_flow_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+4])      \t " *
                            "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))")
                        pᵢ += 2
                    elseif flow_prof_ln[k] == :Voigt
                        println("$(nm)_flow_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+3])      \t " *
                            "Limits: ($(@sprintf "%.3f" minimum(η_ln[k].prior)), $(@sprintf "%.3f" maximum(η_ln[k].prior)))")
                        pᵢ += 1
                    end
                    pᵢ += 3
                else
                    println("$(nm)_flow_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) km/s \t " *
                        "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))")
                    if flow_prof_ln[k] == :GaussHermite
                        println("$(nm)_flow_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+2])      \t " *
                            "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))")
                        println("$(nm)_flow_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+3])      \t " *
                            "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))")
                        pᵢ += 2
                    elseif flow_prof_ln[k] == :Voigt
                        println("$(nm)_flow_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+2])      \t " *
                            "Limits: ($(@sprintf "%.3f" minimum(η_ln[k].prior)), $(@sprintf "%.3f" maximum(η_ln[k].prior)))")
                        pᵢ += 1
                    end
                    pᵢ += 2
                end
            end
            println()
        end 
        println("######################################################################")
    end

    return σ, popt, I_model, comps, n_free

end

function plot_spaxel_fit(λ::Vector{Float64}, I::Vector{Float32}, I_cont::Vector{Float64}, σ::Vector{Float32}, 
    comps::Dict{String, Vector{Float64}}, n_dust_cont::Int, n_dust_features::Int, line_wave::Vector{Float64}, 
    line_names::Vector{Symbol}, χ2red::Float64, name::String, label::String; backend::Symbol=:pyplot)

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
                if occursin("flow", comp)
                    append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"], mode="lines",
                        line=Dict(:color => "#F574F9", :width => 1), name="Flows")])
                else
                    append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"], mode="lines",
                        line=Dict(:color => "rebeccapurple", :width => 1), name="Lines")])
                end
            end
        end
        for (lw, ln) ∈ zip(line_wave, line_names)
            append!(traces, [PlotlyJS.scatter(x=[lw, lw], y=[0., nanmaximum(I)*1.1], mode="lines", 
                line=Dict(:color => occursin("H2", String(ln)) ? "red" : (any(occursin.(["alpha", "beta", "gamma", "delta"], String(ln))) ? "#ff7f0e" : "rebeccapurple"), 
                :width => 0.5, :dash => "dash"))])
        end
        append!(traces, [PlotlyJS.scatter(x=λ, y=comps["extinction"] .* (sum([comps["dust_cont_$i"] for i ∈ 1:n_dust_cont], dims=1)[1] .+ comps["stellar"]), mode="lines",
            line=Dict(:color => "green", :width => 1), name="Dust+Stellar Continuum")])
        layout = PlotlyJS.Layout(
            xaxis_title="\$\\lambda\\ (\\mu{\\rm m})\$",
            yaxis_title="\$I_{\\nu}\\ ({\\rm MJy}\\,{\\rm sr}^{-1})\$",
            title="\$\\tilde{\\chi}^2 = $χ2red\$",
            xaxis_constrain="domain",
            font_family="Georgia, Times New Roman, Serif",
            template="plotly_white",
            # annotations=[
            #     attr(x=lw, y=nanmaximum(I)*.75, text=ll) for (lw, ll) ∈ zip(line_wave, line_latex)
            # ]
        )
        p = PlotlyJS.plot(traces, layout)
        PlotlyJS.savefig(p, isnothing(label) ? "output_$name/spaxel_fits/levmar_fit_spaxel.html" : "output_$name/spaxel_fits/$label.html")

    elseif backend == :pyplot
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(nrows=4, ncols=1, hspace=0.)
        ax1 = fig.add_subplot(py"$(gs)[:-1, :]")
        ax2 = fig.add_subplot(py"$(gs)[-1, :]")
        ax1.plot(λ, I, "k-", label="Data")
        ax1.plot(λ, I_cont, "r-", label="Model")
        ax2.plot(λ, I.-I_cont, "k-")
        ax2.plot(λ, ones(length(λ)), "r-")
        ax2.fill_between(λ, σ, .-σ, color="k", alpha=0.5)
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
                if occursin("flow", comp)
                    ax1.plot(λ, comps[comp] .* comps["extinction"], "-", color="#F574F9", alpha=0.5)
                else
                    ax1.plot(λ, comps[comp] .* comps["extinction"], "-", color=:rebeccapurple, alpha=0.5)
                end
            end
        end
        for (lw, ln) ∈ zip(line_wave, line_names)
            ax1.axvline(lw, linestyle="--", 
                color=occursin("H2", String(ln)) ? :red : (any(occursin.(["alpha", "beta", "gamma", "delta"], String(ln))) ? "#ff7f0e" : :rebeccapurple), lw=0.5, alpha=0.5)
            ax2.axvline(lw, linestyle="--", 
                color=occursin("H2", String(ln)) ? :red : (any(occursin.(["alpha", "beta", "gamma", "delta"], String(ln))) ? "#ff7f0e" : :rebeccapurple), lw=0.5, alpha=0.5)
        end
        ax1.plot(λ, comps["extinction"] .* (sum([comps["dust_cont_$i"] for i ∈ 1:n_dust_cont], dims=1)[1] .+ comps["stellar"]), "g-")
        ax1.set_xlim(minimum(λ), maximum(λ))
        ax2.set_xlim(minimum(λ), maximum(λ))
        ax3.set_ylim(0., 1.1)
        ax3.set_ylabel("Extinction")
        ax1.set_ylabel("\$ I_{\\nu} \$ (MJy sr\$^{-1}\$)")
        ax1.set_ylim(bottom=0.)
        ax2.set_ylabel("Residuals")
        ax2.set_xlabel("\$ \\lambda \$ (\$\\mu\$m)")
        # ax1.legend()
        ax1.set_title("\$\\tilde{\\chi}^2 = $(@sprintf "%.3f" χ2red)\$")
        ax1.tick_params(axis="both", direction="in")
        ax2.tick_params(axis="both", direction="in", labelright=true, right=true)
        ax3.tick_params(axis="both", direction="in")
        plt.savefig(isnothing(label) ? "output_$name/spaxel_fits/levmar_fit_spaxel.pdf" : "output_$name/spaxel_fits/$label.pdf", dpi=300, bbox_inches="tight")
        plt.close()
    end
end

# function pahfit_spaxel(cube_fitter::CubeFitter, spaxel::Tuple{Int,Int})
    
#     # Extract spaxel to be fit
#     λ = cube_fitter.cube.λ
#     I = cube_fitter.cube.Iλ[spaxel..., :]
#     σ = cube_fitter.cube.σI[spaxel..., :]

#     # Filter NaNs
#     if sum(.!isfinite.(I) .| .!isfinite.(σ)) > (size(I, 1) / 10)
#         return
#     end
#     filt = .!isfinite.(I) .& .!isfinite.(σ)

#     # Interpolate the NaNs
#     if sum(filt) > 0
#         # Make sure the wavelength vector is linear, since it is assumed later in the function
#         diffs = diff(λ)
#         @assert diffs[1] ≈ diffs[end]
#         Δλ = diffs[1]

#         # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
#         λknots = λ[length(λ) ÷ 13]:Δλ*25:λ[end-(length(λ) ÷ 13)]
#         # ONLY replace NaN values, keep the rest of the data as-is
#         I[filt] .= Spline1D(λ[isfinite.(I)], I[isfinite.(I)], λknots, k=3, bc="extrapolate").(λ[filt])
#         σ[filt] .= Spline1D(λ[isfinite.(σ)], σ[isfinite.(σ)], λknots, k=3, bc="extrapolate").(λ[filt])
#     end 

#     folder = joinpath(@__DIR__, "idl_data")
#     if !isdir(folder)
#         mkdir(folder)
#     end

#     if !isfile("$folder/spaxel_$(spaxel[1])_$(spaxel[2])_params.csv") || !isfile("$folder/spaxel_$(spaxel[1])_$(spaxel[2])_fit.csv")    
#         CSV.write("$folder/spaxel_$(spaxel[1])_$(spaxel[2]).csv", DataFrame(wave=λ, intensity=I, err=σ))
#         IDL_DIR = ENV["IDL_DIR"]
#         run(`$IDL_DIR/bin/idl $(@__DIR__)/fit_spaxel.pro -args $(@__DIR__)/idl_data/spaxel_$(spaxel[1])_$(spaxel[2]).csv`);
#     end

#     popt = CSV.read("$folder/spaxel_$(spaxel[1])_$(spaxel[2])_params.csv", DataFrame)[!, :params]
#     I_cont = CSV.read("$folder/spaxel_$(spaxel[1])_$(spaxel[2])_fit.csv", DataFrame)[!, :intensity]

#     return popt, I_cont

# end

function calculate_extra_parameters(cube_fitter::CubeFitter, spaxel::Tuple{Int, Int}, comps::Dict{String, Vector{Float64}})

    λ = cube_fitter.cube.λ
    I = cube_fitter.cube.Iλ[spaxel..., :]
    σ = cube_fitter.cube.σI[spaxel..., :]

    p_complex = zeros(2cube_fitter.n_complexes)
    pᵢ = 1
    for c ∈ cube_fitter.complexes
        Iᵢ = zeros(length(λ))
        # Add up the dust feature profiles that belong to this complex
        for (ii, cdf) ∈ enumerate(cube_fitter.dust_features)
            if cdf[:complex] == c
                Iᵢ .+= comps["dust_feat_$ii"]
            end
        end
        # Integrate the intensity of the combined profile
        window = parse(Float64, c)
        p_complex[pᵢ] = NumericalIntegration.integrate(λ, Iᵢ, SimpsonEven())
        p_complex[pᵢ] = p_complex[pᵢ] > 0. ? log10(p_complex[pᵢ]) : -Inf
        # SNR
        # p_complex[pᵢ+1] = maximum(Iᵢ) / 
        #     std(I[(window-2cube_fitter.window_size .< λ .< window-cube_fitter.window_size) .| 
        #         (window+cube_fitter.window_size .< λ .< window+2cube_fitter.window_size)])
        disp = window / cube_fitter.R
        p_complex[pᵢ+1] = p_complex[pᵢ] / 
            (√π * disp * std(I[(window-2cube_fitter.window_size .< λ .< window-cube_fitter.window_size) .| 
            (window+cube_fitter.window_size .< λ .< window+2cube_fitter.window_size)]))
        pᵢ += 2
    end

    p_lines = zeros(2cube_fitter.n_lines)
    pᵢ = 1
    for (k, ln) ∈ enumerate(cube_fitter.lines)

        # Line intensity
        profile = zeros(length(λ))
        profile .+= comps["line_$(k)"]
        if haskey(comps, "line_$(k)_flow")
            profile .+= comps["line_$(k)_flow"]
        end
        window = ln.λ₀
        p_lines[pᵢ] = NumericalIntegration.integrate(λ, profile, SimpsonEven())
        p_lines[pᵢ] = p_lines[pᵢ] > 0. ? log10(p_lines[pᵢ]) : -Inf

        # SNR
        # p_lines[pᵢ+1] = maximum(profile) /
        #     std(I[(window-2cube_fitter.window_size .< λ .< window-cube_fitter.window_size) .| 
        #         (window+cube_fitter.window_size .< λ .< window+2cube_fitter.window_size)])
        disp = window / cube_fitter.R
        p_lines[pᵢ+1] = p_lines[pᵢ] / 
            (√π * disp * std(I[(window-2cube_fitter.window_size .< λ .< window-cube_fitter.window_size) .| 
            (window+cube_fitter.window_size .< λ .< window+2cube_fitter.window_size)]))
    
        pᵢ += 2

    end

    return p_complex, p_lines
end

# Utility function for fitting a single spaxel
function fit_spaxel(cube_fitter::CubeFitter, spaxel::Tuple{Int, Int}; verbose::Bool=false)

    # Skip spaxels with NaNs (post-interpolation)
    λ = cube_fitter.cube.λ
    I = cube_fitter.cube.Iλ[spaxel..., :]
    if any(.!isfinite.(I) .| .!isfinite.(I))
        return
    end

    # Fit the spaxel
    σ, popt_c, I_cont, comps_cont, n_free_c = continuum_fit_spaxel(cube_fitter, spaxel, verbose=verbose)
    _, popt_l, I_line, comps_line, n_free_l = line_fit_spaxel(cube_fitter, spaxel, I_cont, verbose=verbose)

    # Combine the continuum and line models
    I_model = I_cont .+ I_line
    comps = merge(comps_cont, comps_line)

    # Total free parameters
    n_free = n_free_c + n_free_l
    n_data = length(I)

    # Reduced chi^2 of the model
    χ2red = 1 / (n_data - n_free) * sum((I .- I_model).^2 ./ σ.^2)

    # Add dust complex and line parameters (intensity and SNR)
    p_complex, p_lines = calculate_extra_parameters(cube_fitter, spaxel, comps)
    p_out = [popt_c; popt_l; p_complex; p_lines]

    # Plot the fit
    λ0_ln = [ln.λ₀ for ln ∈ cube_fitter.lines]
    if cube_fitter.plot_spaxels != :none
        plot_spaxel_fit(λ, I, I_model, σ, comps, 
            cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, λ0_ln, cube_fitter.line_names,
            χ2red, cube_fitter.name, "spaxel_$(spaxel[1])_$(spaxel[2])", backend=cube_fitter.plot_spaxels)
    end

    # If the reduced chi^2 meets the threshold, set the new starting parameters for the rest of the fits
    # if isnothing(cube_fitter.best_spaxel)
    #     if χ2red ≤ cube_fitter.χ²_thresh
    #         cube_fitter.p_best_cont = popt_c
    #         cube_fitter.p_best_line = popt_l
    #         cube_fitter.χ²_best = χ2red
    #         cube_fitter.best_spaxel = spaxel
    #     end
    # # If they already exist, don't overwrite unless the new chi^2 is better than the old chi^2
    # else
    #     if χ2red < cube_fitter.χ²_best
    #         cube_fitter.p_best_cont = popt_c
    #         cube_fitter.p_best_line = popt_l
    #         cube_fitter.χ²_best = χ2red
    #         cube_fitter.best_spaxel = spaxel
    #     end
    # end

    return p_out, I_model, comps, χ2red
end

function fit_cube(cube_fitter::CubeFitter)

    shape = size(cube_fitter.cube.Iλ)
    # Interpolate NaNs in the cube
    interpolate_cube!(cube_fitter.cube)

    # Prepare output array
    println("Preparing output data structures...")
    out_params = SharedArray(ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 1) .* NaN)

    #########################
    function fit_spax_i(xᵢ::Int, yᵢ::Int)

        result = fit_spaxel(cube_fitter, (xᵢ, yᵢ))
        if !isnothing(result)
            p_out, _, _, χ2red = result
            out_params[xᵢ, yᵢ, :] .= [p_out; χ2red]
        end

        return
    end

    # Sort spaxels by median brightness, so that we fit the brightest ones first
    # (which hopefully have the best reduced chi^2s)
    spaxels = Iterators.product(1:shape[1], 1:shape[2])
    med_I = collect(Iterators.flatten([nanmedian(cube_fitter.cube.Iλ[spaxel..., :]) for spaxel ∈ spaxels]))
    # replace NaNs with -1s
    med_I[.!isfinite.(med_I)] .= -1.
    # reverse sort
    ss = sortperm(med_I, rev=true)
    med_I = med_I[ss]
    # apply sorting to spaxel indices
    spaxels = collect(spaxels)[ss]

    println("Beginning Levenberg-Marquardt least squares fitting...")
    # Use multiprocessing (not threading) to iterate over multiple spaxels at once using multiple CPUs
    if cube_fitter.parallel
        @showprogress pmap(spaxels) do (xᵢ, yᵢ)
            fit_spax_i(xᵢ, yᵢ)
        end
    else
        @showprogress for (xᵢ, yᵢ) ∈ spaxels
            fit_spax_i(xᵢ, yᵢ)
        end
    end

    println("Updating parameter maps and model cubes...")
    for (xᵢ, yᵢ) ∈ Iterators.product(1:shape[1], 1:shape[2])
        # Set the 2D parameter map outputs

        # Stellar continuum amplitude, temp
        cube_fitter.param_maps.stellar_continuum[:amp][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, 1] > 0. ? log10(out_params[xᵢ, yᵢ, 1]) : -Inf 
        cube_fitter.param_maps.stellar_continuum[:temp][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, 2]
        pᵢ = 3
        # Dust continuum amplitude, temp
        for i ∈ 1:cube_fitter.n_dust_cont
            cube_fitter.param_maps.dust_continuum[i][:amp][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ] > 0. ? log10(out_params[xᵢ, yᵢ, pᵢ]) : -Inf
            cube_fitter.param_maps.dust_continuum[i][:temp][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+1]
            pᵢ += 2
        end
        # Dust feature log(amplitude), mean, FWHM
        for df ∈ cube_fitter.df_names
            cube_fitter.param_maps.dust_features[df][:amp][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ] > 0. ? log10(out_params[xᵢ, yᵢ, pᵢ]) : -Inf
            cube_fitter.param_maps.dust_features[df][:mean][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+1]
            cube_fitter.param_maps.dust_features[df][:fwhm][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+2]
            pᵢ += 3
        end
        # Extinction parameters
        cube_fitter.param_maps.extinction[:tau_9_7][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ]
        cube_fitter.param_maps.extinction[:beta][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+1]
        pᵢ += 2

        # End of continuum parameters: recreate the continuum model
        I_cont, comps_c = Util.fit_spectrum(cube_fitter.cube.λ, out_params[xᵢ, yᵢ, 1:pᵢ-1], cube_fitter.n_dust_cont, cube_fitter.n_dust_feat;
            return_components=true)

        # Tied line velocity offsets
        vᵢ = pᵢ
        for vk ∈ cube_fitter.voff_tied_key
            cube_fitter.param_maps.tied_voffs[vk][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ]
            pᵢ += 1
        end
        # Tied flow velocity offsets
        for fvk ∈ cube_fitter.flow_voff_tied_key
            cube_fitter.param_maps.flow_tied_voffs[fvk][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ]
            pᵢ += 1
        end
        for (k, ln) ∈ enumerate(cube_fitter.line_names)
            # Log of line amplitude
            cube_fitter.param_maps.lines[ln][:amp][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ] > 0. ? log10(out_params[xᵢ, yᵢ, pᵢ]) : -Inf
            fwhm_res = Util.C_KMS / cube_fitter.R

            if isnothing(cube_fitter.line_tied[k]) || cube_fitter.flexible_wavesol
                # Individual shift
                cube_fitter.param_maps.lines[ln][:voff][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+1]
                if !isnothing(cube_fitter.line_tied[k])
                    # If velocity is tied while flexible, add the overall shift and the individual shift together
                    vwhere = findfirst(x -> x == cube_fitter.line_tied[k], cube_fitter.voff_tied_key)
                    cube_fitter.param_maps.lines[ln][:voff][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+1] + out_params[xᵢ, yᵢ, vᵢ+vwhere-1]
                end
                # FWHM -> subtract instrumental resolution in quadrature
                if fwhm_res > out_params[xᵢ, yᵢ, pᵢ+2]
                    cube_fitter.param_maps.lines[ln][:fwhm][xᵢ, yᵢ] = 0.
                else
                    cube_fitter.param_maps.lines[ln][:fwhm][xᵢ, yᵢ] = √(out_params[xᵢ, yᵢ, pᵢ+2]^2 - fwhm_res^2)
                end
                # Get Gauss-Hermite 3rd and 4th order moments
                if cube_fitter.line_profiles[k] == :GaussHermite
                    cube_fitter.param_maps.lines[ln][:h3][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+3]
                    cube_fitter.param_maps.lines[ln][:h4][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+4]
                    pᵢ += 2
                elseif cube_fitter.line_profiles[k] == :Voigt
                    cube_fitter.param_maps.lines[ln][:mixing][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+3]
                    pᵢ += 1
                end
                pᵢ += 3
            else
                # FWHM -> subtract instrumental resolution in quadrature
                if fwhm_res > out_params[xᵢ, yᵢ, pᵢ+1]
                    cube_fitter.param_maps.lines[ln][:fwhm][xᵢ, yᵢ] = 0.
                else
                    cube_fitter.param_maps.lines[ln][:fwhm][xᵢ, yᵢ] = √(out_params[xᵢ, yᵢ, pᵢ+1]^2 - fwhm_res^2)
                end
                # Get Gauss-Hermite 3rd and 4th order moments
                if cube_fitter.line_profiles[k] == :GaussHermite
                    cube_fitter.param_maps.lines[ln][:h3][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+2]
                    cube_fitter.param_maps.lines[ln][:h4][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+3]
                    pᵢ += 2
                elseif cube_fitter.line_profiles[k] == :Voigt
                    cube_fitter.param_maps.lines[k][:mixing][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+2]
                    pᵢ += 1
                end
                pᵢ += 2
            end

            if !isnothing(cube_fitter.line_flow_profiles[k])
                cube_fitter.param_maps.lines[ln][:flow_amp][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ] > 0. ? log10(out_params[xᵢ, yᵢ, pᵢ]) : -Inf

                if isnothing(cube_fitter.line_flow_tied[k])
                    # Individual shift
                    cube_fitter.param_maps.lines[ln][:flow_voff][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+1]
                    # FWHM -> subtract instrumental resolution in quadrature
                    if fwhm_res > out_params[xᵢ, yᵢ, pᵢ+2]
                        cube_fitter.param_maps.lines[ln][:flow_fwhm][xᵢ, yᵢ] = 0.
                    else
                        cube_fitter.param_maps.lines[ln][:flow_fwhm][xᵢ, yᵢ] = √(out_params[xᵢ, yᵢ, pᵢ+2]^2 - fwhm_res^2)
                    end
                    # Get Gauss-Hermite 3rd and 4th order moments
                    if cube_fitter.line_flow_profiles[k] == :GaussHermite
                        cube_fitter.param_maps.lines[ln][:flow_h3][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+3]
                        cube_fitter.param_maps.lines[ln][:flow_h4][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+4]
                        pᵢ += 2
                    elseif cube_fitter.line_flow_profiles[k] == :Voigt
                        cube_fitter.param_maps.lines[ln][:flow_mixing][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+3]
                        pᵢ += 1
                    end
                    pᵢ += 3
                else
                    # FWHM -> subtract instrumental resolution in quadrature
                    if fwhm_res > out_params[xᵢ, yᵢ, pᵢ+1]
                        cube_fitter.param_maps.lines[ln][:flow_fwhm][xᵢ, yᵢ] = 0.
                    else
                        cube_fitter.param_maps.lines[ln][:flow_fwhm][xᵢ, yᵢ] = √(out_params[xᵢ, yᵢ, pᵢ+1]^2 - fwhm_res^2)
                    end
                    # Get Gauss-Hermite 3rd and 4th order moments
                    if cube_fitter.line_flow_profiles[k] == :GaussHermite
                        cube_fitter.param_maps.lines[ln][:flow_h3][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+2]
                        cube_fitter.param_maps.lines[ln][:flow_h4][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+3]
                        pᵢ += 2
                    elseif cube_fitter.line_flow_profiles[k] == :Voigt
                        cube_fitter.param_maps.lines[k][:flow_mixing][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+2]
                        pᵢ += 1
                    end
                    pᵢ += 2
                end
            end

        end

        # End of line parameters: recreate the line model
        I_line, comps_l = Util.fit_line_residuals(cube_fitter.cube.λ, out_params[xᵢ, yᵢ, vᵢ:pᵢ-1], cube_fitter.n_lines, cube_fitter.n_voff_tied,
            cube_fitter.voff_tied_key, cube_fitter.line_tied, [ln.profile for ln ∈ cube_fitter.lines], cube_fitter.n_flow_voff_tied, cube_fitter.flow_voff_tied_key,
            cube_fitter.line_flow_tied, [ln.flow_profile for ln ∈ cube_fitter.lines], [ln.λ₀ for ln ∈ cube_fitter.lines], 
            cube_fitter.flexible_wavesol; return_components=true)

        for c ∈ cube_fitter.complexes
            # Dust complex intensity and SNR, from calculate_extra_parameters
            cube_fitter.param_maps.dust_complexes[c][:intI][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ] > 0. ? log10(out_params[xᵢ, yᵢ, pᵢ]) : -Inf
            cube_fitter.param_maps.dust_complexes[c][:SNR][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+1]
            pᵢ += 2
        end

        for (k, ln) ∈ enumerate(cube_fitter.line_names)
            # Line intensity and SNR, from calculate_extra_parameters
            cube_fitter.param_maps.lines[ln][:intI][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ] > 0. ? log10(out_params[xᵢ, yᵢ, pᵢ]) : -Inf
            cube_fitter.param_maps.lines[ln][:SNR][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+1]
            pᵢ += 2
        end

        # Reduced χ^2
        cube_fitter.param_maps.reduced_χ2[xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ]

        # Set the 3D model cube outputs
        I_model = I_cont .+ I_line
        comps = merge(comps_c, comps_l)

        cube_fitter.cube_model.model[xᵢ, yᵢ, :] .= I_model
        cube_fitter.cube_model.stellar[xᵢ, yᵢ, :] .= comps["stellar"]
        for i ∈ 1:cube_fitter.n_dust_cont
            cube_fitter.cube_model.dust_continuum[xᵢ, yᵢ, :, i] .= comps["dust_cont_$i"]
        end
        for j ∈ 1:cube_fitter.n_dust_feat
            cube_fitter.cube_model.dust_features[xᵢ, yᵢ, :, j] .= comps["dust_feat_$j"]
        end
        for k ∈ 1:cube_fitter.n_lines
            cube_fitter.cube_model.lines[xᵢ, yᵢ, :, k] .= comps["line_$k"]
            if haskey(comps, "line_$(k)_flow")
                cube_fitter.cube_model.lines[xᵢ, yᵢ, :, k] .+= comps["line_$(k)_flow"]
            end
        end
        cube_fitter.cube_model.extinction[xᵢ, yᵢ, :] .= comps["extinction"]

    end

    if cube_fitter.plot_maps
        println("Plotting parameter maps...")
        plot_parameter_maps(cube_fitter)
    end

    if cube_fitter.save_fits
        println("Writing FITS outputs...")
        write_fits(cube_fitter)
    end

    println("Done!")

    return cube_fitter

end

# Plotting utility function
function plot_parameter_map(data::Union{Matrix{Float64},SharedMatrix{Float64}}, name::String, name_i::String;
    Ω::Union{Float64,Nothing}=nothing, z::Union{Float64,Nothing}=nothing, cosmo::Union{Cosmology.AbstractCosmology,Nothing}=nothing,
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
    elseif occursin("chi2", String(name_i))
        bunit = "\$\\tilde{\\chi}^2\$"
    elseif occursin("h3", String(name_i))
        bunit = "\$h_3\$"
    elseif occursin("h4", String(name_i))
        bunit = "\$h_4\$"
    elseif occursin("mixing", String(name_i))
        bunit = "\$\\eta\$"
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

    if !isnothing(Ω) && !isnothing(z) && !isnothing(cosmo)
        n_pix = 1/(sqrt(Ω) * 180/π * 3600)
        dL = luminosity_dist(cosmo, z).val * 1e6 / (180/π * 3600)  # l = d * theta (1")
        dL = @sprintf "%.0f" dL
        scalebar = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, "1\$\'\'\$ / $dL pc", "lower left", pad=1, color=:black, 
            frameon=false, size_vertical=0.2, label_top=true)
        ax.add_artist(scalebar)
    end

    fig.colorbar(cdata, ax=ax, label=bunit)
    plt.savefig("output_$(name)/param_maps/$(name_i).pdf", dpi=300, bbox_inches=:tight)
    plt.close()

end

function plot_parameter_maps(cube_fitter::CubeFitter; snr_thresh=3.)

    # Make subdirectory
    if !isdir("output_$(cube_fitter.name)/param_maps")
        mkdir("output_$(cube_fitter.name)/param_maps")
    end

    # Iterate over model parameters and make 2D maps
    for parameter ∈ keys(cube_fitter.param_maps.stellar_continuum)
        data = cube_fitter.param_maps.stellar_continuum[parameter]
        name_i = join(["stellar_continuum", parameter], "_")
        plot_parameter_map(data, cube_fitter.name, name_i, Ω=cube_fitter.cube.Ω, z=cube_fitter.z, cosmo=cube_fitter.cosmology)
    end

    for i ∈ keys(cube_fitter.param_maps.dust_continuum)
        for parameter ∈ keys(cube_fitter.param_maps.dust_continuum[i])
            data = cube_fitter.param_maps.dust_continuum[i][parameter]
            name_i = join(["dust_continuum", i, parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, Ω=cube_fitter.cube.Ω, z=cube_fitter.z, cosmo=cube_fitter.cosmology)
        end
    end

    for df ∈ keys(cube_fitter.param_maps.dust_features)
        for parameter ∈ keys(cube_fitter.param_maps.dust_features[df])
            data = cube_fitter.param_maps.dust_features[df][parameter]
            name_i = join(["dust_features", df, parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, Ω=cube_fitter.cube.Ω, z=cube_fitter.z, cosmo=cube_fitter.cosmology)
        end
    end

    for parameter ∈ keys(cube_fitter.param_maps.extinction)
        data = cube_fitter.param_maps.extinction[parameter]
        name_i = join(["extinction", parameter], "_")
        plot_parameter_map(data, cube_fitter.name, name_i, Ω=cube_fitter.cube.Ω, z=cube_fitter.z, cosmo=cube_fitter.cosmology)
    end

    for c ∈ keys(cube_fitter.param_maps.dust_complexes)
        snr = cube_fitter.param_maps.dust_complexes[c][:SNR]
        for parameter ∈ keys(cube_fitter.param_maps.dust_complexes[c])
            data = cube_fitter.param_maps.dust_complexes[c][parameter]
            name_i = join(["dust_complexes", c, parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, Ω=cube_fitter.cube.Ω, z=cube_fitter.z, cosmo=cube_fitter.cosmology,
                snr_filter=parameter ≠ :SNR ? snr : nothing, snr_thresh=snr_thresh)
        end
    end

    for vk ∈ cube_fitter.voff_tied_key
        data = cube_fitter.param_maps.tied_voffs[vk]
        name_i = join(["tied_voffs", vk], "_")
        plot_parameter_map(data, cube_fitter.name, name_i, Ω=cube_fitter.cube.Ω, z=cube_fitter.z, cosmo=cube_fitter.cosmology)
    end

    for line ∈ keys(cube_fitter.param_maps.lines)
        snr = cube_fitter.param_maps.lines[line][:SNR]
        for parameter ∈ keys(cube_fitter.param_maps.lines[line])
            data = cube_fitter.param_maps.lines[line][parameter]
            name_i = join(["lines", line, parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, Ω=cube_fitter.cube.Ω, z=cube_fitter.z, cosmo=cube_fitter.cosmology,
                snr_filter=parameter ≠ :SNR ? snr : nothing, snr_thresh=snr_thresh)
        end
    end

    data = cube_fitter.param_maps.reduced_χ2
    name_i = "reduced_chi2"
    plot_parameter_map(data, cube_fitter.name, name_i, Ω=cube_fitter.cube.Ω, z=cube_fitter.z, cosmo=cube_fitter.cosmology)

end

function write_fits(cube_fitter::CubeFitter)

    # Header information
    hdr = FITSHeader(
        ["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "PIXAR_SR", "RA", "DEC", "WCSAXES",
            "CDELT1", "CDELT2", "CDELT3", "CTYPE1", "CTYPE2", "CTYPE3", "CRPIX1", "CRPIX2", "CRPIX3",
            "CRVAL1", "CRVAL2", "CRVAL3", "CUNIT1", "CUNIT2", "CUNIT3", "PC1_1", "PC1_2", "PC1_3", 
            "PC2_1", "PC2_2", "PC2_3", "PC3_1", "PC3_2", "PC3_3"],

        # Check if the redshift correction is right for the third WCS axis?
        [cube_fitter.name, cube_fitter.z, cube_fitter.cube.channel, cube_fitter.cube.band, cube_fitter.cube.Ω, cube_fitter.cube.α, cube_fitter.cube.δ, 
         cube_fitter.cube.wcs.naxis, cube_fitter.cube.wcs.cdelt[1], cube_fitter.cube.wcs.cdelt[2], cube_fitter.cube.wcs.cdelt[3]/(1+cube_fitter.z), 
         cube_fitter.cube.wcs.ctype[1], cube_fitter.cube.wcs.ctype[2], cube_fitter.cube.wcs.ctype[3], cube_fitter.cube.wcs.crpix[1], 
         cube_fitter.cube.wcs.crpix[2], cube_fitter.cube.wcs.crpix[3], cube_fitter.cube.wcs.crval[1], cube_fitter.cube.wcs.crval[2], 
         cube_fitter.cube.wcs.crval[3]/(1+cube_fitter.z), cube_fitter.cube.wcs.cunit[1], cube_fitter.cube.wcs.cunit[2], cube_fitter.cube.wcs.cunit[3], 
         cube_fitter.cube.wcs.pc[1,1], cube_fitter.cube.wcs.pc[1,2], cube_fitter.cube.wcs.pc[1,3], cube_fitter.cube.wcs.pc[2,1], cube_fitter.cube.wcs.pc[2,2], 
         cube_fitter.cube.wcs.pc[2,3], cube_fitter.cube.wcs.pc[3,1], cube_fitter.cube.wcs.pc[3,2], cube_fitter.cube.wcs.pc[3,3]],

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

    FITS("output_$(cube_fitter.name)/$(cube_fitter.name)_3D_model.fits", "w") do f
        write(f, Vector{Int}())                                                                                 # Primary HDU (empty)
        write(f, cube_fitter.cube.Iλ; header=hdr, name="DATA")                                                  # Raw data with nans inserted
        write(f, cube_fitter.cube_model.model; header=hdr, name="MODEL")                                        # Full intensity model
        write(f, cube_fitter.cube.Iλ .- cube_fitter.cube_model.model; header=hdr, name="RESIDUALS")             # Residuals (data - model)
        write(f, cube_fitter.cube_model.stellar; header=hdr, name="STELLAR_CONTINUUM")                          # Stellar continuum model
        for i ∈ 1:size(cube_fitter.cube_model.dust_continuum, 4)
            write(f, cube_fitter.cube_model.dust_continuum[:, :, :, i]; header=hdr, name="DUST_CONTINUUM_$i")   # Dust continuum models
        end
        for (j, df) ∈ enumerate(cube_fitter.df_names)
            write(f, cube_fitter.cube_model.dust_features[:, :, :, j]; header=hdr, name="$df")                  # Dust feature profiles
        end
        for (k, line) ∈ enumerate(cube_fitter.line_names)
            write(f, cube_fitter.cube_model.lines[:, :, :, k]; header=hdr, name="$line")                        # Emission line profiles
        end
        write(f, cube_fitter.cube_model.extinction; header=hdr, name="EXTINCTION")                              # Extinction model
        write_key(f["MODEL"], "BUNIT", "MJy/sr")
    end

    FITS("output_$(cube_fitter.name)/$(cube_fitter.name)_parameter_maps.fits", "w") do f

        write(f, Vector{Int}())  # Primary HDU (empty)

        # Iterate over model parameters and make 2D maps
        for parameter ∈ keys(cube_fitter.param_maps.stellar_continuum)
            data = cube_fitter.param_maps.stellar_continuum[parameter]
            name_i = join(["stellar_continuum", parameter], "_")
            if occursin("amp", String(name_i))
                bunit = "log10(I / MJy sr^-1)"
            elseif occursin("temp", String(name_i))
                bunit = "Kelvin"
            end
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end

        for i ∈ keys(cube_fitter.param_maps.dust_continuum)
            for parameter ∈ keys(cube_fitter.param_maps.dust_continuum[i])
                data = cube_fitter.param_maps.dust_continuum[i][parameter]
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

        for df ∈ keys(cube_fitter.param_maps.dust_features)
            for parameter ∈ keys(cube_fitter.param_maps.dust_features[df])
                data = cube_fitter.param_maps.dust_features[df][parameter]
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

        for line ∈ keys(cube_fitter.param_maps.lines)
            for parameter ∈ keys(cube_fitter.param_maps.lines[line])
                data = cube_fitter.param_maps.lines[line][parameter]
                name_i = join(["lines", line, parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / MJy sr^-1)"
                elseif occursin("fwhm", String(name_i)) || occursin("voff", String(name_i))
                    bunit = "km/s"
                elseif occursin("intI", String(name_i))
                    bunit = "log10(I / MJy sr^-1 um)"
                elseif occursin("SNR", String(name_i)) || occursin("h3", String(name_i)) || 
                    occursin("h4", String(name_i)) || occursin("mixing", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)   
            end
        end

        for parameter ∈ keys(cube_fitter.param_maps.extinction)
            data = cube_fitter.param_maps.extinction[parameter]
            name_i = join(["extinction", parameter], "_")
            bunit = "unitless"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)  
        end

        for vk ∈ cube_fitter.voff_tied_key
            data = cube_fitter.param_maps.tied_voffs[vk]
            name_i = join(["tied_voffs", vk], "_")
            bunit = "km/s"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end

        for c ∈ keys(cube_fitter.param_maps.dust_complexes)
            snr = cube_fitter.param_maps.dust_complexes[c][:SNR]
            for parameter ∈ keys(cube_fitter.param_maps.dust_complexes[c])
                data = cube_fitter.param_maps.dust_complexes[c][parameter]
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

        data = cube_fitter.param_maps.reduced_χ2
        name_i = "reduced_chi2"
        bunit = "unitless"
        write(f, data; header=hdr, name=name_i)
        write_key(f[name_i], "BUNIT", bunit)

    end
end

end