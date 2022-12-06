module CubeFit

export CubeFitter, fit_cube, continuum_fit_spaxel, line_fit_spaxel, fit_spaxel, plot_parameter_maps, write_fits

# Parallel computing packages
using Distributed
using SharedArrays

# Math packages
using Distributions
using Statistics
using NaNStatistics
using QuadGK
using Interpolations
using Dierckx

# Optimization packages
using Optim
using CMPFit

# Astronomy packages
using FITSIO
using Cosmology

# File I/O
using TOML
using CSV
using DataFrames

# Plotting packages
using PlotlyJS
using PyPlot

# Misc packages/utilites
using ProgressMeter
using Reexport
using Serialization
using Printf
using Logging
using LoggingExtras
using Dates

# PyCall needed for anchored_artists
using PyCall
# Have to import anchored_artists within the __init__ function so that it works after precompilation
const py_anchored_artists = PyNULL()

# MATPLOTLIB SETTINGS TO MAKE PLOTS LOOK PRETTY :)
const SMALL = 12
const MED = 14
const BIG = 16
function __init__()
    # Import matplotlib's anchored_artists package for scale bars
    copy!(py_anchored_artists, pyimport_conda("mpl_toolkits.axes_grid1.anchored_artists", "matplotlib"))

    plt.switch_backend("Agg")
    plt.rc("font", size=MED)          # controls default text sizes
    plt.rc("axes", titlesize=MED)     # fontsize of the axes title
    plt.rc("axes", labelsize=MED)     # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MED)    # legend fontsize
    plt.rc("figure", titlesize=BIG)   # fontsize of the figure title
    plt.rc("text", usetex=true)       # use LaTeX
    plt.rc("font", family="Times New Roman")   # use Times New Roman font
end

# Import and reexport the parameters functions for use all throughout the code
include("parameters.jl")
@reexport using .Param

# Import and reexport the cubedata functions for use all throughout the code
include("cubedata.jl")
@reexport using .CubeData
# Note cubedata.jl also contains the Utils functions through reexport


############################## OPTIONS/SETUP/PARSING FUNCTIONS ####################################

const date_format = "yyyy-mm-dd HH:MM:SS"


"""
    parse_resolving(z)

Read in the resolving_mrs.csv configuration file to create a cubic spline interpolation of the
MIRI MRS resolving power as a function of wavelength, redshifted to the rest frame of the object
being fit.

# Arguments
- `z::AbstractFloat`: The redshift of the object to be fit
"""
function parse_resolving(z::AbstractFloat)::Dierckx.Spline1D

    @debug "Parsing MRS resoling power from resolving_mrs.csv"

    # Read in the resolving power data
    resolve = CSV.read(joinpath(@__DIR__, "resolving_mrs.csv"), DataFrame)

    # Find points where wavelength jumps down (b/w channels)
    jumps = diff(resolve[!, :wave]) .< 0

    # Define regions of overlapping wavelength space
    wave_left = resolve[BitVector([0; jumps]), :wave]
    wave_right = resolve[BitVector([jumps; 0]), :wave]

    # Smooth the data in overlapping regions
    for i ∈ 1:sum(jumps)
        region = wave_left[i] .< resolve[!, :wave] .< wave_right[i]
        resolve[region, :R] .= movmean(resolve[!, :R], 5)[region]
    end
    # Sort the data to be monotonically increasing in wavelength
    ss = sortperm(resolve[!, :wave])
    wave = resolve[ss, :wave]
    R = resolve[ss, :R]

    # Shift to the rest frame
    wave = Util.rest_frame(wave, z)

    # Define coarse knots
    knots = (wave[1]+0.25):0.25:(wave[end]-0.25)

    # Create an interpolation function so we can evaluate it at the points of interest for our data
    interp_R = Spline1D(wave, R, knots)
    
    return interp_R
end


"""
    parse_options()

Read in the options.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary.  This deals with general/top-level code configurations.
"""
function parse_options()::Dict

    @debug """\n
    Parsing options file
    #######################################################
    """

    # Read in the options file
    options = TOML.parsefile(joinpath(@__DIR__, "options.toml"))
    options_out = Dict()
    keylist1 = ["extinction_curve", "extinction_screen", "chi2_threshold", "overwrite", "cosmology"]
    keylist2 = ["h", "omega_m", "omega_K", "omega_r"]
    # Loop through the keys that should be in the file and confirm that they are there
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
    
    # Set the keys in the options file to our output dictionary
    options_out[:extinction_curve] = options["extinction_curve"]
    @debug "Extinction model - $(options["extinction_curve"])"
    options_out[:extinction_screen] = options["extinction_screen"]
    @debug "Extinction screening? - $(options["extinction_screen"])"
    options_out[:chi2_threshold] = options["chi2_threshold"]
    @debug "Reduced chi^2 threshold - $(options["chi2_threshold"])"
    options_out[:overwrite] = options["overwrite"]
    @debug "Overwrite old fits? - $(options["overwrite"])"

    # Convert cosmology keys into a proper cosmology object
    options_out[:cosmology] = cosmology(h=options["cosmology"]["h"], 
                                        OmegaM=options["cosmology"]["omega_m"],
                                        OmegaK=options["cosmology"]["omega_K"],
                                        OmegaR=options["cosmology"]["omega_r"])
    @debug """\n
    Cosmology:
    h      = $(options["cosmology"]["h"])
    OmegaM = $(options["cosmology"]["omega_m"])
    OmegaK = $(options["cosmology"]["omega_K"])
    OmegaR = $(options["cosmology"]["omega_r"])
    """

    return options_out
end


"""
    parse_dust()

Read in the dust.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary with Parameter objects for dust fitting parameters.
This deals with continuum, PAH features, and extinction options.
"""
function parse_dust()::Dict

    @debug """\n
    Parsing dust file
    #######################################################
    """

    # Read in the dust file
    dust = TOML.parsefile(joinpath(@__DIR__, "dust.toml"))
    dust_out = Dict()
    keylist1 = ["stellar_continuum_temp", "dust_continuum_temps", "dust_features", "extinction"]
    keylist2 = ["wave", "fwhm"]
    keylist3 = ["tau_9_7", "beta"]
    keylist4 = ["val", "plim", "locked"]
    # Loop through all of the keys that should be in the file and confirm that they are there
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

    # Convert the options into Parameter objects, and set them to the output dictionary
    dust_out[:stellar_continuum_temp] = Param.from_dict(dust["stellar_continuum_temp"])
    @debug "Stellar continuum:\nTemp $(dust_out[:stellar_continuum_temp])"

    dust_out[:dust_continuum_temps] = [Param.from_dict(dust["dust_continuum_temps"][i]) for i ∈ 1:length(dust["dust_continuum_temps"])]
    msg = "Dust continuum:"
    for dc ∈ dust_out[:dust_continuum_temps]
        msg *= "\nTemp $dc"
    end
    @debug msg
    
    dust_out[:dust_features] = Dict()
    msg = "Dust features:"
    for df ∈ keys(dust["dust_features"])
        dust_out[:dust_features][df] = Dict()
        dust_out[:dust_features][df][:complex] = "complex" ∈ keys(dust["dust_features"][df]) ? dust["dust_features"][df]["complex"] : split(df, "_")[2]
        msg *= "\nComplex - $(dust_out[:dust_features][df][:complex])"
        dust_out[:dust_features][df][:wave] = Param.from_dict_wave(dust["dust_features"][df]["wave"])
        msg *= "\nWave $(dust_out[:dust_features][df][:wave])"
        dust_out[:dust_features][df][:fwhm] = Param.from_dict_fwhm(dust["dust_features"][df]["fwhm"])
        msg *= "\nFWHM $(dust_out[:dust_features][df][:fwhm])"
    end
    @debug msg

    dust_out[:extinction] = Param.ParamDict()
    msg = "Extinction:"
    dust_out[:extinction][:tau_9_7] = Param.from_dict(dust["extinction"]["tau_9_7"])
    msg *= "\nTau $(dust_out[:extinction][:tau_9_7])"
    dust_out[:extinction][:beta] = Param.from_dict(dust["extinction"]["beta"])
    msg *= "\nBeta $(dust_out[:extinction][:beta])"
    @debug msg

    return dust_out
end


"""
    parse_lines(channel, interp_R, λ)

Read in the lines.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary with Parameter objects for line fitting parameters.
This deals purely with emission line options.

# Arguments
- `channel::Integer`: The MIRI channel that is being fit
- `interp_R::Dierckx.Spline1D`: The MRS resolving power interpolation function, as a function of rest frame wavelength
- `λ::Vector{<:AbstractFloat}`: The rest frame wavelength vector of the spectrum being fit
"""
function parse_lines(channel::Integer, interp_R::Dierckx.Spline1D, λ::Vector{<:AbstractFloat})

    @debug """\n
    Parsing lines file
    #######################################################
    """

    # Read in the lines file
    lines = TOML.parsefile(joinpath(@__DIR__, "lines.toml"))
    lines_out = Param.LineDict()

    keylist1 = ["tie_H2_voff", "tie_IP_voff", "tie_H2_flow_voff", "tie_IP_flow_voff", "tie_voigt_mixing", 
        "voff_plim", "fwhm_pmax", "h3_plim", "h4_plim", 
        "flexible_wavesol", "wavesol_unc", "channels", "lines", "profiles", "flows"]
    # Loop through all the keys that should be in the file and confirm that they are there
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

    # Minimum possible FWHM of a narrow line given the instrumental resolution of MIRI 
    # in the given wavelength range
    fwhm_pmin = Util.C_KMS / maximum(interp_R.(λ))
    @debug "Setting minimum FWHM to $fwhm_pmin km/s"

    # Define the initial values of line parameters given the values in the options file (if present)
    fwhm_init = "fwhm_init" ∈ keys(lines) ? lines["fwhm_init"] : maximum([fwhm_pmin + 1, 100])
    voff_init = "voff_init" ∈ keys(lines) ? lines["voff_init"] : 0.0
    h3_init = "h3_init" ∈ keys(lines) ? lines["h3_init"] : 0.0    # gauss-hermite series start fully gaussian,
    h4_init = "h4_init" ∈ keys(lines) ? lines["h4_init"] : 0.0    # with both h3 and h4 moments starting at 0
    η_init = "η_init" ∈ keys(lines) ? lines["eta_init"] : 1.0     # Voigts start fully gaussian

    # Loop through all the lines
    for line ∈ keys(lines["lines"])

        @debug """\n
        ################# $line #######################
        # Rest wavelength: $(lines["lines"][line]) um #
        """

        # Set the priors for FWHM, voff, h3, h4, and eta based on the values in the options file
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

        # Set the priors for inflow/outflow FWHM, voff, h3, h4, and eta based on the values in the options file
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

        # Check if there are any specific override values present in the options file,
        # and if so, use them
        if haskey(lines, "priors")
            if haskey(lines["priors"], line)
                if haskey(lines["priors"][line], "voff")
                    @debug "Overriding voff prior"
                    voff_prior = eval(Meta.parse(lines["priors"][line]["voff"]["pstr"]))
                    voff_locked = lines["priors"][line]["voff"]["locked"]
                end
                if haskey(lines["priors"][line], "fwhm")
                    @debug "Overriding fwhm prior"
                    fwhm_prior = eval(Meta.parse(lines["priors"][line]["fwhm"]["pstr"]))
                    fwhm_locked = lines["priors"][line]["fwhm"]["locked"]
                end
                if haskey(lines["priors"][line], "h3")
                    @debug "Overriding h3 prior"
                    h3_prior = eval(Meta.parse(lines["priors"][line]["h3"]["pstr"]))
                    h3_locked = lines["priors"][line]["h3"]["locked"]
                end
                if haskey(lines["priors"][line], "h4")
                    @debug "Overriding h4 prior"
                    h4_prior = eval(Meta.parse(lines["priors"][line]["h4"]["pstr"]))
                    h4_locked = lines["priors"][line]["h4"]["locked"]
                end
                if haskey(lines["priors"][line], "eta")
                    @debug "Overriding eta prior"
                    η_prior = eval(Meta.parse(lines["priors"][line]["eta"]["pstr"]))
                    η_locked = lines["priors"][line]["eta"]["locked"]
                end

                if haskey(lines["priors"][line], "flow_voff")
                    @debug "Overriding flow voff prior"
                    flow_voff_prior = eval(Meta.parse(lines["priors"][line]["flow_voff"]["pstr"]))
                    flow_voff_locked = lines["priors"][line]["flow_voff"]["locked"]
                end
                if haskey(lines["priors"][line], "flow_fwhm")
                    @debug "Overriding flow fwhm prior"
                    flow_fwhm_prior = eval(Meta.parse(lines["priors"][line]["flow_fwhm"]["pstr"]))
                    flow_fwhm_locked = lines["priors"][line]["flow_fwhm"]["locked"]
                end
                if haskey(lines["priors"][line], "flow_h3")
                    @debug "Overriding flow h3 prior"
                    flow_h3_prior = eval(Meta.parse(lines["priors"][line]["flow_h3"]["pstr"]))
                    flow_h3_locked = lines["priors"][line]["flow_h3"]["locked"]
                end
                if haskey(lines["priors"][line], "flow_h4")
                    @debug "Overriding flow h4 prior"
                    flow_h4_prior = eval(Meta.parse(lines["priors"][line]["flow_h4"]["pstr"]))
                    flow_h4_locked = lines["priors"][line]["flow_h4"]["locked"]
                end
                if haskey(lines["priors"][line], "flow_eta")
                    @debug "Overriding flow eta prior"
                    flow_η_prior = eval(Meta.parse(lines["priors"][line]["flow_eta"]["pstr"]))
                    flow_η_locked = lines["priors"][line]["flow_eta"]["locked"]
                end
            end
        end

        # Check if the voff should be tied to other voffs based on the line type (H2 or IP)
        tied = nothing
        if lines["tie_H2_voff"] && occursin("H2", line)
            # String representing what lines are tied together, will be the same for all tied lines
            # and `nothing` for untied lines
            tied = "H2"
            @debug "Tying voff to the group: $tied"
            # If the wavelength solution is bad, allow the voff to still be flexible based on 
            # the accuracy of it
            if lines["flexible_wavesol"]
                δv = lines["wavesol_unc"][channel]
                voff_prior = Uniform(-δv, δv)
                @debug "(Using flexible tied voff with lenience of +/-$δv km/s)"
            end
        end
        # Check if the outflow voff should be tied to other outflow voffs based on the line type (H2 or IP)
        flow_tied = nothing
        if lines["tie_H2_flow_voff"] && occursin("H2", line) && !isnothing(flow_profiles[line])
            # String representing what lines are tied together, will be the same for all tied lines
            # and `nothing` for untied lines
            flow_tied = "H2"
            @debug "Tying flow voff to the group: $flow_tied"
            # dont allow tied inflow/outflow voffs to vary, even with flexible_wavesol
        end

        @debug "Profile: $(profiles[line])"

        # Create parameter objects using the priors
        voff = Param.Parameter(voff_init, voff_locked, voff_prior)
        @debug "Voff $voff"
        fwhm = Param.Parameter(fwhm_init, fwhm_locked, fwhm_prior)
        @debug "FWHM $fwhm"
        if profiles[line] ∈ ("Gaussian", "Lorentzian")
            params = Param.ParamDict(:voff => voff, :fwhm => fwhm)
        elseif profiles[line] == "GaussHermite"
            h3 = Param.Parameter(h3_init, h3_locked, h3_prior)
            @debug "h3 $h3"
            h4 = Param.Parameter(h4_init, h4_locked, h4_prior)
            @debug "h4 $h4"
            params = Param.ParamDict(:voff => voff, :fwhm => fwhm, :h3 => h3, :h4 => h4)
        elseif profiles[line] == "Voigt"
            η = Param.Parameter(η_init, η_locked, η_prior)
            @debug "eta $η"
            params = Param.ParamDict(:voff => voff, :fwhm => fwhm, :mixing => η)
        end
        # Do the same for the inflow/outflow parameters, but only if the line has an inflow/outflow component
        if !isnothing(flow_profiles[line])
            @debug "Flow profile: $(flow_profiles[line])"

            flow_voff = Param.Parameter(voff_init, flow_voff_locked, flow_voff_prior)
            @debug "Voff $flow_voff"
            flow_fwhm = Param.Parameter(fwhm_init, flow_fwhm_locked, flow_fwhm_prior)
            @debug "FWHM $flow_fwhm"
            if flow_profiles[line] ∈ ("Gaussian", "Lorentzian")
                flow_params = Param.ParamDict(:flow_voff => flow_voff, :flow_fwhm => flow_fwhm)
            elseif flow_profiles[line] == "GaussHermite"
                flow_h3 = Param.Parameter(h3_init, flow_h3_locked, flow_h3_prior)
                @debug "h3 $flow_h3"
                flow_h4 = Param.Parameter(h4_init, flow_h4_locked, flow_h4_prior)
                @debug "h4 $flow_h4"
                flow_params = Param.ParamDict(:flow_voff => flow_voff, :flow_fwhm => flow_fwhm, :flow_h3 => flow_h3, :flow_h4 => flow_h4)
            elseif flow_profiles[line] == "Voigt"
                flow_η = Param.Parameter(η_init, flow_η_locked, flow_η_prior)
                @debug "eta $flow_η"
                flow_params = Param.ParamDict(:flow_voff => flow_voff, :flow_fwhm => flow_fwhm, :flow_mixing => flow_η)
            end
            params = merge(params, flow_params)
        end

        # Create the TransitionLine object using the above parameters, and add it to the dictionary
        lines_out[Symbol(line)] = Param.TransitionLine(lines["lines"][line], 
            Symbol(profiles[line]), !isnothing(flow_profiles[line]) ? Symbol(flow_profiles[line]) : nothing, params, tied, flow_tied)

    end


    @debug "#######################################################"

    # Create a dictionary containing all of the unique `tie` keys, and the tied voff parameters 
    # corresponding to that tied key
    voff_tied_key = unique([lines_out[line].tied for line ∈ keys(lines_out)])
    voff_tied_key = voff_tied_key[.!isnothing.(voff_tied_key)]
    @debug "voff_tied_key: $voff_tied_key"

    voff_tied = Dict{String, Param.Parameter}()
    msg = ""
    for (i, voff_tie) ∈ enumerate(voff_tied_key)
        prior = Uniform(lines["voff_plim"]...)
        locked = false
        # Check if there is an overwrite option in the lines file
        if haskey(lines, "priors")
            if haskey(lines["priors"], voff_tie)
                prior = lines["priors"][voff_tie]["pstr"]
                locked = lines["priors"][voff_tie]["locked"]
            end
        end
        voff_tied[voff_tie] = Param.Parameter(voff_init, locked, prior)
        msg *= "\nvoff_tied_$i $(voff_tied[voff_tie])"
    end
    @debug msg

    # ^^^ do the same for the inflow/outflow tied velocities \/\/\/
    flow_voff_tied_key = unique([lines_out[line].flow_tied for line ∈ keys(lines_out)])
    flow_voff_tied_key = flow_voff_tied_key[.!isnothing.(flow_voff_tied_key)]
    @debug "flow_voff_tied_key: $flow_voff_tied_key"

    flow_voff_tied = Dict{String, Param.Parameter}()
    msg = ""
    for (j, flow_voff_tie) ∈ enumerate(flow_voff_tied_key)
        flow_prior = Uniform(lines["flow_voff_plim"]...)
        flow_locked = false
        # Check if there is an overwrite option in the lines file
        if haskey(lines, "priors")
            if haskey(lines["priors"], flow_voff_tie)
                flow_prior = lines["priors"][flow_voff_tie]["pstr"]
                flow_locked = lines["priors"][flow_voff_tie]["locked"]
            end
        end
        flow_voff_tied[flow_voff_tie] = Param.Parameter(voff_init, flow_locked, flow_prior)
        msg *= "\nflow_voff_tied_$j $(flow_voff_tied[flow_voff_tie])"
    end
    @debug msg

    # If tie_voigt_mixing is set, all Voigt profiles have the same tied mixing parameter eta
    if lines["tie_voigt_mixing"]
        voigt_mix_tied = Param.Parameter(η_init, false, Uniform(0.0, 1.0))
        @debug "voigt_mix_tied $voigt_mix_tied (tied)"
    else
        voigt_mix_tied = nothing
        @debug "voigt_mix_tied is $voigt_mix_tied (untied)"
    end

    return lines_out, voff_tied, flow_voff_tied, lines["flexible_wavesol"], lines["tie_voigt_mixing"], voigt_mix_tied
end

############################## PARAMETER / MODEL STRUCTURES ####################################

"""
    ParamMaps(stellar_continuum, dust_continuum, dust_features, lines, tied_voffs, flow_tied_voffs,
        tied_voigt_mix, extinction, dust_complexes, reduced_χ2)

A structure for holding 2D maps of fitting parameters generated when fitting a cube.

See ['parammaps_empty`](@ref) for a default constructor function.
"""
struct ParamMaps

    stellar_continuum::Dict{Symbol, Array{Float64, 2}}
    dust_continuum::Dict{Int, Dict{Symbol, Array{Float64, 2}}}
    dust_features::Dict{String, Dict{Symbol, Array{Float64, 2}}}
    lines::Dict{Symbol, Dict{Symbol, Array{Float64, 2}}}
    tied_voffs::Dict{String, Array{Float64, 2}}
    flow_tied_voffs::Dict{String, Array{Float64, 2}}
    tied_voigt_mix::Union{Array{Float64, 2}, Nothing}
    extinction::Dict{Symbol, Array{Float64, 2}}
    dust_complexes::Dict{String, Dict{Symbol, Array{Float64, 2}}}
    reduced_χ2::Array{Float64, 2}

end

"""
    parammaps_empty(shape, n_dust_cont, df_names, complexes, line_names, line_tied, line_profiles,
        line_flow_tied, line_flow_profiles, voff_tied_key, flow_voff_tied_key, flexible_wavesol, 
        tie_voigt_mixing)

A constructor function for making a default empty ParamMaps structure with all necessary fields for a given
fit of a DataCube.

# Arguments
`S<:Integer`
- `shape::Tuple{S,S,S}`: The dimensions of the DataCube being fit, formatted as a tuple of (nx, ny, nz)
- `n_dust_cont::Integer`: The number of dust continuum components in the fit (usually given by the number of temperatures 
    specified in the dust.toml file)
- `df_names::Vector{String}`: List of names of PAH features being fit, i.e. "PAH_12.62", ...
- `complexes::Vector{String}`: List of names of dust complexes being fit; similar to PAH features, but some of them may 
    contain multiple PAH features combined close together, while others may be isolated, i.e. "12.7", ...
- `line_names::Vector{Symbol}`: List of names of lines being fit, i.e. "NeVI_7652", ...
- `line_tied::Vector{Union{String,Nothing}}`: List of line tie keys which specify whether the voff of the given line should be
    tied to other lines. The line tie key itself may be either `nothing` (untied), or a String specifying the group of lines
    being tied, i.e. "H2"
- `line_profiles::Vector{String}`: List of the type of profile to use to fit each emission line. Each entry may be any of
    "Gaussian", "Lorentzian", "GaussHermite", or "Voigt"
- `line_flow_tied::Vector{Union{String,Nothing}}`: Same as line_tied, but for inflow/outflow line components
- `line_flow_profiles::Vector{Union{String,Nothing}}`: Same as line_profiles, but for inflow/outflow line components. An element
    may be `nothing` if the line in question has no inflow/outflow component.
- `voff_tied_key::Vector{String}`: List of only the unique keys in line_tied (not including `nothing`)
- `flow_voff_tied_key::Vector{String}`: Same as voff_tied_key, but for inflow/outflow line components
- `flexible_wavesol::Bool`: Fitting option on whether to allow a small variation in voff components even when they are tied,
    in case the wavelength solution of the data is not calibrated very well
- `tie_voigt_mixing::Bool`: Whether or not to tie the mixing parameters of any Voigt line profiles
"""
function parammaps_empty(shape::Tuple{S,S,S}, n_dust_cont::Integer, df_names::Vector{String}, 
    complexes::Vector{String}, line_names::Vector{Symbol}, line_tied::Vector{Union{String,Nothing}},
    line_profiles::Vector{Symbol}, line_flow_tied::Vector{Union{String,Nothing}}, line_flow_profiles::Vector{Union{Symbol,Nothing}},
    voff_tied_key::Vector{String}, flow_voff_tied_key::Vector{String}, flexible_wavesol::Bool, 
    tie_voigt_mixing::Bool)::ParamMaps where {S<:Integer}

    @debug """\n
    Creating ParamMaps struct with shape $shape
    ###########################################
    """

    # Initialize a default array of nans to be used as a placeholder for all the other arrays
    # until the actual fitting parameters are obtained
    nan_arr = ones(shape[1:2]...) .* NaN

    # Add stellar continuum fitting parameters
    stellar_continuum = Dict{Symbol, Array{Float64, 2}}()
    stellar_continuum[:amp] = copy(nan_arr)
    stellar_continuum[:temp] = copy(nan_arr)
    @debug "stellar continuum maps with keys $(keys(stellar_continuum))"

    # Add dust continuum fitting parameters
    dust_continuum = Dict{Int, Dict{Symbol, Array{Float64, 2}}}()
    for i ∈ 1:n_dust_cont
        dust_continuum[i] = Dict{Symbol, Array{Float64, 2}}()
        dust_continuum[i][:amp] = copy(nan_arr)
        dust_continuum[i][:temp] = copy(nan_arr)
        @debug "dust continuum $i maps with keys $(keys(dust_continuum[i]))"
    end

    # Add dust features fitting parameters
    dust_features = Dict{String, Dict{Symbol, Array{Float64, 2}}}()
    for n ∈ df_names
        dust_features[n] = Dict{Symbol, Array{Float64, 2}}()
        dust_features[n][:amp] = copy(nan_arr)
        dust_features[n][:mean] = copy(nan_arr)
        dust_features[n][:fwhm] = copy(nan_arr)
        @debug "dust feature $n maps with keys $(keys(dust_features[n]))"
    end

    # Nested dictionary -> first layer keys are line names, second layer keys are parameter names, which contain 2D arrays
    lines = Dict{Symbol, Dict{Symbol, Array{Float64, 2}}}()
    for (line, tie, prof, flowtie, flowprof) ∈ zip(line_names, line_tied, line_profiles, line_flow_tied, line_flow_profiles)
        lines[line] = Dict{Symbol, Array{Float64, 2}}()
        # If tied and NOT using a flexible solution, don't include a voff parameter
        pnames = isnothing(tie) || flexible_wavesol ? [:amp, :voff, :fwhm] : [:amp, :fwhm]
        # Add 3rd and 4th order moments (skewness and kurtosis) for Gauss-Hermite profiles
        if prof == :GaussHermite
            pnames = [pnames; :h3; :h4]
        # Add mixing parameter for Voigt profiles, but only if NOT tying it
        elseif prof == :Voigt && !tie_voigt_mixing
            pnames = [pnames; :mixing]
        end
        # Repeat the above but for inflow/outflow components
        if !isnothing(flowprof)
            pnames = isnothing(flowtie) ? [pnames; :flow_amp; :flow_voff; :flow_fwhm] : [pnames; :flow_amp; :flow_fwhm]
            if flowprof == :GaussHermite
                pnames = [pnames; :flow_h3; :flow_h4]
            elseif flowprof == :Voigt && !tie_voigt_mixing
                pnames = [pnames; :flow_mixing]
            end
        end
        # Append parameters for intensity and signal-to-noise ratio, which are NOT fitting parameters, but are of interest
        pnames = [pnames; :intI; :SNR]
        for pname ∈ pnames
            lines[line][pname] = copy(nan_arr)
        end
        @debug "line $line maps with keys $pnames"
    end

    # Tied voff parameters
    tied_voffs = Dict{String, Array{Float64, 2}}()
    for vk ∈ voff_tied_key
        tied_voffs[vk] = copy(nan_arr)
        @debug "tied voff map for group $vk"
    end

    # Tied inflow/outflow voff parameters
    flow_tied_voffs = Dict{String, Array{Float64, 2}}()
    for fvk ∈ flow_voff_tied_key
        flow_tied_voffs[fvk] = copy(nan_arr)
        @debug "tied flow voff map for group $fvk"
    end

    # Tied voigt mixing ratio parameter, if appropriate
    if tie_voigt_mixing
        tied_voigt_mix = copy(nan_arr)
        @debug "tied voigt mixing map"
    else
        tied_voigt_mix = nothing
    end

    # Add extinction fitting parameters
    extinction = Dict{Symbol, Array{Float64, 2}}()
    extinction[:tau_9_7] = copy(nan_arr)
    extinction[:beta] = copy(nan_arr)
    @debug "extinction maps with keys $(keys(extinction))"

    # Add dusts complexes with extrapolated parameters
    dust_complexes = Dict{String, Dict{Symbol, Array{Float64, 2}}}()
    for c ∈ complexes
        dust_complexes[c] = Dict{Symbol, Array{Float64, 2}}()
        # Again, intensity and S/N are not fitted parameters, but are of interest to save
        dust_complexes[c][:intI] = copy(nan_arr)
        dust_complexes[c][:SNR] = copy(nan_arr)
        @debug "dust complex maps $c with keys $(keys(dust_complexes[c]))"
    end

    # Reduced chi^2 of the fits
    reduced_χ2 = copy(nan_arr)
    @debug "reduced chi^2 map"

    return ParamMaps(stellar_continuum, dust_continuum, dust_features, lines, tied_voffs, flow_tied_voffs,
        tied_voigt_mix, extinction, dust_complexes, reduced_χ2)
end


"""
    CubeModel(model, stellar, dust_continuum, dust_features, extinction, lines)

A structure for holding 3D models of intensity, split up into model components, generated when fitting a cube

See [`cubemodel_empty`](@ref) for a default constructor method.
"""
struct CubeModel{T<:AbstractFloat}

    model::Array{T, 3}
    stellar::Array{T, 3}
    dust_continuum::Array{T, 4}
    dust_features::Array{T, 4}
    extinction::Array{T, 3}
    lines::Array{T, 4}

end


"""
    cubemodel_empty(shape, n_dust_cont, df_names, line_names; floattype=floattype)

A constructor function for making a default empty CubeModel object with all the necessary fields for a given
fit of a DataCube.

# Arguments
`S<:Integer`
- `shape::Tuple{S,S,S}`: The dimensions of the DataCube being fit, formatted as a tuple of (nx, ny, nz)
- `n_dust_cont::Integer`: The number of dust continuum components in the fit (usually given by the number of temperatures 
    specified in the dust.toml file)
- `df_names::Vector{String}`: List of names of PAH features being fit, i.e. "PAH_12.62", ...
- `line_names::Vector{Symbol}`: List of names of lines being fit, i.e. "NeVI_7652", ...
- `floattype::DataType=Float32`: The type of float to use in the arrays.
"""
function cubemodel_empty(shape::Tuple{S,S,S}, n_dust_cont::Integer, df_names::Vector{String}, 
    line_names::Vector{Symbol}, floattype::DataType=Float32)::CubeModel where {S<:Integer}

    @debug """\n
    Creating CubeModel struct with shape $shape
    ###########################################
    """

    # Make sure the floattype given is actually a type of float
    @assert floattype <: AbstractFloat

    # Initialize the arrays for each part of the full 3D model
    model = zeros(floattype, shape...)
    @debug "model cube"
    stellar = zeros(floattype, shape...)
    @debug "stellar continuum comp cube"
    dust_continuum = zeros(floattype, shape..., n_dust_cont)
    @debug "dust continuum comp cubes"
    dust_features = zeros(floattype, shape..., length(df_names))
    @debug "dust features comp cubes"
    extinction = zeros(floattype, shape...)
    @debug "extinction comp cube"
    lines = zeros(floattype, shape..., length(line_names))
    @debug "lines comp cubes"

    return CubeModel(model, stellar, dust_continuum, dust_features, extinction, lines)
end


"""
    CubeFitter(cube, z, name, n_procs; window_size=window_size, plot_spaxels=plot_spaxels, 
        plot_maps=plot_maps, parallel=parallel, save_fits=save_fits)

This is the main structure used for fitting IFU cubes, containing all of the necessary data, metadata,
fitting options, and its own instances of ParamMaps and CubeModel structures to handle the outputs of all the fits.  
The actual fitting functions (`fit_spaxel` and `fit_cube`) require an instance of this structure.

# Fields
`T<:AbstractFloat, S<:Integer`
- `cube::CubeData.DataCube`: The main DataCube object containing the cube that is being fit
- `z::AbstractFloat`: The redshift of the target that is being fit
- `n_procs::Integer`: The number of parallel processes that are being used in the fitting procedure
- `cube_model::CubeModel`: The CubeModel object containing 3D arrays to be populated with the full best-fit models
- `param_maps::ParamMaps`: The ParamMaps object containing 2D arrays to be populated with the best-fit model parameters
- `window_size::AbstractFloat=.025`: The window size (currently a deprecated parameter)
- `plot_spaxels::Symbol=:pyplot`: A Symbol specifying the plotting backend to be used when plotting individual spaxel fits, can
    be either `:pyplot` or `:plotly`
- `plot_maps::Bool=true`: Whether or not to plot 2D maps of the best-fit parameters after the fitting is finished
- `parallel::Bool=true`: Whether or not to fit multiple spaxels in parallel using multiprocessing
- `save_fits::Bool=true`: Whether or not to save the final best-fit models and parameters as FITS files
Read from the options files:
- `overwrite::Bool`: Whether or not to overwrite old fits of spaxels when rerunning
- `extinction_curve::String`: The type of extinction curve being used, either `"kvt"` or `"d+"`
- `extinction_screen::Bool`: Whether or not the extinction is modeled as a screen
- `T_s::Param.Parameter`: The stellar temperature parameter
- `T_dc::Vector{Param.Parameter}`: The dust continuum temperature parameters
- `τ_97::Param.Parameter`: The dust opacity at 9.7 um parameter
- `β::Param.Parameter`: The extinction profile mixing parameter
- `n_dust_cont::Integer`: The number of dust continuum profiles
- `df_names::Vector{String}`: The names of each PAH feature profile
- `dust_features::Vector{Dict}`: All of the fitting parameters for each PAH feature
- `n_lines::Integer`: The number of lines being fit
- `line_names::Vector{Symbol}`: The names of each line being fit
- `line_profiles::Vector{Symbol}`: The profiles of each line being fit
- `line_flow_profiles::Vector{Union{Nothing,Symbol}}`: Same as `line_profiles`, but for the inflow/outflow components
- `lines::Vector{Param.TransitionLine}`: All of the fitting parameters for each line
- `n_voff_tied::Integer`: The number of tied velocity offsets
- `line_tied::Vector{Union{String,Nothing}}`: List of line tie keys which specify whether the voff of the given line should be
    tied to other lines. The line tie key itself may be either `nothing` (untied), or a String specifying the group of lines
    being tied, i.e. "H2"
- `voff_tied_key::Vector{String}`: List of only the unique keys in line_tied (not including `nothing`)
- `voff_tied::Vector{Param.Parameter}`: The actual tied voff parameter objects, corresponding to the `voff_tied_key`
- `n_flow_voff_tied::Integer`: Same as `n_voff_tied`, but for inflow/outflow components
- `line_flow_tied::Vector{Union{String,Nothing}}`: Same as `line_tied`, but for inflow/outflow components
- `flow_voff_tied_key::Vector{String}`: Same as `voff_tied_key`, but for inflow/outflow components
- `flow_voff_tied::Vector{Param.Parameter}`: Same as `voff_tied`, but for inflow/outflow components
- `tie_voigt_mixing::Bool`: Whether or not the Voigt mixing parameter is tied between all the lines with Voigt profiles
- `voigt_mix_tied::Param.Parameter`: The actual tied Voigt mixing parameter object, given `tie_voigt_mixing` is true
- `n_complexes::Integer`: The number of PAH complexes in the fitting region
- `complexes::Vector{String}`: The list of names of the PAH complexes in the fitting region
- `n_params_cont::Integer`: The total number of free fitting parameters for the continuum fit (not including emission lines)
- `n_params_lines::Integer`: The total number of free fitting parameters for the emission line fit (not including the continuum)
- `cosmology::Cosmology.AbstractCosmology`: The Cosmology, used solely to create physical scale bars on the 2D parameter plots
- `χ²_thresh::AbstractFloat`: The threshold for reduced χ² values, below which the best fit parameters for a given
    row will be set
- `interp_R::Union{Function,Dierckx.Spline1D}`: Interpolation function for the instrumental resolution as a function of wavelength
- `flexible_wavesol::Bool`: Whether or not to allow small variations in the velocity offsets even when tied, to account
    for a bad wavelength solution
- `p_best_cont::SharedArray{T}`: A rolling collection of best fit continuum parameters for the best fitting spaxels
    along each row, for fits with a reduced χ² below χ²_thresh, which are used for the starting parameters in the following
    fits for the given row
- `p_best_line::SharedArray{T}`: Same as `p_best_cont`, but for the line parameters
- `χ²_best::SharedVector{T}`: The reduced χ² values associated with the `p_best_cont` and `p_best_line` values
    in each row
- `best_spaxel::SharedVector{Tuple{S,S}}`: The locations of the spaxels associated with the `p_best_cont` and `p_best_line`
    values in each row

See [`ParamMaps`](@ref), [`parammaps_empty`](@ref), [`CubeModel`](@ref), [`cubemodel_empty`](@ref), 
    [`fit_spaxel`](@ref), [`fit_cube`](@ref)
"""
mutable struct CubeFitter{T<:AbstractFloat,S<:Integer}
    
    # Data
    cube::CubeData.DataCube
    z::AbstractFloat
    name::String

    # Basic fitting options
    n_procs::Integer
    cube_model::CubeModel
    param_maps::ParamMaps
    window_size::AbstractFloat
    plot_spaxels::Symbol
    plot_maps::Bool
    parallel::Bool
    save_fits::Bool
    overwrite::Bool
    extinction_curve::String
    extinction_screen::Bool

    # Continuum parameters
    T_s::Param.Parameter
    T_dc::Vector{Param.Parameter}
    τ_97::Param.Parameter
    β::Param.Parameter
    n_dust_cont::Integer
    n_dust_feat::Integer
    df_names::Vector{String}
    dust_features::Vector{Dict}

    # Line parameters
    n_lines::Integer
    line_names::Vector{Symbol}
    line_profiles::Vector{Symbol}
    line_flow_profiles::Vector{Union{Nothing,Symbol}}
    lines::Vector{Param.TransitionLine}

    # Tied voffs
    n_voff_tied::Integer
    line_tied::Vector{Union{String,Nothing}}
    voff_tied_key::Vector{String}
    voff_tied::Vector{Param.Parameter}

    # Tied inflow/outflow voffs
    n_flow_voff_tied::Integer
    line_flow_tied::Vector{Union{String,Nothing}}
    flow_voff_tied_key::Vector{String}
    flow_voff_tied::Vector{Param.Parameter}

    # Tied voigt mixing
    tie_voigt_mixing::Bool
    voigt_mix_tied::Param.Parameter

    # Dust complexes
    n_complexes::Integer
    complexes::Vector{String}
    n_params_cont::Integer
    n_params_lines::Integer
    
    # Rolling best fit options
    cosmology::Cosmology.AbstractCosmology
    χ²_thresh::AbstractFloat
    interp_R::Union{Function,Dierckx.Spline1D}
    flexible_wavesol::Bool

    p_best_cont::SharedArray{T}
    p_best_line::SharedArray{T}
    χ²_best::SharedVector{T}
    best_spaxel::SharedVector{Tuple{S,S}}

    # Constructor function
    function CubeFitter(cube::CubeData.DataCube, z::Float64, name::String, n_procs::Int; window_size::Float64=.025, 
        plot_spaxels::Symbol=:pyplot, plot_maps::Bool=true, parallel::Bool=true, save_fits::Bool=true)

        @debug """\n
        Creating CubeFitter struct for $name
        ####################################
        """

        # Get shape
        shape = size(cube.Iλ)
        # Alias
        λ = cube.λ

        # Parse all of the options files to create default options and parameter objects
        interp_R = parse_resolving(z)
        dust = parse_dust() 
        options = parse_options()
        line_list, voff_tied, flow_voff_tied, flexible_wavesol, tie_voigt_mixing, 
            voigt_mix_tied = parse_lines(parse(Int, cube.channel), interp_R, λ)

        # Check that number of processes doesn't exceed first dimension, so the rolling best fit can work as intended
        if n_procs > shape[1]
            error("Number of processes ($n_procs) must be ≤ the size of the first cube dimension ($(shape[1]))!")
        end

        T_s = dust[:stellar_continuum_temp]
        T_dc = dust[:dust_continuum_temps]
        τ_97 = dust[:extinction][:tau_9_7]
        β = dust[:extinction][:beta]

        @debug "### Model will include 1 stellar continuum component ###" *
             "\n### at T = $(T_s.value) K ###"

        #### PREPARE OUTPUTS ####
        n_dust_cont = length(T_dc)
        msg = "### Model will include $n_dust_cont dust continuum components ###"
        for T_dci ∈ T_dc
            msg *= "\n### at T = $(T_dci.value) K ###"
        end
        @debug msg 

        # Only use PAH features within +/-0.5 um of the region being fitting (to include wide tails)
        df_filt = [(minimum(λ)-1//2 < dust[:dust_features][df][:wave].value < maximum(λ)+1//2) for df ∈ keys(dust[:dust_features])]
        df_names = Vector{String}(collect(keys(dust[:dust_features]))[df_filt])
        df_mean = [parse(Float64, split(df, "_")[2]) for df ∈ df_names]
        # Sort by the wavelength values since dictionaries are not sorted by default
        ss = sortperm(df_mean)
        df_names = df_names[ss]
        dust_features = [dust[:dust_features][df] for df ∈ df_names]
        n_dust_features = length(df_names)
        msg = "### Model will include $n_dust_features dust feature (PAH) components ###"
        for df_mn ∈ df_mean
            msg *= "\n### at lambda = $df_mn um ###"
        end
        @debug msg

        # Get all the complexes for the PAH features being used
        complexes = Vector{String}(unique([dust[:dust_features][n][:complex] for n ∈ df_names]))
        # Also sort the complexes by the wavelength values
        ss = sortperm([parse(Float64, c) for c ∈ complexes])
        complexes = complexes[ss]
        n_complexes = length(complexes)
        msg = "### These constitute $n_complexes dust feature (PAH) complexes ###"
        for cx ∈ complexes
            msg *= "\n### at lambda = $cx um ###"
        end
        @debug msg

        # Only use lines within the wavelength range being fit
        line_wave = [line_list[line].λ₀ for line ∈ keys(line_list)]
        ln_filt = [minimum(λ) < lw < maximum(λ) for lw ∈ line_wave]
        line_names = Vector{Symbol}(collect(keys(line_list))[ln_filt])
        # Sort lines by the wavelength values, since dictionaries are unsorted
        ss = sortperm(line_wave[ln_filt])
        line_names = line_names[ss]
        lines = [line_list[line] for line ∈ line_names]
        # Also get the profile and flow profile types for each line
        line_profiles = [line_list[line].profile for line ∈ line_names]
        line_flow_profiles = Vector{Union{Symbol,Nothing}}([line_list[line].flow_profile for line ∈ line_names])
        n_lines = length(line_names)
        msg = "### Model will include $n_lines emission lines ###"
        for (name, ln, prof, flow_prof) ∈ zip(line_names, lines, line_profiles, line_flow_profiles)
            msg *= "\n### $name at lambda = $(ln.λ₀) um with $prof profile and $flow_prof flow profile ###"
        end
        @debug msg

        # Unpack the voff_tied dictionary
        voff_tied_key = collect(keys(voff_tied))
        voff_tied = [voff_tied[voff] for voff ∈ voff_tied_key]
        n_voff_tied = length(voff_tied)
        # Also store the "tied" parameter for each line, which will need to be checked against the voff_tied_key
        # during fitting to find the proper location of the tied voff parameter to use
        line_tied = Vector{Union{Nothing,String}}([line.tied for line ∈ lines])
        msg = "### Model will include $n_voff_tied tied voff parameters ###"
        for lt ∈ voff_tied_key
            msg *= "\n### for group $lt ###"
        end
        @debug msg

        # Repeat for in/outflow velocity offsets, same logic
        flow_voff_tied_key = collect(keys(flow_voff_tied))
        flow_voff_tied = [flow_voff_tied[flow_voff] for flow_voff ∈ flow_voff_tied_key]
        n_flow_voff_tied = length(flow_voff_tied)
        line_flow_tied = Vector{Union{Nothing,String}}([line.flow_tied for line ∈ lines])
        msg = "### Model will include $n_flow_voff_tied tied flow voff parameters ###"
        for lft ∈ flow_voff_tied_key
            msg *= "\n### for group $lft ###"
        end
        @debug msg

        # Total number of parameters for the continuum and line fits
        n_params_cont = (2+2) + 2n_dust_cont + 3n_dust_features + 2n_complexes
        n_params_lines = n_voff_tied + n_flow_voff_tied
        # One η for all voigt profiles
        if (any(line_profiles .== :Voigt) || any(line_flow_profiles .== :Voigt)) && tie_voigt_mixing
            n_params_lines += 1
            @debug "### Model will include 1 tied voigt mixing parameter ###"
        end
        for i ∈ 1:n_lines
            if isnothing(line_tied[i]) || flexible_wavesol
                # amplitude, voff, and FWHM parameters
                n_params_lines += 3
            else
                # no voff parameter, since it's tied
                n_params_lines += 2
            end
            if line_profiles[i] == :GaussHermite
                # extra h3 and h4 parmeters
                n_params_lines += 2
            elseif line_profiles[i] == :Voigt
                # extra mixing parameter, but only if it's not tied
                if !tie_voigt_mixing
                    n_params_lines += 1
                end
            end
            # Repeat above for the inflow/outflow components
            if !isnothing(line_flow_profiles[i])
                if isnothing(line_flow_tied[i])
                    n_params_lines += 3
                else
                    n_params_lines += 2
                end
                if line_flow_profiles[i] == :GaussHermite
                    n_params_lines += 2
                elseif line_flow_profiles[i] == :Voigt
                    if !tie_voigt_mixing
                        n_params_lines += 1
                    end
                end
            end
            # Add extra 2 for the intensity and S/N, which are not fit but we need space for them in
            # the ParamMaps object
            n_params_lines += 2
        end
        @debug "### This totals to $(n_params_cont-2n_complexes) free continuum parameters ###"
        @debug "### This totals to $(n_params_lines-2n_lines) free emission line parameters ###"

        # Full 3D intensity model array
        cube_model = cubemodel_empty(shape, n_dust_cont, df_names, line_names)
        # 2D maps of fitting parameters
        param_maps = parammaps_empty(shape, n_dust_cont, df_names, complexes, line_names, line_tied,
            line_profiles, line_flow_tied, line_flow_profiles, voff_tied_key, flow_voff_tied_key, flexible_wavesol,
            tie_voigt_mixing)

        # Prepare output directories
        @debug "Preparing output directories"
        name = replace(name, " " => "_")
        if !isdir("output_$name")
            mkdir("output_$name")
        end
        if !isdir(joinpath("output_$name", "spaxel_plots"))
            mkdir(joinpath("output_$name", "spaxel_plots"))
        end
        if !isdir(joinpath("output_$name", "spaxel_binaries"))
            mkdir(joinpath("output_$name", "spaxel_binaries"))
        end
        if !isdir(joinpath("output_$name", "param_maps"))
            mkdir(joinpath("output_$name", "param_maps"))
        end
        if !isdir(joinpath("output_$name", "logs"))
            mkdir(joinpath("output_$name", "logs"))
        end

        # Prepare options
        extinction_curve = options[:extinction_curve]
        extinction_screen = options[:extinction_screen]
        χ²_thresh = options[:chi2_threshold]
        overwrite = options[:overwrite]
        cosmo = options[:cosmology]

        # Prepare rolling best fit parameter options
        rows = size(cube.Iλ, 1)
        @debug "Preparing rolling best fit continuum array with $rows rows and $(n_params_cont-2n_complexes) columns"
        p_best_cont = SharedArray(zeros(rows, n_params_cont-2n_complexes))
        @debug "Preparing rolling best fit line array with $rows rows and $(n_params_lines-2n_lines) columns"
        p_best_line = SharedArray(zeros(rows, n_params_lines-2n_lines))
        @debug "Preparing rolling best fit chi^2 vector with $rows items"
        χ²_best = SharedVector(zeros(rows))
        @debug "Preparing rolling best spaxel array with $rows tuples"
        best_spaxel = SharedVector(repeat([(0, 0)], rows))
        # If a fit has been run previously, read in the file containing the rolling best fit parameters
        # to pick up where the fitter left off seamlessly
        if isfile(joinpath("output_$name", "spaxel_binaries", "best_fit_params.LOKI"))
            p_best_dict = deserialize(joinpath("output_$name", "spaxel_binaries", "best_fit_params.LOKI"))
            p_best_cont = SharedArray(p_best_dict[:p_best_cont])
            p_best_line = SharedArray(p_best_dict[:p_best_line])
            χ²_best = SharedVector(p_best_dict[:chi2_best])
            best_spaxel = SharedVector(p_best_dict[:best_spaxel])
        end

        return new{eltype(χ²_best),eltype(eltype(best_spaxel))}(cube, z, name, n_procs, cube_model, param_maps, window_size, plot_spaxels, plot_maps, 
            parallel, save_fits, overwrite, extinction_curve, extinction_screen, T_s, T_dc, τ_97, β, n_dust_cont, n_dust_features, df_names, 
            dust_features, n_lines, line_names, line_profiles, line_flow_profiles, lines, n_voff_tied, line_tied, voff_tied_key, voff_tied, n_flow_voff_tied, 
            line_flow_tied, flow_voff_tied_key, flow_voff_tied, tie_voigt_mixing, voigt_mix_tied, n_complexes, complexes, n_params_cont, n_params_lines, 
            cosmo, χ²_thresh, interp_R, flexible_wavesol, p_best_cont, p_best_line, χ²_best, best_spaxel)
    end

end

############################## FITTING FUNCTIONS AND HELPERS ####################################

"""
    mask_emission_lines(λ, I, σ)

Mask out emission lines in a given spectrum using a series of median filters with varying
window sizes, taking the standard deviation of the flux between each window size, and comparing it to 
the RMS of the spectrum itself at each point.  

This function has been adapted from the BADASS code (Sexton et al. 2020; https://github.com/remingtonsexton/BADASS3).

# Arguments
- `λ::Vector{<:AbstractFloat}`: The wavelength vector of the spectrum
- `I::Vector{<:AbstractFloat}`: The flux vector of the spectrum
- `σ::Vector{<:AbstractFloat}`: The uncertainty vector of the spectrum

See also [`continuum_cubic_spline`](@ref)
"""
function mask_emission_lines(λ::Vector{<:AbstractFloat}, I::Vector{<:AbstractFloat}, σ::Vector{<:AbstractFloat})::BitVector

    # Series of window sizes to perform median filtering
    window_sizes = [2, 5, 10, 50, 100, 250, 500]
    med_spec = zeros(length(λ), length(window_sizes))
    mask = falses(length(λ))

    @debug "Masking emission lines with window sizes $window_sizes"

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


"""
    continuum_cubic_spline(λ, I, σ)

Mask out the emission lines in a given spectrum using `mask_emission_lines` and replace them with
a coarse cubic spline fit to the continuum, using wide knots to avoid reinterpolating the lines or
noise.

# Arguments
- `λ::Vector{<:AbstractFloat}`: The wavelength vector of the spectrum
- `I::Vector{<:AbstractFloat}`: The flux vector of the spectrum
- `σ::Vector{<:AbstractFloat}`: The uncertainty vector of the spectrum 

See also [`mask_emission_lines`](@ref)
"""
function continuum_cubic_spline(λ::Vector{<:AbstractFloat}, I::Vector{<:AbstractFloat}, σ::Vector{<:AbstractFloat})

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
        # @assert diffs[1] ≈ diffs[end]
        Δλ = diffs[1]

        # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
        λknots = λ[51]:Δλ*50:λ[end-51]
        @debug "Performing cubic spline continuum fit with knots at $λknots"

        # Do a full cubic spline remapping of the data
        I_out = Spline1D(λ[isfinite.(I_out)], I_out[isfinite.(I_out)], λknots, k=3, bc="extrapolate").(λ)
        σ_out = Spline1D(λ[isfinite.(σ_out)], σ_out[isfinite.(σ_out)], λknots, k=3, bc="extrapolate").(λ)
    end  

    return mask_lines, I_out, σ_out
end


"""
    continuum_fit_spaxel(cube_fitter, spaxel)

Fit the continuum of a given spaxel in the DataCube, masking out the emission lines, using the 
Levenberg-Marquardt least squares fitting method with the `CMPFit` package.  

This function has been adapted from PAHFIT (with some heavy adjustments -> masking out lines, allowing
PAH parameters to vary, and tying certain parameters together). See Smith, Draine, et al. 2007; 
http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php

# Arguments
`S<:Integer`
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::Tuple{S,S}`: The coordinates of the spaxel to be fit
"""
function continuum_fit_spaxel(cube_fitter::CubeFitter, spaxel::Tuple{S,S}) where {S<:Integer}

    @debug """\n
    #########################################################
    ###   Beginning continuum fit for spaxel $spaxel...   ###
    #########################################################
    """

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

    @debug "Spaxel $spaxel - Adding statistical error of $σ_stat in quadrature"
    
    # Mean and FWHM parameters for PAH profiles
    mean_df = [cdf[:wave] for cdf ∈ cube_fitter.dust_features]
    fwhm_df = [cdf[:fwhm] for cdf ∈ cube_fitter.dust_features]

    # amp_dc_prior = amp_agn_prior = Uniform(0., 1e12)  # just set it arbitrarily large, otherwise infinity gives bad logpdfs
    # amp_df_prior = Uniform(0., maximum(I) > 0. ? maximum(I) : 1e12)

    # stellar_priors = [amp_dc_prior, cube_fitter.T_s.prior]
    # dc_priors = vcat([[amp_dc_prior, Ti.prior] for Ti ∈ cube_fitter.T_dc]...)
    # df_priors = vcat([[amp_df_prior, mi.prior, fi.prior] for (mi, fi) ∈ zip(mean_df, fwhm_df)]...)

    # priors = vcat(stellar_priors, dc_priors, df_priors, [cube_fitter.τ_97.prior, cube_fitter.β.prior])

    # Check if the cube fitter has best fit parameters from a previous fit
    if !all(iszero.(cube_fitter.p_best_cont[spaxel[1], :]))

        @debug "Spaxel $spaxel - Using previous best fit continuum parameters..."

        # Set the parameters to the best parameters
        p₀ = Vector{Float64}(cube_fitter.p_best_cont[spaxel[1], :])

        # scale all flux amplitudes by the difference in medians between spaxels
        scale = nanmedian(cube_fitter.cube.Iλ[spaxel..., :]) / 
            nanmedian(cube_fitter.cube.Iλ[cube_fitter.best_spaxel[spaxel[1]]..., :])
        max_amp = nanmaximum(I)
        
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
            # Make sure the amplitude doesn't exceed the maximum for the given spaxel
            if p₀[pᵢ] > max_amp
                p₀[pᵢ] = max_amp
            end
            pᵢ += 3
        end

    # Otherwise, we estimate the initial parameters based on the data
    else

        @debug "Calculating initial starting points..."

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

        stellar_pars = [A_s, cube_fitter.T_s.value]

        dc_pars = vcat([[Ai, Ti.value] for (Ai, Ti) ∈ zip(A_dc, cube_fitter.T_dc)]...)

        df_pars = vcat([[Ai, mi.value, fi.value] for (Ai, mi, fi) ∈ zip(A_df, mean_df, fwhm_df)]...)

        # Initial parameter vector
        p₀ = Vector{Float64}(vcat(stellar_pars, dc_pars, df_pars, [cube_fitter.τ_97.value, cube_fitter.β.value]))

    end

    @debug "Continuum Parameter labels: \n [stellar_amp, stellar_temp, " * 
        join(["dust_continuum_amp_$i, dust_continuum_temp_$i" for i ∈ 1:cube_fitter.n_dust_cont], ", ") * 
        join(["$(df)_amp, $(df)_mean, $(df)_fwhm" for df ∈ cube_fitter.df_names], ", ") *
        "extinction_tau_97, extinction_beta]"

    # @debug "Priors: \n $priors"
    @debug "Spaxel $spaxel - Continuum Starting Values: \n $p₀"

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

    @debug "Spaxel $spaxel - Continuum Parameters locked? \n $([parinfo[i].fixed for i ∈ 1:length(p₀)])"
    @debug "Spaxel $spaxel - Continuum Lower limits: \n $([parinfo[i].limits[1] for i ∈ 1:length(p₀)])"
    @debug "Spaxel $spaxel - Continuum Upper limits: \n $([parinfo[i].limits[2] for i ∈ 1:length(p₀)])"

    @debug "Spaxel $spaxel - Beginning continuum fitting with Levenberg-Marquardt least squares (CMPFit):"

    res = cmpfit(λ, I, σ, (x, p) -> Util.fit_spectrum(x, p, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat,
        cube_fitter.extinction_curve, cube_fitter.extinction_screen), p₀, parinfo=parinfo, config=config)

    @debug "Spaxel $spaxel - continuum CMPFit status: $(res.status)"

    # Get best fit results
    popt = res.param
    # Count free parameters
    n_free = 0
    for pᵢ ∈ 1:length(popt)
        if iszero(parinfo[pᵢ].fixed)
            n_free += 1
        end
    end

    @debug "Spaxel $spaxel - Best fit continuum parameters: \n $popt"

    # function ln_prior(p)
    #     logpdfs = [logpdf(priors[i], p[i]) for i ∈ 1:length(p)]
    #     return sum(logpdfs)
    # end

    # function nln_probability(p)
    #     model = Util.fit_spectrum(λ, p, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, cube_fitter.extinction_curve,
    #              cube_fitter.extinction_screen)
    #     return -Util.ln_likelihood(I, model, σ) - ln_prior(p)
    # end

    # # res = optimize(nln_probability, minimum.(priors), maximum.(priors), p₀, 
    #     # SAMIN(;rt=0.9, nt=5, ns=5, neps=5, verbosity=0), Optim.Options(iterations=10^6))
    # popt = res.minimizer
    # χ2 = χ2red = -res.minimum
    # n_free = length(p₀)

    # Final optimized fit
    I_model, comps = Util.fit_spectrum(λ, popt, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, 
        cube_fitter.extinction_curve, cube_fitter.extinction_screen, return_components=true)

    msg = "######################################################################\n"
    msg *= "################# SPAXEL FIT RESULTS -- CONTINUUM ####################\n"
    msg *= "######################################################################\n"
    msg *= "\n#> STELLAR CONTINUUM <#\n"
    msg *= "Stellar_amp: \t\t\t $(@sprintf "%.3e" popt[1]) MJy/sr \t Limits: (0, Inf)\n"
    msg *= "Stellar_temp: \t\t\t $(@sprintf "%.0f" popt[2]) K \t (fixed)\n"
    pᵢ = 3
    msg *= "\n#> DUST CONTINUUM <#\n"
    for i ∈ 1:cube_fitter.n_dust_cont
        msg *= "Dust_continuum_$(i)_amp: \t\t $(@sprintf "%.3e" popt[pᵢ]) MJy/sr \t Limits: (0, Inf)\n"
        msg *= "Dust_continuum_$(i)_temp: \t\t $(@sprintf "%.0f" popt[pᵢ+1]) K \t\t\t (fixed)\n"
        msg *= "\n"
        pᵢ += 2
    end
    msg *= "\n#> DUST FEATURES <#\n"
    for (j, df) ∈ enumerate(cube_fitter.df_names)
        msg *= "$(df)_amp:\t\t\t $(@sprintf "%.1f" popt[pᵢ]) MJy/sr \t Limits: " *
            "(0, $(@sprintf "%.1f" nanmaximum(I))\n"
        msg *= "$(df)_mean:  \t\t $(@sprintf "%.3f" popt[pᵢ+1]) μm \t Limits: " *
            "($(@sprintf "%.3f" minimum(mean_df[j].prior)), $(@sprintf "%.3f" maximum(mean_df[j].prior)))" * 
            (mean_df[j].locked ? " (fixed)" : "") * "\n"
        msg *= "$(df)_fwhm:  \t\t $(@sprintf "%.3f" popt[pᵢ+2]) μm \t Limits: " *
            "($(@sprintf "%.3f" minimum(fwhm_df[j].prior)), $(@sprintf "%.3f" maximum(fwhm_df[j].prior)))" * 
            (fwhm_df[j].locked ? " (fixed)" : "") * "\n"
        msg *= "\n"
        pᵢ += 3
    end
    msg *= "\n#> EXTINCTION <#\n"
    msg *= "τ_9.7: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) [-] \t Limits: " *
        "($(@sprintf "%.2f" minimum(cube_fitter.τ_97.prior)), $(@sprintf "%.2f" maximum(cube_fitter.τ_97.prior)))" * 
        (cube_fitter.τ_97.locked ? " (fixed)" : "") * "\n"
    msg *= "β: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+1]) [-] \t Limits: " *
        "($(@sprintf "%.2f" minimum(cube_fitter.β.prior)), $(@sprintf "%.2f" maximum(cube_fitter.β.prior)))" * 
        (cube_fitter.β.locked ? " (fixed)" : "") * "\n"
    msg *= "\n"
    msg *= "######################################################################"
    @debug msg

    return σ, popt, I_model, comps, n_free

end


"""
    line_fit_spaxel(cube_fitter, spaxel, continuum)

Fit the emission lines of a given spaxel in the DataCube, subtracting the continuum, using the 
Simulated Annealing and L-BFGS fitting methods with the `Optim` package.

This function has been adapted from PAHFIT (with some heavy adjustments -> lines are now fit in a 
separate second step, and no longer using the Levenberg-Marquardt method; line width and voff limits are also
adjusted to compensate for the better spectral resolution of JWST compared to Spitzer). 
See Smith, Draine, et al. 2007; http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php

# Arguments
`S<:Integer`
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::Tuple{S,S}`: The coordinates of the spaxel to be fit
"""
function line_fit_spaxel(cube_fitter::CubeFitter, spaxel::Tuple{S,S}) where {S<:Integer}

    @debug """\n
    #########################################################
    ###      Beginning line fit for spaxel $spaxel...     ###
    #########################################################
    """

    # Extract spaxel to be fit
    λ = cube_fitter.cube.λ
    I = cube_fitter.cube.Iλ[spaxel..., :]
    σ = cube_fitter.cube.σI[spaxel..., :]

    # Perform a cubic spline continuum fit
    mask_lines, continuum, _ = continuum_cubic_spline(λ, I, σ)
    N = Float64(abs(nanmaximum(I)))
    N = N ≠ 0. ? N : 1.

    @debug "Spaxel $spaxel - Using normalization N=$N"

    # Add statistical uncertainties to the systematic uncertainties in quadrature
    σ_stat = std(I[.!mask_lines] .- continuum[.!mask_lines])
    σ .= .√(σ.^2 .+ σ_stat.^2)

    @debug "Spaxel $spaxel - Adding statistical error of $σ_stat in quadrature"

    # Normalized flux and uncertainty by subtracting the cubic spline fit and dividing by the maximum
    Inorm = (I .- continuum) ./ N
    σnorm = σ ./ N

    # Organize the voff, FWHM, h3, h4, and η parameters for each line into vectors
    voff_ln = [ln.parameters[:voff] for ln ∈ cube_fitter.lines]
    fwhm_ln = [ln.parameters[:fwhm] for ln ∈ cube_fitter.lines]
    h3_ln = [ln.profile == :GaussHermite ? ln.parameters[:h3] : nothing for ln ∈ cube_fitter.lines]
    h4_ln = [ln.profile == :GaussHermite ? ln.parameters[:h4] : nothing for ln ∈ cube_fitter.lines]
    η_ln = [ln.profile == :Voigt ? ln.parameters[:mixing] : nothing for ln ∈ cube_fitter.lines]
    # Overwrite η_ln with a single value if the voigt mixing should be tied
    if cube_fitter.tie_voigt_mixing
        index = findfirst(x -> !isnothing(x), η_ln)
        if !isnothing(index)
            η_ln = η_ln[index]
        else
            η_ln = nothing
        end
    end

    # Repeat for the inflow/outflow line parameters
    flow_voff_ln = [isnothing(ln.flow_profile) ? nothing : ln.parameters[:flow_voff] for ln ∈ cube_fitter.lines]
    flow_fwhm_ln = [isnothing(ln.flow_profile) ? nothing : ln.parameters[:flow_fwhm] for ln ∈ cube_fitter.lines]
    flow_h3_ln = [ln.flow_profile == :GaussHermite ? ln.parameters[:flow_h3] : nothing for ln ∈ cube_fitter.lines]
    flow_h4_ln = [ln.flow_profile == :GaussHermite ? ln.parameters[:flow_h4] : nothing for ln ∈ cube_fitter.lines]
    if !cube_fitter.tie_voigt_mixing
        flow_η_ln = [ln.flow_profile == :Voigt ? ln.parameters[:flow_mixing] : nothing for ln ∈ cube_fitter.lines]
    else
        flow_η_ln = η_ln
    end

    # Set up the prior vector
    amp_ln_prior = Uniform(0., 1.)
    amp_flow_prior = Uniform(0., 1.)
    λ0_ln = Vector{Float64}()
    prof_ln = Vector{Symbol}()
    flow_prof_ln = Vector{Union{Symbol,Nothing}}()
    ln_priors = Vector{Any}()
    param_names = Vector{String}()
    # Loop through each line and append the new components
    for (i, ln) ∈ enumerate(cube_fitter.lines)
        ln_name = cube_fitter.line_names[i]
        if isnothing(ln.tied) || cube_fitter.flexible_wavesol
            append!(ln_priors, [amp_ln_prior, voff_ln[i].prior, fwhm_ln[i].prior])
            append!(param_names, ["$(ln_name)_amp", "$(ln_name)_voff", "$(ln_name)_fwhm"])
        else
            append!(ln_priors, [amp_ln_prior, fwhm_ln[i].prior])
            append!(param_names, ["$(ln_name)_amp", "$(ln_name)_fwhm"])
        end
        if ln.profile == :GaussHermite
            append!(ln_priors, [h3_ln[i].prior, h4_ln[i].prior])
            append!(param_names, ["$(ln_name)_h3", "$(ln_name)_h4"])
        elseif ln.profile == :Voigt
            if !cube_fitter.tie_voigt_mixing
                append!(ln_priors, [η_ln[i].prior])
                append!(param_names, ["$(ln_name)_eta"])
            end
        end
        if !isnothing(ln.flow_profile)
            if isnothing(ln.flow_tied)
                append!(ln_priors, [amp_flow_prior, flow_voff_ln[i].prior, flow_fwhm_ln[i].prior])
                append!(param_names, ["$(ln_name)_flow_amp", "$(ln_name)_flow_voff", "$(ln_name)_flow_fwhm"])
            else
                append!(ln_priors, [amp_flow_prior, flow_fwhm_ln[i].prior])
                append!(param_names, ["$(ln_name)_flow_amp", "$(ln_name)_flow_fwhm"])
            end
            if ln.flow_profile == :GaussHermite
                append!(ln_priors, [flow_h3_ln[i].prior, flow_h4_ln[i].prior])
                append!(param_names, ["$(ln_name)_flow_h3", "$(ln_name)_flow_h4"])
            elseif ln.flow_profile == :Voigt
                if !cube_fitter.tie_voigt_mixing
                    append!(ln_prior, [flow_η_ln[i].prior])
                    append!(param_names, ["$(ln_name)_flow_eta"])
                end
            end
        end
        append!(λ0_ln, [ln.λ₀])
        append!(prof_ln, [ln.profile])
        append!(flow_prof_ln, [ln.flow_profile])
    end
    # Set up the tied voff parameters as vectors
    voff_tied_priors = [cube_fitter.voff_tied[i].prior for i ∈ 1:cube_fitter.n_voff_tied]
    flow_voff_tied_priors = [cube_fitter.flow_voff_tied[i].prior for i ∈ 1:cube_fitter.n_flow_voff_tied]

    # Initial prior vector
    if !cube_fitter.tie_voigt_mixing
        # If the voigt mixing parameters are untied, place them sequentially in the ln_priors
        priors = vcat(voff_tied_priors, flow_voff_tied_priors, ln_priors)
        param_names = vcat(["voff_tied_$k" for k ∈ cube_fitter.voff_tied_key], 
                           ["flow_voff_tied_$k" for k ∈ cube_fitter.flow_voff_tied_key], param_names)
    else
        # If the voigt mixing parameters are tied, just add the single mixing parameter before the rest of the linen parameters
        priors = vcat(voff_tied_priors, flow_voff_tied_priors, η_ln.prior, ln_priors)
        param_names = vcat(["voff_tied_$k" for k ∈ cube_fitter.voff_tied_key], 
                           ["flow_voff_tied_$k" for k ∈ cube_fitter.flow_voff_tied_key], ["eta_tied"], param_names)
    end

    # Check if there are previous best fit parameters
    if !all(iszero.(cube_fitter.p_best_line[spaxel[1], :]))

        @debug "Spaxel $spaxel - Using previous best fit line parameters..."

        # If so, set the parameters to the previous ones
        p₀ = Vector{Float64}(cube_fitter.p_best_line[spaxel[1], :])

        # Scale amplitudes by the ratio of the median intensities
        scale = nanmedian(cube_fitter.cube.Iλ[spaxel..., :]) / 
            nanmedian(cube_fitter.cube.Iλ[cube_fitter.best_spaxel[spaxel[1]]..., :])
        
        # Skip past the tied voff and tied flow voff parameters at the beginning
        pᵢ = 1 + cube_fitter.n_voff_tied + cube_fitter.n_flow_voff_tied
        # Skip past the tied voigt mixing, if present
        if cube_fitter.tie_voigt_mixing
            pᵢ += 1
        end
        for i ∈ 1:cube_fitter.n_lines
            p₀[pᵢ] *= scale
            # Make sure amplitude is not above the upper limit of 1 (normalized units)
            if p₀[pᵢ] > 1
                p₀[pᵢ] = 1
            end
            if isnothing(cube_fitter.line_tied[i]) || cube_fitter.flexible_wavesol
                # 3 parameters: amplitude, voff, FWHM
                pᵢ += 3
            else
                # 2 parameters: amplitude, FWHM
                pᵢ += 2
            end
            if prof_ln[i] == :GaussHermite
                # 2 extra parameters: h3 and h4
                pᵢ += 2
            elseif prof_ln[i] == :Voigt && !cube_fitter.tie_voigt_mixing
                # 1 extra parameter: eta
                pᵢ += 1
            end
            # Repeat for inflow/outflow components, if present
            if !isnothing(flow_prof_ln[i])
                p₀[pᵢ] *= scale
                if p₀[pᵢ] > 1
                    p₀[pᵢ] = 1
                end
                if isnothing(cube_fitter.line_flow_tied[i])
                    pᵢ += 3
                else
                    pᵢ += 2
                end
                if flow_prof_ln[i] == :GaussHermite
                    pᵢ += 2
                elseif flow_prof_ln[i] == :Voigt && !cube_fitter.tie_voigt_mixing
                    pᵢ += 1
                end
            end
        end

    else

        @debug "Calculating initial starting points..."

        # Start the ampltiudes at 1/2 or 1/4 (in normalized units)
        A_ln = ones(cube_fitter.n_lines) .* 0.5
        A_fl = ones(cube_fitter.n_lines) .* 0.25

        # Initial parameter vector
        ln_pars = Vector{Float64}()
        for (i, ln) ∈ enumerate(cube_fitter.lines)
            if isnothing(ln.tied) || cube_fitter.flexible_wavesol
                # 3 parameters: amplitude, voff, FWHM
                append!(ln_pars, [A_ln[i], voff_ln[i].value, fwhm_ln[i].value])
            else
                # 2 parameters: amplitude, FWHM (since voff is tied)
                append!(ln_pars, [A_ln[i], fwhm_ln[i].value])
            end
            if ln.profile == :GaussHermite
                # 2 extra parameters: h3 and h4
                append!(ln_pars, [h3_ln[i].value, h4_ln[i].value])
            elseif ln.profile == :Voigt
                # 1 extra parameter: eta, but only if not tied
                if !cube_fitter.tie_voigt_mixing
                    append!(ln_pars, [η_ln[i].value])
                end
            end
            # Repeat but for inflow/outflow components, if present
            if !isnothing(ln.flow_profile)
                if isnothing(ln.flow_tied)
                    append!(ln_pars, [A_fl[i], flow_voff_ln[i].value, flow_fwhm_ln[i].value])
                else
                    append!(ln_pars, [A_fl[i], flow_fwhm_ln[i].value])
                end
                if ln.flow_profile == :GaussHermite
                    append!(ln_pars, [flow_h3_ln[i].value, flow_h4_ln[i].value])
                elseif ln.flow_profile == :Voigt
                    if !cube_fitter.tie_voigt_mixing
                        append!(ln_pars, [flow_η_ln[i].value])
                    end
                end
            end
        end
        # Set up tied voff and flow voff parameter vectors
        voff_tied_pars = [cube_fitter.voff_tied[i].value for i ∈ 1:cube_fitter.n_voff_tied]
        flow_voff_tied_pars = [cube_fitter.flow_voff_tied[i].value for i ∈ 1:cube_fitter.n_flow_voff_tied]

        # Set up the parameter vector in the proper order: 
        # (tied voffs, tied flow voffs, tied voigt mixing, [amp, voff, FWHM, h3, h4, eta,
        #     flow_amp, flow_voff, flow_FWHM, flow_h3, flow_h4, flow_eta] for each line)
        if !cube_fitter.tie_voigt_mixing
            p₀ = Vector{Float64}(vcat(voff_tied_pars, flow_voff_tied_pars, ln_pars))
        else
            p₀ = Vector{Float64}(vcat(voff_tied_pars, flow_voff_tied_pars, η_ln.value, ln_pars))
        end

    end

    @debug "Line Parameter labels: \n $param_names"
    @debug "Spaxel $spaxel - Line starting values: \n $p₀"
    @debug "Spaxel $spaxel - Line priors: \n $priors"

    # Internal helper function to calculate the log of the prior probability for all the parameters
    # Most priors will be uniform, i.e. constant, finite prior values as long as the parameter is inbounds.
    # Priors will be -Inf if any parameter goes out of bounds.  There is also an additional penalty applied
    # if any inflow/outflow amplitudes are greater than their corresponding line's amplitudes, or if their 
    # FWHMs are smaller than their corresponding line's FWHMs.  This ensures that inflow/outflow components
    # should always correspond to the actual flow component, instead of allowing them to swap places with
    # the main line, which could happen with no constraints.
    function ln_prior(p)

        # sum the log prior distribution of each parameter
        lnpdf = sum([logpdf(priors[i], p[i]) for i ∈ 1:length(p)])

        pᵢ = 1 + cube_fitter.n_voff_tied + cube_fitter.n_flow_voff_tied
        if cube_fitter.tie_voigt_mixing
            pᵢ += 1
        end
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
            elseif prof_ln[i] == :Voigt && !cube_fitter.tie_voigt_mixing
                pᵢ += 1
            end
            fwhm_res = Util.C_KMS / cube_fitter.interp_R(λ0_ln[i])
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
                elseif flow_prof_ln[i] == :Voigt && !cube_fitter.tie_voigt_mixing
                    pᵢ += 1
                end
                # penalize the likelihood if any in/outflow FWHMs are smaller than the corresponding narrow lines
                # or if the amplitudes are too large
                if flow_fwhm ≤ na_fwhm || flow_amp ≥ na_amp
                    lnpdf += -Inf
                end
            end
        end 

        return lnpdf
    end

    # Internal helper function to calculate the negative of the log of the probability,
    # ln(probability) = ln(likelihood) + ln(prior)
    function negln_probability(p)
        model = Util.fit_line_residuals(λ, p, cube_fitter.n_lines, cube_fitter.n_voff_tied, 
            cube_fitter.voff_tied_key, cube_fitter.line_tied, prof_ln, cube_fitter.n_flow_voff_tied,
            cube_fitter.flow_voff_tied_key, cube_fitter.line_flow_tied, flow_prof_ln, λ0_ln, 
            cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing)
        lnP = Util.ln_likelihood(Inorm, model, σnorm) + ln_prior(p)
        return -lnP 
    end

    lower_bounds = minimum.(priors)
    upper_bounds = maximum.(priors)

    @debug "Spaxel $spaxel - Line Lower limits: \n $lower_bounds"
    @debug "Spaxel $spaxel - Line Upper Limits: \n $upper_bounds"

    @debug "Spaxel $spaxel - Beginning Line fitting with Simulated Annealing:"

    # First, perform a bounded Simulated Annealing search for the optimal parameters with a generous rt and max iterations
    res = optimize(negln_probability, lower_bounds, upper_bounds, p₀, 
        SAMIN(;rt=0.9, nt=5, ns=5, neps=5, verbosity=0), Optim.Options(iterations=10^6))

    p₁ = res.minimizer

    @debug "Spaxel $spaxel - Refining Line best fit with L-BFGS:"

    # Then, refine the solution with a bounded local minimum search with LBFGS
    res = optimize(negln_probability, lower_bounds, upper_bounds, p₁, Fminbox(LBFGS()))

    #################################### DEPRECATED FIT WITH LEVMAR ###################################################

    # # Convert parameter limits into CMPFit object
    # parinfo = CMPFit.Parinfo(length(p₀))

    # # Tied velocity offsets
    # pᵢ = 1
    # for i ∈ 1:cube_fitter.n_voff_tied
    #     parinfo[pᵢ].fixed = cube_fitter.voff_tied[i].locked
    #     if !(cube_fitter.voff_tied[i].locked)
    #         parinfo[pᵢ].limited = (1,1)
    #         parinfo[pᵢ].limits = (minimum(cube_fitter.voff_tied[i].prior), maximum(cube_fitter.voff_tied[i].prior))
    #     end
    #     pᵢ += 1
    # end

    # # Tied in/outflow velocity offsets
    # for j ∈ 1:cube_fitter.n_flow_voff_tied
    #     parinfo[pᵢ].fixed = cube_fitter.flow_voff_tied[j].locked
    #     if !(cube_fitter.flow_voff_tied[j].locked)
    #         parinfo[pᵢ].limited = (1,1)
    #         parinfo[pᵢ].limits = (minimum(cube_fitter.flow_voff_tied[j].prior), maximum(cube_fitter.flow_voff_tied[j].prior))
    #     end
    #     pᵢ += 1
    # end

    # # Tied voigt mixing
    # if cube_fitter.tie_voigt_mixing
    #     parinfo[pᵢ].fixed = cube_fitter.voigt_mix_tied.locked
    #     if !(cube_fitter.voigt_mix_tied.locked)
    #         parinfo[pᵢ].limited = (1,1)
    #         parinfo[pᵢ].limits = (minimum(cube_fitter.voigt_mix_tied.prior), maximum(cube_fitter.voigt_mix_tied.prior))
    #     end
    #     pᵢ += 1
    # end

    # # Emission line amplitude, voff, fwhm
    # for i ∈ 1:cube_fitter.n_lines
    #     line_amp = p₀[pᵢ]
    #     parinfo[pᵢ].limited = (1,1)
    #     parinfo[pᵢ].limits = (0., 1.0)
    #     if isnothing(cube_fitter.line_tied[i]) || cube_fitter.flexible_wavesol
    #         parinfo[pᵢ+1].fixed = voff_ln[i].locked
    #         if !(voff_ln[i].locked)
    #             parinfo[pᵢ+1].limited = (1,1)
    #             parinfo[pᵢ+1].limits = (minimum(voff_ln[i].prior), maximum(voff_ln[i].prior))
    #         end
    #         line_fwhm = p₀[pᵢ+2]
    #         parinfo[pᵢ+2].fixed = fwhm_ln[i].locked
    #         if !(fwhm_ln[i].locked)
    #             parinfo[pᵢ+2].limited = (1,1)
    #             parinfo[pᵢ+2].limits = (minimum(fwhm_ln[i].prior), maximum(fwhm_ln[i].prior))
    #         end
    #         if prof_ln[i] == :GaussHermite
    #             parinfo[pᵢ+3].fixed = h3_ln[i].locked
    #             if !(h3_ln[i].locked)
    #                 parinfo[pᵢ+3].limited = (1,1)
    #                 parinfo[pᵢ+3].limits = (minimum(h3_ln[i].prior), maximum(h3_ln[i].prior))
    #             end
    #             parinfo[pᵢ+4].fixed = h4_ln[i].locked
    #             if !(h4_ln[i].locked)
    #                 parinfo[pᵢ+4].limited = (1,1)
    #                 parinfo[pᵢ+4].limits = (minimum(h4_ln[i].prior), maximum(h4_ln[i].prior))
    #             end
    #             pᵢ += 2
    #         elseif prof_ln[i] == :Voigt && !cube_fitter.tie_voigt_mixing
    #             parinfo[pᵢ+3].fixed = η_ln[i].locked
    #             if !(η_ln[i].locked)
    #                 parinfo[pᵢ+3].limited = (1,1)
    #                 parinfo[pᵢ+3].limits = (minimum(η_ln[i].prior), maximum(η_ln[i].prior))
    #             end
    #             pᵢ += 1
    #         end
    #         pᵢ += 3
    #     else
    #         line_fwhm = p₀[pᵢ+1]
    #         parinfo[pᵢ+1].fixed = fwhm_ln[i].locked
    #         if !(fwhm_ln[i].locked)
    #             parinfo[pᵢ+1].limited = (1,1)
    #             parinfo[pᵢ+1].limits = (minimum(fwhm_ln[i].prior), maximum(fwhm_ln[i].prior))
    #         end
    #         if prof_ln[i] == :GaussHermite
    #             parinfo[pᵢ+2].fixed = h3_ln[i].locked
    #             if !(h3_ln[i].locked)
    #                 parinfo[pᵢ+2].limited = (1,1)
    #                 parinfo[pᵢ+2].limits = (minimum(h3_ln[i].prior), maximum(h3_ln[i].prior))
    #             end
    #             parinfo[pᵢ+3].fixed = h4_ln[i].locked
    #             if !(h4_ln[i].locked)
    #                 parinfo[pᵢ+3].limited = (1,1)
    #                 parinfo[pᵢ+3].limits = (minimum(h4_ln[i].prior), maximum(h4_ln[i].prior))
    #             end
    #             pᵢ += 2       
    #         elseif prof_ln[i] == :Voigt && !cube_fitter.tie_voigt_mixing
    #             parinfo[pᵢ+2].fixed = η_ln[i].locked
    #             if !(η_ln[i].locked)
    #                 parinfo[pᵢ+2].limited = (1,1)
    #                 parinfo[pᵢ+2].limits = (minimum(η_ln[i].prior), maximum(η_ln[i].prior))
    #             end
    #             pᵢ += 1
    #         end
    #         pᵢ += 2
    #     end
    #     if !isnothing(flow_prof_ln[i])
    #         parinfo[pᵢ].limited = (1,1)
    #         parinfo[pᵢ].limits = (0., line_amp)
    #         if isnothing(cube_fitter.line_flow_tied[i])
    #             parinfo[pᵢ+1].fixed = flow_voff_ln[i].locked
    #             if !(flow_voff_ln[i].locked)
    #                 parinfo[pᵢ+1].limited = (1,1)
    #                 parinfo[pᵢ+1].limits = (minimum(flow_voff_ln[i].prior), maximum(flow_voff_ln[i].prior))
    #             end
    #             parinfo[pᵢ+2].fixed = flow_fwhm_ln[i].locked
    #             if !(flow_fwhm_ln[i].locked)
    #                 parinfo[pᵢ+2].limited = (1,1)
    #                 parinfo[pᵢ+2].limits = (line_fwhm, maximum(flow_fwhm_ln[i].prior))
    #             end
    #             if flow_prof_ln[i] == :GaussHermite
    #                 parinfo[pᵢ+3].fixed = flow_h3_ln[i].locked
    #                 if !(flow_h3_ln[i].locked)
    #                     parinfo[pᵢ+3].limited = (1,1)
    #                     parinfo[pᵢ+3].limits = (minimum(flow_h3_ln[i].prior), maximum(flow_h3_ln[i].prior))
    #                 end
    #                 parinfo[pᵢ+4].fixed = flow_h4_ln[i].locked
    #                 if !(flow_h4_ln[i].locked)
    #                     parinfo[pᵢ+4].limited = (1,1)
    #                     parinfo[pᵢ+4].limits = (minimum(flow_h4_ln[i].prior), maximum(flow_h4_ln[i].prior))
    #                 end
    #                 pᵢ += 2
    #             elseif flow_prof_ln[i] == :Voigt && !cube_fitter.tie_voigt_mixing
    #                 parinfo[pᵢ+3].fixed = flow_η_ln[i].locked
    #                 if !(flow_η_ln[i].locked)
    #                     parinfo[pᵢ+3].limited = (1,1)
    #                     parinfo[pᵢ+3].limits = (minimum(flow_η_ln[i].prior), maximum(flow_η_ln[i].prior))
    #                 end
    #                 pᵢ += 1
    #             end
    #             pᵢ += 3
    #         else
    #             parinfo[pᵢ+1].fixed = flow_fwhm_ln[i].locked
    #             if !(flow_fwhm_ln[i].locked)
    #                 parinfo[pᵢ+1].limited = (1,1)
    #                 parinfo[pᵢ+1].limits = (line_fwhm, maximum(flow_fwhm_ln[i].prior))
    #             end
    #             if flow_prof_ln[i] == :GaussHermite
    #                 parinfo[pᵢ+2].fixed = flow_h3_ln[i].locked
    #                 if !(flow_h3_ln[i].locked)
    #                     parinfo[pᵢ+2].limited = (1,1)
    #                     parinfo[pᵢ+2].limits = (minimum(flow_h3_ln[i].prior), maximum(flow_h3_ln[i].prior))
    #                 end
    #                 parinfo[pᵢ+3].fixed = flow_h4_ln[i].locked
    #                 if !(flow_h4_ln[i].locked)
    #                     parinfo[pᵢ+3].limited = (1,1)
    #                     parinfo[pᵢ+3].limits = (minimum(flow_h4_ln[i].prior), maximum(flow_h4_ln[i].prior))
    #                 end
    #                 pᵢ += 2       
    #             elseif flow_prof_ln[i] == :Voigt && !cube_fitter.tie_voigt_mixing
    #                 parinfo[pᵢ+2].fixed = flow_η_ln[i].locked
    #                 if !(flow_η_ln[i].locked)
    #                     parinfo[pᵢ+2].limited = (1,1)
    #                     parinfo[pᵢ+2].limits = (minimum(flow_η_ln[i].prior), maximum(flow_η_ln[i].prior))
    #                 end
    #                 pᵢ += 1
    #             end
    #             pᵢ += 2
    #         end
    #     end
    # end

    # # Create a `config` structure
    # config = CMPFit.Config()

    # res = cmpfit(λ, Inorm, σnorm, (x, p) -> Util.fit_line_residuals(x, p, cube_fitter.n_lines, cube_fitter.n_voff_tied, 
    #     cube_fitter.voff_tied_key, cube_fitter.line_tied, prof_ln, cube_fitter.n_flow_voff_tied, cube_fitter.flow_voff_tied_key,
    #     cube_fitter.line_flow_tied, flow_prof_ln, λ0_ln, cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing), p₁, 
    #     parinfo=parinfo, config=config)

    # # Get the results
    # popt = res.param

    # Count free parameters
    # n_free = 0
    # for pᵢ ∈ 1:length(popt)
    #     if iszero(parinfo[pᵢ].fixed)
    #         n_free += 1
    #     end
    # end

    ######################################################################################################################

    # Get the optimized parameter vector and number of free parameters
    popt = res.minimizer
    n_free = length(p₀)

    @debug "Spaxel $spaxel - Best fit line parameters: \n $popt"

    # Final optimized fit
    I_model, comps = Util.fit_line_residuals(λ, popt, cube_fitter.n_lines, cube_fitter.n_voff_tied, 
        cube_fitter.voff_tied_key, cube_fitter.line_tied, prof_ln, cube_fitter.n_flow_voff_tied,
        cube_fitter.flow_voff_tied_key, cube_fitter.line_flow_tied, flow_prof_ln, λ0_ln, 
        cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing, return_components=true)
    
    # Renormalize
    I_model = I_model .* N
    for comp ∈ keys(comps)
        comps[comp] = comps[comp] .* N
    end

    msg = "######################################################################\n"
    msg *= "############### SPAXEL FIT RESULTS -- EMISSION LINES #################\n"
    msg *= "######################################################################\n"
    pᵢ = 1
    msg *= "\n#> TIED VELOCITY OFFSETS <#\n"
    for (i, vk) ∈ enumerate(cube_fitter.voff_tied_key)
        msg *= "$(vk)_tied_voff: \t\t\t $(@sprintf "%.0f" popt[pᵢ]) km/s \t " *
            "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.voff_tied[i].prior)), $(@sprintf "%.0f" maximum(cube_fitter.voff_tied[i].prior)))\n"
        pᵢ += 1
    end
    for (j, fvk) ∈ enumerate(cube_fitter.flow_voff_tied_key)
        msg *= "$(fvk)_flow_tied_voff:\t\t\t $(@sprintf "%.0f" popt[pᵢ]) km/s \t " *
            "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.flow_voff_tied[j].prior)), $(@sprintf "%.0f" maximum(cube_fitter.flow_voff_tied[j].prior)))\n"
        pᵢ += 1
    end
    msg *= "\n#> TIED VOIGT MIXING <#\n"
    if cube_fitter.tie_voigt_mixing
        msg *= "tied_voigt_mixing: \t\t\t $(@sprintf "%.2f" popt[pᵢ]) [-] \t " * 
            "Limits: ($(@sprintf "%.2f" minimum(cube_fitter.voigt_mix_tied.prior)), $(@sprintf "%.2f" maximum(cube_fitter.voigt_mix_tied.prior)))\n"
        pᵢ += 1
    end
    msg *= "\n#> EMISSION LINES <#\n"
    for (k, (ln, nm)) ∈ enumerate(zip(cube_fitter.lines, cube_fitter.line_names))
        msg *= "$(nm)_amp:\t\t\t $(@sprintf "%.0f" popt[pᵢ]*N) MJy/sr \t Limits: (0, $(@sprintf "%.0f" nanmaximum(I)))\n"
        if isnothing(cube_fitter.line_tied[k]) || cube_fitter.flexible_wavesol
            msg *= "$(nm)_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(voff_ln[k].prior)), $(@sprintf "%.0f" maximum(voff_ln[k].prior)))\n"
            msg *= "$(nm)_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+2]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))\n"
            if prof_ln[k] == :GaussHermite
                msg *= "$(nm)_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+3])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))\n"
                msg *= "$(nm)_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+4])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))\n"
                pᵢ += 2
            elseif prof_ln[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                msg *= "$(nm)_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+3])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(η_ln[k].prior)), $(@sprintf "%.3f" maximum(η_ln[k].prior)))\n"
                pᵢ += 1
            end
            pᵢ += 3
        else
            msg *= "$(nm)_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))\n"
            if prof_ln[k] == :GaussHermite
                msg *= "$(nm)_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+2])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))\n"
                msg *= "$(nm)_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+3])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))\n"
                pᵢ += 2
            elseif prof_ln[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                msg *= "$(nm)_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+2])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(η_ln[k].prior)), $(@sprintf "%.3f" maximum(η_ln[k].prior)))\n"
                pᵢ += 1
            end
            pᵢ += 2
        end
        if !isnothing(flow_prof_ln[k])
            msg *= "\n$(nm)_flow_amp:\t\t\t $(@sprintf "%.0f" popt[pᵢ]*N) MJy/sr \t Limits: (0, $(@sprintf "%.0f" nanmaximum(I)))\n"
            if isnothing(cube_fitter.line_flow_tied[k])
                msg *= "$(nm)_flow_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) km/s \t " *
                    "Limits: ($(@sprintf "%.0f" minimum(voff_ln[k].prior)), $(@sprintf "%.0f" maximum(voff_ln[k].prior)))\n"
                msg *= "$(nm)_flow_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+2]) km/s \t " *
                    "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))\n"
                if flow_prof_ln[k] == :GaussHermite
                    msg *= "$(nm)_flow_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+3])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))\n"
                    msg *= "$(nm)_flow_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+4])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))\n"
                    pᵢ += 2
                elseif flow_prof_ln[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                    msg *= "$(nm)_flow_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+3])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(η_ln[k].prior)), $(@sprintf "%.3f" maximum(η_ln[k].prior)))\n"
                    pᵢ += 1
                end
                pᵢ += 3
            else
                msg *= "$(nm)_flow_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) km/s \t " *
                    "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))\n"
                if flow_prof_ln[k] == :GaussHermite
                    msg *= "$(nm)_flow_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+2])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))\n"
                    msg *= "$(nm)_flow_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+3])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))\n"
                    pᵢ += 2
                elseif flow_prof_ln[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                    msg *= "$(nm)_flow_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+2])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(η_ln[k].prior)), $(@sprintf "%.3f" maximum(η_ln[k].prior)))\n"
                    pᵢ += 1
                end
                pᵢ += 2
            end
        end
        msg *= "\n"
    end 
    msg *= "######################################################################" 
    @debug msg

    return σ, popt, I_model, comps, n_free

end


"""
    plot_spaxel_fit(λ, I, I_cont, σ, comps, n_dust_cont, n_dust_features, line_wave, line_names, z, χ2red, 
        name, label; backend=backend)

Plot the fit for an individual spaxel, `I_cont`, and its individual components `comps`, using the given 
backend (`:pyplot` or `:plotly`).

# Arguments
`T<:AbstractFloat`
- `λ::Vector{<:AbstractFloat}`: The wavelength vector of the spaxel to be plotted
- `I::Vector{<:AbstractFloat}`: The intensity data vector of the spaxel to be plotted
- `I_cont::Vector{<:AbstractFloat}`: The intensity model vector of the spaxel to be plotted
- `σ::Vector{<:AbstractFloat}`: The uncertainty vector of the spaxel to be plotted
- `comps::Dict{String, Vector{T}}`: The dictionary of individual components of the model intensity
- `n_dust_cont::Integer`: The number of dust continuum components in the fit
- `n_dust_features::Integer`: The number of PAH features in the fit
- `line_wave::Vector{<:AbstractFloat}`: List of nominal central wavelengths for each line in the fit
- `line_names::Vector{Symbol}`: List of names for each line in the fit
- `z::AbstractFloat`: The redshift of the object being fit
- `χ2red::AbstractFloat`: The reduced χ^2 value of the fit
- `name::String`: The name of the object being fit
- `label::String`: A label for the individual spaxel being plotted, to be put in the file name
- `backend::Symbol`: The backend to use to plot, either `:pyplot` or `:plotly`
"""
function plot_spaxel_fit(λ::Vector{<:AbstractFloat}, I::Vector{<:AbstractFloat}, I_cont::Vector{<:AbstractFloat}, 
    σ::Vector{<:AbstractFloat}, comps::Dict{String, Vector{T}}, n_dust_cont::Integer, n_dust_features::Integer, 
    line_wave::Vector{<:AbstractFloat}, line_names::Vector{Symbol}, z::AbstractFloat, χ2red::AbstractFloat, name::String, 
    label::String; backend::Symbol=:pyplot) where {T<:AbstractFloat}

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
        PlotlyJS.savefig(p, isnothing(label) ? joinpath("output_$name", "spaxel_plots", "levmar_fit_spaxel.html") : 
            joinpath("output_$name", "spaxel_plots", "$label.html"))

    elseif backend == :pyplot
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(nrows=4, ncols=1, hspace=0.)
        ax1 = fig.add_subplot(py"$(gs)[:-1, :]")
        ax2 = fig.add_subplot(py"$(gs)[-1, :]")
        ax1.plot(λ, I, "k-", label="Data")
        ax1.plot(λ, I_cont, "r-", label="Model")
        ax2.plot(λ, I.-I_cont, "k-")
        ax2.plot(λ, ones(length(λ)), "r-", label="\$\\tilde{\\chi}^2 = $(@sprintf "%.3f" χ2red)\$")
        ax2.fill_between(λ, σ, .-σ, color="k", alpha=0.5)
        ax3 = ax1.twinx()
        ax4 = ax1.twiny()
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
        ax4.set_xlim(minimum(Util.observed_frame(λ, z)), maximum(Util.observed_frame(λ, z)))
        ax1.set_ylabel("\$ I_{\\nu} \$ (MJy sr\$^{-1}\$)")
        ax1.set_ylim(bottom=0.)
        ax2.set_ylabel("Residuals")
        ax2.set_xlabel("\$ \\lambda_{\\rm rest} \$ (\$\\mu\$m)")
        ax4.set_xlabel("\$ \\lambda_{\\rm obs} \$ (\$\\mu\$m)")
        ax2.legend(loc="upper left")
        # ax1.set_title("\$\\tilde{\\chi}^2 = $(@sprintf "%.3f" χ2red)\$")
        ax1.tick_params(axis="both", direction="in")
        ax2.tick_params(axis="both", direction="in", labelright=true, right=true)
        ax3.tick_params(axis="both", direction="in")
        plt.savefig(isnothing(label) ? joinpath("output_$name", "spaxel_plots", "levmar_fit_spaxel.pdf") : 
            joinpath("output_$name", "spaxel_plots", "$label.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    end
end


"""
    calculate_extra_parameters(cube_fitter, spaxel, comps)

Calculate extra parameters that are not fit, but are nevertheless important to know, for a given spaxel.
Currently this includes the integrated intensity and signal to noise ratios of dust complexes and emission lines.

# Arguments
`T<:AbstractFloat, S<:Integer`
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::Tuple{S,S}`: The coordinates of the spaxel to be fit
- `popt_c::Vector{T}`: The best-bit parameter vector for the continuum components of the fit
- `popt_l::Vector{T}`: The best-fit parameter vector for the line components of the fit
"""
function calculate_extra_parameters(cube_fitter::CubeFitter, spaxel::Tuple{S,S}, 
    popt_c::Vector{T}, popt_l::Vector{T}) where {T<:AbstractFloat,S<:Integer}

    @debug "Spaxel $spaxel - Calculating extra parameters"

    # Extract the wavelength, intensity, and uncertainty data
    λ = cube_fitter.cube.λ
    I = cube_fitter.cube.Iλ[spaxel..., :]
    σ = cube_fitter.cube.σI[spaxel..., :]

    Δλ = mean(diff(λ)) / 10

    # Perform a cubic spline fit to the continuum, masking out lines
    mask_lines, continuum, _ = continuum_cubic_spline(λ, I, σ)
    N = Float64(abs(nanmaximum(I)))
    N = N ≠ 0. ? N : 1.

    # Loop through dust complexes
    p_complex = zeros(2cube_fitter.n_complexes)
    pₒ = 1
    for c ∈ cube_fitter.complexes

        # Start with a flat profile at 0
        profile = x -> 0.
        # Initial parameter vector index where dust profiles start
        pᵢ = 3 + 2cube_fitter.n_dust_cont

        # Add up the dust feature profiles that belong to this complex
        for (ii, cdf) ∈ enumerate(cube_fitter.dust_features)
            if cdf[:complex] == c
                # unpack the parameters
                A, μ, fwhm = popt_c[pᵢ:pᵢ+2]
                # add the anonymous functions recursively
                profile = let profile = profile
                    x -> profile(x) + Util.Drude(x, A, μ, fwhm)
                end
            end
            # increment the parameter index
            pᵢ += 3
        end
        λ_arr = (minimum(λ)-3):Δλ:(maximum(λ)+3)
        peak, peak_ind = findmax(profile.(λ_arr))

        # Integrate the intensity of the combined profile using Gauss-Kronrod quadrature
        p_complex[pₒ], _ = quadgk(profile, 0, Inf, order=200)

        # SNR, calculated as (amplitude) / (RMS of the surrounding spectrum)
        p_complex[pₒ+1] = peak / std(I[.!mask_lines] .- continuum[.!mask_lines])
        @debug "Dust complex $c with integrated intensity $(p_complex[pₒ]) and SNR $(p_complex[pₒ+1])"

        pₒ += 2
    end

    # Loop through lines
    p_lines = zeros(2cube_fitter.n_lines)
    pₒ = 1
    # Skip over the tied velocity offsets
    pᵢ = cube_fitter.n_voff_tied + cube_fitter.n_flow_voff_tied + 1
    # Skip over the tied voigt mixing parameter, saving its index
    if cube_fitter.tie_voigt_mixing
        ηᵢ = pᵢ
        pᵢ += 1
    end
    for (k, ln) ∈ enumerate(cube_fitter.lines)

        # (\/ pretty much the same as the fit_line_residuals function, but outputting anonymous line profile functions)
        amp = popt_l[pᵢ]
            
        # Check if voff is tied: if so, use the tied voff parameter, otherwise, use the line's own voff parameter
        if isnothing(cube_fitter.line_tied[k])
            # Unpack the components of the line
            voff = popt_l[pᵢ+1]
            fwhm = popt_l[pᵢ+2]
            if cube_fitter.line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = popt_l[pᵢ+3]
                h4 = popt_l[pᵢ+4]
            elseif cube_fitter.line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                if !cube_fitter.tie_voigt_mixing
                    η = popt_l[pᵢ+3]
                else
                    η = popt_l[ηᵢ]
                end
            end
        elseif !isnothing(cube_fitter.line_tied[k]) && cube_fitter.flexible_wavesol
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and voff_tied_key
            vwhere = findfirst(x -> x == cube_fitter.line_tied[k], cube_fitter.voff_tied_key)
            voff_series = popt_l[vwhere]
            voff_indiv = popt_l[pᵢ+1]
            # Add velocity shifts of the tied lines and the individual offsets together
            voff = voff_series + voff_indiv
            fwhm = popt_l[pᵢ+2]
            if cube_fitter.line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = popt_l[pᵢ+3]
                h4 = popt_l[pᵢ+4]
            elseif cube_fitter.line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                if !cube_fitter.tie_voigt_mixing
                    η = popt_l[pᵢ+3]
                else
                    η = popt_l[ηᵢ]
                end
            end
        else
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and voff_tied_key
            vwhere = findfirst(x -> x == cube_fitter.line_tied[k], cube_fitter.voff_tied_key)
            voff = popt_l[vwhere]
            fwhm = popt_l[pᵢ+1]
            # (dont add any individual voff components)
            if cube_fitter.line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = popt_l[pᵢ+2]
                h4 = popt_l[pᵢ+3]
            elseif cube_fitter.line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                if !cube_fitter.tie_voigt_mixing
                    η = popt_l[pᵢ+2]
                else
                    η = popt_l[ηᵢ]
                end
            end
        end

        # Convert voff in km/s to mean wavelength in μm
        mean_μm = Util.Doppler_shift_λ(ln.λ₀, voff)
        # Convert FWHM from km/s to μm
        fwhm_μm = Util.Doppler_shift_λ(ln.λ₀, fwhm) - ln.λ₀
        # Evaluate line profiles centered at 0 (since the shift doesnt matter for integration)
        # -> evaluating centered at 0 helps quadgk when the fwhm is small compared to the integration region
        if cube_fitter.line_profiles[k] == :Gaussian
            profile = x -> Util.Gaussian(x, amp, 0., fwhm_μm)
        elseif cube_fitter.line_profiles[k] == :Lorentzian
            profile = x -> Util.Lorentzian(x, amp, 0., fwhm_μm)
        elseif cube_fitter.line_profiles[k] == :GaussHermite
            profile = x -> Util.GaussHermite(x, amp, 0., fwhm_μm, h3, h4)
        elseif cube_fitter.line_profiles[k] == :Voigt
            profile = x -> Util.Voigt(x, amp, 0., fwhm_μm, η)
        else
            error("Unrecognized line profile $(cube_fitter.line_profiles[k])!")
        end
    
        # Advance the parameter vector index -> 3 if untied (or tied + flexible_wavesol) or 2 if tied
        pᵢ += isnothing(cube_fitter.line_tied[k]) || cube_fitter.flexible_wavesol ? 3 : 2
        if cube_fitter.line_profiles[k] == :GaussHermite
            # advance and extra 2 if GaussHermite profile
            pᵢ += 2
        elseif cube_fitter.line_profiles[k] == :Voigt
            # advance another extra 1 if untied Voigt profile
            if !cube_fitter.tie_voigt_mixing
                pᵢ += 1
            end
        end

        # Repeat EVERYTHING, minus the flexible_wavesol, for the inflow/outflow components
        if !isnothing(cube_fitter.line_flow_profiles[k])
            flow_amp = popt_l[pᵢ]
            if isnothing(cube_fitter.line_flow_tied[k])
                flow_voff = popt_l[pᵢ+1]
                flow_fwhm = popt_l[pᵢ+2]
                if cube_fitter.line_flow_profiles[k] == :GaussHermite
                    flow_h3 = popt_l[pᵢ+3]
                    flow_h4 = popt_l[pᵢ+4]
                elseif cube_fitter.line_flow_profiles[k] == :Voigt
                    if !cube_fitter.tie_voigt_mixing
                        flow_η = popt_l[pᵢ+3]
                    else
                        flow_η = popt_l[ηᵢ]
                    end
                end
            else
                vwhere = findfirst(x -> x == cube_fitter.line_flow_tied[k], cube_fitter.flow_voff_tied_key)
                flow_voff = popt_l[cube_fitter.n_voff_tied+vwhere]
                flow_fwhm = popt_l[pᵢ+1]
                if cube_fitter.line_flow_profiles[k] == :GaussHermite
                    flow_h3 = popt_l[pᵢ+2]
                    flow_h4 = popt_l[pᵢ+3]
                elseif cube_fitter.line_flow_profiles[k] == :Voigt
                    if !cube_fitter.tie_voigt_mixing
                        flow_η = popt_l[pᵢ+2]
                    else
                        flow_η = popt_l[ηᵢ]
                    end
                end
            end

            # Convert voff in km/s to mean wavelength in μm
            flow_mean_μm = Util.Doppler_shift_λ(ln.λ₀, voff+flow_voff)
            # Convert FWHM from km/s to μm
            flow_fwhm_μm = Util.Doppler_shift_λ(ln.λ₀, flow_fwhm) - ln.λ₀
            # Evaluate line profile, shifted by the same amount as the primary line profile
            if cube_fitter.line_flow_profiles[k] == :Gaussian
                profile = let profile = profile
                    x -> profile(x) + Util.Gaussian(x, flow_amp, flow_mean_μm-mean_μm, flow_fwhm_μm)
                end
            elseif cube_fitter.line_flow_profiles[k] == :Lorentzian
                profile = let profile = profile
                    x -> profile(x) + Util.Lorentzian(x, flow_amp, flow_mean_μm-mean_μm, flow_fwhm_μm)
                end
            elseif cube_fitter.line_profiles[k] == :GaussHermite
                profile = let profile = profile
                    x -> profile(x) + Util.GaussHermite(x, flow_amp, flow_mean_μm-mean_μm, flow_fwhm_μm, flow_h3, flow_h4)
                end
            elseif cube_fitter.line_profiles[k] == :Voigt
                profile = let profile = profile
                    x -> profile(x) + Util.Voigt(x, flow_amp, flow_mean_μm-mean_μm, flow_fwhm_μm, flow_η)
                end
            else
                error("Unrecognized flow line profile $(cube_fitter.line_profiles[k])!")
            end

            # Advance the parameter vector index by the appropriate amount        
            pᵢ += isnothing(cube_fitter.line_flow_tied[k]) ? 3 : 2
            if cube_fitter.line_flow_profiles[k] == :GaussHermite
                pᵢ += 2
            elseif cube_fitter.line_flow_profiles[k] == :Voigt
                if !cube_fitter.tie_voigt_mixing
                    pᵢ += 1
                end
            end
        end

        # Add back in the normalization
        profile = let profile = profile
            x -> N * profile(x)
        end

        λ_arr = (-10fwhm_μm):Δλ:(10fwhm_μm)
        peak, peak_ind = findmax(profile.(λ_arr))

        p_lines[pₒ], _ = quadgk(profile, -Inf, Inf, order=200)

        # SNR, calculated as (amplitude) / (RMS of the surrounding spectrum)
        p_lines[pₒ+1] = peak / std(I[.!mask_lines] .- continuum[.!mask_lines])

        @debug "Line $(cube_fitter.line_names[k]) with integrated intensity $(p_lines[pₒ]) and SNR $(p_lines[pₒ+1])"

        pₒ += 2

    end

    return p_complex, p_lines
end


"""
    fit_spaxel(cube_fitter, spaxel)

Wrapper function to perform a full fit of a single spaxel, calling `continuum_fit_spaxel` and `line_fit_spaxel` and
concatenating the best-fit parameters. The outputs are also saved to files so the fit need not be repeated in the case
of a crash.

# Arguments
`S<:Integer`
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::Tuple{S,S}`: The coordinates of the spaxel to be fit
"""
function fit_spaxel(cube_fitter::CubeFitter, spaxel::Tuple{S,S})::Union{Nothing,Vector{<:AbstractFloat}} where {S<:Integer}

    # Create a local logger for this individual spaxel
    timestamp_logger(logger) = TransformerLogger(logger) do log
        merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
    end

    # This log should be entirely handled by 1 process, since each spaxel is entirely handled by 1 process
    # so there should be no problems with I/O race conditions
    logger = TeeLogger(ConsoleLogger(stdout, Logging.Info), timestamp_logger(MinLevelLogger(FileLogger(
                         joinpath("output_$(cube_fitter.name)", "logs", "loki.spaxel_$(spaxel[1])_$(spaxel[2]).log"); 
                         always_flush=true), Logging.Debug)))

    with_logger(logger) do
        # Define global variable
        global p_out

        # Check if the fit has already been performed
        if !isfile(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).LOKI")) || cube_fitter.overwrite

            # Skip spaxels with NaNs (post-interpolation)
            λ = cube_fitter.cube.λ
            I = cube_fitter.cube.Iλ[spaxel..., :]

            if any(.!isfinite.(I) .| .!isfinite.(I))
                @debug "Too many bad datapoints...skipping the fit for spaxel $spaxel"
                p_out = nothing

            else

                # Fit the spaxel
                σ, popt_c, I_cont, comps_cont, n_free_c = continuum_fit_spaxel(cube_fitter, spaxel)
                _, popt_l, I_line, comps_line, n_free_l = line_fit_spaxel(cube_fitter, spaxel)

                # Combine the continuum and line models
                I_model = I_cont .+ I_line
                comps = merge(comps_cont, comps_line)

                # Total free parameters
                n_free = n_free_c + n_free_l
                n_data = length(I)

                # Reduced chi^2 of the model
                χ2red = 1 / (n_data - n_free) * sum((I .- I_model).^2 ./ σ.^2)

                # Add dust complex and line parameters (intensity and SNR)
                p_complex, p_lines = calculate_extra_parameters(cube_fitter, spaxel, popt_c, popt_l)
                p_out = [popt_c; popt_l; p_complex; p_lines; χ2red]

                # Plot the fit
                λ0_ln = [ln.λ₀ for ln ∈ cube_fitter.lines]
                if cube_fitter.plot_spaxels != :none
                    @debug "Plotting spaxel $spaxel best fit"
                    plot_spaxel_fit(λ, I, I_model, σ, comps, 
                        cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, λ0_ln, cube_fitter.line_names, cube_fitter.z,
                        χ2red, cube_fitter.name, "spaxel_$(spaxel[1])_$(spaxel[2])", backend=cube_fitter.plot_spaxels)
                end

                # Set parameters in each row based on the previous fit, given the reduced chi^2 meets a given threshold
                row = spaxel[1]
                if χ2red ≤ cube_fitter.χ²_thresh
                    @debug "Spaxel $spaxel - reduced chi^2 of $χ2red is less than the threshold $(cube_fitter.χ²_thresh), " *
                     "new rolling best fit parameters will be set"
                    cube_fitter.p_best_cont[row, :] .= popt_c
                    cube_fitter.p_best_line[row, :] .= popt_l
                    cube_fitter.χ²_best[row] = χ2red
                    cube_fitter.best_spaxel[row] = spaxel
                end

            end

            @debug "Saving results to binary for spaxel $spaxel"
            # save output as binary file
            serialize(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).LOKI"), p_out)
            # save running best fit parameters in case the fitting is interrupted
            serialize(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "best_fit_params.LOKI"),
                Dict(:p_best_cont => Array(cube_fitter.p_best_cont),
                     :p_best_line => Array(cube_fitter.p_best_line),
                     :chi2_best => Vector(cube_fitter.χ²_best),
                     :best_spaxel => Vector(cube_fitter.best_spaxel))
                )

        # Otherwise, just grab the results from before
        else
            @debug "Loading results from binary for spaxel $spaxel"
            p_out = deserialize(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).LOKI"))

        end

    end

    return p_out
end


"""
    fit_cube(cube_fitter)

Wrapper function to perform a full fit of an entire IFU cube, calling `fit_spaxel` for each spaxel in a parallel or
serial loop depending on the cube_fitter options.  Results are concatenated and plotted/saved, also based on the
cube_fitter options.

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
"""
function fit_cube(cube_fitter::CubeFitter)::CubeFitter

    @info """\n
    #############################################################################
    ######## BEGINNING FULL CUBE FITTING ROUTINE FOR $(cube_fitter.name) ########
    #############################################################################
    """

    shape = size(cube_fitter.cube.Iλ)
    # Interpolate NaNs in the cube
    interpolate_cube!(cube_fitter.cube)

    # Prepare output array
    @info "===> Preparing output data structures... <==="
    out_params = SharedArray(ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 1) .* NaN)

    #########################
    function fit_spax_i(xᵢ::Int, yᵢ::Int)

        p_out = fit_spaxel(cube_fitter, (xᵢ, yᵢ))
        if !isnothing(p_out)
            out_params[xᵢ, yᵢ, :] .= p_out
        end

        return
    end

    # Sort spaxels by median brightness, so that we fit the brightest ones first
    # (which hopefully have the best reduced chi^2s)
    # spaxels = Iterators.product(1:shape[1], 1:shape[2])
    spaxels = Iterators.product(15:16, 15:16)

    # med_I = collect(Iterators.flatten([nanmedian(cube_fitter.cube.Iλ[spaxel..., :]) for spaxel ∈ spaxels]))
    # # replace NaNs with -1s
    # med_I[.!isfinite.(med_I)] .= -1.
    # # reverse sort
    # ss = sortperm(med_I, rev=true)
    # med_I = med_I[ss]
    # # apply sorting to spaxel indices
    # spaxels = collect(spaxels)[ss]

    @info "===> Beginning spaxel fitting... <==="
    # Use multiprocessing (not threading) to iterate over multiple spaxels at once using multiple CPUs
    if cube_fitter.parallel
        prog = Progress(length(spaxels); showspeed=true)
        progress_pmap(spaxels, progress=prog) do (xᵢ, yᵢ)
            fit_spax_i(xᵢ, yᵢ)
        end
    else
        prog = Progress(length(spaxels); showspeed=true)
        for (xᵢ, yᵢ) ∈ spaxels
            fit_spax_i(xᵢ, yᵢ)
            next!(prog)
        end
    end

    @info "===> Updating parameter maps and model cubes... <==="
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
        I_cont, comps_c = Util.fit_spectrum(cube_fitter.cube.λ, out_params[xᵢ, yᵢ, 1:pᵢ-1], cube_fitter.n_dust_cont, cube_fitter.n_dust_feat,
            cube_fitter.extinction_curve, cube_fitter.extinction_screen; return_components=true)

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
        # Tied voigt mixing
        if cube_fitter.tie_voigt_mixing
            cube_fitter.param_maps.tied_voigt_mix[xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ]
            pᵢ += 1
        end
        for (k, ln) ∈ enumerate(cube_fitter.line_names)
            # Log of line amplitude
            cube_fitter.param_maps.lines[ln][:amp][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ] > 0. ? log10(out_params[xᵢ, yᵢ, pᵢ]) : -Inf
            fwhm_res = Util.C_KMS / cube_fitter.interp_R(cube_fitter.lines[k].λ₀)

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
                elseif cube_fitter.line_profiles[k] == :Voigt && !cube_fitter.tie_voigt_mixing
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
                elseif cube_fitter.line_profiles[k] == :Voigt && !cube_fitter.tie_voigt_mixing
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
                    elseif cube_fitter.line_flow_profiles[k] == :Voigt && !cube_fitter.tie_voigt_mixing
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
                    elseif cube_fitter.line_flow_profiles[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                        cube_fitter.param_maps.lines[k][:flow_mixing][xᵢ, yᵢ] = out_params[xᵢ, yᵢ, pᵢ+2]
                        pᵢ += 1
                    end
                    pᵢ += 2
                end
            end

        end

        # End of line parameters: recreate the line model
        I_line, comps_l = Util.fit_line_residuals(cube_fitter.cube.λ, out_params[xᵢ, yᵢ, vᵢ:pᵢ-1], cube_fitter.n_lines, cube_fitter.n_voff_tied,
            cube_fitter.voff_tied_key, cube_fitter.line_tied, cube_fitter.line_profiles, cube_fitter.n_flow_voff_tied, cube_fitter.flow_voff_tied_key,
            cube_fitter.line_flow_tied, cube_fitter.line_flow_profiles, [ln.λ₀ for ln ∈ cube_fitter.lines], 
            cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing; return_components=true)

        # Renormalize
        N = Float64(abs(nanmaximum(cube_fitter.cube.Iλ[xᵢ, yᵢ, :])))
        N = N ≠ 0. ? N : 1.
        for comp ∈ keys(comps_l)
            comps_l[comp] .*= N
        end
        I_line .*= N
        
        # Combine the continuum and line models
        I_model = I_cont .+ I_line
        comps = merge(comps_c, comps_l)

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

        # Set 3D model cube outputs
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
        @info ">=== Plotting parameter maps... <==="
        plot_parameter_maps(cube_fitter)
    end

    if cube_fitter.save_fits
        @info ">=== Writing FITS outputs... <==="
        write_fits(cube_fitter)
        # println("Cleaning outputs...")
        # rm("output_$(cube_fitter.name)/spaxel_binaries", recursive=true)
    end

    @info """\n
    #############################################################################
    ################################### Done!! ##################################
    #############################################################################
    """

    return cube_fitter

end

############################## OUTPUT / SAVING FUNCTIONS ####################################


"""
    plot_parameter_map(data, name, name_i; Ω=Ω, z=z, cosmo=cosmo, snr_filter=snr_filter, snr_thresh=snr_thresh)

Plotting function for 2D parameter maps which are output by `fit_cube`

# Arguments
`T<:AbstractFloat`
- `data::Matrix{T}`: The 2D array of data to be plotted
- `name::String`: The name of the object whose fitting parameter is being plotted, i.e. "NGC_7469"
- `name_i::String`: The name of the individual parameter being plotted, i.e. "dust_features_PAH_5.24_amp"
- `Ω::Union{AbstractFloat,Nothing}=nothing`: The solid angle subtended by each pixel, in steradians (used for angular scalebar)
- `z::Union{AbstractFloat,Nothing}=nothing`: The redshift of the object (used for physical scalebar)
- `cosmo::Union{Cosmology.AbstractCosmology,Nothing}=nothing`: The cosmology to use to calculate distance for the physical scalebar
- `snr_filter::Union{Matrix{T},Nothing}=nothing`: A 2D array of S/N values to
    be used to filter out certain spaxels from being plotted
- `snr_thresh::Real=3`: The S/N threshold below which to cut out any spaxels using the values in snr_filter
"""
function plot_parameter_map(data::Matrix{T}, name::String, name_i::String;
    Ω=nothing, z=nothing, cosmo=nothing, snr_filter=nothing, snr_thresh::Real=3) where {T<:AbstractFloat}

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

    @debug "Plotting 2D map of $name_i with units $bunit"

    filtered = copy(data)
    if !isnothing(snr_filter)
        filtered[snr_filter .≤ snr_thresh] .= NaN
        @debug "Performing SNR filtering, $(sum(isfinite.(filtered)))/$(length(filtered)) passed"
    end

    fig = plt.figure()
    ax = plt.subplot()
    # Need to filter out any NaNs in order to use quantile
    flatdata = filtered[isfinite.(filtered)]
    vmin = length(flatdata) > 0 ? quantile(flatdata, 0.01) : 0.0
    vmax = length(flatdata) > 0 ? quantile(flatdata, 0.99) : 0.0
    cdata = ax.imshow(filtered', origin=:lower, cmap=:magma, vmin=vmin, vmax=vmax)
    ax.axis(:off)

    # Angular and physical scalebars
    if !isnothing(Ω) && !isnothing(z) && !isnothing(cosmo)
        n_pix = 1/(sqrt(Ω) * 180/π * 3600)
        dL = luminosity_dist(cosmo, z).val * 1e6 / (180/π * 3600)  # l = d * theta (1")
        dL = @sprintf "%.0f" dL
        scalebar = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, "1\$\'\'\$ / $dL pc", "lower left", pad=1, color=:black, 
            frameon=false, size_vertical=0.2, label_top=true)
        ax.add_artist(scalebar)
    end

    fig.colorbar(cdata, ax=ax, label=bunit)
    plt.savefig(joinpath("output_$(name)", "param_maps", "$(name_i).pdf"), dpi=300, bbox_inches=:tight)
    plt.close()

end


"""
    plot_parameter_maps(cube_fitter; snr_thresh=snr_thresh)

Wrapper function for `plot_parameter_map`, iterating through all the parameters in a `CubeFitter`'s `ParamMaps` object
and creating 2D maps of them.

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `snr_thresh::Real`: The S/N threshold to be used when filtering the parameter maps by S/N, for those applicable
"""
function plot_parameter_maps(cube_fitter::CubeFitter; snr_thresh::Real=3.)

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

    if cube_fitter.tie_voigt_mixing
        data = cube_fitter.param_maps.tied_voigt_mix
        name_i = "tied_voigt_mixing"
        plot_parameter_map(data, cube_fitter.name, name_i, Ω=cube_fitter.cube.Ω, z=cube_fitter.z, cosmo=cube_fitter.cosmology)
    end

    for vk ∈ cube_fitter.voff_tied_key
        data = cube_fitter.param_maps.tied_voffs[vk]
        name_i = join(["tied_voffs", vk], "_")
        plot_parameter_map(data, cube_fitter.name, name_i, Ω=cube_fitter.cube.Ω, z=cube_fitter.z, cosmo=cube_fitter.cosmology)
    end

    for fvk ∈ cube_fitter.flow_voff_tied_key
        data = cube_fitter.param_maps.flow_tied_voffs[fvk]
        name_i = join(["flow_tied_voffs", fvk], "_")
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


"""
    write_fits(cube_fitter)

Save the best fit results for the cube into two FITS files: one for the full 3D intensity model of the cube, split up by
individual model components, and one for 2D parameter maps of the best-fit parameters for each spaxel in the cube.

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
"""
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

    # Create the 3D intensity model FITS file
    FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_3D_model.fits"), "w") do f

        @debug "Writing 3D model FITS HDUs"

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
        
        write(f, ["wave_rest", "wave_obs"],                                                                     # 1D Rest frame and observed frame
                 [cube_fitter.cube.λ, Util.observed_frame(cube_fitter.cube.λ, cube_fitter.z)],                  # wavelength vectors
              hdutype=TableHDU, name="WAVELENGTH", units=Dict(:wave_rest => "um", :wave_obs => "um"))

        # Insert physical units into the headers of each HDU -> MegaJansky per steradian for all except
        # the extinction profile, which is a multiplicative constant
        write_key(f["DATA"], "BUNIT", "MJy/sr")
        write_key(f["MODEL"], "BUNIT", "MJy/sr")
        write_key(f["RESIDUALS"], "BUNIT", "MJy/sr")
        write_key(f["STELLAR_CONTINUUM"], "BUNIT", "MJy/sr")
        for i ∈ 1:size(cube_fitter.cube_model.dust_continuum, 4)
            write_key(f["DUST_CONTINUUM_$i"], "BUNIT", "MJy/sr")
        end
        for df ∈ cube_fitter.df_names
            write_key(f["$df"], "BUNIT", "MJy/sr")
        end
        for line ∈ cube_fitter.line_names
            write_key(f["$line"], "BUNIT", "MJy/sr")
        end
        write_key(f["EXTINCTION"], "BUNIT", "unitless")
    end

    # Create the 2D parameter map FITS file
    FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_parameter_maps.fits"), "w") do f

        @debug "Writing 2D parameter map FITS HDUs"

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

        if cube_fitter.tie_voigt_mixing
            data = cube_fitter.param_maps.tied_voigt_mix
            name_i = "tied_voigt_mixing"
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

        for fvk ∈ cube_fitter.flow_voff_tied_key
            data = cube_fitter.param_maps.flow_tied_voffs[fvk]
            name_i = join(["flow_tied_voffs", fvk], "_")
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