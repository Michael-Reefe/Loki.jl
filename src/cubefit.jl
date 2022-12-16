module CubeFit

export CubeFitter, fit_cube, continuum_fit_spaxel, line_fit_spaxel, fit_spaxel, fit_stack!, plot_parameter_maps, write_fits

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
using Unitful
using UnitfulAstro

# File I/O
using TOML
using DelimitedFiles

# Plotting packages
using PlotlyJS

# Misc packages/utilites
using ProgressMeter
using Reexport
# using Serialization
using Printf
using Logging
using LoggingExtras
using Dates
using InteractiveUtils
using TimerOutputs

# PyCall needed for anchored_artists
using PyCall
# Have to import anchored_artists within the __init__ function so that it works after precompilation
const plt::PyObject = PyNULL()
const py_anchored_artists::PyObject = PyNULL()
const py_ticker::PyObject = PyNULL()

# MATPLOTLIB SETTINGS TO MAKE PLOTS LOOK PRETTY :)
const SMALL::UInt8 = 12
const MED::UInt8 = 14
const BIG::UInt8 = 16

function __init__()
    # Import pyplot
    copy!(plt, pyimport_conda("matplotlib.pyplot", "matplotlib"))
    # Import matplotlib's anchored_artists package for scale bars
    copy!(py_anchored_artists, pyimport_conda("mpl_toolkits.axes_grid1.anchored_artists", "matplotlib"))
    copy!(py_ticker, pyimport_conda("matplotlib.ticker", "matplotlib"))

    plt.switch_backend("Agg")
    plt.rc("font", size=MED)          # controls default text sizes
    plt.rc("axes", titlesize=MED)     # fontsize of the axes title
    plt.rc("axes", labelsize=MED)     # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL)    # legend fontsize
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
const timer_output = TimerOutput()
const file_lock = ReentrantLock()


"""
    parse_resolving(z, channel)

Read in the resolving_mrs.csv configuration file to create a cubic spline interpolation of the
MIRI MRS resolving power as a function of wavelength, redshifted to the rest frame of the object
being fit.

# Arguments
- `z::AbstractFloat`: The redshift of the object to be fit
- `channel::Integer`: The channel of the fit
"""
function parse_resolving(z::AbstractFloat, channel::Integer)::Dierckx.Spline1D

    @debug "Parsing MRS resoling power from resolving_mrs.csv for channel $channel"

    # Read in the resolving power data
    resolve = readdlm(joinpath(@__DIR__, "resolving_mrs.csv"), ',', Float64, '\n', header=true)
    wave = resolve[1][:, 1]
    R = resolve[1][:, 2]

    # Find points where wavelength jumps down (b/w channels)
    jumps = diff(wave) .< 0
    indices = 1:length(wave)
    ind_left = indices[BitVector([0; jumps])]
    ind_right = indices[BitVector([jumps; 0])]

    # Channel 1: everything before jump 3
    if channel == 1
        edge_left = 1
        edge_right = ind_right[3]
    # Channel 2: between jumps 3 & 6
    elseif channel == 2
        edge_left = ind_left[3]
        edge_right = ind_right[6]
    # Channel 3: between jumps 6 & 9
    elseif channel == 3
        edge_left = ind_left[6]
        edge_right = ind_right[9]
    # Channel 4: everything above jump 9
    elseif channel == 4
        edge_left = ind_left[9]
        edge_right = length(wave)
    end

    # Filter down to the channel we want
    wave = wave[edge_left:edge_right]
    R = R[edge_left:edge_right]

    # Now get the jumps within the individual channel we're interested in
    jumps = diff(wave) .< 0

    # Define regions of overlapping wavelength space
    wave_left = wave[BitVector([0; jumps])]
    wave_right = wave[BitVector([jumps; 0])]

    # Sort the data to be monotonically increasing in wavelength
    ss = sortperm(wave)
    wave = wave[ss]
    R = R[ss]

    # Smooth the data in overlapping regions
    for i ∈ 1:sum(jumps)
        region = wave_left[i] .≤ wave .≤ wave_right[i]
        R[region] .= movmean(R, 10)[region]
    end
    # Shift to the rest frame
    wave = Util.rest_frame(wave, z)

    # Create a linear interpolation function so we can evaluate it at the points of interest for our data
    interp_R = Spline1D(wave, R, k=1)
    
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
    keylist1 = ["extinction_curve", "extinction_screen", "chi2_threshold", "overwrite", "track_memory", "track_convergence", "cosmology"]
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
    options_out[:track_memory] = options["track_memory"]
    @debug "Track memory allocations? - $(options["track_memory"])"
    options_out[:track_convergence] = options["track_convergence"]
    @debug "Track SAMIN convergence? - $(options["track_convergence"])"

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

    keylist1 = ["tie_H2_voff", "tie_ion_voff", "tie_H2_flow_voff", "tie_ion_flow_voff", "tie_voigt_mixing", 
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
        # lower bound 0 -> since this is ADDED to the main line FWHM
        flow_fwhm_prior = Uniform(0., lines["flow_fwhm_pmax"])
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
        if lines["tie_ion_voff"] && !occursin("H2", line)
            tied = "ion"
            @debug "Tying voff to the group: $tied"
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
        if lines["tie_ion_flow_voff"] && !occursin("H2", line) && !isnothing(flow_profiles[line])
            flow_tied = "ion"
            @debug "Tying flow voff to the group: $flow_tied"
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

            flow_voff = Param.Parameter(0., flow_voff_locked, flow_voff_prior)
            @debug "Voff $flow_voff"
            flow_fwhm = Param.Parameter(1., flow_fwhm_locked, flow_fwhm_prior)
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
        tied_voigt_mix, extinction, reduced_χ2)

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
    reduced_χ2::Array{Float64, 2}

end

"""
    parammaps_empty(shape, n_dust_cont, df_names, line_names, line_tied, line_profiles,
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
    line_names::Vector{Symbol}, line_tied::Vector{Union{String,Nothing}}, line_profiles::Vector{Symbol},
    line_flow_tied::Vector{Union{String,Nothing}}, line_flow_profiles::Vector{Union{Symbol,Nothing}},
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
        dust_features[n][:intI] = copy(nan_arr)
        dust_features[n][:SNR] = copy(nan_arr)
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

    # Reduced chi^2 of the fits
    reduced_χ2 = copy(nan_arr)
    @debug "reduced chi^2 map"

    return ParamMaps(stellar_continuum, dust_continuum, dust_features, lines, tied_voffs, flow_tied_voffs,
        tied_voigt_mix, extinction, reduced_χ2)
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
struct CubeFitter{T<:AbstractFloat,S<:Integer}
    
    # Data
    cube::CubeData.DataCube
    z::T
    name::String

    # Basic fitting options
    n_procs::S
    # cube_model::CubeModel
    # param_maps::ParamMaps
    # param_errs::ParamMaps
    window_size::T
    plot_spaxels::Symbol
    plot_maps::Bool
    parallel::Bool
    save_fits::Bool
    overwrite::Bool
    track_memory::Bool
    track_convergence::Bool
    extinction_curve::String
    extinction_screen::Bool

    # Continuum parameters
    T_s::Param.Parameter
    T_dc::Vector{Param.Parameter}
    τ_97::Param.Parameter
    β::Param.Parameter
    n_dust_cont::S
    n_dust_feat::S
    df_names::Vector{String}
    dust_features::Vector{Dict}

    # Line parameters
    n_lines::S
    line_names::Vector{Symbol}
    line_profiles::Vector{Symbol}
    line_flow_profiles::Vector{Union{Nothing,Symbol}}
    lines::Vector{Param.TransitionLine}

    # Tied voffs
    n_voff_tied::S
    line_tied::Vector{Union{String,Nothing}}
    voff_tied_key::Vector{String}
    voff_tied::Vector{Param.Parameter}

    # Tied inflow/outflow voffs
    n_flow_voff_tied::S
    line_flow_tied::Vector{Union{String,Nothing}}
    flow_voff_tied_key::Vector{String}
    flow_voff_tied::Vector{Param.Parameter}

    # Tied voigt mixing
    tie_voigt_mixing::Bool
    voigt_mix_tied::Param.Parameter

    # Number of parameters
    n_params_cont::S
    n_params_lines::S
    
    # Rolling best fit options
    cosmology::Cosmology.AbstractCosmology
    χ²_thresh::T
    interp_R::Union{Function,Dierckx.Spline1D}
    flexible_wavesol::Bool

    p_init_cont::Vector{T}
    p_init_line::Vector{T}

    # Constructor function
    function CubeFitter(cube::CubeData.DataCube, z::Float64, name::String, n_procs::Int; window_size::Float64=.025, 
        plot_spaxels::Symbol=:pyplot, plot_maps::Bool=true, parallel::Bool=true, save_fits::Bool=true)

        ###### SETTING UP A GLOBAL LOGGER FOR THE CUBE FITTER ######

        # Prepare output directories
        @info "Preparing output directories"
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
        @info "Preparing logger"


        timestamp_logger(logger) = TransformerLogger(logger) do log
            merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
        end

        logger = TeeLogger(ConsoleLogger(stdout, Logging.Info), 
                           timestamp_logger(MinLevelLogger(FileLogger(joinpath("output_$name", "loki.main.log"); 
                                                                      always_flush=true), 
                                                                      Logging.Debug)))

        global_logger(logger)

        #############################################################

        @debug """\n
        Creating CubeFitter struct for $name
        ####################################
        """

        # Get shape
        shape = size(cube.Iλ)
        # Alias
        λ = cube.λ

        # Parse all of the options files to create default options and parameter objects
        interp_R = parse_resolving(z, parse(Int, cube.channel))
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
        n_params_cont = (2+2) + 2n_dust_cont + 5n_dust_features
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
        @debug "### This totals to $(n_params_cont-2n_dust_features) continuum parameters ###"
        @debug "### This totals to $(n_params_lines-2n_lines) emission line parameters ###"

        # Prepare options
        extinction_curve = options[:extinction_curve]
        extinction_screen = options[:extinction_screen]
        χ²_thresh = options[:chi2_threshold]
        overwrite = options[:overwrite]
        track_memory = options[:track_memory]
        track_convergence = options[:track_convergence]
        cosmo = options[:cosmology]

        # Prepare initial best fit parameter options
        @debug "Preparing initial best fit parameter vectors with $(n_params_cont-2n_dust_features) and $(n_params_lines-2n_lines) parameters"
        p_init_cont = zeros(n_params_cont-2n_dust_features)
        p_init_line = zeros(n_params_lines-2n_lines)

        # If a fit has been run previously, read in the file containing the rolling best fit parameters
        # to pick up where the fitter left off seamlessly
        if isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv")) && isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"))
            p_init_cont = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv"), ',', Float64, '\n')[:, 1]
            p_init_line = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"), ',', Float64, '\n')[:, 1]
        end

        # if isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_parameters.LOKI"))
        #     p_init_dict = deserialize(joinpath("output_$name", "spaxel_binaries", "init_fit_parameters.LOKI"))
        #     p_init_cont = p_init_dict[:p_init_cont]
        #     p_init_line = p_init_dict[:p_init_line]
        #     χ²_init = [p_init_dict[:chi2_init]]
        # end

        return new{typeof(z), typeof(n_procs)}(cube, z, name, n_procs, window_size, plot_spaxels, plot_maps, 
            parallel, save_fits, overwrite, track_memory, track_convergence, extinction_curve, extinction_screen, 
            T_s, T_dc, τ_97, β, n_dust_cont, n_dust_features, df_names, dust_features, n_lines, line_names, line_profiles, 
            line_flow_profiles, lines, n_voff_tied, line_tied, voff_tied_key, voff_tied, n_flow_voff_tied, 
            line_flow_tied, flow_voff_tied_key, flow_voff_tied, tie_voigt_mixing, voigt_mix_tied, n_params_cont, n_params_lines, 
            cosmo, χ²_thresh, interp_R, flexible_wavesol, p_init_cont, p_init_line)
    end

end


"""
    generate_cubemodel(cube_fitter)

Generate a CubeModel object corresponding to the options given by the CubeFitter object
"""
function generate_cubemodel(cube_fitter::CubeFitter)::CubeModel
    shape = size(cube_fitter.cube.Iλ)
    # Full 3D intensity model array
    @debug "Generateing full 3D cube models"
    return cubemodel_empty(shape, cube_fitter.n_dust_cont, cube_fitter.df_names, cube_fitter.line_names)
end


"""
    generate_parammaps(cube_fitter)

Generate two ParamMaps objects (for the values and errors) corrresponding to the options given
by the CubeFitter object
"""
function generate_parammaps(cube_fitter::CubeFitter)::Tuple{ParamMaps, ParamMaps}
    shape = size(cube_fitter.cube.Iλ)
    # 2D maps of fitting parameters
    @debug "Generating 2D parameter value & error maps"
    param_maps = parammaps_empty(shape, cube_fitter.n_dust_cont, cube_fitter.df_names, cube_fitter.line_names, cube_fitter.line_tied,
                                 cube_fitter.line_profiles, cube_fitter.line_flow_tied, cube_fitter.line_flow_profiles,
                                 cube_fitter.voff_tied_key, cube_fitter.flow_voff_tied_key, cube_fitter.flexible_wavesol,
                                 cube_fitter.tie_voigt_mixing)
    # 2D maps of fitting parameter 1-sigma errors
    param_errs = parammaps_empty(shape, cube_fitter.n_dust_cont, cube_fitter.df_names, cube_fitter.line_names, cube_fitter.line_tied,
                                 cube_fitter.line_profiles, cube_fitter.line_flow_tied, cube_fitter.line_flow_profiles,
                                 cube_fitter.voff_tied_key, cube_fitter.flow_voff_tied_key, cube_fitter.flexible_wavesol,
                                 cube_fitter.tie_voigt_mixing)
    return param_maps, param_errs
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

    return mask_lines, I_out, σ_out
end


"""
    _interp_func(x, λ)

Function to interpolate the data with least squares quadratic fitting
"""
function _interp_func(x, λ, I)
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



"""
    continuum_fit_spaxel(cube_fitter, spaxel; init=init)

Fit the continuum of a given spaxel in the DataCube, masking out the emission lines, using the 
Levenberg-Marquardt least squares fitting method with the `CMPFit` package.  

This function has been adapted from PAHFIT (with some heavy adjustments -> masking out lines, allowing
PAH parameters to vary, and tying certain parameters together). See Smith, Draine, et al. 2007; 
http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::CartesianIndex`: The coordinates of the spaxel to be fit
- `init::Bool=false`: Flag for the initial fit which fits the sum of all spaxels, to get an estimation for
    the initial parameter vector for individual spaxel fits
"""
function continuum_fit_spaxel(cube_fitter::CubeFitter, spaxel::CartesianIndex; init::Bool=false) 

    @debug """\n
    #########################################################
    ###   Beginning continuum fit for spaxel $spaxel...   ###
    #########################################################
    """

    # Extract spaxel to be fit
    λ = cube_fitter.cube.λ
    I = !init ? cube_fitter.cube.Iλ[spaxel, :] : Util.Σ(cube_fitter.cube.Iλ, (1,2))
    σ = !init ? cube_fitter.cube.σI[spaxel, :] : sqrt.(Util.Σ(cube_fitter.cube.σI.^2, (1,2)))

    # Mask out emission lines so that they aren't included in the continuum fit
    mask_lines, I_cubic, σ_cubic = continuum_cubic_spline(λ, I, σ)
    # Fill in the data where the lines are with the cubic spline interpolation
    I[mask_lines] .= I_cubic[mask_lines]
    σ[mask_lines] .= σ_cubic[mask_lines]
    # Add statistical uncertainties to the systematic uncertainties in quadrature
    σ_stat = std(I .- I_cubic)
    σ .= .√(σ.^2 .+ σ_stat.^2)

    @debug "Adding statistical error of $σ_stat in quadrature"
    
    # Mean and FWHM parameters for PAH profiles
    mean_df = [cdf[:wave] for cdf ∈ cube_fitter.dust_features]
    fwhm_df = [cdf[:fwhm] for cdf ∈ cube_fitter.dust_features]

    # amp_dc_prior = amp_agn_prior = Uniform(0., 1e12)  # just set it arbitrarily large, otherwise infinity gives bad logpdfs
    # amp_df_prior = Uniform(0., maximum(I) > 0. ? maximum(I) : 1e12)

    # stellar_priors = [amp_dc_prior, cube_fitter.T_s.prior]
    # dc_priors = vcat([[amp_dc_prior, Ti.prior] for Ti ∈ cube_fitter.T_dc]...)
    # df_priors = vcat([[amp_df_prior, mi.prior, fi.prior] for (mi, fi) ∈ zip(mean_df, fwhm_df)]...)

    # priors = vcat(stellar_priors, dc_priors, df_priors, [cube_fitter.τ_97.prior, cube_fitter.β.prior])

    # Check if the cube fitter has initial fit parameters 
    if !init

        @debug "Using initial best fit continuum parameters..."

        # Set the parameters to the best parameters
        p₀ = copy(cube_fitter.p_init_cont)

        # scale all flux amplitudes by the difference in medians between spaxels
        scale = nanmedian(I) / nanmedian(Util.Σ(cube_fitter.cube.Iλ, (1,2)))
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
            p₀[pᵢ] = clamp(nanmedian(I)/2, 0., Inf)
            pᵢ += 3
        end

    # Otherwise, we estimate the initial parameters based on the data
    else

        @debug "Calculating initial starting points..."

        # Stellar amplitude
        A_s = clamp(_interp_func(5.5, λ, I) / Util.Blackbody_ν(5.5, cube_fitter.T_s.value), 0., Inf)

        # Dust feature amplitudes
        A_df = repeat([clamp(nanmedian(I)/2, 0., Inf)], cube_fitter.n_dust_feat)

        # Dust continuum amplitudes
        λ_dc = clamp.([Util.Wein(Ti.value) for Ti ∈ cube_fitter.T_dc], minimum(λ), maximum(λ))
        A_dc = clamp.([_interp_func(λ_dci, λ, I) / Util.Blackbody_ν(λ_dci, T_dci.value) for (λ_dci, T_dci) ∈ zip(λ_dc, cube_fitter.T_dc)] .* 
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
    @debug "Continuum Starting Values: \n $p₀"

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

    @debug "Continuum Parameters locked? \n $([parinfo[i].fixed for i ∈ 1:length(p₀)])"
    @debug "Continuum Lower limits: \n $([parinfo[i].limits[1] for i ∈ 1:length(p₀)])"
    @debug "Continuum Upper limits: \n $([parinfo[i].limits[2] for i ∈ 1:length(p₀)])"

    @debug "Beginning continuum fitting with Levenberg-Marquardt least squares (CMPFit):"

    res = cmpfit(λ, I, σ, (x, p) -> Util.fit_spectrum(x, p, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat,
        cube_fitter.extinction_curve, cube_fitter.extinction_screen), p₀, parinfo=parinfo, config=config)

    @debug "Continuum CMPFit status: $(res.status)"

    # Get best fit results
    popt = res.param        # Best fit parameters
    perr = res.perror       # 1-sigma uncertainties
    covar = res.covar       # Covariance matrix

    # Count free parameters
    n_free = 0
    for pᵢ ∈ 1:length(popt)
        if iszero(parinfo[pᵢ].fixed)
            n_free += 1
        end
    end

    @debug "Best fit continuum parameters: \n $popt"
    @debug "Continuum parameter errors: \n $perr"
    @debug "Continuum covariance matrix: \n $covar"

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
        cube_fitter.extinction_curve, cube_fitter.extinction_screen, true)

    msg = "######################################################################\n"
    msg *= "################# SPAXEL FIT RESULTS -- CONTINUUM ####################\n"
    msg *= "######################################################################\n"
    msg *= "\n#> STELLAR CONTINUUM <#\n"
    msg *= "Stellar_amp: \t\t\t $(@sprintf "%.3e" popt[1]) +/- $(@sprintf "%.3e" perr[1]) MJy/sr \t Limits: (0, Inf)\n"
    msg *= "Stellar_temp: \t\t\t $(@sprintf "%.0f" popt[2]) +/- $(@sprintf "%.3e" perr[2]) K \t (fixed)\n"
    pᵢ = 3
    msg *= "\n#> DUST CONTINUUM <#\n"
    for i ∈ 1:cube_fitter.n_dust_cont
        msg *= "Dust_continuum_$(i)_amp: \t\t $(@sprintf "%.3e" popt[pᵢ]) +/- $(@sprintf "%.3e" perr[pᵢ]) MJy/sr \t Limits: (0, Inf)\n"
        msg *= "Dust_continuum_$(i)_temp: \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.3e" perr[pᵢ+1]) K \t\t\t (fixed)\n"
        msg *= "\n"
        pᵢ += 2
    end
    msg *= "\n#> DUST FEATURES <#\n"
    for (j, df) ∈ enumerate(cube_fitter.df_names)
        msg *= "$(df)_amp:\t\t\t $(@sprintf "%.1f" popt[pᵢ]) +/- $(@sprintf "%.1f" perr[pᵢ]) MJy/sr \t Limits: " *
            "(0, $(@sprintf "%.1f" nanmaximum(I)))\n"
        msg *= "$(df)_mean:  \t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1]) μm \t Limits: " *
            "($(@sprintf "%.3f" minimum(mean_df[j].prior)), $(@sprintf "%.3f" maximum(mean_df[j].prior)))" * 
            (mean_df[j].locked ? " (fixed)" : "") * "\n"
        msg *= "$(df)_fwhm:  \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) μm \t Limits: " *
            "($(@sprintf "%.3f" minimum(fwhm_df[j].prior)), $(@sprintf "%.3f" maximum(fwhm_df[j].prior)))" * 
            (fwhm_df[j].locked ? " (fixed)" : "") * "\n"
        msg *= "\n"
        pᵢ += 3
    end
    msg *= "\n#> EXTINCTION <#\n"
    msg *= "τ_9.7: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t Limits: " *
        "($(@sprintf "%.2f" minimum(cube_fitter.τ_97.prior)), $(@sprintf "%.2f" maximum(cube_fitter.τ_97.prior)))" * 
        (cube_fitter.τ_97.locked ? " (fixed)" : "") * "\n"
    msg *= "β: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+1]) +/- $(@sprintf "%.2f" perr[pᵢ+1]) [-] \t Limits: " *
        "($(@sprintf "%.2f" minimum(cube_fitter.β.prior)), $(@sprintf "%.2f" maximum(cube_fitter.β.prior)))" * 
        (cube_fitter.β.locked ? " (fixed)" : "") * "\n"
    msg *= "\n"
    msg *= "######################################################################"
    @debug msg

    return σ, popt, I_model, comps, n_free, perr, covar

end


"""
    _ln_prior(p, priors)

Internal helper function to calculate the log of the prior probability for all the parameters,
specifically for the line residual fitting.
Most priors will be uniform, i.e. constant, finite prior values as long as the parameter is inbounds.
Priors will be -Inf if any parameter goes out of bounds.  
"""
function _ln_prior(p, priors)
    # sum the log prior distribution of each parameter
    lnpdf = sum([logpdf(priors[i], p[i]) for i ∈ 1:length(p)])
    return lnpdf
end


"""
    _negln_probability(p, λ, Inorm, σnorm, cube_fitter, λ0_ln, priors)

Internal helper function to calculate the negative of the log of the probability,
for the line residual fitting.

ln(probability) = ln(likelihood) + ln(prior)
"""
function _negln_probability(p, λ, Inorm, σnorm, cube_fitter, λ0_ln, priors)
    model = Util.fit_line_residuals(λ, p, cube_fitter.n_lines, cube_fitter.n_voff_tied, 
        cube_fitter.voff_tied_key, cube_fitter.line_tied, cube_fitter.line_profiles, cube_fitter.n_flow_voff_tied,
        cube_fitter.flow_voff_tied_key, cube_fitter.line_flow_tied, cube_fitter.line_flow_profiles, 
        λ0_ln, cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing)
    lnP = Util.ln_likelihood(Inorm, model, σnorm) + _ln_prior(p, priors)
    return -lnP 
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
- `spaxel::CartesianIndex`: The coordinates of the spaxel to be fit
- `init::Bool=false`: Flag for the initial fit which fits the sum of all spaxels, to get an estimation for
    the initial parameter vector for individual spaxel fits
"""
function line_fit_spaxel(cube_fitter::CubeFitter, spaxel::CartesianIndex; init::Bool=false)

    @debug """\n
    #########################################################
    ###      Beginning line fit for spaxel $spaxel...     ###
    #########################################################
    """

    # Extract spaxel to be fit
    λ = cube_fitter.cube.λ
    I = !init ? cube_fitter.cube.Iλ[spaxel, :] : Util.Σ(cube_fitter.cube.Iλ, (1,2))
    σ = !init ? cube_fitter.cube.σI[spaxel, :] : sqrt.(Util.Σ(cube_fitter.cube.σI.^2, (1,2)))

    # Perform a cubic spline continuum fit
    mask_lines, continuum, _ = continuum_cubic_spline(λ, I, σ)
    N = Float64(abs(nanmaximum(I)))
    N = N ≠ 0. ? N : 1.

    @debug "Using normalization N=$N"

    # Add statistical uncertainties to the systematic uncertainties in quadrature
    σ_stat = std(I[.!mask_lines] .- continuum[.!mask_lines])
    σ .= .√(σ.^2 .+ σ_stat.^2)

    @debug "Adding statistical error of $σ_stat in quadrature"

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
    if !init

        @debug "Using initial best fit line parameters..."

        # If so, set the parameters to the previous ones
        p₀ = copy(cube_fitter.p_init_line)

    else

        @debug "Calculating initial starting points..."

        # Start the ampltiudes at 1/2 and 1/4 (in normalized units)
        A_ln = ones(cube_fitter.n_lines) .* 0.5
        A_fl = ones(cube_fitter.n_lines) .* 0.5     # (flow amp is multiplied with main amp)

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
    @debug "Line starting values: \n $p₀"
    @debug "Line priors: \n $priors"

    lower_bounds = minimum.(priors)
    upper_bounds = maximum.(priors)

    @debug "Line Lower limits: \n $lower_bounds"
    @debug "Line Upper Limits: \n $upper_bounds"

    @debug "Beginning Line fitting with Simulated Annealing:"

    # First, perform a bounded Simulated Annealing search for the optimal parameters with a generous max iterations
    res = optimize(p -> _negln_probability(p, λ, Inorm, σnorm, cube_fitter, λ0_ln, priors), 
       lower_bounds, upper_bounds, p₀, SAMIN(;rt=0.9, nt=5, ns=5, neps=5, f_tol=1e-3/σ_stat^2, x_tol=0.01/σ_stat^2, verbosity=0), 
       Optim.Options(iterations=10^6))
    p₁ = res.minimizer

    if cube_fitter.track_convergence
        global file_lock
        lock(file_lock) do 
            open(joinpath("output_$(cube_fitter.name)", "loki.convergence.log"), "a") do conv
                redirect_stdout(conv) do
                    println("Spaxel ($(spaxel[1]),$(spaxel[2])) on worker $(myid()):")
                    println(res)
                    println("-------------------------------------------------------")
                end
            end
        end
    end

    @debug "Refining Line best fit with Levenberg-Marquardt:"

    # # Then, refine the solution with a bounded local minimum search with LBFGS
    # res = optimize(negln_probability, lower_bounds, upper_bounds, p₁, 
    #                Fminbox(BFGS()), Optim.Options(store_trace=true, extended_trace=true))

    # # Get the optimized parameter vector and number of free parameters
    # popt = res.minimizer
    # # take the inverse of the hessian to get the covariance matrix
    # covar = res.trace[end].metadata["~inv(H)"]
    # perr = .√(diag(covar))

    # n_free = length(p₀)

    ############################################# FIT WITH LEVMAR ###################################################

    # Convert parameter limits into CMPFit object
    parinfo = CMPFit.Parinfo(length(p₀))

    # Tied velocity offsets
    pᵢ = 1
    for i ∈ 1:cube_fitter.n_voff_tied
        parinfo[pᵢ].fixed = cube_fitter.voff_tied[i].locked
        if !(cube_fitter.voff_tied[i].locked)
            parinfo[pᵢ].limited = (1,1)
            parinfo[pᵢ].limits = (minimum(cube_fitter.voff_tied[i].prior), maximum(cube_fitter.voff_tied[i].prior))
        end
        pᵢ += 1
    end

    # Tied in/outflow velocity offsets
    for j ∈ 1:cube_fitter.n_flow_voff_tied
        parinfo[pᵢ].fixed = cube_fitter.flow_voff_tied[j].locked
        if !(cube_fitter.flow_voff_tied[j].locked)
            parinfo[pᵢ].limited = (1,1)
            parinfo[pᵢ].limits = (minimum(cube_fitter.flow_voff_tied[j].prior), maximum(cube_fitter.flow_voff_tied[j].prior))
        end
        pᵢ += 1
    end

    # Tied voigt mixing
    if cube_fitter.tie_voigt_mixing
        parinfo[pᵢ].fixed = cube_fitter.voigt_mix_tied.locked
        if !(cube_fitter.voigt_mix_tied.locked)
            parinfo[pᵢ].limited = (1,1)
            parinfo[pᵢ].limits = (minimum(cube_fitter.voigt_mix_tied.prior), maximum(cube_fitter.voigt_mix_tied.prior))
        end
        pᵢ += 1
    end

    # Emission line amplitude, voff, fwhm
    for i ∈ 1:cube_fitter.n_lines
        parinfo[pᵢ].limited = (1,1)
        parinfo[pᵢ].limits = (0., 1.)
        if isnothing(cube_fitter.line_tied[i]) || cube_fitter.flexible_wavesol
            parinfo[pᵢ+1].fixed = voff_ln[i].locked
            if !(voff_ln[i].locked)
                parinfo[pᵢ+1].limited = (1,1)
                parinfo[pᵢ+1].limits = (minimum(voff_ln[i].prior), maximum(voff_ln[i].prior))
            end
            line_fwhm = p₀[pᵢ+2]
            parinfo[pᵢ+2].fixed = fwhm_ln[i].locked
            if !(fwhm_ln[i].locked)
                parinfo[pᵢ+2].limited = (1,1)
                parinfo[pᵢ+2].limits = (minimum(fwhm_ln[i].prior), maximum(fwhm_ln[i].prior))
            end
            if prof_ln[i] == :GaussHermite
                parinfo[pᵢ+3].fixed = h3_ln[i].locked
                if !(h3_ln[i].locked)
                    parinfo[pᵢ+3].limited = (1,1)
                    parinfo[pᵢ+3].limits = (minimum(h3_ln[i].prior), maximum(h3_ln[i].prior))
                end
                parinfo[pᵢ+4].fixed = h4_ln[i].locked
                if !(h4_ln[i].locked)
                    parinfo[pᵢ+4].limited = (1,1)
                    parinfo[pᵢ+4].limits = (minimum(h4_ln[i].prior), maximum(h4_ln[i].prior))
                end
                pᵢ += 2
            elseif prof_ln[i] == :Voigt && !cube_fitter.tie_voigt_mixing
                parinfo[pᵢ+3].fixed = η_ln[i].locked
                if !(η_ln[i].locked)
                    parinfo[pᵢ+3].limited = (1,1)
                    parinfo[pᵢ+3].limits = (minimum(η_ln[i].prior), maximum(η_ln[i].prior))
                end
                pᵢ += 1
            end
            pᵢ += 3
        else
            line_fwhm = p₀[pᵢ+1]
            parinfo[pᵢ+1].fixed = fwhm_ln[i].locked
            if !(fwhm_ln[i].locked)
                parinfo[pᵢ+1].limited = (1,1)
                parinfo[pᵢ+1].limits = (minimum(fwhm_ln[i].prior), maximum(fwhm_ln[i].prior))
            end
            if prof_ln[i] == :GaussHermite
                parinfo[pᵢ+2].fixed = h3_ln[i].locked
                if !(h3_ln[i].locked)
                    parinfo[pᵢ+2].limited = (1,1)
                    parinfo[pᵢ+2].limits = (minimum(h3_ln[i].prior), maximum(h3_ln[i].prior))
                end
                parinfo[pᵢ+3].fixed = h4_ln[i].locked
                if !(h4_ln[i].locked)
                    parinfo[pᵢ+3].limited = (1,1)
                    parinfo[pᵢ+3].limits = (minimum(h4_ln[i].prior), maximum(h4_ln[i].prior))
                end
                pᵢ += 2       
            elseif prof_ln[i] == :Voigt && !cube_fitter.tie_voigt_mixing
                parinfo[pᵢ+2].fixed = η_ln[i].locked
                if !(η_ln[i].locked)
                    parinfo[pᵢ+2].limited = (1,1)
                    parinfo[pᵢ+2].limits = (minimum(η_ln[i].prior), maximum(η_ln[i].prior))
                end
                pᵢ += 1
            end
            pᵢ += 2
        end
        if !isnothing(flow_prof_ln[i])
            parinfo[pᵢ].limited = (1,1)
            parinfo[pᵢ].limits = (0., 1.)
            if isnothing(cube_fitter.line_flow_tied[i])
                parinfo[pᵢ+1].fixed = flow_voff_ln[i].locked
                if !(flow_voff_ln[i].locked)
                    parinfo[pᵢ+1].limited = (1,1)
                    parinfo[pᵢ+1].limits = (minimum(flow_voff_ln[i].prior), maximum(flow_voff_ln[i].prior))
                end
                parinfo[pᵢ+2].fixed = flow_fwhm_ln[i].locked
                if !(flow_fwhm_ln[i].locked)
                    parinfo[pᵢ+2].limited = (1,1)
                    parinfo[pᵢ+2].limits = (minimum(flow_fwhm_ln[i].prior), maximum(flow_fwhm_ln[i].prior))
                end
                if flow_prof_ln[i] == :GaussHermite
                    parinfo[pᵢ+3].fixed = flow_h3_ln[i].locked
                    if !(flow_h3_ln[i].locked)
                        parinfo[pᵢ+3].limited = (1,1)
                        parinfo[pᵢ+3].limits = (minimum(flow_h3_ln[i].prior), maximum(flow_h3_ln[i].prior))
                    end
                    parinfo[pᵢ+4].fixed = flow_h4_ln[i].locked
                    if !(flow_h4_ln[i].locked)
                        parinfo[pᵢ+4].limited = (1,1)
                        parinfo[pᵢ+4].limits = (minimum(flow_h4_ln[i].prior), maximum(flow_h4_ln[i].prior))
                    end
                    pᵢ += 2
                elseif flow_prof_ln[i] == :Voigt && !cube_fitter.tie_voigt_mixing
                    parinfo[pᵢ+3].fixed = flow_η_ln[i].locked
                    if !(flow_η_ln[i].locked)
                        parinfo[pᵢ+3].limited = (1,1)
                        parinfo[pᵢ+3].limits = (minimum(flow_η_ln[i].prior), maximum(flow_η_ln[i].prior))
                    end
                    pᵢ += 1
                end
                pᵢ += 3
            else
                parinfo[pᵢ+1].fixed = flow_fwhm_ln[i].locked
                if !(flow_fwhm_ln[i].locked)
                    parinfo[pᵢ+1].limited = (1,1)
                    parinfo[pᵢ+1].limits = (minimum(flow_fwhm_ln[i].prior), maximum(flow_fwhm_ln[i].prior))
                end
                if flow_prof_ln[i] == :GaussHermite
                    parinfo[pᵢ+2].fixed = flow_h3_ln[i].locked
                    if !(flow_h3_ln[i].locked)
                        parinfo[pᵢ+2].limited = (1,1)
                        parinfo[pᵢ+2].limits = (minimum(flow_h3_ln[i].prior), maximum(flow_h3_ln[i].prior))
                    end
                    parinfo[pᵢ+3].fixed = flow_h4_ln[i].locked
                    if !(flow_h4_ln[i].locked)
                        parinfo[pᵢ+3].limited = (1,1)
                        parinfo[pᵢ+3].limits = (minimum(flow_h4_ln[i].prior), maximum(flow_h4_ln[i].prior))
                    end
                    pᵢ += 2       
                elseif flow_prof_ln[i] == :Voigt && !cube_fitter.tie_voigt_mixing
                    parinfo[pᵢ+2].fixed = flow_η_ln[i].locked
                    if !(flow_η_ln[i].locked)
                        parinfo[pᵢ+2].limited = (1,1)
                        parinfo[pᵢ+2].limits = (minimum(flow_η_ln[i].prior), maximum(flow_η_ln[i].prior))
                    end
                    pᵢ += 1
                end
                pᵢ += 2
            end
        end
    end

    # Create a `config` structure
    config = CMPFit.Config()

    res = cmpfit(λ, Inorm, σnorm, (x, p) -> Util.fit_line_residuals(x, p, cube_fitter.n_lines, cube_fitter.n_voff_tied, 
        cube_fitter.voff_tied_key, cube_fitter.line_tied, prof_ln, cube_fitter.n_flow_voff_tied, cube_fitter.flow_voff_tied_key,
        cube_fitter.line_flow_tied, flow_prof_ln, λ0_ln, cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing), p₁, 
        parinfo=parinfo, config=config)


    # Get the results
    popt = res.param
    perr = res.perror
    covar = res.covar

    # Count free parameters
    n_free = 0
    for pᵢ ∈ 1:length(popt)
        if iszero(parinfo[pᵢ].fixed)
            n_free += 1
        end
    end

    ######################################################################################################################

    @debug "Best fit line parameters: \n $popt"
    @debug "Line parameter errors: \n $perr"
    @debug "Line covariance matrix: \n $covar"

    # Final optimized fit
    I_model, comps = Util.fit_line_residuals(λ, popt, cube_fitter.n_lines, cube_fitter.n_voff_tied, 
        cube_fitter.voff_tied_key, cube_fitter.line_tied, prof_ln, cube_fitter.n_flow_voff_tied,
        cube_fitter.flow_voff_tied_key, cube_fitter.line_flow_tied, flow_prof_ln, λ0_ln, 
        cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing, true)
    
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
        msg *= "$(vk)_tied_voff: \t\t\t $(@sprintf "%.0f" popt[pᵢ]) +/- $(@sprintf "%.0f" perr[pᵢ]) km/s \t " *
            "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.voff_tied[i].prior)), $(@sprintf "%.0f" maximum(cube_fitter.voff_tied[i].prior)))\n"
        pᵢ += 1
    end
    for (j, fvk) ∈ enumerate(cube_fitter.flow_voff_tied_key)
        msg *= "$(fvk)_flow_tied_voff:\t\t\t $(@sprintf "%.0f" popt[pᵢ]) +/- $(@sprintf "%.0f" perr[pᵢ]) km/s \t " *
            "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.flow_voff_tied[j].prior)), $(@sprintf "%.0f" maximum(cube_fitter.flow_voff_tied[j].prior)))\n"
        pᵢ += 1
    end
    msg *= "\n#> TIED VOIGT MIXING <#\n"
    if cube_fitter.tie_voigt_mixing
        msg *= "tied_voigt_mixing: \t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t " * 
            "Limits: ($(@sprintf "%.2f" minimum(cube_fitter.voigt_mix_tied.prior)), $(@sprintf "%.2f" maximum(cube_fitter.voigt_mix_tied.prior)))\n"
        pᵢ += 1
    end
    msg *= "\n#> EMISSION LINES <#\n"
    for (k, (ln, nm)) ∈ enumerate(zip(cube_fitter.lines, cube_fitter.line_names))
        msg *= "$(nm)_amp:\t\t\t $(@sprintf "%.3f" popt[pᵢ]) +/- $(@sprintf "%.3f" perr[pᵢ]) [x norm] \t Limits: (0, 1)\n"
        if isnothing(cube_fitter.line_tied[k]) || cube_fitter.flexible_wavesol
            msg *= "$(nm)_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(voff_ln[k].prior)), $(@sprintf "%.0f" maximum(voff_ln[k].prior)))\n"
            msg *= "$(nm)_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+2]) +/- $(@sprintf "%.0f" perr[pᵢ+2]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))\n"
            if prof_ln[k] == :GaussHermite
                msg *= "$(nm)_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))\n"
                msg *= "$(nm)_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+4]) +/- $(@sprintf "%.3f" perr[pᵢ+4])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))\n"
                pᵢ += 2
            elseif prof_ln[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                msg *= "$(nm)_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(η_ln[k].prior)), $(@sprintf "%.3f" maximum(η_ln[k].prior)))\n"
                pᵢ += 1
            end
            pᵢ += 3
        else
            msg *= "$(nm)_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))\n"
            if prof_ln[k] == :GaussHermite
                msg *= "$(nm)_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))\n"
                msg *= "$(nm)_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))\n"
                pᵢ += 2
            elseif prof_ln[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                msg *= "$(nm)_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(η_ln[k].prior)), $(@sprintf "%.3f" maximum(η_ln[k].prior)))\n"
                pᵢ += 1
            end
            pᵢ += 2
        end
        if !isnothing(flow_prof_ln[k])
            msg *= "\n$(nm)_flow_amp:\t\t\t $(@sprintf "%.3f" popt[pᵢ]) +/- $(@sprintf "%.3f" perr[pᵢ]) [x amp] \t Limits: (0, 1)\n"
            if isnothing(cube_fitter.line_flow_tied[k])
                msg *= "$(nm)_flow_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) [+ voff] \t " *
                    "Limits: ($(@sprintf "%.0f" minimum(voff_ln[k].prior)), $(@sprintf "%.0f" maximum(voff_ln[k].prior)))\n"
                msg *= "$(nm)_flow_fwhm:   \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) [x fwhm] \t " *
                    "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))\n"
                if flow_prof_ln[k] == :GaussHermite
                    msg *= "$(nm)_flow_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))\n"
                    msg *= "$(nm)_flow_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+4]) +/- $(@sprintf "%.3f" perr[pᵢ+4])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))\n"
                    pᵢ += 2
                elseif flow_prof_ln[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                    msg *= "$(nm)_flow_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])       \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(η_ln[k].prior)), $(@sprintf "%.3f" maximum(η_ln[k].prior)))\n"
                    pᵢ += 1
                end
                pᵢ += 3
            else
                msg *= "$(nm)_flow_fwhm:   \t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1]) [x fwhm] \t " *
                    "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))\n"
                if flow_prof_ln[k] == :GaussHermite
                    msg *= "$(nm)_flow_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))\n"
                    msg *= "$(nm)_flow_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))\n"
                    pᵢ += 2
                elseif flow_prof_ln[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                    msg *= "$(nm)_flow_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2])      \t " *
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

    return σ, popt, I_model, comps, n_free, perr, covar

end


"""
    plot_spaxel_fit(λ, I, I_cont, σ, comps, n_dust_cont, n_dust_features, line_wave, line_names, screen, z, χ2red, 
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
- `screen::Bool`: The type of model used for extinction screening
- `z::AbstractFloat`: The redshift of the object being fit
- `χ2red::AbstractFloat`: The reduced χ^2 value of the fit
- `name::String`: The name of the object being fit
- `label::String`: A label for the individual spaxel being plotted, to be put in the file name
- `backend::Symbol`: The backend to use to plot, either `:pyplot` or `:plotly`
"""
function plot_spaxel_fit(λ::Vector{<:AbstractFloat}, I::Vector{<:AbstractFloat}, I_cont::Vector{<:AbstractFloat}, 
    σ::Vector{<:AbstractFloat}, comps::Dict{String, Vector{T}}, n_dust_cont::Integer, n_dust_features::Integer, 
    line_wave::Vector{<:AbstractFloat}, line_names::Vector{Symbol}, screen::Bool, z::AbstractFloat, χ2red::AbstractFloat, 
    name::String, label::String; backend::Symbol=:pyplot) where {T<:AbstractFloat}

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

        # Normalize if above 10^4
        power = floor(Int, log10(maximum(I)))
        if power ≥ 4
            norm = 10^power
        else
            norm = 1
        end

        # Set up subplots with gridspec
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(nrows=4, ncols=1, hspace=0.)
        # main plot
        ax1 = fig.add_subplot(py"$(gs)[:-1, :]")
        # residuals plot
        ax2 = fig.add_subplot(py"$(gs)[-1, :]")
        ax1.plot(λ, I ./ norm, "k-", label="Data")
        ax1.plot(λ, I_cont ./ norm, "r-", label="Model")
        ax2.plot(λ, (I.-I_cont) ./ norm, "k-")
        ax2.plot(λ, zeros(length(λ)), "r-", label="\$\\tilde{\\chi}^2 = $(@sprintf "%.3f" χ2red)\$")
        ax2.fill_between(λ, σ./norm, .-σ./norm, color="k", alpha=0.5)
        # twin axes with different labels
        ax3 = ax1.twinx()
        ax4 = ax1.twiny()
        # plot individual model components
        for comp ∈ keys(comps)
            if comp == "extinction"
                ax3.plot(λ, comps[comp], "k--", alpha=0.5)
            elseif comp == "stellar"
                ax1.plot(λ, comps[comp] .* comps["extinction"] ./ norm, "r--", alpha=0.5)
            elseif occursin("dust_cont", comp)
                ax1.plot(λ, comps[comp] .* comps["extinction"] ./ norm, "g--", alpha=0.5)
            elseif occursin("dust_feat", comp)
                ax1.plot(λ, comps[comp] .* comps["extinction"] ./ norm, "b-", alpha=0.5)
            elseif occursin("line", comp)
                if occursin("flow", comp)
                    ax1.plot(λ, comps[comp] .* comps["extinction"] ./ norm, "-", color="#F574F9", alpha=0.5)
                else
                    ax1.plot(λ, comps[comp] .* comps["extinction"] ./ norm, "-", color=:rebeccapurple, alpha=0.5)
                end
            end
        end
        # plot vertical dashed lines for emission line wavelengths
        for (lw, ln) ∈ zip(line_wave, line_names)
            ax1.axvline(lw, linestyle="--", 
                color=occursin("H2", String(ln)) ? :red : (any(occursin.(["alpha", "beta", "gamma", "delta"], String(ln))) ? "#ff7f0e" : :rebeccapurple), lw=0.5, alpha=0.5)
            ax2.axvline(lw, linestyle="--", 
                color=occursin("H2", String(ln)) ? :red : (any(occursin.(["alpha", "beta", "gamma", "delta"], String(ln))) ? "#ff7f0e" : :rebeccapurple), lw=0.5, alpha=0.5)
        end
        # full continuum
        ax1.plot(λ, comps["extinction"] .* (sum([comps["dust_cont_$i"] for i ∈ 1:n_dust_cont], dims=1)[1] .+ comps["stellar"]) ./ norm, "g-")
        # set axes limits and labels
        ax1.set_xlim(minimum(λ), maximum(λ))
        ax2.set_xlim(minimum(λ), maximum(λ))
        ax2.set_ylim(-1.1maximum((I.-I_cont) ./ norm), 1.1maximum((I.-I_cont) ./ norm))
        ax3.set_ylim(0., 1.1)
        if screen
            ax3.set_ylabel("\$ e^{-\\tau_{\\lambda}} \$")
        else
            ax3.set_ylabel("\$ (1-e^{-\\tau_{\\lambda}}) / \\tau_{\\lambda} \$")
        end
        ax4.set_xlim(minimum(Util.observed_frame(λ, z)), maximum(Util.observed_frame(λ, z)))
        if power ≥ 4
            ax1.set_ylabel("\$ I_{\\nu} \$ (\$10^{$power}\$ MJy sr\$^{-1}\$)")
        else
            ax1.set_ylabel("\$ I_{\\nu} \$ (MJy sr\$^{-1}\$)")
        end
        ax1.set_ylim(bottom=0.)
        ax2.set_ylabel("\$ O-C \$")
        ax2.set_xlabel("\$ \\lambda_{\\rm rest} \$ (\$\\mu\$m)")
        ax4.set_xlabel("\$ \\lambda_{\\rm obs} \$ (\$\\mu\$m)")
        ax2.legend(loc="upper left")

        # Set minor ticks as multiples of 0.1 μm and auto for y axis
        ax1.xaxis.set_minor_locator(py_ticker.MultipleLocator(0.1))
        ax1.yaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax2.xaxis.set_minor_locator(py_ticker.MultipleLocator(0.1))
        ax2.yaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax3.yaxis.set_minor_locator(py_ticker.MultipleLocator(0.1))
        ax4.xaxis.set_minor_locator(py_ticker.MultipleLocator(0.1))

        # Set major ticks and formats
        ax1.set_xticklabels([]) # ---> will be covered up by the residuals plot
        ax2.set_yticks([-round(maximum((I.-I_cont) ./ norm) / 2, sigdigits=1), 0.0, round(maximum((I.-I_cont) ./ norm) / 2, sigdigits=1)])
        ax1.tick_params(which="both", axis="both", direction="in")
        ax2.tick_params(which="both", axis="both", direction="in", labelright=true, right=true, top=true)
        ax3.tick_params(which="both", axis="both", direction="in")
        ax4.tick_params(which="both", axis="both", direction="in")
        
        # Save figure as PDF
        plt.savefig(isnothing(label) ? joinpath("output_$name", "spaxel_plots", "levmar_fit_spaxel.pdf") : 
            joinpath("output_$name", "spaxel_plots", "$label.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    end
end


"""
    calculate_extra_parameters(cube_fitter, spaxel, comps)

Calculate extra parameters that are not fit, but are nevertheless important to know, for a given spaxel.
Currently this includes the integrated intensity and signal to noise ratios of dust features and emission lines.

# Arguments
`T<:AbstractFloat`
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::CartesianIndex`: The coordinates of the spaxel to be fit
- `popt_c::Vector{T}`: The best-bit parameter vector for the continuum components of the fit
- `popt_l::Vector{T}`: The best-fit parameter vector for the line components of the fit
"""
function calculate_extra_parameters(cube_fitter::CubeFitter, spaxel::CartesianIndex, 
    popt_c::Vector{T}, popt_l::Vector{T}, perr_c::Vector{T}, perr_l::Vector{T}) where {T<:AbstractFloat}

    @debug "Calculating extra parameters"

    # Extract the wavelength, intensity, and uncertainty data
    λ = cube_fitter.cube.λ
    I = cube_fitter.cube.Iλ[spaxel, :]
    σ = cube_fitter.cube.σI[spaxel, :]

    Δλ = mean(diff(λ)) / 10

    # Perform a cubic spline fit to the continuum, masking out lines
    mask_lines, continuum, _ = continuum_cubic_spline(λ, I, σ)
    N = Float64(abs(nanmaximum(I)))
    N = N ≠ 0. ? N : 1.
    @debug "Normalization: $N"

    # Loop through dust features
    p_dust = zeros(2cube_fitter.n_dust_feat)
    p_dust_err = zeros(2cube_fitter.n_dust_feat)
    pₒ = 1
    # Initial parameter vector index where dust profiles start
    pᵢ = 3 + 2cube_fitter.n_dust_cont

    for (ii, df) ∈ enumerate(cube_fitter.dust_features)

        # unpack the parameters
        A, μ, fwhm = popt_c[pᵢ:pᵢ+2]
        A_err, μ_err, fwhm_err = perr_c[pᵢ:pᵢ+2]
        # Convert peak intensity to CGS units (erg s^-1 cm^-2 μm^-1 sr^-1)
        A_cgs = Util.MJysr_to_cgs(A, μ)
        # Convert the error in the intensity to CGS units
        A_cgs_err = Util.MJysr_to_cgs_err(A, A_err, μ, μ_err)
        # add the integral of the individual Drude profiles using the helper function
        # (integral = π/2 * A * fwhm)
        intensity = Util.∫Drude(A_cgs, fwhm)
        # get the error of the integrated intensity
        if A_cgs == 0.
            i_err = π/2 * fwhm * A_cgs_err
        else
            frac_err2 = (A_cgs_err / A_cgs)^2 + (fwhm_err / fwhm)^2
            i_err = √(frac_err2 * intensity^2)
        end

        @debug "Drude profile with ($A_cgs, $μ, $fwhm) and errors ($A_cgs_err, $μ_err, $fwhm_err)"
        @debug "I=$intensity, err=$i_err"

        # add to profile recursively
        # profile = x -> Util.Drude(x, A, μ, fwhm)

        # increment the parameter index
        pᵢ += 3

        # intensity units: erg s^-1 cm^-2 sr^-1 (integrated over μm)
        p_dust[pₒ] = intensity
        # add errors in quadrature
        p_dust_err[pₒ] = √(sum(i_err.^2))

        # SNR, calculated as (amplitude) / (RMS of the surrounding spectrum)
        p_dust[pₒ+1] = A / std(I[.!mask_lines] .- continuum[.!mask_lines])
        @debug "Dust feature $df with integrated intensity $(p_dust[pₒ]) +/- $(p_dust_err[pₒ]) " *
            "(erg s^-1 cm^-2 sr^-1) and SNR $(p_dust[pₒ+1])"

        pₒ += 2
    end

    # Loop through lines
    p_lines = zeros(2cube_fitter.n_lines)
    p_lines_err = zeros(2cube_fitter.n_lines)
    pₒ = 1
    # Skip over the tied velocity offsets
    pᵢ = cube_fitter.n_voff_tied + cube_fitter.n_flow_voff_tied + 1
    # Skip over the tied voigt mixing parameter, saving its index
    if cube_fitter.tie_voigt_mixing
        ηᵢ = pᵢ
        pᵢ += 1
    end
    for (k, ln) ∈ enumerate(cube_fitter.lines)

        # Start with 0 intensity -> intensity holds the overall integrated intensity in CGS units
        intensity = 0. 
        i_err = []
        # (\/ pretty much the same as the fit_line_residuals function, but calculating the integrated intensities)
        amp = popt_l[pᵢ]
        amp_err = perr_l[pᵢ]
            
        # Check if voff is tied: if so, use the tied voff parameter, otherwise, use the line's own voff parameter
        if isnothing(cube_fitter.line_tied[k])
            # Unpack the components of the line
            voff = popt_l[pᵢ+1]
            voff_err = perr_l[pᵢ+1]
            fwhm = popt_l[pᵢ+2]
            fwhm_err = perr_l[pᵢ+2]
            if cube_fitter.line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = popt_l[pᵢ+3]
                h3_err = perr_l[pᵢ+3]
                h4 = popt_l[pᵢ+4]
                h4_err = perr_l[pᵢ+4]
            elseif cube_fitter.line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                if !cube_fitter.tie_voigt_mixing
                    η = popt_l[pᵢ+3]
                    η_err = perr_l[pᵢ+3]
                else
                    η = popt_l[ηᵢ]
                    η_err = perr_l[ηᵢ]
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
            voff_err = √(perr_l[vwhere]^2 + perr_l[pᵢ+1]^2)
            fwhm = popt_l[pᵢ+2]
            fwhm_err = perr_l[pᵢ+2]
            if cube_fitter.line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = popt_l[pᵢ+3]
                h3_err = perr_l[pᵢ+3]
                h4 = popt_l[pᵢ+4]
                h4_err = perr_l[pᵢ+4]
            elseif cube_fitter.line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                if !cube_fitter.tie_voigt_mixing
                    η = popt_l[pᵢ+3]
                    η_err = perr_l[pᵢ+3]
                else
                    η = popt_l[ηᵢ]
                    η_err = perr_l[ηᵢ]
                end
            end
        else
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and voff_tied_key
            vwhere = findfirst(x -> x == cube_fitter.line_tied[k], cube_fitter.voff_tied_key)
            voff = popt_l[vwhere]
            voff_err = perr_l[vwhere]
            fwhm = popt_l[pᵢ+1]
            fwhm_err = perr_l[pᵢ+1]
            # (dont add any individual voff components)
            if cube_fitter.line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = popt_l[pᵢ+2]
                h3_err = perr_l[pᵢ+2]
                h4 = popt_l[pᵢ+3]
                h4_err = perr_l[pᵢ+3]
            elseif cube_fitter.line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                if !cube_fitter.tie_voigt_mixing
                    η = popt_l[pᵢ+2]
                    η_err = perr_l[pᵢ+2]
                else
                    η = popt_l[ηᵢ]
                    η_err = perr_l[ηᵢ]
                end
            end
        end

        # Convert voff in km/s to mean wavelength in μm
        mean_μm = Util.Doppler_shift_λ(ln.λ₀, voff)
        mean_μm_err = ln.λ₀ / Util.C_KMS * voff_err
        # WARNING:
        # Set to 0 if using flexible tied voffs since they are highly degenerate and result in massive errors
        # if !isnothing(cube_fitter.line_tied[k]) && cube_fitter.flexible_wavesol
        #     mean_μm_err = 0.
        # end
        # Convert FWHM from km/s to μm
        fwhm_μm = Util.Doppler_shift_λ(ln.λ₀, fwhm) - ln.λ₀
        fwhm_μm_err = ln.λ₀ / Util.C_KMS * fwhm_err

        # Convert amplitude to erg s^-1 cm^-2 μm^-1 sr^-1, put back in the normalization
        amp_cgs = Util.MJysr_to_cgs(amp*N, mean_μm)
        amp_cgs_err = Util.MJysr_to_cgs_err(amp*N, amp_err*N, mean_μm, mean_μm_err)

        @debug "Line with ($amp_cgs, $mean_μm, $fwhm_μm) and errors ($amp_cgs_err, $mean_μm_err, $fwhm_μm_err)"

        # Evaluate the line profiles according to whether there is a simple analytic form
        # otherwise, integrate numerically with quadgk
        if cube_fitter.line_profiles[k] == :Gaussian
            ii = Util.∫Gaussian(amp_cgs, fwhm_μm)
            intensity += ii
            frac_err2 = (amp_cgs_err / amp_cgs)^2 + (fwhm_μm_err / fwhm_μm)^2
            err = √(frac_err2 * ii^2)
            append!(i_err, [err])
            @debug "I=$ii, err=$err"

            profile = x -> Util.Gaussian(x, amp, 0., fwhm_μm)
        elseif cube_fitter.line_profiles[k] == :Lorentzian
            ii = Util.∫Lorentzian(amp_cgs)
            intensity += ii
            frac_err2 = (amp_cgs_err / amp_cgs)^2
            err = √(frac_err2 * ii^2)
            append!(i_err, [err])
            @debug "I=$ii, err=$err"

            profile = x -> Util.Lorentzian(x, amp, 0., fwhm_μm)
        elseif cube_fitter.line_profiles[k] == :GaussHermite
            # shift the profile to be centered at 0 since it doesnt matter for the integral, and it makes it
            # easier for quadgk to find a solution
            ii = quadgk(x -> Util.GaussHermite(x, amp_cgs, 0., fwhm_μm, h3, h4), -Inf, Inf, order=200)[1]
            intensity += ii
            # estimate error by evaluating the integral at +/- 1 sigma
            err_l = ii - quadgk(x -> Util.GaussHermite(x, amp_cgs-amp_cgs_err, 0., fwhm_μm-fwhm_μm_err, h3-h3_err, h4-h4_err), -Inf, Inf, order=200)[1]
            err_u = quadgk(x -> Util.GaussHermite(x, amp_cgs+amp_cgs_err, 0., fwhm_μm+fwhm_μm_err, h3+h3_err, h4+h4_err), -Inf, Inf, order=200)[1] - ii
            err = mean([err_l ≥ 0 ? err_l : 0., err_u])
            append!(i_err, [err])
            @debug "I=$ii, err_u=$err_u, err_l=$err_l, err=$err"

            profile = x -> Util.GaussHermite(x, amp, 0., fwhm_μm, h3, h4)
        elseif cube_fitter.line_profiles[k] == :Voigt
            # also use a high order to ensure that all the initial test points dont evaluate to precisely 0
            ii = quadgk(x -> Util.Voigt(x, amp_cgs, 0., fwhm_μm, η), -Inf, Inf, order=200)[1]
            intensity += ii
            # estimate error by evaluating the integral at +/- 1 sigma
            err_l = ii - quadgk(x -> Util.Voigt(x, amp_cgs-amp_cgs_err, 0., fwhm_μm-fwhm_μm_err, η-η_err), -Inf, Inf, order=200)[1]
            err_u = quadgk(x -> Util.Voigt(x, amp_cgs+amp_cgs_err, 0., fwhm_μm+fwhm_μm_err, η+η_err), -Inf, Inf, order=200)[1] - ii
            err = mean([err_l ≥ 0 ? err_l : 0., err_u])
            append!(i_err, [err])
            @debug "I=$ii, err_u=$err_u, err_l=$err_l, err=$err"

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

            flow_amp = amp * popt_l[pᵢ]
            flow_amp_err = √(flow_amp^2 * ((amp_err / amp)^2 + (perr_l[pᵢ] / flow_amp)^2))
            if isnothing(cube_fitter.line_flow_tied[k])
                flow_voff = voff + popt_l[pᵢ+1]
                flow_voff_err = √(voff_err^2 + perr_l[pᵢ+1]^2)
                flow_fwhm = fwhm * popt_l[pᵢ+2]
                flow_fwhm_err = √(flow_fwhm^2 * ((fwhm_err / fwhm)^2 + (perr_l[pᵢ+2] / flow_fwhm)^2))
                if cube_fitter.line_flow_profiles[k] == :GaussHermite
                    flow_h3 = popt_l[pᵢ+3]
                    flow_h3_err = perr_l[pᵢ+3]
                    flow_h4 = popt_l[pᵢ+4]
                    flow_h4_err = perr_l[pᵢ+4]
                elseif cube_fitter.line_flow_profiles[k] == :Voigt
                    if !cube_fitter.tie_voigt_mixing
                        flow_η = popt_l[pᵢ+3]
                        flow_η_err = perr_l[pᵢ+3]
                    else
                        flow_η = popt_l[ηᵢ]
                        flow_η_err = perr_l[ηᵢ+3]
                    end
                end
            else
                vwhere = findfirst(x -> x == cube_fitter.line_flow_tied[k], cube_fitter.flow_voff_tied_key)
                flow_voff = voff + popt_l[cube_fitter.n_voff_tied+vwhere]
                flow_voff_err = √(voff_err^2 + perr_l[cube_fitter.n_voff_tied+vwhere]^2)
                flow_fwhm = fwhm * popt_l[pᵢ+1]
                flow_fwhm_err = √(flow_fwhm^2 * ((fwhm_err / fwhm)^2 + (perr_l[pᵢ+1] / flow_fwhm)^2))
                if cube_fitter.line_flow_profiles[k] == :GaussHermite
                    flow_h3 = popt_l[pᵢ+2]
                    flow_h3_err = perr_l[pᵢ+2]
                    flow_h4 = popt_l[pᵢ+3]
                    flow_h4_err = perr_l[pᵢ+3]
                elseif cube_fitter.line_flow_profiles[k] == :Voigt
                    if !cube_fitter.tie_voigt_mixing
                        flow_η = popt_l[pᵢ+2]
                        flow_η_err = perr_l[pᵢ+2]
                    else
                        flow_η = popt_l[ηᵢ]
                        flow_η_err = perr_l[ηᵢ]
                    end
                end
            end

            # Convert voff in km/s to mean wavelength in μm
            flow_mean_μm = Util.Doppler_shift_λ(ln.λ₀, flow_voff)
            flow_mean_μm_err = ln.λ₀ / Util.C_KMS * flow_voff_err
            # Convert FWHM from km/s to μm
            flow_fwhm_μm = Util.Doppler_shift_λ(ln.λ₀, flow_fwhm) - ln.λ₀
            flow_fwhm_μm_err = ln.λ₀ / Util.C_KMS * flow_fwhm_err

            # Convert amplitude to erg s^-1 cm^-2 μm^-1 sr^-1, put back in the normalization
            flow_amp_cgs = Util.MJysr_to_cgs(flow_amp*N, flow_mean_μm)
            flow_amp_cgs_err = Util.MJysr_to_cgs_err(flow_amp*N, flow_amp_err*N, flow_mean_μm, flow_mean_μm_err)

            @debug "Flow line with ($flow_amp_cgs, $flow_mean_μm, $flow_fwhm_μm) and errors ($flow_amp_cgs_err, $flow_mean_μm_err, $flow_fwhm_μm_err)"

            # Evaluate line profile, shifted by the same amount as the primary line profile
            if cube_fitter.line_flow_profiles[k] == :Gaussian
                ii = Util.∫Gaussian(flow_amp_cgs, flow_fwhm_μm)
                intensity += ii
                frac_err2 = (flow_amp_cgs_err / flow_amp_cgs)^2 + (flow_fwhm_μm_err / flow_fwhm_μm)^2
                err = √(frac_err2 * ii^2)
                # warning: flow component degeneracy with the main line components causes large errors
                append!(i_err, [err])
                @debug "I=$ii, err=$err"

                profile = let profile = profile
                    x -> profile(x) + Util.Gaussian(x, flow_amp, flow_mean_μm-mean_μm, flow_fwhm_μm)
                end
            elseif cube_fitter.line_flow_profiles[k] == :Lorentzian
                ii = Util.∫Lorentzian(flow_amp_cgs)
                intensity += ii
                frac_err2 = (flow_amp_cgs_err / flow_amp_cgs)^2
                err = √(frac_err2 * ii^2)
                append!(i_err, [err])
                @debug "I=$ii, err=$err"

                profile = let profile = profile
                    x -> profile(x) + Util.Lorentzian(x, flow_amp, flow_mean_μm-mean_μm, flow_fwhm_μm)
                end
            elseif cube_fitter.line_profiles[k] == :GaussHermite
                # same as above
                ii = quadgk(x -> Util.GaussHermite(x, flow_amp_cgs, 0., flow_fwhm_μm, flow_h3, flow_h4), -Inf, Inf, order=200)[1]
                intensity += ii
                err_l = ii - quadgk(x -> Util.GaussHermite(x, flow_amp_cgs-flow_amp_cgs_err, 0., flow_fwhm_μm-flow_fwhm_μm_err, 
                                                      flow_h3-flow_h3_err, flow_h4-flow_h4_err), -Inf, Inf, order=200)[1]
                err_u = quadgk(x -> Util.GaussHermite(x, flow_amp_cgs+flow_amp_cgs_err, 0., flow_fwhm_μm+flow_fwhm_μm_err,
                                                      flow_h3+flow_h3_err, flow_h4+flow_h4_err), -Inf, Inf, order=200)[1] - ii
                err = mean([err_l ≥ 0 ? err_l : 0., err_u])
                append!(i_err, [err])
                @debug "I=$ii, err_u=$err_u, err_l=$err_l, err=$err"

                profile = let profile = profile
                    x -> profile(x) + Util.GaussHermite(x, flow_amp, flow_mean_μm-mean_μm, flow_fwhm_μm, flow_h3, flow_h4)
                end
            elseif cube_fitter.line_profiles[k] == :Voigt
                # same as above
                ii = quadgk(x -> Util.Voigt(x, flow_amp_cgs, 0., flow_fwhm_μm, flow_η), -Inf, Inf, order=200)[1]
                intensity += ii
                err_l = ii - quadgk(x -> Util.Voigt(x, flow_amp_cgs-flow_amp_cgs_err, 0., flow_fwhm_μm-flow_fwhm_μm_err, flow_η-flow_η_err),
                               -Inf, Inf, order=200)[1]
                err_u = quadgk(x -> Util.Voigt(x, flow_amp_cgs+flow_amp_cgs_err, 0., flow_fwhm_μm+flow_fwhm_μm_err, flow_η+flow_η_err),
                               -Inf, Inf, order=200)[1] - ii
                err = mean([err_l ≥ 0 ? err_l : 0., err_u])
                append!(i_err, [err])
                @debug "I=$ii, err_u=$err_u, err_l=$err_l, err=$err"

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

        # intensity in units of erg s^-1 cm^-2 sr^-1 (integrated over μm)
        p_lines[pₒ] = intensity
        p_lines_err[pₒ] = √(sum(i_err.^2))

        # Add back in the normalization for the profile, to be used to calculate the S/N
        profile = let profile = profile
            x -> N * profile(x)
        end

        λ_arr = (-10fwhm_μm):Δλ:(10fwhm_μm)
        peak, peak_ind = findmax(profile.(λ_arr))

        # SNR, calculated as (amplitude) / (RMS of the surrounding spectrum)
        p_lines[pₒ+1] = peak / std(I[.!mask_lines] .- continuum[.!mask_lines])

        @debug "Line $(cube_fitter.line_names[k]) with integrated intensity $(p_lines[pₒ]) +/- $(p_lines_err[pₒ]) " *
            "(erg s^-1 cm^-2 sr^-1) and SNR $(p_lines[pₒ+1])"

        pₒ += 2

    end

    return p_dust, p_lines, p_dust_err, p_lines_err
end


"""
    fit_spaxel(cube_fitter, spaxel)

Wrapper function to perform a full fit of a single spaxel, calling `continuum_fit_spaxel` and `line_fit_spaxel` and
concatenating the best-fit parameters. The outputs are also saved to files so the fit need not be repeated in the case
of a crash.

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::CartesianIndex`: The coordinates of the spaxel to be fit
"""
function fit_spaxel(cube_fitter::CubeFitter, spaxel::CartesianIndex) 

    local p_out
    local p_err

    # Skip spaxels with NaNs (post-interpolation)
    λ = cube_fitter.cube.λ
    I = cube_fitter.cube.Iλ[spaxel, :]

    if any(.!isfinite.(I))
        return nothing, nothing
    end

    # Check if the fit has already been performed
    if !isfile(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).csv")) || cube_fitter.overwrite
        
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

            # Fit the spaxel
            σ, popt_c, I_cont, comps_cont, n_free_c, perr_c, covar_c = 
                @timeit timer_output "continuum_fit_spaxel" continuum_fit_spaxel(cube_fitter, spaxel)
            _, popt_l, I_line, comps_line, n_free_l, perr_l, covar_l = 
                @timeit timer_output "line_fit_spaxel" line_fit_spaxel(cube_fitter, spaxel)

            # Combine the continuum and line models
            I_model = I_cont .+ I_line
            comps = merge(comps_cont, comps_line)

            # Total free parameters
            n_free = n_free_c + n_free_l
            n_data = length(I)

            # Reduced chi^2 of the model
            χ2red = 1 / (n_data - n_free) * sum((I .- I_model).^2 ./ σ.^2)

            # Add dust feature and line parameters (intensity and SNR)
            p_dust, p_lines, p_dust_err, p_lines_err = 
                @timeit timer_output "calculate_extra_parameters" calculate_extra_parameters(cube_fitter, spaxel, popt_c, popt_l, perr_c, perr_l)
            p_out = [popt_c; popt_l; p_dust; p_lines; χ2red]
            p_err = [perr_c; perr_l; p_dust_err; p_lines_err; 0.]

            # Plot the fit
            λ0_ln = [ln.λ₀ for ln ∈ cube_fitter.lines]
            if cube_fitter.plot_spaxels != :none
                @debug "Plotting spaxel $spaxel best fit"
                @timeit timer_output "plot_spaxel_fit" plot_spaxel_fit(λ, I, I_model, σ, comps, 
                    cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, λ0_ln, cube_fitter.line_names, cube_fitter.extinction_screen, 
                    cube_fitter.z, χ2red, cube_fitter.name, "spaxel_$(spaxel[1])_$(spaxel[2])", backend=cube_fitter.plot_spaxels)
            end

            @debug "Saving results to binary for spaxel $spaxel"
            # serialize(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).LOKI"), (p_out=p_out, p_err=p_err))
            # save output as csv file
            open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).csv"), "w") do f 
                @timeit timer_output "writedlm" writedlm(f, [p_out p_err], ',')
            end

            if cube_fitter.track_memory
                # save memory allocations & other logistic data to a separate log file
                open(joinpath("output_$(cube_fitter.name)", "logs", "mem.spaxel_$(spaxel[1])_$(spaxel[2]).log"), "w") do f

                    print(f, """
                    ### PROCESS ID: $(getpid()) ###
                    Memory usage stats:
                    CubeFitter - $(Base.summarysize(cube_fitter) ÷ 10^6) MB
                      Cube - $(Base.summarysize(cube_fitter.cube) ÷ 10^6) MB 
                    """)

                    print(f, """
                    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
                    """)

                    print_timer(f, timer_output, sortby=:name)
                end
            end

        end


        return p_out, p_err

    end

    # Otherwise, just grab the results from before
    # results = deserialize(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).LOKI"))
    results = readdlm(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).csv"), ',', Float64, '\n')
    p_out = results[:, 1]
    p_err = results[:, 2]

    return p_out, p_err

end


"""
    fit_stack!(cube_fitter)

Perform an initial fit to the sum of all spaxels (the stack) to get an estimate for the initial parameter
vector to use with all of the individual spaxel fits.  The only input is the CubeFitter object, which is
modified with the resultant fit parameters.  There is no output.
"""
function fit_stack!(cube_fitter::CubeFitter)

    @info "===> Performing initial fit to the sum of all spaxels... <==="
    # Collect the data
    λ_init = cube_fitter.cube.λ
    I_sum_init = Util.Σ(cube_fitter.cube.Iλ, (1,2))

    # Continuum and line fits
    σ_init, popt_c_init, I_c_init, comps_c_init, n_free_c_init, _, _ = continuum_fit_spaxel(cube_fitter, CartesianIndex(0,0); init=true)
    _, popt_l_init, I_l_init, comps_l_init, n_free_l_init, _, _ = line_fit_spaxel(cube_fitter, CartesianIndex(0,0); init=true)

    # Get the overall models
    I_model_init = I_c_init .+ I_l_init
    comps_init = merge(comps_c_init, comps_l_init)

    n_free_init = n_free_c_init + n_free_l_init
    n_data_init = length(I_sum_init)

    # Calculate reduce chi^2
    χ2red_init = 1 / (n_data_init - n_free_init) * sum((I_sum_init .- I_model_init).^2 ./ σ_init.^2)

    # Save the results to the cube fitter
    cube_fitter.p_init_cont[:] .= popt_c_init
    cube_fitter.p_init_line[:] .= popt_l_init

    # Save the results to a file 
    # save running best fit parameters in case the fitting is interrupted
    open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_cont.csv"), "w") do f
        writedlm(f, popt_c_init, ',')
    end
    open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_line.csv"), "w") do f
        writedlm(f, popt_l_init, ',')
    end

    # Plot the fit
    λ0_ln = [ln.λ₀ for ln ∈ cube_fitter.lines]
    if cube_fitter.plot_spaxels != :none
        @debug "Plotting spaxel sum initial fit"
        plot_spaxel_fit(λ_init, I_sum_init, I_model_init, σ_init, comps_init, 
            cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, λ0_ln, cube_fitter.line_names, cube_fitter.extinction_screen, 
            cube_fitter.z, χ2red_init, cube_fitter.name, "initial_sum_fit", backend=cube_fitter.plot_spaxels)
    end

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
    out_errs = SharedArray(ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 1) .* NaN)

    ######################### DO AN INITIAL FIT WITH THE SUM OF ALL SPAXELS ###################

    @debug """
    $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
    """

    # Don't repeat if it's already been done
    if all(iszero.(cube_fitter.p_init_cont))
        fit_stack!(cube_fitter)
    else
        @info "===> Initial fit to the sum of all spaxels has already been performed <==="
    end

    ##############################################################################################

    function fit_spax_i(index::CartesianIndex)

        p_out, p_err = fit_spaxel(cube_fitter, index)
        if !isnothing(p_out)
            out_params[index, :] .= p_out
            out_errs[index, :] .= p_err
        end

        return
    end

    # Sort spaxels by median brightness, so that we fit the brightest ones first
    # (which hopefully have the best reduced chi^2s)
    spaxels = CartesianIndices((1:shape[1], 1:shape[2]))
    # spaxels = CartesianIndices((15:16, 15:16))

    # med_I = collect(Iterators.flatten([nanmedian(cube_fitter.cube.Iλ[spaxel..., :]) for spaxel ∈ spaxels]))
    # # replace NaNs with -1s
    # med_I[.!isfinite.(med_I)] .= -1.
    # # reverse sort
    # ss = sortperm(med_I, rev=true)
    # med_I = med_I[ss]
    # # apply sorting to spaxel indices
    # spaxels = collect(spaxels)[ss]

    @info "===> Beginning individual spaxel fitting... <==="
    # Use multiprocessing (not threading) to iterate over multiple spaxels at once using multiple CPUs
    if cube_fitter.parallel
        prog = Progress(length(spaxels); showspeed=true)
        progress_pmap(spaxels, progress=prog) do index
            fit_spax_i(index)
        end
    else
        prog = Progress(length(spaxels); showspeed=true)
        for index ∈ spaxels
            fit_spax_i(index)
            next!(prog)
        end
    end

    @info "===> Generating parameter maps and model cubes... <==="

    cube_model = generate_cubemodel(cube_fitter)
    param_maps, param_errs = generate_parammaps(cube_fitter)

    for index ∈ spaxels
        # Set the 2D parameter map outputs

        # Conversion factor from MJy sr^-1 to erg s^-1 cm^-2 Hz^-1 sr^-1 = 10^6 * 10^-23 = 10^-17
        # So, log10(A * 1e-17) = log10(A) - 17

        # Stellar continuum amplitude, temp
        param_maps.stellar_continuum[:amp][index] = out_params[index, 1] > 0. ? log10(out_params[index, 1])-17 : -Inf 
        param_errs.stellar_continuum[:amp][index] = out_params[index, 1] > 0. ? out_errs[index, 1] / out_params[index, 1] : NaN
        param_maps.stellar_continuum[:temp][index] = out_params[index, 2]
        param_errs.stellar_continuum[:temp][index] = out_errs[index, 2]
        pᵢ = 3
        # Dust continuum amplitude, temp
        for i ∈ 1:cube_fitter.n_dust_cont
            param_maps.dust_continuum[i][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ])-17 : -Inf
            param_errs.dust_continuum[i][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / out_params[index, pᵢ] : NaN
            param_maps.dust_continuum[i][:temp][index] = out_params[index, pᵢ+1]
            param_errs.dust_continuum[i][:temp][index] = out_errs[index, pᵢ+1]
            pᵢ += 2
        end
        # Dust feature log(amplitude), mean, FWHM
        for df ∈ cube_fitter.df_names
            param_maps.dust_features[df][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ])-17 : -Inf
            param_errs.dust_features[df][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / out_params[index, pᵢ] : NaN
            param_maps.dust_features[df][:mean][index] = out_params[index, pᵢ+1]
            param_errs.dust_features[df][:mean][index] = out_errs[index, pᵢ+1]
            param_maps.dust_features[df][:fwhm][index] = out_params[index, pᵢ+2]
            param_errs.dust_features[df][:fwhm][index] = out_errs[index, pᵢ+2]
            pᵢ += 3
        end
        # Extinction parameters
        param_maps.extinction[:tau_9_7][index] = out_params[index, pᵢ]
        param_errs.extinction[:tau_9_7][index] = out_errs[index, pᵢ]
        param_maps.extinction[:beta][index] = out_params[index, pᵢ+1]
        param_errs.extinction[:beta][index] = out_errs[index, pᵢ+1]
        pᵢ += 2

        # End of continuum parameters: recreate the continuum model
        I_cont, comps_c = Util.fit_spectrum(cube_fitter.cube.λ, out_params[index, 1:pᵢ-1], cube_fitter.n_dust_cont, cube_fitter.n_dust_feat,
            cube_fitter.extinction_curve, cube_fitter.extinction_screen, true)

        # Tied line velocity offsets
        vᵢ = pᵢ
        for vk ∈ cube_fitter.voff_tied_key
            param_maps.tied_voffs[vk][index] = out_params[index, pᵢ]
            param_errs.tied_voffs[vk][index] = out_errs[index, pᵢ]
            pᵢ += 1
        end
        # Tied flow velocity offsets
        for fvk ∈ cube_fitter.flow_voff_tied_key
            param_maps.flow_tied_voffs[fvk][index] = out_params[index, pᵢ]
            param_errs.flow_tied_voffs[fvk][index] = out_errs[index, pᵢ]
            pᵢ += 1
        end
        # Tied voigt mixing
        if cube_fitter.tie_voigt_mixing
            param_maps.tied_voigt_mix[index] = out_params[index, pᵢ]
            param_errs.tied_voigt_mix[index] = out_errs[index, pᵢ]
            pᵢ += 1
        end
        for (k, ln) ∈ enumerate(cube_fitter.line_names)

            amp = out_params[index, pᵢ]
            amp_err = out_errs[index, pᵢ]
            param_maps.lines[ln][:amp][index] = amp
            param_errs.lines[ln][:amp][index] = amp_err
            fwhm_res = Util.C_KMS / cube_fitter.interp_R(cube_fitter.lines[k].λ₀)

            if isnothing(cube_fitter.line_tied[k]) || cube_fitter.flexible_wavesol

                # Individual shift
                voff = out_params[index, pᵢ+1]
                voff_err = out_errs[index, pᵢ+1]
                if !isnothing(cube_fitter.line_tied[k])
                    # If velocity is tied while flexible, add the overall shift and the individual shift together
                    vwhere = findfirst(x -> x == cube_fitter.line_tied[k], cube_fitter.voff_tied_key)
                    voff += out_params[index, vᵢ+vwhere-1]
                    voff_err = √(voff_err^2 + out_errs[index, vᵢ+vwhere-1]^2)
                end
                param_maps.lines[ln][:voff][index] = voff
                param_errs.lines[ln][:voff][index] = voff_err

                # FWHM -> subtract instrumental resolution in quadrature
                fwhm = out_params[index, pᵢ+2]
                fwhm_err = out_errs[index, pᵢ+2]
                if fwhm_res > out_params[index, pᵢ+2]
                    param_maps.lines[ln][:fwhm][index] = 0.
                    param_errs.lines[ln][:fwhm][index] = fwhm_err
                else
                    param_maps.lines[ln][:fwhm][index] = √(fwhm^2 - fwhm_res^2)
                    param_errs.lines[ln][:fwhm][index] = fwhm / √(fwhm^2 - fwhm_res^2) * fwhm_err
                end

                # Get Gauss-Hermite 3rd and 4th order moments
                if cube_fitter.line_profiles[k] == :GaussHermite
                    param_maps.lines[ln][:h3][index] = out_params[index, pᵢ+3]
                    param_errs.lines[ln][:h3][index] = out_errs[index, pᵢ+3]
                    param_maps.lines[ln][:h4][index] = out_params[index, pᵢ+4]
                    param_errs.lines[ln][:h4][index] = out_errs[index, pᵢ+4]
                    pᵢ += 2
                # Get Voigt mixing
                elseif cube_fitter.line_profiles[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                    param_maps.lines[ln][:mixing][index] = out_params[index, pᵢ+3]
                    param_errs.lines[ln][:mixing][index] = out_errs[index, pᵢ+3]
                    pᵢ += 1
                end
                pᵢ += 3

            else
                # Tied shift only
                vwhere = findfirst(x -> x == cube_fitter.line_tied[k], cube_fitter.voff_tied_key)
                voff = out_params[index, vᵢ+vwhere-1]
                voff_err = out_errs[index, vᵢ+vwhere-1]

                # FWHM -> subtract instrumental resolution in quadrature
                fwhm = out_params[index, pᵢ+1]
                fwhm_err = out_errs[index, pᵢ+1]
                if fwhm_res > out_params[index, pᵢ+1]
                    param_maps.lines[ln][:fwhm][index] = 0.
                    param_errs.lines[ln][:fwhm][index] = fwhm_err
                else
                    param_maps.lines[ln][:fwhm][index] = √(fwhm^2 - fwhm_res^2)
                    param_errs.lines[ln][:fwhm][index] = fwhm / √(fwhm^2 - fwhm_res^2) * fwhm_err
                end
                # Get Gauss-Hermite 3rd and 4th order moments
                if cube_fitter.line_profiles[k] == :GaussHermite
                    param_maps.lines[ln][:h3][index] = out_params[index, pᵢ+2]
                    param_errs.lines[ln][:h3][index] = out_errs[index, pᵢ+2]
                    param_maps.lines[ln][:h4][index] = out_params[index, pᵢ+3]
                    param_errs.lines[ln][:h4][index] = out_errs[index, pᵢ+3]
                    pᵢ += 2
                elseif cube_fitter.line_profiles[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                    param_maps.lines[k][:mixing][index] = out_params[index, pᵢ+2]
                    param_errs.lines[k][:mixing][index] = out_errs[index, pᵢ+2]
                    pᵢ += 1
                end
                pᵢ += 2

            end

            if !isnothing(cube_fitter.line_flow_profiles[k])

                param_maps.lines[ln][:flow_amp][index] = amp * out_params[index, pᵢ]
                param_errs.lines[ln][:flow_amp][index] = 
                    √((amp * out_params[index, pᵢ])^2 * ((amp_err / amp)^2 + (out_errs[index, pᵢ] / out_params[index, pᵢ])^2))

                if isnothing(cube_fitter.line_flow_tied[k])
                    # note of caution: the output/saved/plotted values of flow voffs are ALWAYS given relative to
                    # the main line's voff, NOT the rest wavelength of the line; same goes for the errors (which add in quadrature)
                    param_maps.lines[ln][:flow_voff][index] = out_params[index, pᵢ+1]
                    param_errs.lines[ln][:flow_voff][index] = out_errs[index, pᵢ+1]

                    # FWHM -> subtract instrumental resolution in quadrature
                    flow_fwhm = fwhm * out_params[index, pᵢ+2]
                    flow_fwhm_err = √(flow_fwhm^2 * ((fwhm_err / fwhm)^2 + (out_errs[index, pᵢ+2] / out_params[index, pᵢ+2])^2))
                    if fwhm_res > flow_fwhm
                        param_maps.lines[ln][:flow_fwhm][index] = 0.
                        param_errs.lines[ln][:flow_fwhm][index] = flow_fwhm_err
                    else
                        param_maps.lines[ln][:flow_fwhm][index] = √(flow_fwhm^2 - fwhm_res^2)
                        param_errs.lines[ln][:flow_fwhm][index] = flow_fwhm / √(flow_fwhm^2 - fwhm_res^2) * flow_fwhm_err
                    end
                    # Get Gauss-Hermite 3rd and 4th order moments
                    if cube_fitter.line_flow_profiles[k] == :GaussHermite
                        param_maps.lines[ln][:flow_h3][index] = out_params[index, pᵢ+3]
                        param_errs.lines[ln][:flow_h3][index] = out_errs[index, pᵢ+3]
                        param_maps.lines[ln][:flow_h4][index] = out_params[index, pᵢ+4]
                        param_errs.lines[ln][:flow_h4][index] = out_errs[index, pᵢ+4]
                        pᵢ += 2
                    elseif cube_fitter.line_flow_profiles[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                        param_maps.lines[ln][:flow_mixing][index] = out_params[index, pᵢ+3]
                        param_errs.lines[ln][:flow_mixing][index] = out_errs[index, pᵢ+3]
                        pᵢ += 1
                    end
                    pᵢ += 3
                else
                    flow_fwhm = fwhm * out_params[index, pᵢ+1]
                    flow_fwhm_err = √(flow_fwhm^2 * ((fwhm_err / fwhm)^2 + (out_errs[index, pᵢ+1] / out_params[index, pᵢ+1])^2))
                    # FWHM -> subtract instrumental resolution in quadrature
                    if fwhm_res > flow_fwhm
                        param_maps.lines[ln][:flow_fwhm][index] = 0.
                        param_errs.lines[ln][:flow_fwhm][index] = flow_fwhm
                    else
                        param_maps.lines[ln][:flow_fwhm][index] = √(flow_fwhm^2 - fwhm_res^2)
                        param_errs.lines[ln][:flow_fwhm][index] = flow_fwhm / √(flow_fwhm^2 - fwhm_res^2) * flow_fwhm_err
                    end
                    # Get Gauss-Hermite 3rd and 4th order moments
                    if cube_fitter.line_flow_profiles[k] == :GaussHermite
                        param_maps.lines[ln][:flow_h3][index] = out_params[index, pᵢ+2]
                        param_errs.lines[ln][:flow_h3][index] = out_errs[index, pᵢ+2]
                        param_maps.lines[ln][:flow_h4][index] = out_params[index, pᵢ+3]
                        param_errs.lines[ln][:flow_h4][index] = out_errs[index, pᵢ+3]
                        pᵢ += 2
                    elseif cube_fitter.line_flow_profiles[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                        param_maps.lines[k][:flow_mixing][index] = out_params[index, pᵢ+2]
                        param_errs.lines[k][:flow_mixing][index] = out_errs[index, pᵢ+2]
                        pᵢ += 1
                    end
                    pᵢ += 2
                end
            end

        end

        # End of line parameters: recreate the line model
        I_line, comps_l = Util.fit_line_residuals(cube_fitter.cube.λ, out_params[index, vᵢ:pᵢ-1], cube_fitter.n_lines, cube_fitter.n_voff_tied,
            cube_fitter.voff_tied_key, cube_fitter.line_tied, cube_fitter.line_profiles, cube_fitter.n_flow_voff_tied, cube_fitter.flow_voff_tied_key,
            cube_fitter.line_flow_tied, cube_fitter.line_flow_profiles, [ln.λ₀ for ln ∈ cube_fitter.lines], 
            cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing, true)

        # Renormalize
        N = Float64(abs(nanmaximum(cube_fitter.cube.Iλ[index, :])))
        N = N ≠ 0. ? N : 1.
        for comp ∈ keys(comps_l)
            comps_l[comp] .*= N
        end
        I_line .*= N
        
        # Combine the continuum and line models
        I_model = I_cont .+ I_line
        comps = merge(comps_c, comps_l)

        # Dust feature intensity and SNR, from calculate_extra_parameters
        for df ∈ cube_fitter.df_names
            param_maps.dust_features[df][:intI][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]) : -Inf
            param_errs.dust_features[df][:intI][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / out_params[index, pᵢ] : NaN
            param_maps.dust_features[df][:SNR][index] = out_params[index, pᵢ+1]
            pᵢ += 2
        end

        for (k, ln) ∈ enumerate(cube_fitter.line_names)
            # Convert amplitudes to the correct units, then take the log
            amp_norm = param_maps.lines[ln][:amp][index]
            amp_norm_err = param_errs.lines[ln][:amp][index]
            param_maps.lines[ln][:amp][index] = amp_norm > 0 ? log10(amp_norm * N)-17 : -Inf
            param_errs.lines[ln][:amp][index] = amp_norm > 0 ? amp_norm_err / amp_norm : NaN
            if !isnothing(cube_fitter.line_flow_profiles[k])
                flow_amp_norm = param_maps.lines[ln][:flow_amp][index]
                flow_amp_norm_err = param_errs.lines[ln][:flow_amp][index]
                param_maps.lines[ln][:flow_amp][index] = flow_amp_norm > 0 ? log10(amp_norm * flow_amp_norm * N)-17 : -Inf
                param_errs.lines[ln][:flow_amp][index] = flow_amp_norm > 0 ? flow_amp_norm_err / flow_amp_norm : NaN
            end

            # Line intensity and SNR, from calculate_extra_parameters
            param_maps.lines[ln][:intI][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]) : -Inf
            param_errs.lines[ln][:intI][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / out_params[index, pᵢ] : NaN
            param_maps.lines[ln][:SNR][index] = out_params[index, pᵢ+1]
            pᵢ += 2
        end

        # Reduced χ^2
        param_maps.reduced_χ2[index] = out_params[index, pᵢ]

        # Set 3D model cube outputs
        cube_model.model[index, :] .= I_model
        cube_model.stellar[index, :] .= comps["stellar"]
        for i ∈ 1:cube_fitter.n_dust_cont
            cube_model.dust_continuum[index, :, i] .= comps["dust_cont_$i"]
        end
        for j ∈ 1:cube_fitter.n_dust_feat
            cube_model.dust_features[index, :, j] .= comps["dust_feat_$j"]
        end
        for k ∈ 1:cube_fitter.n_lines
            cube_model.lines[index, :, k] .= comps["line_$k"]
            if haskey(comps, "line_$(k)_flow")
                cube_model.lines[index, :, k] .+= comps["line_$(k)_flow"]
            end
        end
        cube_model.extinction[index, :] .= comps["extinction"]

    end

    if cube_fitter.plot_maps
        @info "===> Plotting parameter maps... <==="
        plot_parameter_maps(cube_fitter, param_maps)
    end

    if cube_fitter.save_fits
        @info "===> Writing FITS outputs... <==="
        write_fits(cube_fitter, cube_model, param_maps, param_errs)
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
    plot_parameter_map(data, name, name_i, Ω, z, cosmo; snr_filter=snr_filter, snr_thresh=snr_thresh)

Plotting function for 2D parameter maps which are output by `fit_cube`

# Arguments
- `data::Matrix{Float64}`: The 2D array of data to be plotted
- `name::String`: The name of the object whose fitting parameter is being plotted, i.e. "NGC_7469"
- `name_i::String`: The name of the individual parameter being plotted, i.e. "dust_features_PAH_5.24_amp"
- `Ω::Float64`: The solid angle subtended by each pixel, in steradians (used for angular scalebar)
- `z::Float64`: The redshift of the object (used for physical scalebar)
- `cosmo::Cosmology.AbstractCosmology`: The cosmology to use to calculate distance for the physical scalebar
- `snr_filter::Matrix{Float64}=Matrix{Float64}(undef,0,0)`: A 2D array of S/N values to
    be used to filter out certain spaxels from being plotted - must be the same size as `data` to filter
- `snr_thresh::Float64=3.`: The S/N threshold below which to cut out any spaxels using the values in snr_filter
"""
function plot_parameter_map(data::Matrix{Float64}, name::String, name_i::String, Ω::Float64, z::Float64, 
    cosmo::Cosmology.AbstractCosmology; snr_filter::Union{Nothing,Matrix{Float64}}=nothing, snr_thresh::Float64=3.)

    # I know this is ugly but I couldn't figure out a better way to do it lmao
    if occursin("amp", String(name_i))
        bunit = "\$\\log_{10}(I / \$ erg s\$^{-1}\$ cm\$^{-2}\$ Hz\$^{-1}\$ sr\$^{-1})\$"
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
        bunit = "\$\\log_{10}(I /\$ erg s\$^{-1}\$ cm\$^{-2}\$ sr\$^{-1}\$)"
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
    # flatdata = filtered[isfinite.(filtered)]
    vmin = sum(isfinite.(filtered)) > 0 ? nanminimum(filtered) : 0.0
    vmax = sum(isfinite.(filtered)) > 0 ? nanmaximum(filtered) : 0.0
    cdata = ax.imshow(filtered', origin=:lower, cmap=:cubehelix, vmin=vmin, vmax=vmax)
    ax.axis(:off)

    # Angular and physical scalebars
    n_pix = 1/(sqrt(Ω) * 180/π * 3600)
    @debug "Using angular diameter distance $(angular_diameter_dist(cosmo, z))"
    # Calculate in Mpc
    dL = angular_diameter_dist(u"pc", cosmo, z) / (180/π * 3600)  # l = d * theta (1")
    # Remove units
    dL = uconvert(NoUnits, dL/u"pc")
    # Round to integer
    dL = floor(Int, dL)
    if cosmo.h == 1.0
        scalebar = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, "1\$\'\'\$ / $dL\$h^{-1}\$ pc", "lower left", pad=1, color=:black, 
            frameon=false, size_vertical=0.4, label_top=false)
    else
        scalebar = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, "1\$\'\'\$ / $dL pc", "lower left", pad=1, color=:black,
            frameon=false, size_vertical=0.4, label_top=false)
    end
    ax.add_artist(scalebar)

    fig.colorbar(cdata, ax=ax, label=bunit)
    plt.savefig(joinpath("output_$(name)", "param_maps", "$(name_i).pdf"), dpi=300, bbox_inches=:tight)
    plt.close()

end


"""
    plot_parameter_maps(param_maps; snr_thresh=snr_thresh)

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

    for parameter ∈ keys(param_maps.stellar_continuum)
        data = param_maps.stellar_continuum[parameter]
        name_i = join(["stellar_continuum", parameter], "_")
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology)
    end

    for i ∈ keys(param_maps.dust_continuum)
        for parameter ∈ keys(param_maps.dust_continuum[i])
            data = param_maps.dust_continuum[i][parameter]
            name_i = join(["dust_continuum", i, parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology)
        end
    end

    for df ∈ keys(param_maps.dust_features)
        snr = param_maps.dust_features[df][:SNR]
        for parameter ∈ keys(param_maps.dust_features[df])
            data = param_maps.dust_features[df][parameter]
            name_i = join(["dust_features", df, parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
                snr_filter=parameter ≠ :SNR ? snr : nothing, snr_thresh=snr_thresh)
        end
    end

    for parameter ∈ keys(param_maps.extinction)
        data = param_maps.extinction[parameter]
        name_i = join(["extinction", parameter], "_")
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology)
    end

    if cube_fitter.tie_voigt_mixing
        data = param_maps.tied_voigt_mix
        name_i = "tied_voigt_mixing"
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology)
    end

    for vk ∈ cube_fitter.voff_tied_key
        data = param_maps.tied_voffs[vk]
        name_i = join(["tied_voffs", vk], "_")
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology)
    end

    for fvk ∈ cube_fitter.flow_voff_tied_key
        data = param_maps.flow_tied_voffs[fvk]
        name_i = join(["flow_tied_voffs", fvk], "_")
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology)
    end

    for line ∈ keys(param_maps.lines)
        snr = param_maps.lines[line][:SNR]
        for parameter ∈ keys(param_maps.lines[line])
            data = param_maps.lines[line][parameter]
            name_i = join(["lines", line, parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
                snr_filter=parameter ≠ :SNR ? snr : nothing, snr_thresh=snr_thresh)
        end
    end

    data = param_maps.reduced_χ2
    name_i = "reduced_chi2"
    plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology)

end


"""
    write_fits(cube_fitter)

Save the best fit results for the cube into two FITS files: one for the full 3D intensity model of the cube, split up by
individual model components, and one for 2D parameter maps of the best-fit parameters for each spaxel in the cube.

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
"""
function write_fits(cube_fitter::CubeFitter, cube_model::CubeModel, param_maps::ParamMaps, param_errs::ParamMaps)

    # Header information
    hdr = FITSHeader(
        ["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "PIXAR_SR", "RA", "DEC", "WCSAXES",
            "CDELT1", "CDELT2", "CDELT3", "CTYPE1", "CTYPE2", "CTYPE3", "CRPIX1", "CRPIX2", "CRPIX3",
            "CRVAL1", "CRVAL2", "CRVAL3", "CUNIT1", "CUNIT2", "CUNIT3", "PC1_1", "PC1_2", "PC1_3", 
            "PC2_1", "PC2_2", "PC2_3", "PC3_1", "PC3_2", "PC3_3"],

        # Check if the redshift correction is right for the third WCS axis?
        [cube_fitter.name, cube_fitter.z, cube_fitter.cube.channel, cube_fitter.cube.band, cube_fitter.cube.Ω, cube_fitter.cube.α, cube_fitter.cube.δ, 
         cube_fitter.cube.wcs.naxis, cube_fitter.cube.wcs.cdelt[1], cube_fitter.cube.wcs.cdelt[2], cube_fitter.cube.wcs.cdelt[3], 
         cube_fitter.cube.wcs.ctype[1], cube_fitter.cube.wcs.ctype[2], cube_fitter.cube.wcs.ctype[3], cube_fitter.cube.wcs.crpix[1], 
         cube_fitter.cube.wcs.crpix[2], cube_fitter.cube.wcs.crpix[3], cube_fitter.cube.wcs.crval[1], cube_fitter.cube.wcs.crval[2], 
         cube_fitter.cube.wcs.crval[3], cube_fitter.cube.wcs.cunit[1], cube_fitter.cube.wcs.cunit[2], cube_fitter.cube.wcs.cunit[3], 
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

        write(f, Vector{Int}())                                                                     # Primary HDU (empty)
        write(f, cube_fitter.cube.Iλ; header=hdr, name="DATA")                                      # Raw data with nans inserted
        write(f, cube_model.model; header=hdr, name="MODEL")                                        # Full intensity model
        write(f, cube_fitter.cube.Iλ .- cube_model.model; header=hdr, name="RESIDUALS")             # Residuals (data - model)
        write(f, cube_model.stellar; header=hdr, name="STELLAR_CONTINUUM")                          # Stellar continuum model
        for i ∈ 1:size(cube_model.dust_continuum, 4)
            write(f, cube_model.dust_continuum[:, :, :, i]; header=hdr, name="DUST_CONTINUUM_$i")   # Dust continuum models
        end
        for (j, df) ∈ enumerate(cube_fitter.df_names)
            write(f, cube_model.dust_features[:, :, :, j]; header=hdr, name="$df")                  # Dust feature profiles
        end
        for (k, line) ∈ enumerate(cube_fitter.line_names)
            write(f, cube_model.lines[:, :, :, k]; header=hdr, name="$line")                        # Emission line profiles
        end
        write(f, cube_model.extinction; header=hdr, name="EXTINCTION")                              # Extinction model
        
        write(f, ["wave_rest", "wave_obs"],                                                                     # 1D Rest frame and observed frame
                 [cube_fitter.cube.λ, Util.observed_frame(cube_fitter.cube.λ, cube_fitter.z)],                  # wavelength vectors
              hdutype=TableHDU, name="WAVELENGTH", units=Dict(:wave_rest => "um", :wave_obs => "um"))

        # Insert physical units into the headers of each HDU -> MegaJansky per steradian for all except
        # the extinction profile, which is a multiplicative constant
        write_key(f["DATA"], "BUNIT", "MJy/sr")
        write_key(f["MODEL"], "BUNIT", "MJy/sr")
        write_key(f["RESIDUALS"], "BUNIT", "MJy/sr")
        write_key(f["STELLAR_CONTINUUM"], "BUNIT", "MJy/sr")
        for i ∈ 1:size(cube_model.dust_continuum, 4)
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
        for parameter ∈ keys(param_maps.stellar_continuum)
            data = param_maps.stellar_continuum[parameter]
            name_i = join(["stellar_continuum", parameter], "_")
            if occursin("amp", String(name_i))
                bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
            elseif occursin("temp", String(name_i))
                bunit = "Kelvin"
            end
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end
        for parameter ∈ keys(param_errs.stellar_continuum)
            data = param_errs.stellar_continuum[parameter]
            name_i = join(["stellar_continuum", parameter, "err"], "_")
            if occursin("amp", String(name_i))
                bunit = "dex"
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
                    bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                elseif occursin("temp", String(name_i))
                    bunit = "Kelvin"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)  
            end
        end
        for i ∈ keys(param_errs.dust_continuum)
            for parameter ∈ keys(param_errs.dust_continuum[i])
                data = param_errs.dust_continuum[i][parameter]
                name_i = join(["dust_continuum", i, parameter, "err"], "_")
                if occursin("amp", String(name_i))
                    bunit = "dex"
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
                    bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                elseif occursin("fwhm", String(name_i)) || occursin("mean", String(name_i))
                    bunit = "um"
                elseif occursin("intI", String(name_i))
                    bunit = "log10(I / erg s^-1 cm^-2 sr^-1)"
                elseif occursin("SNR", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)      
            end
        end
        for df ∈ keys(param_errs.dust_features)
            for parameter ∈ keys(param_errs.dust_features[df])
                data = param_errs.dust_features[df][parameter]
                name_i = join(["dust_features", df, parameter, "err"], "_")
                if occursin("amp", String(name_i))
                    bunit = "dex"
                elseif occursin("fwhm", String(name_i)) || occursin("mean", String(name_i))
                    bunit = "um"
                elseif occursin("intI", String(name_i))
                    bunit = "dex"
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
                    bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                elseif occursin("fwhm", String(name_i)) || occursin("voff", String(name_i))
                    bunit = "km/s"
                elseif occursin("intI", String(name_i))
                    bunit = "log10(I / erg s^-1 cm^-2 sr^-1)"
                elseif occursin("SNR", String(name_i)) || occursin("h3", String(name_i)) || 
                    occursin("h4", String(name_i)) || occursin("mixing", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)   
            end
        end
        for line ∈ keys(param_errs.lines)
            for parameter ∈ keys(param_errs.lines[line])
                data = param_errs.lines[line][parameter]
                name_i = join(["lines", line, parameter, "err"], "_")
                if occursin("amp", String(name_i))
                    bunit = "dex"
                elseif occursin("fwhm", String(name_i)) || occursin("voff", String(name_i))
                    bunit = "km/s"
                elseif occursin("intI", String(name_i))
                    bunit = "dex"
                elseif occursin("SNR", String(name_i)) || occursin("h3", String(name_i)) || 
                    occursin("h4", String(name_i)) || occursin("mixing", String(name_i))
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
        for parameter ∈ keys(param_errs.extinction)
            data = param_errs.extinction[parameter]
            name_i = join(["extinction", parameter, "err"], "_")
            bunit = "unitless"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)  
        end

        if cube_fitter.tie_voigt_mixing
            data = param_maps.tied_voigt_mix
            name_i = "tied_voigt_mixing"
            bunit = "unitless"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end
        if cube_fitter.tie_voigt_mixing
            data = param_errs.tied_voigt_mix
            name_i = "tied_voigt_mixing_err"
            bunit = "unitless"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end

        for vk ∈ cube_fitter.voff_tied_key
            data = param_maps.tied_voffs[vk]
            name_i = join(["tied_voffs", vk], "_")
            bunit = "km/s"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end
        for vk ∈ cube_fitter.voff_tied_key
            data = param_errs.tied_voffs[vk]
            name_i = join(["tied_voffs", vk, "err"], "_")
            bunit = "km/s"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end

        for fvk ∈ cube_fitter.flow_voff_tied_key
            data = param_maps.flow_tied_voffs[fvk]
            name_i = join(["flow_tied_voffs", fvk], "_")
            bunit = "km/s"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end
        for fvk ∈ cube_fitter.flow_voff_tied_key
            data = param_errs.flow_tied_voffs[fvk]
            name_i = join(["flow_tied_voffs", fvk, "err"], "_")
            bunit = "km/s"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end

        data = param_maps.reduced_χ2
        name_i = "reduced_chi2"
        bunit = "unitless"
        write(f, data; header=hdr, name=name_i)
        write_key(f[name_i], "BUNIT", bunit)

    end
end

end