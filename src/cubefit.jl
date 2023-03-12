#=
THE CUBEFIT MODULE
------------------

This is the main module for actually fitting IFU cubes.  It contains structs for holding
fitting options and output arrays, as well as all the functions for actually performing the
fitting across a single spaxel or an entire cube. The main calling sequence one will want to 
perform is first loading in the data, presumably from some FITS files, with the CubeData module,
then creating a CubeFitter struct from the CubeData struct, followed by calling fit_cube! on the
CubeFitter. An example of this is provided in the test driver files in the test directory.
=#

module CubeFit

# Export only the functions that the user may want to call
export CubeFitter, fit_optical_depth, fit_cube!, continuum_fit_spaxel, line_fit_spaxel, fit_spaxel, 
    fit_stack!, plot_parameter_maps, write_fits

# Parallel computing packages
using Distributed
using SharedArrays

# Math packages
using Distributions
using Statistics
using NaNStatistics
using QuadGK
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
using Printf
using Logging
using LoggingExtras
using Dates
using InteractiveUtils
using TimerOutputs
using LaTeXStrings

# PyCall needed for some matplotlib modules
using PyCall

# Have to import certain python modules within the __init__ function so that it works after precompilation,
# so these constants are just placeholders before that happens
const plt::PyObject = PyNULL()
const py_anchored_artists::PyObject = PyNULL()
const py_ticker::PyObject = PyNULL()
const py_animation::PyObject = PyNULL()
const py_wcs::PyObject = PyNULL()

# Some constants for setting matplotlib font sizes
const SMALL::UInt8 = 12
const MED::UInt8 = 14
const BIG::UInt8 = 16

function __init__()

    # Import pyplot
    copy!(plt, pyimport_conda("matplotlib.pyplot", "matplotlib"))

    # Import matplotlib submodules for nice plots
    # anchored_artists --> used for scale bars
    copy!(py_anchored_artists, pyimport_conda("mpl_toolkits.axes_grid1.anchored_artists", "matplotlib"))
    # ticker --> used for formatting axis ticks and tick labels
    copy!(py_ticker, pyimport_conda("matplotlib.ticker", "matplotlib"))
    # animation --> used for making mp4 movie files (optional)
    copy!(py_animation, pyimport_conda("matplotlib.animation", "matplotlib"))
    # Import astropy's WCS module to work with matplotlib
    copy!(py_wcs, pyimport_conda("astropy.wcs", "astropy"))

    # Matplotlib settings to make plots look pretty :)
    plt.switch_backend("Agg")                  # agg backend just saves to file, no GUI display
    plt.rc("font", size=MED)                   # controls default text sizes
    plt.rc("axes", titlesize=MED)              # fontsize of the axes title
    plt.rc("axes", labelsize=MED)              # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL)           # fontsize of the x tick labels
    plt.rc("ytick", labelsize=SMALL)           # fontsize of the y tick labels
    plt.rc("legend", fontsize=SMALL)           # legend fontsize
    plt.rc("figure", titlesize=BIG)            # fontsize of the figure title
    plt.rc("text", usetex=true)                # use LaTeX for things like axis offsets
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

# Date format for the log files
const date_format::String = "yyyy-mm-dd HH:MM:SS"

# Timer object for logging the performance of the code, if the settings are enabled
const timer_output::TimerOutput = TimerOutput()

# This lock is used to control write access to the convergence log file, since
# multiple parallel processes may try to write to it at once.
const file_lock::ReentrantLock = ReentrantLock()


"""
    parse_resolving(z, channel)

Read in the resolving_mrs.csv configuration file to create a cubic spline interpolation of the
MIRI MRS resolving power as a function of wavelength, redshifted to the rest frame of the object
being fit.

# Arguments
- `z::Real`: The redshift of the object to be fit
- `channel::Integer`: The channel of the fit
"""
function parse_resolving(z::Real, channel::Integer)::Function

    @debug "Parsing MRS resoling power from resolving_mrs.csv for channel $channel"

    # Read in the resolving power data
    resolve = readdlm(joinpath(@__DIR__, "resolving_mrs.csv"), ',', Float64, '\n', header=true)
    wave = resolve[1][:, 1]
    R = resolve[1][:, 2]

    # Find points where wavelength jumps down (b/w channels)
    jumps = diff(wave) .< 0
    indices = eachindex(wave)
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

    # Create a linear interpolation function so we can evaluate it at the points of interest for our data,
    # corrected to be in the rest frame
    interp_R = wi -> Spline1D(wave, R, k=1)(Util.observed_frame(wi, z))
    
    interp_R
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
    keylist1 = ["extinction_curve", "extinction_screen", "fit_sil_emission", "overwrite", "track_memory", "track_convergence", 
                "save_full_model", "make_movies", "cosmology"]
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
    options_out[:fit_sil_emission] = options["fit_sil_emission"]
    @debug "Fit Silicate Emission? - $(options["fit_sil_emission"])"
    options_out[:overwrite] = options["overwrite"]
    @debug "Overwrite old fits? - $(options["overwrite"])"
    options_out[:track_memory] = options["track_memory"]
    @debug "Track memory allocations? - $(options["track_memory"])"
    options_out[:track_convergence] = options["track_convergence"]
    @debug "Track SAMIN convergence? - $(options["track_convergence"])"
    options_out[:save_full_model] = options["save_full_model"]
    @debug "Save full model? - $(options["save_full_model"])"
    options_out[:make_movies] = options["make_movies"]
    @debug "Make movies? - $(options["make_movies"])"

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

    options_out
end


"""
    parse_dust(τ_guess)

Read in the dust.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary with Parameter objects for dust fitting parameters.
This deals with continuum, PAH features, and extinction options.  The input parameter τ_guess
is an initial guess for the optical depth, estimated by interpolating the continuum.
"""
function parse_dust(τ_guess::Real)::Dict

    @debug """\n
    Parsing dust file
    #######################################################
    """

    # Read in the dust file
    dust = TOML.parsefile(joinpath(@__DIR__, "dust.toml"))
    dust_out = Dict()
    keylist1 = ["stellar_continuum_temp", "dust_continuum_temps", "dust_features", "extinction", "hot_dust"]
    keylist2 = ["wave", "fwhm"]
    keylist3 = ["tau_9_7", "tau_ice", "tau_ch", "beta"]
    keylist4 = ["temp", "frac", "tau_warm", "tau_cold"]
    keylist5 = ["val", "plim", "locked"]

    # Loop through all of the required keys that should be in the file and confirm that they are there
    for key ∈ keylist1
        if !(key ∈ keys(dust))
            error("Missing option $key in dust file!")
        end
    end
    for key ∈ keylist5
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
                if (ex_key == "tau_9_7") && (key == "val")
                    continue
                end
                error("Missing option $key in $ex_key options!")
            end
        end
        for hd_key ∈ keylist4
            if !(hd_key ∈ keys(dust["hot_dust"]))
                error("Missing option $hd_key in hot dust options!")
            end
            if !(key ∈ keys(dust["hot_dust"][hd_key]))
                if (hd_key ∈ ("tau_warm", "tau_cold")) && (key == "val")
                    continue
                end
                error("Missing option $key in $hd_key options!")
            end
        end
    end

    # Convert the options into Parameter objects, and set them to the output dictionary

    # Stellar continuum temperature
    dust_out[:stellar_continuum_temp] = Param.from_dict(dust["stellar_continuum_temp"])
    @debug "Stellar continuum:\nTemp $(dust_out[:stellar_continuum_temp])"

    # Dust continuum temperatures
    dust_out[:dust_continuum_temps] = [Param.from_dict(dust["dust_continuum_temps"][i]) for i ∈ eachindex(dust["dust_continuum_temps"])]
    msg = "Dust continuum:"
    for dc ∈ dust_out[:dust_continuum_temps]
        msg *= "\nTemp $dc"
    end
    @debug msg
    
    # Dust feature central wavelengths and FWHMs
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

    # Extinction parameters, optical depth and mixing ratio
    dust_out[:extinction] = Param.ParamDict()
    msg = "Extinction:"
    # Write tau_9_7 value based on the provided guess
    dust["extinction"]["tau_9_7"]["val"] = τ_guess
    dust_out[:extinction][:tau_9_7] = Param.from_dict(dust["extinction"]["tau_9_7"])
    msg *= "\nTau_sil $(dust_out[:extinction][:tau_9_7])"
    dust_out[:extinction][:tau_ice] = Param.from_dict(dust["extinction"]["tau_ice"])
    msg *= "\nTau_ice $(dust_out[:extinction][:tau_ice])"
    dust_out[:extinction][:tau_ch] = Param.from_dict(dust["extinction"]["tau_ch"])
    msg *= "\nTau_CH $(dust_out[:extinction][:tau_ch])"
    dust_out[:extinction][:beta] = Param.from_dict(dust["extinction"]["beta"])
    msg *= "\nBeta $(dust_out[:extinction][:beta])"
    @debug msg

    # Hot dust parameters, temperature, covering fraction, warm tau, and cold tau
    dust_out[:hot_dust] = Param.ParamDict()
    msg = "Hot Dust:"
    # Write warm_tau and col_tau values based on the provided guess
    dust["hot_dust"]["tau_warm"]["val"] = τ_guess
    dust["hot_dust"]["tau_cold"]["val"] = τ_guess
    dust_out[:hot_dust][:temp] = Param.from_dict(dust["hot_dust"]["temp"])
    msg *= "\nTemp $(dust_out[:hot_dust][:temp])"
    dust_out[:hot_dust][:frac] = Param.from_dict(dust["hot_dust"]["frac"])
    msg *= "\nFrac $(dust_out[:hot_dust][:frac])"
    dust_out[:hot_dust][:tau_warm] = Param.from_dict(dust["hot_dust"]["tau_warm"])
    msg *= "\nTau_Warm $(dust_out[:hot_dust][:tau_warm])"
    dust_out[:hot_dust][:tau_cold] = Param.from_dict(dust["hot_dust"]["tau_cold"])
    msg *= "\nTau_Cold $(dust_out[:hot_dust][:tau_cold])"
    @debug msg

    dust_out
end


"""
    parse_lines(channel, interp_R, λ)

Read in the lines.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary with Parameter objects for line fitting parameters.
This deals purely with emission line options.

# Arguments
- `channel::Integer`: The MIRI channel that is being fit
- `interp_R::Function`: The MRS resolving power interpolation function, as a function of rest frame wavelength
- `λ::Vector{<:Real}`: The rest frame wavelength vector of the spectrum being fit
"""
function parse_lines(channel::Integer, interp_R::Function, λ::Vector{<:Real})

    @debug """\n
    Parsing lines file
    #######################################################
    """

    # Read in the lines file
    lines = TOML.parsefile(joinpath(@__DIR__, "lines.toml"))
    lines_out = Param.LineDict()

    keylist1 = ["tie_voigt_mixing", "voff_plim", "fwhm_pmax", "h3_plim", "h4_plim", "acomp_voff_plim", 
        "acomp_fwhm_plim", "flexible_wavesol", "wavesol_unc", "lines", "profiles", "acomps"]

    # Loop through all the required keys that should be in the file and confirm that they are there
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
    acomp_profiles = Dict{String, Union{String,Nothing}}(ln => nothing for ln ∈ keys(lines["lines"]))
    for line ∈ keys(lines["lines"])
        if haskey(lines, "acomps")
            if haskey(lines["acomps"], line)
                acomp_profiles[line] = lines["acomps"][line]
            end
        end
    end

    # Minimum possible FWHM of a narrow line given the instrumental resolution of MIRI 
    # in the given wavelength range: Δλ/λ = Δv/c ---> Δv = c/(λ/Δλ) = c/R
    fwhm_pmin = Util.C_KMS / maximum(interp_R.(λ))
    @debug "Setting minimum FWHM to $fwhm_pmin km/s"

    # Define the initial values of line parameters given the values in the options file (if present)
    fwhm_init = "fwhm_init" ∈ keys(lines) ? lines["fwhm_init"] : max(fwhm_pmin + 1, 100)
    voff_init = "voff_init" ∈ keys(lines) ? lines["voff_init"] : 0.0
    h3_init = "h3_init" ∈ keys(lines) ? lines["h3_init"] : 0.0        # gauss-hermite series start fully gaussian,
    h4_init = "h4_init" ∈ keys(lines) ? lines["h4_init"] : 0.0        # with both h3 and h4 moments starting at 0
    η_init = "eta_init" ∈ keys(lines) ? lines["eta_init"] : 0.5       # Voigts start half gaussian, half lorentzian

    # Create the kinematic groups
    kinematic_groups = []
    for key ∈ keys(lines)
        if occursin("kinematic_group_", key)
            append!(kinematic_groups, [replace(key, "kinematic_group_" => "")])
        end
    end

    # Loop through all the lines
    for line ∈ keys(lines["lines"])

        @debug """\n
        ################# $line #######################
        # Rest wavelength: $(lines["lines"][line]) um #
        """

        # Set the priors for FWHM, voff, h3, h4, and eta based on the values in the options file
        voff_prior = Uniform(lines["voff_plim"]...)
        voff_locked = false
        fwhm_prior = Uniform(fwhm_pmin, profiles[line] == "GaussHermite" ? 
                             lines["fwhm_pmax"] * 2 #= allow GH prof. to be wide =# : lines["fwhm_pmax"])
        fwhm_locked = false
        if profiles[line] == "GaussHermite"
            h3_prior = truncated(Normal(0.0, 0.1), lines["h3_plim"]... #= normal profile, but truncated with hard limits =#)
            h3_locked = false
            h4_prior = truncated(Normal(0.0, 0.1), lines["h4_plim"]... #= normal profile, but truncated with hard limits =#)
            h4_locked = false
        elseif profiles[line] == "Voigt"
            η_prior = Uniform(0.0, 1.0)
            η_locked = false
        end

        # Set the priors for additional component FWHM, voff, h3, h4, and eta based on the values in the options file
        acomp_voff_prior = Uniform(lines["acomp_voff_plim"]...)
        acomp_voff_locked = false
        acomp_fwhm_prior = Uniform(lines["acomp_fwhm_plim"]...)
        acomp_fwhm_locked = false
        if acomp_profiles[line] == "GaussHermite"
            acomp_h3_prior = truncated(Normal(0.0, 0.1), lines["h3_plim"]...)
            acomp_h3_locked = false
            acomp_h4_prior = truncated(Normal(0.0, 0.1), lines["h4_plim"]...)
            acomp_h4_locked = false
        elseif acomp_profiles[line] == "Voigt"
            acomp_η_prior = Uniform(0.0, 1.0)
            acomp_η_locked = false
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

                if haskey(lines["priors"][line], "acomp_voff")
                    @debug "Overriding acomp voff prior"
                    acomp_voff_prior = eval(Meta.parse(lines["priors"][line]["acomp_voff"]["pstr"]))
                    acomp_voff_locked = lines["priors"][line]["acomp_voff"]["locked"]
                end
                if haskey(lines["priors"][line], "acomp_fwhm")
                    @debug "Overriding acomp fwhm prior"
                    acomp_fwhm_prior = eval(Meta.parse(lines["priors"][line]["acomp_fwhm"]["pstr"]))
                    acomp_fwhm_locked = lines["priors"][line]["acomp_fwhm"]["locked"]
                end
                if haskey(lines["priors"][line], "acomp_h3")
                    @debug "Overriding acomp h3 prior"
                    acomp_h3_prior = eval(Meta.parse(lines["priors"][line]["acomp_h3"]["pstr"]))
                    acomp_h3_locked = lines["priors"][line]["acomp_h3"]["locked"]
                end
                if haskey(lines["priors"][line], "acomp_h4")
                    @debug "Overriding acomp h4 prior"
                    acomp_h4_prior = eval(Meta.parse(lines["priors"][line]["acomp_h4"]["pstr"]))
                    acomp_h4_locked = lines["priors"][line]["acomp_h4"]["locked"]
                end
                if haskey(lines["priors"][line], "acomp_eta")
                    @debug "Overriding acomp eta prior"
                    acomp_η_prior = eval(Meta.parse(lines["priors"][line]["acomp_eta"]["pstr"]))
                    acomp_η_locked = lines["priors"][line]["acomp_eta"]["locked"]
                end
            end
        end

        # Check if the kinematics should be tied to other lines based on the kinematic groups
        tied = acomp_tied = nothing
        for group ∈ kinematic_groups
            for groupmember ∈ lines["kinematic_group_" * group]
                #= Loop through the items in the "kinematic_group_*" list and see if the line name matches any of them.
                 It needn't be a perfect match, the line name just has to contain the value in the kinematic group list.
                 i.e. if you want to automatically tie all FeII lines together, instead of manually listing out each one,
                 you can just include an item "FeII" and it will automatically catch all the FeII lines
                =#
                if occursin(groupmember, line)
                    # Make sure line is not already a member of another group
                    if !isnothing(tied)
                        error("Line $line is already part of the kinematic group $tied, but it also passed filtering criteria"
                            * "to be included in the group $group. Make sure your filters are not too lenient!")
                    end
                    @debug "Tying kinematics for $line to the group: $group"
                    # Use the group label (which can be anything you want) to categorize what lines are tied together
                    tied = group
                    # Only set acomp_tied if the line actually *has* an acomp
                    if !isnothing(acomp_profiles[line])
                        acomp_tied = group
                    end
                    # If the wavelength solution is bad, allow the kinematics to still be flexible based on its accuracy
                    if lines["flexible_wavesol"]
                        δv = lines["wavesol_unc"][channel]
                        voff_prior = Uniform(-δv, δv)
                        @debug "Using flexible tied voff with lenience of +/-$δv km/s"
                    end
                    break
                end
            end
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
        # Do the same for the additional component parameters, but only if the line has an additional component
        if !isnothing(acomp_profiles[line])
            @debug "acomp profile: $(acomp_profiles[line])"

            acomp_voff = Param.Parameter(0., acomp_voff_locked, acomp_voff_prior)
            @debug "Voff $acomp_voff"
            acomp_fwhm = Param.Parameter(1., acomp_fwhm_locked, acomp_fwhm_prior)
            @debug "FWHM $acomp_fwhm"
            if acomp_profiles[line] ∈ ("Gaussian", "Lorentzian")
                acomp_params = Param.ParamDict(:acomp_voff => acomp_voff, :acomp_fwhm => acomp_fwhm)
            elseif acomp_profiles[line] == "GaussHermite"
                acomp_h3 = Param.Parameter(h3_init, acomp_h3_locked, acomp_h3_prior)
                @debug "h3 $acomp_h3"
                acomp_h4 = Param.Parameter(h4_init, acomp_h4_locked, acomp_h4_prior)
                @debug "h4 $acomp_h4"
                acomp_params = Param.ParamDict(:acomp_voff => acomp_voff, :acomp_fwhm => acomp_fwhm, :acomp_h3 => acomp_h3, :acomp_h4 => acomp_h4)
            elseif acomp_profiles[line] == "Voigt"
                acomp_η = Param.Parameter(η_init, acomp_η_locked, acomp_η_prior)
                @debug "eta $acomp_η"
                acomp_params = Param.ParamDict(:acomp_voff => acomp_voff, :acomp_fwhm => acomp_fwhm, :acomp_mixing => acomp_η)
            end
            params = merge(params, acomp_params)
        end

        # Create the TransitionLine object using the above parameters, and add it to the dictionary
        lines_out[Symbol(line)] = Param.TransitionLine(lines["lines"][line], 
            Symbol(profiles[line]), !isnothing(acomp_profiles[line]) ? Symbol(acomp_profiles[line]) : nothing, params, 
            tied, acomp_tied)

    end


    @debug "#######################################################"

    # Create a dictionary containing all of the unique `tie` keys, and the tied voff parameters 
    # corresponding to that tied key
    kin_tied_key = unique([lines_out[line].tied for line ∈ keys(lines_out)])
    kin_tied_key = kin_tied_key[.!isnothing.(kin_tied_key)]
    @debug "kin_tied_key: $kin_tied_key"

    voff_tied = Dict{String, Param.Parameter}()
    fwhm_tied = Dict{String, Param.Parameter}()
    msg = ""
    for (i, kin_tie) ∈ enumerate(kin_tied_key)
        v_prior = Uniform(lines["voff_plim"]...)
        f_prior = Uniform(fwhm_pmin, lines["fwhm_pmax"])
        v_locked = f_locked = false
        # Check if there is an overwrite option in the lines file
        if haskey(lines, "priors")
            if haskey(lines["priors"], kin_tie)
                v_prior = lines["priors"][kin_tie]["voff_pstr"]
                v_locked = lines["priors"][kin_tie]["voff_locked"]
                f_prior = lines["priors"][kin_tie]["fwhm_pstr"]
                f_locked = lines["priors"][kin_tie]["fwhm_locked"]
            end
        end
        voff_tied[kin_tie] = Param.Parameter(voff_init, v_locked, v_prior)
        fwhm_tied[kin_tie] = Param.Parameter(fwhm_init, f_locked, f_prior)
        msg *= "\nvoff_tied_$i $(voff_tied[kin_tie])"
        msg *= "\nfwhm_tied_$i $(fwhm_tied[kin_tie])"
    end
    @debug msg

    # ^^^ do the same for the additional tied kinematics
    acomp_kin_tied_key = unique([lines_out[line].acomp_tied for line ∈ keys(lines_out)])
    acomp_kin_tied_key = acomp_kin_tied_key[.!isnothing.(acomp_kin_tied_key)]
    @debug "acomp_kin_tied_key: $acomp_kin_tied_key"

    acomp_voff_tied = Dict{String, Param.Parameter}()
    acomp_fwhm_tied = Dict{String, Param.Parameter}()
    msg = ""
    for (i, acomp_kin_tie) ∈ enumerate(acomp_kin_tied_key)
        av_prior = Uniform(lines["acomp_voff_plim"]...)
        af_prior = Uniform(lines["acomp_fwhm_plim"]...)
        av_locked = af_locked = false
        # Check if there is an overwrite option in the lines file
        if haskey(lines, "priors")
            if haskey(lines["priors"], acomp_kin_tie)
                av_prior = lines["priors"][acomp_kin_tie]["voff_pstr"]
                av_locked = lines["priors"][acomp_kin_tie]["voff_locked"]
                af_prior = lines["priors"][acomp_kin_tie]["fwhm_pstr"]
                af_locked = lines["priors"][acomp_kin_tie]["fwhm_locked"]
            end
        end
        acomp_voff_tied[acomp_kin_tie] = Param.Parameter(0., av_locked, av_prior)
        acomp_fwhm_tied[acomp_kin_tie] = Param.Parameter(1., af_locked, af_prior)
        msg *= "\nacomp_voff_tied_$i $(acomp_voff_tied[acomp_kin_tie])"
        msg *= "\nacomp_fwhm_tied_$i $(acomp_fwhm_tied[acomp_kin_tie])"
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

    lines_out, voff_tied, fwhm_tied, acomp_voff_tied, acomp_fwhm_tied, lines["flexible_wavesol"], lines["tie_voigt_mixing"], 
        voigt_mix_tied
end


"""
    fit_optical_depth(obs)

Gets an estimated value for tau_9.7, the optical depth at 9.7 um, for each spaxel in an observation.
This requires rebinning the cube to get full spectral coverage across the spatial dimension, then linearly
interpolating the continuum from 6.7-13 um, and taking the ratio with the observed value at 9.7 um.  This
is done four times, each time rebinning the spaxels spatially to match the grid of one of the MIRI channels.
This is necessary to preserve the highest spatail resolution in each of the four channels.

# Arguments
- `obs::Observation`: The Observation object containing the MIRI data in all four channels to be fit.
"""
function fit_optical_depth(obs::Observation)

    @info "Fitting optical depth at 9.7 um with linear interpolation..."
    name = replace(obs.name, " " => "_")

    # Create dictionary to hold outputs
    τ_97 = Dict{Int, Matrix{Float64}}()
    τ_97[0] = zeros(length(keys(obs.channels)),1)
    if !isdir("output_$(name)_optical_depth")
        mkdir("output_$(name)_optical_depth")
    end
    
    # Check for outputs that already have been saved
    c1 = false
    if isfile(joinpath("output_$(name)_optical_depth", "optical_depth_$(name)_sum.csv"))
        @debug "Optical depth sum file found for $(name)"
        τ_97[0] = readdlm(joinpath("output_$(name)_optical_depth", "optical_depth_$(name)_sum.csv"), 
            ',', Float64, '\n')
        c1 = true 
    end

    # Loop through each channel
    for channel ∈ keys(obs.channels)

        # Check if file already exists
        c2 = false
        if isfile(joinpath("output_$(name)_optical_depth", "optical_depth_$(name)_ch$channel.csv"))
            @debug "Optical depth files found for $(name) in channel $channel"
            τ_97[channel] = readdlm(joinpath("output_$(name)_optical_depth", "optical_depth_$(name)_ch$channel.csv"), 
                ',', Float64, '\n')
            c2 = true
        end

        if c1 && c2
            continue
        end

        # Rebin the other channels onto this channel's grid
        cube_combine!(obs, out_grid=channel, out_id=0)
        τ_97[channel] = zeros(size(obs.channels[0].Iν)[1:2])

        # Find specific wavelength points in the vectors
        _, p1 = findmin(x -> abs(x - 6.7), obs.channels[0].λ)
        λ1 = obs.channels[0].λ[p1]
        _, p2 = findmin(x -> abs(x - 13), obs.channels[0].λ)
        λ2 = obs.channels[0].λ[p2]
        _, p3 = findmin(x -> abs(x - 9.7), obs.channels[0].λ)

        # Loop through each spaxel
        @debug "Calculating optical depth for each spaxel in channel $channel"
        for spax ∈ CartesianIndices(size(obs.channels[0].Iν)[1:2])
            # Linear interpolation from 6.7 um to 13 um
            i1 = mean(obs.channels[0].Iν[spax, p1-5:p1+5])
            i2 = mean(obs.channels[0].Iν[spax, p2-5:p2+5])
            slope = (i2 - i1) / (λ2 - λ1)
            # extrapolate to get intrinsic flux value at 9.7 um
            i_97 = i1 + slope * (9.7 - λ1)
            # get the observed flux at 9.7 um
            o_97 = mean(obs.channels[0].Iν[spax, p3-5:p3+5])
            # take the ratio of the observed to intrinsic flux to get the optical depth
            # REFERENCE: Donnan et al. 2023, MNRAS 519, 3691-3705 https://doi.org/10.1093/mnras/stac3729
            ratio = (o_97 / i_97) > 0 ? (o_97 / i_97) : 1.
            ratio == 1. && @debug "Spaxel $spax, ratio <= 0 with obs. $o_97 and intrin. $i_97"
            τ_97[channel][spax] = max(0., -log(ratio) / 0.9)
        end

        # Calculate for the sum of spaxels
        i1 = mean(Util.Σ(obs.channels[0].Iν, (1,2))[p1-5:p1+5])
        i2 = mean(Util.Σ(obs.channels[0].Iν, (1,2))[p2-5:p2+5])
        slope = (i2 - i1) / (λ2 - λ1)
        i_97 = i1 + slope * (9.7 - λ1)
        o_97 = mean(Util.Σ(obs.channels[0].Iν, (1,2))[p3-5:p3+5])
        ratio = (o_97 / i_97) > 0 ? (o_97 / i_97) : 1.
        ratio == 1. && @debug "Sum, ratio <= 0 with obs. $o_97 and intrin. $i_97"
        τ_97[0][channel,1] = max(0., -log(ratio) / 0.9)

        # save outputs to CSV files
        @debug "Writing outputs to file optical_depth_$(name)_ch$channel.csv"
        open(joinpath("output_$(name)_optical_depth", "optical_depth_$(name)_ch$channel.csv"), "w") do file
            writedlm(file, τ_97[channel], ',')
        end

    end

    open(joinpath("output_$(name)_optical_depth", "optical_depth_$(name)_sum.csv"), "w") do file
        writedlm(file, τ_97[0], ',')
    end

    τ_97
end


############################## PARAMETER / MODEL STRUCTURES ####################################


"""
    ParamMaps(stellar_continuum, dust_continuum, dust_features, lines, tied_voffs, acomp_tied_voffs,
        tied_voigt_mix, extinction, hot_dust, reduced_χ2)

A structure for holding 2D maps of fitting parameters generated after fitting a cube.  Each parameter
that is fit (i.e. stellar continuum temperature, optical depth, line ampltidue, etc.) corresponds to one 
2D map with the value of that parameter, with each spaxel's fit value located at the corresponding location
in the parameter map.

# Fields
- `stellar_continuum::Dict{Symbol, Array{Float64, 2}}`: The stellar continuum parameters: amplitude and temperature
- `dust_continuum::Dict{Int, Dict{Symbol, Array{Float64, 2}}}`: The dust continuum parameters: amplitude and temperature for each
    dust continuum component
- `dust_features::Dict{String, Dict{Symbol, Array{Float64, 2}}}`: The dust feature parameters: amplitude, central wavelength, and FWHM
    for each PAH feature
- `lines::Dict{Symbol, Dict{Symbol, Array{Float64, 2}}}`: The emission line parameters: amplitude, voff, FWHM, and any additional 
    line profile parameters for each line
- `tied_voffs::Dict{String, Array{Float64, 2}}`: Tied line velocity offsets
- `tied_fwhms::Dict{String, Array{Float64, 2}}`: Tied line velocity fwhms
- `acomp_tied_voffs::Dict{String, Array{Float64, 2}}`: additional component tied velocity offsets
- `acomp_tied_fwhms::Dict{String, Array{Float64, 2}}`: additional component tied velocity fwhms
- `tied_voigt_mix::Union{Array{Float64, 2}, Nothing}`: Tied Voigt mixing parameter
- `extinction::Dict{Symbol, Array{Float64, 2}}`: Extinction parameters: optical depth at 9.7 μm and mixing ratio
- `hot_dust::Dict{Symbol, Array{Float64, 2}}`: Hot dust parameters: amplitude, temperature, covering fraction, warm tau, and cold tau
- `reduced_χ2::Array{Float64, 2}`: The reduced chi^2 value of each fit

See ['parammaps_empty`](@ref) for a default constructor function.
"""
struct ParamMaps{T<:Real}

    stellar_continuum::Dict{Symbol, Array{T, 2}}
    dust_continuum::Dict{Int, Dict{Symbol, Array{T, 2}}}
    dust_features::Dict{String, Dict{Symbol, Array{T, 2}}}
    lines::Dict{Symbol, Dict{Symbol, Array{T, 2}}}
    tied_voffs::Dict{String, Array{T, 2}}
    tied_fwhms::Dict{String, Array{T, 2}}
    acomp_tied_voffs::Dict{String, Array{T, 2}}
    acomp_tied_fwhms::Dict{String, Array{T, 2}}
    tied_voigt_mix::Union{Array{T, 2}, Nothing}
    extinction::Dict{Symbol, Array{T, 2}}
    hot_dust::Dict{Symbol, Array{T, 2}}
    reduced_χ2::Array{T, 2}

end


"""
    parammaps_empty(shape, n_dust_cont, df_names, line_names, line_tied, line_profiles,
        line_acomp_tied, line_acomp_profiles, kin_tied_key, acomp_kin_tied_key, flexible_wavesol, 
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
- `line_acomp_tied::Vector{Union{String,Nothing}}`: Same as line_tied, but for additional line components
- `line_acomp_profiles::Vector{Union{String,Nothing}}`: Same as line_profiles, but for additional line components. An element
    may be `nothing` if the line in question has no additional component.
- `kin_tied_key::Vector{String}`: List of only the unique keys in line_tied (not including `nothing`)
- `acomp_kin_tied_key::Vector{String}`: Same as kin_tied_key, but for additional line components
- `flexible_wavesol::Bool`: Fitting option on whether to allow a small variation in voff components even when they are tied,
    in case the wavelength solution of the data is not calibrated very well
- `tie_voigt_mixing::Bool`: Whether or not to tie the mixing parameters of any Voigt line profiles
"""
function parammaps_empty(shape::Tuple{S,S,S}, n_dust_cont::Integer, df_names::Vector{String}, 
    line_names::Vector{Symbol}, line_tied::Vector{Union{String,Nothing}}, line_profiles::Vector{Symbol},
    line_acomp_tied::Vector{Union{String,Nothing}}, line_acomp_profiles::Vector{Union{Symbol,Nothing}},
    kin_tied_key::Vector{String}, acomp_kin_tied_key::Vector{String}, flexible_wavesol::Bool, 
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
        dust_features[n][:eqw] = copy(nan_arr)
        dust_features[n][:SNR] = copy(nan_arr)
        @debug "dust feature $n maps with keys $(keys(dust_features[n]))"
    end

    # Nested dictionary -> first layer keys are line names, second layer keys are parameter names, which contain 2D arrays
    lines = Dict{Symbol, Dict{Symbol, Array{Float64, 2}}}()
    for (line, tie, prof, acomptie, acompprof) ∈ zip(line_names, line_tied, line_profiles, line_acomp_tied, line_acomp_profiles)
        lines[line] = Dict{Symbol, Array{Float64, 2}}()
        # If tied and NOT using a flexible solution, don't include a voff parameter
        if isnothing(tie)
            pnames = [:amp, :voff, :fwhm]
        elseif flexible_wavesol
            pnames = [:amp, :voff]
        else
            pnames = [:amp]
        end
        # Add 3rd and 4th order moments (skewness and kurtosis) for Gauss-Hermite profiles
        if prof == :GaussHermite
            pnames = [pnames; :h3; :h4]
        # Add mixing parameter for Voigt profiles, but only if NOT tying it
        elseif prof == :Voigt && !tie_voigt_mixing
            pnames = [pnames; :mixing]
        end
        # Repeat the above but for additional components
        if !isnothing(acompprof)
            pnames = isnothing(acomptie) ? [pnames; :acomp_amp; :acomp_voff; :acomp_fwhm] : [pnames; :acomp_amp]
            if acompprof == :GaussHermite
                pnames = [pnames; :acomp_h3; :acomp_h4]
            elseif acompprof == :Voigt && !tie_voigt_mixing
                pnames = [pnames; :acomp_mixing]
            end
        end
        # Append parameters for intensity and signal-to-noise ratio, which are NOT fitting parameters, but are of interest
        pnames = [pnames; :intI; :eqw; :SNR]
        if !isnothing(acompprof)
            pnames = [pnames; :acomp_intI; :acomp_eqw; :acomp_SNR]
        end
        for pname ∈ pnames
            lines[line][pname] = copy(nan_arr)
        end
        @debug "line $line maps with keys $pnames"
    end

    # Tied voff parameters
    tied_voffs = Dict{String, Array{Float64, 2}}()
    for vk ∈ kin_tied_key
        tied_voffs[vk] = copy(nan_arr)
        @debug "tied voff map for group $vk"
    end
    # Tied fwhm parameters
    tied_fwhms = Dict{String, Array{Float64, 2}}()
    for vk ∈ kin_tied_key
        tied_fwhms[vk] = copy(nan_arr)
        @debug "tied fwhm map for group $vk"
    end

    # Tied additional component voff parameters
    acomp_tied_voffs = Dict{String, Array{Float64, 2}}()
    for fvk ∈ acomp_kin_tied_key
        acomp_tied_voffs[fvk] = copy(nan_arr)
        @debug "tied acomp voff map for group $fvk"
    end
    # Tied additional component fwhm parameters
    acomp_tied_fwhms = Dict{String, Array{Float64, 2}}()
    for fvk ∈ acomp_kin_tied_key
        acomp_tied_fwhms[fvk] = copy(nan_arr)
        @debug "tied acomp fwhm map for group $fvk"
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
    extinction[:tau_ice] = copy(nan_arr)
    extinction[:tau_ch] = copy(nan_arr)
    extinction[:beta] = copy(nan_arr)
    @debug "extinction maps with keys $(keys(extinction))"

    # Add hot dust fitting parameters
    hot_dust = Dict{Symbol, Array{Float64, 2}}()
    hot_dust[:amp] = copy(nan_arr)
    hot_dust[:temp] = copy(nan_arr)
    hot_dust[:frac] = copy(nan_arr)
    hot_dust[:tau_warm] = copy(nan_arr)
    hot_dust[:tau_cold] = copy(nan_arr)
    @debug "hot dust maps with keys $(keys(hot_dust))"

    # Reduced chi^2 of the fits
    reduced_χ2 = copy(nan_arr)
    @debug "reduced chi^2 map"

    ParamMaps{Float64}(stellar_continuum, dust_continuum, dust_features, lines, tied_voffs, tied_fwhms, 
        acomp_tied_voffs, acomp_tied_fwhms, tied_voigt_mix, extinction, hot_dust, reduced_χ2)
end


"""
    CubeModel(model, stellar, dust_continuum, dust_features, extinction, hot_dust, lines)

A structure for holding 3D models of intensity, split up into model components, generated when fitting a cube.
This will be the same shape as the input data, and preferably the same datatype too (i.e., JWST files have flux
and error in Float32 format, so we should also output in Float32 format).  This is useful as a way to quickly
compare the full model, or model components, to the data.

# Fields
- `model::Array{T, 3}`: The full 3D model.
- `stellar::Array{T, 3}`: The stellar component of the continuum.
- `dust_continuum::Array{T, 4}`: The dust components of the continuum. The 4th axis runs over each individual dust component.
- `dust_features::Array{T, 4}`: The dust (PAH) feature profiles. The 4th axis runs over each individual dust profile.
- `extinction::Array{T, 3}`: The extinction profile.
- `hot_dust::Array{T, 3}`: The hot dust emission profile
- `lines::Array{T, 4}`: The line profiles. The 4th axis runs over each individual line.

See [`cubemodel_empty`](@ref) for a default constructor method.
"""
struct CubeModel{T<:Real}

    model::Array{T, 3}
    stellar::Array{T, 3}
    dust_continuum::Array{T, 4}
    dust_features::Array{T, 4}
    extinction::Array{T, 3}
    abs_ice::Array{T, 3}
    abs_ch::Array{T, 3}
    hot_dust::Array{T, 3}
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
- `floattype::DataType=Float32`: The type of float to use in the arrays. Should ideally be the same as the input data,
    which for JWST is Float32.
"""
function cubemodel_empty(shape::Tuple{S,S,S}, n_dust_cont::Integer, df_names::Vector{String}, 
    line_names::Vector{Symbol}, floattype::DataType=Float32)::CubeModel where {S<:Integer}

    @debug """\n
    Creating CubeModel struct with shape $shape
    ###########################################
    """

    # Make sure the floattype given is actually a type of float
    @assert floattype <: AbstractFloat "floattype must be a type of AbstractFloat (Float32 or Float64)!"

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
    abs_ice = zeros(floattype, shape...)
    @debug "abs_ice comp cube"
    abs_ch = zeros(floattype, shape...)
    @debug "abs_ch comp cube"
    hot_dust = zeros(floattype, shape...)
    @debug "hot dust comp cube"
    lines = zeros(floattype, shape..., length(line_names))
    @debug "lines comp cubes"

    CubeModel(model, stellar, dust_continuum, dust_features, extinction, abs_ice, abs_ch, hot_dust, lines)
end


"""
    CubeFitter(cube, z, name, n_procs; plot_spaxels=plot_spaxels, 
        plot_maps=plot_maps, parallel=parallel, save_fits=save_fits)

This is the main structure used for fitting IFU cubes, containing all of the necessary data, metadata,
fitting options, and associated functions for generating ParamMaps and CubeModel structures to handle the outputs 
of all the fits.  This is essentially the "configuration object" that tells the rest of the fitting code how
to run. The actual fitting functions (`fit_spaxel` and `fit_cube!`) require an instance of this structure.

# Fields
`T<:Real, S<:Integer`
- `cube::CubeData.DataCube`: The main DataCube object containing the cube that is being fit
- `z::Real`: The redshift of the target that is being fit
- `τ_guess::Real`: Initial guess for the optical depth
- `n_procs::Integer`: The number of parallel processes that are being used in the fitting procedure
- `plot_spaxels::Symbol=:pyplot`: A Symbol specifying the plotting backend to be used when plotting individual spaxel fits, can
    be either `:pyplot` or `:plotly`
- `plot_maps::Bool=true`: Whether or not to plot 2D maps of the best-fit parameters after the fitting is finished
- `parallel::Bool=true`: Whether or not to fit multiple spaxels in parallel using multiprocessing
- `save_fits::Bool=true`: Whether or not to save the final best-fit models and parameters as FITS files
Read from the options files:
- `overwrite::Bool`: Whether or not to overwrite old fits of spaxels when rerunning
- `track_memory::Bool`: Whether or not to save diagnostic files showing memory usage of the program
- `track_convergence::Bool`: Whether or not to save diagnostic files showing convergence of line fitting for each spaxel
- `make_movies::Bool`: Whether or not to save mp4 files of the final model
- `extinction_curve::String`: The type of extinction curve being used, either `"kvt"` or `"d+"`
- `extinction_screen::Bool`: Whether or not the extinction is modeled as a screen
- `T_s::Param.Parameter`: The stellar temperature parameter
- `T_dc::Vector{Param.Parameter}`: The dust continuum temperature parameters
- `τ_97::Param.Parameter`: The dust opacity at 9.7 um parameter
- `τ_ice::Param.Parameter`: The peak opacity from ice absorption (at around 6.9 um)
- `τ_ch::Param.Parameter`: The peak opacity from CH absorption (at around 6.9 um)
- `β::Param.Parameter`: The extinction profile mixing parameter
- `T_hot::Param.Parameter`: The hot dust temperature
- `Cf_hot::Param.Parameter`: The hot dust covering fraction
- `τ_warm::Param.Parameter`: The warm dust optical depth
- `τ_cold::Param.Parameter`: The cold dust optical depth
- `n_dust_cont::Integer`: The number of dust continuum profiles
- `df_names::Vector{String}`: The names of each PAH feature profile
- `dust_features::Vector{Dict}`: All of the fitting parameters for each PAH feature
- `n_lines::Integer`: The number of lines being fit
- `line_names::Vector{Symbol}`: The names of each line being fit
- `line_profiles::Vector{Symbol}`: The profiles of each line being fit
- `line_acomp_profiles::Vector{Union{Nothing,Symbol}}`: Same as `line_profiles`, but for the additional components
- `lines::Vector{Param.TransitionLine}`: All of the fitting parameters for each line
- `n_kin_tied::Integer`: The number of tied velocity offsets
- `line_tied::Vector{Union{String,Nothing}}`: List of line tie keys which specify whether the voff of the given line should be
    tied to other lines. The line tie key itself may be either `nothing` (untied), or a String specifying the group of lines
    being tied, i.e. "H2"
- `kin_tied_key::Vector{String}`: List of only the unique keys in line_tied (not including `nothing`)
- `voff_tied::Vector{Param.Parameter}`: The actual tied voff parameter objects, corresponding to the `kin_tied_key`
- `fwhm_tied::Vector{Param.Parameter}`: The actual tied fwhm parameter objects, corresponding to the `kin_tied_key`
- `n_acomp_kin_tied::Integer`: Same as `n_kin_tied`, but for additional components
- `line_acomp_tied::Vector{Union{String,Nothing}}`: Same as `line_tied`, but for additional components
- `acomp_kin_tied_key::Vector{String}`: Same as `kin_tied_key`, but for additional components
- `acomp_voff_tied::Vector{Param.Parameter}`: Same as `voff_tied`, but for additional components
- `acomp_fwhm_tied::Vector{Param.Parameter}`: Same as `fwhm_tied`, but for additional components
- `tie_voigt_mixing::Bool`: Whether or not the Voigt mixing parameter is tied between all the lines with Voigt profiles
- `voigt_mix_tied::Param.Parameter`: The actual tied Voigt mixing parameter object, given `tie_voigt_mixing` is true
- `n_params_cont::Integer`: The total number of free fitting parameters for the continuum fit (not including emission lines)
- `n_params_lines::Integer`: The total number of free fitting parameters for the emission line fit (not including the continuum)
- `cosmology::Cosmology.AbstractCosmology`: The Cosmology, used solely to create physical scale bars on the 2D parameter plots
- `χ²_thresh::Real`: The threshold for reduced χ² values, below which the best fit parameters for a given
    row will be set
- `interp_R::Function`: Interpolation function for the instrumental resolution as a function of wavelength
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
    [`fit_spaxel`](@ref), [`fit_cube!`](@ref)
"""
struct CubeFitter{T<:Real,S<:Integer}

    # See explanations for each field in the docstring!
    
    # Data
    cube::CubeData.DataCube
    z::T
    τ_guess::Dict{Int, Matrix{T}}
    name::String

    # Basic fitting options
    n_procs::S
    plot_spaxels::Symbol
    plot_maps::Bool
    plot_range::Union{Vector{<:Tuple},Nothing}
    parallel::Bool
    save_fits::Bool
    save_full_model::Bool
    overwrite::Bool
    track_memory::Bool
    track_convergence::Bool
    make_movies::Bool
    extinction_curve::String
    extinction_screen::Bool
    fit_sil_emission::Bool

    # Continuum parameters
    T_s::Param.Parameter
    T_dc::Vector{Param.Parameter}
    τ_97::Param.Parameter
    τ_ice::Param.Parameter
    τ_ch::Param.Parameter
    β::Param.Parameter
    T_hot::Param.Parameter
    Cf_hot::Param.Parameter
    τ_warm::Param.Parameter
    τ_cold::Param.Parameter
    n_dust_cont::S
    n_dust_feat::S
    df_names::Vector{String}
    dust_features::Vector{Dict}

    # Line parameters
    n_lines::S
    n_acomps::S
    line_names::Vector{Symbol}
    line_profiles::Vector{Symbol}
    line_acomp_profiles::Vector{Union{Nothing,Symbol}}
    lines::Vector{Param.TransitionLine}

    # Tied voffs
    n_kin_tied::S
    line_tied::Vector{Union{String,Nothing}}
    kin_tied_key::Vector{String}
    voff_tied::Vector{Param.Parameter}
    fwhm_tied::Vector{Param.Parameter}

    # Tied additional component voffs
    n_acomp_kin_tied::S
    line_acomp_tied::Vector{Union{String,Nothing}}
    acomp_kin_tied_key::Vector{String}
    acomp_voff_tied::Vector{Param.Parameter}
    acomp_fwhm_tied::Vector{Param.Parameter}

    # Tied voigt mixing
    tie_voigt_mixing::Bool
    voigt_mix_tied::Param.Parameter

    # Number of parameters
    n_params_cont::S
    n_params_lines::S
    n_params_extra::S
    
    # Rolling best fit options
    cosmology::Cosmology.AbstractCosmology
    interp_R::Function
    flexible_wavesol::Bool

    p_init_cont::Vector{T}
    p_init_line::Vector{T}

    # Store the astropy version of the 2D WCS transformation object,
    # for nice plotting with matplotlib
    python_wcs::PyObject

    #= Constructor function --> the inputs taken map directly to fields in the CubeFitter object,
    the rest of the fields are generated in the function from these inputs =#
    function CubeFitter(cube::CubeData.DataCube, z::Real, τ_guess::Dict{Int, Matrix{T}},
        name::String, n_procs::Integer; plot_spaxels::Symbol=:pyplot, plot_maps::Bool=true, 
        plot_range::Union{Vector{<:Tuple},Nothing}=nothing, parallel::Bool=true, save_fits::Bool=true) where {T<:Real}

        # Prepare output directories
        @info "Preparing output directories"
        name = replace(name, #= no spaces! =# " " => "_")

        # Top-level output directory
        if !isdir("output_$name")
            mkdir("output_$name")
        end
        # Sub-folder for 1D plots of spaxel fits
        if !isdir(joinpath("output_$name", "spaxel_plots"))
            mkdir(joinpath("output_$name", "spaxel_plots"))
        end
        if !isdir(joinpath("output_$name", "zoomed_plots")) && !isnothing(plot_range)
            mkdir(joinpath("output_$name", "zoomed_plots"))
        end
        # Sub-folder for data files saving the results of individual spaxel fits
        if !isdir(joinpath("output_$name", "spaxel_binaries"))
            mkdir(joinpath("output_$name", "spaxel_binaries"))
        end
        # Sub-folder for 2D parameter maps 
        if !isdir(joinpath("output_$name", "param_maps"))
            mkdir(joinpath("output_$name", "param_maps"))
        end
        # Sub-folder for log files
        if !isdir(joinpath("output_$name", "logs"))
            mkdir(joinpath("output_$name", "logs"))
        end

        ###### SETTING UP A GLOBAL LOGGER FOR THE CUBE FITTER ######
        
        @info "Preparing logger"

        # Append timestamps to all logging messages
        timestamp_logger(logger) = TransformerLogger(logger) do log
            merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
        end
        # Create a tee logger that writes both to the stdout and to a log file
        logger = TeeLogger(ConsoleLogger(stdout, Logging.Info), 
                           timestamp_logger(MinLevelLogger(FileLogger(joinpath("output_$name", "loki.main.log"); 
                                                                      always_flush=true), 
                                                                      Logging.Debug)))
        # Initialize our logger as the global logger
        global_logger(logger)

        #############################################################

        @debug """\n
        Creating CubeFitter struct for $name at z=$z
        ############################################
        """

        # Get shape
        shape = size(cube.Iν)
        # Alias
        λ = cube.λ

        # Parse all of the options files to create default options and parameter objects
        interp_R = parse_resolving(z, parse(Int, cube.channel))
        dust = parse_dust(τ_guess[0][parse(Int, cube.channel)])
        options = parse_options()
        line_list, voff_tied, fwhm_tied, acomp_voff_tied, acomp_fwhm_tied, flexible_wavesol, tie_voigt_mixing, voigt_mix_tied = 
            parse_lines(parse(Int, cube.channel), interp_R, λ)

        # Check that number of processes doesn't exceed first dimension, so the rolling best fit can work as intended
        if n_procs > shape[1]
            error("Number of processes ($n_procs) must be ≤ the size of the first cube dimension ($(shape[1]))!")
        end

        # Get dust options from the dictionary
        T_s = dust[:stellar_continuum_temp]
        T_dc = dust[:dust_continuum_temps]
        τ_97 = dust[:extinction][:tau_9_7]
        τ_ice = dust[:extinction][:tau_ice]
        τ_ch = dust[:extinction][:tau_ch]
        β = dust[:extinction][:beta]
        T_hot = dust[:hot_dust][:temp]
        Cf_hot = dust[:hot_dust][:frac]
        τ_warm = dust[:hot_dust][:tau_warm]
        τ_cold = dust[:hot_dust][:tau_cold]

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
        # Also get the profile and acomp profile types for each line
        line_profiles = [line_list[line].profile for line ∈ line_names]
        line_acomp_profiles = Vector{Union{Symbol,Nothing}}([line_list[line].acomp_profile for line ∈ line_names])
        n_lines = length(line_names)
        n_acomps = sum(.!isnothing.(line_acomp_profiles))
        msg = "### Model will include $n_lines emission lines ###"
        for (name, ln, prof, acomp_prof) ∈ zip(line_names, lines, line_profiles, line_acomp_profiles)
            msg *= "\n### $name at lambda = $(ln.λ₀) um with $prof profile and $acomp_prof acomp profile ###"
        end
        @debug msg

        # Unpack the kin_tied dictionary
        kin_tied_key = collect(keys(voff_tied))
        voff_tied = [voff_tied[kin] for kin ∈ kin_tied_key]
        fwhm_tied = [fwhm_tied[kin] for kin ∈ kin_tied_key]
        n_kin_tied = length(voff_tied)
        # Also store the "tied" parameter for each line, which will need to be checked against the kin_tied_key
        # during fitting to find the proper location of the tied voff parameter to use
        line_tied = Vector{Union{Nothing,String}}([line.tied for line ∈ lines])
        msg = "### Model will include $(2n_kin_tied) tied voff parameters ###"
        for lt ∈ kin_tied_key
            msg *= "\n### for group $lt ###"
        end
        @debug msg

        # Repeat for additional component velocity offsets, same logic
        acomp_kin_tied_key = collect(keys(acomp_voff_tied))
        acomp_voff_tied = [acomp_voff_tied[akin] for akin ∈ acomp_kin_tied_key]
        acomp_fwhm_tied = [acomp_fwhm_tied[akin] for akin ∈ acomp_kin_tied_key]
        n_acomp_kin_tied = length(acomp_voff_tied)
        line_acomp_tied = Vector{Union{Nothing,String}}([line.acomp_tied for line ∈ lines])
        msg = "### Model will include $(2n_acomp_kin_tied) tied acomp kinematic parameters ###"
        for lft ∈ acomp_kin_tied_key
            msg *= "\n### for group $lft ###"
        end
        @debug msg

        # Total number of parameters for the continuum and line fits
        n_params_cont = (2+4) + 2n_dust_cont + 3n_dust_features + (options[:fit_sil_emission] ? 5 : 0)
        n_params_lines = 2n_kin_tied + 2n_acomp_kin_tied
        # One η for all voigt profiles
        if (any(line_profiles .== :Voigt) || any(line_acomp_profiles .== :Voigt)) && tie_voigt_mixing
            n_params_lines += 1
            @debug "### Model will include 1 tied voigt mixing parameter ###"
        end
        for i ∈ 1:n_lines
            if isnothing(line_tied[i]) || flexible_wavesol
                # amplitude and voff parameters
                n_params_lines += 2
            else
                # no voff or FWHM parameter, since they're tied
                n_params_lines += 1
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
            # Repeat above for the additional components
            if !isnothing(line_acomp_profiles[i])
                if isnothing(line_acomp_tied[i])
                    n_params_lines += 2
                else
                    n_params_lines += 1
                end
                if line_acomp_profiles[i] == :GaussHermite
                    n_params_lines += 2
                elseif line_acomp_profiles[i] == :Voigt
                    if !tie_voigt_mixing
                        n_params_lines += 1
                    end
                end
            end
        end
        n_params_extra = 3 * (n_dust_features + n_lines + n_acomps)
        @debug "### This totals to $(n_params_cont) continuum parameters ###"
        @debug "### This totals to $(n_params_lines) emission line parameters ###"
        @debug "### This totals to $(n_params_extra) extra parameters ###"

        # Prepare options
        extinction_curve = options[:extinction_curve]
        extinction_screen = options[:extinction_screen]
        fit_sil_emission = options[:fit_sil_emission]
        overwrite = options[:overwrite]
        track_memory = options[:track_memory]
        track_convergence = options[:track_convergence]
        save_full_model = options[:save_full_model]
        make_movies = options[:make_movies]
        cosmo = options[:cosmology]

        # Prepare initial best fit parameter options
        @debug "Preparing initial best fit parameter vectors with $(n_params_cont+2) and $(n_params_lines) parameters"
        p_init_cont = zeros(n_params_cont+2)
        p_init_line = zeros(n_params_lines)

        # If a fit has been run previously, read in the file containing the rolling best fit parameters
        # to pick up where the fitter left off seamlessly
        if isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv")) && isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"))
            p_init_cont = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv"), ',', Float64, '\n')[:, 1]
            p_init_line = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"), ',', Float64, '\n')[:, 1]
        end

        # Get WCS transformation as a python object
        cube_wcs = py_wcs.WCS(naxis=2)
        cube_wcs.wcs.cdelt = cube.wcs.cdelt[1:2]
        cube_wcs.wcs.ctype = cube.wcs.ctype[1:2]
        cube_wcs.wcs.crpix = cube.wcs.crpix[1:2]
        cube_wcs.wcs.crval = cube.wcs.crval[1:2]
        cube_wcs.wcs.cunit = cube.wcs.cunit[1:2]
        cube_wcs.wcs.pc = cube.wcs.pc[1:2, 1:2]

        new{typeof(z), typeof(n_procs)}(cube, z, τ_guess, name, n_procs, plot_spaxels, plot_maps, plot_range, 
            parallel, save_fits, save_full_model, overwrite, track_memory, track_convergence, make_movies, extinction_curve, extinction_screen, 
            fit_sil_emission, T_s, T_dc, τ_97, τ_ice, τ_ch, β, T_hot, Cf_hot, τ_warm, τ_cold, n_dust_cont, n_dust_features, df_names, 
            dust_features, n_lines, n_acomps, line_names, line_profiles, line_acomp_profiles, lines, n_kin_tied, line_tied, kin_tied_key, 
            voff_tied, fwhm_tied, n_acomp_kin_tied, line_acomp_tied, acomp_kin_tied_key, acomp_voff_tied, acomp_fwhm_tied, tie_voigt_mixing, 
            voigt_mix_tied, n_params_cont, n_params_lines, n_params_extra, cosmo, interp_R, flexible_wavesol, p_init_cont, p_init_line, 
            cube_wcs)
    end

end


"""
    generate_cubemodel(cube_fitter)

Generate a CubeModel object corresponding to the options given by the CubeFitter object
"""
function generate_cubemodel(cube_fitter::CubeFitter)::CubeModel
    shape = size(cube_fitter.cube.Iν)
    # Full 3D intensity model array
    @debug "Generating full 3D cube models"
    cubemodel_empty(shape, cube_fitter.n_dust_cont, cube_fitter.df_names, cube_fitter.line_names)
end


"""
    generate_parammaps(cube_fitter)

Generate two ParamMaps objects (for the values and errors) corrresponding to the options given
by the CubeFitter object
"""
function generate_parammaps(cube_fitter::CubeFitter)::Tuple{ParamMaps, ParamMaps}
    shape = size(cube_fitter.cube.Iν)
    # 2D maps of fitting parameters
    @debug "Generating 2D parameter value & error maps"
    param_maps = parammaps_empty(shape, cube_fitter.n_dust_cont, cube_fitter.df_names, cube_fitter.line_names, cube_fitter.line_tied,
                                 cube_fitter.line_profiles, cube_fitter.line_acomp_tied, cube_fitter.line_acomp_profiles,
                                 cube_fitter.kin_tied_key, cube_fitter.acomp_kin_tied_key, cube_fitter.flexible_wavesol,
                                 cube_fitter.tie_voigt_mixing)
    # 2D maps of fitting parameter 1-sigma errors
    param_errs = parammaps_empty(shape, cube_fitter.n_dust_cont, cube_fitter.df_names, cube_fitter.line_names, cube_fitter.line_tied,
                                 cube_fitter.line_profiles, cube_fitter.line_acomp_tied, cube_fitter.line_acomp_profiles,
                                 cube_fitter.kin_tied_key, cube_fitter.acomp_kin_tied_key, cube_fitter.flexible_wavesol,
                                 cube_fitter.tie_voigt_mixing)
    param_maps, param_errs
end


############################## FITTING FUNCTIONS AND HELPERS ####################################


"""
    mask_emission_lines(λ, I)

Mask out emission lines in a given spectrum using a numerical second derivative and flagging 
negative spikes (indicating strong concave-downness) up to some tolerance threshold (i.e. 3-sigma)

# Arguments
- `λ::Vector{<:Real}`: The wavelength vector of the spectrum
- `I::Vector{<:Real}`: The flux vector of the spectrum
- `Δ::Integer=3`: The half-width of the numerical second derivative approximation, in pixels
- `W::Real=0.5`: The half-width of the window with which to calculate the standard deviation of the derivative of the spectrum
    in comparison to the point in question
- `thresh::Real=3`: The threshold by which to count a spike as a line that should be masked, in units
    of sigma.
- `n_iter::Integer=2`: How many iterations to perform the derivative test after masking the previous results

See also [`continuum_cubic_spline`](@ref)
"""
function mask_emission_lines(λ::Vector{<:Real}, I::Vector{<:Real}; Δ::Integer=3, W::Real=0.5, 
    thresh::Real=3., n_iter::Integer=1)

    # Numerical derivative width in microns
    h = Δ * median(diff(λ))

    # Calculate numerical second derivative
    d2f = zeros(length(λ))
    @inbounds @simd for i ∈ 1:length(λ)
        d2f[i] = (I[min(length(λ), i+Δ)] - 2I[i] + I[max(1, i-Δ)]) / h^2
    end
    d2f_i = copy(d2f)
    lines = Vector{Int64}()
    mask = falses(length(λ))

    function iter_mask(d2f, lines, mask)
        # Sigma-clip to find the lines based on the *local* noise level
        li = falses(length(λ))
        @inbounds @simd for j ∈ 1:length(λ)
            # Only consider the spectrum within +/- W microns from the point in question
            wi = Int(fld(W, diff(λ)[min(length(λ)-1, j)]))
            li[j] = d2f[j] < -thresh * nanstd(d2f[max(1, j-wi):min(length(λ), j+wi)])
        end
        lines = collect(1:length(λ))[li]
        # Mask out everything within the resolution of the second derivative calculations
        @inbounds for line ∈ lines
            mask[max(1,line-2Δ):min(length(λ),line+2Δ)] .= 1
        end
        lines, mask
    end

    # Use an iterative approach to repeat the process on the masked spectrum
    # -> this can help if, for example, you have two significant lines very close to each other,
    #    but one of them is much brighter and dominates the local stdev, which causes the fainter
    #    line to be missed by the first iteration
    # -> this is OFF by default (1 iteration)
    for k ∈ 1:n_iter
        lines, mask = iter_mask(d2f_i, lines, mask)
        d2f_i[mask] .= 0
    end

    # Don't mask out this region that tends to trick this method sometimes
    mask[11.175 .< λ .< 11.355] .= 0

    # Return the line locations and the mask
    lines, mask

end


"""
    continuum_cubic_spline(λ, I, σ)

Mask out the emission lines in a given spectrum using `mask_emission_lines` and replace them with
a coarse cubic spline fit to the continuum, using wide knots to avoid reinterpolating the lines or
noise.

# Arguments
- `λ::Vector{<:Real}`: The wavelength vector of the spectrum
- `I::Vector{<:Real}`: The flux vector of the spectrum
- `σ::Vector{<:Real}`: The uncertainty vector of the spectrum 

See also [`mask_emission_lines`](@ref)
"""
function continuum_cubic_spline(λ::Vector{<:Real}, I::Vector{<:Real}, σ::Vector{<:Real})

    # Copy arrays
    I_out = copy(I)
    σ_out = copy(σ)

    # Mask out emission lines so that they aren't included in the continuum fit
    lines, mask_lines = mask_emission_lines(λ, I)
    I_out[mask_lines] .= NaN
    σ_out[mask_lines] .= NaN 

    # Interpolate the NaNs
    diffs = diff(λ)
    Δλ = mean(diffs)
    # Break up cubic spline interpolation into knots 0.05 um long
    # (longer than a narrow emission line but not too long)
    scale = 0.05
    offset = findfirst(λ .> (scale + λ[1]))

    # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
    λknots = λ[offset+1]:scale:λ[end-offset-1]
    @debug "Performing cubic spline continuum fit with knots at $λknots"

    # Do a full cubic spline remapping of the data
    I_out = Spline1D(λ[isfinite.(I_out)], I_out[isfinite.(I_out)], λknots, k=3, bc="extrapolate").(λ)
    σ_out = Spline1D(λ[isfinite.(σ_out)], σ_out[isfinite.(σ_out)], λknots, k=3, bc="extrapolate").(λ)

    lines, mask_lines, I_out, σ_out
end


"""
    _interp_func(x, λ, I)

Function to interpolate the data with least squares quadratic fitting

# Arguments
- `x::Real`: The wavelength position to interpolate the intensity at
- `λ::Vector{<:Real}`: The wavelength vector
- `I::Vector{<:Real}`: The intensity vector
"""
function _interp_func(x::Real, λ::Vector{<:Real}, I::Vector{<:Real})::Real
    # Get the index of the value in the wavelength vector closest to x
    ind = findmin(abs.(λ .- x))[2]
    # Get the indices one before / after
    lo = ind - 1
    hi = ind + 2
    # Edge case for the left edge
    while lo ≤ 0
        lo += 1
        hi += 1
    end
    # Edge case for the right edge
    while hi > length(λ)
        hi -= 1
        lo -= 1
    end
    # A matrix with λ^2, λ, 1
    A = [λ[lo:hi].^2 λ[lo:hi] ones(4)]
    # y vector
    y = I[lo:hi]
    # Solve the least squares problem
    param = A \ y

    param[1]*x^2 + param[2]*x + param[3]
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
function continuum_fit_spaxel(cube_fitter::CubeFitter, spaxel::CartesianIndex, mask_lines::BitVector,
    I_spline::Vector{<:Real}, σ_spline::Vector{<:Real}; init::Bool=false) 

    @debug """\n
    #########################################################
    ###   Beginning continuum fit for spaxel $spaxel...   ###
    #########################################################
    """

    # Extract spaxel to be fit
    λ = cube_fitter.cube.λ
    #= 
    Get the data to fit -- either one spaxel, or if "init" is set, the sum of all spaxels.
    The spaxels are in surface brightness units (flux per steradian), so to preserve the units when summing them up, 
    we must divide by the number of spaxels in the sum (this way preserves the proper conversion into flux units, i.e. you would
    multiply the result by N_SPAXELS x STERADIANS_PER_SPAXEL to get the total flux within all the spaxels included in the sum) 
    =#
    I = !init ? cube_fitter.cube.Iν[spaxel, :] : Util.Σ(cube_fitter.cube.Iν, (1,2)) ./ Util.Σ(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    σ = !init ? cube_fitter.cube.σI[spaxel, :] : sqrt.(Util.Σ(cube_fitter.cube.σI.^2, (1,2))) ./ Util.Σ(Array{Int}(.~cube_fitter.cube.mask), (1,2))

    # Fill in the data where the lines are with the cubic spline interpolation
    I[mask_lines] .= I_spline[mask_lines]
    σ[mask_lines] .= σ_spline[mask_lines]
    # Add statistical uncertainties to the systematic uncertainties in quadrature
    σ_stat = std(I[.!mask_lines] .- I_spline[.!mask_lines])
    σ .= .√(σ.^2 .+ σ_stat.^2)

    @debug "Adding statistical error of $σ_stat in quadrature"
    
    # Mean and FWHM parameters for PAH profiles
    mean_df = [cdf[:wave] for cdf ∈ cube_fitter.dust_features]
    fwhm_df = [cdf[:fwhm] for cdf ∈ cube_fitter.dust_features]

    amp_dc_prior = Uniform(0., Inf)  # dont actually use this for getting pdfs or logpdfs, it's just for min/max
    amp_df_prior = Uniform(0., clamp(nanmaximum(I) / exp(-maximum(cube_fitter.τ_97.prior)), 1., Inf))

    stellar_priors = [amp_dc_prior, cube_fitter.T_s.prior]
    stellar_lock = [false, cube_fitter.T_s.locked]
    dc_priors = vcat([[amp_dc_prior, Ti.prior] for Ti ∈ cube_fitter.T_dc]...)
    dc_lock = vcat([[false, Ti.locked] for Ti ∈ cube_fitter.T_dc]...)
    df_priors = vcat([[amp_df_prior, mi.prior, fi.prior] for (mi, fi) ∈ zip(mean_df, fwhm_df)]...)
    df_lock = vcat([[false, mi.locked, fi.locked] for (mi, fi) ∈ zip(mean_df, fwhm_df)]...)
    ext_priors = [cube_fitter.τ_97.prior, cube_fitter.τ_ice.prior, cube_fitter.τ_ch.prior, cube_fitter.β.prior]
    ext_lock = [cube_fitter.τ_97.locked, cube_fitter.τ_ice.locked, cube_fitter.τ_ch.locked, cube_fitter.β.locked]
    hd_priors = cube_fitter.fit_sil_emission ? [amp_dc_prior, cube_fitter.T_hot.prior, cube_fitter.Cf_hot.prior, 
        cube_fitter.τ_warm.prior, cube_fitter.τ_cold.prior] : []
    hd_lock = cube_fitter.fit_sil_emission ? [false, cube_fitter.T_hot.locked, cube_fitter.Cf_hot.locked,
        cube_fitter.τ_warm.locked, cube_fitter.τ_cold.locked] : []

    priors_1 = vcat(stellar_priors, dc_priors, ext_priors, hd_priors, [amp_df_prior, amp_df_prior])
    lock_1 = vcat(stellar_lock, dc_lock, ext_lock, hd_lock, [false, false])
    priors_2 = df_priors
    lock_2 = df_lock

    # Check if the cube fitter has initial fit parameters 
    if !init

        @debug "Using initial best fit continuum parameters..."

        # Set the parameters to the best parameters
        p₀ = copy(cube_fitter.p_init_cont)[1:end-2]
        pah_frac = copy(cube_fitter.p_init_cont)[end-1:end]

        # pull out optical depth that was pre-fit
        τ_97_0 = cube_fitter.τ_guess[parse(Int, cube_fitter.cube.channel)][spaxel]

        # scale all flux amplitudes by the difference in medians between the spaxel and the summed spaxels
        # (should be close to 1 since the sum is already normalized by the number of spaxels included anyways)
        scale = max(nanmedian(I), 1e-10) / nanmedian(Util.Σ(cube_fitter.cube.Iν, (1,2)) ./ Util.Σ(Array{Int}(.~cube_fitter.cube.mask), (1,2)))
        max_amp = nanmaximum(I)

        # PAH template strengths
        pah_frac .*= scale
        
        # Stellar amplitude
        p₀[1] *= scale
        pᵢ = 3

        # Dust continuum amplitudes
        for i ∈ 1:cube_fitter.n_dust_cont
            p₀[pᵢ] *= scale
            pᵢ += 2
        end

        # Set optical depth based on the pre-fitting
        p₀[pᵢ] = τ_97_0
        pᵢ += 4

        if cube_fitter.fit_sil_emission
            # Hot dust amplitude
            p₀[pᵢ] *= scale
            # Warm / cold optical depths
            p₀[pᵢ+3] = τ_97_0
            p₀[pᵢ+4] = τ_97_0
            pᵢ += 5
        end

        # Dust feature amplitudes
        for i ∈ 1:cube_fitter.n_dust_feat
            # dont take the best fit values, just start them all equal otherwise weird stuff happens
            # p₀[pᵢ] = clamp(nanmedian(I)/2, 0., Inf) 
            p₀[pᵢ] = clamp(p₀[pᵢ]*scale, 0., max_amp / exp(-τ_97_0))
            pᵢ += 3
        end

    # Otherwise, we estimate the initial parameters based on the data
    else

        @debug "Calculating initial starting points..."
        pah_frac = repeat([clamp(nanmedian(I)/2, 0., Inf)], 2)

        # Stellar amplitude
        A_s = clamp(_interp_func(5.5, λ, I) / Util.Blackbody_ν(5.5, cube_fitter.T_s.value), 0., Inf) 

        # Dust feature amplitudes
        A_df = repeat([clamp(nanmedian(I)/2, 0., Inf)], cube_fitter.n_dust_feat)

        # Dust continuum amplitudes
        λ_dc = clamp.([Util.Wein(Ti.value) for Ti ∈ cube_fitter.T_dc], minimum(λ), maximum(λ))
        A_dc = clamp.([_interp_func(λ_dci, λ, I) / Util.Blackbody_ν(λ_dci, T_dci.value) for (λ_dci, T_dci) ∈ zip(λ_dc, cube_fitter.T_dc)] .* 
            (λ_dc ./ 9.7).^2 ./ 5., 0., Inf)
        
        # Hot dust amplitude
        A_hd = clamp(_interp_func(5.5, λ, I) / Util.Blackbody_ν(5.5, cube_fitter.T_hot.value), 0., Inf) / 2

        stellar_pars = [A_s, cube_fitter.T_s.value]
        dc_pars = vcat([[Ai, Ti.value] for (Ai, Ti) ∈ zip(A_dc, cube_fitter.T_dc)]...)
        df_pars = vcat([[Ai, mi.value, fi.value] for (Ai, mi, fi) ∈ zip(A_df, mean_df, fwhm_df)]...)
        if cube_fitter.fit_sil_emission
            hd_pars = [A_hd, cube_fitter.T_hot.value, cube_fitter.Cf_hot.value, cube_fitter.τ_warm.value, cube_fitter.τ_cold.value]
        else
            hd_pars = []
        end

        # Initial parameter vector
        p₀ = Vector{Float64}(vcat(stellar_pars, dc_pars, [cube_fitter.τ_97.value, cube_fitter.τ_ice.value, cube_fitter.τ_ch.value, 
            cube_fitter.β.value], hd_pars, df_pars))
        # p₀ = Vector{Float64}(vcat(stellar_pars, dc_pars, df_pars, [cube_fitter.τ_97[0][parse(Int, cube_fitter.cube.channel)], 
            # cube_fitter.τ_ice.value, cube_fitter.τ_ch.value, cube_fitter.β.value]))

    end

    @debug "Continuum Parameter labels: \n [stellar_amp, stellar_temp, " * 
        join(["dust_continuum_amp_$i, dust_continuum_temp_$i" for i ∈ 1:cube_fitter.n_dust_cont], ", ") * 
        "extinction_tau_97, extinction_tau_ice, extinction_tau_ch, extinction_beta, " *  
        (cube_fitter.fit_sil_emission ? "hot_dust_amp, hot_dust_temp, hot_dust_covering_frac, hot_dust_tau, cold_dust_tau, " : "") *
        join(["$(df)_amp, $(df)_mean, $(df)_fwhm" for df ∈ cube_fitter.df_names], ", ") * "]"
        
    # @debug "Priors: \n $priors"
    @debug "Continuum Starting Values: \n $p₀"

    # Split up the parameter vector into the components that we need for each fitting step

    # Step 1: Stellar + Dust blackbodies, 2 new amplitudes for the PAH templates, and the extinction parameters
    pars_1 = vcat(p₀[1:(2+2*cube_fitter.n_dust_cont+4+(cube_fitter.fit_sil_emission ? 5 : 0))], pah_frac)
    # Step 2: The PAH profile amplitudes, centers, and FWHMs
    pars_2 = p₀[(3+2*cube_fitter.n_dust_cont+4+(cube_fitter.fit_sil_emission ? 5 : 0)):end]

    # Convert parameter limits into CMPFit object
    parinfo_1 = CMPFit.Parinfo(length(pars_1))
    parinfo_2 = CMPFit.Parinfo(length(pars_2))

    for pᵢ ∈ 1:length(pars_1)
        parinfo_1[pᵢ].fixed = lock_1[pᵢ]
        if iszero(parinfo_1[pᵢ].fixed)
            parinfo_1[pᵢ].limited = (1,1)
            parinfo_1[pᵢ].limits = (minimum(priors_1[pᵢ]), maximum(priors_1[pᵢ]))
        end
    end

    for pᵢ ∈ 1:length(pars_2)
        parinfo_2[pᵢ].fixed = lock_2[pᵢ]
        if iszero(parinfo_2[pᵢ].fixed)
            parinfo_2[pᵢ].limited = (1,1)
            parinfo_2[pᵢ].limits = (minimum(priors_2[pᵢ]), maximum(priors_2[pᵢ]))
        end
    end

    # Create a `config` structure
    config = CMPFit.Config()

    @debug """\n
    ##########################################################################################################
    ########################## STEP 1 - FIT THE BLACKBODY CONTINUUM WITH PAH TEMPLATES #######################
    ##########################################################################################################
    """
    @debug "Continuum Step 1 Parameters: \n $(pars_1)"
    @debug "Continuum Parameters locked? \n $([parinfo_1[i].fixed for i ∈ eachindex(pars_1)])"
    @debug "Continuum Lower limits: \n $([parinfo_1[i].limits[1] for i ∈ eachindex(pars_1)])"
    @debug "Continuum Upper limits: \n $([parinfo_1[i].limits[2] for i ∈ eachindex(pars_1)])"

    @debug "Beginning continuum fitting with Levenberg-Marquardt least squares (CMPFit):"

    res_1 = cmpfit(λ, I, σ, (x, p) -> Util.fit_spectrum(x, p, cube_fitter.n_dust_cont,
        cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.fit_sil_emission), 
        pars_1, parinfo=parinfo_1, config=config)

    @debug "Continuum CMPFit status: $(res_1.status)"

    # Create continuum without the PAH features
    _, ccomps = Util.fit_spectrum(λ, res_1.param, cube_fitter.n_dust_cont, cube_fitter.extinction_curve, 
    cube_fitter.extinction_screen, cube_fitter.fit_sil_emission, true)

    I_cont = ccomps["stellar"]
    for i ∈ 1:cube_fitter.n_dust_cont
        I_cont .+= ccomps["dust_cont_$i"]
    end
    I_cont .*= ccomps["extinction"] .* ccomps["abs_ice"] .* ccomps["abs_ch"]
    if cube_fitter.fit_sil_emission
        I_cont .+= ccomps["hot_dust"]
    end

    # Count free parameters
    n_free_1 = 0
    for p₁ ∈ eachindex(pars_1)
        n_free_1 += iszero(parinfo_1[p₁].fixed)
    end

    @debug """\n
    ##########################################################################################################
    ################################# STEP 2 - FIT THE PAHs AS DRUDE PROFILES ################################
    ##########################################################################################################
    """
    @debug "Continuum Step 2 Parameters: \n $(pars_2)"
    @debug "Continuum Parameters locked? \n $([parinfo_2[i].fixed for i ∈ eachindex(pars_2)])"
    @debug "Continuum Lower limits: \n $([parinfo_2[i].limits[1] for i ∈ eachindex(pars_2)])"
    @debug "Continuum Upper limits: \n $([parinfo_2[i].limits[2] for i ∈ eachindex(pars_2)])"

    @debug "Beginning continuum fitting with Levenberg-Marquardt least squares (CMPFit):"

    res_2 = cmpfit(λ, I.-I_cont, σ, (x, p) -> Util.fit_pah_residuals(x, p, cube_fitter.n_dust_feat,
        ccomps["extinction"]), pars_2, parinfo=parinfo_2, config=config)

    @debug "Continuum CMPFit status: $(res_2.status)"

    # Count free parameters
    n_free_2 = 0
    for p₂ ∈ eachindex(pars_2)
        n_free_2 += iszero(parinfo_2[p₂].fixed)
    end

    # Get combined best fit results
    popt = vcat(res_1.param[1:end-2], res_2.param)        # Combined Best fit parameters
    perr = vcat(res_1.perror[1:end-2], res_2.perror)      # Combined 1-sigma uncertainties
    covar = (res_1.covar[1:end-2, 1:end-2], res_2.covar)  # Combined Covariance matrix
    n_free = n_free_1 - 2 + n_free_2

    @debug "Best fit continuum parameters: \n $popt"
    @debug "Continuum parameter errors: \n $perr"
    @debug "Continuum covariance matrix: \n $covar"

    # Create the full model
    I_model, comps = Util.fit_full_continuum(λ, popt, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat,
        cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.fit_sil_emission)

    if init
        cube_fitter.p_init_cont[:] .= vcat(popt, res_1.param[end-1:end])
        # Save the results to a file 
        # save running best fit parameters in case the fitting is interrupted
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_cont.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_cont, ',')
        end
    end

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
    msg *= "\n#> EXTINCTION <#\n"
    msg *= "τ_9.7: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t Limits: " *
        "($(@sprintf "%.2f" minimum(cube_fitter.τ_97.prior)), $(@sprintf "%.2f" maximum(cube_fitter.τ_97.prior)))" * 
        (cube_fitter.τ_97.locked ? " (fixed)" : "") * "\n"
    msg *= "τ_ice: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+1]) +/- $(@sprintf "%.2f" perr[pᵢ+1]) [-] \t Limits: " *
        "($(@sprintf "%.2f" minimum(cube_fitter.τ_ice.prior)), $(@sprintf "%.2f" maximum(cube_fitter.τ_ice.prior)))" *
        (cube_fitter.τ_ice.locked ? " (fixed)" : "") * "\n"
    msg *= "τ_ch: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+2]) +/- $(@sprintf "%.2f" perr[pᵢ+2]) [-] \t Limits: " *
        "($(@sprintf "%.2f" minimum(cube_fitter.τ_ch.prior)), $(@sprintf "%.2f" maximum(cube_fitter.τ_ch.prior)))" *
        (cube_fitter.τ_ch.locked ? " (fixed)" : "") * "\n"
    msg *= "β: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+3]) +/- $(@sprintf "%.2f" perr[pᵢ+3]) [-] \t Limits: " *
        "($(@sprintf "%.2f" minimum(cube_fitter.β.prior)), $(@sprintf "%.2f" maximum(cube_fitter.β.prior)))" * 
        (cube_fitter.β.locked ? " (fixed)" : "") * "\n"
    msg *= "\n"
    pᵢ += 4
    if cube_fitter.fit_sil_emission
        msg *= "\n#> HOT DUST <#\n"
        msg *= "Hot_dust_amp: \t\t\t $(@sprintf "%.3e" popt[pᵢ]) +/- $(@sprintf "%.3e" perr[pᵢ]) MJy/sr \t Limits: (0, Inf)\n"
        msg *= "Hot_dust_temp: \t\t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) K \t Limits: " *
            "($(@sprintf "%.0f" minimum(cube_fitter.T_hot.prior)), $(@sprintf "%.0f" maximum(cube_fitter.T_hot.prior)))" *
            (cube_fitter.T_hot.locked ? " (fixed)" : "") * "\n"
        msg *= "Hot_dust_frac: \t\t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) [-] \t Limits: " *
            "($(@sprintf "%.3f" minimum(cube_fitter.Cf_hot.prior)), $(@sprintf "%.3f" maximum(cube_fitter.Cf_hot.prior)))" *
            (cube_fitter.Cf_hot.locked ? " (fixed)" : "") * "\n"
        msg *= "Hot_dust_τ: \t\t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3]) [-] \t Limits: " *
            "($(@sprintf "%.3f" minimum(cube_fitter.τ_warm.prior)), $(@sprintf "%.3f" maximum(cube_fitter.τ_warm.prior)))" *
            (cube_fitter.τ_warm.locked ? " (fixed)" : "") * "\n"
        msg *= "Cold_dust_τ: \t\t\t $(@sprintf "%.3f" popt[pᵢ+4]) +/- $(@sprintf "%.3f" perr[pᵢ+4]) [-] \t Limits: " *
            "($(@sprintf "%.3f" minimum(cube_fitter.τ_cold.prior)), $(@sprintf "%.3f" maximum(cube_fitter.τ_cold.prior)))" *
            (cube_fitter.τ_cold.locked ? " (fixed)" : "") * "\n"
        pᵢ += 5
    end
    msg *= "\n#> DUST FEATURES <#\n"
    for (j, df) ∈ enumerate(cube_fitter.df_names)
        msg *= "$(df)_amp:\t\t\t $(@sprintf "%.1f" popt[pᵢ]) +/- $(@sprintf "%.1f" perr[pᵢ]) MJy/sr \t Limits: " *
            "(0, $(@sprintf "%.1f" (nanmaximum(I) / exp(-popt[end-1]))))\n"
        msg *= "$(df)_mean:  \t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1]) μm \t Limits: " *
            "($(@sprintf "%.3f" minimum(mean_df[j].prior)), $(@sprintf "%.3f" maximum(mean_df[j].prior)))" * 
            (mean_df[j].locked ? " (fixed)" : "") * "\n"
        msg *= "$(df)_fwhm:  \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) μm \t Limits: " *
            "($(@sprintf "%.3f" minimum(fwhm_df[j].prior)), $(@sprintf "%.3f" maximum(fwhm_df[j].prior)))" * 
            (fwhm_df[j].locked ? " (fixed)" : "") * "\n"
        msg *= "\n"
        pᵢ += 3
    end
    msg *= "######################################################################"
    @debug msg

    σ, popt, I_model, comps, n_free, perr, covar
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
    sum([logpdf(priors[i], p[i]) for i ∈ eachindex(p)])
end


"""
    _negln_probability(p, grad, λ, Inorm, σnorm, cube_fitter, λ0_ln, ext_curve, priors)

Internal helper function to calculate the negative of the log of the probability,
for the line residual fitting.

ln(probability) = ln(likelihood) + ln(prior)
"""
function _negln_probability(p, grad, λ, Inorm, σnorm, cube_fitter, λ0_ln, ext_curve, priors)
    # Check for gradient
    if length(grad) > 0
        error("Gradient-based solvers are not currently supported!")
    end
    # First compute the model
    model = Util.fit_line_residuals(λ, p, cube_fitter.n_lines, cube_fitter.n_kin_tied, 
        cube_fitter.kin_tied_key, cube_fitter.line_tied, cube_fitter.line_profiles, cube_fitter.n_acomp_kin_tied,
        cube_fitter.acomp_kin_tied_key, cube_fitter.line_acomp_tied, cube_fitter.line_acomp_profiles, 
        λ0_ln, cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing, ext_curve)
    # Add the log of the likelihood (based on the model) and the prior distribution functions
    lnP = Util.ln_likelihood(Inorm, model, σnorm) + _ln_prior(p, priors)
    # return the NEGATIVE log of the probability
    -lnP 
end


"""
    line_fit_spaxel(cube_fitter, spaxel, continuum, ext_curve, init=init)

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
- `continuum::Vector{<:Real}`: The fitted continuum level of the spaxel being fit (which will be subtracted
    before the lines are fit)
- `ext_curve::Vector{<:Real}`: The extinction curve of the spaxel being fit (which will be used to calculate
    extinction-corrected line amplitudes and fluxes)
- `init::Bool=false`: Flag for the initial fit which fits the sum of all spaxels, to get an estimation for
    the initial parameter vector for individual spaxel fits
"""
function line_fit_spaxel(cube_fitter::CubeFitter, spaxel::CartesianIndex, continuum::Vector{<:Real}, 
    ext_curve::Vector{<:Real}, line_locs::Vector{<:Integer}, mask_lines::BitVector, I_spline::Vector{<:Real}, σ_spline::Vector{<:Real}; 
    init::Bool=false)

    @debug """\n
    #########################################################
    ###      Beginning line fit for spaxel $spaxel...     ###
    #########################################################
    """

    # Extract spaxel to be fit
    λ = cube_fitter.cube.λ
    I = !init ? cube_fitter.cube.Iν[spaxel, :] : Util.Σ(cube_fitter.cube.Iν, (1,2)) ./ Util.Σ(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    σ = !init ? cube_fitter.cube.σI[spaxel, :] : sqrt.(Util.Σ(cube_fitter.cube.σI.^2, (1,2))) ./ Util.Σ(Array{Int}(.~cube_fitter.cube.mask), (1,2))

    # Perform a cubic spline continuum fit
    # line_locs, mask_lines, spline_continuum, _ = continuum_cubic_spline(λ, I, σ)
    N = Float64(abs(nanmaximum(I)))
    N = N ≠ 0. ? N : 1.

    @debug "Using normalization N=$N"

    # Add statistical uncertainties to the systematic uncertainties in quadrature
    σ_stat = std(I[.!mask_lines] .- I_spline[.!mask_lines])
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

    # Repeat for the additional component line parameters
    acomp_voff_ln = [isnothing(ln.acomp_profile) ? nothing : ln.parameters[:acomp_voff] for ln ∈ cube_fitter.lines]
    acomp_fwhm_ln = [isnothing(ln.acomp_profile) ? nothing : ln.parameters[:acomp_fwhm] for ln ∈ cube_fitter.lines]
    acomp_h3_ln = [ln.acomp_profile == :GaussHermite ? ln.parameters[:acomp_h3] : nothing for ln ∈ cube_fitter.lines]
    acomp_h4_ln = [ln.acomp_profile == :GaussHermite ? ln.parameters[:acomp_h4] : nothing for ln ∈ cube_fitter.lines]
    if !cube_fitter.tie_voigt_mixing
        acomp_η_ln = [ln.acomp_profile == :Voigt ? ln.parameters[:acomp_mixing] : nothing for ln ∈ cube_fitter.lines]
    else
        acomp_η_ln = η_ln
    end

    # Set up the prior vector
    amp_ln_prior = Uniform(0., 1.)
    amp_acomp_prior = Uniform(0., 1.)
    λ0_ln = Vector{Float64}()
    prof_ln = Vector{Symbol}()
    acomp_prof_ln = Vector{Union{Symbol,Nothing}}()
    ln_priors = Vector{Any}()
    ln_lock = Vector{Bool}()
    param_names = Vector{String}()
    # Loop through each line and append the new components
    for (i, ln) ∈ enumerate(cube_fitter.lines)
        # name
        ln_name = cube_fitter.line_names[i]
        # check if voff should be tied or untied
        if isnothing(ln.tied)
            # amplitude, voff, FWHM
            append!(ln_priors, [amp_ln_prior, voff_ln[i].prior, fwhm_ln[i].prior])
            append!(ln_lock, [false, voff_ln[i].locked, fwhm_ln[i].locked])
            append!(param_names, ["$(ln_name)_amp", "$(ln_name)_voff", "$(ln_name)_fwhm"])
        elseif cube_fitter.flexible_wavesol
            # amplitude, voff (since FWHM is tied)
            append!(ln_priors, [amp_ln_prior, voff_ln[i].prior])
            append!(ln_lock, [false, voff_ln[i].locked])
            append!(param_names, ["$(ln_name)_amp", "$(ln_name)_voff"])
        else
            # just amplitude (since voff & FWHM are tied)
            append!(ln_priors, [amp_ln_prior])
            append!(ln_lock, [false])
            append!(param_names, ["$(ln_name)_amp"])
        end
        # check for additional profile parameters
        if ln.profile == :GaussHermite
            # add h3 and h4 moments
            append!(ln_priors, [h3_ln[i].prior, h4_ln[i].prior])
            append!(ln_lock, [h3_ln[i].locked, h4_ln[i].locked])
            append!(param_names, ["$(ln_name)_h3", "$(ln_name)_h4"])
        elseif ln.profile == :Voigt
            # add voigt mixing parameter, but only if it's not tied
            if !cube_fitter.tie_voigt_mixing
                append!(ln_priors, [η_ln[i].prior])
                append!(ln_lock, [η_ln[i].locked])
                append!(param_names, ["$(ln_name)_eta"])
            end
        end
        # repeat the above for the acomp components
        if !isnothing(ln.acomp_profile)
            # check tied or untied (but no flexible wavesol)
            if isnothing(ln.acomp_tied)
                # amplitude, voff, FWHM
                append!(ln_priors, [amp_acomp_prior, acomp_voff_ln[i].prior, acomp_fwhm_ln[i].prior])
                append!(ln_lock, [false, acomp_voff_ln[i].locked, acomp_fwhm_ln[i].locked])
                append!(param_names, ["$(ln_name)_acomp_amp", "$(ln_name)_acomp_voff", "$(ln_name)_acomp_fwhm"])
            else
                # just amplitude (voff & FWHM tied)
                append!(ln_priors, [amp_acomp_prior])
                append!(ln_lock, [false])
                append!(param_names, ["$(ln_name)_acomp_amp"])
            end
            # check for additional profile parameters
            if ln.acomp_profile == :GaussHermite
                # h3 and h4 moments
                append!(ln_priors, [acomp_h3_ln[i].prior, acomp_h4_ln[i].prior])
                append!(ln_lock, [acomp_h3_ln[i].locked, acomp_h4_ln[i].locked])
                append!(param_names, ["$(ln_name)_acomp_h3", "$(ln_name)_acomp_h4"])
            elseif ln.acomp_profile == :Voigt
                # voigt mixing parameter, only if untied
                if !cube_fitter.tie_voigt_mixing
                    append!(ln_prior, [acomp_η_ln[i].prior])
                    append!(ln_lock, [acomp_η_ln[i].locked])
                    append!(param_names, ["$(ln_name)_acomp_eta"])
                end
            end
        end
        # add central wavelength, profile, and acomp profile
        append!(λ0_ln, [ln.λ₀])
        append!(prof_ln, [ln.profile])
        append!(acomp_prof_ln, [ln.acomp_profile])
    end
    # Set up the tied voff and fwhm parameters as vectors
    voff_tied_priors = [cube_fitter.voff_tied[i].prior for i ∈ 1:cube_fitter.n_kin_tied]
    voff_tied_lock = [cube_fitter.voff_tied[i].locked for i ∈ 1:cube_fitter.n_kin_tied]
    fwhm_tied_priors = [cube_fitter.fwhm_tied[i].prior for i ∈ 1:cube_fitter.n_kin_tied]
    fwhm_tied_lock = [cube_fitter.fwhm_tied[i].locked for i ∈ 1:cube_fitter.n_kin_tied]

    acomp_voff_tied_priors = [cube_fitter.acomp_voff_tied[i].prior for i ∈ 1:cube_fitter.n_acomp_kin_tied]
    acomp_voff_tied_lock = [cube_fitter.acomp_voff_tied[i].locked for i ∈ 1:cube_fitter.n_acomp_kin_tied]
    acomp_fwhm_tied_priors = [cube_fitter.acomp_fwhm_tied[i].prior for i ∈ 1:cube_fitter.n_acomp_kin_tied]
    acomp_fwhm_tied_lock = [cube_fitter.acomp_fwhm_tied[i].locked for i ∈ 1:cube_fitter.n_acomp_kin_tied]

    # Initial prior vector
    if !cube_fitter.tie_voigt_mixing
        # If the voigt mixing parameters are untied, place them sequentially in the ln_priors
        priors = vcat(voff_tied_priors, fwhm_tied_priors, acomp_voff_tied_priors, acomp_fwhm_tied_priors, ln_priors)
        param_lock = vcat(voff_tied_lock, fwhm_tied_lock, acomp_voff_tied_lock, acomp_fwhm_tied_lock, ln_lock)
        param_names = vcat(["voff_tied_$k" for k ∈ cube_fitter.kin_tied_key], 
                           ["fwhm_tied_$k" for k ∈ cube_fitter.kin_tied_key],
                           ["acomp_voff_tied_$k" for k ∈ cube_fitter.acomp_kin_tied_key], 
                           ["acomp_fwhm_tied_$k" for k ∈ cube_fitter.acomp_kin_tied_key],
                           param_names)
    else
        # If the voigt mixing parameters are tied, just add the single mixing parameter before the rest of the line parameters
        ηᵢ = 2cube_fitter.n_kin_tied + 2cube_fitter.n_acomp_kin_tied + 1
        # If the sum has already been fit, keep eta fixed for the individual spaxels
        η_prior = init ? η_ln.prior : Uniform(cube_fitter.p_init_line[ηᵢ]-1e-10, cube_fitter.p_init_line[ηᵢ]+1e-10)
        priors = vcat(voff_tied_priors, fwhm_tied_priors, acomp_voff_tied_priors, acomp_fwhm_tied_priors, η_prior, ln_priors)
        param_lock = vcat(voff_tied_lock, fwhm_tied_lock, acomp_voff_tied_lock, acomp_fwhm_tied_lock, 
            [cube_fitter.voigt_mix_tied.locked || !init], ln_lock)
        param_names = vcat(["voff_tied_$k" for k ∈ cube_fitter.kin_tied_key], 
                           ["fwhm_tied_$k" for k ∈ cube_fitter.kin_tied_key],
                           ["acomp_voff_tied_$k" for k ∈ cube_fitter.acomp_kin_tied_key], 
                           ["acomp_fwhm_tied_$k" for k ∈ cube_fitter.acomp_kin_tied_key],
                           ["eta_tied"], 
                           param_names)
    end

    # Check if there are previous best fit parameters
    if !init

        @debug "Using initial best fit line parameters..."

        # If so, set the parameters to the previous ones
        p₀ = copy(cube_fitter.p_init_line)

    else

        @debug "Calculating initial starting points..."

        # Intelligently pick starting positions for voffs based on the locations of lines found with the line masking
        voff_inits = zeros(cube_fitter.n_lines)
        for (i, ln) ∈ enumerate(cube_fitter.lines)
            # Constrain to the line fitting region (+/- maximum voff)
            if length(voff_tied_priors) > 0
                max_kms = maximum(voff_tied_priors[1])
            else
                max_kms = maximum(voff_ln[i].prior)
            end
            minw = ln.λ₀ * (1 - max_kms / Util.C_KMS)
            maxw = ln.λ₀ * (1 + max_kms / Util.C_KMS)
            # Match with any lines flagged by the line masking algorithm
            candidates = findall(minw .< λ[line_locs] .< maxw)
            # For each candidate, make sure it's not closer to another line's nominal position
            good = trues(length(candidates))
            for j ∈ 1:length(candidates)
                if !(λ0_ln[argmin(abs.(λ[line_locs][candidates[j]] .- λ0_ln))] ≈ ln.λ₀)
                    good[j] = false
                end
            end
            candidates = candidates[good]
            # Check if we're out of candidates
            if length(candidates) == 0
                voff_inits[i] = 0.
                continue
            end
            # Finally, of the remaining candidates (if there's still more than one), pick the brightest one
            ind = candidates[argmax(I[line_locs][candidates])]
            pos = λ[line_locs][ind]
            voff_inits[i] = (pos/ln.λ₀ - 1) * Util.C_KMS
        end
        @debug "Initial individual voff parameter vector: $voff_inits"
        voff_inits_tied = zeros(cube_fitter.n_kin_tied)
        # For tied velocities, take the averages
        for j ∈ 1:cube_fitter.n_kin_tied
            # consider the lines within the tied group
            tied_group = BitVector([ln.tied == cube_fitter.kin_tied_key[j] for ln ∈ cube_fitter.lines])
            voff_inits_tied[j] = mean(voff_inits[tied_group])
        end
        @debug "Initial tied voff parameter vector: $voff_inits_tied"
        # Double back and make sure all of the voff_inits are consistent with any tied lines
        for (i, ln) ∈ enumerate(cube_fitter.lines)
            if !isnothing(ln.tied)
                vwhere = findfirst(cube_fitter.kin_tied_key .== ln.tied)
                voff_group = voff_inits_tied[vwhere]
                voff_inits[i] -= voff_group
                if cube_fitter.flexible_wavesol
                    # Ensure the individual voff is consistent with the tied voff
                    if voff_inits[i] < minimum(voff_ln[i].prior) 
                        @debug "initial voff for line $i at $(ln.λ₀) is inconsistent with the group: $(voff_inits[i]), adjusting" 
                        voff_inits[i] = minimum(voff_ln[i].prior) + 1
                    elseif voff_inits[i] > maximum(voff_ln[i].prior)
                        @debug "initial voff for line $i at $(ln.λ₀) is inconsistent with the group: $(voff_inits[i]), adjusting" 
                        voff_inits[i] = maximum(voff_ln[i].prior) - 1
                    end
                end
            end
        end
        @debug "Individual voff parameter vector corrected for ties: $voff_inits"

        # Start the ampltiudes at 1/2 and 1/4 (in normalized units)
        A_ln = ones(cube_fitter.n_lines) .* 0.5
        A_fl = ones(cube_fitter.n_lines) .* 0.5     # (acomp amp is multiplied with main amp)

        # Initial parameter vector
        ln_pars = Vector{Float64}()
        for (i, ln) ∈ enumerate(cube_fitter.lines)
            if isnothing(ln.tied)
                # 3 parameters: amplitude, voff, FWHM
                append!(ln_pars, [A_ln[i], voff_inits[i], fwhm_ln[i].value])
            elseif cube_fitter.flexible_wavesol
                # 2 parameters: amplitude, voff (since FWHM is tied but voff is only restricted)
                append!(ln_pars, [A_ln[i], voff_inits[i]])
            else
                # 1 parameter: amplitude (since FWHM and voff are tied)
                append!(ln_pars, [A_ln[i]])
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
            # Repeat but for additional components, if present
            if !isnothing(ln.acomp_profile)
                if isnothing(ln.acomp_tied)
                    append!(ln_pars, [A_fl[i], acomp_voff_ln[i].value, acomp_fwhm_ln[i].value])
                else
                    append!(ln_pars, [A_fl[i]])
                end
                if ln.acomp_profile == :GaussHermite
                    append!(ln_pars, [acomp_h3_ln[i].value, acomp_h4_ln[i].value])
                elseif ln.acomp_profile == :Voigt
                    if !cube_fitter.tie_voigt_mixing
                        append!(ln_pars, [acomp_η_ln[i].value])
                    end
                end
            end
        end
        # Set up tied voff and fwhm parameter vectors
        # voff_tied_pars = [cube_fitter.voff_tied[i].value for i ∈ 1:cube_fitter.n_kin_tied]
        voff_tied_pars = voff_inits_tied
        fwhm_tied_pars = [cube_fitter.fwhm_tied[i].value for i ∈ 1:cube_fitter.n_kin_tied]

        acomp_voff_tied_pars = [cube_fitter.acomp_voff_tied[i].value for i ∈ 1:cube_fitter.n_acomp_kin_tied]
        acomp_fwhm_tied_pars = [cube_fitter.acomp_fwhm_tied[i].value for i ∈ 1:cube_fitter.n_acomp_kin_tied]

        # Set up the parameter vector in the proper order: 
        # (tied voffs, tied acomp voffs, tied voigt mixing, [amp, voff, FWHM, h3, h4, eta,
        #     acomp_amp, acomp_voff, acomp_FWHM, acomp_h3, acomp_h4, acomp_eta] for each line)
        if !cube_fitter.tie_voigt_mixing
            p₀ = Vector{Float64}(vcat(voff_tied_pars, fwhm_tied_pars, 
                acomp_voff_tied_pars, acomp_fwhm_tied_pars, ln_pars))
        else
            p₀ = Vector{Float64}(vcat(voff_tied_pars, fwhm_tied_pars, 
                acomp_voff_tied_pars, acomp_fwhm_tied_pars, η_ln.value, ln_pars))
        end

    end

    @debug "Line Parameter labels: \n $param_names"
    @debug "Line starting values: \n $p₀"
    @debug "Line priors: \n $priors"

    # Lower and upper limits on each parameter, to be fed into the Simulated Annealing (SAMIN) algorithm
    lower_bounds = minimum.(priors)
    upper_bounds = maximum.(priors)

    @debug "Line Lower limits: \n $lower_bounds"
    @debug "Line Upper Limits: \n $upper_bounds"

    if init
        @debug "Beginning Line fitting with Simulated Annealing:"

        # Parameter and function tolerance levels for convergence with SAMIN,
        # these are a bit loose since we're mainly just looking to get into the right global minimum region with SAMIN
        # before refining the fit later with a LevMar local minimum routine
        x_tol = 1e-5
        f_tol = abs(_negln_probability(p₀, [], λ, Inorm, σnorm, cube_fitter, λ0_ln, ext_curve, priors) -
                    _negln_probability(clamp.(p₀ .- x_tol, lower_bounds, upper_bounds), [], λ, Inorm, σnorm, cube_fitter, λ0_ln, ext_curve, priors))

        # First, perform a bounded Simulated Annealing search for the optimal parameters with a generous max iterations and temperature rate (rt)
        res = Optim.optimize(p -> _negln_probability(p, [], λ, Inorm, σnorm, cube_fitter, λ0_ln, ext_curve, priors), lower_bounds, upper_bounds, p₀, 
           SAMIN(;rt=0.9, nt=5, ns=5, neps=5, f_tol=f_tol, x_tol=x_tol, verbosity=0), Optim.Options(iterations=10^6))
        p₁ = res.minimizer

        # Write convergence results to file, if specified
        if cube_fitter.track_convergence
            global file_lock
            # use the ReentrantLock to prevent multiple processes from trying to write to the same file at once
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
    else
        p₁ = p₀
    end    
    
    @debug "Beginning Line fitting with Levenberg-Marquardt:"

    ############################################# FIT WITH LEVMAR ###################################################

    # Convert parameter limits into CMPFit object
    parinfo = CMPFit.Parinfo(length(p₀))
    for pᵢ ∈ 1:length(p₀)
        parinfo[pᵢ].fixed = param_lock[pᵢ]
        if iszero(parinfo[pᵢ].fixed)
            parinfo[pᵢ].limited = (1,1)
            parinfo[pᵢ].limits = (minimum(priors[pᵢ]), maximum(priors[pᵢ]))
        end
    end

    # Create a `config` structure
    config = CMPFit.Config()

    # # Same procedure as with the continuum fit
    res = cmpfit(λ, Inorm, σnorm, (x, p) -> Util.fit_line_residuals(x, p, cube_fitter.n_lines, cube_fitter.n_kin_tied, 
        cube_fitter.kin_tied_key, cube_fitter.line_tied, prof_ln, cube_fitter.n_acomp_kin_tied, cube_fitter.acomp_kin_tied_key,
        cube_fitter.line_acomp_tied, acomp_prof_ln, λ0_ln, cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing, ext_curve), p₁, 
        parinfo=parinfo, config=config)

    # Get the results
    popt = res.param
    perr = res.perror
    covar = res.covar

    # Count free parameters
    n_free = 0
    for pᵢ ∈ eachindex(popt)
        n_free += iszero(parinfo[pᵢ].fixed)
    end

    ######################################################################################################################

    @debug "Best fit line parameters: \n $popt"
    @debug "Line parameter errors: \n $perr"
    @debug "Line covariance matrix: \n $covar"

    # Final optimized fit
    I_model, comps = Util.fit_line_residuals(λ, popt, cube_fitter.n_lines, cube_fitter.n_kin_tied, 
        cube_fitter.kin_tied_key, cube_fitter.line_tied, prof_ln, cube_fitter.n_acomp_kin_tied,
        cube_fitter.acomp_kin_tied_key, cube_fitter.line_acomp_tied, acomp_prof_ln, λ0_ln, 
        cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing, ext_curve, true)
    
    # Renormalize
    I_model = I_model .* N
    for comp ∈ keys(comps)
        comps[comp] = comps[comp] .* N
    end

    if init
        cube_fitter.p_init_line[:] .= copy(popt)
        # Save results to file
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_line.csv"), "w") do f
            writedlm(f, cube_fitter.p_init_line, ',')
        end
    end

    msg = "######################################################################\n"
    msg *= "############### SPAXEL FIT RESULTS -- EMISSION LINES #################\n"
    msg *= "######################################################################\n"
    pᵢ = 1
    msg *= "\n#> TIED VELOCITY OFFSETS <#\n"
    for (i, vk) ∈ enumerate(cube_fitter.kin_tied_key)
        msg *= "$(vk)_tied_voff: \t\t\t $(@sprintf "%.0f" popt[pᵢ]) +/- $(@sprintf "%.0f" perr[pᵢ]) km/s \t " *
            "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.voff_tied[i].prior)), $(@sprintf "%.0f" maximum(cube_fitter.voff_tied[i].prior)))\n"
        pᵢ += 1
    end
    for (ii, vk) ∈ enumerate(cube_fitter.kin_tied_key)
        msg *= "$(vk)_tied_fwhm: \t\t\t $(@sprintf "%.0f" popt[pᵢ]) +/- $(@sprintf "%.0f" perr[pᵢ]) km/s \t " *
            "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.fwhm_tied[ii].prior)), $(@sprintf "%.0f" maximum(cube_fitter.fwhm_tied[ii].prior)))\n"
        pᵢ += 1
    end
    for (j, fvk) ∈ enumerate(cube_fitter.acomp_kin_tied_key)
        msg *= "$(fvk)_acomp_tied_voff:\t\t\t $(@sprintf "%.0f" popt[pᵢ]) +/- $(@sprintf "%.0f" perr[pᵢ]) km/s \t " *
            "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.acomp_voff_tied[j].prior)), $(@sprintf "%.0f" maximum(cube_fitter.acomp_voff_tied[j].prior)))\n"
        pᵢ += 1
    end
    for (jj, fvk) ∈ enumerate(cube_fitter.acomp_kin_tied_key)
        msg *= "$(fvk)_acomp_tied_fwhm:\t\t\t $(@sprintf "%.0f" popt[pᵢ]) +/- $(@sprintf "%.0f" perr[pᵢ]) kmk/s \t " *
            "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.acomp_fwhm_tied[jj].prior)), $(@sprintf "%.0f" maximum(cube_fitter.acomp_fwhm_tied[jj].prior)))\n"
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
        if isnothing(cube_fitter.line_tied[k])
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
        elseif cube_fitter.flexible_wavesol
            msg *= "$(nm)_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(voff_ln[k].prior)), $(@sprintf "%.0f" maximum(voff_ln[k].prior)))\n"
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
            pᵢ += 2
        else
            pᵢ += 1
        end
        if !isnothing(acomp_prof_ln[k])
            msg *= "\n$(nm)_acomp_amp:\t\t\t $(@sprintf "%.3f" popt[pᵢ]) +/- $(@sprintf "%.3f" perr[pᵢ]) [x amp] \t Limits: (0, 1)\n"
            if isnothing(cube_fitter.line_acomp_tied[k])
                msg *= "$(nm)_acomp_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) [+ voff] \t " *
                    "Limits: ($(@sprintf "%.0f" minimum(voff_ln[k].prior)), $(@sprintf "%.0f" maximum(voff_ln[k].prior)))\n"
                msg *= "$(nm)_acomp_fwhm:   \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) [x fwhm] \t " *
                    "Limits: ($(@sprintf "%.0f" minimum(fwhm_ln[k].prior)), $(@sprintf "%.0f" maximum(fwhm_ln[k].prior)))\n"
                if acomp_prof_ln[k] == :GaussHermite
                    msg *= "$(nm)_acomp_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h3_ln[k].prior)), $(@sprintf "%.3f" maximum(h3_ln[k].prior)))\n"
                    msg *= "$(nm)_acomp_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+4]) +/- $(@sprintf "%.3f" perr[pᵢ+4])      \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(h4_ln[k].prior)), $(@sprintf "%.3f" maximum(h4_ln[k].prior)))\n"
                    pᵢ += 2
                elseif acomp_prof_ln[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                    msg *= "$(nm)_acomp_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])       \t " *
                        "Limits: ($(@sprintf "%.3f" minimum(η_ln[k].prior)), $(@sprintf "%.3f" maximum(η_ln[k].prior)))\n"
                    pᵢ += 1
                end
                pᵢ += 3
            else
                pᵢ += 1
            end
        end
        msg *= "\n"
    end 
    msg *= "######################################################################" 
    @debug msg

    σ, popt, I_model, comps, n_free, perr, covar
end


"""
    plot_spaxel_fit(λ, I, I_cont, σ, comps, n_dust_cont, n_dust_features, line_wave, line_names, screen, z, χ2red, 
        name, label; backend=backend)

Plot the fit for an individual spaxel, `I_cont`, and its individual components `comps`, using the given 
backend (`:pyplot` or `:plotly`).

# Arguments
`T<:Real,S<:Integer`
- `λ::Vector{<:Real}`: The wavelength vector of the spaxel to be plotted
- `I::Vector{<:Real}`: The intensity data vector of the spaxel to be plotted
- `I_cont::Vector{<:Real}`: The intensity model vector of the spaxel to be plotted
- `σ::Vector{<:Real}`: The uncertainty vector of the spaxel to be plotted
- `comps::Dict{String, Vector{T}}`: The dictionary of individual components of the model intensity
- `n_dust_cont::Integer`: The number of dust continuum components in the fit
- `n_dust_features::Integer`: The number of PAH features in the fit
- `line_wave::Vector{<:Real}`: List of nominal central wavelengths for each line in the fit
- `line_names::Vector{Symbol}`: List of names for each line in the fit
- `screen::Bool`: The type of model used for extinction screening
- `z::Real`: The redshift of the object being fit
- `χ2red::Real`: The reduced χ^2 value of the fit
- `name::String`: The name of the object being fit
- `label::String`: A label for the individual spaxel being plotted, to be put in the file name
- `backend::Symbol`: The backend to use to plot, either `:pyplot` or `:plotly`
"""
function plot_spaxel_fit(λ::Vector{<:Real}, I::Vector{<:Real}, I_cont::Vector{<:Real}, σ::Vector{<:Real}, comps::Dict{String, Vector{T}}, 
    n_dust_cont::Integer, n_dust_features::Integer, line_wave::Vector{<:Real}, line_names::Vector{Symbol}, screen::Bool, 
    z::Real, χ2red::Real, name::String, label::String; backend::Symbol=:pyplot, 
    range::Union{Tuple,Nothing}=nothing) where {T<:Real}

    # Plotly ---> useful interactive plots for visually inspecting data, but not publication-quality
    if (backend == :plotly || backend == :both) && isnothing(range)
        # Plot the overall data / model
        trace1 = PlotlyJS.scatter(x=λ, y=I, mode="lines", line=Dict(:color => "black", :width => 1), name="Data", showlegend=true)
        trace2 = PlotlyJS.scatter(x=λ, y=I_cont, mode="lines", line=Dict(:color => "red", :width => 1), name="Continuum Fit", showlegend=true)
        traces = [trace1, trace2]
        # Loop over and plot individual model components
        for comp ∈ keys(comps)
            if comp == "extinction"
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["abs_ice"] .* comps["abs_ch"] .* maximum(I_cont) .* 1.1, mode="lines", 
                    line=Dict(:color => "black", :width => 1, :dash => "dash"), name="Extinction")])
            # elseif comp == "stellar"
                # append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"], mode="lines",
                    # line=Dict(:color => "red", :width => 1, :dash => "dash"), name="Stellar Continuum")])
            # elseif occursin("dust_cont", comp)
                # append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"], mode="lines",
                    # line=Dict(:color => "green", :width => 1, :dash => "dash"), name="Dust Continuum")])
            # elseif occursin("dust_feat", comp)
                # append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"], mode="lines",
                    # line=Dict(:color => "blue", :width => 1), name="Dust Features")])
            elseif occursin("hot_dust", comp)
                append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp], mode="lines", line=Dict(:color => "yellow", :width => 1),
                    name="Hot Dust")])
            elseif occursin("line", comp)
                if occursin("acomp", comp)
                    append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"], mode="lines",
                        line=Dict(:color => "#F574F9", :width => 1), name="acomps")])
                else
                    append!(traces, [PlotlyJS.scatter(x=λ, y=comps[comp] .* comps["extinction"], mode="lines",
                        line=Dict(:color => "rebeccapurple", :width => 1), name="Lines")])
                end
            end
        end
        # Add vertical dashed lines for emission line rest wavelengths
        for (lw, ln) ∈ zip(line_wave, line_names)
            append!(traces, [PlotlyJS.scatter(x=[lw, lw], y=[0., nanmaximum(I)*1.1], mode="lines", 
                line=Dict(:color => occursin("H2", String(ln)) ? "red" : 
                              (any(occursin.(["alpha", "beta", "gamma", "delta"], String(ln))) ? "#ff7f0e" : "rebeccapurple"), 
                          :width => 0.5, :dash => "dash"))])
        end
        # Add the summed up continuum
        append!(traces, [PlotlyJS.scatter(x=λ, y=("hot_dust" ∈ keys(comps) ? comps["hot_dust"] : zeros(length(λ))) .+ 
            comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"] .* (sum([comps["dust_cont_$i"] for i ∈ 1:n_dust_cont], dims=1)[1] .+ comps["stellar"]),
            mode="lines", line=Dict(:color => "green", :width => 1), name="Dust+Stellar Continuum")])
        append!(traces, [PlotlyJS.scatter(x=λ, y=sum([comps["dust_feat_$i"] for i ∈ 1:n_dust_features], dims=1)[1] .* comps["extinction"],
            mode="lines", line=Dict(:color => "blue", :width => 1), name="PAHs")])
        # axes labels / titles / fonts
        layout = PlotlyJS.Layout(
            xaxis_title=L"$\lambda\ (\mu{\rm m})$",
            yaxis_title=L"$I_{\nu}\ ({\rm MJy}\,{\rm sr}^{-1})$",
            title=L"$\tilde{\chi}^2 = %$χ2red$",
            xaxis_constrain="domain",
            font_family="Georgia, Times New Roman, Serif",
            template="plotly_white",
            # annotations=[
            #     attr(x=lw, y=nanmaximum(I)*.75, text=ll) for (lw, ll) ∈ zip(line_wave, line_latex)
            # ]
        )
        # save as HTML file
        p = PlotlyJS.plot(traces, layout)
        PlotlyJS.savefig(p, isnothing(label) ? joinpath("output_$name", "spaxel_plots", "levmar_fit_spaxel.html") : 
            joinpath("output_$name", "spaxel_plots", "$label.html"))

    end

    # Pyplot --> actually publication-quality plots finely tuned to be the most useful and visually appealing that I could make them
    if backend == :pyplot || backend == :both

        # If max is above 10^4, normalize so the y axis labels aren't super wide
        power = floor(Int, log10(maximum(I ./ λ)))
        if power ≥ 4
            norm = 10^power
        else
            norm = 1
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
        ax1.plot(λ, I ./ norm ./ λ, "k-", label="Data")
        ax1.plot(λ, I_cont ./ norm ./ λ, "-", color="#ff5d00", label="Model")
        ax2.plot(λ, (I.-I_cont) ./ norm ./ λ, "k-")
        χ2_str = @sprintf "%.3f" χ2red
        ax2.plot(λ, zeros(length(λ)), "-", color="#ff5d00", label=L"$\tilde{\chi}^2 = %$χ2_str$")
        ax2.fill_between(λ, (I.-I_cont.+σ)./norm./λ, (I.-I_cont.-σ)./norm./λ, color="k", alpha=0.5)
        # twin axes with different labels --> extinction for ax3 and observed wavelength for ax4
        ax3 = ax1.twinx()
        ax4 = ax1.twiny()

        # full continuum
        ax1.plot(λ, (comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"] .* 
            (sum([comps["dust_cont_$i"] for i ∈ 1:n_dust_cont], dims=1)[1] .+ comps["stellar"])) ./ norm ./ λ, "k--", alpha=0.5, 
            label="Continuum")
        # full PAH profile
        ax1.plot(λ, sum([comps["dust_feat_$i"] for i ∈ 1:n_dust_features], dims=1)[1] .* comps["extinction"] ./ norm ./ λ, "-", 
            color="#0065ff", label="PAHs")
        # full line profile
        ax1.plot(λ, sum([comps["line_$i"] .+ (haskey(comps, "line_$(i)_acomp") ? comps["line_$(i)_acomp"] : 0.) 
            for i ∈ 1:length(line_wave)], dims=1)[1] .* comps["extinction"] ./ norm ./ λ, "-", color="rebeccapurple", alpha=0.6, 
            label="Lines")
        # plot extinction
        ax3.plot(λ, comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"], "k:", alpha=0.5, label="Extinction")
        # plot hot dust
        if haskey(comps, "hot_dust")
            ax1.plot(λ, comps["hot_dust"] ./ norm ./ λ, "-", color="#8ac800", alpha=0.6, label="Hot Dust")
        end

        # loop over and plot individual model components
        # for comp ∈ keys(comps)
            # if comp == "extinction"
                # ax3.plot(λ, comps[comp] .* comps["abs_ice"] .* comps["abs_ch"], "k--", alpha=0.5)
            # elseif comp == "stellar"
                # ax1.plot(λ, comps[comp] .* comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"] ./ norm ./ λ, "--", color=:orangered, alpha=0.5)
            # elseif occursin("dust_cont", comp)
                # ax1.plot(λ, comps[comp] .* comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"] ./ norm ./ λ, "--", color=:yellowgreen, alpha=0.5)
            # elseif occursin("dust_feat", comp)
                # ax1.plot(λ, comps[comp] .* comps["extinction"] ./ norm ./ λ, "-", color=:cornflowerblue, alpha=0.5)
            # elseif occursin("hot_dust", comp)
                # ax1.plot(λ, comps[comp] ./ norm ./ λ, "-", color="goldenrod")
            # elseif occursin("line", comp)
            #     if occursin("acomp", comp)
            #         ax1.plot(λ, comps[comp] .* comps["extinction"] ./ norm ./ λ, "-", color="fuchsia")
            #     else
            #         ax1.plot(λ, comps[comp] .* comps["extinction"] ./ norm ./ λ, "-", color="ff009b")
            #     end
            # end
        # end

        # plot vertical dashed lines for emission line wavelengths
        for (lw, ln) ∈ zip(line_wave, line_names)
            ax1.axvline(lw, linestyle="--", color="k", lw=0.5, alpha=0.5)
            ax2.axvline(lw, linestyle="--", color="k", lw=0.5, alpha=0.5)
        end

        line_mask = falses(length(λ))
        for ln ∈ line_wave
            window_size = 3000. / Util.C_KMS * ln
            window = (ln - window_size) .< λ .< (ln + window_size)
            line_mask .|= window
        end
        # set axes limits and labels
        if isnothing(range)
            ax1.set_xlim(minimum(λ), maximum(λ))
            ax2.set_xlim(minimum(λ), maximum(λ))
            ax4.set_xlim(minimum(Util.observed_frame(λ, z)), maximum(Util.observed_frame(λ, z)))
            ax1.set_ylim(-0.01, 1.3nanmaximum(I[.!line_mask] ./ norm ./ λ[.!line_mask]))
        else
            ax1.set_xlim(range[1], range[2])
            ax2.set_xlim(range[1], range[2])
            ax4.set_xlim(Util.observed_frame(range[1], z), Util.observed_frame(range[2], z))
            ax1.set_ylim(-0.01, 1.1nanmaximum((I ./ norm ./ λ)[range[1] .< λ .< range[2]]))
        end
        ax2.set_ylim(-1.1maximum(((I.-I_cont) ./ norm ./ λ)[.!line_mask]), 1.1maximum(((I.-I_cont) ./ norm ./ λ)[.!line_mask]))
        ax3.set_yscale("log") # logarithmic extinction axis
        ax3.set_ylim(1e-3, 1.)
        if screen
            ax3.set_ylabel(L"$e^{-\tau_{\lambda}}$")
        else
            ax3.set_ylabel(L"$(1-e^{-\tau_{\lambda}}) / \tau_{\lambda}$")
        end
        if power ≥ 4
            ax1.set_ylabel(L"$I_{\nu}/\lambda$ ($10^{%$power}$ MJy sr$^{-1}$ $\mu$m$^{-1}$)")
        else
            ax1.set_ylabel(L"$I_{\nu}/\lambda$ (MJy sr$^{-1}$ $\mu$m$^{-1}$)")
        end
        ax2.set_ylabel(L"$O-C$")  # ---> residuals, (O)bserved - (C)alculated
        ax2.set_xlabel(L"$\lambda_{\rm rest}$ ($\mu$m)")
        ax4.set_xlabel(L"$\lambda_{\rm obs}$ ($\mu$m)")
        ax2.legend(loc="upper left")

        # Set minor ticks as multiples of 0.1 μm for x axis and automatic for y axis
        ax1.xaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax1.yaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax2.xaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax2.yaxis.set_minor_locator(py_ticker.AutoMinorLocator())
        ax4.xaxis.set_minor_locator(py_ticker.AutoMinorLocator())

        # Set major ticks and formats
        ax1.set_xticklabels([]) # ---> will be covered up by the residuals plot
        ax2.set_yticks([-round(maximum(((I.-I_cont) ./ norm ./ λ)[.!line_mask]) / 2, sigdigits=1), 0.0, round(maximum(((I.-I_cont) ./ norm ./ λ)[.!line_mask]) / 2, sigdigits=1)])
        ax1.tick_params(which="both", axis="both", direction="in")
        ax2.tick_params(which="both", axis="both", direction="in", labelright=true, right=true, top=true)
        ax3.tick_params(which="both", axis="both", direction="in")
        ax4.tick_params(which="both", axis="both", direction="in")
        
        # Save figure as PDF, yay for vector graphics!
        plt.savefig(isnothing(label) ? joinpath("output_$name", "spaxel_plots", "levmar_fit_spaxel.pdf") : 
            joinpath("output_$name", isnothing(range) ? "spaxel_plots" : "zoomed_plots", "$label.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    end
end


"""
    calculate_extra_parameters(cube_fitter, spaxel, comps)

Calculate extra parameters that are not fit, but are nevertheless important to know, for a given spaxel.
Currently this includes the integrated intensity and signal to noise ratios of dust features and emission lines.

# Arguments
`T<:Real`
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
- `spaxel::CartesianIndex`: The coordinates of the spaxel to be fit
- `popt_c::Vector{T}`: The best-bit parameter vector for the continuum components of the fit
- `popt_l::Vector{T}`: The best-fit parameter vector for the line components of the fit
"""
function calculate_extra_parameters(cube_fitter::CubeFitter, spaxel::CartesianIndex, 
    popt_c::Vector{T}, popt_l::Vector{T}, perr_c::Vector{T}, perr_l::Vector{T}, 
    extinction::Vector{T}, mask_lines::BitVector, continuum::Vector{T}) where {T<:Real}

    @debug "Calculating extra parameters"

    # Extract the wavelength, intensity, and uncertainty data
    λ = cube_fitter.cube.λ
    I = cube_fitter.cube.Iν[spaxel, :]
    σ = cube_fitter.cube.σI[spaxel, :]

    # Get the average wavelength resolution, and divide by 10 to subsample pixels
    # (this is only used to calculate the peak location of the profiles, and thus the peak intensity)
    Δλ = mean(diff(λ)) / 10

    # Normalization
    N = Float64(abs(nanmaximum(I)))
    N = N ≠ 0. ? N : 1.
    @debug "Normalization: $N"

    # Loop through dust features
    p_dust = zeros(3cube_fitter.n_dust_feat)
    p_dust_err = zeros(3cube_fitter.n_dust_feat)
    pₒ = 1
    # Initial parameter vector index where dust profiles start
    pᵢ = 3 + 2cube_fitter.n_dust_cont + 4 + (cube_fitter.fit_sil_emission ? 5 : 0)

    for (ii, df) ∈ enumerate(cube_fitter.dust_features)

        # unpack the parameters
        A, μ, fwhm = popt_c[pᵢ:pᵢ+2]
        A_err, μ_err, fwhm_err = perr_c[pᵢ:pᵢ+2]
        # Convert peak intensity to CGS units (erg s^-1 cm^-2 μm^-1 sr^-1)
        A_cgs = Util.MJysr_to_cgs(A, μ)
        # Convert the error in the intensity to CGS units
        A_cgs_err = Util.MJysr_to_cgs_err(A, A_err, μ, μ_err)

        # Get the extinction profile at the center
        ext = extinction[argmin(abs.(λ .- μ))]

        # Calculate the intensity using the utility function
        intensity, i_err = Util.calculate_intensity(:Drude, A_cgs, A_cgs_err, μ, μ_err, fwhm, fwhm_err)
        # Calculate the equivalent width using the utility function
        eqw, e_err = Util.calculate_eqw(popt_c, perr_c, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, 
            cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.fit_sil_emission,
            :Drude, A*ext, A_err*ext, μ, μ_err, fwhm, fwhm_err)

        @debug "Drude profile with ($A_cgs, $μ, $fwhm) and errors ($A_cgs_err, $μ_err, $fwhm_err)"
        @debug "I=$intensity, err=$i_err, EQW=$eqw, err=$e_err"

        # increment the parameter index
        pᵢ += 3

        # intensity units: erg s^-1 cm^-2 sr^-1 (integrated over μm)
        p_dust[pₒ] = intensity
        p_dust_err[pₒ] = i_err
        # equivalent width units: μm
        p_dust[pₒ+1] = eqw
        p_dust_err[pₒ+1] = e_err

        # SNR, calculated as (peak amplitude) / (RMS intensity of the surrounding spectrum)
        # include the extinction factor when calculating the SNR
        p_dust[pₒ+2] = Util.calculate_SNR(Δλ, I[.!mask_lines] .- continuum[.!mask_lines], :Drude, A*ext, μ, fwhm)
        @debug "Dust feature $df with integrated intensity $(p_dust[pₒ]) +/- $(p_dust_err[pₒ]) " *
            "(erg s^-1 cm^-2 sr^-1), equivalent width $(p_dust[pₒ+1]) +/- $(p_dust_err[pₒ+1]) um, " *
            "and SNR $(p_dust[pₒ+2])"

        pₒ += 3
    end

    # Loop through lines
    p_lines = zeros(3cube_fitter.n_lines+3cube_fitter.n_acomps)
    p_lines_err = zeros(3cube_fitter.n_lines+3cube_fitter.n_acomps)
    pₒ = 1
    # Skip over the tied velocity offsets
    pᵢ = 2cube_fitter.n_kin_tied + 2cube_fitter.n_acomp_kin_tied + 1
    # Skip over the tied voigt mixing parameter, saving its index
    if cube_fitter.tie_voigt_mixing
        ηᵢ = pᵢ
        pᵢ += 1
    end
    for (k, ln) ∈ enumerate(cube_fitter.lines)

        # (\/ pretty much the same as the fit_line_residuals function, but calculating the integrated intensities)
        amp = popt_l[pᵢ]
        amp_err = perr_l[pᵢ]
        # fill values with nothings for profiles that may / may not have them
        h3 = h3_err = h4 = h4_err = η = η_err = nothing

        # Check if voff is tied: if so, use the tied voff parameter, otherwise, use the line's own voff parameter
        if isnothing(cube_fitter.line_tied[k])
            # Unpack the components of the line
            voff = popt_l[pᵢ+1]
            voff_err = perr_l[pᵢ+1]
            fwhm = popt_l[pᵢ+2]
            fwhm_err = perr_l[pᵢ+2]
            pᵢ += 3
        elseif !isnothing(cube_fitter.line_tied[k]) && cube_fitter.flexible_wavesol
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and kin_tied_key
            vwhere = findfirst(x -> x == cube_fitter.line_tied[k], cube_fitter.kin_tied_key)
            voff_series = popt_l[vwhere]
            voff_indiv = popt_l[pᵢ+1]
            # Add velocity shifts of the tied lines and the individual offsets together
            voff = voff_series + voff_indiv
            voff_err = √(perr_l[vwhere]^2 + perr_l[pᵢ+1]^2)
            fwhm = popt_l[cube_fitter.n_kin_tied+vwhere]
            fwhm_err = perr_l[cube_fitter.n_kin_tied+vwhere]
            pᵢ += 2
        else
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and kin_tied_key
            vwhere = findfirst(x -> x == cube_fitter.line_tied[k], cube_fitter.kin_tied_key)
            voff = popt_l[vwhere]
            voff_err = perr_l[vwhere]
            fwhm = popt_l[cube_fitter.n_kin_tied+vwhere]
            fwhm_err = perr_l[cube_fitter.n_kin_tied+vwhere]
            pᵢ += 1
        end

        if cube_fitter.line_profiles[k] == :GaussHermite
            # Get additional h3, h4 components
            h3 = popt_l[pᵢ]
            h3_err = perr_l[pᵢ]
            h4 = popt_l[pᵢ+1]
            h4_err = perr_l[pᵢ+1]
            pᵢ += 2
        elseif cube_fitter.line_profiles[k] == :Voigt
            # Get additional mixing component, either from the tied position or the 
            # individual position
            if !cube_fitter.tie_voigt_mixing
                η = popt_l[pᵢ]
                η_err = perr_l[pᵢ]
                pᵢ += 1
            else
                η = popt_l[ηᵢ]
                η_err = perr_l[ηᵢ]
            end
        end

        # Convert voff in km/s to mean wavelength in μm
        mean_μm = Util.Doppler_shift_λ(ln.λ₀, voff)
        mean_μm_err = ln.λ₀ / Util.C_KMS * voff_err
        # WARNING:
        # Probably should set to 0 if using flexible tied voffs since they are highly degenerate and result in massive errors
        # if !isnothing(cube_fitter.line_tied[k]) && cube_fitter.flexible_wavesol
        #     mean_μm_err = 0.
        # end

        # Convert FWHM from km/s to μm
        fwhm_μm = Util.Doppler_shift_λ(ln.λ₀, fwhm/2) - Util.Doppler_shift_λ(ln.λ₀, -fwhm/2)
        fwhm_μm_err = ln.λ₀ / Util.C_KMS * fwhm_err

        # Convert amplitude to erg s^-1 cm^-2 μm^-1 sr^-1, put back in the normalization
        amp_cgs = Util.MJysr_to_cgs(amp*N, mean_μm)
        amp_cgs_err = Util.MJysr_to_cgs_err(amp*N, amp_err*N, mean_μm, mean_μm_err)

        # Get the extinction factor at the line center
        ext = extinction[argmin(abs.(λ .- mean_μm))]

        @debug "Line with ($amp_cgs, $mean_μm, $fwhm_μm) and errors ($amp_cgs_err, $mean_μm_err, $fwhm_μm_err)"

        # Calculate line intensity using the helper function
        p_lines[pₒ], p_lines_err[pₒ] = Util.calculate_intensity(cube_fitter.line_profiles[k], amp_cgs, amp_cgs_err, mean_μm, mean_μm_err,
            fwhm_μm, fwhm_μm_err, h3=h3, h3_err=h3_err, h4=h4, h4_err=h4_err, η=η, η_err=η_err)

        # Calculate the equivalent width using the utility function
        p_lines[pₒ+1], p_lines_err[pₒ+1] = Util.calculate_eqw(popt_c, perr_c, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat,
            cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.fit_sil_emission,
            cube_fitter.line_profiles[k], amp*N*ext, amp_err*N*ext, mean_μm, mean_μm_err, fwhm_μm, fwhm_μm_err, 
            h3=h3, h3_err=h3_err, h4=h4, h4_err=h4_err, η=η, η_err=η_err)

        # SNR
        p_lines[pₒ+2] = amp*N*ext / std(I[.!mask_lines] .- continuum[.!mask_lines])

        @debug "Line $(cube_fitter.line_names[k]) with integrated intensity $(p_lines[pₒ]) +/- $(p_lines_err[pₒ]) " *
            "(erg s^-1 cm^-2 sr^-1), equivalent width $(p_lines[pₒ+1]) +/- $(p_lines_err[pₒ+1]) um, and SNR $(p_lines[pₒ+1])"

        # Advance the output vector index by 3
        pₒ += 3

        # filler values
        acomp_amp = acomp_mean_μm = acomp_fwhm_μm = nothing
        acomp_h3 = acomp_h3_err = acomp_h4 = acomp_h4_err = acomp_η = acomp_η_err = nothing
        # Repeat EVERYTHING, minus the flexible_wavesol, for the additional components
        if !isnothing(cube_fitter.line_acomp_profiles[k])
            acomp_amp = amp * popt_l[pᵢ]
            acomp_amp_err = √(acomp_amp^2 * ((amp_err / amp)^2 + (perr_l[pᵢ] / acomp_amp)^2))
            if isnothing(cube_fitter.line_acomp_tied[k])
                acomp_voff = voff + popt_l[pᵢ+1]
                acomp_voff_err = √(voff_err^2 + perr_l[pᵢ+1]^2)
                acomp_fwhm = fwhm * popt_l[pᵢ+2]
                acomp_fwhm_err = √(acomp_fwhm^2 * ((fwhm_err / fwhm)^2 + (perr_l[pᵢ+2] / acomp_fwhm)^2))
                pᵢ += 3

            else
                vwhere = findfirst(x -> x == cube_fitter.line_acomp_tied[k], cube_fitter.acomp_kin_tied_key)
                acomp_voff = voff + popt_l[2cube_fitter.n_kin_tied+vwhere]
                acomp_voff_err = √(voff_err^2 + perr_l[2cube_fitter.n_kin_tied+vwhere]^2)
                acomp_fwhm = fwhm * popt_l[2cube_fitter.n_kin_tied+cube_fitter.n_acomp_kin_tied+vwhere]
                acomp_fwhm_err = √(acomp_fwhm^2 * ((fwhm_err / fwhm)^2 + 
                    (perr_l[2cube_fitter.n_kin_tied+cube_fitter.n_acomp_kin_tied+vwhere] / acomp_fwhm)^2))      
                pᵢ += 1
            end

            if cube_fitter.line_acomp_profiles[k] == :GaussHermite
                acomp_h3 = popt_l[pᵢ]
                acomp_h3_err = perr_l[pᵢ]
                acomp_h4 = popt_l[pᵢ+1]
                acomp_h4_err = perr_l[pᵢ+1]
                pᵢ += 2
            elseif cube_fitter.line_acomp_profiles[k] == :Voigt
                if !cube_fitter.tie_voigt_mixing
                    acomp_η = popt_l[pᵢ]
                    acomp_η_err = perr_l[pᵢ]
                    pᵢ += 1
                else
                    acomp_η = popt_l[ηᵢ]
                    acomp_η_err = perr_l[ηᵢ]
                end
            end

            # Convert voff in km/s to mean wavelength in μm
            acomp_mean_μm = Util.Doppler_shift_λ(ln.λ₀, acomp_voff)
            acomp_mean_μm_err = ln.λ₀ / Util.C_KMS * acomp_voff_err
            # Convert FWHM from km/s to μm
            acomp_fwhm_μm = Util.Doppler_shift_λ(ln.λ₀, acomp_fwhm/2) - Util.Doppler_shift_λ(ln.λ₀, -acomp_fwhm/2)
            acomp_fwhm_μm_err = ln.λ₀ / Util.C_KMS * acomp_fwhm_err

            # Convert amplitude to erg s^-1 cm^-2 μm^-1 sr^-1, put back in the normalization
            acomp_amp_cgs = Util.MJysr_to_cgs(acomp_amp*N, acomp_mean_μm)
            acomp_amp_cgs_err = Util.MJysr_to_cgs_err(acomp_amp*N, acomp_amp_err*N, acomp_mean_μm, acomp_mean_μm_err)

            @debug "acomp line with ($acomp_amp_cgs, $acomp_mean_μm, $acomp_fwhm_μm) and errors ($acomp_amp_cgs_err, $acomp_mean_μm_err, $acomp_fwhm_μm_err)"
                   
            # Calculate line intensity using the helper function
            p_lines[pₒ], p_lines_err[pₒ] = Util.calculate_intensity(cube_fitter.line_acomp_profiles[k], acomp_amp_cgs, acomp_amp_cgs_err, acomp_mean_μm, 
                acomp_mean_μm_err, acomp_fwhm_μm, acomp_fwhm_μm_err, h3=acomp_h3, h3_err=acomp_h3_err, h4=acomp_h4, h4_err=acomp_h4_err, 
                η=acomp_η, η_err=acomp_η_err)
  
            # Calculate the equivalent width using the utility function
            p_lines[pₒ+1], p_lines_err[pₒ+1] = Util.calculate_eqw(popt_c, perr_c, cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, 
                cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.fit_sil_emission,
                cube_fitter.line_acomp_profiles[k], acomp_amp*N*ext, acomp_amp_err*N*ext, acomp_mean_μm, acomp_mean_μm_err, 
                acomp_fwhm_μm, acomp_fwhm_μm_err, h3=acomp_h3, h3_err=acomp_h3_err, h4=acomp_h4, h4_err=acomp_h4_err, η=acomp_η, 
                η_err=acomp_η_err)

            # SNR
            p_lines[pₒ+2] = acomp_amp*N*ext / std(I[.!mask_lines] .- continuum[.!mask_lines])

            @debug "Acomp line for $(cube_fitter.line_names[k]) with integrated intensity $(p_lines[pₒ]) +/- $(p_lines_err[pₒ]) " *
                "(erg s^-1 cm^-2 sr^-1), equivalent width $(p_lines[pₒ+1]) +/- $(p_lines_err[pₒ+1]) um, and SNR $(p_lines[pₒ+1])"

            # Advance the output vector index by 3
            pₒ += 3
        end

        # SNR, calculated as (amplitude) / (RMS of the surrounding spectrum)
        # p_lines[pₒ+2] = Util.calculate_SNR(Δλ, I[.!mask_lines] .- continuum[.!mask_lines], cube_fitter.line_profiles[k],
        #     amp*N*ext, mean_μm, fwhm_μm, h3=h3, h4=h4, η=η, acomp_prof=cube_fitter.line_acomp_profiles[k], 
        #     acomp_amp=isnothing(acomp_amp) ? acomp_amp : acomp_amp*N*ext,
        #     acomp_peak=acomp_mean_μm, acomp_fwhm=acomp_fwhm_μm, acomp_h3=acomp_h3, acomp_h4=acomp_h4, acomp_η=acomp_η)

        # pₒ += 3

    end

    p_dust, p_lines, p_dust_err, p_lines_err
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
function fit_spaxel(cube_fitter::CubeFitter, spaxel::CartesianIndex; recalculate_params=false)

    local p_out
    local p_err

    # Skip spaxels with NaNs (post-interpolation)
    λ = cube_fitter.cube.λ
    I = cube_fitter.cube.Iν[spaxel, :]
    σ = cube_fitter.cube.σI[spaxel, :]

    # if there are any NaNs, skip over the spaxel
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

            line_locs, mask_lines, I_spline, σ_spline = continuum_cubic_spline(λ, I, σ)

            # Fit the spaxel
            σ, popt_c, I_cont, comps_cont, n_free_c, perr_c, covar_c = 
                @timeit timer_output "continuum_fit_spaxel" continuum_fit_spaxel(cube_fitter, spaxel, mask_lines, I_spline, σ_spline)
            _, popt_l, I_line, comps_line, n_free_l, perr_l, covar_l = 
                @timeit timer_output "line_fit_spaxel" line_fit_spaxel(cube_fitter, spaxel, I_cont, comps_cont["extinction"], line_locs, 
                    mask_lines, I_spline, σ_spline)

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
                @timeit timer_output "calculate_extra_parameters" calculate_extra_parameters(cube_fitter, spaxel, popt_c, popt_l, perr_c, perr_l,
                    comps["extinction"], mask_lines, I_spline)
            p_out = [popt_c; popt_l; p_dust; p_lines; χ2red]
            p_err = [perr_c; perr_l; p_dust_err; p_lines_err; 0.]

            # Plot the fit
            λ0_ln = [ln.λ₀ for ln ∈ cube_fitter.lines]
            if cube_fitter.plot_spaxels != :none
                @debug "Plotting spaxel $spaxel best fit"
                @timeit timer_output "plot_spaxel_fit" plot_spaxel_fit(λ, I, I_model, σ, comps, 
                    cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, λ0_ln, cube_fitter.line_names, cube_fitter.extinction_screen, 
                    cube_fitter.z, χ2red, cube_fitter.name, "spaxel_$(spaxel[1])_$(spaxel[2])", backend=cube_fitter.plot_spaxels)
                if !isnothing(cube_fitter.plot_range)
                    for (i, plot_range) ∈ enumerate(cube_fitter.plot_range)
                        @timeit timer_output "plot_line_fit" plot_spaxel_fit(λ, I, I_model, σ, comps,
                            cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, λ0_ln, cube_fitter.line_names, cube_fitter.extinction_screen,
                            cube_fitter.z, χ2red, cube_fitter.name, "lines_$(spaxel[1])_$(spaxel[2])_$i", backend=cube_fitter.plot_spaxels,
                            range=plot_range)
                    end
                end
            end

            @debug "Saving results to binary for spaxel $spaxel"
            # serialize(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).LOKI"), (p_out=p_out, p_err=p_err))
            # save output as csv file
            open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).csv"), "w") do f 
                @timeit timer_output "writedlm" writedlm(f, [p_out p_err], ',')
            end

            # save memory allocations & other logistic data to a separate log file
            if cube_fitter.track_memory
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
    results = readdlm(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).csv"), ',', Float64, '\n')
    p_out = results[:, 1]
    p_err = results[:, 2]

    # if requested, recalculate the extra parameters obtained with the calculate_extra_parameters function
    if recalculate_params
        # Get the line mask and cubic spline continuum
        line_locs, mask_lines, I_spline, σ_spline = continuum_cubic_spline(λ, I, σ)

        # Separate the fit parameters
        popt_c = p_out[1:cube_fitter.n_params_cont]
        perr_c = p_err[1:cube_fitter.n_params_cont]
        popt_l = p_out[(cube_fitter.n_params_cont+1):(cube_fitter.n_params_cont+cube_fitter.n_params_lines)]
        perr_l = p_err[(cube_fitter.n_params_cont+1):(cube_fitter.n_params_cont+cube_fitter.n_params_lines)]

        # Get extinction parameters
        τ = popt_c[3+2cube_fitter.n_dust_cont]
        β = popt_c[3+2cube_fitter.n_dust_cont+3]

        # Get the extinction curve
        if cube_fitter.extinction_curve == "d+"
            ext_curve = Util.τ_dp.(λ, β)
        elseif cube_fitter.extinction_curve == "kvt"
            ext_curve = Util.τ_kvt.(λ, β)
        elseif cube_fitter.extinction_curve == "ct"
            ext_curve = Util.τ_ct.(λ)
        else
            error("Unrecognized extinction curve: $extinction_curve")
        end
        extinction = Util.Extinction.(ext_curve, τ, screen=cube_fitter.extinction_screen)

        p_dust, p_lines, p_dust_err, p_lines_err = 
            @timeit timer_output "calculate_extra_parameters" calculate_extra_parameters(cube_fitter, spaxel, popt_c, popt_l, perr_c, perr_l,
                extinction, mask_lines, I_spline)

        # Reconstruct the output and error vectors
        p_out = [popt_c; popt_l; p_dust; p_lines; p_out[end]]
        p_err = [perr_c; perr_l; p_dust_err; p_lines_err; p_err[end]]

        # Rewrite outputs
        open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "spaxel_$(spaxel[1])_$(spaxel[2]).csv"), "w") do f 
            @timeit timer_output "writedlm" writedlm(f, [p_out p_err], ',')
        end
    end

    p_out, p_err

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
    I_sum_init = Util.Σ(cube_fitter.cube.Iν, (1,2)) ./ Util.Σ(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    σ_sum_init = sqrt.(Util.Σ(cube_fitter.cube.σI.^2, (1,2))) ./ Util.Σ(Array{Int}(.~cube_fitter.cube.mask), (1,2))

    line_locs, mask_lines, I_spline_init, σ_spline_init = continuum_cubic_spline(λ_init, I_sum_init, σ_sum_init)

    # Continuum and line fits
    σ_init, popt_c_init, I_c_init, comps_c_init, n_free_c_init, _, _ = continuum_fit_spaxel(cube_fitter, CartesianIndex(0,0),
        mask_lines, I_spline_init, σ_spline_init; init=true)
    _, popt_l_init, I_l_init, comps_l_init, n_free_l_init, _, _ = line_fit_spaxel(cube_fitter, CartesianIndex(0,0), I_c_init, 
        comps_c_init["extinction"], line_locs, mask_lines, I_spline_init, σ_spline_init; init=true)

    # Get the overall models
    I_model_init = I_c_init .+ I_l_init
    comps_init = merge(comps_c_init, comps_l_init)

    n_free_init = n_free_c_init + n_free_l_init
    n_data_init = length(I_sum_init)

    # Calculate reduce chi^2
    χ2red_init = 1 / (n_data_init - n_free_init) * sum((I_sum_init .- I_model_init).^2 ./ σ_init.^2)

    # Plot the fit
    λ0_ln = [ln.λ₀ for ln ∈ cube_fitter.lines]
    if cube_fitter.plot_spaxels != :none
        @debug "Plotting spaxel sum initial fit"
        plot_spaxel_fit(λ_init, I_sum_init, I_model_init, σ_init, comps_init,
            cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, λ0_ln, cube_fitter.line_names, cube_fitter.extinction_screen, 
            cube_fitter.z, χ2red_init, cube_fitter.name, "initial_sum_fit", backend=cube_fitter.plot_spaxels)
        if !isnothing(cube_fitter.plot_range)
            for (i, plot_range) ∈ enumerate(cube_fitter.plot_range)
                plot_spaxel_fit(λ_init, I_sum_init, I_model_init, σ_init, comps_init,
                    cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, λ0_ln, cube_fitter.line_names, cube_fitter.extinction_screen, 
                    cube_fitter.z, χ2red_init, cube_fitter.name, "initial_sum_line_$i", backend=cube_fitter.plot_spaxels;
                    range=plot_range)
            end
        end
            
    end

end


"""
    fit_cube!(cube_fitter)

This is the main cube fitting function!! It's essentially a wrapper function to perform a full fit of an 
entire IFU cube, calling `fit_spaxel` for each spaxel in a parallel or serial loop depending on the cube_fitter options.  
Results are then concatenated into `ParamMaps` and `CubeModel` structs and plotted/saved, also based on the 
cube_fitter options.

# Arguments
- `cube_fitter::CubeFitter`: The CubeFitter object containing the data, parameters, and options for the fit
"""
function fit_cube!(cube_fitter::CubeFitter)::Tuple{CubeFitter, ParamMaps, ParamMaps, CubeModel}

    @info """\n
    #############################################################################
    ######## BEGINNING FULL CUBE FITTING ROUTINE FOR $(cube_fitter.name) ########
    #############################################################################
    """

    shape = size(cube_fitter.cube.Iν)
    # Interpolate NaNs in the cube
    interpolate_cube!(cube_fitter.cube)

    # Prepare output array
    @info "===> Preparing output data structures... <==="
    out_params = SharedArray(ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 1) .* NaN)
    out_errs = SharedArray(ones(shape[1:2]..., cube_fitter.n_params_cont + cube_fitter.n_params_lines + 
        cube_fitter.n_params_extra + 1) .* NaN)

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
    spaxels = CartesianIndices(selectdim(cube_fitter.cube.Iν, 3, 1))

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

    # Create the CubeModel and ParamMaps structs to be filled in
    cube_model = generate_cubemodel(cube_fitter)
    param_maps, param_errs = generate_parammaps(cube_fitter)

    # Loop over each spaxel and fill in the associated fitting parameters into the ParamMaps and CubeModel
    # I know this is long and ugly and looks stupid but it works for now and I'll make it pretty later
    prog = Progress(length(spaxels); showspeed=true)
    @inbounds for index ∈ spaxels

        # Set the 2D parameter map outputs

        # Conversion factor from MJy sr^-1 to erg s^-1 cm^-2 Hz^-1 sr^-1 = 10^6 * 10^-23 = 10^-17
        # So, log10(A * 1e-17) = log10(A) - 17

        # Stellar continuum amplitude, temp
        param_maps.stellar_continuum[:amp][index] = out_params[index, 1] > 0. ? log10(out_params[index, 1])-17 : -Inf 
        param_errs.stellar_continuum[:amp][index] = out_params[index, 1] > 0. ? out_errs[index, 1] / (log(10) * out_params[index, 1]) : NaN
        param_maps.stellar_continuum[:temp][index] = out_params[index, 2]
        param_errs.stellar_continuum[:temp][index] = out_errs[index, 2]
        pᵢ = 3

        # Dust continuum amplitude, temp
        for i ∈ 1:cube_fitter.n_dust_cont
            param_maps.dust_continuum[i][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ])-17 : -Inf
            param_errs.dust_continuum[i][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.dust_continuum[i][:temp][index] = out_params[index, pᵢ+1]
            param_errs.dust_continuum[i][:temp][index] = out_errs[index, pᵢ+1]
            pᵢ += 2
        end

        # Extinction parameters
        param_maps.extinction[:tau_9_7][index] = out_params[index, pᵢ]
        param_errs.extinction[:tau_9_7][index] = out_errs[index, pᵢ]
        param_maps.extinction[:tau_ice][index] = out_params[index, pᵢ+1]
        param_errs.extinction[:tau_ice][index] = out_errs[index, pᵢ+1]
        param_maps.extinction[:tau_ch][index] = out_params[index, pᵢ+2]
        param_errs.extinction[:tau_ch][index] = out_errs[index, pᵢ+2]
        param_maps.extinction[:beta][index] = out_params[index, pᵢ+3]
        param_errs.extinction[:beta][index] = out_errs[index, pᵢ+3]
        pᵢ += 4

        if cube_fitter.fit_sil_emission
            # Hot dust parameters
            param_maps.hot_dust[:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ])-17 : -Inf
            param_errs.hot_dust[:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.hot_dust[:temp][index] = out_params[index, pᵢ+1]
            param_errs.hot_dust[:temp][index] = out_errs[index, pᵢ+1]
            param_maps.hot_dust[:frac][index] = out_params[index, pᵢ+2]
            param_errs.hot_dust[:frac][index] = out_errs[index, pᵢ+2]
            param_maps.hot_dust[:tau_warm][index] = out_params[index, pᵢ+3]
            param_errs.hot_dust[:tau_warm][index] = out_errs[index, pᵢ+3]
            param_maps.hot_dust[:tau_cold][index] = out_params[index, pᵢ+4]
            param_errs.hot_dust[:tau_cold][index] = out_errs[index, pᵢ+4]
            pᵢ += 5
        end

        # Dust feature log(amplitude), mean, FWHM
        for df ∈ cube_fitter.df_names
            param_maps.dust_features[df][:amp][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ])-17 : -Inf
            param_errs.dust_features[df][:amp][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.dust_features[df][:mean][index] = out_params[index, pᵢ+1]
            param_errs.dust_features[df][:mean][index] = out_errs[index, pᵢ+1]
            param_maps.dust_features[df][:fwhm][index] = out_params[index, pᵢ+2]
            param_errs.dust_features[df][:fwhm][index] = out_errs[index, pᵢ+2]
            pᵢ += 3
        end

        if cube_fitter.save_full_model
            # End of continuum parameters: recreate the continuum model
            I_cont, comps_c = Util.fit_full_continuum(cube_fitter.cube.λ, out_params[index, 1:pᵢ-1], cube_fitter.n_dust_cont, cube_fitter.n_dust_feat,
                cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.fit_sil_emission)
        end

        # Tied line kinematics
        vᵢ = pᵢ
        for vk ∈ cube_fitter.kin_tied_key
            param_maps.tied_voffs[vk][index] = out_params[index, pᵢ]
            param_errs.tied_voffs[vk][index] = out_errs[index, pᵢ]
            pᵢ += 1
        end
        for vk ∈ cube_fitter.kin_tied_key
            param_maps.tied_fwhms[vk][index] = out_params[index, pᵢ]
            param_errs.tied_fwhms[vk][index] = out_errs[index, pᵢ]
            pᵢ += 1
        end

        # Tied acomp kinematics
        for fvk ∈ cube_fitter.acomp_kin_tied_key
            param_maps.acomp_tied_voffs[fvk][index] = out_params[index, pᵢ]
            param_errs.acomp_tied_voffs[fvk][index] = out_errs[index, pᵢ]
            pᵢ += 1
        end
        for fvk ∈ cube_fitter.acomp_kin_tied_key
            param_maps.acomp_tied_fwhms[fvk][index] = out_params[index, pᵢ]
            param_errs.acomp_tied_fwhms[fvk][index] = out_errs[index, pᵢ]
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

            if isnothing(cube_fitter.line_tied[k])

                # Individual shift
                voff = out_params[index, pᵢ+1]
                voff_err = out_errs[index, pᵢ+1]
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
                pᵢ += 3

            elseif cube_fitter.flexible_wavesol
                voff = out_params[index, pᵢ+1]
                voff_err = out_errs[index, pᵢ+1]
                # If velocity is tied while flexible, add the overall shift and the individual shift together
                vwhere = findfirst(x -> x == cube_fitter.line_tied[k], cube_fitter.kin_tied_key)
                # voff += out_params[index, vᵢ+vwhere-1]
                # voff_err = √(voff_err^2 + out_errs[index, vᵢ+vwhere-1]^2)
                param_maps.lines[ln][:voff][index] = voff
                param_errs.lines[ln][:voff][index] = voff_err

                # FWHM -> subtract instrumental resolution in quadrature
                fwhm = out_params[index, vᵢ+cube_fitter.n_kin_tied+vwhere-1]
                fwhm_err = out_errs[index, vᵢ+cube_fitter.n_kin_tied+vwhere-1]
                pᵢ += 2

            else
                # Tied shift only
                vwhere = findfirst(x -> x == cube_fitter.line_tied[k], cube_fitter.kin_tied_key)
                voff = out_params[index, vᵢ+vwhere-1]
                voff_err = out_errs[index, vᵢ+vwhere-1]
                # FWHM -> subtract instrumental resolution in quadrature
                fwhm = out_params[index, vᵢ+cube_fitter.n_kin_tied+vwhere-1]
                fwhm_err = out_errs[index, vᵢ+cube_fitter.n_kin_tied+vwhere-1]
                pᵢ += 1

            end

            # Get Gauss-Hermite 3rd and 4th order moments
            if cube_fitter.line_profiles[k] == :GaussHermite
                param_maps.lines[ln][:h3][index] = out_params[index, pᵢ]
                param_errs.lines[ln][:h3][index] = out_errs[index, pᵢ]
                param_maps.lines[ln][:h4][index] = out_params[index, pᵢ+1]
                param_errs.lines[ln][:h4][index] = out_errs[index, pᵢ+1]
                pᵢ += 2
            elseif cube_fitter.line_profiles[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                param_maps.lines[k][:mixing][index] = out_params[index, pᵢ]
                param_errs.lines[k][:mixing][index] = out_errs[index, pᵢ]
                pᵢ += 1
            end

            if !isnothing(cube_fitter.line_acomp_profiles[k])

                param_maps.lines[ln][:acomp_amp][index] = amp * out_params[index, pᵢ]
                param_errs.lines[ln][:acomp_amp][index] = 
                    √((amp * out_params[index, pᵢ])^2 * ((amp_err / amp)^2 + (out_errs[index, pᵢ] / out_params[index, pᵢ])^2))

                if isnothing(cube_fitter.line_acomp_tied[k])
                    # note of caution: the output/saved/plotted values of acomp voffs are ALWAYS given relative to
                    # the main line's voff, NOT the rest wavelength of the line; same goes for the errors (which add in quadrature)
                    param_maps.lines[ln][:acomp_voff][index] = out_params[index, pᵢ+1]
                    param_errs.lines[ln][:acomp_voff][index] = out_errs[index, pᵢ+1]

                    # FWHM -> subtract instrumental resolution in quadrature
                    acomp_fwhm = fwhm * out_params[index, pᵢ+2]
                    acomp_fwhm_err = √(acomp_fwhm^2 * ((fwhm_err / fwhm)^2 + (out_errs[index, pᵢ+2] / out_params[index, pᵢ+2])^2))
                    if fwhm_res > acomp_fwhm
                        param_maps.lines[ln][:acomp_fwhm][index] = 0.
                        param_errs.lines[ln][:acomp_fwhm][index] = acomp_fwhm_err
                    else
                        param_maps.lines[ln][:acomp_fwhm][index] = √(acomp_fwhm^2 - fwhm_res^2)
                        param_errs.lines[ln][:acomp_fwhm][index] = acomp_fwhm / √(acomp_fwhm^2 - fwhm_res^2) * acomp_fwhm_err
                    end
                    pᵢ += 3
                else
                    pᵢ += 1
                end
                # Get Gauss-Hermite 3rd and 4th order moments
                if cube_fitter.line_acomp_profiles[k] == :GaussHermite
                    param_maps.lines[ln][:acomp_h3][index] = out_params[index, pᵢ]
                    param_errs.lines[ln][:acomp_h3][index] = out_errs[index, pᵢ]
                    param_maps.lines[ln][:acomp_h4][index] = out_params[index, pᵢ+1]
                    param_errs.lines[ln][:acomp_h4][index] = out_errs[index, pᵢ+1]
                    pᵢ += 2
                elseif cube_fitter.line_acomp_profiles[k] == :Voigt && !cube_fitter.tie_voigt_mixing
                    param_maps.lines[ln][:acomp_mixing][index] = out_params[index, pᵢ]
                    param_errs.lines[ln][:acomp_mixing][index] = out_errs[index, pᵢ]
                    pᵢ += 1
                end
            end

        end

        N = Float64(abs(nanmaximum(cube_fitter.cube.Iν[index, :])))
        N = N ≠ 0. ? N : 1.
        if cube_fitter.save_full_model
            # End of line parameters: recreate the un-extincted (intrinsic) line model
            I_line, comps_l = Util.fit_line_residuals(cube_fitter.cube.λ, out_params[index, vᵢ:pᵢ-1], cube_fitter.n_lines, cube_fitter.n_kin_tied,
                cube_fitter.kin_tied_key, cube_fitter.line_tied, cube_fitter.line_profiles, cube_fitter.n_acomp_kin_tied, cube_fitter.acomp_kin_tied_key,
                cube_fitter.line_acomp_tied, cube_fitter.line_acomp_profiles, [ln.λ₀ for ln ∈ cube_fitter.lines], 
                cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing, comps_c["extinction"], true)

            # Renormalize
            for comp ∈ keys(comps_l)
                # (dont include extinction correction here since it's already included in the fitted line amplitudes)
                comps_l[comp] .*= N
            end
            I_line .*= N
            
            # Combine the continuum and line models, which both have the extinction profile already applied to them 
            I_model = I_cont .+ I_line
            comps = merge(comps_c, comps_l)
        end

        # Dust feature intensity, EQW, and SNR, from calculate_extra_parameters
        for df ∈ cube_fitter.df_names
            param_maps.dust_features[df][:intI][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]) : -Inf
            param_errs.dust_features[df][:intI][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.dust_features[df][:eqw][index] = out_params[index, pᵢ+1]
            param_errs.dust_features[df][:eqw][index] = out_errs[index, pᵢ+1]
            param_maps.dust_features[df][:SNR][index] = out_params[index, pᵢ+2]
            pᵢ += 3
        end

        for (k, ln) ∈ enumerate(cube_fitter.line_names)
            # Convert amplitudes to the correct units, then take the log
            amp_norm = param_maps.lines[ln][:amp][index]
            amp_norm_err = param_errs.lines[ln][:amp][index]
            param_maps.lines[ln][:amp][index] = amp_norm > 0 ? log10(amp_norm * N)-17 : -Inf
            param_errs.lines[ln][:amp][index] = amp_norm > 0 ? amp_norm_err / (log(10) * amp_norm) : NaN

            # Line intensity, EQW, and SNR, from calculate_extra_parameters
            param_maps.lines[ln][:intI][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]) : -Inf
            param_errs.lines[ln][:intI][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / (log(10) * out_params[index, pᵢ]) : NaN
            param_maps.lines[ln][:eqw][index] = out_params[index, pᵢ+1]
            param_errs.lines[ln][:eqw][index] = out_errs[index, pᵢ+1]
            param_maps.lines[ln][:SNR][index] = out_params[index, pᵢ+2]

            if !isnothing(cube_fitter.line_acomp_profiles[k])
                acomp_amp_norm = param_maps.lines[ln][:acomp_amp][index]
                acomp_amp_norm_err = param_errs.lines[ln][:acomp_amp][index]
                param_maps.lines[ln][:acomp_amp][index] = acomp_amp_norm > 0 ? log10(acomp_amp_norm * N)-17 : -Inf
                param_errs.lines[ln][:acomp_amp][index] = acomp_amp_norm > 0 ? acomp_amp_norm_err / (log(10) * acomp_amp_norm) : NaN

                # Line intensity, EQW, and SNR from calculate_extra_parameters
                param_maps.lines[ln][:acomp_intI][index] = out_params[index, pᵢ] > 0. ? log10(out_params[index, pᵢ]) : -Inf
                param_errs.lines[ln][:acomp_intI][index] = out_params[index, pᵢ] > 0. ? out_errs[index, pᵢ] / (log(10) * out_params[index, pᵢ]) : NaN
                param_maps.lines[ln][:acomp_eqw][index] = out_params[index, pᵢ+1]
                param_errs.lines[ln][:acomp_eqw][index] = out_params[index, pᵢ+1]
                param_maps.lines[ln][:acomp_SNR][index] = out_params[index, pᵢ+2]

                pᵢ += 3
            end

            pᵢ += 3
        end

        # Reduced χ^2
        param_maps.reduced_χ2[index] = out_params[index, pᵢ]

        if cube_fitter.save_full_model
            # Set 3D model cube outputs
            cube_model.model[index, :] .= I_model
            cube_model.stellar[index, :] .= comps["stellar"]
            for i ∈ 1:cube_fitter.n_dust_cont
                cube_model.dust_continuum[index, :, i] .= comps["dust_cont_$i"]
            end
            for j ∈ 1:cube_fitter.n_dust_feat
                cube_model.dust_features[index, :, j] .= comps["dust_feat_$j"]
            end
            if cube_fitter.fit_sil_emission
                cube_model.hot_dust[index, :] .= comps["hot_dust"]
            end
            for k ∈ 1:cube_fitter.n_lines
                cube_model.lines[index, :, k] .= comps["line_$k"]
                if haskey(comps, "line_$(k)_acomp")
                    cube_model.lines[index, :, k] .+= comps["line_$(k)_acomp"]
                end
            end
            cube_model.extinction[index, :] .= comps["extinction"]
            cube_model.abs_ice[index, :] .= comps["abs_ice"]
            cube_model.abs_ch[index, :] .= comps["abs_ch"]
        end

        next!(prog)

    end

    # Subtract the average of the individual voffs from the tied voffs, based on the SNR, for each group
    if cube_fitter.flexible_wavesol
        @debug "Adjusting individual voffs due to the flexible_wavesol option"
        for vk ∈ cube_fitter.kin_tied_key
            indiv_voffs = nothing
            snrs = nothing
            # Loop through and create 3D arrays of the voffs and SNRs of each line in the tied kinematic group
            for (name, ln) ∈ zip(cube_fitter.line_names, cube_fitter.lines)
                if ln.tied == vk
                    if isnothing(indiv_voffs)
                        indiv_voffs = param_maps.lines[name][:voff]
                        snrs = param_maps.lines[name][:SNR]
                        continue
                    end
                    indiv_voffs = cat(indiv_voffs, param_maps.lines[name][:voff], dims=3)
                    snrs = cat(snrs, param_maps.lines[name][:SNR], dims=3)
                end
            end
            # Collapse the voff array into an average along the 3rd dimension, ignoring any with an SNR < 3
            if !isnothing(indiv_voffs) && !isnothing(snrs)
                indiv_voffs[snrs .< 3] .= NaN
                avg_offset = dropdims(nanmean(indiv_voffs, dims=3), dims=3)
                # Subtract the average offset from the individual voffs
                # (the goal is to have the average offset of the individual voffs be 0, relative to the tied voff)
                for (name, ln) ∈ zip(cube_fitter.line_names, cube_fitter.lines)
                    if ln.tied == vk
                        param_maps.lines[name][:voff] .-= avg_offset
                    end
                end
                # and add it to the tied voff
                param_maps.tied_voffs[vk] .+= avg_offset
            end
        end
    end

    if cube_fitter.plot_maps
        @info "===> Plotting parameter maps... <==="
        plot_parameter_maps(cube_fitter, param_maps)
    end

    if cube_fitter.save_fits
        @info "===> Writing FITS outputs... <==="
        write_fits(cube_fitter, cube_model, param_maps, param_errs)
    end

    if cube_fitter.make_movies
        @info "===> Writing MP4 movies... (this may take a while) <==="
        make_movie(cube_fitter, cube_model)
    end

    @info """\n
    #############################################################################
    ################################### Done!! ##################################
    #############################################################################
    """

    # Return the (potentially) modified cube_fitter object, along with the param maps/errs and cube model
    cube_fitter, param_maps, param_errs, cube_model

end


############################## OUTPUT / SAVING FUNCTIONS ####################################


"""
    plot_parameter_map(data, name, name_i, Ω, z, cosmo; snr_filter=snr_filter, snr_thresh=snr_thresh,
        cmap=cmap)

Plotting function for 2D parameter maps which are output by `fit_cube!`

# Arguments
- `data::Matrix{Float64}`: The 2D array of data to be plotted
- `name::String`: The name of the object whose fitting parameter is being plotted, i.e. "NGC_7469"
- `name_i::String`: The name of the individual parameter being plotted, i.e. "dust_features_PAH_5.24_amp"
- `Ω::Float64`: The solid angle subtended by each pixel, in steradians (used for angular scalebar)
- `z::Float64`: The redshift of the object (used for physical scalebar)
- `cosmo::Cosmology.AbstractCosmology`: The cosmology to use to calculate distance for the physical scalebar
- `python_wcs::PyObject`: The astropy WCS object used to project the maps onto RA/Dec space
- `snr_filter::Matrix{Float64}=Matrix{Float64}(undef,0,0)`: A 2D array of S/N values to
    be used to filter out certain spaxels from being plotted - must be the same size as `data` to filter
- `snr_thresh::Float64=3.`: The S/N threshold below which to cut out any spaxels using the values in snr_filter
- `cmap::Symbol=:cubehelix`: The colormap used in the plot, defaults to the cubehelix map
"""
function plot_parameter_map(data::Matrix{Float64}, name::String, name_i::String, Ω::Float64, z::Float64, 
    cosmo::Cosmology.AbstractCosmology, python_wcs::PyObject; snr_filter::Union{Nothing,Matrix{Float64}}=nothing, 
    snr_thresh::Float64=3., cmap::Symbol=:cubehelix)

    # I know this is ugly but I couldn't figure out a better way to do it lmao
    if occursin("amp", String(name_i))
        bunit = L"$\log_{10}(I / $ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1})$"
    elseif occursin("temp", String(name_i))
        bunit = L"$T$ (K)"
    elseif occursin("fwhm", String(name_i)) && occursin("PAH", String(name_i))
        bunit = L"FWHM ($\mu$m)"
    elseif occursin("fwhm", String(name_i)) && !occursin("PAH", String(name_i))
        bunit = L"FWHM (km s$^{-1}$)"
    elseif occursin("mean", String(name_i))
        bunit = L"$\mu$ ($\mu$m)"
    elseif occursin("voff", String(name_i))
        bunit = L"$v_{\rm off}$ (km s$^{-1}$)"
    elseif occursin("SNR", String(name_i))
        bunit = L"$S/N$"
    elseif occursin("tau", String(name_i))
        if occursin("warm", String(name_i))
            bunit = L"$\tau_{\rm warm}$"
        elseif occursin("cold", String(name_i))
            bunit = L"$\tau_{\rm cold}$"
        elseif occursin("ice", String(name_i))
            bunit = L"$\tau_{\rm ice}$"
        elseif occursin("ch", String(name_i))
            bunit = L"$\tau_{\rm CH}$"
        else
            bunit = L"$\tau_{9.7}$"
        end
    elseif occursin("intI", String(name_i))
        bunit = L"$\log_{10}(I /$ erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$)"
    elseif occursin("eqw", String(name_i))
        bunit = L"$W_{\rm eq}$ ($\mu$m)"
    elseif occursin("chi2", String(name_i))
        bunit = L"$\tilde{\chi}^2$"
    elseif occursin("h3", String(name_i))
        bunit = L"$h_3$"
    elseif occursin("h4", String(name_i))
        bunit = L"$h_4$"
    elseif occursin("mixing", String(name_i))
        bunit = L"$\eta$"
    elseif occursin("beta", String(name_i))
        bunit = L"$\beta$"
    elseif occursin("frac", String(name_i))
        bunit = L"$C_f$"
    end

    @debug "Plotting 2D map of $name_i with units $bunit"

    filtered = copy(data)
    # Convert Infs into NaNs
    filtered[.!isfinite.(filtered)] .= NaN
    # Filter out low SNR points
    if !isnothing(snr_filter)
        filtered[snr_filter .≤ snr_thresh] .= NaN
        @debug "Performing SNR filtering, $(sum(isfinite.(filtered)))/$(length(filtered)) passed"
    end
    # filter out insane/unphysical equivalent widths (due to ~0 continuum level)
    if occursin("eqw", String(name_i))
        filtered[filtered .> 100] .= NaN
    end
    if occursin("voff", String(name_i))
        # Perform a 5-sigma clip to remove outliers
        f_avg = nanmean(filtered)
        f_std = nanstd(filtered)
        filtered[abs.(filtered .- f_avg) .> 5f_std] .= NaN
    end

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=python_wcs)
    # Need to filter out any NaNs in order to use quantile
    vmin = nanminimum(filtered)
    vmax = nanmaximum(filtered)
    # override vmin/vmax for mixing parameter
    if occursin("mixing", String(name_i))
        vmin = 0.
        vmax = 1.
    end
    # if taking a voff, make sure vmin/vmax are symmetric and change the colormap to coolwarm
    if occursin("voff", String(name_i))
        vabs = max(abs(vmin), abs(vmax))
        vmin = -vabs
        vmax = vabs
        if cmap == :cubehelix
            cmap = :coolwarm
        end
    end
    # default cmap is magma for FWHMs and equivalent widths
    if (occursin("fwhm", String(name_i)) || occursin("eqw", String(name_i))) && cmap == :cubehelix
        cmap = :magma
    end
    cdata = ax.imshow(filtered', origin=:lower, cmap=cmap, vmin=vmin, vmax=vmax)
    # ax.axis(:off)
    ax.tick_params(which="both", axis="both", direction="in")
    ax.set_xlabel("R.A.")
    ax.set_ylabel("Dec.")

    # Angular and physical scalebars
    n_pix = 1/(sqrt(Ω) * 180/π * 3600)
    @debug "Using angular diameter distance $(angular_diameter_dist(cosmo, z))"
    # Calculate in Mpc
    dA = angular_diameter_dist(u"pc", cosmo, z)
    # Remove units
    dA = uconvert(NoUnits, dA/u"pc")
    l = dA * π/180 / 3600  # l = d * theta (1")
    # Round to a nice even number
    l = Int(round(l, sigdigits=1))
     # new angular size for this scale
    θ = l / dA
    n_pix = 1/sqrt(Ω) * θ   # number of pixels = (pixels per radian) * radians
    if cosmo.h == 1.0
        scalebar = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, L"%$l$h^{-1}$ pc", "lower left", pad=1, color=:black, 
            frameon=false, size_vertical=0.4, label_top=false)
    else
        scalebar = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, L"%$l pc", "lower left", pad=1, color=:black,
            frameon=false, size_vertical=0.4, label_top=false)
    end
    ax.add_artist(scalebar)

    fig.colorbar(cdata, ax=ax, label=bunit)
    plt.savefig(joinpath("output_$(name)", "param_maps", "$(name_i).pdf"), dpi=300, bbox_inches=:tight)
    plt.close()

    return

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

    # Stellar continuum parameters
    for parameter ∈ keys(param_maps.stellar_continuum)
        data = param_maps.stellar_continuum[parameter]
        name_i = join(["stellar_continuum", parameter], "_")
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
            cube_fitter.python_wcs)
    end

    # Dust continuum parameters
    for i ∈ keys(param_maps.dust_continuum)
        for parameter ∈ keys(param_maps.dust_continuum[i])
            data = param_maps.dust_continuum[i][parameter]
            name_i = join(["dust_continuum", i, parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
                cube_fitter.python_wcs)
        end
    end

    # Dust feature (PAH) parameters
    for df ∈ keys(param_maps.dust_features)
        snr = param_maps.dust_features[df][:SNR]
        for parameter ∈ keys(param_maps.dust_features[df])
            data = param_maps.dust_features[df][parameter]
            name_i = join(["dust_features", df, parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
                cube_fitter.python_wcs, snr_filter=parameter !== :SNR ? snr : nothing, snr_thresh=snr_thresh)
        end
    end

    # Extinction parameters
    for parameter ∈ keys(param_maps.extinction)
        data = param_maps.extinction[parameter]
        name_i = join(["extinction", parameter], "_")
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
            cube_fitter.python_wcs)
    end

    if cube_fitter.fit_sil_emission
        # Hot dust parameters
        for parameter ∈ keys(param_maps.hot_dust)
            data = param_maps.hot_dust[parameter]
            name_i = join(["hot_dust", parameter], "_")
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
                cube_fitter.python_wcs)
        end
    end

    # Tied Voigt mixing parameter
    # if cube_fitter.tie_voigt_mixing
    #     data = param_maps.tied_voigt_mix
    #     name_i = "tied_voigt_mixing"
    #     plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
    #         cube_fitter.python_wcs)
    # end

    # Tied kinematics
    for vk ∈ cube_fitter.kin_tied_key
        data = param_maps.tied_voffs[vk]
        name_i = join(["tied_voffs", vk], "_")
        # Get the SNRs of each line in the tied kinematic group
        snr = nothing
        for (name, line) ∈ zip(cube_fitter.line_names, cube_fitter.lines)
            if line.tied == vk
                if isnothing(snr)
                    snr = param_maps.lines[name][:SNR]
                    continue
                end
                snr = cat(snr, param_maps.lines[name][:SNR], dims=3)
            end
        end
        # Take the maximum SNR along the 3rd axis
        if ndims(snr) == 3
            snr = dropdims(nanmaximum(snr, dims=3), dims=3)
        end
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
            cube_fitter.python_wcs, snr_filter=snr, snr_thresh=snr_thresh)
    end
    for vk ∈ cube_fitter.kin_tied_key
        data = param_maps.tied_fwhms[vk]
        name_i = join(["tied_fwhms", vk], "_")
        # Get the SNRs of each line in the tied kinematic group
        snr = nothing
        for (name, line) ∈ zip(cube_fitter.line_names, cube_fitter.lines)
            if line.tied == vk
                if isnothing(snr)
                    snr = param_maps.lines[name][:SNR]
                    continue
                end
                snr = cat(snr, param_maps.lines[name][:SNR], dims=3)
            end
        end
        # Take the maximum SNR along the 3rd axis
        if ndims(snr) == 3
            snr = dropdims(nanmaximum(snr, dims=3), dims=3)
        end
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
            cube_fitter.python_wcs, snr_filter=snr, snr_thresh=snr_thresh)
    end

    # Tied acomp kinematics
    for fvk ∈ cube_fitter.acomp_kin_tied_key
        data = param_maps.acomp_tied_voffs[fvk]
        name_i = join(["acomp_tied_voffs", fvk], "_")
        # Get the SNRs of each line in the tied kinematic group
        snr = nothing
        for (name, line) ∈ zip(cube_fitter.line_names, cube_fitter.lines)
            if line.acomp_tied == fvk
                if isnothing(snr)
                    snr = param_maps.lines[name][:acomp_SNR]
                    continue
                end
                snr = cat(snr, param_maps.lines[name][:acomp_SNR], dims=3)
            end
        end
        # Take the maximum SNR along the 3rd axis
        if ndims(snr) == 3
            snr = dropdims(nanmaximum(snr, dims=3), dims=3)
        end
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
            cube_fitter.python_wcs, snr_filter=snr, snr_thresh=snr_thresh)
    end
    for fvk ∈ cube_fitter.acomp_kin_tied_key
        data = param_maps.acomp_tied_fwhms[fvk]
        name_i = join(["acomp_tied_fwhms", fvk], "_")
        # Get the SNRs of each line in the tied kinematic group
        snr = nothing
        for (name, line) ∈ zip(cube_fitter.line_names, cube_fitter.lines)
            if line.acomp_tied == fvk
                if isnothing(snr)
                    snr = param_maps.lines[name][:acomp_SNR]
                    continue
                end
                snr = cat(snr, param_maps.lines[name][:acomp_SNR], dims=3)
            end
        end
        # Take the maximum SNR along the 3rd axis
        if ndims(snr) == 3
            snr = dropdims(nanmaximum(snr, dims=3), dims=3)
        end
        plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
            cube_fitter.python_wcs, snr_filter=snr, snr_thresh=snr_thresh)
    end

    # Line parameters
    for line ∈ keys(param_maps.lines)
        snr = param_maps.lines[line][:SNR]
        asnr = nothing
        if haskey(param_maps.lines[line], :acomp_SNR)
            asnr = param_maps.lines[line][:acomp_SNR]
        end
        for parameter ∈ keys(param_maps.lines[line])
            data = param_maps.lines[line][parameter]
            name_i = join(["lines", line, parameter], "_")
            snr_filter = snr
            if contains(String(parameter), "SNR")
                snr_filter = nothing
            elseif contains(String(parameter), "acomp")
                snr_filter = asnr
            end
            plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
                cube_fitter.python_wcs, snr_filter=snr_filter, snr_thresh=snr_thresh)
        end
    end

    # Reduced chi^2 
    data = param_maps.reduced_χ2
    name_i = "reduced_chi2"
    plot_parameter_map(data, cube_fitter.name, name_i, cube_fitter.cube.Ω, cube_fitter.z, cube_fitter.cosmology,
        cube_fitter.python_wcs)

    return

end


"""
    make_movie(cube_fitter, cube_model; cmap=cmap)

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

    for (full_data, title) ∈ zip([cube_fitter.cube.Iν, cube_model.model], ["DATA", "MODEL"])

        # Writer using FFMpeg to create an mp4 file
        metadata = Dict(:title => title, :artist => "LOKI", :fps => 60)
        writer = py_animation.FFMpegWriter(fps=60, metadata=metadata)

        # Set up plots with gridspec
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=20,  nrows=10)
        ax1 = fig.add_subplot(py"$(gs)[0:8, 0:18]", projection=cube_fitter.python_wcs)
        ax2 = fig.add_subplot(py"$(gs)[9:10, :]")
        ax3 = fig.add_subplot(py"$(gs)[0:8, 18:19]")

        # First wavelength slice of the model
        wave_rest = cube_fitter.cube.λ
        data = full_data[:, :, 1]

        # Get average along the wavelength dimension
        datasum = Util.Σ(full_data, 3)
        dataavg = datasum ./ size(full_data, 3)
        flatavg = dataavg[isfinite.(dataavg)]

        # Plot the first slice
        image = ax1.imshow(data', origin=:lower, cmap=cmap, vmin=quantile(flatavg, 0.01), vmax=quantile(flatavg, 0.99))
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
        for i ∈ 1:size(full_data, 3)
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

    if cube_fitter.save_full_model
        # Create the 3D intensity model FITS file
        FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_3D_model.fits"), "w") do f

            @debug "Writing 3D model FITS HDUs"

            write(f, Vector{Int}())                                                                     # Primary HDU (empty)
            write(f, cube_fitter.cube.Iν; header=hdr, name="DATA")                                      # Raw data with nans inserted
            write(f, cube_model.model; header=hdr, name="MODEL")                                        # Full intensity model
            write(f, cube_fitter.cube.Iν .- cube_model.model; header=hdr, name="RESIDUALS")             # Residuals (data - model)
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
            write(f, cube_model.abs_ice; header=hdr, name="ABS_ICE")                                    # Ice Absorption model
            write(f, cube_model.abs_ch; header=hdr, name="ABS_CH")                                      # CH Absorption model
            if cube_fitter.fit_sil_emission
                write(f, cube_model.hot_dust; header=hdr, name="HOT_DUST")                              # Hot dust model
            end
            
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
            write_key(f["ABS_ICE"], "BUNIT", "unitless")
            write_key(f["ABS_CH"], "BUNIT", "unitless")
            if cube_fitter.fit_sil_emission
                write_key(f["HOT_DUST"], "BUNIT", "MJy/sr")
            end
        end
    end

    # Create the 2D parameter map FITS file
    FITS(joinpath("output_$(cube_fitter.name)", "$(cube_fitter.name)_parameter_maps.fits"), "w") do f

        @debug "Writing 2D parameter map FITS HDUs"

        write(f, Vector{Int}())  # Primary HDU (empty)

        # Iterate over model parameters and make 2D maps

        # Stellar continuum parameters
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

        # Dust continuum parameters
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

        if cube_fitter.fit_sil_emission
            # Hot dust parameters
            for parameter ∈ keys(param_maps.hot_dust)
                data = param_maps.hot_dust[parameter]
                name_i = join(["hot_dust", parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                elseif occursin("temp", String(name_i))
                    bunit = "Kelvin"
                elseif occursin("frac", String(name_i)) || occursin("tau", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)
            end
            for parameter ∈ keys(param_errs.hot_dust)
                data = param_errs.hot_dust[parameter]
                name_i = join(["hot_dust", parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                elseif occursin("temp", String(name_i))
                    bunit = "Kelvin"
                elseif occursin("frac", String(name_i)) || occursin("tau", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)
            end
        end

        # Dust feature (PAH) parameters
        for df ∈ keys(param_maps.dust_features)
            for parameter ∈ keys(param_maps.dust_features[df])
                data = param_maps.dust_features[df][parameter]
                name_i = join(["dust_features", df, parameter], "_")
                if occursin("amp", String(name_i))
                    bunit = "log10(I / erg s^-1 cm^-2 Hz^-1 sr^-1)"
                elseif occursin("fwhm", String(name_i)) || occursin("mean", String(name_i)) || occursin("eqw", String(name_i))
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
                elseif occursin("fwhm", String(name_i)) || occursin("mean", String(name_i)) || occursin("eqw", String(name_i))
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

        # Line parameters
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
                elseif occursin("eqw", String(name_i))
                    bunit = "um"
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
                elseif occursin("eqw", String(name_i))
                    bunit = "um"
                elseif occursin("SNR", String(name_i)) || occursin("h3", String(name_i)) || 
                    occursin("h4", String(name_i)) || occursin("mixing", String(name_i))
                    bunit = "unitless"
                end
                write(f, data; header=hdr, name=name_i)
                write_key(f[name_i], "BUNIT", bunit)   
            end
        end

        # Extinction parameters
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

        # Tied Voigt mixing parameter
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

        # Tied velocity offsets
        for vk ∈ cube_fitter.kin_tied_key
            data = param_maps.tied_voffs[vk]
            name_i = join(["tied_voffs", vk], "_")
            bunit = "km/s"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end
        for vk ∈ cube_fitter.kin_tied_key
            data = param_errs.tied_voffs[vk]
            name_i = join(["tied_voffs", vk, "err"], "_")
            bunit = "km/s"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end

        # Tied acomp velocity offsets
        for fvk ∈ cube_fitter.acomp_kin_tied_key
            data = param_maps.acomp_tied_voffs[fvk]
            name_i = join(["acomp_tied_voffs", fvk], "_")
            bunit = "km/s"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end
        for fvk ∈ cube_fitter.acomp_kin_tied_key
            data = param_errs.acomp_tied_voffs[fvk]
            name_i = join(["acomp_tied_voffs", fvk, "err"], "_")
            bunit = "km/s"
            write(f, data; header=hdr, name=name_i)
            write_key(f[name_i], "BUNIT", bunit)
        end

        # Reduced chi^2
        data = param_maps.reduced_χ2
        name_i = "reduced_chi2"
        bunit = "unitless"
        write(f, data; header=hdr, name=name_i)
        write_key(f[name_i], "BUNIT", bunit)

    end
end

end
