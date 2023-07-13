module Loki

# Importing all of the dependencies

# Parallel computing packages
using Distributed

# Math packages
using Distributions
using Random
using Statistics
using NaNStatistics
using QuadGK
using NumericalIntegration
using Dierckx
using LinearAlgebra
using EllipsisNotation
using SpecialFunctions
using FFTW
using Polynomials

# Optimization packages
using Optim
using CMPFit

# Astronomy packages
using FITSIO
using Cosmology
using AstroAngles
using Unitful
using UnitfulAstro

# File I/O
using TOML
using DelimitedFiles
using CSV
using Serialization
using DataFrames

# Plotting packages
using PlotlyJS

# Misc packages/utilites
using ProgressMeter
using Printf
using Logging
using LoggingExtras
using Dates
using InteractiveUtils
using ColorSchemes
using LaTeXStrings
using Pipe

# PyCall needed for some matplotlib modules
using PyCall

# Warnings
const py_warnings::PyObject = PyNULL()

# Matplotlib modules
const plt::PyObject = PyNULL()
const py_anchored_artists::PyObject = PyNULL()
const py_ticker::PyObject = PyNULL()
const py_colormap::PyObject = PyNULL()
const py_animation::PyObject = PyNULL()

# Astropy modules
const py_wcs::PyObject = PyNULL()
const py_coords::PyObject = PyNULL()
const py_units::PyObject = PyNULL()
const py_photutils::PyObject = PyNULL()
const py_reproject::PyObject = PyNULL()
const py_mosaicking::PyObject = PyNULL()
const py_lineidplot::PyObject = PyNULL()

# FSPS Library
const py_fsps::PyObject = PyNULL()

# Some constants for setting matplotlib font sizes
const SMALL::UInt8 = 12
const MED::UInt8 = 14
const BIG::UInt8 = 16

# Date format for the log files
const date_format::String = "yyyy-mm-dd HH:MM:SS"

# This lock is used to control write access to the convergence log file, since
# multiple parallel processes may try to write to it at once.
const file_lock::ReentrantLock = ReentrantLock()

# Have to import certain python modules within the __init__ function so that it works after precompilation,
# so these PyNULL constants are just placeholders before that happens
function __init__()

    # Import pyplot
    copy!(plt, pyimport_conda("matplotlib.pyplot", "matplotlib"))

    # Import matplotlib submodules for nice plots
    # anchored_artists --> used for scale bars
    copy!(py_anchored_artists, pyimport_conda("mpl_toolkits.axes_grid1.anchored_artists", "matplotlib"))
    # ticker --> used for formatting axis ticks and tick labels
    copy!(py_ticker, pyimport_conda("matplotlib.ticker", "matplotlib"))
    # cm --> used for formatting matplotlib colormaps
    copy!(py_colormap, pyimport_conda("matplotlib.cm", "matplotlib"))
    # animation --> used for making mp4 movie files (optional)
    copy!(py_animation, pyimport_conda("matplotlib.animation", "matplotlib"))
    # python package for adjusting matplotlib text so it doesn't overlap
    copy!(py_lineidplot, pyimport_conda("lineid_plot", "lineid_plot"))

    # Import the WCS, photutils, and reproject packages from astropy
    copy!(py_wcs, pyimport_conda("astropy.wcs", "astropy"))
    copy!(py_coords, pyimport_conda("astropy.coordinates", "astropy"))
    copy!(py_units, pyimport_conda("astropy.units", "astropy"))
    copy!(py_photutils, pyimport_conda("photutils", "photutils"))
    copy!(py_reproject, pyimport_conda("reproject", "reproject"))
    copy!(py_mosaicking, pyimport_conda("reproject.mosaicking", "reproject"))

    # Warnings
    copy!(py_warnings, pyimport_conda("warnings", "warnings"))

    try
        copy!(py_fsps, pyimport_conda("fsps", "fsps"))
    catch
        @warn "Could not find the Python FSPS Library! Optical spectra modeling will not be possible."
    end

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

    # Filter annoying FITS fixed warnings from astropy
    py_warnings.filterwarnings("ignore", category=py_wcs.FITSFixedWarning)

    ###### SETTING UP A GLOBAL LOGGER ######

    # Append timestamps to all logging messages
    timestamp_logger(logger) = TransformerLogger(logger) do log
        merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
    end
    # Create a tee logger that writes both to the stdout and to a log file
    logger = TeeLogger(ConsoleLogger(stdout, Logging.Info), 
                        timestamp_logger(MinLevelLogger(FileLogger(joinpath(@__DIR__, "loki.main.log"); 
                                                                    always_flush=true), 
                                                                    Logging.Debug)))
    # Initialize our logger as the global logger
    global_logger(logger)

end


export DataCube,   # DataCube struct

       # DataCube Functions
       from_fits, 
       to_rest_frame!, 
       apply_mask!, 
       to_vacuum_wavelength!,
       log_rebin!,
       correct!, 
       interpolate_nans!, 
       plot_2d, 
       plot_1d,
       make_aperture,

       # Observation struct
       Observation, 

       # Observation functions
       save_fits,
       adjust_wcs_alignment!, 
       reproject_channels!, 
       # psf_kernel, 
       convolve_psf!,

       # Parameter structs
       Parameter,
       Continuum,
       DustFeatures,
       TransitionLines,
       TiedKinematics,

       # Parameter functions
       from_dict,
       from_dict_fwhm,
       from_dict_wave,

       # ParamMaps and CubeModel structs
       ParamMaps,
       CubeModel,

       # initialization functions
       parammaps_empty,
       cubemodel_empty,

       # CubeFitter struct
       CubeFitter,

       # CubeFitter-related functions
       generate_cubemodel,
       generate_parammaps,
       mask_emission_lines,
       continuum_cubic_spline,
       fit_spaxel,
       fit_stack!,
       fit_cube!,
       plot_parameter_maps,
       make_movie,
       write_fits,

       # Utility functions that the user may wish to take advantage of
       frebin,
       fshift,
       make_python_wcs,
       attenuation_calzetti,
       attenuation_cardelli,  # (AKA ccm_unred)
       resample_conserving_flux,
       air_to_vacuum,
       extend,
       sumdim,
       Doppler_shift_v,
       Doppler_shift_λ,
       Doppler_width_v,
       Doppler_width_λ

# Include all of the files that we need to create the module

include("util/parameters.jl")
include("util/parsing.jl")
include("util/math.jl")

include("core/cubedata.jl")
include("core/cubefit.jl")
include("core/fitting.jl")
include("core/output.jl")

#####################
#= WELCOME TO LOKI =#
#####################

# Namespace:
# First-class => most of cubefit.jl and cubedata.jl
# "Loki" => utils.jl, parameters.jl

end