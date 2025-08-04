module Loki

# Importing all of the dependencies

# Debugging
using Infiltrator

# Parallel computing packages
using Distributed
using SharedArrays

# Math packages
using Distributions
using Random
using StatsBase, Statistics, NaNStatistics
using NumericalIntegration
using Dierckx
using LinearAlgebra
using FFTW
using SpecialFunctions
using NLsolve
using ImageFiltering
using ImageTransformations

# Optimization packages
using Optim, CMPFit, NonNegLeastSquares

# Astronomy packages
using AstroLib
using FITSIO
using Photometry
using Cosmology
using AstroAngles
using SkyCoords
using WCS
using DustExtinction
using Reproject
using SpectralResampling
using VoronoiBinning

# File I/O
using Glob
using DelimitedFiles
using CSV
using TOML
using Serialization
using DataFrames

# Plotting packages
using PlotlyJS

# Misc packages/utilites
using ProgressMeter
using Printf
using Logging, LoggingExtras
using Dates
using InteractiveUtils
using ColorSchemes
using LaTeXStrings, Latexify
using Unitful, UnitfulAstro, UnitfulLatexify, UnitfulParsableString
using ArgParse

# PyCall needed for some matplotlib modules
using PyCall, Conda

# Matplotlib modules
const plt::PyObject = PyNULL()
const py_anchored_artists::PyObject = PyNULL()
const py_ticker::PyObject = PyNULL()
const py_colormap::PyObject = PyNULL()
const py_colors::PyObject = PyNULL()
const py_animation::PyObject = PyNULL()

# Line ID plot package
const py_lineidplot::PyObject = PyNULL()

# FSPS Library
const py_fsps::PyObject = PyNULL()

# Astroquery SVO FPS module 
const py_svo::PyObject = PyNULL()

# Some constants for setting matplotlib font sizes
const SMALL::UInt8 = 12
const MED::UInt8 = 14
const BIG::UInt8 = 16

# Date format for the log files
const date_format::String = "yyyy-mm-dd HH:MM:SS"

# This lock is used to control write access to the convergence log file, since
# multiple parallel processes may try to write to it at once.
const file_lock::ReentrantLock = ReentrantLock()

# MIRI channel boundaries
const channel_boundaries = [5.7, 6.58, 7.58, 8.72, 10.075, 11.625, 13.39, 15.49, 17.84, 20.82, 24.335] .* u"μm"
const channel_edges = [4.90, 5.74, 5.66, 6.63, 6.53, 7.65, 7.51, 8.77, 8.67, 10.13, 10.02, 11.70, 11.55, 
                       13.47, 13.34, 15.57, 15.41, 17.98, 17.70, 20.95, 20.69, 24.48, 24.40, 27.90] .* u"μm"

# NIRSPEC channel boundaries
const channel_boundaries_nir = [0.97, 1.775, 3.02] .* u"μm"
const channel_edges_nir = [0.9, 1.27, 0.97, 1.89, 1.66, 3.17, 2.87, 5.27] .* u"μm"
const chip_gaps_nir = [(1.40780, 1.48580) .* u"μm", (2.36067, 2.49153) .* u"μm", (3.98276, 4.20323) .* u"μm"]

# Have to import certain python modules within the __init__ function so that it works after precompilation,
# so these PyNULL constants are just placeholders before that happens
function __init__()

    # Import pyplot
    try
        copy!(plt, pyimport("matplotlib.pyplot"))
    catch
        Conda.pip_interop(true)
        Conda.pip("install", "matplotlib")
        copy!(plt, pyimport("matplotlib.pyplot"))
    end

    # Import matplotlib submodules for nice plots
    # anchored_artists --> used for scale bars
    copy!(py_anchored_artists, pyimport("mpl_toolkits.axes_grid1.anchored_artists"))
    # ticker --> used for formatting axis ticks and tick labels
    copy!(py_ticker, pyimport("matplotlib.ticker"))
    # cm --> used for formatting matplotlib colormaps
    copy!(py_colormap, pyimport("matplotlib.cm"))
    copy!(py_colors, pyimport("matplotlib.colors"))
    # animation --> used for making mp4 movie files (optional)
    copy!(py_animation, pyimport("matplotlib.animation"))
    # non-conda packages:
    try
        copy!(py_lineidplot, pyimport("lineid_plot"))
    catch
        Conda.pip_interop(true)
        Conda.pip("install", "lineid_plot")
        copy!(py_lineidplot, pyimport("lineid_plot"))
    end
    # FSPS
    try
        copy!(py_fsps, pyimport("fsps"))
    catch
        @warn "Could not find the Python FSPS Library! Optical spectra modeling will not be possible."
        # Conda.pip_interop(true)
        # Conda.pip("install", "fsps")
        # copy!(py_fsps, pyimport("fsps"))
    end
    # Astroquery
    try
        copy!(py_svo, pyimport("astroquery.svo_fps"))
    catch
        # @warn "Could not find the Python astroquery library! Photometry modeling will not be possible."
        # Conda.pip_interop(true)
        # Conda.pip("install", "astroquery")
        # copy!(py_astroquery, pyimport("astroquery"))
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
    # plt.rc("text", usetex=true)                # use LaTeX for things like axis labels
    # plt.rc("text.latex", preamble="\\usepackage{siunitx}")   # use the siunitx LaTeX package
    # plt.rc("font", family="Times New Roman")   # use Times New Roman font

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
       from_data,
       from_cubes,
       to_rest_frame!, 
       apply_mask!, 
       to_vacuum_wavelength!,
       log_rebin!,
       deredden!,
       correct!, 
       interpolate_nans!, 
       calculate_statistical_errors!,
       rotate_to_sky_axes!,
       voronoi_rebin!,
       plot_2d, 
       plot_1d,

       # Observation struct
       Observation, 

       # Observation functions
       save_fits,
       generate_psf_model!,
       splinefit_psf_model!,
       adjust_wcs_alignment!, 
       reproject_channels!, 
       extract_from_aperture!,
       resample_channel_wavelengths!,
       combine_channels!,
       generate_nuclear_template,

       # ParamMaps and CubeModel structs
       ParamMaps,
       CubeModel,
       get_val,
       get_err,
       get_label,

       # CubeFitter struct
       CubeFitter,

       # CubeFitter-related functions
       generate_cubemodel,
       generate_parammaps,
       fit_spaxel,
       fit_stack!,
       fit_cube!,
       post_fit_nuclear_template!,
       plot_parameter_map,
       plot_parameter_maps,
       make_movie,
       write_fits,

       # Spaxel-related objects
       Spaxel,

       make_normalized_spaxel,
       continuum_cubic_spline!,
       calculate_statistical_errors!,
       get_vector_mask,
       fill_bad_pixels!,
       subtract_continuum!,
       normalize!,

       SpaxelFitResult,
       combine!,
       pretty_print_results,

       # Utility functions that the user may wish to take advantage of
       get_area,
       get_patches,
       centroid_com,
       make_aperture,
       resample_conserving_flux,
       extend,
       sumdim,
       Doppler_shift_v,
       Doppler_shift_λ,
       Doppler_width_v,
       Doppler_width_λ,
       evaluate_model

# Include all of the files that we need to create the module

include("util/parameters.jl")
include("core/cubedata.jl")
include("core/cubefit.jl")
include("core/cubefit_helpers.jl")
include("core/parammaps.jl")
include("core/cubemodel.jl")
include("core/spaxelresult.jl")
include("core/spaxel.jl")

include("util/parsing.jl")
include("util/create_params.jl")
include("util/math.jl")
include("core/model.jl")

include("core/fitdata.jl")
include("core/fitplot.jl")
include("core/fitting.jl")

include("core/output_sorters.jl")
include("core/output.jl")

include("core/psf.jl")
include("util/aperture_utils.jl")

end
