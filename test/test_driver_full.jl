using Distributions
using Interpolations
using ProgressMeter
using Distributed
using PlotlyJS
using DataFrames
using CSV
using Printf
using PyPlot
using PyCall

addprocs(maximum([0, Sys.CPU_THREADS ÷ 2]))
@everywhere begin
    using Pkg; Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate(); Pkg.precompile()
end
@everywhere using Loki

py_anchored_artists = pyimport("mpl_toolkits.axes_grid1.anchored_artists")

# MATPLOTLIB SETTINGS TO MAKE PLOTS LOOK PRETTY :)
SMALL = 12
MED = 14
BIG = 16

plt.rc("font", size=MED)          # controls default text sizes
plt.rc("axes", titlesize=MED)     # fontsize of the axes title
plt.rc("axes", labelsize=MED)     # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL)  # fontsize of the tick labels
plt.rc("legend", fontsize=MED)    # legend fontsize
plt.rc("figure", titlesize=BIG)   # fontsize of the figure title
plt.rc("text", usetex=true)
plt.rc("font", family="Times New Roman")


# Load in data
# obs = from_fits(["data/jw01328-o015_t014_miri_ch1-mediumshortlong-_s3d.fits", 
#     "data/jw01328-o015_t014_miri_ch2-mediumshortlong-_s3d.fits", 
#     "data/jw01328-o015_t014_miri_ch3-mediumshortlong-_s3d.fits", 
#     "data/jw01328-o015_t014_miri_ch4-mediumshortlong-_s3d.fits"], 
#     0.016317)

obs = from_fits(["data/jw01039-o005_t001_miri_ch1-shortlongmedium-_s3d.fits",
    "data/jw01039-o005_t001_miri_ch2-shortlongmedium-_s3d.fits",
    "data/jw01039-o005_t001_miri_ch3-shortlongmedium-_s3d.fits",
    "data/jw01039-o005_t001_miri_ch4-shortlongmedium-_s3d.fits"],
    0.0266)

obs = correct(obs)

# Create the cube fitting object
cube_fitter = CubeFitter(obs.channels[2], obs.z, obs.name * "_ch2_test_flex"; parallel=true, plot_spaxels=:pyplot,
    plot_maps=true, save_fits=true)

# Perform the Levenberg-Marquardt least-squares fitting
cube_fitter = @time fit_cube(cube_fitter)
