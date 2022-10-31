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

addprocs(maximum([0, Sys.CPU_THREADS ÷ 2 - 1]))
# addprocs(9)
@everywhere begin
    using Pkg; Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate(); Pkg.precompile()
end
@everywhere using Loki
# using Pkg; Pkg.activate(dirname(@__DIR__))
# Pkg.instantiate(); Pkg.precompile()
# using Loki

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
obs = from_fits(["jw01328-o015_t014_miri_ch1-mediumshortlong-_s3d.fits", 
    "jw01328-o015_t014_miri_ch2-mediumshortlong-_s3d.fits", 
    "jw01328-o015_t014_miri_ch3-mediumshortlong-_s3d.fits", 
    "jw01328-o015_t014_miri_ch4-mediumshortlong-_s3d.fits"], 
    0.016317)

obs = correct(obs)
# plot_2d(obs.channels[1], "test.pdf")

# spax = Dict(1 => (19,25), 2 => (21,19), 3 => (22,25), 4 => (15,12))
# x, y = spax[1]

# # Pick a spaxel to fit
# λ = obs.channels[1].λ
# I = obs.channels[1].Iλ[x, y, :]
# σ = obs.channels[1].σI[x, y, :]

# Fit a spaxel
# CubeFit.continuum_fit_spaxel(λ, I, σ; plot=:plotly)

# Fit the cube
param_maps, I_model = @time fit_cube(obs.channels[2]; parallel=true, 
    plot_continua=:pyplot, plot_lines=:none, plot_maps=true, name=obs.name * "_ch2")
