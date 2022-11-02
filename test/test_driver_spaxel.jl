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

# addprocs(maximum([0, Sys.CPU_THREADS ÷ 2 - 1]))
# # addprocs(9)
# @everywhere begin
#     using Pkg; Pkg.activate(dirname(@__DIR__))
#     Pkg.instantiate(); Pkg.precompile()
# end
# @everywhere using Loki
using Pkg; Pkg.activate(dirname(@__DIR__))
Pkg.instantiate(); Pkg.precompile()
using Loki

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

# Fit a spaxel
x = [28, 25, 10]
y = [12, 20, 14]
for (xi, yi) ∈ zip(x, y)
    # Pick a spaxel to fit
    λ = obs.channels[2].λ
    I = obs.channels[2].Iλ[xi, yi, :]
    σ = obs.channels[2].σI[xi, yi, :]
    CubeFit.levmar_fit_spaxel(λ, I, σ; plot=:plotly, name="test", label="spaxel_$(xi)_$(yi)")
end
