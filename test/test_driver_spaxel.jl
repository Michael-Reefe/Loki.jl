using Distributions
using Interpolations
using Dierckx
using ProgressMeter
using Distributed
using PlotlyJS
using DataFrames
using CSV
using Printf
using PyPlot
using PyCall

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

# Create the cube fitting object
cube_fitter = CubeFitter(obs.channels[2], obs.z, "test"; parallel=false, plot_spaxels=:plotly,
    plot_maps=true, save_fits=true)
interpolate_cube!(cube_fitter.cube)

# Fit some individual spaxels
x = [28, 33, 25, 10, 6]
y = [12, 4, 19, 14, 17]
for (xi, yi) ∈ zip(x, y)

    λ = cube_fitter.cube.λ
    I = cube_fitter.cube.Iλ[xi, yi, :]
    σ = cube_fitter.cube.σI[xi, yi, :]

    # Check emission line masking
    mask, Icube, σcube = CubeFit.continuum_cubic_spline(λ, I, σ)
    Imask = copy(I)
    Imask[mask] .= NaN

    trace1 = PlotlyJS.scatter(x=λ, y=I, line=Dict(:color => "black", :width => 1), name="Data", showlegend=true)
    trace2 = PlotlyJS.scatter(x=λ, y=Imask, line=Dict(:color => "red", :width => 1), name="Masked", showlegend=true)
    trace3 = PlotlyJS.scatter(x=λ, y=Icube, line=Dict(:color => "green", :width => 1), name="Cubic Spline", showlegend=true)
    layout = PlotlyJS.Layout(
        xaxis_title="\$\\lambda\\ (\\mu{\\rm m})\$",
        yaxis_title="\$I_{\\nu}\\ ({\\rm MJy}\\,{\\rm sr}^{-1})\$",
        xaxis_constrain="domain",
        font_family="Georgia, Times New Roman, Serif",
        template="plotly_white"
    )
    p = PlotlyJS.plot([trace1, trace2, trace3], layout)
    PlotlyJS.savefig(p, "output_test/linefilt_$(xi)_$(yi).html")

    # Fit continuum and lines
    p_cont, I_cont, comps_cont, χ2red = @time continuum_fit_spaxel(cube_fitter, (xi, yi), verbose=true)
    p_line, I_line, comps_line = @time line_fit_spaxel(cube_fitter, (xi, yi), I_cont, verbose=true)

    # Combine results
    I_model = I_cont .+ I_line
    comps = merge(comps_cont, comps_line)

    # Plot results
    λ0_ln = [ln.λ₀ for ln ∈ cube_fitter.lines]
    if cube_fitter.plot_spaxels != :none
        CubeFit.plot_spaxel_fit(cube_fitter.cube.λ, cube_fitter.cube.Iλ[xi, yi, :] , I_model, comps, 
            cube_fitter.n_dust_cont, cube_fitter.n_dust_feat, λ0_ln, cube_fitter.line_names,
            χ2red, cube_fitter.name, "spaxel_$(xi)_$(yi)", backend=cube_fitter.plot_spaxels)
    end
end
