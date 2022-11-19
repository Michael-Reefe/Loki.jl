using Pkg; Pkg.activate(dirname(@__DIR__))
Pkg.instantiate(); Pkg.precompile()
using Loki


# Load in data
obs = from_fits(["data/jw01328-o015_t014_miri_ch1-mediumshortlong-_s3d.fits", 
    "data/jw01328-o015_t014_miri_ch2-mediumshortlong-_s3d.fits", 
    "data/jw01328-o015_t014_miri_ch3-mediumshortlong-_s3d.fits", 
    "data/jw01328-o015_t014_miri_ch4-mediumshortlong-_s3d.fits"], 
    0.016317)

# obs = from_fits(["data/jw01039-o005_t001_miri_ch1-shortlongmedium-_s3d.fits",
#     "data/jw01039-o005_t001_miri_ch2-shortlongmedium-_s3d.fits",
#     "data/jw01039-o005_t001_miri_ch3-shortlongmedium-_s3d.fits",
#     "data/jw01039-o005_t001_miri_ch4-shortlongmedium-_s3d.fits"],
#     0.0266)

obs = correct(obs)

# Create the cube fitting object
cube_fitter = CubeFitter(obs.channels[2], obs.z, "test"; parallel=false, plot_spaxels=:plotly,
    plot_maps=true, save_fits=true)
interpolate_cube!(cube_fitter.cube)

# Fit some individual spaxels
x = [28, 33, 25, 13, 10, 6, 15]
y = [12, 4, 19, 18, 14, 17, 38]

# x = [15, 15]
# y = [16, 17]

for (xi, yi) ∈ zip(x, y)

    λ = cube_fitter.cube.λ
    I = cube_fitter.cube.Iλ[xi, yi, :]
    σ = cube_fitter.cube.σI[xi, yi, :]

    # Fit continuum and lines
    @time fit_spaxel(cube_fitter, (xi, yi), verbose=true)

    # # Check emission line masking
    # mask, Icube, σcube = CubeFit.continuum_cubic_spline(λ, I, σ)
    # Imask = copy(I)
    # Imask[mask] .= NaN

    # trace1 = PlotlyJS.scatter(x=λ, y=I, line=Dict(:color => "black", :width => 1), name="Data", showlegend=true)
    # trace2 = PlotlyJS.scatter(x=λ, y=Imask, line=Dict(:color => "red", :width => 1), name="Masked", showlegend=true)
    # trace3 = PlotlyJS.scatter(x=λ, y=Icube, line=Dict(:color => "green", :width => 1), name="Cubic Spline", showlegend=true)
    # layout = PlotlyJS.Layout(
    #     xaxis_title="\$\\lambda\\ (\\mu{\\rm m})\$",
    #     yaxis_title="\$I_{\\nu}\\ ({\\rm MJy}\\,{\\rm sr}^{-1})\$",
    #     xaxis_constrain="domain",
    #     font_family="Georgia, Times New Roman, Serif",
    #     template="plotly_white"
    # )
    # p = PlotlyJS.plot([trace1, trace2, trace3], layout)
    # PlotlyJS.savefig(p, "output_test/linefilt_$(xi)_$(yi).html")

end
