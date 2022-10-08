include("cubedata.jl")
include("parameters.jl")
include("cubefit.jl")

using .CubeData
using .Parameters
using .CubeFit

using PlotlyJS

obs = from_fits(["jw01328-o015_t014_miri_ch1-mediumshortlong-_s3d.fits", 
    "jw01328-o015_t014_miri_ch2-mediumshortlong-_s3d.fits", 
    "jw01328-o015_t014_miri_ch3-mediumshortlong-_s3d.fits", 
    "jw01328-o015_t014_miri_ch4-mediumshortlong-_s3d.fits"], 
    0.016317)

obs = correct(obs)

cube_rebin!(obs, [2,3])

plot_2d(obs.channels[0], "jl-ch2-3.pdf", z=obs.z)
plot_1d(obs.channels[0], "jl-ch2-3-spec.pdf")

# Pick a spaxel to fit
λ = obs.channels[0].λ
I = obs.channels[0].Iλ[21, 21, :] .* obs.channels[1].Ω
σ = obs.channels[0].σI[21, 21, :] .* obs.channels[1].Ω

cont_model, F_cont = pahfit_spaxel(λ, I, σ, "spax_21_21", nostart=true)

# Plotly interactive plots of the model
trace1 = PlotlyJS.scatter(x=λ./1e4, y=I./1e-17, mode="lines", line=Dict(:color => "black", :width => 1), name="Data", showlegend=true)
trace2 = PlotlyJS.scatter(x=λ./1e4, y=F_cont./1e-17, mode="lines", line=Dict(:color => "red", :width => 1), name="Model", showlegend=true)

layout = PlotlyJS.Layout(
    xaxis_title="\$\\lambda\\ ({\\rm \\mu m})\$",
    xaxis_constrain="domain",
    yaxis_title="\$F_{\\lambda}\\ (10^{-17}\\ {\\rm erg\\ s}^{-1}\\ {\\rm cm}^{-2}\\ {\\rm \\mathring{A}}^{-1})\$",
    title="NGC 7469",
    font_family="Georgia, Times New Roman, Serif",
    hovermode="x",
    template="plotly_white"
)

p = PlotlyJS.plot([trace1, trace2], layout)
PlotlyJS.savefig(p, "pahfit_spax_21_21.html")
