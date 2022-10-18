include("../src/Loki.jl")
using .Loki

using Distributions
using ProgressMeter
using PlotlyJS
using DataFrames
using CSV
using Printf
using PyPlot
using PyCall

py_anchored_artists = pyimport("mpl_toolkits.axes_grid1.anchored_artists")


obs = from_fits(["jw01328-o015_t014_miri_ch1-mediumshortlong-_s3d.fits", 
    "jw01328-o015_t014_miri_ch2-mediumshortlong-_s3d.fits", 
    "jw01328-o015_t014_miri_ch3-mediumshortlong-_s3d.fits", 
    "jw01328-o015_t014_miri_ch4-mediumshortlong-_s3d.fits"], 
    0.016317)

obs = correct(obs)

# cube_rebin!(obs, [1,2,3,4])
# x, y = 15, 15

# plot_2d(obs.channels[0], "jl-ch1-2-3-4.pdf", z=obs.z, marker=(x,y))
# plot_1d(obs.channels[0], "jl-ch2-3-spec.pdf")

# # Pick a spaxel to fit
# λ = obs.channels[0].λ
# I = obs.channels[0].Iλ[x, y, :] .* obs.channels[1].Ω
# # I = obs.channels[0].Iλ[x, y, :]
# σ = obs.channels[0].σI[x, y, :] .* obs.channels[1].Ω
# # σ = obs.channels[0].σI[x, y, :]


# cont_model, F_cont = Loki.CubeFit.continuum_fit_spaxel(λ, I, σ, "spax_$(x)_$(y)", nostart=true)
# χ2_pah = sum((F_cont .- I).^2 ./ σ.^2) / (length(I) - 89)

# cafe = CSV.read("CAFE_FIT.CSV", DataFrame)
# good = findall(x -> minimum(λ./1e4) .< x .< maximum(λ./1e4), cafe[!, "wave"])
# cafe_λ = cafe[!, "wave"][good]
# cafe_F = cafe[!, "flux"][good] .* 1e6 .* 1e-23 .* (299792458 .* 1e10) ./ ((cafe_λ .* 1e4).^2) .* obs.channels[1].Ω
# cF_interp = linear_interpolation(cafe_λ, cafe_F, extrapolation_bc=Linear())
# χ2_cafe = sum((cF_interp.(λ./1e4) .- I).^2 ./ σ.^2) / (length(I) - 89)

# idl_pahfit = CSV.read("PAHFIT_FIT.CSV", DataFrame)
# idl_λ = idl_pahfit[!, "wave"]
# idl_F = idl_pahfit[!, "flux"] .* 1e6 .* 1e-23 .* (299792458 .* 1e10) ./ ((idl_λ .* 1e4).^2) .* obs.channels[1].Ω
# χ2_idl = sum((idl_F .- I).^2 ./ σ.^2) / (length(I) - 89)

# # Plotly interactive plots of the model
# trace1 = PlotlyJS.scatter(x=λ./1e4, y=I./1e-17, mode="lines", line=Dict(:color => "black", :width => 1), name="Data", showlegend=true)
# trace2 = PlotlyJS.scatter(x=λ./1e4, y=F_cont./1e-17, mode="lines", line=Dict(:color => "red", :width => 1), name="PAHFIT (Python), " * @sprintf("%.2f", χ2_pah), showlegend=true)
# trace3 = PlotlyJS.scatter(x=cafe_λ, y=cafe_F./1e-17, mode="lines", line=Dict(:color => "green", :width => 1), name="CAFE, " * @sprintf("%.2f", χ2_cafe), showlegend=true)
# trace4 = PlotlyJS.scatter(x=idl_λ, y=idl_F./1e-17, mode="linear", line=Dict(:color => "blue", :width => 1), name="PAHFIT (IDL), " * @sprintf("%.2f", χ2_idl), showlegend=true)

# layout = PlotlyJS.Layout(
#     xaxis_title="\$\\lambda\\ ({\\rm \\mu m})\$",
#     xaxis_constrain="domain",
#     yaxis_title="\$F_{\\lambda}\\ (10^{-17}\\ {\\rm erg\\ s}^{-1}\\ {\\rm cm}^{-2}\\ {\\rm \\mathring{A}}^{-1})\$",
#     title="NGC 7469",
#     font_family="Georgia, Times New Roman, Serif",
#     hovermode="x",
#     template="plotly_white"
# )

# p = PlotlyJS.plot([trace1, trace2, trace3, trace4], layout)
# PlotlyJS.savefig(p, "pahfit_spax_$(x)_$(y).html")




# Fit line at 7.1 μm

# Parameters
params = Param.ParamDict(
    :A => Param.Parameter(5e-6, locked=false, prior=Uniform(0., 1e-2)),        # amplitude in erg/s/cm^2/AA/sr
    :voff => Param.Parameter(0., locked=false, prior=Uniform(-500., 500.)),    # voff in km/s
    :FWHM => Param.Parameter(100., locked=false, prior=Uniform(10., 1000.))    # FWHM in km/s
)

# Line List
line_list = [Param.TransitionLine("S I", 69865., :Gaussian, params)]

# Fit the cube
param_maps, F_model = fit_cube(obs.channels[1], line_list, continuum_method=:linear, loc=68965., progress=true, plot_lines=false)

amps = param_maps["S I"][:A]
FWHMs = param_maps["S I"][:FWHM]
voffs = param_maps["S I"][:voff]
∫Fs = log10.(param_maps["S I"][:∫F])
snrs = param_maps["S I"][:SNR]

for (data, name, label) ∈ zip([amps, FWHMs, voffs, ∫Fs, snrs], 
    ["amp", "fwhm", "voff", "fluxint", "snr"], 
    ["Amp\$_{\\rm S\\ I}\$ (erg s\$^{-1}\$ cm\$^{-2}\$ \${\\rm \\AA}^{-1}\$ sr\$^{-1}\$)", "FWHM\$_{\\rm S\\ I}\$ (km s\$^{-1}\$)", "\$v_{\\rm off, S\\ I}\$ (km s\$^{-1}\$)", "\$\\log_{10}(I_{\\rm S\\ I} /\$ erg s\$^{-1}\$ cm\$^{-2}\$ sr\$^{-1}\$)", "\$S/N_{\\rm S\\ I}\$"])
    
    fig = plt.figure()
    ax = plt.subplot()
    flatdata = data[isfinite.(data)]
    cdata = ax.imshow(data', origin=:lower, cmap=:cubehelix, vmin=quantile(flatdata, 0.01), vmax=quantile(flatdata, 0.99))
    ax.axis(:off)

    n_pix = 1/(sqrt(obs.channels[1].Ω) * 180/π * 3600)
    scalebar = py_anchored_artists.AnchoredSizeBar(ax.transData, n_pix, "1\$\'\'\$", "lower left", pad=1, color=:black, 
        frameon=false, size_vertical=0.2)
    ax.add_artist(scalebar)

    fig.colorbar(cdata, ax=ax, label=label)
    plt.savefig("$(name)_2D_map.pdf", dpi=300, bbox_inches=:tight)
    plt.close()

end
