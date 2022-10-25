using Distributions
using Interpolations
using ProgressMeter
using PlotlyJS
using DataFrames
using CSV
using Printf
using PyPlot
using PyCall

using Loki

py_anchored_artists = pyimport("mpl_toolkits.axes_grid1.anchored_artists")


obs = from_fits(["jw01328-o015_t014_miri_ch1-mediumshortlong-_s3d.fits", 
    "jw01328-o015_t014_miri_ch2-mediumshortlong-_s3d.fits", 
    "jw01328-o015_t014_miri_ch3-mediumshortlong-_s3d.fits", 
    "jw01328-o015_t014_miri_ch4-mediumshortlong-_s3d.fits"], 
    0.016317)

obs = correct(obs)

spax = Dict(1 => (19,25), 2 => (21,19), 3 => (22,25), 4 => (15,12))
x, y = spax[1]

# Pick a spaxel to fit
λ = obs.channels[1].λ
I = obs.channels[1].Iλ[x, y, :]
σ = obs.channels[1].σI[x, y, :]

# # Fit a spaxel
# CubeFit.continuum_fit_spaxel(λ, I, σ; plot=:plotly)

# Fit the cube
param_maps, I_model = fit_cube(obs.channels[1]; progress=true, plot_continua=:none, plot_lines=:none)

amps = param_maps[:SI_69865][:amp]
fwhms = param_maps[:SI_69865][:fwhm]
voffs = param_maps[:SI_69865][:voff]
∫Is = log10.(param_maps[:SI_69865][:∫I])
snrs = param_maps[:SI_69865][:SNR]

for (data, name, label) ∈ zip([amps, fwhms, voffs, ∫Is, snrs], 
    ["amp", "fwhm", "voff", "fluxint", "snr"], 
    ["Amp\$_{\\rm S\\ I}\$ (MJy sr\$^{-1}\$)", "FWHM\$_{\\rm S\\ I}\$ (km s\$^{-1}\$)", "\$v_{\\rm off, S\\ I}\$ (km s\$^{-1}\$)", "\$\\log_{10}(I_{\\rm S\\ I} /\$ MJy sr\$^{-1}\$ \$\\mu\$m)", "\$S/N_{\\rm S\\ I}\$"])
    
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
