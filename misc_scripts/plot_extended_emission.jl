using FITSIO
using TOML
using Cosmology
using Statistics, NaNStatistics
using LaTeXStrings
using PyPlot
using Loki

function plot_decomposed_flux_maps(path::String)

    println("Reading in FITS files...")
    params = FITS(path)
    hdr = read_header(params[1])
    line_dict = TOML.parsefile(joinpath(@__DIR__, "..", "src", "options", "lines.toml"))

    out_path = dirname(path)
    Ω = hdr["PIXAR_SR"]
    z = hdr["REDSHIFT"]
    psf(λ) = 0.033 * λ + 0.106
    cosmo = cosmology(;h=1.0, OmegaM=0.27, OmegaK=0.0, OmegaR=0.0)

    # Get mask for the shape of the data
    mask = .~isfinite.(read(params["stellar_continuum_temp"]))
    lines_to_plot = []

    println("Searching the HDUs for line fluxes to plot...")
    for (i, param) in enumerate(params)
        FITSIO.fits_movabs_hdu(params.fitsfile, i)
        name = lowercase(something(FITSIO.fits_try_read_extname(params.fitsfile), ""))
        if contains(name, "flux") && (contains(name, "point") || contains(name, "extended"))
            line_name = ""
            for key in keys(line_dict["lines"])
                if contains(name, lowercase(key))
                    line_name = key
                end
            end
            push!(lines_to_plot, line_name)
        end
    end
    lines_to_plot = unique(lines_to_plot)

    println("Plotting pointlike and extended emission...")
    for line in lines_to_plot
        # Get basic line properties
        line_wave = line_dict["lines"][line]["wave"]
        line_latex = line_dict["lines"][line]["latex"]
        line_wave *= (1+z)
        psf_fwhm = psf(line_wave)

        # Prepare figure
        fig = plt.figure(figsize=(18,5.4))
        gs = fig.add_gridspec(1, 4, width_ratios=(20, 20, 20, 1), height_ratios=(1,), wspace=0.05, hspace=0.05)
        ax1 = fig.add_subplot(gs[1,1])
        ax2 = fig.add_subplot(gs[1,2])
        ax3 = fig.add_subplot(gs[1,3])
        cax = fig.add_subplot(gs[1,4])
        vmin, vmax = 0., 0.

        # Loop over total, pointlike, and extended emission
        cdata = nothing
        for (line_param, ax) ∈ zip(("flux", "flux_point", "flux_extended"), (ax1, ax2, ax3))
            data = zeros(size(read(params[2])))
            j = 1
            exists = true
            while exists
                try
                    data .+= 10 .^ read(params[join(["lines", line, j, line_param], "_")])
                    j += 1
                catch
                    exists = false
                end
            end
            data = log10.(data)
            save_path = ""
            if contains(line_param, "point")
                snr_filter = nothing
            else
                snr_filter = dropdims(nanmaximum(cat([read(params[join(["lines", line, jj, "SNR"], "_")]) for jj in 1:(j-1)]..., dims=3), dims=3), dims=3)
            end
            if contains(line_param, "point")
                data[mask] .= NaN
            end
            if line_param == "flux"
                nandata = data[isfinite.(data)]
                vmin = quantile(nandata, 0.01)
                vmax = quantile(nandata, 0.99)
            end
            _, _, cdata = plot_parameter_map(data, join(["lines", line, line_param]), save_path, Ω, z, psf_fwhm, cosmo, nothing; snr_filter=snr_filter,
                snr_thresh=3., line_latex=line_latex, modify_ax=(fig, ax), disable_colorbar=true, colorscale_limits=(vmin, vmax))
        end
        plt.colorbar(cdata, cax=cax, label=L"$\log_{10}(F /$ erg s$^{-1}$ cm$^{-2})$")

        # Save the combined flux map
        save_path = joinpath(out_path, "param_maps", "lines", line, "$(line)_decomposed_flux.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    end
    println("Done!")
end

# Equivalent to if __name__ == '__main__' in python
if abspath(PROGRAM_FILE) == @__FILE__
    path = ARGS[1]
    plot_decomposed_flux_maps(path)
end