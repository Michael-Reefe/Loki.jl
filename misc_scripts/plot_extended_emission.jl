using FITSIO
using TOML
using Cosmology
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

    println("Plotting pointlike and extended emission...")
    for (i, param) in enumerate(params)
        FITSIO.fits_movabs_hdu(params.fitsfile, i)
        name = lowercase(something(FITSIO.fits_try_read_extname(params.fitsfile), ""))
        if contains(name, "flux") && (contains(name, "point") || contains(name, "extended"))
            println(name)
            data = read(param)
            line_wave = 0.
            line_latex = ""
            line_name = ""
            for key in keys(line_dict["lines"])
                if contains(name, lowercase(key))
                    line_wave = line_dict["lines"][key]["wave"]
                    line_latex = line_dict["lines"][key]["latex"]
                    line_name = key
                end
            end
            @assert line_wave ≠ 0.
            line_wave *= (1+z)
            psf_fwhm = psf(line_wave)
            snr_filter = contains(name, "extended") ? read(params[replace(replace(name, "flux_point" => "snr"), "flux_extended" => "snr")]) : nothing
            # For the pointlike emission, map NaNs from the regular flux array onto this one 
            if contains(name, "point")
                data[mask] .= NaN
            end
            save_path = joinpath(out_path, "param_maps", "lines", line_name, name * ".pdf")
            plot_parameter_map(data, name, save_path, Ω, z, psf_fwhm, cosmo, nothing; snr_filter=snr_filter,
                snr_thresh=3., line_latex=line_latex)
        end
    end
    println("Done!")
end

# Equivalent to if __name__ == '__main__' in python
if abspath(PROGRAM_FILE) == @__FILE__
    path = ARGS[1]
    plot_decomposed_flux_maps(path)
end