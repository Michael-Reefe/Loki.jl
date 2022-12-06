using Pkg; Pkg.activate(dirname(@__DIR__))
Pkg.instantiate(); Pkg.precompile()
using Loki

using Logging, LoggingExtras

logger = TeeLogger(global_logger(), 
                   MinLevelLogger(FileLogger("loki.log"), Logging.Debug))

with_logger(logger) do

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
    interpolate_cube!(obs.channels[2])

    # cube_rebin!(obs, [1,2,3,4])

    # Create the cube fitting object
    cube_fitter = CubeFitter(obs.channels[2], obs.z, "test_2", 1; parallel=false, plot_spaxels=:plotly,
        plot_maps=true, save_fits=true)

    # Fit some individual spaxels
    x = [28, 28, 33, 25, 13, 10, 6, 15]
    y = [13, 12, 4, 19, 18, 14, 17, 38]

    # x = [31, 15, 15]
    # y = [23, 16, 17]

    # (19,13) roughly corresponds to (28,13)
    # x = [19]
    # y = [13]

    for (xi, yi) ∈ zip(x, y)

        # Fit continuum and lines
        @time fit_spaxel(cube_fitter, (xi, yi))

    end

end
