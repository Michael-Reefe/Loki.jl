using Pkg; Pkg.activate(dirname(@__DIR__))
Pkg.instantiate(); Pkg.precompile()
using Loki

channel = 2

# Load in data
# obs = from_fits(["data/jw01328-o015_t014_miri_ch1-mediumshortlong-_s3d.fits", 
#     "data/jw01328-o015_t014_miri_ch2-mediumshortlong-_s3d.fits", 
#     "data/jw01328-o015_t014_miri_ch3-mediumshortlong-_s3d.fits", 
#     "data/jw01328-o015_t014_miri_ch4-mediumshortlong-_s3d.fits"], 
#     0.016317)

obs = from_fits(["data/NGC_7469_Level3_ch1-shortmediumlong_s3d.fits",
                 "data/NGC_7469_Level3_ch2-shortmediumlong_s3d.fits",
                 "data/NGC_7469_Level3_ch3-shortmediumlong_s3d.fits",
                 "data/NGC_7469_Level3_ch4-shortmediumlong_s3d.fits"],
                 0.016317)

# obs = from_fits(["data/jw01039-o005_t001_miri_ch1-shortlongmedium-_s3d.fits",
#     "data/jw01039-o005_t001_miri_ch2-shortlongmedium-_s3d.fits",
#     "data/jw01039-o005_t001_miri_ch3-shortlongmedium-_s3d.fits",
#     "data/jw01039-o005_t001_miri_ch4-shortlongmedium-_s3d.fits"],
#     0.0266)

obs = correct(obs)
interpolate_cube!(obs.channels[channel])
τ_guess = fit_optical_depth(obs)

# Create the cube fitting object
cube_fitter = CubeFitter(obs.channels[channel], obs.z, τ_guess, "sbplx", 1; parallel=false, plot_spaxels=:pyplot,
    plot_maps=true, save_fits=true)

# Fit some individual spaxels
x = [16, 11, 23, 33, 35, 28, 28, 13, 14, 15, 33, 25, 10, 6, 15, 33, 36, 8]
y = [10, 24, 35, 7, 6, 12, 13, 18, 23, 27, 4, 19, 14, 17, 38, 13, 13, 14]

if all(iszero.(cube_fitter.p_init_cont))
    fit_stack!(cube_fitter)
else
    @info "===> Initial fit to the sum of all spaxels has already been performed <==="
end


for (xi, yi) ∈ zip(x, y)
    # Fit continuum and lines
    fit_spaxel(cube_fitter, CartesianIndex(xi, yi))
end
