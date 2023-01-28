using Distributed
using TimerOutputs

procs = addprocs(Sys.CPU_THREADS)
n_procs = length(procs)
@everywhere begin
    using Pkg; Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate(); Pkg.precompile()
end
@everywhere using Loki

# using Pkg; Pkg.activate(dirname(@__DIR__))
# Pkg.instantiate(); Pkg.precompile()
# using Loki
# n_procs = 1

to = TimerOutput()
channel = 3
# Load in data
# obs = from_fits(["data/jw01328-o015_t014_miri_ch1-mediumshortlong-_s3d.fits", 
#     "data/jw01328-o015_t014_miri_ch2-mediumshortlong-_s3d.fits", 
#     "data/jw01328-o015_t014_miri_ch3-mediumshortlong-_s3d.fits", 
#     "data/jw01328-o015_t014_miri_ch4-mediumshortlong-_s3d.fits"], 
#     0.016317)

# obs = from_fits(["data/Level3_ch1-shortmediumlong_s3d.fits",
#                  "data/Level3_ch2-shortmediumlong_s3d.fits",
#                  "data/Level3_ch3-shortmediumlong_s3d.fits",
#                  "data/Level3_ch4-shortmediumlong_s3d.fits"],
#                  0.016317)

obs = from_fits(["data/NGC_6552_Level3_ch1-shortmediumlong_s3d.fits",
                 "data/NGC_6552_Level3_ch2-shortmediumlong_s3d.fits",
                 "data/NGC_6552_Level3_ch3-shortmediumlong_s3d.fits",
                 "data/NGC_6552_Level3_ch4-shortmediumlong_s3d.fits"],
                 0.0266)

# obs = from_fits(["data/VV_114E_Level3_ch1-shortmediumlong_s3d.fits",
#                  "data/VV_114E_Level3_ch2-shortmediumlong_s3d.fits",
#                  "data/VV_114E_Level3_ch3-shortmediumlong_s3d.fits",
#                  "data/VV_114E_Level3_ch4-shortmediumlong_s3d.fits"],
#                  0.02007)

obs = correct(obs)

# Do the optical depth pre-fitting
τ_guess = fit_optical_depth(obs)

# rebin channels 2-4 onto channel 1
# cube_rebin!(obs; out_grid=1)

# Create the cube fitting object
cube_fitter = CubeFitter(obs.channels[channel], obs.z, τ_guess, obs.name * "_ch$(channel)_nofringe", n_procs; 
    parallel=true, plot_spaxels=:pyplot, plot_maps=true, save_fits=true)

# Perform the Levenberg-Marquardt least-squares fitting
@timeit to "Full Fitting Procedure for Channel $channel" fit_cube!(cube_fitter)
print_timer(to)
