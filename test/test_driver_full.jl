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
# Load in data

obs = from_fits(["data/NGC_7469_Level3_ch1-shortmediumlong_s3d.fits",
                 "data/NGC_7469_Level3_ch2-shortmediumlong_s3d.fits",
                 "data/NGC_7469_Level3_ch3-shortmediumlong_s3d.fits",
                 "data/NGC_7469_Level3_ch4-shortmediumlong_s3d.fits"],
                 0.016317)

# obs = from_fits(["data/NGC_6552_Level3_ch1-shortmediumlong_s3d.fits",
#                  "data/NGC_6552_Level3_ch2-shortmediumlong_s3d.fits",
#                  "data/NGC_6552_Level3_ch3-shortmediumlong_s3d.fits",
#                  "data/NGC_6552_Level3_ch4-shortmediumlong_s3d.fits"],
#                  0.0266)

# obs = from_fits(["data/VV_114E_Level3_ch1-shortmediumlong_s3d.fits",
#                  "data/VV_114E_Level3_ch2-shortmediumlong_s3d.fits",
#                  "data/VV_114E_Level3_ch3-shortmediumlong_s3d.fits",
#                  "data/VV_114E_Level3_ch4-shortmediumlong_s3d.fits"],
#                  0.02007)

# obs = from_fits(["data/NGC_7319_Level3_ch1-shortmediumlong_s3d.fits",
#                  "data/NGC_7319_Level3_ch2-shortmediumlong_s3d.fits",
#                  "data/NGC_7319_Level3_ch3-shortmediumlong_s3d.fits",
#                  "data/NGC_7319_Level3_ch4-shortmediumlong_s3d.fits"],
#                  0.022)

obs = correct(obs)

# Do the optical depth pre-fitting
τ_guess = fit_optical_depth(obs)

# combine channels 1-3 onto the channel 1 grid
cube_combine!(obs, [1,2,3]; out_grid=1, out_id=0)
# rebin into a 4x4 grid
# cube_rebin!(obs, 4, 0; out_id=-1)

# Channel to run the fitting on
channel = 0

# Create the cube fitting object
# plot_range=[(7.61, 7.69), (12.77, 12.85)]
cube_fitter = CubeFitter(obs.channels[channel], obs.z, τ_guess, obs.name * "_ch$(channel)_full_flexible", n_procs; 
    parallel=true, plot_spaxels=:pyplot, plot_maps=true, save_fits=true)

# Perform the Levenberg-Marquardt least-squares fitting
@timeit to "Full Fitting Procedure for Channel $channel" fit_cube!(cube_fitter)
print_timer(to)
