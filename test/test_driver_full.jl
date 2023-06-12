using Distributed

procs = addprocs(Sys.CPU_THREADS)
@everywhere begin
    using Pkg; Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate(); Pkg.precompile()
end
@everywhere using Loki

# using Pkg; Pkg.activate(dirname(@__DIR__))
# Pkg.instantiate(); Pkg.precompile()
# using Loki
# n_procs = 1

# Channel to run the fitting on
# channel = 2

# Load in data
# obs = from_fits(["data/Level3_ch1-long_s3d.fits",
#                  "data/Level3_ch1-medium_s3d.fits",
#                  "data/Level3_ch1-short_s3d.fits",
#                  "data/Level3_ch2-long_s3d.fits",
#                  "data/Level3_ch2-medium_s3d.fits",
#                  "data/Level3_ch2-short_s3d.fits",
#                  "data/Level3_ch3-long_s3d.fits",
#                  "data/Level3_ch3-medium_s3d.fits",
#                  "data/Level3_ch3-short_s3d.fits",
#                  "data/Level3_ch4-long_s3d.fits",
#                  "data/Level3_ch4-medium_s3d.fits",
#                  "data/Level3_ch4-short_s3d.fits"],
#                  0.016317)

# obs = from_fits(["data/VV_114E_ch1-long_s3d.fits",
#                  "data/VV_114E_ch1-medium_s3d.fits",
#                  "data/VV_114E_ch1-short_s3d.fits",
#                  "data/VV_114E_ch2-long_s3d.fits",
#                  "data/VV_114E_ch2-medium_s3d.fits",
#                  "data/VV_114E_ch2-short_s3d.fits",
#                  "data/VV_114E_ch3-long_s3d.fits",
#                  "data/VV_114E_ch3-medium_s3d.fits",
#                  "data/VV_114E_ch3-short_s3d.fits",
#                  "data/VV_114E_ch4-long_s3d.fits",
#                  "data/VV_114E_ch4-medium_s3d.fits",
#                  "data/VV_114E_ch4-short_s3d.fits"],
#                  0.02007)

# obs = from_fits(["data/NGC_6552_ch1-long_s3d.fits",
#                  "data/NGC_6552_ch1-medium_s3d.fits",
#                  "data/NGC_6552_ch1-short_s3d.fits",
#                  "data/NGC_6552_ch2-long_s3d.fits",
#                  "data/NGC_6552_ch2-medium_s3d.fits",
#                  "data/NGC_6552_ch2-short_s3d.fits",
#                  "data/NGC_6552_ch3-long_s3d.fits",
#                  "data/NGC_6552_ch3-medium_s3d.fits",
#                  "data/NGC_6552_ch3-short_s3d.fits",
#                  "data/NGC_6552_ch4-long_s3d.fits",
#                  "data/NGC_6552_ch4-medium_s3d.fits",
#                  "data/NGC_6552_ch4-short_s3d.fits"],
#                  0.0266)

obs = from_fits(["data/NGC_7319_ch1-long_s3d.fits",
                 "data/NGC_7319_ch1-medium_s3d.fits",
                 "data/NGC_7319_ch1-short_s3d.fits",
                 "data/NGC_7319_ch2-long_s3d.fits",
                 "data/NGC_7319_ch2-medium_s3d.fits",
                 "data/NGC_7319_ch2-short_s3d.fits",
                 "data/NGC_7319_ch3-long_s3d.fits",
                 "data/NGC_7319_ch3-medium_s3d.fits",
                 "data/NGC_7319_ch3-short_s3d.fits",
                 "data/NGC_7319_ch4-long_s3d.fits",
                 "data/NGC_7319_ch4-medium_s3d.fits",
                 "data/NGC_7319_ch4-short_s3d.fits"],
                 0.022)

# obs = from_fits(["data/F2M1106_ch1-long_s3d.fits",
#                  "data/F2M1106_ch1-medium_s3d.fits",
#                  "data/F2M1106_ch1-short_s3d.fits",
#                  "data/F2M1106_ch2-long_s3d.fits",
#                  "data/F2M1106_ch2-medium_s3d.fits",
#                  "data/F2M1106_ch2-short_s3d.fits",
#                  "data/F2M1106_ch3-long_s3d.fits",
#                  "data/F2M1106_ch3-medium_s3d.fits",
#                  "data/F2M1106_ch3-short_s3d.fits",
#                  "data/F2M1106_ch4-long_s3d.fits",
#                  "data/F2M1106_ch4-medium_s3d.fits",
#                  "data/F2M1106_ch4-short_s3d.fits"],
#                  0.43744)

# obs_full = from_fits(["data/NGC_7469_Level3_ch1-shortmediumlong_s3d.fits",
#                  "data/NGC_7469_Level3_ch2-shortmediumlong_s3d.fits",
#                  "data/NGC_7469_Level3_ch3-shortmediumlong_s3d.fits",
#                  "data/NGC_7469_Level3_ch4-shortmediumlong_s3d.fits"],
#                  0.016317)
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

channel = 0
nm = replace(obs.name, " " => "_") 
name = nm * "_ch$(channel)_full_06-01-23"

if isfile("processed-data-$nm.loki")
    obs = load!("processed-data-$nm.loki")
else
    # Convert to rest-frame wavelength vector, and mask out bad spaxels
    correct!(obs)
    # Concatenate the subchannels of each channel so that we have one cube for each channel
    for i_channel ∈ 1:4
        reproject_channels!(obs, i_channel, out_id=i_channel, method=:adaptive) 
        # Interpolate NaNs in otherwise good spaxels
        interpolate_nans!(obs.channels[i_channel])
    end
    reproject_channels!(obs, [1,2,3], out_id=0, method=:adaptive)
    interpolate_nans!(obs.channels[0], obs.z)
    save!("processed-data-$nm.loki", obs)
end

# Do the optical depth pre-fitting
# τ_guess = fit_optical_depth(obs)

# Convolve with the PSF FWHM
# convolve_psf!(obs.channels[channel], psf_scale=1., kernel_type=:Tophat)

# combine channels 1-3 onto the channel 1 grid
# cube_combine!(obs, [1,2,3]; out_grid=1, out_id=0)
# rebin into a 4x4 grid
# cube_rebin!(obs, 4, 0; out_id=-1)

# Create the cube fitting object
# plot_range=[(7.61, 7.69), (12.77, 12.85)]
cube_fitter = CubeFitter(obs.channels[channel], obs.z, name; parallel=true, plot_spaxels=:pyplot, plot_maps=true, 
    save_fits=true)

# Fit the cube
fit_cube!(cube_fitter)
