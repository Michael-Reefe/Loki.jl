using Distributed

# procs = addprocs(Sys.CPU_THREADS, exeflags="--threads=$(Threads.nthreads())")
@everywhere begin
    using Pkg; Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate(); Pkg.precompile()
end
@everywhere using Loki

# Load in data
obs = from_fits(["data/Level3_ch1-long_s3d.fits",
                 "data/Level3_ch1-medium_s3d.fits",
                 "data/Level3_ch1-short_s3d.fits",
                 "data/Level3_ch2-long_s3d.fits",
                 "data/Level3_ch2-medium_s3d.fits",
                 "data/Level3_ch2-short_s3d.fits",
                 "data/Level3_ch3-long_s3d.fits",
                 "data/Level3_ch3-medium_s3d.fits",
                 "data/Level3_ch3-short_s3d.fits",
                 "data/Level3_ch4-long_s3d.fits",
                 "data/Level3_ch4-medium_s3d.fits",
                 "data/Level3_ch4-short_s3d.fits"],
                 0.016317)

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

# Convert to rest-frame wavelength vector, and mask out bad spaxels
channel = 0
nm = replace(obs.name, " " => "_") 
name = nm * "_ch$(channel)_aperture_m_nuc_emissivity_hotdustmodel_0_4"

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

# Make aperture
ap = make_aperture(obs.channels[channel], :Circular, "23:03:15.610", "+8:52:26.10", 0.5, auto_centroid=true,
    scale_psf=false)
# ap = make_aperture(obs.channels[channel], :Circular, "23:03:15.575", "+8:52:24.80", 0.5, auto_centroid=false,
#     scale_psf=false)
# ap = make_aperture(obs.channels[channel], :Rectangular, "23:03:15.692", "+8:52:26.622", 0.13, 0.13, 0.,
    # auto_centroid=false, scale_psf=false)

# ap = make_aperture(obs.channels[channel], :Circular, "18:00:07.21", "+66:36:54.5", 0.5, auto_centroid=true,
#     scale_psf=false)
# ap = make_aperture(obs.channels[channel], :Rectangular, "18:00:7.17613", "+66:36:55.19675", 0.13, 0.13, 0., 
#     auto_centroid=false, scale_psf=false)

# ap = make_aperture(obs.channels[channel], :Circular, "01:07:47.525", "-17:30:25.25", 0.5, auto_centroid=true,
#     scale_psf=false)

# Create the cube fitting object
cube_fitter = CubeFitter(obs.channels[channel], obs.z, name; parallel=true, plot_spaxels=:both, 
    plot_maps=true, save_fits=true)

# Fit the cube
fit_cube!(cube_fitter, ap)
