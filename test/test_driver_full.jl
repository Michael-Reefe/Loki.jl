using Distributed

procs = addprocs(maximum([0, Sys.CPU_THREADS - 1]))
n_procs = length(procs)
@everywhere begin
    using Pkg; Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate(); Pkg.precompile()
end
@everywhere using Loki

# TODO: Convert this all into a main() function in Loki.jl

channel = 1

# This is the logger for ONLY the main process
using Logging, LoggingExtras
using Dates
const date_format = "yyyy-mm-dd HH:MM:SS"
timestamp_logger(logger) = TransformerLogger(logger) do log
    merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
end

logger = TeeLogger(ConsoleLogger(stdout, Logging.Info), 
                   timestamp_logger(MinLevelLogger(FileLogger("loki.main.log"; always_flush=true), 
                                                   Logging.Debug)))
global_logger(logger)


# Load in data
obs = from_fits(["data/jw01328-o015_t014_miri_ch1-mediumshortlong-_s3d.fits", 
    "data/jw01328-o015_t014_miri_ch2-mediumshortlong-_s3d.fits", 
    "data/jw01328-o015_t014_miri_ch3-mediumshortlong-_s3d.fits", 
    "data/jw01328-o015_t014_miri_ch4-mediumshortlong-_s3d.fits"], 
    0.016317)

# obs = from_fits([parent * "/jw01039-o005_t001_miri_ch1-shortlongmedium-_s3d.fits",
#     parent * "/jw01039-o005_t001_miri_ch2-shortlongmedium-_s3d.fits",
#     parent * "/jw01039-o005_t001_miri_ch3-shortlongmedium-_s3d.fits",
#     parent * "/jw01039-o005_t001_miri_ch4-shortlongmedium-_s3d.fits"],
#     0.0266)

obs = correct(obs)

# Create the cube fitting object
cube_fitter = CubeFitter(obs.channels[channel], obs.z, obs.name * "_ch$(channel)_test_SAMIN", n_procs; 
    parallel=true, plot_spaxels=:pyplot, plot_maps=true, save_fits=true)

# Perform the Levenberg-Marquardt least-squares fitting
cube_fitter = @time fit_cube(cube_fitter)
