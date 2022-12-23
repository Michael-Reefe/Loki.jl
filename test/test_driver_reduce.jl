using Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()
Pkg.precompile()
using Loki

run_jwst_pipeline(
    "/Users/mreefe/Dropbox/Astrophysics/Phoenix_Cluster/JWST_pipeline/NGC_7469_uncal/",
    "/Users/mreefe/Dropbox/Astrophysics/Phoenix_Cluster/JWST_pipeline/NGC_7469_cal",
    "/Users/mreefe/Dropbox/Astrophysics/Phoenix_Cluster/JWST_pipeline/NGC_7469_uncal_bg/",
    "/Users/mreefe/Dropbox/Astrophysics/Phoenix_Cluster/JWST_pipeline/NGC_7469_cal_bg/"
)
