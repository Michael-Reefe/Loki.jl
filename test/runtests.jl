###############################################################################
# Loki.jl Test Suite
# Run with: julia --project=. test/runtests.jl
# Or from within Julia: import Pkg; Pkg.test()
###############################################################################

using Test
using Loki
using Unitful
using Photometry     # for CircularAperture, EllipticalAperture, etc.
using FITSIO         # for FITSHeader (WCS tests)
using WCS            # for WCSTransform (WCS tests)
using SkyCoords      # for ICRSCoords, FK5Coords (WCS tests)

@testset "Loki.jl" begin
    include("test_math_doppler.jl")
    include("test_math_profiles.jl")
    include("test_math_lines.jl")        # extended: hermite, asym Drude/PearsonIV
    include("test_math_continuum.jl")    # extended: Blackbody dispatch, Blackbody_modified
    include("test_math_stats.jl")
    include("test_parameters.jl")
    include("test_math_extinction.jl")
    include("test_math_misc.jl")         # extended: match_fluxunits
    include("test_cubedata_utils.jl")
    include("test_apertures.jl")
    include("test_parameters_collection.jl")  # extended: lock!, set_val!, check_valid, get_tied_pairs
    include("test_fitdata.jl")
    include("test_parsing.jl")
    include("test_wcs_utils.jl")         # new: WCS utility functions
    include("test_cubefitter.jl")        # new: DataCube functions + CubeFitter-dependent functions
end
