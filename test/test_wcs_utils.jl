###############################################################################
# Tests for WCS utility functions
# Source: src/util/wcs_utils.jl
#
# All tests use synthetically constructed WCSTransform and FITSHeader objects —
# no FITS files or real observations are needed.
#
# Functions tested:
#   create_wcs_from_header(hdr)           — parse FITSHeader into WCSTransform
#   string_to_coordframe(sframe)          — RADESYS string → SkyCoords frame type
#   get_sky_rotation_angle(wcs)           — rotation angle from PC or CD matrix
#   get_pixel_resolutions(wcs)            — pixel scale per axis
#   shrink_wcs_dimensions(wcs; keep_axes) — reduce WCS to fewer axes
#   extend_wcs_dimensions(wcs, λ)         — add a wavelength axis to 2D WCS
###############################################################################

@testset "WCS utilities" begin

    # =========================================================================
    # create_wcs_from_header(hdr::FITSHeader)
    # Parses WCS keywords from a FITS header into a WCSTransform.
    # =========================================================================
    @testset "create_wcs_from_header" begin

        # Minimal valid 2-axis header
        hdr = FITSHeader(
            ["NAXIS", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
             "CDELT1", "CDELT2", "CTYPE1", "CTYPE2"],
            [2, 150.0, 2.0, 50.0, 50.0, -0.01, 0.01, "RA---TAN", "DEC--TAN"],
            fill("", 9)
        )
        wcs = Loki.create_wcs_from_header(hdr)

        # Returns a WCSTransform with correct naxis
        @test wcs isa WCSTransform
        @test wcs.naxis == 2

        # CRVAL is correctly transferred
        @test wcs.crval ≈ [150.0, 2.0]

        # CRPIX is correctly transferred
        @test wcs.crpix ≈ [50.0, 50.0]

        # CDELT is correctly transferred
        @test wcs.cdelt ≈ [-0.01, 0.01]

        # CTYPE strings are preserved
        @test wcs.ctype[1] == "RA---TAN"
        @test wcs.ctype[2] == "DEC--TAN"

        # Missing optional keywords (PC matrix, EQUINOX, etc.) → no error
        @test_nowarn Loki.create_wcs_from_header(hdr)

        # Header with PC matrix
        hdr_pc = FITSHeader(
            ["NAXIS", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
             "CDELT1", "CDELT2", "CTYPE1", "CTYPE2",
             "PC1_1", "PC1_2", "PC2_1", "PC2_2"],
            [2, 150.0, 2.0, 50.0, 50.0, -0.01, 0.01, "RA---TAN", "DEC--TAN",
             1.0, 0.0, 0.0, 1.0],
            fill("", 13)
        )
        wcs_pc = Loki.create_wcs_from_header(hdr_pc)
        @test wcs_pc.pc ≈ [1.0 0.0; 0.0 1.0]

        # Missing NAXIS → error
        hdr_bad = FITSHeader(["CRVAL1"], [150.0], [""])
        @test_throws ErrorException Loki.create_wcs_from_header(hdr_bad)
    end

    # =========================================================================
    # string_to_coordframe(sframe::String)
    # Returns the SkyCoords.jl frame type corresponding to a FITS RADESYS string.
    # =========================================================================
    @testset "string_to_coordframe" begin

        # Known RADESYS strings map to correct frame types
        @test Loki.string_to_coordframe("ICRS") == ICRSCoords
        @test Loki.string_to_coordframe("FK5")  == FK5Coords{2000}
        @test Loki.string_to_coordframe("FK4")  == FK5Coords{1950}

        # Unknown string → warning + defaults to ICRSCoords
        @test_logs (:warn, r"Could not recognize RADESYS") Loki.string_to_coordframe("GALACTIC")
        @test Loki.string_to_coordframe("UNKNOWN") == ICRSCoords

        # Return value is a Type (could be DataType or UnionAll for parametric types)
        @test Loki.string_to_coordframe("ICRS") isa Type
    end

    # =========================================================================
    # get_sky_rotation_angle(wcs::WCSTransform)
    # Returns rotation angle in radians from the PC or CD matrix.
    # =========================================================================
    @testset "get_sky_rotation_angle" begin

        # No PC or CD matrix (all zeros default) → angle = 0
        wcs_norot = WCSTransform(2;
            crval=[150.0, 2.0],
            crpix=[50.0, 50.0],
            cdelt=[-0.01, 0.01],
            ctype=["RA---TAN", "DEC--TAN"]
        )
        angle_norot = Loki.get_sky_rotation_angle(wcs_norot)
        @test ustrip(angle_norot) ≈ 0.0 atol=1e-12

        # Identity PC matrix → no rotation (but CDELT1 < 0 means no sign flip)
        wcs_ident = WCSTransform(2;
            crval=[150.0, 2.0],
            crpix=[50.0, 50.0],
            cdelt=[-0.01, 0.01],
            ctype=["RA---TAN", "DEC--TAN"],
            pc=[1.0 0.0; 0.0 1.0]
        )
        angle_ident = Loki.get_sky_rotation_angle(wcs_ident)
        @test ustrip(angle_ident) ≈ 0.0 atol=1e-12

        # Return type is a Unitful quantity in radians
        @test unit(angle_ident) == u"rad"

        # Finite angle for any valid PC matrix
        @test isfinite(ustrip(angle_ident))
    end

    # =========================================================================
    # get_pixel_resolutions(wcs::WCSTransform)
    # Returns pixel scale in each axis derived from PC/CD/CDELT.
    # =========================================================================
    @testset "get_pixel_resolutions" begin

        # CDELT-only WCS (no PC or CD): resolutions = |CDELT|
        wcs_cdelt = WCSTransform(2;
            crval=[150.0, 2.0],
            crpix=[50.0, 50.0],
            cdelt=[-0.01, 0.01]
        )
        res = Loki.get_pixel_resolutions(wcs_cdelt)

        # Returns a 2×1 matrix (column vector)
        @test length(res) == 2
        @test res[1] ≈ 0.01 rtol=1e-12   # |CDELT1| = 0.01
        @test res[2] ≈ 0.01 rtol=1e-12   # |CDELT2| = 0.01

        # All resolutions are positive
        @test all(res .> 0)

        # With identity PC matrix and CDELT: same result
        wcs_pc = WCSTransform(2;
            crval=[150.0, 2.0],
            crpix=[50.0, 50.0],
            cdelt=[-0.01, 0.01],
            pc=[1.0 0.0; 0.0 1.0]
        )
        res_pc = Loki.get_pixel_resolutions(wcs_pc)
        @test res_pc[1] ≈ 0.01 rtol=1e-12
        @test res_pc[2] ≈ 0.01 rtol=1e-12

        # Number of resolutions equals naxis
        wcs3 = WCSTransform(3;
            crval=[150.0, 2.0, 5.0],
            crpix=[50.0, 50.0, 1.0],
            cdelt=[-0.01, 0.01, 0.001]
        )
        res3 = Loki.get_pixel_resolutions(wcs3)
        @test length(res3) == 3
        @test res3[3] ≈ 0.001 rtol=1e-12
    end

    # =========================================================================
    # shrink_wcs_dimensions(wcs::WCSTransform; keep_axes=1:2)
    # Reduces a 3D WCS to 2D by selecting a subset of axes.
    # =========================================================================
    @testset "shrink_wcs_dimensions" begin

        # Build a 3-axis WCS
        wcs3 = WCSTransform(3;
            crval=[150.0, 2.0, 5.0],
            crpix=[50.0, 50.0, 1.0],
            cdelt=[-0.01, 0.01, 0.001],
            ctype=["RA---TAN", "DEC--TAN", "WAVE"]
        )

        # Shrink to first 2 axes (default keep_axes=1:2)
        wcs2 = Loki.shrink_wcs_dimensions(wcs3; keep_axes=1:2)
        @test wcs2.naxis == 2
        @test wcs2.crval ≈ [150.0, 2.0]
        @test wcs2.crpix ≈ [50.0, 50.0]
        @test wcs2.cdelt ≈ [-0.01, 0.01]
        @test wcs2.ctype[1] == "RA---TAN"
        @test wcs2.ctype[2] == "DEC--TAN"

        # Keep only the wavelength axis
        wcs_wave = Loki.shrink_wcs_dimensions(wcs3; keep_axes=3:3)
        @test wcs_wave.naxis == 1
        @test wcs_wave.crval ≈ [5.0]
        @test wcs_wave.cdelt ≈ [0.001]

        # Keeping all axes returns same dimensionality
        wcs_all = Loki.shrink_wcs_dimensions(wcs3; keep_axes=1:3)
        @test wcs_all.naxis == 3
        @test wcs_all.crval ≈ [150.0, 2.0, 5.0]

        # Output is a WCSTransform
        @test wcs2 isa WCSTransform
    end

    # =========================================================================
    # extend_wcs_dimensions(wcs::WCSTransform, λ::Vector{<:QWave})
    # Adds a wavelength axis to a 2D spatial WCS.
    # =========================================================================
    @testset "extend_wcs_dimensions" begin

        wcs2 = WCSTransform(2;
            crval=[150.0, 2.0],
            crpix=[50.0, 50.0],
            cdelt=[-0.01, 0.01],
            ctype=["RA---TAN", "DEC--TAN"]
        )

        # Linear wavelength grid → CTYPE3 = "WAVE"
        λ_linear = collect(5.0:0.001:5.5) .* u"μm"
        wcs3 = Loki.extend_wcs_dimensions(wcs2, λ_linear)

        @test wcs3.naxis == 3
        @test wcs3.ctype[3] == "WAVE"

        # WCS.jl converts to SI units internally (meters for wavelength)
        # CRVAL3 stores the first wavelength in meters (5 μm = 5e-6 m)
        @test wcs3.crval[3] ≈ 5.0e-6 rtol=1e-6

        # CDELT3 stores wavelength spacing in meters (0.001 μm = 1e-9 m)
        @test wcs3.cdelt[3] ≈ 1.0e-9 rtol=1e-6

        # CRPIX3 = 1.0 (reference pixel is the first)
        @test wcs3.crpix[3] ≈ 1.0

        # Spatial axes are preserved unchanged (degrees, no SI conversion)
        @test wcs3.crval[1:2] ≈ [150.0, 2.0]
        @test wcs3.crpix[1:2] ≈ [50.0, 50.0]
        @test wcs3.cdelt[1:2] ≈ [-0.01, 0.01]

        # WCS.jl normalizes cunit to the SI base unit string ("m" for meters)
        @test wcs3.cunit[3] == "m"

        # Non-linear wavelength grid → CTYPE3 = "WAVE-TAB"
        λ_nonlinear = [5.0, 5.1, 5.3, 5.6, 6.0] .* u"μm"
        wcs3_tab = Loki.extend_wcs_dimensions(wcs2, λ_nonlinear)
        @test wcs3_tab.naxis == 3
        @test wcs3_tab.ctype[3] == "WAVE-TAB"

        # Angstrom wavelength input: also stored in SI (m), cunit = "m"
        λ_aa = collect(5000.0:1.0:5500.0) .* u"angstrom"
        wcs3_aa = Loki.extend_wcs_dimensions(wcs2, λ_aa)
        # 5000 Å = 5000e-10 m = 5e-7 m
        @test wcs3_aa.crval[3] ≈ 5000.0e-10 rtol=1e-6
        @test wcs3_aa.cunit[3] == "m"

        # Output is a WCSTransform
        @test wcs3 isa WCSTransform
    end

end
