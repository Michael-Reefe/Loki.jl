###############################################################################
# Tests for instrument utility functions
# Source: src/core/cubedata.jl  (lines ~12–46)
#
# Functions tested:
#   mrs_lsf(λ)                 — MIRI MRS line-spread function (km/s)
#   mrs_psf(λ)                 — MIRI MRS point-spread function (arcsec)
#   nirspec_lsf(λ, grating)    — NIRSpec LSF by grating name (km/s)
#   nirspec_psf(λ)             — NIRSpec PSF (arcsec)
#   fits_unitstr_to_unitful(s) — FITS unit string → Julia/Unitful-compatible string
###############################################################################

@testset "Instrument utility functions" begin

    # =========================================================================
    # mrs_lsf(λ) = C_KMS / (4603 - 128*(λ/μm) + 10^(-7.4*(λ/μm)))
    # The MIRI MRS spectral resolution R decreases at longer wavelengths,
    # so the LSF in km/s increases at longer wavelengths.
    # =========================================================================
    @testset "mrs_lsf" begin

        # Output is positive, finite, with km/s units
        lsf_5 = Loki.mrs_lsf(5.0u"μm")
        @test unit(lsf_5) == u"km/s"
        @test ustrip(uconvert(u"km/s", lsf_5)) > 0
        @test isfinite(ustrip(lsf_5))

        # Longer wavelength → lower spectral resolution → higher velocity LSF
        lsf_short = Loki.mrs_lsf(5.0u"μm")
        lsf_long  = Loki.mrs_lsf(20.0u"μm")
        @test ustrip(lsf_long) > ustrip(lsf_short)

        # At λ=5 μm: R = 4603 - 128*5 + 10^(-37) ≈ 3963
        # LSF = C_KMS / 3963 ≈ 75.6 km/s
        expected_R_5 = 4603 - 128*5  # 10^(-37) ≈ 0
        expected_lsf_5 = 299792.458 / expected_R_5
        @test ustrip(uconvert(u"km/s", lsf_5)) ≈ expected_lsf_5 rtol=1e-3

        # All values in the MIRI MRS range are positive and finite
        for λ_val in [5.0, 8.0, 12.0, 18.0, 25.0]
            lsf = Loki.mrs_lsf(λ_val * u"μm")
            @test ustrip(lsf) > 0
            @test isfinite(ustrip(lsf))
        end
    end

    # =========================================================================
    # mrs_psf(λ) = (0.033*(λ/μm) + 0.016) * arcsecond
    # MIRI MRS PSF scales linearly with wavelength.
    # =========================================================================
    @testset "mrs_psf" begin

        # Output is positive and finite (unit is arcsecond, but we avoid u"arcsecond"
        # since UnitfulAngles is not a direct test dependency)
        psf_5 = Loki.mrs_psf(5.0u"μm")
        @test ustrip(psf_5) > 0
        @test isfinite(ustrip(psf_5))

        # Longer wavelength → larger PSF (linear scaling)
        psf_short = Loki.mrs_psf(5.0u"μm")
        psf_long  = Loki.mrs_psf(20.0u"μm")
        @test ustrip(psf_long) > ustrip(psf_short)

        # Exact numeric value at λ=5 μm: 0.033*5 + 0.016 = 0.181
        @test ustrip(Loki.mrs_psf(5.0u"μm")) ≈ 0.033*5 + 0.016 rtol=1e-10

        # Exact numeric value at λ=10 μm: 0.033*10 + 0.016 = 0.346
        @test ustrip(Loki.mrs_psf(10.0u"μm")) ≈ 0.033*10 + 0.016 rtol=1e-10
    end

    # =========================================================================
    # nirspec_lsf(λ, grating)
    # Returns C_KMS / R where R depends on the grating name:
    #   Medium:  G140M, G235M, G395M → R = 1000
    #   High:    G140H, G235H, G395H → R = 2700
    #   Other/prism:                 → R = 100
    # =========================================================================
    @testset "nirspec_lsf" begin

        λ = 2.0u"μm"  # wavelength doesn't affect the result (piecewise constant)

        # Medium-resolution gratings: R = 1000 → LSF = C/1000 km/s
        for grating in ["G140M", "G235M", "G395M"]
            lsf = Loki.nirspec_lsf(λ, grating)
            @test unit(lsf) == u"km/s"
            @test ustrip(lsf) ≈ 299792.458 / 1000.0 rtol=1e-10
        end

        # High-resolution gratings: R = 2700 → LSF = C/2700 km/s
        for grating in ["G140H", "G235H", "G395H"]
            lsf = Loki.nirspec_lsf(λ, grating)
            @test unit(lsf) == u"km/s"
            @test ustrip(lsf) ≈ 299792.458 / 2700.0 rtol=1e-10
        end

        # Prism / other: R = 100 → LSF = C/100 km/s
        for grating in ["PRISM", "G140", "unknown"]
            lsf = Loki.nirspec_lsf(λ, grating)
            @test unit(lsf) == u"km/s"
            @test ustrip(lsf) ≈ 299792.458 / 100.0 rtol=1e-10
        end

        # High-res has smaller LSF (higher resolution) than medium-res
        lsf_med  = Loki.nirspec_lsf(λ, "G235M")
        lsf_high = Loki.nirspec_lsf(λ, "G235H")
        @test ustrip(lsf_high) < ustrip(lsf_med)
    end

    # =========================================================================
    # nirspec_psf(λ) = (0.03 + 0.13*(λ/μm - 0.90)/4.37) * arcsecond
    # NIRSpec PSF scales linearly with wavelength.
    # =========================================================================
    @testset "nirspec_psf" begin

        # Output is positive, finite (avoid u"arcsecond" — not a direct test dep)
        psf_1 = Loki.nirspec_psf(1.0u"μm")
        @test ustrip(psf_1) > 0
        @test isfinite(ustrip(psf_1))

        # Exact numeric value at λ=0.9 μm: 0.03 + 0.13*(0-0)/4.37 = 0.03
        @test ustrip(Loki.nirspec_psf(0.90u"μm")) ≈ 0.03 rtol=1e-10

        # Longer wavelength → larger PSF
        psf_short = Loki.nirspec_psf(1.0u"μm")
        psf_long  = Loki.nirspec_psf(3.0u"μm")
        @test ustrip(psf_long) > ustrip(psf_short)
    end

    # =========================================================================
    # fits_unitstr_to_unitful(unit_string)
    # Converts FITS header unit strings to Julia-parseable Unitful strings:
    #   "um" → "μm"
    #   "."  → "*"  (FITS multiplication separator)
    #   digits/signs not preceded by "^" → insert "^" before them
    # =========================================================================
    @testset "fits_unitstr_to_unitful" begin

        # "um" is replaced by the Unicode "μm"
        @test contains(Loki.fits_unitstr_to_unitful("um"), "μm")
        @test contains(Loki.fits_unitstr_to_unitful("MJy/um"), "μm")

        # "." multiplication separator → "*"
        result_dot = Loki.fits_unitstr_to_unitful("MJy.sr")
        @test contains(result_dot, "*")
        @test !contains(result_dot, ".")  # dot is gone

        # "MJy" has no um or dots: should be unchanged
        @test Loki.fits_unitstr_to_unitful("MJy") == "MJy"

        # "cm2" → "cm^2"  (digit after letter, caret inserted)
        result_exp = Loki.fits_unitstr_to_unitful("cm2")
        @test contains(result_exp, "^")
        @test contains(result_exp, "^2")

        # "erg.s-1.cm-2" → "erg*s^-1*cm^-2"
        result_erg = Loki.fits_unitstr_to_unitful("erg.s-1.cm-2")
        @test contains(result_erg, "^-1")
        @test contains(result_erg, "^-2")
        @test contains(result_erg, "*")

        # Plain "MJy/sr" — no um, no dot, no exponent: unchanged
        @test Loki.fits_unitstr_to_unitful("MJy/sr") == "MJy/sr"
    end

end
