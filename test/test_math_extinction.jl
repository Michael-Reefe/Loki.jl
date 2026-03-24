###############################################################################
# Tests for dust extinction and attenuation functions
# Source: src/util/math.jl  (lines ~1187–1199, 909–941, 986–1031)
#
# Functions tested:
#   extinction_factor  — converts silicate absorption curve + optical depth
#                        into a multiplicative attenuation factor
#   extinction_calzetti — Calzetti et al. (2000) UV-optical attenuation law
#   extinction_cardelli — Cardelli et al. (1989) Galactic extinction curve
###############################################################################

@testset "Extinction functions" begin

    # =========================================================================
    # extinction_factor(ext, τ_97; screen)
    #
    # screen=true  (foreground screen):  factor = exp(-τ_97 * ext)
    # screen=false (mixed geometry):     factor = (1 - exp(-τ_97*ext)) / (τ_97*ext)
    #   special case τ_97=0:             factor = 1.0   (avoids 0/0)
    # =========================================================================
    @testset "extinction_factor" begin

        # Screen geometry: exact formula check
        @test Loki.extinction_factor(1.0, 1.0; screen=true) ≈ exp(-1.0) rtol=1e-12
        @test Loki.extinction_factor(0.5, 2.0; screen=true) ≈ exp(-0.5*2.0) rtol=1e-12

        # Mixed geometry: exact formula check
        # (1 - exp(-τ*ext)) / (τ*ext)
        τ, ext = 2.0, 0.8
        expected = (1 - exp(-τ*ext)) / (τ*ext)
        @test Loki.extinction_factor(ext, τ; screen=false) ≈ expected rtol=1e-12

        # Special case: τ_97=0, mixed geometry returns 1 (avoids 0/0)
        @test Loki.extinction_factor(1.0, 0.0; screen=false) == 1.0
        @test Loki.extinction_factor(0.5, 0.0; screen=false) == 1.0

        # Both converge to 1 as τ → 0
        @test Loki.extinction_factor(1.0, 1e-8; screen=true)  ≈ 1.0 atol=1e-6
        @test Loki.extinction_factor(1.0, 1e-8; screen=false) ≈ 1.0 atol=1e-6

        # Screen attenuates more (lower factor) than mixed for the same τ > 0
        # exp(-τ) < (1 - exp(-τ)) / τ  for τ > 0
        for τ_val in [0.1, 0.5, 1.0, 2.0, 5.0]
            f_screen = Loki.extinction_factor(1.0, τ_val; screen=true)
            f_mixed  = Loki.extinction_factor(1.0, τ_val; screen=false)
            @test f_screen < f_mixed
        end

        # ext=0 in mixed mode now also returns 1.0 (special-cased alongside τ_97=0)
        @test Loki.extinction_factor(0.0, 1.0; screen=false) == 1.0
        @test Loki.extinction_factor(0.0, 2.0; screen=false) == 1.0

        # Both factors are in (0, 1] for any ext ≥ 0, τ ≥ 0
        for τ_val in [0.0, 0.5, 1.0, 3.0]
            for ext_val in [0.0, 0.5, 1.0, 2.0]
                f_screen = Loki.extinction_factor(ext_val, τ_val; screen=true)
                f_mixed  = Loki.extinction_factor(ext_val, τ_val; screen=false)
                @test 0 < f_screen ≤ 1
                @test 0 < f_mixed  ≤ 1
            end
        end

        # Monotonically decreasing in τ (for fixed ext > 0)
        τ_vals = [0.5, 1.0, 2.0, 4.0]
        for i in 1:length(τ_vals)-1
            @test Loki.extinction_factor(1.0, τ_vals[i]; screen=true)  >
                  Loki.extinction_factor(1.0, τ_vals[i+1]; screen=true)
            @test Loki.extinction_factor(1.0, τ_vals[i]; screen=false) >
                  Loki.extinction_factor(1.0, τ_vals[i+1]; screen=false)
        end

        # Both approach 0 for very large τ
        @test Loki.extinction_factor(1.0, 100.0; screen=true)  < 1e-10
        @test Loki.extinction_factor(1.0, 100.0; screen=false) < 0.1
    end

    # =========================================================================
    # extinction_calzetti(λ, E_BV[, δ_uv]; Rv=4.05)
    # Two-piece law from Calzetti et al. (2000):
    #   λ ≥ 0.63 μm: kprime = 2.659*(-1.857 + 1.040/λ) + Rv
    #   λ < 0.63 μm: kprime = 2.659*(-2.156 + 1.509/λ - 0.198/λ² + 0.011/λ³) + Rv
    # attenuation = 10^(-0.4 * E_BV * kprime)
    # =========================================================================
    @testset "extinction_calzetti" begin

        # No dust (E_BV=0): attenuation = 1 for any wavelength
        for λ_val in [0.15, 0.3, 0.55, 1.0, 5.0]
            @test Loki.extinction_calzetti(λ_val * u"μm", 0.0) ≈ 1.0 rtol=1e-12
        end

        # More dust (higher E_BV) → lower attenuation (more extinction)
        λ = 0.55u"μm"
        atten_lo = Loki.extinction_calzetti(λ, 0.5)
        atten_hi = Loki.extinction_calzetti(λ, 1.0)
        @test atten_hi < atten_lo
        @test atten_lo < 1.0

        # UV has stronger attenuation than IR (Calzetti law rises to UV)
        atten_uv = Loki.extinction_calzetti(0.2u"μm",  1.0)
        atten_opt = Loki.extinction_calzetti(0.55u"μm", 1.0)
        atten_nir = Loki.extinction_calzetti(1.0u"μm",  1.0)
        @test atten_uv < atten_opt < atten_nir

        # Attenuation is always in (0, 1] for E_BV ≥ 0
        for λ_val in [0.2, 0.55, 1.0, 3.0]
            for ebv in [0.0, 0.5, 1.0, 2.0]
                atten = Loki.extinction_calzetti(λ_val * u"μm", ebv)
                @test 0 < atten ≤ 1
            end
        end

        # Test both piecewise regimes explicitly
        # λ < 0.63 μm (UV regime)
        atten_uv_regime = Loki.extinction_calzetti(0.4u"μm", 1.0)
        @test isfinite(atten_uv_regime)
        @test 0 < atten_uv_regime < 1

        # λ ≥ 0.63 μm (optical/IR regime)
        atten_ir_regime = Loki.extinction_calzetti(0.8u"μm", 1.0)
        @test isfinite(atten_ir_regime)
        @test 0 < atten_ir_regime < 1

        # Angstrom input gives same result as μm input (Unitful conversion)
        λ_um = 0.55u"μm"
        λ_aa = uconvert(u"angstrom", λ_um)
        @test Loki.extinction_calzetti(λ_um, 1.0) ≈ Loki.extinction_calzetti(λ_aa, 1.0) rtol=1e-10
    end

    @testset "extinction_calzetti (UV bump variant)" begin

        # The 3-argument form adds a UV bump via Kriek & Conroy (2013)
        # Eb = 0.85 - 1.9*δ_uv; larger δ_uv → smaller (or negative) bump
        λ = 0.3u"μm"
        E_BV = 1.0

        # Result should still be a valid attenuation (positive, finite)
        for δ_uv in [0.0, 0.3, 0.5, 0.85/1.9]
            atten = Loki.extinction_calzetti(λ, E_BV, δ_uv)
            @test isfinite(atten)
            @test atten > 0
        end

        # δ_uv=0 gives strongest UV bump (Eb=0.85); larger δ_uv → less UV bump
        # Both must be positive for E_BV=0
        @test Loki.extinction_calzetti(λ, 0.0, 0.0)  ≈ 1.0 rtol=1e-12
        @test Loki.extinction_calzetti(λ, 0.0, 0.5)  ≈ 1.0 rtol=1e-12
    end

    # =========================================================================
    # extinction_cardelli(λ, E_BV; Rv=3.10)
    # Cardelli et al. (1989) Milky Way extinction curve.
    # 4 wavelength regimes based on x = 1/λ_μm:
    #   IR (x < 1.1), Optical/NIR (1.1–3.3), Mid-UV (3.3–8), Far-UV (8–11)
    # attenuation = 10^(-0.4 * Rv * E_BV * (a+b/Rv))
    # Out of range (x > 11): returns 1 with a warning
    # =========================================================================
    @testset "extinction_cardelli" begin

        # No dust: attenuation = 1 everywhere
        for λ_val in [0.15, 0.3, 0.55, 1.0, 2.0]
            @test Loki.extinction_cardelli(λ_val * u"μm", 0.0) ≈ 1.0 rtol=1e-12
        end

        # Higher E_BV → lower attenuation
        atten_lo = Loki.extinction_cardelli(0.55u"μm", 0.5)
        atten_hi = Loki.extinction_cardelli(0.55u"μm", 1.5)
        @test atten_hi < atten_lo < 1.0

        # UV has more extinction than optical (x_UV > x_optical)
        atten_uv  = Loki.extinction_cardelli(0.15u"μm", 1.0)   # x = 6.67: Mid-UV
        atten_opt = Loki.extinction_cardelli(0.55u"μm", 1.0)   # x = 1.82: Optical
        atten_nir = Loki.extinction_cardelli(1.0u"μm",  1.0)   # x = 1.0:  IR
        @test atten_uv < atten_opt < atten_nir

        # All 4 piecewise regimes return finite positive values
        # IR: x in (0.3, 1.1) → λ in (0.91, 3.33 μm)
        atten_ir = Loki.extinction_cardelli(1.5u"μm", 1.0)
        @test isfinite(atten_ir) && atten_ir > 0

        # Optical/NIR: x in [1.1, 3.3) → λ in (0.30, 0.91 μm)
        atten_optical = Loki.extinction_cardelli(0.55u"μm", 1.0)
        @test isfinite(atten_optical) && atten_optical > 0

        # Mid-UV: x in [3.3, 8) → λ in (0.125, 0.30 μm)
        atten_muv = Loki.extinction_cardelli(0.2u"μm", 1.0)
        @test isfinite(atten_muv) && atten_muv > 0

        # Far-UV: x in [8, 11] → λ in (0.091, 0.125 μm) = (910, 1250 Å)
        atten_fuv = Loki.extinction_cardelli(0.1u"μm", 1.0)
        @test isfinite(atten_fuv) && atten_fuv > 0

        # Out-of-range (x > 11, i.e. λ < 0.091 μm): returns 1 with a warning
        # @warn uses Julia's logging system, so we check the return value and
        # separately verify the warning is issued via @test_logs
        @test_logs (:warn, r"outside the allowable range") Loki.extinction_cardelli(0.08u"μm", 1.0)
        @test Loki.extinction_cardelli(0.08u"μm", 1.0) ≈ 1.0

        # Attenuation is in (0, 1] for E_BV ≥ 0 in the valid range
        for λ_val in [0.15, 0.55, 1.0, 2.0]
            for ebv in [0.0, 0.5, 1.0]
                atten = Loki.extinction_cardelli(λ_val * u"μm", ebv)
                @test 0 < atten ≤ 1
            end
        end

        # Angstrom input: same result after unit conversion
        λ_um = 0.55u"μm"
        λ_aa = uconvert(u"angstrom", λ_um)
        @test Loki.extinction_cardelli(λ_um, 1.0) ≈ Loki.extinction_cardelli(λ_aa, 1.0) rtol=1e-10

        # Out-of-range wavelength (λ > 3.33 μm, x < 0.3): outside IR regime, also returns 1
        @test Loki.extinction_cardelli(5.0u"μm", 1.0) ≈ 1.0 atol=1e-6
    end

end
