###############################################################################
# Tests for continuum physics functions: blackbody, Wien, and power law
# Source: src/util/math.jl  (lines ~480–537)
#
# The Blackbody and Wien functions use Unitful quantities throughout:
#   Wavelengths are in μm  (Unitful type Qum = typeof(1.0u"μm"))
#   Temperatures are in K   (Unitful type QTemp = typeof(1.0u"K"))
#   The Blackbody functions return spectral intensity with appropriate units.
#
# power_law accepts Unitful wavelength vectors (ratio λ/ref_λ is dimensionless).
#
# Also tested:
#   Blackbody dispatch wrapper — routes to Blackbody_ν or Blackbody_λ by Iunit type
#   Blackbody_modified         — adds dust emissivity ∝ (9.7μm/λ)² factor
###############################################################################

@testset "Continuum functions" begin

    # =========================================================================
    # power_law(λ, α, ref_λ)  =  (λ / ref_λ)^α
    # Reference wavelength always gives 1.0 regardless of exponent.
    # =========================================================================
    @testset "power_law" begin
        ref = 9.7u"μm"

        # Normalization: power law equals 1 at the reference wavelength for any exponent
        for α in [-2.0, 0.0, 1.0, 3.5]
            @test Loki.power_law([ref], α, ref) ≈ [1.0] rtol=1e-12
        end

        # Linear exponent: doubling the wavelength doubles the power law
        @test Loki.power_law([2*ref], 1.0, ref) ≈ [2.0] rtol=1e-12

        # Quadratic exponent: doubling wavelength quadruples power law
        @test Loki.power_law([2*ref], 2.0, ref) ≈ [4.0] rtol=1e-12

        # Negative exponent: larger wavelength → smaller value
        vals = Loki.power_law([1*ref, 2*ref, 4*ref], -1.0, ref)
        @test vals[1] > vals[2] > vals[3]

        # Zero exponent: flat (all ones)
        vals_flat = Loki.power_law([0.5*ref, ref, 2*ref, 4*ref], 0.0, ref)
        @test all(vals_flat .≈ 1.0)

        # Output is always non-negative for real exponents on positive λ
        @test all(Loki.power_law([0.5*ref, ref, 2*ref], 2.0, ref) .≥ 0)
    end

    # =========================================================================
    # Blackbody_ν and Blackbody_λ
    # Physical expectations:
    #   1. Positive finite output for sensible (λ, T)
    #   2. Hotter temperature → higher flux at a given wavelength
    #   3. Peak wavelength obeys Wien's law
    # =========================================================================
    @testset "Blackbody_ν — basic properties" begin
        T1 = 1000.0u"K"
        T2 = 5000.0u"K"
        λ  = 2.0u"μm"

        B1 = Loki.Blackbody_ν(λ, T1)
        B2 = Loki.Blackbody_ν(λ, T2)

        # Positive and finite
        @test ustrip(B1) > 0
        @test isfinite(ustrip(B1))
        @test ustrip(B2) > 0

        # Higher temperature → higher flux
        @test ustrip(B2) > ustrip(B1)
    end

    @testset "Blackbody_λ — basic properties" begin
        T1 = 1000.0u"K"
        T2 = 5000.0u"K"
        λ  = 2.0u"μm"

        B1 = Loki.Blackbody_λ(λ, T1)
        B2 = Loki.Blackbody_λ(λ, T2)

        @test ustrip(B1) > 0
        @test isfinite(ustrip(B1))
        @test ustrip(B2) > ustrip(B1)
    end

    @testset "Blackbody_ν and Blackbody_λ — angstrom inputs" begin
        # Both methods (μm and angstrom) should give the same physical answer
        # after unit conversion
        T   = 3000.0u"K"
        λ_μm = 2.0u"μm"
        λ_AA = uconvert(u"angstrom", λ_μm)

        B_μm = Loki.Blackbody_ν(λ_μm, T)
        B_AA = Loki.Blackbody_ν(λ_AA, T)

        @test ustrip(uconvert(unit(B_μm), B_AA)) ≈ ustrip(B_μm) rtol=1e-10
    end

    # =========================================================================
    # Wien's displacement law: λ_peak = b / T  (b ≈ 2898 μm·K)
    # The peak of the per-wavelength blackbody should occur near Wein(T).
    # =========================================================================
    @testset "Wien displacement law" begin
        for T_val in [500.0, 1000.0, 5778.0, 10000.0]
            T = T_val * u"K"
            λ_peak = Loki.Wein(T)

            # Returns a positive wavelength in μm
            @test ustrip(uconvert(u"μm", λ_peak)) > 0

            # Classic values: hotter star → shorter peak wavelength
            # e.g. Wein(5778 K) ≈ 0.502 μm (solar peak near green)
            @test ustrip(uconvert(u"μm", λ_peak)) ≈ 2897.771955 / T_val rtol=1e-10
        end

        # Hotter objects peak at shorter wavelengths
        @test ustrip(Loki.Wein(10000.0u"K")) < ustrip(Loki.Wein(5000.0u"K"))
        @test ustrip(Loki.Wein(5000.0u"K"))  < ustrip(Loki.Wein(1000.0u"K"))
    end

    @testset "Wien peak matches Blackbody_λ maximum" begin
        # The per-wavelength Blackbody should peak close to the Wien wavelength.
        # We sample a grid around the Wien peak and check it's actually a maximum.
        T = 3000.0u"K"
        λ_wein = Loki.Wein(T)                        # expected peak in μm
        λ_val  = ustrip(uconvert(u"μm", λ_wein))

        # Sample ±50% around the Wien peak
        λ_test = collect(range(0.5*λ_val, 1.5*λ_val, length=1001)) .* u"μm"
        B_vals = [ustrip(Loki.Blackbody_λ(λ, T)) for λ in λ_test]

        idx_max = argmax(B_vals)
        λ_numerical_peak = λ_val * 0.5 + (idx_max - 1) / 1000 * λ_val

        # The numerical peak should be within 5% of Wien's law prediction
        @test abs(λ_numerical_peak - λ_val) / λ_val < 0.05
    end

    # =========================================================================
    # Blackbody dispatch wrapper: Blackbody(λ, T, Iunit)
    # Routes to Blackbody_ν when Iunit is a per-frequency type (QGeneralPerFreq),
    # and to Blackbody_λ when Iunit is a per-wavelength type (QGeneralPerWave).
    # =========================================================================
    @testset "Blackbody dispatch wrapper" begin
        λ = 2.0u"μm"
        T = 1000.0u"K"

        # Per-frequency unit → dispatches to Blackbody_ν
        Iν_unit = 1.0u"erg/s/cm^2/Hz/sr"
        Bν_wrap = Loki.Blackbody(λ, T, Iν_unit)
        Bν_direct = Loki.Blackbody_ν(λ, T)
        @test ustrip(uconvert(unit(Bν_direct), Bν_wrap)) ≈ ustrip(Bν_direct) rtol=1e-12
        @test ustrip(Bν_wrap) > 0
        @test isfinite(ustrip(Bν_wrap))

        # Per-wavelength unit → dispatches to Blackbody_λ
        Iλ_unit = 1.0u"erg/s/cm^2/μm/sr"
        Bλ_wrap = Loki.Blackbody(λ, T, Iλ_unit)
        Bλ_direct = Loki.Blackbody_λ(λ, T)
        @test ustrip(uconvert(unit(Bλ_direct), Bλ_wrap)) ≈ ustrip(Bλ_direct) rtol=1e-12
        @test ustrip(Bλ_wrap) > 0

        # Higher T → higher flux at a fixed wavelength
        Bν_hot = Loki.Blackbody(λ, 5000.0u"K", Iν_unit)
        @test ustrip(Bν_hot) > ustrip(Bν_wrap)
    end

    # =========================================================================
    # Blackbody_modified(λ, T, Iunit)
    # Multiplies Blackbody(λ, T, Iunit) by the dust emissivity factor (9.7μm/λ)².
    #   At λ = 9.7μm: factor = 1.0 → equals Blackbody
    #   At λ > 9.7μm: factor < 1 → Blackbody_modified < Blackbody
    #   At λ < 9.7μm: factor > 1 → Blackbody_modified > Blackbody
    # =========================================================================
    @testset "Blackbody_modified emissivity" begin
        T     = 1000.0u"K"
        Iunit = 1.0u"erg/s/cm^2/Hz/sr"

        # At λ = 9.7μm: emissivity factor = (9.7/9.7)² = 1 → equals Blackbody
        λ_ref  = 9.7u"μm"
        Bmod_ref = Loki.Blackbody_modified(λ_ref, T, Iunit)
        Bbb_ref  = Loki.Blackbody(λ_ref, T, Iunit)
        @test ustrip(Bmod_ref) ≈ ustrip(Bbb_ref) rtol=1e-12

        # At λ > 9.7μm: factor < 1 → Blackbody_modified < Blackbody
        λ_long  = 20.0u"μm"
        Bmod_lo = Loki.Blackbody_modified(λ_long, T, Iunit)
        Bbb_lo  = Loki.Blackbody(λ_long, T, Iunit)
        @test ustrip(Bmod_lo) < ustrip(Bbb_lo)

        # At λ < 9.7μm: factor > 1 → Blackbody_modified > Blackbody
        λ_short = 5.0u"μm"
        Bmod_hi = Loki.Blackbody_modified(λ_short, T, Iunit)
        Bbb_hi  = Loki.Blackbody(λ_short, T, Iunit)
        @test ustrip(Bmod_hi) > ustrip(Bbb_hi)

        # Verify exact emissivity ratio: Bmod / Bbb = (9.7/λ)² at each wavelength
        for λ_val in [5.0, 9.7, 15.0, 25.0]
            λ    = λ_val * u"μm"
            Bmod = Loki.Blackbody_modified(λ, T, Iunit)
            Bbb  = Loki.Blackbody(λ, T, Iunit)
            @test ustrip(Bmod) / ustrip(Bbb) ≈ (9.7 / λ_val)^2 rtol=1e-12
        end

        # Always positive and finite
        for λ_val in [1.0, 5.0, 9.7, 15.0, 25.0]
            Bmod = Loki.Blackbody_modified(λ_val * u"μm", T, Iunit)
            @test ustrip(Bmod) > 0
            @test isfinite(ustrip(Bmod))
        end

        # Also works with per-wavelength Iunit
        Iλ_unit  = 1.0u"erg/s/cm^2/μm/sr"
        Bmod_λ   = Loki.Blackbody_modified(λ_ref, T, Iλ_unit)
        Bbb_λ    = Loki.Blackbody(λ_ref, T, Iλ_unit)
        @test ustrip(Bmod_λ) ≈ ustrip(Bbb_λ) rtol=1e-12
    end

end
