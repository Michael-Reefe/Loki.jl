###############################################################################
# Tests for spectral line and dust-feature profile shapes
# Source: src/util/math.jl  (lines ~776–858, 558–581, 469–477, 800–822)
#
# Profiles tested:
#   Gaussian   — symmetric bell curve, peak = A at x = μ
#   Lorentzian — heavier-tailed bell, peak = A at x = μ
#   Voigt      — mixture of Gaussian and Lorentzian; A controls total *flux*,
#                NOT peak amplitude (the peak height depends on FWHM and η)
#   Drude      — used for PAH dust features; peak = A at x = μ (asym=0);
#                asymmetric version also tested (asym≠0)
#   PearsonIV  — generalized PAH profile; peak = A at x = μ when ν=0;
#                asymmetric version also tested (ν≠0)
#   hermite    — Hermite polynomial helper (vector-only; scalar dispatch missing)
#
# All functions accept plain Float64 numbers — no Unitful needed.
###############################################################################

@testset "Line profile shapes" begin

    # =========================================================================
    # Gaussian
    # =========================================================================
    @testset "Gaussian" begin
        A, μ, FWHM = 500.0, 10.0, 0.5

        # Peak value at center
        @test Loki.Gaussian(μ, A, μ, FWHM) ≈ A rtol=1e-12

        # Half-maximum at x = μ ± FWHM/2
        half = Loki.Gaussian(μ + FWHM/2, A, μ, FWHM)
        @test half ≈ A/2 rtol=1e-10

        # Symmetry about μ
        for Δx in [0.1, 0.25, FWHM, 2*FWHM]
            @test Loki.Gaussian(μ + Δx, A, μ, FWHM) ≈ Loki.Gaussian(μ - Δx, A, μ, FWHM) rtol=1e-12
        end

        # Zero amplitude → zero everywhere
        for x in [μ, μ+1, μ-1]
            @test Loki.Gaussian(x, 0.0, μ, FWHM) == 0.0
        end

        # Profile decays away from center
        @test Loki.Gaussian(μ + FWHM, A, μ, FWHM) < Loki.Gaussian(μ, A, μ, FWHM)
        @test Loki.Gaussian(μ + 2*FWHM, A, μ, FWHM) < Loki.Gaussian(μ + FWHM, A, μ, FWHM)

        # Numerical integral matches analytic ∫Gaussian
        # (uses a fine grid over ±10σ; adequate for 3-digit accuracy)
        x_grid = range(μ - 15*FWHM, μ + 15*FWHM, length=100_000)
        dx = step(x_grid)
        numerical_integral = sum(Loki.Gaussian.(x_grid, A, μ, FWHM)) * dx
        @test numerical_integral ≈ Loki.∫Gaussian(A, FWHM) rtol=1e-4
    end

    # =========================================================================
    # Lorentzian
    # =========================================================================
    @testset "Lorentzian" begin
        A, μ, FWHM = 500.0, 10.0, 0.5

        # Peak value at center
        @test Loki.Lorentzian(μ, A, μ, FWHM) ≈ A rtol=1e-12

        # Half-maximum at x = μ ± FWHM/2
        @test Loki.Lorentzian(μ + FWHM/2, A, μ, FWHM) ≈ A/2 rtol=1e-12

        # Symmetry about μ
        for Δx in [0.1, 0.25, FWHM, 2*FWHM]
            @test Loki.Lorentzian(μ + Δx, A, μ, FWHM) ≈ Loki.Lorentzian(μ - Δx, A, μ, FWHM) rtol=1e-12
        end

        # Zero amplitude → zero everywhere
        @test Loki.Lorentzian(μ, 0.0, μ, FWHM) == 0.0

        # Profile decays away from center
        @test Loki.Lorentzian(μ + FWHM, A, μ, FWHM) < Loki.Lorentzian(μ, A, μ, FWHM)

        # Lorentzian has heavier tails than Gaussian at the same A/FWHM
        x_far = μ + 5*FWHM
        @test Loki.Lorentzian(x_far, A, μ, FWHM) > Loki.Gaussian(x_far, A, μ, FWHM)
    end

    # =========================================================================
    # Voigt  (pseudo-Voigt: linear mix of Gaussian and Lorentzian pdfs)
    #
    # Important: in Loki's Voigt(x, A, μ, FWHM, η), the parameter A controls
    # the *integral* (total flux), NOT the peak amplitude.
    # Voigt(μ, A, μ, FWHM, η)  ≠  A  in general.
    # =========================================================================
    @testset "Voigt" begin
        A, μ, FWHM = 1000.0, 10.0, 0.5

        # Voigt is always non-negative for non-negative A
        x_test = range(μ - 5*FWHM, μ + 5*FWHM, length=50)
        for η in [0.0, 0.3, 0.7, 1.0]
            @test all(Loki.Voigt.(x_test, A, μ, FWHM, η) .≥ 0)
        end

        # Symmetry about μ (both Gaussian and Lorentzian components are symmetric)
        for η in [0.0, 0.5, 1.0]
            for Δx in [0.1, 0.25, FWHM]
                @test Loki.Voigt(μ + Δx, A, μ, FWHM, η) ≈ Loki.Voigt(μ - Δx, A, μ, FWHM, η) rtol=1e-12
            end
        end

        # Zero amplitude → zero
        @test Loki.Voigt(μ, 0.0, μ, FWHM, 0.5) == 0.0

        # Numerical integral ≈ ∫Voigt for each η.
        # The Lorentzian (η→0) has very heavy tails: a simple trapezoidal sum
        # over a finite range introduces ~0.3% truncation error, so we allow
        # rtol=0.005 (0.5%) here.  The analytic ∫Voigt formula is verified
        # more rigorously by the docstring examples in test_math_profiles.jl.
        for (η, half_range) in [(1.0, 20*FWHM), (0.5, 50*FWHM), (0.0, 200*FWHM)]
            x_grid = range(μ - half_range, μ + half_range, length=200_000)
            dx = step(x_grid)
            integral = sum(Loki.Voigt.(x_grid, A, μ, FWHM, η)) * dx
            @test integral ≈ Loki.∫Voigt(A, FWHM, η) rtol=5e-3
        end
    end

    # =========================================================================
    # Drude  (used for PAH emission features)
    # For asym=0 the profile is symmetric and peaks at A.
    # =========================================================================
    @testset "Drude" begin
        A, μ, FWHM = 500.0, 10.0, 0.5

        # Peak value at center with no asymmetry
        @test Loki.Drude(μ, A, μ, FWHM, 0) ≈ A rtol=1e-12

        # Zero amplitude → zero
        @test Loki.Drude(μ, 0.0, μ, FWHM, 0) == 0.0

        # Profile is always non-negative
        x_test = range(μ - 3*FWHM, μ + 3*FWHM, length=50)
        @test all(Loki.Drude.(x_test, A, μ, FWHM, 0) .≥ 0)

        # Profile decays away from center
        @test Loki.Drude(μ + FWHM, A, μ, FWHM, 0) < A
        @test Loki.Drude(μ + 2*FWHM, A, μ, FWHM, 0) < Loki.Drude(μ + FWHM, A, μ, FWHM, 0)
    end

    # =========================================================================
    # PearsonIV  (generalized profile; reduces cleanly when ν=0)
    # =========================================================================
    @testset "PearsonIV" begin
        A, μ, a, m = 500.0, 10.0, 0.3, 2.5

        # Peak at center when ν=0  (normalization factor n=1 when ν=0)
        @test Loki.PearsonIV(μ, A, μ, a, m, 0.0) ≈ A rtol=1e-12

        # Zero amplitude → zero
        @test Loki.PearsonIV(μ, 0.0, μ, a, m, 0.0) == 0.0

        # Profile always non-negative for A > 0
        x_test = range(μ - 5*a, μ + 5*a, length=50)
        @test all(Loki.PearsonIV.(x_test, A, μ, a, m, 0.0) .≥ 0)

        # Profile decays away from center
        @test Loki.PearsonIV(μ + a, A, μ, a, m, 0.0) < A
    end

    # =========================================================================
    # Drude — asymmetric variant (asym ≠ 0)
    # When asym=0, γ = FWHM (already tested above).
    # When asym≠0, γ is position-dependent and the profile is no longer symmetric.
    # =========================================================================
    @testset "Drude (asymmetric)" begin
        A, μ, FWHM = 500.0, 10.0, 0.5

        # asym=0: reduces to symmetric Drude (γ = FWHM at all x)
        # NOTE: the profile x/μ - μ/x is NOT an even function of (x-μ), so
        # Drude(μ+δ) ≠ Drude(μ-δ) even for asym=0 — confirm peak is still at μ.
        @test Loki.Drude(μ, A, μ, FWHM, 0.0)  ≈ A rtol=1e-12  # peak = A

        # asym≠0: breaks symmetry further (profile is higher on one side)
        for δ in [0.2, 0.5, 1.0]
            @test Loki.Drude(μ + δ, A, μ, FWHM, 1.0) ≉
                  Loki.Drude(μ - δ, A, μ, FWHM, 1.0)
        end

        # Peak is still at x=μ regardless of asym
        # At x=μ: γ = 2FWHM/(1+exp(0)) = FWHM and (x/μ - μ/x)=0, so Drude(μ)=A
        @test Loki.Drude(μ, A, μ, FWHM, 1.0)  ≈ A rtol=1e-12
        @test Loki.Drude(μ, A, μ, FWHM, -1.0) ≈ A rtol=1e-12

        # Always non-negative for A > 0, any asym
        x_test = range(μ - 3*FWHM, μ + 3*FWHM, length=50)
        @test all(Loki.Drude.(x_test, A, μ, FWHM,  0.5) .≥ 0)
        @test all(Loki.Drude.(x_test, A, μ, FWHM, -0.5) .≥ 0)

        # Different asym signs produce different (mirrored) profiles
        @test Loki.Drude(μ + 0.3, A, μ, FWHM, 1.0) ≠
              Loki.Drude(μ + 0.3, A, μ, FWHM, -1.0)
    end

    # =========================================================================
    # PearsonIV — asymmetric variant (ν ≠ 0)
    # When ν=0, profile is symmetric; ν≠0 introduces skewness.
    # =========================================================================
    @testset "PearsonIV (asymmetric)" begin
        A, μ, a, m = 500.0, 10.0, 0.3, 2.5

        # ν=0: strictly symmetric about μ
        for δ in [0.1, 0.3, 0.6]
            @test Loki.PearsonIV(μ + δ, A, μ, a, m, 0.0) ≈
                  Loki.PearsonIV(μ - δ, A, μ, a, m, 0.0) rtol=1e-12
        end

        # ν≠0: breaks symmetry
        for δ in [0.1, 0.3, 0.6]
            @test Loki.PearsonIV(μ + δ, A, μ, a, m, 1.0) ≉
                  Loki.PearsonIV(μ - δ, A, μ, a, m, 1.0)
        end

        # Always non-negative for A > 0
        x_test = range(μ - 5*a, μ + 5*a, length=50)
        @test all(Loki.PearsonIV.(x_test, A, μ, a, m,  1.0) .≥ 0)
        @test all(Loki.PearsonIV.(x_test, A, μ, a, m, -1.0) .≥ 0)

        # Always finite
        @test all(isfinite.(Loki.PearsonIV.(x_test, A, μ, a, m, 2.0)))

        # Positive and negative ν produce different profiles
        @test Loki.PearsonIV(μ + 0.5*a, A, μ, a, m, 1.0) ≠
              Loki.PearsonIV(μ + 0.5*a, A, μ, a, m, -1.0)
    end

    # =========================================================================
    # hermite(x::Vector, n::Integer)
    # Hermite polynomial recurrence: H_0=1, H_1=2x, H_n=2x*H_{n-1}-2(n-1)*H_{n-2}
    # NOTE: Only a Vector method exists; scalar dispatch is absent.
    # =========================================================================
    @testset "hermite polynomial" begin

        # H_0(x) = 1 for any x
        @test Loki.hermite([0.0], 0) ≈ [1.0]
        @test Loki.hermite([3.7], 0) ≈ [1.0]

        # H_1(x) = 2x
        @test Loki.hermite([0.0], 1) ≈ [0.0]
        @test Loki.hermite([1.0], 1) ≈ [2.0]
        @test Loki.hermite([-1.0], 1) ≈ [-2.0]

        # H_2(x) = 4x² - 2
        @test Loki.hermite([0.0], 2) ≈ [-2.0]  # 4*0 - 2 = -2
        @test Loki.hermite([1.0], 2) ≈ [2.0]   # 4*1 - 2 = 2

        # H_3(2) = 8*8 - 12*2 = 40  (from docstring example)
        @test Loki.hermite([2.0], 3) ≈ [40.0]

        # Vectorized: H_1([0,1,2]) = [0,2,4]
        @test Loki.hermite([0.0, 1.0, 2.0], 1) ≈ [0.0, 2.0, 4.0]

        # Recurrence relation: H_n = 2x*H_{n-1} - 2*(n-1)*H_{n-2}
        x = [1.5]
        for n in 2:5
            h_n   = Loki.hermite(x, n)
            h_nm1 = Loki.hermite(x, n-1)
            h_nm2 = Loki.hermite(x, n-2)
            @test h_n ≈ 2 .* x .* h_nm1 .- 2*(n-1) .* h_nm2 rtol=1e-12
        end

        # Output length equals input length
        xs = [0.0, 1.0, -1.0, 2.0, 0.5]
        @test length(Loki.hermite(xs, 4)) == length(xs)
    end

end
