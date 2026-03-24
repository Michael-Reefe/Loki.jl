###############################################################################
# Tests for statistical / mathematical helper functions
# Source: src/util/math.jl
#
# Functions tested:
#   ln_likelihood  — Gaussian log-likelihood (lines ~437–450)
#   hermite        — Hermite polynomial recurrence (lines ~453–477)
#   F_test         — Statistical F-test for model comparison (lines ~286–305)
###############################################################################

@testset "Statistical and math helpers" begin

    # =========================================================================
    # ln_likelihood(data, model, err)
    # = -0.5 * Σ [ (data - model)² / err² + log(2π err²) ]
    # =========================================================================
    @testset "ln_likelihood" begin
        data  = [1.1, 1.9, 3.2]
        model = [1.0, 2.0, 3.0]
        err   = [0.1, 0.1, 0.1]

        # Docstring example
        @test Loki.ln_likelihood(data, model, err) ≈ 1.1509396793681144 rtol=1e-10

        # Perfect model (data == model): maximum possible likelihood for these errors
        ll_perfect = Loki.ln_likelihood(data, data, err)
        @test ll_perfect > Loki.ln_likelihood(data, model, err)

        # Effect of error scale on the perfect-model log-likelihood:
        # ln_L = -0.5 * Σ [ (d-m)²/err² + log(2π err²) ]
        # At the perfect model (d==m), only the log term survives:
        # ln_L_perfect = -0.5 * n * log(2π err²)
        # Larger errors → larger log(err²) → MORE NEGATIVE log-likelihood.
        err_big = 10 .* err
        ll_perfect_small_err = Loki.ln_likelihood(data, data, err)
        ll_perfect_big_err   = Loki.ln_likelihood(data, data, err_big)
        @test ll_perfect_small_err > ll_perfect_big_err  # smaller err → higher peak LL

        # Single-element arrays
        @test Loki.ln_likelihood([1.0], [1.0], [0.1]) ≈ Loki.ln_likelihood([1.0], [1.0], [0.1])

        # ln_likelihood is finite for sensible inputs
        @test isfinite(Loki.ln_likelihood(data, model, err))

        # Larger residuals → lower likelihood
        model_bad = [0.0, 0.0, 0.0]
        @test Loki.ln_likelihood(data, model, err) > Loki.ln_likelihood(data, model_bad, err)
    end

    # =========================================================================
    # hermite(x, n)  — probabilist's Hermite polynomials via recurrence
    # H₀(x) = 1,  H₁(x) = 2x,  H₂(x) = 4x²−2,  H₃(x) = 8x³−12x, ...
    # =========================================================================
    @testset "hermite" begin
        # Docstring examples
        @test Loki.hermite([0.0], 1) ≈ [0.0]  rtol=1e-12
        @test Loki.hermite([1.0], 2) ≈ [2.0]  rtol=1e-12
        @test Loki.hermite([2.0], 3) ≈ [40.0] rtol=1e-12

        # Order 0: H₀(x) = 1 for any x
        for x_val in [-2.0, -1.0, 0.0, 1.0, 5.0]
            @test Loki.hermite([x_val], 0) ≈ [1.0] rtol=1e-12
        end

        # Order 1: H₁(x) = 2x
        for x_val in [-3.0, -1.0, 0.0, 1.0, 3.0]
            @test Loki.hermite([x_val], 1) ≈ [2 * x_val] rtol=1e-12
        end

        # Order 2: H₂(x) = 4x² − 2
        for x_val in [-2.0, 0.0, 1.5]
            @test Loki.hermite([x_val], 2) ≈ [4*x_val^2 - 2] rtol=1e-10
        end

        # Order 3: H₃(x) = 8x³ − 12x
        for x_val in [-1.0, 0.0, 1.0, 2.0]
            @test Loki.hermite([x_val], 3) ≈ [8*x_val^3 - 12*x_val] rtol=1e-10
        end

        # Recurrence relation: H_n(x) = 2x·H_{n-1}(x) - 2(n-1)·H_{n-2}(x)
        x_vec = [1.5]
        for n in 3:6
            Hn   = Loki.hermite(x_vec, n)
            Hn_1 = Loki.hermite(x_vec, n-1)
            Hn_2 = Loki.hermite(x_vec, n-2)
            expected = 2 .* x_vec .* Hn_1 .- 2*(n-1) .* Hn_2
            @test Hn ≈ expected rtol=1e-10
        end

        # hermite works on a vector of x values
        x_multi = [-1.0, 0.0, 1.0, 2.0]
        result = Loki.hermite(x_multi, 2)
        @test length(result) == length(x_multi)
        @test result ≈ 4 .* x_multi .^ 2 .- 2 rtol=1e-10
    end

    # =========================================================================
    # F_test(n, p1, p2, χ1, χ2, threshold)
    # Tests whether adding parameters (p1 → p2) significantly improves the fit.
    # Returns (passed::Bool, F_data::Float, F_crit::Float)
    # =========================================================================
    @testset "F_test" begin
        n  = 100   # number of data points
        p1 = 3     # free parameters in simpler model
        p2 = 5     # free parameters in complex model

        # A dramatically better model should pass at any reasonable threshold
        χ1_big = 200.0   # simple model: poor fit
        χ2_small = 50.0  # complex model: much better fit
        passed, F_data, F_crit = Loki.F_test(n, p1, p2, χ1_big, χ2_small, 0.05)
        @test passed == true
        @test F_data > F_crit

        # The same chi-squared: no improvement → F_data = 0 → should NOT pass
        χ_same = 80.0
        passed_same, F_data_same, F_crit_same = Loki.F_test(n, p1, p2, χ_same, χ_same, 0.05)
        @test passed_same == false
        @test F_data_same ≈ 0.0 atol=1e-12

        # F_data > F_crit is always consistent with the Bool return value
        for (χ1, χ2) in [(200.0, 50.0), (100.0, 95.0), (80.0, 80.0)]
            pass, Fd, Fc = Loki.F_test(n, p1, p2, χ1, χ2, 0.05)
            @test pass == (Fd > Fc)
        end

        # Stricter threshold (smaller probability → higher F_crit required)
        pass_loose,  _, F_crit_loose  = Loki.F_test(n, p1, p2, 150.0, 80.0, 0.1)
        pass_strict, _, F_crit_strict = Loki.F_test(n, p1, p2, 150.0, 80.0, 0.001)
        @test F_crit_strict > F_crit_loose

        # Return type check: F_data and F_crit are finite real numbers
        _, Fd2, Fc2 = Loki.F_test(n, p1, p2, 150.0, 80.0, 0.05)
        @test isfinite(Fd2)
        @test isfinite(Fc2)
    end

end
