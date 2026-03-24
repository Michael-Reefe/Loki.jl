###############################################################################
# Tests for profile integral functions
# Source: src/util/math.jl  (lines ~353–434)
#
# These functions compute the analytic area under spectral-line and dust
# feature profiles given an amplitude A and a full-width at half-maximum FWHM.
# They are all pure functions of plain numbers — no file I/O or Unitful needed.
#
# Each function also has a 4- or 6-argument version that returns both the
# integral value AND its uncertainty via error propagation.
###############################################################################

@testset "Profile integrals" begin

    # -------------------------------------------------------------------------
    # ∫Gaussian  =  √(π / (4 log 2)) * A * FWHM
    # -------------------------------------------------------------------------
    @testset "∫Gaussian" begin
        # Docstring examples
        @test Loki.∫Gaussian(1000.0, 0.5) ≈ 532.2335097156131 rtol=1e-10
        @test Loki.∫Gaussian(600.0,  1.2) ≈ 766.4162539904829 rtol=1e-10

        # Linearity in A
        @test Loki.∫Gaussian(2000.0, 0.5) ≈ 2 * Loki.∫Gaussian(1000.0, 0.5) rtol=1e-12

        # Linearity in FWHM
        @test Loki.∫Gaussian(1000.0, 1.0) ≈ 2 * Loki.∫Gaussian(1000.0, 0.5) rtol=1e-12

        # Zero amplitude → zero integral
        @test Loki.∫Gaussian(0.0, 0.5) == 0.0

        # Error-propagation overload: zero errors → zero uncertainty output
        val, err = Loki.∫Gaussian(1000.0, 0.0, 0.5, 0.0)
        @test val ≈ Loki.∫Gaussian(1000.0, 0.5) rtol=1e-12
        @test err ≈ 0.0 atol=1e-14

        # Uncertainty scales with A_err
        _, err_A = Loki.∫Gaussian(1000.0, 10.0, 0.5, 0.0)
        @test err_A > 0

        # Uncertainty scales with FWHM_err
        _, err_F = Loki.∫Gaussian(1000.0, 0.0, 0.5, 0.01)
        @test err_F > 0
    end

    # -------------------------------------------------------------------------
    # ∫Lorentzian  =  (π/2) * A * FWHM
    # -------------------------------------------------------------------------
    @testset "∫Lorentzian" begin
        # Docstring examples
        @test Loki.∫Lorentzian(1000.0, 0.5) ≈ 785.3981633974482 rtol=1e-10
        @test Loki.∫Lorentzian(600.0,  1.2) ≈ 1130.9733552923256 rtol=1e-10

        # Linearity
        @test Loki.∫Lorentzian(2000.0, 0.5) ≈ 2 * Loki.∫Lorentzian(1000.0, 0.5) rtol=1e-12

        # Zero amplitude
        @test Loki.∫Lorentzian(0.0, 0.5) == 0.0

        # Error overload: zero errors → zero uncertainty
        val, err = Loki.∫Lorentzian(1000.0, 0.0, 0.5, 0.0)
        @test val ≈ Loki.∫Lorentzian(1000.0, 0.5) rtol=1e-12
        @test err ≈ 0.0 atol=1e-14
    end

    # -------------------------------------------------------------------------
    # ∫Voigt  (pseudo-Voigt mixing parameter η)
    #   η=1 → pure Gaussian   ⟹  ∫Voigt(A,FWHM,1) == ∫Gaussian(A,FWHM)
    #   η=0 → pure Lorentzian ⟹  ∫Voigt(A,FWHM,0) == ∫Lorentzian(A,FWHM)
    # -------------------------------------------------------------------------
    @testset "∫Voigt" begin
        # Docstring examples
        @test Loki.∫Voigt(1000.0, 0.5, 1.0) ≈ 532.233509715613  rtol=1e-10
        @test Loki.∫Voigt(600.0,  1.2, 0.0) ≈ 1130.9733552923256 rtol=1e-10

        # Limiting cases
        @test Loki.∫Voigt(1000.0, 0.5, 1.0) ≈ Loki.∫Gaussian(1000.0, 0.5)   rtol=1e-12
        @test Loki.∫Voigt(1000.0, 0.5, 0.0) ≈ Loki.∫Lorentzian(1000.0, 0.5) rtol=1e-12

        # Intermediate η is between the two limits
        I_G = Loki.∫Gaussian(1000.0, 0.5)
        I_L = Loki.∫Lorentzian(1000.0, 0.5)
        I_mix = Loki.∫Voigt(1000.0, 0.5, 0.5)
        @test min(I_G, I_L) < I_mix < max(I_G, I_L)

        # Error overload: zero errors → zero uncertainty
        val, err = Loki.∫Voigt(1000.0, 0.0, 0.5, 0.0, 0.5, 0.0)
        @test val ≈ Loki.∫Voigt(1000.0, 0.5, 0.5) rtol=1e-12
        @test err ≈ 0.0 atol=1e-14
    end

    # -------------------------------------------------------------------------
    # ∫Drude  =  (π/2) * A * FWHM  (identical formula to Lorentzian)
    # -------------------------------------------------------------------------
    @testset "∫Drude" begin
        # Docstring examples
        @test Loki.∫Drude(1000.0, 0.5) ≈ 785.3981633974482 rtol=1e-10
        @test Loki.∫Drude(600.0,  1.2) ≈ 1130.9733552923256 rtol=1e-10

        # Identical formula to ∫Lorentzian
        @test Loki.∫Drude(500.0, 0.8) ≈ Loki.∫Lorentzian(500.0, 0.8) rtol=1e-12

        # Zero amplitude
        @test Loki.∫Drude(0.0, 0.5) == 0.0

        # Error overload: zero errors → zero uncertainty
        val, err = Loki.∫Drude(1000.0, 0.0, 0.5, 0.0)
        @test val ≈ Loki.∫Drude(1000.0, 0.5) rtol=1e-12
        @test err ≈ 0.0 atol=1e-14
    end

    # -------------------------------------------------------------------------
    # ∫PearsonIV  (returns a positive area for sensible parameters)
    # -------------------------------------------------------------------------
    @testset "∫PearsonIV" begin
        # For typical PAH-fitting parameters: A>0, a>0, m>1, ν real
        # The integral should be positive
        A, a, m, ν = 500.0, 0.3, 2.5, 0.0
        @test Loki.∫PearsonIV(A, a, m, ν) > 0

        # Linearity in A
        @test Loki.∫PearsonIV(2A, a, m, ν) ≈ 2 * Loki.∫PearsonIV(A, a, m, ν) rtol=1e-12

        # Zero amplitude → zero integral
        @test Loki.∫PearsonIV(0.0, a, m, ν) == 0.0
    end

end
