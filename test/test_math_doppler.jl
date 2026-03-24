###############################################################################
# Tests for Doppler shift and width functions
# Source: src/util/math.jl
#
# These functions convert between wavelength and velocity using special
# relativity (shift functions) or the low-velocity approximation (width
# functions).  All four are exported from the Loki module.
#
# NOTE on units: the internal speed-of-light constant C_KMS has units of
# km/s (a Unitful quantity), so:
#   - Doppler_shift_λ expects velocity in km/s (Unitful)
#   - Doppler_shift_v returns velocity in km/s (Unitful)
#   - Doppler_width_λ expects velocity in km/s (Unitful)
#   - Doppler_width_v returns velocity in km/s (Unitful)
# Wavelengths can be in μm or angstroms (Unitful) or plain numbers.
###############################################################################

@testset "Doppler shift functions" begin

    # --- Doppler_shift_v ---
    # Shift from λ to v: works with plain numbers because λ/λ₀ is dimensionless
    # (the km/s units of C_KMS propagate into the result)

    @testset "Doppler_shift_v — zero shift" begin
        v = Doppler_shift_v(10.0, 10.0)
        @test ustrip(uconvert(u"km/s", v)) ≈ 0.0 atol=1e-10
    end

    @testset "Doppler_shift_v — docstring examples" begin
        # Redshift (λ > λ₀): positive velocity
        v1 = Doppler_shift_v(10.1, 10.0)
        @test ustrip(uconvert(u"km/s", v1)) ≈ 2982.9356991238064 rtol=1e-10

        # Blueshift (λ < λ₀): negative velocity
        v2 = Doppler_shift_v(9.9, 10.0)
        @test ustrip(uconvert(u"km/s", v2)) ≈ -3012.9134458865756 rtol=1e-10
    end

    @testset "Doppler_shift_v — physical sense" begin
        # Redshifted wavelength → positive recession velocity
        @test ustrip(Doppler_shift_v(11.0, 10.0)) > 0
        # Blueshifted wavelength → negative recession velocity
        @test ustrip(Doppler_shift_v(9.0, 10.0)) < 0
    end

    # --- Doppler_shift_λ ---
    # Convert rest-frame wavelength + velocity → observed wavelength.
    # Requires Unitful velocity (km/s) because C_KMS is a Unitful quantity.

    @testset "Doppler_shift_λ — zero velocity" begin
        λ_obs = Doppler_shift_λ(10.0u"μm", 0.0u"km/s")
        @test ustrip(uconvert(u"μm", λ_obs)) ≈ 10.0 atol=1e-10
    end

    @testset "Doppler_shift_λ — docstring values (Unitful calling convention)" begin
        # These reproduce the documented numeric values using proper Unitful units
        λ1 = Doppler_shift_λ(10.0u"μm", 100.0u"km/s")
        @test ustrip(uconvert(u"μm", λ1)) ≈ 10.003336197462627 rtol=1e-10

        λ2 = Doppler_shift_λ(10.0u"μm", -100.0u"km/s")
        @test ustrip(uconvert(u"μm", λ2)) ≈ 9.996664915187521 rtol=1e-10
    end

    @testset "Doppler_shift_λ — physical sense" begin
        # Positive velocity (recession) → observed wavelength is longer (redshift)
        @test ustrip(Doppler_shift_λ(10.0u"μm",  500.0u"km/s")) > 10.0
        # Negative velocity (approach) → observed wavelength is shorter (blueshift)
        @test ustrip(Doppler_shift_λ(10.0u"μm", -500.0u"km/s")) < 10.0
    end

    # --- Round-trip: shift then recover ---
    @testset "Doppler_shift round-trip" begin
        for (λ0, v_kms) in [(5.0, 200.0), (10.0, -300.0), (2.5, 50000.0)]
            λ_obs = Doppler_shift_λ(λ0 * u"μm", v_kms * u"km/s")
            v_rec = Doppler_shift_v(λ_obs, λ0 * u"μm")
            @test ustrip(uconvert(u"km/s", v_rec)) ≈ v_kms rtol=1e-10
        end
    end

    # --- Doppler_width_v ---
    # Low-velocity approximation: Δλ/λ₀ * c

    @testset "Doppler_width_v — zero width" begin
        Δv = Doppler_width_v(0.0, 10.0)
        @test ustrip(uconvert(u"km/s", Δv)) ≈ 0.0 atol=1e-10
    end

    @testset "Doppler_width_v — docstring example" begin
        Δv = Doppler_width_v(0.1, 10.0)
        @test ustrip(uconvert(u"km/s", Δv)) ≈ 2997.92458 rtol=1e-10
    end

    # --- Doppler_width_λ ---
    # Low-velocity approximation: Δv/c * λ₀.  Requires Unitful velocity.

    @testset "Doppler_width_λ — zero width" begin
        Δλ = Doppler_width_λ(0.0u"km/s", 10.0u"μm")
        @test ustrip(uconvert(u"μm", Δλ)) ≈ 0.0 atol=1e-10
    end

    @testset "Doppler_width_λ — docstring example" begin
        Δλ = Doppler_width_λ(3000.0u"km/s", 10.0u"μm")
        @test ustrip(uconvert(u"μm", Δλ)) ≈ 0.10006922855944561 rtol=1e-10
    end

    # --- Width round-trip ---
    @testset "Doppler_width round-trip" begin
        for (λ0, Δv) in [(5.0, 100.0), (10.0, 5000.0)]
            Δλ = Doppler_width_λ(Δv * u"km/s", λ0 * u"μm")
            Δv_rec = Doppler_width_v(Δλ, λ0 * u"μm")
            @test ustrip(uconvert(u"km/s", Δv_rec)) ≈ Δv rtol=1e-10
        end
    end

end
