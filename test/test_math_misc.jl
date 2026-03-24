###############################################################################
# Tests for miscellaneous math utility functions
# Source: src/util/math.jl
#
# Functions tested:
#   sumdim               — NaN-aware sum along specified dimensions (line ~118)
#   extend / extendp     — broadcast 1D array into higher dimensions (lines ~150, 189)
#   fluxconvert          — convert between per-frequency and per-wavelength flux (lines ~265–270)
#   match_fluxunits      — smart unit matching for flux arrays (lines ~274–283)
#   convolveGaussian1D   — variable-FWHM Gaussian convolution (lines ~319–350)
#   multiplicative_exponentials — 4-component multiplicative continuum (lines ~1249–1258)
###############################################################################

@testset "Miscellaneous math utilities" begin

    # =========================================================================
    # sumdim(array, dims; nan=true)
    # Sums array along dims, dropping those dimensions; NaN-aware by default.
    # =========================================================================
    @testset "sumdim" begin

        # Docstring example: sumdim([1 2; 3 NaN], 1) == [4.0, 2.0]
        @test Loki.sumdim([1.0 2.0; 3.0 NaN], 1) ≈ [4.0, 2.0]

        # 3D array: sum along dim 1 → shape (4,5)
        A = ones(3, 4, 5)
        result = Loki.sumdim(A, 1)
        @test size(result) == (4, 5)
        @test all(result .≈ 3.0)

        # Sum along multiple dims: (1,3) of ones(3,4,5) → shape (4,) with value 15
        result2 = Loki.sumdim(A, (1, 3))
        @test size(result2) == (4,)
        @test all(result2 .≈ 15.0)

        # NaN ignored (nan=true, default): sumdim on a 1D array returns a 0-dim array;
        # use only() or [] to extract the scalar
        B = [1.0, NaN, 3.0]
        @test only(Loki.sumdim(B, 1)) ≈ 4.0

        # NaN propagated (nan=false)
        @test isnan(only(Loki.sumdim(B, 1; nan=false)))
    end

    # =========================================================================
    # extend(array1d, shape)
    # Docstring: extend([1,2,3], (4,5)) → 4×5×3 array; array varies on LAST dim.
    # =========================================================================
    @testset "extend" begin

        result = Loki.extend([1, 2, 3], (4, 5))

        # Output shape: (outer_shape..., length(array1d))
        @test size(result) == (4, 5, 3)

        # Each [:, :, k] slice equals the k-th element
        @test all(result[:, :, 1] .== 1)
        @test all(result[:, :, 2] .== 2)
        @test all(result[:, :, 3] .== 3)

        # Works with Float64 input and a 2D shape
        r2 = Loki.extend([1.0, 2.0], (3, 4))
        @test size(r2) == (3, 4, 2)
        @test all(r2[:, :, 1] .== 1.0)
        @test all(r2[:, :, 2] .== 2.0)
    end

    # =========================================================================
    # extendp(array1d, shape)
    # Docstring: extendp([1,2,3], (4,5)) → 3×4×5 array; array varies on FIRST dim.
    # =========================================================================
    @testset "extendp" begin

        result = Loki.extendp([1, 2, 3], (4, 5))

        # Output shape: (length(array1d), outer_shape...)
        @test size(result) == (3, 4, 5)

        # Each [k, :, :] slice equals the k-th element
        @test all(result[1, :, :] .== 1)
        @test all(result[2, :, :] .== 2)
        @test all(result[3, :, :] .== 3)
    end

    # =========================================================================
    # fluxconvert(Iν, λ) and fluxconvert(Iλ, λ)
    # Physical Iν ↔ Iλ conversion: Iλ = Iν * c / λ² and vice versa.
    # =========================================================================
    @testset "fluxconvert" begin

        λ = 2.0u"μm"
        Iν = 1.0u"erg/s/cm^2/Hz/sr"

        # Forward conversion: Iν → Iλ produces positive, finite result
        Iλ = Loki.fluxconvert(Iν, λ)
        @test ustrip(Iλ) > 0
        @test isfinite(ustrip(Iλ))

        # Round-trip: Iν → Iλ → Iν' should recover the original
        Iν_back = Loki.fluxconvert(Iλ, λ)
        @test ustrip(uconvert(unit(Iν), Iν_back)) ≈ ustrip(Iν) rtol=1e-10

        # Shorter wavelength → larger Iλ for same Iν (Iλ ∝ 1/λ²)
        Iλ_1 = Loki.fluxconvert(Iν, 1.0u"μm")
        Iλ_2 = Loki.fluxconvert(Iν, 2.0u"μm")
        @test ustrip(Iλ_1) > ustrip(Iλ_2)

        # Scaling: doubling Iν doubles Iλ
        Iλ_double = Loki.fluxconvert(2.0u"erg/s/cm^2/Hz/sr", λ)
        @test ustrip(Iλ_double) ≈ 2 * ustrip(Iλ) rtol=1e-12
    end

    # =========================================================================
    # convolveGaussian1D(flux, fwhm)
    # Each output pixel is the local Gaussian-weighted average.
    # =========================================================================
    @testset "convolveGaussian1D" begin

        # Output length equals input length
        flux = randn(50)
        fwhm = 2.0 * ones(50)
        result = Loki.convolveGaussian1D(flux, fwhm)
        @test length(result) == length(flux)

        # A constant signal is unchanged in the interior (edge pixels are affected
        # by zero-padding: pad_size = ceil(2*FWHM) pixels on each side are unreliable)
        const_flux = fill(5.0, 50)
        fwhm_const = 3.0 * ones(50)
        result_const = Loki.convolveGaussian1D(const_flux, fwhm_const)
        pad = ceil(Int, 2 * 3.0)   # = 6
        @test result_const[pad+1:end-pad] ≈ const_flux[pad+1:end-pad] rtol=1e-10

        # Smoothing a delta-spike spreads the energy: peak decreases
        spike = zeros(100)
        spike[50] = 1.0
        smoothed = Loki.convolveGaussian1D(spike, 5.0 * ones(100))
        @test smoothed[50] < spike[50]           # peak is reduced
        @test sum(smoothed) ≈ sum(spike) rtol=1e-3  # energy approximately conserved

        # Very small FWHM (clamped to 0.01) barely changes the signal
        small_fwhm = 1e-8 * ones(50)
        result_small = Loki.convolveGaussian1D(const_flux, small_fwhm)
        @test result_small ≈ const_flux rtol=1e-8
    end

    # =========================================================================
    # multiplicative_exponentials(λ, p)
    # Returns a (n, 4) matrix.  Each column is one exponential component:
    #   e1 = p[1]*exp(-p[2]*λ̄)        — decays with λ̄
    #   e2 = p[3]*exp(-p[4]*(1-λ̄))    — decays inversely with λ̄
    #   e3 = p[5]*(1-exp(-p[6]*λ̄))    — rises with λ̄
    #   e4 = p[7]*(1-exp(-p[8]*(1-λ̄)))— rises inversely with λ̄
    # where λ̄ = (λ - λ_min) / (λ_max - λ_min) ∈ [0,1]
    # =========================================================================
    @testset "multiplicative_exponentials" begin

        λ = collect(range(1.0, 10.0, length=20))
        p = [1.0, 2.0, 3.0, 1.5, 0.5, 3.0, 2.0, 1.0]

        result = Loki.multiplicative_exponentials(λ, p)

        # Output is (n, 4)
        @test size(result) == (length(λ), 4)

        # At λ_min (λ̄=0):
        # e1 = p[1], e2 = p[3]*exp(-p[4]), e3 = 0, e4 = p[7]*(1-exp(-p[8]))
        @test result[1, 1] ≈ p[1]                           rtol=1e-12  # e1 at λ̄=0
        @test result[1, 2] ≈ p[3]*exp(-p[4])                rtol=1e-12  # e2 at λ̄=0
        @test result[1, 3] ≈ 0.0                             atol=1e-12  # e3 at λ̄=0
        @test result[1, 4] ≈ p[7]*(1 - exp(-p[8]))          rtol=1e-12  # e4 at λ̄=0

        # At λ_max (λ̄=1):
        # e1 = p[1]*exp(-p[2]), e2 = p[3], e3 = p[5]*(1-exp(-p[6])), e4 = 0
        @test result[end, 1] ≈ p[1]*exp(-p[2])              rtol=1e-12  # e1 at λ̄=1
        @test result[end, 2] ≈ p[3]                         rtol=1e-12  # e2 at λ̄=1
        @test result[end, 3] ≈ p[5]*(1 - exp(-p[6]))        rtol=1e-12  # e3 at λ̄=1
        @test result[end, 4] ≈ 0.0                          atol=1e-12  # e4 at λ̄=1

        # All columns are non-negative for positive p values
        @test all(result .≥ 0)
    end

    # =========================================================================
    # match_fluxunits(I_mod, I_ref, λ)
    # Smart dispatch: 5 overloads covering every pair of per-freq/per-wave types.
    #
    # Type aliases (from parameters.jl):
    #   QPerAng = typeof(1.0u"erg/s/cm^2/angstrom/sr")   — per-Å wavelength
    #   QPerum  = typeof(1.0u"erg/s/cm^2/μm/sr")         — per-μm wavelength
    #   QPerFreq = typeof(1.0u"erg/s/cm^2/Hz/sr")        — per-Hz frequency
    #
    # Overloads:
    #   1. QPerAng, QPerum → uconvert (same kind, different wavelength unit)
    #   2. QPerum, QPerAng → uconvert (same kind, different wavelength unit)
    #   3. same QGeneralPerFreq → identity (no conversion)
    #   4. same QGeneralPerWave → identity (no conversion)
    #   5. mixed QGeneralPerFreq/QGeneralPerWave → fluxconvert
    # =========================================================================
    @testset "match_fluxunits" begin

        λ = 2.0u"μm"

        # --- Case 1: per-Å → per-μm (same physical dimension, different unit) ---
        I_aa  = 1.0u"erg/s/cm^2/angstrom/sr"   # QPerAng
        I_um  = 1.0u"erg/s/cm^2/μm/sr"          # QPerum
        res1  = Loki.match_fluxunits(I_aa, I_um, λ)
        # 1 per Å = 1e4 per μm (1 μm = 1e4 Å → 1/Å = 1e4/μm)
        @test unit(res1) == u"erg/s/cm^2/μm/sr"
        @test ustrip(res1) ≈ 1e4 rtol=1e-10

        # --- Case 2: per-μm → per-Å ---
        res2 = Loki.match_fluxunits(I_um, I_aa, λ)
        @test unit(res2) == u"erg/s/cm^2/angstrom/sr"
        @test ustrip(res2) ≈ 1e-4 rtol=1e-10

        # --- Case 3: same per-frequency type → return unchanged ---
        I_hz = 3.7u"erg/s/cm^2/Hz/sr"           # QPerFreq (subtype of QGeneralPerFreq)
        res3 = Loki.match_fluxunits(I_hz, I_hz, λ)
        @test ustrip(res3) ≈ 3.7
        @test unit(res3) == u"erg/s/cm^2/Hz/sr"

        # --- Case 4: same per-wavelength type → return unchanged ---
        I_um2 = 5.0u"erg/s/cm^2/μm/sr"
        res4 = Loki.match_fluxunits(I_um2, I_um2, λ)
        @test ustrip(res4) ≈ 5.0
        @test unit(res4) == u"erg/s/cm^2/μm/sr"

        # --- Case 5: cross-type (per-freq ↔ per-wave) via fluxconvert ---
        Iν = 1.0u"erg/s/cm^2/Hz/sr"
        Iλ_ref = 1.0u"erg/s/cm^2/μm/sr"
        res5 = Loki.match_fluxunits(Iν, Iλ_ref, λ)
        # Result should be in per-μm units (matching I_ref)
        @test unit(res5) == u"erg/s/cm^2/μm/sr"
        @test ustrip(res5) > 0
        @test isfinite(ustrip(res5))

        # Round-trip: Iν → Iλ → match back to Iν gives original
        Iλ = Loki.fluxconvert(Iν, λ)
        Iν_back = Loki.match_fluxunits(Iλ, Iν, λ)
        @test ustrip(uconvert(unit(Iν), Iν_back)) ≈ ustrip(Iν) rtol=1e-10
    end

end
