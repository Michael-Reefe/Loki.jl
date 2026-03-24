###############################################################################
# Tests for spectral data utility functions
# Source: src/core/fitdata.jl
#
# Functions tested:
#   mask_emission_lines(λ, mask_regions)    — BitVector mask for line regions
#   fill_bad_pixels(I, σ, templates)        — NaN/Inf fill in spectrum arrays
#   calculate_statistical_errors(I, Isp, mask) — local RMS noise estimation
###############################################################################

@testset "Spectral data utilities" begin

    # =========================================================================
    # mask_emission_lines(λ, mask_regions)
    # λ must be Unitful.  mask_regions is a Vector of (lo, hi) tuples.
    # Masking is exclusive on both ends: region[1] < λ < region[2].
    # =========================================================================
    @testset "mask_emission_lines" begin

        λ = collect(1.0:1.0:8.0) .* u"μm"   # [1,2,3,4,5,6,7,8] μm

        # Empty regions → all-false mask
        empty_regions = Vector{Tuple{typeof(1.0u"μm"), typeof(1.0u"μm")}}()
        mask_empty = Loki.mask_emission_lines(λ, empty_regions)
        @test length(mask_empty) == length(λ)
        @test !any(mask_empty)

        # Single region (2.5, 5.5): only λ=3,4,5 are strictly inside
        regions1 = [(2.5u"μm", 5.5u"μm")]
        mask1 = Loki.mask_emission_lines(λ, regions1)
        @test mask1 == BitVector([0,0,1,1,1,0,0,0])

        # Boundaries are exclusive: λ=2 is NOT inside (2.0, 4.0)
        regions2 = [(2.0u"μm", 4.0u"μm")]
        mask2 = Loki.mask_emission_lines(λ, regions2)
        @test mask2[2] == false   # λ=2 excluded (equal to lower bound)
        @test mask2[3] == true    # λ=3 inside
        @test mask2[4] == false   # λ=4 excluded (equal to upper bound)

        # Multiple disjoint regions
        regions3 = [(1.5u"μm", 3.5u"μm"), (5.5u"μm", 7.5u"μm")]
        mask3 = Loki.mask_emission_lines(λ, regions3)
        @test mask3 == BitVector([0,1,1,0,0,1,1,0])

        # Output length equals λ length
        @test length(mask3) == length(λ)
    end

    # =========================================================================
    # fill_bad_pixels(I, σ, templates)
    # Replaces NaN/Inf in I and σ:
    #   - edges (first/last): replaced with nearest finite value
    #   - interior: replaced with average of nearest finite neighbors
    # Returns modified I, σ, templates (modifies in place).
    # =========================================================================
    @testset "fill_bad_pixels" begin

        # No NaNs: arrays returned unchanged (modifies in place, so compare values)
        I_clean = [1.0, 2.0, 3.0, 4.0, 5.0]
        σ_clean = [0.1, 0.1, 0.1, 0.1, 0.1]
        I_out, σ_out, _ = Loki.fill_bad_pixels(copy(I_clean), copy(σ_clean), nothing)
        @test I_out == I_clean
        @test σ_out == σ_clean

        # NaN at start: replaced with first finite value
        I_nan_start = [NaN, 2.0, 3.0, 4.0, 5.0]
        σ_nan_start = [NaN, 0.1, 0.1, 0.1, 0.1]
        I_out2, σ_out2, _ = Loki.fill_bad_pixels(copy(I_nan_start), copy(σ_nan_start), nothing)
        @test isfinite(I_out2[1])
        @test I_out2[1] ≈ 2.0

        # NaN at end: replaced with last finite value (nearest finite neighbor)
        I_nan_end = [1.0, 2.0, 3.0, 4.0, NaN]
        σ_nan_end = [0.1, 0.1, 0.1, 0.1, NaN]
        I_out3, _, _ = Loki.fill_bad_pixels(copy(I_nan_end), copy(σ_nan_end), nothing)
        @test isfinite(I_out3[end])
        @test I_out3[end] ≈ 4.0

        # Interior NaN: replaced with average of left and right finite neighbors
        I_nan_mid = [1.0, NaN, 3.0, 4.0, 5.0]
        σ_nan_mid = [0.1, 0.1, 0.1, 0.1, 0.1]
        I_out4, _, _ = Loki.fill_bad_pixels(copy(I_nan_mid), copy(σ_nan_mid), nothing)
        @test isfinite(I_out4[2])
        @test I_out4[2] ≈ (1.0 + 3.0) / 2  # average of neighbors

        # All outputs must be finite
        I_multi = [NaN, 2.0, NaN, 4.0, NaN]
        σ_multi = [0.1, 0.1, 0.1, 0.1, 0.1]
        I_out5, σ_out5, _ = Loki.fill_bad_pixels(copy(I_multi), copy(σ_multi), nothing)
        @test all(isfinite.(I_out5))
        @test all(isfinite.(σ_out5))

        # With templates: NaN in template also gets filled
        I_t = [1.0, 2.0, 3.0]
        σ_t = [0.1, 0.1, 0.1]
        tmpl = [1.0 NaN; 1.5 2.5; 2.0 3.0]  # NaN in column 2, row 1
        _, _, tmpl_out = Loki.fill_bad_pixels(copy(I_t), copy(σ_t), copy(tmpl))
        @test all(isfinite.(tmpl_out))
    end

    # =========================================================================
    # calculate_statistical_errors(I, I_spline, mask)
    # Estimates per-pixel noise as the local RMS of residuals with a spline.
    # Window size = min(60, number_of_unmasked_pixels).
    # =========================================================================
    @testset "calculate_statistical_errors" begin

        n = 80
        I = randn(n) .+ 10.0         # noisy signal around 10
        I_spline = fill(10.0, n)     # perfect continuum
        mask = falses(n)             # no masked pixels

        σ_stat = Loki.calculate_statistical_errors(I, I_spline, mask)

        # Output length equals input length
        @test length(σ_stat) == n

        # All errors are non-negative (std ≥ 0)
        @test all(σ_stat .≥ 0)

        # Perfect fit (I == I_spline): residuals are all zero, so σ_stat ≈ 0
        I_perfect = fill(5.0, n)
        σ_stat_perfect = Loki.calculate_statistical_errors(I_perfect, I_perfect, mask)
        @test all(isapprox.(σ_stat_perfect, 0.0, atol=1e-12))

        # Masked pixels: output still has correct length
        mask_partial = falses(n)
        mask_partial[30:40] .= true  # 11 masked pixels
        σ_stat2 = Loki.calculate_statistical_errors(I, I_spline, mask_partial)
        @test length(σ_stat2) == n
        @test all(σ_stat2 .≥ 0)

        # Window size bounded at 60 unmasked pixels:
        # with n=80 all unmasked → window = min(60, 80) = 60 (no error expected)
        @test length(σ_stat) == n
    end

end
