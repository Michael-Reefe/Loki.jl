###############################################################################
# Tests for FitParameters collection operations
# Source: src/util/parameters.jl  (lines ~101–112, 261–297, 390–655)
#
# FitParameters is a named, ordered collection of FitParameter objects,
# analogous to an ordered dictionary.  It supports:
#   - Construction from (names, labels, transformations, parameters)
#   - Integer and String indexing
#   - push! / append! / deleteat!
#   - Bulk accessors: get_plims, get_lock, get_val, get_tie
#   - get_λrange — classify a wavelength range as UVOptIR / Infrared / UVOptical
#   - get_gap_masks — split a wavelength vector into gap-separated BitVector masks
#   - lock! / unlock!  — toggle the locked flag by index, name, or vector of names
#   - set_val! / set_plim! — update value or limits in place
#   - check_valid — assert a FitParameter is internally consistent
#   - get_tied_pairs — extract all tied-parameter pairs and their ratios
###############################################################################

# ---------------------------------------------------------------------------
# Helper: build a small FitParameters collection
# ---------------------------------------------------------------------------
function _make_fp()
    p1 = Loki.FitParameter(1.0, false, (0.0, 2.0))
    p2 = Loki.FitParameter(0.5, true,  (0.0, 1.0))
    p3 = Loki.FitParameter(3.0, false, (1.0, 5.0))
    fp = Loki.FitParameters(String[], String[], Vector{Loki.Transformation}[], Loki.FitParameter[])
    push!(fp, "amp",   "A",      Loki.Transformation[], p1)
    push!(fp, "fwhm",  "\\sigma", Loki.Transformation[], p2)
    push!(fp, "voff",  "v",      Loki.Transformation[], p3)
    fp
end

@testset "FitParameters collection" begin

    # =========================================================================
    # Construction
    # =========================================================================
    @testset "Constructor" begin

        # Valid: unique names → succeeds
        p = Loki.FitParameter(1.0, false, (0.0, 2.0))
        fp = Loki.FitParameters(["x"], ["X"], [Loki.Transformation[]], Loki.FitParameter[p])
        @test length(fp) == 1

        # Duplicate names → AssertionError
        @test_throws AssertionError Loki.FitParameters(
            ["x", "x"], ["X", "X"],
            [Loki.Transformation[], Loki.Transformation[]],
            Loki.FitParameter[p, p]
        )

        # Empty collection
        fp_empty = Loki.FitParameters(String[], String[], Vector{Loki.Transformation}[], Loki.FitParameter[])
        @test length(fp_empty) == 0
    end

    # =========================================================================
    # Indexing
    # =========================================================================
    @testset "getindex" begin
        fp = _make_fp()

        # Integer index: returns FitParameter at that position
        @test fp[1].value == 1.0
        @test fp[2].value == 0.5
        @test fp[3].value == 3.0

        # String index: returns FitParameter by name
        @test fp["amp"].value  == 1.0
        @test fp["fwhm"].value == 0.5
        @test fp["voff"].value == 3.0

        # Vector of strings: returns multiple parameters in order
        params = fp[["voff", "amp"]]
        @test params[1].value == 3.0
        @test params[2].value == 1.0
    end

    # =========================================================================
    # push!
    # =========================================================================
    @testset "push!" begin
        fp = _make_fp()
        n_before = length(fp)

        # Push a new parameter
        p_new = Loki.FitParameter(0.1, false, (0.0, 0.5))
        push!(fp, "eta", "\\eta", Loki.Transformation[], p_new)
        @test length(fp) == n_before + 1
        @test fp["eta"].value == 0.1

        # Duplicate name → AssertionError
        @test_throws AssertionError push!(fp, "amp", "A2", Loki.Transformation[], p_new)
    end

    # =========================================================================
    # append!
    # =========================================================================
    @testset "append!" begin
        fp1 = _make_fp()

        # Build a second collection with different names
        p_new = Loki.FitParameter(2.5, false, (0.0, 5.0))
        fp2 = Loki.FitParameters(String[], String[], Vector{Loki.Transformation}[], Loki.FitParameter[])
        push!(fp2, "extra", "E", Loki.Transformation[], p_new)

        n_before = length(fp1)
        append!(fp1, fp2)
        @test length(fp1) == n_before + 1
        @test fp1["extra"].value == 2.5

        # Conflicting names → AssertionError
        @test_throws AssertionError append!(fp1, fp1)
    end

    # =========================================================================
    # deleteat!
    # =========================================================================
    @testset "deleteat!" begin
        fp = _make_fp()

        # Remove by integer index: length decreases, other names survive
        deleteat!(fp, 2)  # removes "fwhm"
        @test length(fp) == 2
        @test fp[1].value == 1.0   # "amp" still first
        @test fp[2].value == 3.0   # "voff" is now second

        # Remove by name string
        fp2 = _make_fp()
        deleteat!(fp2, "voff")
        @test length(fp2) == 2
        @test fp2["amp"].value  == 1.0
        @test fp2["fwhm"].value == 0.5
    end

    # =========================================================================
    # Bulk accessors: get_plims, get_lock, get_val, get_tie
    # =========================================================================
    @testset "Bulk accessors" begin
        fp = _make_fp()

        # get_val by index and name
        @test Loki.get_val(fp, 1) == 1.0
        @test Loki.get_val(fp, "fwhm") == 0.5

        # get_lock by index and name
        @test Loki.get_lock(fp, 1) == false
        @test Loki.get_lock(fp, "fwhm") == true

        # get_plims by index and name
        @test Loki.get_plims(fp, 1) == (0.0, 2.0)
        @test Loki.get_plims(fp, "fwhm") == (0.0, 1.0)

        # get_tie: all start as nothing
        @test isnothing(Loki.get_tie(fp, 1))
        @test isnothing(Loki.get_tie(fp, "amp"))

        # After adding a tie, get_tie returns it
        Loki.tie!(fp["amp"], :ha_group)
        @test !isnothing(Loki.get_tie(fp, "amp"))
        @test Loki.get_tie(fp, "amp").group == :ha_group

        # Bulk _getproperty for values vector
        vals = fp.values
        @test length(vals) == 3
        @test vals[1] == 1.0
        @test vals[2] == 0.5
        @test vals[3] == 3.0

        # Bulk locks vector
        locks = fp.locked
        @test locks[1] == false
        @test locks[2] == true
        @test locks[3] == false
    end

    # =========================================================================
    # get_λrange(λlim)
    # Classifies the wavelength range:
    #   UVOptIR   — straddles both UV/optical and IR (λ[1]<3.3 && λ[2]>2.2 μm)
    #   Infrared  — entirely IR (λ[1] > 3.3 μm)
    #   UVOptical — entirely UV/optical (λ[2] ≤ 2.2 μm)
    # =========================================================================
    @testset "get_λrange" begin

        # Spans UV through IR: UVOptIR
        @test Loki.get_λrange((0.5u"μm", 28.0u"μm")) == Loki.UVOptIR

        # Both λ values > 3.3 μm: Infrared
        @test Loki.get_λrange((5.0u"μm", 28.0u"μm")) == Loki.Infrared

        # Both λ values < 2.2 μm: UVOptical
        @test Loki.get_λrange((0.3u"μm", 1.0u"μm")) == Loki.UVOptical

        # Straddles 2.2 μm but starts below 3.3 μm: UVOptIR
        @test Loki.get_λrange((0.5u"μm", 3.0u"μm")) == Loki.UVOptIR
    end

    # =========================================================================
    # get_gap_masks(λ, gaps)
    # Creates N+1 BitVector masks for N gaps.  Elements in a gap appear in
    # neither mask.  With no gaps, returns a single all-true mask.
    # =========================================================================
    @testset "get_gap_masks" begin

        λ = collect(1.0:1.0:10.0)   # [1,2,...,10]

        # No gaps: single all-true mask of correct length
        masks_0 = Loki.get_gap_masks(λ, Tuple{Float64,Float64}[])
        @test length(masks_0) == 1
        @test all(masks_0[1])
        @test length(masks_0[1]) == length(λ)

        # One gap at (3.5, 6.5): elements 4,5,6 are in gap
        # Mask 1: λ .< 3.5 → [T,T,T,F,F,F,F,F,F,F]
        # Mask 2: 6.5 .< λ → [F,F,F,F,F,F,T,T,T,T]
        gaps1 = [(3.5, 6.5)]
        masks_1 = Loki.get_gap_masks(λ, gaps1)
        @test length(masks_1) == 2
        @test sum(masks_1[1]) == 3   # elements 1,2,3
        @test sum(masks_1[2]) == 4   # elements 7,8,9,10
        @test !any(masks_1[1] .& masks_1[2])  # no overlap

        # Two gaps → three masks
        gaps2 = [(2.5, 4.5), (6.5, 8.5)]
        masks_2 = Loki.get_gap_masks(λ, gaps2)
        @test length(masks_2) == 3

        # Each mask is a BitVector of correct length
        for m in masks_2
            @test m isa BitVector
            @test length(m) == length(λ)
        end

        # No pair of masks has overlapping true elements
        for i in eachindex(masks_2), j in (i+1):lastindex(masks_2)
            @test !any(masks_2[i] .& masks_2[j])
        end
    end

    # =========================================================================
    # lock!(p, key) and unlock!(p, key)
    # Toggle the `locked` field of FitParameter objects by index, name, or
    # vector of names.
    # =========================================================================
    @testset "lock! and unlock!" begin
        fp = _make_fp()

        # Lock by name: locked becomes true
        Loki.lock!(fp, "amp")
        @test fp["amp"].locked == true

        # Unlock by name: locked becomes false
        Loki.unlock!(fp, "amp")
        @test fp["amp"].locked == false

        # Lock by integer index
        Loki.lock!(fp, 1)       # "amp" is index 1
        @test fp[1].locked == true
        Loki.unlock!(fp, 1)
        @test fp[1].locked == false

        # Lock multiple by vector of names
        Loki.lock!(fp, ["amp", "voff"])
        @test fp["amp"].locked  == true
        @test fp["voff"].locked == true
        @test fp["fwhm"].locked == true   # "fwhm" was already locked (p2 initialized with true)

        # Unlock multiple
        Loki.unlock!(fp, ["amp", "voff"])
        @test fp["amp"].locked  == false
        @test fp["voff"].locked == false

        # Locking one parameter does not affect others
        fp2 = _make_fp()
        Loki.lock!(fp2, "amp")
        @test fp2["fwhm"].locked == true   # still locked (was initialized true)
        @test fp2["voff"].locked == false  # still unlocked
    end

    # =========================================================================
    # set_val!(p, key, v) and set_plim!(p, key, limits)
    # Update parameter value or limits in place.
    # =========================================================================
    @testset "set_val! and set_plim!" begin
        fp = _make_fp()

        # set_val! by name
        Loki.set_val!(fp, "amp", 1.5)
        @test fp["amp"].value ≈ 1.5

        # set_val! by integer index
        Loki.set_val!(fp, 2, 0.8)   # "fwhm" is index 2
        @test fp["fwhm"].value ≈ 0.8

        # set_val! with vector of names and values
        Loki.set_val!(fp, ["amp", "voff"], [0.3, 2.5])
        @test fp["amp"].value  ≈ 0.3
        @test fp["voff"].value ≈ 2.5

        # set_plim! by name
        Loki.set_plim!(fp, "amp", (0.0, 5.0))
        @test fp["amp"].limits == (0.0, 5.0)

        # set_plim! by integer index
        Loki.set_plim!(fp, 2, (0.0, 2.0))
        @test fp["fwhm"].limits == (0.0, 2.0)

        # set_plim! reversed limits → AssertionError (lower must be < upper)
        @test_throws AssertionError Loki.set_plim!(fp, "amp", (5.0, 0.0))
        @test_throws AssertionError Loki.set_plim!(fp, "amp", (1.0, 1.0))  # equal limits

        # After set_plim!, value is still accessible
        @test fp["amp"].value ≈ 0.3
    end

    # =========================================================================
    # check_valid(p::FitParameter) and check_valid(p::Parameters)
    # Validates that a FitParameter (or all parameters in a collection) is
    # internally consistent: finite value, limits ordered, value within limits.
    # =========================================================================
    @testset "check_valid" begin

        # Valid FitParameter → no error
        p_valid = Loki.FitParameter(1.0, false, (0.0, 2.0))
        @test_nowarn Loki.check_valid(p_valid)

        # NaN value → AssertionError (isfinite check)
        p_nan = Loki.FitParameter(NaN, false, (0.0, 2.0))
        @test_throws AssertionError Loki.check_valid(p_nan)

        # Value below lower limit → AssertionError
        p_below = Loki.FitParameter(-1.0, false, (0.0, 2.0))
        @test_throws AssertionError Loki.check_valid(p_below)

        # Value above upper limit → AssertionError
        p_above = Loki.FitParameter(3.0, false, (0.0, 2.0))
        @test_throws AssertionError Loki.check_valid(p_above)

        # Value at boundary → no error (limits are inclusive)
        p_lo = Loki.FitParameter(0.0, false, (0.0, 2.0))
        p_hi = Loki.FitParameter(2.0, false, (0.0, 2.0))
        @test_nowarn Loki.check_valid(p_lo)
        @test_nowarn Loki.check_valid(p_hi)

        # FitParameters collection: all valid → no error
        fp_ok = _make_fp()
        @test_nowarn Loki.check_valid(fp_ok)

        # Collection with one invalid parameter → AssertionError
        fp_bad = _make_fp()
        fp_bad["amp"].value = 99.0   # above limits (0.0, 2.0)
        @test_throws AssertionError Loki.check_valid(fp_bad)
    end

    # =========================================================================
    # get_tied_pairs(fp::FitParameters)
    # Extracts all tied-parameter pairs as (anchor_idx, tied_idx, ratio) tuples.
    # FlatTie: ratio = 1.0; RatioTie: ratio = tied.ratio / anchor.ratio.
    # =========================================================================
    @testset "get_tied_pairs" begin

        # No ties → empty pair and index vectors
        fp_notied = _make_fp()
        pairs_empty, idx_empty = Loki.get_tied_pairs(fp_notied)
        @test isempty(pairs_empty)
        @test isempty(idx_empty)

        # One FlatTie group: amp (index 1) and fwhm (index 2) share :ha_group
        fp_flat = _make_fp()
        Loki.tie!(fp_flat["amp"],  :ha_group)   # anchor (first in group)
        Loki.tie!(fp_flat["fwhm"], :ha_group)   # tied to amp

        pairs_flat, idx_flat = Loki.get_tied_pairs(fp_flat)
        @test length(pairs_flat) == 1
        @test pairs_flat[1][1] == 1              # anchor is "amp" (index 1)
        @test pairs_flat[1][2] == 2              # tied is "fwhm" (index 2)
        @test pairs_flat[1][3] ≈ 1.0             # FlatTie → ratio = 1.0
        @test idx_flat == [2]                    # only the non-anchor is in tie_indices

        # One RatioTie group: amp (ratio=1.0), fwhm (ratio=2.0)
        fp_ratio = _make_fp()
        Loki.tie!(fp_ratio["amp"],  :ratio_group, 1.0)
        Loki.tie!(fp_ratio["fwhm"], :ratio_group, 2.0)

        pairs_ratio, idx_ratio = Loki.get_tied_pairs(fp_ratio)
        @test length(pairs_ratio) == 1
        @test pairs_ratio[1][3] ≈ 2.0            # fwhm.ratio / amp.ratio = 2.0/1.0
        @test idx_ratio == [2]

        # Return types
        @test pairs_flat isa Vector{Tuple{Int,Int,Float64}}
        @test idx_flat   isa Vector{Int}
    end

end
