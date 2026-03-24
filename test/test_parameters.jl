###############################################################################
# Tests for FitParameter struct and related operations
# Source: src/util/parameters.jl
#
# FitParameter is the fundamental data structure holding a single model
# parameter: its current value, whether it is fixed (locked), its allowed
# range (limits), and an optional "tie" linking it to another parameter.
#
# None of these are part of the public API, so they are accessed via the
# Loki module prefix: Loki.FitParameter, Loki.lock!, etc.
###############################################################################

@testset "FitParameter" begin

    # =========================================================================
    # Construction
    # =========================================================================
    @testset "Constructor" begin
        # Basic creation: value, locked=false, limits
        p = Loki.FitParameter(1.0, false, (0.0, 2.0))
        @test p.value   == 1.0
        @test p.locked  == false
        @test p.limits  == (0.0, 2.0)
        @test isnothing(p.tie)

        # Locked parameter
        p_locked = Loki.FitParameter(1.0, true, (0.0, 2.0))
        @test p_locked.locked == true

        # Integer value (type T = Int)
        p_int = Loki.FitParameter(1, false, (0, 5))
        @test p_int.value == 1
        @test p_int.limits == (0, 5)

        # Value at the lower limit (boundary is allowed)
        p_lo = Loki.FitParameter(0.0, false, (0.0, 2.0))
        @test p_lo.value == 0.0

        # Value at the upper limit
        p_hi = Loki.FitParameter(2.0, false, (0.0, 2.0))
        @test p_hi.value == 2.0

        # Reversed or equal limits must throw AssertionError
        @test_throws AssertionError Loki.FitParameter(1.0, false, (2.0, 0.0))
        @test_throws AssertionError Loki.FitParameter(1.0, false, (1.0, 1.0))
    end

    # =========================================================================
    # lock! / unlock!
    # =========================================================================
    @testset "lock! and unlock!" begin
        p = Loki.FitParameter(1.0, false, (0.0, 2.0))
        @test p.locked == false

        Loki.lock!(p)
        @test p.locked == true

        Loki.unlock!(p)
        @test p.locked == false

        # Locking an already-locked parameter is idempotent
        Loki.lock!(p)
        Loki.lock!(p)
        @test p.locked == true
    end

    # =========================================================================
    # tie! / untie!
    # =========================================================================
    @testset "tie! and untie!" begin
        p = Loki.FitParameter(1.0, false, (0.0, 2.0))

        # Flat tie: links p to a named group with equal values
        Loki.tie!(p, :ha_group)
        @test p.tie isa Loki.FlatTie
        @test p.tie.group == :ha_group

        # Ratio tie: links p to a named group with a fixed ratio
        Loki.tie!(p, :ratio_group, 2.0)
        @test p.tie isa Loki.RatioTie
        @test p.tie.group  == :ratio_group
        @test p.tie.ratio  == 2.0

        # Untie: removes the tie
        Loki.untie!(p)
        @test isnothing(p.tie)
    end

    # =========================================================================
    # set_val! / set_plim!
    # =========================================================================
    @testset "set_val! and set_plim!" begin
        p = Loki.FitParameter(1.0, false, (0.0, 2.0))

        # Update value
        Loki.set_val!(p, 1.5)
        @test p.value == 1.5

        # Update limits
        Loki.set_plim!(p, (0.5, 3.0))
        @test p.limits == (0.5, 3.0)

        # Reversed limits in set_plim! must throw AssertionError
        @test_throws AssertionError Loki.set_plim!(p, (3.0, 0.5))
    end

    # =========================================================================
    # check_valid
    # =========================================================================
    @testset "check_valid" begin
        # A well-formed parameter passes without error
        p_ok = Loki.FitParameter(1.0, false, (0.0, 2.0))
        @test_nowarn Loki.check_valid(p_ok)

        # Value at boundary is also valid
        p_lo = Loki.FitParameter(0.0, false, (0.0, 2.0))
        @test_nowarn Loki.check_valid(p_lo)

        # Value outside limits fails (constructor doesn't enforce this, only check_valid does)
        p_bad = Loki.FitParameter(5.0, false, (0.0, 2.0))
        @test_throws AssertionError Loki.check_valid(p_bad)
    end

    # =========================================================================
    # parameter_from_dict helpers
    # These create FitParameter objects from a plain Dict, which is how
    # Loki reads parameters from .toml configuration files.
    # =========================================================================
    @testset "parameter_from_dict" begin
        d = Dict("val" => 1.0, "locked" => false, "plim" => [0.0, 2.0])

        # Without units: default NoUnits
        p = Loki.parameter_from_dict(d)
        @test ustrip(p.value)    ≈ 1.0
        @test p.locked == false
        @test ustrip(p.limits[1]) ≈ 0.0
        @test ustrip(p.limits[2]) ≈ 2.0

        # With Unitful units
        p_um = Loki.parameter_from_dict(d; units=u"μm")
        @test ustrip(uconvert(u"μm", p_um.value)) ≈ 1.0
        @test ustrip(uconvert(u"μm", p_um.limits[2])) ≈ 2.0
    end

    @testset "parameter_from_dict_wave" begin
        # val = 5.0, plim = [-0.1, +0.1] → limits = [4.9, 5.1]
        d = Dict("val" => 5.0, "locked" => false, "plim" => [-0.1, 0.1])
        p = Loki.parameter_from_dict_wave(d; units=u"μm")
        @test ustrip(uconvert(u"μm", p.value))    ≈ 5.0 rtol=1e-6
        @test ustrip(uconvert(u"μm", p.limits[1])) ≈ 4.9 rtol=1e-6
        @test ustrip(uconvert(u"μm", p.limits[2])) ≈ 5.1 rtol=1e-6
    end

    @testset "parameter_from_dict_fwhm" begin
        # val = 0.04, plim = [0.5, 2.0] → limits = [0.02, 0.08]
        d = Dict("val" => 0.04, "locked" => false, "plim" => [0.5, 2.0])
        p = Loki.parameter_from_dict_fwhm(d; units=u"μm")
        @test ustrip(uconvert(u"μm", p.value))    ≈ 0.04 rtol=1e-6
        @test ustrip(uconvert(u"μm", p.limits[1])) ≈ 0.02 rtol=1e-6
        @test ustrip(uconvert(u"μm", p.limits[2])) ≈ 0.08 rtol=1e-6
    end

    # =========================================================================
    # fast_indexin — optimized string/symbol search
    # =========================================================================
    @testset "fast_indexin" begin
        haystack = ["alpha", "beta", "gamma", "delta"]

        # Single needle
        @test Loki.fast_indexin("alpha", haystack) == 1
        @test Loki.fast_indexin("beta",  haystack) == 2
        @test Loki.fast_indexin("gamma", haystack) == 3
        @test Loki.fast_indexin("delta", haystack) == 4

        # Missing needle returns nothing
        @test isnothing(Loki.fast_indexin("epsilon", haystack))

        # Vector of needles
        @test Loki.fast_indexin(["delta", "alpha"], haystack) == [4, 1]

        # Works with Symbol needles against String haystack
        @test Loki.fast_indexin(:beta, haystack) == 2

        # Empty vector of needles
        @test Loki.fast_indexin(String[], haystack) == []
    end

end
