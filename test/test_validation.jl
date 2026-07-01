#=
Unit tests for the input-validation layer added in src/util/validation.jl and the extended
file/parameter validators. These exercise the friendly-error paths for malformed settings and data
without running a full fit, plus the numerical helpers (safenorm, validate_fit_result).
=#
using Loki
using Unitful, UnitfulAstro
using Test
using TOML, Logging

@testset "Input validation" begin

    # ---------------------------------------------------------------------------------------------
    @testset "validate_fit_settings" begin
        valid = Dict{Symbol,Any}(
            :n_bootstrap => 0, :line_test_threshold => 0.003, :map_snr_thresh => 3.0,
            :spaxel_timelimit => 3600.0, :ssp_regularize => 0.0, :sys_err => 0.0,
            :linemask_width => 1000.0, :apoly_degree => -1, :mpoly_degree => -1,
            :grain_size => 0.1, :pyroxene_x => 0.5, :olivine_y => 0.5,
            :extinction_curve => "calz", :silicate_absorption => "d+",
            :apoly_type => :Chebyshev, :mpoly_type => :Chebyshev, :plot_spaxels => :pyplot,
            :fit_joint => false, :use_pah_templates => false)
        @test Loki.validate_fit_settings(valid) === nothing

        bad(k, v) = (d = copy(valid); d[k] = v; d)
        @test_throws ErrorException Loki.validate_fit_settings(bad(:n_bootstrap, -1))
        @test_throws ErrorException Loki.validate_fit_settings(bad(:line_test_threshold, 0.0))
        @test_throws ErrorException Loki.validate_fit_settings(bad(:line_test_threshold, 1.5))
        @test_throws ErrorException Loki.validate_fit_settings(bad(:sys_err, 2.0))
        @test_throws ErrorException Loki.validate_fit_settings(bad(:apoly_degree, -2))
        @test_throws ErrorException Loki.validate_fit_settings(bad(:grain_size, 0.0))
        @test_throws ErrorException Loki.validate_fit_settings(bad(:pyroxene_x, 1.5))
        @test_throws ErrorException Loki.validate_fit_settings(bad(:extinction_curve, "bogus"))
        @test_throws ErrorException Loki.validate_fit_settings(bad(:silicate_absorption, "nope"))
        @test_throws ErrorException Loki.validate_fit_settings(bad(:apoly_type, :Hermite))
        @test_throws ErrorException Loki.validate_fit_settings(bad(:plot_spaxels, :ascii))
        # mutually exclusive
        d = copy(valid); d[:fit_joint] = true; d[:use_pah_templates] = true
        @test_throws ErrorException Loki.validate_fit_settings(d)

        # numeric options passed WITH units as kwargs (linemask_width km/s, grain_size μm) must validate
        @test Loki.validate_fit_settings(bad(:linemask_width, 1000.0u"km/s")) === nothing
        @test Loki.validate_fit_settings(bad(:grain_size, 0.1u"μm")) === nothing
        @test Loki.validate_fit_settings(bad(:spaxel_timelimit, 3600.0u"s")) === nothing
        @test_throws ErrorException Loki.validate_fit_settings(bad(:linemask_width, -5.0u"km/s"))
        # enum options accept either a String or a Symbol of a valid value, and reject invalid ones either way
        @test Loki.validate_fit_settings(bad(:extinction_curve, :ccm)) === nothing
        @test Loki.validate_fit_settings(bad(:apoly_type, "Legendre")) === nothing
        @test_throws ErrorException Loki.validate_fit_settings(bad(:extinction_curve, :bogus))

        # degeneracy warning: fit_temp_multexp + mpoly (only when templates are present)
        dwarn = copy(valid); dwarn[:fit_temp_multexp] = true; dwarn[:mpoly_degree] = 2
        dwarn[:templates] = ones(1, 1, 10, 1)
        @test_logs (:warn,) match_mode=:any Loki.validate_fit_settings(dwarn)
        dnone = copy(dwarn); dnone[:templates] = Array{Float64,4}(undef, 1, 1, 10, 0)   # no templates -> no warning
        @test_logs min_level=Logging.Warn Loki.validate_fit_settings(dnone)
        dnope = copy(dwarn); dnope[:mpoly_degree] = -1                                   # mpoly off -> no warning
        @test_logs min_level=Logging.Warn Loki.validate_fit_settings(dnope)
    end

    # ---------------------------------------------------------------------------------------------
    @testset "extinction degeneracy auto-locks" begin
        λunit, Iunit = u"μm", u"erg/s/cm^2/Hz/sr"
        Tμ = typeof(1.0u"μm")
        λlim = (5.0u"μm", 25.0u"μm")
        region = Loki.SpectralRegion(λlim, Tuple{Tμ,Tμ}[], 1, BitVector[trues(2)], Tμ[], Tuple{Tμ,Tμ}[],
            Loki.get_λrange(λlim))
        optical = TOML.parsefile(joinpath(pkgdir(Loki), "src", "options", "optical.toml"))
        infrared = TOML.parsefile(joinpath(pkgdir(Loki), "src", "options", "infrared.toml"))
        baseout() = Dict{Symbol,Any}(:mpoly_degree => -1, :fit_joint => true, :fit_uv_bump => false,
            :silicate_absorption => "d+", :fit_ch_abs => false, :fit_covering_frac => false,
            :extinction_curve => "calz")
        function run_ext(out)
            params = Loki.FitParameter[]; pnames = String[]; plabels = String[]
            ptrans = Vector{Loki.Transformation}[]
            Loki.construct_extinction_params!(params, pnames, plabels, ptrans, out, optical, infrared, λunit, Iunit, region)
            params, pnames
        end
        islocked(p, n, name) = p[findfirst(==(name), n)].locked

        # baseline: mpoly off + joint on -> E_BV free (E_BV_factor keeps its config default, which is locked)
        p, n = run_ext(baseout())
        @test !islocked(p, n, "extinction.E_BV")

        # guard 1: multiplicative polynomial active -> E_BV auto-locked (with a warning)
        o1 = baseout(); o1[:mpoly_degree] = 2
        p1, n1 = @test_logs (:warn,) match_mode=:any run_ext(o1)
        @test islocked(p1, n1, "extinction.E_BV")

        # guard 2: fit_joint=false + E_BV_factor unlocked -> E_BV_factor auto-locked (with a warning)
        o2 = baseout(); o2[:fit_joint] = false
        optical2 = deepcopy(optical); optical2["extinction"]["E_BV_factor"]["locked"] = false
        params = Loki.FitParameter[]; pnames = String[]; plabels = String[]; ptrans = Vector{Loki.Transformation}[]
        @test_logs (:warn,) match_mode=:any Loki.construct_extinction_params!(params, pnames, plabels, ptrans,
            o2, optical2, infrared, λunit, Iunit, region)
        @test islocked(params, pnames, "extinction.E_BV_factor")
    end

    # ---------------------------------------------------------------------------------------------
    @testset "validate_datacube_inputs" begin
        # NOTE: array-SHAPE checks live in the DataCube constructor; validate_datacube_inputs only checks
        # the wavelength vector and PSF/LSF *values* (signature is (λ; psf, lsf)).
        λ = collect(range(5.0, 6.0, length=20)) .* u"μm"
        psf = ones(20) .* u"arcsecond"
        lsf = ones(20) .* u"km/s"
        @test Loki.validate_datacube_inputs(λ; psf=psf, lsf=lsf) === nothing

        @test_throws AssertionError Loki.validate_datacube_inputs(reverse(λ))                          # decreasing
        @test_throws AssertionError Loki.validate_datacube_inputs(collect(range(0.0,1.0,length=20)).*u"μm")  # nonpositive
        bad_λ = copy(λ); bad_λ[5] = bad_λ[6]                                                            # duplicate (not strictly increasing)
        @test_throws AssertionError Loki.validate_datacube_inputs(bad_λ)
        @test_throws AssertionError Loki.validate_datacube_inputs((1:20) .* u"Hz")                      # wrong dimension
        @test_throws AssertionError Loki.validate_datacube_inputs(λ; psf=(-ones(20)).*u"arcsecond")     # negative psf
    end

    # ---------------------------------------------------------------------------------------------
    @testset "parameter_from_dict plim well-formedness" begin
        @test Loki.parameter_from_dict(Dict("val"=>1.0, "locked"=>false, "plim"=>[0.0, 2.0])) isa Loki.FitParameter
        @test_throws AssertionError Loki.parameter_from_dict(Dict("val"=>7.0, "locked"=>false, "plim"=>[10.0, 5.0]))   # inverted
        @test_throws AssertionError Loki.parameter_from_dict(Dict("val"=>100.0, "locked"=>false, "plim"=>[0.0, 50.0])) # out of range
        @test_throws AssertionError Loki.parameter_from_dict(Dict("val"=>1.0, "locked"=>false, "plim"=>[0.0, 2.0, 3.0])) # wrong length
    end

    # ---------------------------------------------------------------------------------------------
    @testset "safenorm" begin
        @test Loki.safenorm([1.0, 2.0, 4.0]) ≈ [0.25, 0.5, 1.0]
        @test all(Loki.safenorm([0.0, 0.0, 0.0]) .== 0.0)          # all-zero -> zeros, not 0/0=NaN
        @test all(isfinite, Loki.safenorm([0.0, 0.0, 0.0]))
        bb = [1.0, 2.0, 3.0] .* u"erg/s/cm^2/Hz/sr"                 # unitful -> dimensionless result
        @test Loki.safenorm(bb) ≈ [1/3, 2/3, 1.0]
    end

    # ---------------------------------------------------------------------------------------------
    @testset "validate_fit_result" begin
        idx = CartesianIndex(1, 1)
        @test Loki.validate_fit_result((status=1, param=[1.0, 2.0], perror=[0.1, 0.1]), "test", idx) == true
        @test Loki.validate_fit_result((status=-1, param=[1.0, 2.0], perror=[0.1, 0.1]), "test", idx) == false
        @test Loki.validate_fit_result((status=1, param=[NaN, 2.0], perror=[0.1, 0.1]), "test", idx) == false
    end

end
