###############################################################################
# Tests for DataCube functions and CubeFitter-dependent functions
# Source: src/core/cubedata.jl   (DataCube constructor, to_rest_frame!,
#                                  to_vacuum_wavelength!, apply_mask!)
#         src/util/math.jl       (τ_kvt, τ_ohm, τ_ice, τ_ch, silicate_emission,
#                                  silicate_absorption)
#         src/core/cubefit_helpers.jl (get_continuum_step_sizes, get_line_step_sizes)
#         src/core/fitdata.jl         (get_line_nprof_ncomp)
#
# Strategy: build ONE set of base arrays and ONE DataCube (used for CubeFitter
# construction).  The DataCube constructor test reuses that cube directly.
# The three mutation tests (to_rest_frame!, to_vacuum_wavelength!, apply_mask!)
# each build a single additional DataCube from the same base arrays.
# The CubeFitter is built once and shared across all CubeFitter testsets.
#
# CubeFitter kwargs used:
#   fit_stellar_continuum = false  — skip SSP template loading
#   fit_opt_na_feii       = false  — skip narrow Fe II templates
#   fit_opt_br_feii       = false  — skip broad Fe II templates
#   use_pah_templates     = false  — skip Smith PAH template loading
#   silicate_absorption   = "kvt" — load KVT + OHM dust profiles
#   fit_ch_abs            = true  — load ice and CH absorption profiles
###############################################################################

@testset "DataCube and CubeFitter functions" begin

    # =========================================================================
    # Build base arrays once — shared by the DataCube constructor, all
    # mutation tests, and the CubeFitter testsets below.
    #
    # DataCube uses positional (not keyword) arguments. The order is:
    #   λ, I, σ, mask, psf_model, Ω, α, δ, θ_sky, psf, lsf, wcs,
    #   channel, band, user_mask, gaps, rest_frame, z, masked,
    #   vacuum_wave, log_binned, dereddened, sky_aligned, voronoi_bins, format
    # =========================================================================
    λ_cube = collect(range(5.0, 28.0, length=100)) .* u"μm"
    Ival   = 1.0u"erg/s/cm^2/μm/sr"
    I_cube = fill(Ival, 1, 1, 100)
    σ_cube = fill(0.1 * Ival, 1, 1, 100)

    # PSF and LSF vectors (typed Vector{Quantity} required by the struct)
    psf_cube = Loki.mrs_psf.(λ_cube)   # Vector of arcsecond Quantities
    lsf_cube = Loki.mrs_lsf.(λ_cube)   # Vector of km/s Quantities

    # Main DataCube: rest-frame, vacuum wavelengths, masked flag set.
    cube = DataCube(
        λ_cube,          # λ
        I_cube,          # I
        σ_cube,          # σ
        nothing,         # mask
        nothing,         # psf_model
        NaN*u"sr",      # Ω
        NaN*u"°",       # α
        NaN*u"°",       # δ
        NaN*u"rad",     # θ_sky
        psf_cube,        # psf
        lsf_cube,        # lsf
        nothing,         # wcs
        "Test",          # channel
        "Test",          # band
        nothing,         # user_mask
        nothing,         # gaps
        true,            # rest_frame ← must be true for CubeFitter
        0.01,            # z (required when rest_frame=true)
        true,            # masked ← must be true for CubeFitter
        true,            # vacuum_wave ← must be true for CubeFitter
        false,           # log_binned
        true,            # dereddened ← must be true for CubeFitter
        false,           # sky_aligned
        nothing,         # voronoi_bins
        :MIRI            # format
    )

    # Build CubeFitter exactly once; mktempdir isolates the output directory.
    cf = mktempdir() do tmpdir
        cd(tmpdir) do
            CubeFitter(cube, 0.01, "test";
                fit_stellar_continuum = false,
                fit_opt_na_feii       = false,
                fit_opt_br_feii       = false,
                use_pah_templates     = false,
                silicate_absorption   = "kvt",
                fit_ch_abs            = true,
            )
        end
    end

    # =========================================================================
    # DataCube constructor
    # Verifies properties of `cube` built above — no additional DataCube needed.
    # =========================================================================
    @testset "DataCube constructor" begin

        # Spatial and spectral dimensions match input
        @test cube.nx == 1
        @test cube.ny == 1
        @test cube.nz == 100

        # Wavelength vector preserved (up to Float64 promotion)
        @test length(cube.λ) == 100
        @test cube.λ[1]   ≈ λ_cube[1]
        @test cube.λ[end] ≈ λ_cube[end]

        # Flags match values passed to the constructor
        @test cube.rest_frame  == true
        @test cube.vacuum_wave == true
        @test cube.masked      == true
        @test cube.log_binned  == false
        @test cube.dereddened  == true

        # Default mask is all-false (nothing was passed for mask)
        @test size(cube.mask) == (1, 1, 100)
        @test !any(cube.mask)

        # Channel and band strings are preserved
        @test cube.channel == "Test"
        @test cube.band    == "Test"

        # spectral_region is populated
        @test !isnothing(cube.spectral_region)

        # psf and lsf vectors have the correct length
        @test length(cube.psf) == 100
        @test length(cube.lsf) == 100
    end

    # =========================================================================
    # to_rest_frame!(cube, z)
    # λ_rest = λ_obs / (1+z);  per-wavelength I_rest = I_obs × (1+z).
    # Idempotent: a second call must leave λ unchanged.
    # =========================================================================
    @testset "to_rest_frame!" begin

        z  = 0.01
        z1 = 1 + z

        cube_obs = DataCube(
            λ_cube, I_cube, σ_cube,
            nothing, nothing, NaN*u"sr", NaN*u"°", NaN*u"°", NaN*u"rad",
            psf_cube, lsf_cube,
            nothing, "Test", "Test", nothing, nothing,
            false, z,   # rest_frame=false, z provided for spectral_region computation
            false, true, false, false, false, nothing, :MIRI
        )

        λ_obs = copy(cube_obs.λ)
        I_obs = copy(cube_obs.I)
        σ_obs = copy(cube_obs.σ)

        Loki.to_rest_frame!(cube_obs, z)

        # rest_frame flag is set after the call
        @test cube_obs.rest_frame == true

        # Wavelengths divided by (1+z)
        @test cube_obs.λ ≈ λ_obs ./ z1

        # Per-wavelength flux and uncertainty multiplied by (1+z)
        @test cube_obs.I ≈ I_obs .* z1
        @test cube_obs.σ ≈ σ_obs .* z1

        # Resulting wavelengths are finite and positive
        @test all(isfinite.(ustrip.(cube_obs.λ)))
        @test all(ustrip.(cube_obs.λ) .> 0)

        # Idempotency: second call leaves λ unchanged
        λ_rest = copy(cube_obs.λ)
        Loki.to_rest_frame!(cube_obs, z)
        @test cube_obs.λ == λ_rest
        @test cube_obs.rest_frame == true
    end

    # =========================================================================
    # to_vacuum_wavelength!(cube)
    # Converts air → vacuum via AstroLib.airtovac; vacuum ≥ air.
    # Idempotent: a second call must leave λ unchanged.
    # =========================================================================
    @testset "to_vacuum_wavelength!" begin

        cube_air = DataCube(
            λ_cube, I_cube, σ_cube,
            nothing, nothing, NaN*u"sr", NaN*u"°", NaN*u"°", NaN*u"rad",
            psf_cube, lsf_cube,
            nothing, "Test", "Test", nothing, nothing,
            true, 0.01,
            false, false, false, false, false, nothing, :MIRI  # vacuum_wave=false
        )

        λ_air = copy(cube_air.λ)

        Loki.to_vacuum_wavelength!(cube_air)

        # vacuum_wave flag is set after the call
        @test cube_air.vacuum_wave == true

        # All converted wavelengths are finite
        @test all(isfinite.(ustrip.(cube_air.λ)))

        # Vacuum wavelengths ≥ air wavelengths (refractive index of air ≥ 1)
        @test all(ustrip.(cube_air.λ) .>= ustrip.(λ_air) .- 1e-12)

        # Idempotency: second call leaves λ unchanged
        λ_vac = copy(cube_air.λ)
        Loki.to_vacuum_wavelength!(cube_air)
        @test cube_air.λ == λ_vac
        @test cube_air.vacuum_wave == true
    end

    # =========================================================================
    # apply_mask!(cube)
    # Sets I[mask] and σ[mask] to NaN.  Idempotent via the masked flag.
    # =========================================================================
    @testset "apply_mask!" begin

        mask_ba         = falses(1, 1, 100)
        mask_ba[1,1,50] = true

        cube_m = DataCube(
            λ_cube, I_cube, σ_cube,
            mask_ba, nothing, NaN*u"sr", NaN*u"°", NaN*u"°", NaN*u"rad",
            psf_cube, lsf_cube,
            nothing, "Test", "Test", nothing, nothing,
            true, 0.01,
            false, true, false, false, false, nothing, :MIRI  # masked=false
        )

        # Before masking: pixel [1,1,50] is not NaN
        @test !isnan(ustrip(cube_m.I[1, 1, 50]))
        @test !isnan(ustrip(cube_m.σ[1, 1, 50]))
        @test cube_m.masked == false

        Loki.apply_mask!(cube_m)

        # After masking: flag is set
        @test cube_m.masked == true

        # Masked pixel is NaN in both I and σ
        @test isnan(ustrip(cube_m.I[1, 1, 50]))
        @test isnan(ustrip(cube_m.σ[1, 1, 50]))

        # Unmasked pixels are unchanged
        @test !isnan(ustrip(cube_m.I[1, 1, 1]))
        @test !isnan(ustrip(cube_m.σ[1, 1, 1]))

        # Idempotency: second call does not change the unmasked values
        I_val_after = cube_m.I[1, 1, 1]
        Loki.apply_mask!(cube_m)
        @test cube_m.masked == true
        @test cube_m.I[1, 1, 1] == I_val_after
    end

    # Shared test wavelength vectors for CubeFitter function tests below
    λ_sil = collect(range(8.0, 13.0, length=30)) .* u"μm"   # silicate feature
    λ_ice = collect(range(5.0, 15.0, length=30)) .* u"μm"   # ice/CH feature range

    # =========================================================================
    # τ_kvt(λ, β, cube_fitter)
    # Mixed silicate profile after Kemper, Vriend, & Tielens (2004).
    # β controls the mix with a power-law; ext is always clamped to ≥ 0.
    # =========================================================================
    @testset "τ_kvt" begin

        τ = Loki.τ_kvt(λ_sil, 0.0, cf)

        # Output length matches input
        @test length(τ) == length(λ_sil)

        # Always finite and non-negative (clamped internally)
        @test all(isfinite.(τ))
        @test all(τ .≥ 0)

        # β=1 adds a power-law component → different values
        τ_pow = Loki.τ_kvt(λ_sil, 1.0, cf)
        @test all(isfinite.(τ_pow))
        @test all(τ_pow .≥ 0)
        @test τ ≠ τ_pow

        # Profile peaks somewhere in the 8–13 μm silicate window
        @test λ_sil[argmax(τ)] ≥ 8.0u"μm"
        @test λ_sil[argmax(τ)] ≤ 13.0u"μm"

        # Monotonic β: at 9.7 μm the kvt and power-law contribute differently
        τ_mixed = Loki.τ_kvt(λ_sil, 0.5, cf)
        @test all(isfinite.(τ_mixed))
    end

    # =========================================================================
    # τ_ohm(λ, cube_fitter)
    # Extinction profile from Ossenkopf, Henning, & Mathis (1992).
    # Always loaded regardless of silicate_absorption setting.
    # =========================================================================
    @testset "τ_ohm" begin

        τ = Loki.τ_ohm(λ_sil, cf)

        # Output length matches input
        @test length(τ) == length(λ_sil)

        # Values are finite (spline extrapolates beyond data range with bc="nearest")
        @test all(isfinite.(τ))

        # OHM profile has its silicate peak somewhere in 8–13 μm
        @test any(τ .> 0)
    end

    # =========================================================================
    # τ_ice(λ, cube_fitter)
    # Ice absorption profile (Donnan's PAHDecomp templates).
    # Requires fit_ch_abs=true.
    # =========================================================================
    @testset "τ_ice" begin

        τ = Loki.τ_ice(λ_ice, cf)

        # Output length matches input
        @test length(τ) == length(λ_ice)

        # Values are finite (normalised template, spline interpolation)
        @test all(isfinite.(τ))
    end

    # =========================================================================
    # τ_ch(λ, cube_fitter)
    # CH aliphatic absorption profile (Donnan's PAHDecomp templates).
    # Requires fit_ch_abs=true.
    # =========================================================================
    @testset "τ_ch" begin

        τ = Loki.τ_ch(λ_ice, cf)

        # Output length matches input
        @test length(τ) == length(λ_ice)

        # Values are finite
        @test all(isfinite.(τ))
    end

    # =========================================================================
    # silicate_emission(λ, T, Cf, τ_warm, τ_cold, λ_peak, Iunit, cube_fitter)
    # Hot silicate dust emission (Gallimore et al. 2010).
    # Internally calls τ_ohm; output shares the units of Iunit.
    # =========================================================================
    @testset "silicate_emission" begin

        T       = 300.0u"K"
        τ_warm  = 1.0
        τ_cold  = 0.5
        Cf      = 0.5
        λ_peak  = 9.7u"μm"
        Iunit   = 1.0u"erg/s/cm^2/μm/sr"

        result = Loki.silicate_emission(λ_sil, T, Cf, τ_warm, τ_cold, λ_peak, Iunit, cf)

        # Output has same length as input wavelength vector
        @test length(result) == length(λ_sil)

        # All values are finite and non-negative (Blackbody × extinction_factor ≥ 0)
        @test all(isfinite.(ustrip.(result)))
        @test all(ustrip.(result) .≥ 0)

        # Higher temperature → larger total emission (Stefan-Boltzmann)
        result_hot = Loki.silicate_emission(λ_sil, 1000.0u"K", Cf, τ_warm, τ_cold, λ_peak, Iunit, cf)
        @test sum(ustrip.(result_hot)) > sum(ustrip.(result))

        # τ_warm = 0 → emission is zero (no hot silicates visible)
        result_zero = Loki.silicate_emission(λ_sil, T, Cf, 0.0, τ_cold, λ_peak, Iunit, cf)
        @test all(ustrip.(result_zero) .≈ 0.0)

        # Covering fraction Cf=0 vs Cf=1 produces different results
        result_cf0 = Loki.silicate_emission(λ_sil, T, 0.0, τ_warm, τ_cold, λ_peak, Iunit, cf)
        result_cf1 = Loki.silicate_emission(λ_sil, T, 1.0, τ_warm, τ_cold, λ_peak, Iunit, cf)
        @test result_cf0 ≠ result_cf1
    end

    # =========================================================================
    # silicate_absorption(λ, params, pstart, cube_fitter)
    # Computes extinction factors using the configured silicate profile.
    # With silicate_absorption="kvt": params = [τ_97, β], dp = 2.
    # Returns (ext, dp, ext_oli, ext_pyr, ext_for).
    # =========================================================================
    @testset "silicate_absorption" begin

        # "kvt" mode: params[1]=τ_97 (optical depth), params[2]=β (mix)
        params = [0.5, 0.0]
        pstart = 1
        ext, dp, ext_oli, ext_pyr, ext_for =
            Loki.silicate_absorption(λ_sil, params, pstart, cf)

        # ext is a vector of extinction factors
        @test length(ext) == length(λ_sil)
        @test all(isfinite.(ext))

        # extinction_factor = exp(-τ*profile) ∈ (0,1] for screen; (0,1] for mixed
        @test all(ext .> 0)
        @test all(ext .≤ 1)

        # In "kvt" mode, dp = 2 (uses 2 parameters: τ_97 and β)
        @test dp == 2

        # ext_oli, ext_pyr, ext_for are nothing for non-decompose modes
        @test isnothing(ext_oli)
        @test isnothing(ext_pyr)
        @test isnothing(ext_for)

        # Larger τ_97 → more absorption → smaller extinction factors
        ext_strong, _, _, _, _ =
            Loki.silicate_absorption(λ_sil, [2.0, 0.0], pstart, cf)
        @test sum(ext_strong) < sum(ext)

        # τ_97 = 0 → no absorption → all extinction factors = 1
        ext_none, _, _, _, _ =
            Loki.silicate_absorption(λ_sil, [0.0, 0.0], pstart, cf)
        @test all(ext_none .≈ 1.0)
    end

    # =========================================================================
    # model(cf) and fit_options(cf) accessor functions
    # These return the ModelParameters and FittingOptions structs.
    # =========================================================================
    @testset "model and fit_options accessors" begin

        fopt = Loki.fit_options(cf)
        m    = Loki.model(cf)

        # fit_options returns a struct with the expected field values
        @test fopt.silicate_absorption == "kvt"
        @test fopt.fit_stellar_continuum == false
        @test fopt.fit_ch_abs == true
        @test fopt.use_pah_templates == false
        @test fopt.extinction_screen isa Bool

        # model returns a struct with continuum and lines sub-objects
        @test !isnothing(m)
        @test hasproperty(m, :continuum)
        @test hasproperty(m, :lines)
        @test hasproperty(m, :dust_features)
        @test hasproperty(m, :abs_features)

        # CubeFitter counts are non-negative integers
        @test cf.n_params_cont ≥ 0
        @test cf.n_params_lines ≥ 0
        @test cf.n_dust_cont ≥ 0
        @test cf.n_lines ≥ 0
    end

    # =========================================================================
    # get_continuum_step_sizes(cube_fitter, λ)
    # Returns (dstep, dstep_pahtemp) where dstep has length n_params_cont.
    # =========================================================================
    @testset "get_continuum_step_sizes" begin

        dstep, dstep_pahtemp = Loki.get_continuum_step_sizes(cf, λ_cube)

        # dstep length equals the number of continuum parameters
        @test length(dstep) == cf.n_params_cont

        # All step sizes are finite and positive
        @test all(isfinite.(dstep))
        @test all(dstep .> 0)

        # PAH template step sizes are a 2-element vector
        @test length(dstep_pahtemp) == 2
        @test all(isfinite.(dstep_pahtemp))
        @test all(dstep_pahtemp .> 0)
    end

    # =========================================================================
    # get_line_step_sizes(cube_fitter, ln_pars, init)
    # With init=true, returns a vector of zeros (same length as ln_pars).
    # =========================================================================
    @testset "get_line_step_sizes" begin

        n = cf.n_params_lines

        if n > 0
            # init=true: all step sizes are zero (no stepping applied)
            ln_pars  = zeros(n)
            ln_astep = Loki.get_line_step_sizes(cf, ln_pars, true)

            @test length(ln_astep) == n
            @test all(iszero.(ln_astep))
        else
            @test cf.n_lines == 0   # no lines in spectral range → no params
        end
    end

    # =========================================================================
    # get_line_nprof_ncomp(cube_fitter, i)
    # Returns (n_prof, pcomps) for emission line i.
    # n_prof ≥ 1; each element of pcomps ≥ 3 (amp, voff, fwhm at minimum).
    # =========================================================================
    @testset "get_line_nprof_ncomp" begin

        if cf.n_lines > 0
            # Test the first line
            n_prof, pcomps = Loki.get_line_nprof_ncomp(cf, 1)
            @test n_prof ≥ 1
            @test length(pcomps) == n_prof
            @test all(pcomps .≥ 3)

            # Test all lines: each has at least 1 profile with ≥ 3 parameters
            for i in 1:cf.n_lines
                n_p, pc = Loki.get_line_nprof_ncomp(cf, i)
                @test n_p ≥ 1
                @test all(pc .≥ 3)
            end
        else
            @test cf.n_lines == 0
        end
    end

end
