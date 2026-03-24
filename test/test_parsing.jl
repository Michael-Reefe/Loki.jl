###############################################################################
# Tests for configuration file parsing helpers
# Source: src/util/parsing.jl
#
# Functions tested:
#   pah_name_to_float(name)     — extract PAH wavelength from feature name
#   validate_options_file(opts) — check all required option keys are present
#   validate_ir_file(dust)      — check IR dust config structure
#   validate_optical_file(opt)  — check optical config structure
#   validate_lines_file(lines)  — check emission line config structure
###############################################################################

# ---------------------------------------------------------------------------
# Helpers: minimal valid configuration dictionaries
# ---------------------------------------------------------------------------

function _valid_options()
    Dict(
        "n_bootstrap"                    => 0,
        "silicate_absorption"            => "kvt",
        "extinction_curve"               => "calz",
        "extinction_screen"              => true,
        "fit_sil_emission"               => false,
        "fit_opt_na_feii"                => false,
        "fit_opt_br_feii"                => false,
        "fit_all_global"                 => false,
        "use_pah_templates"              => false,
        "fit_joint"                      => false,
        "fit_uv_bump"                    => false,
        "fit_covering_frac"              => false,
        "parallel"                       => false,
        "plot_spaxels"                   => false,
        "plot_maps"                      => false,
        "save_fits"                      => false,
        "overwrite"                      => false,
        "track_memory"                   => false,
        "track_convergence"              => false,
        "save_full_model"                => false,
        "line_test_lines"                => String[],
        "line_test_threshold"            => 0.003,
        "plot_line_test"                 => false,
        "make_movies"                    => false,
        "cosmology"                      => Dict("h" => 0.7, "omega_m" => 0.3,
                                                 "omega_K" => 0.0, "omega_r" => 0.0),
        "parallel_strategy"              => "serial",
        "random_seed"                    => 42,
        "sys_err"                        => 0.0,
        "olivine_y"                      => 0.5,
        "pyroxene_x"                     => 0.5,
        "grain_size"                     => "small",
        "fit_stellar_continuum"          => false,
        "fit_temp_multexp"               => false,
        "decompose_lock_column_densities"=> false,
        "linemask_width"                 => 10.0,
        "map_snr_thresh"                 => 3.0,
    )
end

function _valid_dust()
    param = Dict("val" => 1.0, "plim" => [0.0, 2.0], "locked" => false)
    Dict(
        "dust_features" => Dict(
            "PAH_620" => Dict(
                "wave" => Dict("val" => 6.2, "plim" => [-0.1, 0.1], "locked" => false),
                "fwhm" => Dict("val" => 0.18, "plim" => [0.5, 2.0], "locked" => false),
            )
        ),
        "extinction" => Dict(
            "tau_9_7"  => copy(param),
            "tau_ice"  => copy(param),
            "tau_ch"   => copy(param),
            "beta"     => copy(param),
        ),
        "hot_dust" => Dict(
            "temp"      => copy(param),
            "frac"      => copy(param),
            "tau_warm"  => copy(param),
            "tau_cold"  => copy(param),
            "peak"      => copy(param),
        ),
    )
end

function _valid_optical()
    kin  = Dict(
        "vel"   => Dict("val" => 0.0,   "plim" => [-500.0, 500.0], "locked" => false),
        "vdisp" => Dict("val" => 100.0, "plim" => [10.0, 500.0],   "locked" => false),
    )
    p0 = Dict("val" => 0.0, "plim" => [-1.0, 1.0], "locked" => false)
    ext = Dict(
        "E_BV"       => copy(p0),
        "E_BV_factor"=> copy(p0),
        "uv_slope"   => copy(p0),
        "frac"       => copy(p0),
    )
    Dict(
        "extinction"           => ext,
        "stellar_kinematics"   => deepcopy(kin),
        "na_feii_kinematics"   => deepcopy(kin),
        "br_feii_kinematics"   => deepcopy(kin),
    )
end

function _valid_lines()
    Dict(
        "default_sort_order" => 1,
        "tie_voigt_mixing"   => false,
        "lines"   => Dict("halpha" => Dict("wave" => 0.6563, "latex" => "H\\alpha",
                                           "annotate" => true, "unit" => "μm")),
        "profiles"=> Dict("default" => "Gaussian"),
        "acomps"  => Dict(),
        "n_acomps"=> 0,
        "rel_amp" => false,
        "rel_fwhm"=> false,
        "rel_voff"=> false,
        "voff"    => Dict("val" => 0.0,   "plim" => [-500.0, 500.0], "locked" => false),
        "fwhm"    => Dict("val" => 100.0, "plim" => [10.0, 1000.0],  "locked" => false),
        "h3"      => Dict("val" => 0.0,   "plim" => [-0.5, 0.5],     "locked" => false),
        "h4"      => Dict("val" => 0.0,   "plim" => [-0.5, 0.5],     "locked" => false),
    )
end

@testset "Parsing helpers" begin

    # =========================================================================
    # pah_name_to_float(name)
    # Extracts the wavelength from "PREFIX_XYZW" → X.YZ (last 3 digits, decimal before last 2)
    # =========================================================================
    @testset "pah_name_to_float" begin

        @test Loki.pah_name_to_float("PAH_620")  ≈ 6.20  rtol=1e-10
        @test Loki.pah_name_to_float("PAH_1130") ≈ 11.30 rtol=1e-10
        @test Loki.pah_name_to_float("PAH_770")  ≈ 7.70  rtol=1e-10
        @test Loki.pah_name_to_float("PAH_850")  ≈ 8.50  rtol=1e-10

        # Works with any prefix (not just "PAH")
        @test Loki.pah_name_to_float("feature_330") ≈ 3.30 rtol=1e-10

        # Return type is Float64
        @test typeof(Loki.pah_name_to_float("PAH_620")) == Float64
    end

    # =========================================================================
    # validate_options_file(options)
    # =========================================================================
    @testset "validate_options_file" begin

        # Valid dict → no error
        @test_nowarn Loki.validate_options_file(_valid_options())

        # Missing a top-level key → AssertionError
        opts_missing = _valid_options()
        delete!(opts_missing, "n_bootstrap")
        @test_throws AssertionError Loki.validate_options_file(opts_missing)

        # Missing cosmology sub-key → AssertionError
        opts_missing_cosmo = _valid_options()
        delete!(opts_missing_cosmo["cosmology"], "h")
        @test_throws AssertionError Loki.validate_options_file(opts_missing_cosmo)
    end

    # =========================================================================
    # validate_ir_file(dust)
    # =========================================================================
    @testset "validate_ir_file" begin

        # Valid dict → no error
        @test_nowarn Loki.validate_ir_file(_valid_dust())

        # Missing top-level key → AssertionError
        dust_missing = _valid_dust()
        delete!(dust_missing, "hot_dust")
        @test_throws AssertionError Loki.validate_ir_file(dust_missing)

        # Missing extinction sub-key → AssertionError
        dust_missing2 = _valid_dust()
        delete!(dust_missing2["extinction"], "tau_9_7")
        @test_throws AssertionError Loki.validate_ir_file(dust_missing2)

        # Missing "val" inside a nested parameter → AssertionError
        dust_missing3 = _valid_dust()
        delete!(dust_missing3["extinction"]["tau_9_7"], "val")
        @test_throws AssertionError Loki.validate_ir_file(dust_missing3)
    end

    # =========================================================================
    # validate_optical_file(optical)
    # =========================================================================
    @testset "validate_optical_file" begin

        # Valid dict → no error
        @test_nowarn Loki.validate_optical_file(_valid_optical())

        # Missing top-level key → AssertionError
        opt_missing = _valid_optical()
        delete!(opt_missing, "stellar_kinematics")
        @test_throws AssertionError Loki.validate_optical_file(opt_missing)

        # Missing extinction sub-key → AssertionError
        opt_missing2 = _valid_optical()
        delete!(opt_missing2["extinction"], "E_BV")
        @test_throws AssertionError Loki.validate_optical_file(opt_missing2)
    end

    # =========================================================================
    # validate_lines_file(lines)
    # =========================================================================
    @testset "validate_lines_file" begin

        # Valid dict → no error
        @test_nowarn Loki.validate_lines_file(_valid_lines())

        # Missing top-level required key → AssertionError
        lines_missing = _valid_lines()
        delete!(lines_missing, "voff")
        @test_throws AssertionError Loki.validate_lines_file(lines_missing)

        # Missing "default" in profiles → AssertionError
        lines_missing2 = _valid_lines()
        delete!(lines_missing2["profiles"], "default")
        @test_throws AssertionError Loki.validate_lines_file(lines_missing2)

        # Missing a line sub-key ("wave") → AssertionError
        lines_missing3 = _valid_lines()
        delete!(lines_missing3["lines"]["halpha"], "wave")
        @test_throws AssertionError Loki.validate_lines_file(lines_missing3)
    end

end
