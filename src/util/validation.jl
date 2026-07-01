#=
This file contains input-validation helpers that check user-supplied settings and data up front,
raising clear, user-friendly errors for malformed inputs instead of letting them crash deep inside
the fitting routines or silently produce wrong scientific results. None of these are exported; they
are called internally by the DataCube/CubeFitter constructors and the fitting loop.
=#


"""
    validate_fit_settings(out)

Validate the merged options dictionary (TOML defaults overridden by keyword arguments) before a
`CubeFitter` is constructed. Checks numeric ranges, enum-like string/symbol options, and
mutually-exclusive flags, raising a descriptive error that names the offending option and the valid
range/values. Called from the `CubeFitter` constructor after `cubefitter_add_default_options!`, so
symbol-valued options (`plot_spaxels`, `apoly_type`, `mpoly_type`) are already `Symbol`s here.
"""
function validate_fit_settings(out::Dict)

    @debug "validate_fit_settings: validating $(length(keys(out))) merged options"
    chk(cond, msg) = cond || error("Invalid fitting option: " * msg)

    # --- numeric ranges (only checked if present). Some of these are dimensionless from the TOML but may
    #     be passed WITH units as keyword arguments (e.g. linemask_width=1000u"km/s", grain_size=0.1u"μm",
    #     spaxel_timelimit=3600u"s"); ustrip(...) handles both — it is a no-op on a plain number and strips
    #     units off a Quantity — so the comparisons never throw a DimensionError. ---
    haskey(out, :n_bootstrap) && chk(out[:n_bootstrap] isa Integer && out[:n_bootstrap] ≥ 0,
        "n_bootstrap must be a non-negative integer (got $(out[:n_bootstrap])).")
    haskey(out, :line_test_threshold) && chk(0 < ustrip(out[:line_test_threshold]) < 1,
        "line_test_threshold must be a probability in the open interval (0, 1) (got $(out[:line_test_threshold])).")
    haskey(out, :map_snr_thresh) && chk(ustrip(out[:map_snr_thresh]) ≥ 0,
        "map_snr_thresh must be ≥ 0 (got $(out[:map_snr_thresh])).")
    haskey(out, :spaxel_timelimit) && chk(ustrip(out[:spaxel_timelimit]) > 0,
        "spaxel_timelimit must be > 0 (got $(out[:spaxel_timelimit])).")
    haskey(out, :ssp_regularize) && chk(ustrip(out[:ssp_regularize]) ≥ 0,
        "ssp_regularize must be ≥ 0 (got $(out[:ssp_regularize])).")
    haskey(out, :sys_err) && chk(0 ≤ ustrip(out[:sys_err]) < 1,
        "sys_err must be a fractional systematic error in [0, 1) (got $(out[:sys_err])).")
    haskey(out, :linemask_width) && chk(ustrip(out[:linemask_width]) > 0,
        "linemask_width must be > 0 (got $(out[:linemask_width])).")
    haskey(out, :apoly_degree) && chk(out[:apoly_degree] isa Integer && out[:apoly_degree] ≥ -1,
        "apoly_degree must be an integer ≥ -1 (use -1 to disable; got $(out[:apoly_degree])).")
    haskey(out, :mpoly_degree) && chk(out[:mpoly_degree] isa Integer && out[:mpoly_degree] ≥ -1,
        "mpoly_degree must be an integer ≥ -1 (use -1 to disable; got $(out[:mpoly_degree])).")
    haskey(out, :grain_size) && chk(ustrip(out[:grain_size]) > 0,
        "grain_size must be > 0 (got $(out[:grain_size])).")
    haskey(out, :pyroxene_x) && chk(0 ≤ ustrip(out[:pyroxene_x]) ≤ 1,
        "pyroxene_x (pyroxene Mg fraction) must be in [0, 1] (got $(out[:pyroxene_x])).")
    haskey(out, :olivine_y) && chk(0 ≤ ustrip(out[:olivine_y]) ≤ 1,
        "olivine_y (olivine Mg fraction) must be in [0, 1] (got $(out[:olivine_y])).")

    # --- enum-like options. Compare on string(...) so the check is robust whether the option arrives as a
    #     String (extinction_curve, silicate_absorption — TOML strings, also coerced from a Symbol kwarg in
    #     cubefitter_add_default_options!) or a Symbol (apoly_type, mpoly_type, plot_spaxels — converted to
    #     Symbol there). Valid sets mirror the `else error(...)` sites that consume each option. ---
    if haskey(out, :extinction_curve)
        valid = ("calz", "ccm")
        chk(string(out[:extinction_curve]) in valid,
            "extinction_curve must be one of $(valid) (got $(out[:extinction_curve])).")
    end
    if haskey(out, :silicate_absorption)
        valid = ("kvt", "ct", "ohm", "d+", "decompose", "custom")
        chk(string(out[:silicate_absorption]) in valid,
            "silicate_absorption must be one of $(valid) (got $(out[:silicate_absorption])).")
    end
    for key in (:apoly_type, :mpoly_type)
        if haskey(out, key)
            valid = ("Legendre", "Chebyshev")
            chk(string(out[key]) in valid, "$(key) must be one of $(valid) (got $(out[key])).")
        end
    end
    if haskey(out, :plot_spaxels)
        valid = ("pyplot", "plotly", "both", "none")
        chk(string(out[:plot_spaxels]) in valid, "plot_spaxels must be one of $(valid) (got $(out[:plot_spaxels])).")
    end

    # --- mutually-exclusive flags (caught here so the failure is up-front, not deep in fit_spaxel) ---
    if get(out, :fit_joint, false) && get(out, :use_pah_templates, false)
        error("Invalid fitting option: fit_joint and use_pah_templates are mutually exclusive; " *
              "set at most one of them to true.")
    end

    # --- degeneracy warning (no lock): the template multiplicative exponentials and the multiplicative
    #     polynomial both smoothly modulate the template continuum, so they partially trade off. Only warn
    #     when templates are actually present (size(templates, 4) is the template count). ---
    if get(out, :fit_temp_multexp, false) && get(out, :mpoly_degree, 0) ≥ 1 &&
       haskey(out, :templates) && size(out[:templates], 4) > 0
        @warn "fit_temp_multexp=true with mpoly_degree=$(out[:mpoly_degree]): the multiplicative " *
              "exponentials applied to the templates and the multiplicative polynomial both modulate the " *
              "template continuum and can partially trade off (degenerate). Consider disabling one."
    end

    @debug "validate_fit_settings: all option checks passed"
    nothing
end


"""
    validate_datacube_inputs(λ; psf=nothing, lsf=nothing)

Validate the inputs to a `DataCube` that the constructor does not already check. The `DataCube`
constructor asserts the array *shapes* (that `λ` is 1-D, that I/σ/mask are the same size, that the
spectral axis length matches `length(λ)`, and that any PSF/LSF vectors match `length(λ)`); this
function adds the checks it does not perform: that the wavelength vector has units of length and is
finite, strictly positive, and strictly increasing, and that any PSF/LSF FWHM *values* are finite
and positive. It is called from the constructor after those shape asserts have run.
"""
function validate_datacube_inputs(λ::AbstractVector; psf::Union{AbstractVector,Nothing}=nothing,
    lsf::Union{AbstractVector,Nothing}=nothing)

    @debug "validate_datacube_inputs: nλ=$(length(λ)), λ unit=$(unit(λ[1]))"

    # Wavelength vector: must be a length, finite, strictly positive, and strictly increasing
    @assert dimension(eltype(λ)) == dimension(u"m") "The wavelength vector must have units of length " *
        "(e.g. μm or Å); got units of $(unit(λ[1])). If your spectral axis is in frequency or velocity, " *
        "convert it to wavelength before constructing a DataCube."
    @assert all(isfinite, ustrip.(λ)) "The wavelength vector contains NaN or Inf values."
    @assert all(λ .> zero(λ[1])) "The wavelength vector must be strictly positive."
    @assert all(diff(λ) .> zero(λ[1])) "The wavelength vector must be strictly monotonically increasing " *
        "(no repeated or out-of-order values)."

    # PSF / LSF FWHM values must be finite and positive (the constructor already checks their length)
    if !isnothing(psf)
        @assert all(isfinite, ustrip.(psf)) && all(psf .> zero(psf[1])) "The PSF FWHM vector must be " *
            "finite and strictly positive."
    end
    if !isnothing(lsf)
        @assert all(isfinite, ustrip.(lsf)) && all(lsf .> zero(lsf[1])) "The LSF FWHM vector must be " *
            "finite and strictly positive."
    end

    nothing
end


"""
    validate_fit_result(res, label, coords) -> Bool

Inspect a CMPFit result for a failed or degenerate fit. Returns `true` if the result looks usable,
or `false` (after logging a warning naming the spaxel and fit stage) if the solver reported a hard
failure (`status < 0`) or produced non-finite best-fit parameters or uncertainties. The caller
should skip the spaxel when this returns `false`, consistent with Loki's graceful handling of
unusable spaxels (a bad spaxel must never abort a full-cube fit).
"""
function validate_fit_result(res, label::AbstractString, coords)
    if res.status < 0
        @warn "The $label fit for spaxel $coords failed (CMPFit status $(res.status)); the spaxel will be skipped."
        return false
    end
    if !all(isfinite, res.param)
        @warn "The $label fit for spaxel $coords produced non-finite parameters; the spaxel will be skipped."
        return false
    end
    if isdefined(res, :perror) && !isnothing(res.perror) && !all(isfinite, res.perror)
        @warn "The $label fit for spaxel $coords produced non-finite parameter uncertainties; results may be unreliable."
    end
    true
end
