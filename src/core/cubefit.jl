#=
This file contains the CubeFitter struct and its related functions.  The CubeFitter is how
fitting is performed with Loki.
=#

############################## PARAMETER / MODEL STRUCTURES ####################################


"""
    ParamMaps{T<:Real}

A basic structure for holding parameter best-fit values and errors along with the parameter names to 
keep track of where each parameter is located.

# Fields
- `data::Array{T,3}`: A 3D array holding the best-fit parameters for each spaxel.
- `err_upp:Array{T,3}`: A 3D array holding the upper uncertainties for each spaxel.
- `err_low:Array{T,3}`: A 3D array holding the lower uncertainties for each spaxel.
- `names::Vector{String}`: The name of each fit parameter
- `units::Vector{String}`: The units of each fit parameter
- `labels::Vector{AbstractString}`: The labels of each fit parameter
- `restframe_transform::Vector{Int}`: Which parameters need to be transformed back into the observed frame after fitting.
- `log_transform::BitVector`: Which parameters need to be log transformed after fitting.
- `normalize::BitVector`: Which parameters need to be de-normalized after fitting.
- `line_transform::BitVector`: Which emission line parameters need special transformations after fitting.
- `perfreq_transform::BitVector`: Which parameters need to be transformed from per-unit-wavelength to per-unit-frequency flux units.
"""
struct ParamMaps{T<:Real}

    data::Array{T,3}
    err_upp::Array{T,3}
    err_low::Array{T,3}
    names::Vector{String}
    units::Vector{String}
    labels::Vector{AbstractString}

    # 1 = flux transform (multiply for per-frequency flux, divide for per-wavelength flux)
    # 2 = wavelength transform (always multiply), 
    restframe_transform::Vector{Int}  
    # Always assumed log base 10
    log_transform::BitVector
    # Normalization factor depends on spectral region
    normalize::BitVector
    # Relative line parameter transformations depend on the type
    line_transform::BitVector
    # Transform per-wavelength flux units to per-frequency flux units
    perfreq_transform::BitVector

end


"""
    Helper functions for indexing into the ParamMaps object with strings
"""
function get(parammap::ParamMaps, pname::String)
    ind = findfirst(parammap.names .== pname)
    parammap.data[:, :, ind]
end
function get(parammap::ParamMaps, pnames::Vector{String})
    inds = [findfirst(parammap.names .== pname) for pname in pnames]
    parammap.data[:, :, inds]
end
function get(parammap::ParamMaps, index::CartesianIndex, pname::String)
    ind = findfirst(parammap.names .== pname)
    parammap.data[index, ind]
end
function get(parammap::ParamMaps, index::CartesianIndex, pnames::Vector{String})
    inds = [findfirst(parammap.names .== pname) for pname in pnames]
    parammap.data[index, inds]
end

function get_err(parammap::ParamMaps, pname::String)
    ind = findfirst(parammap.names .== pname)
    parammap.err_upp[:, :, ind], parammap.err_low[:, :, ind]
end
function get_err(parammap::ParamMaps, pnames::Vector{String})
    inds = [findfirst(parammap.names .== pname) for pname in pnames]
    parammap.err_upp[:, :, inds], parammap.err_low[:, :, inds]
end
function get_err(parammap::ParamMaps, index::CartesianIndex, pname::String)
    ind = findfirst(parammap.names .== pname)
    parammap.err_upp[index, ind], parammap.err_low[index, ind]
end
function get_err(parammap::ParamMaps, index::CartesianIndex, pnames::Vector{String})
    inds = [findfirst(parammap.names .== pname) for pname in pnames]
    parammap.err_upp[index, inds], parammap.err_low[index, inds]
end

function get_label(parammap::ParamMaps, pname::String)
    ind = findfirst(parammap.names .== pname)
    parammap.labels[ind]
end
function get_label(parammap::ParamMaps, pnames::Vector{String})
    inds = [findfirst(parammap.names .== pname) for pname in pnames]
    parammap.labels[inds]
end


abstract type CubeModel end


"""
    _get_line_names_and_transforms(cf_lines, n_lines, n_comps, flexible_wavesol)

Helper function for getting a vector of line parameter names and boolean vectors 
to determine what transformations to apply on the final parameters.
"""
function _get_line_names_and_transforms(cf_lines::TransitionLines, n_lines::Integer, n_comps::Integer,
        flexible_wavesol::Bool, lines_allow_negative::Bool; perfreq::Int=0)

    line_names = String[]
    line_units = String[]
    line_labels = String[]
    line_restframe = Int[]
    line_log = Int[]
    line_normalize = Int[]
    line_perfreq = Int[]
    line_names_extra = String[]
    line_units_extra = String[]
    line_labels_extra = String[]
    line_extra_restframe = Int[]
    line_extra_log = Int[]
    line_extra_normalize = Int[]
    line_extra_perfreq = Int[]

    for i ∈ 1:n_lines

        line = "lines." * string(cf_lines.names[i])

        for j ∈ 1:n_comps
            if !isnothing(cf_lines.profiles[i, j])

                line_j = "lines." * string(cf_lines.names[i]) * ".$(j)"

                pnames = ["amp", "voff", "fwhm"]
                punits = [lines_allow_negative ? "erg.s-1.cm-2.Hz-1" : "log(erg.s-1.cm-2.Hz-1.sr-1)", "km/s", "km/s"]
                plabels = [lines_allow_negative ? L"$I$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$)" : L"$\log_{10}(I / $ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1})$", 
                    L"$v_{\rm off}$ (km s$^{-1}$)", L"FWHM (km s$^{-1}$)"]
                p_rf = [1, 0, 0]
                p_log = [lines_allow_negative ? 0 : 1, 0, 0]
                p_norm = [1, 0, 0]
                p_pf = [perfreq, 0, 0]
                # Need extra voff parameter if using the flexible_wavesol keyword
                if !isnothing(cf_lines.tied_voff[i, j]) && flexible_wavesol && isone(j)
                    pnames = ["amp", "voff", "voff_indiv", "fwhm"]
                    punits = [lines_allow_negative ? "erg.s-1.cm-2.Hz-1" : "log(erg.s-1.cm-2.Hz-1.sr-1)", "km/s", "km/s", "km/s"]
                    plabels = [lines_allow_negative ? L"$I$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$)" : L"$\log_{10}(I / $ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1})$",
                        L"$v_{\rm off}$ (km s$^{-1}$)", L"$v_{\rm off,indiv}$ (km s$^{-1}$)", L"FWHM (km s$^{-1}$)"]
                    p_rf = [1, 0, 0, 0]
                    p_log = [lines_allow_negative ? 0 : 1, 0, 0, 0]
                    p_norm = [1, 0, 0, 0]
                    p_pf = [perfreq, 0, 0, 0]
                end
                # Add 3rd and 4th order moments (skewness and kurtosis) for Gauss-Hermite profiles
                if cf_lines.profiles[i, j] == :GaussHermite
                    pnames = [pnames; "h3"; "h4"]
                    punits = [punits; "-"; "-"]
                    plabels = [plabels; L"$h_3$"; L"$h_4$"]
                    p_rf = [p_rf; 0; 0]
                    p_log = [p_log; 0; 0]
                    p_norm = [p_norm; 0; 0]
                    p_pf = [p_pf; 0; 0]
                # Add mixing parameter for Voigt profiles, but only if NOT tying it
                elseif cf_lines.profiles[i, j] == :Voigt
                    pnames = [pnames; "mixing"]
                    punits = [punits; "-"]
                    plabels = [plabels; L"$\eta$"]
                    p_rf = [p_rf; 0]
                    p_log = [p_log; 0]
                    p_norm = [p_norm; 0]
                    p_pf = [p_pf; 0]
                end
                pnames_extra = ["flux"; "eqw"; "SNR"]
                punits_extra = [lines_allow_negative ? "erg.s-1.cm-2" : "log(erg.s-1.cm-2)", "um", "-"]
                plabels_extra = [lines_allow_negative ? L"$F$ (erg s$^{-1}$ cm$^{-2}$)" : L"$\log_{10}(F /$ erg s$^{-1}$ cm$^{-2}$)", L"$W_{\rm eq}$ ($\mu$m)", L"$S/N$"]
                p_extra_rf = [0, 2, 0]
                p_extra_log = [lines_allow_negative ? 0 : 1, 0, 0]
                p_extra_norm = [0, 0, 0]
                p_extra_pf = [0, 0, 0]
                # Append parameters for flux, equivalent width, and signal-to-noise ratio, which are NOT fitting parameters, but are of interest
                append!(line_names, [join([line_j, pname], ".") for pname in pnames])
                append!(line_units, punits)
                append!(line_labels, plabels)
                append!(line_restframe, p_rf)
                append!(line_log, p_log)
                append!(line_normalize, p_norm)
                append!(line_perfreq, p_pf)
                append!(line_names_extra, [join([line_j, pname], ".") for pname in pnames_extra])
                append!(line_units_extra, punits_extra)
                append!(line_labels_extra, plabels_extra)
                append!(line_extra_restframe, p_extra_rf)
                append!(line_extra_log, p_extra_log)
                append!(line_extra_normalize, p_extra_norm)
                append!(line_extra_perfreq, p_extra_pf)
            end
        end
        pnames_comp = ["n_comps", "w80", "delta_v", "vmed", "vpeak"]
        punits_comp = ["-", "km/s", "km/s", "km/s", "km/s"]
        plabels_comp = [L"$n_{\rm comp}$", L"$W_{80}$ (km s$^{-1}$)", L"$\Delta v$ (km s$^{-1}$)", L"$v_{\rm med}$ (km s$^{-1}$)",
            L"$v_{\rm peak}$ (km s$^{-1}$)"]
        p_comp_rf = [0, 0, 0, 0, 0]
        p_comp_log = [0, 0, 0, 0, 0]
        p_comp_norm = [0, 0, 0, 0, 0]
        p_comp_pf = [0, 0, 0, 0, 0]
        append!(line_names_extra, [join([line, pname], ".") for pname in pnames_comp])
        append!(line_units_extra, punits_comp)
        append!(line_labels_extra, plabels_comp)
        append!(line_extra_restframe, p_comp_rf)
        append!(line_extra_log, p_comp_log)
        append!(line_extra_normalize, p_comp_norm)
        append!(line_extra_perfreq, p_comp_pf)
    end
    @debug "line maps with keys $line_names"
    @debug "extra line maps with keys $line_names_extra"

    line_names, line_names_extra, line_units, line_units_extra, line_labels, line_labels_extra, line_restframe, line_extra_restframe, 
        line_log, line_extra_log, line_normalize, line_extra_normalize, line_perfreq, line_extra_perfreq
end


"""
    CubeFitter(cube, z, name; <keyword arguments>)

This is the main structure used for fitting IFU cubes, containing all of the necessary data, metadata,
fitting options, and associated functions for generating ParamMaps and CubeModel structures to handle the outputs 
of all the fits.  This is essentially the "configuration object" that tells the rest of the fitting code how
to run. The actual fitting functions (`fit_spaxel` and `fit_cube!`) require an instance of this structure.

Other than `cube`, `z`, and `name`, all fields listed below will be populated by the defaults given in the 
`config.toml`, `dust.toml`, `lines.toml`, and `optical.toml` configuration files, or by quantities derived from
these options. For more detailed explanations on these options, please refer to the options files and the README file.

# Fields {T<:Real, S<:Integer, C<:Complex}

## Data
- `cube::DataCube`: The main DataCube object containing the cube that is being fit
- `z::Real`: The redshift of the target that is being fit
- `name::String`: The name of the fitting run being performed.
- `spectral_region::Symbol`: Either :MIR for mid-infrared, or :OPT for optical.

## Basic configuration options
- `user_mask::Union{Vector{<:Tuple},Nothing}`: An optional mask specifying areas of the spectrum to ignore during fitting.
- `plot_spaxels::Symbol`: A Symbol specifying the plotting backend to be used when plotting individual spaxel fits, can
    be either `:pyplot`, `:plotly`, or `:both`
- `plot_maps::Bool`: Whether or not to plot 2D maps of the best-fit parameters after the fitting is finished
- `plot_range::Union{Vector{<:Tuple},Nothing}`: An optional list of spectral regions to extract zoomed-in plots from (useful for plotting
specific emission lines of interest).
- `parallel::Bool`: Whether or not to fit multiple spaxels in parallel using multiprocessing
- `save_fits::Bool`: Whether or not to save the final best-fit models and parameters as FITS files
- `save_full_model::Bool`: Whether or not to save the full 3D best-fit model as a FITS file.
- `overwrite::Bool`: Whether or not to overwrite old fits of spaxels when rerunning
- `track_memory::Bool`: Whether or not to save diagnostic files showing memory usage of the program
- `track_convergence::Bool`: Whether or not to save diagnostic files showing convergence of line fitting for each spaxel
- `make_movies::Bool`: Whether or not to save mp4 files of the final model

## Basic fitting options
- `sys_err::Real`: Optional systematic uncertainty quantified as a ratio (0-1) of the flux. I.e. to add a 10% systematic 
    uncertainty use fσ_sys = 0.1
- `extinction_curve::String`: The type of extinction curve being used, i.e. `"kvt"` or `"d+"`
- `extinction_screen::Bool`: Whether or not the extinction is modeled as a screen
- `κ_abs::Vector{Spline1D}`: Mass absorption coefficient for amorphous olivine Mg(2y)Fe(2-2y)SiO4, amorphous pyroxene Mg(x)Fe(1-x)SiO3, and 
    crystalline forsterite Mg2SiO4 as interpolating functions over wavelength, only used if extinction_curve == "decompose"
- `custom_ext_template::Union{Spline1D,Nothing}`: A custom dust extinction template given as a matrix of wavelengths and optical depths that is
    converted into an interpolating function.
- `extinction_map::Union{Array{T,3},Nothing}`: An optional map of estimated extinction values. For MIR spectra, this is interpreted 
as tau_9.7 values, whereas for optical spectra it is interpreted as E(B-V) values. The fits will be locked to the value at the 
corresponding spaxel.
- `fit_stellar_continuum::Bool`: Whether or not to fit MIR stellar continuum
- `fit_sil_emission::Bool`: Whether or not to fit MIR hot silicate dust emission
- `fit_ch_abs::Bool`: Whether or not to fit the MIR CH absorption feature
- `fit_temp_multexp::Bool`: Whether or not to apply and fit multiplicative exponentials to any provided templates
- `guess_tau::Union{Vector{<:Tuple},Nothing}`: Whether or not to guess the optical depth at 9.7 microns by interpolating between 
    PAH-free parts of the continuum at the given wavelength windows in microns (must have enough spectral coverage). 
    The fitted value will then be constrained to be at least 80% of the inferred value.
- `fit_opt_na_feii::Bool`: Whether or not to fit optical narrow Fe II emission
- `fit_opt_br_feii::Bool`: Whether or not to fit optical broad Fe II emission
- `fit_all_global::Bool`: Whether or not to fit all spaxels with a global optimizer (Differential Evolution) instead of a local optimizer (Levenberg-Marquardt).
- `use_pah_templates::Bool`: Whether or not to fit the continuum in two steps, where the first step uses PAH templates,
and the second step fits the PAH residuals with the PAHFIT Drude model.
- `fit_joint::Bool`: If true, fit the continuum and lines simultaneously. If false, the lines will first be masked and the
continuum will be fit, then the continuum is subtracted and the lines are fit to the residuals.
- `fit_uv_bump::Bool`: Whether or not to fit the UV bump in the dust attenuation profile. Only applies if the extinction
curve is "calzetti".
- `fit_covering_frac::Bool`: Whether or not to fit a dust covering fraction in the attenuation profile. Only applies if the
extinction curve is "calzetti".
- `tie_template_amps::Bool`: If true, the template amplitudes in each channel are tied to the same value. Otherwise they may have separate
normalizations in each channel. By default this is false.
- `decompose_lock_column_densities::Bool`: If true and if using the "decompose" extinction profile, then the column densities for
pyroxene and forsterite (N_pyr and N_for) will be locked to their values after the initial fit to the integrated spectrum over all 
spaxels. These parameters are measured relative to olivine (N_oli) so this, in effect, locks the relative abundances of these three silicates
over the full FOV of the cube.

## Continuum parameters
- `continuum::Continuum`: A Continuum structure holding information about various continuum fitting parameters.
This object will obviously be different depending on if the spectral region is :MIR or :OPT.

## MIR continuum parameters
- `n_channels::S`: The number of subchannels covered by the spectrum
- `channel_masks::Vector{BitVector}`: Masks that filter out each individual subchannel in the data
- `n_dust_cont::S`: The number of dust continuum components
- `n_power_law::S`: The number of power law continuum components
- `n_dust_feat::S`: The number of PAH features
- `n_abs_feat::S`: The number of absorption features
- `n_templates::S`: The number of generic templates
- `templates::Array{T,4}`: Each template to be used in the fit. The first 2 axes are spatial, the 3rd axis should be wavelength, 
    and the 4th axis should iterate over each individual template. Each template will get an amplitude parameter in the fit.
- `template_names::Vector{String}`: The names of each generic template in the fit.
- `dust_features::Union{DustFeatures,Nothing}`: All of the fitting parameters for each PAH feature
- `abs_features::Union{DustFeatures,Nothing}`: All of the fitting parameters for each absorption feature
- `abs_taus::Union{Vector{Parameter},Nothing}`: A vector of amplitude parameters for the absorption features

## Optical continuum parameters
- `n_ssps::S`: The number of simple stellar population components
- `ssp_λ::Union{Vector{T},Nothing}`: The wavelength grid for the simple stellar population templates
- `ssp_templates::Union{Vector{Spline2D},Nothing}`: A vector of interpolating functions for the simple stellar populations
at each point in the wavelength grid.
- `feii_templates_fft::Union{Matrix{C},Nothing}`: A matrix of the Fourier transforms of the narrow and broad Fe II templates
- `vres::T`: The constant velocity resolution of the wavelength grid, which assumes logarithmic spacing, in km/s/pixel.
- `vsyst_ssp::T`: The systemic velocity offset between the input wavelength grid and the SSP template wavelength grid.
- `vsyst_feii::T`: The systemic velocity offset between the input wavelength grid and the Fe II template wavelength grid.
- `npad_feii::S`: The length of the Fe II templates (NOT the length of the Fourier transformed templates).

## Line parameters
- `n_lines::S`: The number of lines being fit
- `n_acomps::S`: The summed total number of line profiles fit to all lines in the spectrum, including additional components.
- `n_comps::S`: The maximum number of additional profiles that may be fit to any given line.
- `relative_flags::BitVector`: Flags for whether additional line components should be parametrized relative to the primary component,
    ordered as (amp, voff, fwhm) (global settings for all lines).
- `lines::TransitionLines`: A TransitionLines struct containing parameters and tied information for the emission lines.

## Tied Kinematics
- `tied_kinematics::TiedKinematics`: A TiedKinematics struct containing information about which lines have tied velocities and FWHMs.

## Tied voigt mixing
- `tie_voigt_mixing::Bool`: Whether or not the Voigt mixing parameter is tied between all the lines with Voigt profiles
- `voigt_mix_tied::Parameter`: The actual tied Voigt mixing parameter object, given `tie_voigt_mixing` is true

## Number of parameters
- `n_params_cont::S`: The total number of free fitting parameters for the continuum fit (not including emission lines)
- `n_params_lines::S`: The total number of free fitting parameters for the emission line fit (not including the continuum)
- `n_params_extra::S`: The total number of extra parameters calculated for each fit (includes things like line fluxes and equivalent widths)

## General options
- `cosmology::Cosmology.AbstractCosmology`: The Cosmology, used solely to create physical scale bars on the 2D parameter plots
- `flexible_wavesol::Bool`: Whether or not to allow small variations in the velocity offsets even when tied, to account
    for a bad wavelength solution
- `n_bootstrap::S`: The number of bootstrapping iterations that should be performed for each fit.
- `random_seed::S`: An optional number to use as a seed for the RNG utilized during bootstrapping, to ensure consistency between
- `bootstrap_use::Symbol`: Determines what to output for the resulting parameter values when bootstrapping. May be :med for the 
    median or :best for the original best-fit values.
attempts.
- `line_test_lines::Vector{Vector{Symbol}}`: A list of lines which should be tested for additional components. They may be grouped
together (hence the vector-of-a-vector structure) such that lines in a group will all be given the maximum number of parameters that
any one line passes the test for.
- `line_test_threshold::T`: A threshold which must be met in the chi^2 ratio between a fit without an additional line profile and
one with the additional profile, in order to include the additional profile in the final fit.
- `plot_line_test::Bool`: Whether or not to plot the line test results.

## Line masking options
- `linemask_Δ::S`: The half-width, in pixels, to use to calculate the second derivative for the line masking algorithm.
- `linemask_n_inc_thresh::S`: The number of times that the flux must increase to the left/right before the line masking window stops.
- `linemask_thresh::T`: The number of sigmas that a peak in the second derivative must reach to be considered a line detection in
    the line masking algorithm.
- `linemask_overrides::Vector{Tuple{T,T}}`: Optional list of tuples specifying (min, max) wavelength ranges that will be forcibly 
    added to the line mask. This is different from the `user_mask` option since it only applies to the continuum fitting step but
    will be ignored during the line fitting step. NOTE: Using this option will disable the 'automatic' line masking algorithm and just
    use the manually input regions in the mask.
- `map_snr_thresh::T`: The SNR threshold below which to mask out spaxels from parameter maps for emission lines.

## Best fit parameters
- `p_init_cont::Vector{T}`: The best-fit continuum parameters for the initial fit to the sum of all spaxels.
- `p_init_line::Vector{T}`: Same as `p_init_cont`, but for the line parameters
- `p_init_pahtemp::Vector{T}`: Same as `p_init_cont`, but for the PAH templates amplitudes

See [`ParamMaps`](@ref), [`parammaps_empta`](@ref), [`CubeModel`](@ref), [`cubemodel_empty`](@ref), 
    [`fit_spaxel`](@ref), [`fit_cube!`](@ref)
"""
struct CubeFitter{T<:Real,S<:Integer,C<:Complex} 

    # See explanations for each field in the docstring!
    
    # Data
    cube::DataCube
    z::T
    name::String
    spectral_region::Symbol

    # Basic fitting options
    user_mask::Union{Vector{<:Tuple},Nothing}
    plot_spaxels::Symbol
    plot_maps::Bool
    plot_range::Union{Vector{<:Tuple},Nothing}
    parallel::Bool
    parallel_strategy::String
    save_fits::Bool
    save_full_model::Bool
    overwrite::Bool
    track_memory::Bool
    track_convergence::Bool
    make_movies::Bool

    sys_err::Real
    extinction_curve::String
    extinction_screen::Bool
    κ_abs::Vector{Spline1D}
    custom_ext_template::Union{Spline1D,Nothing}
    extinction_map::Union{Array{T,3},Nothing}
    fit_stellar_continuum::Bool
    fit_sil_emission::Bool
    fit_ch_abs::Bool
    fit_temp_multexp::Bool
    guess_tau::Union{Vector{<:Tuple},Nothing}
    fit_opt_na_feii::Bool
    fit_opt_br_feii::Bool
    fit_all_global::Bool
    use_pah_templates::Bool
    fit_joint::Bool
    fit_uv_bump::Bool
    fit_covering_frac::Bool
    tie_template_amps::Bool
    lock_hot_dust::BitVector
    F_test_ext::BitVector
    decompose_lock_column_densities::Bool

    # Continuum parameters
    continuum::Continuum

    # MIR continuum parameters
    n_channels::S
    channel_masks::Vector{BitVector}
    n_dust_cont::S
    n_power_law::S  # (also used for optical power laws)
    n_dust_feat::S
    n_abs_feat::S
    n_templates::S
    # 4D templates: first 3 axes are the template's spatial and spectral axes, while the 4th axis enumerates individual templates
    templates::Array{T, 4}  
    template_names::Vector{String}
    dust_features::Union{DustFeatures,Nothing}
    abs_features::Union{DustFeatures,Nothing}
    abs_taus::Union{Vector{Parameter},Nothing}

    # Optical continuum parameters
    n_ssps::S
    ssp_λ::Union{Vector{T},Nothing}
    ssp_templates::Union{Vector{Spline2D},Nothing}
    feii_templates_fft::Union{Matrix{C},Nothing}
    vres::T
    vsyst_ssp::T
    vsyst_feii::T
    npad_feii::S

    # Line parameters
    n_lines::S
    n_acomps::S
    n_comps::S
    n_fit_comps::Dict{Symbol,Matrix{S}}
    relative_flags::BitVector
    lines::TransitionLines

    # Tied voffs
    tied_kinematics::TiedKinematics

    # Tied voigt mixing
    tie_voigt_mixing::Bool
    voigt_mix_tied::Parameter

    # Number of parameters
    n_params_cont::S
    n_params_lines::S
    n_params_extra::S
    
    # General options
    cosmology::Cosmology.AbstractCosmology
    flexible_wavesol::Bool
    n_bootstrap::S
    random_seed::S
    bootstrap_use::Symbol
    line_test_lines::Vector{Vector{Symbol}}
    line_test_threshold::T
    plot_line_test::Bool
    lines_allow_negative::Bool
    subtract_cubic_spline::Bool

    # Line masking and sorting options
    linemask_Δ::S
    linemask_n_inc_thresh::S
    linemask_thresh::T
    linemask_overrides::Vector{Tuple{T,T}}
    linemask_width::T
    map_snr_thresh::T
    sort_line_components::Union{Symbol,Nothing}

    # Initial parameter vectors
    p_init_cont::Vector{T}
    p_init_line::Vector{T}
    p_init_pahtemp::Vector{T}

    # Initial parameter cubes
    p_init_cube_λ::Union{Vector{T},Nothing}
    p_init_cube_cont::Union{Array{T,3},Nothing}
    p_init_cube_lines::Union{Array{T,3},Nothing}
    p_init_cube_wcs::Union{WCSTransform,Nothing}
    p_init_cube_coords::Union{Vector{Vector{T}},Nothing}
    p_init_cube_Ω::Union{T,Nothing}

    # A flag that is set after fitting the nuclear spectrum
    nuc_fit_flag::BitVector
    # A set of template amplitudes that is relevant only if fitting a model to the nuclear template. In this case,
    # the model fits amplitudes to the PSF model which are helpful to store.
    nuc_temp_amps::Vector{T}

    #= Constructor function --> the default inputs are all taken from the configuration files, but may be overwritten
    by the kwargs object using the same syntax as any keyword argument. The rest of the fields are generated in the function 
    from these inputs =#
    function CubeFitter(cube::DataCube, z::Real, name::String; kwargs...) 
        
        # Some setup variables
        λ = cube.λ
        spectral_region = cube.spectral_region
        name = replace(name, #= no spaces! =# " " => "_")
        name = join([name, lowercase(string(spectral_region))], "_")

        # Prepare options from the configuration file
        options = parse_options()
        # Overwrite with any options provided in the keyword arguments
        out = copy(options)
        for key in keys(kwargs)
            out[key] = kwargs[key]
        end
        # Set up and reformat some default options
        cubefitter_add_default_options!(cube, spectral_region, out)
        # Set up the extinction map and PAH template boolean map 
        extinction_map = cubefitter_prepare_extinction_map(out, cube)
        # Set up the output directories
        cubefitter_prepare_output_directories(name, spectral_region, out)

        #############################################################

        @debug """\n
        Creating CubeFitter struct for $name at z=$z
        ############################################
        """

        # Set default values for everything that is specific to the wavelength region
        n_dust_cont = n_dust_features = n_abs_features = n_channels = n_ssps = 0
        npad_feii = vres = vsyst_ssp = vsyst_feii = 0.
        dust_features = dust_features_0 = abs_features = abs_features_0 = abs_taus = ssp_λ = 
            ssp_templates = feii_templates_fft = nothing
        lock_hot_dust = F_test_ext = false
        channel_masks = []

        if spectral_region == :MIR

            # Calculate the number of subchannels
            n_channels, channel_masks = cubefitter_mir_get_n_channels(λ, z)
            # Collect data structures for the continuum, dust features, & absorption features
            continuum, dust_features_0, dust_features, abs_features_0, abs_features, abs_taus, 
                n_dust_cont, n_power_law, n_dust_features, n_abs_features, n_templates, lock_hot_dust,
                F_test_ext = cubefitter_mir_prepare_continuum(λ, z, out, n_channels)

            n_params_cont = cubefitter_mir_count_cont_parameters(out[:extinction_curve], out[:fit_sil_emission], out[:fit_temp_multexp], 
                n_dust_cont, n_power_law, n_abs_features, n_templates, n_channels, dust_features)

        elseif spectral_region == :OPT

            continuum, ssp_λ, ssp_templates, feii_templates_fft, npad_feii, vres, 
                vsyst_ssp, vsyst_feii, n_ssps, n_power_law, n_templates = cubefitter_opt_prepare_continuum(
                    λ, z, out, name, cube)

            n_params_cont = cubefitter_opt_count_cont_parameters(out[:fit_opt_na_feii], out[:fit_opt_br_feii], out[:fit_uv_bump],
                out[:fit_covering_frac], n_ssps, n_power_law, n_templates)

        end

        lines_0, lines, tied_kinematics, flexible_wavesol, tie_voigt_mixing, voigt_mix_tied, 
            n_lines, n_comps, n_acomps, n_fit_comps, relative_flags = cubefitter_prepare_lines(λ, cube, out)

        # the automatic line masking routine is meh;
        # so for now it's essentially softly disabled by automatically including
        # overrides for all lines in the line list with a width of +/-1000 km/s;
        # this can still be overridden with a manual line mask input
        if !haskey(out, :linemask_overrides)
            overrides = Tuple[]
            for λi in lines.λ₀
                push!(overrides, λi .* (1-out[:linemask_width]/C_KMS, 1+out[:linemask_width]/C_KMS))
            end
            out[:linemask_overrides] = overrides
        end
    
        n_params_lines = cubefitter_count_line_parameters(lines, flexible_wavesol, n_lines, n_comps)
        n_params_extra = cubefitter_count_extra_parameters(n_dust_features, n_lines, n_acomps)

        @debug "### There is a total of $(n_params_cont) continuum parameters ###"
        @debug "### There is a total of $(n_params_lines) emission line parameters ###"
        @debug "### There is a total of $(n_params_extra) extra parameters ###"

        # Pre-calculate mass absorption coefficients for olivine, pyroxene, and forsterite
        κ_oli, κ_pyr, κ_for = read_dust_κ(out[:pyroxene_x], out[:olivine_y], out[:grain_size])

        # Prepare initial best fit parameter options
        @debug "Preparing initial best fit parameter vectors with $(n_params_cont) and $(n_params_lines) parameters"
        p_init_cont = zeros(n_params_cont)
        p_init_line = zeros(n_params_lines)
        p_init_pahtemp = zeros(2)

        # If a fit has been run previously, read in the file containing the rolling best fit parameters
        # to pick up where the fitter left off seamlessly
        if isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv")) && isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"))
            p_init_cont = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv"), ',', Float64, '\n')[:, 1]
            p_init_line = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"), ',', Float64, '\n')[:, 1]
            p_init_pahtemp = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_pahtemp.csv"), ',', Float64, '\n')[:, 1]
        end

        p_init_cube_λ = p_init_cube_cont = p_init_cube_lines = p_init_cube_wcs = p_init_cube_coords = p_init_cube_Ω = nothing
        if haskey(out, :p_init_cube) && spectral_region == :MIR
            p_init_cube_λ, p_init_cube_cont, p_init_cube_lines, p_init_cube_wcs, p_init_cube_coords, p_init_cube_Ω  =
                cubefitter_prepare_p_init_cube_parameters(λ, z, out, cube, spectral_region, dust_features_0, dust_features, 
                abs_features_0, lines_0, flexible_wavesol, n_params_cont, n_params_lines, n_dust_cont, n_power_law, 
                n_abs_features, n_templates, n_channels, n_comps)
        end

        # Nuclear template fitting attributes
        nuc_fit_flag = BitVector([0])
        nuc_temp_amps = ones(Float64, n_channels)
        lock_hot_dust = BitVector([lock_hot_dust])
        F_test_ext = BitVector([F_test_ext])

        ctype = isnothing(feii_templates_fft) ? ComplexF64 : eltype(feii_templates_fft)
        new{typeof(z), typeof(n_lines), ctype}(
            cube, 
            z, 
            name, 
            spectral_region, 
            out[:user_mask], 
            out[:plot_spaxels], 
            out[:plot_maps], 
            out[:plot_range], 
            out[:parallel], 
            out[:parallel_strategy],
            out[:save_fits], 
            out[:save_full_model], 
            out[:overwrite], 
            out[:track_memory], 
            out[:track_convergence], 
            out[:make_movies], 
            out[:sys_err],
            out[:extinction_curve], 
            out[:extinction_screen], 
            [κ_oli, κ_pyr, κ_for], 
            out[:custom_ext_template], 
            extinction_map, 
            out[:fit_stellar_continuum], 
            out[:fit_sil_emission], 
            out[:fit_ch_abs],
            out[:fit_temp_multexp], 
            out[:guess_tau], 
            out[:fit_opt_na_feii], 
            out[:fit_opt_br_feii],
            out[:fit_all_global], 
            out[:use_pah_templates], 
            out[:fit_joint], 
            out[:fit_uv_bump], 
            out[:fit_covering_frac], 
            out[:tie_template_amps],
            lock_hot_dust,
            F_test_ext,
            out[:decompose_lock_column_densities],
            continuum, 
            n_channels,
            channel_masks,
            n_dust_cont, 
            n_power_law, 
            n_dust_features, 
            n_abs_features, 
            n_templates, 
            out[:templates], 
            out[:template_names], 
            dust_features, 
            abs_features, 
            abs_taus, 
            n_ssps, 
            ssp_λ, 
            ssp_templates, 
            feii_templates_fft, 
            vres, 
            vsyst_ssp, 
            vsyst_feii, 
            npad_feii, 
            n_lines, 
            n_acomps, 
            n_comps, 
            n_fit_comps, 
            relative_flags, 
            lines, 
            tied_kinematics, 
            tie_voigt_mixing, 
            voigt_mix_tied, 
            n_params_cont, 
            n_params_lines, 
            n_params_extra, 
            out[:cosmology], 
            flexible_wavesol, 
            out[:n_bootstrap], 
            out[:random_seed], 
            out[:bootstrap_use],
            out[:line_test_lines], 
            out[:line_test_threshold], 
            out[:plot_line_test], 
            out[:lines_allow_negative],
            out[:subtract_cubic_spline], 
            out[:linemask_delta], 
            out[:linemask_n_inc_thresh],
            out[:linemask_thresh], 
            out[:linemask_overrides], 
            out[:linemask_width],
            out[:map_snr_thresh], 
            out[:sort_line_components], 
            p_init_cont, 
            p_init_line, 
            p_init_pahtemp, 
            p_init_cube_λ, 
            p_init_cube_cont, 
            p_init_cube_lines, 
            p_init_cube_wcs, 
            p_init_cube_coords, 
            p_init_cube_Ω,
            nuc_fit_flag,
            nuc_temp_amps
        )
    end
end


# Helper function for making output directoreis when setting up the CubeFitter object
function cubefitter_prepare_output_directories(name::String, spectral_region::Symbol, out::Dict)

    # Prepare output directories
    @info "Preparing output directories"

    # Top-level output directory
    if !isdir("output_$name")
        mkdir("output_$name")
    end
    # Sub-folder for 1D plots of spaxel fits
    if !isdir(joinpath("output_$name", "spaxel_plots"))
        mkdir(joinpath("output_$name", "spaxel_plots"))
    end
    if !isdir(joinpath("output_$name", "zoomed_plots")) && !isnothing(out[:plot_range])
        mkdir(joinpath("output_$name", "zoomed_plots"))
    end
    if !isdir(joinpath("output_$name", "line_tests")) && (length(out[:line_test_lines]) > 0) && out[:plot_line_test]
        mkdir(joinpath("output_$name", "line_tests"))
    end
    # Sub-folder for data files saving the results of individual spaxel fits
    if !isdir(joinpath("output_$name", "spaxel_binaries"))
        mkdir(joinpath("output_$name", "spaxel_binaries"))
    end
    # Sub-folder for 2D parameter maps 
    if !isdir(joinpath("output_$name", "param_maps"))
        mkdir(joinpath("output_$name", "param_maps"))
    end
    # Sub-folder for log files
    if !isdir(joinpath("output_$name", "logs"))
        mkdir(joinpath("output_$name", "logs"))
    end

end


# Helper function for setting up default options when creating a CubeFitter object
function cubefitter_add_default_options!(cube::DataCube, spectral_region::Symbol, out::Dict)

    out[:line_test_lines] = [[Symbol(ln) for ln in group] for group in out[:line_test_lines]]
    out[:plot_spaxels] = Symbol(out[:plot_spaxels])
    out[:bootstrap_use] = Symbol(out[:bootstrap_use])

    if !haskey(out, :plot_range)
        out[:plot_range] = nothing
    else
        out[:plot_range] = [tuple(out[:plot_range][i]...) for i in 1:length(out[:plot_range])]
        for  pair in out[:plot_range]
            @assert pair[1] < pair[2] "plot_range pairs must be in ascending order!"
        end
    end

    if !haskey(out, :user_mask)
        out[:user_mask] = nothing
    else
        out[:user_mask] = [tuple(out[:user_mask][i]...) for i in 1:length(out[:user_mask])]
        for  pair in out[:user_mask]
            @assert pair[1] < pair[2] "user_mask pairs must be in ascending order!"
        end
    end

    if !haskey(out, :templates)
        out[:templates] = Array{Float64, 4}(undef, size(cube.I)..., 0)
    elseif ndims(out[:templates]) == 3
        t4 = Array{Float64, 4}(undef, size(cube.I)..., 1)
        t4[:, :, :, 1] .= out[:templates]
        out[:templates] = t4
    end

    if !haskey(out, :template_names)
        out[:template_names] = String["template_$i" for i in axes(out[:templates], 4)]
    end

    if !haskey(out, :linemask_delta)
        out[:linemask_delta] = 20
    end
    if !haskey(out, :linemask_n_inc_thresh)
        out[:linemask_n_inc_thresh] = 7
    end
    if !haskey(out, :linemask_thresh)
        out[:linemask_thresh] = 3.
    end
    if !haskey(out, :map_snr_thresh)
        out[:map_snr_thresh] = 3.
    end
    if !haskey(out, :guess_tau)
        out[:guess_tau] = nothing
    end
    if !haskey(out, :custom_ext_template)
        out[:custom_ext_template] = nothing
    else
        out[:custom_ext_template] = Spline1D(out[:custom_ext_template][:,1], out[:custom_ext_template][:,2], k=1, bc="extrapolate")
    end

end


# Helper function to prepare extinction map and PAH template boolean map
function cubefitter_prepare_extinction_map(out::Dict, cube::DataCube)

    # Get potential extinction map
    extinction_map = nothing
    if haskey(out, :extinction_map) && !isnothing(out[:extinction_map])
        extinction_map = out[:extinction_map]
        @assert size(extinction_map)[1:2] == size(cube.I)[1:2] "The extinction map must match the shape of the first two dimensions of the intensity map!"
        if ndims(extinction_map) == 2
            extinction_map = reshape(extinction_map, (size(extinction_map)..., 1))
        end
    end
    
    extinction_map 
end


# Helper function for preparing emission line parameters for a CubeFitter object
function cubefitter_prepare_lines(λ::Vector{<:Real}, cube::DataCube, out::Dict)

    lines_0, tied_kinematics, flexible_wavesol, tie_voigt_mixing, voigt_mix_tied = parse_lines()

    # Only use lines within the wavelength range being fit
    ln_filt = minimum(λ) .< lines_0.λ₀ .< maximum(λ)
    if !isnothing(out[:user_mask])
        for pair in out[:user_mask]
            ln_filt .&= .~(pair[1] .< lines_0.λ₀ .< pair[2])
        end
    end
    # Convert to a vectorized "TransitionLines" object
    lines = TransitionLines(lines_0.names[ln_filt], lines_0.latex[ln_filt], lines_0.annotate[ln_filt], lines_0.λ₀[ln_filt], lines_0.sort_order[ln_filt],
                            lines_0.profiles[ln_filt, :], lines_0.tied_amp[ln_filt, :], lines_0.tied_voff[ln_filt, :], lines_0.tied_fwhm[ln_filt, :], 
                            lines_0.acomp_amp[ln_filt, :], lines_0.voff[ln_filt, :], lines_0.fwhm[ln_filt, :], lines_0.h3[ln_filt, :], 
                            lines_0.h4[ln_filt, :], lines_0.η[ln_filt, :], lines_0.combined, lines_0.rel_amp, lines_0.rel_voff, lines_0.rel_fwhm)
    n_lines = length(lines.names)
    n_comps = size(lines.profiles, 2)
    n_acomps = sum(.!isnothing.(lines.profiles[:, 2:end]))
    msg = "### Model will include $n_lines emission lines ###"
    for (k, (name, λ0, prof)) ∈ enumerate(zip(lines.names, lines.λ₀, lines.profiles))
        msg *= "\n### $name at lambda = $λ0 um with $prof profile ###"
        for acomp_prof ∈ lines.profiles[k, 2:end]
            if !isnothing(acomp_prof)
                msg *= "\n###   and $acomp_prof acomp profile   ###"
            end
        end
    end
    @debug msg
    n_fit_comps = Dict{Symbol, Matrix{typeof(n_comps)}}()
    for name ∈ lines.names
        n_fit_comps[name] = ones(typeof(n_comps), size(cube.I)[1:2])
    end

    relative_flags = BitVector([lines.rel_amp, lines.rel_voff, lines.rel_fwhm])
    if !haskey(out, :sort_line_components)
        out[:sort_line_components] = nothing
        if all(.~relative_flags)
            out[:sort_line_components] = :flux
        end
    elseif !isnothing(out[:sort_line_components])
        out[:sort_line_components] = Symbol(out[:sort_line_components])
    end

    # Remove unnecessary rows/keys from the tied_kinematics object after the lines have been filtered
    @debug "TiedKinematics before filtering: $tied_kinematics"
    for j ∈ 1:n_comps
        keep0 = Int64[]
        for (k, key) ∈ enumerate(tied_kinematics.key_amp[j])
            if any(lines.tied_amp[:, j] .== key)
                # Remove the unneeded elements
                append!(keep0, [k])
            end
        end
        tied_kinematics.key_amp[j] = tied_kinematics.key_amp[j][keep0]
        tied_kinematics.amp[j] = tied_kinematics.amp[j][keep0]
        # Remove line ratios for lines that fall outside the fitting region
        for i in eachindex(tied_kinematics.amp[j])
            # Make a copy so we're not modifying the dictionary as we loop through it
            acp = copy(tied_kinematics.amp[j][i])
            for lk in keys(acp)
                if !(lk in lines.names)
                    delete!(tied_kinematics.amp[j][i], lk)
                end
            end
            # Renormalize the ratios so the largest is 1.0
            max_ratio = maximum(values(tied_kinematics.amp[j][i]))
            for lk in keys(tied_kinematics.amp[j][i])
                tied_kinematics.amp[j][i][lk] /= max_ratio
            end
        end

        keep1 = Int64[]
        for (k, key) ∈ enumerate(tied_kinematics.key_voff[j])
            if any(lines.tied_voff[:, j] .== key)
                # Remove the unneeded elements
                append!(keep1, [k])
            end
        end
        tied_kinematics.key_voff[j] = tied_kinematics.key_voff[j][keep1]
        tied_kinematics.voff[j] = tied_kinematics.voff[j][keep1]

        keep2 = Int64[]
        for (k, key) ∈ enumerate(tied_kinematics.key_fwhm[j])
            if any(lines.tied_fwhm[:, j] .== key)
                # Remove the unneeded elements
                append!(keep2, [k])
            end
        end
        tied_kinematics.key_fwhm[j] = tied_kinematics.key_fwhm[j][keep2]
        tied_kinematics.fwhm[j] = tied_kinematics.fwhm[j][keep2]
    end
    @debug "TiedKinematics after filtering: $tied_kinematics"

    # Also store the "tied" parameter for each line, which will need to be checked against the kin_tied_key
    # during fitting to find the proper location of the tied voff parameter to use
    msg = "### Model will include these tied kinematics parameters for each component ###"
    for lt ∈ tied_kinematics.key_voff
        msg *= "\n### voff for group $lt ###"
    end
    for lt ∈ tied_kinematics.key_fwhm
        msg *= "\n### fwhm for group $lt ###"
    end
    @debug msg

    # One η for all voigt profiles
    if any(lines.profiles .== :Voigt) && tie_voigt_mixing
        @debug "### Model will include 1 tied voigt mixing parameter ###"
    end

    lines_0, lines, tied_kinematics, flexible_wavesol, tie_voigt_mixing, voigt_mix_tied, 
        n_lines, n_comps, n_acomps, n_fit_comps, relative_flags
end


# Helper function for counting the total number of emission line parameters
function cubefitter_count_line_parameters(lines::TransitionLines, flexible_wavesol::Bool, n_lines::Integer, n_comps::Integer)
    n_params_lines = 0
    for i ∈ 1:n_lines
        for j ∈ 1:n_comps
            if !isnothing(lines.profiles[i, j])
                # amplitude, voff, and FWHM parameters
                n_params_lines += 3
                if !isnothing(lines.tied_voff[i, j]) && flexible_wavesol && isone(j)
                    # individual voff parameter
                    n_params_lines += 1
                end
                if lines.profiles[i, j] == :GaussHermite
                    # extra h3 and h4 parmeters
                    n_params_lines += 2
                elseif lines.profiles[i, j] == :Voigt
                    # extra mixing parameter
                    n_params_lines += 1
                end
            end
        end
    end

    n_params_lines
end


# Helper function for counting the total number of "extra" (non-fit) parameters
function cubefitter_count_extra_parameters(n_dust_features::Integer, n_lines::Integer, n_acomps::Integer)
    3 * (n_dust_features + n_lines + n_acomps) + 5n_lines
end


# Helper function for creating initial parameter cubes based on a previous fit
function cubefitter_prepare_p_init_cube_parameters(λ::Vector{<:Real}, z::Real, out::Dict, cube::DataCube, 
    spectral_region::Symbol, dust_features_0::DustFeatures, dust_features::DustFeatures, abs_features_0::DustFeatures,
    lines_0::TransitionLines, flexible_wavesol::Bool, n_params_cont::Integer, n_params_lines::Integer, n_dust_cont::Integer, 
    n_power_law::Integer, n_abs_features::Integer, n_templates::Integer, n_channels::Integer, 
    n_comps::Integer)

    # Use fitting parameters from another run (on a potentially different WCS) as initial conditions
    path = out[:p_init_cube]
    oname = replace(basename(path), "output_" => "")
    hdu = FITS(joinpath(path, "$(oname)_full_model.fits"))
    # Wavelength vector
    p_init_cube_λ = read(hdu["WAVELENGTH"], "wave") ./ (1 + z)
    # WCS
    p_init_cube_wcs = WCS.from_header(read_header(hdu[1], String))[1]
    # Angular resolution
    p_init_cube_Ω = read_header(hdu[1])["PIXAR_SR"]
    # Parameters from the fits
    p_init_cube_cont = ones(size(cube.I)[1:2]..., n_params_cont) .* NaN
    p_init_cube_lines = ones(size(cube.I)[1:2]..., n_params_lines) .* NaN

    # Filter out dust features
    df_filt = [((minimum(p_init_cube_λ)-0.1) < dust_features_0.mean[i].value < (maximum(p_init_cube_λ)+0.1)) for i ∈ 1:length(dust_features_0.mean)]
    if !isnothing(out[:user_mask])
        for pair in out[:user_mask]
            df_filt .&= [~(pair[1] < dust_features_0.mean[i].value < pair[2]) for i ∈ 1:length(dust_features_0.mean)]
        end
    end
    initcube_dust_features = DustFeatures(dust_features_0.names[df_filt], 
                                dust_features_0.profiles[df_filt],
                                dust_features_0.mean[df_filt],
                                dust_features_0.fwhm[df_filt],
                                dust_features_0.asym[df_filt],
                                dust_features_0.index[df_filt],
                                dust_features_0.cutoff[df_filt],
                                dust_features_0.complexes[df_filt],
                                dust_features_0._local[df_filt])
    # Count how many dust features are in the cube template but not in the current fitting region
    n_dfparams_left = n_dfparams_right = 0
    for i in 1:length(initcube_dust_features.names)
        if initcube_dust_features.mean[i].value < (minimum(λ)-0.1)
            if initcube_dust_features.profiles[i] == :Drude
                n_dfparams_left += 4
            elseif initcube_dust_features.profiles[i] == :PearsonIV
                n_dfparams_left += 5
            end
        end
        if initcube_dust_features.mean[i].value > (maximum(λ)+0.1)
            if initcube_dust_features.profiles[i] == :Drude
                n_dfparams_right += 4
            elseif initcube_dust_features.profiles[i] == :PearsonIV
                n_dfparams_right += 5
            end
        end
    end

    # Repeat for absorption features
    ab_filt = [((minimum(p_init_cube_λ)-0.1) < abs_features_0.mean[i].value < (maximum(p_init_cube_λ)+0.1)) for i ∈ 1:length(abs_features_0.mean)]
    if !isnothing(out[:user_mask])
        for pair in out[:user_mask]
            ab_filt .&= [~(pair[1] < abs_features_0.mean[i].value < pair[2]) for i ∈ 1:length(abs_features_0.mean)]
        end
    end
    initcube_abs_features = DustFeatures(abs_features_0.names[ab_filt],
                                abs_features_0.profiles[ab_filt],
                                abs_features_0.mean[ab_filt],
                                abs_features_0.fwhm[ab_filt],
                                abs_features_0.asym[ab_filt],
                                abs_features_0.index[ab_filt],
                                abs_features_0.cutoff[ab_filt],
                                abs_features_0.complexes[ab_filt],
                                abs_features_0._local[ab_filt])
    n_abparams_left = n_abparams_right = 0
    for i in 1:length(initcube_abs_features.names)
        if initcube_abs_features.mean[i].value < (minimum(λ)-0.1)
            n_abparams_left += 4
        end
        if initcube_abs_features.mean[i].value > (maximum(λ)+0.1)
            n_abparams_right += 4
        end
    end

    # Repeat for emission lines
    ln_filt = minimum(p_init_cube_λ) .< lines_0.λ₀ .< maximum(p_init_cube_λ)
    if !isnothing(out[:user_mask])
        for pair in out[:user_mask]
            ln_filt .&= .~(pair[1] .< lines_0.λ₀ .< pair[2])
        end
    end
    # Convert to a vectorized "TransitionLines" object
    initcube_lines = TransitionLines(lines_0.names[ln_filt], lines_0.latex[ln_filt], lines_0.annotate[ln_filt], lines_0.λ₀[ln_filt], lines_0.sort_order[ln_filt],
                            lines_0.profiles[ln_filt, :], lines_0.tied_amp[ln_filt, :], lines_0.tied_voff[ln_filt, :], lines_0.tied_fwhm[ln_filt, :], 
                            lines_0.acomp_amp[ln_filt, :], lines_0.voff[ln_filt, :], lines_0.fwhm[ln_filt, :], lines_0.h3[ln_filt, :], 
                            lines_0.h4[ln_filt, :], lines_0.η[ln_filt, :], lines_0.combined, lines_0.rel_amp, lines_0.rel_voff, lines_0.rel_fwhm)
    n_lineparams_left = n_lineparams_right = 0
    n_initcube_lineparams = 0
    for i in 1:length(initcube_lines.names)
        for j in 1:n_comps
            if !isnothing(initcube_lines.profiles[i, j])
                n_initcube_lineparams += 3
                if !isnothing(initcube_lines.tied_voff[i, j]) && flexible_wavesol && isone(j)
                    n_initcube_lineparams += 1
                end
                if initcube_lines.profiles[i, j] == :GaussHermite
                    n_initcube_lineparams += 2
                elseif initcube_lines.profiles[i, j] == :Voigt
                    n_initcube_lineparams += 1
                end
                if initcube_lines.λ₀[i] < minimum(λ)
                    n_lineparams_left += 3
                    if !isnothing(initcube_lines.tied_voff[i, j]) && flexible_wavesol && isone(j)
                        # individual voff parameter
                        n_lineparams_left += 1
                    end
                    if initcube_lines.profiles[i, j] == :GaussHermite
                        # extra h3 and h4 parmeters
                        n_lineparams_left += 2
                    elseif initcube_lines.profiles[i, j] == :Voigt
                        # extra mixing parameter
                        n_lineparams_left += 1
                    end 
                end
                if initcube_lines.λ₀[i] > maximum(λ)
                    n_lineparams_right += 3
                    if !isnothing(initcube_lines.tied_voff[i, j]) && flexible_wavesol && isone(j)
                        # individual voff parameter
                        n_lineparams_right += 1
                    end
                    if initcube_lines.profiles[i, j] == :GaussHermite
                        # extra h3 and h4 parmeters
                        n_lineparams_right += 2
                    elseif initcube_lines.profiles[i, j] == :Voigt
                        # extra mixing parameter
                        n_lineparams_right += 1
                    end 
                end
            end
        end
    end

    # Now loop through the spaxels and assign the initial fitting parameters based on the saved results
    spaxfiles = [f for f in readdir(joinpath(path, "spaxel_binaries")) if contains(f, "spaxel")]
    for sf in spaxfiles
        params = readdlm(joinpath(path, "spaxel_binaries", sf), ',', Float64, '\n')[:,1]
        c1 = (2+4) + (out[:extinction_curve] == "decompose" ? 3 : 1) + 2n_dust_cont + 2n_power_law
        c2 = c1 + n_abparams_left + 4n_abs_features - n_abparams_right
        c3 = c2 + n_abparams_right + (out[:fit_sil_emission] ? 6 : 0) + (out[:fit_temp_multexp] ? 8 : n_templates*n_channels)
        c4 = c3 + n_dfparams_left + 4sum(dust_features.profiles .== :Drude) + 5sum(dust_features.profiles .== :PearsonIV)
        c5 = c4 + n_dfparams_right + n_lineparams_left + n_params_lines
        params_cont1 = params[1:c1]
        params_ab = params[(1+c1+n_abparams_left):c2]
        params_cont2 = params[(1+c2+n_abparams_right):c3]
        params_df = params[(1+c3+n_dfparams_left):c4]
        params_lines = params[(1+c4+n_dfparams_right+n_lineparams_left):c5]
        params_cont = [params_cont1; params_ab; params_cont2; params_df]

        @assert length(params_cont) == size(p_init_cube_cont, 3) "Sizes do not match between init cube cont params and current cube params!"
        @assert length(params_lines) == size(p_init_cube_lines, 3) "Sizes do not match between init cube line params and current cube params!"

        spax = split(replace(sf, ".csv" => ""), "_")[end-1:end]
        spax_x = parse(Int, spax[1])
        spax_y = parse(Int, spax[2])
        p_init_cube_cont[spax_x, spax_y, :] .= params_cont
        p_init_cube_lines[spax_x, spax_y, :] .= params_lines
    end

    # Calculate the cube spaxel coordinates in the current WCS frame
    # Get the coordinates of all spaxels that have fit results
    coords0 = [float.(c.I) for c in CartesianIndices(size(p_init_cube_cont)[1:2]) if !all(isnan.(p_init_cube_cont[c,:]))]
    # Transform to coordinates in the WCS of our current frame
    p_init_cube_coords = [world_to_pix(cube.wcs, pix_to_world(p_init_cube_wcs, [coord..., 1.]))[1:2] for coord in coords0]

    p_init_cube_λ, p_init_cube_cont, p_init_cube_lines, p_init_cube_wcs, p_init_cube_coords, p_init_cube_Ω 
end


"""
    generate_cubemodel(cube_fitter[, aperture])

Generate a CubeModel object corresponding to the options given by the CubeFitter object
"""
function generate_cubemodel(cube_fitter::CubeFitter, aperture::Bool=false)
    shape = aperture ? (1,1,size(cube_fitter.cube.I, 3)) : size(cube_fitter.cube.I)
    # Full 3D intensity model array
    @debug "Generating full 3D cube models"
    if cube_fitter.spectral_region == :MIR
        cube_model = cubemodel_empty(shape, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.names,
            cube_fitter.abs_features.names, cube_fitter.template_names, cube_fitter.lines.names, cube_fitter.extinction_curve)
    elseif cube_fitter.spectral_region == :OPT
        cube_model = cubemodel_empty(shape, cube_fitter.n_ssps, cube_fitter.n_power_law, cube_fitter.lines.names, 
            cube_fitter.template_names)
    end
    cube_model
end


"""
    generate_parammaps(cube_fitter[, aperture])

Generate three ParamMaps objects (for the values and upper/lower errors) corrresponding to the options given
by the CubeFitter object.
"""
function generate_parammaps(cube_fitter::CubeFitter, aperture::Bool=false)
    shape = aperture ? (1,1,size(cube_fitter.cube.I, 3)) : size(cube_fitter.cube.I)
    # 2D maps of fitting parameters
    @debug "Generating 2D parameter value & error maps"
    if cube_fitter.spectral_region == :MIR
        param_maps = parammaps_mir_empty(cube_fitter, shape)
    elseif cube_fitter.spectral_region == :OPT
        param_maps = parammaps_opt_empty(cube_fitter, shape)
    end
    param_maps
end


"""
    split_parameters(pars, dstep, plims, plock)

Sorts parameters into vectors containing only the free parameters and the locked parameters.
Also sorts the step, limits, and lock vectors accordingly.
"""
function split_parameters(pars::Vector{<:Real}, dstep::Vector{<:Real}, plims::Vector{Tuple}, plock::BitVector)

    # Sort vectors
    pfix = pars[plock]
    pfree = pars[.~plock]
    dfree = dstep[.~plock]

    # Count free parameters
    n_free = sum(.~plock)

    # Lower/upper limits
    lbfree = [pl[1] for pl in plims[.~plock]]
    ubfree = [pl[2] for pl in plims[.~plock]]

    pfix, pfree, dfree, lbfree, ubfree, n_free
end


"""
    split_parameters(pars, dstep, plims, plock, tied_indices)

Sorts parameters into vectors containing only the tied, free parameters and separates the
locked parameters into a separate vector. Also sorts the step, limits, and lock vectors 
accordingly.
"""
function split_parameters(pars::Vector{<:Real}, dstep::Vector{<:Real}, plims::Vector{Tuple}, plock::BitVector,
    tied_indices::Vector{<:Integer}; param_names::Union{Nothing,Vector{String}}=nothing)

    do_names = !isnothing(param_names)

    # Copy the input vectors
    pars_tied = copy(pars)
    dstep_tied = copy(dstep)
    plims_tied = copy(plims)
    plock_tied = copy(plock)

    # Delete the entries at points where they should be tied to another entry
    deleteat!(pars_tied, tied_indices)
    deleteat!(dstep_tied, tied_indices)
    deleteat!(plims_tied, tied_indices)
    deleteat!(plock_tied, tied_indices)

    # Sort the parameters into those that are locked and those that are unlocked
    pfix_tied = pars_tied[plock_tied]
    pfree_tied = pars_tied[.~plock_tied]
    dfree_tied = dstep_tied[.~plock_tied]

    # Count free parameters
    n_free = sum(.~plock_tied)
    n_tied = length(pars_tied)

    # Lower/upper bounds
    lb_tied = [pl[1] for pl in plims_tied]
    ub_tied = [pl[2] for pl in plims_tied]
    lbfree_tied = lb_tied[.~plock_tied]
    ubfree_tied = ub_tied[.~plock_tied]

    @debug "Parameters: \n $pars_tied"
    @debug "Parameters locked? \n $plock_tied"
    @debug "Lower limits: \n $lb_tied"
    @debug "Upper limits: \n $ub_tied"

    if do_names
        pnames_tied = copy(param_names)
        deleteat!(pnames_tied, tied_indices)
        @debug "Parameter Names: \n $pnames_tied"
        return pfree_tied, pfix_tied, dfree_tied, plock_tied, lbfree_tied, ubfree_tied, pnames_tied, n_free, n_tied
    end

    return pfree_tied, pfix_tied, dfree_tied, plock_tied, lbfree_tied, ubfree_tied, n_free, n_tied
end


"""
    rebuild_full_parameters(pfree, pfix, plock, n_tot)

The opposite of split_parameters. Takes a split up parameter vector and rebuilds the full vector 
including all of the free+locked parameters.
"""
function rebuild_full_parameters(pfree::Vector{<:Real}, pfix::Vector{<:Real}, plock::BitVector)
    n_tot = length(plock)
    pfull = zeros(eltype(pfree), n_tot)
    pfull[.~plock] .= pfree
    pfull[plock] .= pfix
    return pfull
end


"""
    rebuild_full_parameters(pfree_tied, pfix_tied, plock_tied, tied_pairs, tied_indices, n_tied)

The opposite of split_parameters.  Takes a split up parameter vector and rebuilds the full vector including
all of the free+locked parameters and the tied parameters.
"""
function rebuild_full_parameters(pfree_tied::Vector{<:Real}, pfix_tied::Vector{<:Real}, plock_tied::BitVector, 
    tied_pairs::Vector{Tuple}, tied_indices::Vector{<:Integer}, n_tied::Integer)
    pfull = zeros(eltype(pfree_tied), n_tied)
    pfull[.~plock_tied] .= pfree_tied
    pfull[plock_tied] .= pfix_tied
    for tind in tied_indices
        insert!(pfull, tind, 0.)
    end
    for tie in tied_pairs
        pfull[tie[2]] = pfull[tie[1]] * tie[3]
    end

    return pfull
end


"""
    get_continuum_plimits(cube_fitter)

Get the continuum limits vector for a given CubeFitter object, possibly split up by the 2 continuum fitting steps.
Also returns a boolean vector for which parameters are allowed to vary.
"""
get_continuum_plimits(cube_fitter::CubeFitter, kwargs...) = cube_fitter.spectral_region == :MIR ? 
    get_mir_continuum_plimits(cube_fitter; kwargs...) : 
    get_opt_continuum_plimits(cube_fitter; kwargs...)


"""
    get_continuum_initial_values(cube_fitter, spaxel, λ, I, σ, N, init; split)

Get the vectors of starting values and relative step sizes for the continuum fit for a given CubeFitter object. 
Again, the vector may be split up by the 2 continuum fitting steps in the MIR case.
"""
get_continuum_initial_values(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real},
    N::Real; kwargs...) = cube_fitter.spectral_region == :MIR ? 
    get_mir_continuum_initial_values(cube_fitter, spaxel, λ, I, N; kwargs...) :
    get_opt_continuum_initial_values(cube_fitter, spaxel, λ, I; kwargs...)


"""
    get_continuum_parinfo(n_free, lb, ub, dp)

Get the CMPFit parinfo and config objects for a given CubeFitter object, given the vector of initial valuels,
limits, and relative step sizes.
"""
function get_continuum_parinfo(n_free::S, lb::Vector{T}, ub::Vector{T}, dp::Vector{T}) where {S<:Integer,T<:Real}

    parinfo = CMPFit.Parinfo(n_free)

    for pᵢ ∈ 1:n_free
        parinfo[pᵢ].fixed = 0
        parinfo[pᵢ].limited = (1,1)
        parinfo[pᵢ].limits = (lb[pᵢ], ub[pᵢ])
        # Set the relative step size for finite difference derivative calculations
        parinfo[pᵢ].relstep = dp[pᵢ]
    end

    # Create a `config` structure
    config = CMPFit.Config()
    config.maxiter = 500

    parinfo, config

end


# Version for the split fitting if use_pah_templates is enabled
function get_continuum_parinfo(n_free_1::S, n_free_2::S, lb_1::Vector{T}, ub_1::Vector{T}, 
    lb_2::Vector{T}, ub_2::Vector{T}, dp_1::Vector{T}, dp_2::Vector{T}) where {S<:Integer,T<:Real}

    parinfo_1 = CMPFit.Parinfo(n_free_1)
    parinfo_2 = CMPFit.Parinfo(n_free_2)

    for pᵢ ∈ 1:n_free_1
        parinfo_1[pᵢ].fixed = 0
        parinfo_1[pᵢ].limited = (1,1)
        parinfo_1[pᵢ].limits = (lb_1[pᵢ], ub_1[pᵢ])
        parinfo_1[pᵢ].relstep = dp_1[pᵢ]
    end

    for pᵢ ∈ 1:n_free_2
        parinfo_2[pᵢ].fixed = 0
        parinfo_2[pᵢ].limited = (1,1)
        parinfo_2[pᵢ].limits = (lb_2[pᵢ], ub_2[pᵢ])
        parinfo_2[pᵢ].relstep = dp_2[pᵢ]
    end

    # Create a `config` structure
    config = CMPFit.Config()
    config.maxiter = 500

    parinfo_1, parinfo_2, config

end


"""
    pretty_print_continuum_results(cube_fitter, popt, perr, I)

Print out a nicely formatted summary of the continuum fit results for a given CubeFitter object.
"""
pretty_print_continuum_results(cube_fitter::CubeFitter, popt::Vector{<:Real}, perr::Vector{<:Real},
    I::Vector{<:Real}) = cube_fitter.spectral_region == :MIR ? 
        pretty_print_mir_continuum_results(cube_fitter, popt, perr, I) :
        pretty_print_opt_continuum_results(cube_fitter, popt, perr, I)


# Helper function to loop through each line component and populate the limits, locked, name,
# and tied arrays 
function _line_plimits_component_loop!(cube_fitter::CubeFitter, ln_plims::Vector{Tuple},
    ln_lock::BitVector, ln_names::Vector{String}, amp_plim::Tuple, amp_tied::Vector{Any}, 
    amp_ratios::Vector{Any}, voff_tied::Vector{Any}, fwhm_tied::Vector{Any}, η_tied::Vector{Any})

    ind = 1
    for i ∈ 1:cube_fitter.n_lines
        for j ∈ 1:cube_fitter.n_comps
            if !isnothing(cube_fitter.lines.profiles[i, j])

                # name
                ln_name = string(cube_fitter.lines.names[i]) * "_$(j)"
                amp_ln_plim = isone(j) ? amp_plim : cube_fitter.lines.acomp_amp[i, j-1].limits
                amp_ln_locked = isone(j) ? false : cube_fitter.lines.acomp_amp[i, j-1].locked

                # get the right amp, voff, and fwhm parameters based on if theyre tied or not
                at = vt = ft = false
                ka = kv = kf = nothing
                if !isnothing(cube_fitter.lines.tied_amp[i, j])
                    key_amp = cube_fitter.lines.tied_amp[i, j]
                    ka = findfirst(cube_fitter.tied_kinematics.key_amp[j] .== key_amp)
                    at = true
                end

                if isnothing(cube_fitter.lines.tied_voff[i, j])
                    voff_ln_plim = cube_fitter.lines.voff[i, j].limits
                    voff_ln_locked = cube_fitter.lines.voff[i, j].locked
                    voff_ln_name = "$(ln_name)_voff"
                else
                    key_voff = cube_fitter.lines.tied_voff[i, j]
                    kv = findfirst(cube_fitter.tied_kinematics.key_voff[j] .== key_voff)
                    voff_ln_plim = cube_fitter.tied_kinematics.voff[j][kv].limits
                    voff_ln_locked = cube_fitter.tied_kinematics.voff[j][kv].locked
                    voff_ln_name = "$(key_voff)_$(j)_voff"
                    vt = true
                end
                if isnothing(cube_fitter.lines.tied_fwhm[i, j])
                    fwhm_ln_plim = cube_fitter.lines.fwhm[i, j].limits
                    fwhm_ln_locked = cube_fitter.lines.fwhm[i, j].locked
                    fwhm_ln_name = "$(ln_name)_fwhm"
                else
                    key_fwhm = cube_fitter.lines.tied_fwhm[i, j]
                    kf = findfirst(cube_fitter.tied_kinematics.key_fwhm[j] .== key_fwhm)
                    fwhm_ln_plim = cube_fitter.tied_kinematics.fwhm[j][kf].limits
                    fwhm_ln_locked = cube_fitter.tied_kinematics.fwhm[j][kf].locked
                    fwhm_ln_name = "$(key_fwhm)_$(j)_fwhm"
                    ft = true
                end

                if at
                    append!(amp_tied[j][ka], [ind])
                    append!(amp_ratios[j][ka], [cube_fitter.tied_kinematics.amp[j][ka][cube_fitter.lines.names[i]]])
                end

                # Depending on flexible_wavesol, we need to add 2 voffs instead of 1 voff
                if !isnothing(cube_fitter.lines.tied_voff[i, j]) && cube_fitter.flexible_wavesol && isone(j)
                    append!(ln_plims, [amp_ln_plim, voff_ln_plim, cube_fitter.lines.voff[i, j].limits, fwhm_ln_plim])
                    append!(ln_lock, [amp_ln_locked, voff_ln_locked, cube_fitter.lines.voff[i, j].locked, fwhm_ln_locked])
                    append!(ln_names, ["$(ln_name)_amp", voff_ln_name, "$(ln_name)_voff_indiv", fwhm_ln_name])
                    append!(voff_tied[j][kv], [ind+1])
                    if ft
                        append!(fwhm_tied[j][kf], [ind+3])
                    end
                    ind += 4
                else
                    append!(ln_plims, [amp_ln_plim, voff_ln_plim, fwhm_ln_plim])
                    append!(ln_lock, [amp_ln_locked, voff_ln_locked, fwhm_ln_locked])
                    append!(ln_names, ["$(ln_name)_amp", voff_ln_name, fwhm_ln_name])
                    if vt
                        append!(voff_tied[j][kv], [ind+1])
                    end
                    if ft
                        append!(fwhm_tied[j][kf], [ind+2])
                    end
                    ind += 3
                end

                # check for additional profile parameters
                if cube_fitter.lines.profiles[i, j] == :GaussHermite
                    # add h3 and h4 moments
                    append!(ln_plims, [cube_fitter.lines.h3[i, j].limits, cube_fitter.lines.h4[i, j].limits])
                    append!(ln_lock, [cube_fitter.lines.h3[i, j].locked, cube_fitter.lines.h4[i, j].locked])
                    append!(ln_names, ["$(ln_name)_h3", "$(ln_name)_h4"])
                    ind += 2
                elseif cube_fitter.lines.profiles[i, j] == :Voigt
                    # add voigt mixing parameter, but only if it's not tied
                    if !cube_fitter.tie_voigt_mixing
                        append!(ln_plims, [cube_fitter.lines.η[i, j].limits])
                        append!(ln_lock, [cube_fitter.lines.η[i, j].locked])
                        append!(ln_names, ["$(ln_name)_eta"])
                    else
                        append!(ln_plims, [cube_fitter.voigt_mix_tied.limits])
                        append!(ln_lock, [cube_fitter.voigt_mix_tied.locked])
                        append!(ln_names, ["eta_tied"])
                        append!(η_tied, [ind])
                    end
                    ind += 1
                end
            end
        end
    end

end


# Helper function to sort line fit parameter based on which ones are tied together.
# Returns 2 helpful objects - tied_pairs (the pairs of indices for parameters that are tied to each other),
# and tied_indices (the full list of indicies that have tied parameters in them)
function _line_plimits_organize_tied_comps(cube_fitter::CubeFitter, amp_tied::Vector{Any}, 
    amp_ratios::Vector{Any}, voff_tied::Vector{Any}, fwhm_tied::Vector{Any}, η_tied::Vector{Any})

    # Combine all "tied" vectors
    tied = []
    tied_amp_inds = []
    tied_amp_ratios = []
    for j ∈ 1:cube_fitter.n_comps
        for k ∈ 1:length(cube_fitter.tied_kinematics.key_amp[j])
            append!(tied_amp_inds, [tuple(amp_tied[j][k]...)])
            append!(tied_amp_ratios, [tuple(amp_ratios[j][k]...)])
        end
        for k ∈ 1:length(cube_fitter.tied_kinematics.key_voff[j])
            append!(tied, [tuple(voff_tied[j][k]...)])
        end
        for k ∈ 1:length(cube_fitter.tied_kinematics.key_fwhm[j])
            append!(tied, [tuple(fwhm_tied[j][k]...)])
        end
    end
    append!(tied, [tuple(η_tied...)])

    # Convert the tied vectors into tuples for each pair of parameters
    tied_pairs = Tuple[]
    for group in tied
        if length(group) > 1
            append!(tied_pairs, [(group[1],group[j],1.0) for j in 2:length(group)])
        end
    end
    for groupind in eachindex(tied_amp_inds)
        group = tied_amp_inds[groupind]
        ratio = tied_amp_ratios[groupind]
        if length(group) > 1
            append!(tied_pairs, [(group[1],group[j],ratio[j]/ratio[1]) for j in 2:length(group)])
        end
    end

    # Convert the paired tuples into indices for each tied parameter
    tied_indices = Vector{Int}(sort([tp[2] for tp in tied_pairs]))

    tied_pairs, tied_indices 
end


"""
    get_line_plimits(cube_fitter, init[, ext_curve])

Get the line limits vector for a given CubeFitter object. Also returns boolean locked values and
names of each parameter as strings.
"""
function get_line_plimits(cube_fitter::CubeFitter)

    # if !isnothing(ext_curve)
    #     max_amp = clamp(1 / minimum(ext_curve), 1., Inf) * (nuc_temp_fit ? 1000. : 1.)
    #     if cube_fitter.lines_allow_negative
    #         amp_plim = (-max_amp, max_amp)
    #     else
    #         amp_plim = (0., max_amp)
    #     end
    # else
    if cube_fitter.spectral_region == :MIR
        max_amp = 1 / exp(-cube_fitter.continuum.τ_97.limits[2])
    elseif cube_fitter.extinction_curve == "ccm"
        max_amp = 1 / attenuation_cardelli([cube_fitter.cube.λ[1]], cube_fitter.continuum.E_BV.limits[2])[1]
    elseif cube_fitter.extinction_curve == "calzetti"
        Cf_dust = cube_fitter.fit_covering_frac ? cube_fitter.continuum.frac : 0.
        if cube_fitter.fit_uv_bump 
            max_amp = 1 / attenuation_calzetti([cube_fitter.cube.λ[1]], cube_fitter.continuum.E_BV.limits[2],
                cube_fitter.continuum.δ_uv, Cf=Cf_dust)[1]
        else
            max_amp = 1 / attenuation_calzetti([cube_fitter.cube.λ[1]], cube_fitter.continuum.E_BV.limits[2],
                Cf=Cf_dust)[1]
        end
    end 
    if cube_fitter.lines_allow_negative
        amp_plim = (-max_amp, max_amp)
    else
        amp_plim = (0, max_amp)
    end

    ln_plims = Vector{Tuple}()
    ln_lock = BitVector()
    ln_names = Vector{String}()
    
    amp_tied = []
    amp_ratios = []
    voff_tied = []
    fwhm_tied = []
    for j ∈ 1:cube_fitter.n_comps
        append!(amp_tied, [[[] for _ in cube_fitter.tied_kinematics.key_amp[j]]])
        append!(amp_ratios, [[[] for _ in cube_fitter.tied_kinematics.key_amp[j]]])
        append!(voff_tied, [[[] for _ in cube_fitter.tied_kinematics.key_voff[j]]])
        append!(fwhm_tied, [[[] for _ in cube_fitter.tied_kinematics.key_fwhm[j]]])
    end
    η_tied = []
    
    # Loop through each line and append the new components
    _line_plimits_component_loop!(cube_fitter, ln_plims, ln_lock, ln_names, amp_plim, amp_tied,
        amp_ratios, voff_tied, fwhm_tied, η_tied)

    tied_pairs, tied_indices = _line_plimits_organize_tied_comps(cube_fitter, amp_tied, amp_ratios, 
        voff_tied, fwhm_tied, η_tied)

    ln_plims, ln_lock, ln_names, tied_pairs, tied_indices

end


# Helper function for getting line fit parameters based on an input previous parameter cube
function get_line_initial_values_from_pcube(cube_fitter::CubeFitter, spaxel::CartesianIndex)

    pcube_lines = cube_fitter.p_init_cube_lines
    # Get the coordinates of all spaxels that have fit results
    coords0 = [float.(c.I) for c in CartesianIndices(size(pcube_lines)[1:2]) if !all(isnan.(pcube_lines[c,:]))]
    # Transform to coordinates in the WCS of our current frame
    coords = cube_fitter.p_init_cube_coords
    # Calculate their distances from the current spaxel
    dist = [hypot(spaxel[1]-c[1], spaxel[2]-c[2]) for c in coords]
    closest = coords0[argmin(dist)]
    @debug "Using initial best fit line parameters from coordinates $closest --> $(coords[argmin(dist)])"

    ln_pars = pcube_lines[Int.(closest)..., :]

    ln_pars
end


# Helper function for getting line fit parameters based on estimates from the data
function get_line_initial_values_from_estimation(cube_fitter::CubeFitter)
    
    # Start the ampltiudes at 1/2 and 1/4 (in normalized units)
    A_ln = ones(cube_fitter.n_lines) .* 0.5

    # Initial parameter vector
    ln_pars = Float64[]
    for i ∈ 1:cube_fitter.n_lines
        for j ∈ 1:cube_fitter.n_comps
            if !isnothing(cube_fitter.lines.profiles[i, j])

                amp_ln = isone(j) ? A_ln[i] : cube_fitter.lines.acomp_amp[i, j-1].value
                if !isnothing(cube_fitter.lines.tied_amp[i, j])
                    key_amp = cube_fitter.lines.tied_amp[i, j]
                    ka = findfirst(cube_fitter.tied_kinematics.key_amp[j] .== key_amp)
                    amp_ln *= cube_fitter.tied_kinematics.amp[j][ka][cube_fitter.lines.names[i]]
                end
                if isnothing(cube_fitter.lines.tied_voff[i, j])
                    voff_ln = cube_fitter.lines.voff[i, j].value
                else
                    key_voff = cube_fitter.lines.tied_voff[i, j]
                    kv = findfirst(cube_fitter.tied_kinematics.key_voff[j] .== key_voff)
                    voff_ln = cube_fitter.tied_kinematics.voff[j][kv].value
                end
                if isnothing(cube_fitter.lines.tied_fwhm[i, j])
                    fwhm_ln = cube_fitter.lines.fwhm[i, j].value
                else
                    key_fwhm = cube_fitter.lines.tied_fwhm[i, j]
                    kf = findfirst(cube_fitter.tied_kinematics.key_fwhm[j] .== key_fwhm)
                    fwhm_ln = cube_fitter.tied_kinematics.fwhm[j][kf].value
                end

                # Depending on flexible_wavesol option, we need to add 2 voffs
                if !isnothing(cube_fitter.lines.tied_voff[i, j]) && cube_fitter.flexible_wavesol && isone(j)
                    append!(ln_pars, [amp_ln, voff_ln, cube_fitter.lines.voff[i, j].value, fwhm_ln])
                else
                    append!(ln_pars, [amp_ln, voff_ln, fwhm_ln])
                end

                if cube_fitter.lines.profiles[i, j] == :GaussHermite
                    # 2 extra parameters: h3 and h4
                    append!(ln_pars, [cube_fitter.lines.h3[i, j].value, cube_fitter.lines.h4[i, j].value])
                elseif cube_fitter.lines.profiles[i, j] == :Voigt
                    # 1 extra parameter: eta
                    if !cube_fitter.tie_voigt_mixing
                        # Individual eta parameter
                        append!(ln_pars, [cube_fitter.lines.η[i, j].value])
                    else
                        # Tied eta parameter
                        append!(ln_pars, [cube_fitter.voigt_mix_tied.value])
                    end
                end
            end
        end
    end

    ln_pars
end


# Helper function for getting line parameter step sizes
function get_line_step_sizes(cube_fitter::CubeFitter, ln_pars::Vector{<:Real}, init::Bool)

    # Absolute step size vector (0 tells CMPFit to use a default value)
    ln_astep = zeros(length(ln_pars))

    if !init
        ln_astep = Float64[]
        for i ∈ 1:cube_fitter.n_lines
            for j ∈ 1:cube_fitter.n_comps
                if !isnothing(cube_fitter.lines.profiles[i, j])

                    # A small absolute step size of 1e-5 for the velocities / FWHMs helps in particular when
                    # the starting value is 0 to get out of becoming stuck
                    amp_step = 0.
                    voff_step = 1e-5
                    fwhm_step = j > 1 && cube_fitter.relative_flags[3] ? 0. : 1e-5

                    # Depending on flexible_wavesol option, we need to add 2 voffs
                    if !isnothing(cube_fitter.lines.tied_voff[i, j]) && cube_fitter.flexible_wavesol && isone(j)
                        append!(ln_astep, [amp_step, voff_step, voff_step, fwhm_step])
                    else
                        append!(ln_astep, [amp_step, voff_step, fwhm_step])
                    end

                    if cube_fitter.lines.profiles[i, j] == :GaussHermite
                        # 2 extra parameters: h3 and h4
                        append!(ln_astep, [0., 0.])
                    elseif cube_fitter.lines.profiles[i, j] == :Voigt
                        # 1 extra parameter: eta
                        append!(ln_astep, [0.])
                    end
                end
            end
        end 
    end

    ln_astep
end


"""
    get_line_initial_values(cube_fitter, spaxel, init)

Get the vector of starting values and relative step sizes for the line fit for a given CubeFitter object.
"""
function get_line_initial_values(cube_fitter::CubeFitter, spaxel::CartesianIndex, init::Bool)

    # Check if cube fitter has initial cube
    if !isnothing(cube_fitter.p_init_cube_λ)
        ln_pars = get_line_initial_values_from_pcube(cube_fitter, spaxel)
    # Check if there are previous best fit parameters
    elseif !init
        @debug "Using initial best fit line parameters..."
        # If so, set the parameters to the previous ones
        ln_pars = copy(cube_fitter.p_init_line)
    else
        @debug "Calculating initial starting points..."
        # Otherwise, make initial guesses based on the data
        ln_pars = get_line_initial_values_from_estimation(cube_fitter)
    end
    # Get step sizes for each parameter
    ln_astep = get_line_step_sizes(cube_fitter, ln_pars, init)

    ln_pars, ln_astep

end


"""
    get_line_parinfo(n_free, lb, ub, dp)

Get the CMPFit parinfo and config objects for a given CubeFitter object, given the vector of initial values,
limits, and absolute step sizes.
"""
function get_line_parinfo(n_free, lb, ub, dp)

    # Convert parameter limits into CMPFit object
    parinfo = CMPFit.Parinfo(n_free)
    for pᵢ ∈ 1:n_free
        parinfo[pᵢ].fixed = 0
        parinfo[pᵢ].limited = (1,1)
        parinfo[pᵢ].limits = (lb[pᵢ], ub[pᵢ])
        parinfo[pᵢ].step = dp[pᵢ]
    end

    # Create a `config` structure
    config = CMPFit.Config()
    # Lower tolerance level for lines fit
    config.ftol = 1e-16
    config.xtol = 1e-16
    config.maxiter = 500

    parinfo, config
end


"""
    clean_line_parameters(cube_fitter, popt, lower_bounds, upper_bounds)

Takes the results of an initial global line fit and prepares the parameters for individual spaxel
fits by sorting the line components, fixing the voffs/FWHMs for lines that are not detected, and 
various other small adjustments.
"""
function clean_line_parameters(cube_fitter::CubeFitter, popt::Vector{<:Real}, lower_bounds::Vector{<:Real}, upper_bounds::Vector{<:Real})
    pᵢ = 1
    for i in 1:cube_fitter.n_lines
        pstart = Int[]
        pfwhm = Int[]
        pend = Int[]
        amp_main = popt[pᵢ]
        voff_main = popt[pᵢ+1]
        fwhm_main = (!isnothing(cube_fitter.lines.tied_voff[i, 1]) && cube_fitter.flexible_wavesol) ? popt[pᵢ+3] : popt[pᵢ+2]

        for j in 1:cube_fitter.n_comps
            n_prof = sum(.~isnothing.(cube_fitter.lines.profiles[i, :]))
            if !isnothing(cube_fitter.lines.profiles[i, j])
                push!(pstart, pᵢ)

                # If additional components arent detected, set them to a small nonzero value
                replace_line = iszero(popt[pᵢ])
                if replace_line
                    if j > 1
                        popt[pᵢ] = cube_fitter.relative_flags[1] ? 0.1 * 1/(n_prof-1) : 0.1 * 1/(n_prof-1) * amp_main
                        popt[pᵢ+1] = cube_fitter.relative_flags[2] ? 0.0 : voff_main
                        popt[pᵢ+2] = cube_fitter.relative_flags[3] ? 1.0 : fwhm_main
                    else
                        popt[pᵢ] = 0.9 * popt[pᵢ]
                        if isnothing(cube_fitter.lines.tied_voff[i, j])
                            popt[pᵢ+1] = voff_main = 0. # voff
                        end
                        if isnothing(cube_fitter.lines.tied_fwhm[i, j])
                            popt[pᵢ+2] = fwhm_main = (lower_bounds[pᵢ+2]+upper_bounds[pᵢ+2])/2 # fwhm
                        end
                    end
                end
                # Velocity offsets for the integrated spectrum shouldnt be too large
                # if abs(popt[pᵢ+1]) > 500.
                # if !cube_fitter.fit_all_global
                #     popt[pᵢ+1] = 0.
                # end

                if replace_line && !isnothing(cube_fitter.lines.tied_voff[i, j]) && isone(j) && cube_fitter.flexible_wavesol
                    popt[pᵢ+2] = 0. # individual voff
                end

                # Check if using a flexible_wavesol tied voff -> if so there is an extra voff parameter
                if !isnothing(cube_fitter.lines.tied_voff[i, j]) && cube_fitter.flexible_wavesol && isone(j)
                    pc = 4
                    push!(pfwhm, pᵢ+3)
                else
                    pc = 3
                    push!(pfwhm, pᵢ+2)
                end

                if cube_fitter.lines.profiles[i, j] == :GaussHermite
                    pc += 2
                elseif cube_fitter.lines.profiles[i, j] == :Voigt
                    # Set the Voigt mixing ratios back to 0.5 since a summed fit may lose the structure of the line-spread function
                    if !cube_fitter.tie_voigt_mixing && !cube_fitter.lines.η[i, j].locked
                        popt[pᵢ+pc] = 0.5
                    elseif cube_fitter.tie_voigt_mixing && !cube_fitter.voigt_mix_tied.locked
                        popt[pᵢ+pc] = 0.5
                    end
                    pc += 1
                end

                pᵢ += pc
                push!(pend, pᵢ-1)
            end
        end
        # resort line components by decreasing flux
        if all(.~cube_fitter.relative_flags) && !cube_fitter.flexible_wavesol
            pnew = copy(popt)
            # pstart gives the amplitude indices
            ss = sortperm(popt[pstart].*popt[pfwhm], rev=true)
            for k in eachindex(ss)
                pnew[pstart[k]:pend[k]] .= popt[pstart[ss[k]]:pend[ss[k]]]
            end
            popt = pnew
        end
    end
    return popt
end


"""
    pretty_print_line_results(cube_fitter, popt, perr)

Print out a nicely formatted summary of the line fit results for a given CubeFitter object.
"""
function pretty_print_line_results(cube_fitter::CubeFitter, popt::Vector{<:Real}, perr::Vector{<:Real})

    rel_amp, rel_voff, rel_fwhm = cube_fitter.relative_flags

    msg = "######################################################################\n"
    msg *= "############### SPAXEL FIT RESULTS -- EMISSION LINES #################\n"
    msg *= "######################################################################\n"
    pᵢ = 1
    msg *= "\n#> EMISSION LINES <#\n"
    for (k, name) ∈ enumerate(cube_fitter.lines.names)
        for j ∈ 1:cube_fitter.n_comps
            if !isnothing(cube_fitter.lines.profiles[k, j])
                nm = string(name) * "_$(j)"
                msg *= "$(nm)_amp:\t\t\t $(@sprintf "%.3f" popt[pᵢ]) +/- $(@sprintf "%.3f" perr[pᵢ]) " * ((isone(j) || !rel_amp) ? "[x norm]" : "[x amp_1]") * "\t " * 
                    "Limits: " * (isone(j) ? "(0, 1)" : "($(@sprintf "%.3f" cube_fitter.lines.acomp_amp[k, j-1].limits[1]), $(@sprintf "%.3f" cube_fitter.lines.acomp_amp[k, j-1].limits[2]))") * "\n"
                msg *= "$(nm)_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) " * ((isone(j) || !rel_voff) ? "km/s" : "[+ voff_1]") * " \t " *
                    "Limits: ($(@sprintf "%.0f" cube_fitter.lines.voff[k, j].limits[1]), $(@sprintf "%.0f" cube_fitter.lines.voff[k, j].limits[2]))\n"
                if !isnothing(cube_fitter.lines.tied_voff[k, j]) && cube_fitter.flexible_wavesol && isone(j)
                    msg *= "$(nm)_voff_indiv:   \t\t $(@sprintf "%.0f" popt[pᵢ+2]) +/- $(@sprintf "%.0f" perr[pᵢ+2]) km/s \t " *
                        "Limits: ($(@sprintf "%.0f" cube_fitter.lines.voff[k, j].limits[1]), $(@sprintf "%.0f" cube_fitter.lines.voff[k, j].limits[2]))\n"
                    msg *= "$(nm)_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+3]) +/- $(@sprintf "%.0f" perr[pᵢ+3]) km/s \t " *
                        "Limits: ($(@sprintf "%.0f" cube_fitter.lines.fwhm[k, j].limits[1]), $(@sprintf "%.0f" cube_fitter.lines.fwhm[k, j].limits[2]))\n"
                    pᵢ += 4
                else
                    if isone(j)
                        msg *= "$(nm)_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+2]) +/- $(@sprintf "%.0f" perr[pᵢ+2]) km/s \t " *
                            "Limits: ($(@sprintf "%.0f" cube_fitter.lines.fwhm[k, j].limits[1]), $(@sprintf "%.0f" cube_fitter.lines.fwhm[k, j].limits[2]))\n"
                    else
                        msg *= "$(nm)_fwhm:   \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) " * ((isone(j) || !rel_fwhm) ? "km/s" : "[x fwhm_1]") * "\t " *
                            "Limits: ($(@sprintf "%.3f" cube_fitter.lines.fwhm[k, j].limits[1]), $(@sprintf "%.3f" cube_fitter.lines.fwhm[k, j].limits[2]))\n"
                    end
                    pᵢ += 3
                end
                if cube_fitter.lines.profiles[k, j] == :GaussHermite
                    msg *= "$(nm)_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ]) +/- $(@sprintf "%.3f" perr[pᵢ])      \t " *
                        "Limits: ($(@sprintf "%.3f" cube_fitter.lines.h3[k, j].limits[1]), $(@sprintf "%.3f" cube_fitter.lines.h3[k, j].limits[2]))\n"
                    msg *= "$(nm)_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1])      \t " *
                        "Limits: ($(@sprintf "%.3f" cube_fitter.lines.h4[k, j].limits[1]), $(@sprintf "%.3f" cube_fitter.lines.h4[k, j].limits[2]))\n"
                    pᵢ += 2
                elseif cube_fitter.lines.profiles[k, j] == :Voigt 
                    msg *= "$(nm)_η:     \t\t $(@sprintf "%.3f" popt[pᵢ]) +/- $(@sprintf "%.3f" perr[pᵢ])      \t " *
                        "Limits: ($(@sprintf "%.3f" cube_fitter.lines.η[k, j].limits[1]), $(@sprintf "%.3f" cube_fitter.lines.η[k, j].limits[2]))\n"
                    pᵢ += 1
                end
            end
        end
        msg *= "\n"
    end 
    msg *= "######################################################################" 
    @debug msg

    msg

end

