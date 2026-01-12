#=
This file contains the CubeFitter struct and its related functions.  The CubeFitter is how
fitting is performed with Loki.
=#

############################## PARAMETER / MODEL STRUCTURES ####################################

abstract type Options end

"""
## Output Options ##

- `plot_spaxels`: A Symbol specifying the plotting backend to be used when plotting individual spaxel fits, can
    be either `:pyplot`, `:plotly`, or `:both`
- `plot_maps`: Whether or not to plot 2D maps of the best-fit parameters after the fitting is finished
- `plot_range`: An optional list of spectral regions to extract zoomed-in plots from (useful for plotting
specific emission lines of interest).
- `parallel`: Whether or not to fit multiple spaxels in parallel using multiprocessing
- `save_fits`: Whether or not to save the final best-fit models and parameters as FITS files
- `save_tables`: Whether or not to save the final best-fit parameters as CSV tables (note: these are separate from the
    CSV files that are saved to the "spaxel_binaries" folder during fitting to save progress) 
- `save_full_model`: Whether or not to save the full 3D best-fit model as a FITS file.
- `overwrite`: Whether or not to overwrite old fits of spaxels when rerunning
- `track_memory`: Whether or not to save diagnostic files showing memory usage of the program
- `track_convergence`: Whether or not to save diagnostic files showing convergence of line fitting for each spaxel
- `make_movies`: Whether or not to save mp4 files of the final model
- `map_snr_thresh`: The SNR threshold below which to mask out spaxels from parameter maps for emission lines.
- `sort_line_components`: Defines how sorting should be done (either by flux or by fwhm) for individual line components

"""
struct OutputOptions <: Options
    plot_spaxels::Symbol
    plot_maps::Bool
    plot_range::Union{Vector{<:Tuple},Nothing}
    parallel::Bool
    parallel_strategy::String
    save_fits::Bool
    save_tables::Bool
    save_full_model::Bool
    overwrite::Bool
    track_memory::Bool
    track_convergence::Bool
    make_movies::Bool
    map_snr_thresh::Real
    sort_line_components::Union{Symbol,Nothing}
    plot_line_annotation_positions::Vector{<:Real}
end

"""
## Fitting Options ##

- `parallel`: Whether or not to allow multiprocessing
- `parallel_strategy`: Either `pmap` or `distributed`
- `sys_err`: Optional systematic uncertainty quantified as a ratio (0-1) of the flux. I.e. to add a 10% systematic 
    uncertainty use fσ_sys = 0.1
- `extinction_curve`: The type of extinction curve being used, i.e. `"kvt"` or `"d+"`
- `extinction_screen`: Whether or not the extinction is modeled as a screen
- `κ_abs`: Mass absorption coefficient for amorphous olivine Mg(2y)Fe(2-2y)SiO4, amorphous pyroxene Mg(x)Fe(1-x)SiO3, and 
    crystalline forsterite Mg2SiO4 as interpolating functions over wavelength, only used if extinction_curve == "decompose"
- `custom_ext_template`: A custom dust extinction template given as a matrix of wavelengths and optical depths that is
    converted into an interpolating function.
- `ebv_map`: An optional map of estimated E(B-V) values. The fits will be locked to the value at the corresponding spaxel.
- `sil_abs_map`: An optional map of estimated τ_97 values. The fits will be locked to the value at the corresponding spaxel.
- `fit_stellar_continuum`: Whether or not to fit stellar continuum
- `fit_sil_emission`: Whether or not to fit MIR hot silicate dust emission
- `fit_ch_abs`: Whether or not to fit the MIR CH absorption feature
- `fit_temp_multexp`: Whether or not to apply and fit multiplicative exponentials to any provided templates
- `guess_tau`: Whether or not to guess the optical depth at 9.7 microns by interpolating between 
    PAH-free parts of the continuum at the given wavelength windows in microns (must have enough spectral coverage). 
    The fitted value will then be constrained to be at least 80% of the inferred value.
- `fit_opt_na_feii`: Whether or not to fit optical narrow Fe II emission
- `fit_opt_br_feii`: Whether or not to fit optical broad Fe II emission
- `fit_all_global`: Whether or not to fit all spaxels with a global optimizer (Differential Evolution) instead of a local optimizer (Levenberg-Marquardt).
- `use_pah_templates`: Whether or not to fit the continuum in two steps, where the first step uses PAH templates,
and the second step fits the PAH residuals with the PAHFIT Drude model.
- `fit_joint`: If true, fit the continuum and lines simultaneously. If false, the lines will first be masked and the
continuum will be fit, then the continuum is subtracted and the lines are fit to the residuals.
- `fit_uv_bump`: Whether or not to fit the UV bump in the dust attenuation profile. Only applies if the extinction
curve is "calzetti".
- `fit_covering_frac`: Whether or not to fit a dust covering fraction in the attenuation profile. Only applies if the
extinction curve is "calzetti".
- `tie_template_amps`: If true, the template amplitudes in each channel are tied to the same value. Otherwise they may have separate
- `nirspec_mask_chip_gaps`: If true, will mask data between the gaps in the NIRSpec chips.
normalizations in each channel. By default this is false.
- `decompose_lock_column_densities`: If true and if using the "decompose" extinction profile, then the column densities for
pyroxene and forsterite (N_pyr and N_for) will be locked to their values after the initial fit to the integrated spectrum over all 
spaxels. These parameters are measured relative to olivine (N_oli) so this, in effect, locks the relative abundances of these three silicates
over the full FOV of the cube.
- `n_bootstrap`: The number of bootstrapping iterations that should be performed for each fit.
- `random_seed`: An optional number to use as a seed for the RNG utilized during bootstrapping, to ensure consistency between
- `line_test_lines`: A list of lines which should be tested for additional components. They may be grouped
together (hence the vector-of-a-vector structure) such that lines in a group will all be given the maximum number of parameters that
any one line passes the test for.
- `line_test_threshold`: A threshold which must be met in the chi^2 ratio between a fit without an additional line profile and
one with the additional profile, in order to include the additional profile in the final fit.
- `plot_line_test`: Whether or not to plot the line test results.
- `lines_allow_negative`: Whether or not to allow line amplitdues to go negative
- `subtract_cubic_spline`: If true, subtract a cubic spline fit to the continuum rather than the actual fit to the continuum before
fitting emission lines.  DO NOT USE THIS OPTION IF YOU EXPECT THERE TO BE STELLAR ABSORPTION.

"""
mutable struct FittingOptions{T<:Real,S<:Integer} <: Options
    sys_err::Real
    silicate_absorption::String
    extinction_curve::String
    extinction_screen::Bool
    κ_abs::Union{Vector{Spline1D},Nothing}
    custom_ext_template::Union{Spline1D,Nothing}
    ebv_map::Union{Array{T,3},Nothing}
    sil_abs_map::Union{Array{T,3},Nothing}
    fit_stellar_continuum::Bool
    ssp_regularize::T
    stellar_template_type::String
    custom_stellar_template_wave::Union{Vector{T},Nothing}
    custom_stellar_template_R::Union{Vector{T},Nothing}
    custom_stellar_templates::Union{Array{T,2},Nothing}
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
    lock_hot_dust::Bool
    F_test_ext::Bool
    nirspec_mask_chip_gaps::Bool
    decompose_lock_column_densities::Bool
    n_bootstrap::S
    random_seed::S
    line_test_lines::Vector{Vector{Symbol}}
    line_test_threshold::T
    plot_line_test::Bool
    lines_allow_negative::Bool
    subtract_cubic_spline::Bool
end


"""
    CubeFitter(cube, z, name; <keyword arguments>)

This is the main structure used for fitting IFU cubes, containing all of the necessary data, metadata,
fitting options, and associated functions for generating ParamMaps and CubeModel structures to handle the outputs 
of all the fits.  This is essentially the "configuration object" that tells the rest of the fitting code how
to run. The actual fitting functions (`fit_spaxel` and `fit_cube!`) require an instance of this structure.

Other than `cube`, `z`, and `name`, all fields listed below will be populated by the defaults given in the 
`config.toml`, `infrared.toml`, `lines.toml`, and `optical.toml` configuration files, or by quantities derived from
these options. For more detailed explanations on these options, please refer to the options files and the README file.

### Fields ###

## Data ##

- `cube`: The main DataCube object containing the cube that is being fit
- `z`: The redshift of the target that is being fit
- `name`: The name of the fitting run being performed.
- `spectral_region`: The SpectralRegion object of the data being fit

## Options ##

- `output`: The output options
- `fitting`: The fitting options

## Parameters ##

- `model`: The model parameters

- `templates`: Each template to be used in the fit. The first 2 axes are spatial, the 3rd axis should be wavelength, 
    and the 4th axis should iterate over each individual template. Each template will get an amplitude parameter in the fit.
- `template_names`: The names of each generic template in the fit.

- `ssps`: The Stellar Population templates object
- `vsyst_ssp`: The systemic velocity offset between the input wavelength grid and the SSP template wavelength grid.

- `feii_templates_fft`: A matrix of the Fourier transforms of the narrow and broad Fe II templates
- `vsyst_feii`: The systemic velocity offset between the input wavelength grid and the Fe II template wavelength grid.
- `npad_feii`: The length of the Fe II templates (NOT the length of the Fourier transformed templates).

- `vres`: The constant velocity resolution of the wavelength grid, which assumes logarithmic spacing, in km/s/pixel.
- `n_ssps`: The number of simple stellar population components
- `n_dust_cont`: The number of dust continuum components
- `n_power_law`: The number of power law continuum components
- `n_dust_feat`: The number of PAH features
- `n_abs_feat`: The number of absorption features
- `n_templates`: The number of generic templates
- `n_lines`: The number of lines being fit
- `n_acomps`: The summed total number of line profiles fit to all lines in the spectrum, including additional components.
- `n_comps`: The maximum number of additional profiles that may be fit to any given line.

- `apoly_degree`: The degree of additive polynomials.
- `mpoly_degree`: The degree of multiplicative polynomials.
- `apoly_type`: The type of additive polynomials.
- `mpoly_type`: The type of multiplicative polynomials.

- `n_params_cont`: The total number of free fitting parameters for the continuum fit (not including emission lines)
- `n_params_dust`: The total number of free fitting parameters for the PAH fit (not including the continuum)
- `n_params_lines`: The total number of free fitting parameters for the emission line fit (not including the continuum)
- `n_params_extra`: The total number of extra parameters calculated for each fit (includes things like line fluxes and equivalent widths)

- `cosmology`: The Cosmology, used solely to create physical scale bars on the 2D parameter plots

- `linemask_overrides`: Optional list of tuples specifying (min, max) wavelength ranges that will be forcibly 
    added to the line mask. This is different from the `user_mask` option since it only applies to the continuum fitting step but
    will be ignored during the line fitting step. NOTE: Using this option will disable the 'automatic' line masking algorithm and just
    use the manually input regions in the mask.
- `linemask_width`: The width in km/s to automatically apply a line mask

## Best fit parameters ##

- `p_init_cont`: The best-fit continuum parameters for the initial fit to the sum of all spaxels.
- `p_init_line`: Same as `p_init_cont`, but for the line parameters
- `p_init_pahtemp`: Same as `p_init_cont`, but for the PAH templates amplitudes

See [`ParamMaps`](@ref), [`CubeModel`](@ref), [`OutputOptions`](@ref), [`FittingOptions`](@ref), 
    [`fit_spaxel`](@ref), [`fit_cube!`](@ref)
"""
struct CubeFitter{T<:Real,S<:Integer,Q<:QSIntensity,Qv<:QVelocity,Qw<:QWave}

    # See explanations for each field in the docstring!
    
    # Data
    cube::DataCube
    z::T
    name::String
    spectral_region::SpectralRegion

    # Options for the output
    output::OutputOptions

    # Options for the fitting 
    fitting::FittingOptions

    # Model Parameters object
    model::ModelParameters

    # 4D templates: first 3 axes are the template's spatial and spectral axes, while the 4th axis enumerates individual templates
    templates::Array{Q, 4}  
    template_names::Vector{String}

    # Stellar populations
    ssps::Union{StellarPopulations,Nothing}

    # Fe II templates
    feii::Union{FeIITemplates,Nothing}

    # A few parameters derived from the model parameters object
    vres::Qv
    n_ssps::S
    n_dust_cont::S
    n_power_law::S  
    n_dust_feat::S
    n_abs_feat::S
    n_templates::S
    n_lines::S
    n_acomps::S

    apoly_degree::S
    mpoly_degree::S
    apoly_type::Symbol
    mpoly_type::Symbol

    # Total number of parameters for the continuum, dust features, lines, and extra
    n_params_cont::S
    n_params_dust::S
    n_params_lines::S
    n_params_extra::S
    n_params_total::S

    # Line spread function interpolator
    lsf::Function
    dust_profiles::Dict{String, Tuple{Vector{<:QWave},Vector{<:AbstractFloat}}}
    dust_interpolators::Dict{String, Spline1D}

    # The number of actually fit line profiles in each spaxel
    n_fit_comps::Dict{Symbol,Matrix{S}}
    
    # The cosmology to be used when necessary
    cosmology::Cosmology.AbstractCosmology

    # Line masking and sorting options
    linemask_overrides::Vector{Tuple{Qw,Qw}}
    linemask_width::Qv

    # Initial parameter vectors
    p_init_cont::Vector{Number}
    p_init_line::Vector{Number}
    p_init_pahtemp::Vector{Number}

    #= Constructor function --> the default inputs are all taken from the configuration files, but may be overwritten
    by the kwargs object using the same syntax as any keyword argument. The rest of the fields are generated in the function 
    from these inputs =#
    function CubeFitter(cube::DataCube, z::Real, name::String; 
        options_file::String=joinpath(@__DIR__, "..", "options", "options.toml"),
        lines_file::String=joinpath(@__DIR__, "..", "options", "lines.toml"),
        optical_file::String=joinpath(@__DIR__, "..", "options", "optical.toml"),
        infrared_file::String=joinpath(@__DIR__, "..", "options", "infrared.toml"),
        kwargs...) 

        # Do a few checks on the input data cube
        @assert cube.vacuum_wave "Please make sure the data is in vacuum wavelengths before constructing a CubeFitter! (use function: correct!)"
        @assert cube.rest_frame "Please make sure the data is in the rest-frame before constructing a CubeFitter! (use function: correct!)"
        @assert cube.masked "Please make sure the data is properly masked before constructing a CubeFitter! (use function: correct!)"
        @assert cube.dereddened "Please make sure the data is de-reddened before constructing a CubeFitter! (use function: deredden!)"
        
        # Some setup variables
        λ = cube.λ
        λunit = unit(λ[1])
        Iunit = unit(cube.I[1])
        spectral_region = cube.spectral_region
        name = replace(name, #= no spaces! =# " " => "_")

        # Prepare options from the configuration file
        # joinpath(@__DIR__, "..", "options", "options.toml")
        options = parse_options(options_file)

        # Overwrite with any options provided in the keyword arguments
        out = copy(options)
        for key in keys(kwargs)
            out[key] = kwargs[key]
        end

        # Set up and reformat some default options
        cubefitter_add_default_options!(cube, out)
        # Set up the extinction map and PAH template boolean map 
        ebv_map, sil_abs_map = cubefitter_prepare_extinction_maps(out, cube)
        # Set up potential custom stellar templates 
        custom_stellar_template_wave, custom_stellar_template_R, custom_stellar_templates = 
            cubefitter_prepare_custom_stellar_templates(out, cube)
        # Set up the output directories
        cubefitter_prepare_output_directories(name, out)
        # add the user mask, if given
        if !isnothing(out[:user_mask])
            append!(spectral_region.mask, out[:user_mask])
        end
        # Disable regularization for any non-SSP stellar templates
        if haskey(out, :stellar_template_type) && (out[:stellar_template_type] != "ssp")
            if out[:ssp_regularize] != 0.0
                @warn "Setting ssp_regularize=0.0 since stellar_template_type is not \"ssp\""
                out[:ssp_regularize] = 0.0
            end
        end

        #############################################################

        @debug """\n
        Creating CubeFitter struct for $name at z=$z
        ############################################
        """

        # Create the model parameters, stellar and iron templates, and count various parameters
        model_parameters, ssps, feii, n_ssps, n_power_law, n_dust_cont, n_dust_feat, n_abs_feat, n_templates, vres =
            cubefitter_prepare_continuum(optical_file, infrared_file, lines_file, λ, z, out, λunit, Iunit, spectral_region, 
            name, cube, custom_stellar_template_wave, custom_stellar_template_R, custom_stellar_templates)
        lines, n_lines, n_acomps, n_fit_comps = cubefitter_prepare_lines(lines_file, out, λunit, Iunit, cube, spectral_region)

        # Count total parameters
        n_params_cont = count_cont_parameters(model_parameters; split=false)
        n_params_dust = count_dust_parameters(model_parameters)
        n_params_lines = count_line_parameters(lines)
        n_params_extra = count_extra_parameters(model_parameters)
        n_params_total = n_params_cont + n_params_lines + n_params_extra

        # Create the line mask
        make_linemask!(out, lines, λunit)

        @debug "### There are a total of $(n_params_cont-n_params_dust) continuum parameters ###"
        @debug "### There are a total of $(n_params_dust) PAH parameters ###"
        @debug "### There are a total of $(n_params_lines) emission line parameters ###"
        @debug "### There are a total of $(n_params_extra) extra parameters ###"

        # Pre-calculate mass absorption coefficients for olivine, pyroxene, and forsterite
        κ_abs = nothing 
        if out[:silicate_absorption] == "decompose"
            gunit = typeof(out[:grain_size]) <: QLength ? 1.0 : u"μm"
            κ_oli, κ_pyr, κ_for = read_dust_κ(out[:pyroxene_x], out[:olivine_y], out[:grain_size]*gunit, λunit)
            κ_abs = [κ_oli, κ_pyr, κ_for]
        end

        # Prepare initial best fit parameter options
        @debug "Preparing initial best fit parameter vectors with $(n_params_cont) and $(n_params_lines) parameters"
        p_init_cont = zeros(n_params_cont)
        p_init_line = zeros(n_params_lines)
        p_init_pahtemp = zeros(2)

        # If a fit has been run previously, read in the file containing the best fit parameters
        # to pick up where the fitter left off seamlessly
        if isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv")) && isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"))
            p_init_cont, = read_fit_results_csv(name, "init_fit_cont")
            p_init_line, = read_fit_results_csv(name, "init_fit_line")
            p_init_pahtemp, = read_fit_results_csv(name, "init_fit_pahtemp")
        end

        # Load templates into memory
        dust_profiles, dust_interpolators = _load_dust_templates(out[:silicate_absorption], out[:fit_ch_abs], out[:use_pah_templates], 
            λunit, Iunit)

        # Create the LSF interpolator
        lsf_interp = Spline1D(ustrip.(cube.λ), ustrip.(cube.lsf), k=1)  # rest frame wavelengths
        lsf = λi -> lsf_interp(ustrip(uconvert(unit(cube.λ[1]), λi)))*unit(cube.lsf[1])

        @debug "Preparing line annotation positions for line ID plotting..."
        line_annotation_positions = zeros(sum(model_parameters.lines.config.annotate))
        if isfile(joinpath("output_$name", "line_annotation_positions.csv"))
            line_annotation_positions = readdlm(joinpath("output_$name", "line_annotation_positions.csv"), ',', Float64, '\n')[:,1]
        end

        # Reset all global variables 
        # (these are defined in math.jl)
        global nnls_workspace = NNLSWorkspace(0, 0)
        global rfft_cache = nothing
        global irfft_plan = nothing
        global A_cache    = nothing 
        global b_cache    = nothing 
        global A1_cache   = nothing 
        global b1_cache   = nothing

        # Create options structs
        output_options = OutputOptions(
            out[:plot_spaxels],
            out[:plot_maps],
            out[:plot_range],
            out[:parallel],
            out[:parallel_strategy],
            out[:save_fits],
            out[:save_tables],
            out[:save_full_model],
            out[:overwrite],
            out[:track_memory],
            out[:track_convergence],
            out[:make_movies],
            out[:map_snr_thresh],
            out[:sort_line_components],
            line_annotation_positions
        )
        fitting_options = FittingOptions(
            out[:sys_err],
            out[:silicate_absorption],
            out[:extinction_curve],
            out[:extinction_screen],
            κ_abs,
            out[:custom_ext_template],
            ebv_map,
            sil_abs_map,
            out[:fit_stellar_continuum],
            out[:ssp_regularize],
            out[:stellar_template_type],
            custom_stellar_template_wave,
            custom_stellar_template_R,
            custom_stellar_templates,
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
            out[:lock_hot_dust],
            out[:F_test_ext],
            out[:nirspec_mask_chip_gaps],
            out[:decompose_lock_column_densities],
            out[:n_bootstrap],
            out[:random_seed],
            out[:line_test_lines],
            out[:line_test_threshold],
            out[:plot_line_test],
            out[:lines_allow_negative],
            out[:subtract_cubic_spline]
        )

        new{typeof(z), typeof(n_params_cont), eltype(out[:templates]), typeof(vres), eltype(cube.λ)}(
            cube, 
            z, 
            name, 
            spectral_region, 
            output_options,
            fitting_options,
            model_parameters,
            out[:templates],
            out[:template_names],
            ssps,
            feii,
            vres,
            n_ssps,
            n_dust_cont,
            n_power_law,
            n_dust_feat,
            n_abs_feat,
            n_templates,
            n_lines,
            n_acomps,
            out[:apoly_degree],
            out[:mpoly_degree],
            out[:apoly_type],
            out[:mpoly_type],
            n_params_cont,
            n_params_dust,
            n_params_lines,
            n_params_extra,
            n_params_total,
            lsf,
            dust_profiles,
            dust_interpolators,
            n_fit_comps,
            out[:cosmology],
            out[:linemask_overrides],
            out[:linemask_width],
            p_init_cont,
            p_init_line,
            p_init_pahtemp
        )
    end
end


# Interface functions for the cube fitter
fit_options(cube_fitter::CubeFitter) = cube_fitter.fitting
out_options(cube_fitter::CubeFitter) = cube_fitter.output
model(cube_fitter::CubeFitter) = cube_fitter.model


# Helper function for making output directoreis when setting up the CubeFitter object
function cubefitter_prepare_output_directories(name::String, out::Dict)

    # Prepare output directories
    @info "Preparing output directories"

    # Top-level output directory
    if !isdir("output_$name")
        mkdir("output_$name")
    end
    # Sub-folders 
    for foldername in ("spaxel_plots", "spaxel_binaries", "param_maps", "logs")
        if !isdir(joinpath("output_$name", foldername))
            mkdir(joinpath("output_$name", foldername))
        end
    end
    if !isdir(joinpath("output_$name", "zoomed_plots")) && !isnothing(out[:plot_range])
        mkdir(joinpath("output_$name", "zoomed_plots"))
    end
    if !isdir(joinpath("output_$name", "line_tests")) && (length(out[:line_test_lines]) > 0) && out[:plot_line_test]
        mkdir(joinpath("output_$name", "line_tests"))
    end

end


# Helper function for setting up default options when creating a CubeFitter object
function cubefitter_add_default_options!(cube::DataCube, out::Dict)

    out[:line_test_lines] = Vector{Symbol}[Symbol[Symbol(ln) for ln in group] for group in out[:line_test_lines]]
    out[:plot_spaxels] = Symbol(out[:plot_spaxels])
    out[:apoly_type] = Symbol(out[:apoly_type])
    out[:mpoly_type] = Symbol(out[:mpoly_type])

    λunit = unit(cube.λ[1])
    if !haskey(out, :plot_range)
        out[:plot_range] = nothing
    elseif length(out[:plot_range]) > 0
        punit = typeof(out[:plot_range][1][1]) <: Quantity{<:Real, u"𝐋"} ? 1.0 : λunit
        out[:plot_range] = [tuple(out[:plot_range][i].*punit...) for i in 1:length(out[:plot_range])]
        for  pair in out[:plot_range]
            @assert pair[1] < pair[2] "plot_range pairs must be in ascending order!"
        end
    end

    if !haskey(out, :user_mask)
        out[:user_mask] = nothing
    elseif length(out[:user_mask]) > 0
        punit = typeof(out[:user_mask][1][1]) <: Quantity{<:Real, u"𝐋"} ? 1.0 : λunit
        out[:user_mask] = [tuple(out[:user_mask][i].*punit...) for i in 1:length(out[:user_mask])]
        for  pair in out[:user_mask]
            @assert pair[1] < pair[2] "user_mask pairs must be in ascending order!"
        end
    end

    Itype = eltype(cube.I)
    if !haskey(out, :templates)
        out[:templates] = Array{Itype, 4}(undef, size(cube.I)..., 0)
    elseif ndims(out[:templates]) == 3
        t4 = Array{Itype, 4}(undef, size(cube.I)..., 1)
        t4[:, :, :, 1] .= out[:templates]
        out[:templates] = t4
    end
    @assert eltype(out[:templates]) == Itype "The templates must have the same intensity units as the data!"

    if !haskey(out, :template_names)
        out[:template_names] = String["template_$i" for i in axes(out[:templates], 4)]
    end

    if !haskey(out, :guess_tau)
        out[:guess_tau] = nothing
    end
    if !haskey(out, :custom_ext_template)
        out[:custom_ext_template] = nothing
    else
        out[:custom_ext_template] = Spline1D(out[:custom_ext_template][:,1], out[:custom_ext_template][:,2], k=1, bc="extrapolate")
    end

    # If we are using AGN templates, lock the hot dust component to 0
    n_templates = size(out[:templates], 4)
    if !haskey(out, :lock_hot_dust)
        out[:lock_hot_dust] = n_templates > 0
    end

    # check for F test for extinction
    if !haskey(out, :F_test_ext)
        out[:F_test_ext] = false
    end

    if !haskey(out, :nirspec_mask_chip_gaps)
        out[:nirspec_mask_chip_gaps] = false
    end

end


# Helper function to prepare extinction map and PAH template boolean map
function cubefitter_prepare_extinction_maps(out::Dict, cube::DataCube)
    # Get potential extinction map
    ebv_map = sil_abs_map = nothing
    if haskey(out, :ebv_map) && !isnothing(out[:ebv_map])
        ebv_map = out[:ebv_map]
        @assert size(ebv_map)[1:2] == size(cube.I)[1:2] "The extinction map must match the shape of the first two dimensions of the intensity map!"
        if ndims(ebv_map) == 2
            ebv_map = reshape(ebv_map, (size(ebv_map)..., 1))
        end
    end
    if haskey(out, :sil_abs_map) && !isnothing(out[:sil_abs_map])
        sil_abs_map = out[:sil_abs_map]
        @assert size(sil_abs_map)[1:2] == size(cube.I)[1:2] "The extinction map must match the shape of the first two dimensions of the intensity map!"
        if ndims(sil_abs_map) == 2
            sil_abs_map = reshape(sil_abs_map, (size(sil_abs_map)..., 1))
        end
    end
    ebv_map, sil_abs_map
end


# Helper function to prepare custom stellar templates
function cubefitter_prepare_custom_stellar_templates(out::Dict, cube::DataCube)
    temp_λ = nothing
    temp_R = nothing
    temps = nothing
    λunit = unit(cube.λ[1])
    Iunit = unit(cube.I[1])
    if haskey(out, :custom_stellar_template_wave) && !isnothing(out[:custom_stellar_template_wave])
        temp_λ = out[:custom_stellar_template_wave]
        @assert ndims(temp_λ) == 1 "Please input a 1-dimensional wavelength vector for the stellar templates"
        try
            temp_λ = uconvert.(λunit, temp_λ)
        catch
            error("Please input custom_stellar_template_wave in units that can be converted into $(λunit)")
        end
        temp_λ = ustrip.(temp_λ)
    end
    if haskey(out, :custom_stellar_template_R) && !isnothing(out[:custom_stellar_template_R])
        temp_R = out[:custom_stellar_template_R]
        @assert ndims(temp_R) == 1 "Please input a 1-dimensional resolution vector for the stellar templates"
        @assert size(temp_R, 1) == size(temp_λ, 1) "Please ensure the custom_stellar_template_R has the same length as the " *
            "custom_stellar_template_wave"
        temp_R = ustrip.(temp_R)
    end
    if haskey(out, :custom_stellar_templates) && !isnothing(out[:custom_stellar_templates])
        temps = out[:custom_stellar_templates]
        @assert ndims(temps) == 2 "Please input a 2-dimensional array for the stellar templates"
        @assert size(temps, 1) == size(temp_λ, 1) "Please ensure the first axis of custom_stellar_templates matches " *
            "the first axis of custom_stellar_template_wave"
        try
            temps = match_fluxunits.(temps, 1.0Iunit, temp_λ.*λunit)
        catch
            error("Please input custom_stellar_templates in units that can be converted into $(Iunit)" *
            " (Note: the normalization does not matter, but per-unit-frequency or per-unit-wavelength does matter)")
        end
        temps = ustrip.(temps)
    end
    temp_λ, temp_R, temps
end


# Helper function for counting the total number of emission line parameters
function count_line_parameters(lines::FitFeatures)
    length(get_flattened_fit_parameters(lines))
end


# Helper function for counting the total number of "extra" (non-fit) parameters
function count_extra_parameters(model_parameters::ModelParameters)
    length(get_flattened_nonfit_parameters(model_parameters))
end


function make_linemask!(out::Dict, lines::FitFeatures, λunit::Unitful.Units)
    # overrides for all lines in the line list with a width of +/-1000 km/s;
    # this can still be overridden with a manual line mask input
    if haskey(out, :linemask_width) && !(typeof(out[:linemask_width]) <: QVelocity)
        out[:linemask_width] *= u"km/s"
    end
    if !haskey(out, :linemask_overrides)
        overrides = Tuple[]
        for λi in lines.λ₀
            push!(overrides, λi .* (1-out[:linemask_width]/C_KMS, 1+out[:linemask_width]/C_KMS))
        end
        out[:linemask_overrides] = overrides
    else
        punit = typeof(out[:linemask_overrides][1][1]) <: Quantity{<:Real, u"𝐋"} ? 1.0 : λunit
        out[:linemask_overrides] = [tuple(out[:linemask_overrides][i].*punit...) for i in 1:length(out[:linemask_overrides])]
    end
end


"""
    split_parameters(pars, dstep, plims, plock)

Sorts parameters into vectors containing only the free parameters and the locked parameters.
Also sorts the step, limits, and lock vectors accordingly.
"""
function split_parameters(pars::Vector{<:Real}, dstep::Vector{<:Real}, plims::Vector{<:Tuple}, plock::BitVector)

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
function split_parameters(pars::Vector{<:Real}, dstep::Vector{<:Real}, plims::Vector{<:Tuple}, plock::BitVector,
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
    tied_pairs::Vector{<:Tuple}, tied_indices::Vector{<:Integer}, n_tied::Integer)
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


function strip_units(params::Vector{<:Number}, plims::Vector{<:Tuple})
    # Get the units of the parameters and strip them off of the actual parameter vector
    punits = unit.(params)
    pars_out = ustrip.(params)
    plims_out = [ustrip.(plims[i]) for i in eachindex(plims)]
    pars_out, plims_out, punits
end


function strip_units(params_1::Vector{<:Number}, params_2::Vector{<:Number}, 
    plims_1::Vector{<:Tuple}, plims_2::Vector{<:Tuple})
    pars_out_1, plims_out_1, punits_1 = strip_units(params_1, plims_1)
    pars_out_2, plims_out_2, punits_2 = strip_units(params_2, plims_2)
    pars_out_1, plims_out_1, punits_1, pars_out_2, plims_out_2, punits_2
end

