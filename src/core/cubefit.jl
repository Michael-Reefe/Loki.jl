#=
This file contains the CubeFitter struct and its related functions.  The CubeFitter is how
fitting is performed with Loki.
=#

############################## PARAMETER / MODEL STRUCTURES ####################################


"""
    ParamMaps{T<:Real}

A basic structure for holding parameter best-fit values and errors along with the parameters to 
keep track of where each parameter is located.

# Fields
- `data`: A 3D array holding the best-fit parameters for each spaxel.
- `err_upp`: A 3D array holding the upper uncertainties for each spaxel.
- `err_low`: A 3D array holding the lower uncertainties for each spaxel.
- `parameters`: A ModelParameters object containing the names, labels, etc. for all of the parameters
"""
struct ParamMaps

    data::Array{<:Number,3}
    err_upp::Array{<:Number,3}
    err_low::Array{<:Number,3}
    parameters::AllParameters

    function ParamMaps(model::ModelParameters, shape::Tuple{S,S})::ParamMaps where {S<:Integer}

        @debug """\n
        Creating ParamMaps struct with shape $shape
        ##############################################
        """

        # Initialize a default array of nans to be used as a placeholder for all the other arrays
        # until the actual fitting parameters are obtained
        data = ones(shape..., 0)
        mp = get_flattened_parameters(model)
        for i in eachindex(mp._parameters)
            if typeof(mp[i]) <: FitParameter
                out_type = typeof(mp[i].value)
            else
                out_type = mp[i]._type
            end
            di = ones(out_type, shape...) .* NaN
            data = cat(data, di, dims=3)
        end

        new(data, copy(data), copy(data), mp)

    end
end


function get_val(parammap::ParamMaps, pname::String)
    param = get_flattened_parameters(parammap.parameters)
    ind = findfirst(param.names .== pname)
    parammap.data[:, :, ind]
end
function get_val(parammap::ParamMaps, pnames::Vector{String})
    param = get_flattened_parameters(parammap.parameters)
    inds = [findfirst(param.names .== pname) for pname in pnames]
    parammap.data[:, :, inds]
end
get_val(parammap::ParamMaps, index::CartesianIndex, pname::String) = get_val(parammap, pname)[index]
get_val(parammap::ParamMaps, index::CartesianIndex, pnames::Vector{String}) = get_val(parammap, pnames)[index]

function get_err(parammap::ParamMaps, pname::String)
    param = get_flattened_parameters(parammap.parameters)
    ind = findfirst(param.names .== pname)
    parammap.err_upp[:, :, ind], parammap.err_low[:, :, ind]
end
function get_err(parammap::ParamMaps, pnames::Vector{String})
    param = get_flattened_parameters(parammap.parameters)
    inds = [findfirst(param.names .== pname) for pname in pnames]
    parammap.err_upp[:, :, inds], parammap.err_low[:, :, inds]
end
function get_err(parammap::ParamMaps, index::CartesianIndex, pname::String)
    upp, low = get_err(parammap, pname)
    upp[index], low[index]
end
function get_err(parammap::ParamMaps, index::CartesianIndex, pnames::Vector{String})
    upp, low = get_err(parammap, pnames)
    upp[index], low[index]
end

function get_label(parammap::ParamMaps, pname::String)
    param = get_flattened_parameters(parammap.parameters)
    ind = findfirst(param.names .== pname)
    param.labels[ind]
end
function get_label(parammap::ParamMaps, pnames::String)
    param = get_flattened_parameters(parammap.parameters)
    inds = [findfirst(param.names .== pname) for pname in pnames]
    param.labels[inds]
end


abstract type CubeModel end

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
    save_full_model::Bool
    overwrite::Bool
    track_memory::Bool
    track_convergence::Bool
    make_movies::Bool
    map_snr_thresh::Real
    sort_line_components::Union{Symbol,Nothing}
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
normalizations in each channel. By default this is false.
- `decompose_lock_column_densities`: If true and if using the "decompose" extinction profile, then the column densities for
pyroxene and forsterite (N_pyr and N_for) will be locked to their values after the initial fit to the integrated spectrum over all 
spaxels. These parameters are measured relative to olivine (N_oli) so this, in effect, locks the relative abundances of these three silicates
over the full FOV of the cube.
- `n_bootstrap`: The number of bootstrapping iterations that should be performed for each fit.
- `random_seed`: An optional number to use as a seed for the RNG utilized during bootstrapping, to ensure consistency between
- `bootstrap_use`: Determines what to output for the resulting parameter values when bootstrapping. May be :med for the 
    median or :best for the original best-fit values.
attempts.
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
    decompose_lock_column_densities::Bool
    n_bootstrap::S
    random_seed::S
    bootstrap_use::Symbol
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
struct CubeFitter{T<:Real,S<:Integer,Q<:QSIntensity,Qv<:QVelocity} 

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
    n_templates::S
    n_lines::S
    n_acomps::S

    # Total number of parameters for the continuum, dust features, lines, and extra
    n_params_cont::S
    n_params_dust::S
    n_params_lines::S
    n_params_extra::S
    n_params_total::S

    # The number of actually fit line profiles in each spaxel
    n_fit_comps::Dict{Symbol,Matrix{S}}
    
    # The cosmology to be used when necessary
    cosmology::Cosmology.AbstractCosmology

    # Line masking and sorting options
    linemask_overrides::Vector{Tuple{T,T}}
    linemask_width::Qv

    # Initial parameter vectors
    p_init_cont::Vector{T}
    p_init_line::Vector{T}
    p_init_pahtemp::Vector{T}

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
        λunit = unit(λ[1])
        Iunit = unit(cube.I[1])
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
        cubefitter_add_default_options!(cube, out)
        # Set up the extinction map and PAH template boolean map 
        ebv_map, sil_abs_map = cubefitter_prepare_extinction_maps(out, cube)
        # Set up the output directories
        cubefitter_prepare_output_directories(name, out)

        #############################################################

        @debug """\n
        Creating CubeFitter struct for $name at z=$z
        ############################################
        """

        # Create the model parameters, stellar and iron templates, and count various parameters
        model_parameters, ssps, feii, n_ssps, n_power_law, n_dust_cont, n_dust_feat, n_templates, vres =
            cubefitter_prepare_continuum(λ, z, out, λunit, Iunit, spectral_region, name, cube)
        lines, n_lines, n_acomps, n_fit_comps = cubefitter_prepare_lines(out, λunit, Iunit, cube, spectral_region)

        # Count total parameters
        n_params_cont = count_cont_parameters(model_parameters; split=out[:use_pah_templates])
        n_params_dust = count_dust_parameters(model_parameters)
        n_params_lines = count_line_parameters(lines)
        n_params_extra = count_extra_parameters(model_parameters)
        n_params_total = n_params_cont + (out[:ust_pah_templates] ? n_params_dust : 0) + n_params_lines + n_params_extra

        # Create the line mask
        make_linemask!(out, lines, λunit)

        @debug "### There are a total of $(n_params_cont) $(out[:use_pah_templates] ? "continuum" : "continuum+PAH") parameters ###"
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
            p_init_cont = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv"), ',', Float64, '\n')[:, 1]
            p_init_line = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"), ',', Float64, '\n')[:, 1]
            p_init_pahtemp = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_pahtemp.csv"), ',', Float64, '\n')[:, 1]
        end

        # Nuclear template fitting attributes
        nuc_fit_flag = BitVector([0])
        nuc_temp_amps = ones(Float64, nchannels(spectral_region))

        # Load templates into memory
        _load_dust_templates(out[:silicate_absorption], out[:extinction_curve], out[:fit_ch_abs], out[:use_pah_templates])

        # Create options structs
        output_options = OutputOptions(
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
            out[:map_snr_thresh],
            out[:sort_line_components]
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
            out[:decompose_lock_column_densities],
            out[:n_bootstrap],
            out[:random_seed],
            out[:bootstrap_use],
            out[:line_test_lines],
            out[:line_test_threshold],
            out[:plot_line_test],
            out[:lines_allow_negative],
            out[:subtract_cubic_spline]
        )

        new{typeof(z), typeof(n_params_cont), eltype(out[:templates]), typeof(vres)}(
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
            n_templates,
            n_lines,
            n_acomps,
            n_params_cont,
            n_params_dust,
            n_params_lines,
            n_params_extra,
            n_params_total,
            n_fit_comps,
            out[:cosmology],
            out[:linemask_overrides],
            out[:linemask_width],
            p_init_cont,
            p_init_line,
            p_init_pahtemp,
            nuc_fit_flag,
            nuc_temp_amps
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

    out[:line_test_lines] = [[Symbol(ln) for ln in group] for group in out[:line_test_lines]]
    out[:plot_spaxels] = Symbol(out[:plot_spaxels])
    out[:bootstrap_use] = Symbol(out[:bootstrap_use])

    λunit = unit(cube.λ[1])
    if !haskey(out, :plot_range)
        out[:plot_range] = nothing
    else
        punit = typeof(out[:plot_rangee][1][1]) <: Quantity{<:Real, u"𝐋"} ? 1.0 : λunit
        out[:plot_range] = [tuple(out[:plot_range][i].*punit...) for i in 1:length(out[:plot_range])]
        for  pair in out[:plot_range]
            @assert pair[1] < pair[2] "plot_range pairs must be in ascending order!"
        end
    end

    if !haskey(out, :user_mask)
        out[:user_mask] = nothing
    else
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
    if !haskey(out, :linemask_overrides)
        overrides = Tuple[]
        for λi in lines.λ₀
            vunit = typeof(out[:linemask_width]) <: QVelocity ? 1.0 : u"km/s"
            push!(overrides, λi .* (1-out[:linemask_width]*vunit/C_KMS, 1+out[:linemask_width]*vunit/C_KMS))
        end
        out[:linemask_overrides] = overrides
    else
        punit = typeof(out[:linemask_overrides][1][1]) <: Quantity{<:Real, u"𝐋"} ? 1.0 : λunit
        out[:linemask_overrides] = [tuple(out[:linemask_overrides][i].*punit...) for i in 1:length(out[:linemask_overrides])]
    end
end


# TODO: COME BACK TO THIS!!! 
# """
#     generate_cubemodel(cube_fitter[, aperture])

# Generate a CubeModel object corresponding to the options given by the CubeFitter object
# """
# function generate_cubemodel(cube_fitter::CubeFitter, aperture::Bool=false)
#     shape = aperture ? (1,1,size(cube_fitter.cube.I, 3)) : size(cube_fitter.cube.I)
#     # Full 3D intensity model array
#     @debug "Generating full 3D cube models"
#     if cube_fitter.spectral_region == :MIR
#         cube_model = cubemodel_empty(shape, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.names,
#             cube_fitter.abs_features.names, cube_fitter.template_names, cube_fitter.lines.names, cube_fitter.extinction_curve)
#     elseif cube_fitter.spectral_region == :OPT
#         cube_model = cubemodel_empty(shape, cube_fitter.n_ssps, cube_fitter.n_power_law, cube_fitter.lines.names, 
#             cube_fitter.template_names)
#     end
#     cube_model
# end


"""
    generate_parammaps(cube_fitter[, oneD])

Generate three ParamMaps objects (for the values and upper/lower errors) corrresponding to the options given
by the CubeFitter object.
"""
function generate_parammaps(cube_fitter::CubeFitter, oneD::Bool=false)
    shape = oneD ? (1,1,size(cube_fitter.cube.I, 3)) : size(cube_fitter.cube.I)
    ParamMaps(cube_fitter.model, shape)
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
    get_continuum_parameter_vectors(cube_fitter, I, σ; init, force_noext)

Get the continuum initial values, limits, and locked vectors for a given CubeFitter object.
"""
function get_continuum_parameter_vectors(cube_fitter::CubeFitter, I::Vector{<:QSIntensity}, σ::Vector{<:QSIntensity}; 
    init::Bool=false, force_noext::Bool=false, split::Bool=false)

    continuum = model(cube_fitter).continuum
    dust_features = model(cube_fitter).dust_features
    fopt = fit_options(cube_fitter)

    # Get the default lower/upper limits from the model object
    pnames = continuum.names
    plims = continuum.limits
    plock = continuum.locked
    df_params = get_flattened_fit_parameters(dust_features)
    df_names = df_params.names
    df_plims = df_params.limits
    df_plock = df_params.locked

    # Initial values
    # if init
    #     p₀ = get_initial_parameter_values_from_estimate(...)
    # else
    #     p₀ = get_initial_parameter_values_from_previous(...)
    # end

    # Make modifications at the individual spaxel level (this doesnt change the model object since these are copies)

    # Lock E(B-V) if an extinction map has been provided
    ext_criterion = force_noext || nanmedian(I) ≤ nanmedian(σ)
    if (!isnothing(fopt.ebv_map) && !init) || ext_criterion
        prefix = "extinction."
        pwhere = prefix .* ["E_BV", "E_BV_factor"]
        inds = fast_indexin(pwhere, pnames)
        plock[inds] .= true
    end
    if (!isnothing(fopt.sil_abs_map) && !init) || ext_criterion
        if fopt.silicate_absorption == "decompose"
            pwhere = "extinction." .* ["N_oli", "N_pyr", "N_for"]
        else
            pwhere = ["extinction.tau_9_7"]
        end
        inds = fast_indexin(pwhere, pnames)
        plock[inds] .= true
    end
    # Lock the hottest dust component if some flags have been set
    if fopt.lock_hot_dust[1] || cube_fitter.nuc_fit_flag[1]
        pwhere = "continuum.dust.1." .* ["amp", "temp"]
        inds = fast_indexin(pwhere, pnames)
        plock[inds] .= true
    end
    # Lock N_pyr and N_for if the option is given
    if fopt.silicate_absorption == "decompose" && fopt.decompose_lock_column_densities && !init
        pwhere = "extinction." .* ["N_pyr", "N_for"]
        inds = fast_indexin(pwhere, pnames)
        plock[inds] .= true
    end

    # Get the tied pairs/indices
    tied_pairs, tied_indices = get_tied_pairs(continuum)

    if split
        # add the 2 amplitudes for the PAH templates to the continuum, and return the PAH parameters separately
        pnames = [pnames; "dust_features.templates.1.amp"; "dust_features.templates.2.amp"]
        plims = [plims; (0., Inf); (0., Inf)]
        plock = BitVector([plock; 0; 0])
        plims, df_plims, plock, df_plock, tied_pairs, tied_indices
    else
        # add all of the PAH parameters
        pnames = [pnames; df_names]
        plims = [plims; df_plims]
        plock = [plock; df_plock]
        pnames, plims, plock, tied_pairs, tied_indices
    end
end


function estimate_initial_reddening(extinction_curve::String, p₀::Vector, pnames::Vector{String}, λ::QWave)
    inds = fast_indexin("extinction." .* ["E_BV", "E_BV_factor"], pnames)
    E_BV, E_BV_factor = p₀[inds]
    if extinction_curve == "ccm"
        attenuation_cardelli([λ], E_BV*E_BV_factor)[1]
    elseif extinction_curve == "calz"
        attenuation_calzetti([λ], E_BV*E_BV_factor)[1]
    else
        error("Uncrecognized extinction curve $(extinction_curve)")
    end
end


# Helper function for getting initial MIR parameters based on estimates from the data
# --- remember I is already normalized at this point!
function get_mir_continuum_initial_values_from_estimation(cube_fitter::CubeFitter, λ::Vector{<:QWave}, 
    I::Vector{<:Real}, N::QSIntensity, tau_guess::Real)

    # Get the initial parameter vector read from the config files
    fopt = fit_options(cube_fitter)
    continuum = model(cube_fitter).continuum
    p₀ = continuum.values
    pnames = continuum.names

    dust_features = model(cube_fitter).dust_features
    df_params = get_flattened_fit_parameters(dust_features)
    df_p₀ = df_params.values
    df_names = df_params.names
    df_feature_names = dust_features.config.all_feature_names

    # We need to fill in some missing values - particularly the amplitudes of different components
    inv_atten_med = 1 / estimate_initial_reddening(fopt.extinction_curve, p₀, pnames, median(λ))
    I_med = clamp(nanmedian(I), 0., Inf)
    λlim = extrema(λ)

    # SSP Masses (amplitudes at this point)
    if fopt.fit_stellar_continuum
        for i ∈ 1:cube_fitter.n_ssps
            # get the mass, age, metallicity
            prefix = "continuum.stellar_populations.$(i)."
            m_ind, a_ind, z_ind = fast_indexin(prefix .* ["mass", "age", "metallicity"], pnames)
            # evaluate the starting spectrum of the SSP
            ssp_i = [cube_fitter.ssps.templates[j](p₀[a_ind], p₀[z_ind]) for j ∈ eachindex(λ)]
            # find the wavlenegth and intensity where it peaks
            indmax = argmax(ssp_i)
            # estimate the initial reddening and apply it inversely
            inv_atten_i = 1 / estimate_initial_reddening(fopt.extinction_curve, p₀, pnames, λ[indmax])
            m_i = clamp(nanmean(I[max(indmax-5,1):min(indmax+5,length(I))]), 0., Inf) * inv_atten_i
            # adjust based on if there are other continuum components expected to be overlapping 
            if cube_fitter.n_power_law > 0
                m_i *= 0.5
            end
            if cube_fitter.n_templates > 0
                m_i *= 0.5
            end
            if fopt.fit_sil_emission
                m_i *= 0.8
            end
            p₀[m_ind] = m_i
        end
    end

    # Fe II amplitudes
    if fopt.fit_opt_na_feii
        ind = fast_indexin(["continuum.feii.na.amp"], pnames)
        p₀[ind] = 0.1 * I_med * inv_atten_med
    end
    if fopt.fit_opt_br_feii
        ind = fast_indexin(["continuum.feii.br.amp"], pnames)
        p₀[ind] = 0.1 * I_med * inv_atten_med
    end
    
    # Power laws
    for j ∈ 1:cube_fitter.n_power_law
        inds = fast_indexin("continuum.power_law.$(j)." .* ["amp", "index"], pnames)
        if p₀[inds[2]] > 0   # index > 0 means it gets brighter at longer wavelengths
            inv_atten_j = estimate_initial_reddening(fopt.extinction_curve, p₀, pnames, λlim[2])
            A_i = 0.5 * clamp(nanmean(I[end-10:end]), 0., Inf) * inv_atten_j / cube_fitter.n_power_law
        else                 # index < 0 means it gets brighter at shorter wavelengths
            inv_atten_j = estimate_initial_reddening(fopt.extinction_curve, p₀, pnames, λlim[1])
            A_i = 0.5 * clamp(nanmean(I[1:10]), 0., Inf) * inv_atten_j / cube_fitter.n_power_law
        end
        if cube_fitter.n_templates > 0
            A_i *= 0.5
        end
        p₀[inds[1]] = A_i
    end

    # Dust continuum amplitudes
    for k ∈ 1:cube_fitter.n_dust_cont
        inds = fast_indexin("continuum.dust.$(k)." .* ["amp", "temp"], pnames)
        if fopt.lock_hot_dust || cube_fitter.nuc_fit_flag[1]
            p₀[inds[1]] = 0.
            continue
        end
        T_dc = p₀[inds[2]]
        λ_dc = clamp(Wein(T_dc), λlim...)
        i_dc = argmin(x -> abs(x - λ_dc), λ)
        bb = match_fluxunits(Blackbody_ν(λ[i_dc], T_dc), N, λ[i_dc]) * uconvert(NoUnits, 9.7u"μm"/λ[i_dc])^2
        A_i = I[max(i_dc-5,1):min(i_dc+5,length(I))] * N / bb
        if cube_fitter.n_templates > 0
            A_i *= 0.5
        end
        if fopt.fit_sil_emission && T_dc > 500 
            A_i *= 0.8
        end
        p₀[inds[1]] = clamp(A_i, 0., Inf)
    end

    # Hot dust amplitude
    if fopt.fit_sil_emission
        inds = fast_indexin("continuum.hot_dust." .* ["amp", "temp", "frac", "tau_warm", "tau_cold", "sil_peak"], pnames)
        T_hd, Cf, τw, τc, peak = p₀[inds[2:end]]
        hd = silicate_emission(λ, 1.0, T_hd, Cf, τw, τc, peak)
        mhd = argmax(hd)
        A_hd = 0.2 * I[mhd] * N / hd[mhd]
        p₀[inds[1]] = clamp(A_hd, 0., Inf)
    end

    # Templates
    if fopt.fit_temp_multexp
        inds = fast_indexin("continuum.templates.amp_" .* string.(1:4), pnames)
        p₀[inds] .= 0.25
    end
    if cube_fitter.nuc_fit_flag[1]
        for tname in cube_fitter.template_names
            inds = findall(x -> contains(x, "continuum.templates.$(tname).amp_"), pnames)
            p₀[inds] ./= cube_fitter.nuc_temp_amps
        end
    end

    # Dust features
    for df ∈ df_feature_names
        ind = fast_indexin(["dust_features.$(df).amp"], df_names)
        # do a quick and dirty check around 7.7um or 11.3um to see if the PAHs are strong
        checks = [7.7, 7.7, 11.3, 11.3, 12.7, 12.7] .* u"μm"
        refs = [7.0, 10.0, 10.0, 12.0, 12.0, 13.2] .* u"μm"
        diffs = Float64[]
        for (ref, check) in zip(refs, checks)
            if (λlim[1] < check < λlim[2]) && (λlim[1] < ref < λlim[2])
                icheck = argmin(x -> abs(x - check), λ)
                iref = argmin(x -> abs(x - ref), λ)
                Icheck = nanmean(I[max(icheck-5,1):min(icheck+5,length(I))])
                Iref = nanmean(I[max(iref-5,1):min(iref+5,length(I))])
                push!(diffs, clamp(Icheck - Iref, 0., Inf))
            end
        end
        if length(diffs) > 0
            df_p₀[ind] = nanmean(diffs)
        else
            df_p₀[ind] = clamp(I_med/4, 0., Inf)
        end
    end

    # For the PAH templates
    pah_frac = repeat([clamp(nanmedian(I)/4, 0., Inf)], 2)

    p₀, pah_frac
end


# Helper function for getting initial MIR parameters based on the initial fit
function get_continuum_initial_values_from_previous(cube_fitter::CubeFitter, spaxel::CartesianIndex, pnames::Vector{String},
    I::Vector{<:Real}, N::QSIntensity, tau_guess::Real)

    fopt = fit_options(cube_fitter)

    # Set the parameters to the best parameters
    p₀ = copy(cube_fitter.p_init_cont)

    # scale all flux amplitudes by the difference in medians between the spaxel and the summed spaxels
    I_init = sumdim(cube_fitter.cube.I, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    scale = max(nanmedian(I), 1e-10) * N / nanmedian(I_init)

    # pₑ = 1 + 3cube_fitter.n_ssps + 2
    # ebv_orig = p₀[pₑ]
    # ebv_factor = p₀[pₑ+1]

    if !isnothing(fopt.ebv_map)
        @debug "Using the provided E(B-V) values from the extinction_map"
        if !isnothing(cube_fitter.cube.voronoi_bins)
            data_indices = findall(vbin -> vbin == Tuple(spaxel)[1], cube_fitter.cube.voronoi_bins)
            ebv_new = mean(fopt.ebv_map[data_indices, 1])
        else
            ebv_new = fopt.ebv_map[spaxel, 1]
        end

    end

    # Rescale to keep the continuum at a good starting point
    if cube_fitter.extinction_curve == "ccm"
        scale /= median(attenuation_cardelli(λ, ebv_new*ebv_factor_new) ./ attenuation_cardelli(λ, ebv_orig*ebv_factor))
    elseif cube_fitter.extinction_curve == "calzetti"
        scale /= median(attenuation_calzetti(λ, ebv_new*ebv_factor_new) ./ attenuation_calzetti(λ, ebv_orig*ebv_factor))
    else
        error("Unrecognized extinction curve $(cube_fitter.extinction_curve)")
    end

    # Set the new values
    p₀[pₑ] = ebv_new
    p₀[pₑ+1] = ebv_factor_new

    # SSP amplitudes
    pᵢ = 1
    for _ ∈ 1:cube_fitter.n_ssps
        p₀[pᵢ] *= scale
        pᵢ += 3
    end

    # If stellar velocities hit any limits, reset them to sensible starting values
    if (p₀[pᵢ] == continuum.stel_vel.limits[1]) || (p₀[pᵢ] == continuum.stel_vel.limits[2])
        p₀[pᵢ] = 0.
    end
    if (p₀[pᵢ+1] == continuum.stel_vdisp.limits[1]) || (p₀[pᵢ+1] == continuum.stel_vdisp.limits[2])
        p₀[pᵢ+1] = 100.
    end
    pᵢ += 2

    if cube_fitter.fit_uv_bump && cube_fitter.extinction_curve == "calzetti"
        pᵢ += 1
    end
    if cube_fitter.fit_covering_frac && cube_fitter.extinction_curve == "calzetti"
        pᵢ += 1
    end
    pᵢ += 2

    # Fe II amplitudes
    if cube_fitter.fit_opt_na_feii
        p₀[pᵢ] *= scale
        pᵢ += 3
    end
    if cube_fitter.fit_opt_br_feii
        p₀[pᵢ] *= scale
        pᵢ += 3
    end

    # Power law amplitudes
    for _ ∈ 1:cube_fitter.n_power_law
        p₀[pᵢ] *= scale
        pᵢ += 2
    end

    # Template amplitudes (not rescaled)
    if cube_fitter.fit_temp_multexp
        tamp = sum(p₀[[pᵢ,pᵢ+2,pᵢ+4,pᵢ+6]]) / 4
        for _ ∈ 1:4
            p₀[pᵢ] = tamp
            pᵢ += 2
        end
    else
        for _ ∈ 1:(cube_fitter.n_templates)
            p₀[pᵢ] = 1/cube_fitter.n_templates
            pᵢ += 1
        end
    end

    p₀

    # Set the parameters to the best parameters
    p₀ = copy(cube_fitter.p_init_cont)
    pah_frac = copy(cube_fitter.p_init_pahtemp)

    # τ_97_0 = cube_fitter.τ_guess[parse(Int, cube_fitter.cube.channel)][spaxel]
    # max_τ = cube_fitter.continuum.τ_97.limits[2]

    # scale all flux amplitudes by the difference in medians between the spaxel and the summed spaxels
    # (should be close to 1 since the sum is already normalized by the number of spaxels included anyways)
    I_init = sumdim(cube_fitter.cube.I, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    scale = max(nanmedian(I), 1e-10) * N / nanmedian(I_init)
    # max_amp = 1 / exp(-max_τ)

    # Stellar amplitude (rescaled)
    p₀[1] = p₀[1] * scale 
    pᵢ = 3

    # Dust continuum amplitudes (rescaled)
    for di ∈ 1:cube_fitter.n_dust_cont
        p₀[pᵢ] = p₀[pᵢ] * scale 
        if (cube_fitter.lock_hot_dust[1] || cube_fitter.nuc_fit_flag[1]) && isone(di)
            p₀[pᵢ] = 0.
        end
        pᵢ += 2
    end

    # Power law amplitudes (NOT rescaled)
    for _ ∈ 1:cube_fitter.n_power_law
        # p₀[pᵢ] = p₀[pᵢ] 
        pᵢ += 2
    end

    # Set optical depth based on the initial guess or the initial fit (whichever is larger)
    if cube_fitter.extinction_curve != "decompose"
        p₀[pᵢ] = max(cube_fitter.continuum.τ_97.value, p₀[pᵢ])
    end

    # Set τ_9.7 and τ_CH to 0 if the continuum is within 1 std dev of 0
    # lock_abs = false
    # if nanmedian(I) ≤ 2nanmedian(σ)
    #     lock_abs = true
    #     if cube_fitter.extinction_curve != "decompose"
    #         p₀[pᵢ] = 0.
    #         p₀[pᵢ+2] = 0.
    #     else
    #         p₀[pᵢ:pᵢ+2] .= 0.
    #         p₀[pᵢ+4] = 0.
    #     end
    # end

    # Set τ_9.7 to the guess if the guess_tau flag is set
    if !isnothing(cube_fitter.guess_tau) && (cube_fitter.extinction_curve != "decompose")
        p₀[pᵢ] = tau_guess
    end

    # Override if an extinction_map was provided
    if !isnothing(cube_fitter.extinction_map)
        @debug "Using the provided τ_9.7 values from the extinction_map"
        pₑ = [pᵢ]
        if cube_fitter.extinction_curve == "decompose"
            append!(pₑ, [pᵢ+1, pᵢ+2])
        end
        if !isnothing(cube_fitter.cube.voronoi_bins)
            data_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(spaxel)[1])
            for i in eachindex(pₑ)
                p₀[pₑ[i]] = mean(cube_fitter.extinction_map[data_indices, i])
            end
        else
            data_index = spaxel
            for i in eachindex(pₑ)
                p₀[pₑ[i]] = cube_fitter.extinction_map[data_index, i]
            end
        end
    end

    # Do not adjust absorption feature amplitudes since they are multiplicative
    pᵢ += 4 + (cube_fitter.extinction_curve == "decompose" ? 3 : 1)
    for _ ∈ 1:cube_fitter.n_abs_feat
        # if lock_abs
        #     p₀[pᵢ] = 0.
        # end
        pᵢ += 4
    end

    # Hot dust amplitude (rescaled)
    if cube_fitter.fit_sil_emission
        p₀[pᵢ] *= scale
        pᵢ += 6
    end

    # Template amplitudes (not rescaled)
    if cube_fitter.fit_temp_multexp
        tamp = sum(p₀[[pᵢ,pᵢ+2,pᵢ+4,pᵢ+6]]) / 4
        for _ ∈ 1:4
            p₀[pᵢ] = tamp
            pᵢ += 2
        end
    else
        for _ ∈ 1:(cube_fitter.n_templates*cube_fitter.n_channels)
            p₀[pᵢ] = 1/cube_fitter.n_templates
            pᵢ += 1
        end
    end

    # Dust feature amplitudes (not rescaled)
    # for i ∈ 1:cube_fitter.n_dust_feat
    #     pᵢ += 3
    #     if cube_fitter.dust_features.profiles[i] == :PearsonIV
    #         pᵢ += 2
    #     end
    # end

    p₀, pah_frac
end


"""
    get_continuum_initial_values(cube_fitter, spaxel, λ, I, σ, N, init; split)

Get the vectors of starting values and relative step sizes for the continuum fit for a given CubeFitter object. 
Again, the vector may be split up by the 2 continuum fitting steps in the MIR case.
"""
function get_continuum_initial_values(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real},
    N::Real; init::Bool=false, nuc_temp_fit::Bool=false)

    continuum = model(cube_fitter).continuum

    # Check if the cube fitter has initial fit parameters 
    if !init
        @debug "Using initial best fit continuum parameters..."
        p₀ = get_continuum_initial_values_from_previous(cube_fitter, spaxel, λ, I)
    else
        @debug "Calculating initial starting points..." 
        p₀ = get_continuum_initial_values_from_estimation(cube_fitter, λ, I)
    end

    if nuc_temp_fit
        pwhere = findall(contains("templates.", ))
        p₀[end-cube_fitter.n_templates+1:end] .= 1.0
    end

    @debug "Continuum Parameter labels: \n [" *
        join(["SSP_$(i)_mass, SSP_$(i)_age, SSP_$(i)_metallicity" for i in 1:cube_fitter.n_ssps], ", ") * 
        "stel_vel, stel_vdisp, " * 
        "E_BV, E_BV_factor, " * (cube_fitter.fit_uv_bump ? "delta_uv, " : "") *
        (cube_fitter.fit_covering_frac ? "covering_frac, " : "") * 
        (cube_fitter.fit_opt_na_feii ? "na_feii_amp, na_feii_vel, na_feii_vdisp, " : "") *
        (cube_fitter.fit_opt_br_feii ? "br_feii_amp, br_feii_vel, br_feii_vdisp, " : "") *
        join(["power_law_$(j)_amp, power_law_$(j)_index, " for j in 1:cube_fitter.n_power_law], ", ") * 
        (cube_fitter.fit_temp_multexp ? "temp_multexp_amp1, temp_multexp_ind1, temp_multexp_amp2, " * 
        "temp_multexp_ind2, temp_multexp_amp3, temp_multexp_ind3, temp_multexp_amp4, temp_multexp_ind4, " : 
        join(["$(tp)_amp_1" for tp ∈ cube_fitter.template_names], ", ")) "]"

    dstep = get_opt_continuum_step_sizes(cube_fitter) 

    @debug "Continuum Starting Values: \n $p₀"
    @debug "Continuum relative step sizes: \n $dstep"

    p₀, dstep

    continuum = cube_fitter.continuum
    n_split = cubefitter_mir_count_cont_parameters(cube_fitter.extinction_curve, cube_fitter.fit_sil_emission, 
        cube_fitter.fit_temp_multexp, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_abs_feat, 
        cube_fitter.n_templates, cube_fitter.n_channels, cube_fitter.dust_features; split=true)

    # guess optical depth from the dip in the continuum level
    tau_guess = 0.
    if !isnothing(cube_fitter.guess_tau) && (cube_fitter.extinction_curve != "decompose")
        tau_guess = guess_optical_depth(cube_fitter, λ, init)
    end

    # Check if cube fitter has initial cube
    if !isnothing(cube_fitter.p_init_cube_λ) && !init
        @debug "Using parameter cube best fit continuum parameters..."
        p₀, pah_frac = get_mir_continuum_initial_values_from_pcube(cube_fitter, spaxel, n_split)
    # Check if the cube fitter has initial fit parameters 
    elseif !init
        @debug "Using initial best fit continuum parameters..."
        p₀, pah_frac = get_mir_continuum_initial_values_from_previous(cube_fitter, spaxel, I, N, tau_guess)
    # Otherwise, we estimate the initial parameters based on the data
    else
        @debug "Calculating initial starting points..."
        p₀, pah_frac = get_mir_continuum_initial_values_from_estimation(cube_fitter, λ, I, N, tau_guess)
    end
    if force_noext
        pₑ = [3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law]
        p₀[pₑ] .= 0.
    end

    @debug "Continuum Parameter labels: \n [stellar_amp, stellar_temp, " * 
        join(["dust_continuum_amp_$i, dust_continuum_temp_$i" for i ∈ 1:cube_fitter.n_dust_cont], ", ") * 
        join(["power_law_amp_$i, power_law_index_$i" for i ∈ 1:cube_fitter.n_power_law], ", ") *
        (cube_fitter.extinction_curve == "decompose" ? ", extinction_N_oli, extinction_N_pyr, extinction_N_for" : ", extinction_tau_97") *
        ", extinction_tau_ice, extinction_tau_ch, extinction_beta, extinction_Cf, " *  
        join(["$(ab)_tau, $(ab)_mean, $(ab)_fwhm" for ab ∈ cube_fitter.abs_features.names], ", ") *
        (cube_fitter.fit_sil_emission ? ", hot_dust_amp, hot_dust_temp, hot_dust_covering_frac, hot_dust_tau_warm, hot_dust_tau_cold, hot_dust_sil_peak, " : ", ") *
        (cube_fitter.fit_temp_multexp ? "temp_multexp_amp1, temp_multexp_ind1, temp_multexp_amp2, temp_multexp_ind2, temp_multexp_amp3, temp_multexp_ind3, " * 
        "temp_multexp_amp4, temp_multexp_ind4, " : join(["$(tp)_amp_$i" for i in 1:cube_fitter.n_channels for tp ∈ cube_fitter.template_names], ", ")) *
        join(["$(df)_amp, $(df)_mean, $(df)_fwhm" * (cube_fitter.dust_features.profiles[n] == :PearsonIV ? ", $(df)_index, $(df)_cutoff" : 
            "$(df)_asym") for (n, df) ∈ enumerate(cube_fitter.dust_features.names)], ", ") * "]"
    @debug "Continuum Starting Values: \n $p₀"

    deps, dstep = get_mir_continuum_step_sizes(cube_fitter, λ)
    @debug "Continuum relative step sizes: \n $dstep"

    if !split
        p₀, dstep
    else
        # Step 1: Stellar + Dust blackbodies, 2 new amplitudes for the PAH templates, and the extinction parameters
        pars_1 = vcat(p₀[1:n_split], pah_frac)
        dstep_1 = vcat(dstep[1:n_split], [deps, deps])
        # Step 2: The PAH profile amplitudes, centers, and FWHMs
        pars_2 = p₀[(n_split+1):end]
        dstep_2 = dstep[(n_split+1):end]

        pars_1, pars_2, dstep_1, dstep_2
    end

end


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

