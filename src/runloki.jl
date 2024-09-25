# This file contains the CLI for LOKI
using Loki
using ArgParse
using Glob
using Distributed


function main(_args)

    # Set up the argument parser with a bunch of different commands 
    s = ArgParseSettings("LOKI: Likelihood Optimization of gas Kinematics in IFUs | Note: CLI options are limited and intended for simple / quick fits")

    # Top-level command-line arguments for the different commands that can be run
    @add_arg_table! s begin
        "prepare", "prep"
            action = :command
            help = "Prepare multiple data cubes for fitting and combine them into one cube"
        "fit"
            action = :command
            help = "Fit a prepared data cube"
    end

    # Options for the 'prepare' command
    @add_arg_table! s["prepare"] begin
        "label"
            required = true
            arg_type = String
            help = "A channel name for the corrected, combined data cube (can be anything, but note the target name is already included!)"
        "redshift"
            required = true
            arg_type = Float64
            help = "The (systemic) redshift of the target"
        "cubes"
            nargs = '+'
            required = true
            arg_type = String
            help = "The data cube(s) to be combined - input individual files or a folder"
        "--make-psf", "-p"
            action = :store_true
            help = "Create a 3D PSF model for the data cube(s)"
        "--psf-spline-length", "-s"
            arg_type = Int
            default = 100
            help = "The length (in pixels) of the cubic splines used to smooth the PSF model -- " * 
                "set to 0 to disable spline fitting"
        "--adjust-wcs", "-w"
            action = :store_true
            help = "Adjust the WCS values of each data cube based on the peak brightness in overlapping regions"
        "--extract-ap", "-a"
            arg_type = Float64
            default = 0.
            help = "The size of the aperture to extract spectra from at each spaxel, in units of the PSF FWHM"
        "--min-wave"
            arg_type = Float64
            default = 0.
            help = "The minimum wavelength to allow in the output data cube"
        "--max-wave"
            arg_type = Float64
            default = Inf
            help = "The maximum wavelength to allow in the output data cube"
        "--rotate-sky"
            action = :store_true
            help = "Rotate the data cube to align with the RA/Dec axes on the sky"
        "--replace-errors"
            action = :store_true
            help = "Replace the errors in the data cube with statistical errors based on the variance in a cubic spline fit"
    end

    # Options for the 'fit' command
    @add_arg_table! s["fit"] begin
        "label"
            required = true
            arg_type = String
            help = "A label for this fitting run (can be anything)"
        "cube"
            required = true
            arg_type = String
            help = "The data cube to be fit - input a single FITS file"
        "--parallel", "-p"
            arg_type = Int
            default = 0
            help = "The number of cores to use for parallel processing"
        "--plot", "-P"
            arg_type = String
            default = "pyplot"
            help = "Which backend to use for plotting; may be 'pyplot', 'plotly', or 'both'"
        "--aperture", "-a"
            nargs = '*'
            help = "Extract and fit a spectrum from an aperture: must be followed by the parameters of the aperture " *
                " [AP_SHAPE] [RA] [DEC] [P1] [P2] [P3] where [AP_SHAPE] is one of 'circular', 'elliptical', 'rectangular', or 'all';" *
                " [RA] is in sexagesimal hour angles OR decimal degrees; [DEC] is in sexagesimal degrees OR decimal degrees;" *
                " [P1]...[P3] are dependent on [AP_SHAPE] - for circular apertures, [P1] is the radius in arcseconds, for" *
                " elliptical apertures, [P1] and [P2] are the semimajor and semiminor axes in arcseconds and [P3] is the " *
                " position angle in degrees, and for rectangular apertures, [P1] and [P2] are the width and height in arcseconds" *
                " and [P3] is the position angle in degrees; if [AP_SHAPE] is 'all', the aperture is taken to be the whole FOV"
        "--extinction", "-e"
            arg_type = String
            default = "auto"
            help = "Which dust extinction template to use; for IR fits, may be 'kvt', 'ct', 'ohm', 'd+', or 'decompose'; " *
                "for optical fits, may be 'calzetti' or 'ccm'"
        "--mixed-dust", "-m"
            action = :store_true
            help = "Change the dust geometry from a screen (default) to mixed"
        "--no-pah-templates", "-t"
            action = :store_true
            help = "Disable the usage of PAH template spectra in the fitting - the PAHs and continuum are fit together"
        "--sil-emission", "-s"
            action = :store_true
            help = "Enable the fitting of a silicate emission component (common in type 1 AGN)"
        "--ch-abs", "-c"
            action = :store_true
            help = "Enable the fitting of the CH + water ice absorption features at ~7 um"
        "--joint", "-j"
            action = :store_true
            help = "Enable joint fitting of the continuum and lines simultaneously; must also pass --no-pah-templates"
        "--ir-no-stellar", "-S"
            action = :store_true
            help = "Disable the fitting of a stellar continuum component in the IR"
        "--i-like-my-storage-space-actually", "--akhil", "-A"
            action = :store_true
            help = "Disable saving the full model as a FITS file (which can be a few GB)"
        "--global", "-g"
            action = :store_true
            help = "Force all fits to be globally optimized with simulated annealing"
        "--qso-template", "-q"
            action = :store_true
            help = "Enable the usage of a QSO template component in the fit - the data cube MUST have had a PSF model already generated" * 
                " (see the '--make-psf' option in the 'prepare' command)"
        "--sub-cubic", "-C"
            action = :store_true
            help = "When fitting line residuals, subtract a cubic spline fit to the continuum instead of the actual fit to the continuum"
        "--bootstrap", "-b"
            arg_type = Int
            default = 0
            help = "Set the number of bootstrapping iterations to estimate uncertainties"
    end


    # Parse the arguments
    args = parse_args(_args, s)

    # Get the command that was run
    cmd = args["%COMMAND%"]
    cmd_args = args[args["%COMMAND%"]]

    # Run the appropriate function
    if cmd == "prepare"
        prepare(cmd_args["label"], cmd_args["cubes"], cmd_args["redshift"], cmd_args["make-psf"], cmd_args["psf-spline-length"],
            cmd_args["adjust-wcs"], cmd_args["extract-ap"], cmd_args["min-wave"], cmd_args["max-wave"], cmd_args["rotate-sky"],
            cmd_args["replace-errors"])
    elseif cmd == "fit"
        fit(cmd_args["label"], cmd_args["cube"], cmd_args["parallel"], cmd_args["plot"], cmd_args["aperture"], cmd_args["extinction"], 
            cmd_args["mixed-dust"], cmd_args["no-pah-templates"], cmd_args["sil-emission"], cmd_args["ch-abs"], cmd_args["joint"], 
            cmd_args["ir-no-stellar"], cmd_args["i-like-my-storage-space-actually"], cmd_args["global"], cmd_args["qso-template"], 
            cmd_args["sub-cubic"], cmd_args["bootstrap"])
    end

end


"""
    prepare(label, cubes, redshift, makepsf, spline_length, adjust_wcs, extract_ap, min_wave, max_wave,
        rotate_sky, replace_errors)

A CLI function to prepare data cubes for fitting.  This function is NOT for formatting - the input data cubes must 
already have the right format to be read.  What this function does is apply various corrections and adjustments to 
the data, including shifting it to the rest frame, creating a 3D PSF model, converting air wavelengths to 
vacuum wavelengths, combining data from multiple cubes into a single cube (through reprojection), calculating
errors through a statistical analysis, etc.

"""
function prepare(label::String, cubes::Vector{String}, redshift::Float64, makepsf::Bool, spline_length::Int,
    adjust_wcs::Bool, extract_ap::Float64, min_wave::Float64, max_wave::Float64, rotate_sky::Bool, replace_errors::Bool)

    label = Symbol(label)

    files = String[]
    for cube in cubes
        if isdir(cube)
            _files = glob("*.fits", cube)
            append!(files, _files)
        else
            push!(files, cube)
        end
    end
    println("Found the following FITS files as input:")
    for file in files
        println(file)
    end

    # Create the observation objects for each cube
    obs = from_fits(files, redshift)

    # Get the channels in wavelength order
    channels = collect(keys(obs.channels))
    ch0 = []
    for channel in channels
        push!(ch0, minimum(obs.channels[channel].λ))
    end
    ss = sortperm(ch0)
    channels = channels[ss]

    if makepsf
        # Generate the PSF models
        generate_psf_model!(obs)
        # Spline fit the PSF models
        if spline_length > 0
            for channel in channels
                splinefit_psf_model!(obs.channels[channel], spline_length)
            end
        end
    end

    # Convert to the rest-frame, vacuum wavelengths, and mask out bad pixels
    correct!(obs)
    # For optical data, logarithmically rebin the wavelength vector
    if obs.spectral_region == :OPT
        log_rebin!(obs)
    end

    if length(channels) > 1
        # Combine multiple channels into one composite data cube
        combine_channels!(obs, channels, out_id=label, order=1, rescale_channels=nothing, adjust_wcs_headerinfo=adjust_wcs,
            extract_from_ap=extract_ap, min_λ=min_wave, max_λ=max_wave, rescale_all_psf=false, scale_psf_only=false)
    end

    # Rotate the data cube to align with the sky axes
    if rotate_sky
        rotate_to_sky_axes!(obs.channels[label])
    end

    # Interpolate NaNs in otherwise good spaxels
    interpolate_nans!(obs.channels[label])

    # Replace errors with statistical errors
    if replace_errors
        calculate_statistical_errors!(obs.channels[label], 20, 5, 3.0)
    end

    # Save results
    save_fits(".", obs, label)

    obs

end


"""
    fit(label, cube, parallel, plot, aperture, extinction, mixed_dust, no_pah_templates, sil_emission, 
        ch_abs, joint, ir_no_stellar, i_like_storage_space, all_global, qso_template, sub_cubic, bootstrap)

A CLI function to fit a data cube.
Note:   The CLI is meant for simple/quick use cases and does not allow for the customization of all options.
        If you want to do more advanced things, like inputting a custom template for the extinction profile,
        adjusting the automatic line masking options, or changing the line F-test options, you'll need to 
        set up your own script and input these things manually.  Though you can also change options in the
        options.toml, lines.toml, dust.toml, and optical.toml options files if those are sufficient for
        your usage. Otherwise, see the jupyter notebook in the examples/ directory and the instructions in 
        the README for a guide on how to set up your own driver file.  You can also use the contents of this 
        function itself as a guide as well.

"""
function fit(label::String, cube::String, parallel::Int, plot::String, aperture::Vector, extinction::String, mixed_dust::Bool,
    no_pah_templates::Bool, sil_emission::Bool, ch_abs::Bool, joint::Bool, ir_no_stellar::Bool, i_like_storage_space::Bool, 
    all_global::Bool, qso_template::Bool, sub_cubic::Bool, bootstrap::Int)

    if parallel > 0
        # Get the current project being used so that the other processes can be 
        # activated in the same project
        project = dirname(Base.active_project())
        procs = addprocs(parallel, exeflags=["--heap-size-hint=4G", "--project=$project"])
        # Have to do this workaround with eval and macroexpand since 'using' statements dont work
        # inside functions
        @everywhere eval(Meta.parse("using Loki"))
    end

    # Read in the data
    obs = from_fits([cube])
    ch = collect(keys(obs.channels))[1]

    # Create templates
    templates = Array{Float64, 4}(undef, size(obs.channels[ch].I)..., 0)
    template_names = String[]

    if qso_template
        nuc_temp = generate_nuclear_template(obs.channels[ch], 0.)
        templates = cat(templates, nuc_temp, dims=4)
        push!(template_names, "nuclear")
    end

    # Sanity checking
    @assert plot in ("pyplot", "plotly", "both") "plot must be one of 'pyplot', 'plotly', or 'both'"
    if extinction == "auto"
        if obs.spectral_region == :MIR
            extinction = "d+"
        else
            extinction = "calzetti"
        end
    end
    @assert extinction in ("kvt", "ct", "ohm", "d+", "decompose", "calzetti", "ccm") "extinction must be one of " * 
        "'kvt', 'ct', 'ohm', 'd+', 'decompose', 'calzetti', or 'ccm'"
    
    # Create the cubefitter object
    cube_fitter = CubeFitter(
        obs.channels[ch],
        obs.z,
        label;
        parallel=parallel > 0,
        plot_spaxels=Symbol(plot),
        extinction_curve=extinction,
        extinction_screen=!mixed_dust,
        use_pah_templates=!no_pah_templates,
        fit_sil_emission=sil_emission,
        fit_ch_abs=ch_abs,
        fit_joint=joint,
        fit_stellar_continuum=!ir_no_stellar,
        save_full_model=!i_like_storage_space,
        fit_all_global=all_global,
        templates=templates,
        template_names=template_names,
        subtract_cubic_spline=sub_cubic,
        n_bootstrap=bootstrap
    )

    # Parsing the aperture options into actual aperture objects
    ap = nothing
    if length(aperture) > 0
        ap_shape = lowercase(aperture[1])
        if ap_shape == "all"
            ap = ap_shape
        elseif ap_shape == "circular"
            ap = make_aperture(obs.channels[ch], :Circular, aperture[2], aperture[3], aperture[4], 
                auto_centroid=false, scale_psf=false)
        elseif ap_shape == "elliptical"
            ap = make_aperture(obs.channels[ch], :Elliptical, aperture[2], aperture[3], aperture[4], aperture[5], aperture[6], 
                auto_centroid=false, scale_psf=false)
        elseif ap_shape == "rectangular"
            ap = make_aperture(obs.channels[ch], :Rectangular, aperture[2], aperture[3], aperture[4], aperture[5], aperture[6],
                auto_centroid=false, scale_psf=false)
        end
    end        

    # Decide whether to do an aperture fit or a full cube fit
    if !isnothing(ap)
        fit_cube!(cube_fitter, ap)
    else
        try
            fit_cube!(cube_fitter)
        finally
            if parallel > 0
                rmprocs(procs)
            end
        end
    end

end


# If the user is runing this as the main file, do argument parsing 
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
