# This file contains the CLI for LOKI
using Loki
using ArgParse
using Glob

function main(_args)

    # Set up the argument parser with a bunch of different commands 
    s = ArgParseSettings("LOKI: Likelihood Optimization of gas Kinematics in IFUs")

    # Top-level command-line arguments for the different commands that can be run
    @add_arg_table! s begin
        "prepare", "prep"
            action = :command
            help = "Prepare multiple data cubes for fitting and combine them into one cube"
    end

    # Options for the 'combine' command
    @add_arg_table! s["prepare"] begin
        "label"
            required = true
            arg_type = String
            help = "A channel name for the corrected, combined data cube (the target name is already included!)"
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
    end

end


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

# If the user is runing this as the main file, do argument parsing 
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
