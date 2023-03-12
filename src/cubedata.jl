#=
THE CUBEDATA MODULE
------------------

This is the module that handles loading in and preparing 3D IFU cube data to be fit. It contains
structs for holding the 3D IFU cubes, and functions that apply certain corrections to the data, such
as shifting the wavelength to the rest frame, and masking out bad spaxels.  The first step in any
fitting sequence should be loading in the data, likely with the "from_fits" function to load from a 
Flexible Image Transfer System (FITS) file, followed by applying the two aforementioned corrections
using the "correct" function.  The data should then be handed off to the structs/functions in the 
"CubeFit" module, which handles actually fitting the cubes.
=#

module CubeData

# Export only the functions that the user may want to call
export DataCube, Observation, from_fits, to_rest_frame, apply_mask, correct, interpolate_cube!, 
    plot_2d, plot_1d, cube_combine!, cube_rebin!

# Import packages
using Statistics
using NaNStatistics  # => statistics functions, but ignoring NaNs
using ProgressMeter  # => nifty way of showing progress bars for long loops
using Printf

# Import astronomy packages
using FITSIO
using Cosmology
using WCS

# Interpolation packages
using Dierckx

# Misc utilities
using ColorSchemes
using LaTeXStrings
using Reexport

# Plotting with python
# PyCall is only needed to import an additional matplotlib package
using PyCall

# Importing it within the __init__ function is necessary so that it works after precompilation
const plt::PyObject = PyNULL()
const py_anchored_artists::PyObject = PyNULL()

const SMALL::UInt8 = 12
const MED::UInt8 = 14
const BIG::UInt8 = 16

function __init__()

    # Import pyplot
    copy!(plt, pyimport_conda("matplotlib.pyplot", "matplotlib"))
    # Import the anchored_artists package from matplotlib
    copy!(py_anchored_artists, pyimport_conda("mpl_toolkits.axes_grid1.anchored_artists", "matplotlib"))

    # MATPLOTLIB SETTINGS TO MAKE PLOTS LOOK PRETTY :)
    plt.switch_backend("Agg")         # switch to agg backend so that nothing is displayed, just saved to files
    plt.rc("font", size=MED)          # controls default text sizes
    plt.rc("axes", titlesize=MED)     # fontsize of the axes title
    plt.rc("axes", labelsize=MED)     # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MED)    # legend fontsize
    plt.rc("figure", titlesize=BIG)   # fontsize of the figure title
    plt.rc("text", usetex=true)       # use LaTeX
    plt.rc("font", family="Times New Roman")  # use Times New Roman font

end

# Import and reexport the utils functions for use all throughout the code
include("utils.jl")
@reexport using .Util


############################## DATACUBE STRUCTURE AND FUNCTIONS ####################################


""" 
    DataCube(λ, Iν, σI[, mask, Ω, α, δ, wcs, channel, band, rest_frame, masked])

An object for holding 3D IFU spectroscopy data. 

# Fields
- `λ::Vector{<:Real}`: 1D array of wavelengths, in Angstroms
- `Iν::Array{<:Real,3}`: 3D array of intensity, in MJy/sr
- `σI::Array{<:Real,3}`: 3D array of uncertainties, in MJy/sr
- `mask::BitArray{3}=falses(size(Iν))`: 3D array of booleans acting as a mask for the flux/error data
- `Ω::Real=NaN`: the solid angle subtended by each spaxel, in steradians
- `α::Real=NaN`: the right ascension of the observation, in decimal degrees
- `δ::Real=NaN`: the declination of the observation, in decimal degrees
- `wcs::Union{WCSTransform,Nothing}=nothing`: a World Coordinate System conversion object, optional
- `channel::String="Generic Channel"`: the MIRI channel of the observation, from 1-4
- `band::String="Generic Band"`: the MIRI band of the observation, i.e. 'MULTIPLE'
- `nx::Integer=size(Iν,1)`: the length of the x dimension of the cube
- `ny::Integer=size(Iν,2)`: the length of the y dimension of the cube
- `nz::Integer=size(Iν,3)`: the length of the z dimension of the cube
- `rest_frame::Bool=false`: whether or not the DataCube wavelength vector is in the rest-frame
- `masked::Bool=false`: whether or not the DataCube has been masked
"""
struct DataCube

    λ::Vector{<:Real}
    Iν::Array{<:Real,3}
    σI::Array{<:Real,3}
 
    mask::BitArray{3}

    Ω::Real
    α::Real
    δ::Real

    wcs::Union{WCSTransform,Nothing}

    channel::String
    band::String

    nx::Integer
    ny::Integer
    nz::Integer

    rest_frame::Bool
    masked::Bool

    # This is the constructor for the DataCube struct; see the DataCube docstring for details
    function DataCube(λ::Vector{<:Real}, Iν::Array{<:Real,3}, σI::Array{<:Real,3}, mask::Union{BitArray{3},Nothing}=nothing, 
        Ω::Real=NaN, α::Real=NaN, δ::Real=NaN, wcs::Union{WCSTransform,Nothing}=nothing, channel::String="Generic Channel", 
        band::String="Generic Band", rest_frame::Bool=false, masked::Bool=false)

        # Make sure inputs have the right dimensions
        @assert ndims(λ) == 1 "Wavelength vector must be 1-dimensional!"
        @assert (ndims(Iν) == 3) && (size(Iν)[end] == size(λ)[1]) "The last axis of the intensity cube must be the same length as the wavelength!"
        @assert size(Iν) == size(σI) "The intensity and error cubes must be the same size!"
        nx, ny, nz = size(Iν)

        # If no mask is given, make the default mask to be all falses (i.e. don't mask out anything)
        if isnothing(mask)
            @info "DataCube initialization: No mask was given, all spaxels will be unmasked"
            mask = falses(size(Iν))
        end

        # Return a new instance of the DataCube struct
        new(λ, Iν, σI, mask, Ω, α, δ, wcs, channel, band, nx, ny, nz, rest_frame, masked)
    end

end


"""
    from_fits(filename::String)

Utility class-method for creating DataCube structures directly from JWST-formatted FITS files.

# Arguments
- `filename::String`: the filepath to the JWST-formatted FITS file
"""
function from_fits(filename::String)::DataCube

    @info "Initializing DataCube struct from $filename"

    # Open the fits file
    hdu = FITS(filename, "r")
    if read_header(hdu[1])["DATAMODL"] ≠ "IFUCubeModel"
        error("The FITS file must contain IFU data!")
    end
    
    # Read the FITS header
    hdr = read_header(hdu["SCI"])
    # Unpack data cube dimensions
    nx, ny, nz = hdr["NAXIS1"], hdr["NAXIS2"], hdr["NAXIS3"]
    # Solid angle of each spaxel
    Ω = hdr["PIXAR_SR"]
    # Intensity and error arrays
    Iν = read(hdu["SCI"])
    σI = read(hdu["ERR"])

    @debug "FITS data dimensions: ($nx, $ny, $nz), solid angle per spaxel: $Ω"

    # Construct 3D World coordinate system to convert from pixels to (RA,Dec,wave)
    wcs = WCSTransform(3)
    wcs.cdelt = [hdr["CDELT1"], hdr["CDELT2"], hdr["CDELT3"]]
    wcs.ctype = [hdr["CTYPE1"], hdr["CTYPE2"], hdr["CTYPE3"]]
    wcs.crpix = [hdr["CRPIX1"], hdr["CRPIX2"], hdr["CRPIX3"]]
    wcs.crval = [hdr["CRVAL1"], hdr["CRVAL2"], hdr["CRVAL3"]]
    wcs.cunit = [hdr["CUNIT1"], hdr["CUNIT2"], hdr["CUNIT3"]]
    wcs.pc = [hdr["PC1_1"] hdr["PC1_2"] hdr["PC1_3"]; hdr["PC2_1"] hdr["PC2_2"] hdr["PC2_3"]; hdr["PC3_1"] hdr["PC3_2"] hdr["PC3_3"]]

    # Wavelength vector
    λ = hdr["CRVAL3"] .+ hdr["CDELT3"] .* (collect(0:hdr["NAXIS3"]-1) .+ hdr["CRPIX3"] .- 1)
    # Alternative method using the WCS directly:
    # λ = pix_to_world(wcs, Matrix(hcat(ones(nz), ones(nz), collect(1:nz))'))[3,:] ./ 1e-6

    # Data quality map (i.e. the mask)
    # dq = 0 if the data is good, > 0 if the data is bad
    dq = read(hdu["DQ"])
    # also make sure to mask any points with Inf/NaN in the intensity or error, in case they were 
    # missed by the DQ map
    mask = (dq .≠ 0) .|| .!isfinite.(Iν) .|| .!isfinite.(σI)

    # Target info from the header
    hdr0 = read_header(hdu[1])
    name = hdr0["TARGNAME"]    # name of the target
    ra = hdr0["TARG_RA"]       # right ascension in deg
    dec = hdr0["TARG_DEC"]     # declination in deg
    channel = hdr0["CHANNEL"]  # MIRI channel (1-4)
    band = hdr0["BAND"]        # MIRI band (long,med,short,multiple)

    @debug """\n
    ##################################################################
    #################### TARGET INFORMATION ##########################
    ##################################################################
    name: \t\t $name
    RA: \t\t\t $ra
    Dec: \t\t\t $dec
    MIRI Channel: \t $channel
    MIRI Band: \t\t $band
    ##################################################################
    """

    # Make sure intensity units are MegaJansky per steradian and wavelength 
    # units are microns (this is assumed in the fitting code)
    if hdr["BUNIT"] ≠ "MJy/sr"
        error("Unrecognized flux unit: $(hdr["BUNIT"])")
    end
    if hdr["CUNIT3"] ≠ "um"
        error("Unrecognized wavelength unit: $(hdr["CUNIT3"])")
    end

    @debug "Intensity units: $(hdr["BUNIT"]), Wavelength units: $(hdr["CUNIT3"])"

    DataCube(λ, Iν, σI, mask, Ω, ra, dec, wcs, channel, band, false, false)
end


"""
    to_rest_frame(cube::DataCube, z)

Convert a DataCube object's wavelength vector to the rest frame

# Arguments
- `cube::DataCube`: The DataCube object to be converted
- `z::Real`: The redshift to be used to convert to the rest frame

See also [`DataCube`](@ref), [`Util.rest_frame`](@ref)
"""
function to_rest_frame(cube::DataCube, z::Real)::DataCube

    # Only convert using redshift if it hasn't already been converted
    if !cube.rest_frame
        @debug "Converting the wavelength vector of cube with channel $(cube.channel), band $(cube.band)" *
        " to the rest frame using redshift z=$z"

        new_λ = Util.rest_frame(cube.λ, z)
        return DataCube(new_λ, cube.Iν, cube.σI, cube.mask, cube.Ω, cube.α, cube.δ, cube.wcs, cube.channel, 
            cube.band, true, cube.masked)
    end

    @info "This cube's wavelength vector has already been converted to the rest frame. Returning" *
        " the unmodified cube."
    return cube

end


"""
    apply_mask(cube::DataCube)

Apply the mask to the intensity & error arrays in the DataCube

# Arguments
- `cube::DataCube`: The DataCube to mask

See also [`DataCube`](@ref)
"""
function apply_mask(cube::DataCube)::DataCube

    # Only apply the mask if it hasn't already been applied
    if !cube.masked
        @debug "Masking the intensity and error maps of cube with channel $(cube.channel), band $(cube.band)"

        Iν = copy(cube.Iν)
        σI = copy(cube.σI)

        Iν[cube.mask] .= NaN
        σI[cube.mask] .= NaN

        return DataCube(cube.λ, Iν, σI, cube.mask, cube.Ω, cube.α, cube.δ, cube.wcs, cube.channel, cube.band, 
            cube.rest_frame, true)
    end

    @info "This cube's intensity and error maps have already been masked. Returning the" *
        " unmodified cube."
    return cube

end


"""
    interpolate_cube!(cube)

Function to interpolate bad pixels in individual spaxels.  Does not interpolate any spaxels
where more than 10% of the datapoints are bad.  Uses a wide cubic spline interpolation to
get the general shape of the continuum but not fit noise or lines.

# Arguments
- `cube::DataCube`: The DataCube object to interpolate

See also [`DataCube`](@ref)
"""
function interpolate_cube!(cube::DataCube)

    λ = cube.λ
    @debug "Interpolating NaNs in cube with channel $(cube.channel), band $(cube.band):"

    for index ∈ CartesianIndices(selectdim(cube.Iν, 3, 1))

        I = cube.Iν[index, :]
        σ = cube.σI[index, :]

        # Filter NaNs
        if sum(.!isfinite.(I) .| .!isfinite.(σ)) > (size(I, 1) / 10)
            # Keep NaNs in spaxels that are a majority NaN (i.e., we do not want to fit them)
            @debug "Too many NaNs in spaxel $index -- this spaxel will not be fit"
            continue
        end
        filt = .!isfinite.(I) .& .!isfinite.(σ)

        # Interpolate the NaNs
        if sum(filt) > 0
            @debug "NaNs found in spaxel $index -- interpolating"

            # Make sure the wavelength vector is linear, since it is assumed later in the function
            diffs = diff(λ)
            Δλ = mean(diffs[1])
            scale = length(λ) ÷ 40

            # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
            λknots = λ[scale+1]:Δλ*scale:λ[end-scale-1]
            # ONLY replace NaN values, keep the rest of the data as-is
            I[filt] .= Spline1D(λ[isfinite.(I)], I[isfinite.(I)], λknots, k=3, bc="extrapolate").(λ[filt])
            σ[filt] .= Spline1D(λ[isfinite.(σ)], σ[isfinite.(σ)], λknots, k=3, bc="extrapolate").(λ[filt])

            # Reassign data in cube structure
            cube.Iν[index, :] .= I
            cube.σI[index, :] .= σ

        end 
    end

    return
end

############################## PLOTTING FUNCTIONS ####################################


"""
    plot_2d(data, fname; <keyword arguments>)

A plotting utility function for 2D maps of the raw intensity / error

# Arguments
- `data::DataCube`: The DataCube object to plot data from
- `fname::String`: The file name of the plot to be saved
- `intensity::Bool=true`: If true, plot the intensity.
- `err::Bool=true`: If true, plot the error
- `logᵢ::Union{Integer,Nothing}=10`: The base of the logarithm to take for intensity data. 
    Set to nothing to not take the logarithm.
- `logₑ::Union{Integer,Nothing}=nothing`: The base of the logarithm to take for the error data. 
    Set to nothing to not take the logarithm.
- `colormap::Symbol=:cubehelix`: Matplotlib colormap for the data.
- `name::Union{String,Nothing}=nothing`: Name to put in the title of the plot.
- `slice::Union{Integer,Nothing}=nothing`: Index along the wavelength axis to plot. 
    If nothing, sums the data along the wavelength axis.
- `z::Union{Real,Nothing}=nothing`: The redshift of the source, used to calculate 
    the distance and thus the spatial scale in kpc.
- `marker::Union{Tuple{<:Real,<:Real},Nothing}=nothing`: Position in (x,y) coordinates to place a marker.

See also [`DataCube`](@ref), [`plot_1d`](@ref)
"""
function plot_2d(data::DataCube, fname::String; intensity::Bool=true, err::Bool=true, logᵢ::Union{Integer,Nothing}=10,
    logₑ::Union{Integer,Nothing}=nothing, colormap::Symbol=:cubehelix, name::Union{String,Nothing}=nothing, 
    slice::Union{Integer,Nothing}=nothing, z::Union{Real,Nothing}=nothing, marker::Union{Tuple{<:Real,<:Real},Nothing}=nothing)

    @debug "Plotting 2D intensity/error map for cube with channel $(cube.channel), band $(cube.band)"

    if isnothing(slice)
        # Sum up data along wavelength dimension
        I = Util.Σ(data.Iν, 3)
        σ = sqrt.(Util.Σ(data.σI.^2, 3))
        # Reapply masks
        I[I .≤ 0.] .= NaN
        σ[σ .≤ 0.] .= NaN
        sub = ""
    else
        # Take the wavelength slice
        I = data.Iν[:, :, slice]
        σ = data.σI[:, :, slice]
        λᵢ = data.λ[slice]
        sub = @sprintf "\\lambda%.2f" λᵢ
    end

    # Take logarithm if specified
    if !isnothing(logₑ)
        σ .= σ ./ abs.(log(logₑ) .* I)
    end
    if !isnothing(logᵢ)
        I .= log.(I) ./ log(logᵢ)
    end

    # Format units
    unit_str = isnothing(slice) ? L"MJy$\,$sr$^{-1}\,\mu$m" : L"MJy$\,$sr$^{-1}$"

    ax1 = ax2 = nothing

    # 1D, no NaNs/Infs
    flatI = I[isfinite.(I)]
    flatσ = σ[isfinite.(σ)]

    if !isnothing(z)
        # Get the luminosity distance given the redshift
        ΛCDM = cosmology(h=0.70, OmegaM=0.3, OmegaK=0.0)
        DL = luminosity_dist(ΛCDM, z).val
    end

    fig = plt.figure(figsize=intensity && err ? (12, 6) : (12,12))
    if intensity
        # Plot intensity on a 2D map
        ax1 = fig.add_subplot(121)
        ax1.set_title(isnothing(name) ? "" : name)
        cdata = ax1.imshow(I', origin=:lower, cmap=colormap, vmin=quantile(flatI, 0.01), vmax=quantile(flatI, 0.99))
        fig.colorbar(cdata, ax=ax1, fraction=0.046, pad=0.04, 
            label=(isnothing(logᵢ) ? "" : L"$\log_{%$logᵢ}$(") * L"$I_{%$sub}\,/\,$" * unit_str * (isnothing(logᵢ) ? "" : ")"))
        ax1.axis(:off)
        
        # Add scale bars for reference
        n_pix = 1/(sqrt(data.Ω) * 180/π * 3600)
        scalebar = py_anchored_artists.AnchoredSizeBar(ax1.transData, n_pix, L"1$''$", "lower left", pad=1, color=:black, 
            frameon=false, size_vertical=0.2)
        ax1.add_artist(scalebar)

        if !isnothing(z)
            # kpc per pixel
            scale_kpc = DL * 1000 * sqrt(data.Ω)
            # pixels per kpc
            n_pix = 1/scale_kpc
            scalebar = py_anchored_artists.AnchoredSizeBar(ax1.transData, n_pix, "1 kpc", "upper right", pad=1, color=:black,
                frameon=false, size_vertical=0.2)
            ax1.add_artist(scalebar)
        end

        if !isnothing(marker)
            ax1.plot(marker[1], marker[2], "rx", ms=5)
        end

    end

    if err
        # Plot error on a 2D map
        ax2 = fig.add_subplot(122)
        ax2.set_title(isnothing(name) ? "" : name)
        cdata = ax2.imshow(σ', origin=:lower, cmap=colormap, vmin=quantile(flatσ, 0.01), vmax=quantile(flatσ, 0.99))
        fig.colorbar(cdata, ax=ax2, fraction=0.046, pad=0.04,
            label=isnothing(logₑ) ? L"$\sigma_{I_{%$sub}}\,/\,$" * unit_str : L"$\sigma_{\log_{%$logᵢ}I_{%$sub}}$")
        ax2.axis(:off)

        # Add scale bar for reference
        n_pix = 1/(sqrt(data.Ω) * 180/π * 3600)
        scalebar = py_anchored_artists.AnchoredSizeBar(ax2.transData, n_pix, L"1$''$", "lower left", pad=1, color=:black, 
            frameon=false, size_vertical=0.2)
        ax2.add_artist(scalebar)

        if !isnothing(z)
            # kpc per pixel
            scale_kpc = DL * 1000 * sqrt(data.Ω)
            # pixels per kpc
            n_pix = 1/scale_kpc
            scalebar = py_anchored_artists.AnchoredSizeBar(ax2.transData, n_pix, "1 kpc", "upper right", pad=1, color=:black,
                frameon=false, size_vertical=0.2)
            ax2.add_artist(scalebar)
        end

        if !isnothing(marker)
            ax2.plot(marker[1], marker[2], "rx", ms=5)
        end

    end

    @debug "Saving 2D plot to $fname"

    # Save and close plot
    plt.savefig(fname, dpi=300, bbox_inches=:tight)
    plt.close()

end


"""
    plot_1d(data, fname; <keyword arguments>)

A plotting utility function for 1D spectra of individual spaxels or the full cube.

# Arguments
`S<:Integer`
- `data::DataCube`: The DataCube object to plot data from
- `fname::String`: The file name of the plot to be saved
- `intensity::Bool=true`: If true, plot the intensity
- `err::Bool=true`: If true, plot the error
- `logᵢ::Integer=false`: The base of the logarithm to take for the flux/error data. 
    Set to false to not take the logarithm.
- `spaxel::Union{Tuple{S,S}=nothing`: Tuple of (x,y) spaxel coordinates 
    to plot the 1D spectrum for; otherwise sum over all spaxels.
- `linestyle::String="-"`: The matplotlib linestyle argument.
- `name::Union{String,Nothing}=nothing`: Name to put in the title of the plot.

See also [`DataCube`](@ref), [`plot_2d`](@ref)
"""
function plot_1d(data::DataCube, fname::String; intensity::Bool=true, err::Bool=true, logᵢ::Integer=false,
    spaxel::Union{Tuple{S,S},Nothing}=nothing, linestyle::String="-", name::Union{String,Nothing}=nothing) where {S<:Integer}

    @debug "Plotting 1D intensity/error map for cube with channel $(cube.channel), band $(cube.band)"

    # Alias
    λ = data.λ
    if isnothing(spaxel)
        # Sum up data along spatial dimensions
        I = Util.Σ(data.Iν, (1,2))
        σ = sqrt.(Util.Σ(data.σI.^2, (1,2)))
        # Reapply masks
        I[I .≤ 0.] .= NaN
        σ[σ .≤ 0.] .= NaN

        # If intensity was not per spaxel, we pick up an extra spaxel/sr term that must be divided out
        I ./= Util.Σ(Array{Int}(.~data.mask), (1,2))
        σ ./= Util.Σ(Array{Int}(.~data.mask), (1,2))
    else
        # Take the spaxel
        I = data.Iν[spaxel..., :]
        σ = data.σI[spaxel..., :]
        sub = "\\lambda"
    end

    # If specified, take the logarithm of the data with the given base
    if logᵢ ≠ false
        σ .= σ ./ abs.(I .* log.(logᵢ)) 
        ν_Hz = (Util.C_MS .* 1e6) ./ data.λ
        I .= log.(ν_Hz .* I) / log.(logᵢ)
    end

    # Formatting for x and y units on the axis labels
    xunit = L"${\rm \mu m}$"
    yunit = logᵢ == false ? L"MJy$\,$sr$^{-1}$" : L"MJy$\,$sr$^{-1}\,$Hz"

    # Plot formatting
    fig, ax = plt.subplots(figsize=(10,5))
    if intensity
        ax.plot(λ, I, "k", linestyle=linestyle, label="Data")
    end
    if err && !intensity
        ax.plot(λ, σ, "k", linestyle=linestyle, label=L"$1\sigma$ Error")
    end
    if intensity && !err
        ax.fill_between(λ, I.-σ, I.+σ, color="k", alpha=0.5, label=L"$1\sigma$ Error")
    end
    ax.set_xlabel(L"$\lambda$ (%$xunit)")
    if logᵢ == false
        ax.set_ylabel(L"$I_{\nu}$ (%$yunit)")
    else
        ax.set_ylabel(L"$\log_{%$logᵢ}{\nu}{I}_{\nu}$ (%$yunit)")
    end
    ax.legend(loc="upper right", frameon=false)
    ax.set_xlim(minimum(λ), maximum(λ))
    ax.set_title(isnothing(name) ? "" * (isnothing(spaxel) ? "" : "Spaxel ($(spaxel[1]),$(spaxel[2]))") : name)
    ax.tick_params(direction="in")

    @debug "Saving 1D plot to $fname"

    plt.savefig(fname, dpi=300, bbox_inches=:tight)
    plt.close()

end


############################## OBSERVATION STRUCTURE AND FUNCTIONS ####################################


"""
    Observation([channels, name, z, α, δ, instrument, detector, rest_frame, masked])

A struct for holding DataCube objects in different channels for the same observation of some target

# Fields
- `channels::Dict{<:Integer,DataCube}=Dict{<:Integer,DataCube}()`: A dictionary containing the channel 
    numbers (1-4) as keys and the individual DataCube observations as values.
- `name::String="Generic Observation"`: The label for the observation/data
- `z::Real=NaN`: the redshift of the source
- `α::Real=NaN`: the right ascension of the source, in decimal degrees
- `δ::Real=NaN`: the declination of the source, in decimal degrees
- `instrument::String="Generic Instrument"`: the instrument name, i.e. "MIRI"
- `detector::String="Generic Detector"`: the detector name, i.e. "MIRIFULONG"
- `rest_frame::Bool=false`: whether or not the individual DataCubes have been converted to the rest frame
- `masked::Bool=false`: whether or not the individual DataCubes have been masked

See also [`DataCube`](@ref)
"""
struct Observation

    channels::Dict{<:Integer,DataCube}

    name::String
    z::Real
    α::Real
    δ::Real
    instrument::String
    detector::String
    rest_frame::Bool
    masked::Bool

    function Observation(channels::Dict{<:Integer,DataCube}=Dict{<:Integer,DataCube}(), name::String="Generic Observation",
        z::Real=NaN, α::Real=NaN, δ::Real=NaN, instrument::String="Generic Instrument", 
        detector::String="Generic Detector", rest_frame::Bool=false, masked::Bool=false)

        new(channels, name, z, α, δ, instrument, detector, rest_frame, masked)
    end
    
end


"""
    from_fits(filenames::Vector{String}, z)

Create an Observation object from a series of fits files with IFU cubes in different channels.

# Arguments
- `filenames::Vector{String}`: A vector of filepaths to the FITS files
- `z::Real`: The redshift of the object.
"""
function from_fits(filenames::Vector{String}, z::Real)::Observation


    # Grab object information from the FITS header of the first file
    channels = Dict{Int,DataCube}()
    hdu = FITS(filenames[1])
    hdr = read_header(hdu[1])
    name = hdr["TARGNAME"]
    ra = hdr["TARG_RA"]
    dec = hdr["TARG_DEC"]
    inst = hdr["INSTRUME"]
    detector = hdr["DETECTOR"]

    @debug """\n
    Initializing Observation struct for $name, with redshift z=$z
    #############################################################
    """
    
    # Loop through the files and call the individual DataCube method of the from_fits function
    for (i, filepath) ∈ enumerate(filenames)
        channels[i] = from_fits(filepath)
    end

    Observation(channels, name, z, ra, dec, inst, detector, false, false)
end


"""
    to_rest_frame(obs::Observation)

Convert each wavelength channel into the rest-frame given by the redshift

# Arguments
- `obs::Observation`: The Observation object to convert
"""
function to_rest_frame(obs::Observation)::Observation

    @debug """\n
    Converting observation of $(obs.name) to the rest frame
    #######################################################
    """

    new_channels = Dict{Int,DataCube}()
    # Loop through the channels and call the individual DataCube method of the to_rest_frame function
    for (i, chan) ∈ zip(keys(obs.channels), values(obs.channels))
        if !isnothing(chan)
            new_channels[i] = to_rest_frame(chan, obs.z)
        end
    end
    Observation(new_channels, obs.name, obs.z, obs.α, obs.δ, obs.instrument, obs.detector, true, obs.masked)

end


"""
    apply_mask(obs::Observation)

Apply the mask onto each intensity/error map in the observation

# Arguments
- `obs::Observation`: The Observation object to mask
"""
function apply_mask(obs::Observation)::Observation

    @debug """\n
    Masking out bad spaxels in observation of $(obs.name)
    #####################################################
    """

    new_channels = Dict{Int,DataCube}()
    # Loop through the channels and call the individual DataCube method of the apply_mask function
    for (i, chan) ∈ zip(keys(obs.channels), values(obs.channels))
        if !isnothing(chan)
            new_channels[i] = apply_mask(chan)
        end
    end
    Observation(new_channels, obs.name, obs.z, obs.α, obs.δ, obs.instrument, obs.detector, obs.rest_frame, true)

end


"""
    correct(obs::Observation)

A composition of the `apply_mask` and `to_rest_frame` functions for Observation objects

See [`apply_mask`](@ref) and [`to_rest_frame`](@ref)
"""
correct = apply_mask ∘ to_rest_frame


"""
    cube_combine!(obs[, channels])

Perform a 2D rebinning of the given channels such that all spaxels lie on the same grid. The grid is chosen to be
the highest channel given (or, if no channels are given, the highest channel in the obs object) since this will also
be the coarsest grid.

# Arguments
`S<:Integer`
- `obs::Observation`: The Observation object to rebin
- `channels::Union{Vector{S},Nothing}=nothing`: The list of channels to be rebinned. If nothing, rebin all channels.
- `out_grid::S=1`: Index for the channel whose grid the other channels should be rebinned to, defaults to 1 (first in the channels array).
- `out_id::S=0`: The dictionary key corresponding to the newly rebinned cube, defaults to 0.
"""
function cube_combine!(obs::Observation, channels::Union{Vector{S},Nothing}=nothing; 
    out_grid::S=1, out_id::S=0) where {S<:Integer}

    # Default channels to include
    if isnothing(channels)
        channels = [1, 2, 3, 4]
    end

    # Reference channel
    ch_ref = channels[out_grid]
    # DEPRECATED: Output grid is determined by the highest channel (lowest spatial resolution)
    # NEW: Output grid is determined by the input value, defaults to the first channel given
    # Examples: input ch [1,2,3,4], output ch 1
    #           input ch [2,3],     output ch 2
    #           input ch 2,         output ch 2

    # Output wavelength is just the individual wavelength vectors concatenated
    λ_out = vcat([obs.channels[ch_i].λ for ch_i ∈ channels]...)
    I_out = zeros(size(obs.channels[ch_ref].Iν)[1:end-1]..., size(λ_out)...)
    σ_out = copy(I_out)

    shape_ref = size(obs.channels[ch_ref].Iν)

    # Loop through all other channels
    cumsum = 0
    for ch_in ∈ channels

        @info "Rebinning $(obs.name), channel $ch_in to channel $ch_ref..."

        wi_size = size(obs.channels[ch_in].λ)[1]

        if ch_in == ch_ref
            # append this channel in as normal with no rebinning,
            # since by definition its grid is the one we rebinned to
            ch_Iν = obs.channels[ch_ref].Iν
            ch_Iν[.!isfinite.(ch_Iν)] .= 0.
            ch_σI = obs.channels[ch_ref].σI
            ch_σI[.!isfinite.(ch_σI)] .= 0.    

            I_out[:, :, cumsum+1:cumsum+wi_size] = ch_Iν
            σ_out[:, :, cumsum+1:cumsum+wi_size] = ch_σI

            cumsum += wi_size
            continue
        end

        # Function to transform output coordinates into input coordinates
        # Add in the wavelength coordinate, which doesn't change
        @inline function pix_transform(x, y)
            coords3d = [x, y, 1.]
            cprime = world_to_pix(obs.channels[ch_in].wcs, pix_to_world(obs.channels[ch_ref].wcs, coords3d))
            cprime[1], cprime[2]
        end

        # Convert NaNs to 0s
        ch_Iν = obs.channels[ch_in].Iν
        ch_Iν[.!isfinite.(ch_Iν)] .= 0.
        ch_σI = obs.channels[ch_in].σI
        ch_σI[.!isfinite.(ch_σI)] .= 0.
        shape_in = size(ch_Iν)

        # Heaviside step function
        @inline θ(x) = 0.5 * (sign(x) + 1)

        prog = Progress(wi_size, dt=0.01, showspeed=true)
        for wi ∈ 1:wi_size
            # 2D Cubic spline interpolations at each wavelength bin with flat boundary conditions
            interp_func_I = Spline2D(1:shape_in[1], 1:shape_in[2], ch_Iν[:, :, wi]; kx=3, ky=3)
            interp_func_σ = Spline2D(1:shape_in[1], 1:shape_in[2], ch_σI[:, :, wi]; kx=3, ky=3)
            # interp_func_I = extrapolate(interpolate(ch_Iν[:, :, wi], BSpline(Cubic(Interpolations.Flat(OnGrid())))), Interpolations.Flat())
            # interp_func_σ = extrapolate(interpolate(ch_σI[:, :, wi], BSpline(Cubic(Interpolations.Flat(OnGrid())))), Interpolations.Flat())
            
            # Loop through all pairs of coordinates in the refrerence grid and set the intensity and error
            # at that point to the interpolated values of the corresponding location in the input grid, given by
            # the pix_transform function
            # @inbounds for (xᵣ, yᵣ) ∈ collect(Iterators.product(1:shape_ref[1], 1:shape_ref[2]))
            @inbounds @simd for xᵣ ∈ 1:shape_ref[1]
                @simd for yᵣ ∈ 1:shape_ref[2]
                    xᵢ, yᵢ = pix_transform(xᵣ, yᵣ)
                    # Fill with zeros for any points outside the boundaries of the input data
                    inb = θ(shape_in[1] - xᵢ) * θ(xᵢ - 1) * θ(shape_in[2] - yᵢ) * θ(yᵢ - 1)
                    I_out[xᵣ, yᵣ, cumsum+wi] = inb * interp_func_I(xᵢ, yᵢ)
                    σ_out[xᵣ, yᵣ, cumsum+wi] = inb * interp_func_σ(xᵢ, yᵢ)
                end
            end
            # Iterate the progress bar
            next!(prog)
        end

        cumsum += wi_size
    end

    # find overlapping regions
    jumps = findall(diff(λ_out) .< 0.)
    # rescale each channel so the flux level is continuous
    for jump ∈ jumps
        # find the full scale of the overlapping region
        wave_left, wave_right = λ_out[jump+1], λ_out[jump]
        _, i1 = findmin(abs.(λ_out[1:jump] .- wave_left))
        _, i2 = findmin(abs.(λ_out[jump+1:end] .- wave_right))
        i2 += jump
        # get the median fluxes from both channels over the full region
        med_left = dropdims(nanmedian(I_out[:, :, i1:jump], dims=3), dims=3)
        med_right = dropdims(nanmedian(I_out[:, :, jump+1:i2], dims=3), dims=3)
        # rescale the flux in the right channel to match the left channel
        I_out[:, :, jump+1:end] .*= med_left ./ med_right
    end

    # deal with overlapping wavelength data -> sort wavelength vector to be monotonically increasing
    ss = sortperm(λ_out)
    λ_out = λ_out[ss]
    I_out = I_out[:, :, ss]
    σ_out = σ_out[:, :, ss]

    @info "Masking bins with bad data..."

    # apply strict masking -- only retain pixels that have data for all channels
    mask_out = falses(size(I_out))
    # 1e-3 from empirically testing what works well
    atol = median(I_out[I_out .> 0]) * 1e-3
    @inbounds for (i, j) ∈ collect(Iterators.product(1:shape_ref[1], 1:shape_ref[2]))
        # Has to have  significant number close to 0 to make sure that an entire channel has been cut out
        # (as opposed to single-pixel noise dips)
        mask_out[i, j, :] .= sum(isapprox.(I_out[i, j, :], 0.; atol=atol)) > 100
    end

    if obs.masked
        I_out[mask_out] .= NaN
        σ_out[mask_out] .= NaN
    end

    # Define the rebinned cube as the zeroth channel (since this is not taken up by anything else)
    obs.channels[out_id] = DataCube(λ_out, I_out, σ_out, mask_out, 
        obs.channels[ch_ref].Ω, obs.α, obs.δ, obs.channels[ch_ref].wcs, obs.channels[ch_ref].channel, 
        obs.channels[ch_ref].band, obs.rest_frame, obs.masked)
    
    @info "Done!"
    
    obs.channels[out_id]

end


"""
    cube_rebin!(obs, binsize, channel; out_id)

Perform a 2D rebinning of spaxels in a single channel onto a square grid of size (binsize)x(binsize).  The purpose of this is
to allow quicker test-runs of fits for a given cube.  The binned spaxels will have a higher S/N are coarser resolution,
which is ideal for quickly seeing how well the fitting will do for a given cube without running the full modeling (which
may be time-consuing).

# Arguments
`S<:Integer`
- `obs::Observation`: The Observation object to rebin
- `binsize::S`: The side length of the new (square) bins -- must be an integer > 1
- `channel::S`: The channel to perform the binning on
- `out_id::S=0`: The ID to assign the newly binned channel within the observation object's `channels` dictionary
"""
function cube_rebin!(obs::Observation, binsize::S, channel::S; out_id::S=0) where {S<:Integer}

    @info "Rebinning channel $channel onto a $binsize x $binsize grid"

    # Get input arrays
    λ = obs.channels[channel].λ
    I_in = obs.channels[channel].Iν
    σ_in = obs.channels[channel].σI
    mask_in = obs.channels[channel].mask

    # Get the new dimensions for the binned arrays
    dims = Tuple([Int.(cld.(size(I_in)[1:2], binsize))...; size(I_in, 3)])

    # Create empty binned arrays
    I_out = zeros(eltype(I_in), dims...)
    σ_out = zeros(eltype(σ_in), dims...)
    mask_out = falses(dims...)

    @debug "Rebinned dimensions: $dims"

    # Iterate over the binned dimensions
    for x ∈ 1:dims[1]
        for y ∈ 1:dims[2]
            # Get the axes indices for the unbinned arrays
            xmin = (x-1)*binsize+1
            xmax = min(x*binsize, size(I_in, 1))
            ymin = (y-1)*binsize+1
            ymax = min(y*binsize, size(I_in, 2))

            # @debug "($xmin:$xmax, $ymin:$ymax) --> ($x,$y)"
            @inbounds @simd for z ∈ 1:dims[3]
                # Sum fluxes within the bin
                I_out[x,y,z] = Util.Σ(I_in[xmin:xmax, ymin:ymax, z], (1,2), nan=false)
                # Sum errors in quadrature within the bin
                σ_out[x,y,z] = √(Util.Σ(σ_in[xmin:xmax, ymin:ymax, z].^2, (1,2), nan=false))
                # Add masks with binary and
                mask_out[x,y,z] = sum(mask_in[xmin:xmax, ymin:ymax, z]) > binsize^2/2
            end
        end
    end

    # Apply mask
    if obs.masked
        I_out[mask_out] .= NaN
        σ_out[mask_out] .= NaN
    end

    # New WCS (not tested)
    wcs_out = obs.channels[channel].wcs
    wcs_out.cdelt[1:2] .*= binsize
    wcs_out.crpix[1:2] .-= (0.5 - 0.5/binsize)

    # Set new binned channel
    obs.channels[out_id] = DataCube(λ, I_out, σ_out, mask_out, 
        obs.channels[channel].Ω * binsize^2, obs.α, obs.δ, obs.channels[channel].wcs, obs.channels[channel].channel,
        obs.channels[channel].band, obs.rest_frame, obs.masked)

    @info "Done!"

    obs.channels[out_id]

end

end
