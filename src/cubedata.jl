module CubeData

export DataCube, Observation, from_fits, to_rest_frame, apply_mask, correct, plot_2d, plot_1d, cube_rebin!

# Import packages
using Statistics
using NaNStatistics  # => statistics functions, but ignoring NaNs
using ProgressMeter

# Import astronomy packages
using FITSIO
using Cosmology
using WCS
using Interpolations

# Plotting utilities
using ColorSchemes
using LaTeXStrings
using Reexport

# Plotting with Python
using PyPlot
plt.switch_backend("Agg")

# Python imports
# ENV["PYTHON"] = "/opt/homebrew/Caskroom/miniforge/base/envs/jwst_env/bin/python3"
# using Pkg
# Pkg.build("PyCall")

using PyCall
py_anchored_artists = pyimport("mpl_toolkits.axes_grid1.anchored_artists")

include("utils.jl")
@reexport using .Util

# MATPLOTLIB SETTINGS TO MAKE PLOTS LOOK PRETTY :)
SMALL = 12
MED = 14
BIG = 16

plt.rc("font", size=MED)          # controls default text sizes
plt.rc("axes", titlesize=MED)     # fontsize of the axes title
plt.rc("axes", labelsize=MED)     # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL)  # fontsize of the tick labels
plt.rc("legend", fontsize=MED)    # legend fontsize
plt.rc("figure", titlesize=BIG)   # fontsize of the figure title
plt.rc("text", usetex=true)
plt.rc("font", family="Times New Roman")


struct DataCube{T1<:Real,T2<:Real,S<:Int}
    """ 
    An object for holding 3D IFU spectroscopy data. 

    :param λ: array
        1D array of wavelengths, in Angstroms
    :param Iλ: array
        3D array of intensity, in erg s^-1 cm^-2 Angstrom^-1 sr^-1
    :param σI: array
        3D array of uncertainties, in erg s^-1 cm^-2 Angstrom^-1 sr^-1
    :param mask: array, optional
        3D array of booleans acting as a mask for the flux/error data
    :param wcs: WCS, optional
        a World Coordinate System conversion object
    :param Ω: float, optional
        the solid angle subtended by each spaxel, in sr/spaxel
    :param channel: Int, optional
        the channel, i.e. 4
    :param band: String, optional
        the band, i.e. 'MULTIPLE'
    :param rest_frame: Bool
        whether or not the DataCubes are in the rest-frame
    :param masked: Bool
        whether or not the DataCubes have been masked
    """

    λ::Vector{T1}
    Iλ::Array{T2,3}
    σI::Array{T2,3}
 
    mask::BitArray{3}

    Ω::T1

    α::T1
    δ::T1

    wcs::Union{WCSTransform,Nothing}

    channel::String
    band::String

    nx::S
    ny::S
    nz::S

    rest_frame::Bool
    masked::Bool

    function DataCube(λ::Vector{T1}, Iλ::Array{T2,3}, σI::Array{T2,3}, mask::Union{BitArray{3},Nothing}=nothing, 
        Ω::T1=NaN, α::T1=NaN, δ::T1=NaN, wcs::Union{WCSTransform,Nothing}=nothing, channel::String="Generic Channel", 
        band::String="Generic Band", rest_frame::Bool=false, masked::Bool=false) where {T1,T2}
        """
        A constructor for DataCube objects without keywords
        """

        # Make sure inputs have the right dimensions
        @assert ndims(λ) == 1
        @assert (ndims(Iλ) == 3) && (size(Iλ)[end] == size(λ)[1])
        @assert (ndims(σI) == 3) && (size(σI)[end] == size(λ)[1])
        nx, ny, nz = size(Iλ)

        # Make default mask
        if isnothing(mask)
            mask = falses(shape(Iλ))
        end

        return new{eltype(λ),eltype(Iλ),typeof(nx)}(λ, Iλ, σI, mask, Ω, α, δ, wcs, channel, band, nx, ny, nz, rest_frame, masked)
    end

end

# FITS Constructor 
function from_fits(filename::String)
    """
    Utility class-method for creating DataCube structures directly from JWST-formatted FITS files.
    """

    # Open the fits file
    hdu = FITS(filename)
    if read_header(hdu[1])["DATAMODL"] ≠ "IFUCubeModel"
        error("The FITS file must contain IFU data!")
    end
    
    hdr = read_header(hdu["SCI"])
    # Wavelength dimension
    nx, ny, nz = hdr["NAXIS1"], hdr["NAXIS2"], hdr["NAXIS3"]
    # Solid angle of each spaxel
    Ω = hdr["PIXAR_SR"]
    # Construct wavelength array
    λ = hdr["CRVAL3"] .+ hdr["CDELT3"] .* collect(0:nz-1)
    # Intensity and error arrays
    Iλ = read(hdu["SCI"])
    σI = read(hdu["ERR"])

    # Construct 2D World coordinate system
    wcs = WCSTransform(2)
    wcs.cdelt = [hdr["CDELT1"], hdr["CDELT2"]]
    wcs.ctype = [hdr["CTYPE1"], hdr["CTYPE2"]]
    wcs.crpix = [hdr["CRPIX1"], hdr["CRPIX2"]]
    wcs.crval = [hdr["CRVAL1"], hdr["CRVAL2"]]
    wcs.cunit = [hdr["CUNIT1"], hdr["CUNIT2"]]
    wcs.pc = [hdr["PC1_1"] hdr["PC1_2"]; hdr["PC2_1"] hdr["PC2_2"]]

    # Data quality map to mask
    dq = read(hdu["DQ"])
    mask = (dq .≠ 0) .|| .!isfinite.(Iλ) .|| .!isfinite.(σI)

    # Target info from the header
    hdr0 = read_header(hdu[1])
    name = hdr0["TARGNAME"]
    ra = hdr0["TARG_RA"]
    dec = hdr0["TARG_DEC"]
    channel = hdr0["CHANNEL"]
    band = hdr0["BAND"]

    # Check units
    if hdr["BUNIT"] ≠ "MJy/sr"
        error("Unrecognized flux unit: $(hdr["BUNIT"])")
    end
    if hdr["CUNIT3"] ≠ "um"
        error("Unrecognized wavelength unit: $(hdr["CUNIT3"])")
    end
    # Convert units
    # 1 μm = 10^4 Å
    Å = 1e4
    λ .*= Å
    # Extend wavelength into same dimensionality as I and σI
    ext_λ = Util.extend(λ, (nx,ny)) 

    # 1 Jy = 10^-23 erg s^-1 cm^-2 Hz^-1
    # 1 MJy = 10^6 Jy
    # Fνdν = Fλdλ
    # Fλ = Fν(dν/dλ) = Fν(c/λ²)
    # 1 erg s^-1 cm^-2 Å^-1 sr^-1 = Ω * 1 erg s^-1 cm^-2 Å^-1 spax^-1
    Iλ = Iλ .* 1e-7 .* Util.C_MS ./ ext_λ.^2
    σI = σI .* 1e-7 .* Util.C_MS ./ ext_λ.^2

    return DataCube(λ, Iλ, σI, mask, Ω, ra, dec, wcs, channel, band, false, false)
end

function to_rest_frame(cube::DataCube, z::Float64)
    """
    Convert a DataCube to a rest-frame DataCube

    :param cube: DataCube
        The DataCube to convert
    :param z: Float
        The redshift

    :return: new DataCube with redshift-corrected λ array
    """

    if !cube.rest_frame
        new_λ = Util.rest_frame(cube.λ, z)
        return DataCube(new_λ, cube.Iλ, cube.σI, cube.mask, cube.Ω, cube.α, cube.δ, cube.wcs, cube.channel, cube.band, true, cube.masked)
    end
    return cube

end

function apply_mask(cube::DataCube)
    """
    Apply the mask to the intensity & error arrays

    :param cube: DataCube
        The DataCube to convert
    
    :return: new DataCube with masked Iλ and σI
    """

    if !cube.masked
        Iλ = copy(cube.Iλ)
        σI = copy(cube.σI)

        Iλ[cube.mask] .= NaN
        σI[cube.mask] .= NaN

        return DataCube(cube.λ, Iλ, σI, cube.mask, cube.Ω, cube.α, cube.δ, cube.wcs, cube.channel, cube.band, cube.rest_frame, true)
    end
    return cube

end

# Plotting functions
function plot_2d(data::DataCube, fname::String; intensity::Union{Bool,Symbol}=:sr, err::Symbol=:sr, logᵢ::Union{Int,Nothing}=10,
    logₑ::Union{Int,Nothing}=nothing, colormap::Symbol=:cubehelix, space::Symbol=:wave, name::Union{String,Nothing}=nothing, 
    slice::Union{Int,Nothing}=nothing, z::Union{Float64,Nothing}=nothing, marker::Union{Tuple{T,T},Nothing} where {T<:Real}=nothing)
    """
    A plotting utility function for 2D maps of the intensity / error

    :param fname: String
        The file name of the plot to be saved.
    :param intensity: Bool, Symbol
        If 'sr', plot the intensity in flux sr^-1 units. If 'spax' or 'pix', plot in flux spax^-1 units. If false,
        do not plot the intensity.
    :param err: Bool, Symbol
        If 'sr', plot the error in flux sr^-1 units. If 'spax' or 'pix', plot in flux spax^-1 units. If false,
        do not plot the error.
    :param log_i: Float, optional
        The base of the logarithm to take for intensity data. Set to None to not take the logarithm.
    :param log_e: Float, optional
        The base of the logarithm to take for the error data. Set to None to not take the logarithm.
    :param colormap: Symbol
        Matplotlib colormap for the data.
    :param space: Symbol
        Specifies if in wavelength or frequency space, accepts 'wave', 'wavelength', 'freq', 'frequency'
    :param name: String
        Name to put in the title of the plot.
    :param slice: Int, optional
        Index along the wavelength axis to plot. If nothing, sums the data along the wavelength axis.
    :param z: Float, optional
        The redshift of the source, used to calculate the distance and thus the spatial scale in kpc.
    :param marker: Tuple, optional
        Position in (x,y) coordinates to place a marker.
    
    :return nothing:
    """

    if intensity ≠ false && err ≠ false && intensity ≠ err
        error("intensity and error should be plotted in the same units!")
    end
    
    # Copy arrays
    Iλ = copy(data.Iλ)
    σI = copy(data.σI)
    ext_λ = Util.extend(data.λ, (data.nx, data.ny))
    
    # Convert to frequency domain if necessary
    if space ∈ (:freq, :frequency, :ν)
        # Fλdλ = Fνdν  --->  Fν = Fλ|dλ/dν| = Fν(λ^2/c)
        Iλ .*= ext_λ.^2 ./ (Util.C_MS .* 1e10)
        σI .*= ext_λ.^2 ./ (Util.C_MS .* 1e10)
    elseif !(space ∈ (:wave, :wavelength, :λ))
        error("Unrecognized argument for 'space': $space")
    end

    if isnothing(slice)
        # Sum up data along wavelength dimension
        I = Util.Σ(Iλ, 3)
        σ = sqrt.(Util.Σ(σI.^2, 3))
        # Reapply masks
        I[I .≤ 0.] .= NaN
        σ[σ .≤ 0.] .= NaN
        sub = ""
    else
        # Take the wavelength slice
        I = Iλ[:, :, slice]
        σ = σI[:, :, slice]
        λᵢ = trunc(Int, data.λ[slice])
        sub = "\\lambda$λᵢ"
    end

    # Convert to spaxel units if plotting w/r/t spaxels
    if intensity ∈ (:spax, :pix) || err ∈ (:spax, :pix)
        I .*= data.Ω
        σ .*= data.Ω
    elseif intensity ≠ :sr
        error("Unrecognized intensity solid angle unit: $intensity")
    end

    # Take logarithm if specified
    if !isnothing(logₑ)
        σ .= σ ./ abs.(log(logₑ) .* I)
    end
    if !isnothing(logᵢ)
        I .= log.(I) ./ log(logᵢ)
    end

    # Format units
    unit_str = "erg\$\\,\$s\$^{-1}\\,\$cm\$^{-2}\\,\$" * 
        (isnothing(slice) ? "" : (space ∈ (:wave, :wavelength, :λ) ? "\${\\rm \\AA}^{-1}\\,\$" : "Hz\$^{-1}\\,\$")) * 
        (intensity == :sr || err == :sr ? "sr\$^{-1}\$" : "spax\$^{-1}\$")

    ax1 = ax2 = nothing

    # 1D, no NaNs/Infs
    flatI = I[isfinite.(I)]
    flatσ = σ[isfinite.(σ)]

    if !isnothing(z)
        # Get the luminosity distance given the redshift
        ΛCDM = cosmology(h=0.70, OmegaM=0.3, OmegaK=0.0)
        DL = luminosity_dist(ΛCDM, z).val
    end

    fig = plt.figure(figsize=intensity ≠ false && err ≠ false ? (12, 6) : (12,12))
    if intensity ≠ false
        # Plot intensity on a 2D map
        ax1 = fig.add_subplot(121)
        ax1.set_title(isnothing(name) ? "" : name)
        cdata = ax1.imshow(I', origin=:lower, cmap=colormap, vmin=quantile(flatI, 0.01), vmax=quantile(flatI, 0.99))
        fig.colorbar(cdata, ax=ax1, fraction=0.046, pad=0.04, 
            label=(isnothing(logᵢ) ? "" : "\$\\log_{$logᵢ}\$(") * "\$I_{$sub}\\,/\\,\$" * unit_str * (isnothing(logᵢ) ? "" : ")"))
        ax1.axis(:off)
        
        # Add scale bars for reference
        n_pix = 1/(sqrt(data.Ω) * 180/π * 3600)
        scalebar = py_anchored_artists.AnchoredSizeBar(ax1.transData, n_pix, "1\$\'\'\$", "lower left", pad=1, color=:black, 
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

    if err ≠ false
        # Plot error on a 2D map
        ax2 = fig.add_subplot(122)
        ax2.set_title(isnothing(name) ? "" : name)
        cdata = ax2.imshow(σ', origin=:lower, cmap=colormap, vmin=quantile(flatσ, 0.01), vmax=quantile(flatσ, 0.99))
        fig.colorbar(cdata, ax=ax2, fraction=0.046, pad=0.04,
            label=isnothing(logₑ) ? "\$\\sigma_{I_{$sub}}\\,/\\,\$" * unit_str : "\$\\sigma_{\\log_{$logᵢ}I_{$sub}}\$")
        ax2.axis(:off)

        # Add scale bar for reference
        n_pix = 1/(sqrt(data.Ω) * 180/π * 3600)
        scalebar = py_anchored_artists.AnchoredSizeBar(ax2.transData, n_pix, "1\$\'\'\$", "lower left", pad=1, color=:black, 
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

    # Save and close plot
    plt.savefig(fname, dpi=300, bbox_inches=:tight)
    plt.close()

end


function plot_1d(data::DataCube, fname::String; intensity::Union{Bool,Symbol}=:sr, err::Union{Bool,Symbol}=:sr, logᵢ::Union{Bool,Int}=false,
    space::Symbol=:wave, spaxel::Union{Tuple{Int,Int},Nothing}=nothing, linestyle::String="-", name::Union{String,Nothing}=nothing)
    """
    A plotting utility function for 1D spectra of individual spaxels or the full cube.
    
    :param fname: String
        The file name of the plot to be saved.
    :param intensity: Bool, Symbol
        If 'sr', plot the intensity in flux sr^-1 units. If 'spax' or 'pix', plot in flux spax^-1 units. If False,
        do not plot the intensity.
    :param err: Bool
        If 'sr', plot the error in flux sr^-1 units. If 'spax' or 'pix', plot in flux spax^-1 units. If False,
        do not plot the error.
    :param log: Int, optional
        The base of the logarithm to take for the flux/error data. Set to None to not take the logarithm.
    :param space: String
        Specifies if in wavelength or frequency space, accepts 'wave', 'wavelength', 'freq', 'frequency'
    :param spaxel: Tuple{Int,Int}, optional
        Tuple of (x,y) spaxel coordinates to plot the 1D spectrum for; otherwise sum over all spaxels.
    :param linestyle: String
        The matplotlib linestyle argument.
    :param name: String
        Name to put in the title of the plot.
    
    :return nothing:
    """
    if intensity ≠ false && err ≠ false && intensity ≠ err
        error("intensity and error should be plotted in the same units!")
    end

    # Copy arrays
    Iλ = copy(data.Iλ)
    σI = copy(data.σI)
    ext_λ = repeat(reshape(data.λ, (1,1,length(data.λ))), outer=[data.nx,data.ny,1]) 

    # Convert to frequency domain if necessary
    if space ∈ (:freq, :frequency, :ν)
        sub = "\\nu"
        # Fλdλ = Fνdν  --->  Fν = Fλ|dλ/dν| = Fν(λ^2/c)
        Iλ .*= ext_λ.^2 ./ (Util.C_MS .* 1e10)
        σI .*= ext_λ.^2 ./ (Util.C_MS .* 1e10)
        # factor of 1e10 for AA/m but factor of 1/1e12 for THz
        xval = (Util.C_MS / 100) ./ data.λ
    elseif space ∈ (:wave, :wavelength, :λ)
        sub = "\\lambda"
        # convert to um
        xval = data.λ ./ 1e4
    else
        error("Unrecognized argument for 'space': $space")
    end 

    if intensity ∈ (:spax, :pix) || err ∈ (:spax, :pix)
        Iλ .*= data.Ω
        σI .*= data.Ω
    end

    if isnothing(spaxel)
        # Sum up data along spatial dimensions
        I = Util.Σ(Iλ, (1,2))
        σ = sqrt.(Util.Σ(σI.^2, (1,2)))
        # Reapply masks
        I[I .≤ 0.] .= NaN
        σ[σ .≤ 0.] .= NaN

        # If intensity was not per spaxel, we pick up an extra spaxel/sr term that must be divided out
        if intensity == :sr
            I ./= Util.Σ(Array{Int}(.~data.mask), (1,2))
        end
        if err == :sr
            σ ./= Util.Σ(Array{Int}(.~data.mask), (1,2))
        end

    else
        # Take the spaxel
        I = Iλ[spaxel..., :]
        σ = σI[spaxel..., :]
        sub = "\\lambda"
    end

    if logᵢ ≠ false && space ∈ (:wave, :wavelength, :λ)
        σ .= σ ./ abs.(I .* log.(logᵢ))
        I .= log.(data.λ .* I) / log.(logᵢ)
    elseif logᵢ ≠ false && space ∈ (:freq, :frequency, :ν)
        σ .= σ ./ abs.(I .* log.(logᵢ)) 
        ν_Hz = (Util.C_MS .* 1e10) ./ data.λ
        I .= log.(ν_Hz .* I) / log.(logᵢ)
    end

    xunit = space ∈ (:wave, :wavelength, :λ) ? "\${\\rm \\mu m}\$" : "\${\\rm THz}\$"
    yunit = "erg\$\\,\$s\$^{-1}\\,\$cm\$^{-2}\\,\$" * 
        (space ∈ (:wave, :wavelength, :λ) && logᵢ == false ? "\${\\rm \\AA}^{-1}\\,\$" : (space ∈ (:freq, :frequency, :ν) && logᵢ == false ? "Hz\$^{-1}\\,\$" : "")) * 
        (intensity == :sr || err == :sr ? "sr\$^{-1}\$" : (intensity ∈ (:spax, :pix) || err ∈ (:spax, :pix)) && !isnothing(spaxel) ? "spax\$^{-1}\$" : "")
    yunittype = (intensity ∈ (:spax, :pix) || err ∈ (:spax, :pix)) && isnothing(spaxel) ? "F" : "I"

    # Plot formatting
    fig, ax = plt.subplots(figsize=(10,5))
    if intensity ≠ false
        ax.plot(xval, I, "k", linestyle=linestyle, label="Data")
    end
    if err ≠ false && intensity == false
        ax.plot(xval, σ, "k", linestyle=linestyle, label="\$1\\sigma\$ Error")
    end
    if intensity ≠ false && err ≠ false
        ax.fill_between(xval, I.-σ, I.+σ, color="k", alpha=0.5, label="\$1\\sigma\$ Error")
    end
    ax.set_xlabel("\$$sub\$ ($xunit)")
    if logᵢ == false
        ax.set_ylabel("\$$(yunittype)_{$sub}\$ ($yunit)")
    else
        ax.set_ylabel("\$\\log_{$logᵢ}{$sub}{$yunittype}_{$sub}\$ ($yunit)")
    end
    ax.legend(loc="upper right", frameon=false)
    ax.set_xlim(minimum(xval), maximum(xval))
    ax.set_title(isnothing(name) ? "" * (isnothing(spaxel) ? "" : "Spaxel ($(spaxel[1]),$(spaxel[2]))") : name)
    ax.tick_params(direction="in")
    plt.savefig(fname, dpi=300, bbox_inches=:tight)
    plt.close()

end


struct Observation
    """
    A struct for holding DataCube objects in different channels for the same object

    :param name: String, optional
        a label for the source / data
    :param z: Float, optional
        the redshift of the source
    :param α: Float, optional
        the right ascension of the source, in units convertible to decimal degrees
    :param δ: Float, optional
        the declination of the source, in units convertible to decimal degrees
    :param instrument: String, optional
        the instrument name, i.e. 'MIRI'
    :param detector: String, optional
        the detector name, i.e. 'MIRIFULONG'
    """

    channels::Dict{Int,DataCube}

    name::String
    z::Float64
    α::Float64
    δ::Float64
    instrument::String
    detector::String
    rest_frame::Bool
    masked::Bool

    function Observation(channels::Dict{Int,DataCube}=Dict{Int,DataCube}(), name::String="Generic Observation",
        z::Float64=NaN, α::Float64=NaN, δ::Float64=NaN, instrument::String="Generic Instrument", detector::String="Generic Detector",
        rest_frame::Bool=false, masked::Bool=false)
        """
        A constructor for Observations without keywords
        """

        return new(channels, name, z, α, δ, instrument, detector, rest_frame, masked)
    end
    
end

function from_fits(filenames::Vector{String}, z::Float64)
    """
    Create an Observation object from a series of fits files with IFU cubes in different channels.

    :param filepaths: Vector{String}
        A vector of filepaths to the FITS files
    :param z: Float
        The redshift of the object.
    """

    channels = Dict{Int,DataCube}()
    hdu = FITS(filenames[1])
    hdr = read_header(hdu[1])
    name = hdr["TARGNAME"]
    ra = hdr["TARG_RA"]
    dec = hdr["TARG_DEC"]
    inst = hdr["INSTRUME"]
    detector = hdr["DETECTOR"]
    
    for (i, filepath) ∈ enumerate(filenames)
        channels[i] = from_fits(filepath)
    end

    return Observation(channels, name, z, ra, dec, inst, detector, false, false)
end

# Convert observations to rest-frame
function to_rest_frame(obs::Observation)
    """
    Convert each wavelength channel into the rest-frame given by the redshift

    :param obs: Observation
        The Observation object to convert
    """

    new_channels = Dict{Int,DataCube}()
    for (i, chan) ∈ zip(keys(obs.channels), values(obs.channels))
        if !isnothing(chan)
            new_channels[i] = to_rest_frame(chan, obs.z)
        end
    end
    return Observation(new_channels, obs.name, obs.z, obs.α, obs.δ, obs.instrument, obs.detector, true, obs.masked)

end

function apply_mask(obs::Observation)
    """
    Apply the mask onto each intensity/error map in the observation

    :param obs: Observation
        The Observation object to mask
    """

    new_channels = Dict{Int,DataCube}()
    for (i, chan) ∈ zip(keys(obs.channels), values(obs.channels))
        if !isnothing(chan)
            new_channels[i] = apply_mask(chan)
        end
    end
    return Observation(new_channels, obs.name, obs.z, obs.α, obs.δ, obs.instrument, obs.detector, obs.rest_frame, true)

end


correct = apply_mask ∘ to_rest_frame


function cube_rebin!(obs::Observation, channels::Union{Vector{Int},Nothing}=nothing)

    # Default channels to include
    if isnothing(channels)
        channels = [1, 2, 3, 4]
    end

    # Reference channel
    ch_ref = channels[end]
    # Output grid is determined by the highest channel (lowest resolution)
    # Examples: input ch [1,2,3], output ch 4
    #           input ch [1,2],   output ch 3
    #           input ch 2,       output ch 3

    # Output wavelength is just the individual wavelength vectors concatenated
    λ_out = vcat([obs.channels[ch_i].λ for ch_i ∈ channels]...)
    I_out = zeros(size(obs.channels[ch_ref].Iλ)[1:end-1]..., size(λ_out)...)
    σ_out = copy(I_out)

    shape_ref = size(obs.channels[ch_ref].Iλ)

    # Loop through all other channels
    cumsum = 0
    for ch_in ∈ channels[1:end-1]
        wi_size = size(obs.channels[ch_in].λ)[1]

        # Function to transform output coordinates into input coordinates
        # Add in the wavelength coordinate, which doesn't change
        function pix_transform(x, y)
            coords3d = [x, y, 1.]
            cprime = world_to_pix(obs.channels[ch_in].wcs, pix_to_world(obs.channels[ch_ref].wcs, coords3d))
            return cprime[1], cprime[2]
        end

        # Convert NaNs to 0s
        ch_Iλ = obs.channels[ch_in].Iλ
        ch_Iλ[.!isfinite.(ch_Iλ)] .= 0.
        ch_σI = obs.channels[ch_in].σI
        ch_σI[.!isfinite.(ch_σI)] .= 0.

        # 2D Cubic interpolations at each wavelength bin
        prog = Progress(wi_size, dt=0.01, desc="Interpolating channel $ch_in...", showspeed=true)
        for wi ∈ 1:wi_size
            interp_func_I = extrapolate(interpolate(ch_Iλ[:, :, wi], BSpline(Cubic(Interpolations.Flat(OnGrid())))), Interpolations.Flat())
            interp_func_σ = extrapolate(interpolate(ch_σI[:, :, wi], BSpline(Cubic(Interpolations.Flat(OnGrid())))), Interpolations.Flat())
            
            for (xᵣ, yᵣ) ∈ collect(Iterators.product(1:shape_ref[1], 1:shape_ref[2]))
                xᵢ, yᵢ = pix_transform(xᵣ, yᵣ)
                I_out[xᵣ, yᵣ, cumsum+wi] = interp_func_I(xᵢ, yᵢ)
                σ_out[xᵣ, yᵣ, cumsum+wi] = interp_func_σ(xᵢ, yᵢ)
            end

            next!(prog)
        end

        cumsum += wi_size
    end

    # append the last channel in as normal with no rebinning
    wf = shape_ref[3]
    ch_Iλ = obs.channels[ch_ref].Iλ
    ch_Iλ[.!isfinite.(ch_Iλ)] .= 0.
    ch_σI = obs.channels[ch_ref].σI
    ch_σI[.!isfinite.(ch_σI)] .= 0.    

    I_out[:, :, cumsum+1:cumsum+wf] = ch_Iλ
    σ_out[:, :, cumsum+1:cumsum+wf] = ch_σI

    # deal with overlapping wavelength data -> sort wavelength vector to be monotonically increasing
    ss = sortperm(λ_out)
    λ_out = λ_out[ss]
    I_out = I_out[:, :, ss]
    σ_out = σ_out[:, :, ss]

    # apply strict masking -- only retain pixels that have data for all channels
    mask_out = falses(size(I_out))
    atol = median(I_out[I_out .> 0]) * 1e-3
    for (i, j) ∈ collect(Iterators.product(1:shape_ref[1], 1:shape_ref[2]))
        # Has to have  significant number close to 0 to make sure that an entire channel has been cut out
        # (as opposed to single-pixel noise dips)
        mask_out[i, j, :] .= sum(isapprox.(I_out[i, j, :], 0.; atol=atol)) > 100
    end

    if obs.masked
        I_out[mask_out] .= NaN
        σ_out[mask_out] .= NaN
    end

    # Define the rebinned cube as the zeroth channel (since this is not taken up by anything else)
    obs.channels[0] = DataCube(λ_out, I_out, σ_out, mask_out, 
        obs.channels[ch_ref].Ω, obs.α, obs.δ, obs.channels[ch_ref].wcs, obs.channels[ch_ref].channel, 
        obs.channels[ch_ref].band, obs.rest_frame, true)
    
    return obs.channels[0]

end

end