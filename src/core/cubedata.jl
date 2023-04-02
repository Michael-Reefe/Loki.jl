#=
This is the file that handles loading in and preparing 3D IFU cube data to be fit. It contains
structs for holding the 3D IFU cubes, and functions that apply certain corrections to the data, such
as shifting the wavelength to the rest frame, and masking out bad spaxels.  The first step in any
fitting sequence should be loading in the data, likely with the "from_fits" function to load from a 
Flexible Image Transfer System (FITS) file, followed by applying the two aforementioned corrections
using the "correct" function.  The data should then be handed off to the structs/functions in the 
"CubeFit" file, which handles actually fitting the cubes.
=#


############################## DATACUBE STRUCTURE AND FUNCTIONS ####################################


""" 
    DataCube(λ, Iν, σI[, mask, Ω, α, δ, psf_fwhm, wcs, channel, band, rest_frame, masked])

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
mutable struct DataCube

    λ::Vector{<:Real}
    Iν::Array{<:Real,3}
    σI::Array{<:Real,3}
 
    mask::BitArray{3}

    Ω::Real
    α::Real
    δ::Real
    psf_fwhm::Vector{<:Real}

    wcs::Union{PyObject,Nothing}

    channel::String
    band::String

    nx::Integer
    ny::Integer
    nz::Integer

    rest_frame::Bool
    masked::Bool

    # This is the constructor for the DataCube struct; see the DataCube docstring for details
    function DataCube(λ::Vector{<:Real}, Iν::Array{<:Real,3}, σI::Array{<:Real,3}, mask::Union{BitArray{3},Nothing}=nothing, 
        Ω::Real=NaN, α::Real=NaN, δ::Real=NaN, psf_fwhm::Union{Vector{<:Real},Nothing}=nothing, wcs::Union{PyObject,Nothing}=nothing, 
        channel::String="Generic Channel", band::String="Generic Band", rest_frame::Bool=false, masked::Bool=false)

        # Make sure inputs have the right dimensions
        @assert ndims(λ) == 1 "Wavelength vector must be 1-dimensional!"
        @assert (ndims(Iν) == 3) && (size(Iν)[end] == size(λ)[1]) "The last axis of the intensity cube must be the same length as the wavelength!"
        @assert size(Iν) == size(σI) "The intensity and error cubes must be the same size!"
        if !isnothing(psf_fwhm)
            @assert size(psf_fwhm) == size(λ) "The PSF FWHM vector should match the size of the wavelength vector!"
        end
        nx, ny, nz = size(Iν)

        # If no mask is given, make the default mask to be all falses (i.e. don't mask out anything)
        if isnothing(mask)
            @info "DataCube initialization: No mask was given, all spaxels will be unmasked"
            mask = falses(size(Iν))
        else
            @assert size(mask) == size(Iν) "The mask must be the same size as the intensity cube!"
        end

        # Return a new instance of the DataCube struct
        new(λ, Iν, σI, mask, Ω, α, δ, psf_fwhm, wcs, channel, band, nx, ny, nz, rest_frame, masked)
    end

end


"""
    from_fits(filename::String; mirror_size)

Utility class-method for creating DataCube structures directly from JWST-formatted FITS files.

# Arguments
- `filename::String`: the filepath to the JWST-formatted FITS file
- `mirror_size::Real=6.5`: size of the telescope mirror in meters, defaults to 6.5m for JWST
"""
function from_fits(filename::String; mirror_size::Real=6.5)::DataCube

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
    # wcs = py_wcs.WCS(naxis=3)
    # wcs.wcs.cdelt = [hdr["CDELT1"], hdr["CDELT2"], hdr["CDELT3"]]
    # wcs.wcs.ctype = [hdr["CTYPE1"], hdr["CTYPE2"], hdr["CTYPE3"]]
    # wcs.wcs.crpix = [hdr["CRPIX1"], hdr["CRPIX2"], hdr["CRPIX3"]]
    # wcs.wcs.crval = [hdr["CRVAL1"], hdr["CRVAL2"], hdr["CRVAL3"]]
    # wcs.wcs.cunit = [hdr["CUNIT1"], hdr["CUNIT2"], hdr["CUNIT3"]]
    # wcs.wcs.pc = [hdr["PC1_1"] hdr["PC1_2"] hdr["PC1_3"]; hdr["PC2_1"] hdr["PC2_2"] hdr["PC2_3"]; hdr["PC3_1"] hdr["PC3_2"] hdr["PC3_3"]]
    wcs = py_wcs.WCS(read_header(hdu["SCI"], String))

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

    # Get the PSF FWHM in arcseconds assuming diffraction-limited optics (theta = lambda/D)
    # Using the OBSERVED-FRAME wavelength
    # psf_fwhm = @. (λ * 1e-6 / mirror_size) * 180/π * 3600
    psf_fwhm = @. 0.033 * λ + 0.106

    @debug "Intensity units: $(hdr["BUNIT"]), Wavelength units: $(hdr["CUNIT3"])"

    DataCube(λ, Iν, σI, mask, Ω, ra, dec, psf_fwhm, wcs, channel, band, false, false)
end


"""
    to_rest_frame!(cube::DataCube, z)

Convert a DataCube object's wavelength vector to the rest frame

# Arguments
- `cube::DataCube`: The DataCube object to be converted
- `z::Real`: The redshift to be used to convert to the rest frame

See also [`DataCube`](@ref), [`rest_frame`](@ref)
"""
function to_rest_frame!(cube::DataCube, z::Real)

    # Only convert using redshift if it hasn't already been converted
    if !cube.rest_frame
        @debug "Converting the wavelength vector of cube with channel $(cube.channel), band $(cube.band)" *
        " to the rest frame using redshift z=$z"

        cube.λ = rest_frame(cube.λ, z)
        cube.rest_frame = true
    end

    cube

end


"""
    apply_mask!(cube::DataCube)

Apply the mask to the intensity & error arrays in the DataCube

# Arguments
- `cube::DataCube`: The DataCube to mask

See also [`DataCube`](@ref)
"""
function apply_mask!(cube::DataCube)

    # Only apply the mask if it hasn't already been applied
    if !cube.masked
        @debug "Masking the intensity and error maps of cube with channel $(cube.channel), band $(cube.band)"

        cube.Iν[cube.mask] .= NaN
        cube.σI[cube.mask] .= NaN
        cube.masked = true

    end

    cube

end


"""
    interpolate_nans!(cube)

Function to interpolate bad pixels in individual spaxels.  Does not interpolate any spaxels
where more than 10% of the datapoints are bad.  Uses a wide cubic spline interpolation to
get the general shape of the continuum but not fit noise or lines.

# Arguments
- `cube::DataCube`: The DataCube object to interpolate

See also [`DataCube`](@ref)
"""
function interpolate_nans!(cube::DataCube)

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
            scale = 0.025
            finite = isfinite.(I)
            offset = findfirst(λ[finite] .> (scale + λ[finite][1]))

            # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
            λknots = λ[finite][offset+1]:scale:λ[finite][end-offset-1]
            good = []
            for i ∈ 1:length(λknots)
                _, λc = findmin(abs.(λknots[i] .- λ))
                if !isnan(I[λc])
                    append!(good, [i])
                end
            end
            λknots = λknots[good]
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

    @debug "Plotting 2D intensity/error map for cube with channel $(data.channel), band $(data.band)"

    if isnothing(slice)
        # Sum up data along wavelength dimension
        I = sumdim(data.Iν, 3)
        σ = sqrt.(sumdim(data.σI.^2, 3))
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
    aperture::Union{Tuple{S,S,S,S},Nothing}=nothing, linestyle::String="-", name::Union{String,Nothing}=nothing) where {S<:Integer}

    @debug "Plotting 1D intensity/error map for cube with channel $(data.channel), band $(data.band)"

    # Alias
    λ = data.λ
    if isnothing(aperture)
        # Sum up data along spatial dimensions
        I = sumdim(data.Iν, (1,2))
        σ = sqrt.(sumdim(data.σI.^2, (1,2)))
        # Reapply masks
        I[I .≤ 0.] .= NaN
        σ[σ .≤ 0.] .= NaN

        # If intensity was not per spaxel, we pick up an extra spaxel/sr term that must be divided out
        I ./= sumdim(Array{Int}(.~data.mask), (1,2))
        σ ./= sumdim(Array{Int}(.~data.mask), (1,2))
    else
        # Take the spaxel
        I = sumdim(data.Iν[aperture[1]:aperture[2], aperture[3]:aperture[4], :], (1,2))
        σ = .√(sumdim(data.σI[aperture[1]:aperture[2], aperture[3]:aperture[4], :].^2, (1,2)))
        sub = "\\lambda"
    end

    # If specified, take the logarithm of the data with the given base
    if logᵢ ≠ false
        σ .= σ ./ abs.(I .* log.(logᵢ)) 
        ν_Hz = (C_MS .* 1e6) ./ data.λ
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
    ax.set_title(isnothing(name) ? "" * (isnothing(aperture) ? "" : "Spaxel ($(aperture[1]):$(aperture[2]),$(aperture[3]):$(aperture[4]))") : name)
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
mutable struct Observation

    channels::Dict{Any,DataCube}

    name::String
    z::Real
    α::Real
    δ::Real
    instrument::String
    detector::String
    rest_frame::Bool
    masked::Bool

    function Observation(channels::Dict{Any,DataCube}=Dict{Any,DataCube}(), name::String="Generic Observation",
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
- `mirror_size::Real=6.5`: The mirror size of the telescope in meters, default = 6.5 m for JWST
"""
function from_fits(filenames::Vector{String}, z::Real; mirror_size::Real=6.5)::Observation


    # Grab object information from the FITS header of the first file
    channels = Dict{Any,DataCube}()
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

    bands = Dict("SHORT" => "A", "MEDIUM" => "B", "LONG" => "C")
    
    # Loop through the files and call the individual DataCube method of the from_fits function
    for (i, filepath) ∈ enumerate(filenames)
        cube = from_fits(filepath; mirror_size=mirror_size)
        if cube.band == "MULTIPLE"
            channels[parse(Int, cube.channel)] = cube
            continue
        end
        channels[Symbol(bands[cube.band] * cube.channel)] = cube
    end

    Observation(channels, name, z, ra, dec, inst, detector, false, false)
end


"""
    to_rest_frame!(obs::Observation)

Convert each wavelength channel into the rest-frame given by the redshift

# Arguments
- `obs::Observation`: The Observation object to convert
"""
function to_rest_frame!(obs::Observation)

    @debug """\n
    Converting observation of $(obs.name) to the rest frame
    #######################################################
    """
    # Loop through the channels and call the individual DataCube method of the to_rest_frame function
    for k ∈ keys(obs.channels)
        to_rest_frame!(obs.channels[k], obs.z)
    end
    obs.rest_frame = true

    obs

end


"""
    apply_mask!(obs::Observation)

Apply the mask onto each intensity/error map in the observation

# Arguments
- `obs::Observation`: The Observation object to mask
"""
function apply_mask!(obs::Observation)

    @debug """\n
    Masking out bad spaxels in observation of $(obs.name)
    #####################################################
    """
    # Loop through the channels and call the individual DataCube method of the apply_mask function
    for k ∈ keys(obs.channels)
        apply_mask!(obs.channels[k])
    end
    obs.masked = true

    obs

end


"""
    correct!(obs::Observation)

A composition of the `apply_mask!` and `to_rest_frame!` functions for Observation objects

See [`apply_mask!`](@ref) and [`to_rest_frame!`](@ref)
"""
correct! = apply_mask! ∘ to_rest_frame!


"""
    adjust_wcs_alignment(obs, channels; box_size=9)

Adjust the WCS alignments of each channel such that they match.
This is performed in order to remove discontinuous jumps in the flux level when crossing between
sub-channels.
"""
function adjust_wcs_alignment!(obs::Observation, channels; box_size::Integer=11)

    @info "Aligning World Coordinate Systems for channels $channels..."

    # Prepare array of centroids for each channel
    c_coords = zeros(2*length(channels)-2, 2)
    offsets = zeros(length(channels), 2)

    k = 1
    for (i, channel) ∈ enumerate(channels)

        @assert haskey(obs.channels, channel) "Channel $channel does not exist!"

        # Get summed intensity map and WCS object for each channel
        ch_data = obs.channels[channel]
        
        # Get the continuum fluxes at the beginning and end of each channel (except the first/last)
        filter_right = ch_data.λ .> (ch_data.λ[end] - 0.1)
        filter_left = ch_data.λ .< (ch_data.λ[1] + 0.1)
        if i == 1
            filters = [filter_right]
        elseif i < length(channels)
            filters = [filter_left, filter_right]
        else
            filters = [filter_left]
        end

        for filter in filters
            data = sumdim(ch_data.Iν[:, :, filter], 3)
            err = sqrt.(sumdim(ch_data.σI[:, :, filter].^2, 3))
            wcs = ch_data.wcs
            # Find the peak brightness pixels and place boxes around them
            peak = argmax(data)
            box_half = fld(box_size, 2)
            mask = trues(size(data))
            mask[peak[1]-box_half:peak[1]+box_half, peak[2]-box_half:peak[2]+box_half] .= 0
            # Find the centroid using photutils' 2D-gaussian fitting
            # NOTE 1: Make sure to transpose the data array because of numpy's reversed axis order
            # NOTE 2: Add 1 because of python's 0-based indexing
            x_cent, y_cent = py_photutils.centroids.centroid_2dg(data', error=err', mask=mask') .+ 1

            fig, ax = plt.subplots()
            data[data .≤ 0] .= NaN
            logdata = log10.(data)
            ax.imshow(logdata', origin=:lower, cmap=:cubehelix, vmin=nanquantile(logdata, 0.01), vmax=nanquantile(logdata, 0.99))
            ax.plot(x_cent-1, y_cent-1, "rx", ms=10)
            plt.savefig(obs.name * "_centroid_$(channel)_$k.pdf", dpi=300, bbox_inches=:tight)
            plt.close()

            # Convert pixel coordinates to physical coordinates using the WCS transformation with the argument for 1-based indexing
            c_coords[k, :] .= wcs.all_pix2world([[x_cent, y_cent, 1.]], 1)[1:2]
            @debug "The centroid location of channel $channel is ($x_cent, $y_cent) pixels " *
                "or $c_coords in RA (deg), dec (deg)"
            k += 1
        end

        if i == 1
            # Do not offset the first channel
            offsets[i, :] .= [0., 0.]
        elseif i < length(channels)
            # Offset between the left edge of this channel and the right edge of the previous channel
            offsets[i, :] .= c_coords[k-2, :] .- c_coords[k-3, :]
        else
            # Offset between the left edge of the final channel and the right edge of the penultimate channel
            offsets[i, :] .= c_coords[k-1, :] .- c_coords[k-2, :]
        end
    end

    # Adjust the crval entries accordingly
    for (i, channel) ∈ enumerate(channels)
        ch_data = obs.channels[channel]
        offset = sumdim(offsets[1:i, :], 1)
        ch_data.wcs.wcs.crval = [ch_data.wcs.wcs.crval[1:2] .- offset; ch_data.wcs.wcs.crval[3]]
        @info "The centroid offset relative to channel $(channels[1]) for channel $channel is $(offset)"
    end

end


"""
    reproject_channels!(obs[, channels])

Perform a 3D interpolation of the given channels such that all spaxels lie on the same grid.

# Arguments
`S<:Integer`
- `obs::Observation`: The Observation object to rebin
- `channels=nothing`: The list of channels to be rebinned. If nothing, rebin all channels.
- `out_id=0`: The dictionary key corresponding to the newly rebinned cube, defaults to 0.
- `scrub_output=true`: Whether or not to delete the individual channels that were interpolated after assigning the new interpolated channel.
"""
function reproject_channels!(obs::Observation, channels=nothing, concat_type=:full; out_id=0, scrub_output::Bool=false,
    method=:adaptive, fudge=true)

    # Default to all 4 channels
    if isnothing(channels)
        channels = [1, 2, 3, 4]
        concat_type = :full
    end
    # If a single channel is given, interpret as interpolating all of the subchannels
    if !(typeof(channels) <: Vector)
        channels = [Symbol("A" * string(channels)), Symbol("B" * string(channels)), Symbol("C" * string(channels))]
        concat_type = :sub
    end

    # First and foremost -- adjust the WCS alignment of each channel so that they are consistent with each other
    adjust_wcs_alignment!(obs, channels; box_size=11)

    # Output angular size
    Ω_out = minimum([obs.channels[channel].Ω for channel ∈ channels])

    # Find the optimal output WCS using the reproject python package
    if concat_type == :sub
        shapes = [size(obs.channels[channel].Iν)[1:2] for channel ∈ channels]
        wcs2ds = [obs.channels[channel].wcs[0] for channel ∈ channels]
        wcs_opt, size_optimal = py_mosaicking.find_optimal_celestial_wcs([(reverse(shapes[i]), wcs2ds[i]) 
            for i ∈ 1:length(channels)], resolution=sqrt(Ω_out) * py_units.rad, projection="TAN")
        wcs_optimal = wcs_opt
        size_optimal = reverse(size_optimal)
    else
        wcs_opt = obs.channels[channels[1]].wcs
        wcs_optimal = wcs_opt[0]
        size_optimal = size(obs.channels[channels[1]].Iν)[1:2]
    end

    # Output wavelength is just the individual wavelength vectors concatenated (for now -- will be interpolated later)
    λ_out = vcat([obs.channels[ch_i].λ for ch_i ∈ channels]...)
    # Output intensity/error have the same grid as the reference channel but with 3rd axis defined by lambda_out
    I_out = zeros(size_optimal..., size(λ_out)...)
    σ_out = zeros(size(I_out))
    mask_out = falses(size(I_out))

    # Iteration variables to keep track of
    cumsum = 0
    for (i, ch_in) ∈ enumerate(channels)

        # The size of the current channel's wavelength veector
        wi_size = length(obs.channels[ch_in].λ)

        @info "Reprojecting $(obs.name) channel $ch_in onto the optimal $(size_optimal) WCS grid..."

        # Get the intensity and error arrays
        # NOTE 1: Dont forget to permute and re-permute the dimensions from the numpy row-major order back to the julia column-major order
        # NOTE 2: We need to resample the FLUX, not the intensity, because the pixel sizes may be different between the input and output images,
        #          and flux scales with the pixel size whereas intensity does not (assuming it's an extended source).
        F_in = permutedims(obs.channels[ch_in].Iν .* obs.channels[ch_in].Ω, (3,2,1))
        σF_in = permutedims(obs.channels[ch_in].σI .* obs.channels[ch_in].Ω, (3,2,1))
        mask_in = permutedims(obs.channels[ch_in].mask, (3,2,1))

        # Replace NaNs with 0s for the interpolation
        F_in[.!isfinite.(F_in)] .= 0.
        σF_in[.!isfinite.(σF_in)] .= 0.

        # Use the adaptive DeForest (2004) algorithm for resampling the cubes onto the same WCS grid
        # - Conserves flux if the option is enabled
        # - Can use the Hann or Gaussian kernels
        # - Boundary mode determines what to do when some or all of the output region is masked 
        if method == :adaptive
            F_out = permutedims(py_reproject.reproject_adaptive((F_in, obs.channels[ch_in].wcs[0]), wcs_optimal, 
                (size(F_in, 1), size_optimal[2], size_optimal[1]), conserve_flux=true, kernel="gaussian", boundary_mode="strict",
                return_footprint=false), (3,2,1))
            σF_out = permutedims(py_reproject.reproject_adaptive((σF_in, obs.channels[ch_in].wcs[0]), wcs_optimal, 
                (size(σF_in, 1), size_optimal[2], size_optimal[1]), conserve_flux=true, kernel="gaussian", boundary_mode="strict",
                return_footprint=false), (3,2,1))
        elseif method == :interp
            F_out = permutedims(py_reproject.reproject_interp((F_in, obs.channels[ch_in].wcs[0]), wcs_optimal, 
                (size(F_in, 1), size_optimal[2], size_optimal[1]), order=3, block_size=(10,10),
                return_footprint=false), (3,2,1))
            σF_out = permutedims(py_reproject.reproject_interp((σF_in, obs.channels[ch_in].wcs[0]), wcs_optimal, 
                (size(σF_in, 1), size_optimal[2], size_optimal[1]), order=3, block_size=(10,10),
                return_footprint=false), (3,2,1))
            # Flux conservation
            for wi ∈ 1:wi_size
                F_out[:, :, wi] .*= nansum(F_in[wi, :, :]) ./ nansum(F_out[:, :, wi])
                σF_out[:, :, wi] .*= nansum(σF_in[wi, :, :]) ./ nansum(σF_out[:, :, wi])
            end
        end

        # Convert back to intensity
        I_out[:, :, cumsum+1:cumsum+wi_size] .= F_out ./ Ω_out
        σ_out[:, :, cumsum+1:cumsum+wi_size] .= σF_out ./ Ω_out

        # Use nearest-neighbor interpolation for the mask since it's a binary 1 or 0
        mask_out_temp = permutedims(py_reproject.reproject_interp((mask_in, obs.channels[ch_in].wcs[0]), 
            wcs_optimal, (size(mask_in, 1), size_optimal[2], size_optimal[1]), order="nearest-neighbor", return_footprint=false), (3,2,1))

        # Set all NaNs to 1s for the mask (i.e. they are masked out)
        mask_out_temp[.!isfinite.(mask_out_temp) .| .!isfinite.(I_out[:, :, cumsum+1:cumsum+wi_size]) .| 
            .!isfinite.(σ_out[:, :, cumsum+1:cumsum+wi_size])] .= 1
        mask_out[:, :, cumsum+1:cumsum+wi_size] .= mask_out_temp

        cumsum += wi_size
    end
    
    # If an entire channel is masked out, we want to throw away all channels
    # This has to be done after the first loop so that mask_out is not overwritten to be unmasked after it has been masked
    cumsum = 0
    for (i, ch_in) ∈ enumerate(channels)
        wi_size = length(obs.channels[ch_in].λ)
        for xᵣ ∈ 1:size_optimal[1], yᵣ ∈ 1:size_optimal[2]
            if all(mask_out[xᵣ, yᵣ, cumsum+1:cumsum+wi_size])
                mask_out[xᵣ, yᵣ, :] .= 1
            end
        end
        cumsum += wi_size
    end

    # Need an additional correction (fudge) factor for overall channels
    if concat_type == :full && fudge
        # find overlapping regions
        jumps = findall(diff(λ_out) .< 0.)
        # rescale channels so the flux level is continuous
        for (i, jump) ∈ enumerate(jumps)
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
            @info "Minimum/Maximum fudge factor for channel $(i+1): ($(nanminimum(med_left./med_right)), $(nanmaximum(med_left./med_right)))"
        end
    end

    # deal with overlapping wavelength data -> sort wavelength vector to be monotonically increasing
    ss = sortperm(λ_out)
    λ_out = λ_out[ss]
    I_out = I_out[:, :, ss]
    σ_out = σ_out[:, :, ss]
    mask_out = mask_out[:, :, ss]

    # Now we interpolate the wavelength dimension using a flux-conserving approach
    Δλ = median(diff(λ_out))
    if concat_type == :sub
        @info "Resampling wavelength onto a uniform, monotonic grid"
        λ_lin = collect(λ_out[1]:Δλ:λ_out[end])
        I_out, σ_out, mask_out = resample_conserving_flux(λ_lin, λ_out, I_out, σ_out, mask_out)
        λ_out = λ_lin
    else
        @warn "The wavelength dimension will not be resampled to be linear when concatenating multiple full channels! " *
              "Caution: as a result, the WCS's wavelength axis will not be accurate on the output cube!"
    end
    # New WCS object should be the same as the optimal wcs except in the wavelength dimension
    λ_init = λ_out[1]
    λ_diff = Δλ
    if obs.rest_frame
        λ_init = observed_frame(λ_out[1], obs.z)
        λ_diff = observed_frame(Δλ, obs.z)
    end
    wcs_out = py_wcs.WCS(naxis=3)
    wcs_out.wcs.ctype = [wcs_opt.wcs.ctype[1]; wcs_opt.wcs.ctype[2]; "WAVE"]
    wcs_out.wcs.cunit = [wcs_opt.wcs.cunit[1]; wcs_opt.wcs.cunit[2]; "m"]
    wcs_out.wcs.crpix = [wcs_opt.wcs.crpix[1:2]; 1.]
    wcs_out.wcs.crval = [wcs_opt.wcs.crval[1:2]; λ_init * 1e-6]
    wcs_out.wcs.cdelt = [wcs_opt.wcs.cdelt[1:2]; λ_diff * 1e-6]
    wcs_out.wcs.pc = [wcs_opt.wcs.pc[1,1] wcs_opt.wcs.pc[1,2] 0;
                      wcs_opt.wcs.pc[2,1] wcs_opt.wcs.pc[2,2] 0;
                      0                       0                       1]
    wcs_out._naxis = [wcs_opt._naxis[1:2]; length(λ_out)]

    # New PSF FWHM vector from the wavelength vector
    psf_fwhm_out = @. 0.033 * λ_out + 0.106

    if obs.masked
        @info "Masking bins with bad data..."
        I_out[mask_out] .= NaN
        σ_out[mask_out] .= NaN
    end

    # Define the interpolated cube as the zeroth channel (since this is not taken up by anything else)
    obs.channels[out_id] = DataCube(λ_out, I_out, σ_out, mask_out, Ω_out, obs.α, obs.δ, psf_fwhm_out, wcs_out, 
        "MULTIPLE", "MULTIPLE", obs.rest_frame, obs.masked)
    
    # Delete all of the individual channels
    if scrub_output
        @info "Deleting individual channels that are no longer needed"
        for channel ∈ channels
            delete!(obs.channels, channel)
        end
    end

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
                I_out[x,y,z] = sumdim(I_in[xmin:xmax, ymin:ymax, z], (1,2), nan=false)
                # Sum errors in quadrature within the bin
                σ_out[x,y,z] = √(sumdim(σ_in[xmin:xmax, ymin:ymax, z].^2, (1,2), nan=false))
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
    wcs_out.wcs.cdelt[1:2] .*= binsize
    wcs_out.wcs.crpix[1:2] .-= (0.5 - 0.5/binsize)

    # Set new binned channel
    obs.channels[out_id] = DataCube(λ, I_out, σ_out, mask_out, 
        obs.channels[channel].Ω * binsize^2, obs.α, obs.δ, obs.channels[channel].psf_fwhm, obs.channels[channel].wcs, 
        obs.channels[channel].channel, obs.channels[channel].band, obs.rest_frame, obs.masked)

    @info "Done!"

    obs.channels[out_id]

end


"""
    psf_kernel(cube; psf_scale, kernel_size)
    
Compute the kernel that will be convolved with the IFU cube to extract individual spaxels while still respecting
the PSF FWHM of the observations.
"""
function psf_kernel(cube::DataCube; aperture_scale::Real=1., kernel_size::Union{Integer,Nothing}=nothing, kernel_type::Symbol=:Tophat)

    @debug "Creating the PSF kernel..."

    # Make sure kernel size is odd
    if !isnothing(kernel_size)
        @assert kernel_size % 2 == 1 "The kernel size must be an odd integer!"
    end

    # Pixel scale in arcseconds
    pix_as = sqrt(cube.Ω) * 180/π * 3600
    # Find the PSF FWHM distance in pixel units (cube value is in arcseconds)
    psf_fwhm_pix = cube.psf_fwhm ./ pix_as
    # ap_pix is the HWHM of the aperture being used (or in the tophat case, the radius)
    # aperture_scale is essentially a scaling factor for how much larger the aperture should be compared to the PSF FWHM
    ap_pix = aperture_scale .* psf_fwhm_pix

    @debug "PSF FWHM in arcseconds: $(cube.psf_fwhm[1]) at $(cube.λ[1]) microns"
    @debug "PSF FWHM in pixels: $(psf_fwhm_pix[1]) at $(cube.λ[1]) microns"
    @debug "Aperture radius in pixels (scale = x$(aperture_scale)): $(ap_pix[1]) at $(cube.λ[1]) microns"

    if isnothing(kernel_size)
        if kernel_type == :Gaussian
            # Require the kernel size to cover 1-sigma (68%) of the PSF distribution
            kernel_size = Int(ceil(2 * maximum(ap_pix) / (√(2log(2)))))
        elseif kernel_type == :Tophat
            # The diameter of the kernel is 2x the PSF FWHM for an aperture_scale of 1
            kernel_size = Int(ceil(2 * maximum(ap_pix)))
        end
        # Make sure the size is odd so the center position covers the spaxel being convolved
        if kernel_size % 2 == 0
            kernel_size += 1
        end
        @debug "Using kernel size of $kernel_size pixels"
    end

    # Initialize kernel
    kernel = zeros(kernel_size, kernel_size, length(cube.λ))
    # Get the coordinate of the center of the kernel (i.e. [2,2] for a kernel size of 3)
    c = cld(kernel_size, 2)
    f = fld(kernel_size, 2)

    # Loop through each wavelength point
    for wi ∈ 1:length(cube.λ)

        if kernel_type == :Gaussian
            # For each pixel in the kernel, calculate the distance from the center (in pixels)
            # and calculate the value of the normal distribution at that point
            for xᵢ ∈ 1:kernel_size, yᵢ ∈ 1:kernel_size
                # Set the kernel to the normal probability distribution function at the given coordinates
                dist = hypot(xᵢ-c, yᵢ-c)
                # (This is not normalized properly but it doesnt matter because we divide out the normalization anyways)
                kernel[xᵢ, yᵢ, wi] = Gaussian(dist, 1., 0., 2*ap_pix[wi])
            end
        elseif kernel_type == :Tophat
            kernel[:, :, wi] .= py_photutils.geometry.circular_overlap_grid(-f-0.5, f+0.5, -f-0.5, f+0.5, kernel_size, kernel_size, 
                ap_pix[wi], true, 5)'
        end

        # Renormalize such that it integrates to 1
        kernel[:,:,wi] ./= sum(kernel[:,:,wi])

    end

    kernel

end


"""
    convolve_psf!(cube, psf_kernel)

Convolve the flux and error cubes with the PSF kernel to obtain a cube that has been smoothed out by the PSF
"""
function convolve_psf!(cube::DataCube; aperture_scale::Real=1., kernel_size::Union{Integer,Nothing}=nothing, kernel_type::Symbol=:Tophat)

    kernel = psf_kernel(cube; aperture_scale=aperture_scale, kernel_size=kernel_size, kernel_type=kernel_type)

    kernel_size = size(kernel)[1]
    kernel_cent = cld(kernel_size, 2)
    kernel_len = fld(kernel_size, 2)

    @info "The median PSF FWHM is $(median(cube.psf_fwhm)) arcseconds"
    @info "Convolving the IFU data with a $(kernel_size)x$(kernel_size) $(kernel_type) PSF FWHM kernel with a median " *
          "$(kernel_type == :Tophat ? "diameter" : "FWHM") of $(2*aperture_scale*median(cube.psf_fwhm)) arcseconds"

    # Step along the spatial directions and apply the kernel at each pixel
    @showprogress for xᵢ ∈ 1:size(cube.Iν,1), yᵢ ∈ 1:size(cube.Iν,2)

        if any(cube.mask[xᵢ, yᵢ, :])
            # Do not convolve spaxels that are masked
            continue
        end

        # Copy the kernel
        k = copy(kernel)

        # Min/max indices to consider
        xmin = xᵢ - kernel_len
        xmax = xᵢ + kernel_len
        ymin = yᵢ - kernel_len
        ymax = yᵢ + kernel_len

        # Truncate the kernel if we're at one of the bounds
        if xmin < 1
            k = k[kernel_cent:end, :, :]
            xmin = xᵢ
        elseif xmax > size(cube.Iν,1)
            k = k[1:kernel_cent, :, :]
            xmax = xᵢ
        end
        if ymin < 1
            k = k[:, kernel_cent:end, :]
            ymin = yᵢ
        elseif ymax > size(cube.Iν,2)
            k = k[:, 1:kernel_cent, :]
            ymax = yᵢ
        end

        # Check if any of the spaxels that would be covered by the kernel are masked
        mask_reg = [any(cube.mask[xx, yy, :]) for xx ∈ xmin:xmax, yy ∈ ymin:ymax] 
        # If so, renormalize the kernel such that only the unmasked spaxels are weighted
        k[mask_reg, :] .= 0.
        if iszero(sum(k))
            # If every spaxel in the region is masked out, we dont need to do any convolving
            continue
        end
        for wi ∈ 1:size(k,3)
            k[:, :, wi] ./= sum(k[:, :, wi])
        end

        # Perform the convolution with the kernel
        I = cube.Iν[xmin:xmax, ymin:ymax, :]
        cube.Iν[xᵢ, yᵢ, :] .= sumdim(k .* I, (1,2))

        # Same for the error
        σ = cube.σI[xmin:xmax, ymin:ymax, :]
        cube.σI[xᵢ, yᵢ, :] .= .√(sumdim(k .* σ.^2, (1,2)))

    end

    cube

end

