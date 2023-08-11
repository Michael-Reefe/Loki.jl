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
    DataCube(λ, I, σ[, mask, Ω, α, δ, psf_fwhm, wcs, channel, band, rest_frame, masked])

An object for holding 3D IFU spectroscopy data. 

# Fields
- `λ::Vector{<:Real}`: 1D array of wavelengths, in μm
- `I::Array{<:Real,3}`: 3D array of intensity, in MJy/sr (MIR) or erg/s/cm^2/ang/sr (OPT)
- `σ::Array{<:Real,3}`: 3D array of uncertainties, in MJy/sr (MIR) or erg/s/cm^2/ang/sr (OPT)
- `mask::BitArray{3}=falses(size(Iν))`: 3D array of booleans acting as a mask for the flux/error data
- `Ω::Real=NaN`: the solid angle subtended by each spaxel, in steradians
- `α::Real=NaN`: the right ascension of the observation, in decimal degrees
- `δ::Real=NaN`: the declination of the observation, in decimal degrees
- `psf::Function`: the FWHM of the spatial point-spread function in arcseconds as a function of (observed-frame) wavelength in microns
- `lsf::Function`: the FWHM of the spectral line-spread function in km/s as a function of (observed-frame) wavelength in microns
- `wcs::Union{WCSTransform,Nothing}=nothing`: a World Coordinate System conversion object, optional
- `channel::String="Generic Channel"`: the MIRI channel of the observation, from 1-4
- `band::String="Generic Band"`: the MIRI band of the observation, i.e. 'MULTIPLE'
- `nx::Integer=size(Iν,1)`: the length of the x dimension of the cube
- `ny::Integer=size(Iν,2)`: the length of the y dimension of the cube
- `nz::Integer=size(Iν,3)`: the length of the z dimension of the cube
- `spectral_region::Symbol`: which spectral region does the DataCube cover. Must be either :MIR for mid-infrared or :OPT for optical.
- `rest_frame::Bool=false`: whether or not the DataCube wavelength vector is in the rest-frame
- `masked::Bool=false`: whether or not the DataCube has been masked
- `vacuum_wave::Bool=true`: whether or not the wavelength vector is in vacuum wavelengths; if false, it is assumed to be air wavelengths
"""
mutable struct DataCube

    λ::Vector{<:Real}
    I::Array{<:Real,3}
    σ::Array{<:Real,3}
 
    mask::BitArray{3}

    Ω::Real
    α::Real
    δ::Real
    psf::Vector{<:Real}
    lsf::Vector{<:Real}

    wcs::Union{PyObject,Nothing}

    channel::String
    band::String

    nx::Integer
    ny::Integer
    nz::Integer

    spectral_region::Symbol
    rest_frame::Bool
    masked::Bool
    vacuum_wave::Bool

    voronoi_bins::Union{Matrix{<:Integer},Nothing}

    # This is the constructor for the DataCube struct; see the DataCube docstring for details
    function DataCube(λ::Vector{<:Real}, I::Array{<:Real,3}, σ::Array{<:Real,3}, mask::Union{BitArray{3},Nothing}=nothing, 
        Ω::Real=NaN, α::Real=NaN, δ::Real=NaN, psf::Union{Vector{<:Real},Nothing}=nothing, lsf::Union{Vector{<:Real},Nothing}=nothing, 
        wcs::Union{PyObject,Nothing}=nothing, channel::String="Generic Channel", band::String="Generic Band", spectral_region::Symbol=:MIR, 
        rest_frame::Bool=false, masked::Bool=false, vacuum_wave::Bool=true, voronoi_bins::Union{Matrix{<:Integer},Nothing}=nothing)

        # Make sure inputs have the right dimensions
        @assert ndims(λ) == 1 "Wavelength vector must be 1-dimensional!"
        @assert (ndims(I) == 3) && (size(I)[end] == size(λ)[1]) "The last axis of the intensity cube must be the same length as the wavelength!"
        @assert size(I) == size(σ) "The intensity and error cubes must be the same size!"
        if !isnothing(psf)
            @assert size(psf) == size(λ) "The PSF FWHM vector must be the same size as the wavelength vector!"
        end
        if !isnothing(lsf)
            @assert size(lsf) == size(λ) "The LSF FWHM vector must be the same size as the wavelength vector!"
        end

        nx, ny, nz = size(I)

        # If no mask is given, make the default mask to be all falses (i.e. don't mask out anything)
        if isnothing(mask)
            @info "DataCube initialization: No mask was given, all spaxels will be unmasked"
            mask = falses(size(I))
        else
            @assert size(mask) == size(I) "The mask must be the same size as the intensity cube!"
        end

        # Return a new instance of the DataCube struct
        new(λ, I, σ, mask, Ω, α, δ, psf, lsf, wcs, channel, band, nx, ny, nz, spectral_region, rest_frame, masked, vacuum_wave, voronoi_bins)
    end

end


"""
    from_fits(filename::String)

Utility class-method for creating DataCube structures directly from FITS files.

# Arguments
- `filename::String`: the filepath to the JWST-formatted FITS file
"""
function from_fits(filename::String; format=:JWST)
    if format == :JWST
        from_fits_jwst(filename)
    else
        error("Unrecognized format $format")
    end
end


function from_fits_jwst(filename::String)::DataCube

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

    # Only save the spatial axes as the WCS since we create the spectral axis separately
    wcs = py_wcs.WCS(read_header(hdu["SCI"], String), naxis=2)

    # Wavelength vector
    λ = try
        read(hdu["AUX"], "wave")
    catch
        hdr["CRVAL3"] .+ hdr["CDELT3"] .* (collect(0:hdr["NAXIS3"]-1) .+ hdr["CRPIX3"] .- 1)
    end
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
    channel = string(hdr0["CHANNEL"])  # MIRI channel (1-4)
    band = hdr0["BAND"]        # MIRI band (long,med,short,multiple)

    @debug """\n
    ##################################################################
    #################### TARGET INFORMATION ##########################
    ##################################################################
    name: \t\t $name
    RA: \t\t\t $ra
    Dec: \t\t\t $dec
    Channel: \t $channel
    Band: \t\t $band
    ##################################################################
    """

    # Make sure intensity units are MegaJansky per steradian and wavelength 
    # units are microns (this is assumed in the fitting code)
    # if hdr["BUNIT"] ≠ "MJy/sr"
    #     error("Unrecognized flux unit: $(hdr["BUNIT"])")
    # end
    # if hdr["CUNIT3"] ≠ "um"
    #     error("Unrecognized wavelength unit: $(hdr["CUNIT3"])")
    # end

    # Get the PSF FWHM in arcseconds assuming diffraction-limited optics (theta = lambda/D)
    # Using the OBSERVED-FRAME wavelength
    # psf_fwhm = @. (λ * 1e-6 / mirror_size) * 180/π * 3600
    psf = try
        read(hdu["AUX"], "psf")
    catch
        @. 0.033 * λ + 0.016
    end
    lsf = try
        read(hdu["AUX"], "lsf")
    catch
        parse_resolving(channel).(λ)
    end

    # @debug "Intensity units: $(hdr["BUNIT"]), Wavelength units: $(hdr["CUNIT3"])"

    spectral_region = :MIR
    if haskey(hdr0, "SPECREG")
        spectral_region = Symbol(hdr0["SPECREG"])
    end
    rest_frame = false
    if haskey(hdr0, "RESTFRAM")
        rest_frame = hdr0["RESTFRAM"]
    end
    masked = false
    if haskey(hdr0, "MASKED")
        masked = hdr0["MASKED"]
    end
    vacuum_wave = true
    if haskey(hdr0, "VACWAVE")
        vacuum_wave = hdr0["VACWAVE"]
    end

    DataCube(λ, Iν, σI, mask, Ω, ra, dec, psf, lsf, wcs, channel, band, spectral_region, rest_frame, masked, vacuum_wave)
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

        # Wavelength is shorter in the rest frame
        cube.λ = @. cube.λ / (1 + z)
        if cube.spectral_region == :MIR
            # To conserve flux, which here is measured per unit frequency, we must also divide by the same factor
            cube.I = @. cube.I / (1 + z)
            # Uncertainty follows the same scaling as flux
            cube.σ = @. cube.σ / (1 + z)
        elseif cube.spectral_region == :OPT
            # Optical spectra are measured per unit wavelength, so we multiply by the 1+z factor instead
            cube.I = @. cube.I * (1 + z)
            cube.σ = @. cube.σ * (1 + z)
        end

        cube.rest_frame = true
    end

    cube

end


"""
    to_vacuum_wavelength!(cube::DataCube, linear_resample=true)

Convert a DataCube object's wavelength vector from air wavelengths to vacuum wavelengths.
"""
function to_vacuum_wavelength!(cube::DataCube; linear_resample::Bool=true)

    # Only convert if it isn't already in vacuum wavelengths
    if !cube.vacuum_wave
        @debug "Converting the wavelength vector of cube with channel $(cube.channel), band $(cube.band)" *
            " to vacuum wavelengths."
        # Convert to vacuum wavelengths
        cube.λ = air_to_vacuum.(cube.λ)
        # Optionally resample back onto a linear grid
        if linear_resample
            λlin = range(minimum(cube.λ), maximum(cube.λ), length=length(cube.λ))
            cube.I, cube.σ, cube.mask = resample_conserving_flux(λlin, cube.λ, cube.I, cube.σ, cube.mask)
            cube.λ = collect(λlin)
        end
        cube.vacuum_wave = true
    else
        @debug "The wavelength vector is already in vacuum wavelengths for cube with channel $(cube.channel), " *
            "band $(cube.band)."
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

        cube.I[cube.mask] .= NaN
        cube.σ[cube.mask] .= NaN
        cube.masked = true

    end

    cube

end


"""
    log_rebin!(cube::DataCube)

Rebin a DataCube onto a logarithmically spaced wavelength vector, conserving flux.
"""
function log_rebin!(cube::DataCube)

    # check if it is already log-rebinned
    log_check = (cube.λ[2]/cube.λ[1]) ≈ (cube.λ[end]/cube.λ[end-1])
    if cube.spectral_region == :MIR
        @warn "The LOKI code does not require logarithmic rebinning for MIR spectra, only for optical spectra." *
            " In fact, it is discouraged to logarithmically rebin MIR spectra since it is unnecessary." *
            " However, doing so should not drastically affect fit results, so this is merely a warning."
    end

    # rebin onto a logarithmically spaced wavelength grid
    if !log_check
        lnλ = get_logarithmic_λ([minimum(cube.λ), maximum(cube.λ)], length(cube.λ), oversample=1)
        cube.I, cube.σ, cube.mask = resample_conserving_flux(lnλ, cube.λ, cube.I, cube.σ, cube.mask)
        cube.λ = lnλ
    else
        @warn "Cube is already log-rebinned! Will not be rebinned again."
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

    for index ∈ CartesianIndices(selectdim(cube.I, 3, 1))

        I = cube.I[index, :]
        σ = cube.σ[index, :]

        # Filter NaNs
        if sum(.!isfinite.(I) .| .!isfinite.(σ)) > (size(I, 1) / 2)
            # Keep NaNs in spaxels that are a majority NaN (i.e., we do not want to fit them)
            @debug "Too many NaNs in spaxel $index -- this spaxel will not be fit"
            continue
        end
        filt = .!isfinite.(I) .& .!isfinite.(σ)

        # Interpolate the NaNs
        if sum(filt) > 0
            @debug "NaNs found in spaxel $index -- interpolating"

            finite = isfinite.(I)
            scale = 7

            # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
            λknots = λ[finite][(1+scale):scale:(length(λ[finite])-scale)]
            good = []
            for i ∈ eachindex(λknots) 
                _, λc = findmin(abs.(λknots[i] .- λ))
                if !isnan(I[λc])
                    append!(good, [i])
                end
            end
            λknots = λknots[good]
            # ONLY replace NaN values, keep the rest of the data as-is
            I[filt] .= Spline1D(λ[isfinite.(I)], I[isfinite.(I)], λknots, k=1, bc="extrapolate").(λ[filt])
            σ[filt] .= Spline1D(λ[isfinite.(σ)], σ[isfinite.(σ)], λknots, k=1, bc="extrapolate").(λ[filt])

            # Reassign data in cube structure
            cube.I[index, :] .= I
            cube.σ[index, :] .= σ

        end 
    end

    return
end


"""
    calculate_statistical_errors!(cube)

This function calculates the 'statistical' errors of the given IFU cube, replacing its error cube.
The statistical errors are defined as the standard deviation of the residuals between the flux and a cubic spline
fit to the flux, within a small window (60 pixels). Emission lines are masked out during this process.
"""
function calculate_statistical_errors!(cube::DataCube, Δ::Union{Integer,Nothing}=nothing, 
    n_inc_thresh::Union{Integer,Nothing}=nothing, thresh::Union{Real,Nothing}=nothing,
    overrides::Vector{Tuple{T,T}}=Vector{Tuple{Real,Real}}(); median::Bool=false) where {T<:Real}

    λ = cube.λ
    if isnothing(Δ)
        Δ = cube.spectral_region == :MIR ? 3 : 20
    end
    if isnothing(n_inc_thresh)
        n_inc_thresh = cube.spectral_region == :MIR ? 3 : 7
    end
    if isnothing(thresh)
        thresh = 3.0
    end

    println("Calculating statistical errors for each spaxel...")
    @showprogress for spaxel ∈ CartesianIndices(size(cube.I)[1:2])
        # Get the flux/error for this spaxel
        I = cube.I[spaxel, :]
        σ = cube.σ[spaxel, :]
        # Perform a cubic spline fit, also obtaining the line mask
        mask_lines, I_spline, _ = continuum_cubic_spline(λ, I, σ, Δ, n_inc_thresh, thresh, overrides)
        mask_bad = cube.mask[spaxel, :]
        mask = mask_lines .| mask_bad

        l_mask = sum(.~mask)
        if iszero(l_mask)
            continue
        end
        # Statistical uncertainties based on the local RMS of the residuals with a cubic spline fit
        σ_stat = zeros(l_mask)
        for i in 1:l_mask
            indices = sortperm(abs.((1:l_mask) .- i))[1:min(60,l_mask)]
            σ_stat[i] = std(I[.~mask][indices] .- I_spline[.~mask][indices])
        end
        # We insert at the locations of the lines since the cubic spline does not include them
        l_all = length(λ)
        line_inds = (1:l_all)[mask]
        for line_ind ∈ line_inds
            insert!(σ_stat, line_ind, σ_stat[max(line_ind-1, 1)])
        end
        @debug "Statistical uncertainties for spaxel $spaxel: ($(σ_stat[1]) - $(σ_stat[end]))"
        # σ = hypot.(σ, σ_stat)

        # Replace the cube's error with the statistical errors
        if median
            σ_stat .= nanmedian(σ_stat)
        end
        cube.σ[spaxel, :] .= σ_stat
    end

end 


"""
    voronoi_rebin!(cube, target_SN)

Calculate Voronoi bins for the cube such that each bin has a signal to noise ratio roughly equal to `target_SN`.
Modifies the cube object in-place with the `voronoi_bins` attribute, which is a 2D array that gives unique integer
labels to each voronoi bin.
"""
function voronoi_rebin!(cube::DataCube, target_SN::Real)

    @info "Performing Voronoi rebinning with target S/N=$target_SN"
    # Get the signal and noise 
    signal = dropdims(nanmedian(cube.I, dims=3), dims=3)
    noise = dropdims(nanmedian(cube.σ, dims=3), dims=3)
    # x/y coordinate arrays
    x = [i for i in axes(signal,1), _ in axes(signal,2)]
    y = [j for _ in axes(signal,1), j in axes(signal,2)]
    # mask out bad spaxels
    mask = (.~isfinite.(signal)) .| (.~isfinite.(noise))
    # flatten arrays
    signal = signal[.~mask]
    noise = noise[.~mask]
    x = x[.~mask]
    y = y[.~mask]
    # make sure signals are nonnegative
    signal = clamp.(signal, 0., Inf)
    # perform voronoi rebinning
    bin_numbers, _, _, _, _, _, _, _ = py_vorbin.voronoi_2d_binning(x, y, signal, noise, target_SN, pixelsize=1.0, 
        cvt=true, wvt=true, plot=false)
    # reformat bin numbers as a 2D array so that we don't need the x/y vectors anymore
    voronoi_bins = zeros(Int, size(cube.I)[1:2])
    for i in eachindex(bin_numbers)
        voronoi_bins[x[i], y[i]] = bin_numbers[i] + 1  # python 0-based indexing -> julia 1-based indexing
    end
    # Set the voronoi_bins value in the cube object
    cube.voronoi_bins = voronoi_bins

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
- `cosmo::Union{AbstractCosmology,Nothing}=nothing`: Cosmology object for calculating physical distance scales
- `aperture::Union{Aperture.AbstractAperture,Nothing}=nothing`: Aperture object that may be plotted to show its location/size

See also [`DataCube`](@ref), [`plot_1d`](@ref)
"""
function plot_2d(data::DataCube, fname::String; intensity::Bool=true, err::Bool=true, logᵢ::Union{Integer,Nothing}=10,
    logₑ::Union{Integer,Nothing}=nothing, colormap::Symbol=:cubehelix, name::Union{String,Nothing}=nothing, 
    slice::Union{Integer,Nothing}=nothing, z::Union{Real,Nothing}=nothing, cosmo::Union{Cosmology.AbstractCosmology,Nothing}=nothing, 
    aperture::Union{Aperture.AbstractAperture,Nothing}=nothing)

    @debug "Plotting 2D intensity/error map for cube with channel $(data.channel), band $(data.band)"

    if isnothing(slice)
        # Sum up data along wavelength dimension
        I = sumdim(data.I, 3)
        σ = sqrt.(sumdim(data.σ.^2, 3))
        # Reapply masks
        I[I .≤ 0.] .= NaN
        σ[σ .≤ 0.] .= NaN
        sub = ""
    else
        # Take the wavelength slice
        I = data.I[:, :, slice]
        σ = data.σ[:, :, slice]
        I[I .≤ 0.] .= NaN
        σ[σ .≤ 0.] .= NaN
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
    if data.spectral_region == :MIR
        unit_str = isnothing(slice) ? L"MJy$\,$sr$^{-1}\,\mu$m" : L"MJy$\,$sr$^{-1}$"
    else
        unit_str = isnothing(slice) ? L"erg$\,$s$^{-1}\,$cm$^{-2}\,$sr$^{-1}$" : L"erg$\,$s$^{-1}\,$cm$^{-2}\,\mathring{A}^{-1}\,$sr$^{-1}$"
    end

    ax1 = ax2 = nothing

    # 1D, no NaNs/Infs
    pix_as = sqrt(data.Ω) * 180/π * 3600
    if !isnothing(z) && !isnothing(cosmo)
        # Angular and physical scalebars
        n_pix = 1/pix_as
        @debug "Using angular diameter distance $(angular_diameter_dist(cosmo, z))"
        # Calculate in Mpc
        dA = angular_diameter_dist(u"pc", cosmo, z)
        # Remove units
        dA = uconvert(NoUnits, dA/u"pc")
        l = dA * π/180 / 3600  # l = d * theta (1")
        # Round to a nice even number
        l = Int(round(l, sigdigits=1))
        # new angular size for this scale
        θ = l / dA
        n_pix = 1/sqrt(data.Ω) * θ   # number of pixels = (pixels per radian) * radians
        unit = "pc"
        # convert to kpc if l is more than 1000 pc
        if l ≥ 1000
            l = Int(l / 1000)
            unit = "kpc"
        end
    end

    fig = plt.figure(figsize=intensity && err ? (12, 6) : (12,12))
    if intensity
        # Plot intensity on a 2D map
        ax1 = fig.add_subplot(121, projection=data.wcs)
        ax1.set_title(isnothing(name) ? "" : name)
        cdata = ax1.imshow(I', origin=:lower, cmap=colormap, vmin=nanquantile(I, 0.01), vmax=nanquantile(I, 0.99))
        fig.colorbar(cdata, ax=ax1, fraction=0.046, pad=0.04, 
            label=(isnothing(logᵢ) ? "" : L"$\log_{%$logᵢ}$(") * L"$I_{%$sub}\,/\,$" * unit_str * (isnothing(logᵢ) ? "" : ")"))
        # ax1.axis(:off)
        ax1.tick_params(which="both", axis="both", direction="in")
        ax1.set_xlabel("R.A.")
        ax1.set_ylabel("Dec.")
        
        if !isnothing(z) && !isnothing(cosmo)
            if cosmo.h ≈ 1.0
                scalebar = py_anchored_artists.AnchoredSizeBar(ax1.transData, n_pix, L"%$l$h^{-1}$ %$unit", "lower left", pad=1, color=:black, 
                    frameon=false, size_vertical=0.4, label_top=false)
            else
                scalebar = py_anchored_artists.AnchoredSizeBar(ax1.transData, n_pix, "$l $unit", "lower left", pad=1, color=:black,
                    frameon=false, size_vertical=0.4, label_top=false)
            end
            ax1.add_artist(scalebar)
        end

        psf = plt.Circle(size(I) .* 0.9, (isnothing(slice) ? median(data.psf) : data.psf[slice]) / pix_as / 2, color="k")
        ax1.add_patch(psf)
        ax1.annotate("PSF", size(I) .* 0.9 .- (0., median(data.psf) / pix_as / 2 * 1.5 + 1.75), ha=:center, va=:center,
            bbox=Dict(:facecolor => "white", :edgecolor => "white", :pad => 5.0))    

        if !isnothing(aperture)
            patches = get_patches(aperture)
            for patch in patches
                ax1.add_patch(patch)
            end
        end

    end

    if err
        # Plot error on a 2D map
        ax2 = fig.add_subplot(122, projection=data.wcs)
        ax2.set_title(isnothing(name) ? "" : name)
        cdata = ax2.imshow(σ', origin=:lower, cmap=colormap, vmin=nanquantile(σ, 0.01), vmax=nanquantile(σ, 0.99))
        fig.colorbar(cdata, ax=ax2, fraction=0.046, pad=0.04,
            label=isnothing(logₑ) ? L"$\sigma_{I_{%$sub}}\,/\,$" * unit_str : L"$\sigma_{\log_{%$logᵢ}I_{%$sub}}$")
        # ax2.axis(:off)
        ax2.tick_params(which="both", axis="both", direction="in")
        ax2.set_xlabel("R.A.")
        ax2.set_ylabel("Dec.")

        if !isnothing(z) && !isnothing(cosmo)
            if cosmo.h ≈ 1.0
                scalebar = py_anchored_artists.AnchoredSizeBar(ax2.transData, n_pix, L"%$l$h^{-1}$ %$unit", "lower left", pad=1, color=:black, 
                    frameon=false, size_vertical=0.4, label_top=false)
            else
                scalebar = py_anchored_artists.AnchoredSizeBar(ax2.transData, n_pix, "$l $unit", "lower left", pad=1, color=:black,
                    frameon=false, size_vertical=0.4, label_top=false)
            end
            ax2.add_artist(scalebar)
        end

        psf = plt.Circle(size(I) .* 0.9, (isnothing(slice) ? median(data.psf) : data.psf[slice]) / pix_as / 2, color="k")
        ax1.add_patch(psf)
        ax1.annotate("PSF", size(I) .* 0.9 .- (0., median(data.psf) / pix_as / 2 * 1.5 + 1.75), ha=:center, va=:center,
            bbox=Dict(:facecolor => "white", :edgecolor => "white", :pad => 5.0))

        if !isnothing(aperture)
            patches = get_patches(aperture)
            for patch in patches
                ax2.add_patch(patch)
            end
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

# Arguments {S<:Integer}
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

    @debug "Plotting 1D intensity/error map for cube with channel $(data.channel), band $(data.band)"

    # Alias
    λ = data.λ
    if isnothing(spaxel)
        # Sum up data along spatial dimensions
        I = sumdim(data.I, (1,2))
        σ = sqrt.(sumdim(data.σ.^2, (1,2)))
        # Reapply masks
        I[I .≤ 0.] .= NaN
        σ[σ .≤ 0.] .= NaN

        # If intensity was not per spaxel, we pick up an extra spaxel/sr term that must be divided out
        I ./= sumdim(Array{Int}(.~data.mask), (1,2))
        σ ./= sumdim(Array{Int}(.~data.mask), (1,2))
    else
        # Take the spaxel
        I = data.I[spaxel..., :]
        σ = data.σ[spaxel..., :]
        sub = "\\lambda"
    end

    # If specified, take the logarithm of the data with the given base
    if logᵢ ≠ false
        σ .= σ ./ abs.(I .* log.(logᵢ)) 
        ν_Hz = (C_KMS .* 1e9) ./ data.λ
        if data.spectral_region == :MIR
            I .= log.(ν_Hz .* I) / log.(logᵢ)
        else
            I .= log.(data.λ .* 1e4 .* I) / log.(logᵢ)
        end
    end

    # Formatting for x and y units on the axis labels
    xunit = L"${\rm \mu m}$"
    if data.spectral_region == :MIR
        yunit = logᵢ == false ? L"MJy$\,$sr$^{-1}$" : L"MJy$\,$sr$^{-1}\,$Hz"
    else
        yunit = logᵢ == false ? L"erg$\,$s$^{-1}\,$cm$^{-2}\,\mathring{A}^{-1}\,$sr$^{-1}$" : L"erg$\,$s$^{-1}\,$cm$^{-2}\,$sr$^{-1}$"
    end
 
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
- `spectral_region::Symbol`: Either :MIR for mid-infrared data or :OPT for optical data
- `rest_frame::Bool=false`: whether or not the individual DataCubes have been converted to the rest frame
- `masked::Bool=false`: whether or not the individual DataCubes have been masked
- `vacuum_wave::Bool=true`: whether or not the wavelengths are specified in vacuum wavelengths; if false, they are assumed
to be in air wavelengths

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
    spectral_region::Symbol
    rest_frame::Bool
    masked::Bool
    vacuum_wave::Bool

    function Observation(channels::Dict{Any,DataCube}=Dict{Any,DataCube}(), name::String="Generic Observation",
        z::Real=NaN, α::Real=NaN, δ::Real=NaN, instrument::String="Generic Instrument", 
        detector::String="Generic Detector", spectral_region::Symbol=:MIR, rest_frame::Bool=false, masked::Bool=false,
        vacuum_wave::Bool=true)

        new(channels, name, z, α, δ, instrument, detector, spectral_region, rest_frame, masked, vacuum_wave)
    end
    
end


"""
    save_fits(path, obs[, channels])

Save a pre-processed Observation object as a FITS file in a format that can be read by the "from_fits" function.
This saves time upon re-running the code assuming one wishes to keep the same pre-processing.
"""
save_fits(path::String, obs::Observation) = save_fits(path, obs, collect(keys(obs.channels)))
save_fits(path::String, obs::Observation, channels::Integer) = save_fits(path, obs, [channels])

function save_fits(path::String, obs::Observation, channels::Vector)
    for channel in channels

        # Header information
        hdr = FITSHeader(
            Vector{String}(["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "PIXAR_SR", "TARG_RA", "TARG_DEC", "INSTRUME", "DETECTOR", 
                "SPECREG", "RESTFRAM", "MASKED", "VACWAVE", "DATAMODL", "NAXIS1", "NAXIS2", "NAXIS3", "WCSAXES", "CDELT1", "CDELT2", "CTYPE1", 
                "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CUNIT1", "CUNIT2", "PC1_1", "PC1_2", "PC2_1", "PC2_2"]),

            [obs.name, obs.z, string(channel), obs.channels[channel].band, obs.channels[channel].Ω, 
                obs.α, obs.δ, obs.instrument, obs.detector, string(obs.spectral_region), obs.rest_frame, obs.masked, obs.vacuum_wave, 
                "IFUCubeModel", obs.channels[channel].nx, obs.channels[channel].ny, obs.channels[channel].nz, obs.channels[channel].wcs.wcs.naxis, 
                obs.channels[channel].wcs.wcs.cdelt[1], obs.channels[channel].wcs.wcs.cdelt[2], 
                obs.channels[channel].wcs.wcs.ctype[1], obs.channels[channel].wcs.wcs.ctype[2], 
                obs.channels[channel].wcs.wcs.crpix[1], obs.channels[channel].wcs.wcs.crpix[2], 
                obs.channels[channel].wcs.wcs.crval[1], obs.channels[channel].wcs.wcs.crval[2], 
                obs.channels[channel].wcs.wcs.cunit[1].name, obs.channels[channel].wcs.wcs.cunit[2].name, 
                obs.channels[channel].wcs.wcs.pc[1,1], obs.channels[channel].wcs.wcs.pc[1,2], 
                obs.channels[channel].wcs.wcs.pc[2,1], obs.channels[channel].wcs.wcs.pc[2,2]],

            Vector{String}(["Target name", "Target redshift", "Channel", "Band",
                "Solid angle per pixel (rad.)", "Right ascension of target (deg.)", "Declination of target (deg.)",
                "Instrument name", "Detector name", "spectral region", "data in rest frame", "data masked", "vacuum wavelengths", "data model",
                "length of the first axis", "length of the second axis", "length of the third axis",
                "number of World Coordinate System axes", 
                "first axis increment per pixel", "second axis increment per pixel",
                "first axis coordinate type", "second axis coordinate type",
                "axis 1 coordinate of the reference pixel", "axis 2 coordinate of the reference pixel",
                "first axis value at the reference pixel", "second axis value at the reference pixel",
                "first axis units", "second axis units",
                "linear transformation matrix element", "linear transformation matrix element",
                "linear transformation matrix element", "linear transformation matrix element"])
        )
        FITS(joinpath(path, "$(replace(obs.name, " " => "_")).channel$(channel)$(obs.rest_frame ? ".rest_frame" : "").fits"), "w") do f
            @info "Writing FITS file from Observation object"

            write(f, Vector{Int}(); header=hdr)                          # Primary HDU (empty)
            write(f, obs.channels[channel].I; name="SCI", header=hdr)   # Data HDU
            write(f, obs.channels[channel].σ; name="ERR", header=hdr)   # Error HDU
            write(f, UInt8.(obs.channels[channel].mask); name="DQ", header=hdr)  # Mask HDU
            write(f, ["wave", "psf", "lsf"],                                     # Auxiliary HDU
                     [obs.channels[channel].λ, obs.channels[channel].psf, obs.channels[channel].lsf], 
                     hdutype=TableHDU, name="AUX", units=Dict(:wave => "um", :psf => "arcsec", :lsf => "km/s"))
            
            write_key(f["SCI"], "BUNIT", obs.spectral_region == :MIR ? "MJy/sr" : "erg/s/cm^2/ang/sr")
            write_key(f["ERR"], "BUNIT", obs.spectral_region == :MIR ? "MJy/sr" : "erg/s/cm^2/ang/sr")
        end
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

    spectral_region = :MIR
    if haskey(hdr, "SPECREG")
        spectral_region = Symbol(hdr["SPECREG"])
    end
    rest_frame = false
    if haskey(hdr, "RESTFRAM")
        rest_frame = hdr["RESTFRAM"]
    end
    masked = false
    if haskey(hdr, "MASKED")
        masked = hdr["MASKED"]
    end
    vacuum_wave = true
    if haskey(hdr, "VACWAVE")
        vacuum_wave = hdr["VACWAVE"]
    end
        
    # Loop through the files and call the individual DataCube method of the from_fits function
    for (i, filepath) ∈ enumerate(filenames)
        cube = from_fits(filepath)
        if spectral_region == :MIR
            if cube.band == "MULTIPLE"
                channels[parse(Int, cube.channel)] = cube
                continue
            end
            channels[Symbol(bands[cube.band] * cube.channel)] = cube
        else
            channels[i] = cube
        end
    end

    Observation(channels, name, z, ra, dec, inst, detector, spectral_region, rest_frame, masked, vacuum_wave)
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
    to_vacuum_wavelength!(obs::Observation)

Convert each wavelength channel into vacuum wavelengths.
"""
function to_vacuum_wavelength!(obs::Observation; linear_resample::Bool=true)

    @debug """\n
    Converting observation of $(obs.name) to vacuum wavelengths
    ###########################################################
    """
    # Loop through the channels and call the individual DataCube method of the to_vacuum_wavelength function
    for k ∈ keys(obs.channels)
        to_vacuum_wavelength!(obs.channels[k]; linear_resample=linear_resample)
    end
    obs.vacuum_wave = true

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
    correct!

A composition of the `apply_mask!`, `to_rest_frame!`, and `to_vacuum_wavelength!` functions for Observation objects

See [`apply_mask!`](@ref) and [`to_rest_frame!`](@ref)
"""
correct! = apply_mask! ∘ to_rest_frame! ∘ to_vacuum_wavelength!


#################################### CHANNEL ALIGNMENT AND REPROJECTION ######################################


"""
    adjust_wcs_alignment(obs, channels; box_size=9)

Adjust the WCS alignments of each channel such that they match.
This is performed in order to remove discontinuous jumps in the flux level when crossing between
sub-channels.
"""
function adjust_wcs_alignment!(obs::Observation, channels; box_size::Integer=11)

    @assert obs.spectral_region == :MIR "The adjust_wcs_alignment! function is only supported for MIR cubes!"

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
            data = sumdim(ch_data.I[:, :, filter], 3)
            wcs = ch_data.wcs
            # Find the peak brightness pixels and place boxes around them
            peak = argmax(data)
            box_half = fld(box_size, 2)
            mask = trues(size(data))
            mask[peak[1]-box_half:peak[1]+box_half, peak[2]-box_half:peak[2]+box_half] .= 0
            x_cent, y_cent = centroid_com(data, mask)

            fig, ax = plt.subplots()
            data[data .≤ 0] .= NaN
            logdata = log10.(data)
            ax.imshow(logdata', origin=:lower, cmap=:cubehelix, vmin=nanquantile(logdata, 0.01), vmax=nanquantile(logdata, 0.99))
            ax.plot(x_cent-1, y_cent-1, "rx", ms=10)
            nm = replace(obs.name, " " => "_")
            plt.savefig(nm * "_centroid_$(channel)_$k.pdf", dpi=300, bbox_inches=:tight)
            plt.close()

            # Convert pixel coordinates to physical coordinates using the WCS transformation with the argument for 1-based indexing
            c_coords[k, :] .= wcs.all_pix2world([[x_cent, y_cent]], 1)'
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
        ch_data.wcs.wcs.crval = ch_data.wcs.wcs.crval .- offset
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
- `concat_type=:full`: Should be :full for overall channels or :sub for subchannels (bands).
- `out_id=0`: The dictionary key corresponding to the newly rebinned cube, defaults to 0.
- `res=nothing`: Determines which channel's angular resolution to use for the output. Defaults to the first channel in `channels`.
- `scrub_output::Bool=true`: Whether or not to delete the individual channels that were interpolated after assigning the new interpolated channel.
- `rescale_channels::Union{Real,Nothing}=nothing`: Whether or not to rescale the flux of subchannels so that they match at the overlapping regions.
    If a real number is provided, this is interpreted as an observed-frame wavelength to choose the reference channel that all other channels
    are rescaled to match.
- `adjust_wcs_headerinfo::Bool=true`: Whether or not to try to automatically adjust the WCS header info of the channels by 
    calculating centroids at the boundaries between the channels and forcing them to match.  On by default.
- `min_λ::Real=-Inf`: Minimum wavelength cutoff for the output cube.
- `max_λ::Real=Inf`: Maximum wavelength cutoff for the output cube.
- `rescale_limits::Tuple{<:Real,<:Real}=(0.5, 1.5)`: Lower/upper limits on the rescaling factor between channels.
- `rescale_snr::Real=10.0`: SNR threshold that a spaxel must pass before being rescaled.
"""
function reproject_channels!(obs::Observation, channels=nothing, concat_type=:full; out_id=0, res=nothing, scrub_output::Bool=false,
    method=:adaptive, rescale_channels::Union{Real,Nothing}=nothing, adjust_wcs_headerinfo::Bool=true, min_λ::Real=-Inf, max_λ::Real=Inf,
    rescale_limits::Tuple{<:Real,<:Real}=(0.5, 1.5), rescale_snr::Real=10.0, output_wcs_frame=1)

    @assert obs.spectral_region == :MIR "The reproject_channels! function is only supported for MIR cubes!"

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
    if adjust_wcs_headerinfo
        adjust_wcs_alignment!(obs, channels; box_size=11)
    end

    # Output angular size
    if isnothing(res)
        res = channels[1]
    end
    Ω_out = obs.channels[res].Ω

    wcs_optimal = obs.channels[channels[output_wcs_frame]].wcs
    size_optimal = size(obs.channels[channels[output_wcs_frame]].I)[1:2]

    # Output wavelength is just the individual wavelength vectors concatenated (for now -- will be interpolated later)
    λ_out = vcat([obs.channels[ch_i].λ for ch_i ∈ channels]...)
    # Output intensity/error have the same grid as the reference channel but with 3rd axis defined by lambda_out
    I_out = zeros(size_optimal..., size(λ_out)...)
    σ_out = zeros(size(I_out))
    mask_out = falses(size(I_out))

    # Iteration variables to keep track of
    for i ∈ eachindex(channels)

        # The size of the current channel's wavelength veector
        ch_in = channels[i]
        wi_size = length(obs.channels[ch_in].λ)
        wsum = isone(i) ? 0 : sum([length(obs.channels[ch_i].λ) for ch_i in channels[1:(i-1)]])

        @info "Reprojecting $(obs.name) channel $ch_in onto the optimal $(size_optimal) WCS grid..."

        # Get the intensity and error arrays
        # NOTE 1: Dont forget to permute and re-permute the dimensions from the numpy row-major order back to the julia column-major order
        # NOTE 2: We need to resample the FLUX, not the intensity, because the pixel sizes may be different between the input and output images,
        #          and flux scales with the pixel size whereas intensity does not (assuming it's an extended source).
        F_in = permutedims(obs.channels[ch_in].I .* obs.channels[ch_in].Ω, (3,2,1))
        σF_in = permutedims(obs.channels[ch_in].σ .* obs.channels[ch_in].Ω, (3,2,1))
        mask_in = permutedims(obs.channels[ch_in].mask, (3,2,1))

        # Replace NaNs with 0s for the interpolation
        F_in[.!isfinite.(F_in)] .= 0.
        σF_in[.!isfinite.(σF_in)] .= 0.

        # Use the adaptive DeForest (2004) algorithm for resampling the cubes onto the same WCS grid
        # - Conserves flux if the option is enabled
        # - Can use the Hann or Gaussian kernels
        # - Boundary mode determines what to do when some or all of the output region is masked 
        if method == :adaptive
            F_out = permutedims(py_reproject.reproject_adaptive((F_in, obs.channels[ch_in].wcs), wcs_optimal, 
                (size(F_in, 1), size_optimal[2], size_optimal[1]), conserve_flux=true, kernel="gaussian", boundary_mode="strict",
                return_footprint=false), (3,2,1))
            σF_out = permutedims(py_reproject.reproject_adaptive((σF_in.^2, obs.channels[ch_in].wcs), wcs_optimal, 
                (size(σF_in, 1), size_optimal[2], size_optimal[1]), conserve_flux=true, kernel="gaussian", boundary_mode="strict",
                return_footprint=false), (3,2,1))
        elseif method == :interp
            F_out = permutedims(py_reproject.reproject_interp((F_in, obs.channels[ch_in].wcs), wcs_optimal, 
                (size(F_in, 1), size_optimal[2], size_optimal[1]), order=1, return_footprint=false), (3,2,1))
            σF_out = permutedims(py_reproject.reproject_interp((σF_in.^2, obs.channels[ch_in].wcs), wcs_optimal, 
                (size(σF_in, 1), size_optimal[2], size_optimal[1]), order=1, return_footprint=false), (3,2,1))
        elseif method == :exact
            F_out = permutedims(py_reproject.reproject_exact((F_in, obs.channels[ch_in].wcs), wcs_optimal, 
                (size(F_in, 1), size_optimal[2], size_optimal[1]), return_footprint=false), (3,2,1))
            σF_out = permutedims(py_reproject.reproject_exact((σF_in.^2, obs.channels[ch_in].wcs), wcs_optimal, 
                (size(σF_in, 1), size_optimal[2], size_optimal[1]), return_footprint=false), (3,2,1))
        elseif method == :none
            F_out = permutedims(F_in, (3,2,1))
            σF_out = permutedims(σF_in.^2, (3,2,1))
        end
        σF_out = sqrt.(σF_out)

        # Convert back to intensity
        I_out[:, :, wsum+1:wsum+wi_size] .= F_out ./ Ω_out
        σ_out[:, :, wsum+1:wsum+wi_size] .= σF_out ./ Ω_out

        # Use nearest-neighbor interpolation for the mask since it's a binary 1 or 0
        if method != :none
            mask_out_temp = permutedims(py_reproject.reproject_interp((mask_in, obs.channels[ch_in].wcs), 
                wcs_optimal, (size(mask_in, 1), size_optimal[2], size_optimal[1]), order="nearest-neighbor", return_footprint=false), (3,2,1))
        else
            mask_out_temp = permutedims(mask_in, (3,2,1))
        end

        # Set all NaNs to 1s for the mask (i.e. they are masked out)
        mask_out_temp[.!isfinite.(mask_out_temp) .| .!isfinite.(I_out[:, :, wsum+1:wsum+wi_size]) .| 
            .!isfinite.(σ_out[:, :, wsum+1:wsum+wi_size])] .= 1
        mask_out[:, :, wsum+1:wsum+wi_size] .= mask_out_temp
    end
    
    # If an entire channel is masked out, we want to throw away all channels
    # This has to be done after the first loop so that mask_out is not overwritten to be unmasked after it has been masked
    for i ∈ eachindex(channels)
        ch_in = channels[i]
        wi_size = length(obs.channels[ch_in].λ)
        wsum = isone(i) ? 0 : sum([length(obs.channels[ch_i].λ) for ch_i in channels[1:(i-1)]])
        for xᵣ ∈ 1:size_optimal[1], yᵣ ∈ 1:size_optimal[2]
            if all(mask_out[xᵣ, yᵣ, wsum+1:wsum+wi_size])
                mask_out[xᵣ, yᵣ, :] .= 1
            end
        end
    end

    # Need an additional correction (fudge) factor for overall channels
    jumps = findall(diff(λ_out) .< 0.)
    scale_factors = Dict{Int, Matrix{Float64}}()
    if !isnothing(rescale_channels)
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
            # DO NOT rescale low S/N spaxels
            SN = dropdims(nanmedian(I_out ./ σ_out, dims=3), dims=3)
            med_left[SN .< rescale_snr, :] .= 1.
            med_right[SN .< rescale_snr, :] .= 1.
            # rescale the flux to match between the channels, using the given reference point
            if wave_left < rescale_channels
                scale = clamp.(med_right ./ med_left, rescale_limits...)
                I_out[:, :, 1:jump] .*= scale
                σ_out[:, :, 1:jump] .*= scale
                @info "Minimum/Maximum scale factor for channel $(i+1): $(nanextrema(scale))"
            else
                scale = clamp.(med_left ./ med_right, rescale_limits...)
                I_out[:, :, jump+1:end] .*= scale
                σ_out[:, :, jump+1:end] .*= scale
                @info "Minimum/Maximum scale factor for channel $(i+1): $(nanextrema(scale))"
            end
            scale_factors[i+1] = scale
        end
    end
    if concat_type == :full
        λ_con = zeros(eltype(λ_out), 0)
        I_con = zeros(eltype(I_out), size(I_out)[1:2]..., 0)
        σ_con = zeros(eltype(σ_out), size(σ_out)[1:2]..., 0)
        mask_con = falses(size(mask_out)[1:2]..., 0)
        prev_i2 = 1

        for (i, jump) ∈ enumerate(jumps)
            # find the full scale of the overlapping region
            wave_left, wave_right = λ_out[jump+1], λ_out[jump]
            _, i1 = findmin(abs.(λ_out[1:jump] .- wave_left))
            _, i2 = findmin(abs.(λ_out[jump+1:end] .- wave_right))
            i2 += jump
            # resample fluxes in the overlapping regions
            λ_res = median([diff(λ_out[i1:jump])[1], diff(λ_out[jump+1:i2])[1]])
            λ_resamp = λ_out[i1]:λ_res:(λ_out[i2]-eps())
            ss = sortperm(λ_out[i1:i2])
            I_resamp, σ_resamp, mask_resamp = resample_conserving_flux(λ_resamp, 
                λ_out[i1:i2][ss], I_out[:, :, (i1:i2)[ss]], σ_out[:, :, (i1:i2)[ss]], mask_out[:, :, (i1:i2)[ss]])
            # replace overlapping regions in outputs
            λ_con = [λ_con; λ_out[prev_i2:i1-1]; λ_resamp]
            I_con = cat(I_con, I_out[:, :, prev_i2:i1-1], I_resamp, dims=3)
            σ_con = cat(σ_con, σ_out[:, :, prev_i2:i1-1], σ_resamp, dims=3)
            mask_con =cat(mask_con, mask_out[:, :, prev_i2:i1-1], mask_resamp, dims=3)
            prev_i2 = i2
        end

        λ_out = [λ_con; λ_out[prev_i2:end]]
        I_out = cat(I_con, I_out[:, :, prev_i2:end], dims=3)
        σ_out = cat(σ_con, σ_out[:, :, prev_i2:end], dims=3)
        mask_out = cat(mask_con, mask_out[:, :, prev_i2:end], dims=3)
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
        @warn "The wavelength dimension has not be resampled to be linear when concatenating multiple full channels! " *
              "Only overlapping regions between channels have been resampled to a median resolution!"
    end

    # Cut off at large wavelength, if specified
    λmask = min_λ .≤ λ_out .≤ max_λ
    λ_out = λ_out[λmask]
    I_out = I_out[:, :, λmask]
    σ_out = σ_out[:, :, λmask]
    mask_out = mask_out[:, :, λmask]

    # New PSF FWHM function with input in the rest frame
    if obs.rest_frame
        psf_fwhm_out = @. 0.033 * λ_out * (1 + obs.z) + 0.106
    else
        psf_fwhm_out = @. 0.033 * λ_out + 0.106
    end
    # New LSF FWHM function with input in the rest frame
    if obs.rest_frame
        lsf_fwhm_out = parse_resolving("MULTIPLE").(λ_out .* (1 .+ obs.z))
    else
        lsf_fwhm_out = parse_resolving("MULTIPLE").(λ_out)
    end

    if obs.masked
        @info "Masking bins with bad data..."
        I_out[mask_out] .= NaN
        σ_out[mask_out] .= NaN
    end

    # Define the interpolated cube as the zeroth channel (since this is not taken up by anything else)
    obs.channels[out_id] = DataCube(λ_out, I_out, σ_out, mask_out, Ω_out, obs.α, obs.δ, psf_fwhm_out, lsf_fwhm_out, wcs_optimal, 
        "MULTIPLE", "MULTIPLE", obs.spectral_region, obs.rest_frame, obs.masked, obs.vacuum_wave)
    
    # Delete all of the individual channels
    if scrub_output
        @info "Deleting individual channels that are no longer needed"
        for channel ∈ channels
            delete!(obs.channels, channel)
        end
    end

    @info "Done!"

    return scale_factors

end


"""
    fshift(array, Δx, Δy)

Shift a 2D image by a non-integer amount Δx and Δy using bilinear interpolation.
Originally written in IDL for the IDLAstronomy Library: https://idlastro.gsfc.nasa.gov/ftp/contrib/malumuth/fshift.pro

Original docstring is copied below:

;+
;			fshift
;
; Routine to shift an image by non-integer values
;
; CALLING SEQUENCE:
;	results = fshift(image,delx,dely)
;
; INPUTS:
;	image - 2D image to be shifted
;	delx - shift in x (same direction as IDL SHIFT function)
;	dely - shift in y
;
; OUTPUTS:
;	shifted image is returned as the function results
;
; HISTORY:
;	version 2  D. Lindler  May, 1992 - rewritten for IDL version 2
;	19-may-1992	JKF/ACC		- move to GHRS DAF.
;-
;--------------------------------------------------------------------

"""
function fshift(array::AbstractArray, Δx::T, Δy::T) where {T<:Real}

    # Separate shift into an integer and fractional shift
    intx = floor(Int, Δx)
    inty = floor(Int, Δy)
    fracx = Δx - intx
    fracy = Δy - inty
    if fracx < 0
        fracx += 1
        intx -= 1
    end
    if fracy < 0
        fracy += 1
        inty -= 1
    end

    # Shift by the integer portion
    s = circshift(array, (intx, inty))
    if iszero(fracx) && iszero(fracy)
        return s
    end

    # Use bilinear interpolation between four pixels
    return s .* ((1 .- fracx) .* (1 .- fracy)) .+ 
           circshift(s, (0,1)) .* ((1 .- fracx) .* fracy) .+
           circshift(s, (1,0)) .* (fracx .* (1 .- fracy)) .+
           circshift(s, (1,1)) .* fracx .* fracy

end


"""
    frebin(array, nsout, nlout=1, total=false)

Rebin a 1D or 2D array onto a new pixel grid that may or may not be an integer fraction or multiple
of the original grid. Originally written in IDL for the IDLAstronomy Library: https://idlastro.gsfc.nasa.gov/ftp/pro/image/frebin.pro

Original docstring is copied below:

;+
; NAME:
;   FREBIN
;
; PURPOSE:
;   Shrink or expand the size of an array an arbitrary amount using interpolation
;
; EXPLANATION: 
;   FREBIN is an alternative to CONGRID or REBIN.    Like CONGRID it
;   allows expansion or contraction by an arbitrary amount. ( REBIN requires 
;   integral factors of the original image size.)    Like REBIN it conserves 
;   flux by ensuring that each input pixel is equally represented in the output
;   array.       
;
; CALLING SEQUENCE:
;   result = FREBIN( image, nsout, nlout, [ /TOTAL] )
;
; INPUTS:
;    image - input image, 1-d or 2-d numeric array
;    nsout - number of samples in the output image, numeric scalar
;
; OPTIONAL INPUT:
;    nlout - number of lines in the output image, numeric scalar
;            If not supplied, then set equal to 1
;
; OPTIONAL KEYWORD INPUTS:
;   /total - if set, the output pixels will be the sum of pixels within
;          the appropriate box of the input image.  Otherwise they will
;          be the average.    Use of the /TOTAL keyword conserves total counts.
; 
; OUTPUTS:
;    The resized image is returned as the function result.    If the input
;    image is of type DOUBLE or FLOAT then the resized image is of the same
;    type.     If the input image is BYTE, INTEGER or LONG then the output
;    image is usually of type FLOAT.   The one exception is expansion by
;    integral amount (pixel duplication), when the output image is the same
;    type as the input image.  
;     
; EXAMPLE:
;     Suppose one has an 800 x 800 image array, im, that must be expanded to
;     a size 850 x 900 while conserving the total counts:
;
;     IDL> im1 = frebin(im,850,900,/total) 
;
;     im1 will be a 850 x 900 array, and total(im1) = total(im)
; NOTES:
;    If the input image sizes are a multiple of the output image sizes
;    then FREBIN is equivalent to the IDL REBIN function for compression,
;    and simple pixel duplication on expansion.
;
;    If the number of output pixels are not integers, the output image
;    size will be truncated to an integer.  The platescale, however, will
;    reflect the non-integer number of pixels.  For example, if you want to
;    bin a 100 x 100 integer image such that each output pixel is 3.1
;    input pixels in each direction use:
;           n = 100/3.1   ; 32.2581
;          image_out = frebin(image,n,n)
;
;     The output image will be 32 x 32 and a small portion at the trailing
;     edges of the input image will be ignored.
; 
; PROCEDURE CALLS:
;    None.
; HISTORY:
;    Adapted from May 1998 STIS  version, written D. Lindler, ACC
;    Added /NOZERO, use INTERPOLATE instead of CONGRID, June 98 W. Landsman  
;    Fixed for nsout non-integral but a multiple of image size  Aug 98 D.Lindler
;    DJL, Oct 20, 1998, Modified to work for floating point image sizes when
;		expanding the image. 
;    Improve speed by addressing arrays in memory order W.Landsman Dec/Jan 2001
;-
;----------------------------------------------------------------------------
"""
function frebin(array::AbstractArray, nsout::S, nlout::S=1; total::Bool=false) where {S<:Integer}

    # Determine the size of the input array
    ns = size(array, 1)
    nl = length(array)/ns

    # Determine if the new sizes are integral factors of the original sizes
    sbox = ns/nsout
    lbox = nl/nlout

    # Contraction by an integral amount
    if (sbox == round(Int, sbox)) && (lbox == round(Int, lbox)) && (ns % nsout == 0) && (nl % nlout == 0)
        array_shaped = reshape(array, (Int(sbox), nsout, Int(lbox), nlout))
        return dropdims((total ? sum : mean)(array_shaped, dims=(1,3)), dims=(1,3))
    end

    # Expansion by an integral amount
    if (nsout % ns == 0) && (nlout % nl == 0)
        xindex = (1:nsout) / (nsout/ns)
        if isone(nl)  # 1D case, linear interpolation
            return Spline1D(1:ns, array, k=1)(xindex) * (total ? sbox : 1.)
        end
        yindex = (1:nlout) / (nlout/nl)
        interpfunc = Spline2D(1:ns, 1:Int(nl), array, kx=1, ky=1)
        return [interpfunc(x, y) for x in xindex, y in yindex] .* (total ? sbox.*lbox : 1.)
    end

    ns1 = ns-1
    nl1 = nl-1

    # Do 1D case separately
    if isone(nl)
        result = zeros(eltype(array), nsout)
        for i ∈ 0:nsout-1
            rstart = i*sbox                # starting position for each box
            istart = floor(Int, rstart)
            rstop = rstart + sbox          # ending position for each box
            istop = Int(clamp(floor(rstop), 0, ns1))
            frac1 = rstart-istart
            frac2 = 1.0 - (rstop-istop)

            # add pixel values from istart to istop and subtract fractional pixel from istart to start and
            # fractional pixel from rstop to istop

            result[i+1] = sum(array[istart+1:istop+1]) - frac1*array[istart+1] - frac2*array[istop+1]
        end
        return result .* (total ? 1.0 : 1 ./ (sbox.*lbox))
    end

    # Now, do 2D case
    # First, bin second dimension
    temp = zeros(eltype(array), ns, nlout)
    # Loop on output image lines
    for i ∈ 0:nlout-1
        rstart = i*lbox                # starting position for each box 
        istart = floor(Int, rstart)
        rstop = rstart + lbox
        istop = Int(clamp(floor(rstop), 0, nl1))
        frac1 = rstart-istart
        frac2 = 1.0 - (rstop-istop)

        # add pixel values from istart to istop and subtract fractional pixel from istart to start and
        # fractional pixel from rstop to istop

        if istart == istop
            temp[:,i+1] .= (1 .- frac1 .- frac2).*array[:,istart+1]
        else
            temp[:,i+1] .= sumdim(array[:,istart+1:istop+1], 2) .- frac1.*array[:,istart+1] .- frac2.*array[:,istop+1]
        end
    end
    temp = temp'
    # Bin in first dimension
    result = zeros(eltype(array), nlout, nsout)
    # Loop on output image samples
    for i ∈ 0:nsout-1
        rstart = i*sbox                # starting position for each box
        istart = floor(Int, rstart)
        rstop = rstart + sbox          # ending position for each box
        istop = Int(clamp(floor(rstop), 0, ns1))
        frac1 = rstart-istart
        frac2 = 1.0 - (rstop-istop)

        # add pixel values from istart to istop and subtract fractional pixel from istart to start and
        # fractional pixel from rstop to istop

        if istart == istop
            result[:,i+1] .= (1 .- frac1 .- frac2).*temp[:,istart+1]
        else
            result[:,i+1] .= sumdim(temp[:,istart+1:istop+1], 2) .- frac1.*temp[:,istart+1] .- frac2.*temp[:,istop+1]
        end
    end
    return transpose(result) .* (total ? 1.0 : 1 ./ (sbox.*lbox))

end


function make_python_wcs(x_cent::T, y_cent::T, ra_cent::T, dec_cent::T, x_scale::T, y_scale::T) where {T<:Real}

    wcs = py_wcs.WCS(naxis=2)

    # First set the coordinate system
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Central coordinate in pixel space and RA/Dec space
    wcs.wcs.crpix = [x_cent, y_cent]
    wcs.wcs.crval = [ra_cent, dec_cent]

    # Scale in deg/pix
    wcs.wcs.cdelt = [x_scale, y_scale]

    # Units 
    wcs.wcs.cunit = ["deg", "deg"]

    # Coordinate transform matrix 
    # (negative in the first position because RA increases to the left)
    wcs.wcs.pc = [-1.0 0.0; 0.0 1.0]

    wcs

end

