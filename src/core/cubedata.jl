#=
This is the file that handles loading in and preparing 3D IFU cube data to be fit. It contains
structs for holding the 3D IFU cubes, and functions that apply certain corrections to the data, such
as shifting the wavelength to the rest frame, and masking out bad spaxels.  The first step in any
fitting sequence should be loading in the data, likely with the "from_fits" function to load from a 
Flexible Image Transfer System (FITS) file, followed by applying the two aforementioned corrections
using the "correct" function.  The data should then be handed off to the structs/functions in the 
"CubeFit" file, which handles actually fitting the cubes.
=#

# quick check for determining whether the intensities are per unit frequency or per unit wavelength 
@enum IntensityUnits begin
    PerFrequency
    PerWavelength
end

# latex string representations of units 
get_latex_str(u) = latexify(u)
get_latex_str(::typeof(u"erg/s/cm^2/Hz/sr")) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{Hz}^{-1}\,\mathrm{sr}^{-1}$"
get_latex_str(::typeof(u"erg/s/cm^2/angstrom/sr")) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}\,\mathrm{sr}^{-1}$"
get_latex_str(::typeof(u"erg/s/cm^2/μm/sr")) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\mu{}m}^{-1}\,\mathrm{sr}^{-1}$"

# Functions approximating the MIRI MRS line-spread function and point-spread function
# Note: the inputs of both are the OBSERVED-FRAME wavelengths
@inline mrs_lsf(λ::Quantity) = @. C_KMS / (4603 - 128*(λ/u"μm") + 10^(-7.4*(λ/u"μm")))
@inline mrs_psf(λ::Quantity) = @. (0.033*(λ/u"μm") + 0.016) * u"arcsecond"

# very simple parser; bound to not work in all cases but that's a lotta work
function fits_unitstr_to_unitful(unit_string::String)
    unit_string = replace(unit_string, "u" => "μ", "." => "*")
    i = 1
    while i ≤ lastindex(unit_string)
        # insert an exponent before each number in the string
        char = unit_string[i]
        if char in ('-','+') || (isdigit(char) && !(unit_string[i-1] in ['-','+','^']))
            unit_string = unit_string[1:prevind(unit_string, i)] * "^" * unit_string[i:end]
            i = nextind(unit_string, i)
        end
        i = nextind(unit_string, i)
    end
    unit_string
end

############################## DATACUBE STRUCTURE AND FUNCTIONS ####################################


""" 
    DataCube(λ, I, σ[, mask, Ω, α, δ, psf_fwhm, wcs, channel, band, rest_frame, masked])

An object for holding 3D IFU spectroscopy data. 

# Fields
- `λ`: 1D array of wavelengths, in μm or angstroms
- `I`: 3D array of intensity, in erg/s/cm^2/Hz/sr, erg/s/cm^2/ang/sr, or erg/s/cm^2/μm/sr
- `σ`: 3D array of uncertainties, in the same units as I
- `mask`: 3D array of booleans acting as a mask for the flux/error data
- `psf_model`: 3D model of the point-spread function which should be normalized such that the sum
    of the 2D point-spread function at each wavelength slice integrates to 1. Defaults to `nothing`, but can be generated using the
    `generate_psf_model` function from `psf.jl`.
- `Ω`: the solid angle subtended by each spaxel, in steradians
- `α`: the right ascension of the observation, in decimal degrees
- `δ`: the declination of the observation, in decimal degrees
- `θ_sky`: the rotation angle between the MIRI/MRS IFU axes and the RA/Dec sky axes, in radians
- `psf`: the FWHM of the spatial point-spread function in arcseconds as a function of (observed-frame) wavelength in microns
- `lsf`: the FWHM of the spectral line-spread function in km/s as a function of (observed-frame) wavelength in microns
- `wcs`: a World Coordinate System conversion object, optional
- `channel`: the channel name of the observation
- `band`: the band of the observation, i.e. "SHORT", "MEDIUM", "LONG", or "MULTIPLE"
- `nx`: the length of the x dimension of the cube
- `ny`: the length of the y dimension of the cube
- `nz`: the length of the z dimension of the cube
- `spectral_region`: which spectral region does the DataCube cover
- `rest_frame`: whether or not the DataCube wavelength vector is in the rest-frame
- `masked`: whether or not the DataCube has been masked
- `vacuum_wave`: whether or not the wavelength vector is in vacuum wavelengths; if false, it is assumed to be air wavelengths
- `sky_aligned`: whether or not the data cube is aligned to the sky (RA/Dec) axes or the IFU (α/β) axes
- `voronoi_bins`: a map giving unique labels to each spaxel which place them within voronoi bins
"""
mutable struct DataCube{T, S} where {
        T<:Vector{<:Union{typeof(1.0u"μm"),typeof(1.0u"angstrom")}},   # units must be either angstroms or microns
        S<:Array{<:Union{                                              # units must be one of:
            typeof(1.0u"erg*s^-1*cm^-2*Hz^-1*sr^-1"),                  #    erg s^-1 cm^-2 Hz^-1 sr^-1
            typeof(1.0u"erg*s^-1*cm^-2*μm^-1*sr^-1"),                  #    erg s^-1 cm^-2 μm^-1 sr^-1
            typeof(1.0u"erg*s^-1*cm^-2*angstrom^-1*sr^-1")             #    erg s^-1 cm^-2 Å^-1 sr^-1
        }, 3}
    }

    λ::T
    I::S
    σ::S
    mask::BitArray{3}

    psf_model::Union{Array{<:Real,3},Nothing}   # unitless

    Ω::typeof(1.0u"sr")        
    α::typeof(1.0u"°")         
    δ::typeof(1.0u"°")    
    θ_sky::typeof(1.0u"rad") 
    psf::Vector{typeof(1.0u"arcsecond")}
    lsf::Vector{typeof(1.0u"km/s")}

    wcs::Union{WCSTransform,Nothing}

    channel::String
    band::String

    nx::Integer
    ny::Integer
    nz::Integer

    spectral_region::SpectralRegion
    intensity_units::IntensityUnits
    rest_frame::Bool
    masked::Bool
    vacuum_wave::Bool
    sky_aligned::Bool

    voronoi_bins::Union{Matrix{<:Integer},Nothing}

    # This is the constructor for the DataCube struct; see the DataCube docstring for details
    function DataCube(λ::T, I::S, σ::S, mask::Union{BitArray{3},Nothing}=nothing, psf_model::Union{Array{<:Real,3},Nothing}=nothing, 
        Ω=NaN*u"sr", α=NaN*u"°", δ=NaN*u"°", θ_sky=NaN*u"rad", psf::Union{U,Nothing}=nothing, lsf::Union{V,Nothing}=nothing, 
        wcs::Union{WCSTransform,Nothing}=nothing, channel::String="Generic Channel", band::String="Generic Band", n_channels::Int=1,
        rest_frame::Bool=false, masked::Bool=false, vacuum_wave::Bool=true, sky_aligned::Bool=false, 
        voronoi_bins::Union{Matrix{<:Integer},Nothing}=nothing) where {
            T<:Vector{<:Quantity},S<:Array{<:Quantity,3},U<:Vector{<:Quantity},V<:Vector{<:Quantity}
            }

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

        # Make sure units are consistent with each other
        I_unit = unit(I[1])
        @assert I_unit == unit(σ[1]) "The intensity and error cubes must have the same units!"

        # If units arent in the code's standard, try to convert them 
        if dimension(I_unit) == u"𝐌*𝐓^-2"
            new_I_unit = u"erg*s^-1*cm^-2*Hz^-1*sr^-1"
            intensity_units = PerFrequency
        elseif dimension(I_unit) == u"𝐌*𝐋^-1*𝐓^-3"
            λ_unit = unit(λ[1])
            new_I_unit = uparse("erg*s^-1*cm^-2*$(λ_unit)^-1*sr^-1", unit_context=UnitfulAstro)
            intensity_units = PerWavelength
        end
        I = uconvert.(new_I_unit, I)
        σ = uconvert.(new_I_unit, σ)

        nx, ny, nz = size(I)

        # If no mask is given, make the default mask to be all falses (i.e. don't mask out anything)
        if isnothing(mask)
            @info "DataCube initialization: No mask was given, all spaxels will be unmasked"
            mask = falses(size(I))
        else
            @assert size(mask) == size(I) "The mask must be the same size as the intensity cube!"
        end
        if !isnothing(psf_model)
            @assert size(psf_model) == size(I) "The PSF model must be the same size as the intensity cube!"
        end

        λlim = extrema(λ)
        # the range parameter mainly determines what to do about extinction curves
        # since calzetti and CCM are not defined past ~2-3 um
        λrange = if λlim[1] < 2.2u"μm" && λlim[2] > 2.2u"μm"
            UVOptIR
        elseif λlim[1] > 2.2u"μm"
            Infrared
        else
            UVOptical
        end
        spectral_region = SpectralRegion(λlim, [], n_channels, λrange)

        # Return a new instance of the DataCube struct
        new(λ, I, σ, mask, psf_model, Ω, α, δ, θ_sky, psf, lsf, wcs, channel, band, nx, ny, nz, spectral_region, intensity_units, 
            rest_frame, masked, vacuum_wave, sky_aligned, voronoi_bins)
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
    Ω = hdr["PIXAR_SR"] * u"sr"
    # Intensity and error arrays
    # JWST cubes come in units of MJy/sr
    bunit = uparse(fits_unitstr_to_unitful(hdr["BUNIT"]); unit_context=UnitfulAstro)
    Iν = read(hdu["SCI"]) * bunit
    σI = read(hdu["ERR"]) * bunit

    @debug "FITS data dimensions: ($nx, $ny, $nz), solid angle per spaxel: $Ω"

    # Construct 3D World coordinate system to convert from pixels to (RA,Dec,wave) and vice versa
    wcs = WCS.from_header(read_header(hdu["SCI"], String))[1]
    cunit3 = uparse(fits_unitstr_to_unitful(hdr["CUNIT3"]); unit_context=UnitfulAstro)

    # Wavelength vector
    λ = try
        read(hdu["AUX"], "wave")
    catch
        hdr["CRVAL3"] .+ hdr["CDELT3"] .* (collect(0:hdr["NAXIS3"]-1) .+ hdr["CRPIX3"] .- 1)
    end * cunit3
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
    name = hdr0["TARGNAME"]           # name of the target
    ra = hdr0["TARG_RA"] * u"°"       # right ascension in deg
    dec = hdr0["TARG_DEC"] * u"°"     # declination in deg
    channel = string(hdr0["CHANNEL"]) # MIRI channel (1-4)
    band = hdr0["BAND"]               # MIRI band (long,med,short,multiple)

    # Sky rotation angle
    cosθ = -hdr["PC1_1"]    # negative because of the flipped RA axis (RA increases to the left)
    sinθ = -hdr["PC1_2"]    # negative because of how the rotation matrix is defined
    θ_sky = atan(sinθ, cosθ) * u"rad"
    sky_aligned = iszero(θ_sky)

    @debug """\n
    ##################################################################
    #################### TARGET INFORMATION ##########################
    ##################################################################
    name: \t\t $name
    RA: \t\t\t $ra
    Dec: \t\t\t $dec
    Rotation angle: \t\t $(uconvert(u"°", θ_sky))
    Channel: \t $channel
    Band: \t\t $band
    ##################################################################
    """

    # Get the PSF models (if any)
    psf_model = try
        read(hdu["PSF"])
    catch
        nothing
    end

    # Get the rough PSF FWHM in arcseconds
    psf = try
        read(hdu["AUX"], "psf") * u"arcsecond"
    catch
        mrs_psf(λ)
    end
    lsf = try
        read(hdu["AUX"], "lsf") * u"km/s"
    catch
        mrs_lsf(λ)
    end

    n_channels = nothing
    if band in ("SHORT", "MEDIUM", "LONG")
        n_channels = 1
    end

    if haskey(hdr0, "N_CHANNELS")
        n_channels = parse(Int, hdr0["N_CHANNELS"])
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

    if isnothing(n_channels)
        error("Please only input single-band data cubes! You can combine them into multi-channel cubes using LOKI routines.")
    end

    DataCube(λ, Iν, σI, mask, psf_model, Ω, ra, dec, θ_sky, psf, lsf, wcs, channel, band, n_channels, rest_frame, masked, vacuum_wave, sky_aligned)
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

        z1 = 1 + z

        # Wavelength is always shorter in the rest frame
        cube.λ = @. cube.λ / z1

        # To conserve flux, we have to either multiply or divide by the 1+z factor depending on whether it's
        # measure per unit wavelength or per unit frequency
        if cube.intensity_units == PerFrequency
            cube.I = @. cube.I / z1
            cube.σ = @. cube.σ / z1
        elseif cube.intensity_units == PerWavelength
            cube.I = @. cube.I * z1
            cube.σ = @. cube.σ * z1
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
        # Convert to vacuum wavelengths (airtovac uses Angstroms, 1 Angstrom = 10^-4 μm)
        λ_unit = unit(cube.λ)
        # airtovac always works in angstroms, so we need to convert to them and back
        cube.λ = uconvert(λ_unit, airtovac.(uconvert(u"angstrom", cube.λ) |> ustrip)*u"angstrom")
        # Optionally resample back onto a linear grid
        if linear_resample
            λlin = range(minimum(cube.λ), maximum(cube.λ), length=length(cube.λ))
            cube.I, cube.σ, cube.mask = resample_flux_permuted3D(λlin, cube.λ, cube.I, cube.σ, cube.mask)
            if !isnothing(cube.psf_model)
                cube.psf_model = resample_flux_permuted3D(λlin, cube.λ, cube.psf_model)
            end
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

        cube.I[cube.mask] .*= NaN
        cube.σ[cube.mask] .*= NaN
        if !isnothing(cube.psf_model)
            cube.psf_model[cube.mask] .*= NaN
        end
        cube.masked = true

    end

    cube

end


"""
    log_rebin!(cube, factor)

Rebin a DataCube onto a logarithmically spaced wavelength vector, conserving flux.
Optionally input a rebinning factor > 1 to resample onto a coarser wavelength grid.
"""
function log_rebin!(cube::DataCube, factor::Integer=1)

    # check if it is already log-rebinned
    log_check = (cube.λ[2]/cube.λ[1]) ≈ (cube.λ[end]/cube.λ[end-1])

    # rebin onto a logarithmically spaced wavelength grid
    if !log_check
        dλ = (maximum(cube.λ) - minimum(cube.λ)) / length(cube.λ) * factor
        linλ = minimum(cube.λ):dλ:maximum(cube.λ)
        lnλ = get_logarithmic_λ(ustrip.(linλ)) * unit(linλ[1])
        cube.I, cube.σ, cube.mask = resample_flux_permuted3D(lnλ, cube.λ, cube.I, cube.σ, cube.mask)
        if !isnothing(cube.psf_model)
            cube.psf_model = resample_flux_permuted3D(lnλ, cube.λ, cube.psf_model)
        end
        cube.psf = Spline1D(ustrip.(cube.λ), ustrip.(cube.psf), k=1, bc="extrapolate")(ustrip.(lnλ)) * unit(cube.psf[1])
        cube.lsf = Spline1D(ustrip.(cube.λ), ustrip.(cube.lsf), k=1, bc="extrapolate")(ustrip.(lnλ)) * unit(cube.lsf[1])
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

    λ = ustrip.(cube.λ)
    @info "Interpolating NaNs in cube with channel $(cube.channel), band $(cube.band):"

    for index ∈ CartesianIndices(selectdim(cube.I, 3, 1))

        I = ustrip.(cube.I[index, :])
        σ = ustrip.(cube.σ[index, :])
        psf = nothing
        if !isnothing(cube.psf_model)
            psf = cube.psf_model[index, :]
        end

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
            cube.I[index, filt] .= Spline1D(λ[isfinite.(I)], I[isfinite.(I)], λknots, k=1, bc="extrapolate")(λ[filt]) * unit(cube.I[1])
            cube.σ[index, filt] .= Spline1D(λ[isfinite.(σ)], σ[isfinite.(σ)], λknots, k=1, bc="extrapolate")(λ[filt]) * unit(cube.σ[1])

            # Do for PSF models as well
            if !isnothing(psf)
                cube.psf_model[index, filt] .= Spline1D(λ[isfinite.(psf)], psf[isfinite.(psf)], λknots, k=1, bc="extrapolate")(λ[filt])
            end

        end 
    end

    return
end


"""
    rotate_to_sky_axes!(cube)

Rotate an IFU-aligned DataCube object to the sky axes using the `θ_sky` value obtained
from the FITS headers.
"""
function rotate_to_sky_axes!(cube::DataCube)

    if !cube.sky_aligned

        @debug "Rotating cube to sky axes"

        # negative because `imrotate` rotates clockwise
        out_shape2d = size(imrotate(cube.I[:,:,1], ustrip(cube.θ_sky)))

        # prepare output arrays
        I_rot = zeros(eltype(cube.I), out_shape2d..., size(cube.I, 3))
        σ_rot = zeros(eltype(cube.σ), out_shape2d..., size(cube.σ, 3))
        mask_rot = falses(out_shape2d..., size(cube.mask, 3))
        psf_rot = nothing
        if !isnothing(cube.psf_model)
            psf_rot = zeros(eltype(cube.psf_model), out_shape2d..., size(cube.psf_model, 3))
        end

        # rotate
        for k ∈ axes(I_rot, 3)
            I_rot[:, :, k] = imrotate(cube.I[:, :, k], ustrip(cube.θ_sky))
            σ_rot[:, :, k] = imrotate(cube.σ[:, :, k], ustrip(cube.θ_sky))
            mask_rot[:, :, k] = imrotate(cube.mask[:, :, k], ustrip(cube.θ_sky)) .> 0.
            if !isnothing(psf_rot)
                psf_rot[:, :, k] = imrotate(cube.psf_model[:, :, k], ustrip(cube.θ_sky))
            end
        end

        # set the new parameters
        cube.I = I_rot
        cube.σ = σ_rot
        cube.mask = mask_rot
        cube.psf_model = psf_rot

        # update WCS parameters
        cube.wcs.pc = [-1. 0. 0.; 0. 1. 0.; 0. 0. 1.]
        cube.wcs.cdelt = abs.(cube.wcs.cdelt)
        cube.wcs.crpix = [((size(cube.I)[1:2]./2).+0.5)...; cube.wcs.crpix[3]]

        cube.nx = size(I_rot, 1)
        cube.ny = size(I_rot, 2)
        cube.nz = size(I_rot, 3)

    end
    cube.sky_aligned = true

end


"""
    calculate_statistical_errors!(cube)

This function calculates the 'statistical' errors of the given IFU cube, replacing its error cube.
The statistical errors are defined as the standard deviation of the residuals between the flux and a cubic spline
fit to the flux, within a small window (60 pixels). Emission lines are masked out during this process.
"""
function calculate_statistical_errors!(cube::DataCube, Δ::Union{Integer,Nothing}=nothing, 
    n_inc_thresh::Union{Integer,Nothing}=nothing, thresh::Union{Real,Nothing}=nothing,
    overrides::Union{Vector{Tuple{Quantity{<:Real, u"𝐋"},Quantity{<:Real, u"𝐋"}}},Nothing}=nothing; 
    mask_width::typeof(1.0u"km/s")=1000u"km/s", median::Bool=false)

    λ = cube.λ
    if isnothing(Δ)
        Δ = 20
    end
    if isnothing(n_inc_thresh)
        n_inc_thresh = 7
    end
    if isnothing(thresh)
        thresh = 3.0
    end

    if isnothing(overrides)
        _, _, _, cent_vals = parse_lines(cube.spectral_region)
        overrides = Vector{Tuple{eltype(cent_vals),eltype(cent_vals)}}()
        for λi in cent_vals
            push!(overrides, λi .* (1-mask_width/C_KMS, 1+mask_width/C_KMS))
        end
    end

    @info "Calculating statistical errors for each spaxel..."
    @showprogress for spaxel ∈ CartesianIndices(size(cube.I)[1:2])

        # Get the flux/error for this spaxel
        I = cube.I[spaxel, :]
        σ = cube.σ[spaxel, :]

        # Perform a cubic spline fit, also obtaining the line mask
        mask_lines, I_spline = continuum_cubic_spline(λ, I, σ, overrides; do_err=false)
        mask_bad = cube.mask[spaxel, :]
        mask = mask_lines .| mask_bad

        l_mask = sum(.~mask)
        if iszero(l_mask)
            continue
        end
        # Statistical uncertainties based on the local RMS of the residuals with a cubic spline fit
        σ_stat = zeros(eltype(σ), l_mask)
        for i in 1:l_mask
            σ_stat[i] = std(I[.~mask][max(1,i-30):min(l_mask,i+30)] .- I_spline[.~mask][max(1,i-30):min(l_mask,i+30)])
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
    signal = clamp.(signal, 0*unit(signal), Inf*unit(signal))
    # perform voronoi rebinning
    bin_numbers, _, _, _, _, _, _, _ = py_vorbin.voronoi_2d_binning(x, y, ustrip.(signal), ustrip.(noise), target_SN, 
        pixelsize=1.0, cvt=true, wvt=true, plot=false)
    # reformat bin numbers as a 2D array so that we don't need the x/y vectors anymore
    voronoi_bins = zeros(Int, size(cube.I)[1:2])
    for i in eachindex(bin_numbers)
        voronoi_bins[x[i], y[i]] = bin_numbers[i] + 1  # python 0-based indexing -> julia 1-based indexing
    end
    # Set the voronoi_bins value in the cube object
    cube.voronoi_bins = voronoi_bins

end


function get_physical_scales(data::DataCube, cosmo::Union{Cosmology.AbstractCosmology,Nothing}=nothing, 
    z::Union{Real,Nothing}=nothing)

    # Angular and physical scalebars
    pix_as = uconvert(u"arcsecond", sqrt(data.Ω))
    n_pix = 1.0/ustrip(pix_as)

    if !isnothing(cosmo) && !isnothing(z)
        @debug "Using angular diameter distance $(angular_diameter_dist(cosmo, z))"
        # Calculate in pc
        dA = angular_diameter_dist(u"pc", cosmo, z)
        # Remove units
        dA = uconvert(NoUnits, dA/u"pc")
        # l = d * theta (") where theta is chosen as 1/5 the horizontal extent of the image
        l = dA * ustrip(uconvert(u"rad", size(data.I, 1) * pix_as / 5))
        # Round to a nice even number
        l = round(typeof(1u"pc"), l, sigdigits=1)
        # new angular size for this scale
        θ = l / dA * u"rad"
        θ_as = round(u"arcsecond", θ, digits=1)  # should be close to the original theta, by definition
        n_pix = 1.0/sqrt(data.Ω) * θ   # number of pixels = (pixels per radian) * radians
        new_unit = u"pc"
        # convert to kpc if l is more than 1000 pc
        if l ≥ 1e3u"pc"
            new_unit = u"kpc"
        elseif l ≥ 1e6u"pc"
            new_unit = u"Mpc"
        elseif l ≥ 1e9u"pc"
            new_unit = u"Gpc"
        end
        l = uconvert(new_unit, l)
        l_val = ustrip(l)
        scalebar_text_dist = cosmo.h ≈ 1.0 ? L"%$(l_val)$h^{-1}$ %$(new_unit)" : L"$%$(l_val)$ %$(new_unit)"
    else
        θ_as = round(u"arcsecond", size(data.I, 1) * pix_as / 5, digits=1)
        n_pix = 1.0/sqrt(data.Ω) * uconvert(u"rad", θ_as)
        scalebar_text_dist = ""
    end

    scalebar_text_ang = L"$\ang[angle-symbol-over-decimal]{;;%$(ustrip(θ_as))}$"
    if θ_as > 60u"arcsecond"
        θ_as = round(u"arcminute", θ_as, digits=1)  # convert to arcminutes
        scalebar_text_ang = L"$\ang[angle-symbol-over-decimal]{;%$(ustrip(θ_as));}$"
    end

    n_pix, scalebar_text_dist, scalebar_text_ang
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
    logₑ::Union{Integer,Nothing}=nothing, colormap=py_colormap.cubehelix, name::Union{String,Nothing}=nothing, 
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
    n_pix, scalebar_text_dist, scalebar_text_ang = get_physical_scales(data, cosmo, z)
    colormap.set_bad(color="k")

    fig = plt.figure(figsize=intensity && err ? (12, 6) : (12,12))
    if intensity
        # Plot intensity on a 2D map
        ax1 = fig.add_subplot(121)
        ax1.set_title(isnothing(name) ? "" : name)
        cdata = ax1.imshow(I', origin=:lower, cmap=colormap, vmin=nanquantile(I, 0.01), vmax=nanquantile(I, 0.99))
        fig.colorbar(cdata, ax=ax1, fraction=0.046, pad=0.04, 
            label=(isnothing(logᵢ) ? "" : L"$\log_{%$logᵢ}$(") * L"$I_{%$sub}\,/\,$" * unit_str * (isnothing(logᵢ) ? "" : ")"))
        ax1.axis(:off)
        ax1.tick_params(which="both", axis="both", direction="in")
        
        if !isnothing(z) && !isnothing(cosmo)
            scalebar_1 = py_anchored_artists.AnchoredSizeBar(ax1.transData, n_pix, scalebar_text_dist, "upper center", pad=0, borderpad=0, 
                color="w", frameon=false, size_vertical=0.1, label_top=false, bbox_to_anchor=(0.17, 0.1), bbox_transform=ax1.transAxes)
            ax1.add_artist(scalebar_1)
        end
        scalebar_2 = py_anchored_artists.AnchoredSizeBar(ax1.transData, n_pix, scalebar_text_ang, "lower center", pad=0, borderpad=0, 
            color="w", frameon=false, size_vertical=0.1, label_top=true, bbox_to_anchor=(0.17, 0.1), bbox_transform=ax1.transAxes)
        ax1.add_artist(scalebar_2)

        r = (isnothing(slice) ? median(data.psf) : data.psf[slice]) / pix_as / 2
        psf = plt.Circle(size(I) .* (0.93, 0.05) .+ (-r, r), r, color="w")
        ax1.add_patch(psf)
        ax1.annotate("PSF", size(I) .* (0.93, 0.05) .+ (-r, 2.5r + 1.75), ha=:center, va=:center, color="w")    

        if !isnothing(aperture)
            patches = get_patches(aperture)
            for patch in patches
                ax1.add_patch(patch)
            end
        end

    end

    if err
        # Plot error on a 2D map
        ax2 = fig.add_subplot(122)
        ax2.set_title(isnothing(name) ? "" : name)
        cdata = ax2.imshow(σ', origin=:lower, cmap=colormap, vmin=nanquantile(σ, 0.01), vmax=nanquantile(σ, 0.99))
        fig.colorbar(cdata, ax=ax2, fraction=0.046, pad=0.04,
            label=isnothing(logₑ) ? L"$\sigma_{I_{%$sub}}\,/\,$" * unit_str : L"$\sigma_{\log_{%$logᵢ}I_{%$sub}}$")
        ax2.axis(:off)
        ax2.tick_params(which="both", axis="both", direction="in")

        if !isnothing(z) && !isnothing(cosmo)
            scalebar_1 = py_anchored_artists.AnchoredSizeBar(ax2.transData, n_pix, scalebar_text_dist, "upper center", pad=0, borderpad=0, 
                color="w", frameon=false, size_vertical=0.1, label_top=false, bbox_to_anchor=(0.17, 0.1), bbox_transform=ax2.transAxes)
            ax2.add_artist(scalebar_1)
        end
        scalebar_2 = py_anchored_artists.AnchoredSizeBar(ax2.transData, n_pix, scalebar_text_ang, "lower center", pad=0, borderpad=0, 
            color="w", frameon=false, size_vertical=0.1, label_top=true, bbox_to_anchor=(0.17, 0.1), bbox_transform=ax2.transAxes)
        ax2.add_artist(scalebar_2)

        r = (isnothing(slice) ? median(data.psf) : data.psf[slice]) / pix_as / 2
        psf = plt.Circle(size(I) .* (0.93, 0.05) .+ (-r, r), r, color="w")
        ax1.add_patch(psf)
        ax1.annotate("PSF", size(I) .* (0.93, 0.05) .+ (-r, 2.5r + 1.75), ha=:center, va=:center, color="w")    

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
- `θ_sky::Real=NaN`: the rotation angle between the MIRI/MRS IFU axes and the RA/Dec sky axes, in radians
- `instrument::String="Generic Instrument"`: the instrument name, i.e. "MIRI"
- `detector::String="Generic Detector"`: the detector name, i.e. "MIRIFULONG"
- `spectral_region::Symbol`: Either :MIR for mid-infrared data or :OPT for optical data
- `rest_frame::Bool=false`: whether or not the individual DataCubes have been converted to the rest frame
- `masked::Bool=false`: whether or not the individual DataCubes have been masked
- `vacuum_wave::Bool=true`: whether or not the wavelengths are specified in vacuum wavelengths; if false, they are assumed
- `sky_aligned::Bool=false`: whether or not the data cube is aligned to the sky (RA/Dec) axes or the IFU (α/β) axes
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
    sky_aligned::Bool

    function Observation(channels::Dict{Any,DataCube}=Dict{Any,DataCube}(), name::String="Generic Observation",
        z::Real=NaN, α::Real=NaN, δ::Real=NaN, instrument::String="Generic Instrument", 
        detector::String="Generic Detector", spectral_region::Symbol=:MIR, rest_frame::Bool=false, masked::Bool=false,
        vacuum_wave::Bool=true, sky_aligned::Bool=false)

        new(channels, name, z, α, δ, instrument, detector, spectral_region, rest_frame, masked, vacuum_wave, sky_aligned)
    end
    
end


"""
    save_fits(path, obs[, channels])

Save a pre-processed Observation object as a FITS file in a format that can be read by the "from_fits" function.
This saves time upon re-running the code assuming one wishes to keep the same pre-processing.
"""
save_fits(path::String, obs::Observation) = save_fits(path, obs, collect(keys(obs.channels)))
save_fits(path::String, obs::Observation, channel) = save_fits(path, obs, [channel])

function save_fits(path::String, obs::Observation, channels::Vector)
    for channel in channels

        # Header information
        hdr = FITSHeader(
            Vector{String}(["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "PIXAR_SR", "TARG_RA", "TARG_DEC", "INSTRUME", "DETECTOR", 
                "ROTANGLE", "N_CHANNELS", "RESTFRAM", "MASKED", "VACWAVE", "DATAMODL", "NAXIS1", "NAXIS2", "NAXIS3", "WCSAXES", "CDELT1", "CDELT2", "CDELT3", 
                "CTYPE1", "CTYPE2", "CTYPE3", "CRPIX1", "CRPIX2", "CRPIX3", "CRVAL1", "CRVAL2", "CRVAL3", "CUNIT1", "CUNIT2", "CUNIT3", 
                "PC1_1", "PC1_2", "PC1_3", "PC2_1", "PC2_2", "PC2_3", "PC3_1", "PC3_2", "PC3_3"]),

            [obs.name, obs.z, string(channel), obs.channels[channel].band, obs.channels[channel].Ω, 
                obs.α, obs.δ, obs.instrument, obs.detector, obs.channels[channel].θ_sky, nchannels(obs.channels[channel].spectral_region), 
                obs.rest_frame, obs.masked, obs.vacuum_wave, "IFUCubeModel", size(obs.channels[channel].I, 1), size(obs.channels[channel].I, 2), 
                size(obs.channels[channel].I, 3), obs.channels[channel].wcs.naxis, 
                obs.channels[channel].wcs.cdelt[1], obs.channels[channel].wcs.cdelt[2], obs.channels[channel].wcs.cdelt[3],
                obs.channels[channel].wcs.ctype[1], obs.channels[channel].wcs.ctype[2], obs.channels[channel].wcs.ctype[3],
                obs.channels[channel].wcs.crpix[1], obs.channels[channel].wcs.crpix[2], obs.channels[channel].wcs.crpix[3],
                obs.channels[channel].wcs.crval[1], obs.channels[channel].wcs.crval[2], obs.channels[channel].wcs.crval[3],
                obs.channels[channel].wcs.cunit[1], obs.channels[channel].wcs.cunit[2], obs.channels[channel].wcs.cunit[3],
                obs.channels[channel].wcs.pc[1,1], obs.channels[channel].wcs.pc[1,2], obs.channels[channel].wcs.pc[1,3],
                obs.channels[channel].wcs.pc[2,1], obs.channels[channel].wcs.pc[2,2], obs.channels[channel].wcs.pc[2,3],
                obs.channels[channel].wcs.pc[3,1], obs.channels[channel].wcs.pc[3,2], obs.channels[channel].wcs.pc[3,3]],

            Vector{String}(["Target name", "Target redshift", "Channel", "Band",
                "Solid angle per pixel (rad.)", "Right ascension of target (deg.)", "Declination of target (deg.)",
                "Instrument name", "Detector name", "rotation angle to sky axes", "number of combined channels in the data", 
                "data in rest frame", "data masked", "vacuum wavelengths", "data model", "length of the first axis", 
                "length of the second axis", "length of the third axis", "number of World Coordinate System axes", 
                "first axis increment per pixel", "second axis increment per pixel", "third axis increment per pixel",
                "first axis coordinate type", "second axis coordinate type", "third axis coordinate type",
                "axis 1 coordinate of the reference pixel", "axis 2 coordinate of the reference pixel", "axis 3 coordinate of the reference pixel",
                "first axis value at the reference pixel", "second axis value at the reference pixel", "third axis value at the reference pixel",
                "first axis units", "second axis units", "third axis units",
                "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element",
                "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element",
                "linear transformation matrix element", "linear transformation matrix element", "linear transformation matrix element"])
        )
        FITS(joinpath(path, "$(replace(obs.name, " " => "_")).channel$(channel)$(obs.rest_frame ? ".rest_frame" : "").fits"), "w") do f
            @info "Writing FITS file from Observation object"

            write(f, Vector{Int}(); header=hdr)                          # Primary HDU (empty)
            write(f, obs.channels[channel].I; name="SCI", header=hdr)   # Data HDU
            write(f, obs.channels[channel].σ; name="ERR")   # Error HDU
            write(f, UInt8.(obs.channels[channel].mask); name="DQ")  # Mask HDU
            if !isnothing(obs.channels[channel].psf_model)
                write(f, obs.channels[channel].psf_model; name="PSF") # PSF HDU
            end
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
function from_fits(filenames::Vector{String}, z::Real=0.)::Observation


    # Grab object information from the FITS header of the first file
    channels = Dict{Any,DataCube}()
    hdu = FITS(filenames[1])
    hdr = read_header(hdu[1])
    name = hdr["TARGNAME"]
    ra = hdr["TARG_RA"]
    dec = hdr["TARG_DEC"]
    inst = hdr["INSTRUME"]
    detector = hdr["DETECTOR"]
    redshift = z
    if haskey(hdr, "REDSHIFT")
        redshift = hdr["REDSHIFT"]
    end
    if redshift == 0.
        error("Did not find a redshift for this object! Redshift information must be included!")
    end

    @debug """\n
    Initializing Observation struct for $name, with redshift z=$redshift
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
    sky_aligned = all([channels[ch].sky_aligned for ch in keys(channels)])

    Observation(channels, name, redshift, ra, dec, inst, detector, spectral_region, rest_frame, masked, vacuum_wave, sky_aligned)
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
    rotate_to_sky_axes!(obs)

Rotate each DataCube to its sky axes orientation.

# Arguments
- `obs::Observation`: The Observation object to rotate
"""
function rotate_to_sky_axes!(obs::Observation)

    @debug """\n
    Rotating observation of $(obs.name) to sky axes
    ###############################################
    """

    # Loop through the channels and call the individual DataCube method of the rotate_to_sky_axes! function
    for k ∈ keys(obs.channels)
        rotate_to_sky_axes!(obs.channels[k])
    end
    obs.sky_aligned = true

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
    adjust_wcs_alignment(obs, channels; box_size_as=1.5)

Adjust the WCS alignments of each channel such that they match.
This is performed in order to remove discontinuous jumps in the flux level when crossing between
sub-channels.
"""
function adjust_wcs_alignment!(obs::Observation, channels; box_size_as::Float64=1.5)

    @assert obs.spectral_region == :MIR "The adjust_wcs_alignment! function is only supported for MIR cubes!"

    @info "Aligning World Coordinate Systems for channels $channels..."

    # Prepare array of centroids for each channel
    c_coords = zeros(2*length(channels)-2, 2)
    offsets = zeros(length(channels), 2)

    # box_size in arcseconds -- adjust so that it's the same *physical* size in all channels
    box_sizes = zeros(Int, length(channels))
    for i in eachindex(channels)
        as_per_pix = sqrt(obs.channels[channels[i]].Ω) * 180/π * 3600
        box_sizes[i] = fld(box_size_as, as_per_pix)
    end

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
            box_half = fld(box_sizes[i], 2)
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

            # Convert pixel coordinates to physical coordinates using the WCS transformation 
            c_coords[k, :] .= pix_to_world(wcs, [x_cent; y_cent; 1.0])[1:2]
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
        ch_data.wcs.crval = [ch_data.wcs.crval[1:2] .- offset; ch_data.wcs.crval[3]]
        offset_pix = offset ./ ch_data.wcs.cdelt[1:2]
        @info "The centroid offset relative to channel $(channels[1]) for channel $channel is " *
            "($(@sprintf "%.2g" offset_pix[1]), $(@sprintf "%.2g" offset_pix[2])) spaxels"
    end

end


"""
    reproject_channels!(obs, channels; order, output_wcs_frame)

Reproject all channels onto a common WCS grid. Modifies the Observation object in-place.

# Arguments
- `obs::Observation`: The Observation object to rebin
- `channels=nothing`: The list of channels to be rebinned. If nothing, rebin all channels.
- `order::Integer=1`: The order of interpolation for the reprojection. 
        -1 = do not do any reprojection (all input channels must be on the same WCS grid)
        0 = nearest-neighbor
        1 = linear
        2 = quadratic
- `output_wcs_frame::Integer=1`: Which WCS frame to project the inputs onto. Defaults to 1 (the first channel in channels).
"""
function reproject_channels!(obs::Observation, channels::Vector; order::Integer=1, output_wcs_frame::Integer=1)

    # Output angular size
    Ω_out = obs.channels[channels[output_wcs_frame]].Ω
    # Pixel sizes in each channel (in arcseconds)
    do_psf_model = all([!isnothing(obs.channels[ch_i].psf_model) for ch_i ∈ channels])

    # Convert WCS to a 2-dimensional version
    wcs_optimal_3d = obs.channels[channels[output_wcs_frame]].wcs
    wcs_optimal = WCSTransform(2; 
                               cdelt=wcs_optimal_3d.cdelt[1:2],
                               crpix=wcs_optimal_3d.crpix[1:2],
                               crval=wcs_optimal_3d.crval[1:2],
                               ctype=wcs_optimal_3d.ctype[1:2],
                               cunit=wcs_optimal_3d.cunit[1:2],
                               pc=wcs_optimal_3d.pc[1:2, 1:2],
                               radesys=wcs_optimal_3d.radesys)
    size_optimal = size(obs.channels[channels[output_wcs_frame]].I)[1:2]

    # Iteration variables to keep track of
    for i ∈ eachindex(channels)

        ch_in = channels[i]
        @info "Reprojecting $(obs.name) channel $ch_in onto the optimal $(size_optimal) WCS grid..."

        # Get the intensity and error arrays
        # NOTE 1: Dont forget to permute and re-permute the dimensions from the numpy row-major order back to the julia column-major order
        # NOTE 2: We need to resample the INTENSITY, not the flux, because the pixel sizes may be different between the input and output images,
        #          and flux scales with the pixel size whereas intensity does not (assuming it's an extended source).
        I_in = obs.channels[ch_in].I 
        σI_in = obs.channels[ch_in].σ
        mask_in = obs.channels[ch_in].mask
        psf_in = obs.channels[ch_in].psf_model

        # Replace NaNs with 0s for the interpolation
        I_in[.~isfinite.(I_in)] .= 0.
        σI_in[.~isfinite.(σI_in)] .= 0.
        if do_psf_model
            psf_in[.~isfinite.(psf_in)] .= 0.
        end

        # Get 2D WCS representation
        wcs_channel = WCSTransform(2; 
                                   cdelt=obs.channels[ch_in].wcs.cdelt[1:2],
                                   crpix=obs.channels[ch_in].wcs.crpix[1:2],
                                   crval=obs.channels[ch_in].wcs.crval[1:2],
                                   ctype=obs.channels[ch_in].wcs.ctype[1:2],
                                   cunit=obs.channels[ch_in].wcs.cunit[1:2],
                                   pc=obs.channels[ch_in].wcs.pc[1:2, 1:2],
                                   radesys=obs.channels[ch_in].wcs.radesys)

        I_out_ch = zeros(size_optimal..., size(I_in, 3))
        σI_out_ch = zeros(size_optimal..., size(I_in, 3))
        mask_out_temp = zeros(size_optimal..., size(I_in, 3))
        if do_psf_model
            psf_out_ch = zeros(size_optimal..., size(I_in, 3))
        end

        if (order == -1) || (i == output_wcs_frame)
            @assert size(I_in)[1:2] == size_optimal
            I_out_ch = I_in
            σI_out_ch = σI_in
            mask_out_temp = mask_in
            if do_psf_model
                psf_out_ch = psf_in
            end

        else
            @showprogress for wi in axes(I_in, 3)
                # Reproject using interpolation
                I_out_ch[:, :, wi], _ = reproject((I_in[:, :, wi], wcs_channel), wcs_optimal, shape_out=size_optimal, order=order)
                σ²_temp, _ = reproject((σI_in[:, :, wi].^2, wcs_channel), wcs_optimal, shape_out=size_optimal, order=order)
                σ²_temp[σ²_temp .< 0] .= 0.
                σI_out_ch[:, :, wi] .= sqrt.(σ²_temp)
                if do_psf_model
                    psf_out_ch[:, :, wi], _ = reproject((psf_in[:, :, wi], wcs_channel), wcs_optimal, shape_out=size_optimal, order=order)
                end

                # Use nearest-neighbor interpolation for the mask since it's a binary 1 or 0
                mask_out_temp[:, :, wi], _ = reproject((Matrix{Float64}(mask_in[:, :, wi]), wcs_channel), wcs_optimal, shape_out=size_optimal, order=0)
            end
        end

        # Set all NaNs to 1s for the mask (i.e. they are masked out)
        mask_out_temp[.!isfinite.(mask_out_temp) .| .!isfinite.(I_out_ch) .| .!isfinite.(σI_out_ch)] .= 1
        mask_out_ch = BitArray(mask_out_temp)

        # Update the observation object in-place
        obs.channels[ch_in].I = I_out_ch
        obs.channels[ch_in].σ = σI_out_ch
        obs.channels[ch_in].mask = mask_out_ch
        if do_psf_model
            obs.channels[ch_in].psf_model = psf_out_ch
        end
        obs.channels[ch_in].wcs = obs.channels[channels[output_wcs_frame]].wcs
        obs.channels[ch_in].θ_sky = obs.channels[channels[output_wcs_frame]].θ_sky
        obs.channels[ch_in].sky_aligned = obs.channels[channels[output_wcs_frame]].sky_aligned

    end
    
    # If an entire channel is masked out, we want to throw away all channels
    # This has to be done after the first loop so that mask_out is not overwritten to be unmasked after it has been masked
    for c ∈ CartesianIndices(size(obs.channels[channels[1]].I)[1:2])
        test = [all(obs.channels[channels[i]].mask[c, :]) for i ∈ eachindex(channels)]
        if any(test)
            for i ∈ eachindex(channels)
                obs.channels[channels[i]].mask[c, :] .= 1
            end
        end
    end

    Ω_out, wcs_optimal_3d
end


"""
    extract_from_aperture!(obs, channels, ap_r)

At each spaxel location, extract a spectrum from an aperture of radius `ap_r` in units of the PSF FWHM. The aperture
thus grows with wavelength at the same rate as the PSF FWHM does. The Observation object is modified in-place.
This is essentially convolving the data with a tophat kernel, and it may be necessary for MIRI data to reduce resampling noise.

# Arguments
- `obs::Observation`: The Observation object to rebin
- `channels=nothing`: The list of channels to be rebinned. If nothing, rebin all channels.
- `ap_r::Real`: The size of the aperture to extract from at each spaxel, in units of the PSF FWHM.
"""
function extract_from_aperture!(obs::Observation, channels::Vector, ap_r::Real; output_wcs_frame::Integer=1)

    pixel_scale = sqrt(obs.channels[channels[output_wcs_frame]].Ω) * 180/π * 3600
    for ch_in ∈ channels
        mm = .~isfinite.(obs.channels[ch_in].I) .| .~isfinite.(obs.channels[ch_in].σ)
        obs.channels[ch_in].I[mm] .= 0.
        obs.channels[ch_in].σ[mm] .= 0.
        if !isnothing(obs.channels[ch_in].psf_model)
            obs.channels[ch_in].psf_model[mm] .= 0.
        end

        @info "Extracting from an aperture of size $ap_r x FWHM..."
        @showprogress for wi in axes(obs.channels[ch_in].I, 3)
            # aperture radius in units of the PSF FWHM
            ap_size = ap_r * obs.channels[ch_in].psf[wi] / pixel_scale

            for c ∈ CartesianIndices(size(obs.channels[ch_in].I)[1:2])
                aperture_spax = CircularAperture(c.I..., ap_size)
                obs.channels[ch_in].I[c, wi] = photometry(aperture_spax, obs.channels[ch_in].I[:, :, wi]).aperture_sum / get_area(aperture_spax)
                obs.channels[ch_in].σ[c, wi] = sqrt(photometry(aperture_spax, obs.channels[ch_in].σ[:, :, wi].^2).aperture_sum) / get_area(aperture_spax)
                if !isnothing(obs.channels[ch_in].psf_model)
                    obs.channels[ch_in].psf_model[c, wi] = photometry(aperture_spax, obs.channels[ch_in].psf_model[:, :, wi]).aperture_sum / get_area(aperture_spax)
                end
            end
        end

        obs.channels[ch_in].I[mm] .= NaN
        obs.channels[ch_in].σ[mm] .= NaN
        if !isnothing(obs.channels[ch_in].psf_model)
            obs.channels[ch_in].psf_model[mm] .= NaN
        end
    end
end


function calc_psf_fwhm(cube::DataCube)

    fwhms = zeros(size(cube.I, 3))

    for wi in axes(cube.I, 3)
        
        # Get centroid
        region = max(wi-10,1):min(wi+10,size(cube.I,3))
        data2d = dropdims(nansum(cube.psf_model[:, :, region], dims=3), dims=3)
        data2d[.~isfinite.(data2d)] .= 0.
        _, mx = findmax(data2d)
        c = centroid_com(data2d[mx[1]-10:mx[1]+10, mx[2]-10:mx[2]+10]) .+ (mx.I .- 10) .- 1

        # Create a 1D PSF profile
        r = 0:0.5:10
        f = zeros(20)
        for i in 1:20
            ap = CircularAnnulus(c..., r[i], r[i+1])
            f[i] = photometry(ap, cube.psf_model[:, :, wi]).aperture_sum / (π * (r[i+1]^2 - r[i]^2))
        end

        # Fit a gaussian
        gaussian_model(x, p) = @. p[1] * exp(-(x - p[2])^2 / (2p[3]^2))

        parinfo = CMPFit.Parinfo(3)
        parinfo[1].limited = (1,1)    # amplitude
        parinfo[1].limits = (0, Inf)
        parinfo[2].fixed = 1          # shift = 0
        parinfo[3].limited = (1,1)    # dispersion 
        parinfo[3].limits = (0, Inf)
    
        m = isfinite.(f)
        res = cmpfit(r[1:end-1][m], f[m], ones(length(f[m])), gaussian_model, [1., 0., 1.], parinfo=parinfo)

        fwhms[wi] = res.param[3] * 2.355
    end

    fwhms
end


"""
    apply_gaussian_smoothing!(obs, channels; max_λ)

Blur the data using a gaussian kernel such that the psf size matches the maximum psf size in the full data cube.
"""
function apply_gaussian_smoothing!(obs::Observation, channels::Vector; output_wcs_frame::Integer=1, max_λ::Real=Inf)

    pixel_scale = sqrt(obs.channels[channels[output_wcs_frame]].Ω) * 180/π * 3600
    λ_out = vcat([obs.channels[ch_i].λ for ch_i ∈ channels]...)
    psf_out = []
    for ch_i ∈ channels
        fwhm_i = obs.channels[ch_i].psf ./ pixel_scale
        if !isnothing(obs.channels[ch_i].psf_model)
            fwhm_i = calc_psf_fwhm(obs.channels[ch_i])
        end
        append!(psf_out, fwhm_i)
    end

    region = λ_out .< max_λ
    out_fwhm = maximum(psf_out[region])  # target FWHM in pixels

    for ch_in ∈ channels
        mm = .~isfinite.(obs.channels[ch_in].I) .| .~isfinite.(obs.channels[ch_in].σ)
        obs.channels[ch_in].I[mm] .= 0.
        obs.channels[ch_in].σ[mm] .= 0.
        psf_ch = obs.channels[ch_in].psf ./ pixel_scale
        if !isnothing(obs.channels[ch_in].psf_model)
            obs.channels[ch_in].psf_model[mm] .= 0.
            psf_ch = calc_psf_fwhm(obs.channels[ch_in])
        end

        @info "Smoothing the data with a gaussian kernel to match PSF sizes"
        @showprogress for wi in axes(obs.channels[ch_in].I, 3)
            in_fwhm = psf_ch[wi]                    # current FWHM in pixels
            FWHM_blur = √(out_fwhm^2 - in_fwhm^2)   # FWHM needed to blur the data to
            σ_blur = FWHM_blur/2.355                # sigma needed to blur the data to in pixels 

            # Apply the filtering
            obs.channels[ch_in].I[:, :, wi] .= imfilter(obs.channels[ch_in].I[:, :, wi], Kernel.gaussian(σ_blur))
            σ² = imfilter(obs.channels[ch_in].σ[:, :, wi].^2, Kernel.gaussian(σ_blur))
            σ²[σ² .≤ 0.] .= 0.
            obs.channels[ch_in].σ[:, :, wi] .= sqrt.(σ²)
            if !isnothing(obs.channels[ch_in].psf_model)
                obs.channels[ch_in].psf_model[:, :, wi] .= imfilter(obs.channels[ch_in].psf_model[:, :, wi], Kernel.gaussian(σ_blur))
            end
        end

        obs.channels[ch_in].I[mm] .= NaN
        obs.channels[ch_in].σ[mm] .= NaN
        if !isnothing(obs.channels[ch_in].psf_model)
            obs.channels[ch_in].psf_model[mm] .= NaN
        end
    end

    out_fwhm * pixel_scale
end


"""
    rescale_channels!(λ_out, I_out, σ_out[, psf_out, ref_λ]; force_match_psf_scalefactors)

Calculate and apply scaling factors between channels/subchannels in-place. The scaling factors are calculated as the median ratios
between the fluxes in the overlapping regions between the channels. The input wavelength/intensity/error arrays must already be combined
into a single array covering all of the different channels. The boundary regions are found by searching for where the wavelength jumps
backwards.

# Arguments
- `λ_out::Vector{<:Real}`: The 1D concatenated wavelength vector for all channels
- `I_out::Array{<:Real,3}`: The 3D concatenated intensity array for all channels
- `σ_out::Array{<:Real,3}`: The 3D concatenated error array for all channels
- `psf_out::Union{Array{<:Real,3},Nothing}=nothing`: The 3D concatenated PSF model array for all channels (optional)
- `ref_λ::Real=8.0`: The reference wavelength that is used to choose which channel to rescale to
- `rescale_limits::Tuple{<:Real,<:Real}=(0., Inf)`: Lower/upper limits on the rescaling factor between channels.
- `rescale_snr::Real=0.0`: SNR threshold that a spaxel must pass before being rescaled.
- `force_match_psf_scalefactors::Bool=false`: If true, the rescaling factors for the PSF models are forced to be the same as those
    for the actual data.
- `rescale_all_psf::Bool=false`: If true, force all PSF spaxels to be rescaled (use this if you generated PSF models using
    webbpsf, but not if you are using PSF models from real data)
- `scale_psf_only::Bool=false`: Rescale just the PSF 
"""
function rescale_channels!(λ_out::Vector{<:Real}, I_out::Array{<:Real,3}, σ_out::Array{<:Real,3}, 
    mask_out::BitArray{3}, psf_out::Union{Array{<:Real,3},Nothing}=nothing, ref_λ::Real=8.0; rescale_limits::Tuple=(0., Inf),
    rescale_snr::Real=0., force_match_psf_scalefactors::Bool=false, rescale_all_psf::Bool=false, scale_psf_only::Bool=false)

    # Need an additional correction (fudge) factor for overall channels
    jumps = findall(diff(λ_out) .< 0.)
    scale_factors = Dict{String, Matrix{Float64}}()
    do_psf_model = !isnothing(psf_out)

    mask2d = [all(mask_out[i,j,:]) for i in axes(mask_out,1), j in axes(mask_out,2)]

    # rescale channels so the flux level is continuous
    for (i, jump) ∈ enumerate(jumps)
        # find the full scale of the overlapping region
        wave_left, wave_right = λ_out[jump+1], λ_out[jump]
        _, i1 = findmin(abs.(λ_out[1:jump] .- wave_left))
        _, i2 = findmin(abs.(λ_out[jump+1:end] .- wave_right))
        i2 += jump

        function get_scale_factors(I, signal, noise, rescale_snr)

            # Get the flux in the left and right channels
            I_left = I[:, :, i1:jump]
            I_right = resample_flux_permuted3D(λ_out[i1:jump], λ_out[jump+1:i2], I[:, :, jump+1:i2])

            # Get the ratios of the left/right flux
            scale_left = I_right ./ I_left
            scale_right = I_left ./ I_right
            # get the median fluxes from both channels over the full region
            med_left = dropdims(nanmedian(I[:, :, i1:jump], dims=3), dims=3)
            med_right = dropdims(nanmedian(I[:, :, jump+1:i2], dims=3), dims=3)
            # Check for any medians close to or below 0 and dont rescale them
            thresh_left = dropdims(nanstd(I[:, :, i1:jump], dims=3), dims=3)
            thresh_right = dropdims(nanstd(I[:, :, jump+1:i2], dims=3), dims=3)
            mask_left_right = (med_left .< thresh_left) .| (med_right .< thresh_right)
            scale_left[mask_left_right, :] .= 1.
            scale_right[mask_left_right, :] .= 1.
            # Check for any median differences smaller than the RMS
            # thresh_both = dropdims(nanmean(cat(thresh_left, thresh_right, dims=3), dims=3), dims=3)
            # mask_left_right = abs.(med_left .- med_right) .< 2thresh_both
            # scale_left[mask_left_right, :] .= 1.
            # scale_right[mask_left_right, :] .= 1.
            # Check for non-finite values or zeros
            scale_left[(scale_left .≤ 0.) .| .~isfinite.(scale_left)] .= 1.
            scale_right[(scale_right .≤ 0.) .| .~isfinite.(scale_right)] .= 1.
            # Do not rescale masked spaxels
            scale_left[mask2d, :] .= 1.
            scale_right[mask2d, :] .= 1.
            # Do not rescale low S/N spaxels
            SN_left = dropdims(nanmedian(signal[:, :, i1:jump] ./ noise[:, :, i1:jump], dims=3), dims=3)
            SN_right = dropdims(nanmedian(signal[:, :, jump+1:i2] ./ noise[:, :, jump+1:i2], dims=3), dims=3)
            SN = dropdims(nanminimum(cat(SN_left, SN_right, dims=3), dims=3), dims=3)
            scale_left[SN .< rescale_snr, :] .= 1.
            scale_right[SN .< rescale_snr, :] .= 1.

            # Take the median along the wavelength axis
            scale_left = dropdims(nanmedian(scale_left, dims=3), dims=3)
            scale_right = dropdims(nanmedian(scale_right, dims=3), dims=3)

            # Clamp within the limits
            scale_left = clamp.(scale_left, rescale_limits...)
            scale_right = clamp.(scale_right, rescale_limits...)

            scale_left, scale_right
        end

        scale_left_data, scale_right_data = get_scale_factors(I_out, I_out, σ_out, rescale_snr)
        scale_left_psf, scale_right_psf = scale_left_data, scale_right_data
        if (!force_match_psf_scalefactors) && do_psf_model
            rescale_psf_snr = rescale_all_psf ? 0. : rescale_snr
            scale_left_psf, scale_right_psf = get_scale_factors(psf_out, I_out, σ_out, rescale_psf_snr)
        end

        # Rescale the PSF model so that the jumps in the psf match the jumps in the data
        if do_psf_model && scale_psf_only
            data2d = dropdims(nansum(I_out, dims=3), dims=3)
            data2d[.~isfinite.(data2d)] .= 0.
            _, mx = findmax(data2d)
            nuc1d = I_out[mx, :] ./ psf_out[mx, :]
            nuc3d = [nuc1d[kk] * psf_out[ii, jj, kk] for ii in axes(psf_out,1), jj in axes(psf_out,2), kk in axes(psf_out, 3)]
            scale_left_psf, scale_right_psf = get_scale_factors(nuc3d, I_out, σ_out, rescale_psf_snr)
            scale_left_psf ./= scale_left_data
            scale_right_psf ./= scale_right_data
        end

        # Apply the scale factors
        if wave_left < ref_λ
            if !scale_psf_only
                I_out[:, :, 1:jump] .*= scale_left_data
                σ_out[:, :, 1:jump] .*= scale_left_data
            scale_factors["$(i+1)"] = scale_left_data
            end
            if do_psf_model
                psf_out[:, :, 1:jump] .*= scale_left_psf
                scale_factors["$(i+1)_PSF"] = scale_left_psf
            end
            @info "Rescaling LEFT channel"
            @info "Minimum/Maximum scale factor for channel $(i+1): $(nanextrema(scale_left_data))"
        else
            if !scale_psf_only
                I_out[:, :, jump+1:end] .*= scale_right_data
                σ_out[:, :, jump+1:end] .*= scale_right_data
                scale_factors["$(i+1)"] = scale_right_data
            end
            if do_psf_model
                psf_out[:, :, jump+1:end] .*= scale_right_psf
                scale_factors["$(i+1)_PSF"] = scale_right_psf
            end
            @info "Rescaling RIGHT channel"
            @info "Minimum/Maximum scale factor for channel $(i+1): $(nanextrema(scale_right_data))"
        end
    end

    I_out, σ_out, mask_out, psf_out, scale_factors, jumps
end


"""
    resample_channel_wavelengths!(λ_out, z, I_out, σ_out, mask_out[, psf_out, concat_type, fix_4c])

If resampling subchannels (concat_type = :sub), resamples `I_out`, `σ_out`, `mask_out`, and `psf_out` onto
a linear wavelength grid while conserving flux. Otherwise (concat_type = :full), only resample the parts of the
spectrum that are overlapping between multiple subchannels -- which are resampled onto a median resolution between
the surrounding channels. The inputs are expected to already be concatenated from all of the different channels/subchannels.

# Arguments
- `λ_out::Vector{<:Real}`: The 1D concatenated wavelength vector for all channels
- `z::Real`: The redshift
- `jumps::Vector{<:Integer}`: The indices in `λ_out` that specify the channel boundaries (i.e. where the diff of λ_out is negative)
- `I_out::Array{<:Real,3}`: The 3D concatenated intensity array for all channels
- `σ_out::Array{<:Real,3}`: The 3D concatenated error array for all channels
- `mask_out::BitArray{3}`: The 3D concatenated mask array for all channels
- `psf_out::Union{Array{<:Real,3},Nothing}=nothing`: The 3D concatenated PSF model array for all channels (optional)
- `concat_type::Symbol=:sub`: Either :sub for subchannels or :full for full channels

"""
function resample_channel_wavelengths!(λ_out::Vector{<:Real}, z::Real, jumps::Vector{<:Integer}, I_out::Array{<:Real,3}, σ_out::Array{<:Real,3},
    mask_out::BitArray{3}, psf_out::Union{Array{<:Real,3},Nothing}=nothing, concat_type::Symbol=:sub, fix_4c::Bool=false)

    do_psf_model = !isnothing(psf_out)

    if concat_type == :full
        λ_con = zeros(eltype(λ_out), 0)
        I_con = zeros(eltype(I_out), size(I_out)[1:2]..., 0)
        σ_con = zeros(eltype(σ_out), size(σ_out)[1:2]..., 0)
        mask_con = falses(size(mask_out)[1:2]..., 0)
        if do_psf_model
            psf_con = zeros(eltype(psf_out), size(I_out)[1:2]..., 0)
        end
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
            I_resamp, σ_resamp, mask_resamp = resample_flux_permuted3D(λ_resamp, 
                λ_out[i1:i2][ss], I_out[:, :, (i1:i2)[ss]], σ_out[:, :, (i1:i2)[ss]], mask_out[:, :, (i1:i2)[ss]])
            if do_psf_model
                psf_resamp = resample_flux_permuted3D(λ_resamp, λ_out[i1:i2][ss], psf_out[:, :, (i1:i2)[ss]])
            end
            # replace overlapping regions in outputs
            λ_con = [λ_con; λ_out[prev_i2:i1-1]; λ_resamp]
            I_con = cat(I_con, I_out[:, :, prev_i2:i1-1], I_resamp, dims=3)
            σ_con = cat(σ_con, σ_out[:, :, prev_i2:i1-1], σ_resamp, dims=3)
            mask_con = cat(mask_con, mask_out[:, :, prev_i2:i1-1], mask_resamp, dims=3)
            if do_psf_model
                psf_con = cat(psf_con, psf_out[:, :, prev_i2:i1-1], psf_resamp, dims=3)
            end
            prev_i2 = i2
        end

        λ_out = [λ_con; λ_out[prev_i2:end]]
        I_out = cat(I_con, I_out[:, :, prev_i2:end], dims=3)
        σ_out = cat(σ_con, σ_out[:, :, prev_i2:end], dims=3)
        mask_out = cat(mask_con, mask_out[:, :, prev_i2:end], dims=3)
        if do_psf_model
            psf_out = cat(psf_con, psf_out[:, :, prev_i2:end], dims=3)
        end
    end

    # deal with overlapping wavelength data -> sort wavelength vector to be monotonically increasing
    ss = sortperm(λ_out)
    λ_out = λ_out[ss]
    I_out = I_out[:, :, ss]
    σ_out = σ_out[:, :, ss]
    mask_out = mask_out[:, :, ss]
    if do_psf_model
        psf_out = psf_out[:, :, ss]
    end

    # Now we interpolate the wavelength dimension using a flux-conserving approach
    Δλ = median(diff(λ_out))
    if concat_type == :sub
        @info "Resampling wavelength onto a uniform, monotonic grid"
        λ_lin = collect(λ_out[1]:Δλ:λ_out[end])
        I_out, σ_out, mask_out = resample_flux_permuted3D(λ_lin, λ_out, I_out, σ_out, mask_out)
        if do_psf_model
            psf_out = resample_flux_permuted3D(λ_lin, λ_out, psf_out)
        end
        λ_out = λ_lin
    else
        @warn "The wavelength dimension has not be resampled to be linear when concatenating multiple full channels! " *
              "Only overlapping regions between channels have been resampled to a median resolution!"
    end

    λ_out, I_out, σ_out, mask_out, psf_out
end


"""
    combine_channels!(obs[, channels])

Perform a number of transformations on data from different channels/subchannels to combine them into a single 3D cube:
    1. Reproject all of the channels onto the same spatial WCS grid using interpolation
    2. Optionally extract from an aperture larger than 1 spaxel at the location of each spaxel (effectively convolving the data spatially)
       -OR-
       Apply a gaussian smoothing kernel onto the data with some size to ensure the PSFs of each channel match
    3. Optionally apply scaling factors to make the flux levels match at the boundaries of the subchannels
    4. Resample along the spectral axis onto a linear wavelength grid while conserving flux

# Arguments
`S<:Integer`
- `obs::Observation`: The Observation object to rebin
- `channels=nothing`: The list of channels to be rebinned. If nothing, rebin all channels.
- `concat_type=:full`: Should be :full for overall channels or :sub for subchannels (bands).
- `out_id=0`: The dictionary key corresponding to the newly rebinned cube, defaults to 0.
- `order::Union{Integer,String}=1`: The order of interpolation for the reprojection. 
        -1 = do not do any reprojection (all input channels must be on the same WCS grid)
        0 = nearest-neighbor
        1 = linear
        2 = quadratic
- `rescale_channels::Union{Real,Nothing}=nothing`: Whether or not to rescale the flux of subchannels so that they match at the overlapping regions.
    If a real number is provided, this is interpreted as an observed-frame wavelength to choose the reference channel that all other channels
    are rescaled to match.
- `adjust_wcs_headerinfo::Bool=true`: Whether or not to try to automatically adjust the WCS header info of the channels by 
    calculating centroids at the boundaries between the channels and forcing them to match.  On by default.
- `min_λ::Real=0.`: Minimum wavelength cutoff for the output cube.
- `max_λ::Real=27.`: Maximum wavelength cutoff for the output cube. Default of 27 um due to degradation above this in MIRI.
- `rescale_limits::Tuple{<:Real,<:Real}=(0., Inf)`: Lower/upper limits on the rescaling factor between channels.
- `rescale_snr::Real=0.0`: SNR threshold that a spaxel must pass before being rescaled.
- `output_wcs_frame::Integer=1`: Which WCS frame to project the inputs onto. Defaults to 1 (the first channel in channels).
- `extract_from_ap::Real=0.`: The size of the aperture to extract from at each spaxel, in units of the PSF FWHM. If 0, just takes the single spaxel.
    This may be necessary for MIRI data to reduce resampling noise.
- `match_psf::Bool=true`: If true, convolves the data with a gaussian kernel such that the spatial PSF sizes match
- `force_match_psf_scalefactors::Bool=false`: If a PSF model is present, this will force the rescaling factors between the subchannels to be the
    same between the data and PSF models (the PSF models will use the scaling factors derived from the data).1
- `rescale_all_psf::Bool=false`: If true, force all PSF spaxels to be rescaled (use this if you generated PSF models using
    webbpsf, but not if you are using PSF models from real data)
- `scale_psf_only::Bool=false`: If true, only apply scaling factors to the PSF. The scaling factors that are applied in this case are
    made such that the jumps in the PSF match the jumps in the data.
- `fix_4c::Bool=false`: If set to true, the PSF model in the transition between channels 4B/4C will be set to a constant value.
"""
function combine_channels!(obs::Observation, channels=nothing, concat_type=:full; out_id=0,
    order::Union{Integer,String}=1, rescale_channels::Union{Real,Nothing}=nothing, adjust_wcs_headerinfo::Bool=false, 
    min_λ::Real=0., max_λ::Real=27., rescale_limits::Tuple{<:Real,<:Real}=(0., Inf), rescale_snr::Real=0.0, 
    output_wcs_frame::Integer=1, extract_from_ap::Real=0., match_psf::Bool=false, force_match_psf_scalefactors::Bool=false,
    rescale_all_psf::Bool=false, scale_psf_only::Bool=false, fix_4c::Bool=false)

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

    # 0. First and foremost -- adjust the WCS alignment of each channel so that they are consistent with each other
    if adjust_wcs_headerinfo
        adjust_wcs_alignment!(obs, channels; box_size_as=1.5)
    end

    # 1. Reproject the channels onto the same WCS grid
    Ω_out, wcs_optimal_3d = reproject_channels!(obs, channels; order=order, output_wcs_frame=output_wcs_frame)

    # 2. Optionally extract from an aperture
    if extract_from_ap > 0
        extract_from_aperture!(obs, channels, extract_from_ap, output_wcs_frame=output_wcs_frame)
    end
    # Smooth with the psf
    out_fwhm = 0.
    if match_psf
        out_fwhm = apply_gaussian_smoothing!(obs, channels; output_wcs_frame=output_wcs_frame, max_λ=max_λ)
    end

    # Now we concatenate the data from each channel/subchannel along the wavelength axis
    λ_out = vcat([obs.channels[ch_i].λ for ch_i ∈ channels]...)
    I_out = cat([obs.channels[ch_i].I for ch_i ∈ channels]..., dims=3)
    σ_out = cat([obs.channels[ch_i].σ for ch_i ∈ channels]..., dims=3)
    mask_out = cat([obs.channels[ch_i].mask for ch_i ∈ channels]..., dims=3)
    do_psf_model = all([!isnothing(obs.channels[ch_i].psf_model) for ch_i ∈ channels])
    psf_out = nothing
    if do_psf_model
        psf_out = cat([obs.channels[ch_i].psf_model for ch_i ∈ channels]..., dims=3)
    end

    # 3. Optionally apply scaling factors between the subchannels
    if !isnothing(rescale_channels)
        I_out, σ_out, mask_out, psf_out, scale_factors, jumps = rescale_channels!(λ_out, I_out, σ_out, mask_out, psf_out, 
            rescale_channels, rescale_limits=rescale_limits, rescale_snr=rescale_snr, force_match_psf_scalefactors=force_match_psf_scalefactors,
            rescale_all_psf=rescale_all_psf, scale_psf_only=scale_psf_only)
    else
        jumps = findall(diff(λ_out) .< 0)
        scale_factors = nothing
    end

    # 4. Resample along the spectral axis onto a linear wavelength grid while conserving flux
    λ_out, I_out, σ_out, mask_out, psf_out = resample_channel_wavelengths!(λ_out, obs.z, jumps, I_out, σ_out, mask_out, psf_out, concat_type, fix_4c)

    # Cut off at large wavelength, if specified
    λmask = min_λ .≤ λ_out .≤ max_λ
    λ_out = λ_out[λmask]
    I_out = I_out[:, :, λmask]
    σ_out = σ_out[:, :, λmask]
    mask_out = mask_out[:, :, λmask]
    if do_psf_model
        psf_out = psf_out[:, :, λmask]
    end

    # New PSF FWHM function with input in the rest frame
    if obs.rest_frame
        psf_fwhm_out = mrs_psf(@. λ_out * (1 + obs.z))
    else
        psf_fwhm_out = mrs_psf(λ_out)
    end
    if match_psf
        psf_fwhm_out = repeat([out_fwhm], length(λ_out))
    end

    # New LSF FWHM function with input in the rest frame
    if obs.rest_frame
        lsf_fwhm_out = mrs_lsf(@. λ_out * (1 + obs.z))
    else
        lsf_fwhm_out = mrs_lsf(λ_out)
    end

    if obs.masked
        @info "Masking bins with bad data..."
        I_out[mask_out] .= NaN
        σ_out[mask_out] .= NaN
        if do_psf_model
            psf_out[mask_out] .= NaN
        end
    end

    # Do a final PSF renormalization
    if do_psf_model
        psf_out[psf_out .< 0] .= 0.
        for k ∈ axes(psf_out, 3)
            psf_out[:, :, k] ./= nansum(psf_out[:, :, k])
        end
    end

    # Define the interpolated cube as the zeroth channel (since this is not taken up by anything else)
    obs.channels[out_id] = DataCube(λ_out, I_out, σ_out, mask_out, psf_out, Ω_out, obs.α, obs.δ, obs.channels[channels[output_wcs_frame]].θ_sky, psf_fwhm_out, 
        lsf_fwhm_out, wcs_optimal_3d, "MULTIPLE", "MULTIPLE", obs.spectral_region, obs.rest_frame, obs.masked, obs.vacuum_wave, obs.sky_aligned)

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

