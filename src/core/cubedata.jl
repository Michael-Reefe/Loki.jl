#=
This is the file that handles loading in and preparing 3D IFU cube data to be fit. It contains
structs for holding the 3D IFU cubes, and functions that apply certain corrections to the data, such
as shifting the wavelength to the rest frame, and masking out bad spaxels.  The first step in any
fitting sequence should be loading in the data, likely with the "from_fits" function to load from a 
Flexible Image Transfer System (FITS) file, followed by applying the two aforementioned corrections
using the "correct" function.  The data should then be handed off to the structs/functions in the 
"CubeFit" file, which handles actually fitting the cubes.
=#


# Functions approximating the MIRI MRS line-spread function and point-spread function
# Note: the inputs of both are the OBSERVED-FRAME wavelengths
@inline mrs_lsf(λ::Quantity) = uconvert(u"km/s", C_KMS / (4603 - 128*(λ/u"μm") + 10^(-7.4*(λ/u"μm"))))
@inline mrs_psf(λ::Quantity) = (0.033*ustrip(λ/u"μm") + 0.016) * u"arcsecond"

# Equivalent functions for NIRSpec

@inline nirspec_lsf(λ::Quantity, grating::String) = if grating in ["G140M", "G235M", "G395M"]
    C_KMS / 1000.0
elseif grating in ["G140H", "G235H", "G395H"]
    C_KMS / 2700.0
else
    C_KMS / 100.0
end
# guesstimated based on the NIRSpec docs claim that the PSF ranges from 0.03-0.16" here: 
# https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-observing-strategies/nirspec-dithering-recommended-strategies#gsc.tab=0
@inline nirspec_psf(λ::Quantity) = (0.03 + 0.13*(λ/u"μm" - 0.90)/4.37) * u"arcsecond"


# very simple parser; bound to not work in all cases but that's a lotta work
function fits_unitstr_to_unitful(unit_string::String)
    @debug "fits_unitstr_to_unitful: input=\"$unit_string\""
    unit_string = replace(unit_string, "um" => "μm", "." => "*")
    i = 1
    while i ≤ lastindex(unit_string)
        # insert an exponent before each number in the string
        char = unit_string[i]
        if (char in ('-','+') && !(unit_string[prevind(unit_string, i)] == '^')) || 
            (isdigit(char) && !(unit_string[prevind(unit_string, i)] in ['-','+','^']))
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
- `log_binned`: whether or not the wavelength vector is logarithmically spaceed
- `sky_aligned`: whether or not the data cube is aligned to the sky (RA/Dec) axes or the IFU (α/β) axes
- `voronoi_bins`: a map giving unique labels to each spaxel which place them within voronoi bins
"""
mutable struct DataCube{T<:Vector{<:QWave}, S<:Array{<:QSIntensity, 3}}

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
    rest_frame::Bool
    masked::Bool
    vacuum_wave::Bool
    log_binned::Bool
    dereddened::Bool
    sky_aligned::Bool

    voronoi_bins::Union{Matrix{<:Integer},Nothing}

    # This is the constructor for the DataCube struct; see the DataCube docstring for details
    function DataCube(λ::T, I::S, σ::S, mask::Union{BitArray{3},Nothing}=nothing, psf_model::Union{Array{<:Real,3},Nothing}=nothing, 
        Ω=NaN*u"sr", α=NaN*u"°", δ=NaN*u"°", θ_sky=NaN*u"rad", psf::Union{U,Nothing}=nothing, lsf::Union{V,Nothing}=nothing, 
        wcs::Union{WCSTransform,Nothing}=nothing, channel::String="Generic Channel", band::String="Generic Band", 
        user_mask::Union{Vector{Tuple{W,W}},Nothing}=nothing, gaps::Union{Vector{Tuple{W,W}},Nothing}=nothing, 
        rest_frame::Bool=false, z::Union{Real,Nothing}=nothing, masked::Bool=false, vacuum_wave::Bool=true, 
        log_binned::Bool=false, dereddened::Bool=false, sky_aligned::Bool=false, voronoi_bins::Union{Matrix{<:Integer},Nothing}=nothing, 
        format::Symbol=:MIRI, instrument_channel_edges::Union{T,Nothing}=nothing) where {
            T<:Vector{<:Quantity},S<:Array{<:Quantity,3},U<:Vector{<:Quantity},V<:Vector{<:Quantity},W<:Quantity
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
        elseif dimension(I_unit) == u"𝐌*𝐋^-1*𝐓^-3"
            λ_unit = string(unit(λ[1]))
            new_I_unit = uparse("erg*s^-1*cm^-2*$(λ_unit)^-1*sr^-1"; unit_context=[Unitful, UnitfulAstro])
        else
            error("Intensities must be given in units that are convertible to erg/s/cm2/Hz/sr or erg/s/cm2/Ang/sr")
        end
        I = uconvert.(new_I_unit, I)
        σ = uconvert.(new_I_unit, σ)

        # Also upgrade to Float64 if necessary
        ftype = typeof(1.0)
        λ = ftype.(λ); I = ftype.(I); σ = ftype.(σ)
        Ω = ftype(Ω); α = ftype(α); δ = ftype.(δ); θ_sky = ftype.(θ_sky)
        if !isnothing(psf)
            psf = ftype.(psf)
        end
        if !isnothing(lsf)
            lsf = ftype.(lsf)
        end

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

        restframe_factor = rest_frame ? 1.0 : 1 / (1 + z)
        λlim = extrema(λ .* restframe_factor)
        λrange = get_λrange(λlim)
        umask = !isnothing(user_mask) ? user_mask : Vector{Tuple{eltype(λ),eltype(λ)}}()
        n_channels, ch_bounds, channel_masks = get_n_channels(λ, rest_frame, z; format=format, 
            instrument_channel_edges=instrument_channel_edges)
        if isnothing(gaps)
            gaps = Vector{Tuple{eltype(λ),eltype(λ)}}()
        end
        for gap in gaps
            @assert sum(gap[1] .< (λ.*restframe_factor) .< gap[2]) == 0 "Data was detected within the gap $gap"
        end
        spectral_region = SpectralRegion(λlim, umask, n_channels, channel_masks, ch_bounds, gaps, λrange)

        # Return a new instance of the DataCube struct
        @debug "DataCube: channel=$channel, band=$band, shape=($nx,$ny,$nz), λ_range=($(ustrip(λ[1])),$(ustrip(λ[end]))) $(unit(λ[1])), rest_frame=$rest_frame"
        Tnew = typeof(λ)
        Snew = typeof(I)
        new{Tnew, Snew}(λ, I, σ, mask, psf_model, Ω, α, δ, θ_sky, psf, lsf, wcs, channel, band, nx, ny, nz, spectral_region,
            rest_frame, masked, vacuum_wave, log_binned, dereddened, sky_aligned, voronoi_bins)
    end

end


"""
    from_fits(filename::String)

Utility function for creating DataCube structures directly from FITS files.

# Arguments
- `filename::String`: the filepath to the JWST-formatted FITS file
"""
function from_fits(filename::String, z::Union{Real,Nothing}=nothing)::DataCube

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
    bunit = uparse(fits_unitstr_to_unitful(hdr["BUNIT"]); unit_context=[Unitful, UnitfulAstro])
    Iν = read(hdu["SCI"]) * bunit
    σI = read(hdu["ERR"]) * bunit

    @debug "FITS data dimensions: ($nx, $ny, $nz), solid angle per spaxel: $Ω"

    # Construct 3D World coordinate system to convert from pixels to (RA,Dec,wave) and vice versa
    # Note: We need to be careful here because WCS.jl cannot handle tabular wavelength solutions where,
    #       for example, CTYPE3 = 'WAVE-TAB', so we need to handle those manually.
    if haskey(hdr, "CTYPE3")
        if hdr["CTYPE3"] == "WAVE-TAB"
            wcs = create_wcs_from_header(hdr)
            wave_tab = true
        else
            wcs = WCS.from_header(read_header(hdu["SCI"], String))[1]
            wave_tab = false
        end
    else
        error("Header is missing key CTYPE3, unable to determine wavelength coordinate system")
    end

    if haskey(hdr, "CUNIT3")
        cunit3 = uparse(fits_unitstr_to_unitful(hdr["CUNIT3"]); unit_context=[Unitful, UnitfulAstro])
    else
        error("Header is missing key CUNIT3, unable to determine wavelength units")
    end

    # Construct the wavelength vector
    λ = if wave_tab
        if haskey(hdr, "PS3_0") && haskey(hdr, "PS3_1")
            wave_hdu_name = hdr["PS3_0"]
            wave_col_name = hdr["PS3_1"]
        else
            error("Header is missing key(s) PS3_0 or PS3_1, unable to find wavelength table")
        end
        vec(read(hdu[wave_hdu_name], wave_col_name)) .* cunit3
    else
        # Legacy for backwards-compatibility with cubes made with old versions of LOKI
        if haskey(hdu, "AUX") && ("wave" in FITSIO.colnames(hdu["AUX"]))
            @warn "Detected an outdated cube generated by old versions of Loki."
            tblhdr = read_header(hdu["AUX"])
            vec(read(hdu["AUX"], "wave")) .* uparse(fits_unitstr_to_unitful(tblhdr["UNIT1"]); unit_context=[Unitful,UnitfulAstro])
        # "WAVE" WCS cubes with a linear wavelength vector
        else
            @assert haskey(hdr, "CRVAL3") && haskey(hdr, "CDELT3") && haskey(hdr, "CRPIX3") && haskey(hdr, "NAXIS3") "Header " * 
                "is missing key(s) CRVAL3, CRPIX3, CDELT3, or NAXIS3"
            (hdr["CRVAL3"] .+ hdr["CDELT3"] .* (collect(0:hdr["NAXIS3"]-1) .+ hdr["CRPIX3"] .- 1)) .* cunit3
        end
    end

    # Alternative method using the WCS directly:
    # λ = pix_to_world(wcs, Matrix(hcat(ones(nz), ones(nz), collect(1:nz))'))[3,:] ./ 1e-6

    # Data quality map (i.e. the mask)
    # dq = 0 if the data is good, > 0 if the data is bad
    dq = read(hdu["DQ"])
    # also make sure to mask any points with Inf/NaN in the intensity or error, in case they were 
    # missed by the DQ map
    mask = (dq .≠ 0) .| .~isfinite.(Iν) .| .~isfinite.(σI)

    # Target info from the header
    hdr0 = read_header(hdu[1])
    name = hdr0["TARGNAME"]           # name of the target
    ra = hdr0["TARG_RA"] * u"°"       # right ascension in deg
    dec = hdr0["TARG_DEC"] * u"°"     # declination in deg

    # format type
    format = Symbol(hdr0["INSTRUME"])
    if format == :MIRI
        channel = string(hdr0["CHANNEL"]) # MIRI channel (1-4)
        band = hdr0["BAND"]               # MIRI band (long,med,short,multiple)
    elseif (format == :NIRSPEC) && haskey(hdr0, "FILTER") && haskey(hdr0, "GRATING")
        channel = hdr0["FILTER"]  # NIRSpec filter (F070LP, F100LP, F170LP, F290LP, CLEAR)
        band = hdr0["GRATING"]    # NIRSpec grating (G140H/M, G235H/M, G395H/M, PRISM)
    else
        channel = string(hdr0["CHANNEL"]) 
        band = hdr0["BAND"]               
    end

    # Sky rotation angle
    θ_sky = get_sky_rotation_angle(wcs)
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
        if format == :MIRI
            mrs_psf.(λ)
        elseif format == :NIRSPEC
            nirspec_psf.(λ)
        else
            error("File $filename does not have a specified PSF FWHM!")
        end
    end
    lsf = try
        read(hdu["AUX"], "lsf") * u"km/s"
    catch
        if format == :MIRI
            mrs_lsf.(λ)
        elseif format == :NIRSPEC 
            nirspec_lsf.(λ, band)
        else 
            error("File $filename does not have a specified LSF FWHM!")
        end
    end

    n_channels = nothing
    # MIRI 
    if band in ("SHORT", "MEDIUM", "LONG")
        n_channels = 1
    end
    # NIRSPEC
    if band in ("G140M", "G235M", "G395M", "G140H", "G235H", "G395H", "PRISM")
        n_channels = 1
    end

    if haskey(hdr0, "NCHAN")
        n_channels = hdr0["NCHAN"]
    end
    if haskey(hdr0, "N_CHANNELS")  # another legacy which I've changed to avoid HIERARCH keywords
        n_channels = hdr0["N_CHANNELS"]
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
    log_binned = false
    if haskey(hdr0, "LOGBIN")
        log_binned = hdr0["LOGBIN"]
    end
    dereddened = false
    if haskey(hdr0, "DERED")
        dereddened = hdr0["DERED"]
    end

    redshift = z
    if haskey(hdr, "REDSHIFT")
        redshift = hdr["REDSHIFT"]
    end

    gaps = try
        tblhdr = read_header(hdu["GAP"])
        wu1 = uparse(tblhdr["UNIT1"]; unit_context=[Unitful, UnitfulAstro])
        wu2 = uparse(tblhdr["UNIT2"]; unit_context=[Unitful, UnitfulAstro])
        g1 = read(hdu["GAP"], "gaps_1") .* wu1
        g2 = read(hdu["GAP"], "gaps_2") .* wu2
        [(gi1, gi2) for (gi1, gi2) in zip(g1, g2)]
    catch
        nothing
    end

    DataCube(λ, Iν, σI, mask, psf_model, Ω, ra, dec, θ_sky, psf, lsf, wcs, channel, band, nothing, gaps, 
        rest_frame, redshift, masked, vacuum_wave, log_binned, dereddened, sky_aligned, nothing, format, nothing)
end


"""
    from_data(Ω, z, λ, I[, σ]; <keyword_args>)

Utility function for creating DataCube structures directly from input data.

# Arguments
- `Ω`: The solid angle per spaxel (or for 1D spectra, the total solid angle that the spectrum covers) (must include units)
- `z`: The redshift 
- `λ`: The wavelength vector (must include units)
- `I`: The specific intensity vector (must include units)
- `σ`: [Optional] The error in the specificy intensity (must include units)

# Optional Keyword Arguments
- `mask`: A bitarray specifying which pixels should be masked out
- `α`: The right ascension of the central brightest point (must include units)
- `δ`: The declination of the central brightest point (must include units)
- `θ_sky`: The position angle between the IFU axes and the sky axes (must include units)
- `psf_fwhm`: The FWHM of the PSF, either as a single value or a vector (must include units)
- `psf_model`: A full 3D model of the PSF of the same shape as I and σ (required for QSO PSF decomposition)
- `R`: The spectral FWHM resolution, either as a single value or a vector
- `wcs`: The world coordinate system of the 3D arrays
- `channel`: The name of the wavelength channel of the observations (string)
- `band`: The name of the wavelength band of the observations (string)
- `user_mask`: A series of tuples of (min, max) wavelengths specifying regions in the spectrum to mask out
- `gaps`: A series of tuples of (min, max) wavelengths specifying regions in the spectrum that are missing data 
    (i.e. in the case where multiwavelength data has been stitched together)
- `rest_frame`: A boolean specifying whether the input wavelength/intensity/error are in the rest frame or observed frame
- `masked`: A boolean specifying whether the input intensities/errors have already been masked
- `vacuum_wave`: A boolean specifying whether the input wavelengths are in vacuum or in air
- `log_binned`: A boolean specifying whether the input wavelengths are logarithmically binned
    (note: due to the potential presence of wavelength gaps in the spectrum, this cannot be automatically inferred)
- `dereddened`: A boolean specifying whether the input intensities/errors have been corrected for galactic (Milky Way) dust extinction
- `sky_aligned`: A boolean specifying whether the input IFU axes are aligned to the sky RA/Dec axes
- `instrument_channel_edges`: A series of wavelengths specifying the edges of individual wavelength channels 
    (this is only relevant if doing QSO PSF decomposition with multi-channel data)
"""
function from_data(Ω::typeof(1.0u"sr"), z::Real, λ::AbstractVector{<:Quantity},
    I::AbstractArray{<:Quantity}, σ::Union{AbstractArray{<:Quantity},Nothing}=nothing; mask::Union{BitArray,Nothing}=nothing,
    α=0.0*u"°", δ=0.0*u"°", θ_sky=0.0*u"rad", psf_fwhm::Union{Quantity,AbstractVector{<:Quantity},Nothing}=nothing,
    psf_model::Union{AbstractArray{<:Real},Nothing}=nothing, R::Union{Real,AbstractVector{<:Real},Nothing}=nothing,
    wcs::Union{WCSTransform,Nothing}=nothing, channel::String="Generic Channel", band::String="Generic Band",
    user_mask::Union{Vector{<:Tuple},Nothing}=nothing, gaps::Union{Vector{<:Tuple},Nothing}=nothing,
    rest_frame::Union{Bool,Nothing}=nothing, masked::Union{Bool,Nothing}=nothing, vacuum_wave::Union{Bool,Nothing}=nothing,
    log_binned::Union{Bool,Nothing}=nothing, dereddened::Union{Bool,Nothing}=nothing, sky_aligned::Union{Bool,Nothing}=nothing,
    instrument_channel_edges::Union{Vector{<:Quantity},Nothing}=nothing)

    @debug "from_data: channel=$channel, band=$band, I_shape=$(size(I)), z=$z, n_λ=$(length(λ)), n_I_nan=$(count(isnan, ustrip.(I))), n_I_inf=$(count(isinf, ustrip.(I)))"
    # convert to a normal vector
    _λ = Float64.(collect(λ))

    # if I/σ/mask are 1D, convert to 3D as expected by the code
    _I = I
    if ndims(I) == 1
        _I = reshape(I, (1,1,length(I)))
    end
    @assert ndims(_I) == 3 "The input intensity dimensions must be 1 or 3"
    @assert occursin("sr", string(unit(_I[1]))) "The input intensity must be measured per sr! (if you input a flux, divide it by Ω)"
    _I = Float64.(_I)

    _σ = σ
    if isnothing(σ)
        _σ = ones(eltype(_I), size(_I)...) .* nanmedian(ustrip.(_I)[ustrip.(_I) .> 0.])./10
    end
    if ndims(_σ) == 1
        _σ = reshape(_σ, (1,1,length(_σ)))
    end
    @assert ndims(_σ) == 3 "The input error dimensions must be 1 or 3"
    @assert occursin("sr", string(unit(_σ[1]))) "The input error must be measured per sr! (if you input a flux, divide it by Ω)"
    _σ = Float64.(_σ)

    _mask = mask
    if isnothing(mask)
        # Mask out blank spaxels
        _mask = [all(iszero.(_I[i,j,:]) .| .~isfinite.(_I[i,j,:])) for i in axes(_I, 1), j in axes(_I, 2)]
        _mask = BitArray(repeat(_mask, outer=(1,1,size(_I,3))))
    end
    if ndims(_mask) == 1
        _mask = reshape(_mask, (1,1,length(_mask)))
    end
    @assert ndims(_mask) == 3 "The input mask dimensions must be 1 or 3"

    # the input PSF model
    _psf_model = psf_model
    if !isnothing(psf_model)
        if ndims(psf_model) == 1
            _psf_model = reshape(_psf_model, (1,1,length(_psf_model)))
        end
        @assert ndims(_psf_model) == 3 "The input PSF model dimensions must be 1 or 3"
        _psf_model = Float64.(_psf_model)
    end

    # if no PSF is given, assume that it's the size of 3 pixels
    _psf = psf_fwhm
    if isnothing(psf_fwhm)
        _psf = 3 * uconvert(u"arcsecond", sqrt(Ω))
        @warn "No PSF size [FWHM] was given in the input. It will be assumed that the PSF is 3 spatial pixels wide ($(_psf))."
    end
    if length(_psf) == 1
        _psf = repeat([_psf], length(_λ))
    end
    _psf = Float64.(_psf)

    # if no spectral resolution FWHM is given, assume that it's the size of 3 pixels
    _R = R
    if isnothing(R)
        _R = _λ ./ (3 .* [_λ[2]-_λ[1]; diff(_λ)])
        @warn "No spectral resolution [FWHM] was given in the input. It will be assumed that the FWHM is 3 spectral pixels wide ($(_R[1])-$(_R[end]))."
    end
    if length(_R) == 1
        _R = repeat([_R], length(_λ))
    end
    # convert R to km/s
    _lsf = C_KMS ./ _R
    _lsf = Float64.(_lsf)

    # if no WCS is given, create a generic default one
    _wcs = wcs
    if isnothing(wcs)
        _, mx = findmax(sumdim(ustrip.(_I), 3))
        _α = α
        _δ = δ
        if unit(_α) == NoUnits
            _α = α*u"°"
        end
        if unit(_δ) == NoUnits
            _δ = δ*u"°"
        end
        _α = uconvert(u"°", _α)
        _δ = uconvert(u"°", _δ)
        pix_res_deg = uconvert(u"°", sqrt(Ω))
        x_cent, y_cent = (1.0,1.0)
        if size(_I)[1:2] ≠ (1,1)
            x_cent, y_cent = centroid_com(sumdim(ustrip.(_I), 3)[
                max(1,mx[1]-5):min(size(_I,1),mx[1]+5), max(1,mx[2]-5):min(size(_I,2),mx[2]+5)
                ]) .+ (mx.I .- 5) .- 1
        end
        waveunit_str = replace(string(unit(_λ[1])), 'μ' => 'u', "Å" => "angstrom")
        _wcs = WCSTransform(3; 
            crpix=[x_cent, y_cent, 1.], 
            crval=[ustrip(_α), ustrip(_δ), 1.], 
            cdelt=[ustrip(pix_res_deg), ustrip(pix_res_deg), 1.], 
            cunit=["deg", "deg", waveunit_str], 
            ctype=["RA---TAN", "DEC--TAN", "WAVE-TAB"], 
            pc=[-cos(θ_sky) -sin(θ_sky) 0.; sin(θ_sky) cos(θ_sky) 0.; 0. 0. 1.], 
            radesys="ICRS",
            specsys="BARYCENT"
            )
    end

    _rest_frame = rest_frame
    if isnothing(rest_frame)
        @warn "Assuming input wavelengths are not redshift corrected; if they already are in the rest frame, " *
              "please provide the keyword rest_frame=true."
        _rest_frame = false
    end

    # check for wavelength gaps
    _gaps = gaps
    if isnothing(gaps)
        _gaps = Tuple{eltype(_λ),eltype(_λ)}[]
        dλ = diff(_λ)
        restframe_factor = _rest_frame ? 1.0 : 1/(1+z)
        for i in 2:(length(dλ)-1)
            if (dλ[i] > 10dλ[i-1]) && (dλ[i] > 10dλ[i+1])
                _gap = ((_λ[i]+sqrt(eps())*dλ[i])*restframe_factor, (_λ[i+1]-sqrt(eps())*dλ[i])*restframe_factor)
                @info "Autodetected a wavelength gap at $(_gap)"
                push!(_gaps, _gap)
            end
        end
    end

    # Print some warnings for users
    _masked = masked
    if isnothing(masked)
        # (dont really need a warning for this one as applying it twice does not hurt anything)
        _masked = false
    end
    _vacuum_wave = vacuum_wave
    if isnothing(vacuum_wave)
        @warn "Assuming input wavelengths are in vacuum; if they are in air, please provide the keyword vacuum_wave=false."
        _vacuum_wave = true
    end
    _log_binned = log_binned
    if isnothing(log_binned)
        @warn "Assuming input wavelengths are not logarithmically binned; if they are, please provide the keyword log_binned=true."
        _log_binned = false
    end
    _dereddened = dereddened
    if isnothing(dereddened)
        @warn "Assuming input intensities are not reddening-corrected; if they are already reddening-corrected, " * 
              "please provide the keyword dereddened=true"
        _dereddened = false
    end
    _sky_aligned = sky_aligned
    if isnothing(sky_aligned)
        @warn "Assuming the input intensity cube is aligned with the sky RA/Dec axes; if it is not, please provide " * 
              "the keyword sky_aligned=false and the position angle θ_sky=..."
        _sky_aligned = true
    end

    DataCube(_λ, _I, _σ, _mask, _psf_model, Ω, _α, _δ, θ_sky, _psf, _lsf, _wcs, channel, band, user_mask, 
        _gaps, _rest_frame, z, _masked, _vacuum_wave, _log_binned, _dereddened, _sky_aligned, nothing, :Generic, 
        instrument_channel_edges)
end


# Helper function for calculating the number of subchannels covered by MIRI observations
function get_n_channels(_λ::Vector{<:QWave}, rest_frame::Bool, z::Union{Real,Nothing}; format=:MIRI,
    instrument_channel_edges::Union{Vector{<:QWave},Nothing}=nothing)

    @debug "get_n_channels: nλ=$(length(_λ)), rest_frame=$rest_frame, z=$z, format=$format"
    # NOTE: do not use n_channels to count the ACTUAL number of channels/bands in an observation,
    #  as n_channels counts the overlapping regions between channels as separate "channels" altogether
    #  to allow them to have different normalizations
    n_channels = 0
    channel_masks = BitVector[]
    if format == :MIRI
        ch_edge_sort = sort(channel_edges)
        ch_bounds = channel_boundaries
    elseif format == :NIRSPEC 
        ch_edge_sort = sort(channel_edges_nir)
        ch_bounds = channel_boundaries_nir
    else
        if isnothing(instrument_channel_edges) 
            @warn "The DataCube format is not JWST! Please manually input the channel edges (if any), otherwise it will be assumed " *
                  "that there is only one channel"
            return 1, eltype(_λ)[], [trues(length(_λ))]
        else
            ch_edge_sort = sort(instrument_channel_edges)
            ch_bounds = Vector{eltype(_λ)}()
            for i in 2:2:(length(instrument_channel_edges)-1)
                push!(ch_bounds, (instrument_channel_edges[i]+instrument_channel_edges[i+1])/2)
            end
        end
    end

    λ = _λ
    if rest_frame
        @assert !isnothing(z) "Please input the redshift if the cube is already in the rest frame!"
        λ = _λ .* (1 .+ z)
    end

    for i in 2:(length(ch_edge_sort))
        left = ch_edge_sort[i-1]
        right = ch_edge_sort[i]
        ch_mask = left .< λ .< right
        n_region = sum(ch_mask)

        if n_region > 0
            n_channels += 1
            push!(channel_masks, ch_mask)
        end
    end
    # filter out small beginning/end sections
    if (length(channel_masks) ≥ 2) && (sum(channel_masks[1]) < 200)
        channel_masks[2] .|= channel_masks[1]
        popfirst!(channel_masks)
        n_channels -= 1
    end
    if (length(channel_masks) ≥ 2) && (sum(channel_masks[end]) < 200)
        channel_masks[end-1] .|= channel_masks[end]
        pop!(channel_masks)
        n_channels -= 1
    end
    ch_bound_out = ch_bounds[minimum(λ) .< ch_bounds .< maximum(λ)]

    @debug "get_n_channels: n_channels=$n_channels, n_ch_bounds=$(length(ch_bound_out))"
    n_channels, ch_bound_out, channel_masks
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
        # Wavelength is always shorter in the rest frame
        z1 = 1 + z
        cube.λ = @. cube.λ / z1
        # Multiple dispatch decides whether to divide or multiply by 1+z based on the units of I and σ
        _restframe!(cube, z1)
        cube.rest_frame = true
    end
    cube
end

# Internal functions
function _restframe!(cube::DataCube{<:Vector{<:QWave}, <:Array{<:QPerFreq,3}}, z1::Real)
    @debug "_restframe!(QPerFreq): channel=$(cube.channel), band=$(cube.band), z1=$z1"
    cube.I = @. cube.I / z1
    cube.σ = @. cube.σ / z1
end
function _restframe!(cube::DataCube{<:Vector{<:QWave}, <:Array{<:QPerWave,3}}, z1::Real)
    @debug "_restframe!(QPerWave): channel=$(cube.channel), band=$(cube.band), z1=$z1"
    cube.I = @. cube.I * z1
    cube.σ = @. cube.σ * z1
end



"""
    to_vacuum_wavelength!(cube::DataCube, linear_resample=true)

Convert a DataCube object's wavelength vector from air wavelengths to vacuum wavelengths.
"""
function to_vacuum_wavelength!(cube::DataCube)

    # Only convert if it isn't already in vacuum wavelengths
    if !cube.vacuum_wave
        @debug "Converting the wavelength vector of cube with channel $(cube.channel), band $(cube.band)" *
            " to vacuum wavelengths."
        # Convert to vacuum wavelengths (airtovac uses Angstroms, 1 Angstrom = 10^-4 μm)
        λ_unit = unit(cube.λ[1])
        # airtovac always works in angstroms, so we need to convert to them and back
        cube.λ = uconvert.(λ_unit, airtovac.(ustrip.(uconvert.(u"angstrom", cube.λ))).*u"angstrom")
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
    log_rebin!(cube, z[, logscale, factor])

Rebin a DataCube onto a logarithmically spaced wavelength vector, conserving flux.
Optionally input a rebinning factor > 1 to resample onto a coarser wavelength grid.
"""
function log_rebin!(cube::DataCube, z::Real;
    logscale::Real=nanmedian(log.(cube.λ[2:end]./cube.λ[1:end-1])),
    factor::Integer=1)

    @assert factor > 0 "factor must be > 0"
    @debug "log_rebin!(DataCube): channel=$(cube.channel), band=$(cube.band), nλ=$(length(cube.λ)), factor=$factor, already_binned=$(cube.log_binned)"
    if !cube.log_binned
        # rebin onto a logarithmically spaced wavelength grid
        # get masks for each gap region and the logarithmic spacing
        gap_masks = get_gap_masks(cube.λ, cube.spectral_region.gaps)
        # prepare buffers
        λ_out = Vector{eltype(cube.λ)}()
        I_out = Array{eltype(cube.I), 3}(undef, size(cube.I)[1:2]..., 0)
        σ_out = Array{eltype(cube.σ), 3}(undef, size(cube.σ)[1:2]..., 0)
        mask_out = BitArray(undef, size(cube.mask)[1:2]..., 0)
        if !isnothing(cube.psf_model)
            psf_model_out = Array{eltype(cube.psf_model), 3}(undef, size(cube.psf_model)[1:2]..., 0)
        end
        psf_out = Vector{eltype(cube.psf)}()
        lsf_out = Vector{eltype(cube.lsf)}()
        # loop through regions and append to the buffers
        for gap_mask in gap_masks
            dλ = (maximum(cube.λ[gap_mask]) - minimum(cube.λ[gap_mask])) / (length(cube.λ[gap_mask])-1) * factor
            linλ = minimum(cube.λ[gap_mask]):dλ:maximum(cube.λ[gap_mask])
            lnλ = get_logarithmic_λ(ustrip.(linλ), logscale) * unit(linλ[1])
            I, σ, mask = resample_flux_permuted3D(lnλ, cube.λ, cube.I, cube.σ, cube.mask)
            psf = Spline1D(ustrip.(cube.λ), ustrip.(cube.psf), k=1, bc="extrapolate")(ustrip.(lnλ)) * unit(cube.psf[1])
            lsf = Spline1D(ustrip.(cube.λ), ustrip.(cube.lsf), k=1, bc="extrapolate")(ustrip.(lnλ)) * unit(cube.lsf[1])
            λ_out = cat(λ_out, lnλ, dims=1)
            I_out = cat(I_out, I, dims=3)
            σ_out = cat(σ_out, σ, dims=3)
            mask_out = cat(mask_out, mask, dims=3)
            if !isnothing(cube.psf_model)
                psf_model = resample_flux_permuted3D(lnλ, cube.λ, cube.psf_model)
                psf_model_out = cat(psf_model_out, psf_model, dims=3)
            end
            psf_out = cat(psf_out, psf, dims=1)
            lsf_out = cat(lsf_out, lsf, dims=1)
        end
        # re-calculate channel masks
        for i in eachindex(cube.spectral_region.channel_masks)
            ch_mask = Spline1D(ustrip.(cube.λ), cube.spectral_region.channel_masks[i], k=1)(ustrip.(λ_out)) .> 0
            cube.spectral_region.channel_masks[i] = ch_mask
        end
        # this is a dumb lazy way to do it so we have to make sure there's no pixels where more than one channel mask is set 
        for i in eachindex(λ_out)
            chmaski = [cube.spectral_region.channel_masks[j][i] for j in eachindex(cube.spectral_region.channel_masks)]
            if sum(chmaski) > 1
                ind = findfirst(chmaski)
                for k in ind+1:length(chmaski)
                    cube.spectral_region.channel_masks[k][i] = 0
                end
            end
        end
        # set them back into the cube object
        cube.λ = λ_out
        cube.I = I_out
        cube.σ = σ_out
        cube.mask = mask_out
        if !isnothing(cube.psf_model)
            cube.psf_model = psf_model_out
        end
        cube.psf = psf_out
        cube.lsf = lsf_out

        cube.log_binned = true
    else
        @debug "The cube has already been log-rebinned! Will not be log-rebinned again."
    end

    cube
end


# Apply a Cardelli extinction correction
function deredden!(cube::DataCube)

    @debug "deredden!(DataCube): channel=$(cube.channel), band=$(cube.band), already_dereddened=$(cube.dereddened)"
    if !cube.dereddened

        # get galactic coordinates
        c = ICRSCoords(cube.α, cube.δ)
        g = convert(GalCoords, c)

        # get E(B-V) from galactic dust maps
        dustmap = SFD98Map()
        E_BV = dustmap(g.l, g.b)
        @info "Using SFD98 dust map at (α=$(cube.α), δ=$(cube.δ)): E(B-V)=$E_BV"

        # use the CCM89 extinction law with Rv = 3.1 for the Milky Way
        # unred = 10 .^ (0.4 .* Av .* ustrip.(CCM89(Rv=Rv).(cube.λ)))
        unred = 1 ./ extinction_cardelli.(cube.λ, E_BV)
        unred = extend(unred, size(cube.I)[1:2])
        cube.I .*= unred
        cube.σ .*= unred

        cube.dereddened = true
    else
        @debug "Cube has already been de-reddened! Will not be de-reddened again."
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

    @debug "interpolate_nans!: channel=$(cube.channel), band=$(cube.band), shape=$(size(cube.I)), n_nan_I=$(count(isnan, ustrip.(cube.I))), n_nan_σ=$(count(isnan, ustrip.(cube.σ)))"
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
        filt = .!isfinite.(I) .| .!isfinite.(σ)

        # Interpolate the NaNs
        if sum(filt) > 0
            @debug "NaNs found in spaxel $index -- interpolating"

            # Literal edge cases
            if !isfinite(I[1])
                iind = findfirst(isfinite.(I))
                I[1:iind-1] .= I[iind]
                σ[1:iind-1] .= σ[iind].*100
            end
            if !isfinite(I[end])
                find = findlast(isfinite.(I))
                I[find+1:end] .= I[find]
                σ[find+1:end] .= σ[find].*100
            end
            cube.I[index, :] .= I .* unit(cube.I[1])
            cube.σ[index, :] .= σ .* unit(cube.σ[1])

            if !isnothing(psf)
                if !isfinite(psf[1])
                    iind = findfirst(isfinite.(psf))
                    psf[1:iind-1] .= psf[iind]
                end
                if !isfinite(psf[end])
                    find = findlast(isfinite.(psf))
                    psf[find+1:end] .= psf[find]
                end
                cube.psf_model[index, :] .= psf .* unit(cube.psf_model[1])
            end

            scale = 7
            finite = isfinite.(I)

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
            cube.I[index, filt] .= Spline1D(λ[isfinite.(I)], I[isfinite.(I)], λknots, k=1, bc="extrapolate")(λ[filt]) .* unit(cube.I[1])
            cube.σ[index, filt] .= Spline1D(λ[isfinite.(σ)], σ[isfinite.(σ)], λknots, k=1, bc="extrapolate")(λ[filt]) .* unit(cube.σ[1])

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
        cube.mask = mask_rot .| iszero.(I_rot)
        cube.psf_model = psf_rot

        # reapply the new mask
        cube.I[cube.mask] .*= NaN
        cube.σ[cube.mask] .*= NaN
        if !isnothing(psf_rot)
            cube.psf_model[cube.mask] .*= NaN
        end

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
function calculate_statistical_errors!(cube::DataCube, 
    lines_file::String=joinpath(@__DIR__, "..", "options", "lines.toml"),
    overrides::Union{Vector{Tuple{T,T}},Nothing}=nothing; 
    mask_width::typeof(1.0u"km/s")=1000.0u"km/s", median::Bool=false) where {T<:QLength}

    λ = cube.λ

    if isnothing(overrides)
        _, cent_vals = parse_lines(lines_file, cube.spectral_region, unit(λ[1]))
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

        # Inflate errors on the "bad" pixels
        σ_stat[mask_bad] .= nanmaximum(ustrip.(σ_stat)) .* 100 .* unit(σ_stat[1])

        # Replace the cube's error with the statistical errors
        if median
            σ_stat .= nanmedian(ustrip.(σ_stat)) .* unit(σ_stat[1])
        end
        cube.σ[spaxel, :] .= σ_stat
    end

end 


"""
    voronoi_rebin!(cube, target_SN[, window])

Calculate Voronoi bins for the cube such that each bin has a signal to noise ratio roughly equal to `target_SN`.
Modifies the cube object in-place with the `voronoi_bins` attribute, which is a 2D array that gives unique integer
labels to each voronoi bin.
"""
function voronoi_rebin!(cube::DataCube, target_SN::Real, window::Union{Tuple{QWave,QWave},Nothing}=nothing,
    bin_strategy::BinningStrategy=WeightedVoronoi(); noise_sigma_clip::Real=10.)

    @debug "voronoi_rebin!: channel=$(cube.channel), band=$(cube.band), shape=$(size(cube.I)), target_SN=$target_SN, window=$window"
    @info "Performing Voronoi rebinning with target S/N=$target_SN"

    # Get the signal and noise 
    if isnothing(window)
        signal = dropdims(nanmedian(ustrip.(cube.I), dims=3), dims=3) .* unit(cube.I[1])
        noise = dropdims(nanmedian(ustrip.(cube.σ), dims=3), dims=3) .* unit(cube.σ[1])
    else
        snmask = window[1] .< cube.λ .< window[2]
        signal = dropdims(nansum(ustrip.(cube.I)[:,:,snmask], dims=3), dims=3) .* unit(cube.I[1])
        noise = sqrt.(dropdims(nansum(ustrip.(cube.σ)[:,:,snmask].^2, dims=3), dims=3)) .* unit(cube.σ[1])
    end
    # x/y coordinate arrays
    x = [i for i in axes(signal,1), _ in axes(signal,2)]
    y = [j for _ in axes(signal,1), j in axes(signal,2)]
    # mask out bad spaxels
    mask = (.~isfinite.(signal)) .| (.~isfinite.(noise)) .| (iszero.(signal)) .| (iszero.(noise))
    # flatten arrays
    signal = signal[.~mask]
    noise = noise[.~mask]
    x = x[.~mask]
    y = y[.~mask]
    # make sure signals are nonnegative
    signal = clamp.(signal, 0*unit(signal[1]), Inf*unit(signal[1]))
    noise = clamp.(noise, 0*unit(noise[1]), Inf*unit(noise[1]))
    # do some sigma clipping on the noise to make sure the overall S/N calculation is not dominated by 
    # a few pixels with super high noise levels. Default clip threshold is 10 sigma, where sigma is 
    # calculated as 1.4826 * MAD (median absolute deviation), which is more robust against outliers than
    # the standard deviation.
    noise_thresh = (nanmedian(ustrip.(noise)) + 1.4826*nanmad(ustrip.(noise))*noise_sigma_clip) * unit(noise[1])
    noise_mask = noise .< noise_thresh
    x = x[noise_mask]
    y = y[noise_mask]
    signal = signal[noise_mask]
    noise = noise[noise_mask]
    # perform voronoi rebinning
    bin_numbers, = voronoi2Dbinning(x, y, ustrip.(signal), ustrip.(noise), target_SN, 1.0, bin_strategy)
    # reformat bin numbers as a 2D array so that we don't need the x/y vectors anymore
    voronoi_bins = zeros(Int, size(cube.I)[1:2])
    for i in eachindex(bin_numbers)
        voronoi_bins[x[i], y[i]] = bin_numbers[i]
    end
    # Set the voronoi_bins value in the cube object
    cube.voronoi_bins = voronoi_bins
    @debug "voronoi_rebin!: done — n_bins=$(length(unique(voronoi_bins[voronoi_bins .> 0])))"
end


function get_physical_scales(shape::Tuple, Ω::typeof(1.0u"sr"), cosmo::Union{Cosmology.AbstractCosmology,Nothing}=nothing, 
    z::Union{Real,Nothing}=nothing)

    # Angular and physical scalebars
    pix_as = uconvert(u"arcsecond", sqrt(Ω))
    n_pix = 1.0/ustrip(pix_as)

    if !isnothing(cosmo) && !isnothing(z)
        @debug "Using angular diameter distance $(angular_diameter_dist(cosmo, z))"
        # Calculate in pc
        dA = angular_diameter_dist(u"pc", cosmo, z)
        # l = d * theta (") where theta is chosen as 1/5 the horizontal extent of the image
        l = dA * ustrip(uconvert(u"rad", shape[1] * pix_as / 5))
        # Round to a nice even number
        l = round(typeof(1u"pc"), l, sigdigits=1)
        # new angular size for this scale
        θ = l / dA * u"rad"
        θ_as = round(u"arcsecond", θ, digits=1)           # should be close to the original theta, by definition
        n_pix = uconvert(NoUnits, 1.0/sqrt(Ω) * θ)   # number of pixels = (pixels per radian) * radians
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
        l_val = Int(ustrip(l))
        scalebar_text_dist = cosmo.h ≈ 1.0 ? L"%$(l_val)$h^{-1}$ %$(new_unit)" : L"$%$(l_val)$ %$(new_unit)"
    else
        θ_as = round(u"arcsecond", shape[1] * pix_as / 5, digits=1)
        n_pix = 1.0/sqrt(Ω) * uconvert(u"sr^(1/2)", θ_as)
        scalebar_text_dist = ""
    end

    # scalebar_text_ang = L"$\ang[angle-symbol-over-decimal]{;;%$(ustrip(θ_as))}$"
    scalebar_text_ang = L"$%$(ustrip(θ_as))''$"
    if θ_as > 60u"arcsecond"
        θ_as = round(u"arcminute", θ_as, digits=1)  # convert to arcminutes
        # scalebar_text_ang = L"$\ang[angle-symbol-over-decimal]{;%$(ustrip(θ_as));}$"
        scalebar_text_ang = L"$%$(ustrip(θ_as))''$"
    end

    pix_as, n_pix, scalebar_text_dist, scalebar_text_ang
end


"""
    make_subcube(data, redshift[, xmin, xmax, ymin, ymax, zmin, zmax])

Extract a subcube from a full cube, restricting to a minimum/maximum range in x, y, and z 
(where z is the wavelength axis).
"""
function make_subcube(data::DataCube, redshift::Real,
                      xmin::Integer=1, xmax::Integer=size(data.I, 1),
                      ymin::Integer=1, ymax::Integer=size(data.I, 2),
                      zmin::Integer=1, zmax::Integer=size(data.I, 3))
    
    @assert 1 ≤ xmin ≤ size(data.I, 1) "xmin must be in range (1,$(size(data.I,1)))"
    @assert 1 ≤ xmax ≤ size(data.I, 1) "xmax must be in range (1,$(size(data.I,1)))"
    @assert 1 ≤ ymin ≤ size(data.I, 2) "ymin must be in range (1,$(size(data.I,2)))"
    @assert 1 ≤ ymax ≤ size(data.I, 2) "ymax must be in range (1,$(size(data.I,2)))"
    @assert 1 ≤ zmin ≤ size(data.I, 3) "zmin must be in range (1,$(size(data.I,3)))"
    @assert 1 ≤ zmax ≤ size(data.I, 3) "zmax must be in range (1,$(size(data.I,3)))"

    @info "Making a sub-cube with slices ($xmin:$xmax,$ymin:$ymax,$zmin:$zmax)"

    # make a deep copy of data so we dont modify any of the original DataCube
    out_data = deepcopy(data)
    
    # modify each attribute in-place
    out_data.nx   = xmax-xmin+1
    out_data.ny   = ymax-ymin+1
    out_data.nz   = zmax-zmin+1

    out_data.λ    = out_data.λ[zmin:zmax]
    out_data.I    = out_data.I[xmin:xmax, ymin:ymax, zmin:zmax]
    out_data.σ    = out_data.σ[xmin:xmax, ymin:ymax, zmin:zmax]
    out_data.mask = out_data.mask[xmin:xmax, ymin:ymax, zmin:zmax]
    if !isnothing(out_data.psf_model)
        out_data.psf_model = out_data.psf_model[xmin:xmax, ymin:ymax, zmin:zmax]
    end
    out_data.psf = out_data.psf[zmin:zmax]
    out_data.lsf = out_data.lsf[zmin:zmax]

    # we need to update the WCS to account for the new pixel coordinates
    if !isnothing(out_data.wcs)
        out_data.wcs = make_offset_wcs_crpix(out_data.wcs, -(xmin-1), -(ymin-1), -(zmin-1))
    end

    # if we slice along the wavelength axis, we need to recalculate the spectral region
    if (zmin != 1) || (zmax != size(data.I, 3))

        restframe_factor = out_data.rest_frame ? 1.0 : 1 / (1 + redshift)
        λlim = extrema(out_data.λ .* restframe_factor)
        λrange = get_λrange(λlim)

        new_umask = umask(out_data.spectral_region)
        umask_to_remove = Int[]
        for i in eachindex(new_umask)
            if new_umask[i][2] < minimum(out_data.λ)
                push!(umask_to_remove, i)
            end
            if new_umask[i][1] > maximum(out_data.λ)
                push!(umask_to_remove, i)
            end
        end
        deleteat!(new_umask, umask_to_remove)

        new_gaps = gaps(out_data.spectral_region)
        gaps_to_remove = Int[]
        for i in eachindex(new_gaps)
            if new_gaps[i][2] < minimum(out_data.λ)
                push!(gaps_to_remove, i)
            end
            if new_gaps[i][1] > maximum(out_data.λ)
                push!(gaps_to_remove, i)
            end
        end
        deleteat!(new_gaps, gaps_to_remove)

        # calculate new n_channels, ch_bounds, channel_masks
        new_n_channels = 0
        new_channel_masks = BitVector[]
        for i in 1:out_data.spectral_region.n_channels
            new_mask = out_data.spectral_region.channel_masks[i][zmin:zmax]
            if sum(new_mask) > 0
                push!(new_channel_masks, new_mask)
                new_n_channels += 1
            end
        end
        new_ch_bounds = out_data.spectral_region.ch_bounds[
            minimum(out_data.λ) .< out_data.spectral_region.ch_bounds .< maximum(out_data.λ)
        ]

        # new spectral region
        out_data.spectral_region = SpectralRegion(
            λlim, new_umask, new_n_channels, new_channel_masks, new_ch_bounds, new_gaps, λrange
        )
    end

    out_vorbins = nothing
    if !isnothing(out_data.voronoi_bins)
        new_vorbins = out_data.voronoi_bins[xmin:xmax, ymin:ymax]
        out_vorbins = zeros(Int, size(new_vorbins))
        # need to remap voronoi bin labels because some may have been completely lost
        @warn "Voronoi bins were found in the full cube. Their labels will be remapped in the subcube."
        bin_labels = unique(new_vorbins[new_vorbins .> 0])
        sort!(bin_labels)
        for (i, bin_label) in enumerate(bin_labels)
            # dont modify new_vorbins in-place because the new labels "i" might overlap with some 
            # old "bin_label" which could mess everything up
            out_vorbins[new_vorbins .== bin_label] .= i
        end
        out_data.voronoi_bins = out_vorbins
    end

    return out_data
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
    disable_psfcirc::Bool=false, aperture::Union{Aperture.AbstractAperture,Nothing}=nothing)

    @debug "Plotting 2D intensity/error map for cube with channel $(data.channel), band $(data.band)"

    if isnothing(slice)
        # Integrate up data along wavelength dimension
        I = [sum(.~data.mask[i,j,:]) > 1 ? 
            NumericalIntegration.integrate(data.λ[.~data.mask[i,j,:]], data.I[i,j,.~data.mask[i,j,:]]) : 
            0*unit(data.I[1])*unit(data.λ[1])
            for i in axes(data.I,1), j in axes(data.I,2)]
        σ = [sum(.~data.mask[i,j,:]) > 1 ? 
            NumericalIntegration.integrate(data.λ[.~data.mask[i,j,:]], data.σ[i,j,.~data.mask[i,j,:]]) : 
            0*unit(data.I[1])*unit(data.λ[1])
            for i in axes(data.σ,1), j in axes(data.σ,2)]
        # Reapply masks
        tzero = I |> eltype |> zero
        I[I .≤ tzero] .*= NaN
        σ[σ .≤ tzero] .*= NaN
        sub = ""
    else
        # Take the wavelength slice
        I = data.I[:, :, slice]
        σ = data.σ[:, :, slice]
        tzero = I |> eltype |> zero
        I[I .≤ tzero] .*= NaN
        σ[σ .≤ tzero] .*= NaN
        λᵢ = data.λ[slice]
        sub = @sprintf "\\lambda%.2f" ustrip(λᵢ)
    end

    # Format units BEFORE taking the log
    unit_str = latex(unit(I[1]))

    # Take logarithm if specified
    if !isnothing(logₑ)
        σ = σ ./ abs.(log(logₑ) .* I)
    end
    if !isnothing(logᵢ)
        I = log.(ustrip.(I)) ./ log(logᵢ)
    end

    ax1 = ax2 = nothing

    # 1D, no NaNs/Infs
    pix_as = uconvert(u"arcsecond", sqrt(data.Ω)) 
    _, n_pix, scalebar_text_dist, scalebar_text_ang = get_physical_scales(size(data.I), data.Ω, cosmo, z)
    colormap.set_bad(color="k")

    fig = plt.figure(figsize=intensity && err ? (12, 6) : (12,12))

    _I = ustrip.(I)
    _σ = ustrip.(σ)
    if intensity
        # Plot intensity on a 2D map
        ax1 = fig.add_subplot(121)
        ax1.set_title(isnothing(name) ? "" : name)
        cdata = ax1.imshow(_I', origin=:lower, cmap=colormap, vmin=nanquantile(_I, 0.01), vmax=nanquantile(_I, 0.99))
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

        if !disable_psfcirc
            r = (isnothing(slice) ? median(data.psf) : data.psf[slice]) / pix_as / 2
            psf = plt.Circle(size(_I) .* (0.93, 0.05) .+ (-r, r), r, color="w")
            ax1.add_patch(psf)
            ax1.annotate("PSF", size(_I) .* (0.93, 0.05) .+ (-r, 2.5r + 1.75), ha=:center, va=:center, color="w")    
        end

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
        cdata = ax2.imshow(_σ', origin=:lower, cmap=colormap, vmin=nanquantile(_σ, 0.01), vmax=nanquantile(_σ, 0.99))
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

        if !disable_psfcirc
            r = (isnothing(slice) ? median(data.psf) : data.psf[slice]) / pix_as / 2
            psf = plt.Circle(size(_σ) .* (0.93, 0.05) .+ (-r, r), r, color="w")
            ax2.add_patch(psf)
            ax2.annotate("PSF", size(_σ) .* (0.93, 0.05) .+ (-r, 2.5r + 1.75), ha=:center, va=:center, color="w")    
        end

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
    Iunit = unit(data.I[1])
    tzero = data.I |> eltype |> zero

    # Alias
    λ = data.λ
    if isnothing(spaxel)
        # Sum up data along spatial dimensions
        I = sumdim(ustrip.(data.I), (1,2)) * Iunit
        σ = sqrt.(sumdim(ustrip.(data.σ).^2, (1,2))) * Iunit
        # Reapply masks
        I[I .≤ tzero] .*= NaN
        σ[σ .≤ tzero] .*= NaN

        # Since this is intensity (per unit solid angle), we pick up an extra spaxel/sr term that must be divided out
        # (in other words, while flux sums over an aperture, intensity averages since the solid angle gets larger)
        I ./= sumdim(Array{Int}(.~data.mask), (1,2))
        σ ./= sumdim(Array{Int}(.~data.mask), (1,2))
    else
        # Just take the spaxel
        I = data.I[spaxel..., :]
        σ = data.σ[spaxel..., :]
    end
    yunit_symb = eltype(I) <: QPerFreq ? "I_{\\nu}" : "I_{\\lambda}"

    if logᵢ ≠ false
        if eltype(I) <: QPerFreq
            ν = uconvert.(u"Hz", C_KMS ./ λ)
            I = ν.*I
            yunit_symb = "{\\nu}I_{\\nu}"
        elseif eltype(I) <: QPerWave
            I = λ.*I
            yunit_symb = "{\\lambda}I_{\\lambda}"
        end
    end

    # Formatting for x and y units on the axis labels
    xunit = latex(unit(λ[1]))
    yunit = latex(unit(I[1]))

    # If specified, take the logarithm of the data with the given base
    if logᵢ ≠ false
        σ = σ ./ abs.(I .* log.(logᵢ)) 
        I = log.(ustrip.(I)) ./ log.(logᵢ)
        λ = log.(ustrip.(λ)) ./ log.(logᵢ)
    end

    # Plot formatting
    _λ = ustrip.(λ)
    _I = ustrip.(I)
    _σ = ustrip.(σ)

    fig, ax = plt.subplots(figsize=(10,5))
    if intensity
        ax.plot(_λ, _I, "k", linestyle=linestyle, label="Data")
    end
    if err && !intensity
        ax.plot(_λ, _σ, "k", linestyle=linestyle, label=L"$1\sigma$ Error")
    end
    if intensity && err
        ax.fill_between(_λ, _I.-_σ, _I.+_σ, color="k", alpha=0.5, label=L"$1\sigma$ Error")
    end
    if logᵢ == false
        ax.set_ylabel(L"$%$(yunit_symb)$ (%$yunit)")
        ax.set_xlabel(L"$\lambda$ (%$xunit)")
    else
        ax.set_ylabel(L"$\log_{%$logᵢ}(%$(yunit_symb)$ / %$yunit)")
        ax.set_xlabel(L"$\log_{%$logᵢ}(\lambda$ / %$xunit)")
    end
    ax.legend(loc="upper right", frameon=false)
    ax.set_xlim(minimum(_λ), maximum(_λ))
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
- `channels dictionary containing the channel 
    numbers as keys and the individual DataCube observations as values.
- `name`: The label for the observation/data
- `z`: the redshift of the source
- `α`: the right ascension of the source, in decimal degrees
- `δ`: the declination of the source, in decimal degrees
- `instrument`: the instrument name, i.e. "MIRI"
- `detector`: the detector name, i.e. "MIRIFULONG"
- `spectral_region`: the spectral region specifications
- `rest_frame`: whether or not the individual DataCubes have been converted to the rest frame
- `masked`: whether or not the individual DataCubes have been masked
- `vacuum_wave`: whether or not the wavelengths are specified in vacuum wavelengths; if false, they are assumed
    to be in air wavelengths
- `sky_aligned`: whether or not the data cube is aligned to the sky (RA/Dec) axes or the IFU (α/β) axes

See also [`DataCube`](@ref)
"""
mutable struct Observation

    channels::Dict{Any,DataCube}

    name::String
    z::Real
    α::typeof(1.0u"°")
    δ::typeof(1.0u"°")
    instrument::String
    rest_frame::Bool
    masked::Bool
    vacuum_wave::Bool
    log_binned::Bool
    dereddened::Bool
    sky_aligned::Bool

    function Observation(channels::Dict{Any,DataCube}=Dict{Any,DataCube}(), name::String="Generic Observation",
        z::Real=NaN, α=NaN*u"°", δ=NaN*u"°", instrument::String="Generic Instrument", rest_frame::Bool=false, 
        masked::Bool=false, vacuum_wave::Bool=true, log_binned::Bool=false, dereddened::Bool=false, 
        sky_aligned::Bool=false)
        new(channels, name, z, α, δ, instrument, rest_frame, masked, vacuum_wave, log_binned, dereddened, sky_aligned)
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
    @debug "save_fits: path=$path, channels=$channels, obs=$(obs.name)"
    for channel in channels

        # Header information
        hdr0_keys = String["TARGNAME", "REDSHIFT", "CHANNEL", "BAND", "TARG_RA", "TARG_DEC", "INSTRUME",
                           "ROTANGLE", "NCHAN", "RESTFRAM", "MASKED", "VACWAVE", "LOGBIN", "DERED", "DATAMODL", "NAXIS"]
        hdr1_keys =  String["NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "PIXAR_SR", "PIXAR_A2"]
            
        hdr0_vals = Any[obs.name, obs.z, string(channel), obs.channels[channel].band, ustrip(obs.α), ustrip(obs.δ), 
                        obs.instrument, ustrip(obs.channels[channel].θ_sky), nchannels(obs.channels[channel].spectral_region), 
                        obs.rest_frame, obs.masked, obs.vacuum_wave, obs.log_binned, obs.dereddened, "IFUCubeModel", 0]
        hdr1_vals = Any[3, size(obs.channels[channel].I, 1), size(obs.channels[channel].I, 2), size(obs.channels[channel].I, 3),
                        ustrip(obs.channels[channel].Ω), ustrip(uconvert(u"arcsecond^2", obs.channels[channel].Ω))]
                
        hdr0_comments = String["Target name", "Target redshift", "Channel", "Band",
                               "Right ascension of target (deg.)", "Declination of target (deg.)",
                               "Instrument name", "rotation angle to sky axes", "number of individual wavelength channels/bands in the data", 
                               "data in rest frame?", "data masked?", "vacuum wavelengths?", "log binned?", "dereddened?", "data model", 
                               "Number of data axes"]
        hdr1_comments = String["length of the first axis", "length of the second axis", "length of the third axis", "Number of data axes",
                               "Nominal pixel area in steradians", "Nominal pixel area in arcsec^2"]
        
        wave_unit_str = string(unit(obs.channels[channel].λ[1]))
        wave_unit_str = replace(wave_unit_str, "Å" => "angstrom", 'μ' => 'u', ' ' => '.', '*' => '.')

        # WCS information to add to header 
        wcsstr = WCS.to_header(obs.channels[channel].wcs)
        for m in eachmatch(r"(.{8})\=(.+?)\ \/\ (.{45})", wcsstr)
            key = strip(m[1])
            push!(hdr1_keys, key)
            val = strip(m[2])
            if occursin("\'", val)
                val = replace(val, "\'" => "")
            else
                val = parse(Float64, val)
            end
            if key == "CTYPE3"
                val = "WAVE-TAB"   # overwrite to make sure we output as a tabular wavelength
            end
            if key == "CUNIT3"
                val = wave_unit_str
            end
            if key in ("CRPIX3", "CRVAL3", "CDELT3")
                val = 1.
            end
            push!(hdr1_vals, val)
            comment = strip(m[3])
            push!(hdr1_comments, comment)
        end

        # Fill in the rest of the PC elements with defaults if they weren't included in the
        # to_header formulation of the WCS
        if any(obs.channels[channel].wcs.pc .> 0)
            for i in 1:3
                for j in 1:3
                    if !("PC$(i)_$(j)" in hdr1_keys)
                        pcij = i == j ? 1. : 0.
                        push!(hdr1_keys, "PC$(i)_$(j)")
                        push!(hdr1_vals, pcij)
                        push!(hdr1_comments, "Coordinate transformation matrix element")
                    end
                end
            end
        end

        # Add MJD-OBS = MJD-AVG because for some reason the JWST cubes dont have it, and it is necessary
        # for astropy's wcs module to perform pixel/world coordinate transforms with tabular wavelength data
        if ("MJD-OBS" ∉ hdr1_keys)
            for alternate in ("MJD-AVG", "MJD-BEG", "MJD-END")
                if alternate in hdr1_keys
                    push!(hdr1_vals, hdr1_vals[findfirst(hdr1_keys .== alternate)]) 
                    push!(hdr1_keys, "MJD-OBS")
                    push!(hdr1_comments, "[d] MJD of observation")
                    break
                end
            end
        end

        # Since we are using a WAVE-TAB header, extra values need to be added
        append!(hdr1_keys, ["PS3_0", "PS3_1"])
        append!(hdr1_vals, ["WCS-TABLE", "wavelength"])
        append!(hdr1_comments, ["Coordinate table extension name", "Coordinate table column name"])

        hdr0 = FITSHeader(hdr0_keys, hdr0_vals, hdr0_comments)
        hdr1 = FITSHeader(hdr1_keys, hdr1_vals, hdr1_comments)

        FITS(joinpath(path, "$(replace(obs.name, " " => "_")).channel$(channel)$(obs.rest_frame ? ".rest_frame" : "").fits"), "w") do f
            @info "Writing FITS file from Observation object"

            write(f, Vector{Int}(), header=hdr0)                                 # Primary HDU (empty)
            write(f, ustrip.(obs.channels[channel].I); name="SCI", header=hdr1)  # Data HDU
            write(f, ustrip.(obs.channels[channel].σ); name="ERR")               # Error HDU
            write(f, UInt8.(obs.channels[channel].mask); name="DQ")              # Mask HDU
            if !isnothing(obs.channels[channel].psf_model)
                write(f, obs.channels[channel].psf_model; name="PSF")            # PSF HDU
            end

            # WCS-TABLE holds the wavelength information
            # Note: we have to reshape to a (1 x N_WAVE x 1) array so that it saves to the FITS file as a row vector 
            #       rather than a column vector.  It seems like this is the standard when using tabular WCS data, based
            #       on how the JWST multi-channel cubes are formatted.
            write(f, ["wavelength"], [reshape(ustrip.(obs.channels[channel].λ), (1, length(obs.channels[channel].λ), 1))], 
                     hdutype=TableHDU, name="WCS-TABLE")

            # Auxiliary HDU
            write(f, ["psf", "lsf"], [ustrip.(obs.channels[channel].psf), ustrip.(obs.channels[channel].lsf)],
                     hdutype=TableHDU, name="AUX")

            # Gaps HDU
            write(f, ["gaps_1", "gaps_2"],
                      [Float64[ustrip.(g[1]) for g in obs.channels[channel].spectral_region.gaps], 
                       Float64[ustrip.(g[2]) for g in obs.channels[channel].spectral_region.gaps]], 
                      hdutype=TableHDU, name="GAP")

            unit_str = string(unit(obs.channels[channel].I[1]))
            unit_str = replace(unit_str, "Å" => "angstrom", 'μ' => 'u', ' ' => '.', '*' => '.')
            write_key(f["SCI"], "BUNIT", unit_str)
            write_key(f["ERR"], "BUNIT", unit_str)
            write_key(f["WCS-TABLE"], "TDIM2", "(1,$(length(obs.channels[channel].λ)))", "Wavetable dimension")
            write_key(f["GAP"], "UNIT1", wave_unit_str)
            write_key(f["GAP"], "UNIT2", wave_unit_str)
            unit_str = string(unit(obs.channels[channel].psf[1]))
            write_key(f["AUX"], "UNIT1", unit_str)
            unit_str = string(unit(obs.channels[channel].lsf[1]))
            write_key(f["AUX"], "UNIT2", unit_str)
        end
    end
end


"""
    from_fits(filenames[, z, user_mask, format])

Create an Observation object from a series of fits files with IFU cubes in different channels.

# Arguments
- `filenames`: A vector of filepaths to the FITS files
- `z`: The redshift of the object.
- `user_mask`: An optional vector of tuples specifying regions to be masked out
- `format`: The format of the FITS files 
"""
function from_fits(filenames::Vector{String}, z::Real=0.; user_mask=nothing)

    # Grab object information from the FITS header of the first file
    channels = Dict{Any,DataCube}()
    hdu = FITS(filenames[1])
    hdr = read_header(hdu[1])
    name = hdr["TARGNAME"]
    ra = hdr["TARG_RA"] * u"°"
    dec = hdr["TARG_DEC"] * u"°"
    inst = hdr["INSTRUME"]
    format = Symbol(inst)
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

    gratings_nir = ["G140M", "G235M", "G395M", "G140H", "G235H", "G395H", "PRISM"]
    filters_nir = ["F070LP", "F100LP", "F170LP", "F290LP", "CLEAR"]

    # Loop through the files and call the individual DataCube method of the from_fits function
    for (i, filepath) ∈ enumerate(filenames)
        cube = from_fits(filepath, z)
        # checks for JWST-formatted individual channel/band cubes
        if (format == :MIRI) && (cube.band in keys(bands)) && (
            (cube.channel in string.(1:4)) ||
            (cube.channel in "A" .* string.(1:4)) || 
            (cube.channel in "B" .* string.(1:4)) || 
            (cube.channel in "C" .* string.(1:4))
            )
            channels[Symbol(bands[cube.band] * cube.channel[end:end])] = cube
            continue
        end
        if (format == :NIRSPEC) && (cube.band in gratings_nir) && (
            (cube.channel in filters_nir) || 
            (cube.channel in ("G140M_F070LP", "G140M_F100LP", "G235M_F170LP", "G395M_F290LP")) ||
            (cube.channel in ("G140H_F070LP", "G140H_F100LP", "G235H_F170LP", "G395H_F290LP")) || 
            (cube.channel == "PRISM_CLEAR")
            )
            channels[Symbol(cube.band * "_" * split(cube.channel, "_")[end])] = cube
            continue
        end
        try
            # otherwise just use the channel number
            channels[parse(Int, cube.channel)] = cube
            continue
        catch
            # fall back on the channel label itself
            channels[cube.channel] = cube
        end
    end

    from_cubes(name, redshift, collect(values(channels)), collect(keys(channels)), inst=inst)
end


function from_cubes(name::String, redshift::Real, cubes::Vector{<:DataCube}, channel_names::Vector;
    inst::String="Generic Instrument")

    @debug "from_cubes: name=$name, redshift=$redshift, n_cubes=$(length(cubes)), channel_names=$channel_names"
    channels = Dict{Any,DataCube}()
    for (chan, cube) in zip(channel_names, cubes)
        channels[chan] = cube
    end
    rest_frame = all([channels[ch].rest_frame for ch in keys(channels)])
    masked = all([channels[ch].masked for ch in keys(channels)])
    vacuum_wave = all([channels[ch].vacuum_wave for ch in keys(channels)])
    log_binned = all([channels[ch].log_binned for ch in keys(channels)])
    dereddened = all([channels[ch].dereddened for ch in keys(channels)])
    sky_aligned = all([channels[ch].sky_aligned for ch in keys(channels)])
    ra = cubes[1].α
    dec = cubes[1].δ

    Observation(channels, name, redshift, ra, dec, inst, rest_frame, masked, vacuum_wave, log_binned, 
        dereddened, sky_aligned)
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
function to_vacuum_wavelength!(obs::Observation)

    @debug """\n
    Converting observation of $(obs.name) to vacuum wavelengths
    ###########################################################
    """
    # Loop through the channels and call the individual DataCube method of the to_vacuum_wavelength function
    for k ∈ keys(obs.channels)
        to_vacuum_wavelength!(obs.channels[k])
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


function log_rebin!(obs::Observation, factor::Integer=1)

    @debug """\n
    Rebinning observation of $(obs.name) to logarithmic wavelength vector
    #####################################################################
    """

    for k ∈ keys(obs.channels)
        log_rebin!(obs.channels[k], obs.z; factor=factor)
    end
    obs.log_binned = true
    
    obs
end


"""
    deredden!(obs[, E_BV])

De-redden each DataCube using a given E(B-V) value

# Arguments 
- `obs::Observation`: The Observation object to de-redden
"""
function deredden!(obs::Observation)

    @debug """\n
    Dereddening observation of $(obs.name)
    ######################################
    """

    for k ∈ keys(obs.channels)
        deredden!(obs.channels[k])
    end
    obs.dereddened = true

    obs
end


"""
    correct!

A composition of the many functions for Observation objects that convert the data into a fittable format:
    - Convert to vacuum wavelengths
    - Convert to rest-frame wavelengths
    - Logarithmically rebin 
    - Apply the bad pixel mask
    - De-redden the data

See [`apply_mask!`](@ref) and [`to_rest_frame!`](@ref)
"""
correct! = apply_mask! ∘ log_rebin! ∘ to_rest_frame! ∘ deredden! ∘ to_vacuum_wavelength!


"""
    make_subcube(obs[, xmin, xmax, ymin, ymax])

Extract a sub-cube from each DataCube in an observation, restricted to coordinates 
from (xmin:xmax, ymin:ymax). This requires all cubes in the observation to be projected 
onto the same coordinate grid.
"""
function make_subcube(obs::Observation,
                      xmin::Integer=1, xmax::Integer=size(first(values(obs.channels)).I, 1),
                      ymin::Integer=1, ymax::Integer=size(first(values(obs.channels)).I, 2))
    
    # make subcubes for each datacube in the observation.
    # note: this only works if all subcubes are the same size!
    fsize = size(first(values(obs.channels)).I)
    size_match = [size(cube.I) == fsize for cube in values(obs.channels)]
    @assert all(size_match) "Cannot make a subcube of an observation with mismatching DataCube sizes! Project everything onto the same coordinate axis first."
    
    new_channels = Dict{Any,DataCube}()
    for channel in keys(obs.channels)
        new_channels[channel] = make_subcube(obs.channels[channel], obs.z, xmin, xmax, ymin, ymax)
    end

    # make a new observation (dont overwrite the old one)
    return Observation(new_channels, obs.name, obs.z, obs.α, obs.δ, obs.instrument, 
                       obs.rest_frame, obs.masked, obs.vacuum_wave, obs.log_binned, obs.dereddened,
                       obs.sky_aligned)
end


#################################### CHANNEL ALIGNMENT AND REPROJECTION ######################################


"""
    adjust_wcs_alignment(obs, channels; box_size_as=1.5)

Adjust the WCS alignments of each channel such that they match.
This is performed in order to remove discontinuous jumps in the flux level when crossing between
sub-channels.
"""
function adjust_wcs_alignment!(obs::Observation, channels; box_size_as::Float64=1.5)

    @assert obs.instrument in ("MIRI", "NIRSPEC") "Adjust WCS alignment is currently only supported for MIRI/MRS or NIRSPEC/IFU observations!"
    @info "Aligning World Coordinate Systems for channels $channels..."

    # Prepare array of centroids for each channel
    c_coords = zeros(2*length(channels)-2, 2)
    offsets = zeros(length(channels), 2)

    # box_size in arcseconds -- adjust so that it's the same *physical* size in all channels
    box_sizes = zeros(Int, length(channels))
    for i in eachindex(channels)
        as_per_pix = uconvert(u"arcsecond", sqrt(obs.channels[channels[i]].Ω))
        box_sizes[i] = fld(box_size_as, ustrip(as_per_pix))
    end

    k = 1
    for (i, channel) ∈ enumerate(channels)

        @assert haskey(obs.channels, channel) "Channel $channel does not exist!"

        # Get summed intensity map and WCS object for each channel
        ch_data = obs.channels[channel]
        
        # Get the continuum fluxes at the beginning and end of each channel (except the first/last)
        filter_right = ch_data.λ .> (ch_data.λ[end] - 0.1u"μm")
        filter_left = ch_data.λ .< (ch_data.λ[begin] + 0.1u"μm")
        if i == 1
            filters = [filter_right]
        elseif i < length(channels)
            filters = [filter_left, filter_right]
        else
            filters = [filter_left]
        end

        for filter in filters
            data = sumdim(ustrip.(ch_data.I[:, :, filter]), 3)
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
    get_optimal_2dwcs(observations; frame, refpoint, resolution, projection)

Given 2 or more datacubes of one object, find the optimal WCS projection that
will cover the full range of all datacubes.
"""
function get_optimal_2dwcs(cubes::Vector{<:DataCube}; frame::Union{String,Nothing}=nothing, refpoint=nothing,
    resolution=nothing, projection::String="TAN")

    @debug "get_optimal_2dwcs: n_cubes=$(length(cubes)), frame=$frame, projection=$projection"
    # First we need to create an optimal output WCS to reproject all of the inputs onto.
    # This code is heavily inspired (copied) by the python reproject.mosaicking package's "find_optimal_celestial_wcs" function
    n_cubes = length(cubes)
    @assert n_cubes ≥ 2 "Please input at least two cubes to compute an optimal WCS for!"

    # Keep track of all corners, reference coordinates of those corners, and resolutions 
    corners = []
    references = []
    resolutions = []

    out_sframe = cubes[1].wcs.radesys
    out_frame = string_to_coordframe(out_sframe)
    if !isnothing(frame)
        out_sframe = frame
        out_frame = string_to_coordframe(frame)
    end

    for (i, cube) in enumerate(cubes)

        wcs = cube.wcs
        nx, ny = (cube.nx, cube.ny)

        # Get the reference frame and units of the WCS 
        sframe = wcs.radesys
        frame_i = string_to_coordframe(sframe)
        units_i = parse_cunits(wcs)[1:2]

        # pixel coordinates of the corners
        xc = [0.5, nx+0.5, nx+0.5, 0.5]
        yc = [0.5, 0.5, ny+0.5, ny+0.5]
        wc = ones(Float64, 4)  # (we dont care about the wavelength axis)

        # get the coordinates of the corners in the world frame
        wc = pix_to_world(wcs, Matrix([xc yc wc]'))
        for i in axes(wc, 2)
            sc = frame_i((wc[1:2,i].*units_i)...)   # this creates a SkyCoords.jl object
            fc = convert(out_frame, sc)          # transform to the ouptut frame 
            push!(corners, fc)
        end

        # now we need the reference coordinate for this image in the frame of the first image
        xp, yp, _ = wcs.crpix
        wp = pix_to_world(wcs, [xp, yp, 1.])
        sp = frame_i((wp[1:2].*units_i)...)
        push!(references, convert(out_frame, sp))

        # finally, we need the resolution of the image
        pixel_scales = get_pixel_resolutions(wcs)
        append!(resolutions, pixel_scales[1:2].*units_i)
    end

    # If no reference point is explicitly given, just default to the mean of all the reference points
    if isnothing(refpoint)
        refpoint = out_frame(
            mean([references[i].ra  for i in eachindex(references)]),
            mean([references[i].dec for i in eachindex(references)])
        )
    end
    refpoint = convert(out_frame, refpoint)

    # If no resolution is specified, use the minimum of the input resolutions
    if isnothing(resolution)
        resolution = minimum(resolutions)
    end

    # Construct the output WCS
    wcs_out = WCSTransform(2;
        crval=ustrip.(uconvert.(u"°", [refpoint.ra, refpoint.dec])),
        cdelt=ustrip.(uconvert.(u"°", [-1*resolution, resolution])),
        crpix=[1., 1.],  # temporary
        ctype=["RA---", "DEC--"] .* projection,
        cunit=["deg", "deg"],
        pc=diagm(ones(2))
    )

    # Figure out where all of the corners are in the output WCS projection
    corners_sorted = zeros(2, length(corners))
    for i in axes(corners_sorted, 2)
        corners_sorted[:,i] .= (corners[i].ra, corners[i].dec)
    end
    pp = world_to_pix(wcs_out, ustrip.(uconvert.(u"°", corners_sorted)))
    xp = pp[1,:]
    yp = pp[2,:]

    xmin, xmax = extrema(xp)
    ymin, ymax = extrema(yp)

    # Update crpix, remembering that the bottom-left corner should be at (0.5, 0.5) since the first pixel is centered at (1,1)
    setproperty!(wcs_out, :crpix, [(1-xmin)+0.5, (1-ymin)+0.5])

    # this is stupid but it's the only way that works for some reason...it has to be set *last* as changing any other property 
    # causes it to go back to being blank
    setproperty!(wcs_out, :radesys, out_sframe)

    naxis1 = round(Int, xmax - xmin + 1)
    naxis2 = round(Int, ymax - ymin + 1)

    return wcs_out, (naxis1, naxis2)
end


"""
    reproject_cubes!(cubes; order, output_wcs_frame, enforce_all_cubes_out, <get_optimal_2dwcs options>)

Reproject all channels onto a common WCS grid. Modifies the Observation object in-place.

# Arguments
- `cubes::Vector{<:DataCube}`: The DataCube objects to reproject
- `order::Integer=1`: The order of interpolation for the reprojection. 
        -1 = do not do any reprojection (all input channels must be on the same WCS grid)
        0 = nearest-neighbor
        1 = linear
        2 = quadratic
- `output_wcs_frame::Integer=0`: Which WCS frame to project the inputs onto (index into the cubes vector).
        The default option, 0, will calculate a new WCS frame that is optimal for projection for all cubes.
- `enforce_all_cubes_out::Bool=false`: If set to true, will check each spaxel after reprojecting cubes to see
    if there are any spaxels where one cube gets completely masked out (i.e. its outside its footprint), and 
    removes all data from other channels for this spaxel.  If projecting cubes from *different* channels, it's
    recommended to enable this option.  If projecting cubes from *the same* channel, then it's recommended to
    keep this disabled.
"""
function reproject_cubes!(cubes::Vector{<:DataCube}; order::Integer=1, output_wcs_frame::Integer=0,
    enforce_all_cubes_out::Bool=false, frame::Union{String,Nothing}=nothing, refpoint=nothing, resolution=nothing,
    projection::String="TAN")

    @debug "reproject_cubes!: n_cubes=$(length(cubes)), order=$order, output_wcs_frame=$output_wcs_frame, enforce_all=$enforce_all_cubes_out"
    # Pixel sizes in each channel (in arcseconds)
    do_psf_model = all([!isnothing(cube.psf_model) for cube ∈ cubes])

    # Get the optimal WCS for reprojection
    if output_wcs_frame == 0
        wcs_optimal, size_optimal = get_optimal_2dwcs(cubes; frame=frame, refpoint=refpoint, resolution=resolution,
            projection=projection)
    else
        wcs_optimal_3d = cubes[output_wcs_frame].wcs
        wcs_optimal = shrink_wcs_dimensions(wcs_optimal_3d)
        size_optimal = size(cubes[output_wcs_frame].I)[1:2]
    end
    θ_sky_out = get_sky_rotation_angle(wcs_optimal)

    # Iteration variables to keep track of
    for i ∈ eachindex(cubes)

        cube = cubes[i]
        @info "Reprojecting channel $(cube.channel) $(cube.band) onto the optimal $(size_optimal) WCS grid..."

        # Get the intensity and error arrays
        # NOTE: We need to resample the INTENSITY, not the flux, because the pixel sizes may be different between the input and output images,
        #       and flux scales with the pixel size whereas intensity does not (assuming it's an extended source).
        I_in = ustrip.(cube.I)
        σI_in = ustrip.(cube.σ)
        mask_in = cube.mask
        psf_in = cube.psf_model

        # Replace NaNs with 0s for the interpolation
        I_in[.~isfinite.(I_in)] .= 0.
        σI_in[.~isfinite.(σI_in)] .= 0.
        if do_psf_model
            psf_in[.~isfinite.(psf_in)] .= 0.
        end

        # Get 2D WCS representation
        wcs_channel = shrink_wcs_dimensions(cube.wcs)
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
        cube.I = I_out_ch .* unit(cube.I[1])
        cube.σ = σI_out_ch .* unit(cube.I[1])
        cube.mask = mask_out_ch
        if do_psf_model
            cube.psf_model = psf_out_ch
        end
        cube.wcs = extend_wcs_dimensions(wcs_optimal, cube.λ)
        cube.θ_sky = θ_sky_out
        cube.sky_aligned = iszero(θ_sky_out)
    end
    
    if enforce_all_cubes_out
        # If an entire cube is masked out, we want to throw away all cubes for this spaxel
        # This has to be done after the first loop so that mask_out is not overwritten to be unmasked after it has been masked
        for c ∈ CartesianIndices(size(cubes[1].I)[1:2])
            test = [all(cubes[i].mask[c, :]) for i ∈ eachindex(cubes)]
            if any(test)
                for i ∈ eachindex(cubes)
                    cubes[i].mask[c, :] .= 1
                end
            end
        end
    end
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
function extract_from_aperture!(obs::Observation, channels::Vector, ap_r::Real; output_wcs_frame::Integer=1,
    conical::Bool=false)

    @debug "extract_from_aperture!: obs=$(obs.name), channels=$channels, ap_r=$ap_r, conical=$conical"
    pixel_scale = uconvert(u"arcsecond", sqrt(obs.channels[channels[output_wcs_frame]].Ω))

    for ch_in ∈ channels
        mm = .~isfinite.(obs.channels[ch_in].I) .| .~isfinite.(obs.channels[ch_in].σ)
        obs.channels[ch_in].I[mm] .= zero(eltype(obs.channels[ch_in].I))
        obs.channels[ch_in].σ[mm] .= zero(eltype(obs.channels[ch_in].σ))
        if !isnothing(obs.channels[ch_in].psf_model)
            obs.channels[ch_in].psf_model[mm] .= zero(eltype(obs.channels[ch_in].psf_model))
        end

        @info "Channel $ch_in: Extracting from an aperture of size $ap_r x FWHM..."
        @showprogress for wi in axes(obs.channels[ch_in].I, 3)
            # aperture radius in units of the PSF FWHM
            if conical
                ap_size = ap_r * obs.channels[ch_in].psf[wi] / pixel_scale
            else
                ap_size = ap_r * maximum(obs.channels[ch_in].psf) / pixel_scale
            end

            I_slice = copy(obs.channels[ch_in].I[:,:,wi])
            σ_slice = copy(obs.channels[ch_in].σ[:,:,wi])
            psf_slice = nothing
            if !isnothing(obs.channels[ch_in].psf_model)
                psf_slice = copy(obs.channels[ch_in].psf_model[:,:,wi])
            end
            for c ∈ CartesianIndices(size(obs.channels[ch_in].I)[1:2])
                aperture_spax = CircularAperture(c.I..., ap_size)
                obs.channels[ch_in].I[c, wi] = photometry(aperture_spax, I_slice).aperture_sum / get_area(aperture_spax)
                obs.channels[ch_in].σ[c, wi] = sqrt(photometry(aperture_spax, σ_slice.^2).aperture_sum) / get_area(aperture_spax)
                if !isnothing(obs.channels[ch_in].psf_model)
                    obs.channels[ch_in].psf_model[c, wi] = photometry(aperture_spax, psf_slice).aperture_sum / get_area(aperture_spax)
                end
            end
        end

        obs.channels[ch_in].I[mm] .*= NaN
        obs.channels[ch_in].σ[mm] .*= NaN
        if !isnothing(obs.channels[ch_in].psf_model)
            obs.channels[ch_in].psf_model[mm] .*= NaN
        end
    end
end


# Calculate the PSF FWHM in pixel units
function calc_psf_fwhm(cube::DataCube)

    @debug "calc_psf_fwhm: channel=$(cube.channel), band=$(cube.band), nλ=$(size(cube.I,3))"
    fwhms = zeros(size(cube.I, 3))
    data2d = dropdims(nansum(cube.psf_model, dims=3), dims=3)
    data2d[.~isfinite.(data2d)] .= 0.
    _, mx = findmax(data2d)
    # Get centroid
    c = centroid_com(data2d[mx[1]-5:mx[1]+5, mx[2]-5:mx[2]+5]) .+ (mx.I .- 5) .- 1

    for wi in axes(cube.I, 3)

        # Create a 1D PSF profile
        r = 0:0.5:10
        f = zeros(20)
        for i in eachindex(f)
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

    @debug "calc_psf_fwhm: fwhm_range=($(minimum(fwhms)),$(maximum(fwhms))) pixels"
    fwhms
end


"""
    apply_gaussian_smoothing!(obs, channels; max_λ)

Blur the data using a gaussian kernel such that the psf size matches the maximum psf size in the full data cube.
"""
function apply_gaussian_smoothing!(obs::Observation, channels::Vector; output_wcs_frame::Integer=1,
    max_λ::QLength=Inf*u"μm")

    @debug "apply_gaussian_smoothing!: obs=$(obs.name), channels=$channels, max_λ=$(ustrip(max_λ)) $(unit(max_λ))"
    pixel_scale = uconvert(u"arcsecond", sqrt(obs.channels[channels[output_wcs_frame]].Ω))
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
    tzero = obs.channels[channels[1]].I[1] |> typeof |> zero
    szero = obs.channels[channels[1]].σ[1]^2 |> typeof |> zero

    for ch_in ∈ channels
        mm = .~isfinite.(obs.channels[ch_in].I) .| .~isfinite.(obs.channels[ch_in].σ)
        obs.channels[ch_in].I[mm] .*= 0.
        obs.channels[ch_in].σ[mm] .*= 0.
        psf_ch = obs.channels[ch_in].psf ./ pixel_scale
        if !isnothing(obs.channels[ch_in].psf_model)
            obs.channels[ch_in].psf_model[mm] .*= 0.
            psf_ch = calc_psf_fwhm(obs.channels[ch_in])
        end

        @info "Smoothing the data with a gaussian kernel to match PSF sizes"
        @showprogress for wi in axes(obs.channels[ch_in].I, 3)
            in_fwhm = psf_ch[wi]                             # current FWHM in pixels
            FWHM_blur = √(max(0., out_fwhm^2 - in_fwhm^2))   # FWHM needed to blur the data to
            σ_blur = FWHM_blur/2.355                         # sigma needed to blur the data to in pixels 

            # Apply the filtering
            obs.channels[ch_in].I[:, :, wi] .= imfilter(obs.channels[ch_in].I[:, :, wi], Kernel.gaussian(σ_blur))
            σ² = imfilter(obs.channels[ch_in].σ[:, :, wi].^2, Kernel.gaussian(σ_blur))
            σ²[σ² .≤ szero] .= szero
            obs.channels[ch_in].σ[:, :, wi] .= sqrt.(σ²)
            if !isnothing(obs.channels[ch_in].psf_model)
                obs.channels[ch_in].psf_model[:, :, wi] .= imfilter(obs.channels[ch_in].psf_model[:, :, wi], Kernel.gaussian(σ_blur))
            end
        end

        obs.channels[ch_in].I[mm] .*= NaN
        obs.channels[ch_in].σ[mm] .*= NaN
        if !isnothing(obs.channels[ch_in].psf_model)
            obs.channels[ch_in].psf_model[mm] .*= NaN
        end
    end

    out_fwhm * pixel_scale
end


"""
    resample_channel_wavelengths!(λ_out, jumps, I_out, σ_out, mask_out[, psf_out, concat_type])

If resampling subchannels (concat_type = :sub), resamples `I_out`, `σ_out`, `mask_out`, and `psf_out` onto
a linear wavelength grid while conserving flux. Otherwise (concat_type = :full), only resample the parts of the
spectrum that are overlapping between multiple subchannels -- which are resampled onto a median resolution between
the surrounding channels. The inputs are expected to already be concatenated from all of the different channels/subchannels.

# Arguments
- `λ_out`: The 1D concatenated wavelength vector for all channels
- `jumps`: The indices in `λ_out` that specify the channel boundaries (i.e. where the diff of λ_out is negative)
- `I_out`: The 3D concatenated intensity array for all channels
- `σ_out`: The 3D concatenated error array for all channels
- `mask_out`: The 3D concatenated mask array for all channels
- `psf_out`: The 3D concatenated PSF model array for all channels (optional)
- `concat_type`: Either :sub for subchannels or :full for full channels

"""
function resample_channel_wavelengths!(λ_out::Vector{<:QWave}, jumps::Vector{<:Integer}, I_out::Array{<:QSIntensity,3},
    σ_out::Array{<:QSIntensity,3}, mask_out::BitArray{3}, psf_out::Union{Array{<:Real,3},Nothing}=nothing,
    concat_type::Symbol=:sub)

    @debug "resample_channel_wavelengths!: nλ=$(length(λ_out)), n_jumps=$(length(jumps)), concat_type=$concat_type, do_psf=$(psf_out !== nothing)"
    do_psf_model = !isnothing(psf_out)

    if concat_type == :full
        λ_con = zeros(eltype(λ_out), 0)
        I_con = zeros(eltype(I_out), size(I_out)[1:2]..., 0)
        σ_con = zeros(eltype(σ_out), size(σ_out)[1:2]..., 0)
        mask_con = falses(size(mask_out)[1:2]..., 0)
        if do_psf_model
            psf_con = zeros(eltype(psf_out), size(I_out)[1:2]..., 0)
        end
        prev_i2 = 0

        for (i, jump) ∈ enumerate(jumps)
            # find the full scale of the overlapping region
            wave_left, wave_right = λ_out[jump+1], λ_out[jump]
            _, i1 = findmin(abs.(λ_out[1:jump] .- wave_left))
            _, i2 = findmin(abs.(λ_out[jump+1:end] .- wave_right))
            i2 += jump
            # resample fluxes in the overlapping regions
            λ_res = median([diff(λ_out[i1:jump])[1], diff(λ_out[jump+1:i2])[1]])
            λ_resamp = collect((λ_out[i1]*(1+eps())):λ_res:(λ_out[i2]*(1-eps())))
            ss = sortperm(λ_out[i1:i2])
            I_resamp, σ_resamp, mask_resamp = resample_flux_permuted3D(λ_resamp, 
                λ_out[i1:i2][ss], I_out[:, :, (i1:i2)[ss]], σ_out[:, :, (i1:i2)[ss]], mask_out[:, :, (i1:i2)[ss]])
            if do_psf_model
                psf_resamp = resample_flux_permuted3D(λ_resamp, λ_out[i1:i2][ss], psf_out[:, :, (i1:i2)[ss]])
            end
            # replace overlapping regions in outputs
            λ_con = [λ_con; λ_out[prev_i2+1:i1-1]; λ_resamp]
            I_con = cat(I_con, I_out[:, :, prev_i2+1:i1-1], I_resamp, dims=3)
            σ_con = cat(σ_con, σ_out[:, :, prev_i2+1:i1-1], σ_resamp, dims=3)
            mask_con = cat(mask_con, mask_out[:, :, prev_i2+1:i1-1], mask_resamp, dims=3)
            if do_psf_model
                psf_con = cat(psf_con, psf_out[:, :, prev_i2+1:i1-1], psf_resamp, dims=3)
            end
            prev_i2 = i2
        end

        λ_out = [λ_con; λ_out[prev_i2+1:end]]
        I_out = cat(I_con, I_out[:, :, prev_i2+1:end], dims=3)
        σ_out = cat(σ_con, σ_out[:, :, prev_i2+1:end], dims=3)
        mask_out = cat(mask_con, mask_out[:, :, prev_i2+1:end], dims=3)
        if do_psf_model
            psf_out = cat(psf_con, psf_out[:, :, prev_i2+1:end], dims=3)
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

    # fix some weirdness that may happen to the pixel mask at channel boundaries
    dmask = diff(sumdim(mask_out, (1,2)))
    dmask = [dmask[1]; dmask]
    bad = findall(dmask .> 100)
    for b in bad[bad .> 1]
        mask_out[:,:,b] .= mask_out[:,:,b-1]
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
    3. Resample along the spectral axis onto a linear wavelength grid while conserving flux

# Arguments
- `obs`: The Observation object to rebin
- `channels`: The list of channels to be rebinned. If nothing, rebin all channels.
- `concat_type`: Should be :full for overall channels or :sub for subchannels (bands).
- `out_id`: The dictionary key corresponding to the newly rebinned cube, defaults to 0.
- `instrument_channel_edges`: (Optional) Pairs of wavelengths defining the left and right edges of each channel that will be 
    in the combined cube. Not necessary to specify for MIRI/MRS cubes, but necessary for all other cubes.
- `order`: The order of interpolation for the reprojection. 
    -1 = do not do any reprojection (all input channels must be on the same WCS grid)
    0 = nearest-neighbor
    1 = linear
    2 = quadratic
- `adjust_wcs_headerinfo`: Whether or not to try to automatically adjust the WCS header info of the channels by 
    calculating centroids at the boundaries between the channels and forcing them to match.  On by default.
- `min_λ`: Minimum wavelength cutoff for the output cube.
- `max_λ`: Maximum wavelength cutoff for the output cube. Default of 27 um due to degradation above this in MIRI.
- `output_wcs_frame`: Which WCS frame to project the inputs onto. Defaults to 1 (the first channel in channels).
- `extract_from_ap`: The size of the aperture to extract from at each spaxel, in units of the PSF FWHM. If 0, just takes the single spaxel.
    This may be necessary for MIRI data to reduce resampling noise.
- `match_psf`: If true, convolves the data with a gaussian kernel such that the spatial PSF sizes match
- `enforce_all_cubes_out`: If true, will enforce the same field of view over all combined cubes. That is, the output cube will
    cover only regions where all input cubes have data.  Otherwise, all input data will be retained regardless of whether it
    has corresponding data in other channels.
- `user_mask`: An optional set of tuples with wavelength pairs - wavelengths between each pair will be masked out
    during fitting of the output cube
"""
function combine_channels!(obs::Observation, channels=nothing, concat_type=:full; out_id=0,
    instrument_channel_edges::Union{Vector{<:QWave},Nothing}=nothing,
    order::Union{Integer,String}=1, adjust_wcs_headerinfo::Bool=false, min_λ::QWave=0.0*u"μm", max_λ=27.0*u"μm", 
    output_wcs_frame::Integer=1, extract_from_ap::Real=0., match_psf::Bool=false, enforce_all_cubes_out::Bool=true,
    user_mask::Union{Vector{Tuple{W,W}},Nothing}=nothing, gap_mask::Union{Vector{Tuple{W,W}},Nothing}=nothing) where {W<:QWave}

    @debug "combine_channels!: obs=$(obs.name), channels=$channels, concat_type=$concat_type, order=$order, extract_from_ap=$extract_from_ap, match_psf=$match_psf"
    format = Symbol(obs.instrument)

    # Default to all 4 channels
    if (obs.instrument == "MIRI") && isnothing(channels)
        channels = [1, 2, 3, 4]
        concat_type = :full
    end
    # If a single channel is given, interpret as interpolating all of the subchannels
    if (obs.instrument == "MIRI") && !(typeof(channels) <: Vector)
        channels = [Symbol("A" * string(channels)), Symbol("B" * string(channels)), Symbol("C" * string(channels))]
        concat_type = :sub
    end
    if (obs.instrument == "NIRSPEC") && isnothing(channels)
        error("Please specify at least two channels(grating/filter combos) to combine.")
    end
    if (obs.instrument == "NIRSPEC") && !(typeof(channels) <: Vector)
        error("Please specify at least two channels(grating/filter combos) to combine.")
    end
    # Check that channels and channel edges aren't still nothing
    if isnothing(channels)
        error("It appears that this isn't a MIRI/MRS cube. Please explicitly specify the channels to be combined.")
    end
    if isnothing(instrument_channel_edges) && (obs.instrument != "MIRI") && (obs.instrument != "NIRSPEC")
        error("It appears that this isn't a MIRI/MRS cube. Please explicity specify the edges of each channel for this instrument.")
    end

    # 0. First and foremost -- adjust the WCS alignment of each channel so that they are consistent with each other
    if adjust_wcs_headerinfo && (length(channels) > 1)
        adjust_wcs_alignment!(obs, channels; box_size_as=1.5)
    end

    # 1. Reproject the channels onto the same WCS grid
    Ω_out = obs.channels[channels[output_wcs_frame]].Ω
    wcs_optimal_3d = obs.channels[channels[output_wcs_frame]].wcs
    if length(channels) > 1
        cubes = [obs.channels[channel] for channel in channels]
        reproject_cubes!(cubes; order=order, output_wcs_frame=output_wcs_frame, enforce_all_cubes_out=enforce_all_cubes_out)
    end

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

    # 3. Resample along the spectral axis onto a linear wavelength grid while conserving flux
    jumps = findall(diff(λ_out) .< 0*unit(λ_out[1]))
    λ_out, I_out, σ_out, mask_out, psf_out = resample_channel_wavelengths!(λ_out, jumps, I_out, σ_out, mask_out, psf_out, concat_type)

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
        if obs.instrument == "MIRI"
            psf_fwhm_out = mrs_psf.(@. λ_out * (1 + obs.z))
        elseif obs.instrument == "NIRSPEC"
            psf_fwhm_out = nirspec_psf.(@. λ_out * (1 + obs.z))
        end
    else
        if obs.instrument == "MIRI"
            psf_fwhm_out = mrs_psf.(λ_out)
        elseif obs.instrument == "NIRSPEC"
            psf_fwhm_out = nirspec_psf.(λ_out)
        end
    end
    if match_psf
        psf_fwhm_out = repeat([out_fwhm], length(λ_out))
    end

    # New LSF FWHM function with input in the rest frame
    if obs.rest_frame
        if obs.instrument == "MIRI"
            lsf_fwhm_out = mrs_lsf.(@. λ_out * (1 + obs.z))
        elseif obs.instrument == "NIRSPEC"
            lsf_fwhm_out = nirspec_lsf.(λ_out .* (1 .+ obs.z), obs.channels[channels[1]].band)
        end
    else
        if obs.instrument == "MIRI"
            lsf_fwhm_out = mrs_lsf.(λ_out)
        elseif obs.instrument == "NIRSPEC"
            lsf_fwhm_out = nirspec_lsf.(λ_out, obs.channels[channels[1]].band)
        end
    end

    if obs.masked
        @info "Masking bins with bad data..."
        I_out[mask_out] .*= NaN
        σ_out[mask_out] .*= NaN
        if do_psf_model
            psf_out[mask_out] .*= NaN
        end
    end

    # Do a final PSF renormalization
    if do_psf_model
        psf_out[psf_out .< 0] .= 0.
        for k ∈ axes(psf_out, 3)
            psf_out[:, :, k] ./= nansum(psf_out[:, :, k])
        end
    end

    # Get total channel number
    n_channels = 0
    for channel in channels
        n_channels += nchannels(obs.channels[channel].spectral_region)
    end
    # Get combined user masks
    if isnothing(user_mask)
        user_mask = Vector{Tuple{eltype(λ_out),eltype(λ_out)}}()
        for channel in channels
            append!(user_mask, umask(obs.channels[channel].spectral_region))
        end
    end
    # Get combined gaps
    if isnothing(gap_mask)
        gap_mask = Vector{Tuple{eltype(λ_out),eltype(λ_out)}}()
        for channel in channels
            append!(gap_mask, gaps(obs.channels[channel].spectral_region))
        end
    end

    # Construct a "proper" 3D WCS for the output cube
    wcs_out = shrink_wcs_dimensions(wcs_optimal_3d)
    wcs_out = extend_wcs_dimensions(wcs_out, λ_out)
    θ_sky_out = get_sky_rotation_angle(wcs_out)

    # Define the interpolated cube as the out_id channel 
    obs.channels[out_id] = DataCube(
        λ_out, I_out, σ_out, mask_out, psf_out, Ω_out, obs.α, obs.δ, 
        θ_sky_out, psf_fwhm_out, lsf_fwhm_out, wcs_out, "MULTIPLE", "MULTIPLE",
        user_mask, gap_mask, obs.rest_frame, obs.z, obs.masked, obs.vacuum_wave, 
        false, obs.dereddened, obs.sky_aligned,
        obs.channels[channels[output_wcs_frame]].voronoi_bins, format, 
        instrument_channel_edges
    )

    # redo log binning if necessary
    if obs.log_binned
        @info "Redoing logarithmic wavelength binning..."
        log_rebin!(obs)
    end

    @info "Done!"
end




"""
    combine_observations(observations; frames, refpoints, resolutions, projections, channels,
        order, instrument_channel_edges)

Given 2 or more observations of one object, combine them into a single mosaicked Observation object.
This requires that all input Observation objects contain the same observed channels.
This procedure will, for each channel that the Observations share:
    1) Reproject all of the observations of this channel onto a common 3D grid
    2) Combine the individual observations into a single output cube using a weighted average 
        (where the weights are 1/errors^2)

The output WCS, by default, is calculated to contain the full field of view of all observations, at the
highest resolution of any individual observation in the list.

# Arguments 
- `observations`: A list of the observation objects which should be combined
- `frames`: A list of coordinate frames (i.e. ICRSCoords, FK5Coords, ...) for the output WCS in each channel.
    Defaults to the coordinate frame of the first cube in each channel.
- `refpoints`: A list of reference points for the output WCS's CRVAL keywords (must be a list of SkyCoords).
    Defaults to the mean of the reference points of all the input cubes in each channel.
- `resolutions`: A list of resolutions for the output WCS's CDELT keywords (must be a list of quantities with angular units).
    Defaults to the smallest resolution of the cubes in each channel.
- `projections`: A list of projection types for the output WCS's CTYPE keywords. 
    Always defaults to "TAN" (tangent).
- `channels`: A list specifying which channels should be combined for the output observation.
    Defaults to all channels in the first input cube (which must also exist in all other cubes).
- `order`: The order of interpolation to be used during the reprojection. 
    Defaults to 1 (linear).
- `pad_mask`: If true, pads the edges of the IFU footprint by 1 extra pixel, to cut out any edge artifacts that can arise
    from the resampling and combining process. Default is true.
- `instrument_channel_edges`: If the input cubes are not from JWST MIRI or NIRSpec, one should provide a list of wavelengths
    which specify the boundaries of the channels of the given instrument.  If there is only one channel, or the channel distinction
    is not important to the user, one can simply input the minimum and maximum wavelength of the given cubes.
- `name_out`: A string specifying the name of the output observation object.
    Defaults to "[name of obs 1]_[name of obs 2]_...[name of obs N]_mosaicked"
"""
function combine_observations(observations::Vector{Observation}; frames=nothing,
    refpoints=nothing, resolutions=nothing, projections=nothing, channels=nothing,
    order::Integer=1, pad_mask::Bool=true, instrument_channel_edges::Union{Vector{<:QWave},Nothing}=nothing,
    name_out::Union{String,Nothing}=nothing)

    @debug "combine_observations: n_obs=$(length(observations)), order=$order, pad_mask=$pad_mask"
    # Input checking
    n_obs = length(observations)
    @assert n_obs ≥ 2 "Please input at least two Observations to combine!"
    if isnothing(channels)
        channels = keys(observations[1].channels)
    end
    for channel in channels
        @assert all([haskey(obs.channels, channel) for obs in observations]) "Please ensure all " *
            "input Observations have all of the requested channels to combine"
    end
    @assert all([obs.instrument .== observations[1].instrument for obs in observations]) "Observations \"instrument\" attribute do not match!"
    @assert all([obs.rest_frame .== observations[1].rest_frame for obs in observations]) "Observations \"rest_frame\" attribute do not match!"
    @assert all([obs.z .== observations[1].z for obs in observations]) "Observations \"z\" attribute do not match!"
    @assert all([obs.masked .== observations[1].masked for obs in observations]) "Observations \"masked\" attribute do not match!"
    @assert observations[1].masked "Please only input masked Observations to combine_observations! (use correct! or apply_mask!)"
    @assert all([obs.vacuum_wave .== observations[1].vacuum_wave for obs in observations]) "Observations \"vacuum_wave\" attribute do not match!"
    @assert all([obs.log_binned .== observations[1].log_binned for obs in observations]) "Observations \"log_binned\" attribute do not match!"
    @assert all([obs.dereddened .== observations[1].dereddened for obs in observations]) "Observations \"dereddened\" attribute do not match!"
    @assert all([obs.sky_aligned .== observations[1].sky_aligned for obs in observations]) "Observations \"sky_aligned\" attribute do not match!"

    if isnothing(instrument_channel_edges) && (observations[1].instrument != "MIRI") && (observations[1].instrument != "NIRSPEC")
        error("It appears that this isn't a MIRI/MRS cube. Please explicity specify the edges of each channel for this instrument.")
    end

    ra_all = []
    dec_all = []
    channels_out = Dict{Any,DataCube}()

    for (i, channel) in enumerate(channels)

        @info "Channel: $channel"
        cubes = [obs.channels[channel] for obs in observations]
        band = cubes[1].band
        gaps = cubes[1].spectral_region.gaps
        # more input checking
        for cube in cubes
            @assert cube.band == band "Please ensure all input Observations for channel $channel are in the same band!"
            @assert isnothing(cube.voronoi_bins) "Cannot combine voronoi binned cubes! (channel $channel)"
            @assert length(cube.spectral_region.mask) == 0 "Cannot combine cubes with user masks already applied -- " *
                "please apply them at the end (channel $channel)"
            @assert cube.spectral_region.gaps == gaps "Cannot combine cubes with different gaps in their spectra, all cubes in " *
                "channel $channel should have the same wavelength vector"
        end

        # Get optional arguments
        frame = isnothing(frames) ? nothing : frames[i]
        refpoint = isnothing(refpoints) ? nothing : refpoints[i]
        resolution = isnothing(resolutions) ? nothing : resolutions[i]
        projection = isnothing(projections) ? "TAN" : projections[i]

        @info "Reprojecting all channels onto the optimal WCS"
        reproject_cubes!(cubes; order=order, output_wcs_frame=0, enforce_all_cubes_out=false, frame=frame,
            refpoint=refpoint, resolution=resolution, projection=projection)

        # Prepare the values in each cube to be combined
        λ_out = cubes[1].λ
        psf1d_out = cubes[1].psf
        lsf1d_out = cubes[1].lsf
        I_all = zeros(eltype(cubes[1].I), (size(cubes[1].I)..., n_obs))
        σ_all = zeros(eltype(cubes[1].σ), (size(cubes[1].σ)..., n_obs))
        mask_all = zeros(eltype(cubes[1].mask), (size(cubes[1].mask)..., n_obs))
        psfs_all = nothing
        do_psf = all([!isnothing(cube.psf_model) for cube in cubes])
        if do_psf
            psfs_all = zeros(eltype(cubes[1].psf_model), (size(cubes[1].psf_model)..., n_obs))
        end
        for (j, cube) in enumerate(cubes)
            if !(λ_out == cube.λ) 
                @warn "Cubes for channel $channel have differing wavelength vectors! They will be resampled in the wavelength axis"
                I_all[:,:,:,j], σ_all[:,:,:,j], mask_all[:,:,:,j] = resample_flux_permuted3D(λ_out, cube.λ, cube.I, cube.σ, cube.mask)
                if do_psf
                    psfs_all[:,:,:,j] = resample_flux_permuted3D(λ_out, cube.λ, cube.psf_model)
                end
            else
                I_all[:,:,:,j] .= cube.I
                σ_all[:,:,:,j] .= cube.σ
                mask_all[:,:,:,j] .= cube.mask
                if do_psf
                    psfs_all[:,:,:,j] .= cube.psf_model
                end
            end
        end
        
        # Combine cubes, weighting by the errors, ignoring NaNs and 0s
        I_all[iszero.(I_all) .| mask_all] .*= NaN
        σ_all[iszero.(σ_all) .| mask_all] .*= NaN

        # Expand the masked edges by 1 pixel to prevent edge effects
        if pad_mask
            new_mask = zeros(eltype(mask_all), size(mask_all))
            for cube_i in axes(new_mask, 4)
                for wave_i in axes(new_mask, 3)
                    for xi in axes(new_mask, 1)
                        for yi in axes(new_mask, 2)
                            xs = max(1,xi-1):min(xi+1,size(new_mask,1))
                            ys = max(1,yi-1):min(yi+1,size(new_mask,2))
                            new_mask[xi, yi, wave_i, cube_i] = any(mask_all[xs, ys, wave_i, cube_i])
                        end
                    end
                end
            end
            mask_all = new_mask
        end

        weights = dropdims(nansum(ustrip.(1 ./ σ_all.^2), dims=4), dims=4)
        I_out = dropdims(nansum(ustrip.(I_all ./ σ_all.^2), dims=4), dims=4) ./ weights .* unit(I_all[1])
        σ_out = sqrt.(1 ./ weights) .* unit(σ_all[1])
        # Take the minimum of the masks -- so that at least one of the input cubes must be good
        mask_out = BitArray(dropdims(nanminimum(mask_all, dims=4), dims=4))
        mask_out .|= .~isfinite.(I_out) .| .~isfinite.(σ_out)
        psf_out = nothing
        # Combine the PSF models
        if do_psf
            psfs_all[iszero.(psfs_all) .| mask_all] .*= NaN
            psf_out = dropdims(nansum(ustrip.(psfs_all ./ σ_all.^2), dims=4), dims=4) ./ weights .* unit(psfs_all[1])
            mask_out .|= .~isfinite.(psf_out)
        end

        # Reapply masks
        I_out[mask_out] .*= NaN
        σ_out[mask_out] .*= NaN
        if do_psf
            psf_out[mask_out] .*= NaN
        end

        # Get a few ancillary parameters 
        wcs_out = cubes[1].wcs
        cunits = parse_cunits(wcs_out)
        pixel_scales = get_pixel_resolutions(wcs_out)
        Ω_out = uconvert(u"sr", prod((pixel_scales.*cunits)[1:2]))
        α_out, δ_out = uconvert.(u"°", wcs_out.crval[1:2].*cunits[1:2])
        θ_out = get_sky_rotation_angle(wcs_out)

        channels_out[channel] = DataCube(
            λ_out, I_out, σ_out, mask_out, psf_out, Ω_out, α_out, δ_out, θ_out, psf1d_out, lsf1d_out, wcs_out, string(channel),
            string(band), nothing, gaps, observations[1].rest_frame, observations[1].z, observations[1].masked,
            observations[1].vacuum_wave, observations[1].log_binned, observations[1].dereddened, observations[1].sky_aligned,
            nothing, Symbol(observations[1].instrument), instrument_channel_edges
        )
        push!(ra_all, α_out)
        push!(dec_all, δ_out)

    end

    if isnothing(name_out)
        name_out = ""
        for obs in observations
            name_out *= obs.name * "_"
        end
        name_out *= "_mosaicked"
    end

    Observation(
        channels_out, 
        name_out, 
        observations[1].z, 
        nanmean(ustrip.(uconvert.(u"°", ra_all)))*u"°",
        nanmean(ustrip.(uconvert.(u"°", dec_all)))*u"°",
        observations[1].instrument,
        observations[1].rest_frame,
        observations[1].masked,
        observations[1].vacuum_wave,
        observations[1].log_binned,
        observations[1].dereddened,
        observations[1].sky_aligned
    )
end

