

"""
    get_area(ap)

Get the area in pixels that a Photometry.jl Aperture or Annulus object covers.
"""
get_area(ap::CircularAperture) = π * ap.r^2
get_area(ap::EllipticalAperture) = π * ap.a * ap.b
get_area(ap::RectangularAperture) = ap.w * ap.h
get_area(ap::CircularAnnulus) = π * (ap.r_out^2 - ap.r_in^2)
get_area(ap::EllipticalAnnulus) = π * (ap.a_out * ap.b_out - ap.a_in * ap.b_in)
get_area(ap::RectangularAnnulus) = ap.w_out * ap.h_out - ap.w_in * ap.h_in


"""
    get_patches(aperture)

Get matplotlib patches for a given Photometry.jl Aperture or Annulus object.
"""
get_patches(aperture::CircularAperture) = [plt.Circle((aperture.x-1, aperture.y-1), aperture.r, color="k", lw=2, fill=false)]
get_patches(aperture::EllipticalAperture) = [plt.Ellipse((aperture.x-1, aperture.y-1), 2aperture.a, 2aperture.b, angle=aperture.theta, color="k", lw=2, fill=false)]
# Rectangle needs an extra -0.5 since the aperture is defined by the center but in matplotlib its defined as the bottom-left corner
get_patches(aperture::RectangularAperture) = [plt.Rectangle((aperture.x-1.5, aperture.y-1.5), aperture.w, aperture.h, angle=aperture.theta, color="k", lw=2, fill=false)]
get_patches(aperture::CircularAnnulus) = [plt.Circle((aperture.x-1, aperture.y-1), aperture.r_in, color="k", lw=2, fill=false),
                                          plt.Circle((aperture.x-1, aperture.y-1), aperture.r_out, color="k", lw=2, fill=false)]
get_patches(aperture::EllipticalAnnulus) = [plt.Ellipse((aperture.x-1, aperture.y-1), 2aperture.a_in, 2aperture.b_in, angle=aperture.theta, color="k", lw=2, fill=false),
                                            plt.Ellipse((aperture.x-1, aperture.y-1), 2aperture.a_out, 2aperture.b_out, angle=aperture.theta, color="k", lw=2, fill=false)]
get_patches(aperture::RectangularAnnulus) = [plt.Rectangle((aperture.x-1.5, aperture.y-1.5), aperture.w_in, aperture.h_in, angle=aperture.theta, color="k", lw=2, fill=false),
                                             plt.Rectangle((aperture.x-1.5, aperture.y-1.5), aperture.w_out, aperture.h_out, angle=aperture.theta, color="k", lw=2, fill=false)]

"""
    centroid_com(data[, mask])

Get the center of mass coordinates of a given n-dimensional array `data`.
"""
centroid_com(data) = centroid_com(data, falses(size(data)))

function centroid_com(data::AbstractArray, mask::BitArray)
    # Mask out any non-finite values
    _data = copy(data)
    _data[.~isfinite.(data) .| mask] .= 0.

    total = sum(_data)
    out_cent = zeros(ndims(_data))
    for index ∈ CartesianIndices(_data)
        out_cent .+= Tuple(index) .* _data[index]
    end
    out_cent ./ total
end


"""
    make_aperture(cube, type, ra, dec, params...; [auto_centroid, scale_psf, box_size])

Create an aperture using the Photometry Library.

# Arguments
- `cube::DataCube`: The DataCube struct to create the aperture for
- `type::Symbol`: Must be one of :Circular, :Elliptical, or :Rectangular
- `ra::Union{String,Real}`: Right ascension. If string, should be in sexagesimal hours, if real, should be decimal degrees.
- `dec::Union{String,Real}`: Declination. If string, should be in sexagesimal degrees, if real, should be decimal degrees.
- `params...`: Varying number of parameters for the aperture depending on `type`.
    For circular apertures, the only parameter is the radius in arcseconds.
    For elliptical apertures, the parameters are the semimajor and semiminor axes in arcseconds, and the position angle in degrees
    For rectangular apertures, the parameters are the width and height in arcseconds, and the position angle in degrees
- `auto_centroid::Bool=false`: if true, adjusts the center (ra,dec) to the closest peak in brightness 
- `scale_psf::Bool=false`: if true, creates a vector of apertures that scale up in radius at the same rate that the PSF scales up
- `box_size::Integer=11`: if `auto_centroid` is true, this gives the box size to search for a local peak in brightness, in pixels
"""
function make_aperture(cube::DataCube, ap_type::Symbol, ra::Union{String,Real}, dec::Union{String,Real}, 
    params...; auto_centroid=false, scale_psf::Bool=false, box_size::Integer=11)

    if ap_type == :Circular
        @assert length(params) == 1 "CircularAperture takes 1 parameter (radius in pix)"
    elseif ap_type == :Rectangular
        @assert length(params) == 3 "RectangularAperture takes 3 parameters (width in pix, height in pix, position angle in deg.)"
    elseif ap_type == :Elliptical
        @assert length(params) == 3 "EllipticalAperture takes 3 parameters (semimajor axis in pix, semiminor axis in pix, position angle in deg.)"
    end
    @info "Creating a circular aperture at $ra, $dec"

    # Get the WCS frame from the datacube
    frame = lowercase(cube.wcs.radesys)
    if frame == "icrs"
        coords = ICRSCoords
    elseif frame == "fk5"
        coords = FK5Coords
    end

    # Dictionary mapping symbols to aperture types
    ap_class = Dict(:Circular => CircularAperture, :Elliptical => EllipticalAperture, :Rectangular => RectangularAperture)

    # If given as strings, assume ra/dec units are parsable with AstroAngles, otherwise assume decimal degrees
    ra_deg = ra isa String ? ra |> parse_hms |> hms2deg : ra
    dec_deg = dec isa String ? dec |> parse_dms |> dms2deg : dec
    # Convert to radians
    ra_rad = ra_deg |> deg2rad
    dec_rad = dec_deg |> deg2rad
    # Create SkyCoords object
    sky_center = coords(ra_rad, dec_rad)
    
    # Convert the sky position to a pixel position
    x_cent, y_cent = world_to_pix(cube.wcs, [ra_deg, dec_deg, 1.0])[1:2]

    # Take a point directly north by 1 arcsecond and convert it to pixel coordinates to get the pixel scale
    sky_offset = offset(sky_center, 1/3600*π/180, 0)
    x_offset, y_offset = world_to_pix(cube.wcs, [sky_offset.ra |> rad2deg, sky_offset.dec |> rad2deg, 1.0])[1:2]
    dx = x_offset - x_cent
    dy = y_offset - y_cent
    pixscale = hypot(dx, dy)         # scale in pixels per arcsecond
    angle = atan(dy, dx) |> rad2deg  # rotation angle in degrees

    # Convert parameters to pixel units
    pix_params = [params...]
    if length(params) == 1
        pix_params[1] *= pixscale
    elseif length(params) == 3
        pix_params[1:2] .*= pixscale
        pix_params[3] += angle
    end

    # If auto_centroid is true, readjust the center of the aperture to the peak in the local brightness
    if auto_centroid
        data = sumdim(cube.I, 3)
        mask = trues(size(data))
        p0 = round.(Int, [x_cent, y_cent])
        box_half = fld(box_size, 2)
        mask[p0[1]-box_half:p0[1]+box_half, p0[2]-box_half:p0[2]+box_half] .= 0

        # Find the centroid using a center-of-mass estimation
        x_cent, y_cent = centroid_com(data, mask)
        ra_cent, dec_cent = pix_to_world(cube.wcs, [x_cent, y_cent, 1.0])[1:2]
        @info "Aperture centroid adjusted to $(format_angle(ha2hms(ra_cent/15); delim=["h","m","s"])), " *
            "$(format_angle(deg2dms(dec_cent); delim=["d","m","s"]))"
    end

    # Make the pixel aperture
    pix_ap = ap_class[ap_type]((x_cent, y_cent), pix_params...)

    if scale_psf
        # Scale the aperture size based on the change in the PSF size
        pix_aps = Vector{typeof(pix_ap)}(undef, length(cube.λ))
        for i ∈ eachindex(pix_aps)
            if ap_type == :Circular
                pix_aps[i] = CircularAperture((x_cent, y_cent), pix_params[1] * cube.psf[i] / cube.psf[1])
            elseif ap_type == :Elliptical
                pix_aps[i] = EllipticalAperture((x_cent, y_cent), (pix_params[1:2] .* cube.psf[i] ./ cube.psf[1])..., pix_params[3])
            elseif ap_type == :Rectangular
                pix_aps[i] = RectangularAperture((x_cent, y_cent), (pix_params[1:2] .* cube.psf[i] ./ cube.psf[1])..., pix_params[3])
            end
        end
        return pix_aps
    end

    pix_ap
end
