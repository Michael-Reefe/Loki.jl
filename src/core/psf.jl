

"""
    generate_psf_model!(cube, psf_model_dir; spline_width, interpolate_leak_artifact)

Create a PSF model with for the DataCube `cube`. The model uses observations of standard stars that should be present
in the `psf_model_dir` directory, which are then averaged and shifted such that their centroids match the centroid of the data. They
may also be smoothed along the spectral axis using a cubic spline interpolation with a width of `spline_width` in pixels, and/or interpolated 
over the 12.22 um region to correct for the spectral leak artifact. The final PSF model outputs are normalized such that each wavelength slice 
integrates to 1.  The DataCube object is modified in-place with the PSF model, and the PSF model is also returned by the function.
"""
function generate_psf_model!(cube::DataCube, psf_model_dir::String=""; spline_width::Integer=7, interpolate_leak_artifact::Bool=true,
    z::Real=0.)

    # Load in data from observations of bright stars (HD 163466, HD 37962, 16 Cyg B)
    hdus = []
    if psf_model_dir == ""
        psf_model_dir = joinpath(@__DIR__, "..", "templates", "psf")
    end
    objects = glob("*", psf_model_dir)
    bands = Dict("SHORT" => 'A', "MEDIUM" => 'B', "LONG" => 'C')
    for object ∈ objects
        f = FITS(joinpath(object, "Level3_ch$(cube.channel)-$(lowercase(cube.band))_s3d.fits"))
        push!(hdus, f)
    end

    # Make sure the input cube is aligned to the IFU axes and not the sky axes
    chband = Symbol("$(bands[cube.band])$(cube.channel)")
    @assert !cube.sky_aligned "Must input an IFU-aligned datacube, not a sky-aligned one! (The cubes can be rotated later)"

    @info "Combining PSFs from standard bright star observations..."
    psfs = []
    for hdu ∈ hdus

        data = read(hdu["SCI"])
        data[.~isfinite.(data)] .= 0.
        objname = read_header(hdu[1])["TARGNAME"]
        @debug "Reading in the PSF model for $objname"

        # Shift the centroid to match the observation data
        @debug "Shifting the centroids for $objname to match the observations"
        data_ref2d = dropdims(nansum(cube.I, dims=3), dims=3)
        _, mx = findmax(data_ref2d)
        c1 = centroid_com(data_ref2d[mx[1]-5:mx[1]+5, mx[2]-5:mx[2]+5]) .+ (mx.I .- 5) .- 1
        data2d = dropdims(nansum(data, dims=3), dims=3)
        _, mx2 = findmax(data2d)
        c2 = centroid_com(data2d[mx2[1]-5:mx2[1]+5, mx2[2]-5:mx2[2]+5]) .+ (mx2.I .- 5) .- 1
        dx = c1 .- c2
        
        data_shift = zeros(eltype(data), size(data))
        for i in axes(data_shift, 3)
            data_shift[:, :, i] = fshift(data[:, :, i], dx...)
        end

        # Mask to match the size of the observation data
        size_psf = size(data_shift)[1:2]
        size_obs = size(cube.I)[1:2]
        if size_psf != size_obs
            @debug "Sizes of PSF and Observation data do not match: $size_psf and $size_obs. Masking the edges of the PSF data."
            data_shift = data_shift[1:size_obs[1], 1:size_obs[2], :]
        end

        # Cut down to match the size of the reference data
        push!(psfs, data_shift)
    end

    # Take an average of each individual PSF model
    psf = dropdims(nansum(cat(psfs..., dims=4), dims=4), dims=4)

    # Interpolate over the 12.22 μm region to account for the JWST spectral leak artifact,
    # which is very prominent for blue sources such as these stars
    if interpolate_leak_artifact && (chband == :A3)
        @info "Interpolating over the 12.22um spectral leak artifact"

        # Get the OBSERVED frame wavelengths
        λ = copy(cube.λ)
        if cube.rest_frame
            if iszero(z)
                error("The input cube is already in the rest frame. Please input the redshift using the keyword argument `z` so that the " * 
                      "observed-frame wavelengths can be recalculated, or input a cube that is in the observed frame.")
            end
            λ .*= (1 .+ z)
        end

        for c ∈ CartesianIndices(size(psf)[1:2])

            # Get indices to the left/right of the 12.22 μm leak
            lind = argmin(abs.(λ .- 11.93))
            rind = argmin(abs.(λ .- 12.39))
            # Create widely spaced knots for the interpolation
            Δλ = 0.1
            knots1 = λ[2]:Δλ:λ[lind]
            knots2 = λ[rind]:Δλ:λ[end-1]
            knots = [knots1; knots2]
            # Only use the data to the left/right of the leak
            λinterp = [λ[1:lind]; λ[rind:length(λ)]]
            psfinterp = [psf[c, 1:lind]; psf[c, rind:length(λ)]]
            interp = Spline1D(λinterp, psfinterp, knots, k=3, bc="extrapolate")

            # Fill in the data in the leak region with the interpolation
            psf[c, lind:rind] .= interp(λ[lind:rind])
        end
    end

    # Optionally do some smoothing along the wavelength axis 
    if spline_width > 0
        @info "Performing cubic spline interpolation of the PSF models"
        λ = cube.λ
        λknots = λ[1+spline_width:spline_width:end-spline_width]
        for c ∈ CartesianIndices(size(psf)[1:2])
            psf[c, :] .= Spline1D(λ, psf[c, :], λknots, k=3, bc="extrapolate")(λ)
        end
    end

    # Normalize such that the PSF integrates to 1 at each wavelength slice
    for i ∈ axes(psf, 3)
        psf[:, :, i] ./= nansum(psf[:, :, i])
    end

    # set negative values to 0
    psf[psf .< 0.] .= 0.

    # Save the PSF model to the DataCube object
    cube.psf_model = psf

    psf
end


# A method that applies the `generate_psf_model!` function to each channel in an Observation object
function generate_psf_model!(obs::Observation, psf_model_dir::String=""; spline_width::Integer=7, interpolate_leak_artifact::Bool=true)
    for ch in keys(obs.channels)
        generate_psf_model!(obs.channels[ch], psf_model_dir; spline_width=spline_width, interpolate_leak_artifact=interpolate_leak_artifact)
    end
end


"""
    generate_nuclear_template(cube, ap_r, spline_width)

Extract a point-source nuclear template from the brightest region of a DataCube with a given aperture `ap_r`, 
where `ap_r` is in units of the PSF FWHM (thus, the aperture size increases with wavelength at the same rate 
that the PSF FWHM does). If `ap_r` is 0, then the template is just extracted from the brightest spaxel. If 
`spline_width` is > 0, a cubic spline interpolation is performed with knots spaced by `spline_width` pixels.
The 1D template is then combined with a 3D PSF model to create a full 3D nuclear template.
"""
function generate_nuclear_template(cube::DataCube, ap_r::Real=0., spline_width::Integer=7)

    # Get the brightest spaxel
    data2d = dropdims(nansum(cube.I, dims=3), dims=3)
    _, mx = findmax(data2d)

    # Get the centroid
    c = centroid_com(data2d[mx[1]-5:mx[1]+5, mx[2]-5:mx[2]+5]) .+ (mx.I .- 5) .- 1

    if iszero(ap_r)
        # Just take the brightest spaxel
        nuc1d = cube.I[mx, :]
    else
        # Arcseconds per pixel
        pixel_scale = sqrt(cube.Ω) * 180/π * 3600
        # Prepare outputs
        nuc1d = zeros(eltype(cube.I), length(cube.λ))
        # Do aperture photometry
        for i ∈ eachindex(nuc)
            # Convert ap_r into pixels
            ap_size = ap_r * cube.psf[i] / pixel_scale
            ap = CircularAperture(mx.I..., ap_size)
            nuc1d[i] = photometry(ap, cube.I[:, :, i]).aperture_sum / get_area(ap)
        end
    end

    # Do a cubic spline fit to get a good S/N
    λknots = cube.λ[1+spline_width:spline_width:end-spline_width]
    nuc1d = Spline1D(cube.λ, nuc1d, λknots, k=3, bc="extrapolate")(cube.λ)

    # Normalize the PSF models and combine them
    psf_model = copy(cube.psf_model)
    for i ∈ axes(psf_model, 3)
        psf_model[:, :, i] ./= nanmaximum(psf_model[:, :, i])
    end

    nuc = [nuc1d[k] * psf_model[i,j,k] for i ∈ axes(psf_model, 1), j ∈ axes(psf_model, 2), k ∈ axes(psf_model, 3)]
end



