

"""
    generate_psf_model!(cube, psf_model_dir; interpolate_leak_artifact)

Create a PSF model with for the DataCube `cube`. The model uses observations of standard stars that should be present
in the `psf_model_dir` directory, which are then averaged and shifted such that their centroids match the centroid of the data. They
may also be interpolated over the 12.22 um region to correct for the spectral leak artifact. The final PSF model outputs are normalized such that each wavelength slice 
integrates to 1.  The DataCube object is modified in-place with the PSF model, and the PSF model is also returned by the function.
"""
function generate_psf_model!(cube::DataCube, instrument::String, psf_model_dir::String=""; interpolate_leak_artifact::Bool=true, z::Real=0.)

    # Load in data from observations of bright stars (HD 163466, HD 37962, 16 Cyg B)
    hdus = []
    if (instrument == "MIRI") && (psf_model_dir == "")
        psf_model_dir = joinpath(@__DIR__, "..", "templates", "psfs_stars")
    elseif (instrument == "NIRSPEC") && (psf_model_dir == "")
        psf_model_dir = joinpath(@__DIR__, "..", "templates", "psfs_stars_nirspec")
    else 
        error("generate_psf_model! is only supported for MIRI/MRS or NIRSPEC/IFU observations!")
    end
    files = glob(joinpath("**", "*.fits"), psf_model_dir)
    for file ∈ sort(files)
        m = match(Regex("(.+?)($(cube.channel)[-_.]?$(cube.band))(.+?).fits?", "i"), file)
        if !isnothing(m)
            push!(hdus, FITS(file))
        end
    end

    # Make sure the input cube is aligned to the IFU axes and not the sky axes
    @assert !cube.sky_aligned "Must input an IFU-aligned datacube, not a sky-aligned one! (The cubes can be rotated later)"

    @info "Loading $(length(hdus)) PSF models from $psf_model_dir..."
    psfs = []
    for hdu ∈ hdus

        data = read(hdu["SCI"])
        data[.~isfinite.(data)] .= 0.
        objname = read_header(hdu[1])["TARGNAME"]
        @debug "Reading in the PSF model for $objname"

        # Mask to match the size of the observation data
        size_psf = size(data)[1:2]
        size_obs = size(cube.I)[1:2]
        if size_psf != size_obs
            @debug "Sizes of PSF and Observation data do not match: $size_psf and $size_obs. Masking the edges of the PSF data."
            while size_obs[1] > size_psf[1]
                data_slice = ones(1, size(data)[2:3]...) .* NaN
                data = cat(data_slice, data, data_slice, dims=1)
                size_psf = size(data)[1:2]
            end
            while size_obs[2] > size_psf[2]
                data_slice = ones(size(data, 1), 1, size(data, 3)) .* NaN
                data = cat(data_slice, data, data_slice, dims=2)
                size_psf = size(data)[1:2]
            end
            while size_obs[1] < size_psf[1]
                if size_psf[1] - size_obs[1] >= 2
                    data = data[2:end-1, :, :]
                else
                    data = data[1:end-1, :, :]
                end
                size_psf = size(data)[1:2]
            end
            while size_obs[2] < size_psf[2]
                if size_psf[2] - size_obs[2] >= 2
                    data = data[:, 2:end-1, :]
                else
                    data = data[:, 1:end-1, :]
                end
                size_psf = size(data)[1:2]
            end
            @assert size(data)[1:2] == size(cube.I)[1:2]
        end

        # Shift the centroid to match the observation data
        @debug "Shifting the centroids for $objname to match the observations"
        data_ref2d = dropdims(nansum(ustrip(cube.I), dims=3), dims=3)
        _, mx = findmax(data_ref2d)
        c1 = centroid_com(data_ref2d[mx[1]-5:mx[1]+5, mx[2]-5:mx[2]+5]) .+ (mx.I .- 5) .- 1

        data2d = dropdims(nansum(data, dims=3), dims=3)
        data2d[data2d .< 0] .= 0.
        _, mx2 = findmax(data2d)
        c2 = centroid_com(data2d[mx2[1]-5:mx2[1]+5, mx2[2]-5:mx2[2]+5]) .+ (mx2.I .- 5) .- 1

        dsmooth = copy(data)
        for c in CartesianIndices(size(data)[1:2])
            dsmooth[c, :] .= movmean(data[c, :], 7)
        end

        # Subtract the background within an annulus
        pix_size = uconvert(u"arcsecond", sqrt(cube.Ω))
        for i in axes(data, 3)
            r_in = 5 * cube.psf[i] / pix_size
            r_out = 10 * cube.psf[i] / pix_size
            back_ann = CircularAnnulus(c2..., r_in, r_out)
            pedestal = photometry(back_ann, dsmooth[:,:,i]).aperture_sum / photometry(back_ann, ones(size(dsmooth)[1:2])).aperture_sum
            if i == 1
                # println("$(cube.channel) $(cube.band)")
                # println(pedestal)
                # fig, ax = plt.subplots(1,2)
                # d = copy(data[:, :, i])
                # d[d .< 0] .= 0
                # d2 = data[:, :, i] .- pedestal
                # d2[d2 .< 0] .= 0
                # ax[1].imshow(log10.(d)', origin=:lower, cmap=:cubehelix)
                # ax[2].imshow(log10.(d2)', origin=:lower, cmap=:cubehelix)
                # patches = get_patches(back_ann)
                # for patch in patches
                #     ax[1].add_patch(patch)
                # end
                # plt.show()
                # plt.close()
            end
            data[:, :, i] .-= pedestal
        end

        # Recalculate shift now that pedestal has been subtracted
        data2d = dropdims(nansum(data, dims=3), dims=3)
        data2d[data2d .< 0] .= 0.
        _, mx2 = findmax(data2d)
        c2 = centroid_com(data2d[mx2[1]-5:mx2[1]+5, mx2[2]-5:mx2[2]+5]) .+ (mx2.I .- 5) .- 1
        dx = c1 .- c2
        
        data_shift = zeros(eltype(data), size(data))
        for i in axes(data_shift, 3)
            data_shift[:, :, i] = fshift(data[:, :, i], dx...)
        end

        push!(psfs, data_shift)
    end

    # Take an average of each individual PSF model
    psf = dropdims(nanmean(cat(psfs..., dims=4), dims=4), dims=4)

    # Interpolate over the 12.22 μm region to account for the JWST spectral leak artifact,
    # which is very prominent for blue sources such as these stars
    if instrument == "MIRI"
        bands = Dict("SHORT" => 'A', "MEDIUM" => 'B', "LONG" => 'C')
        chband = Symbol("$(bands[cube.band])$(cube.channel)")
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
                lind = argmin(abs.(λ .- 11.93u"μm"))
                rind = argmin(abs.(λ .- 12.39u"μm"))
                # Create widely spaced knots for the interpolation
                Δλ = 0.1u"μm"
                knots1 = λ[2]:Δλ:λ[lind]
                knots2 = λ[rind]:Δλ:λ[end-1]
                knots = [knots1; knots2]
                # Only use the data to the left/right of the leak
                λinterp = [λ[1:lind]; λ[rind:length(λ)]]
                psfinterp = [psf[c, 1:lind]; psf[c, rind:length(λ)]]
                interp = Spline1D(ustrip(λinterp), psfinterp, ustrip(knots), k=3, bc="extrapolate")

                # Fill in the data in the leak region with the interpolation
                psf[c, lind:rind] .= interp(ustrip(λ[lind:rind]))
            end
        end
    end

    psf[psf .< 0] .= 0.
    # Normalize such that the PSF integrates to 1 at each wavelength slice
    for i ∈ axes(psf, 3)
        psf[:, :, i] ./= nansum(psf[:, :, i])
    end

    # Save the PSF model to the DataCube object
    cube.psf_model = psf

    psf
end


# A method that applies the `generate_psf_model!` function to each channel in an Observation object
function generate_psf_model!(obs::Observation, psf_model_dir::String=""; interpolate_leak_artifact::Bool=true,
    centroid_4c_4b::Bool=false)

    for ch in keys(obs.channels)
        generate_psf_model!(obs.channels[ch], obs.instrument, psf_model_dir; interpolate_leak_artifact=interpolate_leak_artifact, z=obs.z)
    end

    # if centroid_4c_4b is true: do not trust the channel 4C centroiding
    if :C4 in keys(obs.channels) && :B4 in keys(obs.channels) && centroid_4c_4b

        data2d = dropdims(nansum(ustrip(obs.channels[:B4].I), dims=3), dims=3)
        _, mx = findmax(data2d)
        c1 = centroid_com(data2d[mx[1]-5:mx[1]+5, mx[2]-5:mx[2]+5]) .+ (mx.I .- 5) .- 1

        psf = obs.channels[:C4].psf_model
        psf2d = dropdims(nansum(psf, dims=3), dims=3)
        _, mx = findmax(psf2d)
        c2 = centroid_com(psf2d[mx[1]-5:mx[1]+5, mx[2]-5:mx[2]+5]) .+ (mx.I .- 5) .- 1

        dx = c1 .- c2
        for i in axes(psf, 3)
            psf[:, :, i] .= fshift(psf[:, :, i], dx...)
        end
    end
end


"""
    splinefit_psf_model!(cube, spline_width)

Fit a PSF model with a cubic spline with knots spaced by `spline_width` pixels.
"""
function splinefit_psf_model!(cube::DataCube, spline_width::Integer)
    @assert !isnothing(cube.psf_model) "Please input a DataCube that already has a PSF model!"

    @info "Fitting PSF with a cubic spline with knots spaced by $spline_width pixels"
    for c ∈ CartesianIndices(size(cube.psf_model)[1:2])
        filt = isfinite.(cube.psf_model[c, :])
        if sum(filt) < 2spline_width+3
            continue
        end
        λknots = cube.λ[filt][1+spline_width:spline_width:end-spline_width]
        cube.psf_model[c, :] .= Spline1D(ustrip(cube.λ[filt]), cube.psf_model[c, filt], ustrip(λknots), 
            k=3, bc="extrapolate")(ustrip(cube.λ))
    end

    # Renormalize
    cube.psf_model[cube.psf_model .< 0] .= 0.
    for k ∈ axes(cube.psf_model, 3)
        cube.psf_model[:, :, k] ./= nansum(cube.psf_model[:, :, k])
    end
end


# Method that applies `splinefit_psf_model!` to each channel in an Observation object
function splinefit_psf_model!(obs::Observation, spline_width::Integer)
    for ch in keys(obs.channels)
        splinefit_psf_model!(obs.channels[ch], spline_width)
    end
end

# function shift_psf_model!(cube::DataCube, z::Real)

#     for i in 1:(length(channel_boundaries)+1)
        
#         left = i > 1 ? channel_boundaries[i-1] : 0.
#         right = i ≤ length(channel_boundaries) ? channel_boundaries[i] : Inf

#         λ = cube.λ
#         if cube.rest_frame
#             λ = cube.λ .* (1 .+ z)
#         end
#         region = left .< λ .< right
#         # Skip channels where there is no data
#         if sum(region) == 0
#             continue
#         end

#         fch = dropdims(nansum(cube.I[:, :, region], dims=3), dims=3)
#         fch[.~isfinite.(fch)] .= 0.
#         _, mx = findmax(fch)
#         c1 = centroid_com(fch[mx[1]-10:mx[1]+10, mx[2]-10:mx[2]+10]) .+ (mx.I .- 10) .- 1
#         println(c1)

#         for ri in findall(region)
#             psf = cube.psf_model[:, :, ri]
#             m = .~isfinite.(psf)
#             psf[m] .= 0.
#             _, mx = findmax(psf)
#             c2 = centroid_com(psf[mx[1]-10:mx[1]+10, mx[2]-10:mx[2]+10]) .+ (mx.I .- 10) .- 1
#             shift = c1 .- c2
#             cube.psf_model[:, :, ri] .= fshift(psf, shift...)
#             cube.psf_model[m, ri] .= NaN
#         end
#     end

#     cube
# end



"""
    generate_nuclear_template(cube, ap_r, spline_width)

Extract a point-source nuclear template from the brightest region of a DataCube with a given aperture `ap_r`, 
where `ap_r` is in units of the PSF FWHM (thus, the aperture size increases with wavelength at the same rate 
that the PSF FWHM does). If `ap_r` is 0, then the template is just extracted from the brightest spaxel. If 
`spline_width` is > 0, a cubic spline interpolation is performed with knots spaced by `spline_width` pixels.
The 1D template is then combined with a 3D PSF model to create a full 3D nuclear template.
"""
function generate_nuclear_template(cube::DataCube, ap_r::Real=0.; spline_width::Integer=7) 

    Iunit = unit(cube.I[1])
    data2d = dropdims(nansum(ustrip(cube.I), dims=3), dims=3)
    data2d[.~isfinite.(data2d)] .= 0.
    _, mx = findmax(data2d)

    if iszero(ap_r)
        # Just take the brightest spaxel
        nuc1d = cube.I[mx, :] ./ cube.psf_model[mx, :]
    else
        # Arcseconds per pixel
        pixel_scale = uconvert(u"arcsecond", sqrt(cube.Ω))
        # Prepare outputs
        nuc1d = zeros(eltype(cube.I), length(cube.λ))
        # Do aperture photometry
        for i ∈ eachindex(nuc1d)
            # Convert ap_r into pixels
            ap_size = ap_r * cube.psf[i] / pixel_scale
            ap = CircularAperture(mx.I..., ap_size)
            nuc1d[i] = photometry(ap, cube.I[:, :, i]).aperture_sum / get_area(ap)
            nuc1d[i] /= cube.psf_model[:, :, i][mx]
        end
    end

    # Do a cubic spline fit to get a good S/N and fill in holes
    if spline_width > 0
        good = isfinite.(nuc1d) .& (ustrip(abs.(nuc1d)) .< 1e10)
        λknots = cube.λ[good][1+spline_width:spline_width:end-spline_width]
        nuc1d = Spline1D(ustrip(cube.λ[good]), ustrip(nuc1d[good]), ustrip(λknots), k=3, bc="extrapolate")(ustrip(cube.λ)) .* Iunit
    end

    [nuc1d[k] * cube.psf_model[i,j,k] for i ∈ axes(cube.psf_model, 1), j ∈ axes(cube.psf_model, 2), k ∈ axes(cube.psf_model, 3)]
end
