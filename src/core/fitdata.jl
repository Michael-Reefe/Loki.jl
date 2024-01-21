

"""
    mask_emission_lines(λ, I; [Δ, n_inc_thresh, thresh])

Mask out emission lines in a given spectrum using a numerical second derivative and flagging 
negative(positive) spikes, indicating strong concave-downness(upness) up to some tolerance threshold (i.e. 3-sigma).
The widths of the lines are estimated using the number of pixels for which the numerical first derivative
is above some (potentially different) tolerance threshold. Returns the mask as a BitVector.

# Arguments
- `λ::Vector{<:Real}`: The wavelength vector of the spectrum
- `I::Vector{<:Real}`: The intensity vector of the spectrum
- `spectral_region::Symbol`: Either :MIR for mid-infrared or :OPT for optical
- `Δ::Union{Integer,Nothing}=nothing`: The resolution (width) of the numerical derivative calculations, in pixels
- `n_inc_thresh::Union{Integer,Nothing}=nothing`
- `thresh::Real=3.`: The sensitivity threshold for flagging lines with the second derivative test, in units of the RMS.

See also [`continuum_cubic_spline`](@ref)
"""
function mask_emission_lines(λ::Vector{<:Real}, I::Vector{<:Real}, Δ::Integer, n_inc_thresh::Integer, thresh::Real,
    overrides::Vector{Tuple{T,T}}=Vector{Tuple{Real,Real}}()) where {T<:Real}


    mask = falses(length(λ))
    # manual regions that wish to be masked out
    if length(overrides) > 0
        for override in overrides
            mask[override[1] .< λ .< override[2]] .= 1
        end
        return mask
    end

    # Wavelength difference vector
    diffs = diff(λ)
    diffs = [diffs; diffs[end]]

    # Calculate the numerical second derivative
    d2f = zeros(length(λ))
    @simd for i ∈ 1:length(λ)
        d2f[i] = (I[min(length(λ), i+Δ)] - 2I[i] + I[max(1, i-Δ)]) / (Δ * diffs[i])^2
    end
    W = (10 * Δ, 1000)

    # Find where the second derivative is significantly concave-down 
    for j ∈ 7:(length(λ)-7)
        # Only consider the spectrum within +/- W pixels from the point in question
        if any([abs(d2f[j]) > thresh * nanstd(d2f[max(1, j-Wi):min(length(λ), j+Wi)]) for Wi ∈ W])

            n_pix_l = 0
            n_pix_r = 0

            # Travel left/right from the center until the flux goes up 3 times
            n_inc = 0
            while n_inc < n_inc_thresh
                dI = I[j-n_pix_l-1] - I[j-n_pix_l]
                if dI > 0
                    n_inc += 1
                end
                if j-n_pix_l-1 == 1
                    break
                end
                n_pix_l += 1
            end
            n_inc = 0
            while n_inc < n_inc_thresh
                dI = I[j+n_pix_r+1] - I[j+n_pix_r]
                if dI > 0
                    n_inc += 1
                end
                if j+n_pix_r+1 == length(I)
                    break
                end
                n_pix_r += 1
            end

            mask[max(j-n_pix_l,1):min(j+n_pix_r,length(mask))] .= 1

        end
    end

    # Don't mask out these regions that tends to trick this method sometimes
    mask[11.10 .< λ .< 11.15] .= 0
    mask[11.17 .< λ .< 11.24] .= 0
    mask[11.26 .< λ .< 11.355] .= 0
    # mask[12.5 .< λ .< 12.79] .= 0

    # Force the beginning/end few pixels to be unmasked to prevent runaway splines at the edges
    mask[1:7] .= 0
    mask[end-7:end] .= 0

    # Return the line locations and the mask
    mask

end


"""
    get_chi2_mask(cube_fitter, λ, mask_bad, mask_lines)

Creates a chi2 mask to be used when calculating chi2 values.
Uses the bad pixel mask, plus the line mask (but only at locations where
emission lines are not expected to occur based on the model, hopefully
so that it only picks up noise spikes or other calibration errors).
"""
function get_chi2_mask(cube_fitter::CubeFitter, λ::Vector{<:Real}, mask_bad::BitVector, mask_lines::BitVector)
    line_reference = falses(length(λ))
    for i ∈ 1:cube_fitter.n_lines
        max_voff = maximum(abs, cube_fitter.lines.voff[i, 1].limits)
        max_fwhm = maximum(cube_fitter.lines.fwhm[i, 1].limits)
        center = cube_fitter.lines.λ₀[i]
        region = center .* (1 - (max_voff+2max_fwhm)/C_KMS, 1 + (max_voff+2max_fwhm)/C_KMS)
        line_reference[region[1] .< λ .< region[2]] .= 1
    end
    # If any fall outside of this region, do not include these pixels in the chi^2 calculations
    mask_chi2 = mask_bad .| (mask_lines .& .~line_reference)
end


"""
    get_normalized_vectors(λ, I, σ, thing1, thing2, N)

Normalizes I and σ and returns copies of λ, templates, and channel_masks so that the originals are not modified.
"""
function get_normalized_vectors(λ::Vector{<:Real}, I::Vector{<:Real}, σ::Vector{<:Real}, thing1, thing2, N::Real)

    λ_spax = copy(λ)
    I_spax = I ./ N
    σ_spax = σ ./ N
    thing1 = !isnothing(thing1) ? copy(thing1) : nothing
    thing2 = !isnothing(thing2) ? copy(thing2) : nothing

    λ_spax, I_spax, σ_spax, thing1, thing2
end


"""
    get_normalized_vectors(λ, I, σ, ext_curve, template_psfnuc, N, continuum, nuc_temp_fit)

Normalizes I and σ and returns copies of λ, ext_curve, and template_psfnuc so that the originals are not modified.
Modified version for emission line fitting also subtracts the (already normalized) continuum fit from I.
"""
function get_normalized_vectors(λ::Vector{<:Real}, I::Vector{<:Real}, σ::Vector{<:Real}, ext_curve::Vector{<:Real}, 
    template_psfnuc::Union{Vector{<:Real},Nothing}, N::Real, continuum::Vector{<:Real}, nuc_temp_fit::Bool)

    λnorm, Inorm, σnorm, ext_curve_norm, template_norm = get_normalized_vectors(λ, I, σ, ext_curve, template_psfnuc, N)
    Inorm .-= continuum
    template_norm = nuc_temp_fit ? template_norm : nothing

    λnorm, Inorm, σnorm, ext_curve_norm, template_norm
end


"""
    get_normalized_vectors(λ, I, I_spline, σ, templates, channel_masks, N)

Normalizes I, I_spline, and σ and returns copies of λ, templates, and channel_masks so that the originals are not modified.
"""
function get_normalized_vectors(λ::Vector{<:Real}, I::Vector{<:Real}, I_spline::Vector{<:Real}, σ::Vector{<:Real}, 
    templates::Matrix{<:Real}, channel_masks::Vector{BitVector}, N::Real)

    λ_spax, I_spax, σ_spax, templates_spax, channel_masks_spax = get_normalized_vectors(λ, I, σ, templates, 
        channel_masks, N)
    I_spline_spax = I_spline ./ N

    λ_spax, I_spax, I_spline_spax, σ_spax, templates_spax, channel_masks_spax
end


"""
    interpolate_over_lines!(λ_spax, I_spax, σ_spax, templates_spax, mask_lines, scale)

Fills in the data where the emission lines are with linear interpolation of the continuum.
"""
function interpolate_over_lines!(λ_spax::Vector{<:Real}, I_spax::Vector{<:Real}, σ_spax::Vector{<:Real},
    templates_spax::Matrix{<:Real}, mask_lines::BitVector, scale::Integer; only_templates::Bool=false)

    # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
    λknots = λ_spax[.~mask_lines][(1+scale):scale:(length(λ_spax[.~mask_lines])-scale)]
    # Replace the masked lines with a linear interpolation
    if !only_templates
        I_spax[mask_lines] .= Spline1D(λ_spax[.~mask_lines], I_spax[.~mask_lines], λknots, k=1, bc="extrapolate").(λ_spax[mask_lines]) 
        σ_spax[mask_lines] .= Spline1D(λ_spax[.~mask_lines], σ_spax[.~mask_lines], λknots, k=1, bc="extrapolate").(λ_spax[mask_lines])
    end
    for s in axes(templates_spax, 2)
        m = .~isfinite.(templates_spax[:, s])
        templates_spax[mask_lines .| m, s] .= Spline1D(λ_spax[.~mask_lines .& .~m], templates_spax[.~mask_lines .& .~m, s], λknots, 
            k=1, bc="extrapolate")(λ_spax[mask_lines .| m])
    end

end


"""
    fill_bad_pixels(cube_fitter, I, σ, template)

Helper function to fill in NaNs/Infs in a 1D intensity/error vector and 2D templates vector.
"""
function fill_bad_pixels(cube_fitter::CubeFitter, I::Vector{<:Real}, σ::Vector{<:Real}, templates::Array{<:Real,2})

    # Edge cases
    if !isfinite(I[1])
        I[1] = nanmedian(I)
    end
    if !isfinite(I[end])
        I[end] = nanmedian(I)
    end
    if !isfinite(σ[1])
        σ[1] = nanmedian(σ)
    end
    if !isfinite(σ[end])
        σ[end] = nanmedian(σ)
    end
    for s in 1:cube_fitter.n_templates
        if !isfinite(templates[1,s])
            templates[1,s] = nanmedian(templates[:,s])
        end
        if !isfinite(templates[end,s])
            templates[end,s] = nanmedian(templates[:,s])
        end
    end

    bad = findall(.~isfinite.(I) .| .~isfinite.(σ))
    # Replace with the average of the points to the left and right
    l = length(I)
    for badi in bad
        lind = findfirst(isfinite, I[max(badi-1,1):-1:1])
        rind = findfirst(isfinite, I[min(badi+1,l):end])
        I[badi] = (I[max(badi-1,1):-1:1][lind] + I[min(badi+1,l):end][rind]) / 2
        σ[badi] = (σ[max(badi-1,1):-1:1][lind] + σ[min(badi+1,l):end][rind]) / 2
    end
    @assert all(isfinite.(I) .& isfinite.(σ)) "Error: Non-finite values found in the summed intensity/error arrays!"
    for s in 1:cube_fitter.n_templates
        bad = findall(.~isfinite.(templates[:, s]))
        l = length(templates[:, s])
        for badi in bad
            lind = findfirst(isfinite, templates[max(badi-1,1):-1:1, s])
            rind = findfirst(isfinite, templates[min(badi+1,l):end, s])
            templates[badi, s] = (templates[max(badi-1,1):-1:1, s][lind] + templates[min(badi+1,l):end, s][rind]) / 2
        end
    end
    @assert all(isfinite.(templates)) "Error: Non-finite values found in the summed template arrays!"

    return I, σ, templates
end


"""
    mask_vectors!(mask_bad, user_mask, λ, I, σ, templates, channel_masks[, I_spline])

Apply two masks (mask_bad and user_mask) to a set of vectors (λ, I, σ, templates, channel_masks) to prepare
them for fitting. The mask_bad is a pixel-by-pixel mask flagging bad data, while the user_mask is a set of pairs
of wavelengths specifying entire regions to mask out.  The vectors are modified in-place.
"""
function mask_vectors!(mask_bad::BitVector, user_mask::Union{Nothing,Vector{<:Tuple}}, λ::Vector{<:Real}, I::Vector{<:Real}, 
    σ::Vector{<:Real}, templates::Matrix{<:Real}, channel_masks::Vector{BitVector}, I_spline::Union{Nothing,Vector{<:Real}}=nothing)

    do_spline = !isnothing(I_spline)

    λ = λ[.~mask_bad]
    I = I[.~mask_bad]
    σ = σ[.~mask_bad]
    templates = templates[.~mask_bad, :]
    channel_masks = [ch_mask[.~mask_bad] for ch_mask in channel_masks]
    if do_spline 
        I_spline = I_spline[.~mask_bad]
    end

    if !isnothing(user_mask)
        for pair in user_mask
            region = pair[1] .< λ .< pair[2]
            λ = λ[.~region]
            I = I[.~region]
            σ = σ[.~region]
            templates = templates[.~region, :]
            channel_masks = [ch_mask[.~region] for ch_mask in channel_masks]
            if do_spline
                I_spline = I_spline[.~region]
            end
        end
    end
    
end


"""
    continuum_cubic_spline(λ, I, σ, spectral_region)

Mask out the emission lines in a given spectrum using `mask_emission_lines` and replace them with
a coarse cubic spline fit to the continuum, using wide knots to avoid reinterpolating the lines or
noise. Returns the line mask, spline-interpolated I, and spline-interpolated σ.

# Arguments
- `λ::Vector{<:Real}`: The wavelength vector of the spectrum
- `I::Vector{<:Real}`: The flux vector of the spectrum
- `σ::Vector{<:Real}`: The uncertainty vector of the spectrum 
- `spectral_region::Symbol`: Either :MIR for mid-infrared or :OPT for optical

See also [`mask_emission_lines`](@ref)
"""
function continuum_cubic_spline(λ::Vector{<:Real}, I::Vector{<:Real}, σ::Vector{<:Real}, linemask_Δ::Integer, 
    linemask_n_inc_thresh::Integer, linemask_thresh::Real, overrides::Vector{Tuple{T,T}}=Vector{Tuple{Real,Real}}()) where {T<:Real}

    # Mask out emission lines so that they aren't included in the continuum fit
    mask_lines = mask_emission_lines(λ, I, linemask_Δ, linemask_n_inc_thresh, linemask_thresh, overrides)

    # Interpolate the NaNs
    # Break up cubic spline interpolation into knots 7 pixels long
    # (longer than a narrow emission line but not too long)
    scale = 7

    # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
    λknots = λ[1+scale:scale:end-scale]
    # Remove any knots that happen to fall within a masked pixel
    good = []
    for i ∈ eachindex(λknots)
        _, ind = findmin(abs.(λ .- λknots[i]))
        if ~mask_lines[ind]
            append!(good, [i])
        end
    end
    λknots = λknots[good]
    @debug "Performing cubic spline continuum fit with $(length(λknots)) knots"

    # Do a full cubic spline interpolation of the data
    I_spline = Spline1D(λ[.~mask_lines], I[.~mask_lines], λknots, k=3, bc="extrapolate").(λ)
    σ_spline = Spline1D(λ[.~mask_lines], σ[.~mask_lines], λknots, k=3, bc="extrapolate").(λ)
    # Linear interpolation over the lines
    I_spline[mask_lines] .= Spline1D(λ[.~mask_lines], I[.~mask_lines], λknots, k=1, bc="extrapolate").(λ[mask_lines])
    σ_spline[mask_lines] .= Spline1D(λ[.~mask_lines], σ[.~mask_lines], λknots, k=1, bc="extrapolate").(λ[mask_lines])

    mask_lines, I_spline, σ_spline
end


"""
    calculate_statistical_errors(I, I_spline, mask)

Uses the residuals with a cubic spline fit to calculate statistical errors
purely based on the scatter in the data.
"""
function calculate_statistical_errors(I::Vector{<:Real}, I_spline::Vector{<:Real}, mask::BitVector)

    l_mask = sum(.~mask)
    # Statistical uncertainties based on the local RMS of the residuals with a cubic spline fit
    σ_stat = zeros(l_mask)
    for i in 1:l_mask
        indices = sortperm(abs.((1:l_mask) .- i))[1:60]
        σ_stat[i] = std(I[.~mask][indices] .- I_spline[.~mask][indices])
    end
    # We insert at the locations of the lines since the cubic spline does not include them
    l_all = length(I)
    line_inds = (1:l_all)[mask]
    for line_ind ∈ line_inds
        insert!(σ_stat, line_ind, σ_stat[max(line_ind-1, 1)])
    end
    @debug "Statistical uncertainties: ($(σ_stat[1]) - $(σ_stat[end]))"

    σ_stat

end


# Helper function to generate pre-computed stellar templates if all ages/metallicities are locked
function precompute_stellar_templates(cube_fitter::CubeFitter, pars_0::Vector{<:Real}, plock::BitVector)
    lock_ssps = true
    ages = []
    logzs = []
    pᵢ = 1
    for _ in 1:cube_fitter.n_ssps
        lock_ssps &= plock[pᵢ+1] & plock[pᵢ+2]
        push!(ages, pars_0[pᵢ+1])
        push!(logzs, pars_0[pᵢ+2])
        pᵢ += 3
    end
    if lock_ssps
        @debug "Pre-calculating SSPs since all ages and metallicities are locked."
        stellar_templates = zeros(length(cube_fitter.ssp_λ), cube_fitter.n_ssps)
        for i in 1:cube_fitter.n_ssps
            stellar_templates[:, i] .= [cube_fitter.ssp_templates[j](ages[i], logzs[i]) for j in eachindex(cube_fitter.ssp_λ)]
        end
    else
        @debug "SSPs will be interpolated during the fit since ages and/or metallicities are free."
        stellar_templates = cube_fitter.ssp_templates
    end

    stellar_templates
end


# Helper function for estimating PAH template amplitude when PAH templates are not actually
# used during the fit.
function estimate_pah_template_amplitude(cube_fitter::CubeFitter, λ::Vector{<:Real}, comps::Dict)
    if cube_fitter.spectral_region == :MIR
        pahtemp = zeros(length(λ))
        for i in 1:cube_fitter.n_dust_feat
            pahtemp .+= comps["dust_feat_$i"]
        end
        pah_amp = repeat([maximum(pahtemp)/2], 2)
    else
        pah_amp = zeros(2)
    end

    pah_amp
end


# Helper function to get the continuum to be subtracted for the line fit
function get_continuum_for_line_fit(cube_fitter::CubeFitter, λ::Vector{<:Real}, I::Vector{<:Real}, I_cont::Vector{<:Real},
    comps_cont::Dict, norm::Real, nuc_temp_fit::Bool)
    line_cont = copy(I_cont)
    if cube_fitter.subtract_cubic_spline
        full_cont = I ./ norm
        temp_cont = zeros(eltype(line_cont), length(line_cont))
        # subtract any templates first
        if !nuc_temp_fit
            for comp ∈ keys(comps_cont)
                if contains(comp, "templates_")
                    temp_cont .+= comps_cont[comp]
                end
            end
        end
        # do cubic spline fit
        _, notemp_cont, _ = continuum_cubic_spline(λ, full_cont .- temp_cont, zeros(eltype(line_cont), length(line_cont)), 
            cube_fitter.linemask_Δ, cube_fitter.linemask_n_inc_thresh, cube_fitter.linemask_thresh, cube_fitter.linemask_overrides)
        line_cont = temp_cont .+ notemp_cont
    end
    line_cont
end


# Helper function for getting the total integrated intensity/error/solid angle over the whole FOV
function get_total_integrated_intensities(cube_fitter::CubeFitter; shape::Union{Tuple,Nothing}=nothing)

    @info "Integrating spectrum across the whole cube..."
    I = sumdim(cube_fitter.cube.I, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    σ = sqrt.(sumdim(cube_fitter.cube.σ.^2, (1,2))) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    area_sr = cube_fitter.cube.Ω .* sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    templates = zeros(size(I)..., cube_fitter.n_templates)
    for s in 1:cube_fitter.n_templates
        templates[:,s] .= sumdim(cube_fitter.templates, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    end
    I, σ, templates = fill_bad_pixels(cube_fitter, I, σ, templates)

    # I[.~isfinite.(I)] .= Spline1D(cube_fitter.cube.λ[isfinite.(I[1,1,:])], I[isfinite.(I)], k=1, bc="extrapolate")(cube_fitter.cube.λ[.~isfinite.(I[1,1,:])])
    # σ[.~isfinite.(σ)] .= Spline1D(cube_fitter.cube.λ[isfinite.(σ[1,1,:])], σ[isfinite.(σ)], k=1, bc="extrapolate")(cube_fitter.cube.λ[.~isfinite.(σ[1,1,:])])
    # for s in 1:cube_fitter.n_templates
    #     templates[1,1,.~isfinite.(templates[1,1,:,s]),s] .= Spline1D(cube_fitter.cube.λ[isfinite.(templates[1,1,:,s])], 
    #         templates[1,1,:,s][isfinite.(templates[1,1,:,s])], k=1, bc="extrapolate")(cube_fitter.cube.λ[.~isfinite.(templates[1,1,:,s])])
    # end

    if !isnothing(shape)
        I = reshape(I, shape...)
        σ = reshape(σ, shape...)
        area_sr = reshape(area_sr, shape...)
        templates = reshape(templates, shape..., cube_fitter.n_templates)
    end

    I, σ, templates, area_sr
end


# Helper function for getting the total integrated intensity/error/solid angle over an aperture
function get_aperture_integrated_intensities(cube_fitter::CubeFitter, shape::Tuple, 
    aperture::Vector{<:Aperture.AbstractAperture})

    # If using an aperture, overwrite the cube_data object with the quantities within
    # the aperture, which are calculated here.
    # Prepare the 1D arrays
    I = zeros(Float32, shape)
    σ = zeros(Float32, shape)
    templates = zeros(Float32, shape..., cube_fitter.n_templates)
    area_sr = zeros(shape)

    @info "Performing aperture photometry to get an integrated spectrum..."
    for z ∈ 1:shape[3]

        # Sum up the FLUX within the aperture
        Fz = cube_fitter.cube.I[:, :, z] .* cube_fitter.cube.Ω
        e_Fz = cube_fitter.cube.σ[:, :, z] .* cube_fitter.cube.Ω
        # Zero out the masked spaxels
        Fz[cube_fitter.cube.mask[:, :, z]] .= 0.
        e_Fz[cube_fitter.cube.mask[:, :, z]] .= 0.
        # Perform the aperture photometry
        (_, _, F_ap, eF_ap) = photometry(aperture[z], Fz, e_Fz)

        # Convert back to intensity by dividing out the aperture area
        area_sr[1,1,z] = get_area(aperture[z]) * cube_fitter.cube.Ω
        I[1,1,z] = F_ap / area_sr[1,1,z]
        σ[1,1,z] = eF_ap / area_sr[1,1,z]

        # repeat for templates
        for s in 1:cube_fitter.n_templates
            Ftz = cube_fitter.templates[:, :, z, s] .* cube_fitter.cube.Ω
            Ftz[.~isfinite.(Ftz) .| cube_fitter.cube.mask[:, :, z]] .= 0.
            Ft_ap = photometry(aperture[z], Ftz).aperture_sum
            templates[1,1,z,s] = Ft_ap / area_sr[1,1,z]
        end
    end

    I, σ, templates, area_sr
end


# Helper function for getting the number of profiles and number of parameters for each profile
# for a given emission line
function get_line_nprof_ncomp(cube_fitter::CubeFitter, i::Integer)

    n_prof = 0
    pcomps = Int[]
    for j in 1:cube_fitter.n_comps
        if !isnothing(cube_fitter.lines.profiles[i, j])
            n_prof += 1
            # Check if using a flexible_wavesol tied voff -> if so there is an extra voff parameter
            if !isnothing(cube_fitter.lines.tied_voff[i, j]) && cube_fitter.flexible_wavesol && isone(j)
                pc = 4
            else
                pc = 3
            end
            if cube_fitter.lines.profiles[i, j] == :GaussHermite
                pc += 2
            elseif cube_fitter.lines.profiles[i, j] == :Voigt
                pc += 1
            end
            push!(pcomps, pc)
        end
    end

    n_prof, pcomps
end


"""
    create_cube_data(cube_fitter, shape)

Makes a named tuple object that holds the wavelengths, intensities, errors,
templates, and solid angles.
"""
function create_cube_data(cube_fitter::CubeFitter, shape::Tuple)

    cube_data = (λ=cube_fitter.cube.λ, I=cube_fitter.cube.I, σ=cube_fitter.cube.σ, 
        templates=cube_fitter.n_templates > 0 ? cube_fitter.templates : Array{Float64}(undef, shape..., 0),
        area_sr=cube_fitter.cube.Ω .* ones(shape))

    n_bins = 0
    vorbin = !isnothing(cube_fitter.cube.voronoi_bins)
    if vorbin
        # Reformat cube data as a 2D array with the first axis slicing each voronoi bin
        n_bins = maximum(cube_fitter.cube.voronoi_bins)
        I_vorbin = zeros(n_bins, shape[3])
        σ_vorbin = zeros(n_bins, shape[3])
        area_vorbin = zeros(n_bins, shape[3])
        template_vorbin = zeros(n_bins, shape[3], cube_fitter.n_templates)
        for n in 1:n_bins
            w = cube_fitter.cube.voronoi_bins .== n
            I_vorbin[n, :] .= sumdim(cube_fitter.cube.I[w, :], 1) ./ sum(w)
            σ_vorbin[n, :] .= sqrt.(sumdim(cube_fitter.cube.σ[w, :].^2, 1)) ./ sum(w)
            area_vorbin[n, :] .= sum(w) .* cube_fitter.cube.Ω
            for s in 1:cube_fitter.n_templates
                template_vorbin[n, :, s] .= sumdim(cube_fitter.templates[w, :, s], 1) ./ sum(w)
            end
        end
        cube_data = (λ=cube_fitter.cube.λ, I=I_vorbin, σ=σ_vorbin, area_sr=area_vorbin, templates=template_vorbin)
    end

    cube_data, vorbin, n_bins
end


"""
    create_cube_data_nuctemp(cube_fitter, shape)

Makes a named tuple object that holds the wavelengths, intensities, errors,
templates, and solid angles for a nuclear template fit.
"""
function create_cube_data_nuctemp(cube_fitter::CubeFitter, shape::Tuple)
    cube = cube_fitter.cube

    data2d = dropdims(nansum(cube.I, dims=3), dims=3)
    data2d[.~isfinite.(data2d)] .= 0.
    _, mx = findmax(data2d)

    I = zeros(Float32, shape)
    σ = zeros(Float32, shape)
    templates = zeros(Float32, shape..., cube_fitter.n_templates)
    area_sr = zeros(shape)

    # Take the brightest spaxel
    I[1,1,:] .= cube.I[mx,:]
    σ[1,1,:] .= cube.σ[mx,:]
    for s in 1:cube_fitter.n_templates
        templates[1,1,:,s] .= cube_fitter.templates[mx, :, s]
    end
    area_sr[1,1,:] .= cube_fitter.cube.Ω
    cube_data = (λ=cube_fitter.cube.λ, I=I, σ=σ, templates=templates, area_sr=area_sr)

    cube_data, mx
end


"""
    create_cube_data_postnuctemp(cube_fitter, agn_templates)

Makes a named tuple object that holds the wavelengths, intensities, errors,
templates, and solid angles for a post-nuclear-template-fit fit.
"""
function create_cube_data_postnuctemp(cube_fitter::CubeFitter, agn_templates::Array{<:Real,3})
    cube = cube_fitter.cube

    # Get the AGN model over the whole cube
    I_agn = nansum(agn_templates, dims=(1,2)) ./ nansum(.~cube.mask, dims=(1,2))
    σ_agn = nanmedian(I_agn) .* ones(Float64, size(I_agn))  # errors will be overwritten by statistical errors
    templates = Array{Float64,4}(undef, size(I_agn)..., 0)
    area_sr = cube_fitter.cube.Ω .* nansum(cube.mask, dims=(1,2))
    I, σ, temps = fill_bad_pixels(cube_fitter, I_agn[1,1,:], σ_agn[1,1,:], templates[1,1,:,:])

    I_agn[1,1,:] .= I
    σ_agn[1,1,:] .= σ
    templates[1,1,:,:] .= temps

    cube_data = (λ=cube_fitter.cube.λ, I=I_agn, σ=σ_agn, templates=templates, area_sr=area_sr)

    cube_data
end


# Helper function to collect the results of a fit into a parameter vector, error vector,
# intensity vector as a function of wavelength, and a comps dict giving individual components
# of the final model
function collect_fit_results(res::CMPFit.Result, pfix_tied::Vector{<:Real}, plock_tied::BitVector, 
    tied_pairs::Vector{Tuple}, tied_indices::Vector{<:Integer}, n_tied::Integer, 
    stellar_templates::Union{Vector{Spline2D},Matrix{<:Real}}, cube_fitter::CubeFitter, 
    λ::Vector{<:Real}, templates::Matrix{<:Real}, N::Real; nuc_temp_fit::Bool=false, 
    bootstrap_iter::Bool=false)

    popt_0 = rebuild_full_parameters(res.param, pfix_tied, plock_tied, tied_pairs, tied_indices, n_tied)

    # if fitting the nuclear template, minimize the distance from 1.0 for the PSF template amplitudes and move
    # the residual amplitude to the continuum
    if nuc_temp_fit && cube_fitter.spectral_region == :MIR
        nuc_temp_fit_minimize_psftemp_amp!(cube_fitter, popt_0)
    end

    popt = popt_0
    perr = zeros(Float64, length(popt))
    if !bootstrap_iter
        perr = rebuild_full_parameters(res.perror, zeros(length(pfix_tied)), plock_tied, tied_pairs, tied_indices, n_tied)
    end

    @debug "Best fit continuum parameters: \n $popt"
    @debug "Continuum parameter errors: \n $perr"
    # @debug "Continuum covariance matrix: \n $covar"

    # Create the full model, again only if not bootstrapping
    if cube_fitter.spectral_region == :MIR
        I_model, comps = model_continuum(λ, popt, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.profiles,
            cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.κ_abs, cube_fitter.custom_ext_template, 
            cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, false, templates, cube_fitter.channel_masks, nuc_temp_fit, true)

    else
        I_model, comps = model_continuum(λ, popt, N, cube_fitter.vres, cube_fitter.vsyst_ssp, cube_fitter.vsyst_feii, cube_fitter.npad_feii,
            cube_fitter.n_ssps, cube_fitter.ssp_λ, stellar_templates, cube_fitter.feii_templates_fft, cube_fitter.n_power_law, cube_fitter.fit_uv_bump, 
            cube_fitter.fit_covering_frac, cube_fitter.fit_opt_na_feii, cube_fitter.fit_opt_br_feii, cube_fitter.extinction_curve, templates,
            cube_fitter.fit_temp_multexp, nuc_temp_fit, true)
    end

    popt, perr, I_model, comps
end


# Alternative dispatch for the collect_fit_results function for step 1 of the multistep fitting procedure
function collect_fit_results(res_1::CMPFit.Result, p1fix_tied::Vector{<:Real}, lock_1_tied::BitVector, 
    tied_pairs::Vector{Tuple}, tied_indices::Vector{<:Integer}, n_tied_1::Integer, cube_fitter::CubeFitter,
    λ_spax::Vector{<:Real}, fit_step1::Function; nuc_temp_fit::Bool=false)

    popt_0 = rebuild_full_parameters(res_1.param, p1fix_tied, lock_1_tied, tied_pairs, tied_indices, n_tied_1)

    # if fitting the nuclear template, minimize the distance from 1.0 for the PSF template amplitudes and move
    # the residual amplitude to the continuum
    if nuc_temp_fit
        nuc_temp_fit_minimize_psftemp_amp!(cube_fitter, popt_0)
    end
    popt_0_tied = copy(popt_0)
    deleteat!(popt_0_tied, tied_indices)
    popt_0_free_tied = popt_0_tied[.~lock_1_tied]

    # Create continuum without the PAH features
    _, ccomps = fit_step1(λ_spax, popt_0_free_tied, true)

    I_cont = ccomps["obscured_continuum"] .+ ccomps["unobscured_continuum"]
    if cube_fitter.fit_sil_emission
        abs_tot = ones(length(λ_spax))
        for k ∈ 1:cube_fitter.n_abs_feat
            abs_tot .*= ccomps["abs_feat_$k"]
        end
        I_cont .+= ccomps["hot_dust"] .* abs_tot
    end
    template_norm = nothing
    for l ∈ 1:cube_fitter.n_templates
        if !nuc_temp_fit
            I_cont .+= ccomps["templates_$l"]
        else
            template_norm = ccomps["templates_$l"]
        end
    end
    if nuc_temp_fit
        I_cont .*= template_norm
    end

    I_cont, ccomps, template_norm
end


# Alternative dispatch for the collect_fit_results function for step 2 of the multistep fitting procedure
function collect_fit_results(res_1::CMPFit.Result, p1fix_tied::Vector{<:Real}, lock_1_tied::BitVector,
    tied_pairs::Vector{Tuple}, tied_indices::Vector{<:Integer}, n_tied_1::Integer, res_2::CMPFit.Result,
    p2fix::Vector{<:Real}, lock_2::BitVector, n_free_1::Integer, n_free_2::Integer, cube_fitter::CubeFitter,
    λ::Vector{<:Real}, templates::Matrix{<:Real}, N::Real; bootstrap_iter::Bool=false, nuc_temp_fit::Bool=false)

    popt_1 = rebuild_full_parameters(res_1.param, p1fix_tied, lock_1_tied, tied_pairs, tied_indices, n_tied_1)
    # remove the PAH template amplitudes
    pahtemp = popt_1[end-1:end]
    popt_1 = popt_1[1:end-2]
    perr_1 = zeros(length(popt_1))
    if !bootstrap_iter
        perr_1 = rebuild_full_parameters(res_1.perror, zeros(length(p1fix_tied)), lock_1_tied, tied_pairs, tied_indices, n_tied_1)
        perr_1 = perr_1[1:end-2]
    end

    popt_2 = rebuild_full_parameters(res_2.param, p2fix, lock_2)
    perr_2 = zeros(length(lock_2))
    if !bootstrap_iter
        perr_2 = rebuild_full_parameters(res_2.perror, zeros(length(p2fix)), lock_2)
    end

    popt = [popt_1; popt_2]
    perr = [perr_1; perr_2]

    n_free = n_free_1 + n_free_2 - 2

    # Create the full model, again only if not bootstrapping
    I_model, comps = model_continuum(λ, popt, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.profiles,
        cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.κ_abs, cube_fitter.custom_ext_template, 
        cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, false, templates, cube_fitter.channel_masks, nuc_temp_fit, true)

    popt, perr, n_free, pahtemp, I_model, comps
end


# Alternative dispatch for the collect_fit_results function for the line fitting procedure
function collect_fit_results(res::CMPFit.Result, pfix_tied::Vector{<:Real}, param_lock_tied::BitVector,
    tied_pairs::Vector{Tuple}, tied_indices::Vector{<:Integer}, n_tied::Integer, cube_fitter::CubeFitter,
    λ::Vector{<:Real}, ext_curve::Vector{<:Real}, lsf_interp_func::Function, template_norm::Union{Vector{<:Real},Nothing},
    nuc_temp_fit::Bool; bootstrap_iter::Bool=false)

    # Get the results and errors
    popt = rebuild_full_parameters(res.param, pfix_tied, param_lock_tied, tied_pairs, tied_indices, n_tied)

    # Dont both with uncertainties if bootstrapping
    perr = zeros(Float64, length(popt))
    if !bootstrap_iter
        perr = rebuild_full_parameters(res.perror, zeros(length(pfix_tied)), param_lock_tied, tied_pairs, tied_indices, n_tied)
    end

    # Final optimized fit
    I_model, comps = model_line_residuals(λ, popt, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, 
        cube_fitter.flexible_wavesol, ext_curve, lsf_interp_func, cube_fitter.relative_flags, template_norm, nuc_temp_fit, true)

    popt, perr, I_model, comps
end


# Alternative dispatch for the collect_fit_results function for the joint continuum+line fitting procedure
function collect_fit_results(res::CMPFit.Result, pfix_cont_tied::Vector{<:Real}, lock_cont_tied::BitVector, 
    tied_pairs_cont::Vector{Tuple}, tied_indices_cont::Vector{<:Integer}, n_tied_cont::Integer, n_free_cont::Integer,
    pfix_lines_tied::Vector{<:Real}, lock_lines_tied::BitVector, tied_pairs_lines::Vector{Tuple}, 
    tied_indices_lines::Vector{<:Integer}, n_tied_lines::Integer, n_free_lines::Integer, 
    stellar_templates::Union{Vector{Spline2D},Matrix{<:Real}}, cube_fitter::CubeFitter,
    λ::Vector{<:Real}, N::Real, templates::Matrix{<:Real}, lsf_interp_func::Function, nuc_temp_fit::Bool;
    bootstrap_iter::Bool=bootstrap_iter)

    popt_cont = rebuild_full_parameters(res.param[1:n_free_cont], pfix_cont_tied, lock_cont_tied, tied_pairs_cont, tied_indices_cont, n_tied_cont)
    perr_cont = zeros(Float64, length(popt_cont))
    if !bootstrap_iter
        perr_cont = rebuild_full_parameters(res.perror[1:n_free_cont], zeros(length(pfix_cont_tied)), lock_cont_tied, tied_pairs_cont, 
            tied_indices_cont, n_tied_cont)
    end

    popt_lines = rebuild_full_parameters(res.param[n_free_cont+1:end], pfix_lines_tied, lock_lines_tied, tied_pairs_lines, tied_indices_lines, n_tied_lines)
    perr_lines = zeros(Float64, length(popt_lines))
    if !bootstrap_iter
        perr_lines = rebuild_full_parameters(res.perror[n_free_cont+1:end], zeros(length(pfix_lines_tied)), lock_lines_tied, tied_pairs_lines, 
            tied_indices_lines, n_tied_lines)
    end

    @debug "Best fit continuum parameters: \n $popt_cont"
    @debug "Continuum parameter errors: \n $perr_cont"
    @debug "Best fit line parameters: \n $popt_lines"
    @debug "Line parameter errors: \n $perr_lines"

    # Create the full model
    if cube_fitter.spectral_region == :MIR
        Icont, comps_cont = model_continuum(λ, popt_cont, N, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.dust_features.profiles,
            cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, cube_fitter.κ_abs, cube_fitter.custom_ext_template, 
            cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, false, templates, cube_fitter.channel_masks, nuc_temp_fit, true)
        ext_key = "extinction"
    else
        Icont, comps_cont = model_continuum(λ, popt_cont, N, cube_fitter.vres, cube_fitter.vsyst_ssp, cube_fitter.vsyst_feii, cube_fitter.npad_feii,
            cube_fitter.n_ssps, cube_fitter.ssp_λ, stellar_templates, cube_fitter.feii_templates_fft, cube_fitter.n_power_law, cube_fitter.fit_uv_bump, 
            cube_fitter.fit_covering_frac, cube_fitter.fit_opt_na_feii, cube_fitter.fit_opt_br_feii, cube_fitter.extinction_curve, templates,
            cube_fitter.fit_temp_multexp, nuc_temp_fit, true)
        ext_key = "attenuation_gas"
    end
    Ilines, comps_lines = model_line_residuals(λ, popt_lines, cube_fitter.n_lines, cube_fitter.n_comps, cube_fitter.lines, cube_fitter.flexible_wavesol,
        comps_cont[ext_key], lsf_interp_func, cube_fitter.relative_flags, nuc_temp_fit ? comps_cont["templates_1"] : nothing, nuc_temp_fit, true)
    
    popt_cont, perr_cont, Icont, comps_cont, popt_lines, perr_lines, Ilines, comps_lines
end


# Helper function to get the total fit results for continuum and line fits
function collect_total_fit_results(I::Vector{<:Real}, σ::Vector{<:Real}, I_cont::Vector{<:Real}, 
    I_line::Vector{<:Real}, comps_cont::Dict, comps_line::Dict, n_free_c::Integer, n_free_l::Integer, 
    norm::Real, mask_chi2::BitVector, nuc_temp_fit::Bool)

    # Combine the continuum and line models
    I_model = I_cont .+ I_line
    comps = merge(comps_cont, comps_line)

    # Renormalize
    I_model .*= norm
    for comp ∈ keys(comps)
        if (comp == "extinction") || contains(comp, "ext_") || contains(comp, "abs_") || contains(comp, "attenuation")
            continue
        end
        if nuc_temp_fit && contains(comp, "template")
            continue
        end
        comps[comp] .*= norm
    end

    # Total free parameters
    n_free = n_free_c + n_free_l
    n_data = length(I)

    # Degrees of freedom
    dof = n_data - n_free

    # chi^2 and reduced chi^2 of the model
    χ2 = sum(@. (I[.~mask_chi2] - I_model[.~mask_chi2])^2 / σ[.~mask_chi2]^2)

    I_model, comps, χ2, dof
end


# Helper function to get bootstrapped fit results 
function collect_bootstrapped_results(cube_fitter::CubeFitter, p_boot::Matrix{<:Real}, 
    λ::Vector{<:Real}, I::Vector{<:Real}, σ::Vector{<:Real}, I_model_boot::Matrix{<:Real}, 
    norm::Real, split1::Integer, split2::Integer, lsf_interp_func::Function, mask_chi2::BitVector, 
    templates::Matrix{<:Real}, nuc_temp_fit::Bool)

    # Filter out any large (>5 sigma) outliers
    p_med = dropdims(nanquantile(p_boot, 0.50, dims=2), dims=2) 
    p_mad = nanmad(p_boot, dims=2) .* 1.4826   # factor of 1.4826 to normalize the MAD it so it is interpretable as a standard deviation
    p_mask = (p_boot .< (p_med .- 5 .* p_mad)) .| (p_boot .> (p_med .+ 5 .* p_mad))
    p_boot[p_mask] .= NaN

    if cube_fitter.bootstrap_use == :med
        p_out = p_med
    end
    # (if set to :best, p_out is already the best-fit values from earlier)
    p_err_lo = p_med .- dropdims(nanquantile(p_boot, 0.159, dims=2), dims=2)
    p_err_up = dropdims(nanquantile(p_boot, 0.841, dims=2), dims=2) .- p_med
    p_err = [p_err_lo p_err_up]

    # Get the minimum/maximum pointwise bootstrapped models
    I_boot_min = dropdims(nanminimum(I_model_boot, dims=2), dims=2)
    I_boot_max = dropdims(nanmaximum(I_model_boot, dims=2), dims=2)


    # Replace the best-fit model with the 50th percentile model to be consistent with p_out
    if cube_fitter.spectral_region == :MIR
        I_boot_cont, comps_boot_cont = model_continuum(λ, p_out[1:split1], norm, cube_fitter.n_dust_cont, cube_fitter.n_power_law, 
            cube_fitter.dust_features.profiles, cube_fitter.n_abs_feat, cube_fitter.extinction_curve, cube_fitter.extinction_screen, 
            cube_fitter.κ_abs, cube_fitter.custom_ext_template, cube_fitter.fit_sil_emission, cube_fitter.fit_temp_multexp, false, 
            templates, cube_fitter.channel_masks, nuc_temp_fit, true)
        ext_key = "extinction"
    else
        I_boot_cont, comps_boot_cont = model_continuum(λ, p_out[1:split1], norm, cube_fitter.vres, cube_fitter.vsyst_ssp, 
            cube_fitter.vsyst_feii, cube_fitter.npad_feii, cube_fitter.n_ssps, cube_fitter.ssp_λ, cube_fitter.ssp_templates,
            cube_fitter.feii_templates_fft, cube_fitter.n_power_law, cube_fitter.fit_uv_bump, cube_fitter.fit_covering_frac, 
            cube_fitter.fit_opt_na_feii, cube_fitter.fit_opt_br_feii, cube_fitter.extinction_curve, templates, cube_fitter.fit_temp_multexp,
            nuc_temp_fit, true)
        ext_key = "attenuation_gas"
    end
    I_boot_line, comps_boot_line = model_line_residuals(λ, p_out[split1+1:split2], cube_fitter.n_lines, cube_fitter.n_comps,
        cube_fitter.lines, cube_fitter.flexible_wavesol, comps_boot_cont[ext_key], lsf_interp_func, cube_fitter.relative_flags,
        nuc_temp_fit ? comps_boot_cont["templates_1"] : nothing, nuc_temp_fit, true)

    # Reconstruct the full model
    I_model, comps, χ2, _ = collect_total_fit_results(I, σ, I_boot_cont, I_boot_line, comps_boot_cont,
        comps_boot_line, 0, 0, norm, mask_chi2, nuc_temp_fit)
    # Recalculate chi^2 based on the median model
    p_out[end-1] = χ2

    p_out, p_err, I_boot_min, I_boot_max, I_model, comps, χ2
end


# Helper function that decides whether a fit should be redone with extinction locked to 0,
# based on how much the templates dominate the fit
function determine_fit_redo_with0extinction(cube_fitter::CubeFitter, λ::Vector{<:Real}, I::Vector{<:Real},
    σ::Vector{<:Real}, N::Real, popt::Vector{<:Real}, plock::BitVector, I_model::Vector{<:Real}, comps::Dict, 
    init::Bool, force_noext::Bool)

    redo_fit = false
    for s in 1:cube_fitter.n_templates
        pₑ = [3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law]
        sil_abs_region = 9.0 .< λ .< 11.0
        ext_not_locked = any(.~plock[pₑ]) && !init && !force_noext && any(popt[pₑ] .≠ 0)
        region_is_valid = sum(sil_abs_region) > 100
        template_dominates = sum((abs.(I_model .- comps["templates_$s"])[sil_abs_region]) .< (σ[sil_abs_region]./N)) > (sum(sil_abs_region)/2)
        if ext_not_locked && region_is_valid && template_dominates
            redo_fit = true
            break
        end
    end

    redo_fit
end

