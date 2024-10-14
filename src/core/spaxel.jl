

# Datatype for holding a 1D spectrum for a single spaxel
mutable struct Spaxel{Q1<:QWave} 

    coords::CartesianIndex
    λ::AbstractVector{Q1}
    I::AbstractVector
    σ::AbstractVector
    mask_lines::BitVector
    mask_bad::BitVector
    I_spline::Union{Nothing,AbstractVector}
    N::Number
    aux::Dict   # for storing various auxiliary data
    normalized::Bool

    function Spaxel(coords::CartesianIndex, λ::AbstractVector{Q1}, I::AbstractVector{Q2}, σ::AbstractVector{Q2}, 
        mask_lines::BitVector, mask_bad::BitVector, I_spline::Union{AbstractVector{Q2},Nothing}=nothing; 
        N::Q3=1.0, aux::Dict=Dict()) where {Q1<:QWave, Q2<:Number, Q3<:Number}

        @assert eltype(I) == eltype(σ)
        if !isnothing(I_spline)
            @assert eltype(I) == eltype(I_spline)
        end
        @assert length(λ) == length(I) == length(σ) == length(mask_lines) == length(mask_bad)
        if unit(I[1]) <: NoUnits
            @assert unit(N) <: QSIntensity "I and σ are already normalized!"
            normalized = true
        end
        if unit(I[1]) <: QSIntensity
            @assert unit(N) <: NoUnits "I and σ are not yet normalized!" 
            normalized = false
        end

        new{Q1,Q2,Q3}(coords, λ, I, σ, mask_lines, mask_bad, I_spline, N, aux, normalized)
    end
end


function Base.copy(s::Spaxel)
    Spaxel(s.coords, copy(s.λ), copy(s.I), copy(s.σ), copy(s.mask_lines), copy(s.mask_bad), copy(s.I_spline);
        N=s.N, aux=copy(s.aux))
end


# create a new normalized spaxel
function normalize!(s::Spaxel{T1,T2,T3}, N::T3) where {T1,T2,T3}
    s.I = s.I ./ N
    s.σ = s.σ ./ N
    if !isnothing(s.I_spline)
        s.I_spline = s.I_spline ./ N
    end
    if haskey(s.aux, "templates")
        s.aux["templates"] = s.aux["templates"] ./ N
    end
    unit_check(unit(s.I[1]), NoUnits)
    unit_check(unit(s.I[1]), unit(s.σ[1]))
    if !isnothing(s.I_spline)
        unit_check(unit(s.I[1]), unit(s.I_spline[1]))
    end
    if haskey(s.aux, "templates")
        unit_check(unit(s.I[1]), unit(s.aux["templates"]))
    end
    s.normalized[1] = true
end


function subtract_continuum!(s::Spaxel, continuum::AbstractVector{<:Real})
    unit_check(unit(s.I[1]), unit(continuum[1]))
    s.I .-= continuum
end


"""
    interpolate_over_lines!(s, mask_lines, scale[, only_templates])

Fills in the data where the emission lines are with linear interpolation of the continuum.
"""
function interpolate_over_lines!(s::Spaxel, scale::Integer; only_templates::Bool=false)

    # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
    λknots = s.λ[.~s.mask_lines][(1+scale):scale:(length(s.λ[.~s.mask_lines])-scale)]

    # Replace the masked lines with a linear interpolation
    if !only_templates
        s.I[s.mask_lines] .= Spline1D(ustrip.(s.λ[.~s.mask_lines]), s.I[.~s.mask_lines], ustrip.(λknots), 
            k=1, bc="extrapolate").(ustrip.(s.λ[s.mask_lines])) .* unit(s.I[1])
        s.σ[s.mask_lines] .= Spline1D(ustrip.(s.λ[.~s.mask_lines]), s.σ[.~s.mask_lines], ustrip.(λknots), 
            k=1, bc="extrapolate").(ustrip.(s.λ[s.mask_lines])) .* unit(s.σ[1])
    end
    # Do it for the templates
    if haskey(s.aux, "templates")
        for s in axes(s.aux["templates"], 2)
            m = .~isfinite.(s.aux["templates"][:, s])
            s.aux["templates"][s.mask_lines.|m, s] .= Spline1D(ustrip.(s.λ[.~s.mask_lines .& .~m]), 
                ustrip.(s.aux["templates"][.~s.mask_lines .& .~m, s]), ustrip.(λknots), k=1, 
                bc="extrapolate")(ustrip.(s.λ[s.mask_lines .| m]))
        end
    end
end


"""
    fill_bad_pixels!(s)

Helper function to fill in NaNs/Infs in a 1D intensity/error vector and 2D templates vector.
"""
function fill_bad_pixels!(s::Spaxel)

    # Edge cases
    if !isfinite(s.I[1])
        s.I[1] = nanmedian(s.I)
    end
    if !isfinite(s.I[end])
        s.I[end] = nanmedian(s.I)
    end
    if !isfinite(s.σ[1])
        s.σ[1] = nanmedian(s.σ)
    end
    if !isfinite(s.σ[end])
        s.σ[end] = nanmedian(s.σ)
    end
    if haskey(s.aux, "templates")
        for s in axes(s.aux["templates"], 2)
            if !isfinite(s.aux["templates"][1,s])
                s.aux["templates"][1,s] = nanmedian(s.aux["templates"][:,s])
            end
            if !isfinite(s.aux["templates"][end,s])
                s.aux["templates"][end,s] = nanmedian(s.aux["templates"][:,s])
            end
        end
    end

    bad = findall(.~isfinite.(s.I) .| .~isfinite.(s.σ))
    # Replace with the average of the points to the left and right
    l = length(s.I)
    for badi in bad
        lind = findfirst(isfinite, s.I[max(badi-1,1):-1:1])
        rind = findfirst(isfinite, s.I[min(badi+1,l):end])
        s.I[badi] = (s.I[max(badi-1,1):-1:1][lind] + s.I[min(badi+1,l):end][rind]) / 2
        s.σ[badi] = (s.σ[max(badi-1,1):-1:1][lind] + s.σ[min(badi+1,l):end][rind]) / 2
    end
    @assert all(isfinite.(s.I) .& isfinite.(s.σ)) "Error: Non-finite values found in the summed intensity/error arrays!"
    if haskey(s.aux, "templates")
        for s in axes(s.aux["templates"], 2)
            bad = findall(.~isfinite.(s.aux["templates"][:, s]))
            l = length(s.aux["templates"][:, s])
            for badi in bad
                lind = findfirst(isfinite, s.aux["templates"][max(badi-1,1):-1:1, s])
                rind = findfirst(isfinite, s.aux["templates"][min(badi+1,l):end, s])
                s.aux["templates"][badi, s] = (s.aux["templates"][max(badi-1,1):-1:1, s][lind] + s.aux["templates"][min(badi+1,l):end, s][rind]) / 2
            end
        end
        @assert all(isfinite.(s.aux["templates"])) "Error: Non-finite values found in the summed template arrays!"
    end

end


function fill_bad_pixels!(s::Spaxel)
    temp = haskey(s.aux, "templates") ? s.aux["templates"] : nothing
    I, σ, templates = fill_bad_pixels(s.I, s.σ, temp)
    s.I = I
    s.σ = σ
    if haskey(s.aux, "templates")
        s.aux["templates"] = templates
    end
end


"""
    mask_vectors!(s, mask_bad[, user_mask])

Apply two masks (mask_bad and user_mask) to a set of vectors (λ, I, σ, templates, channel_masks) to prepare
them for fitting. The mask_bad is a pixel-by-pixel mask flagging bad data, while the user_mask is a set of pairs
of wavelengths specifying entire regions to mask out.  The vectors are modified in-place.
"""
function mask_vectors!(s::Spaxel, user_mask::Union{Nothing,Vector{<:Tuple}}=nothing) 

    do_spline = !isnothing(s.I_spline)
    do_templates = haskey(s.aux, "templates")
    do_chmasks = haskey(s.aux, "channel_masks")

    s.λ = s.λ[.~s.mask_bad]
    s.I = s.I[.~s.mask_bad]
    s.σ = s.σ[.~s.mask_bad]
    if do_spline 
        s.I_spline = s.I_spline[.~s.mask_bad]
    end
    if do_templates
        s.aux["templates"] = s.aux["templates"][.~s.mask_bad, :]
    end
    if do_chmasks
        s.aux["channel_masks"] = [ch_mask[.~s.mask_bad] for ch_mask in s.aux["channel_masks"]]
    end

    if !isnothing(user_mask)
        for pair in user_mask
            region = pair[1] .< s.λ .< pair[2]
            s.λ = s.λ[.~region]
            s.I = s.I[.~region]
            s.σ = s.σ[.~region]
            if do_spline
                s.I_spline = s.I_spline[.~region]
            end
            if do_templates
                s.aux["templates"] = s.aux["templates"][.~region, :]
            end
            if do_chmasks
                s.aux["channel_masks"] = [ch_mask[.~region] for ch_mask in s.aux["channel_masks"]]
            end
        end
    end

    @assert length(s.λ) == length(s.I) == length(s.σ)
    if do_spline
        @assert length(s.λ) == length(s.I_spline)
    end
    if do_templates
        @assert length(s.λ) == size(s.aux["templates"], 1)
    end
    if do_chmasks
        for ch_mask in s.aux["channel_masks"]
            @assert length(s.λ) == length(ch_mask)
        end
    end
end


function continuum_cubic_spline!(s::Spaxel, overrides)
    mask_lines, I_spline = continuum_cubic_spline(s.λ, s.I, s.σ, overrides; do_err=false)
    s.I_spline = I_spline
    s.aux["mask_lines"] = mask_lines
end


function calculate_statistical_errors!(s::Spaxel)
    @assert haskey(s.aux, "mask_lines") "Need to have a line mask first! Run continuum_cubic_spline! to generate it."
    calculate_statistical_errors(s.I, s.I_spline, s.aux["mask_lines"])
end
