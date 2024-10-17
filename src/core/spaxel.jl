

# Datatype for holding a 1D spectrum for a single spaxel
mutable struct Spaxel{Q1<:QWave} 

    coords::CartesianIndex
    λ::Vector{Q1}
    I::Vector
    σ::Vector
    area_sr::Vector
    mask_lines::BitVector
    mask_bad::BitVector
    vres::Union{Number,Nothing}
    I_spline::Union{Nothing,Vector}
    N::Number
    aux::Dict   # for storing various auxiliary data
    normalized::Bool

    function Spaxel(coords::CartesianIndex, λ::Vector{Q1}, I::Vector{Q2}, σ::Vector{Q2}, 
        area_sr::Vector{Q4}, mask_lines::BitVector, mask_bad::BitVector, vres::Union{Nothing,QVelocity}=nothing, 
        I_spline::Union{Vector{Q2},Nothing}=nothing; N::Q3=1.0, aux::Dict=Dict()) where {
            Q1<:QWave, Q2<:Number, Q3<:Number, Q4<:Number
        }

        @assert eltype(I) == eltype(σ)
        if !isnothing(I_spline)
            @assert eltype(I) == eltype(I_spline)
        end
        @assert length(λ) == length(I) == length(σ) == length(area_sr) == length(mask_lines) == length(mask_bad)
        if unit(I[1]) == NoUnits
            @assert typeof(N) <: QSIntensity "I and σ are already normalized!"
            normalized = true
        end
        if eltype(I) <: QSIntensity
            @assert unit(N) == NoUnits "I and σ are not yet normalized!" 
            normalized = false
        end

        new{Q1}(coords, λ, I, σ, area_sr, mask_lines, mask_bad, vres, I_spline, N, aux, normalized)
    end
end


function Base.copy(s::Spaxel)
    Spaxel(s.coords, copy(s.λ), copy(s.I), copy(s.σ), copy(s.mask_lines), copy(s.mask_bad), s.vres, copy(s.I_spline);
        N=s.N, aux=copy(s.aux))
end


# create a new normalized spaxel
function normalize!(s::Spaxel, N::T) where {T}
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
        unit_check(unit(s.I[1]), unit(s.aux["templates"][1]))
    end
    s.normalized = true
    s.N = N
end


function subtract_continuum!(s::Spaxel, continuum::AbstractVector{<:Real})
    unit_check(unit(s.I[1]), unit(continuum[1]))
    s.I .-= continuum
end


# """
#     interpolate_over_lines!(s, mask_lines, scale[, only_templates])

# Fills in the data where the emission lines are with linear interpolation of the continuum.
# """
# function interpolate_over_lines!(s::Spaxel, scale::Integer; only_templates::Bool=false)

#     # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
#     λknots = s.λ[.~s.mask_lines][(1+scale):scale:(length(s.λ[.~s.mask_lines])-scale)]

#     #### => IMPORTANT <= ####
#     # Lines shouldnt be interpolated over in optical wavelength ranges where you might expect 
#     # stellar absorption to be present on top of nebular emission.  Instead these regions should 
#     # just be masked out.

#     # However, in the infrared, there is no stellar absorption.  But there are PAH emission features,
#     # which sometimes may be fit with amplitudes too large if they fall on top of a masked region due
#     # to a line. So here, it's better to interpolate over the lines.
#     mask_ir = s.mask_lines .& s.λ .≥ 3.0u"μm"
#     # Replace the masked lines with a linear interpolation
#     if !only_templates
#         s.I[mask_ir] .= Spline1D(ustrip.(s.λ[.~mask_ir]), s.I[.~mask_ir], ustrip.(λknots), 
#             k=1, bc="extrapolate").(ustrip.(s.λ[mask_ir])) .* unit(s.I[1])
#         s.σ[mask_ir] .= Spline1D(ustrip.(s.λ[.~mask_ir]), s.σ[.~mask_ir], ustrip.(λknots), 
#             k=1, bc="extrapolate").(ustrip.(s.λ[mask_ir])) .* unit(s.σ[1])
#     end
#     # Do it for the templates
#     if haskey(s.aux, "templates")
#         for s in axes(s.aux["templates"], 2)
#             m = .~isfinite.(s.aux["templates"][:, s])
#             s.aux["templates"][mask_ir.|m, s] .= Spline1D(ustrip.(s.λ[.~mask_ir .& .~m]), 
#                 ustrip.(s.aux["templates"][.~mask_ir .& .~m, s]), ustrip.(λknots), k=1, 
#                 bc="extrapolate")(ustrip.(s.λ[mask_ir .| m]))
#         end
#     end
# end


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
    get_vector_mask(s, mask_bad[, user_mask])

Apply two masks (mask_bad and user_mask) to a set of vectors (λ, I, σ, templates, channel_masks) to prepare
them for fitting. The mask_bad is a pixel-by-pixel mask flagging bad data, while the user_mask is a set of pairs
of wavelengths specifying entire regions to mask out.  The vectors are modified in-place.
"""
function get_vector_mask(s::Spaxel; lines::Bool=true, user_mask::Union{Nothing,Vector{<:Tuple}}=nothing) 
    # bad pixels
    mask_all = copy(s.mask_bad)
    # optical emission lines 
    if lines
        mask_all .|= s.mask_lines
    end
    # user-specified regions
    if !isnothing(user_mask)
        for pair in user_mask
            mask_all .|= pair[1] .< s.λ .< pair[2]
        end
    end
    mask_all
end


function continuum_cubic_spline!(s::Spaxel, overrides)
    mask_lines, I_spline = continuum_cubic_spline(s.λ, s.I, s.σ, overrides; do_err=false)
    s.I_spline = I_spline
    s.mask_lines = mask_lines
end


function calculate_statistical_errors!(s::Spaxel)
    @assert haskey(s.aux, "mask_lines") "Need to have a line mask first! Run continuum_cubic_spline! to generate it."
    calculate_statistical_errors(s.I, s.I_spline, s.aux["mask_lines"])
end


function make_normalized_spaxel(cube_data::NamedTuple, coords::CartesianIndex, cube_fitter::CubeFitter;
    use_ap::Bool=false, use_vorbins::Bool=false, σ_min::Quantity=0.0*unit(cube_data.σ[1]))

    fopt = fit_options(cube_fitter)

    # Gather the data from the appropriate spaxel
    λ = cube_data.λ
    I = cube_data.I[coords, :]
    σ = cube_data.σ[coords, :]
    area_sr = cube_data.area_sr[coords, :]
    templates = cube_data.templates[coords, :, :]

    # Perform a cubic spline fit, also obtaining the line mask
    mask_lines, I_spline = continuum_cubic_spline(λ, I, σ, cube_fitter.linemask_overrides; do_err=false)
    # Bad pixel mask
    mask_bad = (use_ap || use_vorbins) ? iszero.(I) .| iszero.(σ) : cube_fitter.cube.mask[coords, :]
    mask = mask_lines .| mask_bad
    # Calculate statistical errors if necessary
    if use_ap || use_vorbins
        σ .= calculate_statistical_errors(I, I_spline, mask)
        # Add in quadrature with minimum uncertainty
        σ .= sqrt.(σ.^2 .+ σ_min.^2)
    end
    # Add systematic error in quadrature
    σ .= sqrt.(σ.^2 .+ (fopt.sys_err .* I).^2)

    # Use a fixed normalization for the line fits so that the bootstrapped amplitudes are consistent with each other
    norm = Float64(abs(nanmaximum(I)))
    norm = norm ≠ 0. ? norm : 1.

    vres = isapprox((λ[2]/λ[1]), (λ[end]/λ[end-1]), rtol=1e-6) ? log(λ[2]/λ[1]) * C_KMS : nothing

    # Interpolate over the bad pixels in the templates 
    λknots = λ[(1+scale):scale:(length(λ)-scale)]
    for s in axes(templates, 2)
        m = .~isfinite.(templates[:, s])
        templates[m, s] .= Spline1D(ustrip.(λ[.~m]), ustrip.(templates[.~m, s]), ustrip.(λknots), k=1, 
            bc="extrapolate")(ustrip.(s.λ[m]))
    end

    # Create the spaxel object and normalize it
    aux = Dict("templates" => templates, "channel_masks" => cube_fitter.spectral_region.channel_masks)
    spax = Spaxel(coords, λ, I, σ, area_sr, mask_lines, mask_bad, vres, I_spline; aux=aux)
    normalize!(spax, norm)
    fill_bad_pixels!(spax)

    spax
end