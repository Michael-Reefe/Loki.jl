module Util

using NaNStatistics
using Dierckx

# CONSTANTS

const C_MS = 299792458.           # Speed of light in m/s
const C_KMS = C_MS / 1000         # Speed of light in km/s
const h_ERGS = 6.62607015e-27     # Planck constant in erg*s
const kB_ERGK = 1.380649e-16      # Boltzmann constant in erg/K

const Bν_1 = 3.97289e13           # First constant for Planck function, in MJy/sr/μm
const Bν_2 = 1.4387752e4          # Second constant for Planck function, in μm*K

const kvt_prof = [8.0  0.06;
                  8.2  0.09;
                  8.4  0.16;
                  8.6  0.275;
                  8.8  0.415;
                  9.0  0.575;
                  9.2  0.755;
                  9.4  0.895;
                  9.6  0.98;
                  9.7  0.99;
                  9.75 1.00;
                  9.8  0.99;
                  10.0 0.94;
                  10.2 0.83;
                  10.4 0.745;
                  10.6 0.655;
                  10.8 0.58;
                  11.0 0.525;
                  11.2 0.43;
                  11.4 0.35;
                  11.6 0.27;
                  11.8 0.20;
                  12.0 0.13;
                  12.2 0.09;
                  12.4 0.06;
                  12.6 0.045;
                  12.7 0.04314]

# UTILITY FUNCTIONS

# Sum along specific dimensions, ignoring nans, and dropping those dimensions
Σ(array, dims) = dropdims(nansum(array, dims=dims), dims=dims)

# Extend a 1D array into other dimensions
extend(array1d, shape) = repeat(reshape(array1d, (1,1,length(array1d))), outer=[shape...,1])

# Convert wavelength to rest-frame
rest_frame(λ, z) = @. λ / (1 + z)

# Relativistic doppler shift
Doppler_shift_λ(λ₀, v) = λ₀ * √((1+v/C_KMS)/(1-v/C_KMS))
Doppler_shift_v(λ, λ₀) = ((λ/λ₀)^2 - 1)/((λ/λ₀)^2 + 1) * C_KMS

# Doppler shift approximation for v << c
Doppler_width_v(Δλ, λ₀) = Δλ / λ₀ * C_KMS
Doppler_width_λ(Δv, λ₀) = Δv / C_KMS * λ₀

# Integral of a Gaussian
∫Gaussian(A, σ) = √(2π) * A * σ

# Convert units
to_cgs(F, λ) = F .* 1e-7 .* Util.C_MS ./ λ.^2    # λ in angstroms, F (MJy/sr) -> (erg/s/cm^2/A/sr)
to_MJy_sr(F, λ) = F .* 1e7 ./ Util.C_MS .* λ.^2  # λ in angstroms, F (erg/s/cm^2/A/sr) -> (MJy/sr)

# Log of the likelihood for a given model
function ln_likelihood(data::Vector{T}, model::Vector{T}, err::Vector{T}) where {T<:Real}
    return -0.5 * sum((data .- model).^2 ./ err.^2 .+ log.(2π .* err.^2))
end

# BLACKBODY PROFILE

function Blackbody_ν(λ::Union{T,Vector{T}}, Temp::T) where {T<:Real}
    """
    Return the Blackbody function Bν (per unit FREQUENCY) in MJy/sr,
    given a wavelength in μm and a temperature in Kelvins.
    """
    return @. Bν_1/λ^3 / (exp(Bν_2/(λ*Temp))-1)
end

# PAH PROFILES

function Drude(x::Union{T,Vector{T}}, params::AbstractVector{T}) where {T<:Real}
    """
    Calculate a Drude profile
    """
    A, μ, FWHM = params
    return @. A * (FWHM/μ)^2 / ((x/μ - μ/x)^2 + (FWHM/μ)^2)
end

# EXTINCTION

function τ_kvt(λ::Vector{T}, β::T) where {T<:Real}
    """
    Calculate extinction curve
    """
    ext = zeros(length(λ))

    mx, mn = argmax(kvt_prof[:, 1]), argmin(kvt_prof[:, 1])
    λ_mx, λ_mn = kvt_prof[mx, 1], kvt_prof[mn, 1]

    low = findall(x -> x < λ_mn, λ)
    if length(low) > 0
        ext[low] .= kvt_prof[mn, 2] .* exp.(2.03 .* (λ[low] .- λ_mn))
    end

    on_profile = findall(x -> (λ_mn < x < λ_mx), λ)
    if length(on_profile) > 0
        ext[on_profile] .= Spline1D(kvt_prof[:, 1], kvt_prof[:, 2], k=1, bc="extrapolate").(λ[on_profile])
    end

    fit = findall(x -> ((λ_mx - 2.5) ≤ x ≤ λ_mx), λ)
    anchor = findall(x -> ((λ_mx + 2) ≤ x ≤ (λ_mx + 3)), λ)
    fit = [fit; anchor]

    extend = findall(x -> (λ_mx < x < (λ_mx + 2)), λ)
    if length(extend) > 0
        ext[extend] .= Spline1D(λ[fit], ext[fit], k=2, bc="extrapolate").(λ[extend])
    end

    ext[ext .< 0] .= 0.
    ext .+= Drude(λ, [0.4, 18., 4.446])

    ext = @. (1 - β) * ext + β * (9.7/λ)^1.7
    return ext
end

function Extinction(τ_97::T, ext::Vector{T}) where {T<:Real}
    """
    Calculate the overall extinction factor
    """
    return iszero(τ_97) ? ones(length(ext)) : (1 .- exp.(-τ_97.*ext)) ./ (τ_97.*ext)
    # return exp.(-τ_97.*ext)
end

function Continuum(λ::Vector{T}, params::AbstractVector{T},
    n_dust_cont::Int, n_dust_features::Int; return_components::Bool=false) where {T<:Real}

    # Adapted from PAHFIT (IDL)

    comps = Dict{String, Vector{Float64}}()
    contin = zeros(length(λ))

    # Stellar blackbody continuum (usually at 5000 K)
    comps["stellar"] = params[1] .* Blackbody_ν(λ, params[2])
    contin .+= comps["stellar"]
    pᵢ = 3

    # Add dust continua at various temperatures
    for i ∈ 1:n_dust_cont
        comps["dust_cont_$i"] = params[pᵢ] .* (9.7 ./ λ).^2 .* Blackbody_ν(λ, params[pᵢ+1])
        contin .+= comps["dust_cont_$i"] 
        pᵢ += 2
    end

    # Add dust features with drude profiles
    for i ∈ 1:n_dust_features
        comps["dust_feat_$i"] = Drude(λ, params[pᵢ:pᵢ+2])
        contin .+= comps["dust_feat_$i"]
        pᵢ += 3
    end

    # Extinction 
    ext_curve = τ_kvt(λ, params[pᵢ+1])
    comps["extinction"] = Extinction(params[pᵢ], ext_curve)
    contin .*= comps["extinction"]
    
    if return_components
        return contin, comps
    end
    return contin

end

# LINE PROFILES

function Gaussian(x::AbstractVector{T}, params::AbstractVector{U}) where {T<:Real,U<:Real}
    """
    Gaussian profile parameterized in terms of the FWHM
    """
    A, μ, FWHM = params
    return @. A * exp(-(x-μ)^2 / (2(FWHM/(2√(2log(2))))^2))
end

end