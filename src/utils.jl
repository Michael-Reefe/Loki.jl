module Util

using NaNStatistics
using Interpolations

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
function ln_likelihood(data::Vector{T}, model::AbstractVector{V}, err::Vector{T}) where {T<:Real,V<:Real}
    return -0.5 * sum((data .- model).^2 ./ err.^2 .+ log.(2π .* err.^2))
end

# BLACKBODY PROFILE

function Blackbody_ν(λ::T, Temp::V) where {T<:Real,V<:Real}
    """
    Return the Blackbody function Bν (per unit FREQUENCY) in MJy/sr,
    given a wavelength in μm and a temperature in Kelvins.
    """
    return Bν_1/λ^3 / (exp(Bν_2/(λ*Temp))-1)
end

# PAH PROFILES

function Drude(x::T, A::V, μ::V, FWHM::V) where {T<:Real,V<:Real}
    """
    Calculate a Drude profile
    """
    return A * (FWHM/μ)^2 / ((x/μ - μ/x)^2 + (FWHM/μ)^2)
end

# EXTINCTION

function τ_kvt(λ::T, β::V) where {T<:Real,V<:Real}
    """
    Calculate extinction curve
    """
    mx, mn = argmax(kvt_prof[:, 1]), argmin(kvt_prof[:, 1])
    λ_mx, λ_mn = kvt_prof[mx, 1], kvt_prof[mn, 1]

    if λ ≤ λ_mn
        ext = kvt_prof[mn, 2] * exp(2.03 * (λ - λ_mn))
    elseif λ_mn < λ < λ_mx
        ext = linear_interpolation(kvt_prof[:, 1], kvt_prof[:, 2])(λ)
    elseif λ_mx < λ < λ_mx + 2
        ext = cubic_spline_interpolation(8.0:0.2:12.6, [kvt_prof[1:9, 2]; kvt_prof[12:26, 2]], extrapolation_bc=Line())(λ)
    else
        ext = 0.
    end
    ext = ext < 0 ? 0. : ext
    ext += Drude(λ, 0.4, 18., 4.446)

    return (1 - β) * ext + β * (9.7/λ)^1.7
end

function Extinction(ext::T, τ_97::V) where {T<:Real,V<:Real}
    """
    Calculate the overall extinction factor
    """
    return iszero(τ_97) ? 1. : (1 - exp(-τ_97*ext)) / (τ_97*ext)
end

function Continuum(λ::Vector{T}, params::AbstractVector{V}, n_dust_cont::Int, n_dust_features::Int; 
    return_components::Bool=false) where {T<:Real,V<:Real}

    # Adapted from PAHFIT (IDL)

    comps = Dict{String, Vector{V}}()
    contin = zeros(V, length(λ))

    # Stellar blackbody continuum (usually at 5000 K)
    comps["stellar"] = params[1] .* Blackbody_ν.(λ, params[2])
    contin .+= comps["stellar"]
    pᵢ = 3

    # Add dust continua at various temperatures
    for i ∈ 1:n_dust_cont
        comps["dust_cont_$i"] = params[pᵢ] .* (9.7 ./ λ).^2 .* Blackbody_ν.(λ, params[pᵢ+1])
        contin .+= comps["dust_cont_$i"] 
        pᵢ += 2
    end

    # Add dust features with drude profiles
    for i ∈ 1:n_dust_features
        comps["dust_feat_$i"] = Drude.(λ, params[pᵢ:pᵢ+2]...)
        contin .+= comps["dust_feat_$i"]
        pᵢ += 3
    end

    # Extinction 
    ext_curve = τ_kvt.(λ, params[pᵢ+1])
    comps["extinction"] = Extinction.(ext_curve, params[pᵢ])
    contin .*= comps["extinction"]
    
    if return_components
        return contin, comps
    end
    return contin

end

# LINE PROFILES

function Gaussian(x::Vector{T}, params::AbstractVector{T}) where {T<:Real}
    """
    Gaussian profile parameterized in terms of the FWHM
    """
    A, μ, FWHM = params
    return @. A * exp(-(x-μ)^2 / (2(FWHM/(2√(2log(2))))^2))
end

end