module Util

using NaNStatistics

# CONSTANTS

const C_MS = 299792458
const C_KMS = C_MS / 1000

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

# Log of the likelihood for a given model
function ln_likelihood(data::Vector{Float64}, model::Vector{Float64}, err::Vector{Float64})
    return -0.5 * sum((data .- model).^2 ./ err.^2 .+ log.(2π .* err.^2))
end

# LINE PROFILES

function Gaussian(x::AbstractVector{<:Real}, params::AbstractVector{<:Real})
    """
    Gaussian profile parameterized in terms of the FWHM
    """
    A, μ, FWHM = params
    return @. A * exp(-(x-μ)^2 / (2(FWHM/(2√(2log(2))))^2))
end

end