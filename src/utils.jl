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
function ln_likelihood(data::Vector{Float64}, model::Vector{Float64}, err::Vector{Float64})
    return -0.5 * sum((data .- model).^2 ./ err.^2 .+ log.(2π .* err.^2))
end

# Hermite polynomial of order n computed using the recurrence relations
# (gets slow around/above order ~30)
function hermite(x::Float64, n::Int)
    if iszero(n)
        return 1.
    elseif isone(n)
        return 2x
    else
        return 2x * hermite(x, n-1) - 2(n-1) * hermite(x, n-2)
    end
end


# BLACKBODY PROFILE

function Blackbody_ν(λ::Float64, Temp::Float64) 
    """
    Return the Blackbody function Bν (per unit FREQUENCY) in MJy/sr,
    given a wavelength in μm and a temperature in Kelvins.
    """
    return Bν_1/λ^3 / (exp(Bν_2/(λ*Temp))-1)
end

# PAH PROFILES

function Drude(x::Float64, A::Float64, μ::Float64, FWHM::Float64)
    """
    Calculate a Drude profile
    """
    return A * (FWHM/μ)^2 / ((x/μ - μ/x)^2 + (FWHM/μ)^2)
end

# EXTINCTION

function τ_kvt(λ::Float64, β::Float64)
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

function Extinction(ext::Float64, τ_97::Float64, screen::Bool=true)
    """
    Calculate the overall extinction factor
    """
    if screen
        return exp(-τ_97*ext)
    end
    return iszero(τ_97) ? 1. : (1 - exp(-τ_97*ext)) / (τ_97*ext)
end


function fit_spectrum(λ::Vector{Float64}, params::Vector{Float64}, n_dust_cont::Int64, n_dust_features::Int64; 
    return_components::Bool=false, verbose::Bool=false)

    # Adapted from PAHFIT (IDL)

    comps = Dict{String, Vector{Float64}}()
    contin = zeros(Float64, length(λ))

    # Stellar blackbody continuum (usually at 5000 K)
    if verbose
        println("Stellar continuum:")
        println("$(params[1]), $(params[2])")
    end
    comps["stellar"] = params[1] .* Blackbody_ν.(λ, params[2])
    contin .+= comps["stellar"]
    pᵢ = 3

    # Add dust continua at various temperatures
    if verbose
        println("Dust continua:")
    end
    for i ∈ 1:n_dust_cont
        if verbose
            println("$(params[pᵢ]), $(params[pᵢ+1])")
        end
        comps["dust_cont_$i"] = params[pᵢ] .* (9.7 ./ λ).^2 .* Blackbody_ν.(λ, params[pᵢ+1])
        contin .+= comps["dust_cont_$i"] 
        pᵢ += 2
    end

    # Add dust features with drude profiles
    if verbose
        println("Dust features:")
    end
    for j ∈ 1:n_dust_features
        if verbose
            println("$(params[pᵢ]), $(params[pᵢ+1]), $(params[pᵢ+2])")
        end
        comps["dust_feat_$j"] = Drude.(λ, params[pᵢ:pᵢ+2]...)
        contin .+= comps["dust_feat_$j"]
        pᵢ += 3
    end

    # Extinction 
    if verbose
        println("Extinction:")
        println("$(params[pᵢ]), $(params[pᵢ+1])")
    end
    ext_curve = τ_kvt.(λ, params[pᵢ+1])
    comps["extinction"] = Extinction.(ext_curve, params[pᵢ])
    contin .*= comps["extinction"]
    pᵢ += 2

    if return_components
        return contin, comps
    end
    return contin

end

function fit_line_residuals(λ::Vector{Float64}, params::Vector{Float64}, n_lines::Int64, n_voff_tied::Int64, 
    voff_tied_key::Vector{String}, line_tied::Vector{Union{String,Nothing}}, line_profiles::Vector{Symbol}, 
    line_restwave::Vector{Float64}, flexible_wavesol::Bool; return_components::Bool=false, verbose::Bool=false)

    comps = Dict{String, Vector{Float64}}()
    contin = zeros(Float64, length(λ))
    pᵢ = n_voff_tied + 1

    # Add emission lines
    for k ∈ 1:n_lines
        # Check if voff is tied: if so, use the tied voff parameter, otherwise, use the line's own voff parameter
        if verbose
            println("Line at $(line_restwave[k]):")
        end
        if isnothing(line_tied[k])
            voff = params[pᵢ+1]
            fwhm = params[pᵢ+2]
            msg = "$(params[pᵢ]), $(params[pᵢ+1]) (united), $(params[pᵢ+2])"
            if line_profiles[k] == :GaussHermite
                h3 = params[pᵢ+3]
                h4 = params[pᵢ+4]
                msg *= ", $(params[pᵢ+3]), $(params[pᵢ+4])"
            end
            if verbose
                println(msg)
            end
        elseif !isnothing(line_tied[k]) && flexible_wavesol
            vwhere = findfirst(x -> x == line_tied[k], voff_tied_key)
            voff_series = params[vwhere]
            voff_indiv = params[pᵢ+1]
            # Add velocity shifts of the tied lines and the individual offsets
            voff = voff_series + voff_indiv
            fwhm = params[pᵢ+2]
            mgs = "$(params[pᵢ]), $(params[vwhere]) (tied) + $(params[pᵢ+1]) (united), $(params[pᵢ+2])"
            if line_profiles[k] == :GaussHermite
                h3 = params[pᵢ+3]
                h4 = params[pᵢ+4]
                msg *= ", $(params[pᵢ+3]), $(params[pᵢ+4])"
            end
            if verbose
                println(msg)
            end
        else
            vwhere = findfirst(x -> x == line_tied[k], voff_tied_key)
            voff = params[vwhere]
            fwhm = params[pᵢ+1]
            msg = "$(params[pᵢ]), $(params[vwhere]) (tied), $(params[pᵢ+1])"
            if line_profiles[k] == :GaussHermite
                h3 = params[pᵢ+2]
                h4 = params[pᵢ+3]
                msg *= ", $(params[pᵢ+2]), $(params[pᵢ+3])"
            end
            if verbose
                println(msg)
            end
        end
        # Convert voff in km/s to mean wavelength in μm
        mean_μm = Doppler_shift_λ(line_restwave[k], voff)
        # Convert FWHM from km/s to μm
        fwhm_μm = Doppler_shift_λ(line_restwave[k], fwhm) - line_restwave[k]
        # Evaluate line profile
        if line_profiles[k] == :Gaussian
            comps["line_$k"] = Gaussian(λ, params[pᵢ], mean_μm, fwhm_μm)
        elseif line_profiles[k] == :GaussHermite
            comps["line_$k"] = GaussHermite(λ, params[pᵢ], mean_μm, fwhm_μm, [h3, h4])
        else
            error("Unrecognized line profile $(line_profiles[k])!")
        end

        contin .+= comps["line_$k"]        
        pᵢ += isnothing(line_tied[k]) || flexible_wavesol ? 3 : 2
        if line_profiles[k] == :GaussHermite
            pᵢ += 2
        end

    end

    if return_components
        return contin, comps
    end
    return contin

end


# LINE PROFILES

function Gaussian(x::Vector{Float64}, A::Float64, μ::Float64, FWHM::Float64)
    """
    Gaussian profile parameterized in terms of the FWHM
    """
    # Reparametrize FWHM as dispersion σ
    σ = FWHM / (2√(2log(2)))
    return @. A * exp(-(x-μ)^2 / (2σ^2))
end

function GaussHermite(x::Vector{Float64}, A::Float64, μ::Float64, FWHM::Float64, h::Vector{Float64})
    """
    Gauss-Hermite line profiles 
    (see Riffel et al. 2010)
    """
    # Reparametrize FWHM as dispersion σ
    σ = FWHM / (2√(2log(2)))
    # Gaussian exponential argument w
    w = @. (x - μ) / σ
    # Normalized Gaussian
    α = @. 1/√(2π * σ^2) * exp(-w^2 / 2)

    # Calculate coefficients for the Hermite basis
    n = 3:(length(h)+2)
    norm = .√(factorial.(n) .* 2 .^ n)
    coeff = vcat([1, 0, 0], h./norm)
    # Calculate hermite basis
    Herm = hcat((coeff[nᵢ] .* hermite.(w, nᵢ-1) for nᵢ ∈ 1:length(coeff))...)
    # Collapse sum along the moment axis
    Herm = dropdims(sum(Herm, dims=2), dims=2)

    # Combine the Gaussian and Hermite profiles
    GH = α .* Herm
    # Renormalize
    GH ./= maximum(GH)
    GH .*= A

    return GH
end

end