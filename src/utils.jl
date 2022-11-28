module Util

using NaNStatistics
using Interpolations
using Dierckx
using CSV
using DataFrames

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


function read_irs_data(path::String)
    datatable = CSV.read(path, DataFrame, comment="#", delim=' ', ignorerepeated=true, stripwhitespace=true,
        header=["rest_wave", "flux", "e_flux", "enod", "order", "module", "nod1flux", "nod2flux", "e_nod1flux", "e_nod2flux"])
    return datatable[!, "rest_wave"], datatable[!, "flux"], datatable[!, "e_flux"]
end

function silicate_dp()
    # Read in IRS 08572+3915 data from 00000003_0.ideos.mrt
    λ_irs, F_irs, σ_irs = read_irs_data(joinpath(@__DIR__, "00000003_0.ideos.mrt"))
    # Get flux values at anchor points + endpoints
    anchors = [4.9, 5.5, 7.8, 13.0, 14.5, 26.5, λ_irs[end]]
    values = zeros(length(anchors))
    for (i, anchor) ∈ enumerate(anchors)
        _, ki = findmin(k -> abs(k - anchor), λ_irs)
        values[i] = F_irs[ki]
    end

    # Cubic spline fit with specific anchor points
    cubic_spline_irs = Spline1D(anchors, values; k=3)

    # Get optical depth
    τ_DS = log.(cubic_spline_irs.(λ_irs) ./ F_irs)
    # Smooth data and remove features < ~7.5 um
    τ_smooth = movmean(τ_DS, 10)
    v1, p1 = findmin(τ_DS[λ_irs .< 6])
    v2, p2 = findmin(τ_DS[7 .< λ_irs .< 8])
    slope_beg = (v2 - v1) / (λ_irs[7 .< λ_irs .< 8][p2] - λ_irs[λ_irs .< 6][p1])
    beg_filt = λ_irs .< λ_irs[7 .< λ_irs .< 8][p2]
    τ_smooth[beg_filt] .= v1 .+ slope_beg .* (λ_irs[beg_filt] .- λ_irs[1])

    # Normalize to value at 9.7
    τ_97 = τ_smooth[findmin(abs.(λ_irs .- 9.7))[2]]
    τ_λ = τ_smooth ./ τ_97

    # Return the cubic spline interpolator function
    return λ_irs, τ_λ
end

const DPlus_prof = silicate_dp()


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
function ln_likelihood(data::Union{Vector{Float32},Vector{Float64}}, model::Union{Vector{Float32},Vector{Float64}}, 
    err::Union{Vector{Float32},Vector{Float64}})
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

function τ_dp(λ::Float64, β::Float64)
    """
    Calculate Donnan et al. extinction curve
    """
    # Simple cubic spline interpolation
    ext = Spline1D(DPlus_prof[1], DPlus_prof[2]; k=3).(λ)

    # Add 1.7 power law, as in PAHFIT
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


function fit_spectrum(λ::Vector{Float64}, params::Vector{Float64}, n_dust_cont::Int64, n_dust_features::Int64,
    extinction_curve::String; return_components::Bool=false, verbose::Bool=false)

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
    if extinction_curve == "d+"
        ext_curve = τ_dp.(λ, params[pᵢ+1])
    elseif extinction_curve == "kvt"
        ext_curve = τ_kvt.(λ, params[pᵢ+1])
    else
        error("Unrecognized extinction curve: $extinction_curve")
    end
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
    n_flow_voff_tied::Int64, flow_voff_tied_key::Vector{String}, line_flow_tied::Vector{Union{String,Nothing}},
    line_flow_profiles::Vector{Union{Symbol,Nothing}}, line_restwave::Vector{Float64}, 
    flexible_wavesol::Bool, tie_voigt_mixing::Bool; return_components::Bool=false, verbose::Bool=false)

    comps = Dict{String, Vector{Float64}}()
    contin = zeros(Float64, length(λ))
    pᵢ = n_voff_tied + n_flow_voff_tied + 1
    if tie_voigt_mixing
        ηᵢ = pᵢ
        pᵢ += 1
    end

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
            elseif line_profiles[k] == :Voigt
                if !tie_voigt_mixing
                    η = params[pᵢ+3]
                else
                    η = params[ηᵢ]
                end
                msg *= ", $η"
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
            msg = "$(params[pᵢ]), $(params[vwhere]) (tied) + $(params[pᵢ+1]) (united), $(params[pᵢ+2])"
            if line_profiles[k] == :GaussHermite
                h3 = params[pᵢ+3]
                h4 = params[pᵢ+4]
                msg *= ", $(params[pᵢ+3]), $(params[pᵢ+4])"
            elseif line_profiles[k] == :Voigt
                if !tie_voigt_mixing
                    η = params[pᵢ+3]
                else
                    η = params[ηᵢ]
                end
                msg *= ", $η"
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
            elseif line_profiles[k] == :Voigt
                if !tie_voigt_mixing
                    η = params[pᵢ+2]
                else
                    η = params[ηᵢ]
                end
                msg *= ", $η"
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
            comps["line_$k"] = Gaussian.(λ, params[pᵢ], mean_μm, fwhm_μm)
        elseif line_profiles[k] == :GaussHermite
            comps["line_$k"] = GaussHermite.(λ, params[pᵢ], mean_μm, fwhm_μm, h3, h4)
        elseif line_profiles[k] == :Voigt
            comps["line_$k"] = Voigt.(λ, params[pᵢ], mean_μm, fwhm_μm, η)
        else
            error("Unrecognized line profile $(line_profiles[k])!")
        end

        contin .+= comps["line_$k"]        
        pᵢ += isnothing(line_tied[k]) || flexible_wavesol ? 3 : 2
        if line_profiles[k] == :GaussHermite
            pᵢ += 2
        elseif line_profiles[k] == :Voigt
            if !tie_voigt_mixing
                pᵢ += 1
            end
        end

        if !isnothing(line_flow_profiles[k])
            if verbose
                println("Flow component:")
            end
            if isnothing(line_flow_tied[k])
                flow_voff = params[pᵢ+1]
                flow_fwhm = params[pᵢ+2]
                msg = "$(params[pᵢ]), $(params[pᵢ+1]) (united), $(params[pᵢ+2])"
                if line_flow_profiles[k] == :GaussHermite
                    flow_h3 = params[pᵢ+3]
                    flow_h4 = params[pᵢ+4]
                    msg *= ", $(params[pᵢ+3]), $(params[pᵢ+4])"
                elseif line_flow_profiles[k] == :Voigt
                    if !tie_voigt_mixing
                        flow_η = params[pᵢ+3]
                    else
                        flow_η = params[ηᵢ]
                    end
                    msg *= ", $flow_η"
                end
                if verbose
                    println(msg)
                end 
            else
                vwhere = findfirst(x -> x == line_flow_tied[k], flow_voff_tied_key)
                flow_voff = params[n_voff_tied + vwhere]
                flow_fwhm = params[pᵢ+1]
                msg = "$(params[pᵢ]), $(params[vwhere]) (tied), $(params[pᵢ+1])"
                if line_flow_profiles[k] == :GaussHermite
                    flow_h3 = params[pᵢ+2]
                    flow_h4 = params[pᵢ+3]
                    msg *= ", $(params[pᵢ+2]), $(params[pᵢ+3])"
                elseif line_flow_profiles[k] == :Voigt
                    if !tie_voigt_mixing
                        flow_η = params[pᵢ+2]
                    else
                        flow_η = params[ηᵢ]
                    end
                    msg *= ", $flow_η"
                end
                if verbose
                    println(msg)
                end
            end
            # Convert voff in km/s to mean wavelength in μm
            flow_mean_μm = Doppler_shift_λ(line_restwave[k], voff+flow_voff)
            # Convert FWHM from km/s to μm
            flow_fwhm_μm = Doppler_shift_λ(line_restwave[k], flow_fwhm) - line_restwave[k]
            # Evaluate line profile
            if line_flow_profiles[k] == :Gaussian
                comps["line_$(k)_flow"] = Gaussian.(λ, params[pᵢ], flow_mean_μm, flow_fwhm_μm)
            elseif line_profiles[k] == :GaussHermite
                comps["line_$(k)_flow"] = GaussHermite.(λ, params[pᵢ], flow_mean_μm, flow_fwhm_μm, flow_h3, flow_h4)
            elseif line_profiles[k] == :Voigt
                comps["line_$(k)_flow"] = Voigt.(λ, params[pᵢ], flow_mean_μm, flow_fwhm_μm, flow_η)
            else
                error("Unrecognized flow line profile $(line_profiles[k])!")
            end

            contin .+= comps["line_$(k)_flow"]        
            pᵢ += isnothing(line_flow_tied[k]) ? 3 : 2
            if line_flow_profiles[k] == :GaussHermite
                pᵢ += 2
            elseif line_flow_profiles[k] == :Voigt
                if !tie_voigt_mixing
                    pᵢ += 1
                end
            end
        end

    end

    if return_components
        return contin, comps
    end
    return contin

end


# LINE PROFILES

function Gaussian(x::Float64, A::Float64, μ::Float64, FWHM::Float64)
    """
    Gaussian profile parameterized in terms of the FWHM
    """
    # Reparametrize FWHM as dispersion σ
    σ = FWHM / (2√(2log(2)))
    return A * exp(-(x-μ)^2 / (2σ^2))
end

function GaussHermite(x::Float64, A::Float64, μ::Float64, FWHM::Float64, h₃::Float64, h₄::Float64)
    """
    Gauss-Hermite line profiles 
    (see Riffel et al. 2010)
    """
    h = [h₃, h₄]
    # Reparametrize FWHM as dispersion σ
    σ = FWHM / (2√(2log(2)))
    # Gaussian exponential argument w
    w = (x - μ) / σ
    # Normalized Gaussian
    α = exp(-w^2 / 2)

    # Calculate coefficients for the Hermite basis
    n = 3:(length(h)+2)
    norm = .√(factorial.(n) .* 2 .^ n)
    coeff = vcat([1, 0, 0], h./norm)
    # Calculate hermite basis
    Herm = sum([coeff[nᵢ] * hermite(w, nᵢ-1) for nᵢ ∈ 1:length(coeff)])

    # Calculate peak height (i.e. value of function at w=0)
    Herm0 = sum([coeff[nᵢ] * hermite(0., nᵢ-1) for nᵢ ∈ 1:length(coeff)])

    # Combine the Gaussian and Hermite profiles
    GH = A * α * Herm / Herm0

    return GH
end

function Lorentzian(x::Float64, A::Float64, μ::Float64, FWHM::Float64)
    return A/π * (FWHM/2) / ((x-μ)^2 + (FWHM/2)^2)
end

function Voigt(x::Float64, A::Float64, μ::Float64, FWHM::Float64, η::Float64)

    # Reparametrize FWHM as dispersion σ
    σ = FWHM / (2√(2log(2))) 
    # Normalized Gaussian
    G = 1/√(2π * σ^2) * exp(-(x-μ)^2 / (2σ^2))
    # Normalized Lorentzian
    L = 1/π * (FWHM/2) / ((x-μ)^2 + (FWHM/2)^2)

    # Intensity parameter given the peak height A
    I = A * FWHM * π / (2 * (1 + (√(π*log(2)) - 1)*η))

    # Mix the two distributions with the mixing parameter η
    pV = I * (η * G + (1 - η) * L)

    return pV
end

end