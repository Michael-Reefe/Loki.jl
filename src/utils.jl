#=
THE UTIL MODULE
---------------

This module is not intended to be directly accessed by the user when fitting
IFU cubes. Rather, it contains miscellaneous utility functions to aid in correcting
and fitting data.  As such, nothing in this module is exported, although the module
itself IS exported within the CubeData, CubeFit, and Loki modules, so it may be
accessed with the "Util" prefix.
=#

module Util

# Import math packages
using Statistics
using NaNStatistics
using Dierckx
using CSV
using DataFrames
using QuadGK

# CONSTANTS

const C_KMS::Float64 = 299792.458          # Speed of light in km/s
const h_ERGS::Float64 = 6.62607015e-27     # Planck constant in erg*s
const kB_ERGK::Float64 = 1.380649e-16      # Boltzmann constant in erg/K

const Bν_1::Float64 = 3.97289e13           # First constant for Planck function, in MJy/sr/μm
const Bν_2::Float64 = 1.4387752e4          # Second constant for Planck function, in μm*K

const b_Wein::Float64 = 2897.771955        # Wein's law constant of proportionality in μm*K

# Saved Kemper, Vriend, & Tielens (2004) extinction profile
const kvt_prof::Matrix{Float64} =  [8.0  0.06;
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


"""
    read_irs_data(path)

Setup function for reading in the configuration IRS spectrum of IRS 08572+3915

# Arguments
- `path::String`: The file path pointing to the IRS 08572+3915 spectrum 
"""
function read_irs_data(path::String)

    @debug "Reading in IRS data from: $path"

    datatable = CSV.read(path, DataFrame, comment="#", delim=' ', ignorerepeated=true, stripwhitespace=true,
        header=["rest_wave", "flux", "e_flux", "enod", "order", "module", "nod1flux", "nod2flux", "e_nod1flux", "e_nod2flux"])

    datatable[!, "rest_wave"], datatable[!, "flux"], datatable[!, "e_flux"]
end


"""
    silicate_dp()

Setup function for creating a silicate extinction profile based on Donnan et al. (2022)
"""
function silicate_dp()

    @debug "Creating Donnan+2022 optical depth profile..."

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
    τ_DS = log10.(cubic_spline_irs.(λ_irs) ./ F_irs)
    # Smooth data and remove features < ~7.5 um
    τ_smooth = movmean(τ_DS, 10)
    v1, p1 = findmin(τ_DS[λ_irs .< 6])
    v2, p2 = findmin(τ_DS[7 .< λ_irs .< 8])
    slope_beg = (v2 - v1) / (λ_irs[7 .< λ_irs .< 8][p2] - λ_irs[λ_irs .< 6][p1])
    beg_filt = λ_irs .< λ_irs[7 .< λ_irs .< 8][p2]
    τ_smooth[beg_filt] .= v1 .+ slope_beg .* (λ_irs[beg_filt] .- λ_irs[1])
    mid_filt = (λ_irs[7 .< λ_irs .< 8][p2] .- 0.5) .< λ_irs .< (λ_irs[7 .< λ_irs .< 8][p2] .+ 0.5)  
    τ_smooth[mid_filt] = movmean(τ_smooth, 5)[mid_filt]

    # Normalize to value at 9.7
    τ_98 = τ_smooth[findmin(abs.(λ_irs .- 9.8))[2]]
    τ_λ = τ_smooth ./ τ_98

    λ_irs, τ_λ
end

# Save the Donnan et al. 2022 profile as a constant
const DPlus_prof = silicate_dp()
const DP_interp = Spline1D(DPlus_prof[1], DPlus_prof[2]; k=3)

# Setup function for creating the extinction profile from Chiar+Tielens 2006
function silicate_ct()
    data = CSV.read(joinpath(@__DIR__, "chiar+tielens_2005.dat"), DataFrame, skipto=15, delim=' ', 
        ignorerepeated=true, header=["wave", "a_galcen", "a_local"])
    data[!, "wave"], data[!, "a_galcen"]
end

# Save the Chiar+Tielens 2005 profile as a constant
const CT_prof = silicate_ct()
const CT_interp = Spline1D(CT_prof[1], CT_prof[2]; k=3)

########################################### UTILITY FUNCTIONS ###############################################

"""
    Σ(array, dims)

Sum `array` along specific dimensions `dims`, ignoring nans, and dropping those dimensions

# Example
```jldoctest
julia> Σ([1 2; 3 NaN], 1)
2-element Vector{Float64}:
 4.0
 2.0
```
"""
Σ(array, dims) = dropdims(nansum(array, dims=dims), dims=dims)


"""
    extend(array1d, shape)

Extend `array1d` into other dimensions specified by `shape`

# Example
```jldoctest
julia> extend([1,2,3], (4,5))
4×5×3 Array{Int64, 3}:
[:, :, 1] =
 1  1  1  1  1
 1  1  1  1  1
 1  1  1  1  1
 1  1  1  1  1

[:, :, 2] =
 2  2  2  2  2
 2  2  2  2  2
 2  2  2  2  2
 2  2  2  2  2

[:, :, 3] =
 3  3  3  3  3
 3  3  3  3  3
 3  3  3  3  3
 3  3  3  3  3
```
"""
extend(array1d, shape) = repeat(reshape(array1d, (1,1,length(array1d))), outer=[shape...,1])


"""
    rest_frame(λ, z)

Convert observed-frame wavelength `λ` to rest-frame specified by redshift `z`

# Example
```jldoctest
julia> rest_frame([5, 5.1, 5.2], 0.1)
3-element Vector{Float64}:
 4.545454545454545
 4.636363636363636
 4.727272727272727
```
"""
rest_frame(λ, z) = @. λ / (1 + z)


"""
    observed_frame(λ, z)

Convert rest-frame wavelength `λ` specified by redshift `z` to observed-frame

# Example
```jldoctest
julia> observed_frame([5, 5.1, 5.2], 0.1)
3-element Vector{Float64}:
 5.5
 5.61
 5.720000000000001
```
"""
observed_frame(λ, z) = @. λ * (1 + z)


"""
    Doppler_shift_λ(λ₀, v)

Convert rest-frame wavelength `λ₀` to observed frame using the relativistic doppler shift at a 
given velocity `v` in km/s

# Examples
```jldoctest
julia> Doppler_shift_λ(10, 100)
10.003336197462627
julia> Doppler_shift_λ(10, -100)
9.996664915187521
julia> Doppler_shift_λ(10, 0)
10.0
```
"""
Doppler_shift_λ(λ₀, v) = λ₀ * √((1+v/C_KMS)/(1-v/C_KMS))


"""
    Doppler_shift_v(λ, λ₀)

Calculate the velocity in km/s required for the observed shift in wavelength between
`λ` and `λ₀`

# Examples
```jldoctest
julia> Doppler_shift_v(10.1, 10)
2982.9356991238064
julia> Doppler_shift_v(9.9, 10)
-3012.9134458865756
julia> Doppler_shift_v(10, 10)
0.0
```
"""
Doppler_shift_v(λ, λ₀) = ((λ/λ₀)^2 - 1)/((λ/λ₀)^2 + 1) * C_KMS


"""
    Doppler_width_v(Δλ, λ₀)

Doppler shift approximation for v << c: given a rest-frame wavelength `λ₀` and wavelength
shift `Δλ`, calculate the required velocity difference `Δv`

# Examples
```jldoctest
julia> Doppler_width_v(0.1, 10)
2997.92458
julia> Doppler_width_v(0.0, 10)
0.0
```
"""
Doppler_width_v(Δλ, λ₀) = Δλ / λ₀ * C_KMS


"""
    Doppler_width_λ(Δv, λ₀)

Doppler shift approximation for v << c: given a rest-frame wavelength `λ₀` and velocity
shift `Δv`, calculate the wavelength shift `Δλ`

# Examples
```jldoctest
julia> Doppler_width_λ(3000, 10)
0.10006922855944561
julia> Doppler_width_λ(0, 10)
0.0
```
"""
Doppler_width_λ(Δv, λ₀) = Δv / C_KMS * λ₀


"""
    ∫Gaussian(A, FWHM)

Integral of a Gaussian with amplitude `A` and full-width at half-max `FWHM`

# Examples
```jldoctest
julia> ∫Gaussian(1000, 0.5)
532.2335097156131
julia> ∫Gaussian(600, 1.2)
766.4162539904829
```
"""
∫Gaussian(A, FWHM) = √(π / (4log(2))) * A * FWHM


"""
    ∫Lorentzian(A)

Integral of a Lorentzian with amplitude `A`

# Examples
```jldoctest
julia> ∫Lorentzian(1000)
1000
julia> ∫Lorentzian(600)
600
"""
∫Lorentzian(A) = A


"""
    ∫Drude(A, FWHM)

Integral of a Drude with amplitude `A` and full-width at half-max `FWHM`

# Examples
```jldoctest
julia> ∫Drude(1000, 0.5)
785.3981633974482
julia> ∫Drude(600, 1.2)
1130.9733552923256

See CAFE (Marshall et al. 2007), PAHFIT (Smith, Draine et al. 2007) 
```
"""
∫Drude(A, FWHM) = π/2 * A * FWHM


"""
    MJysr_to_cgs(MJy, λ)

Convert specific intensity in MegaJanskys per steradian to CGS units 
-> erg s^-1 cm^-2 μm^-1 sr^-1, given the wavelength `λ` in μm

This converts from intensity per unit frequency to per unit wavelength (Fλ = Fν|dλ/dν| = Fν * c/λ^2)
"""
MJysr_to_cgs(MJy, λ) = MJy * 1e6 * 1e-23 * (C_KMS * 1e9) / λ^2


"""
    MJy_to_cgs_err(MJy, MJy_err, λ, λ_err)

Calculate the uncertainty in intensity in CGS units (erg s^-1 cm^-2 μm^-1 sr^-1)
given the uncertainty in intensity in MJy sr^-1 and the uncertainty in wavelength in μm
"""
function MJysr_to_cgs_err(MJy, MJy_err, λ, λ_err)
    if MJy == 0.
        cgs = 0.
        err = 1e6 * 1e-23 * (C_KMS * 1e9) / λ^2 * MJy_err
    else
        # Get the CGS value of the intensity
        cgs = MJysr_to_cgs(MJy, λ)
        # σ_cgs^2 / cgs^2 = σ_MJy^2 / MJy^2 + 4σ_λ^2 / λ^2
        frac_err2 = (MJy_err / MJy)^2 + 4(λ_err / λ)^2
        # rearrange to solve for σ_cgs
        err = √(frac_err2 * cgs^2)
    end
    if !isfinite(err)
        err = 0.
    end
    err
end


"""
    ln_likelihood(data, model, err)

Natural log of the likelihood for a given `model`, `data`, and `err`

# Example
```jldoctest
julia> ln_likelihood([1.1, 1.9, 3.2], [1., 2., 3.], [0.1, 0.1, 0.1])
1.1509396793681144
```
"""
function ln_likelihood(data::Vector{<:AbstractFloat}, model::Vector{<:AbstractFloat}, 
    err::Vector{<:AbstractFloat})::AbstractFloat
    -0.5 * sum(@. (data - model)^2 / err^2 + log(2π * err^2))
end


"""
    hermite(x, n)

Hermite polynomial of order n computed using the recurrence relations
(gets slow around/above order ~30)

# Examples
```jldoctest
julia> hermite(0., 1)
0.0
julia> hermite(1., 2)
2.0
julia> hermite(2., 3)
40.0
```
"""
function hermite(x::AbstractFloat, n::Integer)::AbstractFloat
    if iszero(n)
        1.
    elseif isone(n)
        2x
    else
        2x * hermite(x, n-1) - 2(n-1) * hermite(x, n-2)
    end
end


############################################### BLACKBODY PROFILE ##############################################


"""
    Blackbody_ν(λ, Temp)

Return the Blackbody function Bν (per unit FREQUENCY) in MJy/sr,
given a wavelength in μm and a temperature in Kelvins.

Function adapted from PAHFIT: Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
"""
function Blackbody_ν(λ::AbstractFloat, Temp::Real)::AbstractFloat
    Bν_1/λ^3 / (exp(Bν_2/(λ*Temp))-1)
end


"""
    Wein(Temp)

Return the peak wavelength (in μm) of a Blackbody spectrum at a given temperature `Temp`,
using Wein's Displacement Law.
"""
function Wein(Temp::AbstractFloat)::AbstractFloat
    b_Wein / Temp
end


################################################# PAH PROFILES ################################################


"""
    Drude(x, A, μ, FWHM)

Calculate a Drude profile at location `x`, with amplitude `A`, central value `μ`, and full-width at half-max `FWHM`

Function adapted from PAHFIT: Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
"""
function Drude(x::AbstractFloat, A::AbstractFloat, μ::AbstractFloat, FWHM::AbstractFloat)::AbstractFloat
    A * (FWHM/μ)^2 / ((x/μ - μ/x)^2 + (FWHM/μ)^2)
end

############################################## EXTINCTION PROFILES #############################################


"""
    τ_kvt(λ, β)

Calculate the mixed silicate extinction profile based on Kemper, Vriend, & Tielens (2004) 

Function adapted from PAHFIT: Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)
"""
function τ_kvt(λ::AbstractFloat, β::AbstractFloat)::AbstractFloat

    # Get limits of the values that we have datapoints for via the kvt_prof constant
    mx, mn = argmax(kvt_prof[:, 1]), argmin(kvt_prof[:, 1])
    λ_mx, λ_mn = kvt_prof[mx, 1], kvt_prof[mn, 1]

    # Interpolate based on the region of data 
    if λ ≤ λ_mn
        ext = kvt_prof[mn, 2] * exp(2.03 * (λ - λ_mn))
    elseif λ_mn < λ < λ_mx
        # ext = linear_interpolation(kvt_prof[:, 1], kvt_prof[:, 2])(λ)
        ext = Spline1D(kvt_prof[:, 1], kvt_prof[:, 2], k=1, bc="nearest")(λ)
    elseif λ_mx < λ < λ_mx + 2
        # ext = cubic_spline_interpolation(8.0:0.2:12.6, [kvt_prof[1:9, 2]; kvt_prof[12:26, 2]], extrapolation_bc=Line())(λ)
        ext = Spline1D(kvt_prof[:, 1], kvt_prof[:, 2], k=3, bc="extrapolate")(λ)
    else
        ext = 0.
    end
    ext = ext < 0 ? 0. : ext

    # Add a drude profile around 18 microns
    ext += Drude(λ, 0.4, 18., 4.446)

    (1 - β) * ext + β * (9.7/λ)^1.7
end


function τ_ct(λ::AbstractFloat)::AbstractFloat

    mx = argmax(CT_prof[1])
    λ_mx = CT_prof[1][mx]
    if λ > λ_mx
        ext = CT_prof[2][mx] * (λ_mx/λ)^1.7
    else
        ext = CT_interp(λ)
    end

    _, wh = findmin(x -> abs(x - 9.7), CT_prof[1])
    ext /= CT_prof[2][wh]

    ext
end


"""
    τ_dp(λ, β)

Calculate the mixed silicate extinction profile based on Donnan et al. (2022)
"""
function τ_dp(λ::AbstractFloat, β::AbstractFloat)::AbstractFloat

    # Simple cubic spline interpolation
    ext = DP_interp(λ)

    # Add 1.7 power law, as in PAHFIT
    (1 - β) * ext + β * (9.8/λ)^1.7
end


function Extinction(ext::AbstractFloat, τ_97::AbstractFloat; screen::Bool=false)::AbstractFloat
    """
    Calculate the overall extinction factor
    """
    if screen
        exp(-τ_97*ext)
    else
        iszero(τ_97) ? 1. : (1 - exp(-τ_97*ext)) / (τ_97*ext)
    end
end


############################################## FITTING FUNCTIONS #############################################


"""
    fit_spectrum(λ, params, n_dust_cont, n_dust_features, extinction_curve, extinction_screen;
        return_components=return_components)

Create a model of the continuum (including stellar+dust continuum, PAH features, and extinction, excluding emission lines)
at the given wavelengths `λ`, given the parameter vector `params`.

Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments
- `λ::Vector{<:AbstractFloat}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:AbstractFloat}`: Parameter vector. Parameters should be ordered as: 
    `[stellar amp, stellar temp, (amp, temp for each dust continuum), (amp, mean, FWHM for each PAH profile), 
    extinction τ, extinction β]`
- `n_dust_cont::Integer`: Number of dust continuum profiles to be fit
- `n_dust_features::Integer`: Number of PAH dust features to be fit
- `extinction_curve::String`: The type of extinction curve to use, "kvt" or "d+"
- `extinction_screen::Bool`: Whether or not to use a screen model for the extinction curve
- `return_components::Bool=false`: Whether or not to return the individual components of the fit as a dictionary, in 
    addition to the overall fit
"""
function fit_spectrum(λ::Vector{<:AbstractFloat}, params::Vector{<:AbstractFloat}, n_dust_cont::Integer, n_dust_features::Integer,
    extinction_curve::String, extinction_screen::Bool, return_components::Bool)

    # Prepare outputs
    comps = Dict{String, Vector{Float64}}()
    contin = zeros(Float64, length(λ))

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
    for j ∈ 1:n_dust_features
        comps["dust_feat_$j"] = Drude.(λ, params[pᵢ:pᵢ+2]...)
        contin .+= comps["dust_feat_$j"]
        pᵢ += 3
    end

    # Extinction 
    if extinction_curve == "d+"
        ext_curve = τ_dp.(λ, params[pᵢ+1])
    elseif extinction_curve == "kvt"
        ext_curve = τ_kvt.(λ, params[pᵢ+1])
    elseif extinction_curve == "ct"
        ext_curve = τ_ct.(λ)
    else
        error("Unrecognized extinction curve: $extinction_curve")
    end
    comps["extinction"] = Extinction.(ext_curve, params[pᵢ], screen=extinction_screen)
    contin .*= comps["extinction"]
    pᵢ += 2

    # Return components if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency --> not allocating the dictionary improves performance DRAMATICALLY
function fit_spectrum(λ::Vector{<:AbstractFloat}, params::Vector{<:AbstractFloat}, n_dust_cont::Integer, n_dust_features::Integer,
    extinction_curve::String, extinction_screen::Bool)

    # Prepare outputs
    contin = zeros(Float64, length(λ))

    # Stellar blackbody continuum (usually at 5000 K)
    contin .+= params[1] .* Blackbody_ν.(λ, params[2])
    pᵢ = 3

    # Add dust continua at various temperatures
    for i ∈ 1:n_dust_cont
        contin .+= params[pᵢ] .* (9.7 ./ λ).^2 .* Blackbody_ν.(λ, params[pᵢ+1])
        pᵢ += 2
    end

    # Add dust features with drude profiles
    for j ∈ 1:n_dust_features
        contin .+= Drude.(λ, params[pᵢ:pᵢ+2]...)
        pᵢ += 3
    end

    # Extinction 
    if extinction_curve == "d+"
        ext_curve = τ_dp.(λ, params[pᵢ+1])
    elseif extinction_curve == "kvt"
        ext_curve = τ_kvt.(λ, params[pᵢ+1])
    elseif extinction_curve == "ct"
        ext_curve = τ_ct.(λ)
    else
        error("Unrecognized extinction curve: $extinction_curve")
    end
    contin .*= Extinction.(ext_curve, params[pᵢ], screen=extinction_screen)
    pᵢ += 2

    contin

end


"""
    fit_line_residuals(λ, params, n_lines, n_voff_tied, voff_tied_key, line_tied, line_profiles,
        n_acomp_voff_tied, acomp_voff_tied_key, line_acomp_tied, line_acomp_profiles, line_restwave,
        flexible_wavesol, tie_voigt_mixing; return_components=return_components) 

Create a model of the emission lines at the given wavelengths `λ`, given the parameter vector `params`.

Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments
- `λ::Vector{<:AbstractFloat}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:AbstractFloat}`: Parameter vector. Parameters should be ordered as: 
    `[tied velocity offsets, tied acomp velocity offsets, tied voigt mixing, 
    (amp[, voff], FWHM[, h3, h4, η], [acomp_amp, acomp_voff, acomp_FWHM, acomp_h3, acomp_h4, acomp_η] for each line)]`
- `n_lines::Integer`: Number of lines being fit
- `n_voff_tied::Integer`: Number of tied velocity offsets
- `voff_tied_key::Vector{String}`: Unique identifiers for each tied velocity offset parameter
- `line_tied::Vector{Union{String,Nothing}}`: Vector, length of n_lines, giving the identifier corresponding to
    the values in `voff_tied_key` that corresponds to which tied velocity offset should be used for a given line,
    if any.
- `line_profiles::Vector{Symbol}`: Vector, length of n_lines, that gives the profile type that should be used to
    fit each line. The profiles should be one of `:Gaussian`, `:Lorentzian`, `:GaussHermite`, or `:Voigt`
- `n_acomp_voff_tied::Integer`: Same as `n_voff_tied`, but for additional line components
- `acomp_voff_tied_key::Vector{String}`: Same as `voff_tied_key`, but for additional line components
- `line_acomp_tied::Vector{Union{String,Nothing}}`: Same as `line_tied`, but for additional line components
- `line_acomp_profiles::Vector{Union{Symbol,Nothing}}`: Same as `line_profiles`, but for additional line components
- `line_restwave::Vector{<:AbstractFloat}`: Vector, length of n_lines, giving the rest wavelengths of each line
- `flexible_wavesol::Bool`: Whether or not to allow small variations in tied velocity offsets, to account for a poor
    wavelength solution in the data
- `tie_voigt_mixing::Bool`: Whether or not to tie the mixing parameters of all Voigt profiles together
- `return_components::Bool=false`: Whether or not to return the individual components of the fit as a dictionary, in 
    addition to the overall fit
"""
function fit_line_residuals(λ::Vector{<:AbstractFloat}, params::Vector{<:AbstractFloat}, n_lines::Integer, n_voff_tied::Integer, 
    voff_tied_key::Vector{String}, line_tied::Vector{Union{String,Nothing}}, line_profiles::Vector{Symbol}, 
    n_acomp_voff_tied::Integer, acomp_voff_tied_key::Vector{String}, line_acomp_tied::Vector{Union{String,Nothing}},
    line_acomp_profiles::Vector{Union{Symbol,Nothing}}, line_restwave::Vector{<:AbstractFloat}, 
    flexible_wavesol::Bool, tie_voigt_mixing::Bool, return_components::Bool)

    # Prepare outputs
    comps = Dict{String, Vector{Float64}}()
    contin = zeros(Float64, length(λ))

    # Skip ahead of the tied velocity offsets of the lines and acomp components
    pᵢ = n_voff_tied + n_acomp_voff_tied + 1
    # If applicable, skip ahead of the tied voigt mixing
    if tie_voigt_mixing
        ηᵢ = pᵢ
        pᵢ += 1
    end

    # Add emission lines
    for k ∈ 1:n_lines
        # Check if voff is tied: if so, use the tied voff parameter, otherwise, use the line's own voff parameter
        amp = params[pᵢ]
        if isnothing(line_tied[k])
            # Unpack the components of the line
            voff = params[pᵢ+1]
            fwhm = params[pᵢ+2]
            if line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = params[pᵢ+3]
                h4 = params[pᵢ+4]
            elseif line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                if !tie_voigt_mixing
                    η = params[pᵢ+3]
                else
                    η = params[ηᵢ]
                end
            end
        elseif !isnothing(line_tied[k]) && flexible_wavesol
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and voff_tied_key
            vwhere = findfirst(x -> x == line_tied[k], voff_tied_key)
            voff_series = params[vwhere]
            voff_indiv = params[pᵢ+1]
            # Add velocity shifts of the tied lines and the individual offsets together
            voff = voff_series + voff_indiv
            fwhm = params[pᵢ+2]
            if line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = params[pᵢ+3]
                h4 = params[pᵢ+4]
            elseif line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                if !tie_voigt_mixing
                    η = params[pᵢ+3]
                else
                    η = params[ηᵢ]
                end
            end
        else
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and voff_tied_key
            vwhere = findfirst(x -> x == line_tied[k], voff_tied_key)
            voff = params[vwhere]
            fwhm = params[pᵢ+1]
            # (dont add any individual voff components)
            if line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = params[pᵢ+2]
                h4 = params[pᵢ+3]
            elseif line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                if !tie_voigt_mixing
                    η = params[pᵢ+2]
                else
                    η = params[ηᵢ]
                end
            end
        end

        # Convert voff in km/s to mean wavelength in μm
        mean_μm = Doppler_shift_λ(line_restwave[k], voff)
        # Convert FWHM from km/s to μm
        fwhm_μm = Doppler_shift_λ(line_restwave[k], fwhm/2) - Doppler_shift_λ(line_restwave[k], -fwhm/2)
        # Evaluate line profile
        if line_profiles[k] == :Gaussian
            comps["line_$k"] = Gaussian.(λ, amp, mean_μm, fwhm_μm)
        elseif line_profiles[k] == :Lorentzian
            comps["line_$k"] = Lorentzian.(λ, amp, mean_μm, fwhm_μm)
        elseif line_profiles[k] == :GaussHermite
            comps["line_$k"] = GaussHermite.(λ, amp, mean_μm, fwhm_μm, h3, h4)
        elseif line_profiles[k] == :Voigt
            comps["line_$k"] = Voigt.(λ, amp, mean_μm, fwhm_μm, η)
        else
            error("Unrecognized line profile $(line_profiles[k])!")
        end

        # Add the line profile into the overall model
        contin .+= comps["line_$k"]        
        # Advance the parameter vector index -> 3 if untied (or tied + flexible_wavesol) or 2 if tied
        pᵢ += isnothing(line_tied[k]) || flexible_wavesol ? 3 : 2
        if line_profiles[k] == :GaussHermite
            # advance and extra 2 if GaussHermite profile
            pᵢ += 2
        elseif line_profiles[k] == :Voigt
            # advance another extra 1 if untied Voigt profile
            if !tie_voigt_mixing
                pᵢ += 1
            end
        end

        # Repeat EVERYTHING, minus the flexible_wavesol, for the additional components
        if !isnothing(line_acomp_profiles[k])
            # Parametrize acomp amplitude in terms of the default line amplitude times some fractional value
            # this way, we can constrain the acomp_amp parameter to be from (0,1) to ensure the acomp amplitude is always
            # less than the line amplitude
            acomp_amp = amp * params[pᵢ]

            if isnothing(line_acomp_tied[k])
                # Parametrize the acomp voff in terms of the default line voff plus some difference value
                # this way, we can constrain the acomp component to be within +/- some range from the line itself
                acomp_voff = voff + params[pᵢ+1]
                # Parametrize the acomp FWHM in terms of the default line FWHM times some fractional value
                # this way, we can constrain the acomp_fwhm parameter to be > 0 to ensure the acomp FWHM is always
                # greater than the line FWHM
                acomp_fwhm = fwhm * params[pᵢ+2]
                if line_acomp_profiles[k] == :GaussHermite
                    acomp_h3 = params[pᵢ+3]
                    acomp_h4 = params[pᵢ+4]
                elseif line_acomp_profiles[k] == :Voigt
                    if !tie_voigt_mixing
                        acomp_η = params[pᵢ+3]
                    else
                        acomp_η = params[ηᵢ]
                    end
                end
            else
                vwhere = findfirst(x -> x == line_acomp_tied[k], acomp_voff_tied_key)
                acomp_voff = voff + params[n_voff_tied + vwhere]
                acomp_fwhm = fwhm * params[pᵢ+1]
                if line_acomp_profiles[k] == :GaussHermite
                    acomp_h3 = params[pᵢ+2]
                    acomp_h4 = params[pᵢ+3]
                elseif line_acomp_profiles[k] == :Voigt
                    if !tie_voigt_mixing
                        acomp_η = params[pᵢ+2]
                    else
                        acomp_η = params[ηᵢ]
                    end
                end
            end

            # Convert voff in km/s to mean wavelength in μm
            acomp_mean_μm = Doppler_shift_λ(line_restwave[k], acomp_voff)
            # Convert FWHM from km/s to μm
            acomp_fwhm_μm = Doppler_shift_λ(line_restwave[k], acomp_fwhm/2) - Doppler_shift_λ(line_restwave[k], -acomp_fwhm/2)
            # Evaluate line profile
            if line_acomp_profiles[k] == :Gaussian
                comps["line_$(k)_acomp"] = Gaussian.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm)
            elseif line_acomp_profiles[k] == :Lorentzian
                comps["line_$(k)_acomp"] = Lorentzian.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm)
            elseif line_profiles[k] == :GaussHermite
                comps["line_$(k)_acomp"] = GaussHermite.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm, acomp_h3, acomp_h4)
            elseif line_profiles[k] == :Voigt
                comps["line_$(k)_acomp"] = Voigt.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm, acomp_η)
            else
                error("Unrecognized acomp line profile $(line_profiles[k])!")
            end

            # Add the additional component into the overall model
            contin .+= comps["line_$(k)_acomp"]
            # Advance the parameter vector index by the appropriate amount        
            pᵢ += isnothing(line_acomp_tied[k]) ? 3 : 2
            if line_acomp_profiles[k] == :GaussHermite
                pᵢ += 2
            elseif line_acomp_profiles[k] == :Voigt
                if !tie_voigt_mixing
                    pᵢ += 1
                end
            end
        end

    end

    # Return components if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency --> not allocating the dictionary improves performance DRAMATICALLY
function fit_line_residuals(λ::Vector{<:AbstractFloat}, params::Vector{<:AbstractFloat}, n_lines::Integer, n_voff_tied::Integer, 
    voff_tied_key::Vector{String}, line_tied::Vector{Union{String,Nothing}}, line_profiles::Vector{Symbol}, 
    n_acomp_voff_tied::Integer, acomp_voff_tied_key::Vector{String}, line_acomp_tied::Vector{Union{String,Nothing}},
    line_acomp_profiles::Vector{Union{Symbol,Nothing}}, line_restwave::Vector{<:AbstractFloat}, 
    flexible_wavesol::Bool, tie_voigt_mixing::Bool)

    # Prepare outputs
    contin = zeros(Float64, length(λ))

    # Skip ahead of the tied velocity offsets of the lines and acomp components
    pᵢ = n_voff_tied + n_acomp_voff_tied + 1
    # If applicable, skip ahead of the tied voigt mixing
    if tie_voigt_mixing
        ηᵢ = pᵢ
        pᵢ += 1
    end

    # Add emission lines
    for k ∈ 1:n_lines
        # Check if voff is tied: if so, use the tied voff parameter, otherwise, use the line's own voff parameter
        amp = params[pᵢ]
        if isnothing(line_tied[k])
            # Unpack the components of the line
            voff = params[pᵢ+1]
            fwhm = params[pᵢ+2]
            if line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = params[pᵢ+3]
                h4 = params[pᵢ+4]
            elseif line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                if !tie_voigt_mixing
                    η = params[pᵢ+3]
                else
                    η = params[ηᵢ]
                end
            end
        elseif !isnothing(line_tied[k]) && flexible_wavesol
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and voff_tied_key
            vwhere = findfirst(x -> x == line_tied[k], voff_tied_key)
            voff_series = params[vwhere]
            voff_indiv = params[pᵢ+1]
            # Add velocity shifts of the tied lines and the individual offsets together
            voff = voff_series + voff_indiv
            fwhm = params[pᵢ+2]
            if line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = params[pᵢ+3]
                h4 = params[pᵢ+4]
            elseif line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                if !tie_voigt_mixing
                    η = params[pᵢ+3]
                else
                    η = params[ηᵢ]
                end
            end
        else
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and voff_tied_key
            vwhere = findfirst(x -> x == line_tied[k], voff_tied_key)
            voff = params[vwhere]
            fwhm = params[pᵢ+1]
            # (dont add any individual voff components)
            if line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = params[pᵢ+2]
                h4 = params[pᵢ+3]
            elseif line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
                if !tie_voigt_mixing
                    η = params[pᵢ+2]
                else
                    η = params[ηᵢ]
                end
            end
        end

        # Convert voff in km/s to mean wavelength in μm
        mean_μm = Doppler_shift_λ(line_restwave[k], voff)
        # Convert FWHM from km/s to μm
        fwhm_μm = Doppler_shift_λ(line_restwave[k], fwhm/2) - Doppler_shift_λ(line_restwave[k], -fwhm/2)
        # Evaluate line profile
        if line_profiles[k] == :Gaussian
            contin .+= Gaussian.(λ, amp, mean_μm, fwhm_μm)
        elseif line_profiles[k] == :Lorentzian
            contin .+= Lorentzian.(λ, amp, mean_μm, fwhm_μm)
        elseif line_profiles[k] == :GaussHermite
            contin .+= GaussHermite.(λ, amp, mean_μm, fwhm_μm, h3, h4)
        elseif line_profiles[k] == :Voigt
            contin .+= Voigt.(λ, amp, mean_μm, fwhm_μm, η)
        else
            error("Unrecognized line profile $(line_profiles[k])!")
        end

        # Advance the parameter vector index -> 3 if untied (or tied + flexible_wavesol) or 2 if tied
        pᵢ += isnothing(line_tied[k]) || flexible_wavesol ? 3 : 2
        if line_profiles[k] == :GaussHermite
            # advance and extra 2 if GaussHermite profile
            pᵢ += 2
        elseif line_profiles[k] == :Voigt
            # advance another extra 1 if untied Voigt profile
            if !tie_voigt_mixing
                pᵢ += 1
            end
        end

        # Repeat EVERYTHING, minus the flexible_wavesol, for the additional components
        if !isnothing(line_acomp_profiles[k])
            # Parametrize acomp amplitude in terms of the default line amplitude times some fractional value
            # this way, we can constrain the acomp_amp parameter to be from (0,1) to ensure the acomp amplitude is always
            # less than the line amplitude
            acomp_amp = amp * params[pᵢ]

            if isnothing(line_acomp_tied[k])
                # Parametrize the acomp voff in terms of the default line voff plus some difference value
                # this way, we can constrain the acomp component to be within +/- some range from the line itself
                acomp_voff = voff + params[pᵢ+1]
                # Parametrize the acomp FWHM in terms of the default line FWHM times some fractional value
                # this way, we can constrain the acomp_fwhm parameter to be > 0 to ensure the acomp FWHM is always
                # greater than the line FWHM
                acomp_fwhm = fwhm * params[pᵢ+2]
                if line_acomp_profiles[k] == :GaussHermite
                    acomp_h3 = params[pᵢ+3]
                    acomp_h4 = params[pᵢ+4]
                elseif line_acomp_profiles[k] == :Voigt
                    if !tie_voigt_mixing
                        acomp_η = params[pᵢ+3]
                    else
                        acomp_η = params[ηᵢ]
                    end
                end
            else
                vwhere = findfirst(x -> x == line_acomp_tied[k], acomp_voff_tied_key)
                acomp_voff = voff + params[n_voff_tied + vwhere]
                acomp_fwhm = fwhm * params[pᵢ+1]
                if line_acomp_profiles[k] == :GaussHermite
                    acomp_h3 = params[pᵢ+2]
                    acomp_h4 = params[pᵢ+3]
                elseif line_acomp_profiles[k] == :Voigt
                    if !tie_voigt_mixing
                        acomp_η = params[pᵢ+2]
                    else
                        acomp_η = params[ηᵢ]
                    end
                end
            end

            # Convert voff in km/s to mean wavelength in μm
            acomp_mean_μm = Doppler_shift_λ(line_restwave[k], acomp_voff)
            # Convert FWHM from km/s to μm
            acomp_fwhm_μm = Doppler_shift_λ(line_restwave[k], acomp_fwhm/2) - Doppler_shift_λ(line_restwave[k], -acomp_fwhm/2)
            # Evaluate line profile
            if line_acomp_profiles[k] == :Gaussian
                contin .+= Gaussian.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm)
            elseif line_acomp_profiles[k] == :Lorentzian
                contin .+= Lorentzian.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm)
            elseif line_profiles[k] == :GaussHermite
                contin .+= GaussHermite.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm, acomp_h3, acomp_h4)
            elseif line_profiles[k] == :Voigt
                contin .+= Voigt.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm, acomp_η)
            else
                error("Unrecognized acomp line profile $(line_profiles[k])!")
            end

            # Advance the parameter vector index by the appropriate amount        
            pᵢ += isnothing(line_acomp_tied[k]) ? 3 : 2
            if line_acomp_profiles[k] == :GaussHermite
                pᵢ += 2
            elseif line_acomp_profiles[k] == :Voigt
                if !tie_voigt_mixing
                    pᵢ += 1
                end
            end
        end

    end

    contin

end


############################################## LINE PROFILES #############################################


"""
    Gaussian(x, A, μ, FWHM)

Evaluate a Gaussian profile at `x`, parameterized by the amplitude `A`, mean value `μ`, and 
full-width at half-maximum `FWHM`
"""
function Gaussian(x::AbstractFloat, A::AbstractFloat, μ::AbstractFloat, FWHM::AbstractFloat)::AbstractFloat
    # Reparametrize FWHM as dispersion σ
    σ = FWHM / (2√(2log(2)))
    A * exp(-(x-μ)^2 / (2σ^2))
end


"""
    GaussHermite(x, A, μ, FWHM, h₃, h₄)

Evaluate a Gauss-Hermite quadrature at `x`, parametrized by the amplitude `A`, mean value `μ`,
full-width at half-maximum `FWHM`, 3rd moment / skewness `h₃`, and 4th moment / kurtosis `h₄`

See Riffel et al. (2010)
"""
function GaussHermite(x::AbstractFloat, A::AbstractFloat, μ::AbstractFloat, FWHM::AbstractFloat, 
    h₃::AbstractFloat, h₄::AbstractFloat)::AbstractFloat

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
    Herm = sum([coeff[nᵢ] * hermite(w, nᵢ-1) for nᵢ ∈ eachindex(coeff)])

    # Calculate peak height (i.e. value of function at w=0)
    Herm0 = sum([coeff[nᵢ] * hermite(0., nᵢ-1) for nᵢ ∈ eachindex(coeff)])

    # Combine the Gaussian and Hermite profiles
    A * α * Herm / Herm0
end


"""
    Lorentzian(x, A, μ, FWHM)

Evaluate a Lorentzian profile at `x`, parametrized by the amplitude `A`, mean value `μ`,
and full-width at half-maximum `FWHM`
"""
function Lorentzian(x::AbstractFloat, A::AbstractFloat, μ::AbstractFloat, FWHM::AbstractFloat)::AbstractFloat
    A/π * (FWHM/2) / ((x-μ)^2 + (FWHM/2)^2)
end


"""
    Voigt(x, A, μ, FWHM, η)

Evaluate a pseudo-Voigt profile at `x`, parametrized by the amplitude `A`, mean value `μ`,
full-width at half-maximum `FWHM`, and mixing ratio `η`
"""
function Voigt(x::AbstractFloat, A::AbstractFloat, μ::AbstractFloat, FWHM::AbstractFloat, 
    η::AbstractFloat)::AbstractFloat

    # Reparametrize FWHM as dispersion σ
    σ = FWHM / (2√(2log(2))) 
    # Normalized Gaussian
    G = 1/√(2π * σ^2) * exp(-(x-μ)^2 / (2σ^2))
    # Normalized Lorentzian
    L = 1/π * (FWHM/2) / ((x-μ)^2 + (FWHM/2)^2)

    # Intensity parameter given the peak height A
    I = A * FWHM * π / (2 * (1 + (√(π*log(2)) - 1)*η))

    # Mix the two distributions with the mixing parameter η
    I * (η * G + (1 - η) * L)
end


"""
    _continuum(x, popt_c, n_dust_cont, n_dust_feat)

Calculate the pure blackbody continuum (with or without PAHs) of the spectrum given
optimization parameters popt_c, at a location x.
"""
function _continuum(x::AbstractFloat, popt_c::Vector{<:AbstractFloat}, n_dust_cont::Integer,
    n_dust_feat::Integer=0)
    c = 0.
    # stellar continuum
    c += popt_c[1] * Blackbody_ν(x, popt_c[2])
    pₓ = 3
    # dust continua
    for i ∈ 1:n_dust_cont
        c += popt_c[pₓ] * (9.7 / x)^2 * Blackbody_ν(x, popt_c[pₓ+1])
        pₓ += 2
    end
    # dust features
    for j ∈ 1:n_dust_feat
        c += Drude(x, popt_c[pₓ:pₓ+2]...)
        pₓ += 3
    end
    # add small epsilon to avoid divide by zero errors
    if c ≤ 0
        c = eps()
    end
    # dont include extinction here since it's not included in the PAH/line profiles either
    c
end


"""
    _continuum_errs(x, popt_c, perr_c, n_dust_cont, n_dust_feat)

Calculate the error in the pure blackbody continuum (with or without PAHs) of the spectrum given
optimization parameters popt_c and errors perr_c, at a location x.
"""
function _continuum_errs(x::AbstractFloat, popt_c::Vector{<:AbstractFloat},
    perr_c::Vector{<:AbstractFloat}, n_dust_cont::Integer, n_dust_feat::Integer=0)
    c_l = 0.
    c_u = 0.
    # stellar continuum
    c_l += max(popt_c[1] - perr_c[1], 0.) * Blackbody_ν(x, max(popt_c[2] - perr_c[2], 0.))
    c_u += (popt_c[1] + perr_c[1]) * Blackbody_ν(x, popt_c[2] + perr_c[2])
    pₓ = 3
    # dust continua
    for i ∈ 1:n_dust_cont
        c_l += max(popt_c[pₓ] - perr_c[pₓ], 0.) * (9.7 / x)^2 * Blackbody_ν(x, max(popt_c[pₓ+1] - perr_c[pₓ+1], 0.))
        c_u += (popt_c[pₓ] + perr_c[pₓ]) * (9.7 / x)^2 * Blackbody_ν(x, popt_c[pₓ+1] + perr_c[pₓ+1])
        pₓ += 2
    end
    # dust features
    for j ∈ 1:n_dust_feat
        c_l += Drude(x, max(popt_c[pₓ] - perr_c[pₓ], 0.), popt_c[pₓ+1], max(popt_c[pₓ+2] - perr_c[pₓ+2], eps()))
        c_u += Drude(x, popt_c[pₓ] + perr_c[pₓ], popt_c[pₓ+1], popt_c[pₓ+2] + perr_c[pₓ+2])
        pₓ += 3
    end
    # add small epsilon to avoid divide by zero errors
    if c_l ≤ 0
        c_l = eps()
    end
    if c_u ≤ 0
        c_u = eps()
    end
    # dont include extinction
    c_l, c_u
end


"""
    calculate_intensity(profile, amp, amp_err, peak, peak_err, fwhm, fwhm_err; <keyword_args>)

Calculate the integrated intensity of a spectral feature, i.e. a PAH or emission line. Calculates the integral
of the feature profile, using an analytic form if available, otherwise integrating numerically with QuadGK.
"""
function calculate_intensity(profile::Symbol, amp::T, amp_err::T, peak::T, peak_err::T, fwhm::T, fwhm_err::T;
    h3::Union{T,Nothing}=nothing, h3_err::Union{T,Nothing}=nothing, h4::Union{T,Nothing}=nothing, 
    h4_err::Union{T,Nothing}=nothing, η::Union{T,Nothing}=nothing, η_err::Union{T,Nothing}=nothing) where {T<:AbstractFloat}

    # Evaluate the line profiles according to whether there is a simple analytic form
    # otherwise, integrate numerically with quadgk
    if profile == :Drude
        # (integral = π/2 * A * fwhm)
        intensity = ∫Drude(amp, fwhm)
        if iszero(amp)
            i_err = π/2 * fwhm * amp_err
        else
            frac_err2 = (amp_err / amp)^2 + (fwhm_err / fwhm)^2
            i_err = √(frac_err2 * intensity^2)
        end
    elseif profile == :Gaussian
        intensity = ∫Gaussian(amp, fwhm)
        frac_err2 = (amp_err / amp)^2 + (fwhm_err / fwhm)^2
        i_err = √(frac_err2 * intensity^2)
    elseif profile == :Lorentzian
        intensity = ∫Lorentzian(amp)
        i_err = amp_err
    elseif profile == :GaussHermite
        # shift the profile to be centered at 0 since it doesnt matter for the integral, and it makes it
        # easier for quadgk to find a solution
        intensity = quadgk(x -> GaussHermite(x+peak, amp, peak, fwhm, h3, h4), -Inf, Inf, order=200)[1]
        # estimate error by evaluating the integral at +/- 1 sigma
        err_l = intensity - quadgk(x -> GaussHermite(x+peak, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), h3-h3_err, h4-h4_err), -Inf, Inf, order=200)[1]
        err_u = quadgk(x -> GaussHermite(x+peak, amp+amp_err, peak, fwhm+fwhm_err, h3+h3_err, h4+h4_err), -Inf, Inf, order=200)[1] - intensity
        err_l = err_l ≥ 0 ? err_l : 0.
        err_u = abs(err_u)
        i_err = (err_l + err_u)/2
    elseif profile == :Voigt
        # also use a high order to ensure that all the initial test points dont evaluate to precisely 0
        intensity = quadgk(x -> Voigt(x+peak, amp, peak, fwhm, η), -Inf, Inf, order=200)[1]
        # estimate error by evaluating the integral at +/- 1 sigma
        err_l = intensity - quadgk(x -> Voigt(x+peak, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), η-η_err), -Inf, Inf, order=200)[1]
        err_u = quadgk(x -> Voigt(x+peak, amp+amp_err, peak, fwhm+fwhm_err, η+η_err), -Inf, Inf, order=200)[1] - intensity
        err_l = err_l ≥ 0 ? err_l : 0.
        err_u = abs(err_u)
        i_err = (err_l + err_u)/2
    else
        error("Unrecognized line profile $profile")
    end

    @debug "I=$intensity, err=$i_err"
    intensity, i_err
end


"""
    calculate_eqw(popt_c, perr_c, n_dust_cont, n_dust_feat, profile, amp, amp_err, 
        peak, peak_err, fwhm, fwhm_err; <keyword_args>)

Calculate the equivalent width (in microns) of a spectral feature, i.e. a PAH or emission line. Calculates the
integral of the ratio of the feature profile to the underlying continuum, calculated using the _continuum function.
"""
function calculate_eqw(popt_c::Vector{T}, perr_c::Vector{T}, n_dust_cont::Integer, n_dust_feat::Integer, profile::Symbol, 
    amp::T, amp_err::T, peak::T, peak_err::T, fwhm::T, fwhm_err::T; h3::Union{T,Nothing}=nothing, h3_err::Union{T,Nothing}=nothing, 
    h4::Union{T,Nothing}=nothing, h4_err::Union{T,Nothing}=nothing, η::Union{T,Nothing}=nothing, 
    η_err::Union{T,Nothing}=nothing) where {T<:AbstractFloat}

    # If the line is not present, the equivalent width is 0
    if iszero(amp)
        @debug "eqw=0, err=0"
        return 0., 0.
    end

    # Integrate the flux ratio to get equivalent width
    if profile == :Drude
        # do not shift the drude profiles since x=0 and mu=0 cause problems;
        # the wide wings should allow quadgk to find the solution even without shifting it
        eqw = quadgk(x -> Drude(x, amp, peak, fwhm) / _continuum(x, popt_c, n_dust_cont), max(peak-10fwhm, 3.), peak+10fwhm, order=200)[1]
        # errors
        err_l = eqw - quadgk(x -> Drude(x, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps())) / _continuum_errs(x, popt_c, perr_c, n_dust_cont)[1], 
            max(peak-10fwhm, 3.), peak+10fwhm, order=200)[1]
        err_u = quadgk(x -> Drude(x, amp+amp_err, peak, fwhm+fwhm_err) / _continuum_errs(x, popt_c, perr_c, n_dust_cont)[2], 
            max(peak-10fwhm, 3.), peak+10fwhm, order=200)[1] - eqw
        err_l = err_l ≥ 0 ? err_l : 0.
        err_u = abs(err_u)
        err = (err_l + err_u)/2
    elseif profile == :Gaussian
        eqw = quadgk(x -> Gaussian(x+peak, amp, peak, fwhm) / _continuum(x+peak, popt_c, n_dust_cont, n_dust_feat), -10fwhm, 10fwhm, order=200)[1]
        err_l = eqw - quadgk(x -> Gaussian(x+peak, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps())) / _continuum_errs(x+peak, popt_c, perr_c, n_dust_cont, n_dust_feat)[1], 
            -10fwhm, 10fwhm, order=200)[1]
        err_u = quadgk(x -> Gaussian(x+peak, amp+amp_err, peak, fwhm+fwhm_err) / _continuum_errs(x+peak, popt_c, perr_c, n_dust_cont, n_dust_feat)[2], 
            -10fwhm, 10fwhm, order=200)[1] - eqw
        err_l = err_l ≥ 0 ? err_l : 0.
        err_u = abs(err_u)
        err = (err_l + err_u)/2
    elseif profile == :Lorentzian
        eqw = quadgk(x -> Lorentzian(x+peak, amp, peak, fwhm) / _continuum(x+peak, popt_c, n_dust_cont, n_dust_feat), -10fwhm, 10fwhm, order=200)[1]
        err_l = eqw - quadgk(x -> Lorentzian(x+peak, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps())) / _continuum_errs(x+peak, popt_c, perr_c, n_dust_cont, n_dust_feat)[1], 
            -10fwhm, 10fwhm, order=200)[1]
        err_u = quadgk(x -> Lorentzian(x+peak, amp+amp_err, peak, fwhm+fwhm_err) / _continuum_errs(x+peak, popt_c, perr_c, n_dust_cont, n_dust_feat)[2], 
            -10fwhm, 10fwhm, order=200)[1] - eqw
        err_l = err_l ≥ 0 ? err_l : 0.
        err_u = abs(err_u)
        err = (err_l + err_u)/2
    elseif profile == :GaussHermite
        eqw = quadgk(x -> GaussHermite(x+peak, amp, peak, fwhm, h3, h4) / _continuum(x+peak, popt_c, n_dust_cont, n_dust_feat), -10fwhm, 10fwhm, order=200)[1]
        err_l = eqw - quadgk(x -> GaussHermite(x+peak, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), h3-h3_err, h4-h4_err) / 
            _continuum_errs(x+peak, popt_c, perr_c, n_dust_cont, n_dust_feat)[1], -10fwhm, 10fwhm, order=200)[1]
        err_u = quadgk(x -> GaussHermite(x+peak, amp+amp_err, peak, fwhm+fwhm_err, h3+h3_err, h4+h4_err) / 
            _continuum_errs(x+peak, popt_c, perr_c, n_dust_cont, n_dust_feat)[2], -10fwhm, 10fwhm, order=200)[1] - eqw
        err_l = err_l ≥ 0 ? err_l : 0.
        err_u = abs(err_u)
        err = (err_l + err_u)/2
    elseif profile == :Voigt
        eqw = quadgk(x -> Voigt(x+peak, amp, peak, fwhm, η) / _continuum(x+peak, popt_c, n_dust_cont, n_dust_feat), -10fwhm, 10fwhm, order=200)[1]
        err_l = eqw - quadgk(x -> Voigt(x+peak, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), η-η_err) / 
            _continuum_errs(x+peak, popt_c, perr_c, n_dust_cont, n_dust_feat)[1], -10fwhm, 10fwhm, order=200)[1]
        err_u = quadgk(x -> Voigt(x+peak, amp+amp_err, peak, fwhm+fwhm_err, η+η_err) / 
            _continuum_errs(x+peak, popt_c, perr_c, n_dust_cont, n_dust_feat)[2], -10fwhm, 10fwhm, order=200)[1] - eqw
        err_l = err_l ≥ 0 ? err_l : 0.
        err_u = abs(err_u)
        err = (err_l + err_u)/2
    else
        error("Unrecognized line profile $profile")
    end

    @debug "eqw=$eqw, err=$err"
    eqw, err
end


"""
    calculate_SNR(resolution, continuum, prof, amp, peak, fwhm; <keyword_args>)

Calculate the signal to noise ratio of a spectral feature, i.e. a PAH or emission line. Calculates the ratio
of the peak intensity of the feature over the root-mean-square (RMS) deviation of the surrounding spectrum.
"""
function calculate_SNR(resolution::T, continuum::Vector{T}, prof::Symbol, amp::T, peak::T, 
    fwhm::T; h3::Union{T,Nothing}=nothing, h4::Union{T,Nothing}=nothing, 
    η::Union{T,Nothing}=nothing, acomp_prof::Union{Symbol,Nothing}=nothing, 
    acomp_amp::Union{T,Nothing}=nothing, acomp_peak::Union{T,Nothing}=nothing, 
    acomp_fwhm::Union{T,Nothing}=nothing, acomp_h3::Union{T,Nothing}=nothing, 
    acomp_h4::Union{T,Nothing}=nothing, acomp_η::Union{T,Nothing}=nothing) where {T<:AbstractFloat}

    # PAH / Drude profiles do not have extra components, so it's a simple A/RMS
    if prof == :Drude
        return amp / std(continuum)
    end

    # Prepare an anonymous function for calculating the line profile
    if prof == :Gaussian
        profile = x -> Gaussian(x, amp, peak, fwhm)
    elseif prof == :Lorentzian
        profile = x -> Lorentzian(x, amp, peak, fwhm)
    elseif prof == :GaussHermite
        profile = x -> GaussHermite(x, amp, peak, fwhm, h3, h4)
    elseif prof == :Voigt
        profile = x -> Voigt(x, amp, peak, fwhm, η)
    else
        error("Unrecognized line profile $(cube_fitter.line_profiles[k])!")
    end

    # Add in additional component profiles if necessary
    if !isnothing(acomp_prof)
        if acomp_prof == :Gaussian
            profile = let profile = profile
                x -> profile(x) + Gaussian(x, acomp_amp, acomp_peak, acomp_fwhm)
            end
        elseif acomp_prof == :Lorentzian
            profile = let profile = profile
                x -> profile(x) + Lorentzian(x, acomp_amp, acomp_peak, acomp_fwhm)
            end
        elseif acomp_prof == :GaussHermite
            profile = let profile = profile
                x -> profile(x) + GaussHermite(x, acomp_amp, acomp_peak, acomp_fwhm, acomp_h3, acomp_h4)
            end
        elseif acomp_prof == :Voigt
            profile = let profile = profile
                x -> profile(x) + Voigt(x, acomp_amp, acomp_peak, acomp_fwhm, acomp_η)
            end
        else
            error("Unrecognized acomp line profile $(cube_fitter.line_acomp_profiles[k])!")
        end
    end

    λ_arr = (peak-10fwhm):resolution:(peak+10fwhm)
    i_max, _ = findmax(profile.(λ_arr))
    # Factor in extinction

    i_max / std(continuum)
end


end
