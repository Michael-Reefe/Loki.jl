module Util

using NaNStatistics
using Interpolations
using Dierckx
using CSV
using DataFrames

# CONSTANTS

const C_KMS = 299792.458          # Speed of light in km/s
const h_ERGS = 6.62607015e-27     # Planck constant in erg*s
const kB_ERGK = 1.380649e-16      # Boltzmann constant in erg/K

const Bν_1 = 3.97289e13           # First constant for Planck function, in MJy/sr/μm
const Bν_2 = 1.4387752e4          # Second constant for Planck function, in μm*K

# Saved Kemper, Vriend, & Tielens (2004) extinction profile
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


"""
    read_irs_data(path)

Setup function for reading in the configuration IRS spectrum of IRS 08572+3915

# Arguments
- `path::String`: The file path pointing to the IRS 08572+3915 spectrum 
"""
function read_irs_data(path::String)
    datatable = CSV.read(path, DataFrame, comment="#", delim=' ', ignorerepeated=true, stripwhitespace=true,
        header=["rest_wave", "flux", "e_flux", "enod", "order", "module", "nod1flux", "nod2flux", "e_nod1flux", "e_nod2flux"])
    return datatable[!, "rest_wave"], datatable[!, "flux"], datatable[!, "e_flux"]
end


"""
    silicate_dp()

Setup function for creating a silicate extinction profile based on Donnan et al. (2022)
"""
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

    return λ_irs, τ_λ
end

# Save the Donnan et al. 2022 profile as a constant
const DPlus_prof = silicate_dp()


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
    ∫Gaussian(A, σ)

Integral of a Gaussian with amplitude `A` and standard deviation `σ`

# Examples
```jldoctest
julia> ∫Gaussian(1, 1)
2.5066282746310002
julia> ∫Gaussian(0, 1)
0.0
julia> ∫Gaussian(-1, 1)
-2.5066282746310002
```
"""
∫Gaussian(A, σ) = √(2π) * A * σ


# Convert units
# to_cgs(F, λ) = F .* 1e-7 .* Util.C_MS ./ λ.^2    # λ in angstroms, F (MJy/sr) -> (erg/s/cm^2/A/sr)
# to_MJy_sr(F, λ) = F .* 1e7 ./ Util.C_MS .* λ.^2  # λ in angstroms, F (erg/s/cm^2/A/sr) -> (MJy/sr)


"""
    ln_likelihood(data, model, err)

Natural log of the likelihood for a given `model`, `data`, and `err`

# Example
```jldoctest
julia> ln_likelihood([1.1, 1.9, 3.2], [1., 2., 3.], [0.1, 0.1, 0.1])
1.1509396793681144
```
"""
function ln_likelihood(data::Vector{<:AbstractFloat}, model::Vector{<:AbstractFloat}, err::Vector{<:AbstractFloat})
    return -0.5 * sum((data .- model).^2 ./ err.^2 .+ log.(2π .* err.^2))
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
function hermite(x::AbstractFloat, n::Integer)
    if iszero(n)
        return 1.
    elseif isone(n)
        return 2x
    else
        return 2x * hermite(x, n-1) - 2(n-1) * hermite(x, n-2)
    end
end


############################################### BLACKBODY PROFILE ##############################################


"""
    Blackbody_ν(λ, Temp)

Return the Blackbody function Bν (per unit FREQUENCY) in MJy/sr,
given a wavelength in μm and a temperature in Kelvins.

Function adapted from PAHFIT: Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
"""
function Blackbody_ν(λ::AbstractFloat, Temp::Real) 
    return Bν_1/λ^3 / (exp(Bν_2/(λ*Temp))-1)
end


################################################# PAH PROFILES ################################################


"""
    Drude(x, A, μ, FWHM)

Calculate a Drude profile at location `x`, with amplitude `A`, central value `μ`, and full-width at half-max `FWHM`

Function adapted from PAHFIT: Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
"""
function Drude(x::Float64, A::Float64, μ::Float64, FWHM::Float64)
    return A * (FWHM/μ)^2 / ((x/μ - μ/x)^2 + (FWHM/μ)^2)
end

############################################## EXTINCTION PROFILES #############################################


"""
    τ_kvt(λ, β)

Calculate the mixed silicate extinction profile based on Kemper, Vriend, & Tielens (2004) 

Function adapted from PAHFIT: Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)
"""
function τ_kvt(λ::AbstractFloat, β::AbstractFloat)

    # Get limits of the values that we have datapoints for via the kvt_prof constant
    mx, mn = argmax(kvt_prof[:, 1]), argmin(kvt_prof[:, 1])
    λ_mx, λ_mn = kvt_prof[mx, 1], kvt_prof[mn, 1]

    # Interpolate based on the region of data 
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

    # Add a drude profile around 18 microns
    ext += Drude(λ, 0.4, 18., 4.446)

    return (1 - β) * ext + β * (9.7/λ)^1.7
end


"""
    τ_dp(λ, β)

Calculate the mixed silicate extinction profile based on Donnan et al. (2022)
"""
function τ_dp(λ::AbstractFloat, β::AbstractFloat)

    # Simple cubic spline interpolation
    ext = Spline1D(DPlus_prof[1], DPlus_prof[2]; k=3).(λ)

    # Add 1.7 power law, as in PAHFIT
    return (1 - β) * ext + β * (9.7/λ)^1.7
end


function Extinction(ext::Float64, τ_97::Float64; screen::Bool=false)
    """
    Calculate the overall extinction factor
    """
    if screen
        return exp(-τ_97*ext)
    end
    return iszero(τ_97) ? 1. : (1 - exp(-τ_97*ext)) / (τ_97*ext)
end


############################################## FITTING FUNCTIONS #############################################


"""
    fit_spectrum(λ, params, n_dust_cont, n_dust_features, extinction_curve, extinction_screen;
        return_components=return_components, verbose=verbose)

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
- `verbose::Bool=false`: Whether or not to print out status messages
"""
function fit_spectrum(λ::Vector{<:AbstractFloat}, params::Vector{<:AbstractFloat}, n_dust_cont::Integer, n_dust_features::Integer,
    extinction_curve::String, extinction_screen::Bool; return_components::Bool=false, verbose::Bool=false)

    # Prepare outputs
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
    comps["extinction"] = Extinction.(ext_curve, params[pᵢ], screen=extinction_screen)
    contin .*= comps["extinction"]
    pᵢ += 2

    # Return components if necessary
    if return_components
        return contin, comps
    end
    return contin

end


"""
    fit_line_residuals(λ, params, n_lines, n_voff_tied, voff_tied_key, line_tied, line_profiles,
        n_flow_voff_tied, flow_voff_tied_key, line_flow_tied, line_flow_profiles, line_restwave,
        flexible_wavesol, tie_voigt_mixing; return_components=return_components, verbose=verbose)

Create a model of the emission lines at the given wavelengths `λ`, given the parameter vector `params`.

Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments
- `λ::Vector{<:AbstractFloat}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:AbstractFloat}`: Parameter vector. Parameters should be ordered as: 
    `[tied velocity offsets, tied flow velocity offsets, tied voigt mixing, 
    (amp[, voff], FWHM[, h3, h4, η], [flow_amp, flow_voff, flow_FWHM, flow_h3, flow_h4, flow_η] for each line)]`
- `n_lines::Integer`: Number of lines being fit
- `n_voff_tied::Integer`: Number of tied velocity offsets
- `voff_tied_key::Vector{String}`: Unique identifiers for each tied velocity offset parameter
- `line_tied::Vector{Union{String,Nothing}}`: Vector, length of n_lines, giving the identifier corresponding to
    the values in `voff_tied_key` that corresponds to which tied velocity offset should be used for a given line,
    if any.
- `line_profiles::Vector{Symbol}`: Vector, length of n_lines, that gives the profile type that should be used to
    fit each line. The profiles should be one of `:Gaussian`, `:Lorentzian`, `:GaussHermite`, or `:Voigt`
- `n_flow_voff_tied::Integer`: Same as `n_voff_tied`, but for inflow/outflow line components
- `flow_voff_tied_key::Vector{String}`: Same as `voff_tied_key`, but for inflow/outflow line components
- `line_flow_tied::Vector{Union{String,Nothing}}`: Same as `line_tied`, but for inflow/outflow line components
- `line_flow_profiles::Vector{Union{Symbol,Nothing}}`: Same as `line_profiles`, but for inflow/outflow line components
- `line_restwave::Vector{<:AbstractFloat}`: Vector, length of n_lines, giving the rest wavelengths of each line
- `flexible_wavesol::Bool`: Whether or not to allow small variations in tied velocity offsets, to account for a poor
    wavelength solution in the data
- `tie_voigt_mixing::Bool`: Whether or not to tie the mixing parameters of all Voigt profiles together
- `return_components::Bool=false`: Whether or not to return the individual components of the fit as a dictionary, in 
    addition to the overall fit
- `verbose::Bool=false`: Whether or not to print out status messages
"""
function fit_line_residuals(λ::Vector{<:AbstractFloat}, params::Vector{<:AbstractFloat}, n_lines::Integer, n_voff_tied::Integer, 
    voff_tied_key::Vector{String}, line_tied::Vector{Union{String,Nothing}}, line_profiles::Vector{Symbol}, 
    n_flow_voff_tied::Integer, flow_voff_tied_key::Vector{String}, line_flow_tied::Vector{Union{String,Nothing}},
    line_flow_profiles::Vector{Union{Symbol,Nothing}}, line_restwave::Vector{<:AbstractFloat}, 
    flexible_wavesol::Bool, tie_voigt_mixing::Bool; return_components::Bool=false, verbose::Bool=false)

    # Prepare outputs
    comps = Dict{String, Vector{Float64}}()
    contin = zeros(Float64, length(λ))

    # Skip ahead of the tied velocity offsets of the lines and flow components
    pᵢ = n_voff_tied + n_flow_voff_tied + 1
    # If applicable, skip ahead of the tied voigt mixing
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
            # Unpack the components of the line
            voff = params[pᵢ+1]
            fwhm = params[pᵢ+2]
            msg = "$(params[pᵢ]), $(params[pᵢ+1]) (united), $(params[pᵢ+2])"
            if line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = params[pᵢ+3]
                h4 = params[pᵢ+4]
                msg *= ", $(params[pᵢ+3]), $(params[pᵢ+4])"
            elseif line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
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
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and voff_tied_key
            vwhere = findfirst(x -> x == line_tied[k], voff_tied_key)
            voff_series = params[vwhere]
            voff_indiv = params[pᵢ+1]
            # Add velocity shifts of the tied lines and the individual offsets together
            voff = voff_series + voff_indiv
            fwhm = params[pᵢ+2]
            msg = "$(params[pᵢ]), $(params[vwhere]) (tied) + $(params[pᵢ+1]) (united), $(params[pᵢ+2])"
            if line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = params[pᵢ+3]
                h4 = params[pᵢ+4]
                msg *= ", $(params[pᵢ+3]), $(params[pᵢ+4])"
            elseif line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
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
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and voff_tied_key
            vwhere = findfirst(x -> x == line_tied[k], voff_tied_key)
            voff = params[vwhere]
            fwhm = params[pᵢ+1]
            # (dont add any individual voff components)
            msg = "$(params[pᵢ]), $(params[vwhere]) (tied), $(params[pᵢ+1])"
            if line_profiles[k] == :GaussHermite
                # Get additional h3, h4 components
                h3 = params[pᵢ+2]
                h4 = params[pᵢ+3]
                msg *= ", $(params[pᵢ+2]), $(params[pᵢ+3])"
            elseif line_profiles[k] == :Voigt
                # Get additional mixing component, either from the tied position or the 
                # individual position
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
        elseif line_profiles[k] == :Lorentzian
            comps["line_$k"] = Lorentzian.(λ, params[pᵢ], mean_μm, fwhm_μm)
        elseif line_profiles[k] == :GaussHermite
            comps["line_$k"] = GaussHermite.(λ, params[pᵢ], mean_μm, fwhm_μm, h3, h4)
        elseif line_profiles[k] == :Voigt
            comps["line_$k"] = Voigt.(λ, params[pᵢ], mean_μm, fwhm_μm, η)
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

        # Repeat EVERYTHING, minus the flexible_wavesol, for the inflow/outflow components
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
            elseif line_flow_profiles[k] == :Lorentzian
                comps["line_$(k)_flow"] = Lorentzian.(λ, params[pᵢ], flow_mean_μm, flow_fwhm_μm)
            elseif line_profiles[k] == :GaussHermite
                comps["line_$(k)_flow"] = GaussHermite.(λ, params[pᵢ], flow_mean_μm, flow_fwhm_μm, flow_h3, flow_h4)
            elseif line_profiles[k] == :Voigt
                comps["line_$(k)_flow"] = Voigt.(λ, params[pᵢ], flow_mean_μm, flow_fwhm_μm, flow_η)
            else
                error("Unrecognized flow line profile $(line_profiles[k])!")
            end

            # Add the inflow/outflow component into the overall model
            contin .+= comps["line_$(k)_flow"]
            # Advance the parameter vector index by the appropriate amount        
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

    # Return components if necessary
    if return_components
        return contin, comps
    end
    return contin

end


############################################## LINE PROFILES #############################################


"""
    Gaussian(x, A, μ, FWHM)

Evaluate a Gaussian profile at `x`, parameterized by the amplitude `A`, mean value `μ`, and 
full-width at half-maximum `FWHM`
"""
function Gaussian(x::AbstractFloat, A::AbstractFloat, μ::AbstractFloat, FWHM::AbstractFloat)
    # Reparametrize FWHM as dispersion σ
    σ = FWHM / (2√(2log(2)))
    return A * exp(-(x-μ)^2 / (2σ^2))
end


"""
    GaussHermite(x, A, μ, FWHM, h₃, h₄)

Evaluate a Gauss-Hermite quadrature at `x`, parametrized by the amplitude `A`, mean value `μ`,
full-width at half-maximum `FWHM`, 3rd moment / skewness `h₃`, and 4th moment / kurtosis `h₄`

See Riffel et al. (2010)
"""
function GaussHermite(x::AbstractFloat, A::AbstractFloat, μ::AbstractFloat, FWHM::AbstractFloat, h₃::AbstractFloat, h₄::AbstractFloat)

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


"""
    Lorentzian(x, A, μ, FWHM)

Evaluate a Lorentzian profile at `x`, parametrized by the amplitude `A`, mean value `μ`,
and full-width at half-maximum `FWHM`
"""
function Lorentzian(x::AbstractFloat, A::AbstractFloat, μ::AbstractFloat, FWHM::AbstractFloat)
    return A/π * (FWHM/2) / ((x-μ)^2 + (FWHM/2)^2)
end


"""
    Voigt(x, A, μ, FWHM, η)

Evaluate a pseudo-Voigt profile at `x`, parametrized by the amplitude `A`, mean value `μ`,
full-width at half-maximum `FWHM`, and mixing ratio `η`
"""
function Voigt(x::AbstractFloat, A::AbstractFloat, μ::AbstractFloat, FWHM::AbstractFloat, η::AbstractFloat)

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