#=
This file is not intended to be directly accessed by the user when fitting
IFU cubes. Rather, it contains miscellaneous utility functions to aid in correcting
and fitting data.  As such, nothing in this file is exported, so if the user wishes
to access it they must use the "Loki" prefix.
=#

# CONSTANTS

const C_KMS::Float64 = 299792.458          # Speed of light in km/s
const h_ERGS::Float64 = 6.62607015e-27     # Planck constant in erg*s
const kB_ERGK::Float64 = 1.380649e-16      # Boltzmann constant in erg/K

# First constant for Planck function, in MJy/sr/μm:
# 2hν^3/c^2 = 2h(c/λ)^3/c^2 = (2h/c^2 erg/s/cm^2/Hz/sr) * (1e23 Jy per erg/s/cm^2/Hz) / (1e6 MJy/Jy) * (c * 1e9 μm/km)^3 / (λ μm)^3
const Bν_1::Float64 = 2h_ERGS/(C_KMS*1e5)^2 * 1e23 / 1e6 * (C_KMS*1e9)^3

# Second constant for Planck function, in μm*K  
# hν/kT = hc/λkT = (hc/k cm*K) * (1e4 μm/cm) / (λ μm)
const Bν_2::Float64 = h_ERGS*(C_KMS*1e5) / kB_ERGK * 1e4

# Wein's law constant of proportionality in μm*K
const b_Wein::Float64 = 2897.771955        

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


# Save the Donnan et al. 2022 profile as a constant
const DP_prof = silicate_dp()
const DP_interp = Spline1D(DP_prof[1], DP_prof[2]; k=3)

# Save the Chiar+Tielens 2005 profile as a constant
const CT_prof = silicate_ct()
const CT_interp = Spline1D(CT_prof[1], CT_prof[2]; k=3)

# Save the OHM 1992 profile as a constant
const OHM_prof = silicate_ohm()
const OHM_interp = Spline1D(OHM_prof[1], OHM_prof[2]; k=3)

# Save the Smith+2006 PAH templates as constants
const SmithTemps = read_smith_temps()
const Smith3_interp = Spline1D(SmithTemps[1], SmithTemps[2]; k=3)
const Smith4_interp = Spline1D(SmithTemps[3], SmithTemps[4]; k=3)

# Save the Ice+CH optical depth template as a constant
const IceCHTemp = read_ice_ch_temps()
const Ice_interp = Spline1D(IceCHTemp[1], IceCHTemp[2]; k=3)
const CH_interp = Spline1D(IceCHTemp[3], IceCHTemp[4]; k=3)

########################################### UTILITY FUNCTIONS ###############################################


"""
    sumdim(array, dims, nan=true)

Sum `array` along specific dimensions `dims`, optinally ignoring nans, and dropping those dimensions

# Example
```jldoctest
julia> sumdim([1 2; 3 NaN], 1)
2-element Vector{Float64}:
 4.0
 2.0
```
"""
@inline sumdim(array, dims; nan=true) = dropdims((nan ? nansum : sum)(array, dims=dims), dims=dims)


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
@inline extend(array1d, shape) = repeat(reshape(array1d, (1,1,length(array1d))), outer=[shape...,1])


"""
    quadratic_interp_func(x, λ, I)

Function to interpolate data with least squares quadratic fitting

# Arguments
- `x::Real`: The wavelength position to interpolate the intensity at
- `λ::Vector{<:Real}`: The wavelength vector
- `I::Vector{<:Real}`: The intensity vector
"""
function quadratic_interp_func(x::Real, λ::Vector{<:Real}, I::Vector{<:Real})::Real
    # Get the index of the value in the wavelength vector closest to x
    ind = findmin(abs.(λ .- x))[2]
    # Get the indices one before / after
    lo = ind - 1
    hi = ind + 2
    # Edge case for the left edge
    while lo ≤ 0
        lo += 1
        hi += 1
    end
    # Edge case for the right edge
    while hi > length(λ)
        hi -= 1
        lo -= 1
    end
    # A matrix with λ^2, λ, 1
    A = [λ[lo:hi].^2 λ[lo:hi] ones(4)]
    # y vector
    y = I[lo:hi]
    # Solve the least squares problem
    param = A \ y

    param[1]*x^2 + param[2]*x + param[3]
end


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
@inline rest_frame(λ, z) = @. λ / (1 + z)


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
@inline observed_frame(λ, z) = @. λ * (1 + z)


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
@inline Doppler_shift_λ(λ₀, v) = λ₀ * √((1+v/C_KMS)/(1-v/C_KMS))


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
@inline Doppler_shift_v(λ, λ₀) = ((λ/λ₀)^2 - 1)/((λ/λ₀)^2 + 1) * C_KMS


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
@inline Doppler_width_v(Δλ, λ₀) = Δλ / λ₀ * C_KMS


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
@inline Doppler_width_λ(Δv, λ₀) = Δv / C_KMS * λ₀


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
@inline ∫Gaussian(A, FWHM) = √(π / (4log(2))) * A * FWHM


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
@inline ∫Lorentzian(A) = A


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
@inline ∫Drude(A, FWHM) = π/2 * A * FWHM


"""
    MJysr_to_cgs(MJy, λ)

Convert specific intensity in MegaJanskys per steradian to CGS units 
-> erg s^-1 cm^-2 μm^-1 sr^-1, given the wavelength `λ` in μm

This converts from intensity per unit frequency to per unit wavelength (Fλ = Fν|dλ/dν| = Fν * c/λ^2)
"""
@inline MJysr_to_cgs(MJy, λ) = MJy * 1e6 * 1e-23 * (C_KMS * 1e9) / λ^2


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
@inline function ln_likelihood(data::Vector{T}, model::Vector{T}, err::Vector{T}) where {T<:Real}
    -0.5 * sum(@. (data - model)^2 / err^2 + log(2π * err^2))
end


"""
    ln_prior(p, priors)

Internal helper function to calculate the log of the prior probability for all the parameters,
specifically for the line residual fitting.
Most priors will be uniform, i.e. constant, finite prior values as long as the parameter is inbounds.
Priors will be -Inf if any parameter goes out of bounds.  
"""
@inline function ln_prior(p, priors)
    # sum the log prior distribution of each parameter
    sum([logpdf(priors[i], p[i]) for i ∈ eachindex(p)])
end


"""
    negln_probability(p, grad, λ, Inorm, σnorm, cube_fitter, ext_curve, priors)

Internal helper function to calculate the negative of the log of the probability,
for the line residual fitting.

ln(probability) = ln(likelihood) + ln(prior)
"""
function negln_probability(p, grad, λ, Inorm, σnorm, n_lines, n_comps, lines, n_kin_tied, tied_kinematics,
    flexible_wavesol, tie_voigt_mixing, ext_curve, priors)
    # Check for gradient
    @assert length(grad) == 0 "Gradient-based solvers are not currently supported!"
    # First compute the model
    model = fit_line_residuals(λ, p, n_lines, n_comps, lines, n_kin_tied, tied_kinematics, flexible_wavesol, 
        tie_voigt_mixing, ext_curve)
    # Add the log of the likelihood (based on the model) and the prior distribution functions
    lnP = ln_likelihood(Inorm, model, σnorm) + ln_prior(p, priors)
    # return the NEGATIVE log of the probability
    -lnP 
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
function hermite(x::Real, n::Integer)
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
@inline function Blackbody_ν(λ::Real, Temp::Real)
    Bν_1/λ^3 / (exp(Bν_2/(λ*Temp))-1)
end


"""
    Wein(Temp)

Return the peak wavelength (in μm) of a Blackbody spectrum at a given temperature `Temp`,
using Wein's Displacement Law.
"""
@inline function Wein(Temp::Real)
    b_Wein / Temp
end


################################################# PAH PROFILES ################################################


"""
    Drude(x, A, μ, FWHM)

Calculate a Drude profile at location `x`, with amplitude `A`, central value `μ`, and full-width at half-max `FWHM`

Function adapted from PAHFIT: Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
"""
@inline function Drude(x::Real, A::Real, μ::Real, FWHM::Real)
    A * (FWHM/μ)^2 / ((x/μ - μ/x)^2 + (FWHM/μ)^2)
end

############################################## EXTINCTION PROFILES #############################################


"""
    τ_kvt(λ, β)

Calculate the mixed silicate extinction profile based on Kemper, Vriend, & Tielens (2004) 

Function adapted from PAHFIT: Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)
"""
function τ_kvt(λ::Real, β::Real)

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


function τ_ct(λ::Real)

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


function τ_ohm(λ::Real)

    ext = OHM_interp(λ)
    _, wh = findmin(x -> abs(x - 9.7), OHM_prof[1])
    ext /= OHM_prof[2][wh]

    ext
end


"""
    τ_dp(λ, β)

Calculate the mixed silicate extinction profile based on Donnan et al. (2022)
"""
function τ_dp(λ::Real, β::Real)

    # Simple cubic spline interpolation
    ext = DP_interp(λ)

    # Add 1.7 power law, as in PAHFIT
    (1 - β) * ext + β * (9.8/λ)^1.7
end


"""
    τ_ice(λ)

Calculate the ice extinction profiles
"""
function τ_ice(λ::Real)

    # Simple cubic spline interpolation
    ext = Ice_interp(λ)
    ext /= maximum(IceCHTemp[2])

    ext
end


"""
    τ_ch(λ)

Calculate the CH extinction profiles
"""
function τ_ch(λ::Real)

    # Simple cubic spline interpolation
    ext = CH_interp(λ)
    ext /= maximum(IceCHTemp[4])

    ext
end


"""
    Extinction(ext, τ_97; screen)

Calculate the overall extinction factor
"""
function Extinction(ext::Real, τ_97::Real; screen::Bool=false)
    if screen
        exp(-τ_97*ext)
    else
        iszero(τ_97) ? 1. : (1 - exp(-τ_97*ext)) / (τ_97*ext)
    end
end


"""
    make_bins(array)

Calculate the bin edges and bin widths for an array given the distances between each entry

NOTES:
This function has been taken and adapted from the SpectRes python package, https://github.com/ACCarnall/SpectRes
"""
function make_bins(array::AbstractArray)
    # Get the bin edges 
    edges = zeros(size(array,1)+1)
    edges[1] = array[1] - (array[2] - array[1])/2
    edges[end] = array[end] + (array[end] - array[end-1])/2
    edges[2:end-1] .= (array[2:end] + array[1:end-1])/2
    # Get the bin widths
    widths = zeros(size(array,1))
    widths[end] = array[end] - array[end-1]
    widths[1:end-1] = edges[2:end-1] - edges[1:end-2]
    # Return the edges and widths
    edges, widths
end


"""
    resample_conserving_flux(new_wave, old_wave, flux, err=nothing, mask=nothing; fill=NaN)

Resample a spectrum (and optionally errors) onto a new wavelength grid, while convserving the flux.

NOTES:
This function has been taken and adapted from the SpectRes python package, https://github.com/ACCarnall/SpectRes
"""
function resample_conserving_flux(new_wave::AbstractVector, old_wave::AbstractVector, flux::AbstractArray, 
    err::Union{AbstractArray,Nothing}=nothing, mask::Union{AbstractArray,Nothing}=nothing; fill::Real=NaN)

    # Find the edges and widths of the new wavelength bins given the old ones
    old_edges, old_widths = make_bins(old_wave)
    new_edges, new_widths = make_bins(new_wave)

    # Instantiate flux and error arrays with zeros
    new_fluxes = zeros((size(flux[.., 1])..., size(new_wave)...))
    if !isnothing(err)
        if size(err) ≠ size(flux)
            error("error must be the same shape as flux")
        else
            new_errs = copy(new_fluxes)
        end
    end
    if !isnothing(mask)
        if size(mask) ≠ size(flux)
            error("mask must be the same shape as flux")
        else
            new_mask = falses(size(new_fluxes))
        end
    end

    start = 1
    stop = 1
    warned = false

    # Calculate new flux and uncertainty values, looping over new bins
    for j ∈ 1:size(new_wave, 1)
        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_edges[j] < old_edges[1]) || (new_edges[j+1] > old_edges[end])
            new_fluxes[.., j] .= fill
            if !isnothing(err)
                new_errs[.., j] .= fill
            end
            if !isnothing(mask)
                new_mask[.., j] .= 1
            end
            if (j == 1 || j == size(new_wave, 1)) && !warned
                @warn "\nSpectres: new_wave contains values outside the range " *
                      "in old_wave, new_fluxes and new_errs will be filled " *
                      "with the value set in the 'fill' keyword argument. \n"
                warned = true
            end
            continue
        end
    
        # Find first old bin which is partially covered by the new bin
        while old_edges[start+1] ≤ new_edges[j]
            start += 1
        end

        # Find last old bin which is partially covered by the new bin
        while old_edges[stop+1] < new_edges[j+1]
            stop += 1
        end

        # If new bin is fully inside an old bin start and stop are equal
        if start == stop
            new_fluxes[.., j] .= flux[.., start]
            if !isnothing(err)
                new_errs[.., j] .= err[.., start]
            end
            if !isnothing(mask)
                new_mask[.., j] .= mask[.., start]
            end
        # Otherwise multiply the first and last old bin widths by P_ij
        else
            start_factor = (old_edges[start+1] - new_edges[j]) / (old_edges[start+1] - old_edges[start])
            stop_factor = (new_edges[j+1] - old_edges[stop]) / (old_edges[stop+1] - old_edges[stop])

            old_widths[start] *= start_factor
            old_widths[stop] *= stop_factor

            # Populate new fluxes and errors
            f_widths = old_widths[start:stop] .* permutedims(flux[.., start:stop], (ndims(flux), range(1, ndims(flux)-1)...))
            new_fluxes[.., j] .= sumdim(f_widths, 1, nan=false)            # -> preserve NaNs
            new_fluxes[.., j] ./= sum(old_widths[start:stop])

            if !isnothing(err)
                e_widths = old_widths[start:stop] .* permutedims(err[.., start:stop], (ndims(err), range(1, ndims(err)-1)...))
                new_errs[.., j] .= .√(sumdim(e_widths.^2, 1, nan=false))   # -> preserve NaNS
                new_errs[.., j] ./= sum(old_widths[start:stop])
            end

            # Put the old bin widths back to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= stop_factor

            # Combine the mask data with a bitwise or
            if !isnothing(mask)
                new_mask[.., j] .= [any(mask[xi, start:stop]) for xi ∈ CartesianIndices(selectdim(mask, 3, j))]
            end
        end
    end

    # Only return the quantities that were given in the input
    if !isnothing(err) && !isnothing(mask)
        new_fluxes, new_errs, new_mask
    elseif !isnothing(err)
        new_fluxes, new_errs
    elseif !isnothing(mask)
        new_fluxes, new_mask
    else
        new_fluxes
    end

end


############################################## FITTING FUNCTIONS #############################################


"""
    fit_spectrum(λ, params, n_dust_cont, extinction_curve, extinction_screen, fit_sil_emission;
        return_components=return_components)

Create a model of the continuum (including stellar+dust continuum, PAH features, and extinction, excluding emission lines)
at the given wavelengths `λ`, given the parameter vector `params`.

Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments
- `λ::Vector{<:AbstractFloat}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:AbstractFloat}`: Parameter vector. 
- `n_dust_cont::Integer`: Number of dust continuum profiles to be fit
- `n_dust_feat::Integer`: Number of PAH dust features to be fit
- `extinction_curve::String`: The type of extinction curve to use, "kvt" or "d+"
- `extinction_screen::Bool`: Whether or not to use a screen model for the extinction curve
- `fit_sil_emission::Bool`: Whether or not to fit silicate emission with a hot dust continuum component
- `return_components::Bool=false`: Whether or not to return the individual components of the fit as a dictionary, in 
    addition to the overall fit
"""
function fit_spectrum(λ::Vector{T}, params::Vector{T}, n_dust_cont::Integer, extinction_curve::String, 
    extinction_screen::Bool, fit_sil_emission::Bool, return_components::Bool) where {T<:Real}

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

    # Extinction 
    if extinction_curve == "d+"
        ext_curve = τ_dp.(λ, params[pᵢ+3])
    elseif extinction_curve == "kvt"
        ext_curve = τ_kvt.(λ, params[pᵢ+3])
    elseif extinction_curve == "ct"
        ext_curve = τ_ct.(λ)
    elseif extinction_curve == "ohm"
        ext_curve = τ_ohm.(λ)
    else
        error("Unrecognized extinction curve: $extinction_curve")
    end
    comps["extinction"] = Extinction.(ext_curve, params[pᵢ], screen=extinction_screen)

    # Ice+CH Absorption
    ext_ice = τ_ice.(λ)
    comps["abs_ice"] = Extinction.(ext_ice, params[pᵢ+1], screen=true)
    ext_ch = τ_ch.(λ)
    comps["abs_ch"] = Extinction.(ext_ch, params[pᵢ+2], screen=true)

    contin .*= comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"]
    pᵢ += 4

    if fit_sil_emission
        # Add Silicate emission from hot dust (amplitude, temperature, covering fraction, warm tau, cold tau)
        # Ref: Gallimore et al. 2010
        comps["hot_dust"] = params[pᵢ] .* Blackbody_ν.(λ, params[pᵢ+1])
        comps["hot_dust"] .*= (1 .- Extinction.(ext_curve, params[pᵢ+3], screen=true))
        comps["hot_dust"] .*= (1 .- params[pᵢ+2]) .* (1 .- Extinction.(ext_curve, params[pᵢ+4], screen=true))
        contin .+= comps["hot_dust"]
        pᵢ += 5
    end

    # Add Smith+2006 PAH templates
    pah3 = Smith3_interp.(λ)
    comps["pah_temp_3"] = params[pᵢ] .* pah3 / maximum(pah3)
    contin .+= comps["pah_temp_3"] .* comps["extinction"]
    pah4 = Smith4_interp.(λ)
    comps["pah_temp_4"] = params[pᵢ+1] .* pah4 / maximum(pah4)
    contin .+= comps["pah_temp_4"] .* comps["extinction"]
    # (Not affected by Ice+CH absorption)
    pᵢ += 2

    # Return components if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency --> not allocating the dictionary improves performance DRAMATICALLY
function fit_spectrum(λ::Vector{T}, params::Vector{T}, n_dust_cont::Integer, extinction_curve::String, 
    extinction_screen::Bool, fit_sil_emission::Bool) where {T<:Real}

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

    # Extinction 
    if extinction_curve == "d+"
        ext_curve = τ_dp.(λ, params[pᵢ+3])
    elseif extinction_curve == "kvt"
        ext_curve = τ_kvt.(λ, params[pᵢ+3])
    elseif extinction_curve == "ct"
        ext_curve = τ_ct.(λ)
    elseif extinction_curve == "ohm"
        ext_curve = τ_ohm.(λ)
    else
        error("Unrecognized extinction curve: $extinction_curve")
    end
    ext = Extinction.(ext_curve, params[pᵢ], screen=extinction_screen)

    # Ice+CH absorption
    ext_ice = τ_ice.(λ)
    abs_ice = Extinction.(ext_ice, params[pᵢ+1], screen=true)
    ext_ch = τ_ch.(λ)
    abs_ch = Extinction.(ext_ch, params[pᵢ+2], screen=true)

    contin .*= ext .* abs_ice .* abs_ch
    pᵢ += 4

    if fit_sil_emission
        # Add Silicate emission from hot dust (amplitude, temperature, covering fraction, warm tau, cold tau)
        # Ref: Gallimore et al. 2010
        hot_dust = params[pᵢ] .* Blackbody_ν.(λ, params[pᵢ+1])
        hot_dust .*= (1 .- Extinction.(ext_curve, params[pᵢ+3], screen=true))
        hot_dust .*= (1 .- params[pᵢ+2]) .* (1 .- Extinction.(ext_curve, params[pᵢ+4], screen=true))
        contin .+= hot_dust
        pᵢ += 5
    end

    # Add Smith+2006 PAH templates
    pah3 = Smith3_interp.(λ)
    contin .+= params[pᵢ] .* pah3 / maximum(pah3) .* ext
    pah4 = Smith4_interp.(λ)
    contin .+= params[pᵢ+1] .* pah4 / maximum(pah4) .* ext
    # (Not affected by Ice+CH absorption)
    pᵢ += 2

    contin

end


"""
    fit_pah_residuals(λ, params, n_dust_feat, return_components)
Create a model of the PAH features at the given wavelengths `λ`, given the parameter vector `params`.
Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)
# Arguments
- `λ::Vector{<:AbstractFloat}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:AbstractFloat}`: Parameter vector. Parameters should be ordered as: `(amp, center, fwhm) for each PAH profile`
- `n_dust_feat::Integer`: The number of PAH features that are being fit
- `ext_curve::Vector{<:AbstractFloat}`: The extinction curve that was fit using fit_spectrum
- `return_components::Bool`: Whether or not to return the individual components of the fit as a dictionary, in
    addition to the overall fit
"""
function fit_pah_residuals(λ::Vector{T}, params::Vector{T}, n_dust_feat::Integer,
    ext_curve::Vector{T}, return_components::Bool) where {T<:Real}

    # Prepare outputs
    comps = Dict{String, Vector{Float64}}()
    contin = zeros(Float64, length(λ))

    # Add dust features with drude profiles
    pᵢ = 1
    for j ∈ 1:n_dust_feat
        comps["dust_feat_$j"] = Drude.(λ, params[pᵢ:pᵢ+2]...)
        contin .+= comps["dust_feat_$j"]
        pᵢ += 3
    end

    # Apply extinction
    contin .*= ext_curve

    # Return components, if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency
function fit_pah_residuals(λ::Vector{T}, params::Vector{T}, n_dust_feat::Integer,
    ext_curve::Vector{T}) where {T<:Real}

    # Prepare outputs
    contin = zeros(Float64, length(λ))

    # Add dust features with drude profiles
    pᵢ = 1
    for j ∈ 1:n_dust_feat
        contin .+= Drude.(λ, params[pᵢ:pᵢ+2]...)
        pᵢ += 3
    end

    # Apply extinction
    contin .*= ext_curve

    contin

end


# Combine fit_spectrum and fit_pah_residuals to get the full continuum in one function (after getting the optimized parameters)
function fit_full_continuum(λ::Vector{T}, params::Vector{T}, n_dust_cont::Integer,
    n_dust_feat::Integer, extinction_curve::String, extinction_screen::Bool, fit_sil_emission::Bool) where {T<:Real}

    pars_1 = vcat(params[1:(2+2n_dust_cont+4+(fit_sil_emission ? 5 : 0))], [0., 0.])
    pars_2 = params[(3+2n_dust_cont+4+(fit_sil_emission ? 5 : 0)):end]

    contin_1, ccomps = fit_spectrum(λ, pars_1, n_dust_cont, extinction_curve, extinction_screen, fit_sil_emission, true)
    contin_2, pcomps = fit_pah_residuals(λ, pars_2, n_dust_feat, ccomps["extinction"], true)

    contin = contin_1 .+ contin_2
    comps = merge(ccomps, pcomps)

    contin, comps

end


"""
    fit_line_residuals(λ, params, n_lines, n_kin_tied, kin_tied_key, line_tied, line_profiles,
        n_acomp_kin_tied, acomp_kin_tied_key, line_acomp_tied, line_acomp_profiles, line_restwave,
        flexible_wavesol, tie_voigt_mixing, ext_curve; return_components=return_components) 

Create a model of the emission lines at the given wavelengths `λ`, given the parameter vector `params`.

Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments
- `λ::Vector{<:AbstractFloat}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:AbstractFloat}`: Parameter vector. Parameters should be ordered as: 
    `[tied velocity offsets, tied acomp velocity offsets, tied voigt mixing, 
    (amp[, voff], FWHM[, h3, h4, η], [acomp_amp, acomp_voff, acomp_FWHM, acomp_h3, acomp_h4, acomp_η] for each line)]`
- `n_lines::Integer`: Number of lines being fit
- `lines::TransitionLines`: Object containing information about each transition line being fit.
- `n_kin_tied::Integer`: Number of tied velocity offsets
- `n_acomp_kin_tied::Integer`: Same as `n_kin_tied`, but for additional line components
- `tied_kinematics::TiedKinematics`: Object containing information about tied kinematics parameters.
- `flexible_wavesol::Bool`: Whether or not to allow small variations in tied velocity offsets, to account for a poor
    wavelength solution in the data
- `tie_voigt_mixing::Bool`: Whether or not to tie the mixing parameters of all Voigt profiles together
- `ext_curve::Vector{<:AbstractFloat}`: The extinction curve fit with fit_spectrum
- `return_components::Bool=false`: Whether or not to return the individual components of the fit as a dictionary, in 
    addition to the overall fit
"""
function fit_line_residuals(λ::Vector{T}, params::Vector{T}, n_lines::S, n_comps::S, lines::TransitionLines, 
    n_kin_tied::Vector{S}, tied_kinematics::TiedKinematics, flexible_wavesol::Bool, tie_voigt_mixing::Bool, ext_curve::Vector{T}, 
    return_components::Bool) where {T<:Real,S<:Integer}

    # Prepare outputs
    comps = Dict{String, Vector{Float64}}()
    contin = zeros(Float64, length(λ))

    # Skip ahead of the tied kinematics of the lines and acomp components
    pᵢ = 2 * sum(n_kin_tied) + 1
    # If applicable, skip ahead of the tied voigt mixing
    if tie_voigt_mixing
        ηᵢ = pᵢ
        pᵢ += 1
    end

    # Add emission lines
    for k ∈ 1:n_lines
        # Check if voff is tied: if so, use the tied voff parameter, otherwise, use the line's own voff parameter
        amp = params[pᵢ]
        if isnothing(lines.tied[k, 1])
            # Unpack the components of the line
            voff = params[pᵢ+1]
            fwhm = params[pᵢ+2]
            pᵢ += 3

        elseif flexible_wavesol
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and kin_tied_key
            vwhere = findfirst(x -> x == lines.tied[k, 1], tied_kinematics.key[1])
            voff_series = params[vwhere]
            voff_indiv = params[pᵢ+1]
            # Add velocity shifts of the tied lines and the individual offsets together
            voff = voff_series + voff_indiv
            # Position of the tied fwhm that should be used
            fwhm = params[n_kin_tied[1]+vwhere]
            pᵢ += 2

        else
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and kin_tied_key
            vwhere = findfirst(x -> x == lines.tied[k, 1], tied_kinematics.key[1])
            voff = params[vwhere]
            fwhm = params[n_kin_tied[1]+vwhere]
            pᵢ += 1

        end

        if lines.profiles[k, 1] == :GaussHermite
            # Get additional h3, h4 components
            h3 = params[pᵢ]
            h4 = params[pᵢ+1]
            pᵢ += 2
        elseif lines.profiles[k, 1] == :Voigt
            # Get additional mixing component, either from the tied position or the 
            # individual position
            if !tie_voigt_mixing
                η = params[pᵢ]
                pᵢ += 1
            else
                η = params[ηᵢ]
            end
        end

        # Convert voff in km/s to mean wavelength in μm
        mean_μm = Doppler_shift_λ(lines.λ₀[k], voff)
        # Convert FWHM from km/s to μm
        fwhm_μm = Doppler_shift_λ(lines.λ₀[k], fwhm/2) - Doppler_shift_λ(lines.λ₀[k], -fwhm/2)
        # Evaluate line profile
        if lines.profiles[k, 1] == :Gaussian
            comps["line_$(k)_1"] = Gaussian.(λ, amp, mean_μm, fwhm_μm)
        elseif lines.profiles[k, 1] == :Lorentzian
            comps["line_$(k)_1"] = Lorentzian.(λ, amp, mean_μm, fwhm_μm)
        elseif lines.profiles[k, 1] == :GaussHermite
            comps["line_$(k)_1"] = GaussHermite.(λ, amp, mean_μm, fwhm_μm, h3, h4)
        elseif lines.profiles[k, 1] == :Voigt
            comps["line_$(k)_1"] = Voigt.(λ, amp, mean_μm, fwhm_μm, η)
        else
            error("Unrecognized line profile $(lines.profiles[k, 1])!")
        end

        # Add the line profile into the overall model
        contin .+= comps["line_$(k)_1"]

        # Repeat EVERYTHING, minus the flexible_wavesol, for the additional components
        for j ∈ 2:n_comps
            if !isnothing(lines.profiles[k, j])
                # Parametrize acomp amplitude in terms of the default line amplitude times some fractional value
                # this way, we can constrain the acomp_amp parameter to be from (0,1) to ensure the acomp amplitude is always
                # less than the line amplitude
                acomp_amp = amp * params[pᵢ]

                if isnothing(lines.tied[k, j])
                    # Parametrize the acomp voff in terms of the default line voff plus some difference value
                    # this way, we can constrain the acomp component to be within +/- some range from the line itself
                    acomp_voff = voff + params[pᵢ+1]
                    # Parametrize the acomp FWHM in terms of the default line FWHM times some fractional value
                    # this way, we can constrain the acomp_fwhm parameter to be > 0 to ensure the acomp FWHM is always
                    # greater than the line FWHM
                    acomp_fwhm = fwhm * params[pᵢ+2]
                    pᵢ += 3

                else
                    vwhere = findfirst(x -> x == lines.tied[k, j], tied_kinematics.key[j])
                    acomp_voff = voff + params[2*sum(n_kin_tied[1:(j-1)])+vwhere]
                    acomp_fwhm = fwhm * params[2*sum(n_kin_tied[1:(j-1)])+n_kin_tied[j]+vwhere]
                    pᵢ += 1

                end

                if lines.profiles[k, j] == :GaussHermite
                    acomp_h3 = params[pᵢ]
                    acomp_h4 = params[pᵢ+1]
                    pᵢ += 2
                elseif lines.profiles[k, j] == :Voigt
                    if !tie_voigt_mixing
                        acomp_η = params[pᵢ]
                        pᵢ += 1
                    else
                        acomp_η = params[ηᵢ]
                    end
                end

                # Convert voff in km/s to mean wavelength in μm
                acomp_mean_μm = Doppler_shift_λ(lines.λ₀[k], acomp_voff)
                # Convert FWHM from km/s to μm
                acomp_fwhm_μm = Doppler_shift_λ(lines.λ₀[k], acomp_fwhm/2) - Doppler_shift_λ(lines.λ₀[k], -acomp_fwhm/2)
                # Evaluate line profile
                if lines.profiles[k, j] == :Gaussian
                    comps["line_$(k)_$(j)"] = Gaussian.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm)
                elseif lines.profiles[k, j] == :Lorentzian
                    comps["line_$(k)_$(j)"] = Lorentzian.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm)
                elseif lines.profiles[k, j] == :GaussHermite
                    comps["line_$(k)_$(j)"] = GaussHermite.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm, acomp_h3, acomp_h4)
                elseif lines.profiles[k, j] == :Voigt
                    comps["line_$(k)_$(j)"] = Voigt.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm, acomp_η)
                else
                    error("Unrecognized acomp line profile $(lines.profiles[k, j])!")
                end

                # Add the additional component into the overall model
                contin .+= comps["line_$(k)_$(j)"]

            end
        end

    end
    # Apply extinction
    contin .*= ext_curve

    # Return components if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency --> not allocating the dictionary improves performance DRAMATICALLY
function fit_line_residuals(λ::Vector{T}, params::Vector{T}, n_lines::S, n_comps::S, lines::TransitionLines, 
    n_kin_tied::Vector{S}, tied_kinematics::TiedKinematics, flexible_wavesol::Bool, tie_voigt_mixing::Bool, 
    ext_curve::Vector{T}) where {T<:Real,S<:Integer}

    # Prepare outputs
    contin = zeros(Float64, length(λ))

    # Skip ahead of the tied kinematics of the lines and acomp components
    pᵢ = 2 * sum(n_kin_tied) + 1
    # If applicable, skip ahead of the tied voigt mixing
    if tie_voigt_mixing
        ηᵢ = pᵢ
        pᵢ += 1
    end

    # Add emission lines
    for k ∈ 1:n_lines
        # Check if voff is tied: if so, use the tied voff parameter, otherwise, use the line's own voff parameter
        amp = params[pᵢ]
        if isnothing(lines.tied[k, 1])
            # Unpack the components of the line
            voff = params[pᵢ+1]
            fwhm = params[pᵢ+2]
            pᵢ += 3

        elseif flexible_wavesol
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and kin_tied_key
            vwhere = findfirst(x -> x == lines.tied[k, 1], tied_kinematics.key[1])
            voff_series = params[vwhere]
            voff_indiv = params[pᵢ+1]
            # Add velocity shifts of the tied lines and the individual offsets together
            voff = voff_series + voff_indiv
            # Position of the tied fwhm that should be used
            fwhm = params[n_kin_tied[1]+vwhere]
            pᵢ += 2

        else
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and kin_tied_key
            vwhere = findfirst(x -> x == lines.tied[k, 1], tied_kinematics.key[1])
            voff = params[vwhere]
            fwhm = params[n_kin_tied[1]+vwhere]
            pᵢ += 1

        end

        if lines.profiles[k, 1] == :GaussHermite
            # Get additional h3, h4 components
            h3 = params[pᵢ]
            h4 = params[pᵢ+1]
            pᵢ += 2
        elseif lines.profiles[k, 1] == :Voigt
            # Get additional mixing component, either from the tied position or the 
            # individual position
            if !tie_voigt_mixing
                η = params[pᵢ]
                pᵢ += 1
            else
                η = params[ηᵢ]
            end
        end

        # Convert voff in km/s to mean wavelength in μm
        mean_μm = Doppler_shift_λ(lines.λ₀[k], voff)
        # Convert FWHM from km/s to μm
        fwhm_μm = Doppler_shift_λ(lines.λ₀[k], fwhm/2) - Doppler_shift_λ(lines.λ₀[k], -fwhm/2)
        # Evaluate line profile
        if lines.profiles[k, 1] == :Gaussian
            contin .+= Gaussian.(λ, amp, mean_μm, fwhm_μm)
        elseif lines.profiles[k, 1] == :Lorentzian
            contin .+= Lorentzian.(λ, amp, mean_μm, fwhm_μm)
        elseif lines.profiles[k, 1] == :GaussHermite
            contin .+= GaussHermite.(λ, amp, mean_μm, fwhm_μm, h3, h4)
        elseif lines.profiles[k, 1] == :Voigt
            contin .+= Voigt.(λ, amp, mean_μm, fwhm_μm, η)
        else
            error("Unrecognized line profile $(lines.profiles[k, 1])!")
        end

        # Repeat EVERYTHING, minus the flexible_wavesol, for the additional components
        for j ∈ 2:n_comps
            if !isnothing(lines.profiles[k, j])
                # Parametrize acomp amplitude in terms of the default line amplitude times some fractional value
                # this way, we can constrain the acomp_amp parameter to be from (0,1) to ensure the acomp amplitude is always
                # less than the line amplitude
                acomp_amp = amp * params[pᵢ]

                if isnothing(lines.tied[k, j])
                    # Parametrize the acomp voff in terms of the default line voff plus some difference value
                    # this way, we can constrain the acomp component to be within +/- some range from the line itself
                    acomp_voff = voff + params[pᵢ+1]
                    # Parametrize the acomp FWHM in terms of the default line FWHM times some fractional value
                    # this way, we can constrain the acomp_fwhm parameter to be > 0 to ensure the acomp FWHM is always
                    # greater than the line FWHM
                    acomp_fwhm = fwhm * params[pᵢ+2]
                    pᵢ += 3

                else
                    vwhere = findfirst(x -> x == lines.tied[k, j], tied_kinematics.key[j])
                    acomp_voff = voff + params[2*sum(n_kin_tied[1:(j-1)])+vwhere]
                    acomp_fwhm = fwhm * params[2*sum(n_kin_tied[1:(j-1)])+n_kin_tied[j]+vwhere]
                    pᵢ += 1

                end

                if lines.profiles[k, j] == :GaussHermite
                    acomp_h3 = params[pᵢ]
                    acomp_h4 = params[pᵢ+1]
                    pᵢ += 2
                elseif lines.profiles[k, j] == :Voigt
                    if !tie_voigt_mixing
                        acomp_η = params[pᵢ]
                        pᵢ += 1
                    else
                        acomp_η = params[ηᵢ]
                    end
                end

                # Convert voff in km/s to mean wavelength in μm
                acomp_mean_μm = Doppler_shift_λ(lines.λ₀[k], acomp_voff)
                # Convert FWHM from km/s to μm
                acomp_fwhm_μm = Doppler_shift_λ(lines.λ₀[k], acomp_fwhm/2) - Doppler_shift_λ(lines.λ₀[k], -acomp_fwhm/2)
                # Evaluate line profile
                if lines.profiles[k, j] == :Gaussian
                    contin .+= Gaussian.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm)
                elseif lines.profiles[k, j] == :Lorentzian
                    contin .+= Lorentzian.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm)
                elseif lines.profiles[k, j] == :GaussHermite
                    contin .+= GaussHermite.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm, acomp_h3, acomp_h4)
                elseif lines.profiles[k, j] == :Voigt
                    contin .+= Voigt.(λ, acomp_amp, acomp_mean_μm, acomp_fwhm_μm, acomp_η)
                else
                    error("Unrecognized acomp line profile $(lines.profiles[k, j])!")
                end

            end
        end

    end
    # Apply extinction
    contin .*= ext_curve

    contin

end


"""
    calculate_extra_parameters(λ, I, σ, n_dust_cont, n_dust_feat, extinction_curve, extinction_screen,
        fit_sil_emission, n_lines, n_acomps, n_kin_tied, kin_tied_key, line_tied, line_profiles,
        n_acomp_kin_tied, acomp_kin_tied_key, line_acomp_tied, line_acomp_profiles, line_restwave, flexible_wavesol,
        tie_voigt_mixing, popt_c, popt_l, perr_c, perr_l, mask_lines, continuum)

Calculate extra parameters that are not fit, but are nevertheless important to know, for a given spaxel.
Currently this includes the integrated intensity and signal to noise ratios of dust features and emission lines.
"""
function calculate_extra_parameters(λ::Vector{T}, I::Vector{T}, σ::Vector{T}, n_dust_cont::Integer, 
    n_dust_feat::Integer, extinction_curve::String, extinction_screen::Bool, fit_sil_emission::Bool, 
    n_lines::Integer, n_acomps::Integer, n_comps::Integer, lines::TransitionLines, n_kin_tied::Vector{S}, 
    tied_kinematics::TiedKinematics, flexible_wavesol::Bool, tie_voigt_mixing::Bool, popt_c::Vector{T}, 
    popt_l::Vector{T}, perr_c::Vector{T}, perr_l::Vector{T}, extinction::Vector{T}, mask_lines::BitVector, 
    continuum::Vector{T}) where {T<:Real,S<:Integer}

    @debug "Calculating extra parameters"

    # Get the average wavelength resolution, and divide by 10 to subsample pixels
    # (this is only used to calculate the peak location of the profiles, and thus the peak intensity)
    Δλ = mean(diff(λ)) / 10

    # Normalization
    N = Float64(abs(nanmaximum(I)))
    N = N ≠ 0. ? N : 1.
    @debug "Normalization: $N"

    # Loop through dust features
    p_dust = zeros(3n_dust_feat)
    p_dust_err = zeros(3n_dust_feat)
    pₒ = 1
    # Initial parameter vector index where dust profiles start
    pᵢ = 3 + 2n_dust_cont + 4 + (fit_sil_emission ? 5 : 0)

    for ii ∈ 1:n_dust_feat

        # unpack the parameters
        A, μ, fwhm = popt_c[pᵢ:pᵢ+2]
        A_err, μ_err, fwhm_err = perr_c[pᵢ:pᵢ+2]
        # Convert peak intensity to CGS units (erg s^-1 cm^-2 μm^-1 sr^-1)
        A_cgs = MJysr_to_cgs(A, μ)
        # Convert the error in the intensity to CGS units
        A_cgs_err = MJysr_to_cgs_err(A, A_err, μ, μ_err)

        # Get the extinction profile at the center
        ext = extinction[argmin(abs.(λ .- μ))]

        # Calculate the intensity using the utility function
        intensity, i_err = calculate_intensity(:Drude, A_cgs, A_cgs_err, μ, μ_err, fwhm, fwhm_err)
        # Calculate the equivalent width using the utility function
        eqw, e_err = calculate_eqw(popt_c, perr_c, n_dust_cont, n_dust_feat, extinction_curve, extinction_screen, 
            fit_sil_emission, :Drude, A*ext, A_err*ext, μ, μ_err, fwhm, fwhm_err)

        @debug "PAH feature with ($A_cgs, $μ, $fwhm) and errors ($A_cgs_err, $μ_err, $fwhm_err)"
        @debug "I=$intensity, err=$i_err, EQW=$eqw, err=$e_err"

        # increment the parameter index
        pᵢ += 3

        # intensity units: erg s^-1 cm^-2 sr^-1 (integrated over μm)
        p_dust[pₒ] = intensity
        p_dust_err[pₒ] = i_err
        # equivalent width units: μm
        p_dust[pₒ+1] = eqw
        p_dust_err[pₒ+1] = e_err

        # SNR, calculated as (peak amplitude) / (RMS intensity of the surrounding spectrum)
        # include the extinction factor when calculating the SNR
        p_dust[pₒ+2] = A*ext / std(I[.!mask_lines] .- continuum[.!mask_lines])
        @debug "integrated intensity $(p_dust[pₒ]) +/- $(p_dust_err[pₒ]) " *
            "(erg s^-1 cm^-2 sr^-1), equivalent width $(p_dust[pₒ+1]) +/- $(p_dust_err[pₒ+1]) um, " *
            "and SNR $(p_dust[pₒ+2])"

        pₒ += 3
    end

    # Loop through lines
    p_lines = zeros(3n_lines+3n_acomps)
    p_lines_err = zeros(3n_lines+3n_acomps)
    pₒ = 1
    # Skip over the tied velocity offsets
    pᵢ = 2 * sum(n_kin_tied) + 1
    # Skip over the tied voigt mixing parameter, saving its index
    if tie_voigt_mixing
        ηᵢ = pᵢ
        pᵢ += 1
    end
    for (k, λ0) ∈ enumerate(lines.λ₀)

        # (\/ pretty much the same as the fit_line_residuals function, but calculating the integrated intensities)
        amp = popt_l[pᵢ]
        amp_err = perr_l[pᵢ]
        # fill values with nothings for profiles that may / may not have them
        h3 = h3_err = h4 = h4_err = η = η_err = nothing

        # Check if voff is tied: if so, use the tied voff parameter, otherwise, use the line's own voff parameter
        if isnothing(lines.tied[k, 1])
            # Unpack the components of the line
            voff = popt_l[pᵢ+1]
            voff_err = perr_l[pᵢ+1]
            fwhm = popt_l[pᵢ+2]
            fwhm_err = perr_l[pᵢ+2]
            pᵢ += 3
        elseif flexible_wavesol
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and kin_tied_key
            vwhere = findfirst(x -> x == lines.tied[k, 1], tied_kinematics.key[1])
            voff_series = popt_l[vwhere]
            voff_indiv = popt_l[pᵢ+1]
            # Add velocity shifts of the tied lines and the individual offsets together
            voff = voff_series + voff_indiv
            voff_err = √(perr_l[vwhere]^2 + perr_l[pᵢ+1]^2)
            fwhm = popt_l[n_kin_tied[1]+vwhere]
            fwhm_err = perr_l[n_kin_tied[1]+vwhere]
            pᵢ += 2
        else
            # Find the position of the tied velocity offset that should be used
            # based on matching the keys in line_tied and kin_tied_key
            vwhere = findfirst(x -> x == lines.tied[k, 1], tied_kinematics.key[1])
            voff = popt_l[vwhere]
            voff_err = perr_l[vwhere]
            fwhm = popt_l[n_kin_tied[1]+vwhere]
            fwhm_err = perr_l[n_kin_tied[1]+vwhere]
            pᵢ += 1
        end

        if lines.profiles[k, 1] == :GaussHermite
            # Get additional h3, h4 components
            h3 = popt_l[pᵢ]
            h3_err = perr_l[pᵢ]
            h4 = popt_l[pᵢ+1]
            h4_err = perr_l[pᵢ+1]
            pᵢ += 2
        elseif lines.profiles[k, 1] == :Voigt
            # Get additional mixing component, either from the tied position or the 
            # individual position
            if !tie_voigt_mixing
                η = popt_l[pᵢ]
                η_err = perr_l[pᵢ]
                pᵢ += 1
            else
                η = popt_l[ηᵢ]
                η_err = perr_l[ηᵢ]
            end
        end

        # Convert voff in km/s to mean wavelength in μm
        mean_μm = Doppler_shift_λ(λ0, voff)
        mean_μm_err = λ0 / C_KMS * voff_err
        # WARNING:
        # Probably should set to 0 if using flexible tied voffs since they are highly degenerate and result in massive errors
        # if !isnothing(cube_fitter.line_tied[k]) && cube_fitter.flexible_wavesol
        #     mean_μm_err = 0.
        # end

        # Convert FWHM from km/s to μm
        fwhm_μm = Doppler_shift_λ(λ0, fwhm/2) - Doppler_shift_λ(λ0, -fwhm/2)
        fwhm_μm_err = λ0 / C_KMS * fwhm_err

        # Convert amplitude to erg s^-1 cm^-2 μm^-1 sr^-1, put back in the normalization
        amp_cgs = MJysr_to_cgs(amp*N, mean_μm)
        amp_cgs_err = MJysr_to_cgs_err(amp*N, amp_err*N, mean_μm, mean_μm_err)

        # Get the extinction factor at the line center
        ext = extinction[argmin(abs.(λ .- mean_μm))]

        @debug "Line with ($amp_cgs, $mean_μm, $fwhm_μm) and errors ($amp_cgs_err, $mean_μm_err, $fwhm_μm_err)"

        # Calculate line intensity using the helper function
        p_lines[pₒ], p_lines_err[pₒ] = calculate_intensity(lines.profiles[k, 1], amp_cgs, amp_cgs_err, mean_μm, mean_μm_err,
            fwhm_μm, fwhm_μm_err, h3=h3, h3_err=h3_err, h4=h4, h4_err=h4_err, η=η, η_err=η_err)

        # Calculate the equivalent width using the utility function
        p_lines[pₒ+1], p_lines_err[pₒ+1] = calculate_eqw(popt_c, perr_c, n_dust_cont, n_dust_feat,
            extinction_curve, extinction_screen, fit_sil_emission, lines.profiles[k, 1], amp*N*ext, amp_err*N*ext, 
            mean_μm, mean_μm_err, fwhm_μm, fwhm_μm_err, h3=h3, h3_err=h3_err, h4=h4, h4_err=h4_err, η=η, η_err=η_err)

        # SNR
        p_lines[pₒ+2] = amp*N*ext / std(I[.!mask_lines] .- continuum[.!mask_lines])

        @debug "integrated intensity $(p_lines[pₒ]) +/- $(p_lines_err[pₒ]) " *
            "(erg s^-1 cm^-2 sr^-1), equivalent width $(p_lines[pₒ+1]) +/- $(p_lines_err[pₒ+1]) um, and SNR $(p_lines[pₒ+1])"

        # Advance the output vector index by 3
        pₒ += 3

        for j ∈ 2:n_comps
            # filler values
            acomp_amp = acomp_mean_μm = acomp_fwhm_μm = nothing
            acomp_h3 = acomp_h3_err = acomp_h4 = acomp_h4_err = acomp_η = acomp_η_err = nothing

            # Repeat EVERYTHING, minus the flexible_wavesol, for the additional components
            if !isnothing(lines.profiles[k, j])
                acomp_amp = amp * popt_l[pᵢ]
                acomp_amp_err = √(acomp_amp^2 * ((amp_err / amp)^2 + (perr_l[pᵢ] / acomp_amp)^2))
                if isnothing(lines.tied[k, j])
                    acomp_voff = voff + popt_l[pᵢ+1]
                    acomp_voff_err = √(voff_err^2 + perr_l[pᵢ+1]^2)
                    acomp_fwhm = fwhm * popt_l[pᵢ+2]
                    acomp_fwhm_err = √(acomp_fwhm^2 * ((fwhm_err / fwhm)^2 + (perr_l[pᵢ+2] / acomp_fwhm)^2))
                    pᵢ += 3

                else
                    vwhere = findfirst(x -> x == lines.tied[k, j], tied_kinematics.key[j])
                    acomp_voff = voff + popt_l[2*sum(n_kin_tied[1:(j-1)])+vwhere]
                    acomp_voff_err = √(voff_err^2 + perr_l[2*sum(n_kin_tied[1:(j-1)])+vwhere]^2)
                    acomp_fwhm = fwhm * popt_l[2*sum(n_kin_tied[1:(j-1)])+n_kin_tied[j]+vwhere]
                    acomp_fwhm_err = √(acomp_fwhm^2 * ((fwhm_err / fwhm)^2 + 
                        (perr_l[2*sum(n_kin_tied[1:(j-1)])+n_kin_tied[j]+vwhere] / acomp_fwhm)^2))      
                    pᵢ += 1
                end

                if lines.profiles[k, j] == :GaussHermite
                    acomp_h3 = popt_l[pᵢ]
                    acomp_h3_err = perr_l[pᵢ]
                    acomp_h4 = popt_l[pᵢ+1]
                    acomp_h4_err = perr_l[pᵢ+1]
                    pᵢ += 2
                elseif lines.profiles[k, j] == :Voigt
                    if !tie_voigt_mixing
                        acomp_η = popt_l[pᵢ]
                        acomp_η_err = perr_l[pᵢ]
                        pᵢ += 1
                    else
                        acomp_η = popt_l[ηᵢ]
                        acomp_η_err = perr_l[ηᵢ]
                    end
                end

                # Convert voff in km/s to mean wavelength in μm
                acomp_mean_μm = Doppler_shift_λ(λ0, acomp_voff)
                acomp_mean_μm_err = λ0 / C_KMS * acomp_voff_err
                # Convert FWHM from km/s to μm
                acomp_fwhm_μm = Doppler_shift_λ(λ0, acomp_fwhm/2) - Doppler_shift_λ(λ0, -acomp_fwhm/2)
                acomp_fwhm_μm_err = λ0 / C_KMS * acomp_fwhm_err

                # Convert amplitude to erg s^-1 cm^-2 μm^-1 sr^-1, put back in the normalization
                acomp_amp_cgs = MJysr_to_cgs(acomp_amp*N, acomp_mean_μm)
                acomp_amp_cgs_err = MJysr_to_cgs_err(acomp_amp*N, acomp_amp_err*N, acomp_mean_μm, acomp_mean_μm_err)

                @debug "acomp line with ($acomp_amp_cgs, $acomp_mean_μm, $acomp_fwhm_μm) and errors ($acomp_amp_cgs_err, $acomp_mean_μm_err, $acomp_fwhm_μm_err)"
                    
                # Calculate line intensity using the helper function
                p_lines[pₒ], p_lines_err[pₒ] = calculate_intensity(lines.profiles[k, j], acomp_amp_cgs, acomp_amp_cgs_err, acomp_mean_μm, 
                    acomp_mean_μm_err, acomp_fwhm_μm, acomp_fwhm_μm_err, h3=acomp_h3, h3_err=acomp_h3_err, h4=acomp_h4, h4_err=acomp_h4_err, 
                    η=acomp_η, η_err=acomp_η_err)
    
                # Calculate the equivalent width using the utility function
                p_lines[pₒ+1], p_lines_err[pₒ+1] = calculate_eqw(popt_c, perr_c, n_dust_cont, n_dust_feat, 
                    extinction_curve, extinction_screen, fit_sil_emission, lines.profiles[k, j], acomp_amp*N*ext, acomp_amp_err*N*ext, 
                    acomp_mean_μm, acomp_mean_μm_err, acomp_fwhm_μm, acomp_fwhm_μm_err, h3=acomp_h3, h3_err=acomp_h3_err, h4=acomp_h4, 
                    h4_err=acomp_h4_err, η=acomp_η, η_err=acomp_η_err)

                # SNR
                p_lines[pₒ+2] = acomp_amp*N*ext / std(I[.!mask_lines] .- continuum[.!mask_lines])

                @debug "integrated intensity $(p_lines[pₒ]) +/- $(p_lines_err[pₒ]) " *
                    "(erg s^-1 cm^-2 sr^-1), equivalent width $(p_lines[pₒ+1]) +/- $(p_lines_err[pₒ+1]) um, and SNR $(p_lines[pₒ+1])"

                # Advance the output vector index by 3
                pₒ += 3
            end
        end

        # SNR, calculated as (amplitude) / (RMS of the surrounding spectrum)
        # p_lines[pₒ+2] = calculate_SNR(Δλ, I[.!mask_lines] .- continuum[.!mask_lines], cube_fitter.line_profiles[k],
        #     amp*N*ext, mean_μm, fwhm_μm, h3=h3, h4=h4, η=η, acomp_prof=cube_fitter.line_acomp_profiles[k], 
        #     acomp_amp=isnothing(acomp_amp) ? acomp_amp : acomp_amp*N*ext,
        #     acomp_peak=acomp_mean_μm, acomp_fwhm=acomp_fwhm_μm, acomp_h3=acomp_h3, acomp_h4=acomp_h4, acomp_η=acomp_η)

        # pₒ += 3

    end

    p_dust, p_lines, p_dust_err, p_lines_err
end


############################################## LINE PROFILES #############################################


"""
    Gaussian(x, A, μ, FWHM)

Evaluate a Gaussian profile at `x`, parameterized by the amplitude `A`, mean value `μ`, and 
full-width at half-maximum `FWHM`
"""
@inline function Gaussian(x::Real, A::Real, μ::Real, FWHM::Real)
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
function GaussHermite(x::Real, A::Real, μ::Real, FWHM::Real, h₃::Real, h₄::Real)

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
@inline function Lorentzian(x::Real, A::Real, μ::Real, FWHM::Real)
    A * (FWHM/2)^2 / ((x-μ)^2 + (FWHM/2)^2)
end


"""
    Voigt(x, A, μ, FWHM, η)

Evaluate a pseudo-Voigt profile at `x`, parametrized by the amplitude `A`, mean value `μ`,
full-width at half-maximum `FWHM`, and mixing ratio `η`

https://docs.mantidproject.org/nightly/fitting/fitfunctions/PseudoVoigt.html
"""
function Voigt(x::Real, A::Real, μ::Real, FWHM::Real, η::Real)

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
    calculate_intensity(profile, amp, amp_err, peak, peak_err, fwhm, fwhm_err; <keyword_args>)

Calculate the integrated intensity of a spectral feature, i.e. a PAH or emission line. Calculates the integral
of the feature profile, using an analytic form if available, otherwise integrating numerically with QuadGK.
"""
function calculate_intensity(profile::Symbol, amp::T, amp_err::T, peak::T, peak_err::T, fwhm::T, fwhm_err::T;
    h3::Union{T,Nothing}=nothing, h3_err::Union{T,Nothing}=nothing, h4::Union{T,Nothing}=nothing, 
    h4_err::Union{T,Nothing}=nothing, η::Union{T,Nothing}=nothing, η_err::Union{T,Nothing}=nothing) where {T<:Real}

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
function calculate_eqw(popt_c::Vector{T}, perr_c::Vector{T}, n_dust_cont::Integer, n_dust_feat::Integer,
    extinction_curve::String, extinction_screen::Bool, fit_sil_emission::Bool, profile::Symbol, 
    amp::T, amp_err::T, peak::T, peak_err::T, fwhm::T, fwhm_err::T; h3::Union{T,Nothing}=nothing, h3_err::Union{T,Nothing}=nothing, 
    h4::Union{T,Nothing}=nothing, h4_err::Union{T,Nothing}=nothing, η::Union{T,Nothing}=nothing, 
    η_err::Union{T,Nothing}=nothing) where {T<:Real}

    # If the line is not present, the equivalent width is 0
    if iszero(amp)
        @debug "eqw=0, err=0"
        return 0., 0.
    end

    # Integrate the flux ratio to get equivalent width
    if profile == :Drude
        # do not shift the drude profiles since x=0 and mu=0 cause problems;
        # the wide wings should allow quadgk to find the solution even without shifting it
        cont = x -> fit_spectrum([x], popt_c, n_dust_cont, extinction_curve, extinction_screen, fit_sil_emission)[1]
        eqw = quadgk(x -> Drude(x, amp, peak, fwhm) / cont(x), max(peak-10fwhm, 3.), peak+10fwhm, order=200)[1]
        # errors
        err_l = eqw - quadgk(x -> Drude(x, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps())) / cont(x),
            max(peak-10fwhm, 3.), peak+10fwhm, order=200)[1]
        err_u = quadgk(x -> Drude(x, amp+amp_err, peak, fwhm+fwhm_err) / cont(x),
            max(peak-10fwhm, 3.), peak+10fwhm, order=200)[1] - eqw
        err_l = err_l ≥ 0 ? err_l : 0.
        err_u = abs(err_u)
        err = (err_l + err_u)/2
    elseif profile == :Gaussian
        # Make sure to use [x] as a vector and take the first element [1] of the result, since the continuum functions
        # were written to be used with vector inputs + outputs
        cont = x -> fit_full_continuum([x], popt_c, n_dust_cont, n_dust_feat, extinction_curve, extinction_screen, fit_sil_emission)[1]
        eqw = quadgk(x -> Gaussian(x+peak, amp, peak, fwhm) / cont(x+peak), max(-10fwhm, -peak+1), 10fwhm, order=200)[1]
        err_l = eqw - quadgk(x -> Gaussian(x+peak, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps())) / cont(x+peak),
            -10fwhm, 10fwhm, order=200)[1]
        err_u = quadgk(x -> Gaussian(x+peak, amp+amp_err, peak, fwhm+fwhm_err) / cont(x+peak),
            -10fwhm, 10fwhm, order=200)[1] - eqw
        err_l = err_l ≥ 0 ? err_l : 0.
        err_u = abs(err_u)
        err = (err_l + err_u)/2
    elseif profile == :Lorentzian
        cont = x -> fit_full_continuum([x], popt_c, n_dust_cont, n_dust_feat, extinction_curve, extinction_screen, fit_sil_emission)[1]
        eqw = quadgk(x -> Lorentzian(x+peak, amp, peak, fwhm) / cont(x+peak), max(-10fwhm, -peak+1), 10fwhm, order=200)[1]
        err_l = eqw - quadgk(x -> Lorentzian(x+peak, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps())) / cont(x+peak),
            -10fwhm, 10fwhm, order=200)[1]
        err_u = quadgk(x -> Lorentzian(x+peak, amp+amp_err, peak, fwhm+fwhm_err) / cont(x+peak),
            -10fwhm, 10fwhm, order=200)[1] - eqw
        err_l = err_l ≥ 0 ? err_l : 0.
        err_u = abs(err_u)
        err = (err_l + err_u)/2
    elseif profile == :GaussHermite
        cont = x -> fit_spectrum([x], popt_c, n_dust_cont, extinction_curve, extinction_screen, fit_sil_emission)[1]
        eqw = quadgk(x -> GaussHermite(x+peak, amp, peak, fwhm, h3, h4) / cont(x+peak), max(-10fwhm, -peak+1), 10fwhm, order=200)[1]
        err_l = eqw - quadgk(x -> GaussHermite(x+peak, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), h3-h3_err, h4-h4_err) / 
            cont(x+peak), -10fwhm, 10fwhm, order=200)[1]
        err_u = quadgk(x -> GaussHermite(x+peak, amp+amp_err, peak, fwhm+fwhm_err, h3+h3_err, h4+h4_err) / 
            cont(x+peak), -10fwhm, 10fwhm, order=200)[1] - eqw
        err_l = err_l ≥ 0 ? err_l : 0.
        err_u = abs(err_u)
        err = (err_l + err_u)/2
    elseif profile == :Voigt
        cont = x -> fit_spectrum([x], popt_c, n_dust_cont, extinction_curve, extinction_screen, fit_sil_emission)[1]
        eqw = quadgk(x -> Voigt(x+peak, amp, peak, fwhm, η) / cont(x+peak), max(-10fwhm, -peak+1), 10fwhm, order=200)[1]
        err_l = eqw - quadgk(x -> Voigt(x+peak, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), η-η_err) / 
            cont(x+peak), -10fwhm, 10fwhm, order=200)[1]
        err_u = quadgk(x -> Voigt(x+peak, amp+amp_err, peak, fwhm+fwhm_err, η+η_err) / 
            cont(x+peak), -10fwhm, 10fwhm, order=200)[1] - eqw
        err_l = err_l ≥ 0 ? err_l : 0.
        err_u = abs(err_u)
        err = (err_l + err_u)/2
    else
        error("Unrecognized line profile $profile")
    end

    @debug "eqw=$eqw, err=$err"
    eqw, err
end
