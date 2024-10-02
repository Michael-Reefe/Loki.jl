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

# Convenience function for interpolation with constant extrapolation at the edges
extrapolate_constant(itp::DataInterpolations.AbstractInterpolation) = x -> itp(clamp.(x, extrema(itp.t)...))

# 1:1 interpolation methods (all data points will be on the spline)
linear_interp(x, y) = DataInterpolations.LinearInterpolation(y, x; extrapolate=false) |> extrapolate_constant
quadratic_spline_interp(x, y) = DataInterpolations.QuadraticSpline(y, x; extrapolate=false) |> extrapolate_constant
cubic_spline_interp(x, y) = DataInterpolations.CubicSpline(y, x; extrapolate=false) |> extrapolate_constant

# smoothing interpolation methods (not all data points will be on the spline)
linear_spline_smooth(x, y, n_knots::Integer) = DataInterpolations.BSplineApprox(y, x, 1, n_knots, :Uniform, :Uniform; extrapolate=true)
cubic_spline_smooth(x, y, n_knots::Integer) = DataInterpolations.BSplineApprox(y, x, 3, n_knots, :Uniform, :Uniform; extrapolate=true)

# Save the Donnan et al. 2022 profile as a constant
const DP_prof = silicate_dp()
const DP_interp = cubic_spline_interp(DP_prof[1], DP_prof[2])

# Save the Chiar+Tielens 2005 profile as a constant
const CT_prof = silicate_ct()
const CT_interp = cubic_spline_interp(CT_prof[1], CT_prof[2])

# Save the KVT profile as a constant
const KVT_interp = quadratic_spline_interp(kvt_prof[:, 1], kvt_prof[:, 2])
const KVT_interp_end = linear_interp([kvt_prof[end, 1], kvt_prof[end, 1]+2], [kvt_prof[end, 2], 0.])

# Save the OHM 1992 profile as a constant
const OHM_prof = silicate_ohm()
const OHM_interp = cubic_spline_interp(OHM_prof[1], OHM_prof[2])

# Save the Smith+2006 PAH templates as constants
const SmithTemps = read_smith_temps()
const Smith3_interp = cubic_spline_interp(SmithTemps[1], SmithTemps[2])
const Smith4_interp = cubic_spline_interp(SmithTemps[3], SmithTemps[4])

# Save the Ice+CH optical depth template as a constant
const IceCHTemp = read_ice_ch_temps()
const Ice_interp = cubic_spline_interp(IceCHTemp[1], IceCHTemp[2])
const CH_interp = cubic_spline_interp(IceCHTemp[3], IceCHTemp[4])

# Polynomials for calculating the CCM extinction curve
const CCM_optPoly_a = Polynomial([1.0, 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505])
const CCM_optPoly_b = Polynomial([0.0, 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347])
const CCM_fuvPoly_a = Polynomial([-1.703, -0.628, 0.137, -0.070])
const CCM_fuvPoly_b = Polynomial([13.670, 4.257, -0.420, 0.374])

# Polynomials for calculating the Calzetti et al. (2000) extinction curve
const Calz_poly_a = Polynomial([-1.857, 1.040])
const Calz_poly_b = Polynomial([-2.156, 1.509, -0.198, 0.011])

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
The other dimensions are assumed to be the first N dimensions

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
    extendp(array1d, shape)

Extend `array1d` into other dimensions specified by `shape`
The other dimensions are assumed to be the last N dimensions

# Example
```jldoctest
julia> extendp([1,2,3], (4,5))
3×4×5 Array{Int64, 3}:
[:, :, 1] =
 1  1  1  1
 2  2  2  2
 3  3  3  3

[:, :, 2] =
 1  1  1  1
 2  2  2  2
 3  3  3  3

[:, :, 3] =
 1  1  1  1
 2  2  2  2
 3  3  3  3

[:, :, 4] =
 1  1  1  1
 2  2  2  2
 3  3  3  3

[:, :, 5] =
 1  1  1  1
 2  2  2  2
 3  3  3  3
```
"""
@inline extendp(array1d, shape) = repeat(reshape(array1d, (length(array1d),1,1)), outer=[1,shape...])


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
    F_test(n, p1, p2, χ1, χ2, threshold)

Perform a statistical F-test on two models with free parameters `p1` and `p2` and
chi2 values `χ1` and `χ2`, fit to data with `n` data points. The F-value calculated
from the data must be greater than the critical value of the F distribution for the
given degrees of freedom up to the specified `threshold` level. `threshold` must be
given as a probability, for example a threshold of 0.003 corresponds 
to 1-0.003 -> 99.7% or a 3-sigma confidence level.
"""
function F_test(n, p1, p2, χ1, χ2, threshold)
    # Generate an F distribution with these parameters
    F = FDist(p2 - p1, n - p2)
    # Calculate the critical value at some confidence threshold set by the user
    F_crit = invlogccdf(F, log(threshold))
    # Calculate the F value from the data
    F_data = ((χ1 - χ2) / (p2 - p1)) / (χ2 / (n - p2))
    # Compare to the critical value
    F_data > F_crit, F_data, F_crit
end


"""
    convolveGaussian1D(flux, fwhm)

Convolve a spectrum by a Gaussian with different FWHM for every pixel.
This function simply loops through each pixel in the array and applies a convolution
with a Gaussian kernel of FWHM given by the corresponding value in the `fwhm` vector.

The concept for this function was based on a similar function in the pPXF python code
(Cappellari 2017, https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract),
however the implementation is different.
"""
function convolveGaussian1D(flux::AbstractVector, fwhm::AbstractVector)

    # clamp with a minimum of 0.01 so as to not cause problems with Gaussians with 0 std dev
    fwhm_clamped = clamp.(fwhm, 0.01, Inf)

    # Prepare output array
    flux_convolved = zeros(eltype(flux), length(flux))

    # Pad edges of the input array with 0s
    pad_size = ceil(Int, maximum(2fwhm_clamped))
    flux_padded = [zeros(eltype(flux), pad_size); flux; zeros(eltype(flux), pad_size)]

    # Loop through pixels
    @inbounds for i ∈ (1+pad_size):(length(flux_padded)-pad_size)

        ii = i - pad_size
        # Create a normal distribution kernel of the corresponding size
        pixel = ceil(Int, 2fwhm_clamped[ii])
        x = -pixel:1:pixel
        kernel = Gaussian.(x, 1.0, 0.0, fwhm_clamped[ii])

        tot_kernel = 0.
        for j ∈ eachindex(x)
            flux_convolved[ii] += kernel[j] * flux_padded[i+x[j]]
            tot_kernel += kernel[j]
        end
        # Make sure to normalize by the kernel
        flux_convolved[ii] /= tot_kernel
    end

    return flux_convolved
end


"""
    ∫Gaussian(A[, A_err], FWHM[, FWHM_err])

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
∫Gaussian(A, A_err, FWHM, FWHM_err) = ∫Gaussian(A, FWHM), √(π / (4log(2))) * hypot(A*FWHM_err, FWHM*A_err)


"""
    ∫Lorentzian(A[, A_err], FWHM[, FWHM_err])

Integral of a Lorentzian with amplitude `A` and full-width at half-max `FWHM`

# Examples
```jldoctest
julia> ∫Lorentzian(1000, 0.5)
785.3981633974482
julia> ∫Lorentzian(600, 1.2)
1130.9733552923256
```
"""
∫Lorentzian(A, FWHM) = π/2 * A * FWHM
∫Lorentzian(A, A_err, FWHM, FWHM_err) = ∫Lorentzian(A, FWHM), π/2 * hypot(A*FWHM_err, FWHM*A_err)


"""
    ∫Voigt(A[, A_err], FWHM[, FWHM_err], η[, η_err])

Integral of a (pseudo) Voigt function with amplitude `A`, full-width at half-max `FWHM`,
and mixing parameter `η`

# Examples
```jldoctest
julia> ∫Voigt(1000, 0.5, 1.0)
532.233509715613
julia> ∫Voigt(600, 1.2, 0.0)
1130.9733552923256
```
"""
∫Voigt(A, FWHM, η) =  A * FWHM * π / (2 * (1 + (√(π*log(2)) - 1)*η))
∫Voigt(A, A_err, FWHM, FWHM_err, η, η_err) = ∫Voigt(A, FWHM, η), 
    π / (2 * (1 + (√(π*log(2)) - 1)*η)) * hypot(A*FWHM_err, FWHM*A_err, A*FWHM*(√(π*log(2)) - 1)/(1 + (√(π*log(2)) - 1)*η)*η_err)


"""
    ∫Drude(A[, A_err], FWHM[, FWHM_err])

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
∫Drude(A, A_err, FWHM, FWHM_err) = ∫Drude(A, FWHM), π/2 * hypot(A*FWHM_err, FWHM*A_err)


"""
    ∫PearsonIV(A, a, ν, m)

Integral of a Pearson type-IV profile.
"""
∫PearsonIV(A, a, m, ν) = begin 
    n = (1 + (-ν/(2m))^2)^-m * exp(-ν * atan(-ν/(2m)))
    k = 1/(√(π)*a) * gamma(m) / gamma(m - 1/2) * abs2(gamma(m + im*ν/2) / gamma(m))
    A/n/k
end


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
function hermite(x, n::Integer)
    if iszero(n)
        1.
    elseif isone(n)
        2x
    else
        2x * hermite(x, n-1) - 2(n-1) * hermite(x, n-2)
    end
end


############################################### CONTINUUM FUNCTIONS ##############################################


"""
    Blackbody_ν(λ, Temp)

Return the Blackbody function Bν (per unit FREQUENCY) in MJy/sr,
given a wavelength in μm and a temperature in Kelvins.
"""
@inline function Blackbody_ν(λ, Temp)
    @. Bν_1/λ^3 / expm1(Bν_2/(λ*Temp))
end


"""
    Wein(Temp)

Return the peak wavelength (in μm) of a Blackbody spectrum at a given temperature `Temp`,
using Wein's Displacement Law.
"""
@inline function Wein(Temp)
    b_Wein / Temp
end


"""
    power_law(λ, α)

Simple power law function where the flux is proportional to the wavelength to the power alpha,
normalized at 9.7 um.
"""
@inline function power_law(λ, α, ref_λ=9.7)
    @. (λ/ref_λ)^α
end


"""
    silicate_emission(λ, A, T, Cf, τ_warm, τ_cold, λ_peak)

A hot silicate dust emission profile, i.e. Gallimore et al. (2010), with an amplitude A,
temperature T, covering fraction Cf, and optical depths τ_warm and τ_cold.
"""
function silicate_emission(λ, A, T, Cf, τ_warm, τ_cold, λ_peak)
    o_peak = 10.0178
    Δλ = o_peak - λ_peak
    λshift = λ .+ Δλ
    ext_curve = τ_ohm(λshift)

    bb = @. A * Blackbody_ν(λ, T) * (1 - extinction(ext_curve, τ_warm, screen=true))
    @. bb * (1 - Cf) + bb * Cf * extinction(ext_curve, τ_cold, screen=true)
end

################################################# PAH PROFILES ################################################


"""
    Drude(x, A, μ, FWHM, asym)

Calculate a Drude profile at location `x`, with amplitude `A`, central value `μ`, and full-width at half-max `FWHM`
Optional asymmetry parameter `asym`
"""
@inline function Drude(x, A, μ, FWHM, asym)
    γ = @. 2FWHM / (1 + exp(asym*(x-μ)))
    @. A * (γ/μ)^2 / ((x/μ - μ/x)^2 + (γ/μ)^2)
end


"""
    PearsonIV(x, A, μ, a, m, ν)

Calculate a Pearson Type-IV profile at location `x`, with amplitude `A`, unextinguished central value `μ`, width
parameter `a`, index `m`, and exponential cutoff `ν`.

See Pearson (1895), and https://iopscience.iop.org/article/10.3847/1538-4365/ac4989/pdf
"""
function PearsonIV(x, A, μ, a, m, ν)
    n = (1 + (-ν/(2m))^2)^-m * exp(-ν * atan(-ν/(2m)))
    @. A/n * (1 + ((x - μ)/a)^2)^-m * exp(-ν * atan((x - μ)/a))
end

########################################## STELLAR POP FUNCTIONS #########################################


"""
    convolve_losvd(templates, vsyst, v, σ, npix)

Convolve a set of stellar population templates with a line-of-sight velocity distribution (LOSVD)
to produce templates according to the fitted stellar kinematics. Uses the Fourier Transforms of 
the templates and the LOSVD to quickly calculate the convolution.

The idea for this function was based on similar functions in the pPXF and BADASS python codes, but
the specific implementation is different. See:
- Cappellari (2017): http://adsabs.harvard.edu/abs/2017MNRAS.466..798C
- Sexton et al. (2021): https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.2871S/abstract 
"""
function convolve_losvd(_templates::AbstractArray, vsyst, v, σ, vres, npix::Integer;
    temp_fft::Bool=false, npad_in::Integer=0)

    if temp_fft
        @assert npad_in > 0 "npad_in must be specified for inputs that are already FFT'ed!"
    end

    templates = copy(_templates)
    if ndims(_templates) == 1
        templates = reshape(_templates, (length(templates), 1))
    end

    # Check if the templates are already FFT'ed
    if !temp_fft
        # Pad with 0s up to a factor of small primes to increase efficiency 
        s = size(templates)
        npad = nextprod([2,3,5], s[1])
        temps = zeros(typeof(v), npad, s[2])
        for j in axes(templates, 2)
            temps[1:s[1], j] = templates[:, j]
        end
        # Calculate the Fourier transform of the templates
        # Note: in general we cannot calculate this a priori because the templates may be different for each iteration of the fit
        #       if the user is marginalizing over the stellar ages or metallicities. The `temp_fft` option is mainly for usage with
        #       the Fe II templates, which do not have any such parameters that can be marginalized over.
        temp_rfft = rfft(temps, 1)
    else
        npad = npad_in
        temp_rfft = templates
    end

    n_ft = size(temp_rfft, 1)

    # Remember to normalize velocities and sigmas by the velocity resolution
    V = (vsyst + v)/vres
    Σ = σ/vres
    ω = range(0, π, n_ft)

    # Calculate the analytic Fourier transform of the LOSVD: See Cappellari (2017) eq. (38)
    ft_losvd = conj.(exp.(1im .* ω .* V .- Σ.^2 .* ω.^2 ./ 2))

    # Calculate the inverse Fourier transform of the resultant distribution to get the convolution
    # see, i.e. https://en.wikipedia.org/wiki/Convolution_theorem
    template_convolved = irfft(temp_rfft .* ft_losvd, npad, 1)

    # Take only enough pixels to match the length of the input spectrum
    template_convolved[1:npix, :]
end


"""
    _calzetti_kprime_curve(λ, E_BV, Rv)

A small helper function to calculate the k'(λ) attenuation curve from Calzetti et al. (2000).
This function is used by the different methods in `attenuation_calzetti` to reduce repetition,
since all methods include this same calculation.
"""
function _calzetti_kprime_curve(λ::AbstractVector, E_BV, Rv)

    # eq. (4) from Calzetti et al. (2000)
    kprime = zeros(eltype(E_BV), length(λ))
    good = λ .≥ 0.63
    kprime[good] .= 2.659 .* Calz_poly_a.(1 ./ λ[good]) .+ Rv
    kprime[.~good] .= 2.659 .* Calz_poly_b.(1 ./ λ[.~good]) .+ Rv

    kprime
end


"""
    attenuation_calzetti(λ, E_BV[, δ_uv]; Cf=0., Rv=4.05)

Calculate dust attenuation factor using the Calzetti et al. (2000) attenuation law:
http://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C.  

If the `δ_uv` parameter is specified, a UV bump with a slope of `δ_uv` will be added following the
prescription by Kriek & Conroy (2013):
https://ui.adsabs.harvard.edu/abs/2013ApJ...775L..16K.

One may also specify a dust covering fraction `Cf` to apply a partial covering of dust
for more complitcated geometries.

This idea for this function was based on a similar function from pPXF (Cappellari 2017), but
the implementation is different.
"""
function attenuation_calzetti(λ::AbstractVector, E_BV; Cf=0., Rv=4.05)

    # Get the actual extinction curve
    kprime = _calzetti_kprime_curve(λ, E_BV, Rv)

    # dust attenuation factor using E(B-V) in magnitudes
    atten = @. 10^(-0.4 * E_BV * kprime)

    # apply covering fraction
    @. Cf + (1 - Cf)*atten
end

function attenuation_calzetti(λ::AbstractVector, E_BV, δ_uv; Cf=0., Rv=4.05)

    # Get the actual extinction curve
    kprime = _calzetti_kprime_curve(λ, E_BV, Rv)

    # Calculate the UV bump 
    # Kriek & Conroy (2013) eq. (3): relation between UV bump amplitude Eb and slope δ_uv
    Eb = 0.85 - 1.9*δ_uv
    # Drude profile parametrizes the UV bump (Kriek & Conroy 2013, eq. (2))
    Dλ = Drude.(λ, Eb, 0.2175, 0.035, 0.)

    # Kriek & Conroy (2013) eq. (1) 
    atten = @. 10^(-0.4 * E_BV * (kprime + Dλ) * (λ / 0.55)^δ_uv)

    # apply covering fraction
    @. Cf + (1 - Cf)*atten
end


"""
    attenuation_cardelli(λ, E_BV[, Rv])

Calculate the attenuation factor for a given wavelength range `λ` with a 
reddening of `E_BV` and selective extinction ratio `Rv`, using the
Cardelli et al. (1989) galactic extinction curve.

This function has been adapted from BADASS (Sexton et al. 2021), which
in turn has been adapted from the IDL Astrolib library.

# Arguments
- `λ::Vector{<:Real}`: The wavelength vector in microns
- `E_BV::Real`: The color excess E(B-V) in magnitudes
- `Rv::Real=3.1`: The ratio of total selective extinction R(V) = A(V)/E(B-V)

# Returns
- `Vector{<:Real}`: The extinction factor, 10^(-0.4*A(V)*(a(λ)+b(λ)/R(V)))

Notes from the original function (Copyright (c) 2014, Wayne Landsman)
have been pasted below:

1. This function was converted from the IDL Astrolib procedure
   last updated in April 1998. All notes from that function
   (provided below) are relevant to this function 
2. (From IDL:) The CCM curve shows good agreement with the Savage & Mathis (1979)
   ultraviolet curve shortward of 1400 A, but is probably
   preferable between 1200 and 1400 A.
3. (From IDL:) Many sightlines with peculiar ultraviolet interstellar extinction 
   can be represented with a CCM curve, if the proper value of 
   R(V) is supplied.
4. (From IDL:) Curve is extrapolated between 912 and 1000 A as suggested by
   Longo et al. (1989, ApJ, 339,474)
5. (From IDL:) Use the 4 parameter calling sequence if you wish to save the 
   original flux vector.
6. (From IDL:) Valencic et al. (2004, ApJ, 616, 912) revise the ultraviolet CCM
   curve (3.3 -- 8.0 um-1).	But since their revised curve does
   not connect smoothly with longer and shorter wavelengths, it is
   not included here.
7. For the optical/NIR transformation, the coefficients from 
   O'Donnell (1994) are used
"""
function attenuation_cardelli(λ::AbstractVector, E_BV, Rv=3.10)
    # inverse wavelength (microns)
    x = 1.0 ./ λ

    # Correction invalid for any x > 11
    if any(x .> 11.0)
        @warn "Input wavelength vector has values outside the allowable range of 1/λ < 11. Returning ones."
        return ones(length(x))
    end

    a = zeros(length(x))
    b = zeros(length(x))

    # Infrared
    good = 0.3 .< x .< 1.1
    a[good] = @. 0.574x[good]^1.61
    b[good] = @. -0.527x[good]^1.61

    # Optical/NIR
    good = 1.1 .≤ x .< 3.3
    y = x .- 1.82
    a[good] = CCM_optPoly_a.(y[good])
    b[good] = CCM_optPoly_b.(y[good])

    # Mid-UV
    good = 3.3 .≤ x .< 8.0
    Fa = zeros(sum(good))
    Fb = zeros(sum(good))

    good1 = x .> 5.9
    if sum(good1) > 0
        y = x .- 5.9
        Fa[good1] = @. -0.04473y[good1]^2 - 0.009779y[good1]^3
        Fb[good1] = @. 0.2130y[good1]^2 + 0.1207y[good1]^3
    end

    a[good] = @. 1.752 - 0.316x[good] - 0.104/((x[good] - 4.67)^2 + 0.341) + Fa
    b[good] = @. -3.090 + 1.825x[good] + 1.206/((x[good] - 4.62)^2 + 0.263) + Fb

    # Far-UV
    good = 8.0 .≤ x .≤ 11.0
    y = x .- 8.0
    a[good] = CCM_fuvPoly_a.(y[good])
    b[good] = CCM_fuvPoly_b.(y[good])

    # Calculate the extintion
    Av = Rv .* E_BV
    aλ = Av .* (a .+ b./Rv)
    @. 10^(-0.4 * aλ)
end


############################################## LINE PROFILES #############################################


"""
    Gaussian(x, A, μ, FWHM)

Evaluate a Gaussian profile at `x`, parameterized by the amplitude `A`, mean value `μ`, and 
full-width at half-maximum `FWHM`
"""
@inline function Gaussian(x, A, μ, FWHM)
    # Reparametrize FWHM as dispersion σ
    σ = FWHM / (2√(2log(2)))
    @. A * exp(-(x-μ)^2 / (2σ^2))
end


"""
    GaussHermite(x, A, μ, FWHM, h₃, h₄)

Evaluate a Gauss-Hermite quadrature at `x`, parametrized by the amplitude `A`, mean value `μ`,
full-width at half-maximum `FWHM`, 3rd moment / skewness `h₃`, and 4th moment / kurtosis `h₄`

See Riffel et al. (2010)
"""
function GaussHermite(x, A, μ, FWHM, h₃, h₄)

    h = [h₃, h₄]
    # Reparametrize FWHM as dispersion σ
    σ = FWHM / (2√(2log(2)))
    # Gaussian exponential argument w
    w = @. (x - μ) / σ
    # Normalized Gaussian
    α = @. exp(-w^2 / 2)

    # Calculate coefficients for the Hermite basis
    n = 3:(length(h)+2)
    norm = .√(factorial.(n) .* 2 .^ n)
    coeff = vcat([1, 0, 0], h./norm)
    # Calculate hermite basis
    Herm = sum([coeff[nᵢ] .* hermite(w, nᵢ-1) for nᵢ ∈ eachindex(coeff)])

    # Calculate peak height (i.e. value of function at w=0)
    Herm0 = sum([coeff[nᵢ] .* hermite(0., nᵢ-1) for nᵢ ∈ eachindex(coeff)])

    # Combine the Gaussian and Hermite profiles
    @. A * α * Herm / Herm0
end


"""
    Lorentzian(x, A, μ, FWHM)

Evaluate a Lorentzian profile at `x`, parametrized by the amplitude `A`, mean value `μ`,
and full-width at half-maximum `FWHM`
"""
@inline function Lorentzian(x, A, μ, FWHM)
    @. A * (FWHM/2)^2 / ((x-μ)^2 + (FWHM/2)^2)
end


"""
    Voigt(x, A, μ, FWHM, η)

Evaluate a pseudo-Voigt profile at `x`, parametrized by the amplitude `A`, mean value `μ`,
full-width at half-maximum `FWHM`, and mixing ratio `η`

https://docs.mantidproject.org/nightly/fitting/fitfunctions/PseudoVoigt.html
"""
function Voigt(x, A, μ, FWHM, η)

    # Reparametrize FWHM as dispersion σ
    σ = FWHM / (2√(2log(2))) 
    # Normalized Gaussian
    G = @. 1/√(2π * σ^2) * exp(-(x-μ)^2 / (2σ^2))
    # Normalized Lorentzian
    L = @. 1/π * (FWHM/2) / ((x-μ)^2 + (FWHM/2)^2)

    # Normalize the function so that the integral is given by this
    I = ∫Voigt(A, FWHM, η)

    # Mix the two distributions with the mixing parameter η
    @. I * (η * G + (1 - η) * L)
end


############################################## EXTINCTION PROFILES #############################################


"""
    τ_kvt(λ, β)

Calculate the mixed silicate extinction profile based on Kemper, Vriend, & Tielens (2004) 

Function adapted from PAHFIT: Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)
"""
function τ_kvt(λ::AbstractVector, β)

    # Get limits of the values that we have datapoints for via the kvt_prof constant
    mx, mn = argmax(kvt_prof[:, 1]), argmin(kvt_prof[:, 1])
    λ_mx, λ_mn = kvt_prof[mx, 1], kvt_prof[mn, 1]

    # Interpolate based on the region of data 
    ext = zeros(typeof(β), length(λ))
    r1 = λ .≤ λ_mn
    if sum(r1) > 0
        ext[r1] .= @. kvt_prof[mn, 2] * exp(2.03 * (λ[r1] - λ_mn))
    end
    r2 = λ_mn .< λ .< λ_mx
    if sum(r2) > 0
        ext[r2] .= KVT_interp(λ[r2])
    end
    r3 = λ_mx .< λ .< λ_mx+2
    if sum(r3) > 0
        ext[r3] .= KVT_interp_end(λ[r3])
    end
    ext[ext .< 0] .= 0.

    # Add a drude profile around 18 microns
    ext .+= Drude.(λ, 0.4, 18., 4.446, 0.)

    @. (1 - β) * ext + β * (9.7/λ)^1.7
end


"""
    τ_ct(λ)

Calculate the extinction profile based on Chiar & Tielens (2005)
"""
function τ_ct(λ::AbstractVector)

    mx = argmax(CT_prof[1])
    λ_mx = CT_prof[1][mx]

    ext = CT_interp(λ)
    w_mx = findall(λ .> λ_mx)
    ext[w_mx] .= CT_prof[2][mx] .* (λ_mx./λ[w_mx]).^1.7

    _, wh = findmin(x -> abs(x - 9.7), CT_prof[1])
    ext ./= CT_prof[2][wh]

    ext
end


"""
    τ_ohm(λ)

Calculate the extinction profile based on Ossenkopf, Henning, & Mathis (1992)
"""
function τ_ohm(λ::AbstractVector)

    ext = OHM_interp(λ)
    _, wh = findmin(x -> abs(x - 9.7), OHM_prof[1])
    ext ./= OHM_prof[2][wh]

    ext
end


"""
    τ_dp(λ, β)

Calculate the mixed silicate extinction profile based on Donnan et al. (2022)
"""
function τ_dp(λ::AbstractVector, β)

    # Simple cubic spline interpolation
    ext = DP_interp(λ)

    # Add 1.7 power law, as in PAHFIT
    @. (1 - β) * ext + β * (9.8/λ)^1.7
end


"""
    τ_decompose(λ, N_col, κ_abs)
Calculate the total silicate absorption optical depth given a series of column densities and
mass absorption coefficients.
"""
function τ_decompose(λ::AbstractVector, N_col::AbstractVector, κ_abs::AbstractVector{<:Base.Callable})
    sum([N_col[i] .* κ_abs[i](λ) for i in eachindex(N_col)])
end


"""
    τ_ice(λ)

Calculate the ice extinction profiles
"""
function τ_ice(λ::AbstractVector)

    # Simple cubic spline interpolation
    ext = Ice_interp(λ)
    ext ./= maximum(IceCHTemp[2])

    ext
end


"""
    τ_ch(λ)

Calculate the CH extinction profiles
"""
function τ_ch(λ::AbstractVector)

    # Simple cubic spline interpolation
    ext = CH_interp(λ)
    ext ./= maximum(IceCHTemp[4])

    ext
end


"""
    Extinction(ext, τ_97; [screen])

Calculate the extinction factor given the normalized curve `ext` and the optical depth
at 9.7 microns, `τ_97`, either assuming a screen or mixed geometry.
"""
function extinction(ext, τ_97; screen::Bool=false)
    if screen
        @. exp(-τ_97*ext)
    else
        @. ifelse(iszero(τ_97), 1., (1 - exp(-τ_97*ext)) / (τ_97*ext))
    end
end


"""
    resample_flux_permuted3D(new_wave, old_wave, flux[, err, mask])

This is a wrapper around SpectralResampling.jl's `resample_conserving_flux` function that applies a
permutation such that the third axis of the inputs is interpreted as the wavelength axis. The outputs
are also permuted such that they have the same shape as the inputs. This is useful since most datacube arrays
in this code have the wavelength axis as the third axis.
"""
function resample_flux_permuted3D(new_wave::AbstractVector, old_wave::AbstractVector, flux::AbstractArray,
    err::Union{AbstractArray,Nothing}=nothing, mask::Union{AbstractArray,Nothing}=nothing)

    # permute the third axis to the first axis for each input
    fluxp = permutedims(flux, (3,1,2))
    errp = nothing
    if !isnothing(err)
        errp = permutedims(err, (3,1,2))
    end
    maskp = nothing
    if !isnothing(mask)
        maskp = permutedims(mask, (3,1,2))
    end

    # apply the function from SpectralResampling.jl
    if !isnothing(errp) && !isnothing(mask) 
        out = resample_conserving_flux(new_wave, old_wave, fluxp, errp, maskp)
        Tuple(permutedims(out[i], (2,3,1)) for i in eachindex(out))
    elseif !isnothing(errp)
        out = resample_conserving_flux(new_wave, old_wave, fluxp, errp)
        Tuple(permutedims(out[i], (2,3,1)) for i in eachindex(out))
    elseif !isnothing(maskp)
        out = resample_conserving_flux(new_wave, old_wave, fluxp, maskp)
        Tuple(permutedims(out[i], (2,3,1)) for i in eachindex(out))
    else
        out = resample_conserving_flux(new_wave, old_wave, fluxp)
        permutedims(out, (2,3,1))
    end
end


function multiplicative_exponentials(λ::AbstractVector, p::AbstractVector)
    λmin, λmax = extrema(λ)
    λ̄ = @. (λ - λmin) / (λmax - λmin)
    # Equation 2 of Rupke et al. (2017): https://arxiv.org/pdf/1708.05139.pdf
    e1 = @. p[1] * exp(-p[2] * λ̄)
    e2 = @. p[3] * exp(-p[4] * (1 - λ̄))
    e3 = @. p[5] * (1 - exp(-p[6] * λ̄))
    e4 = @. p[7] * (1 - exp(-p[8] * (1 - λ̄)))
    return [e1 e2 e3 e4]
end

"""
    test_line_snr(λ0, half_window_size, λ, I)

Perform a quick test to estimate an emission line's signal-to-noise ratio just based on the
maximum of the spectrum within a given window over the RMS of the surrounding region. Some 
processing is done to remove outliers (convolution) and any linear trends in the continuum.

# Arguments {T<:Real}
- `λ0::Real`: The central wavelength of the line to be tested in microns
- `half_window_size::Real`: Half the size of the window in microns
- `λ::Vector{T}`: The wavelength vector
- `I::vector{T}`: The intensity vector
"""
function test_line_snr(λ0::Real, half_window_size::Real, λ::Vector{T}, I::Vector{T}) where {T<:Real}

    # Line testing region
    region = (λ0 - half_window_size) .< λ .< (λ0 + half_window_size)
    @assert sum(region) > 10 "The spectrum does not cover the line in question sufficiently!"

    # Subtract linear trend
    m = (mean(I[region][end-19:end]) - mean(I[region][1:20])) / (λ[region][end-9] - λ[region][10])
    Ilin = mean(I[region][1:20]) .+ m.*(λ[region] .- λ[region][10])
    λsub = λ[region]
    Isub = I[region] .- Ilin

    # Smooth with a width of 3 pixels
    Iconv = convolveGaussian1D([zeros(9); Isub; zeros(9)], 7 .* ones(length(Isub)+18))

    # Maximum within the center of the region of the SMOOTHED spectrum
    central = (λ0 - half_window_size/3) .< λsub .< (λ0 + half_window_size/3)
    # RMS to the left/right of the region of the UNSMOOTHED spectrum
    sides = ((λ0 - half_window_size) .< λsub .< (λ0 - half_window_size/3)) .| ((λ0 + half_window_size/3) .< λsub .< (λ0 + half_window_size))
    rms = nanstd(Isub[sides])

    # Sigma clipping
    mask = abs.(Isub[central] .- Iconv[10:end-9][central]) .> 3rms
    amp = nanmaximum(Iconv[10:end-9][central][.~mask])

    # Sigma clipping
    mask = abs.(Isub[sides] .- Iconv[10:end-9][sides]) .> 3rms
    rms = nanstd(Isub[sides][.~mask])

    # Rough estimate of the signal-to-noise ratio of the line
    amp/rms

end


"""
    model_line_residuals(λ, params, n_lines, n_comps, lines, flexible_wavesol, ext_curve, lsf,
        relative_flags, return_components) 

Create a model of the emission lines at the given wavelengths `λ`, given the parameter vector `params`.

Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments {S<:Integer}
- `λ::Vector{<:Real}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:Real}`: Parameter vector.
- `n_lines::S`: Number of lines being fit
- `n_comps::S`: Maximum number of profiles to be fit to any given line
- `lines::TransitionLines`: Object containing information about each transition line being fit.
- `flexible_wavesol::Bool`: Whether or not to allow small variations in tied velocity offsets, to account for a poor
wavelength solution in the data
- `ext_curve::Vector{<:Real}`: The extinction curve fit with model_{mir|opt}_continuum
- `lsf::Function`: A function giving the FWHM of the line-spread function in km/s as a function of rest-frame wavelength in microns.
- `relative_flags::BitVector`: BitVector giving flags for whether the amp, voff, and fwhm of additional line profiles should be
    parametrized relative to the main profile or not.
- `template_norm::Union{Nothing,Vector{<:Real}}`: The normalization PSF template that was fit using model_continuum
- `nuc_temp_fit::Bool`: Whether or not to apply the PSF normalization template
- `return_components::Bool=false`: Whether or not to return the individual components of the fit as a dictionary, in 
addition to the overall fit
"""
function model_line_residuals(λ::AbstractVector, params::AbstractVector, n_lines::S, n_comps::S, lines::TransitionLines, 
    flexible_wavesol::Bool, ext_curve::AbstractVector, lsf::Function, relative_flags::BitVector, template_norm::Union{Nothing,AbstractVector},
    nuc_temp_fit::Bool, return_components::Bool) where {S<:Integer}

    # Prepare outputs
    out_type = eltype(params)
    comps = Dict{String, Vector{out_type}}()
    contin = zeros(out_type, length(λ))

    # Flags for additional components being relative
    rel_amp, rel_voff, rel_fwhm = relative_flags

    pᵢ = 1
    # Add emission lines
    for k ∈ 1:n_lines
        amp_1 = voff_1 = fwhm_1 = nothing
        for j ∈ 1:n_comps
            if !isnothing(lines.profiles[k, j])
                # Unpack the components of the line
                amp = params[pᵢ]
                voff = params[pᵢ+1]
                # Check if using a flexible_wavesol tied voff -> if so there is an extra voff parameter
                if !isnothing(lines.tied_voff[k, j]) && flexible_wavesol && isone(j)
                    voff += params[pᵢ+2]
                    fwhm = params[pᵢ+3]
                    pᵢ += 4
                else
                    fwhm = params[pᵢ+2]
                    pᵢ += 3
                end
                if lines.profiles[k, j] == :GaussHermite
                    # Get additional h3, h4 components
                    h3 = params[pᵢ]
                    h4 = params[pᵢ+1]
                    pᵢ += 2
                elseif lines.profiles[k, j] == :Voigt
                    # Get additional mixing component, either from the tied position or the 
                    # individual position
                    η = params[pᵢ]
                    pᵢ += 1
                end

                # Save the j = 1 parameters for reference 
                if isone(j)
                    amp_1 = amp
                    voff_1 = voff
                    fwhm_1 = fwhm
                # For the additional components, we (optionally) parametrize them this way to essentially give them soft constraints
                # relative to the primary component
                else
                    if rel_amp
                        amp *= amp_1
                    end
                    if rel_voff
                        voff += voff_1
                    end
                    if rel_fwhm
                        fwhm *= fwhm_1
                    end
                end

                # Broaden the FWHM by the instrumental FWHM at the location of the line
                fwhm_inst = lsf(lines.λ₀[k])
                fwhm = hypot(fwhm, fwhm_inst)

                # Convert voff in km/s to mean wavelength in μm
                mean_μm = Doppler_shift_λ(lines.λ₀[k], voff)
                # Convert FWHM from km/s to μm
                fwhm_μm = Doppler_shift_λ(lines.λ₀[k], fwhm/2) - Doppler_shift_λ(lines.λ₀[k], -fwhm/2)

                # Evaluate line profile
                if lines.profiles[k, j] == :Gaussian
                    comps["line_$(k)_$(j)"] = Gaussian.(λ, amp, mean_μm, fwhm_μm)
                elseif lines.profiles[k, j] == :Lorentzian
                    comps["line_$(k)_$(j)"] = Lorentzian.(λ, amp, mean_μm, fwhm_μm)
                elseif lines.profiles[k, j] == :GaussHermite
                    comps["line_$(k)_$(j)"] = GaussHermite.(λ, amp, mean_μm, fwhm_μm, h3, h4)
                elseif lines.profiles[k, j] == :Voigt
                    comps["line_$(k)_$(j)"] = Voigt.(λ, amp, mean_μm, fwhm_μm, η)
                else
                    error("Unrecognized line profile $(lines.profiles[k, j])!")
                end

                # Add the line profile into the overall model
                contin .+= comps["line_$(k)_$(j)"]
            end
        end
    end

    # Apply extinction
    contin .*= ext_curve
    # Apply PSF normalization
    if nuc_temp_fit
        contin .*= template_norm
    end

    # Return components if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency --> not allocating the dictionary improves performance DRAMATICALLY
function model_line_residuals(λ::AbstractVector, params::AbstractVector, n_lines::S, n_comps::S, lines::TransitionLines, 
    flexible_wavesol::Bool, ext_curve::AbstractVector, lsf::Function, relative_flags::BitVector, template_norm::Union{Nothing,AbstractVector},
    nuc_temp_fit::Bool) where {S<:Integer}

    # Prepare outputs
    out_type = eltype(params)
    contin = zeros(out_type, length(λ))

    # Flags for additional components being relative
    rel_amp, rel_voff, rel_fwhm = relative_flags

    pᵢ = 1
    # Add emission lines
    for k ∈ 1:n_lines
        amp_1 = voff_1 = fwhm_1 = nothing
        for j ∈ 1:n_comps
            if !isnothing(lines.profiles[k, j])
                # Unpack the components of the line
                amp = params[pᵢ]
                voff = params[pᵢ+1]
                # Check if using a flexible_wavesol tied voff -> if so there is an extra voff parameter
                if !isnothing(lines.tied_voff[k, j]) && flexible_wavesol && isone(j)
                    voff += params[pᵢ+2]
                    fwhm = params[pᵢ+3]
                    pᵢ += 4
                else
                    fwhm = params[pᵢ+2]
                    pᵢ += 3
                end
                if lines.profiles[k, j] == :GaussHermite
                    # Get additional h3, h4 components
                    h3 = params[pᵢ]
                    h4 = params[pᵢ+1]
                    pᵢ += 2
                elseif lines.profiles[k, j] == :Voigt
                    # Get additional mixing component, either from the tied position or the 
                    # individual position
                    η = params[pᵢ]
                    pᵢ += 1
                end

                # Save the j = 1 parameters for reference 
                if isone(j)
                    amp_1 = amp
                    voff_1 = voff
                    fwhm_1 = fwhm
                # For the additional components, we (optionally) parametrize them this way to essentially give them soft constraints
                # relative to the primary component
                else
                    if rel_amp
                        amp *= amp_1
                    end
                    if rel_voff
                        voff += voff_1
                    end
                    if rel_fwhm
                        fwhm *= fwhm_1
                    end
                end

                # Broaden the FWHM by the instrumental FWHM at the location of the line
                fwhm_inst = lsf(lines.λ₀[k])
                fwhm = hypot(fwhm, fwhm_inst)

                # Convert voff in km/s to mean wavelength in μm
                mean_μm = Doppler_shift_λ(lines.λ₀[k], voff)
                # Convert FWHM from km/s to μm
                fwhm_μm = Doppler_shift_λ(lines.λ₀[k], fwhm/2) - Doppler_shift_λ(lines.λ₀[k], -fwhm/2)

                # Evaluate line profile
                if lines.profiles[k, j] == :Gaussian
                    contin .+= Gaussian.(λ, amp, mean_μm, fwhm_μm)
                elseif lines.profiles[k, j] == :Lorentzian
                    contin .+= Lorentzian.(λ, amp, mean_μm, fwhm_μm)
                elseif lines.profiles[k, j] == :GaussHermite
                    contin .+= GaussHermite.(λ, amp, mean_μm, fwhm_μm, h3, h4)
                elseif lines.profiles[k, j] == :Voigt
                    contin .+= Voigt.(λ, amp, mean_μm, fwhm_μm, η)
                else
                    error("Unrecognized line profile $(lines.profiles[k, j])!")
                end
            end
        end
    end
    
    # Apply extinction
    contin .*= ext_curve
    # Apply PSF normalization
    if nuc_temp_fit
        contin .*= template_norm
    end

    contin

end


"""
    calculate_flux(profile, amp, amp_err, peak, peak_err, fwhm, fwhm_err; <keyword_args>)

Calculate the integrated flux of a spectral feature, i.e. a PAH or emission line. Calculates the integral
of the feature profile, using an analytic form if available, otherwise integrating numerically with QuadGK.
"""
function calculate_flux(profile::Symbol, λ::AbstractVector, amp::T, amp_err::T, peak::T, peak_err::T, fwhm::T, fwhm_err::T;
    asym::Union{T,Nothing}=nothing, asym_err::Union{T,Nothing}=nothing, m::Union{T,Nothing}=nothing, m_err::Union{T,Nothing}=nothing, 
    ν::Union{T,Nothing}=nothing, ν_err::Union{T,Nothing}=nothing, h3::Union{T,Nothing}=nothing, 
    h3_err::Union{T,Nothing}=nothing, h4::Union{T,Nothing}=nothing, h4_err::Union{T,Nothing}=nothing, 
    η::Union{T,Nothing}=nothing, η_err::Union{T,Nothing}=nothing, propagate_err::Bool=true) where {T<:Real}

    # Evaluate the line profiles according to whether there is a simple analytic form
    # otherwise, integrate numerically with quadgk
    if profile == :Drude
        if isnothing(asym) || iszero(asym) 
            # (integral = π/2 * A * fwhm)
            flux, f_err = propagate_err ? ∫Drude(amp, amp_err, fwhm, fwhm_err) : (∫Drude(amp, fwhm), 0.)
        else
            flux = NumericalIntegration.integrate(λ, Drude.(λ, amp, peak, fwhm, asym), Trapezoidal())
            if propagate_err
                err_l = abs(NumericalIntegration.integrate(λ, Drude.(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), asym-asym_err), Trapezoidal()))
                err_u = abs(NumericalIntegration.integrate(λ, Drude.(λ, amp+amp_err, peak, fwhm+fwhm_err, asym+asym_err), Trapezoidal()))
                f_err = (err_l + err_u)/2
            else
                f_err = 0.
            end
        end
    elseif profile == :PearsonIV
        flux = ∫PearsonIV(amp, fwhm, m, ν)
        if propagate_err
            e_upp = ∫PearsonIV(amp+amp_err, fwhm+fwhm_err, m+m_err, ν+ν_err) - flux
            e_low = flux - ∫PearsonIV(max(amp-amp_err, 0.), max(fwhm-fwhm_err, eps()), max(m-m_err, 0.5), ν-ν_err)
            f_err = (e_upp + e_low) / 2
        else
            f_err = 0.
        end
    elseif profile == :Gaussian
        # (integral = √(π / (4log(2))) * A * fwhm)
        flux, f_err = propagate_err ? ∫Gaussian(amp, amp_err, fwhm, fwhm_err) : (∫Gaussian(amp, fwhm), 0.)
    elseif profile == :Lorentzian
        # (integral is the same as a Drude profile)
        flux, f_err = propagate_err ? ∫Lorentzian(amp, amp_err, fwhm, fwhm_err) : (∫Lorentzian(amp, fwhm), 0.)
    elseif profile == :Voigt
        # (integral is an interpolation between Gaussian and Lorentzian)
        flux, f_err = propagate_err ? ∫Voigt(amp, amp_err, fwhm, fwhm_err, η, η_err) : (∫Voigt(amp, fwhm, η), 0.)
    elseif profile == :GaussHermite
        # there is no general analytical solution (that I know of) for integrated Gauss-Hermite functions
        # (one probably exists but I'm too lazy to find it)
        # so we just do numerical integration for this case (trapezoid rule)
        flux = NumericalIntegration.integrate(λ, GaussHermite.(λ, amp, peak, fwhm, h3, h4), Trapezoidal())
        # estimate error by evaluating the integral at +/- 1 sigma
        if propagate_err
            err_l = abs(NumericalIntegration.integrate(λ, GaussHermite.(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), 
                        h3-h3_err, h4-h4_err), Trapezoidal()))
            err_u = abs(NumericalIntegration.integrate(λ, GaussHermite.(λ, amp+amp_err, peak, fwhm+fwhm_err, 
                        h3+h3_err, h4+h4_err), Trapezoidal()))
            f_err = (err_l + err_u)/2
        else
            f_err = 0.
        end
    else
        error("Unrecognized line profile $profile")
    end

    flux, f_err
end


"""
    calculate_composite_params(λ, flux, λ0, fwhm_inst)

Calculate the w80 (width containing 80% of the flux) and Δv (asymmetry parameter), both in km/s,
for a line profile.
"""
function calculate_composite_params(λ::AbstractVector, flux::AbstractVector, λ0::Real, fwhm_inst::Real)

    # Get the cumulative distribution function
    m = flux .> 0
    if sum(m) < 2
        return 0., 0., 0., 0.
    end

    line_cdf = cumsum(flux[m] ./ sum(flux[m]))
    velocity = Doppler_shift_v.(λ[m], λ0)

    # Cut below a threshold, otherwise Spline1D produces NaNs for some reason
    w = (line_cdf .> 0.001) .& (line_cdf .< 0.999)
    line_cdf = line_cdf[w]
    velocity = velocity[w]
    if length(line_cdf) < 4
        return 0., 0., 0., 0.
    end
    # Cut any pixels that are not increasing the CDF (otherwise may cause the spline fit to fail)
    wd = BitVector([1; diff(line_cdf) .> 0.])
    line_cdf = line_cdf[wd]
    velocity = velocity[wd]
    if length(line_cdf) < 4
        return 0., 0., 0., 0
    end

    # Interpolate to find where velocity is at 5, 10 and 90, and 95%
    vinterp = CubicSpline(velocity, line_cdf; extrapolate=true)
    v5 = vinterp(0.05)
    v10 = vinterp(0.10)
    vmed = vinterp(0.50)
    v90 = vinterp(0.90)
    v95 = vinterp(0.95)

    # Calculate W80
    w80 = v90 - v10
    # Correct for intrumental line spread function (w80 = 1.09FWHM for a Gaussian)
    w80_inst = 1.09 * fwhm_inst
    w80 = sqrt(clamp(w80^2 - w80_inst^2, 0., Inf))

    # Calculate peak velocity
    finterp = CubicSpline(flux[m][w][wd], velocity; extrapolate=true)
    guess = velocity[nanargmax(flux[m][w][wd])]
    res = Optim.optimize(v -> -finterp(v), guess-50., guess+50.)
    vpeak = res.minimizer[1]

    # Calculate Δv (see Harrison et al. 2014: https://ui.adsabs.harvard.edu/abs/2014MNRAS.441.3306H/abstract)
    Δv = (v5 + v95)/2 

    w80, Δv, vmed, vpeak
end
