#=
This file is not intended to be directly accessed by the user when fitting
IFU cubes. Rather, it contains miscellaneous utility functions to aid in correcting
and fitting data.  As such, nothing in this file is exported, so if the user wishes
to access it they must use the "Loki" prefix.
=#

# CONSTANTS

const C_KMS = 299792.458*u"km/s"           # Speed of light in km/s
const h_ERGS = 6.62607015e-27*u"erg*s"     # Planck constant in erg*s
const kB_ERGK = 1.380649e-16*u"erg/K"      # Boltzmann constant in erg/K

# Constants for the Planck function
const Bν_1_μm = uconvert(u"erg/s/cm^2/Hz/sr*μm^3", 2h_ERGS*C_KMS)
const Bν_2_μm = uconvert(u"μm*K", h_ERGS*C_KMS/kB_ERGK)
const Bν_1_AA = uconvert(u"erg/s/cm^2/Hz/sr*angstrom^3", Bν_1_μm)
const Bν_2_AA = uconvert(u"angstrom*K", Bν_2_μm)
const Bλ_1_μm = uconvert(u"erg/s/cm^2/μm/sr*μm^5", 2h_ERGS*C_KMS^2)
const Bλ_2_μm = Bν_2_μm
const Bλ_1_AA = uconvert(u"erg/s/cm^2/angstrom/sr*angstrom^5", Bλ_1_μm)
const Bλ_2_AA = Bν_2_AA

# Wein's law constant of proportionality in μm*K
const b_Wein = 2897.771955*u"μm*K"

# A few other random constants
const o_peak = 10.0178u"μm"


# Load the appropriate templates that we need into the cache
function _load_dust_templates(silicate_absorption::String, fit_ch_abs::Bool, use_pah_templates::Bool, 
    λunit::Unitful.Units, Iunit::Unitful.Units)

    dust_profiles = Dict{String, Tuple{Vector{<:QWave},Vector{<:AbstractFloat}}}()
    dust_interpolators = Dict{String, Spline1D}()

    if silicate_absorption == "d+"
        # Save the Donnan et al. 2022 profile as a constant
        dust_profiles["dp"] = silicate_dp()
        dust_interpolators["dp"] = Spline1D(ustrip.(uconvert.(λunit, dust_profiles["dp"][1])), 
            dust_profiles["dp"][2]; k=3, bc="nearest")
    elseif silicate_absorption == "ct"
        # Save the Chiar+Tielens 2005 profile as a constant
        dust_profiles["ct"] = silicate_ct()
        dust_interpolators["ct"] = Spline1D(ustrip.(uconvert.(λunit, dust_profiles["ct"][1])), 
            dust_profiles["ct"][2]; k=3, bc="nearest")
    elseif silicate_absorption == "kvt"
        # Save the KVT profile as a constant
        dust_profiles["kvt"] = silicate_kvt()
        dust_interpolators["kvt"] = Spline1D(ustrip.(uconvert.(λunit, dust_profiles["kvt"][1])), 
            dust_profiles["kvt"][2], k=2, bc="nearest")
        dust_interpolators["kvt_end"] = Spline1D(ustrip.(uconvert.(λunit, [dust_profiles["kvt"][1][end], 
            dust_profiles["kvt"][1][end]+2u"μm"])), [dust_profiles["kvt"][2][end], 0.], k=1, bc="nearest")
    end
    # Save the OHM 1992 profile as a constant
    dust_profiles["ohm"] = silicate_ohm()
    dust_interpolators["ohm"] = Spline1D(ustrip.(uconvert.(λunit, dust_profiles["ohm"][1])), 
        dust_profiles["ohm"][2]; k=3, bc="nearest")

    if fit_ch_abs
        # Save the Ice+CH optical depth template as a constant
        ice_wave, ice_prof, ch_wave, ch_prof = read_ice_ch_temps()
        dust_profiles["ice"] = (ice_wave, ice_prof)
        dust_profiles["ch"] = (ch_wave, ch_prof)
        dust_interpolators["ice"] = Spline1D(ustrip.(uconvert.(λunit, dust_profiles["ice"][1])), 
            dust_profiles["ice"][2]; k=3)
        dust_interpolators["ch"] = Spline1D(ustrip.(uconvert.(λunit, dust_profiles["ch"][1])), 
            dust_profiles["ch"][2]; k=3)
    end

    # Save the Smith+2006 PAH templates as constants
    if use_pah_templates
        SmithTemps = read_smith_temps()
        # may need to do a unit conversion 
        ST2 = match_fluxunits.(SmithTemps[2].*u"erg/s/cm^2/Hz/sr", 1.0*Iunit, SmithTemps[1])
        ST2 = ST2 ./ maximum(ST2)
        dust_interpolators["smith3"] = Spline1D(ustrip.(uconvert.(λunit, SmithTemps[1])), ST2; k=3, bc="nearest")
        ST4 = match_fluxunits.(SmithTemps[4].*u"erg/s/cm^2/Hz/sr", 1.0*Iunit, SmithTemps[3])
        ST4 = ST4 ./ maximum(ST4)
        dust_interpolators["smith4"] = Spline1D(ustrip.(uconvert.(λunit, SmithTemps[3])), ST4; k=3, bc="nearest")
    end

    return dust_profiles, dust_interpolators
end

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


# Function for converting between per-unit-frequency units and per-unit-wavelength units
function fluxconvert(Iν::QGeneralPerFreq, λ::QLength)
    uconvert(unit(Iν)*u"Hz"/unit(λ), Iν * C_KMS / λ^2)
end
function fluxconvert(Iλ::QGeneralPerWave, λ::QLength)
    uconvert(unit(Iλ)*unit(λ)/u"Hz", Iλ * λ^2 / C_KMS)
end

# Function that smartly decides whether or not to convert the units of the first argument to match 
# the units of the second argument
match_fluxunits(I_mod::Q1, I_ref::Q2, ::QLength) where {Q1<:QPerAng,Q2<:QPerum} = uconvert(unit(I_ref), I_mod)
match_fluxunits(I_mod::Q1, I_ref::Q2, ::QLength) where {Q1<:QPerum,Q2<:QPerAng} = uconvert(unit(I_ref), I_mod)
match_fluxunits(I_mod::Q, ::Q, ::QLength) where {Q<:QGeneralPerFreq} = I_mod
match_fluxunits(I_mod::Q, ::Q, ::QLength) where {Q<:QGeneralPerWave} = I_mod
function match_fluxunits(I_mod::Q1, I_ref::Q2, λ::QLength) where {
    Q1<:Union{QGeneralPerWave,QGeneralPerFreq},
    Q2<:Union{QGeneralPerWave,QGeneralPerFreq}
    }
    uconvert(unit(I_ref), fluxconvert(I_mod, λ))
end


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
function convolveGaussian1D(flux::Vector{T}, fwhm::Vector{<:Real}) where {T<:Number}

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
function hermite(x::Vector{T}, n::Integer) where {T<:Number}
    if iszero(n)
        ones(T, length(x))
    elseif isone(n)
        2 .* x
    else
        2 .* x .* hermite(x, n-1) .- 2 .* (n.-1) .* hermite(x, n-2)
    end
end


############################################### CONTINUUM FUNCTIONS ##############################################


"""
    Blackbody_ν(λ, Temp)

Return the Blackbody function Bν (per unit FREQUENCY) in erg/s/cm^2/Hz/sr.
The two methods ensure that the wavelength can be given in either μm or angstroms
and the returned units will still be the same.

This function will be called frequently inside the fitting routine, so it's
gotta go fast.
"""
@inline function Blackbody_ν(λ::Qum, Temp::QTemp)
    Bν_1_μm/λ^3 / expm1(Bν_2_μm/(λ*Temp))
end
@inline function Blackbody_ν(λ::QAng, Temp::QTemp)
    Bν_1_AA/λ^3 / expm1(Bν_2_AA/(λ*Temp))
end
@inline function Blackbody_λ(λ::Qum, Temp::QTemp)
    Bλ_1_μm/λ^5 / expm1(Bλ_2_μm/(λ*Temp))
end
@inline function Blackbody_λ(λ::QAng, Temp::QTemp)
    Bλ_1_AA/λ^5 / expm1(Bλ_2_AA/(λ*Temp))
end
@inline function Blackbody(λ::QWave, Temp::QTemp, ::QGeneralPerFreq)
    Blackbody_ν(λ, Temp)
end
@inline function Blackbody(λ::QWave, Temp::QTemp, ::QGeneralPerWave)
    Blackbody_λ(λ, Temp)
end

# modified blackbody with a dust emissivity proportional to 1/λ^2
@inline function Blackbody_modified(λ::QWave, Temp::QTemp, Iunit::QSIntensity)
    Blackbody(λ, Temp, Iunit) * uconvert(NoUnits, 9.7u"μm"/λ)^2
end


"""
    Wein(Temp)

Return the peak wavelength (in μm) of a Blackbody spectrum at a given temperature `Temp`,
using Wein's Displacement Law.
"""
@inline function Wein(Temp::QTemp)
    b_Wein / Temp
end


"""
    power_law(λ, α, ref_λ)

Simple power law function where the flux is proportional to the wavelength to the power alpha,
normalized at 9.7 um.
"""
@inline function power_law(λ::Vector{T}, α::Real, ref_λ::T) where {T<:Number}
    uconvert.(NoUnits, λ./ref_λ).^α
end


"""
    silicate_emission(λ, A, T, Cf, τ_warm, τ_cold, λ_peak)

A hot silicate dust emission profile, i.e. Gallimore et al. (2010), with an amplitude A,
temperature T, covering fraction Cf, and optical depths τ_warm and τ_cold.
"""
function silicate_emission(λ::Vector{S}, T::QTemp, Cf::Real, τ_warm::Real, 
    τ_cold::Real, λ_peak::S, Iunit::QSIntensity, cube_fitter::CubeFitter) where {S<:QWave}
    # Δλ = uconvert(unit(λ_peak), o_peak - λ_peak)
    # λshift = λ .+ Δλ
    ext_curve = τ_ohm(λ, cube_fitter)
    bb = Blackbody.(λ, T, Iunit) .* (1 .- extinction_factor.(ext_curve, τ_warm, screen=true))
    bb .* (1 .- Cf) .+ bb .* Cf .* extinction_factor.(ext_curve, τ_cold, screen=true)
end

################################################# PAH PROFILES ################################################


"""
    Drude(x, A, μ, FWHM, asym)

Calculate a Drude profile at location `x`, with amplitude `A`, central value `μ`, and full-width at half-max `FWHM`
Optional asymmetry parameter `asym`
"""
@inline function Drude(x, A, μ, FWHM, asym) 
    γ = 2FWHM / (1 + exp(ustrip(asym*(x-μ))))
    A * (γ/μ)^2 / ((x/μ - μ/x)^2 + (γ/μ)^2)
end


"""
    PearsonIV(x, A, μ, a, m, ν)

Calculate a Pearson Type-IV profile at location `x`, with amplitude `A`, unextinguished central value `μ`, width
parameter `a`, index `m`, and exponential cutoff `ν`.

See Pearson (1895), and https://iopscience.iop.org/article/10.3847/1538-4365/ac4989/pdf
"""
function PearsonIV(x::T, A::Number, μ::T, a::T, m::Number, ν::Number) where {T<:Number}
    n = (1 + (-ν/(2m))^2)^-m * exp(-ν * atan(-ν/(2m)))
    A/n * (1 + ((x - μ)/a)^2)^-m * exp(-ν * atan((x - μ)/a))
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
function convolve_losvd(_templates::AbstractArray{T}, vsyst::S, v::S, σ::S, vres::S, 
    npix::U; temp_fft::Bool=false, npad_in::U=0) where {T<:Number,S<:Number,U<:Integer}

    if temp_fft
        @assert npad_in > 0 "npad_in must be specified for inputs that are already FFT'ed!"
    end

    templates = _templates
    if ndims(_templates) == 1
        templates = reshape(_templates, (length(_templates), 1))
    end

    # Check if the templates are already FFT'ed
    if !temp_fft
        # Pad with 0s up to a factor of small primes to increase efficiency 
        s = size(templates)
        npad = nextprod([2,3,5], s[1])
        if npad > s[1]
            temps = zeros(eltype(templates), npad, s[2])
            for j in axes(templates, 2)
                temps[1:s[1], j] .= templates[:, j]
            end
        else
            temps = templates
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
    # and then get the inverse Fourier transform of the resultant distribution to get the convolution
    # see, i.e. https://en.wikipedia.org/wiki/Convolution_theorem
    template_convolved = irfft(temp_rfft .* conj.(exp.(1im .* ω .* V .- Σ.^2 .* ω.^2 ./ 2)), npad, 1)

    # Take only enough pixels to match the length of the input spectrum
    @view template_convolved[1:npix, :]
end


# get constraint equations for regularization for the least-squares fitting of the stellar templates
function add_reg_constraints!(A::Matrix{<:Real}, nλ::Int, cube_fitter::CubeFitter)
    # reshape into N_ages x N_logzs sized array
    reg_dims = (length(cube_fitter.ssps.ages), length(cube_fitter.ssps.logzs))
    # get a "view" so modifying a also modifies A
    @assert size(A, 2) == prod(reg_dims)
    a = reshape(A, (size(A, 1), reg_dims...))
    reg_diffs = [1., -2., 1.] .* cube_fitter.fitting.ssp_regularize
    # add constraint equations for regularization
    i = nλ+1
    for j in axes(a, 2)
        for k in axes(a, 3)
            if 1 < k < size(a, 3)
                a[i, j, k-1:k+1] .= reg_diffs
            end
            if 1 < j < size(a, 2)
                a[i, j-1:j+1, k] .+= reg_diffs
            end
            i += 1
        end
    end
end


# perform non-negative least squares fitting on the stellar template grids 
function stellar_populations_nnls(s::Spaxel, contin::Vector{<:Real}, ext_stars::Vector{<:Real}, 
    stel_vel::QVelocity, stel_sig::QVelocity, cube_fitter::CubeFitter; do_gaps::Bool=true, mask_lines::Bool=true)

    # prepare buffer arrays for NNLS
    nλ = length(s.λ)
    if cube_fitter.fitting.ssp_regularize > 0.
        reg_dims = (length(cube_fitter.ssps.ages), length(cube_fitter.ssps.logzs))
    else
        reg_dims = (0, 0)
    end
    A = zeros(nλ+prod(reg_dims), cube_fitter.n_ssps)
    b = zeros(nλ+prod(reg_dims))

    # subtract everything else from the data to create a residual stellar spectrum 
    b[1:nλ] .= s.I .- contin

    # calculate the convolved stellar templates
    gap_masks = do_gaps ? get_gap_masks(s.λ, cube_fitter.spectral_region.gaps) : [trues(length(s.λ))]
    for (gi, gap_mask) in enumerate(gap_masks)
        # split if the spectrum has a few separated regions
        A[(1:nλ)[gap_mask], :] .= convolve_losvd(ustrip.(cube_fitter.ssps.templates), 
            cube_fitter.ssps.vsysts[gi], stel_vel, stel_sig, s.vres, sum(gap_mask))
    end

    # divide out the solid angle and apply the extinction
    A[1:nλ, :] .*= ext_stars ./ s.area_sr
    # normalize the stellar templates
    stellar_N = haskey(s.aux, "stellar_norm") ? ustrip(s.aux["stellar_norm"]) : nanmedian(A[1:nλ, :])
    # stellar_N = ustrip(nanmedian(cube_fitter.ssps.templates) * nanmedian(ext_stars ./ s.area_sr))
    A[1:nλ, :] ./= stellar_N
    stellar_norm = stellar_N*unit(cube_fitter.ssps.templates[1])/u"sr"  # should be specific intensity per unit mass
    # weight by the errors
    A[1:nλ, :] ./= s.σ
    b[1:nλ] ./= s.σ
    # add the regularization constraints
    # (we dont need to modify b here as it is already initialized with 0s)
    if cube_fitter.fitting.ssp_regularize > 0.
        add_reg_constraints!(A, nλ, cube_fitter)
    end
    # perform a non-negative least-squares fit
    if !haskey(s.aux, "stellar_weights") || isnothing(s.aux["stellar_weights"])
        ml_extended = falses(length(b))
        # if doing a joint fit, DONT mask out the lines
        if mask_lines
            ml_extended[1:nλ] .= s.mask_lines
        end
        weights = nonneg_lsq(A[.~ml_extended, :], b[.~ml_extended], alg=:fnnls)  # mask out the emission lines!
    else
        weights = reshape(s.aux["stellar_weights"], cube_fitter.n_ssps, 1)
    end

    # get the final stellar continuum with a matrix multiplication
    ssp_contin = ((A[1:nλ, :].*s.σ) * weights)[:,1]

    return ssp_contin, stellar_norm, weights
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
    A * exp(-(x-μ)^2 / (2σ^2))
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
@inline function Lorentzian(x, A, μ, FWHM) 
    A * (FWHM/2)^2 / ((x-μ)^2 + (FWHM/2)^2)
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
    G = 1/√(2π * σ^2) * exp(-(x-μ)^2 / (2σ^2))
    # Normalized Lorentzian
    L = 1/π * (FWHM/2) / ((x-μ)^2 + (FWHM/2)^2)

    # Normalize the function so that the integral is given by this
    I = ∫Voigt(A, FWHM, η)

    # Mix the two distributions with the mixing parameter η
    I * (η * G + (1 - η) * L)
end


############################################## EXTINCTION PROFILES #############################################


# Return the extinction factor for stars and gas separately
function extinction_profiles(λ::Vector{<:QWave}, params::Vector{<:Real}, pstart::Integer, fit_uv_bump::Bool,
    extinction_curve::String)

    E_BV, E_BV_factor = params[pstart], params[pstart+1]
    δ_uv = 0.85/1.9
    dp = 2
    if fit_uv_bump
        δ_uv = params[pstart+2]
        dp += 1
    end

    if extinction_curve == "calz"
        if fit_uv_bump
            att_gas = extinction_calzetti.(λ, E_BV, δ_uv)
        else
            att_gas = extinction_calzetti.(λ, E_BV)
        end
    elseif extinction_curve == "ccm"
        att_gas = extinction_cardelli.(λ, E_BV)
    else
        error("Unrecognized extinction curve: $(extinction_curve)")
    end
    att_stars = att_gas.^E_BV_factor

    att_gas, att_stars, dp
end


"""
    extinction_calzetti(λ, E_BV; δ_uv=0., Cf=0., Rv=4.05)

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
function extinction_calzetti(λ::QWave, E_BV::Real; Rv::Real=4.05)
    # eq. (4) from Calzetti et al. (2000)
    10^(-0.4 * E_BV * _calzetti_kprime_curve(ustrip(uconvert(u"μm", λ)), Rv))
end

function extinction_calzetti(λ::QWave, E_BV::Real, δ_uv::Real; Rv::Real=4.05)

    # eq. (4) from Calzetti et al. (2000)
    λ_um = uconvert(u"μm", λ)
    kprime = _calzetti_kprime_curve(ustrip(λ_um), Rv)

    # Calculate the UV bump 
    # Kriek & Conroy (2013) eq. (3): relation between UV bump amplitude Eb and slope δ_uv
    Eb = 0.85 - 1.9*δ_uv
    # Drude profile parametrizes the UV bump (Kriek & Conroy 2013, eq. (2))
    kprime += Drude(λ_um, Eb, 0.2175u"μm", 0.035u"μm", 0.0/u"μm")

    # Kriek & Conroy (2013) eq. (1) 
    10^(-0.4 * E_BV * kprime * (λ_um / 0.55u"μm")^δ_uv)
end

function _calzetti_kprime_curve(λ::Real, Rv::Real)

    # eq. (4) from Calzetti et al. (2000)
    x = 1/λ
    if λ ≥ 0.63
        kprime = 2.659 * (-1.857 + 1.040x) + Rv
    else
        kprime = 2.659 * (-2.156 + 1.509x - 0.198x^2 + 0.011x^3) + Rv
    end

    kprime
end



"""
    extinction_cardelli(λ, E_BV[, Rv])

Calculate the attenuation factor for a given wavelength range `λ` with a 
reddening of `E_BV` and selective extinction ratio `Rv`, using the
Cardelli et al. (1989) galactic extinction curve.

This function has been adapted from BADASS (Sexton et al. 2021), which
in turn has been adapted from the IDL Astrolib library.

# Arguments
- `λ`: The wavelength vector in microns
- `E_BV`: The color excess E(B-V) in magnitudes
- `Rv`: The ratio of total selective extinction R(V) = A(V)/E(B-V)

# Returns
- The extinction factor, 10^(-0.4*A(V)*(a(λ)+b(λ)/R(V)))

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
function extinction_cardelli(λ::QWave, E_BV::Real; Rv::Real=3.10)

    # inverse wavelength (microns)
    x = ustrip(1.0/uconvert(u"μm", λ))

    # Correction invalid for any x > 11
    if x > 11.0
        @warn "Input wavelength vector has values outside the allowable range of 1/λ < 11. Returning ones."
        return one(typeof(ustrip(λ)))
    end
    a = zero(typeof(ustrip(λ)))
    b = zero(typeof(ustrip(λ)))

    # Infrared
    if 0.3 < x < 1.1
        a = 0.574x^1.61
        b = -0.527x^1.61
    # Optical/NIR
    elseif 1.1 ≤ x < 3.3
        y = x - 1.82
        a = 1.0 + 0.104y - 0.609y^2 + 0.701y^3 + 1.137y^4 - 1.718y^5 - 0.827y^6 + 1.647y^7 - 0.505y^8
        b = 1.952y + 2.908y^2 - 3.989y^3 - 7.985y^4 + 11.102y^5 + 5.491y^6 - 10.805y^7 + 3.347y^8
    # Mid-UV
    elseif 3.3 ≤ x < 8.0
        y = x
        Fa = zero(typeof(ustrip(λ)))
        Fb = zero(typeof(ustrip(λ)))
        if y > 5.9
            Fa = -0.04473y^2 - 0.009779y^3
            Fb = 0.2130y^2 + 0.1207y^3
        end
        a = 1.752 - 0.316x - 0.104/((x - 4.67)^2 + 0.341) + Fa
        b = -3.090 + 1.825x + 1.206/((x - 4.62)^2 + 0.263) + Fb
    # Far-UV
    # ((just kind of let it apply to x > 11 even though it technically shouldnt...I'm sure this will never become a problem))
    elseif 8.0 ≤ x ≤ 11.0
        y = x - 8.0
        a = -1.703 - 0.628y + 0.137y^2 - 0.07y^3
        b = 13.67 + 4.257y - 0.42y^2 + 0.347y^3
    end

    # Calculate the extintion
    Av = Rv * E_BV
    aλ = Av * (a + b/Rv)
    10^(-0.4 * aλ)
end


function silicate_absorption(λ::Vector{<:QWave}, params::Vector{<:Real}, pstart::Integer, 
    cube_fitter::CubeFitter)

    fopt = fit_options(cube_fitter)
    screen = fopt.extinction_screen
    τ_97 = params[pstart]
    dp = 2
    ext_oli = ext_pyr = ext_for = nothing
    # First retrieve the normalized absorption profile
    if fopt.silicate_absorption == "kvt"
        β = params[pstart+1]
        ext = extinction_factor.(τ_kvt(λ, β, cube_fitter), τ_97, screen=screen)
    elseif fopt.silicate_absorption == "ct"
        ext = extinction_factor.(τ_ct(λ, cube_fitter), τ_97, screen=screen)
    elseif fopt.silicate_absorption == "ohm"
        ext = extinction_factor.(τ_ohm(λ, cube_fitter), τ_97, screen=screen)
    elseif fopt.silicate_absorption == "d+"
        β = params[pstart+1]
        ext = extinction_factor.(τ_dp(λ, β, cube_fitter), τ_97, screen=screen)
    elseif fopt.silicate_absorption == "decompose"
        τ_norm, τ_oli, τ_pyr, τ_for = τ_decompose(λ, params[pstart:pstart+3], fopt.κ_abs)
        τ_97 = 1.0
        ext_oli = extinction_factor.(τ_oli, τ_97, screen=screen)
        ext_pyr = extinction_factor.(τ_pyr, τ_97, screen=screen)
        ext_for = extinction_factor.(τ_for, τ_97, screen=screen)
        ext = extinction_factor.(τ_norm, τ_97, screen=screen)
        dp = 4
    elseif fopt.silicate_absorption == "custom"
        ext_curve = fopt.custom_ext_template(λ)
        ext = extinction_factor.(ext_curve, τ_97, screen=screen)
    else
        error("Unrecognized absorption type: $(fopt.silicate_absorption)")
    end
    # Then apply it at the appropriate level of 9.7um optical depth
    ext, dp, ext_oli, ext_pyr, ext_for
end


"""
    τ_kvt(λ, β)

Calculate the mixed silicate extinction profile based on Kemper, Vriend, & Tielens (2004) 

Function adapted from PAHFIT: Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)
"""
function τ_kvt(λ::Vector{<:QWave}, β::Real, cube_fitter::CubeFitter)

    # Get limits of the values that we have datapoints for via the kvt_prof constant
    λ_mx, λ_mn = cube_fitter.dust_profiles["kvt"][1][1], cube_fitter.dust_profiles["kvt"][1][end]

    # Interpolate based on the region of data 
    ext = zeros(typeof(β), length(λ))
    r1 = λ .≤ λ_mn
    if sum(r1) > 0
        ext[r1] .= @. cube_fitter.dust_profiles["kvt"][2][1] * exp(2.03 * ustrip(uconvert(u"μm", λ[r1] - λ_mn)))
    end
    r2 = λ_mn .< λ .< λ_mx
    if sum(r2) > 0
        ext[r2] .= cube_fitter.dust_interpolators["kvt"](ustrip.(λ[r2]))
    end
    r3 = λ_mx .< λ .< λ_mx+2u"μm"
    if sum(r3) > 0
        ext[r3] .= cube_fitter.dust_interpolators["kvt_end"](ustrip.(λ[r3]))
    end
    ext[ext .< 0] .= 0.

    # Add a drude profile around 18 microns
    ext .+= Drude.(λ, 0.4, 18.0u"μm", 4.446u"μm", 0.0/u"μm")

    @. (1 - β) * ext + β * (9.7u"μm"/λ)^1.7
end


"""
    τ_ct(λ)

Calculate the extinction profile based on Chiar & Tielens (2005)
"""
function τ_ct(λ::Vector{<:QWave}, cube_fitter::CubeFitter)

    mx = argmax(cube_fitter.dust_profiles["ct"][1])
    λ_mx = cube_fitter.dust_profiles["ct"][1][mx]

    ext = cube_fitter.dust_interpolators["ct"](ustrip.(λ))
    w_mx = findall(λ .> λ_mx)
    ext[w_mx] .= cube_fitter.dust_profiles["ct"][2][mx] .* (λ_mx./λ[w_mx]).^1.7

    ext
end


"""
    τ_ohm(λ)

Calculate the extinction profile based on Ossenkopf, Henning, & Mathis (1992)
"""
function τ_ohm(λ::Vector{<:QWave}, cube_fitter::CubeFitter)
    cube_fitter.dust_interpolators["ohm"](ustrip.(λ))
end


"""
    τ_dp(λ, β)

Calculate the mixed silicate extinction profile based on Donnan et al. (2022)
"""
function τ_dp(λ::Vector{<:QWave}, β::Real, cube_fitter::CubeFitter)
    # Add 1.7 power law, as in PAHFIT
    (1 .- β) .* cube_fitter.dust_interpolators["dp"](ustrip.(λ)) .+ β .* (9.8u"μm"./λ).^1.7
end


"""
    τ_decompose(λ, N_col, κ_abs)
Calculate the total silicate absorption optical depth given a series of column densities and
mass absorption coefficients.
"""
function τ_decompose(λ::Vector{<:QWave}, params::Vector{<:Number}, κ_abs::Vector{Spline1D})
    Ncol, β = params[1:3], params[end]
    τ_oli = Ncol[1] .* κ_abs[1](ustrip.(λ))
    τ_pyr = Ncol[1] .* Ncol[2] .* κ_abs[2](ustrip.(λ))
    τ_for = Ncol[1] .* Ncol[3] .* κ_abs[3](ustrip.(λ))
    τ_tot = @. τ_oli + τ_pyr + τ_for
    ind = argmin(abs.(λ .- 9.7u"μm"))
    τ_97 = τ_tot[ind] 
    ext = @. (1 - β) * τ_tot + β * τ_97 * (9.7/λ)^1.7
    ext, τ_oli, τ_pyr, τ_for
end


"""
    τ_ice(λ)

Calculate the ice extinction profiles
"""
function τ_ice(λ::Vector{<:QWave}, cube_fitter::CubeFitter)
    # Simple cubic spline interpolation
    cube_fitter.dust_interpolators["ice"](ustrip.(λ))
end


"""
    τ_ch(λ)

Calculate the CH extinction profiles
"""
function τ_ch(λ::Vector{<:QWave}, cube_fitter::CubeFitter)
    # Simple cubic spline interpolation
    cube_fitter.dust_interpolators["ch"](ustrip.(λ))
end


"""
    extinction_factor(ext, τ_97; [screen])

Calculate the extinction factor given the silicate absorption curve `ext` and the optical depth
at 9.7 microns, `τ_97`, either assuming a screen or mixed geometry.
"""
function extinction_factor(ext::Real, τ_97::Real; screen::Bool=false)
    if screen
        exp(-τ_97*ext)
    else
        iszero(τ_97) ? 1. : (1 - exp(-τ_97*ext)) / (τ_97*ext)
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

    # remove units
    funit = unit(flux[begin])
    out_units = Any[funit]

    # permute the third axis to the first axis for each input
    new_wavep = ustrip(new_wave)
    old_wavep = ustrip(old_wave)
    fluxp = ustrip(permutedims(flux, (3,1,2)))
    errp = nothing
    if !isnothing(err)
        errp = ustrip(permutedims(err, (3,1,2)))
        push!(out_units, funit)
    end
    maskp = nothing
    if !isnothing(mask)
        maskp = permutedims(mask, (3,1,2))
        push!(out_units, true)
    end

    # apply the function from SpectralResampling.jl
    if !isnothing(errp) && !isnothing(mask) 
        out = resample_conserving_flux(new_wavep, old_wavep, fluxp, errp, maskp)
        Tuple(permutedims(out[i], (2,3,1))*out_units[i] for i in eachindex(out))
    elseif !isnothing(errp)
        out = resample_conserving_flux(new_wavep, old_wavep, fluxp, errp)
        Tuple(permutedims(out[i], (2,3,1))*out_units[i] for i in eachindex(out))
    elseif !isnothing(maskp)
        out = resample_conserving_flux(new_wavep, old_wavep, fluxp, maskp)
        Tuple(permutedims(out[i], (2,3,1))*out_units[i] for i in eachindex(out))
    else
        out = resample_conserving_flux(new_wavep, old_wavep, fluxp)
        permutedims(out, (2,3,1))*out_units[1]
    end
end


function multiplicative_exponentials(λ::Vector{<:Number}, p::Vector{<:Number})
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
function test_line_snr(λ0::T, half_window_size::T, λ::Vector{T}, I::Vector{<:Number}) where {T<:QWave}

    # Line testing region
    region = (λ0 - half_window_size) .< λ .< (λ0 + half_window_size)
    @assert sum(region) > 40 "The spectrum does not cover the line in question sufficiently!"

    # Subtract linear trend
    m = (mean(I[region][end-19:end]) - mean(I[region][1:20])) / (λ[region][end-9] - λ[region][10])
    Ilin = mean(I[region][1:20]) .+ m.*(λ[region] .- λ[region][10])
    λsub = λ[region]
    Isub = I[region] .- Ilin

    # Smooth with a width of 3 pixels
    Iconv = convolveGaussian1D([zeros(eltype(Isub), 9); Isub; zeros(eltype(Isub), 9)], 7 .* ones(length(Isub)+18))

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

