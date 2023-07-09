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

# Save the KVT profile as a constant
const KVT_interp = Spline1D(kvt_prof[:, 1], kvt_prof[:, 2], k=1, bc="nearest")
const KVT_interp_end = Spline1D([kvt_prof[end, 1], kvt_prof[end, 1]+2], [kvt_prof[end, 2], 0.], k=1, bc="nearest")

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
    air_to_vacuum(λair)

Convert an air wavelength to a vacuum wavelength, correcting for the index of refraction of air.
Only converts wavelengths > 2000 angstroms.

Reference: FIREFLY code, Kyle B. Westfall, https://www.icg.port.ac.uk/firefly/
"""
function air_to_vacuum(λair::Real)
    # Do not modify < 2000 angstroms
    if λair < 0.2
        return λair
    end
    λvac = λair
    for _ in 1:3
        # wavenumber squared
        s² = (1/λvac)^2
        n = 1.0 + 5.792105e-2/(238.0185 - s²) + 1.67917e-3/(57.362 - s²)
        λvac = λair * n
    end
    λvac
end


"""
    convolveGaussian1D(flux, σ)

Convolve a spectrum by a Gaussian with different sigma for every pixel.
Extension of scipy.ndimage.gaussian_filter1d originally written by Michele Cappellari for pPXF.
"""
function convolveGaussian1D(flux::Vector{<:Real}, σ::Vector{<:Real})
    # Impose a minimum width of 0.01 pixels
    σ_conv = clamp.(σ, 0.01, Inf)

    # Kernel size
    p = ceil(Int, maximum(3σ_conv))
    m = 2p+1
    x2 = range(-p, p, length=m).^2
    # Fill in the kernel with shifted copies of the data
    n = length(flux)
    kernel = zeros(m, n)
    for j in 1:m
        kernel[j, 1+p:end-p] = flux[j:n-m+j]
    end
    # Normalized gaussian component
    gauss = exp.(.-x2 ./ (2 .* repeat(reshape(σ_conv, (1,length(σ_conv))), outer=(length(x2),1)).^2))
    gauss ./= sum(gauss, dims=1)

    conv_flux = sumdim(kernel .* gauss, 1)

    conv_flux, p
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
function hermite(x::Real, n::Integer)
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


"""
    power_law(λ, α)

Simple power law function where the flux is proportional to the wavelength to the power alpha,
normalized at 9.7 um.
"""
@inline function power_law(λ::Real, α::Real)
    (λ/9.7)^α
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
    Drude(x, A, μ, FWHM)

Calculate a Drude profile at location `x`, with amplitude `A`, central value `μ`, and full-width at half-max `FWHM`

Function adapted from PAHFIT: Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
"""
@inline function Drude(x::Real, A::Real, μ::Real, FWHM::Real)
    A * (FWHM/μ)^2 / ((x/μ - μ/x)^2 + (FWHM/μ)^2)
end


"""
    PearsonIV(x, A, μ, a, m, ν)

Calculate a Pearson Type-IV profile at location `x`, with amplitude `A`, unextinguished central value `μ`, width
parameter `a`, index `m`, and exponential cutoff `ν`.

See Pearson (1895), and https://iopscience.iop.org/article/10.3847/1538-4365/ac4989/pdf
"""
function PearsonIV(x::Real, A::Real, μ::Real, a::Real, m::Real, ν::Real)
    n = (1 + (-ν/(2m))^2)^-m * exp(-ν * atan(-ν/(2m)))
    A/n * (1 + ((x - μ)/a)^2)^-m * exp(-ν * atan((x - μ)/a))
end

########################################## STELLAR POP FUNCTIONS #########################################


"""
    convolve_losvd(ssp_templates, vsyst, v, σ, npix)

Convolve a set of stellar population templates with a line-of-sight velocity distribution (LOSVD)
to produce templates according to the fitted stellar kinematics. Uses the Fourier Transforms of 
the templates and the LOSVD to quickly calculate the convolution.

Adapted and simplified from the pPXF and BADASS python codes.
Cappellari (2017) http://adsabs.harvard.edu/abs/2017MNRAS.466..798C
"""
function convolve_losvd(ssps::Array{T, 2}, vsyst::Real, v::Real, σ::Real, velscale::Real, npix::Integer) where {T<:Real}

    # Normalize velocities to pixel units
    vsysts = vsyst / velscale
    vs = v / velscale
    σs = σ / velscale

    # Pad with 0s up to a factor of small primes to increase efficiency 
    s = size(ssps)
    npad = nextprod([2,3,5], s[1])
    ssp_temps = zeros(npad, s[2])
    for j in axes(ssps, 2)
        ssp_temps[1:s[1], j] = ssps[:, j]
    end

    # Calculate the Fourier transform of the templates
    ssp_rfft = rfft(ssp_temps, 1)
    # Calculate the Fourier transform of the LOSVD
    nl = size(ssp_rfft, 1)
    lvd_rfft = losvd_rfft(vsysts, vs, σs, nl, σ_diff=0., factor=1.)

    # Calculate the inverse Fourier transform of the resultant distribution
    conv_temp = irfft(ssp_rfft .* lvd_rfft, npad, 1)

    # Take only enough pixels to match the length of the input spectrum
    conv_temp[1:npix, :]

end


"""
    losvd_rfft(vsyst, v, σ, nl, σ_diff=0., factor=1.)

Analytic Fourier Transform (of real input) of the Gauss-Hermite LOSVD.
Equation (38) of Cappellari M., 2017, MNRAS, 466, 798
http://adsabs.harvard.edu/abs/2017MNRAS.466..798C

Adapted and simplified from the pPXF and BADASS python codes.
Only supports 2 moments (v and σ) and 1-sided fitting.
"""
function losvd_rfft(vsyst::Real, v::Real, σ::Real, nl::Integer; σ_diff::Real=0., factor::Real=1.)

    # Prepare quantities
    a = (vsyst + v)/σ
    b = σ_diff/σ
    ω = isfinite(σ) ? range(0., π*σ*factor, nl) : ones(nl) .* NaN  # handle cases where the spaxel isn't fit 
    # Calculate Fourier transform
    lvd_rfft = @. exp(1im*a*ω - 0.5*(1-b^2)*ω^2)
    # Take complex conjugate
    conj(lvd_rfft)

end


"""
    attenuation(λ_ang, Av, δ=nothing, f_nodust=nothing, uv_bump=nothing)

Adapted from pPXF (Cappellari 2017), original docstring below:

Combines the attenuation curves from    
`Kriek & Conroy (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...775L..16K>`_
hereafter KC13, 
`Calzetti et al. (2000) <http://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C>`_
hereafter C+00,
`Noll et al. (2009) <https://ui.adsabs.harvard.edu/abs/2009A%26A...499...69N>`_,
and `Lower et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022ApJ...931...14L>`_.

When ``delta = uv_bump = f_nodust = None`` this function returns the C+00 
reddening curve. When ``uv_bump = f_nodust = None`` this function uses the 
``delta - uv_bump`` relation by KC13. The parametrization of the UV bump 
comes from Noll+09. The modelling of the attenuated fraction follows Lower+22.

"""
function attenuation(λ::Vector{<:Real}, E_BV::Real; δ::Union{Real,Nothing}=nothing,
    f_nodust::Union{Real,Nothing}=nothing, uv_bump::Union{Real,Nothing}=nothing)

    Rv = 4.05         # C+00 eqn (5)

    # C+00 equations (3)-(4) but extrapolate for lam < 0.12 or lam > 2.2
    k₁ = @. Rv + ifelse(λ > 0.63, 2.76536/λ - 4.93776, ((0.029249/λ - 0.526482)/λ + 4.01243)/λ - 5.7328)
    
    if isnothing(δ) && isnothing(uv_bump)
        aλ = E_BV .* k₁
    else
        if isnothing(uv_bump)
            uv_bump = 0.85 - 1.9*δ  # eq.(3) KC13
        end
        λ0 = 0.2175                  # Peak wavelength of UV bump in micron
        δλ = 0.035                   # Width of UV bump in micron
        dλ = @. uv_bump*(λ*δλ)^2 / ((λ^2 - λ0^2)^2 + (λ*δλ)^2)    # eq.(2) KC13
        λv = 0.55                    # Effective V-band wavelength in micron
        aλ = @. E_BV*(k₁ + dλ)*(λ/λv)^δ      
    end
    
    frac = @. 10^(-0.4 * clamp(aλ, 0., Inf))
    if !isnothing(f_nodust)
        frac = @. f_nodust + (1 - f_nodust)*frac
    end

    frac
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

    # Normalize the function so that the integral is given by this
    I = ∫Voigt(A, FWHM, η)

    # Mix the two distributions with the mixing parameter η
    I * (η * G + (1 - η) * L)
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
        ext = KVT_interp(λ)
    elseif λ_mx < λ < λ_mx + 2
        ext = KVT_interp_end(λ)
    else
        ext = 0.
    end
    ext = ext < 0 ? 0. : ext

    # Add a drude profile around 18 microns
    ext += Drude(λ, 0.4, 18., 4.446)

    (1 - β) * ext + β * (9.7/λ)^1.7
end


function τ_ct(λ::Vector{<:Real})

    mx = argmax(CT_prof[1])
    λ_mx = CT_prof[1][mx]

    ext = CT_interp(λ)
    w_mx = findall(λ .> λ_mx)
    ext[w_mx] .= CT_prof[2][mx] .* (λ_mx./λ[w_mx]).^1.7

    _, wh = findmin(x -> abs(x - 9.7), CT_prof[1])
    ext ./= CT_prof[2][wh]

    ext
end


function τ_ohm(λ::Vector{<:Real})

    ext = OHM_interp(λ)
    _, wh = findmin(x -> abs(x - 9.7), OHM_prof[1])
    ext ./= OHM_prof[2][wh]

    ext
end


"""
    τ_dp(λ, β)

Calculate the mixed silicate extinction profile based on Donnan et al. (2022)
"""
function τ_dp(λ::Vector{<:Real}, β::Real)

    # Simple cubic spline interpolation
    ext = DP_interp(λ)

    # Add 1.7 power law, as in PAHFIT
    @. (1 - β) * ext + β * (9.8/λ)^1.7
end


"""
    τ_ice(λ)

Calculate the ice extinction profiles
"""
function τ_ice(λ::Vector{<:Real})

    # Simple cubic spline interpolation
    ext = Ice_interp(λ)
    ext ./= maximum(IceCHTemp[2])

    ext
end


"""
    τ_ch(λ)

Calculate the CH extinction profiles
"""
function τ_ch(λ::Vector{<:Real})

    # Simple cubic spline interpolation
    ext = CH_interp(λ)
    ext ./= maximum(IceCHTemp[4])

    ext
end


"""
    Extinction(ext, τ_97; screen)

Calculate the overall extinction factor
"""
function extinction(ext::Real, τ_97::Real; screen::Bool=false)
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
    get_logarithmic_λ(λlim, n, logscale=nothing, oversample=1)

Calculate a logarithmically spaced wavelength grid with limits of (λlim[1], λlim[2]) with a number of samples
either given by a scaling factor `logscale` between points, or an `oversample` factor based on the input number of points `n`.
"""
function get_logarithmic_λ(λlim::Vector{<:Real}, n::Integer; logscale::Union{Real,Nothing}=nothing, oversample::Integer=1)
    @assert length(λlim) == 2 "λlim must have exactly 2 elements"
    @assert λlim[2] > λlim[1] "λlim must satisfty λlim[2] > λlim[1]"
    m = n * oversample

    dλ = diff(λlim)[1] / (n - 1)              # assume constant dlam
    lnlim = log.(λlim ./ dλ .+ [-0.5, 0.5])   # all in units of dlam

    if isnothing(logscale)
        logscale = diff(lnlim)/m
    else
        m = floor(Int, diff(lnlim)[1]/logscale)
        lnlim[2] = lnlim[1] + m*logscale
    end
    new_borders = exp.(range(lnlim..., length=m+1))
    lnλ = .√(new_borders[2:end] .* new_borders[1:end-1]) .* dλ  # geometric mean

    # return logarithmically spaced wavelength vector according to either logscale or oversample
    lnλ
end


# function log_rebin(λ::Vector{<:Real}, flux::Vector{<:Real}; logscale::Union{Real,Nothing}=nothing, oversample::Integer=1)
#     @assert length(λ) == length(flux) "Wavelength and flux must be the same length!"
#     λlim = [minimum(λ), maximum(λ)]
#     n = length(flux)
#     m = n * oversample

#     dλ = diff(λlim)[1] / (n - 1)              # assume constant dlam
#     lim = λlim ./ dλ .+ [-0.5, 0.5]
#     borders = range(lim..., n+1)
#     lnlim = log.(lim)                         # all in units of dlam

#     if isnothing(logscale)
#         logscale = diff(lnlim)[1]/m
#     else
#         m = floor(Int, diff(lnlim)[1]/logscale)
#         lnlim[2] = lnlim[1] + m*logscale
#     end
#     new_borders = exp.(range(lnlim..., length=m+1))
#     k = floor.(Int, clamp.(new_borders .- lim[1], 0., n-1)) .+ 1
#     # integrate the flux over the windows given by k
#     newflux = [k[i] < k[i+1] ? reduce(+, flux[k[i]:k[i+1]-1]) : 0. for i in eachindex(k[1:end-1])]
#     newflux .+= diff((new_borders .- borders[k]) .* flux[k])
#     newflux ./= diff(new_borders)
    
#     newλ = .√(new_borders[2:end] .* new_borders[1:end-1]) .* dλ  # geometric mean

#     newλ, newflux
# end


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
                @debug "\nSpectres: new_wave contains values outside the range " *
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
                # e_widths = old_widths[start:stop] .* permutedims(err[.., start:stop], (ndims(err), range(1, ndims(err)-1)...))
                # new_errs[.., j] .= .√(sumdim(e_widths.^2, 1, nan=false))   # -> preserve NaNs
                # new_errs[.., j] ./= sum(old_widths[start:stop])
                e_widths = permutedims(err[.., start:stop], (ndims(err), range(1, ndims(err)-1)...))
                new_errs[.., j] .= dropdims(median(e_widths, dims=1), dims=1)
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
    model_continuum(λ, params, n_dust_cont, n_power_law, extinction_curve, extinction_screen, fit_sil_emission;
        return_components=return_components)

Create a model of the continuum (including stellar+dust continuum, PAH features, and extinction, excluding emission lines)
at the given wavelengths `λ`, given the parameter vector `params`.

Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)

# Arguments
- `λ::Vector{<:AbstractFloat}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:AbstractFloat}`: Parameter vector. 
- `n_dust_cont::Integer`: Number of dust continuum profiles to be fit
- `n_power_law::Integer`: Number of power laws to be fit
- `n_dust_feat::Integer`: Number of PAH dust features to be fit
- `extinction_curve::String`: The type of extinction curve to use, "kvt" or "d+"
- `extinction_screen::Bool`: Whether or not to use a screen model for the extinction curve
- `fit_sil_emission::Bool`: Whether or not to fit silicate emission with a hot dust continuum component
- `return_components::Bool=false`: Whether or not to return the individual components of the fit as a dictionary, in 
    addition to the overall fit
"""
function model_continuum(λ::Vector{T}, params::Vector{T}, N::Real, n_dust_cont::Integer, n_power_law::Integer, dust_prof::Vector{Symbol},
    n_abs_feat::Integer, extinction_curve::String, extinction_screen::Bool, fit_sil_emission::Bool, use_pah_templates::Bool, 
    return_components::Bool) where {T<:Real}

    # Prepare outputs
    comps = Dict{String, Vector{Float64}}()
    contin = zeros(Float64, length(λ))

    # Stellar blackbody continuum (usually at 5000 K)
    comps["stellar"] = params[1] .* Blackbody_ν.(λ, params[2]) ./ N
    contin .+= comps["stellar"]
    pᵢ = 3

    # Add dust continua at various temperatures
    for i ∈ 1:n_dust_cont
        comps["dust_cont_$i"] = params[pᵢ] .* (9.7 ./ λ).^2 .* Blackbody_ν.(λ, params[pᵢ+1]) ./ N
        contin .+= comps["dust_cont_$i"] 
        pᵢ += 2
    end

    # Add power laws with various indices
    for j ∈ 1:n_power_law
        comps["power_law_$j"] = params[pᵢ] .* power_law.(λ, params[pᵢ+1])
        contin .+= comps["power_law_$j"]
        pᵢ += 2
    end

    # Extinction 
    if extinction_curve == "d+"
        ext_curve = τ_dp(λ, params[pᵢ+3])
    elseif extinction_curve == "kvt"
        ext_curve = τ_kvt.(λ, params[pᵢ+3])
    elseif extinction_curve == "ct"
        ext_curve = τ_ct(λ)
    elseif extinction_curve == "ohm"
        ext_curve = τ_ohm(λ)
    else
        error("Unrecognized extinction curve: $extinction_curve")
    end
    comps["extinction"] = extinction.(ext_curve, params[pᵢ], screen=extinction_screen)

    # Ice+CH Absorption
    ext_ice = τ_ice(λ)
    comps["abs_ice"] = extinction.(ext_ice, params[pᵢ+1] * params[pᵢ+2], screen=true)
    ext_ch = τ_ch(λ)
    comps["abs_ch"] = extinction.(ext_ch, params[pᵢ+2], screen=true)
    pᵢ += 4

    # Other absorption features
    abs_tot = ones(Float64, length(λ))
    for k ∈ 1:n_abs_feat
        prof = Drude.(λ, 1.0, params[pᵢ+1:pᵢ+2]...)
        comps["abs_feat_$k"] = extinction.(prof, params[pᵢ], screen=true)
        abs_tot .*= comps["abs_feat_$k"]
        pᵢ += 3
    end

    contin .*= comps["extinction"] .* comps["abs_ice"] .* comps["abs_ch"] .* abs_tot

    if fit_sil_emission
        # Add Silicate emission from hot dust (amplitude, temperature, covering fraction, warm tau, cold tau)
        # Ref: Gallimore et al. 2010
        comps["hot_dust"] = silicate_emission(λ, params[pᵢ:pᵢ+5]...) ./ N
        contin .+= comps["hot_dust"] .* comps["abs_ice"] .* comps["abs_ch"] .* abs_tot
        pᵢ += 6
    end

    if use_pah_templates
        pah3 = Smith3_interp.(λ)
        contin .+= params[pᵢ] .* pah3 ./ maximum(pah3) .* comps["extinction"]
        pah4 = Smith4_interp.(λ)
        contin .+= params[pᵢ+1] .* pah4 ./ maximum(pah4) .* comps["extinction"]
    else
        for (j, prof) ∈ enumerate(dust_prof)
            if prof == :Drude
                comps["dust_feat_$j"] = Drude.(λ, params[pᵢ:pᵢ+2]...)
                pᵢ += 3
            elseif prof == :PearsonIV
                comps["dust_feat_$j"] = PearsonIV.(λ, params[pᵢ:pᵢ+4]...)
                pᵢ += 5
            end
            contin .+= comps["dust_feat_$j"] .* comps["extinction"] 
        end
    end

    # Return components if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency --> not allocating the dictionary improves performance DRAMATICALLY
function model_continuum(λ::Vector{T}, params::Vector{T}, N::Real, n_dust_cont::Integer, n_power_law::Integer, dust_prof::Vector{Symbol},
    n_abs_feat::Integer, extinction_curve::String, extinction_screen::Bool, fit_sil_emission::Bool, use_pah_templates::Bool) where {T<:Real}

    # Prepare outputs
    contin = zeros(Float64, length(λ))

    # Stellar blackbody continuum (usually at 5000 K)
    contin .+= params[1] .* Blackbody_ν.(λ, params[2]) ./ N
    pᵢ = 3

    # Add dust continua at various temperatures
    for i ∈ 1:n_dust_cont
        contin .+= params[pᵢ] .* (9.7 ./ λ).^2 .* Blackbody_ν.(λ, params[pᵢ+1]) ./ N
        pᵢ += 2
    end

    # Add power laws with various indices
    for j ∈ 1:n_power_law
        contin .+= params[pᵢ] .* power_law.(λ, params[pᵢ+1])
        pᵢ += 2
    end

    # Extinction 
    if extinction_curve == "d+"
        ext_curve = τ_dp(λ, params[pᵢ+3])
    elseif extinction_curve == "kvt"
        ext_curve = τ_kvt.(λ, params[pᵢ+3])
    elseif extinction_curve == "ct"
        ext_curve = τ_ct(λ)
    elseif extinction_curve == "ohm"
        ext_curve = τ_ohm(λ)
    else
        error("Unrecognized extinction curve: $extinction_curve")
    end
    ext = extinction.(ext_curve, params[pᵢ], screen=extinction_screen)

    # Ice+CH absorption
    ext_ice = τ_ice(λ)
    abs_ice = extinction.(ext_ice, params[pᵢ+1] * params[pᵢ+2], screen=true)
    ext_ch = τ_ch(λ)
    abs_ch = extinction.(ext_ch, params[pᵢ+2], screen=true)
    pᵢ += 4

    # Other absorption features
    abs_tot = ones(Float64, length(λ))
    for k ∈ 1:n_abs_feat
        prof = Drude.(λ, 1.0, params[pᵢ+1:pᵢ+2]...)
        abs_tot .*= extinction.(prof, params[pᵢ], screen=true)
        pᵢ += 3
    end
    
    contin .*= ext .* abs_ice .* abs_ch .* abs_tot

    if fit_sil_emission
        # Add Silicate emission from hot dust (amplitude, temperature, covering fraction, warm tau, cold tau)
        # Ref: Gallimore et al. 2010
        contin .+= silicate_emission(λ, params[pᵢ:pᵢ+5]...) ./ N .* abs_ice .* abs_ch .* abs_tot
        pᵢ += 6
    end

    if use_pah_templates
        pah3 = Smith3_interp.(λ)
        contin .+= params[pᵢ] .* pah3 ./ maximum(pah3) .* ext
        pah4 = Smith4_interp.(λ)
        contin .+= params[pᵢ+1] .* pah4 ./ maximum(pah4) .* ext
    else
        if all(dust_prof .== :Drude)
            for j ∈ 1:length(dust_prof) 
                contin .+= Drude.(λ, params[pᵢ:pᵢ+2]...) .* ext
                pᵢ += 3
            end
        else
            for (j, prof) ∈ enumerate(dust_prof)
                if prof == :Drude
                    df = Drude.(λ, params[pᵢ:pᵢ+2]...)
                    pᵢ += 3
                elseif prof == :PearsonIV
                    df = PearsonIV.(λ, params[pᵢ:pᵢ+4]...)
                    pᵢ += 5
                end
                contin .+= df .* ext
            end
        end
    end

    contin

end

# Optical fitting version of the function
function model_continuum(λ::Vector{T}, params::Vector{T}, N::Real, velscale::Real, vsyst::Real, n_ssps::Integer, 
    ssp_λ::Vector{T}, ssp_templates::Vector{Spline2D}, fit_uv_bump::Bool, fit_covering_frac::Bool, Ω::Vector{<:Real},
    return_components::Bool) where {T<:Real}    

    # Prepare outputs
    comps = Dict{String, Vector{Float64}}()
    contin = zeros(Float64, length(λ))
    pᵢ = 1

    ssps = zeros(Float64, length(ssp_λ), n_ssps)
    # Interpolate the SSPs to the right ages/metallicities (this is slow)
    for i in 1:n_ssps
        # normalize the templates by their median so that the amplitude is properly separated from the age and metallicity during fitting
        temp = [ssp_templates[j](params[pᵢ+1], params[pᵢ+2]) for j in eachindex(ssp_λ)]
        ssps[:, i] = params[pᵢ] .* temp ./ median(temp)
        pᵢ += 3
    end

    # Convolve with a line-of-sight velocity distribution (LOSVD) according to the stellar velocity and dispersion
    conv_ssps = convolve_losvd(ssps, vsyst, params[pᵢ], params[pᵢ+1], velscale, length(λ))
    pᵢ += 2

    # Combine the convolved stellar templates together with the weights
    for i in 1:n_ssps
        comps["SSP_$i"] = conv_ssps[:, i] ./ Ω
        contin .+= comps["SSP_$i"]
    end

    # Apply attenuation law
    E_BV = params[pᵢ]
    δ = f_nodust = nothing
    if fit_uv_bump && fit_covering_frac
        δ = params[pᵢ+1]
        f_nodust = params[pᵢ+2]
        pᵢ += 2
    elseif fit_uv_bump
        δ = params[pᵢ+1]
        pᵢ += 1
    elseif fit_covering_frac
        f_nodust = params[pᵢ+1]
        pᵢ += 1
    end
    pᵢ += 1
    comps["extinction"] = attenuation(λ, E_BV, δ=δ, f_nodust=f_nodust)
    contin .*= comps["extinction"]

    if return_components
        return contin, comps
    end
    contin

end


function model_continuum(λ::Vector{T}, params::Vector{T}, N::Real, velscale::Real, vsyst::Real, n_ssps::Integer, 
    ssp_λ::Vector{T}, ssp_templates::Vector{Spline2D}, fit_uv_bump::Bool, fit_covering_frac::Bool, Ω::Vector{<:Real}) where {T<:Real}    

    # Prepare outputs
    contin = zeros(Float64, length(λ))
    pᵢ = 1

    ssps = zeros(Float64, length(ssp_λ), n_ssps)
    # Interpolate the SSPs to the right ages/metallicities (this is slow)
    for i in 1:n_ssps
        # normalize the templates by their median so that the amplitude is properly separated from the age and metallicity during fitting
        temp = [ssp_templates[j](params[pᵢ+1], params[pᵢ+2]) for j in eachindex(ssp_λ)]
        ssps[:, i] = params[pᵢ] .* temp ./ median(temp)
        pᵢ += 3
    end

    # Convolve with a line-of-sight velocity distribution (LOSVD) according to the stellar velocity and dispersion
    conv_ssps = convolve_losvd(ssps, vsyst, params[pᵢ], params[pᵢ+1], velscale, length(λ))
    pᵢ += 2

    # Combine the convolved stellar templates together with the weights
    contin .+= sumdim(conv_ssps, 2) ./ Ω

    # Apply attenuation law
    E_BV = params[pᵢ]
    δ = f_nodust = nothing
    if fit_uv_bump && fit_covering_frac
        δ = params[pᵢ+1]
        f_nodust = params[pᵢ+2]
        pᵢ += 2
    elseif fit_uv_bump
        δ = params[pᵢ+1]
        pᵢ += 1
    elseif fit_covering_frac
        f_nodust = params[pᵢ+1]
        pᵢ += 1
    end
    pᵢ += 1
    att = attenuation(λ, E_BV, δ=δ, f_nodust=f_nodust)
    contin .*= att

    contin

end


"""
    model_pah_residuals(λ, params, n_dust_feat, return_components)
Create a model of the PAH features at the given wavelengths `λ`, given the parameter vector `params`.
Adapted from PAHFIT, Smith, Draine, et al. (2007); http://tir.astro.utoledo.edu/jdsmith/research/pahfit.php
(with modifications)
# Arguments
- `λ::Vector{<:AbstractFloat}`: Wavelength vector of the spectrum to be fit
- `params::Vector{<:AbstractFloat}`: Parameter vector. Parameters should be ordered as: `(amp, center, fwhm) for each PAH profile`
- `n_dust_feat::Integer`: The number of PAH features that are being fit
- `ext_curve::Vector{<:AbstractFloat}`: The extinction curve that was fit using model_continuum
- `return_components::Bool`: Whether or not to return the individual components of the fit as a dictionary, in
    addition to the overall fit
"""
function model_pah_residuals(λ::Vector{T}, params::Vector{T}, dust_prof::Vector{Symbol}, ext_curve::Vector{T}, 
    return_components::Bool) where {T<:Real}

    # Prepare outputs
    comps = Dict{String, Vector{Float64}}()
    contin = zeros(Float64, length(λ))

    # Add dust features with drude profiles
    pᵢ = 1
    for (j, prof) ∈ enumerate(dust_prof)
        if prof == :Drude
            df = Drude.(λ, params[pᵢ:pᵢ+2]...)
            pᵢ += 3
        elseif prof == :PearsonIV
            df = PearsonIV.(λ, params[pᵢ:pᵢ+4]...)
            pᵢ += 5
        end
        contin .+= df
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
function model_pah_residuals(λ::Vector{T}, params::Vector{T}, dust_prof::Vector{Symbol}, ext_curve::Vector{T}) where {T<:Real}

    # Prepare outputs
    contin = zeros(Float64, length(λ))

    # Add dust features with drude profiles
    pᵢ = 1
    if all(dust_prof .== :Drude)
        for j ∈ 1:length(dust_prof) 
            contin .+= Drude.(λ, params[pᵢ:pᵢ+2]...)
            pᵢ += 3
        end
    else
        for (j, prof) ∈ enumerate(dust_prof)
            if prof == :Drude
                df = Drude.(λ, params[pᵢ:pᵢ+2]...)
                pᵢ += 3
            elseif prof == :PearsonIV
                df = PearsonIV.(λ, params[pᵢ:pᵢ+4]...)
                pᵢ += 5
            end
            contin .+= df
        end
    end

    # Apply extinction
    contin .*= ext_curve

    contin

end


"""
    model_line_residuals(λ, params, n_lines, n_kin_tied, kin_tied_key, line_tied, line_profiles,
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
- `flexible_wavesol::Bool`: Whether or not to allow small variations in tied velocity offsets, to account for a poor
    wavelength solution in the data
- `ext_curve::Vector{<:AbstractFloat}`: The extinction curve fit with model_continuum
- `lsf::Function`: A function giving the FWHM of the line-spread function in km/s as a function of rest-frame wavelength in microns.
- `return_components::Bool=false`: Whether or not to return the individual components of the fit as a dictionary, in 
    addition to the overall fit
"""
function model_line_residuals(λ::Vector{T}, params::Vector{T}, n_lines::S, n_comps::S, lines::TransitionLines, 
    flexible_wavesol::Bool, ext_curve::Vector{T}, lsf::Function, return_components::Bool) where {T<:Real,S<:Integer}

    # Prepare outputs
    comps = Dict{String, Vector{Float64}}()
    contin = zeros(Float64, length(λ))

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
                # For the additional components, we parametrize them this way to essentially give them soft constraints
                # relative to the primary component
                else
                    amp *= amp_1
                    voff += voff_1
                    fwhm *= fwhm_1
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

    # Return components if necessary
    if return_components
        return contin, comps
    end
    contin

end


# Multiple dispatch for more efficiency --> not allocating the dictionary improves performance DRAMATICALLY
function model_line_residuals(λ::Vector{T}, params::Vector{T}, n_lines::S, n_comps::S, lines::TransitionLines, 
    flexible_wavesol::Bool, ext_curve::Vector{T}, lsf::Function) where {T<:Real,S<:Integer}

    # Prepare outputs
    contin = zeros(Float64, length(λ))

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
                # For the additional components, we parametrize them this way to essentially give them soft constraints
                # relative to the primary component
                else
                    amp *= amp_1
                    voff += voff_1
                    fwhm *= fwhm_1
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

    contin

end


"""
    calculate_extra_parameters(λ, I, σ, n_dust_cont, n_dust_feat, extinction_curve, extinction_screen,
        fit_sil_emission, n_lines, n_acomps, n_cops, lines, flexible_wavesol, lsf, popt_c, popt_l, 
        perr_c, perr_l, extinction, mask_lines, continuum, Ω)

Calculate extra parameters that are not fit, but are nevertheless important to know, for a given spaxel.
Currently this includes the integrated intensity and signal to noise ratios of dust features and emission lines.
"""
function calculate_extra_parameters(λ::Vector{<:Real}, I::Vector{<:Real}, N::Real, comps::Dict, n_dust_cont::Integer,
    n_power_law::Integer, n_dust_feat::Integer, dust_profiles::Vector{Symbol}, n_abs_feat::Integer, fit_sil_emission::Bool, 
    n_lines::Integer, n_acomps::Integer, n_comps::Integer, lines::TransitionLines, flexible_wavesol::Bool, 
    lsf::Function, popt_c::Vector{T}, popt_l::Vector{T}, perr_c::Vector{T}, perr_l::Vector{T}, 
    extinction::Vector{T}, mask_lines::BitVector, continuum::Vector{T}, 
    area_sr::Vector{T}, propagate_err::Bool=true) where {T<:Real}

    @debug "Calculating extra parameters"

    # Normalization
    @debug "Normalization: $N"

    # Loop through dust features
    p_dust = zeros(3n_dust_feat)
    p_dust_err = zeros(3n_dust_feat)
    pₒ = 1
    # Initial parameter vector index where dust profiles start
    pᵢ = 3 + 2n_dust_cont + 2n_power_law + 4 + 3n_abs_feat + (fit_sil_emission ? 6 : 0)

    for ii ∈ 1:n_dust_feat

        # unpack the parameters
        A, μ, fwhm = popt_c[pᵢ:pᵢ+2]
        A_err, μ_err, fwhm_err = perr_c[pᵢ:pᵢ+2]
        # Convert peak intensity to CGS units (erg s^-1 cm^-2 μm^-1 sr^-1)
        A_cgs = MJysr_to_cgs(A*N, μ)
        # Convert the error in the intensity to CGS units
        A_cgs_err = propagate_err ? MJysr_to_cgs_err(A*N, A_err*N, μ, μ_err) : 0.

        # Get the index of the central wavelength
        cent_ind = argmin(abs.(λ .- μ))
        
        # Integrate over the solid angle
        A_cgs *= area_sr[cent_ind]
        if propagate_err
            A_cgs_err *= area_sr[cent_ind]
        end

        # Get the extinction profile at the center
        ext = extinction[cent_ind] 

        prof = dust_profiles[ii]
        if prof == :PearsonIV
            m, m_err = popt_c[pᵢ+3], perr_c[pᵢ+3]
            ν, ν_err = popt_c[pᵢ+4], perr_c[pᵢ+4]
        else
            m, m_err = 0., 0.
            ν, ν_err = 0., 0.
        end

        # Calculate the flux using the utility function
        flux, f_err = calculate_flux(prof, A_cgs, A_cgs_err, μ, μ_err, fwhm, fwhm_err, 
            m=m, m_err=m_err, ν=ν, ν_err=ν_err, propagate_err=propagate_err)
        
        # Calculate the equivalent width using the utility function
        eqw, e_err = calculate_eqw(λ, prof, comps, false, n_dust_cont, n_power_law, n_abs_feat, n_dust_feat, fit_sil_emission,
            A*N, A_err*N, μ, μ_err, fwhm, fwhm_err, m=m, m_err=m_err, ν=ν, ν_err=ν_err, propagate_err=propagate_err)
        
        snr = A*N*ext / std(I[.!mask_lines .& (abs.(λ .- μ) .< 2fwhm)] .- continuum[.!mask_lines .& (abs.(λ .- μ) .< 2fwhm)])

        @debug "PAH feature ($prof) with ($A_cgs, $μ, $fwhm, $m, $ν) and errors ($A_cgs_err, $μ_err, $fwhm_err, $m_err, $ν_err)"
        @debug "Flux=$flux +/- $f_err, EQW=$eqw +/- $e_err, SNR=$snr"

        # increment the parameter index
        pᵢ += 3
        if prof == :PearsonIV
            pᵢ += 2
        end

        # flux units: erg s^-1 cm^-2 sr^-1 (integrated over μm)
        p_dust[pₒ] = flux
        p_dust_err[pₒ] = f_err

        # eqw units: μm
        p_dust[pₒ+1] = eqw
        p_dust_err[pₒ+1] = e_err

        # SNR, calculated as (peak amplitude) / (RMS intensity of the surrounding spectrum)
        # include the extinction factor when calculating the SNR
        p_dust[pₒ+2] = snr

        pₒ += 3
    end

    # Loop through lines
    p_lines = zeros(3n_lines+3n_acomps)
    p_lines_err = zeros(3n_lines+3n_acomps)
    pₒ = pᵢ = 1
    for (k, λ0) ∈ enumerate(lines.λ₀)
        amp_1 = amp_1_err = voff_1 = voff_1_err = fwhm_1 = fwhm_1_err = nothing
        for j ∈ 1:n_comps
            if !isnothing(lines.profiles[k, j])

                # (\/ pretty much the same as the model_line_residuals function, but calculating the integrated intensities)
                amp = popt_l[pᵢ]
                amp_err = propagate_err ? perr_l[pᵢ] : 0.
                voff = popt_l[pᵢ+1]
                voff_err = propagate_err ? perr_l[pᵢ+1] : 0.
                # fill values with nothings for profiles that may / may not have them
                h3 = h3_err = h4 = h4_err = η = η_err = nothing

                if !isnothing(lines.tied_voff[k, j]) && flexible_wavesol && isone(j)
                    voff += popt_l[pᵢ+2]
                    voff_err = propagate_err ? hypot(voff_err, perr_l[pᵢ+2]) : 0.
                    fwhm = popt_l[pᵢ+3]
                    fwhm_err = propagate_err ? perr_l[pᵢ+3] : 0.
                    pᵢ += 4
                else
                    fwhm = popt_l[pᵢ+2]
                    fwhm_err = propagate_err ? perr_l[pᵢ+2] : 0.
                    pᵢ += 3
                end

                if lines.profiles[k, j] == :GaussHermite
                    # Get additional h3, h4 components
                    h3 = popt_l[pᵢ]
                    h3_err = propagate_err ? perr_l[pᵢ] : 0.
                    h4 = popt_l[pᵢ+1]
                    h4_err = propagate_err ? perr_l[pᵢ+1] : 0.
                    pᵢ += 2
                elseif lines.profiles[k, j] == :Voigt
                    # Get additional mixing component, either from the tied position or the 
                    # individual position
                    η = popt_l[pᵢ]
                    η_err = propagate_err ? perr_l[pᵢ] : 0.
                    pᵢ += 1
                end

                # Save the j = 1 parameters for reference 
                if isone(j)
                    amp_1 = amp
                    amp_1_err = amp_err
                    voff_1 = voff
                    voff_1_err = voff_err
                    fwhm_1 = fwhm
                    fwhm_1_err = fwhm_err
                # For the additional components, we parametrize them this way to essentially give them soft constraints
                # relative to the primary component
                else
                    amp_err = propagate_err ? hypot(amp_1_err*amp, amp_err*amp_1) : 0.
                    amp *= amp_1
                    
                    voff_err = propagate_err ? hypot(voff_err, voff_1_err) : 0.
                    voff += voff_1

                    fwhm_err = propagate_err ? hypot(fwhm_1_err*fwhm, fwhm_err*fwhm_1) : 0.
                    fwhm *= fwhm_1
                end

                # Broaden the FWHM by the instrumental FWHM at the location of the line
                fwhm_inst = lsf(λ0)
                fwhm_err = propagate_err ? fwhm / hypot(fwhm, fwhm_inst) * fwhm_err : 0.
                fwhm = hypot(fwhm, fwhm_inst)

                # Convert voff in km/s to mean wavelength in μm
                mean_μm = Doppler_shift_λ(λ0, voff)
                mean_μm_err = propagate_err ? λ0 / C_KMS * voff_err : 0.
                # WARNING:
                # Probably should set to 0 if using flexible tied voffs since they are highly degenerate and result in massive errors
                # if !isnothing(cube_fitter.line_tied[k]) && cube_fitter.flexible_wavesol
                #     mean_μm_err = 0.
                # end

                # Convert FWHM from km/s to μm
                fwhm_μm = Doppler_shift_λ(λ0, fwhm/2) - Doppler_shift_λ(λ0, -fwhm/2)
                fwhm_μm_err = propagate_err ? λ0 / C_KMS * fwhm_err : 0.

                # Convert amplitude to erg s^-1 cm^-2 μm^-1 sr^-1, put back in the normalization
                amp_cgs = MJysr_to_cgs(amp*N, mean_μm)
                amp_cgs_err = propagate_err ? MJysr_to_cgs_err(amp*N, amp_err*N, mean_μm, mean_μm_err) : 0.

                # Get the index of the central wavelength
                cent_ind = argmin(abs.(λ .- mean_μm))

                # Integrate over the solid angle
                amp_cgs *= area_sr[cent_ind]
                if propagate_err
                    amp_cgs_err *= area_sr[cent_ind]
                end

                # Get the extinction factor at the line center
                ext = extinction[cent_ind]

                # Calculate line flux using the helper function
                p_lines[pₒ], p_lines_err[pₒ] = calculate_flux(lines.profiles[k, j], amp_cgs, amp_cgs_err, mean_μm, mean_μm_err,
                    fwhm_μm, fwhm_μm_err, h3=h3, h3_err=h3_err, h4=h4, h4_err=h4_err, η=η, η_err=η_err, propagate_err=propagate_err)
                
                # Calculate equivalent width using the helper function
                p_lines[pₒ+1], p_lines_err[pₒ+1] = calculate_eqw(λ, lines.profiles[k, j], comps, true, n_dust_cont, n_power_law, n_abs_feat,
                    n_dust_feat, fit_sil_emission, amp*N, amp_err*N, mean_μm, mean_μm_err, fwhm_μm, fwhm_μm_err, h3=h3, h3_err=h3_err, 
                    h4=h4, h4_err=h4_err, η=η, η_err=η_err, propagate_err=propagate_err)

                # SNR
                p_lines[pₒ+2] = amp*N*ext / std(I[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)] .- continuum[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)])
                
                @debug "Line with ($amp_cgs, $mean_μm, $fwhm_μm) and errors ($amp_cgs_err, $mean_μm_err, $fwhm_μm_err)"
                @debug "Flux=$(p_lines[pₒ]) +/- $(p_lines_err[pₒ]), EQW=$(p_lines[pₒ+1]) +/- $(p_lines_err[pₒ+1]), SNR=$(p_lines[pₒ+2])"

                # Advance the output vector index by 3
                pₒ += 3
            end
        end
    end

    p_dust, p_lines, p_dust_err, p_lines_err
end


function calculate_extra_parameters(λ::Vector{<:Real}, I::Vector{<:Real}, N::Real, comps::Dict, n_ssps::Integer,
    n_lines::Integer, n_acomps::Integer, n_comps::Integer, lines::TransitionLines, flexible_wavesol::Bool, 
    lsf::Function, popt_l::Vector{T}, perr_l::Vector{T}, extinction::Vector{T}, mask_lines::BitVector, 
    continuum::Vector{T}, area_sr::Vector{T}, propagate_err::Bool=true) where {T<:Real}

    @debug "Calculating extra parameters"

    # Normalization
    @debug "Normalization: $N"

    # Loop through lines
    p_lines = zeros(3n_lines+3n_acomps)
    p_lines_err = zeros(3n_lines+3n_acomps)
    pₒ = pᵢ = 1
    for (k, λ0) ∈ enumerate(lines.λ₀)
        amp_1 = amp_1_err = voff_1 = voff_1_err = fwhm_1 = fwhm_1_err = nothing
        for j ∈ 1:n_comps
            if !isnothing(lines.profiles[k, j])

                # (\/ pretty much the same as the model_line_residuals function, but calculating the integrated intensities)
                amp = popt_l[pᵢ]
                amp_err = propagate_err ? perr_l[pᵢ] : 0.
                voff = popt_l[pᵢ+1]
                voff_err = propagate_err ? perr_l[pᵢ+1] : 0.
                # fill values with nothings for profiles that may / may not have them
                h3 = h3_err = h4 = h4_err = η = η_err = nothing

                if !isnothing(lines.tied_voff[k, j]) && flexible_wavesol && isone(j)
                    voff += popt_l[pᵢ+2]
                    voff_err = propagate_err ? hypot(voff_err, perr_l[pᵢ+2]) : 0.
                    fwhm = popt_l[pᵢ+3]
                    fwhm_err = propagate_err ? perr_l[pᵢ+3] : 0.
                    pᵢ += 4
                else
                    fwhm = popt_l[pᵢ+2]
                    fwhm_err = propagate_err ? perr_l[pᵢ+2] : 0.
                    pᵢ += 3
                end

                if lines.profiles[k, j] == :GaussHermite
                    # Get additional h3, h4 components
                    h3 = popt_l[pᵢ]
                    h3_err = propagate_err ? perr_l[pᵢ] : 0.
                    h4 = popt_l[pᵢ+1]
                    h4_err = propagate_err ? perr_l[pᵢ+1] : 0.
                    pᵢ += 2
                elseif lines.profiles[k, j] == :Voigt
                    # Get additional mixing component, either from the tied position or the 
                    # individual position
                    η = popt_l[pᵢ]
                    η_err = propagate_err ? perr_l[pᵢ] : 0.
                    pᵢ += 1
                end

                # Save the j = 1 parameters for reference 
                if isone(j)
                    amp_1 = amp
                    amp_1_err = amp_err
                    voff_1 = voff
                    voff_1_err = voff_err
                    fwhm_1 = fwhm
                    fwhm_1_err = fwhm_err
                # For the additional components, we parametrize them this way to essentially give them soft constraints
                # relative to the primary component
                else
                    amp_err = propagate_err ? hypot(amp_1_err*amp, amp_err*amp_1) : 0.
                    amp *= amp_1
                    
                    voff_err = propagate_err ? hypot(voff_err, voff_1_err) : 0.
                    voff += voff_1

                    fwhm_err = propagate_err ? hypot(fwhm_1_err*fwhm, fwhm_err*fwhm_1) : 0.
                    fwhm *= fwhm_1
                end

                # Broaden the FWHM by the instrumental FWHM at the location of the line
                fwhm_inst = lsf(λ0)
                fwhm_err = propagate_err ? fwhm / hypot(fwhm, fwhm_inst) * fwhm_err : 0.
                fwhm = hypot(fwhm, fwhm_inst)

                # Convert voff in km/s to mean wavelength in μm
                mean_μm = Doppler_shift_λ(λ0, voff)
                mean_μm_err = propagate_err ? λ0 / C_KMS * voff_err : 0.
                # WARNING:
                # Probably should set to 0 if using flexible tied voffs since they are highly degenerate and result in massive errors
                # if !isnothing(cube_fitter.line_tied[k]) && cube_fitter.flexible_wavesol
                #     mean_μm_err = 0.
                # end

                # Convert FWHM from km/s to μm
                fwhm_μm = Doppler_shift_λ(λ0, fwhm/2) - Doppler_shift_λ(λ0, -fwhm/2)
                fwhm_μm_err = propagate_err ? λ0 / C_KMS * fwhm_err : 0.

                # Convert from erg s^-1 cm^-2 Ang^-1 sr^-1 to erg s^-1 cm^-2 μm^-1 sr^-1, putting back in the normalization
                amp_cgs = amp * N * 1e4
                amp_cgs_err = propagate_err ? amp_err * N * 1e4 : 0.

                # Get the index of the central wavelength
                cent_ind = argmin(abs.(λ .- mean_μm))

                # Integrate over the solid angle
                amp_cgs *= area_sr[cent_ind]
                if propagate_err
                    amp_cgs_err *= area_sr[cent_ind]
                end

                # Get the extinction factor at the line center
                ext = extinction[cent_ind]

                # Calculate line flux using the helper function
                p_lines[pₒ], p_lines_err[pₒ] = calculate_flux(lines.profiles[k, j], amp_cgs, amp_cgs_err, mean_μm, mean_μm_err,
                    fwhm_μm, fwhm_μm_err, h3=h3, h3_err=h3_err, h4=h4, h4_err=h4_err, η=η, η_err=η_err, propagate_err=propagate_err)
                
                # Calculate equivalent width using the helper function
                p_lines[pₒ+1], p_lines_err[pₒ+1] = calculate_eqw(λ, lines.profiles[k, j], comps, n_ssps, amp*N, amp_err*N, mean_μm, 
                    mean_μm_err, fwhm_μm, fwhm_μm_err, h3=h3, h3_err=h3_err, h4=h4, h4_err=h4_err, η=η, η_err=η_err, propagate_err=propagate_err)

                # SNR
                p_lines[pₒ+2] = amp*N*ext / std(I[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)] .- continuum[.!mask_lines .& (abs.(λ .- mean_μm) .< 0.1)])
                
                @debug "Line with ($amp_cgs, $mean_μm, $fwhm_μm) and errors ($amp_cgs_err, $mean_μm_err, $fwhm_μm_err)"
                @debug "Flux=$(p_lines[pₒ]) +/- $(p_lines_err[pₒ]), EQW=$(p_lines[pₒ+1]) +/- $(p_lines_err[pₒ+1]), SNR=$(p_lines[pₒ+2])"

                # Advance the output vector index by 3
                pₒ += 3
            end
        end
    end

    p_lines, p_lines_err
end


"""
    calculate_flux(profile, amp, amp_err, peak, peak_err, fwhm, fwhm_err; <keyword_args>)

Calculate the integrated flux of a spectral feature, i.e. a PAH or emission line. Calculates the integral
of the feature profile, using an analytic form if available, otherwise integrating numerically with QuadGK.
"""
function calculate_flux(profile::Symbol, amp::T, amp_err::T, peak::T, peak_err::T, fwhm::T, fwhm_err::T;
    m::Union{T,Nothing}=nothing, m_err::Union{T,Nothing}=nothing, ν::Union{T,Nothing}=nothing,
    ν_err::Union{T,Nothing}=nothing, h3::Union{T,Nothing}=nothing, h3_err::Union{T,Nothing}=nothing, 
    h4::Union{T,Nothing}=nothing, h4_err::Union{T,Nothing}=nothing, η::Union{T,Nothing}=nothing, 
    η_err::Union{T,Nothing}=nothing, propagate_err::Bool=true) where {T<:Real}

    # Evaluate the line profiles according to whether there is a simple analytic form
    # otherwise, integrate numerically with quadgk
    if profile == :Drude
        # (integral = π/2 * A * fwhm)
        flux, f_err = propagate_err ? ∫Drude(amp, amp_err, fwhm, fwhm_err) : (∫Drude(amp, fwhm), 0.)
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
        # shift the profile to be centered at 0 since it doesnt matter for the integral, and it makes it
        # easier for quadgk to find a solution; we also use a high order to ensure the peak is not missed if it is narrow
        flux = amp * quadgk(x -> GaussHermite(x+peak, 1, peak, fwhm, h3, h4), -Inf, Inf, order=200)[1]
        # estimate error by evaluating the integral at +/- 1 sigma
        if propagate_err
            err_l = flux - quadgk(x -> GaussHermite(x+peak, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), h3-h3_err, h4-h4_err), -Inf, Inf, order=200)[1]
            err_u = quadgk(x -> GaussHermite(x+peak, amp+amp_err, peak, fwhm+fwhm_err, h3+h3_err, h4+h4_err), -Inf, Inf, order=200)[1] - flux
            err_l = err_l ≥ 0 ? err_l : 0.
            err_u = abs(err_u)
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
    calculate_eqw(λ, profile, comps, line, n_dust_cont, n_power_law, n_abs_feat, n_dust_feat,
        fit_sil_emission)

Calculate the equivalent width (in microns) of a spectral feature, i.e. a PAH or emission line. Calculates the
integral of the ratio of the feature profile to the underlying continuum.
"""
function calculate_eqw(λ::Vector{T}, profile::Symbol, comps::Dict, line::Bool,
    n_dust_cont::Integer, n_power_law::Integer, n_abs_feat::Integer, n_dust_feat::Integer, 
    fit_sil_emission::Bool, amp::T, amp_err::T, peak::T, peak_err::T, fwhm::T, fwhm_err::T; 
    h3::Union{T,Nothing}=nothing, h3_err::Union{T,Nothing}=nothing, 
    h4::Union{T,Nothing}=nothing, h4_err::Union{T,Nothing}=nothing, 
    η::Union{T,Nothing}=nothing, η_err::Union{T,Nothing}=nothing, 
    m::Union{T,Nothing}=nothing, m_err::Union{T,Nothing}=nothing,
    ν::Union{T,Nothing}=nothing, ν_err::Union{T,Nothing}=nothing,
    propagate_err::Bool=true) where {T<:Real}

    contin = zeros(length(λ))
    contin .+= comps["stellar"]
    for i ∈ 1:n_dust_cont
        contin .+= comps["dust_cont_$i"]
    end
    for j ∈ 1:n_power_law
        contin .+= comps["power_law_$j"]
    end
    contin .*= comps["extinction"]
    if fit_sil_emission
        contin .+= comps["hot_dust"]
    end
    contin .*= comps["abs_ice"] .* comps["abs_ch"]
    for k ∈ 1:n_abs_feat
        contin .*= comps["abs_feat_$k"]
    end
    # For line EQWs, we consider PAHs as part of the "continuum"
    if line
        for l ∈ 1:n_dust_feat
            contin .+= comps["dust_feat_$l"] .* comps["extinction"]
        end
    end

    # Integrate the flux ratio to get equivalent width
    if profile == :Drude
        feature = Drude.(λ, amp, peak, fwhm)
        if propagate_err
            feature_err = hcat(Drude.(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps())),
                           Drude.(λ, amp+amp_err, peak, fwhm+fwhm_err))
        end
    elseif profile == :PearsonIV
        feature = PearsonIV.(λ, amp, peak, fwhm, m, ν)
        if propagate_err
            feature_err = hcat(PearsonIV.(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), m-m_err, ν-ν_err),
                            PearsonIV.(λ, amp+amp_err, peak, fwhm+fwhm_err, m+m_err, ν+ν_err))
        end
    elseif profile == :Gaussian
        feature = Gaussian.(λ, amp, peak, fwhm)
        if propagate_err
            feature_err = hcat(Gaussian.(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps())),
                           Gaussian.(λ, amp+amp_err, peak, fwhm+fwhm_err))
        end
    elseif profile == :Lorentzian
        feature = Lorentzian.(λ, amp, peak, fwhm)
        if propagate_err
            feature_err = hcat(Lorentzian.(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps())),
                           Lorentzian.(λ, amp+amp_err, peak, fwhm+fwhm_err))
        end
    elseif profile == :GaussHermite
        feature = GaussHermite.(λ, amp, peak, fwhm, h3, h4)
        if propagate_err
            feature_err = hcat(GaussHermite.(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), h3-h3_err, h4-h4_err),
                           GaussHermite.(λ, amp+amp_err, peak, fwhm+fwhm_err, h3+h3_err, h4+h4_err))
        end
    elseif profile == :Voigt
        feature = Voigt.(λ, amp, peak, fwhm, η)
        if propagate_err
            feature_err = hcat(Voigt.(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), max(η-η_err, 0.)),
                           Voigt.(λ, amp+amp_err, peak, fwhm+fwhm_err, min(η+η_err, 1.)))
        end
    else
        error("Unrecognized line profile $profile")
    end
    # Continuum is extincted, so make sure the feature is too
    feature .*= comps["extinction"]

    # May blow up for spaxels where the continuum is close to 0
    eqw = NumericalIntegration.integrate(λ, feature ./ contin, Trapezoidal())
    err = 0.
    if propagate_err
        feature_err[:,1] .*= comps["extinction"]
        feature_err[:,2] .*= comps["extinction"]
        err_lo = eqw - NumericalIntegration.integrate(λ, feature_err[:,1] ./ contin, Trapezoidal())
        err_up = NumericalIntegration.integrate(λ, feature_err[:,2] ./ contin, Trapezoidal()) - eqw
        err = (err_up + err_lo) / 2
    end

    eqw, err

end


function calculate_eqw(λ::Vector{T}, profile::Symbol, comps::Dict, n_ssps::Integer, 
    amp::T, amp_err::T, peak::T, peak_err::T, fwhm::T, fwhm_err::T; 
    h3::Union{T,Nothing}=nothing, h3_err::Union{T,Nothing}=nothing, 
    h4::Union{T,Nothing}=nothing, h4_err::Union{T,Nothing}=nothing, 
    η::Union{T,Nothing}=nothing, η_err::Union{T,Nothing}=nothing, 
    propagate_err::Bool=true) where {T<:Real}

    contin = zeros(length(λ))
    for i ∈ 1:n_ssps
        contin .+= comps["SSP_$i"]
    end
    contin .*= comps["extinction"]

    # Integrate the flux ratio to get equivalent width
    if profile == :Gaussian
        feature = Gaussian.(λ, amp, peak, fwhm)
        if propagate_err
            feature_err = hcat(Gaussian.(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps())),
                           Gaussian.(λ, amp+amp_err, peak, fwhm+fwhm_err))
        end
    elseif profile == :Lorentzian
        feature = Lorentzian.(λ, amp, peak, fwhm)
        if propagate_err
            feature_err = hcat(Lorentzian.(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps())),
                           Lorentzian.(λ, amp+amp_err, peak, fwhm+fwhm_err))
        end
    elseif profile == :GaussHermite
        feature = GaussHermite.(λ, amp, peak, fwhm, h3, h4)
        if propagate_err
            feature_err = hcat(GaussHermite.(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), h3-h3_err, h4-h4_err),
                           GaussHermite.(λ, amp+amp_err, peak, fwhm+fwhm_err, h3+h3_err, h4+h4_err))
        end
    elseif profile == :Voigt
        feature = Voigt.(λ, amp, peak, fwhm, η)
        if propagate_err
            feature_err = hcat(Voigt.(λ, max(amp-amp_err, 0.), peak, max(fwhm-fwhm_err, eps()), max(η-η_err, 0.)),
                           Voigt.(λ, amp+amp_err, peak, fwhm+fwhm_err, min(η+η_err, 1.)))
        end
    else
        error("Unrecognized line profile $profile")
    end
    # Continuum is extincted, so make sure the feature is too
    feature .*= comps["extinction"]

    # May blow up for spaxels where the continuum is close to 0
    eqw = NumericalIntegration.integrate(λ, feature ./ contin, Trapezoidal())
    err = 0.
    if propagate_err
        feature_err[:,1] .*= comps["extinction"]
        feature_err[:,2] .*= comps["extinction"]
        err_lo = eqw - NumericalIntegration.integrate(λ, feature_err[:,1] ./ contin, Trapezoidal())
        err_up = NumericalIntegration.integrate(λ, feature_err[:,2] ./ contin, Trapezoidal()) - eqw
        err = (err_up + err_lo) / 2
    end

    eqw, err

end