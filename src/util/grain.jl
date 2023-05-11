
# Save silicate and graphite grain efficiencies
Q_sil_interp, Q_gra_interp = read_draine_q()

# Convert grain efficiency to cross-section in cm^2
grain_σ(a, λ, type) = type == :sil ? Q_sil_interp.abs(a, λ) * π * (1e-4 * a)^2 :
    type == :gra ? Q_gra_interp.abs(a, λ) * π * (1e-4 * a)^2 : error("unrecognized grain type: $type")

# Convert MJy/sr to erg/s/cm^2/μm/sr and multiply by cross-section
freq_to_wave(λ, Iν) = 1e-17 * (C_KMS * 1e9) / λ^2 * Iν
emissivity(λ, a, type, Iν) = freq_to_wave(λ, Iν) * grain_σ(a, λ, type)


function ISRF(λ)
    if λ < 0.0912
        0
    elseif λ < 0.11
        38.57 / (C_KMS * 1e9 * 1e-17) * λ^5.4172
    elseif λ < 0.134
        2.045e-2 / (C_KMS * 1e9 * 1e-17) * λ^2
    elseif λ < 0.246
        7.115e-4 / (C_KMS * 1e9 * 1e-17) * λ^0.3322
    else
        4π * (1e-14 * Blackbody_ν(λ, 7500) + 1e-13 * Blackbody_ν(λ, 4000) + 4e-13 * Blackbody_ν(λ, 3000))
    end
end


function grain_integrals()
    @info "Pre-computing integrated silicate and graphite intensities"

    # Get vector of logarithmically-spaced blackbody temperatures and grain radii
    T_bb = exp10.(range(log10(3), log10(1750), 30))
    a = exp10.(range(log10(0.00031622776601683897), log10(10), 91))
    λ = exp10.(range(-3, 3, 241)) 

    # Integrate for various blackbody temperatures and grain radii
    sil_ints = [NumericalIntegration.integrate(λ, emissivity.(λ, ai, :sil, Blackbody_ν.(λ, Ti)), Trapezoidal()) for ai in a, Ti in T_bb]
    gra_ints = [NumericalIntegration.integrate(λ, emissivity.(λ, ai, :gra, Blackbody_ν.(λ, Ti)), Trapezoidal()) for ai in a, Ti in T_bb]

    # Create interpolating function over the grain radius
    a_interp = Spline1D(a, 1:length(a), k=1)
    # Create interpolating functions over the *Temperature*
    sil_interp = Vector{Spline1D}(undef, length(a))
    gra_interp = Vector{Spline1D}(undef, length(a))
    for i in eachindex(a)
        sil_interp[i] = Spline1D(log10.(sil_ints[i, :]), log10.(T_bb), k=1, bc="extrapolate")
        gra_interp[i] = Spline1D(log10.(gra_ints[i, :]), log10.(T_bb), k=1, bc="extrapolate")
    end

    a_interp, sil_interp, gra_interp
end

a_interp, sil_interp, gra_interp = grain_integrals()

# Compute the equilibrium Temperature of dust grains at a given radius and source radiation 
function grain_Teq(a::Real, T_bb::Real, Iν_source::Symbol=:stellar)
    # Construct functions to be integrated based on the radiation source
    λ = exp10.(range(-3, 3, 241))
    if Iν_source == :stellar
        Iλ = freq_to_wave.(λ, ISRF.(λ))
    else
        error("unrecognized radiation source $Iν_source")
    end
    Eλ_sil = @. Iλ * grain_σ(a, λ, :sil)
    Eλ_gra = @. Iλ * grain_σ(a, λ, :gra)
    # pure integrated intensity
    I_int = NumericalIntegration.integrate(λ, Iλ, Trapezoidal())
    E_int_sil = NumericalIntegration.integrate(λ, Eλ_sil, Trapezoidal()) / I_int
    E_int_gra = NumericalIntegration.integrate(λ, Eλ_gra, Trapezoidal()) / I_int
    # Rescale to the value of a pure blackbody integral with no emissivity
    E_int_sil = σ_SB/π * T_bb^4 * E_int_sil
    E_int_gra = σ_SB/π * T_bb^4 * E_int_gra
    # Find the interpolation for the grain radius based on the grainradii vector
    grain_loc = a_interp(a)
    # Nearest left/right indices
    indices = floor(Int, grain_loc), ceil(Int, grain_loc)
    # Weights are the distances to the closest integers -- doing a linear interpolation
    if indices[1] == indices[2]
        weights = 0.5, 0.5
    else
        weights = (1 - (grain_loc - indices[1])) / (indices[2]-indices[1]),  (1 - (indices[2] - grain_loc)) / (indices[2]-indices[1])
    end
    # Interpolate the temperature weighted by the weights
    T_eq_sil = exp10(sil_interp[indices[1]](log10(E_int_sil))) * weights[1] + 
               exp10(sil_interp[indices[2]](log10(E_int_sil))) * weights[2]
    T_eq_gra = exp10(gra_interp[indices[1]](log10(E_int_gra))) * weights[1] + 
               exp10(gra_interp[indices[2]](log10(E_int_gra))) * weights[2]

    T_eq_sil, T_eq_gra
end


const b_C = 55.8       # Carbon abundance (C/H) in VSG components [ppm]
const a_min = 3.5e-4   # Absolute minimum grain-size [um]
const a_max = 10.0     # Absolute maximum grain-size [um]
const C_G = 9.99e-12   # Graphite abundance
const C_S = 1.0e-13    # Silicate abundance
const α_G = 1.54       # Power-law Index
const α_S = 2.21
const aT_G = 0.0107    # Turnover size [um]
const aT_S = 0.164
const aC_G = 0.428     # Cutoff size [um]
const aC_S = 0.1
const β_G = -0.165     # Curvature function F(a) argument
const β_S = 0.300
const vsg_σ = 0.4      # Log-normal width of the VSG DF
const ρ_G = 2.24       # Density of graphite in g/cm^3
const m_C = 1.99e-23   # Mass of Carbon in g
const a0_i = [3.5, 30]
const bC_i = [0.75, 0.25]

# Coefficients for the VSG DF
const B = @. 3/(2π)^(3/2) * exp(-4.5*vsg_σ^2) / (ρ_G * (a0_i * 1e-8)^3 * vsg_σ) * bC_i * 1e-6 * b_C * m_C / 
    (1 + erf(3vsg_σ/sqrt(2) + log(a0_i/3.5)/vsg_σ*sqrt(2)))


function grain_size_distribution_gra(a::Real, T_eq::Real)
    # Check for hard cutoffs
    T_sub = 1750
    if !(a_min < a < a_max) || T_eq > T_sub
        return 0.
    end

    # Calculate the very small grain distribution function
    dnda_VSG = sum(@. B/a * exp(-0.5 * log(a*1e4/a0_i)^2/vsg_σ^2))

    # Calculate the curvature
    F = β_G > 0 ? 1 + β_G * a/aT_G : 1 / (1 - β_G * a/aT_G)

    # Initial distribution function without any cutoffs
    dnda_G_0 = (C_G/a) * (a/aT_G)^-α_G * F

    # Including soft cutoffs
    dnda_G = a > aT_G ? dnda_G_0 * exp(-(((a - aT_G)/aC_G)^3)) : dnda_G_0

    return dnda_VSG + dnda_G
end


function grain_size_distribution_sil(a::Real, T_eq::Real)
    # Check for hard cutoffs
    T_sub = 1400
    if !(a_min < a < a_max) || T_eq > T_sub
        return 0.
    end

    # Calculate the curvature
    F = β_S > 0 ? 1 + β_S * a/aT_S : 1 / (1 - β_S * a/aT_S)

    # Initial distribution function without any cutoffs
    dnda_S_0 = (C_S/a) * (a/aT_S)^-α_S * F

    # Including soft cutoffs
    dnda_S = a > aT_S ? dnda_S_0 * exp(-(((a - aT_S)/aC_S)^3)) : dnda_S_0

    return dnda_S
end


function grain_size_distribution(a::Real, T_eq::Tuple{<:Real,<:Real})
    grain_size_distribution_sil(a, T_eq[1]), grain_size_distribution_gra(a, T_eq[2])
end


function grain_emissivity(λ::Vector{<:Real}, T_bb::Real)
    a = exp10.(range(log10(0.00031622776601683897), log10(10), 91))
    σ_sil = [grain_σ.(a[j], λ[i], :sil) for i in eachindex(λ), j in eachindex(a)]
    σ_gra = [grain_σ.(a[j], λ[i], :gra) for i in eachindex(λ), j in eachindex(a)]
    T_eq = grain_Teq.(a, T_bb)
    T_eq_sil = [T_eq[i][1] for i in eachindex(T_eq)]
    T_eq_gra = [T_eq[i][2] for i in eachindex(T_eq)]
    dnda = grain_size_distribution.(a, T_eq)
    dnda_sil = [dnda[i][1] for i in eachindex(dnda)]
    dnda_gra = [dnda[i][2] for i in eachindex(dnda)]

    Eν_sil = [NumericalIntegration.integrate(a, dnda_sil .* σ_sil[i,:] .* Blackbody_ν.(λ[i], T_eq_sil), Trapezoidal()) for i in eachindex(λ)]
    Eν_gra = [NumericalIntegration.integrate(a, dnda_gra .* σ_gra[i,:] .* Blackbody_ν.(λ[i], T_eq_gra), Trapezoidal()) for i in eachindex(λ)]

    # integrand(a) = begin 
    #     T_eq_sil, T_eq_gra = grain_Teq(a, T_bb)
    #     E_sil = grain_σ(a, λ, :sil) * grain_size_distribution_sil(a, T_eq_sil) * Blackbody_ν(λ, T_eq_sil)
    #     E_gra = grain_σ(a, λ, :gra) * grain_size_distribution_gra(a, T_eq_gra) * Blackbody_ν(λ, T_eq_gra)
    #     E_sil + E_gra
    # end
    # quadgk(integrand, 3.1622776601683897e-4, 10, maxevals=100)[1]

    Eν_sil .+ Eν_gra
end