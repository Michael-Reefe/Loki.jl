#=
This file, like utils, is not intended to be directly accessed by the 
user when fitting IFU cubes. Rather, it contains various Parameter structures
that are helpful for containing certain combinations of model parameters and
related quantities.
=#

"""
    Parameter(value, locked, limits)

A struct for holding information about parameters' intial values and priors

# Fields
- `value::Number`: The initial value of the parameter
- `locked::Bool`: false to allow the parameter to vary based on the prior, true to keep it fixed
- `limits::Tuple`: lower/upper limits on the parameter, if it is allowed to vary
"""
mutable struct Parameter{T<:Number}

    value::T
    locked::Bool
    limits::Tuple{T, T}
    
    # Constructor function
    function Parameter(value::T, locked::Bool, limits::Tuple{T, T}) where {T<:Number}

        # Make sure the upper limit is greater than the lower limit
        @assert limits[2] > limits[1]

        new{typeof(value)}(value, locked, limits)
    end

end


"""
    show([io,] p)

Show the parameter object as a nicely formatted string
"""
function Base.show(io::IO, p::Parameter)
    Base.show(io, "Parameter: value = $(p.value) | locked = $(p.locked) | limits = $(p.limits)")
end


"""
    from_dict(dict)

A constructor function for Parameter structs given a Dictionary
"""
function from_dict(dict::Dict)::Parameter

    # Unpack the dictionary into fields of the Parameter
    value = dict["val"]
    locked = dict["locked"]
    # plim: absolute upper/lower limits
    lims = (dict["plim"]...,)

    Parameter(value, locked, lims)
end 


"""
    from_dict_wave(dict)

A constructor function for Parameter structs given a Dictionary,
using deltas on upper/lower limits, i.e. if val = 5 and plim = [-0.1, 0.1],
then the true limits will be 5 .+ [-0.1, 0.1] = [4.9, 5.1]
"""
function from_dict_wave(dict::Dict)::Parameter

    # Unpack the dictionary into fields of the Parameter
    value = dict["val"]
    locked = dict["locked"]
    # plim: absolute upper/lower limits
    lims = (dict["plim"].+value...,)

    Parameter(value, locked, lims)
end


"""
    from_dict_fwhm(dict)

A constructor function for Parameter structs given a Dictionary,
using fractional values on upper/lower limits, i.e. if val = 5 and
plim = [0.5, 2], then the true limits will be 5 .* [0.5, 2] = [2.5, 10]
"""
function from_dict_fwhm(dict::Dict)::Parameter

    # Unpack the dictionary into fields of the Parameter
    value = dict["val"]
    locked = dict["locked"]
    # plim: absolute upper/lower limits
    lims = (dict["plim"].*value...,)

    Parameter(value, locked, lims)
end


abstract type Continuum end

"""
    MIRContinuum(T_s, T_dc, α, τ_97, τ_ice, τ_ch, β, T_hot, Cf_hot, τ_warm, τ_cold, sil_peak, temp_amp)

A container for various MIR continuum modeling parameters.
"""
struct MIRContinuum <: Continuum

    # MIR Continuum parameters
    T_s::Parameter
    T_dc::Vector{Parameter}
    α::Vector{Parameter}
    τ_97::Parameter
    N_oli::Parameter
    N_pyr::Parameter
    N_for::Parameter
    τ_ice::Parameter
    τ_ch::Parameter
    β::Parameter
    Cf::Parameter
    T_hot::Parameter
    Cf_hot::Parameter
    τ_warm::Parameter
    τ_cold::Parameter
    sil_peak::Parameter
    temp_amp::Vector{Parameter}

end


"""
    OpticalContinuum(ssp_ages, ssp_metallicities, stel_vel, stel_vdisp,
        na_feii_vel, na_feii_vdisp, br_feii_vel, br_feii_vdisp, α, E_BV,
        E_BV_factor, δ_uv, frac)

A container for various optical continuum modeling parameters.
"""
struct OpticalContinuum <: Continuum

    # Simple Stellar Population parameters
    ssp_ages::Vector{Parameter}
    ssp_metallicities::Vector{Parameter}

    # Stellar kinematics
    stel_vel::Parameter
    stel_vdisp::Parameter

    # Fe II kinematics
    na_feii_vel::Parameter
    na_feii_vdisp::Parameter
    br_feii_vel::Parameter
    br_feii_vdisp::Parameter
    
    # Power law indices
    α::Vector{Parameter}

    # Dust attenuation
    E_BV::Parameter
    E_BV_factor::Parameter
    δ_uv::Parameter
    frac::Parameter

end


"""
    DustFeatures(names, profiles, mean, fwhm, index, cutoff, complexes, _local)

A container for the modeling parameters relating to PAH dust features
and absorption features.
"""
struct DustFeatures

    names::Vector{String}
    profiles::Vector{Symbol}
    mean::Vector{Parameter}
    fwhm::Vector{Parameter}
    index::Vector{Union{Parameter,Nothing}}
    cutoff::Vector{Union{Parameter,Nothing}}
    complexes::Vector{Union{String,Nothing}}
    _local::BitVector

end


"""
    TransitionLines(names, latex, annotate, λ₀, profiles, tied_amp, tied_voff, tied_fwhm,
        acomp_amp, voff, fwhm, h3, h4, η, combined)

A container for ancillary information and modeling parameters relating to transition lines.
"""
struct TransitionLines

    # 1st axis: labels each transition line
    names::Vector{Symbol}
    latex::Vector{String}
    annotate::BitVector
    λ₀::Vector{AbstractFloat}
    sort_order::Vector{Int}
    
    # 1st axis: labels each transition line
    # 2nd axis: labels the components of each line
    profiles::Matrix{Union{Symbol,Nothing}}
    tied_amp::Matrix{Union{Symbol,Nothing}}
    tied_voff::Matrix{Union{Symbol,Nothing}}
    tied_fwhm::Matrix{Union{Symbol,Nothing}}

    # Model Parameters
    acomp_amp::Matrix{Union{Parameter,Nothing}}
    voff::Matrix{Union{Parameter,Nothing}}
    fwhm::Matrix{Union{Parameter,Nothing}}
    h3::Matrix{Union{Parameter,Nothing}}
    h4::Matrix{Union{Parameter,Nothing}}
    η::Matrix{Union{Parameter,Nothing}}

    # Combined lines (for map purposes)
    combined::Vector{Vector{Symbol}}

    # Relative flags for additional components
    rel_amp::Bool
    rel_voff::Bool
    rel_fwhm::Bool

end


"""
    TiedKinematics(key_amp, amp, key_voff, voff, key_fwhm, fwhm)

A container for tied amplitude and kinematic parameter information.
"""
struct TiedKinematics

    # Vectors of vectors, rather than Matrices, since the size of each element may be inhomogeneous 
    key_amp::Vector{Vector{Symbol}}
    amp::Vector{Vector{Dict{Symbol, <:Real}}}
    key_voff::Vector{Vector{Symbol}}
    voff::Vector{Vector{Parameter}}
    key_fwhm::Vector{Vector{Symbol}}
    fwhm::Vector{Vector{Parameter}}

end