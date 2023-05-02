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


"""
    Continuum(T_s, T_dc, τ_97, τ_ice, τ_ch, β, T_hot, Cf_hot, τ_warm, τ_cold)

A container for various continuum modeling parameters.
"""
struct Continuum

    # Continuum parameters
    T_s::Parameter
    T_dc::Vector{Parameter}
    α::Vector{Parameter}
    τ_97::Parameter
    τ_ice::Parameter
    τ_ch::Parameter
    β::Parameter
    T_hot::Parameter
    Cf_hot::Parameter
    τ_warm::Parameter
    τ_cold::Parameter

end


"""
    DustFeatures(names, profiles, mean, fwhm)

A container for the modeling parameters relating to PAH dust features.
"""
struct DustFeatures

    names::Vector{String}
    profiles::Vector{Symbol}
    mean::Vector{Parameter}
    fwhm::Vector{Parameter}

end


"""
    TransitionLines(names, λ₀, profiles, tied, voff, fwhm, h3, h4, η)

A container for ancillary information and modeling parameters relating to transition lines.
"""
struct TransitionLines

    # 1st axis: labels each transition line
    names::Vector{Symbol}
    λ₀::Vector{AbstractFloat}
    
    # 1st axis: labels each transition line
    # 2nd axis: labels the components of each line
    profiles::Matrix{Union{Symbol,Nothing}}
    tied_voff::Matrix{Union{Symbol,Nothing}}
    tied_fwhm::Matrix{Union{Symbol,Nothing}}

    # Model Parameters
    voff::Matrix{Union{Parameter,Nothing}}
    fwhm::Matrix{Union{Parameter,Nothing}}
    h3::Matrix{Union{Parameter,Nothing}}
    h4::Matrix{Union{Parameter,Nothing}}
    η::Matrix{Union{Parameter,Nothing}}

end


"""
    TiedKinematics

A container for tied kinematic parameter information.
"""
struct TiedKinematics

    # Vectors of vectors, rather than Matrices, since the size of each element may be inhomogeneous 
    key_voff::Vector{Vector{Symbol}}
    voff::Vector{Vector{Parameter}}
    key_fwhm::Vector{Vector{Symbol}}
    fwhm::Vector{Vector{Parameter}}

end