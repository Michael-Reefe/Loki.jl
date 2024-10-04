#=
This file, like utils, is not intended to be directly accessed by the 
user when fitting IFU cubes. Rather, it contains various Parameter structures
that are helpful for containing certain combinations of model parameters and
related quantities.
=#

# abstract type Parameter end

"""
    FitParameter(value, locked, limits)

A struct for holding information about parameters' intial values and priors

# Fields
- `value::Number`: The initial value of the parameter
- `locked::Bool`: false to allow the parameter to vary based on the prior, true to keep it fixed
- `limits::Tuple`: lower/upper limits on the parameter, if it is allowed to vary
"""
mutable struct FitParameter{T<:Number}

    value::T
    locked::Bool
    limits::Tuple{T, T}
    
    # Constructor function
    function FitParameter(value::T, locked::Bool, limits::Tuple{T, T}) where {T<:Number}
        # Make sure the upper limit is strictly greater than the lower limit
        @assert limits[2] > limits[1]
        new{T}(value, locked, limits)
    end

end


# lock or unlock the parameter
lock!(p::FitParameter) = (p.locked = true)
unlock!(p::FitParameter) = (p.locked = false)

# update the limits or value of the parameter
function set_plim!(p::FitParameter, limits::Tuple) 
    @assert limits[1] < limits[2]
    _, _, l1, l2 = promote(p.limits[1], p.limits[2], limits[1], limits[2])
    p.limits = (l1, l2)
end
function set_val!(p::FitParameter{T}, v::Number) where {T}
    _, vnew = promote(p.value, v)
    p.value = vnew
end

# check if everything is good
function check_valid(p::FitParameter)
    @assert isfinite(p.value)
    @assert typeof(p.value) == typeof(p.limits[1]) == typeof(p.limits[2])
    @assert p.limits[1] < p.limits[2]
    @assert p.limits[1] ≤ p.value ≤ p.limits[2]
end



"""
    show([io,] p)

Show the parameter object as a nicely formatted string
"""
function Base.show(io::IO, p::FitParameter)
    Base.show(io, "Parameter: value = $(p.value) | locked = $(p.locked) | limits = $(p.limits)")
end


"""
    parameter_from_dict(dict)

A constructor function for Parameter structs given a Dictionary
"""
function parameter_from_dict(dict::Dict)

    # Unpack the dictionary into fields of the Parameter
    value = dict["val"]
    locked = dict["locked"]
    # plim: absolute upper/lower limits
    lims = (dict["plim"]...,)

    FitParameter(value, locked, lims)
end 


"""
    parameter_from_dict_wave(dict)

A constructor function for Parameter structs given a Dictionary,
using deltas on upper/lower limits, i.e. if val = 5 and plim = [-0.1, 0.1],
then the true limits will be 5 .+ [-0.1, 0.1] = [4.9, 5.1]
"""
function parameter_from_dict_wave(dict::Dict)

    # Unpack the dictionary into fields of the Parameter
    value = dict["val"]
    locked = dict["locked"]
    # plim: absolute upper/lower limits
    lims = (dict["plim"].+value...,)

    FitParameter(value, locked, lims)
end


"""
    parameter_from_dict_fwhm(dict)

A constructor function for Parameter structs given a Dictionary,
using fractional values on upper/lower limits, i.e. if val = 5 and
plim = [0.5, 2], then the true limits will be 5 .* [0.5, 2] = [2.5, 10]
"""
function parameter_from_dict_fwhm(dict::Dict)

    # Unpack the dictionary into fields of the Parameter
    value = dict["val"]
    locked = dict["locked"]
    # plim: absolute upper/lower limits
    lims = (dict["plim"].*value...,)

    FitParameter(value, locked, lims)
end


struct FitParameters{T<:Number}
    names::Vector{String}                  # names of all the parameters
    _parameters::Vector{FitParameter{T}}   # the internal storage of the parameter objects
end


# methods for obtaining a named parameter
Base.getindex(p::FitParameters, ind) = p._parameters[ind]
function Base.getindex(p::FitParameters, name::String)
    ind = findfirst(p.names .== name)
    p._parameters[ind]
end
function Base.getindex(p::FitParameters, names::AbstractVector{String})
    inds = [findfirst(p.names .== name) for name in names]
    p._parameters[inds]
end

# methods for obtaining the vector of parameter limits
get_plims(p::FitParameters, ind::Int) = p[ind].limits   
get_plims(p::FitParameters, name::String) = p[name].limits   
function get_plims(p::FitParameters, names::Vector{String})
    plims = Vector{Tuple}(undef, length(names))
    for (i, param) in enumerate(p[names])
        plims[i] = param.limits
    end
    plims
end
function get_plims(p::FitParameters)
    get_plims(p, p.names)
end

# methods for obtaining the lock vector
get_lock(p::FitParameters, ind::Int) = p[ind].locked
get_lock(p::FitParameters, name::String) = p[name].locked
function get_lock(p::FitParameters, names::Vector{String})
    locks = BitVector(undef, length(names))
    for (i, param) in enumerate(p[names])
        locks[i] = param.locked
    end
    locks
end
function get_lock(p::FitParameters)
    get_lock(p, p.names)
end

# methods for obtaining the value vector
get_val(p::FitParameters, ind::Int) = p[ind].value
get_val(p::FitParameters, name::String) = p[name].value
function get_val(p::FitParameters{T}, names::Vector{String}) where {T}
    vals = Vector{T}(undef, length(names))
    for (i, param) in enumerate(p[names])
        vals[i] = param.value
    end
    vals
end
function get_val(p::FitParameters)
    get_val(p, p.names)
end

# methods for locking a named parameter
lock!(p::FitParameters, ind::Int) = lock!(p[ind])
lock!(p::FitParameters, name::String) = lock!(p[name])
function lock!(p::FitParameters, names::AbstractVector{String})
    for param in p[names]
        lock!(param)
    end
end

# methods for unlocking a named parameter
unlock!(p::FitParameters, ind::Int) = unlock!(p[ind])
unlock!(p::FitParameters, name::String) = unlock!(p[name])
function unlock!(p::FitParameters, names::AbstractVector{String})
    for param in p[names]
        unlock!(param)
    end
end

# methods for updating parameter limits
set_plim!(p::FitParameters, ind::Int, limits::Tuple) = set_plim!(p[ind], limits)
set_plim!(p::FitParameters, name::String, limits::Tuple) = set_plim!(p[name], limits)
function set_plim!(p::FitParameters, names::AbstractVector{String}, limits::AbstractVector{<:Tuple})
    for (param, plimit) in zip(p[names], limits)
        set_plim!(param, plimit)
    end
end

# methods for updating parameter values
set_val!(p::FitParameters, ind::Int, v::Number) = set_val!(p[ind], v)
set_val!(p::FitParameters, name::String, v::Number) = set_val!(p[name], v)
function set_val!(p::FitParameters, names::AbstractVector{String}, vs::AbstractVector{<:Number})
    for (param, v) in zip(p[names], vs)
        set_val!(param, v)
    end
end

function check_valid(p::FitParameters)
    for param in p[:]
        check_valid(param)
    end
end


"""
    NonFitParameter(name, value)

A parameter that is not being fit for, but is nevertheless of interest to us

# Fields
- `name::String`: The name of the parameter
- `value::Number`: The value of the parameter
"""
mutable struct NonFitParameter{T<:Number}
    const name::String
    value::T
end


"""
"""
struct FitProfiles{T<:Number,S<:Union{Symbol,String}}
    profiles::Vector{S}
    complexes::Vector{Union{String,Nothing}}
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
    make_single_transitionline_object(lines, i)

Takes a TransitionLine object and an index i and creates a new TransitionLine
object which contains only the single line at index i in the original object.
"""
function make_single_transitionline_object(lines::TransitionLines, i::Integer)
    n_comps = size(lines.profiles, 2)

    TransitionLines(
        [lines.names[i]], 
        [lines.latex[i]], 
        [lines.annotate[i]],
        [lines.λ₀[i]], 
        [lines.sort_order[i]],
        reshape(lines.profiles[i, :], (1, n_comps)), 
        reshape(lines.tied_amp[i, :], (1, n_comps)),
        reshape(lines.tied_voff[i, :], (1, n_comps)),
        reshape(lines.tied_fwhm[i, :], (1, n_comps)), 
        reshape(lines.acomp_amp[i, :], (1, n_comps-1)), 
        reshape(lines.voff[i, :], (1, n_comps)), 
        reshape(lines.fwhm[i, :], (1, n_comps)), 
        reshape(lines.h3[i, :], (1, n_comps)),
        reshape(lines.h4[i, :], (1, n_comps)), 
        reshape(lines.η[i, :], (1, n_comps)),
        lines.combined, 
        lines.rel_amp,
        lines.rel_voff,
        lines.rel_fwhm
    )
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