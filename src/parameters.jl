module Param

# Import packages
using Distributions
using JSON

"""
    Parameter(value, locked, prior)

A struct for holding information about parameters' intial values and priors

# Fields
- `value::Number`: The initial value of the parameter
- `locked::Bool`: false to allow the parameter to vary based on the prior, true to keep it fixed
- `prior::UnivariateDistribution`: distribution that defines the prior (i.e. Normal, Uniform, LogUniform)
- `mcmc_scale::Number`: the MCMC walker search scale for the parameter
"""
mutable struct Parameter

    value::Number
    locked::Bool
    prior::UnivariateDistribution
    mcmc_scale::Number
    
    function Parameter(value::Number, locked::Bool, prior::UnivariateDistribution)
        if typeof(prior) <: Normal
            mcmc_scale = std(prior) / 10
        elseif typeof(prior) <: Uniform
            dx₁ = abs(maximum(prior) - value)
            dx₂ = abs(value - minimum(prior))
            mcmc_scale = minimum([dx₁, dx₂]) / 100
        else
            mcmc_scale = value / 100
        end

        return new(value, locked, prior, mcmc_scale)
    end

end


"""
    show([io,] p)

Show the parameter object as a nicely formatted string
"""
function Base.show(io::IO, p::Parameter)
    Base.show(io, "Parameter: value = $(p.value) | locked = $(p.locked) | prior = $(p.prior)")
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
    if "plim" ∈ keys(dict) && !("pval" ∈ keys(dict))
        prior = truncated(Uniform(-Inf,Inf), dict["plim"]...)
    end
    # pval: prior values, which may be normal (mean, std) or anything else
    if "pval" ∈ keys(dict)
        pval = dict["pval"]
        prior = eval(Meta.parse(dict["prior"] * "($pval...)"))
        prior = truncated(prior, dict["plim"]...)
    end

    return Parameter(value, locked, prior)
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
    if "plim" ∈ keys(dict) && !("pval" ∈ keys(dict))
        plim = dict["plim"] .+ value
        prior = truncated(Uniform(-Inf,Inf), plim...)
    end
    # pval: prior values, which may be normal (mean, std) or anything else
    if "pval" ∈ keys(dict)
        pval = dict["pval"]
        prior = eval(Meta.parse(dict["prior"] * "($pval...)"))
        prior = truncated(prior, dict["plim"]...)
    end

    return Parameter(value, locked, prior)
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
    if "plim" ∈ keys(dict) && !("pval" ∈ keys(dict))
        plim = dict["plim"] .* value
        prior = truncated(Uniform(-Inf,Inf), plim...)
    end
    # pval: prior values, which may be normal (mean, std) or anything else
    if "pval" ∈ keys(dict)
        pval = dict["pval"]
        prior = eval(Meta.parse(dict["prior"] * "($pval...)"))
        prior = truncated(prior, dict["plim"]...)
    end

    return Parameter(value, locked, prior)
end


# Aliasing a helpful composite type
ParamDict = Dict{Union{Symbol,String}, Parameter}


"""
    TransitionLine(λ₀, profile, flow_profile, parameters, tied, flow_tied)

A struct for an emission/absorption Line

# Fields
- `λ₀::AbstractFloat`: The central wavelength of the line in the rest frame
- `profile::Symbol`: The type of profile to fit with, i.e. `:Gaussian`, `:Lorentzian`, `:GaussHermite`, or `:Voigt`
- `flow_profile::Union{Symbol,Nothing}`: Same as `profile`, but for an inflow/outflow component. Leave as `nothing` to
    not include any inflow/outflow components
- `parameters::ParamDict`: All the necessary fitting parameters for the line, based on the profile, as Parameter objects
    (i.e. amplitude, voff, FWHM, etc.)
- `tied::Union{String,Nothing}`: If the voff should be tied to other lines, this should be a String that is the same
    between all of the lines that share a voff. Otherwise, keep it as `nothing` to have an untied voff.
- `flow_tied::Union{String,Nothing}`: Same as `tied`, but for an inflow/outflow voff component
"""
struct TransitionLine
    """
    A structure for an emission/absorption line with a given name, rest wavelength, and fitting parameters
    """
    λ₀::AbstractFloat
    profile::Symbol
    flow_profile::Union{Symbol,Nothing}
    parameters::ParamDict
    tied::Union{String,Nothing}
    flow_tied::Union{String,Nothing}

end


# Another alias for a helpful composite type
LineDict = Dict{Union{Symbol,String}, TransitionLine}

end