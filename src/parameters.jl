module Param

# Import packages
using Distributions
using JSON

mutable struct Parameter{T<:UnivariateDistribution}
    """
    A struct for holding information about parameters' initial value and priors
    
    :param value: Float
        the initial value of the parameter
    :param locked: Bool
        false to allow the parameter to vary based on the prior, true to keep it fixed
    :param prior: UnivariateDistribution
        distribution structure that defines the prior (i.e. Normal, Uniform, LogUniform)
    :param mcmc_scale: Float
        the mcmc walker scale for the parameter
    """

    value::Float64
    locked::Bool
    prior::T
    mcmc_scale::Float64
    
    function Parameter(value::Float64, locked::Bool, prior::T) where {T<:UnivariateDistribution}
        """
        Constructor function with keyword arguments
        """
        if typeof(prior) <: Normal
            mcmc_scale = std(prior) / 10
        elseif typeof(prior) <: Uniform
            dx₁ = abs(maximum(prior) - value)
            dx₂ = abs(value - minimum(prior))
            mcmc_scale = minimum([dx₁, dx₂]) / 100
        else
            mcmc_scale = value / 100
        end

        return new{T}(value, locked, prior, mcmc_scale)
    end

end


function from_dict(dict::Dict)
    """
    Constructor function from dictionary
    """
    value = dict["val"]
    locked = dict["locked"]
    pval = dict["pval"]
    prior = eval(Meta.parse(dict["prior"] * "($pval...)"))

    return Parameter(value, locked, prior)
end 

function from_dict_wave(dict::Dict)
    value = dict["val"]
    locked = dict["locked"]
    pval = dict["val"] .+ dict["pval"]
    prior = eval(Meta.parse(dict["prior"] * "($pval...)"))

    return Parameter(value, locked, prior)
end

function from_dict_fwhm(dict::Dict)
    value = dict["val"]
    locked = dict["locked"]
    pval = dict["val"] .* dict["pval"]
    prior = eval(Meta.parse(dict["prior"] * "($pval...)"))

    return Parameter(value, locked, prior)
end

# Aliasing
ParamDict = Dict{Union{Symbol,String}, Parameter}

struct TransitionLine
    """
    A structure for an emission/absorption line with a given name, rest wavelength, and fitting parameters
    """
    λ₀::Float64
    profile::Symbol
    parameters::ParamDict

end

# More aliasing
LineDict = Dict{Union{Symbol,String}, TransitionLine}

end