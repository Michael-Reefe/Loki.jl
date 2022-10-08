module Parameters

export Parameter, ParamDict, to_json, from_json

# Import packages
using Distributions
using JSON

struct Parameter{T<:UnivariateDistribution}
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
        Constructor function without keyword arguments
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
    
    function Parameter(value::Float64; locked::Bool, prior::T) where {T<:UnivariateDistribution}
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

ParamDict = Dict{Symbol, Parameter}

function to_json(parameters::ParamDict, fname::String)
    # Convert dict of parameters to serializable format
    serial = Dict()
    for key ∈ keys(parameters)
        serial[key] = Dict(ki=>getfield(parameters[key], ki) for ki ∈ fieldnames(Parameter))
        serial[key][:prior_name] = split(string(serial[key][:prior]), "{")[1]
        serial[key][:prior_param] = params(serial[key][:prior])
        delete!(serial[key], :prior)
    end

    # Save to json file
    open(fname, "w") do f
        JSON.print(f, serial, 4)
    end
end

function from_json(fname::String)
    # Read the JSON file
    serial = JSON.parsefile(fname)
    parameters = ParamDict()

    # Convert json strings to prior objects
    for key ∈ keys(serial)
        prior = eval(Meta.parse(serial[key]["prior_name"] * "($(serial[key]["prior_param"])...)"))
        param = Parameter(serial[key]["value"], serial[key]["locked"], prior)
        parameters[Symbol(key)] = param
    end

    return parameters
end

end