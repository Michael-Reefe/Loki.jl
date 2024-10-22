

"""
    ParamMaps{T<:Real}

A basic structure for holding parameter best-fit values and errors along with the parameters to 
keep track of where each parameter is located.

# Fields
- `data`: A 3D array holding the best-fit parameters for each spaxel.
- `err_upp`: A 3D array holding the upper uncertainties for each spaxel.
- `err_low`: A 3D array holding the lower uncertainties for each spaxel.
- `parameters`: A ModelParameters object containing the names, labels, etc. for all of the parameters
"""
struct ParamMaps

    data::Array{<:Number,3}
    err_upp::Array{<:Number,3}
    err_low::Array{<:Number,3}
    parameters::AllParameters

    function ParamMaps(model::ModelParameters, shape::Tuple{S,S}; stellar_params::Int=0)::ParamMaps where {S<:Integer}

        @debug """\n
        Creating ParamMaps struct with shape $shape
        ##############################################
        """

        # Initialize a default array of nans to be used as a placeholder for all the other arrays
        # until the actual fitting parameters are obtained
        mp = get_flattened_parameters(model)
        if stellar_params > 0
            # add a parameter for the total stellar mass (its ALREADY rest-frame transformed)
            push!(mp, "continuum.stellar_populations.mass", L"$\log_{10}(M/M_\odot)$", 
                [LogTransform], NonFitParameter{typeof(1.0u"Msun")}())
            # add N parameters for each age and each metallicity
            for i in 1:stellar_params
                push!(mp, "continuum.stellar_populations.$i.age", "Age (Gyr)",
                    Transformation[], NonFitParameter{typeof(1.0u"Gyr")}())
            end
            for i in 1:stellar_params
                push!(mp, "continuum.stellar_populations.$i.metallicity", L"$\log_{10}(Z/Z_\odot)$",
                    Transformation[], NonFitParameter{Float64}())
            end
        end
        n_param = length(mp)
            
        data = Array{Quantity{Float64}}(undef, shape..., n_param)
        data .= NaN

        new(data, copy(data), copy(data), mp)

    end
end


function get_val(parammap::ParamMaps, pname::String)
    ind = findfirst(parammap.parameters.names .== pname)
    parammap.data[:, :, ind]
end
function get_val(parammap::ParamMaps, pnames::Vector{String})
    inds = [findfirst(parammap.parameters.names .== pname) for pname in pnames]
    parammap.data[:, :, inds]
end
get_val(parammap::ParamMaps, index::CartesianIndex, pname::String) = get_val(parammap, pname)[index]
get_val(parammap::ParamMaps, index::CartesianIndex, pnames::Vector{String}) = get_val(parammap, pnames)[index]

function get_err(parammap::ParamMaps, pname::String)
    ind = findfirst(parammap.parameters.names .== pname)
    parammap.err_upp[:, :, ind], parammap.err_low[:, :, ind]
end
function get_err(parammap::ParamMaps, pnames::Vector{String})
    inds = [findfirst(parammap.parameters.names .== pname) for pname in pnames]
    parammap.err_upp[:, :, inds], parammap.err_low[:, :, inds]
end
function get_err(parammap::ParamMaps, index::CartesianIndex, pname::String)
    upp, low = get_err(parammap, pname)
    upp[index], low[index]
end
function get_err(parammap::ParamMaps, index::CartesianIndex, pnames::Vector{String})
    upp, low = get_err(parammap, pnames)
    upp[index], low[index]
end

function get_label(parammap::ParamMaps, pname::String)
    ind = findfirst(parammap.parameters.names .== pname)
    parammap.parameters.labels[ind]
end
function get_label(parammap::ParamMaps, pnames::Vector{String})
    inds = [findfirst(parammap.parameters.names .== pname) for pname in pnames]
    parammap.parameters.labels[inds]
end


"""
    generate_parammaps(cube_fitter[, do_1d])

Generate three ParamMaps objects (for the values and upper/lower errors) corrresponding to the options given
by the CubeFitter object.
"""
function generate_parammaps(cube_fitter::CubeFitter; do_1d::Bool=false, stellar_params::Int=0)
    shape = do_1d ? (1,1) : size(cube_fitter.cube.I)[1:2]
    ParamMaps(cube_fitter.model, shape, stellar_params=stellar_params)
end
