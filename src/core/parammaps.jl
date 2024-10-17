

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

    function ParamMaps(model::ModelParameters, shape::Tuple{S,S})::ParamMaps where {S<:Integer}

        @debug """\n
        Creating ParamMaps struct with shape $shape
        ##############################################
        """

        # Initialize a default array of nans to be used as a placeholder for all the other arrays
        # until the actual fitting parameters are obtained
        mp = get_flattened_parameters(model)
        data = Array{Quantity{Float64}}(undef, shape..., length(mp))
        data .= NaN

        new(data, copy(data), copy(data), mp)

    end
end


function get_val(parammap::ParamMaps, pname::String)
    param = get_flattened_parameters(parammap.parameters)
    ind = findfirst(param.names .== pname)
    parammap.data[:, :, ind]
end
function get_val(parammap::ParamMaps, pnames::Vector{String})
    param = get_flattened_parameters(parammap.parameters)
    inds = [findfirst(param.names .== pname) for pname in pnames]
    parammap.data[:, :, inds]
end
get_val(parammap::ParamMaps, index::CartesianIndex, pname::String) = get_val(parammap, pname)[index]
get_val(parammap::ParamMaps, index::CartesianIndex, pnames::Vector{String}) = get_val(parammap, pnames)[index]

function get_err(parammap::ParamMaps, pname::String)
    param = get_flattened_parameters(parammap.parameters)
    ind = findfirst(param.names .== pname)
    parammap.err_upp[:, :, ind], parammap.err_low[:, :, ind]
end
function get_err(parammap::ParamMaps, pnames::Vector{String})
    param = get_flattened_parameters(parammap.parameters)
    inds = [findfirst(param.names .== pname) for pname in pnames]
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
    param = get_flattened_parameters(parammap.parameters)
    ind = findfirst(param.names .== pname)
    param.labels[ind]
end
function get_label(parammap::ParamMaps, pnames::Vector{String})
    param = get_flattened_parameters(parammap.parameters)
    inds = [findfirst(param.names .== pname) for pname in pnames]
    param.labels[inds]
end


"""
    generate_parammaps(cube_fitter[, oneD])

Generate three ParamMaps objects (for the values and upper/lower errors) corrresponding to the options given
by the CubeFitter object.
"""
function generate_parammaps(cube_fitter::CubeFitter, oneD::Bool=false)
    shape = oneD ? (1,1,size(cube_fitter.cube.I, 3)) : size(cube_fitter.cube.I)
    ParamMaps(cube_fitter.model, shape)
end
