

# """
#     OpticalCubeModel(model, stellar, lines)

# A structure for holding 3D models of intensity, split up into model components, generated when fitting a cube.
# This will be the same shape as the input data, and preferably the same datatype too (i.e., JWST files have flux
# and error in Float32 format, so we should also output in Float32 format).  This is useful as a way to quickly
# compare the full model, or model components, to the data.

# # Fields {T<:Real}
# - `model::Array{T, 3}`: The full 3D model.
# - `stellar::Array{T, 4}`: The simple stellar population components of the continuum. The 4th axis runs over each individual population.
# - `na_feii::Array{T, 3}`: The narrow Fe II emission component.
# - `br_feii::Array{T, 3}`: The broad Fe II emission component.
# - `power_law::Array{T, 4}`: The power law components of the continuum. The 4th axis runs over each individual power law.
# - `attenuation_stars::Array{T, 3}`: The dust attenuation on the stellar population.
# - `attenuation_gas::Array{T, 3}`: The dust attenuation on the gas, related to attenuation_stars by E(B-V)_stars = E(B-V)_factor * E(B-V)_gas
# - `templates::Array{T, 4}`: The generic template profiles.
# - `lines::Array{T, 4}`: The line profiles. The 4th axis runs over each individual line.

# See [`cubemodel_empty`](@ref) for a default constructor method.
# """
# struct OpticalCubeModel{T<:Real} <: CubeModel

#     model::Array{T, 3}
#     stellar::Array{T, 4}
#     na_feii::Array{T, 3}
#     br_feii::Array{T, 3}
#     power_law::Array{T, 4}
#     attenuation_stars::Array{T, 3}
#     attenuation_gas::Array{T, 3}
#     templates::Array{T, 4}
#     lines::Array{T, 4}

# end


# """
#     cubemodel_empty(shape, n_ssps, n_power_law, line_names[, floattype])

# A constructor function for making a default empty OpticalCubeModel object with all the necessary fields for a given
# fit of a DataCube.
    
# # Arguments
# - `shape::Tuple`: The dimensions of the DataCube being fit, formatted as a tuple of (nx, ny, nz)
# - `n_ssps::Integer`: The number of simple stellar population continuum components in the fit.
# - `n_power_law::Integer`: The number of power law continuum components in the fit.
# - `line_names::Vector{Symbol}`: List of names of lines being fit, i.e. "NeVI_7652", ...
# - `temp_names::Vector{String}`: List of names of generic templates in the fit, i.e. "nuclear", ...
# - `floattype::DataType=Float32`: The type of float to use in the arrays. Should ideally be the same as the input data,
# which for JWST is Float32.
# """
# function cubemodel_empty(shape::Tuple, n_ssps::Integer, n_power_law::Integer, line_names::Vector{Symbol}, temp_names::Vector{String}, 
#     floattype::DataType=Float32)::OpticalCubeModel

#     @debug """\n
#     Creating OpticalCubeModel struct with shape $shape
#     ##################################################
#     """

#     @assert floattype <: AbstractFloat "floattype must be a type of AbstractFloat (Float32 or Float64)!"
#     # Swap the wavelength axis to be the FIRST axis since it is accessed most often and thus should be continuous in memory
#     shape2 = (shape[end], shape[1:end-1]...)

#     # Initialize the arrays for each part of the full 3D model
#     model = zeros(floattype, shape2...)
#     @debug "model cube"
#     stellar = zeros(floattype, shape2..., n_ssps)
#     @debug "stellar population comp cubes"
#     na_feii = zeros(floattype, shape2...)
#     @debug "narrow Fe II emission comp cube"
#     br_feii = zeros(floattype, shape2...)
#     @debug "broad Fe II emission comp cube"
#     power_law = zeros(floattype, shape2..., n_power_law)
#     @debug "power law comp cubes"
#     attenuation_stars = zeros(floattype, shape2...)
#     @debug "attenuation_stars comp cube"
#     attenuation_gas = zeros(floattype, shape2...)
#     @debug "attenuation_gas comp cube"
#     templates = zeros(floattype, shape2..., length(temp_names))
#     @debug "templates comp cube"
#     lines = zeros(floattype, shape2..., length(line_names))
#     @debug "lines comp cubes"

#     OpticalCubeModel(model, stellar, na_feii, br_feii, power_law, attenuation_stars, attenuation_gas, templates, lines)

# end
