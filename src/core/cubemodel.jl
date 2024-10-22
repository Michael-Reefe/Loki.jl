
"""
    CubeModel

A structure for holding 3D models of intensity, split up into model components, generated when fitting a cube.
This will be the same shape as the input data.

See [`cubemodel_empty`](@ref) for a default constructor method.
"""
struct CubeModel{T<:Real,
                 S<:Union{typeof(1.0*u"erg/s/cm^2/Hz/sr"),typeof(1.0f0*u"erg/s/cm^2/Hz/sr"),
                          typeof(1.0*u"erg/s/cm^2/μm/sr"),typeof(1.0f0*u"erg/s/cm^2/μm/sr"),
                          typeof(1.0*u"erg/s/cm^2/angstrom/sr"),typeof(1.0f0*u"erg/s/cm^2/angstrom/sr")}
                }
    # full model
    model::Array{S, 3}

    # individual components
    extinction_stars::Array{T, 3}
    extinction_gas::Array{T, 3}
    absorption_silicates::Array{T, 4}  # (may be just tau_97 or N_oli, N_pyr, N_for)
    abs_ice::Union{Array{T, 3}, Nothing}
    abs_ch::Union{Array{T, 3}, Nothing}
    stellar::Array{S, 3}
    na_feii::Union{Array{S, 3}, Nothing}
    br_feii::Union{Array{S, 3}, Nothing}
    power_law::Array{S, 4}
    dust_continuum::Array{S, 4}
    hot_dust::Union{Array{S, 3}, Nothing}
    templates::Array{S, 4}

    # features
    abs_features::Array{T, 4}
    dust_features::Array{S, 4}
    lines::Array{S, 4}
end


# A constructor function for making a default empty CubeModel object with all the necessary fields for a given
# fit of a DataCube.
function generate_cubemodel(cube_fitter::CubeFitter, floattype::DataType=Float32; do_1d::Bool=false)

    @debug "Generating full 3D cube models"
    shape = do_1d ? (1,1,size(cube_fitter.cube.I, 3)) : size(cube_fitter.cube.I)

    @debug """\n
    Creating CubeModel struct with shape $shape
    ##############################################
    """

    # Make sure the floattype given is actually a type of float
    @assert floattype <: AbstractFloat "floattype must be a type of AbstractFloat (Float32 or Float64)!"

    # Swap the wavelength axis to be the FIRST axis since it is accessed most often and thus should be continuous in memory
    shape2 = (shape[end], shape[1:end-1]...)

    fopt = fit_options(cube_fitter)
    Iunit = unit(cube_fitter.cube.I[1])
    qtype = typeof(one(floattype) * Iunit)

    # Initialize the arrays for each part of the full 3D model
    model = zeros(qtype, shape2...)
    extinction_stars = zeros(floattype, shape2...)
    extinction_gas = zeros(floattype, shape2...)
    absorption_silicates = zeros(floattype, shape2..., fopt.extinction_curve == "decompose" ? 4 : 1)
    abs_ice = fopt.fit_ch_abs ? zeros(floattype, shape2...) : nothing
    abs_ch = fopt.fit_ch_abs ? zeros(floattype, shape2...) : nothing
    stellar = zeros(qtype, shape2...)
    na_feii = fopt.fit_opt_na_feii ? zeros(qtype, shape2...) : nothing
    br_feii = fopt.fit_opt_br_feii ? zeros(qtype, shape2...) : nothing
    power_law = zeros(qtype, shape2..., cube_fitter.n_power_law)
    dust_continuum = zeros(qtype, shape2..., cube_fitter.n_dust_cont)
    hot_dust = fopt.fit_sil_emission ? zeros(qtype, shape2...) : nothing
    templates = zeros(qtype, shape2..., cube_fitter.n_templates)

    abs_features = zeros(floattype, shape2..., cube_fitter.n_abs_feat)
    dust_features = zeros(qtype, shape2..., cube_fitter.n_dust_feat)
    lines = zeros(qtype, shape2..., cube_fitter.n_lines)

    CubeModel(model, extinction_stars, extinction_gas, absorption_silicates, abs_ice, abs_ch, stellar, na_feii, br_feii,
        power_law, dust_continuum, hot_dust, templates, abs_features, dust_features, lines)
end

