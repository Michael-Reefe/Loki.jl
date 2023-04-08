#=
This file contains the CubeFitter struct and its related functions.  The CubeFitter is how
fitting is performed with Loki.
=#

############################## PARAMETER / MODEL STRUCTURES ####################################

"""
    ParamMaps(stellar_continuum, dust_continuum, dust_features, lines, tied_voffs, acomp_tied_voffs,
        tied_voigt_mix, extinction, hot_dust, reduced_χ2)

A structure for holding 2D maps of fitting parameters generated after fitting a cube.  Each parameter
that is fit (i.e. stellar continuum temperature, optical depth, line ampltidue, etc.) corresponds to one 
2D map with the value of that parameter, with each spaxel's fit value located at the corresponding location
in the parameter map.

# Fields
- `stellar_continuum::Dict{Symbol, Array{Float64, 2}}`: The stellar continuum parameters: amplitude and temperature
- `dust_continuum::Dict{Int, Dict{Symbol, Array{Float64, 2}}}`: The dust continuum parameters: amplitude and temperature for each
    dust continuum component
- `dust_features::Dict{String, Dict{Symbol, Array{Float64, 2}}}`: The dust feature parameters: amplitude, central wavelength, and FWHM
    for each PAH feature
- `lines::Dict{Symbol, Dict{Symbol, Array{Float64, 2}}}`: The emission line parameters: amplitude, voff, FWHM, and any additional 
    line profile parameters for each line
- `tied_voffs::Dict{String, Array{Float64, 2}}`: Tied line velocity offsets
- `tied_fwhms::Dict{String, Array{Float64, 2}}`: Tied line velocity fwhms
- `acomp_tied_voffs::Dict{String, Array{Float64, 2}}`: additional component tied velocity offsets
- `acomp_tied_fwhms::Dict{String, Array{Float64, 2}}`: additional component tied velocity fwhms
- `tied_voigt_mix::Union{Array{Float64, 2}, Nothing}`: Tied Voigt mixing parameter
- `extinction::Dict{Symbol, Array{Float64, 2}}`: Extinction parameters: optical depth at 9.7 μm and mixing ratio
- `hot_dust::Dict{Symbol, Array{Float64, 2}}`: Hot dust parameters: amplitude, temperature, covering fraction, warm tau, and cold tau
- `reduced_χ2::Array{Float64, 2}`: The reduced chi^2 value of each fit

See ['parammaps_empty`](@ref) for a default constructor function.
"""
struct ParamMaps{T<:Real}

    stellar_continuum::Dict{Symbol, Array{T, 2}}
    dust_continuum::Dict{Int, Dict{Symbol, Array{T, 2}}}
    dust_features::Dict{String, Dict{Symbol, Array{T, 2}}}
    lines::Dict{Symbol, Dict{Symbol, Array{T, 2}}}
    tied_voffs::Dict{Symbol, Array{T, 2}}
    tied_fwhms::Dict{Symbol, Array{T, 2}}
    tied_voigt_mix::Union{Array{T, 2}, Nothing}
    extinction::Dict{Symbol, Array{T, 2}}
    hot_dust::Dict{Symbol, Array{T, 2}}
    reduced_χ2::Array{T, 2}

end


"""
    parammaps_empty(shape, n_dust_cont, df_names, line_names, line_tied, line_profiles,
        line_acomp_tied, line_acomp_profiles, kin_tied_key, acomp_kin_tied_key, flexible_wavesol, 
        tie_voigt_mixing)

A constructor function for making a default empty ParamMaps structure with all necessary fields for a given
fit of a DataCube.
"""
function parammaps_empty(shape::Tuple{S,S,S}, n_dust_cont::Integer, df_names::Vector{String}, 
    n_lines::S, n_comps::S, line_names::Vector{Symbol}, line_tied::Matrix{Union{Symbol,Nothing}}, 
    line_profiles::Matrix{Union{Symbol,Nothing}}, kin_tied_key::Vector{Vector{Symbol}}, flexible_wavesol::Bool, 
    tie_voigt_mixing::Bool)::ParamMaps where {S<:Integer}

    @debug """\n
    Creating ParamMaps struct with shape $shape
    ###########################################
    """

    # Initialize a default array of nans to be used as a placeholder for all the other arrays
    # until the actual fitting parameters are obtained
    nan_arr = ones(shape[1:2]...) .* NaN

    # Add stellar continuum fitting parameters
    stellar_continuum = Dict{Symbol, Array{Float64, 2}}()
    stellar_continuum[:amp] = copy(nan_arr)
    stellar_continuum[:temp] = copy(nan_arr)
    @debug "stellar continuum maps with keys $(keys(stellar_continuum))"

    # Add dust continuum fitting parameters
    dust_continuum = Dict{Int, Dict{Symbol, Array{Float64, 2}}}()
    for i ∈ 1:n_dust_cont
        dust_continuum[i] = Dict{Symbol, Array{Float64, 2}}()
        dust_continuum[i][:amp] = copy(nan_arr)
        dust_continuum[i][:temp] = copy(nan_arr)
        @debug "dust continuum $i maps with keys $(keys(dust_continuum[i]))"
    end

    # Add dust features fitting parameters
    dust_features = Dict{String, Dict{Symbol, Array{Float64, 2}}}()
    for n ∈ df_names
        dust_features[n] = Dict{Symbol, Array{Float64, 2}}()
        dust_features[n][:amp] = copy(nan_arr)
        dust_features[n][:mean] = copy(nan_arr)
        dust_features[n][:fwhm] = copy(nan_arr)
        dust_features[n][:flux] = copy(nan_arr)
        dust_features[n][:eqw] = copy(nan_arr)
        dust_features[n][:SNR] = copy(nan_arr)
        @debug "dust feature $n maps with keys $(keys(dust_features[n]))"
    end

    # Nested dictionary -> first layer keys are line names, second layer keys are parameter names, which contain 2D arrays
    lines = Dict{Symbol, Dict{Symbol, Array{Float64, 2}}}()
    for i ∈ 1:n_lines

        line = line_names[i]
        lines[line] = Dict{Symbol, Array{Float64, 2}}()

        # If tied and NOT using a flexible solution, don't include a voff parameter
        if isnothing(line_tied[i, 1])
            pnames = [:amp, :voff, :fwhm]
        elseif flexible_wavesol
            pnames = [:amp, :voff]
        else
            pnames = [:amp]
        end
        # Add 3rd and 4th order moments (skewness and kurtosis) for Gauss-Hermite profiles
        if line_profiles[i, 1] == :GaussHermite
            pnames = [pnames; :h3; :h4]
        # Add mixing parameter for Voigt profiles, but only if NOT tying it
        elseif line_profiles[i, 1] == :Voigt && !tie_voigt_mixing
            pnames = [pnames; :mixing]
        end
        # Repeat the above but for additional components
        for j ∈ 2:n_comps
            if !isnothing(line_profiles[i, j])
                pnames = isnothing(line_tied[i, j]) ? [pnames; Symbol("acomp_amp", "_$j"); Symbol("acomp_voff", "_$j"); 
                    Symbol("acomp_fwhm", "_$j")] : [pnames; Symbol("acomp_amp", "_$j")]
                if line_profiles[i, j] == :GaussHermite
                    pnames = [pnames; Symbol("acomp_h3", "_$j"); Symbol("acomp_h4", "_$j")]
                elseif line_profiles[i, j] == :Voigt && !tie_voigt_mixing
                    pnames = [pnames; Symbol("acomp_mixing", "_$j")]
                end
            end
        end
        # Append parameters for flux, equivalent width, and signal-to-noise ratio, which are NOT fitting parameters, but are of interest
        pnames = [pnames; :flux; :eqw; :SNR]
        for j ∈ 2:n_comps
            if !isnothing(line_profiles[i, j])
                pnames = [pnames; Symbol("acomp_flux", "_$j"); Symbol("acomp_eqw", "_$j"); Symbol("acomp_SNR", "_$j")]
            end
        end
        for pname ∈ pnames
            lines[line][pname] = copy(nan_arr)
        end
        @debug "line $line maps with keys $pnames"
    end

    # Tied voff parameters
    tied_voffs = Dict{Symbol, Array{Float64, 2}}()
    for j ∈ 1:n_comps
        for vk ∈ kin_tied_key[j]
            vkj = Symbol(vk, "_$j")
            tied_voffs[vkj] = copy(nan_arr)
            @debug "tied voff map for group $vkj"
        end
    end
    # Tied fwhm parameters
    tied_fwhms = Dict{Symbol, Array{Float64, 2}}()
    for j ∈ 1:n_comps
        for vk ∈ kin_tied_key[j]
            vkj = Symbol(vk, "_$j")
            tied_fwhms[vkj] = copy(nan_arr)
            @debug "tied fwhm map for group $vkj"
        end
    end

    # Tied voigt mixing ratio parameter, if appropriate
    if tie_voigt_mixing
        tied_voigt_mix = copy(nan_arr)
        @debug "tied voigt mixing map"
    else
        tied_voigt_mix = nothing
    end

    # Add extinction fitting parameters
    extinction = Dict{Symbol, Array{Float64, 2}}()
    extinction[:tau_9_7] = copy(nan_arr)
    extinction[:tau_ice] = copy(nan_arr)
    extinction[:tau_ch] = copy(nan_arr)
    extinction[:beta] = copy(nan_arr)
    @debug "extinction maps with keys $(keys(extinction))"

    # Add hot dust fitting parameters
    hot_dust = Dict{Symbol, Array{Float64, 2}}()
    hot_dust[:amp] = copy(nan_arr)
    hot_dust[:temp] = copy(nan_arr)
    hot_dust[:frac] = copy(nan_arr)
    hot_dust[:tau_warm] = copy(nan_arr)
    hot_dust[:tau_cold] = copy(nan_arr)
    @debug "hot dust maps with keys $(keys(hot_dust))"

    # Reduced chi^2 of the fits
    reduced_χ2 = copy(nan_arr)
    @debug "reduced chi^2 map"

    ParamMaps{Float64}(stellar_continuum, dust_continuum, dust_features, lines, tied_voffs, tied_fwhms, 
        tied_voigt_mix, extinction, hot_dust, reduced_χ2)
end


"""
    CubeModel(model, stellar, dust_continuum, dust_features, extinction, hot_dust, lines)

A structure for holding 3D models of intensity, split up into model components, generated when fitting a cube.
This will be the same shape as the input data, and preferably the same datatype too (i.e., JWST files have flux
and error in Float32 format, so we should also output in Float32 format).  This is useful as a way to quickly
compare the full model, or model components, to the data.

# Fields
- `model::Array{T, 3}`: The full 3D model.
- `stellar::Array{T, 3}`: The stellar component of the continuum.
- `dust_continuum::Array{T, 4}`: The dust components of the continuum. The 4th axis runs over each individual dust component.
- `dust_features::Array{T, 4}`: The dust (PAH) feature profiles. The 4th axis runs over each individual dust profile.
- `extinction::Array{T, 3}`: The extinction profile.
- `hot_dust::Array{T, 3}`: The hot dust emission profile
- `lines::Array{T, 4}`: The line profiles. The 4th axis runs over each individual line.

See [`cubemodel_empty`](@ref) for a default constructor method.
"""
struct CubeModel{T<:Real}

    model::Array{T, 3}
    stellar::Array{T, 3}
    dust_continuum::Array{T, 4}
    dust_features::Array{T, 4}
    extinction::Array{T, 3}
    abs_ice::Array{T, 3}
    abs_ch::Array{T, 3}
    hot_dust::Array{T, 3}
    lines::Array{T, 4}

end


"""
    cubemodel_empty(shape, n_dust_cont, df_names, line_names; floattype=floattype)

A constructor function for making a default empty CubeModel object with all the necessary fields for a given
fit of a DataCube.

# Arguments
`S<:Integer`
- `shape::Tuple{S,S,S}`: The dimensions of the DataCube being fit, formatted as a tuple of (nx, ny, nz)
- `n_dust_cont::Integer`: The number of dust continuum components in the fit (usually given by the number of temperatures 
    specified in the dust.toml file)
- `df_names::Vector{String}`: List of names of PAH features being fit, i.e. "PAH_12.62", ...
- `line_names::Vector{Symbol}`: List of names of lines being fit, i.e. "NeVI_7652", ...
- `floattype::DataType=Float32`: The type of float to use in the arrays. Should ideally be the same as the input data,
    which for JWST is Float32.
"""
function cubemodel_empty(shape::Tuple{S,S,S}, n_dust_cont::Integer, df_names::Vector{String}, 
    line_names::Vector{Symbol}, floattype::DataType=Float32)::CubeModel where {S<:Integer}

    @debug """\n
    Creating CubeModel struct with shape $shape
    ###########################################
    """

    # Make sure the floattype given is actually a type of float
    @assert floattype <: AbstractFloat "floattype must be a type of AbstractFloat (Float32 or Float64)!"

    # Initialize the arrays for each part of the full 3D model
    model = zeros(floattype, shape...)
    @debug "model cube"
    stellar = zeros(floattype, shape...)
    @debug "stellar continuum comp cube"
    dust_continuum = zeros(floattype, shape..., n_dust_cont)
    @debug "dust continuum comp cubes"
    dust_features = zeros(floattype, shape..., length(df_names))
    @debug "dust features comp cubes"
    extinction = zeros(floattype, shape...)
    @debug "extinction comp cube"
    abs_ice = zeros(floattype, shape...)
    @debug "abs_ice comp cube"
    abs_ch = zeros(floattype, shape...)
    @debug "abs_ch comp cube"
    hot_dust = zeros(floattype, shape...)
    @debug "hot dust comp cube"
    lines = zeros(floattype, shape..., length(line_names))
    @debug "lines comp cubes"

    CubeModel(model, stellar, dust_continuum, dust_features, extinction, abs_ice, abs_ch, hot_dust, lines)
end


"""
    CubeFitter(cube, z, name; plot_spaxels=plot_spaxels, 
        plot_maps=plot_maps, parallel=parallel, save_fits=save_fits)

This is the main structure used for fitting IFU cubes, containing all of the necessary data, metadata,
fitting options, and associated functions for generating ParamMaps and CubeModel structures to handle the outputs 
of all the fits.  This is essentially the "configuration object" that tells the rest of the fitting code how
to run. The actual fitting functions (`fit_spaxel` and `fit_cube!`) require an instance of this structure.

# Fields
`T<:Real, S<:Integer`
- `cube::DataCube`: The main DataCube object containing the cube that is being fit
- `z::Real`: The redshift of the target that is being fit
- `plot_spaxels::Symbol=:pyplot`: A Symbol specifying the plotting backend to be used when plotting individual spaxel fits, can
    be either `:pyplot` or `:plotly`
- `plot_maps::Bool=true`: Whether or not to plot 2D maps of the best-fit parameters after the fitting is finished
- `parallel::Bool=true`: Whether or not to fit multiple spaxels in parallel using multiprocessing
- `save_fits::Bool=true`: Whether or not to save the final best-fit models and parameters as FITS files
Read from the options files:
- `overwrite::Bool`: Whether or not to overwrite old fits of spaxels when rerunning
- `track_memory::Bool`: Whether or not to save diagnostic files showing memory usage of the program
- `track_convergence::Bool`: Whether or not to save diagnostic files showing convergence of line fitting for each spaxel
- `make_movies::Bool`: Whether or not to save mp4 files of the final model
- `extinction_curve::String`: The type of extinction curve being used, either `"kvt"` or `"d+"`
- `extinction_screen::Bool`: Whether or not the extinction is modeled as a screen
- `T_s::Parameter`: The stellar temperature parameter
- `T_dc::Vector{Parameter}`: The dust continuum temperature parameters
- `τ_97::Parameter`: The dust opacity at 9.7 um parameter
- `τ_ice::Parameter`: The peak opacity from ice absorption (at around 6.9 um)
- `τ_ch::Parameter`: The peak opacity from CH absorption (at around 6.9 um)
- `β::Parameter`: The extinction profile mixing parameter
- `T_hot::Parameter`: The hot dust temperature
- `Cf_hot::Parameter`: The hot dust covering fraction
- `τ_warm::Parameter`: The warm dust optical depth
- `τ_cold::Parameter`: The cold dust optical depth
- `n_dust_cont::Integer`: The number of dust continuum profiles
- `df_names::Vector{String}`: The names of each PAH feature profile
- `dust_features::Vector{Dict}`: All of the fitting parameters for each PAH feature
- `n_lines::Integer`: The number of lines being fit
- `line_names::Vector{Symbol}`: The names of each line being fit
- `line_profiles::Vector{Symbol}`: The profiles of each line being fit
- `line_acomp_profiles::Vector{Union{Nothing,Symbol}}`: Same as `line_profiles`, but for the additional components
- `lines::Vector{TransitionLine}`: All of the fitting parameters for each line
- `n_kin_tied::Integer`: The number of tied velocity offsets
- `line_tied::Vector{Union{String,Nothing}}`: List of line tie keys which specify whether the voff of the given line should be
    tied to other lines. The line tie key itself may be either `nothing` (untied), or a String specifying the group of lines
    being tied, i.e. "H2"
- `kin_tied_key::Vector{String}`: List of only the unique keys in line_tied (not including `nothing`)
- `voff_tied::Vector{Parameter}`: The actual tied voff parameter objects, corresponding to the `kin_tied_key`
- `fwhm_tied::Vector{Parameter}`: The actual tied fwhm parameter objects, corresponding to the `kin_tied_key`
- `n_acomp_kin_tied::Integer`: Same as `n_kin_tied`, but for additional components
- `line_acomp_tied::Vector{Union{String,Nothing}}`: Same as `line_tied`, but for additional components
- `acomp_kin_tied_key::Vector{String}`: Same as `kin_tied_key`, but for additional components
- `acomp_voff_tied::Vector{Parameter}`: Same as `voff_tied`, but for additional components
- `acomp_fwhm_tied::Vector{Parameter}`: Same as `fwhm_tied`, but for additional components
- `tie_voigt_mixing::Bool`: Whether or not the Voigt mixing parameter is tied between all the lines with Voigt profiles
- `voigt_mix_tied::Parameter`: The actual tied Voigt mixing parameter object, given `tie_voigt_mixing` is true
- `n_params_cont::Integer`: The total number of free fitting parameters for the continuum fit (not including emission lines)
- `n_params_lines::Integer`: The total number of free fitting parameters for the emission line fit (not including the continuum)
- `cosmology::Cosmology.AbstractCosmology`: The Cosmology, used solely to create physical scale bars on the 2D parameter plots
- `χ²_thresh::Real`: The threshold for reduced χ² values, below which the best fit parameters for a given
    row will be set
- `interp_R::Function`: Interpolation function for the instrumental resolution as a function of wavelength
- `flexible_wavesol::Bool`: Whether or not to allow small variations in the velocity offsets even when tied, to account
    for a bad wavelength solution
- `p_best_cont::SharedArray{T}`: A rolling collection of best fit continuum parameters for the best fitting spaxels
    along each row, for fits with a reduced χ² below χ²_thresh, which are used for the starting parameters in the following
    fits for the given row
- `p_best_line::SharedArray{T}`: Same as `p_best_cont`, but for the line parameters
- `χ²_best::SharedVector{T}`: The reduced χ² values associated with the `p_best_cont` and `p_best_line` values
    in each row
- `best_spaxel::SharedVector{Tuple{S,S}}`: The locations of the spaxels associated with the `p_best_cont` and `p_best_line`
    values in each row

See [`ParamMaps`](@ref), [`parammaps_empty`](@ref), [`CubeModel`](@ref), [`cubemodel_empty`](@ref), 
    [`fit_spaxel`](@ref), [`fit_cube!`](@ref)
"""
struct CubeFitter{T<:Real,S<:Integer}

    # See explanations for each field in the docstring!
    
    # Data
    cube::DataCube
    z::T
    name::String

    # Basic fitting options
    plot_spaxels::Symbol
    plot_maps::Bool
    plot_range::Union{Vector{<:Tuple},Nothing}
    parallel::Bool
    save_fits::Bool
    save_full_model::Bool
    subtract_cubic::Bool
    overwrite::Bool
    track_memory::Bool
    track_convergence::Bool
    make_movies::Bool
    extinction_curve::String
    extinction_screen::Bool
    fit_sil_emission::Bool
    fit_all_samin::Bool

    # Continuum parameters
    continuum::Continuum

    # Dust Feature parameters
    n_dust_cont::S
    n_dust_feat::S
    dust_features::DustFeatures

    # Line parameters
    n_lines::S
    n_acomps::S
    n_comps::S
    lines::TransitionLines

    # Tied voffs
    n_kin_tied::Vector{S}
    tied_kinematics::TiedKinematics

    # Tied voigt mixing
    tie_voigt_mixing::Bool
    voigt_mix_tied::Parameter

    # Number of parameters
    n_params_cont::S
    n_params_lines::S
    n_params_extra::S
    
    # General options
    cosmology::Cosmology.AbstractCosmology
    interp_R::Function
    flexible_wavesol::Bool

    p_init_cont::Vector{T}
    p_init_line::Vector{T}

    #= Constructor function --> the inputs taken map directly to fields in the CubeFitter object,
    the rest of the fields are generated in the function from these inputs =#
    function CubeFitter(cube::DataCube, z::Real, name::String; plot_spaxels::Symbol=:pyplot, plot_maps::Bool=true, 
        plot_range::Union{Vector{<:Tuple},Nothing}=nothing, parallel::Bool=true, save_fits::Bool=true) 

        # Prepare output directories
        @info "Preparing output directories"
        name = replace(name, #= no spaces! =# " " => "_")

        # Top-level output directory
        if !isdir("output_$name")
            mkdir("output_$name")
        end
        # Sub-folder for 1D plots of spaxel fits
        if !isdir(joinpath("output_$name", "spaxel_plots"))
            mkdir(joinpath("output_$name", "spaxel_plots"))
        end
        if !isdir(joinpath("output_$name", "zoomed_plots")) && !isnothing(plot_range)
            mkdir(joinpath("output_$name", "zoomed_plots"))
        end
        # Sub-folder for data files saving the results of individual spaxel fits
        if !isdir(joinpath("output_$name", "spaxel_binaries"))
            mkdir(joinpath("output_$name", "spaxel_binaries"))
        end
        # Sub-folder for 2D parameter maps 
        if !isdir(joinpath("output_$name", "param_maps"))
            mkdir(joinpath("output_$name", "param_maps"))
        end
        # Sub-folder for log files
        if !isdir(joinpath("output_$name", "logs"))
            mkdir(joinpath("output_$name", "logs"))
        end

        #############################################################

        @debug """\n
        Creating CubeFitter struct for $name at z=$z
        ############################################
        """
        # Alias
        λ = cube.λ

        # Parse all of the options files to create default options and parameter objects
        interp_R = parse_resolving(cube.channel)
        # Get the limiting value of the instrumental FWHM
        fwhm_inst = C_KMS / maximum(interp_R.(λ .* (1 .+ z)))

        continuum, dust_features = parse_dust()
        options = parse_options()
        lines, tied_kinematics, flexible_wavesol, tie_voigt_mixing, voigt_mix_tied = parse_lines(fwhm_inst)

        @debug "### Model will include 1 stellar continuum component ###" *
             "\n### at T = $(continuum.T_s.value) K ###"

        #### PREPARE OUTPUTS ####
        n_dust_cont = length(continuum.T_dc)
        msg = "### Model will include $n_dust_cont dust continuum components ###"
        for T_dci ∈ continuum.T_dc
            msg *= "\n### at T = $(T_dci.value) K ###"
        end
        @debug msg 

        # Only use PAH features within +/-0.5 um of the region being fitting (to include wide tails)
        df_filt = [(minimum(λ)-0.5 < dust_features.mean[i].value < maximum(λ)+0.5) for i ∈ 1:length(dust_features.mean)]
        dust_features = DustFeatures(dust_features.names[df_filt], 
                                     dust_features.profiles[df_filt],
                                     dust_features.mean[df_filt],
                                     dust_features.fwhm[df_filt])
        n_dust_features = length(dust_features.names)
        msg = "### Model will include $n_dust_features dust feature (PAH) components ###"
        for df_mn ∈ dust_features.mean
            msg *= "\n### at lambda = $df_mn um ###"
        end
        @debug msg

        # Only use lines within the wavelength range being fit
        ln_filt = minimum(λ) .< lines.λ₀ .< maximum(λ)
        # Convert to a vectorized "TransitionLines" object
        lines = TransitionLines(lines.names[ln_filt], lines.λ₀[ln_filt], lines.profiles[ln_filt, :],
                                lines.tied[ln_filt, :], lines.voff[ln_filt, :], lines.fwhm[ln_filt, :],
                                lines.h3[ln_filt, :], lines.h4[ln_filt, :], lines.η[ln_filt, :])
        n_lines = length(lines.names)
        n_comps = size(lines.profiles, 2)
        n_acomps = sum(.!isnothing.(lines.profiles[:, 2:end]))
        msg = "### Model will include $n_lines emission lines ###"
        for (k, (name, λ0, prof)) ∈ enumerate(zip(lines.names, lines.λ₀, lines.profiles))
            msg *= "\n### $name at lambda = $λ0 um with $prof profile ###"
            for acomp_prof ∈ lines.profiles[k, 2:end]
                if !isnothing(acomp_prof)
                    msg *= "\n###   and $acomp_prof acomp profile   ###"
                end
            end
        end
        @debug msg

        # Remove unnecessary rows/keys from the tied_kinematics object after the lines have been filtered
        @debug "TiedKinematics before filtering: $tied_kinematics"
        for j ∈ 1:n_comps
            keep = Int64[]
            for (k, key) ∈ enumerate(tied_kinematics.key[j])
                if any(lines.tied[:, j] .== key)
                    # Remove the unneeded elements
                    append!(keep, [k])
                end
            end
            tied_kinematics.key[j] = tied_kinematics.key[j][keep]
            tied_kinematics.voff[j] = tied_kinematics.voff[j][keep]
            tied_kinematics.fwhm[j] = tied_kinematics.fwhm[j][keep]
        end
        @debug "TiedKinematics after filtering: $tied_kinematics"

        # Number of tied kinematic parameters for each line component
        n_kin_tied = [length(tied_kinematics.key[j]) for j ∈ 1:n_comps]

        # Also store the "tied" parameter for each line, which will need to be checked against the kin_tied_key
        # during fitting to find the proper location of the tied voff parameter to use
        msg = "### Model will include $(n_kin_tied) tied kinematics parameters for each component ###"
        for lt ∈ tied_kinematics.key
            msg *= "\n### for group $lt ###"
        end
        @debug msg

        # Total number of parameters for the continuum and line fits
        n_params_cont = (2+4) + 2n_dust_cont + 3n_dust_features + (options[:fit_sil_emission] ? 5 : 0)
        n_params_lines = 2 * sum(n_kin_tied)
        # One η for all voigt profiles
        if any(lines.profiles .== :Voigt) && tie_voigt_mixing
            n_params_lines += 1
            @debug "### Model will include 1 tied voigt mixing parameter ###"
        end
        for i ∈ 1:n_lines
            if isnothing(lines.tied[i, 1])
                # amplitude, voff, and FWHM parameters
                n_params_lines += 3
            elseif flexible_wavesol
                # amplitude and voff_individual parameters
                n_params_lines += 2
            else
                # no voff or FWHM parameter, since they're tied
                n_params_lines += 1
            end
            if lines.profiles[i, 1] == :GaussHermite
                # extra h3 and h4 parmeters
                n_params_lines += 2
            elseif lines.profiles[i, 1] == :Voigt
                # extra mixing parameter, but only if it's not tied
                if !tie_voigt_mixing
                    n_params_lines += 1
                end
            end
            # Repeat above for the additional components
            for j ∈ 2:n_comps
                if !isnothing(lines.profiles[i, j])
                    if isnothing(lines.tied[i, j])
                        n_params_lines += 3
                    else
                        n_params_lines += 1
                    end
                    if lines.profiles[i, j] == :GaussHermite
                        n_params_lines += 2
                    elseif lines.profiles[i, j] == :Voigt
                        if !tie_voigt_mixing
                            n_params_lines += 1
                        end
                    end
                end
            end
        end
        n_params_extra = 3 * (n_dust_features + n_lines + n_acomps)
        @debug "### This totals to $(n_params_cont) continuum parameters ###"
        @debug "### This totals to $(n_params_lines) emission line parameters ###"
        @debug "### This totals to $(n_params_extra) extra parameters ###"

        # Prepare options
        extinction_curve = options[:extinction_curve]
        extinction_screen = options[:extinction_screen]
        fit_sil_emission = options[:fit_sil_emission]
        fit_all_samin = options[:fit_all_samin]
        subtract_cubic = options[:subtract_cubic]
        overwrite = options[:overwrite]
        track_memory = options[:track_memory]
        track_convergence = options[:track_convergence]
        save_full_model = options[:save_full_model]
        make_movies = options[:make_movies]
        cosmo = options[:cosmology]

        # Prepare initial best fit parameter options
        @debug "Preparing initial best fit parameter vectors with $(n_params_cont+2) and $(n_params_lines) parameters"
        p_init_cont = zeros(n_params_cont+2)
        p_init_line = zeros(n_params_lines)

        # If a fit has been run previously, read in the file containing the rolling best fit parameters
        # to pick up where the fitter left off seamlessly
        if isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv")) && isfile(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"))
            p_init_cont = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_cont.csv"), ',', Float64, '\n')[:, 1]
            p_init_line = readdlm(joinpath("output_$name", "spaxel_binaries", "init_fit_line.csv"), ',', Float64, '\n')[:, 1]
        end

        new{typeof(z), typeof(n_lines)}(cube, z, name, plot_spaxels, plot_maps, plot_range, parallel, 
            save_fits, save_full_model, subtract_cubic, overwrite, track_memory, track_convergence, make_movies, extinction_curve, 
            extinction_screen, fit_sil_emission, continuum, n_dust_cont, n_dust_features, dust_features, n_lines, n_acomps, n_comps,
            lines, n_kin_tied, tied_kinematics, tie_voigt_mixing, voigt_mix_tied, n_params_cont, n_params_lines, 
            n_params_extra, cosmo, interp_R, flexible_wavesol, p_init_cont, p_init_line)
    end

end


"""
    generate_cubemodel(cube_fitter)

Generate a CubeModel object corresponding to the options given by the CubeFitter object
"""
function generate_cubemodel(cube_fitter::CubeFitter)::CubeModel
    shape = size(cube_fitter.cube.Iν)
    # Full 3D intensity model array
    @debug "Generating full 3D cube models"
    cubemodel_empty(shape, cube_fitter.n_dust_cont, cube_fitter.dust_features.names, cube_fitter.lines.names)
end


"""
    generate_parammaps(cube_fitter)

Generate two ParamMaps objects (for the values and errors) corrresponding to the options given
by the CubeFitter object
"""
function generate_parammaps(cube_fitter::CubeFitter)::Tuple{ParamMaps, ParamMaps}
    shape = size(cube_fitter.cube.Iν)
    # 2D maps of fitting parameters
    @debug "Generating 2D parameter value & error maps"
    param_maps = parammaps_empty(shape, cube_fitter.n_dust_cont, cube_fitter.dust_features.names, cube_fitter.n_lines,
                                 cube_fitter.n_comps, cube_fitter.lines.names, cube_fitter.lines.tied, cube_fitter.lines.profiles, 
                                 cube_fitter.tied_kinematics.key, cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing)
    # 2D maps of fitting parameter 1-sigma errors
    param_errs = parammaps_empty(shape, cube_fitter.n_dust_cont, cube_fitter.dust_features.names, cube_fitter.n_lines,
                                 cube_fitter.n_comps, cube_fitter.lines.names, cube_fitter.lines.tied, cube_fitter.lines.profiles, 
                                 cube_fitter.tied_kinematics.key, cube_fitter.flexible_wavesol, cube_fitter.tie_voigt_mixing)
    param_maps, param_errs
end


"""
    get_continuum_priors(cube_fitter, I)

Get the continuum prior vector for a given CubeFitter object, split up by the 2 continuum fitting steps.
Also returns a boolean vector for which parameters are allowed to vary.
"""
function get_continuum_priors(cube_fitter::CubeFitter, I::Vector{<:Real}=[Inf])

    dust_features = cube_fitter.dust_features
    continuum = cube_fitter.continuum

    amp_dc_prior = Uniform(0., Inf)  # dont actually use this for getting pdfs or logpdfs, it's just for min/max
    amp_df_prior = Uniform(0., clamp(nanmaximum(I) / exp(-maximum(continuum.τ_97.prior)), 1., Inf))

    stellar_priors = [amp_dc_prior, continuum.T_s.prior]
    stellar_lock = [false, continuum.T_s.locked]
    dc_priors = vcat([[amp_dc_prior, Ti.prior] for Ti ∈ continuum.T_dc]...)
    dc_lock = vcat([[false, Ti.locked] for Ti ∈ continuum.T_dc]...)
    df_priors = vcat([[amp_df_prior, mi.prior, fi.prior] for (mi, fi) ∈ zip(dust_features.mean, dust_features.fwhm)]...)
    df_lock = vcat([[false, mi.locked, fi.locked] for (mi, fi) ∈ zip(dust_features.mean, dust_features.fwhm)]...)
    ext_priors = [continuum.τ_97.prior, continuum.τ_ice.prior, continuum.τ_ch.prior, continuum.β.prior]
    ext_lock = [continuum.τ_97.locked, continuum.τ_ice.locked, continuum.τ_ch.locked, continuum.β.locked]
    hd_priors = cube_fitter.fit_sil_emission ? [amp_dc_prior, continuum.T_hot.prior, continuum.Cf_hot.prior, 
        continuum.τ_warm.prior, continuum.τ_cold.prior] : []
    hd_lock = cube_fitter.fit_sil_emission ? [false, continuum.T_hot.locked, continuum.Cf_hot.locked,
        continuum.τ_warm.locked, continuum.τ_cold.locked] : []

    # Split up for the two different stages of continuum fitting -- with templates and then with the PAHs
    priors_1 = Vector{Distribution}(vcat(stellar_priors, dc_priors, ext_priors, hd_priors, [amp_df_prior, amp_df_prior]))
    lock_1 = BitVector(vcat(stellar_lock, dc_lock, ext_lock, hd_lock, [false, false]))
    priors_2 = Vector{Distribution}(df_priors)
    lock_2 = BitVector(df_lock)

    priors_1, priors_2, lock_1, lock_2

end


"""
    get_continuum_initial_values(cube_fitter, I, init)

Get the vector of starting values for the continuum fit for a given CubeFitter object. Again, the
vector is split up by the 2 continuum fitting steps.
"""
function get_continuum_initial_values(cube_fitter::CubeFitter, λ::Vector{<:Real}, I::Vector{<:Real}, init::Bool)

    # Check if the cube fitter has initial fit parameters 
    if !init

        @debug "Using initial best fit continuum parameters..."

        # Set the parameters to the best parameters
        p₀ = copy(cube_fitter.p_init_cont)[1:end-2]
        pah_frac = copy(cube_fitter.p_init_cont)[end-1:end]

        # pull out optical depth that was pre-fit
        # τ_97_0 = cube_fitter.τ_guess[parse(Int, cube_fitter.cube.channel)][spaxel]
        max_τ = maximum(cube_fitter.continuum.τ_97.prior)

        # scale all flux amplitudes by the difference in medians between the spaxel and the summed spaxels
        # (should be close to 1 since the sum is already normalized by the number of spaxels included anyways)
        scale = max(nanmedian(I), 1e-10) / nanmedian(sumdim(cube_fitter.cube.Iν, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2)))
        max_amp = nanmaximum(I) / exp(-max_τ)

        # PAH template strengths
        pah_frac .*= scale
        
        # Stellar amplitude
        p₀[1] *= scale
        pᵢ = 3

        # Dust continuum amplitudes
        for i ∈ 1:cube_fitter.n_dust_cont
            p₀[pᵢ] *= scale
            pᵢ += 2
        end

        # Set optical depth based on the pre-fitting
        # p₀[pᵢ] = τ_97_0
        pᵢ += 4

        if cube_fitter.fit_sil_emission
            # Hot dust amplitude
            p₀[pᵢ] *= scale
            # Warm / cold optical depths
            # p₀[pᵢ+3] = τ_97_0
            # p₀[pᵢ+4] = τ_97_0
            pᵢ += 5
        end

        # Dust feature amplitudes
        for i ∈ 1:cube_fitter.n_dust_feat
            p₀[pᵢ] = clamp(p₀[pᵢ]*scale, 0., max_amp)
            pᵢ += 3
        end

    # Otherwise, we estimate the initial parameters based on the data
    else

        continuum = cube_fitter.continuum

        @debug "Calculating initial starting points..."
        pah_frac = repeat([clamp(nanmedian(I)/2, 0., Inf)], 2)

        # Stellar amplitude
        A_s = clamp(quadratic_interp_func(5.5, λ, I) / Blackbody_ν(5.5, continuum.T_s.value), 0., Inf) 

        # Dust feature amplitudes
        A_df = repeat([clamp(nanmedian(I)/2, 0., Inf)], cube_fitter.n_dust_feat)

        # Dust continuum amplitudes
        λ_dc = clamp.([Wein(Ti.value) for Ti ∈ continuum.T_dc], minimum(λ), maximum(λ))
        A_dc = clamp.([quadratic_interp_func(λ_dci, λ, I) / Blackbody_ν(λ_dci, T_dci.value) for (λ_dci, T_dci) ∈ 
            zip(λ_dc, continuum.T_dc)] .* (λ_dc ./ 9.7).^2 ./ 5., 0., Inf)
        
        # Hot dust amplitude
        A_hd = clamp(quadratic_interp_func(5.5, λ, I) / Blackbody_ν(5.5, continuum.T_hot.value), 0., Inf) / 2

        stellar_pars = [A_s, continuum.T_s.value]
        dc_pars = vcat([[Ai, Ti.value] for (Ai, Ti) ∈ zip(A_dc, continuum.T_dc)]...)
        df_pars = vcat([[Ai, mi.value, fi.value] for (Ai, mi, fi) ∈ zip(A_df, cube_fitter.dust_features.mean, cube_fitter.dust_features.fwhm)]...)
        if cube_fitter.fit_sil_emission
            hd_pars = [A_hd, continuum.T_hot.value, continuum.Cf_hot.value, continuum.τ_warm.value, continuum.τ_cold.value]
        else
            hd_pars = []
        end

        extinction_pars = [continuum.τ_97.value, continuum.τ_ice.value, continuum.τ_ch.value, continuum.β.value]

        # Initial parameter vector
        p₀ = Vector{Float64}(vcat(stellar_pars, dc_pars, extinction_pars, hd_pars, df_pars))

    end

    @debug "Continuum Parameter labels: \n [stellar_amp, stellar_temp, " * 
        join(["dust_continuum_amp_$i, dust_continuum_temp_$i" for i ∈ 1:cube_fitter.n_dust_cont], ", ") * 
        "extinction_tau_97, extinction_tau_ice, extinction_tau_ch, extinction_beta, " *  
        (cube_fitter.fit_sil_emission ? "hot_dust_amp, hot_dust_temp, hot_dust_covering_frac, hot_dust_tau, cold_dust_tau, " : "") *
        join(["$(df)_amp, $(df)_mean, $(df)_fwhm" for df ∈ cube_fitter.dust_features.names], ", ") * "]"
        
    # @debug "Priors: \n $priors"
    @debug "Continuum Starting Values: \n $p₀"

    # Step 1: Stellar + Dust blackbodies, 2 new amplitudes for the PAH templates, and the extinction parameters
    pars_1 = vcat(p₀[1:(2+2*cube_fitter.n_dust_cont+4+(cube_fitter.fit_sil_emission ? 5 : 0))], pah_frac)
    # Step 2: The PAH profile amplitudes, centers, and FWHMs
    pars_2 = p₀[(3+2*cube_fitter.n_dust_cont+4+(cube_fitter.fit_sil_emission ? 5 : 0)):end]

    pars_1, pars_2

end


"""
    get_continuum_parinfo(cube_fitter, pars_1, priors_1, lock_1, pars_2, priors_2, lock_2)

Get the CMPFit parinfo and config objects for a given CubeFitter object, given the vector of initial valuels,
priors, and boolean locked values.
"""
function get_continuum_parinfo(n_free_1::S, n_free_2::S, lb_1::Vector{T}, ub_1::Vector{T}, 
    lb_2::Vector{T}, ub_2::Vector{T}) where {S<:Integer,T<:Real}

    parinfo_1 = CMPFit.Parinfo(n_free_1)
    parinfo_2 = CMPFit.Parinfo(n_free_2)

    for pᵢ ∈ 1:n_free_1
        parinfo_1[pᵢ].fixed = 0
        parinfo_1[pᵢ].limited = (1,1)
        parinfo_1[pᵢ].limits = (lb_1[pᵢ], ub_1[pᵢ])
    end

    for pᵢ ∈ 1:n_free_2
        parinfo_2[pᵢ].fixed = 0
        parinfo_2[pᵢ].limited = (1,1)
        parinfo_2[pᵢ].limits = (lb_2[pᵢ], ub_2[pᵢ])
    end

    # Create a `config` structure
    config = CMPFit.Config()

    parinfo_1, parinfo_2, config

end


"""
    pretty_print_continuum_results(cube_fitter, popt, perr)

Print out a nicely formatted summary of the continuum fit results for a given CubeFitter object.
"""
function pretty_print_continuum_results(cube_fitter::CubeFitter, popt::Vector{<:Real}, perr::Vector{<:Real},
    I::Vector{<:Real})

    continuum = cube_fitter.continuum

    msg = "######################################################################\n"
    msg *= "################# SPAXEL FIT RESULTS -- CONTINUUM ####################\n"
    msg *= "######################################################################\n"
    msg *= "\n#> STELLAR CONTINUUM <#\n"
    msg *= "Stellar_amp: \t\t\t $(@sprintf "%.3e" popt[1]) +/- $(@sprintf "%.3e" perr[1]) MJy/sr \t Limits: (0, Inf)\n"
    msg *= "Stellar_temp: \t\t\t $(@sprintf "%.0f" popt[2]) +/- $(@sprintf "%.3e" perr[2]) K \t (fixed)\n"
    pᵢ = 3
    msg *= "\n#> DUST CONTINUUM <#\n"
    for i ∈ 1:cube_fitter.n_dust_cont
        msg *= "Dust_continuum_$(i)_amp: \t\t $(@sprintf "%.3e" popt[pᵢ]) +/- $(@sprintf "%.3e" perr[pᵢ]) MJy/sr \t Limits: (0, Inf)\n"
        msg *= "Dust_continuum_$(i)_temp: \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.3e" perr[pᵢ+1]) K \t\t\t (fixed)\n"
        msg *= "\n"
        pᵢ += 2
    end
    msg *= "\n#> EXTINCTION <#\n"
    msg *= "τ_9.7: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t Limits: " *
        "($(@sprintf "%.2f" minimum(continuum.τ_97.prior)), $(@sprintf "%.2f" maximum(continuum.τ_97.prior)))" * 
        (continuum.τ_97.locked ? " (fixed)" : "") * "\n"
    msg *= "τ_ice: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+1]) +/- $(@sprintf "%.2f" perr[pᵢ+1]) [-] \t Limits: " *
        "($(@sprintf "%.2f" minimum(continuum.τ_ice.prior)), $(@sprintf "%.2f" maximum(continuum.τ_ice.prior)))" *
        (continuum.τ_ice.locked ? " (fixed)" : "") * "\n"
    msg *= "τ_ch: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+2]) +/- $(@sprintf "%.2f" perr[pᵢ+2]) [-] \t Limits: " *
        "($(@sprintf "%.2f" minimum(continuum.τ_ch.prior)), $(@sprintf "%.2f" maximum(continuum.τ_ch.prior)))" *
        (continuum.τ_ch.locked ? " (fixed)" : "") * "\n"
    msg *= "β: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+3]) +/- $(@sprintf "%.2f" perr[pᵢ+3]) [-] \t Limits: " *
        "($(@sprintf "%.2f" minimum(continuum.β.prior)), $(@sprintf "%.2f" maximum(continuum.β.prior)))" * 
        (continuum.β.locked ? " (fixed)" : "") * "\n"
    msg *= "\n"
    pᵢ += 4
    if cube_fitter.fit_sil_emission
        msg *= "\n#> HOT DUST <#\n"
        msg *= "Hot_dust_amp: \t\t\t $(@sprintf "%.3e" popt[pᵢ]) +/- $(@sprintf "%.3e" perr[pᵢ]) MJy/sr \t Limits: (0, Inf)\n"
        msg *= "Hot_dust_temp: \t\t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) K \t Limits: " *
            "($(@sprintf "%.0f" minimum(continuum.T_hot.prior)), $(@sprintf "%.0f" maximum(continuum.T_hot.prior)))" *
            (continuum.T_hot.locked ? " (fixed)" : "") * "\n"
        msg *= "Hot_dust_frac: \t\t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) [-] \t Limits: " *
            "($(@sprintf "%.3f" minimum(continuum.Cf_hot.prior)), $(@sprintf "%.3f" maximum(continuum.Cf_hot.prior)))" *
            (continuum.Cf_hot.locked ? " (fixed)" : "") * "\n"
        msg *= "Hot_dust_τ: \t\t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3]) [-] \t Limits: " *
            "($(@sprintf "%.3f" minimum(continuum.τ_warm.prior)), $(@sprintf "%.3f" maximum(continuum.τ_warm.prior)))" *
            (continuum.τ_warm.locked ? " (fixed)" : "") * "\n"
        msg *= "Cold_dust_τ: \t\t\t $(@sprintf "%.3f" popt[pᵢ+4]) +/- $(@sprintf "%.3f" perr[pᵢ+4]) [-] \t Limits: " *
            "($(@sprintf "%.3f" minimum(continuum.τ_cold.prior)), $(@sprintf "%.3f" maximum(continuum.τ_cold.prior)))" *
            (continuum.τ_cold.locked ? " (fixed)" : "") * "\n"
        pᵢ += 5
    end
    msg *= "\n#> DUST FEATURES <#\n"
    for (j, df) ∈ enumerate(cube_fitter.dust_features.names)
        msg *= "$(df)_amp:\t\t\t $(@sprintf "%.1f" popt[pᵢ]) +/- $(@sprintf "%.1f" perr[pᵢ]) MJy/sr \t Limits: " *
            "(0, $(@sprintf "%.1f" (nanmaximum(I) / exp(-popt[end-1]))))\n"
        msg *= "$(df)_mean:  \t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1]) μm \t Limits: " *
            "($(@sprintf "%.3f" minimum(cube_fitter.dust_features.mean[j].prior)), $(@sprintf "%.3f" maximum(cube_fitter.dust_features.mean[j].prior)))" * 
            (cube_fitter.dust_features.mean[j].locked ? " (fixed)" : "") * "\n"
        msg *= "$(df)_fwhm:  \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) μm \t Limits: " *
            "($(@sprintf "%.3f" minimum(cube_fitter.dust_features.fwhm[j].prior)), $(@sprintf "%.3f" maximum(cube_fitter.dust_features.fwhm[j].prior)))" * 
            (cube_fitter.dust_features.fwhm[j].locked ? " (fixed)" : "") * "\n"
        msg *= "\n"
        pᵢ += 3
    end
    msg *= "######################################################################"
    @debug msg

    msg

end


"""
    get_line_priors(cube_fitter)

Get the line prior vector for a given CubeFitter object. Also returns boolean locked values and
names of each parameter as strings.
"""
function get_line_priors(cube_fitter::CubeFitter, init::Bool)

    # Set up the prior vector
    amp_ln_prior = Uniform(0., 1.)
    amp_acomp_prior = Uniform(0., 1.)
    ln_priors = Vector{Any}()
    ln_lock = Vector{Bool}()
    ln_names = Vector{String}()
    
    # Loop through each line and append the new components
    for i ∈ 1:cube_fitter.n_lines
        # name
        ln_name = cube_fitter.lines.names[i]
        # check if voff should be tied or untied
        if isnothing(cube_fitter.lines.tied[i, 1])
            # amplitude, voff, FWHM
            append!(ln_priors, [amp_ln_prior, cube_fitter.lines.voff[i, 1].prior, cube_fitter.lines.fwhm[i, 1].prior])
            append!(ln_lock, [false, cube_fitter.lines.voff[i, 1].locked, cube_fitter.lines.fwhm[i, 1].locked])
            append!(ln_names, ["$(ln_name)_amp", "$(ln_name)_voff", "$(ln_name)_fwhm"])
        elseif cube_fitter.flexible_wavesol
            # amplitude, voff (since FWHM is tied)
            append!(ln_priors, [amp_ln_prior, cube_fitter.lines.voff[i, 1].prior])
            append!(ln_lock, [false, cube_fitter.lines.voff[i, 1].locked])
            append!(ln_names, ["$(ln_name)_amp", "$(ln_name)_voff"])
        else
            # just amplitude (since voff & FWHM are tied)
            append!(ln_priors, [amp_ln_prior])
            append!(ln_lock, [false])
            append!(ln_names, ["$(ln_name)_amp"])
        end
        # check for additional profile parameters
        if cube_fitter.lines.profiles[i, 1] == :GaussHermite
            # add h3 and h4 moments
            append!(ln_priors, [cube_fitter.lines.h3[i, 1].prior, cube_fitter.lines.h4[i, 1].prior])
            append!(ln_lock, [cube_fitter.lines.h3[i, 1].locked, cube_fitter.lines.h4[i, 1].locked])
            append!(ln_names, ["$(ln_name)_h3", "$(ln_name)_h4"])
        elseif cube_fitter.lines.profiles[i, 1] == :Voigt
            # add voigt mixing parameter, but only if it's not tied
            if !cube_fitter.tie_voigt_mixing
                append!(ln_priors, [cube_fitter.lines.η[i, 1].prior])
                append!(ln_lock, [cube_fitter.lines.η[i, 1].locked])
                append!(ln_names, ["$(ln_name)_eta"])
            end
        end
        
        # repeat the above for the acomp components
        for j ∈ 2:cube_fitter.n_comps
            if !isnothing(cube_fitter.lines.profiles[i, j])
                # check tied or untied (but no flexible wavesol)
                if isnothing(cube_fitter.lines.tied[i, j])
                    # amplitude, voff, FWHM
                    append!(ln_priors, [amp_acomp_prior, cube_fitter.lines.voff[i, j].prior, cube_fitter.lines.fwhm[i, j].prior])
                    append!(ln_lock, [false, cube_fitter.lines.voff[i, j].locked, cube_fitter.lines.fwhm[i, j].locked])
                    append!(ln_names, ["$(ln_name)_acomp_amp", "$(ln_name)_acomp_voff", "$(ln_name)_acomp_fwhm"])
                else
                    # just amplitude (voff & FWHM tied)
                    append!(ln_priors, [amp_acomp_prior])
                    append!(ln_lock, [false])
                    append!(ln_names, ["$(ln_name)_acomp_amp"])
                end
                # check for additional profile parameters
                if cube_fitter.lines.profiles[i, j] == :GaussHermite
                    # h3 and h4 moments
                    append!(ln_priors, [cube_fitter.lines.h3[i, j].prior, cube_fitter.lines.h4[i, j].prior])
                    append!(ln_lock, [cube_fitter.lines.h3[i, j].locked, cube_fitter.lines.h4[i, j].locked])
                    append!(ln_names, ["$(ln_name)_acomp_h3", "$(ln_name)_acomp_h4"])
                elseif cube_fitter.lines.profiles[i, j] == :Voigt
                    # voigt mixing parameter, only if untied
                    if !cube_fitter.tie_voigt_mixing
                        append!(ln_priors, [cube_fitter.lines.η[i, j].prior])
                        append!(ln_lock, [cube_fitter.lines.η[i, j].locked])
                        append!(ln_names, ["$(ln_name)_acomp_eta"])
                    end
                end
            end
        end
    end

    priors = []
    param_lock = []
    param_names = []
    for j ∈ 1:cube_fitter.n_comps
        voff_tied_priors = []
        voff_tied_lock = []
        voff_tied_names = []
        fwhm_tied_priors = []
        fwhm_tied_lock = []
        fwhm_tied_names = []
        for (k, key) ∈ enumerate(cube_fitter.tied_kinematics.key[j])
            append!(voff_tied_priors, [cube_fitter.tied_kinematics.voff[j][k].prior])
            append!(voff_tied_lock, [cube_fitter.tied_kinematics.voff[j][k].locked])
            append!(voff_tied_names, ["voff_tied_$(key)_$(j)"])
            append!(fwhm_tied_priors, [cube_fitter.tied_kinematics.fwhm[j][k].prior])
            append!(fwhm_tied_lock, [cube_fitter.tied_kinematics.fwhm[j][k].locked])
            append!(fwhm_tied_names, ["fwhm_tied_$(key)_$(j)"])
        end
        # Sanity checking
        @assert length(voff_tied_priors) == length(fwhm_tied_priors) == cube_fitter.n_kin_tied[j]
        # Append to prior vector
        priors = vcat(priors, voff_tied_priors, fwhm_tied_priors)
        param_lock = vcat(param_lock, voff_tied_lock, fwhm_tied_lock)
        param_names = vcat(param_names, voff_tied_names, fwhm_tied_names)
    end

    ηᵢ = 2 * sum(cube_fitter.n_kin_tied) + 1
    # If the sum has already been fit, keep eta fixed for the individual spaxels
    η_prior = cube_fitter.tie_voigt_mixing ? [cube_fitter.voigt_mix_tied.prior] : []
    η_prior = init ? η_prior : Uniform(cube_fitter.p_init_line[ηᵢ]-1e-10, cube_fitter.p_init_line[ηᵢ]+1e-10)
    η_lock = cube_fitter.tie_voigt_mixing ? [cube_fitter.voigt_mix_tied.locked || !init] : []
    η_name = cube_fitter.tie_voigt_mixing ? ["eta_tied"] : []

    priors = Vector{Distribution}(vcat(priors, η_prior, ln_priors))
    param_lock = BitVector(vcat(param_lock, η_lock, ln_lock))
    param_names = Vector{String}(vcat(param_names, η_name, ln_names))

    priors, param_lock, param_names

end


"""
    get_line_initial_values(cube_fitter, init)

Get the vector of starting values for the line fit for a given CubeFitter object.
"""
function get_line_initial_values(cube_fitter::CubeFitter, init::Bool)

    # Check if there are previous best fit parameters
    if !init

        @debug "Using initial best fit line parameters..."

        # If so, set the parameters to the previous ones
        p₀ = copy(cube_fitter.p_init_line)

    else

        @debug "Calculating initial starting points..."
        
        # Start the ampltiudes at 1/2 and 1/4 (in normalized units)
        A_ln = ones(cube_fitter.n_lines) .* 0.5
        A_fl = ones(cube_fitter.n_lines) .* 0.5     # (acomp amp is multiplied with main amp)

        # Initial parameter vector
        ln_pars = Vector{Float64}()
        for i ∈ 1:cube_fitter.n_lines
            if isnothing(cube_fitter.lines.tied[i, 1])
                # 3 parameters: amplitude, voff, FWHM
                append!(ln_pars, [A_ln[i], cube_fitter.lines.voff[i, 1].value, cube_fitter.lines.fwhm[i, 1].value])
            elseif cube_fitter.flexible_wavesol
                # 2 parameters: amplitude, voff (since FWHM is tied but voff is only restricted)
                append!(ln_pars, [A_ln[i], cube_fitter.lines.voff[i, 1].value])
            else
                # 1 parameter: amplitude (since FWHM and voff are tied)
                append!(ln_pars, [A_ln[i]])
            end
            if cube_fitter.lines.profiles[i, 1] == :GaussHermite
                # 2 extra parameters: h3 and h4
                append!(ln_pars, [cube_fitter.lines.h3[i, 1].value, cube_fitter.lines.h4[i, 1].value])
            elseif cube_fitter.lines.profiles[i, 1] == :Voigt
                # 1 extra parameter: eta, but only if not tied
                if !cube_fitter.tie_voigt_mixing
                    append!(ln_pars, [cube_fitter.lines.η[i].value])
                end
            end
            # Repeat but for additional components, if present
            for j ∈ 2:cube_fitter.n_comps
                if !isnothing(cube_fitter.lines.profiles[i, j])
                    if isnothing(cube_fitter.lines.tied[i, j])
                        append!(ln_pars, [A_fl[i], cube_fitter.lines.voff[i, j].value, cube_fitter.lines.fwhm[i, j].value])
                    else
                        append!(ln_pars, [A_fl[i]])
                    end
                    if cube_fitter.lines.profiles[i, j] == :GaussHermite
                        append!(ln_pars, [cube_fitter.lines.h3[i, j].value, cube_fitter.lines.h4[i, j].value])
                    elseif cube_fitter.lines.profiles[i, j] == :Voigt
                        if !cube_fitter.tie_voigt_mixing
                            append!(ln_pars, [cube_fitter.lines.η[i, j].value])
                        end
                    end
                end
            end
        end

        p₀ = []
        # Set up tied voff and fwhm parameter vectors
        for j ∈ 1:cube_fitter.n_comps
            voff_tied_pars = []
            fwhm_tied_pars = []
            for (k, key) ∈ enumerate(cube_fitter.tied_kinematics.key[j])
                append!(voff_tied_pars, [cube_fitter.tied_kinematics.voff[j][k].value])
                append!(fwhm_tied_pars, [cube_fitter.tied_kinematics.fwhm[j][k].value])
            end
            # Sanity checking
            @assert length(voff_tied_pars) == length(fwhm_tied_pars) == cube_fitter.n_kin_tied[j]
            # Append to parameter vector
            p₀ = vcat(p₀, voff_tied_pars, fwhm_tied_pars)
        end

        η_par = cube_fitter.tie_voigt_mixing ? [cube_fitter.voigt_mix_tied.value] : []

        # Set up the parameter vector in the proper order: 
        # (tied voffs, tied fwhms, tied acomp voffs, tied acomp fwhms, tied voigt mixing, [amp, voff, FWHM, h3, h4, eta,
        #     acomp_amp, acomp_voff, acomp_FWHM, acomp_h3, acomp_h4, acomp_eta] for each line)
        p₀ = Vector{Float64}(vcat(p₀, η_par, ln_pars))

    end

    p₀

end


"""
    get_line_parinfo(cube_fitter, p₀, priors, param_lock)

Get the CMPFit parinfo and config objects for a given CubeFitter object, given the vector of initial valuels,
priors, and boolean locked values.
"""
function get_line_parinfo(n_free, lb, ub)

    # Convert parameter limits into CMPFit object
    parinfo = CMPFit.Parinfo(n_free)
    for pᵢ ∈ 1:n_free
        parinfo[pᵢ].fixed = 0
        parinfo[pᵢ].limited = (1,1)
        parinfo[pᵢ].limits = (lb[pᵢ], ub[pᵢ])
    end

    # Create a `config` structure
    config = CMPFit.Config()

    parinfo, config
end


"""
    pretty_print_line_results(cube_fitter, popt, perr, prof_ln, acomp_prof_ln)

Print out a nicely formatted summary of the line fit results for a given CubeFitter object.
"""
function pretty_print_line_results(cube_fitter::CubeFitter, popt::Vector{<:Real}, perr::Vector{<:Real})

    msg = "######################################################################\n"
    msg *= "############### SPAXEL FIT RESULTS -- EMISSION LINES #################\n"
    msg *= "######################################################################\n"
    pᵢ = 1
    msg *= "\n#> TIED VELOCITY OFFSETS <#\n"
    for j ∈ 1:cube_fitter.n_comps
        for (i, vk) ∈ enumerate(cube_fitter.tied_kinematics.key[j])
            msg *= "$(vk)_$(j)_tied_voff: \t\t\t $(@sprintf "%.0f" popt[pᵢ]) +/- $(@sprintf "%.0f" perr[pᵢ]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.tied_kinematics.voff[j][i].prior)), $(@sprintf "%.0f" maximum(cube_fitter.tied_kinematics.voff[j][i].prior)))\n"
            pᵢ += 1
        end
        for (ii, vk) ∈ enumerate(cube_fitter.tied_kinematics.key[j])
            msg *= "$(vk)_$(j)_tied_fwhm: \t\t\t $(@sprintf "%.0f" popt[pᵢ]) +/- $(@sprintf "%.0f" perr[pᵢ]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.tied_kinematics.fwhm[j][ii].prior)), $(@sprintf "%.0f" maximum(cube_fitter.tied_kinematics.fwhm[j][ii].prior)))\n"
            pᵢ += 1
        end
    end
    msg *= "\n#> TIED VOIGT MIXING <#\n"
    if cube_fitter.tie_voigt_mixing
        msg *= "tied_voigt_mixing: \t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t " * 
            "Limits: ($(@sprintf "%.2f" minimum(cube_fitter.voigt_mix_tied.prior)), $(@sprintf "%.2f" maximum(cube_fitter.voigt_mix_tied.prior)))\n"
        pᵢ += 1
    end
    msg *= "\n#> EMISSION LINES <#\n"
    for (k, nm) ∈ enumerate(cube_fitter.lines.names)
        msg *= "$(nm)_amp:\t\t\t $(@sprintf "%.3f" popt[pᵢ]) +/- $(@sprintf "%.3f" perr[pᵢ]) [x norm] \t Limits: (0, 1)\n"
        if isnothing(cube_fitter.lines.tied[k, 1])
            msg *= "$(nm)_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.lines.voff[k, 1].prior)), $(@sprintf "%.0f" maximum(cube_fitter.lines.voff[k, 1].prior)))\n"
            msg *= "$(nm)_fwhm:   \t\t $(@sprintf "%.0f" popt[pᵢ+2]) +/- $(@sprintf "%.0f" perr[pᵢ+2]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.lines.fwhm[k, 1].prior)), $(@sprintf "%.0f" maximum(cube_fitter.lines.fwhm[k, 1].prior)))\n"
            if cube_fitter.lines.profiles[k, 1] == :GaussHermite
                msg *= "$(nm)_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(cube_fitter.lines.h3[k, 1].prior)), $(@sprintf "%.3f" maximum(cube_fitter.lines.h3[k, 1].prior)))\n"
                msg *= "$(nm)_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+4]) +/- $(@sprintf "%.3f" perr[pᵢ+4])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(cube_fitter.lines.h4[k, 1].prior)), $(@sprintf "%.3f" maximum(cube_fitter.lines.h4[k, 1].prior)))\n"
                pᵢ += 2
            elseif cube_fitter.lines.profiles[k, 1] == :Voigt && !cube_fitter.tie_voigt_mixing
                msg *= "$(nm)_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(cube_fitter.lines.η[k, 1].prior)), $(@sprintf "%.3f" maximum(cube_fitter.lines.η[k, 1].prior)))\n"
                pᵢ += 1
            end
            pᵢ += 3
        elseif cube_fitter.flexible_wavesol
            msg *= "$(nm)_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) km/s \t " *
                "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.lines.voff[k, 1].prior)), $(@sprintf "%.0f" maximum(cube_fitter.lines.voff[k, 1].prior)))\n"
            if cube_fitter.lines.profiles[k, 1] == :GaussHermite
                msg *= "$(nm)_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(cube_fitter.lines.h3[k, 1].prior)), $(@sprintf "%.3f" maximum(cube_fitter.lines.h3[k, 1].prior)))\n"
                msg *= "$(nm)_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+4]) +/- $(@sprintf "%.3f" perr[pᵢ+4])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(cube_fitter.lines.h4[k, 1].prior)), $(@sprintf "%.3f" maximum(cube_fitter.lines.h4[k, 1].prior)))\n"
                pᵢ += 2
            elseif cube_fitter.lines.profiles[k, 1] == :Voigt && !cube_fitter.tie_voigt_mixing
                msg *= "$(nm)_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])      \t " *
                    "Limits: ($(@sprintf "%.3f" minimum(cube_fitter.lines.η[k, 1].prior)), $(@sprintf "%.3f" maximum(cube_fitter.lines.η[k, 1].prior)))\n"
                pᵢ += 1
            end
            pᵢ += 2
        else
            pᵢ += 1
        end
        for j ∈ 2:cube_fitter.n_comps
            if !isnothing(cube_fitter.lines.profiles[k, j])
                msg *= "\n$(nm)_acomp_$(j)_amp:\t\t\t $(@sprintf "%.3f" popt[pᵢ]) +/- $(@sprintf "%.3f" perr[pᵢ]) [x amp] \t Limits: (0, 1)\n"
                if isnothing(cube_fitter.lines.tied[k, j])
                    msg *= "$(nm)_acomp_$(j)_voff:   \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) [+ voff] \t " *
                        "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.lines.voff[k, j].prior)), $(@sprintf "%.0f" maximum(cube_fitter.lines.voff[k, j].prior)))\n"
                    msg *= "$(nm)_acomp_$(j)_fwhm:   \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) [x fwhm] \t " *
                        "Limits: ($(@sprintf "%.0f" minimum(cube_fitter.lines.fwhm[k, j].prior)), $(@sprintf "%.0f" maximum(cube_fitter.lines.fwhm[k, j].prior)))\n"
                    if cube_fitter.lines.profiles[k, j] == :GaussHermite
                        msg *= "$(nm)_acomp_$(j)_h3:    \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])      \t " *
                            "Limits: ($(@sprintf "%.3f" minimum(cube_fitter.lines.h3[k, j].prior)), $(@sprintf "%.3f" maximum(cube_fitter.lines.h3[k, j].prior)))\n"
                        msg *= "$(nm)_acomp_$(j)_h4:    \t\t $(@sprintf "%.3f" popt[pᵢ+4]) +/- $(@sprintf "%.3f" perr[pᵢ+4])      \t " *
                            "Limits: ($(@sprintf "%.3f" minimum(cube_fitter.lines.h4[k, j].prior)), $(@sprintf "%.3f" maximum(cube_fitter.lines.h4[k, j].prior)))\n"
                        pᵢ += 2
                    elseif cube_fitter.lines.profiles[k, j] == :Voigt && !cube_fitter.tie_voigt_mixing
                        msg *= "$(nm)_acomp_$(j)_η:     \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3])       \t " *
                            "Limits: ($(@sprintf "%.3f" minimum(cube_fitter.lines.η[k, j].prior)), $(@sprintf "%.3f" maximum(cube_fitter.lines.η[k, j].prior)))\n"
                        pᵢ += 1
                    end
                    pᵢ += 3
                else
                    pᵢ += 1
                end
            end
            msg *= "\n"
        end
    end 
    msg *= "######################################################################" 
    @debug msg

    msg

end

