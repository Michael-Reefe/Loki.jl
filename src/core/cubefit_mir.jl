


# """
#     MIRCubeModel(model, stellar, dust_continuum, dust_features, extinction, hot_dust, lines)

# A structure for holding 3D models of intensity, split up into model components, generated when fitting a cube.
# This will be the same shape as the input data, and preferably the same datatype too (i.e., JWST files have flux
# and error in Float32 format, so we should also output in Float32 format).  This is useful as a way to quickly
# compare the full model, or model components, to the data.

# # Fields {T<:Real}
# - `model::Array{T, 3}`: The full 3D model.
# - `stellar::Array{T, 3}`: The stellar component of the continuum.
# - `dust_continuum::Array{T, 4}`: The dust components of the continuum. The 4th axis runs over each individual dust component.
# - `power_law::Array{T, 4}`: The power law components of the continuum. The 4th axis runs over each individual power law.
# - `dust_features::Array{T, 4}`: The dust (PAH) feature profiles. The 4th axis runs over each individual dust profile.
# - `abs_features::Array{T, 4}`: The absorption feature profiles. The 4th axis runs over each individual absorption profile.
# - `extinction::Array{T, 3}`: The extinction profile.
# - `abs_ice::Array{T, 3}`: The water-ice absorption feature profile.
# - `abs_ch::Array{T, 3}`: The CH absorption feature profile.
# - `hot_dust::Array{T, 3}`: The hot dust emission profile.
# - `templates::Array{T, 4}`: The generic template profiles.
# - `lines::Array{T, 4}`: The line profiles. The 4th axis runs over each individual line.

# See [`cubemodel_empty`](@ref) for a default constructor method.
# """
# struct CubeModel{T<:Real} <: CubeModel

    # COME BACK TO THIS AFTER SETTING UP THE MODEL 

    # model::Array{T, 3}
    # unobscured_continuum::Array{T, 3}
    # obscured_continuum::Array{T, 3}
    # stellar::Array{T, 3}
    # dust_continuum::Array{T, 4}
    # power_law::Array{T, 4}
    # dust_features::Array{T, 4}
    # abs_features::Array{T, 4}
    # extinction::Array{T, 4}
    # abs_ice::Array{T, 3}
    # abs_ch::Array{T, 3}
    # hot_dust::Array{T, 3}
    # templates::Array{T, 4}
    # lines::Array{T, 4}

# end


# """
#     cubemodel_empty(shape, n_dust_cont, n_power_law, df_names, ab_names, line_names[, floattype])

# A constructor function for making a default empty MIRCubeModel object with all the necessary fields for a given
# fit of a DataCube.

# # Arguments
# - `shape::Tuple`: The dimensions of the DataCube being fit, formatted as a tuple of (nx, ny, nz)
# - `n_dust_cont::Integer`: The number of dust continuum components in the fit (usually given by the number of temperatures 
#     specified in the dust.toml file)
# - `n_power_law::Integer`: The number of power law continuum components in the fit.
# - `df_names::Vector{String}`: List of names of PAH features being fit, i.e. "PAH_12.62", ...
# - `ab_names::Vector{String}`: List of names of absorption features being fit, i.e. "abs_HCO+_12.1", ...
# - `temp_names::Vector{String}`: List of names of generic templates in the fit, i.e. "nuclear", ...
# - `line_names::Vector{Symbol}`: List of names of lines being fit, i.e. "NeVI_7652", ...
# - `floattype::DataType=Float32`: The type of float to use in the arrays. Should ideally be the same as the input data,
#     which for JWST is Float32.
# """
# function cubemodel_empty(shape::Tuple, n_dust_cont::Integer, n_power_law::Integer, df_names::Vector{String}, 
#     ab_names::Vector{String}, temp_names::Vector{String}, line_names::Vector{Symbol}, extinction_curve::String, 
#     floattype::DataType=Float32)::MIRCubeModel

#     @debug """\n
#     Creating MIRCubeModel struct with shape $shape
#     ##############################################
#     """

#     # Make sure the floattype given is actually a type of float
#     @assert floattype <: AbstractFloat "floattype must be a type of AbstractFloat (Float32 or Float64)!"
#     # Swap the wavelength axis to be the FIRST axis since it is accessed most often and thus should be continuous in memory
#     shape2 = (shape[end], shape[1:end-1]...)

#     # Initialize the arrays for each part of the full 3D model
#     model = zeros(floattype, shape2...)
#     @debug "model cube"
#     unobscured_continuum = zeros(floattype, shape2...)
#     @debug "unobscured continuum cube"
#     obscured_continuum = zeros(floattype, shape2...)
#     @debug "obscured continuum cube"
#     stellar = zeros(floattype, shape2...)
#     @debug "stellar continuum comp cube"
#     dust_continuum = zeros(floattype, shape2..., n_dust_cont)
#     @debug "dust continuum comp cubes"
#     power_law = zeros(floattype, shape2..., n_power_law)
#     @debug "power law comp cubes"
#     dust_features = zeros(floattype, shape2..., length(df_names))
#     @debug "dust features comp cubes"
#     abs_features = zeros(floattype, shape2..., length(ab_names))
#     @debug "absorption features comp cubes"
#     extinction = zeros(floattype, shape2..., extinction_curve == "decompose" ? 4 : 1)
#     @debug "extinction comp cube"
#     abs_ice = zeros(floattype, shape2...)
#     @debug "abs_ice comp cube"
#     abs_ch = zeros(floattype, shape2...)
#     @debug "abs_ch comp cube"
#     hot_dust = zeros(floattype, shape2...)
#     @debug "hot dust comp cube"
#     templates = zeros(floattype, shape2..., length(temp_names))
#     @debug "templates comp cube"
#     lines = zeros(floattype, shape2..., length(line_names))
#     @debug "lines comp cubes"

#     MIRCubeModel(model, unobscured_continuum, obscured_continuum, stellar, dust_continuum, power_law, dust_features, abs_features, 
#         extinction, abs_ice, abs_ch, hot_dust, templates, lines)
# end




# Helper function for preparing continuum and dust feature parameters for 
# a CubeFitter object
function cubefitter_prepare_continuum(λ::Vector{<:QWave}, z::Real, out::Dict, λunit::Unitful.Units, 
    Iunit::Unitful.Units, region::SpectralRegion, name::String, cube::DataCube)

    # Construct the ModelParameters object
    model_parameters = construct_model_parameters(out, λunit, Iunit, region, z)
    vres = NaN*u"km/s"   # (vres doesnt make sense if the wavelength vector isnt logarithmic)

    # Count a few different parameters 
    params = get_flattened_fit_parameters(model_parameters)
    pnames = params.names

    # Stellar populations
    ssps = nothing
    n_ssps = div(sum(contains.(pnames, "continuum.stellar_populations.")), 3)  # Each SSP has 3 parameters (mass, age, metallicity)
    if n_ssps > 0
        # Create the simple stellar population templates with FSPS
        ssp_λ, ages, metals, ssp_templates = generate_stellar_populations(λ, Iunit, cube.lsf, z, out[:cosmology], name)
        ssp_unit = unit(ssp_templates[1])
        # Create a 2D linear interpolation over the ages/metallicities
        ssp_templates = [Spline2D(ustrip.(ages), ustrip.(metals), ustrip.(ssp_templates[:, :, i]), kx=1, ky=1) for i in eachindex(ssp_λ)]
        # Make the object
        vsyst_ssp = log(ssp_λ[1]/λ[1]) * C_KMS
        ssps = StellarPopulations(ssp_λ, ssp_templates, ssp_unit, vsyst_ssp)
    end

    # Fe II templates
    feii = nothing
    if out[:fit_opt_na_feii] || out[:fit_opt_br_feii]
        # Load in the Fe II templates from Veron-Cetty et al. (2004)
        npad_feii, feii_λ, na_feii_fft, br_feii_fft = generate_feii_templates(λ, Iunit, cube.lsf)
        # Make the object
        vsyst_feii = log(feii_λ[1]/λ[1]) * C_KMS
        feii = FeIITemplates(feii_λ, npad_feii, na_feii_fft, br_feii_fft, vsyst_feii)
    end

    # Velocity resolution
    if n_ssps > 0 || out[:fit_opt_na_feii] || out[:fit_opt_br_feii]
        vres = log(λ[2]/λ[1]) * C_KMS
    end

    # Power laws
    n_power_law = div(sum(contains.(pnames, "continuum.power_law.")), 2)   # Each power law has 2 parameters (amp, slope)
    # Dust continua
    n_dust_cont = div(sum(contains.(pnames, "continuum.dust.")), 2)   # Each dust continuum has 2 parameters (amp, temperature)
    # PAH features
    n_dust_feat = total_num_profiles(model_parameters.dust_features)
    # Templates
    n_templates = size(out[:templates], 4)

    model_parameters, ssps, feii, n_ssps, n_power_law, n_dust_cont, n_dust_feat, n_templates, vres
end


# Helper function for preparing emission line parameters for a CubeFitter object
function cubefitter_prepare_lines(out::Dict, λunit::Unitful.Units, Iunit::Unitful.Units,
    cube::DataCube, region::SpectralRegion)

    # Construct the FitFeatures object for the lines
    lines = construct_line_parameters(out, λunit, Iunit, region)

    # Count the parameters
    n_lines = length(lines.names)
    n_acomps = total_num_profiles(lines)

    # Buffer array for the number of fit line profiles in each spaxel
    n_fit_comps = Dict{Symbol, Matrix{Int}}()
    for name ∈ lines.names
        n_fit_comps[name] = ones(Int, size(cube.I)[1:2])
    end

    # Adjust the sort_line_components option based on the relative flags in the lines object
    relative_flags = BitVector([lines.rel_amp, lines.rel_voff, lines.rel_fwhm])
    if !haskey(out, :sort_line_components)
        out[:sort_line_components] = nothing
        if all(.~relative_flags)
            out[:sort_line_components] = :flux
        end
    elseif !isnothing(out[:sort_line_components])
        out[:sort_line_components] = Symbol(out[:sort_line_components])
    end

    lines, n_lines, n_acomps, n_fit_comps
end


# Helper function for counting the total number of MIR continuum parameters
function count_cont_parameters(model::ModelParameters; split::Bool=false)
    # Continuum parameters
    n_params_cont = length(model.continuum)
    # If the "split" option is true, keep the continuum and PAHs separate, otherwise they are combined
    if !split
        n_params_cont += count_dust_parameters(model)
    end
    n_params_cont
end


function count_dust_parameters(model::ModelParameters)
    length(get_flattened_fit_parameters(model.dust_features))
end


# Helper function for calculating an estimate of optical depth based on continuum slopes
function guess_optical_depth(cube_fitter::CubeFitter, λ::Vector{<:Real}, init::Bool)

    continuum = cube_fitter.continuum
    i1 = nanmedian(I[cube_fitter.guess_tau[1][1] .< λ .< cube_fitter.guess_tau[1][2]])
    i2 = nanmedian(I[cube_fitter.guess_tau[2][1] .< λ .< cube_fitter.guess_tau[2][2]])
    m = (i2 - i1) / (mean(cube_fitter.guess_tau[2]) - mean(cube_fitter.guess_tau[1]))
    contin_unextinct = i1 + m * (10.0 - mean(cube_fitter.guess_tau[1]))  # linear extrapolation over the silicate feature
    contin_extinct = clamp(nanmedian(I[9.9 .< λ .< 10.1]), 0., Inf)
    # Optical depth at 10 microns
    r = contin_extinct / contin_unextinct
    tau_10 = r > 0 ? clamp(-log(r), continuum.τ_97.limits...) : 0.
    if !cube_fitter.extinction_screen && r > 0
        # solve nonlinear equation
        f(τ) = r - (1 - exp(-τ[1]))/τ[1]
        try
            soln = nlsolve(f, [tau_10])
            tau_10 = clamp(soln.zero[1], continuum.τ_97.limits...)
        catch
            tau_10 = 0.
        end
    end

    pₑ = 3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law
    # Get the extinction curve
    β = init ? continuum.β.value : cube_fitter.p_init_cont[pₑ+3]
    if cube_fitter.extinction_curve == "d+"
        ext_10 = τ_dp([10.0], β)[1]
    elseif cube_fitter.extinction_curve == "kvt"
        ext_10 = τ_kvt([10.0], β)[1]
    elseif cube_fitter.extinction_curve == "ct"
        ext_10 = τ_ct([10.0])[1]
    elseif cube_fitter.extinction_curve == "ohm"
        ext_10 = τ_ohm([10.0])[1]
    else
        error("Unrecognized extinction curve: $(cube_fitter.extinction_curve)")
    end
    # Convert tau at 10 microns to tau at 9.7 microns
    tau_guess = clamp(tau_10 / ext_10, continuum.τ_97.limits...)

    tau_guess
end


# Helper function for getting initial MIR parameters based on an input previous parameter cube
function get_mir_continuum_initial_values_from_pcube(cube_fitter::CubeFitter, spaxel::CartesianIndex, n_split::Integer)

    pcube_cont = cube_fitter.p_init_cube_cont
    # Get the coordinates of all spaxels that have fit results
    coords0 = [float.(c.I) for c in CartesianIndices(size(pcube_cont)[1:2]) if !all(isnan.(pcube_cont[c,:]))]
    coords = cube_fitter.p_init_cube_coords
    # Calculate their distances from the current spaxel
    dist = [hypot(spaxel[1]-c[1], spaxel[2]-c[2]) for c in coords]
    closest = coords0[argmin(dist)]
    @debug "Using initial best fit continuum parameters from coordinates $closest --> $(coords[argmin(dist)])"

    p₀ = pcube_cont[Int.(closest)..., :]
    pahtemp = model_pah_residuals(cube_fitter.cube.λ, p₀[(n_split+1):end], cube_fitter.dust_features.profiles, ones(length(cube_fitter.cube.λ)),
        nothing, false)
    pah_frac = repeat([maximum(pahtemp)/2], 2)

    p₀, pah_frac
end


# Helper function for getting initial MIR parameters based on the initial fit
function get_mir_continuum_initial_values_from_previous(cube_fitter::CubeFitter, spaxel::CartesianIndex,
    I::Vector{<:Real}, N::Real, tau_guess::Real)

    # Set the parameters to the best parameters
    p₀ = copy(cube_fitter.p_init_cont)
    pah_frac = copy(cube_fitter.p_init_pahtemp)

    # τ_97_0 = cube_fitter.τ_guess[parse(Int, cube_fitter.cube.channel)][spaxel]
    # max_τ = cube_fitter.continuum.τ_97.limits[2]

    # scale all flux amplitudes by the difference in medians between the spaxel and the summed spaxels
    # (should be close to 1 since the sum is already normalized by the number of spaxels included anyways)
    I_init = sumdim(cube_fitter.cube.I, (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    scale = max(nanmedian(I), 1e-10) * N / nanmedian(I_init)
    # max_amp = 1 / exp(-max_τ)

    # Stellar amplitude (rescaled)
    p₀[1] = p₀[1] * scale 
    pᵢ = 3

    # Dust continuum amplitudes (rescaled)
    for di ∈ 1:cube_fitter.n_dust_cont
        p₀[pᵢ] = p₀[pᵢ] * scale 
        if (cube_fitter.lock_hot_dust[1] || cube_fitter.nuc_fit_flag[1]) && isone(di)
            p₀[pᵢ] = 0.
        end
        pᵢ += 2
    end

    # Power law amplitudes (NOT rescaled)
    for _ ∈ 1:cube_fitter.n_power_law
        # p₀[pᵢ] = p₀[pᵢ] 
        pᵢ += 2
    end

    # Set optical depth based on the initial guess or the initial fit (whichever is larger)
    if cube_fitter.extinction_curve != "decompose"
        p₀[pᵢ] = max(cube_fitter.continuum.τ_97.value, p₀[pᵢ])
    end

    # Set τ_9.7 and τ_CH to 0 if the continuum is within 1 std dev of 0
    # lock_abs = false
    # if nanmedian(I) ≤ 2nanmedian(σ)
    #     lock_abs = true
    #     if cube_fitter.extinction_curve != "decompose"
    #         p₀[pᵢ] = 0.
    #         p₀[pᵢ+2] = 0.
    #     else
    #         p₀[pᵢ:pᵢ+2] .= 0.
    #         p₀[pᵢ+4] = 0.
    #     end
    # end

    # Set τ_9.7 to the guess if the guess_tau flag is set
    if !isnothing(cube_fitter.guess_tau) && (cube_fitter.extinction_curve != "decompose")
        p₀[pᵢ] = tau_guess
    end

    # Override if an extinction_map was provided
    if !isnothing(cube_fitter.extinction_map)
        @debug "Using the provided τ_9.7 values from the extinction_map"
        pₑ = [pᵢ]
        if cube_fitter.extinction_curve == "decompose"
            append!(pₑ, [pᵢ+1, pᵢ+2])
        end
        if !isnothing(cube_fitter.cube.voronoi_bins)
            data_indices = findall(cube_fitter.cube.voronoi_bins .== Tuple(spaxel)[1])
            for i in eachindex(pₑ)
                p₀[pₑ[i]] = mean(cube_fitter.extinction_map[data_indices, i])
            end
        else
            data_index = spaxel
            for i in eachindex(pₑ)
                p₀[pₑ[i]] = cube_fitter.extinction_map[data_index, i]
            end
        end
    end

    # Do not adjust absorption feature amplitudes since they are multiplicative
    pᵢ += 4 + (cube_fitter.extinction_curve == "decompose" ? 3 : 1)
    for _ ∈ 1:cube_fitter.n_abs_feat
        # if lock_abs
        #     p₀[pᵢ] = 0.
        # end
        pᵢ += 4
    end

    # Hot dust amplitude (rescaled)
    if cube_fitter.fit_sil_emission
        p₀[pᵢ] *= scale
        pᵢ += 6
    end

    # Template amplitudes (not rescaled)
    if cube_fitter.fit_temp_multexp
        tamp = sum(p₀[[pᵢ,pᵢ+2,pᵢ+4,pᵢ+6]]) / 4
        for _ ∈ 1:4
            p₀[pᵢ] = tamp
            pᵢ += 2
        end
    else
        for _ ∈ 1:(cube_fitter.n_templates*cube_fitter.n_channels)
            p₀[pᵢ] = 1/cube_fitter.n_templates
            pᵢ += 1
        end
    end

    # Dust feature amplitudes (not rescaled)
    # for i ∈ 1:cube_fitter.n_dust_feat
    #     pᵢ += 3
    #     if cube_fitter.dust_features.profiles[i] == :PearsonIV
    #         pᵢ += 2
    #     end
    # end

    p₀, pah_frac
end


# Helper function for getting initial MIR parameters based on estimates from the data
function get_mir_continuum_initial_values_from_estimation(cube_fitter::CubeFitter, λ::Vector{<:Real}, 
    I::Vector{<:Real}, N::Real, tau_guess::Real)

    cubic_spline = Spline1D(λ, I, k=3)
    continuum = cube_fitter.continuum

    # Stellar amplitude
    λ_s = minimum(λ) < 5 ? minimum(λ)+0.1 : 5.1
    A_s = clamp(cubic_spline(λ_s) * N / Blackbody_ν(λ_s, continuum.T_s.value), 0., Inf)
    if !cube_fitter.fit_stellar_continuum
        A_s = 0.
    end

    # Dust feature amplitudes
    A_df = repeat([clamp(nanmedian(I)/2, 0., Inf)], cube_fitter.n_dust_feat)
    # PAH templates
    pah_frac = repeat([clamp(nanmedian(I)/2, 0., Inf)], 2)

    # Absorption feature amplitudes
    A_ab = [tau.value for tau ∈ cube_fitter.abs_taus]

    # Dust continuum amplitudes
    λ_dc = clamp.([Wein(Ti.value) for Ti ∈ continuum.T_dc], minimum(λ), maximum(λ))
    A_dc = clamp.([cubic_spline(λ_dci) * N / Blackbody_ν(λ_dci, T_dci.value) for (λ_dci, T_dci) ∈ 
        zip(λ_dc, continuum.T_dc)] .* (λ_dc ./ 9.7).^2 ./ (cube_fitter.n_dust_cont / 2), 0., Inf)
    if cube_fitter.lock_hot_dust[1] || cube_fitter.nuc_fit_flag[1]
        A_dc[1] = 0.
    end
    
    # Power law amplitudes
    A_pl = [clamp(nanmedian(I), 0., Inf)/exp(-continuum.τ_97.value)/cube_fitter.n_power_law for αi ∈ continuum.α]
    
    # Hot dust amplitude
    hd = silicate_emission(λ, 1.0, continuum.T_hot.value, continuum.Cf_hot.value, continuum.τ_warm.value, 
        continuum.τ_cold.value, continuum.sil_peak.value)
    mhd = argmax(hd)
    A_hd = clamp(cubic_spline(λ[mhd]) * N / hd[mhd] / 5, 0., Inf)

    stellar_pars = [A_s, continuum.T_s.value]
    dc_pars = vcat([[Ai, Ti.value] for (Ai, Ti) ∈ zip(A_dc, continuum.T_dc)]...)
    pl_pars = vcat([[Ai, αi.value] for (Ai, αi) ∈ zip(A_pl, continuum.α)]...)
    
    df_pars = Float64[]
    for n in 1:length(cube_fitter.dust_features.names)
        append!(df_pars, [A_df[n], cube_fitter.dust_features.mean[n].value, cube_fitter.dust_features.fwhm[n].value])
        if cube_fitter.dust_features.profiles[n] == :PearsonIV
            append!(df_pars, [cube_fitter.dust_features.index[n].value, cube_fitter.dust_features.cutoff[n].value])
        else
            push!(df_pars, cube_fitter.dust_features.asym[n].value)
        end
    end
    
    ab_pars = vcat([[Ai, mi.value, fi.value, ai.value] for (Ai, mi, fi, ai) ∈ 
        zip(A_ab, cube_fitter.abs_features.mean, cube_fitter.abs_features.fwhm, cube_fitter.abs_features.asym)]...)
    if cube_fitter.fit_sil_emission
        hd_pars = [A_hd, continuum.T_hot.value, continuum.Cf_hot.value, continuum.τ_warm.value, continuum.τ_cold.value,
            continuum.sil_peak.value]
    else
        hd_pars = []
    end

    if cube_fitter.extinction_curve != "decompose"
        extinction_pars = [continuum.τ_97.value, continuum.τ_ice.value, continuum.τ_ch.value, continuum.β.value, continuum.Cf.value]
    else
        extinction_pars = [continuum.N_oli.value, continuum.N_pyr.value, continuum.N_for.value,
                        continuum.τ_ice.value, continuum.τ_ch.value, continuum.β.value, continuum.Cf.value]
    end
    if !isnothing(cube_fitter.guess_tau) && (cube_fitter.extinction_curve != "decompose")
        extinction_pars[1] = tau_guess
    end

    if cube_fitter.fit_temp_multexp
        temp_pars = [0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0]
    else
        temp_pars = [ta.value for ta in continuum.temp_amp]
    end
    # apply the nuclear template amplitudes for the initial fit
    if cube_fitter.nuc_fit_flag[1]
        temp_pars ./= cube_fitter.nuc_temp_amps
    end

    # Initial parameter vector
    p₀ = Vector{Float64}(vcat(stellar_pars, dc_pars, pl_pars, extinction_pars, ab_pars, hd_pars, temp_pars, df_pars))

    p₀, pah_frac
end


# Helper function for getting MIR parameter step sizes 
function get_mir_continuum_step_sizes(cube_fitter::CubeFitter, λ::Vector{<:Real})

    # Calculate relative step sizes for finite difference derivatives
    dλ = (λ[end] - λ[1]) / length(λ)
    deps = sqrt(eps())
    continuum = cube_fitter.continuum

    stellar_dstep = [deps, 1e-4]
    dc_dstep = vcat([[deps, 1e-4] for _ in continuum.T_dc]...)
    pl_dstep = vcat([[deps, deps] for _ in continuum.α]...)
    df_dstep = Float64[]
    for n in 1:length(cube_fitter.dust_features.names)
        append!(df_dstep, [deps, dλ/10/cube_fitter.dust_features.mean[n].value, dλ/1000/cube_fitter.dust_features.fwhm[n].value])
        if cube_fitter.dust_features.profiles[n] == :PearsonIV
            append!(df_dstep, [deps, deps])
        else
            push!(df_dstep, deps)
        end
    end
    ab_dstep = vcat([[deps, dλ/10/mi.value, dλ/1000/fi.value, deps] for (mi, fi) in zip(cube_fitter.abs_features.mean, cube_fitter.abs_features.fwhm)]...)
    if cube_fitter.fit_sil_emission
        hd_dstep = [deps, 1e-4, deps, deps, deps, dλ/10/continuum.sil_peak.value]
    else
        hd_dstep = []
    end
    extinction_dstep = repeat([deps], cube_fitter.extinction_curve == "decompose" ? 7 : 5)
    temp_dstep = [deps for _ in 1:(cube_fitter.fit_temp_multexp ? 8 : cube_fitter.n_templates*cube_fitter.n_channels)]
    dstep = Vector{Float64}(vcat(stellar_dstep, dc_dstep, pl_dstep, extinction_dstep, ab_dstep, hd_dstep, temp_dstep, df_dstep))

    deps, dstep
end


# MIR implementation of the get_continuum_initial_values function
function get_mir_continuum_initial_values(cube_fitter::CubeFitter, spaxel::CartesianIndex, λ::Vector{<:Real}, I::Vector{<:Real}, 
    N::Real; init::Bool=false, split::Bool=false, force_noext::Bool=false)

    continuum = cube_fitter.continuum
    n_split = cubefitter_mir_count_cont_parameters(cube_fitter.extinction_curve, cube_fitter.fit_sil_emission, 
        cube_fitter.fit_temp_multexp, cube_fitter.n_dust_cont, cube_fitter.n_power_law, cube_fitter.n_abs_feat, 
        cube_fitter.n_templates, cube_fitter.n_channels, cube_fitter.dust_features; split=true)

    # guess optical depth from the dip in the continuum level
    tau_guess = 0.
    if !isnothing(cube_fitter.guess_tau) && (cube_fitter.extinction_curve != "decompose")
        tau_guess = guess_optical_depth(cube_fitter, λ, init)
    end

    # Check if cube fitter has initial cube
    if !isnothing(cube_fitter.p_init_cube_λ) && !init
        @debug "Using parameter cube best fit continuum parameters..."
        p₀, pah_frac = get_mir_continuum_initial_values_from_pcube(cube_fitter, spaxel, n_split)
    # Check if the cube fitter has initial fit parameters 
    elseif !init
        @debug "Using initial best fit continuum parameters..."
        p₀, pah_frac = get_mir_continuum_initial_values_from_previous(cube_fitter, spaxel, I, N, tau_guess)
    # Otherwise, we estimate the initial parameters based on the data
    else
        @debug "Calculating initial starting points..."
        p₀, pah_frac = get_mir_continuum_initial_values_from_estimation(cube_fitter, λ, I, N, tau_guess)
    end
    if force_noext
        pₑ = [3 + 2cube_fitter.n_dust_cont + 2cube_fitter.n_power_law]
        p₀[pₑ] .= 0.
    end

    @debug "Continuum Parameter labels: \n [stellar_amp, stellar_temp, " * 
        join(["dust_continuum_amp_$i, dust_continuum_temp_$i" for i ∈ 1:cube_fitter.n_dust_cont], ", ") * 
        join(["power_law_amp_$i, power_law_index_$i" for i ∈ 1:cube_fitter.n_power_law], ", ") *
        (cube_fitter.extinction_curve == "decompose" ? ", extinction_N_oli, extinction_N_pyr, extinction_N_for" : ", extinction_tau_97") *
        ", extinction_tau_ice, extinction_tau_ch, extinction_beta, extinction_Cf, " *  
        join(["$(ab)_tau, $(ab)_mean, $(ab)_fwhm" for ab ∈ cube_fitter.abs_features.names], ", ") *
        (cube_fitter.fit_sil_emission ? ", hot_dust_amp, hot_dust_temp, hot_dust_covering_frac, hot_dust_tau_warm, hot_dust_tau_cold, hot_dust_sil_peak, " : ", ") *
        (cube_fitter.fit_temp_multexp ? "temp_multexp_amp1, temp_multexp_ind1, temp_multexp_amp2, temp_multexp_ind2, temp_multexp_amp3, temp_multexp_ind3, " * 
        "temp_multexp_amp4, temp_multexp_ind4, " : join(["$(tp)_amp_$i" for i in 1:cube_fitter.n_channels for tp ∈ cube_fitter.template_names], ", ")) *
        join(["$(df)_amp, $(df)_mean, $(df)_fwhm" * (cube_fitter.dust_features.profiles[n] == :PearsonIV ? ", $(df)_index, $(df)_cutoff" : 
            "$(df)_asym") for (n, df) ∈ enumerate(cube_fitter.dust_features.names)], ", ") * "]"
    @debug "Continuum Starting Values: \n $p₀"

    deps, dstep = get_mir_continuum_step_sizes(cube_fitter, λ)
    @debug "Continuum relative step sizes: \n $dstep"

    if !split
        p₀, dstep
    else
        # Step 1: Stellar + Dust blackbodies, 2 new amplitudes for the PAH templates, and the extinction parameters
        pars_1 = vcat(p₀[1:n_split], pah_frac)
        dstep_1 = vcat(dstep[1:n_split], [deps, deps])
        # Step 2: The PAH profile amplitudes, centers, and FWHMs
        pars_2 = p₀[(n_split+1):end]
        dstep_2 = dstep[(n_split+1):end]

        pars_1, pars_2, dstep_1, dstep_2
    end
end


"""
    nuc_temp_fit_minimize_psftemp_amp!(cube_fitter, popt_0)

If fitting the nuclear spectrum with individual PSF template amplitudes, this function cleans the posteriors
by minimizing their mean distance from 1. It does so by moving the amplitude in the PSF template amplitudes to
the amplitudes of the other parts of the continuum. The resulting model should be identical.
"""
function nuc_temp_fit_minimize_psftemp_amp!(cube_fitter::CubeFitter, popt_0::Vector{<:Real})
    p_temp_0 = 1 + 2 + 2*cube_fitter.n_dust_cont + 2*cube_fitter.n_power_law + 4 + (cube_fitter.extinction_curve == "decompose" ? 3 : 1) +
        3*cube_fitter.n_abs_feat + (cube_fitter.fit_sil_emission ? 6 : 0)
    p_temp_1 = p_temp_0 + (cube_fitter.fit_temp_multexp ? 8 : cube_fitter.n_templates*cube_fitter.n_channels) - 1

    resid_amp = nanmean(popt_0[p_temp_0:p_temp_1])
    # normalize the residual amplitude from the PSF normalizations
    popt_0[p_temp_0:p_temp_1] ./= resid_amp

    # add the same amount of amplitude back into the continuum
    popt_0[1] *= resid_amp
    pᵢ = 3
    for _ in 1:cube_fitter.n_dust_cont
        popt_0[pᵢ] *= resid_amp
        pᵢ += 2
    end
    for _ in 1:cube_fitter.n_power_law
        popt_0[pᵢ] *= resid_amp
        pᵢ += 2
    end
    pᵢ += cube_fitter.extinction_curve == "decompose" ? 3 : 1
    pᵢ += 4
    pᵢ += 3*cube_fitter.n_abs_feat
    if cube_fitter.fit_sil_emission
        popt_0[pᵢ] *= resid_amp
        pᵢ += 6
    end
    pᵢ += cube_fitter.fit_temp_multexp ? 8 : cube_fitter.n_channels*cube_fitter.n_templates
    for prof in enumerate(cube_fitter.dust_features.profiles)
        popt_0[pᵢ] *= resid_amp
        if prof == :Drude
            pᵢ += 4
        elseif prof == :PearsonIV
            pᵢ += 5
        end
    end
    return popt_0
end


# MIR implementation of the pretty_print_continuum_results function
function pretty_print_mir_continuum_results(cube_fitter::CubeFitter, popt::Vector{<:Real}, perr::Vector{<:Real},
    I::Vector{<:Real})

    continuum = cube_fitter.continuum

    msg = "######################################################################\n"
    msg *= "################# SPAXEL FIT RESULTS -- CONTINUUM ####################\n"
    msg *= "######################################################################\n"
    msg *= "\n#> STELLAR CONTINUUM <#\n"
    msg *= "Stellar_amp: \t\t\t $(@sprintf "%.3g" popt[1]) +/- $(@sprintf "%.3g" perr[1]) [-] \t Limits: (0, Inf)\n"
    msg *= "Stellar_temp: \t\t\t $(@sprintf "%.0f" popt[2]) +/- $(@sprintf "%.3e" perr[2]) K \t (fixed)\n"
    pᵢ = 3
    msg *= "\n#> DUST CONTINUUM <#\n"
    for i ∈ 1:cube_fitter.n_dust_cont
        msg *= "Dust_continuum_$(i)_amp: \t\t $(@sprintf "%.3g" popt[pᵢ]) +/- $(@sprintf "%.3g" perr[pᵢ]) [-] \t Limits: (0, Inf)\n"
        msg *= "Dust_continuum_$(i)_temp: \t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.3e" perr[pᵢ+1]) K \t\t\t (fixed)\n"
        msg *= "\n"
        pᵢ += 2
    end
    msg *= "\n#> POWER LAWS <#\n"
    for k ∈ 1:cube_fitter.n_power_law
        msg *= "Power_law_$(k)_amp: \t\t $(@sprintf "%.3g" popt[pᵢ]) +/- $(@sprintf "%.3g" perr[pᵢ]) [x norm] \t Limits: (0, Inf)\n"
        msg *= "Power_law_$(k)_index: \t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1]) [-] \t Limits: " *
            "($(@sprintf "%.3f" continuum.α[k].limits[1]), $(@sprintf "%.3f" continuum.α[k].limits[2]))" *
            (continuum.α[k].locked ? " (fixed)" : "") * "\n"
        pᵢ += 2
    end
    msg *= "\n#> EXTINCTION <#\n"
    if cube_fitter.extinction_curve != "decompose"
        msg *= "τ_9.7: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t Limits: " *
            "($(@sprintf "%.2f" continuum.τ_97.limits[1]), $(@sprintf "%.2f" continuum.τ_97.limits[2]))" * 
            (continuum.τ_97.locked ? " (fixed)" : "") * "\n"
        pᵢ += 1
    else
        msg *= "N_oli: \t\t\t\t $(@sprintf "%.2g" popt[pᵢ]) +/- $(@sprintf "%.2g" perr[pᵢ]) [-] \t Limits: " *
            "($(@sprintf "%.2g" continuum.N_oli.limits[1]), $(@sprintf "%.2g" continuum.N_oli.limits[2]))" * 
            (continuum.N_oli.locked ? " (fixed)" : "") * "\n"
        msg *= "N_pyr: \t\t\t\t $(@sprintf "%.2g" popt[pᵢ+1]) +/- $(@sprintf "%.2g" perr[pᵢ+1]) [-] \t Limits: " *
            "($(@sprintf "%.2g" continuum.N_pyr.limits[1]), $(@sprintf "%.2g" continuum.N_pyr.limits[2]))" * 
            (continuum.N_pyr.locked ? " (fixed)" : "") * "\n"
        msg *= "N_for: \t\t\t\t $(@sprintf "%.2g" popt[pᵢ+2]) +/- $(@sprintf "%.2g" perr[pᵢ+2]) [-] \t Limits: " *
            "($(@sprintf "%.2g" continuum.N_for.limits[1]), $(@sprintf "%.2g" continuum.N_for.limits[2]))" * 
            (continuum.N_for.locked ? " (fixed)" : "") * "\n"
        pᵢ += 3
    end
    msg *= "τ_ice: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ]) +/- $(@sprintf "%.2f" perr[pᵢ]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.τ_ice.limits[1]), $(@sprintf "%.2f" continuum.τ_ice.limits[2]))" *
        (continuum.τ_ice.locked ? " (fixed)" : "") * "\n"
    msg *= "τ_ch: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+1]) +/- $(@sprintf "%.2f" perr[pᵢ+1]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.τ_ch.limits[1]), $(@sprintf "%.2f" continuum.τ_ch.limits[2]))" *
        (continuum.τ_ch.locked ? " (fixed)" : "") * "\n"
    msg *= "β: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+2]) +/- $(@sprintf "%.2f" perr[pᵢ+2]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.β.limits[1]), $(@sprintf "%.2f" continuum.β.limits[2]))" * 
        (continuum.β.locked ? " (fixed)" : "") * "\n"
    msg *= "Cf: \t\t\t\t $(@sprintf "%.2f" popt[pᵢ+3]) +/- $(@sprintf "%.2f" perr[pᵢ+3]) [-] \t Limits: " *
        "($(@sprintf "%.2f" continuum.Cf.limits[1]), $(@sprintf "%.2f" continuum.Cf.limits[2]))" * 
        (continuum.Cf.locked ? " (fixed)" : "") * "\n"
    msg *= "\n"
    pᵢ += 4
    msg *= "\n#> ABSORPTION FEATURES <#\n"
    for (j, ab) ∈ enumerate(cube_fitter.abs_features.names)
        msg *= "$(ab)_τ:\t\t\t $(@sprintf "%.5f" popt[pᵢ]) +/- $(@sprintf "%.5f" perr[pᵢ]) [x norm] \t Limits: " *
            "($(@sprintf "%.3f" cube_fitter.abs_taus[j].limits[1]), $(@sprintf "%.3f" cube_fitter.abs_taus[j].limits[2]))\n"
        msg *= "$(ab)_mean:  \t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1]) μm \t Limits: " *
            "($(@sprintf "%.3f" cube_fitter.abs_features.mean[j].limits[1]), $(@sprintf "%.3f" cube_fitter.abs_features.mean[j].limits[2]))" * 
            (cube_fitter.abs_features.mean[j].locked ? " (fixed)" : "") * "\n"
        msg *= "$(ab)_fwhm:  \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) μm \t Limits: " *
            "($(@sprintf "%.3f" cube_fitter.abs_features.fwhm[j].limits[1]), $(@sprintf "%.3f" cube_fitter.abs_features.fwhm[j].limits[2]))" * 
            (cube_fitter.abs_features.fwhm[j].locked ? " (fixed)" : "") * "\n"
        msg *= "$(ab)_asym: \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3]) [-] \t Limits: " *
            "($(@sprintf "%.3f" cube_fitter.abs_features.asym[j].limits[1]), $(@sprintf "%.3f" cube_fitter.abs_features.asym[j].limits[2]))" *
            (cube_fitter.abs_features.asym[j].locked ? " (fixed)" : "") * "\n"
        msg *= "\n"
        pᵢ += 4
    end 
    if cube_fitter.fit_sil_emission
        msg *= "\n#> HOT DUST <#\n"
        msg *= "Hot_dust_amp: \t\t\t $(@sprintf "%.3g" popt[pᵢ]) +/- $(@sprintf "%.3g" perr[pᵢ]) [-] \t Limits: (0, Inf)\n"
        msg *= "Hot_dust_temp: \t\t\t $(@sprintf "%.0f" popt[pᵢ+1]) +/- $(@sprintf "%.0f" perr[pᵢ+1]) K \t Limits: " *
            "($(@sprintf "%.0f" continuum.T_hot.limits[1]), $(@sprintf "%.0f" continuum.T_hot.limits[2]))" *
            (continuum.T_hot.locked ? " (fixed)" : "") * "\n"
        msg *= "Hot_dust_frac: \t\t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) [-] \t Limits: " *
            "($(@sprintf "%.3f" continuum.Cf_hot.limits[1]), $(@sprintf "%.3f" continuum.Cf_hot.limits[2]))" *
            (continuum.Cf_hot.locked ? " (fixed)" : "") * "\n"
        msg *= "Hot_dust_τ: \t\t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3]) [-] \t Limits: " *
            "($(@sprintf "%.3f" continuum.τ_warm.limits[1]), $(@sprintf "%.3f" continuum.τ_warm.limits[2]))" *
            (continuum.τ_warm.locked ? " (fixed)" : "") * "\n"
        msg *= "Cold_dust_τ: \t\t\t $(@sprintf "%.3f" popt[pᵢ+4]) +/- $(@sprintf "%.3f" perr[pᵢ+4]) [-] \t Limits: " *
            "($(@sprintf "%.3f" continuum.τ_cold.limits[1]), $(@sprintf "%.3f" continuum.τ_cold.limits[2]))" *
            (continuum.τ_cold.locked ? " (fixed)" : "") * "\n"
        msg *= "Hot_dust_peak: \t\t\t $(@sprintf "%.3f" popt[pᵢ+5]) +/- $(@sprintf "%.3f" perr[pᵢ+5]) [-] \t Limits: " *
            "($(@sprintf "%.3f" continuum.sil_peak.limits[1]), $(@sprintf "%.3f" continuum.sil_peak.limits[2]))" *
            (continuum.sil_peak.locked ? " (fixed)" : "") * "\n"
        pᵢ += 6
    end
    msg *= "\n#> TEMPLATES <#\n"
    if !cube_fitter.fit_temp_multexp
        for (q, tp) ∈ enumerate(cube_fitter.template_names)
            for qi ∈ 1:cube_fitter.n_channels
                msg *= "$(tp)_amp_$qi:\t\t\t $(@sprintf "%.5f" popt[pᵢ]) +/- $(@sprintf "%.5f" perr[pᵢ]) [x norm] \t Limits: (0, 1)\n"
                pᵢ += 1
            end
        end
    else
        for q ∈ 1:4
            msg *= "temp_multexp_amp$q:\t\t\t $(@sprintf "%.5f" popt[pᵢ]) +/- $(@sprintf "%.5f" perr[pᵢ]) [x norm] \t Limits: (0, 1)\n"
            msg *= "temp_multexp_ind$q:\t\t\t $(@sprintf "%.5f" popt[pᵢ+1]) +/- $(@sprintf "%.5f" perr[pᵢ+1]) [-] \t Limits: (0, 1)\n"
            pᵢ += 2
        end
    end
    msg *= "\n#> DUST FEATURES <#\n"
    for (j, df) ∈ enumerate(cube_fitter.dust_features.names)
        msg *= "$(df)_amp:\t\t\t $(@sprintf "%.5f" popt[pᵢ]) +/- $(@sprintf "%.5f" perr[pᵢ]) [x norm] \t Limits: " *
            "(0, $(@sprintf "%.5f" (nanmaximum(I) / exp(-continuum.τ_97.limits[1]))))\n"
        msg *= "$(df)_mean:  \t\t $(@sprintf "%.3f" popt[pᵢ+1]) +/- $(@sprintf "%.3f" perr[pᵢ+1]) μm \t Limits: " *
            "($(@sprintf "%.3f" cube_fitter.dust_features.mean[j].limits[1]), $(@sprintf "%.3f" cube_fitter.dust_features.mean[j].limits[2]))" * 
            (cube_fitter.dust_features.mean[j].locked ? " (fixed)" : "") * "\n"
        msg *= "$(df)_fwhm:  \t\t $(@sprintf "%.3f" popt[pᵢ+2]) +/- $(@sprintf "%.3f" perr[pᵢ+2]) μm \t Limits: " *
            "($(@sprintf "%.3f" cube_fitter.dust_features.fwhm[j].limits[1]), $(@sprintf "%.3f" cube_fitter.dust_features.fwhm[j].limits[2]))" * 
            (cube_fitter.dust_features.fwhm[j].locked ? " (fixed)" : "") * "\n"
        if cube_fitter.dust_features.profiles[j] == :PearsonIV
            msg *= "$(df)_index:  \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3]) μm \t Limits: " *
                "($(@sprintf "%.3f" cube_fitter.dust_features.index[j].limits[1]), $(@sprintf "%.3f" cube_fitter.dust_features.index[j].limits[2]))" * 
                (cube_fitter.dust_features.index[j].locked ? " (fixed)" : "") * "\n"
            msg *= "$(df)_cutoff:  \t\t $(@sprintf "%.3f" popt[pᵢ+4]) +/- $(@sprintf "%.3f" perr[pᵢ+4]) μm \t Limits: " *
                "($(@sprintf "%.3f" cube_fitter.dust_features.cutoff[j].limits[1]), $(@sprintf "%.3f" cube_fitter.dust_features.cutoff[j].limits[2]))" * 
                (cube_fitter.dust_features.cutoff[j].locked ? " (fixed)" : "") * "\n"
            pᵢ += 2
        else
            msg *= "$(df)_asym:  \t\t $(@sprintf "%.3f" popt[pᵢ+3]) +/- $(@sprintf "%.3f" perr[pᵢ+3]) [-] \t Limits: " *
                "($(@sprintf "%.3f" cube_fitter.dust_features.asym[j].limits[1]), $(@sprintf "%.3f" cube_fitter.dust_features.asym[j].limits[2]))" * 
                (cube_fitter.dust_features.asym[j].locked ? " (fixed)" : "") * "\n"
            pᵢ += 1
        end
        msg *= "\n"
        pᵢ += 3
    end
    msg *= "######################################################################"
    @debug msg

    msg

end
