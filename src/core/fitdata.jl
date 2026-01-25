

function mask_emission_lines(λ::Vector{S}, mask_regions::Vector{Tuple{T,T}}) where {S<:Quantity, T<:Quantity}
    mask = falses(length(λ))
    # manual regions that wish to be masked out
    for region in mask_regions
        mask[region[1] .< λ .< region[2]] .= 1
    end
    mask
end


"""
    fill_bad_pixels(cube_fitter, I, σ, template)

Helper function to fill in NaNs/Infs in a 1D intensity/error vector and 2D templates vector.
"""
function fill_bad_pixels(I::Vector{<:Number}, σ::Vector{<:Number}, templates::Union{Array{<:Number,2},Nothing})

    # Edge cases
    if !isfinite(I[1])
        i = findfirst(isfinite, I)
        I[1] = I[i]
    end
    if !isfinite(I[end])
        i = findfirst(isfinite, I[end:-1:1])
        I[end] = I[i]
    end
    if !isfinite(σ[1])
        i = findfirst(isfinite, σ)
        σ[1] = σ[i]
    end
    if !isfinite(σ[end])
        i = findfirst(isfinite, σ[end:-1:1])
        σ[end] = σ[i]
    end
    if !isnothing(templates)
        for s in axes(templates, 2)
            if !isfinite(templates[1,s])
                i = findfirst(isfinite, templates[:,s])
                templates[1,s] = templates[i,s]
            end
            if !isfinite(templates[end,s])
                i = findfirst(isfinite, templates[end:-1:1,s])
                templates[end,s] = templates[i,s]
            end
        end
    end

    # Replace with the average of the points to the left and right
    bad = findall(.~isfinite.(I) .| .~isfinite.(σ))
    l = length(I)
    for badi in bad
        lind = findfirst(isfinite, I[max(badi-1,1):-1:1])
        rind = findfirst(isfinite, I[min(badi+1,l):end])
        I[badi] = (I[max(badi-1,1):-1:1][lind] + I[min(badi+1,l):end][rind]) / 2
        lind = findfirst(isfinite, σ[max(badi-1,1):-1:1])
        rind = findfirst(isfinite, σ[min(badi+1,l):end])
        σ[badi] = (σ[max(badi-1,1):-1:1][lind] + σ[min(badi+1,l):end][rind]) / 2
    end
    @assert all(isfinite.(I) .& isfinite.(σ)) "Error: Non-finite values found in the summed intensity/error arrays!"
    if !isnothing(templates)
        for s in axes(templates, 2)
            bad = findall(.~isfinite.(templates[:, s]))
            l = length(templates[:, s])
            for badi in bad
                lind = findfirst(isfinite, templates[max(badi-1,1):-1:1, s])
                rind = findfirst(isfinite, templates[min(badi+1,l):end, s])
                templates[badi, s] = (templates[max(badi-1,1):-1:1, s][lind] + templates[min(badi+1,l):end, s][rind]) / 2
            end
        end
        @assert all(isfinite.(templates)) "Error: Non-finite values found in the summed template arrays!"
    end

    return I, σ, templates
end


"""
    continuum_cubic_spline(λ, I, mask_regions)

Mask out the emission lines in a given spectrum using `mask_emission_lines` and replace them with
a coarse cubic spline fit to the continuum, using wide knots to avoid reinterpolating the lines or
noise. Returns the line mask, spline-interpolated I, and spline-interpolated σ.

# Arguments
- `λ` The wavelength vector of the spectrum
- `I`: The flux vector of the spectrum
- `mask_regions`: The emission line regions to be masked out 

See also [`mask_emission_lines`](@ref)
"""
function continuum_cubic_spline(λ::Vector{<:Quantity}, I::Vector{S}, σ::Vector{S}, 
    overrides::Vector{Tuple{T,T}}=Vector{Tuple{Quantity,Quantity}}(); do_err::Bool=true,
    scale::Int=7) where {S<:Number,T<:Quantity}

    # Mask out emission lines so that they aren't included in the continuum fit
    mask_lines = mask_emission_lines(λ, overrides)

    _λ = ustrip.(λ)
    _I = ustrip.(I)
    if do_err
        _σ = ustrip.(σ)
    end

    # Make coarse knots to perform a smooth interpolation across any gaps of NaNs in the data
    λknots = _λ[.~mask_lines][1+scale:scale:end-scale]
    # Remove any knots that happen to fall within a masked pixel
    good = []
    for i ∈ eachindex(λknots)
        _, ind = findmin(abs.(_λ .- λknots[i]))
        if ~mask_lines[ind]
            append!(good, [i])
        end
    end
    λknots = λknots[good]
    if length(λknots) == 0
        λknots = [nanmedian(_λ[.~mask_lines])]
    end
    @debug "Performing cubic spline continuum fit with $(length(λknots)) knots"

    # Do a full cubic spline interpolation of the data
    I_spline = Spline1D(_λ[.~mask_lines], _I[.~mask_lines], λknots, k=3, bc="extrapolate").(_λ) .* unit(I[1])
    # Linear interpolation over the lines
    I_spline[mask_lines] .= Spline1D(_λ[.~mask_lines], _I[.~mask_lines], λknots, k=1, bc="extrapolate").(_λ[mask_lines]) .* unit(I[1])

    if do_err
        σ_spline = Spline1D(_λ[.~mask_lines], _σ[.~mask_lines], λknots, k=3, bc="extrapolate").(_λ) * unit(σ[1])
        σ_spline[mask_lines] .= Spline1D(_λ[.~mask_lines], _σ[.~mask_lines], λknots, k=1, bc="extrapolate").(_λ[mask_lines]) .* unit(σ[1])

        return mask_lines, I_spline, σ_spline
    end

    mask_lines, I_spline
end


"""
    calculate_statistical_errors(I, I_spline, mask)

Uses the residuals with a cubic spline fit to calculate statistical errors
purely based on the scatter in the data.
"""
function calculate_statistical_errors(I::Vector{S}, I_spline::Vector{S}, mask::BitVector) where {S<:Number}

    l_mask = sum(.~mask)
    # Statistical uncertainties based on the local RMS of the residuals with a cubic spline fit
    σ_stat = zeros(eltype(I), l_mask)
    window_size = min(60, l_mask)
    for i in 1:l_mask
        indices = sortperm(abs.((1:l_mask) .- i))[1:window_size]
        σ_stat[i] = std(I[.~mask][indices] .- I_spline[.~mask][indices])
    end
    # We insert at the locations of the lines since the cubic spline does not include them
    l_all = length(I)
    line_inds = (1:l_all)[mask]
    for line_ind ∈ line_inds
        insert!(σ_stat, line_ind, σ_stat[max(line_ind-1, 1)])
    end
    @debug "Statistical uncertainties: ($(σ_stat[1]) - $(σ_stat[end]))"

    σ_stat

end


# Helper function for estimating PAH template amplitude when PAH templates are not actually
# used during the fit.
function estimate_pah_template_amplitude(cube_fitter::CubeFitter, λ::Vector, comps::Dict)
    pahtemp = zeros(length(λ))
    for (k, dcomplex) in enumerate(model(cube_fitter).dust_features.profiles)   # <- iterates over PAH complexes
        for j in 1:length(dcomplex)                                             # <- iterates over each individual component
            pahtemp .+= comps["dust_feat_$(k)_$(j)"]
        end
    end
    repeat([maximum(pahtemp)/2], 2)
end


# # Helper function to get the continuum to be subtracted for the line fit
function get_continuum_for_line_fit(spaxel::Spaxel, cube_fitter::CubeFitter, I_cont::Vector{<:Real},
    comps_cont::Dict)

    fopt = fit_options(cube_fitter)
    @assert spaxel.normalized

    # default to using the actual fit continuum
    line_cont = copy(I_cont)
    if fopt.subtract_cubic_spline
        # do cubic spline fit to the residuals of the data
        _, notemp_cont = continuum_cubic_spline(spaxel.λ, spaxel.I .- I_cont, zeros(eltype(line_cont), length(line_cont)), 
            cube_fitter.linemask_overrides; do_err=false)
        line_cont .+= notemp_cont
    end

    line_cont
end


# Helper function for getting the total integrated intensity/error/solid angle over the whole FOV
function get_total_integrated_intensities(cube_fitter::CubeFitter; shape::Union{Tuple,Nothing}=nothing)

    @info "Integrating spectrum across the whole cube..."
    Iunit = unit(cube_fitter.cube.I[1])
    Imasked = copy(cube_fitter.cube.I)
    Imasked[cube_fitter.cube.mask] .*= NaN
    σmasked = copy(cube_fitter.cube.σ)
    σmasked[cube_fitter.cube.mask] .*= NaN

    I = sumdim(ustrip.(Imasked), (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2)) .* Iunit
    σ = sqrt.(sumdim(ustrip.(σmasked).^2, (1,2))) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2)) .* Iunit
    area_sr = cube_fitter.cube.Ω .* sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2))
    # sometimes the mask can cause area_sr to be 0 at some points, which is bad
    area_sr[area_sr .== 0.0u"sr"] .= median(area_sr[area_sr .> 0.0u"sr"])

    templates = zeros(eltype(I), size(I)..., cube_fitter.n_templates)
    for s in 1:cube_fitter.n_templates
        templates[:,s] .= sumdim(ustrip.(cube_fitter.templates[:,:,:,s]), (1,2)) ./ sumdim(Array{Int}(.~cube_fitter.cube.mask), (1,2)) .* Iunit
    end

    # Mask out the chip gaps in NIRSPEC observations 
    if fit_options(cube_fitter).nirspec_mask_chip_gaps
        λobs = cube_fitter.cube.λ .* (1 .+ cube_fitter.z)
        for chip_gap in chip_gaps_nir
            mask = (chip_gap[1] .< λobs .< chip_gap[2])
            I[mask] .= NaN .* Iunit
            σ[mask] .= NaN .* Iunit
            edges = findall(diff(mask) .≠ 0)
            if length(edges) > 0
                area_sr[mask] .= area_sr[max(edges[1]-1,1)]
            end
            for s in 1:cube_fitter.n_templates
                templates[mask,s] .= NaN .* Iunit
            end
        end
    end
    I, σ, templates = fill_bad_pixels(I, σ, templates)

    if !isnothing(shape)
        I = reshape(I, shape...)
        σ = reshape(σ, shape...)
        area_sr = reshape(area_sr, shape...)
        templates = reshape(templates, shape..., cube_fitter.n_templates)
    end

    I, σ, templates, area_sr
end


# Helper function for getting the total integrated intensity/error/solid angle over an aperture
function get_aperture_integrated_intensities(cube_fitter::CubeFitter, shape::Tuple, 
    aperture::Vector{<:Aperture.AbstractAperture})

    # If using an aperture, overwrite the cube_data object with the quantities within
    # the aperture, which are calculated here.
    # Prepare the 1D arrays
    I = zeros(eltype(cube_fitter.cube.I), shape)
    σ = zeros(eltype(cube_fitter.cube.σ), shape)
    templates = zeros(eltype(cube_fitter.templates), shape..., cube_fitter.n_templates)
    area_sr = zeros(typeof(cube_fitter.cube.Ω), shape)
    Funit = unit(I[1])*unit(cube_fitter.cube.Ω)

    @info "Performing aperture photometry to get an integrated spectrum..."
    for z ∈ 1:shape[3]

        # Sum up the FLUX within the aperture
        Fz = cube_fitter.cube.I[:, :, z] .* cube_fitter.cube.Ω
        e_Fz = cube_fitter.cube.σ[:, :, z] .* cube_fitter.cube.Ω
        # Zero out the masked spaxels
        Fz[cube_fitter.cube.mask[:, :, z]] .= 0.0*Funit
        e_Fz[cube_fitter.cube.mask[:, :, z]] .= 0.0*Funit
        # Perform the aperture photometry
        (_, _, F_ap, eF_ap) = photometry(aperture[z], Fz, e_Fz)

        # Convert back to intensity by dividing out the aperture area
        area_sr[1,1,z] = get_area(aperture[z]) * cube_fitter.cube.Ω
        I[1,1,z] = F_ap / area_sr[1,1,z]
        σ[1,1,z] = eF_ap / area_sr[1,1,z]

        # repeat for templates
        for s in 1:cube_fitter.n_templates
            Ftz = cube_fitter.templates[:, :, z, s] .* cube_fitter.cube.Ω
            Ftz[.~isfinite.(Ftz) .| cube_fitter.cube.mask[:, :, z]] .= 0.0*Funit
            Ft_ap = photometry(aperture[z], Ftz).aperture_sum
            templates[1,1,z,s] = Ft_ap / area_sr[1,1,z]
        end
    end

    # Mask out the chip gaps in NIRSPEC observations 
    if fit_options(cube_fitter).nirspec_mask_chip_gaps
        λobs = cube_fitter.cube.λ .* (1 .+ cube_fitter.z)
        for chip_gap in chip_gaps_nir
            mask = (chip_gap[1] .< λobs .< chip_gap[2])
            I[1,1,mask] .= NaN .* unit(I[1])
            σ[1,1,mask] .= NaN .* unit(I[1])
            edges = findall(diff(mask) .≠ 0)
            if length(edges) > 0
                area_sr[1,1,mask] .= area_sr[1,1,max(edges[1]-1,1)]
            end
            for s in 1:cube_fitter.n_templates
                templates[1,1,mask,s] .= NaN .* unit(I[1])
            end
        end
        I, σ, templates = fill_bad_pixels(I[1,1,:], σ[1,1,:], templates[1,1,:,:])
        I = reshape(I, shape...)
        σ = reshape(σ, shape...)
        templates = reshape(templates, shape..., cube_fitter.n_templates)
    end

    I, σ, templates, area_sr
end


# Helper function for getting the number of profiles and number of parameters for each profile
# for a given emission line
function get_line_nprof_ncomp(cube_fitter::CubeFitter, i::Integer)

    profiles = model(cube_fitter).lines.profiles[i]
    n_prof = length(profiles)
    pcomps = Int[]
    for j in 1:n_prof
        pc = 3
        if profiles[j].profile == :GaussHermite
            pc += 2
        elseif profiles[j].profile == :Voigt
            pc += 1
        end
        push!(pcomps, pc)
    end

    n_prof, pcomps
end


"""
    create_cube_data(cube_fitter, shape)

Makes a named tuple object that holds the wavelengths, intensities, errors,
templates, and solid angles.
"""
function create_cube_data(cube_fitter::CubeFitter, shape::Tuple)

    cube_data = (λ=cube_fitter.cube.λ, I=cube_fitter.cube.I, σ=cube_fitter.cube.σ, 
        templates=cube_fitter.n_templates > 0 ? cube_fitter.templates : Array{Float64}(undef, shape..., 0),
        area_sr=cube_fitter.cube.Ω .* ones(shape))

    n_bins = 0
    vorbin = !isnothing(cube_fitter.cube.voronoi_bins)
    if vorbin
        # Reformat cube data as a 2D array with the first axis slicing each voronoi bin
        n_bins = maximum(cube_fitter.cube.voronoi_bins)
        I_vorbin = zeros(eltype(cube_data.I), n_bins, shape[3])
        σ_vorbin = zeros(eltype(cube_data.σ), n_bins, shape[3])
        area_vorbin = zeros(eltype(cube_data.area_sr), n_bins, shape[3])
        template_vorbin = zeros(eltype(cube_data.templates), n_bins, shape[3], cube_fitter.n_templates)
        for n in 1:n_bins
            w = findall(cube_fitter.cube.voronoi_bins .== n)
            # sum up the spaxels in the voronoi bin, ignoring where they are masked out
            npix_vorbin = zeros(Int, shape[3])
            for wi in w
                mask_vorbin = cube_fitter.cube.mask[wi, :]
                npix_vorbin .+= .~mask_vorbin
                I_vorbin[n, .~mask_vorbin] .+= ustrip.(cube_data.I[wi, .~mask_vorbin]) .* unit(cube_data.I[1])
                σ_vorbin[n, .~mask_vorbin] .+= ustrip.(cube_data.σ[wi, .~mask_vorbin].^2) .* unit(cube_data.σ[1])
                for s in 1:cube_fitter.n_templates
                    template_vorbin[n, :, s] .+= ustrip.(cube_fitter.templates[wi, :, s]) .* unit(cube_fitter.templates[n, 1, s])
                end
            end
            npix_vorbin[iszero.(npix_vorbin)] .= 1
            area_vorbin[n, :] .= npix_vorbin .* cube_fitter.cube.Ω
            I_vorbin[n, :] ./= npix_vorbin
            I_vorbin[n, .~isfinite.(I_vorbin[n, :])] .= 0*unit(cube_data.I[1])
            σ_vorbin[n, :] .= sqrt.(ustrip.(σ_vorbin[n, :])) ./ npix_vorbin .* unit(cube_data.σ[1])
            σ_vorbin[n, .~isfinite.(σ_vorbin[n, :])] .= nanmedian(ustrip.(σ_vorbin[n, isfinite.(σ_vorbin[n, :])]))*unit(cube_data.σ[1])
            for s in 1:cube_fitter.n_templates
                template_vorbin[n, :, s] ./= length(w)
                template_vorbin[n, .~isfinite.(template_vorbin), s] .= 0*unit(cube_fitter.templates[n, 1, s])
            end
        end
        cube_data = (λ=cube_fitter.cube.λ, I=I_vorbin, σ=σ_vorbin, area_sr=area_vorbin, templates=template_vorbin)
    end

    cube_data, vorbin, n_bins
end


"""
    create_cube_data_nuctemp(cube_fitter, shape)

Makes a named tuple object that holds the wavelengths, intensities, errors,
templates, and solid angles for a nuclear template fit.
"""
function create_cube_data_nuctemp(cube_fitter::CubeFitter, shape::Tuple)

    cube = cube_fitter.cube
    data2d = dropdims(nansum(ustrip.(cube.I), dims=3), dims=3)
    data2d[.~isfinite.(data2d)] .= 0.
    _, mx = findmax(data2d)

    I = zeros(eltype(cube.I), shape)
    σ = zeros(eltype(cube.σ), shape)
    templates = zeros(eltype(cube_fitter.templates), shape..., cube_fitter.n_templates)
    area_sr = zeros(typeof(cube.Ω), shape)

    # Take the brightest spaxel
    I[1,1,:] .= cube.I[mx,:]
    σ[1,1,:] .= cube.σ[mx,:]
    for s in 1:cube_fitter.n_templates
        templates[1,1,:,s] .= cube_fitter.templates[mx, :, s]
    end
    area_sr[1,1,:] .= cube_fitter.cube.Ω
    cube_data = (λ=cube_fitter.cube.λ, I=I, σ=σ, templates=templates, area_sr=area_sr)

    cube_data, mx
end


"""
    create_cube_data_postnuctemp(cube_fitter, agn_templates)

Makes a named tuple object that holds the wavelengths, intensities, errors,
templates, and solid angles for a post-nuclear-template-fit fit.
"""
function create_cube_data_postnuctemp(cube_fitter::CubeFitter, agn_templates::Array{<:QSIntensity,3})
    cube = cube_fitter.cube

    # Get the AGN model over the whole cube
    if size(agn_templates)[1:2] == (1,1)
        I_agn = agn_templates
    else
        I_agn = nansum(ustrip.(agn_templates), dims=(1,2)) ./ nansum(.~cube.mask, dims=(1,2)) .* unit(agn_templates[1])
    end
    # errors will be overwritten by statistical errors
    σ_agn = nanmedian(ustrip.(I_agn)) .* ones(Float64, size(I_agn)) .* unit(agn_templates[1])  
    templates = Array{eltype(I_agn),4}(undef, size(I_agn)..., 0)
    area_sr = cube_fitter.cube.Ω .* nansum(.~cube.mask, dims=(1,2))

    I, σ, t_out = fill_bad_pixels(I_agn[1,1,:], σ_agn[1,1,:], templates[1,1,:,:])

    # Create a "spaxel" and fill in the bad data
    I_agn[1,1,:] .= I
    σ_agn[1,1,:] .= σ
    templates[1,1,:,:] .= t_out

    cube_data = (λ=cube_fitter.cube.λ, I=I_agn, σ=σ_agn, templates=templates, area_sr=area_sr)

    cube_data
end


# Calculate stellar masses, ages, and metallicities based on a fit
function calculate_stellar_parameters(cube_fitter::CubeFitter, norms::Dict, N::Number)

    # reshape SSPs into a 3D array with axes (wavelength, age, logz)
    p_dims = (length(cube_fitter.ssps.ages), length(cube_fitter.ssps.logzs))
    # rest-frame transform 
    if eltype(cube_fitter.cube.I) <: QPerFreq
        restframe_factor = 1 + cube_fitter.z
    else
        restframe_factor = 1 / (1 + cube_fitter.z)
    end

    stellar_N = norms["continuum.stellar_norm"]
    weights = reshape(norms["continuum.stellar_weights"], p_dims...)

    if fit_options(cube_fitter).stellar_template_type == "ssp"
        f = N / stellar_N   
        unit_check(unit(f), u"Msun")  # should have units of Msun
        # stellar mass that each SSP template contributes
        masses = weights .* f .* restframe_factor
        # total mass
        mtot = sum(masses)
        mfracs = masses ./ mtot

        # renormalize weights so they represent fractions of the bolometric luminosity
        Lbol = reshape(nansum(ustrip.(cube_fitter.ssps.templates), dims=1), p_dims...) 
        wl = weights .* Lbol
        wl ./= sum(wl)

        # detect "peaks" in the age/logz axes and report their values
        w = weights ./ sum(weights)
        minds = findlocalmaxima(w)
        # cut off after 10 so we dont get TOO excessive here
        if length(minds) > 10
            minds = minds[1:10]
        end
        ages = [cube_fitter.ssps.ages[m.I[1]] for m in minds]
        logzs = [cube_fitter.ssps.logzs[m.I[2]] for m in minds]
    else
        # cannot constrain masses, ages, or metallicities if not using SSPs
        mtot = NaN * u"Msun"
        mfracs = weights .* NaN
        wl = weights .* NaN 
        ages = Vector{eltype(cube_fitter.ssps.ages)}()
        logzs = Vector{eltype(cube_fitter.ssps.logzs)}()
    end

    StellarResult(stellar_N, mtot, mfracs, wl, weights, ages, logzs)
end


# Helper function to collect the results of a fit into a parameter vector, error vector,
# intensity vector as a function of wavelength, and a comps dict giving individual components
# of the final model
function collect_cont_fit_results(res::CMPFit.Result, pfix_tied::Vector{<:Real}, plock_tied::BitVector, 
    punits::Vector{<:Unitful.Units}, tied_pairs::Vector{<:Tuple}, tied_indices::Vector{<:Integer}, n_tied::Integer, 
    spaxel::Spaxel, cube_fitter::CubeFitter; bootstrap_iter::Bool=false, step1::Bool=false)

    fopt = fit_options(cube_fitter)
    popt = rebuild_full_parameters(res.param, pfix_tied, plock_tied, tied_pairs, tied_indices, n_tied)
    perr = zeros(Float64, length(popt))
    if !bootstrap_iter
        perr = rebuild_full_parameters(res.perror, zeros(length(pfix_tied)), plock_tied, tied_pairs, tied_indices, n_tied)
    end
    popt = popt .* punits
    perr = perr .* punits

    @debug "Best fit continuum parameters: \n $popt"
    @debug "Continuum parameter errors: \n $perr"

    result_stellar = nothing
    if fopt.fit_stellar_continuum && !step1
        _, _, norms = model_continuum(spaxel, spaxel.N, ustrip.(popt), punits, cube_fitter, fopt.use_pah_templates, true, true, true)
        result_stellar = calculate_stellar_parameters(cube_fitter, norms, spaxel.N)
    end
    spaxel_model = copy(spaxel)
    if (length(cube_fitter.spectral_region.gaps) > 0) && !step1
        spaxel_model = get_model_spaxel(cube_fitter, spaxel, result_stellar)
    else
        add_stellar_weights!(spaxel_model, result_stellar)
    end
    I_model, comps, norms = model_continuum(spaxel_model, spaxel_model.N, ustrip.(popt), punits, cube_fitter, 
        fopt.use_pah_templates, spaxel_model == spaxel, true, true)

    popt, perr, I_model, comps, norms, result_stellar, spaxel_model
end


# Alternative dispatch for the collect_fit_results function for step 2 of the multistep fitting procedure
function collect_cont_fit_results(res_1::CMPFit.Result, p1fix_tied::Vector{<:Real}, lock_1_tied::BitVector, 
    punit_1::Vector{<:Unitful.Units}, tied_pairs::Vector{<:Tuple}, tied_indices::Vector{<:Integer}, n_tied_1::Integer, 
    res_2::CMPFit.Result, p2fix::Vector{<:Real}, lock_2::BitVector, punit_2::Vector{<:Unitful.Units}, n_free_1::Integer, 
    n_free_2::Integer, cube_fitter::CubeFitter, spaxel::Spaxel; bootstrap_iter::Bool=false)

    popt_1 = rebuild_full_parameters(res_1.param, p1fix_tied, lock_1_tied, tied_pairs, tied_indices, n_tied_1)

    # remove the PAH template amplitudes
    fopt = fit_options(cube_fitter)
    pahtemp = popt_1[end-1:end]
    popt_1 = popt_1[1:end-2]
    perr_1 = zeros(length(popt_1))
    if !bootstrap_iter
        perr_1 = rebuild_full_parameters(res_1.perror, zeros(length(p1fix_tied)), lock_1_tied, tied_pairs, tied_indices, n_tied_1)
        perr_1 = perr_1[1:end-2]
    end
    popt_1 = popt_1 .* punit_1[1:end-2]
    perr_1 = perr_1 .* punit_1[1:end-2]

    popt_2 = rebuild_full_parameters(res_2.param, p2fix, lock_2)
    perr_2 = zeros(length(lock_2))
    if !bootstrap_iter
        perr_2 = rebuild_full_parameters(res_2.perror, zeros(length(p2fix)), lock_2)
    end
    popt_2 = popt_2 .* punit_2
    perr_2 = perr_2 .* punit_2

    popt = [popt_1; popt_2]
    perr = [perr_1; perr_2]

    @debug "Best fit continuum parameters: \n $popt"
    @debug "Continuum parameter errors: \n $perr"

    n_free = n_free_1 + n_free_2 - 2

    # Create the full model, again only if not bootstrapping
    result_stellar = nothing
    if fopt.fit_stellar_continuum
        _, _, norms = model_continuum(spaxel, spaxel.N, ustrip.(popt), [punit_1[1:end-2]; punit_2], cube_fitter, 
            false, true, true, true)
        result_stellar = calculate_stellar_parameters(cube_fitter, norms, spaxel.N)
    end
    spaxel_model = copy(spaxel)
    if length(cube_fitter.spectral_region.gaps) > 0
        spaxel_model = get_model_spaxel(cube_fitter, spaxel, result_stellar)
    else
        add_stellar_weights!(spaxel_model, result_stellar)
    end
    I_model, comps, norms = model_continuum(spaxel_model, spaxel_model.N, ustrip.(popt), [punit_1[1:end-2]; punit_2], cube_fitter, 
        false, spaxel_model == spaxel, true, true)

    popt, perr, n_free, pahtemp, I_model, comps, norms, result_stellar, spaxel_model
end


# Alternative dispatch for the collect_fit_results function for the line fitting procedure
function collect_line_fit_results(res::CMPFit.Result, pfix_tied::Vector{<:Real}, param_lock_tied::BitVector,
    punits::Vector{<:Unitful.Units}, tied_pairs::Vector{<:Tuple}, tied_indices::Vector{<:Integer}, n_tied::Integer, 
    spaxel_model::Spaxel, extinction_curve::Vector{<:Real}, cube_fitter::CubeFitter; bootstrap_iter::Bool=false)

    # Get the results and errors
    popt = rebuild_full_parameters(res.param, pfix_tied, param_lock_tied, tied_pairs, tied_indices, n_tied)

    # Dont both with uncertainties if bootstrapping
    perr = zeros(length(popt))
    if !bootstrap_iter
        perr = rebuild_full_parameters(res.perror, zeros(length(pfix_tied)), param_lock_tied, tied_pairs, tied_indices, n_tied)
    end
    popt = popt .* punits
    perr = perr .* punits

    @debug "Best fit line parameters: \n $popt"
    @debug "Line parameter errors: \n $perr"

    # Final optimized fit
    I_model, comps = model_line_residuals(spaxel_model, ustrip.(popt), punits, model(cube_fitter).lines, cube_fitter.lsf, 
        extinction_curve, trues(length(spaxel_model.λ)), true)

    popt, perr, I_model, comps
end


# Alternative dispatch for the collect_fit_results function for the joint continuum+line fitting procedure
function collect_fit_results(res::CMPFit.Result, pfix_cont_tied::Vector{<:Real}, lock_cont_tied::BitVector, 
    punits_cont::Vector{<:Unitful.Units}, tied_pairs_cont::Vector{<:Tuple}, tied_indices_cont::Vector{<:Integer}, 
    n_tied_cont::Integer, n_free_cont::Integer, pfix_lines_tied::Vector{<:Real}, lock_lines_tied::BitVector, 
    punits_lines::Vector{<:Unitful.Units}, tied_pairs_lines::Vector{<:Tuple}, tied_indices_lines::Vector{<:Integer}, 
    n_tied_lines::Integer, n_free_lines::Integer, spaxel::Spaxel, cube_fitter::CubeFitter; 
    bootstrap_iter::Bool=bootstrap_iter)

    fopt = fit_options(cube_fitter)
    popt_cont = rebuild_full_parameters(res.param[1:n_free_cont], pfix_cont_tied, lock_cont_tied, tied_pairs_cont, tied_indices_cont, n_tied_cont)
    perr_cont = zeros(length(popt_cont))
    if !bootstrap_iter
        perr_cont = rebuild_full_parameters(res.perror[1:n_free_cont], zeros(length(pfix_cont_tied)), lock_cont_tied, tied_pairs_cont, 
            tied_indices_cont, n_tied_cont)
    end
    popt_cont = popt_cont .* punits_cont
    perr_cont = perr_cont .* punits_cont

    popt_lines = rebuild_full_parameters(res.param[n_free_cont+1:end], pfix_lines_tied, lock_lines_tied, tied_pairs_lines, tied_indices_lines, n_tied_lines)
    perr_lines = zeros(Float64, length(popt_lines))
    if !bootstrap_iter
        perr_lines = rebuild_full_parameters(res.perror[n_free_cont+1:end], zeros(length(pfix_lines_tied)), lock_lines_tied, tied_pairs_lines, 
            tied_indices_lines, n_tied_lines)
    end
    popt_lines = popt_lines .* punits_lines
    perr_lines = perr_lines .* punits_lines

    @debug "Best fit continuum parameters: \n $popt_cont"
    @debug "Continuum parameter errors: \n $perr_cont"
    @debug "Best fit line parameters: \n $popt_lines"
    @debug "Line parameter errors: \n $perr_lines"

    result_stellar = nothing
    if fopt.fit_stellar_continuum
        ext_gas_0, _, _ = extinction_profiles(spaxel.λ, ustrip.(popt_cont), 1, fopt.fit_uv_bump, fopt.extinction_curve)
        I_lines_0, _ = model_line_residuals(spaxel, ustrip.(popt_lines), punits_lines, model(cube_fitter).lines, 
            cube_fitter.lsf, ext_gas_0, trues(length(spaxel.λ)), true)
        spaxel.I .-= I_lines_0
        _, _, norms = model_continuum(spaxel, spaxel.N, ustrip.(popt_cont), punits_cont, cube_fitter, 
            fopt.use_pah_templates, true, false, true)
        spaxel.I .+= I_lines_0
        result_stellar = calculate_stellar_parameters(cube_fitter, norms, spaxel.N)
    end
    spaxel_model = copy(spaxel)
    if length(cube_fitter.spectral_region.gaps) > 0
        spaxel_model = get_model_spaxel(cube_fitter, spaxel, result_stellar)
    else
        add_stellar_weights!(spaxel_model, result_stellar)
    end

    # Create the full model, again only if not bootstrapping
    # (we dont need to do that ugly stuff in the main fit_joint function because the stellar weights are already added 
    #  to the spaxel_model object, so there wont actually be any NNLS fitting being done, it'll just use the pre-calculated
    #  weights)
    I_cont, comps_cont, norms = model_continuum(spaxel_model, spaxel_model.N, ustrip.(popt_cont), punits_cont, cube_fitter, 
        fopt.use_pah_templates, spaxel_model == spaxel, false, true)
    I_lines, comps_lines = model_line_residuals(spaxel_model, ustrip.(popt_lines), punits_lines, model(cube_fitter).lines, cube_fitter.lsf, 
        comps_cont["total_extinction_gas"], trues(length(spaxel_model.λ)), true)
    
    popt_cont, perr_cont, I_cont, comps_cont, norms, popt_lines, perr_lines, I_lines, comps_lines, result_stellar, spaxel_model
end


# Helper function to get the total fit results for continuum and line fits
function collect_total_fit_results(spaxel::Spaxel, spaxel_model::Spaxel, cube_fitter::CubeFitter, 
    I_cont::Vector{<:Real}, I_line::Vector{<:Real}, comps_cont::Dict, comps_line::Dict, n_free_c::Integer, 
    n_free_l::Integer)

    # Combine the continuum and line models
    I_model = I_cont .+ I_line
    comps = merge(comps_cont, comps_line)

    # chi^2 and reduced chi^2 of the model BEFORE renormalizing
    mask_chi2 = copy(spaxel.mask_bad)
    for pair in cube_fitter.spectral_region.mask
        mask_chi2 .|= pair[1] .< spaxel.λ .< pair[2]
    end
    I_model_d = I_model
    if spaxel != spaxel_model
        # gap interpolation has been done on the model -- bring it back to the same wavelength grid
        I_model_d = Spline1D(ustrip.(spaxel_model.λ), I_model, k=1)(ustrip.(spaxel.λ))
    end
    χ2 = sum((spaxel.I[.~mask_chi2] .- I_model_d[.~mask_chi2]).^2 ./ spaxel.σ[.~mask_chi2].^2)
    # Degrees of freedom
    n_free = n_free_c + n_free_l
    n_data = length(spaxel.I[.~mask_chi2])
    dof = n_data - n_free

    # Renormalize
    I_model = I_model .* spaxel.N
    for comp ∈ keys(comps)
        if contains(comp, "extinction") || contains(comp, "absorption") || contains(comp, "mpoly")
            continue
        end
        comps[comp] = comps[comp] .* spaxel.N
    end

    I_model, comps, χ2, dof
end


# Helper function to get bootstrapped fit results 
function collect_bootstrapped_results(spaxel::Spaxel, spaxel_model::Spaxel, cube_fitter::CubeFitter, 
    p_boot::Matrix{<:Quantity}, I_model_boot::Matrix{<:QSIntensity}, split1::Integer, split2::Integer)

    # Filter out any large (>5 sigma) outliers
    punits = unit.(p_boot[:, 1])
    p_med = dropdims(nanquantile(ustrip.(p_boot), 0.50, dims=2), dims=2) .* punits
    # factor of 1.4826 to normalize the MAD it so it is interpretable as a standard deviation
    p_mad = dropdims(nanmad(ustrip.(p_boot), dims=2), dims=2) .* 1.4826  .* punits
    p_mask = (p_boot .< (p_med .- 5 .* p_mad)) .| (p_boot .> (p_med .+ 5 .* p_mad))
    p_boot[p_mask] .*= NaN
    p_out = p_med

    # (if set to :best, p_out is already the best-fit values from earlier)
    p_err_lo = p_med .- dropdims(nanquantile(ustrip.(p_boot), 0.159, dims=2), dims=2) .* punits
    p_err_up = dropdims(nanquantile(ustrip.(p_boot), 0.841, dims=2), dims=2) .* punits .- p_med
    p_err = [p_err_lo p_err_up]

    # Get the minimum/maximum pointwise bootstrapped models
    I_boot_min = dropdims(nanminimum(ustrip.(I_model_boot), dims=2), dims=2) .* unit(I_model_boot[1])
    I_boot_max = dropdims(nanmaximum(ustrip.(I_model_boot), dims=2), dims=2) .* unit(I_model_boot[1])

    # Replace the best-fit model with the 50th percentile model to be consistent with p_out
    fopt = fit_options(cube_fitter)
    I_boot_cont, comps_boot_cont, norms = model_continuum(spaxel_model, spaxel_model.N, ustrip.(p_out[1:split1]), unit.(p_out[1:split1]), 
        cube_fitter, false, spaxel_model == spaxel, !fopt.fit_joint, true)
    I_boot_line, comps_boot_line = model_line_residuals(spaxel_model, ustrip.(p_out[split1+1:split2]), unit.(p_out[split1+1:split2]), 
        model(cube_fitter).lines, cube_fitter.lsf, comps_boot_cont["total_extinction_gas"], trues(length(spaxel_model.λ)), true)

    # Reconstruct the full model
    I_model, comps, χ2, = collect_total_fit_results(spaxel, spaxel_model, cube_fitter, I_boot_cont, I_boot_line,
        comps_boot_cont, comps_boot_line, 0, 0)

    # Recalculate chi^2 based on the median model
    p_out[end-1] = χ2

    p_out, p_err, I_boot_min, I_boot_max, I_model, comps, norms, χ2
end


# Helper function that decides whether a fit should be redone with extinction locked to 0,
# based on how much the templates dominate the fit
function determine_fit_redo_with0extinction(cube_fitter::CubeFitter, λ::Vector{<:QWave}, σ::Vector{<:Real}, 
    pnames::Vector{String}, popt::Vector{<:Real}, plock::BitVector, I_model::Vector{<:Real}, comps::Dict, 
    init::Bool, force_noext::Bool)

    redo_fit = false
    fopt = fit_options(cube_fitter)
    for s in 1:cube_fitter.n_templates
        pₑ = fopt.silicate_absorption == "decompose" ? fast_indexin("extinction.N_oli", pnames) : fast_indexin("extinction.tau_97", pnames)
        sil_abs_region = 9.0u"μm" .< λ .< 11.0u"μm"
        ext_not_locked = !plock[pₑ] && !init && !force_noext && popt[pₑ] ≠ 0
        region_is_valid = sum(sil_abs_region) > 100
        # check if more than half of the pixels b/w 9-11 um, after subtracting the nuclear template, are within 3 sigma of 0.
        template_dominates = sum((abs.(I_model .- comps["templates_$s"])[sil_abs_region]) .< (σ[sil_abs_region])) > (sum(sil_abs_region)/2)
        if ext_not_locked && region_is_valid && (template_dominates || fopt.F_test_ext)
            redo_fit = true
            break
        end
    end

    redo_fit
end

