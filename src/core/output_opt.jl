############################## OUTPUT / SAVING FUNCTIONS ####################################

# Optical implementation of the assign_outputs function
function assign_outputs_opt(out_params::AbstractArray{<:Real}, out_errs::AbstractArray{<:Real}, cube_fitter::CubeFitter,
    cube_data::NamedTuple, z::Real, aperture::Bool=false, nuc_temp_fit::Bool=false)

    # Create the CubeModel and ParamMaps structs to be filled in
    cube_model = generate_cubemodel(cube_fitter, aperture)
    param_maps = generate_parammaps(cube_fitter, aperture)

    # Unpack relative flags for lines
    rel_amp, rel_voff, rel_fwhm = cube_fitter.relative_flags

    # Loop over each spaxel and fill in the associated fitting parameters into the ParamMaps and CubeModel
    # I know this is long and ugly and looks stupid but it works for now and I'll make it pretty later
    spaxels = CartesianIndices(size(out_params)[1:2])
    prog = Progress(length(spaxels); showspeed=true)
    for index ∈ spaxels

        # Get the normalization to un-normalized the fitted parameters
        data_index = !isnothing(cube_fitter.cube.voronoi_bins) ? cube_fitter.cube.voronoi_bins[index] : index
        if Tuple(data_index)[1] > 0
            N = Float64(abs(nanmaximum(cube_data.I[data_index, :])))
            N = N ≠ 0. ? N : 1.
        else
            N = 1.
        end

        # Set the 2D parameter map outputs

        # Conversion factor from MJy sr^-1 to erg s^-1 cm^-2 Hz^-1 sr^-1 = 10^6 * 10^-23 = 10^-17
        # So, log10(A * 1e-17) = log10(A) - 17

        Ω_med = median(cube_fitter.cube.Ω)

        # Loop through parameters and save them in the parammaps data structure 
        for (pᵢ, pname) in enumerate(param_maps.names)
            # Get the values
            val = out_params[index, pᵢ]
            err_upp, err_low = out_errs[index, pᵢ, 1:2]

            # Handle line parameters
            if param_maps.line_transform[pᵢ]
                line_comp = split(pname, ".")[end-1]
                if isdigit(line_comp[1])
                    line_comp = parse(Int, line_comp)
                    if !isone(line_comp) && ((contains(pname, "amp") && rel_amp) || (contains(pname, "fwhm") && rel_fwhm))
                        # get the amp/fwhm of the first line component
                        pstr = contains(pname, "amp") ? "amp" : "fwhm"
                        ind_1 = findfirst(param_maps.names .== replace(pname, "$line_comp.$pstr" => "1.$pstr"))
                        # apply multiplicative transformation
                        err_upp = √((out_errs[index, ind_1, 1] * val)^2 + (err_upp * out_params[index, ind_1])^2) 
                        err_low = √((out_errs[index, ind_1, 2] * val)^2 + (err_low * out_params[index, ind_1])^2) 
                        val *= out_params[index, ind_1]
                    elseif contains(pname, "voff") && !contains(pname, "voff_indiv") && !isone(line_comp) && rel_voff
                        # get the voff of the first line component
                        voff_1_ind = findfirst(param_maps.names .== replace(pname, "$line_comp.voff" => "1.voff"))
                        err_upp = √(err_upp^2 + out_errs[index, voff_1_ind, 1]^2) 
                        err_low = √(err_low^2 + out_errs[index, voff_1_ind, 2]^2) 
                        # apply additive transformation
                        val += out_params[index, voff_1_ind]
                    end
                end
            end

            # Apply other transformations
            # Shift back to the observed frame
            # In the optical case, it matters whether "restframe_transform" is 1 (flux transformation) or 2 (wavelength transformation)
            # since they need to have opposite factors applied
            if param_maps.restframe_transform[pᵢ] == 1
                val /= 1 + z
                err_upp /= 1 + z
                err_low /= 1 + z
            elseif param_maps.restframe_transform[pᵢ] == 2
                val *= 1 + z
                err_upp *= 1 + z
                err_low *= 1 + z
            end
            # Normalize units
            if param_maps.normalize[pᵢ]
                val *= N
                err_upp *= N
                err_low *= N
            end
            # Special case for the stellar pop masses
            if contains(pname, "stellar_populations") && contains(pname, "mass")
                ssp_max = maximum([cube_fitter.ssp_templates[j](out_params[index, pᵢ+1], out_params[index, pᵢ+2]) 
                    for j in eachindex(cube_fitter.ssp_λ)])
                val *= Ω_med / ssp_max
                err_upp *= Ω_med / ssp_max
                err_low *= Ω_med / ssp_max
            end

            # Convert from per unit wavelength (angstrom) to per unit frequency (Hz)
            if param_maps.perfreq_transform[pᵢ]
                if contains(pname, "feii.na")
                    λ0 = 1e4 * (1+z) * cube_data.λ[argmax(convolve_losvd(cube_fitter.feii_templates_fft[:, 1], cube_fitter.vsyst_feii, 
                        out_params[index, pᵢ+1], out_params[index, pᵢ+2], cube_fitter.vres, length(cube_data.λ), temp_fft=true, npad_in=cube_fitter.npad_feii))]
                elseif contains(pname, "feii.br")
                    λ0 = 1e4 * (1+z) * cube_data.λ[argmax(convolve_losvd(cube_fitter.feii_templates_fft[:, 2], cube_fitter.vsyst_feii,
                        out_params[index, pᵢ+1], out_params[index, pᵢ+2], cube_fitter.vres, length(cube_data.λ), temp_fft=true, npad_in=cube_fitter.npad_feii))]
                elseif contains(pname, "power_law")
                    λ0 = 5100. * (1+z)
                elseif contains(pname, "lines")
                    # Get the name of the line itself, without the component index
                    lname = split(pname, ".")[2]
                    k = findfirst(cube_fitter.lines.names .== Symbol(lname))
                    # voff is always the next parameter and if it's relative it will have already been corrected at this point
                    λ0 = 1e4 * (1+z) * cube_fitter.lines.λ₀[k] * (1 + out_params[index, pᵢ+1]/C_KMS)
                end
                # apply the transformation
                val *= λ0^2 / (C_KMS * 1e13)
                err_upp *= λ0^2 / (C_KMS * 1e13)
                err_low *= λ0^2 / (C_KMS * 1e13)
            end
            # Take the log10
            if param_maps.log_transform[pᵢ]
                err_upp = err_upp / (log(10) * val)
                err_low = err_low / (log(10) * val)
                val = log10(val)
            end

            # Set the values
            param_maps.data[index, pᵢ] = val
            param_maps.err_upp[index, pᵢ] = err_upp
            param_maps.err_low[index, pᵢ] = err_low
        end

        pᵢ = 1 + 3cube_fitter.n_ssps + 2 + 2 + (cube_fitter.fit_uv_bump && cube_fitter.extinction_curve == "calzetti" ? 1 : 0) +
            (cube_fitter.fit_covering_frac && cube_fitter.extinction_curve == "calzetti" ? 1 : 0) + (cube_fitter.fit_opt_na_feii ? 3 : 0) +
            (cube_fitter.fit_opt_br_feii ? 3 : 1) + 2cube_fitter.n_power_law

        if cube_fitter.save_full_model
            # End of continuum parameters: recreate the continuum model
            I_cont, comps_c = model_continuum(cube_fitter.cube.λ, out_params[index, 1:pᵢ-1], N, cube_fitter.vres, cube_fitter.vsyst_ssp,
                cube_fitter.vsyst_feii, cube_fitter.npad_feii, cube_fitter.n_ssps, cube_fitter.ssp_λ, cube_fitter.ssp_templates, 
                cube_fitter.feii_templates_fft, cube_fitter.n_power_law, cube_fitter.fit_uv_bump, cube_fitter.fit_covering_frac, 
                cube_fitter.fit_opt_na_feii, cube_fitter.fit_opt_br_feii, cube_fitter.extinction_curve, cube_data.templates[index, :, :], 
                cube_fitter.fit_temp_multexp, nuc_temp_fit, true)
        end

        # Save marker of the point where the continuum parameters end and the line parameters begin
        vᵢ = pᵢ
        # Get the index where the line parameters end
        for k ∈ 1:cube_fitter.n_lines
            for j ∈ 1:cube_fitter.n_comps
                pᵢ += 3
                if !isnothing(cube_fitter.lines.tied_voff[k,j]) && cube_fitter.flexible_wavesol && isone(j)
                    pᵢ += 1
                end
                if cube_fitter.lines.profiles[k,j] == :GaussHermite
                    pᵢ += 2
                end
                if cube_fitter.lines.profiles[k,j] == :Voigt
                    pᵢ += 1
                end
            end
        end

        if cube_fitter.save_full_model

            # Interpolate the LSF
            lsf_interp = Spline1D(cube_fitter.cube.λ, cube_fitter.cube.lsf, k=1)
            lsf_interp_func = x -> lsf_interp(x)
            templates_psfnuc = nuc_temp_fit ? comps_c["templates_1"] : nothing

            # End of line parameters: recreate the un-extincted (intrinsic) line model
            I_line, comps_l = model_line_residuals(cube_fitter.cube.λ, out_params[index, vᵢ:pᵢ-1], cube_fitter.n_lines, cube_fitter.n_comps,
                cube_fitter.lines, cube_fitter.flexible_wavesol, comps_c["attenuation_gas"], lsf_interp_func, cube_fitter.relative_flags,
                templates_psfnuc, nuc_temp_fit, true)

            # Combine the continuum and line models
            I_model = I_cont .+ I_line
            comps = merge(comps_c, comps_l)

            # Renormalize
            I_model .*= N
            for comp ∈ keys(comps)
                if !((comp == "extinction") || contains(comp, "ext_") || contains(comp, "abs_") || contains(comp, "attenuation"))
                    if !(nuc_temp_fit && contains(comp, "template"))
                        comps[comp] .*= N
                    end
                end
            end
            
        end

        # Sort the parameters for multicomponent lines
        sort_line_components!(cube_fitter, param_maps, index, cube_data)

        if cube_fitter.save_full_model
            # Set 3D model cube outputs, shifted back to the observed frame
            # Remember the wavelength axis is the first axis here to increase efficiency
            cube_model.model[:, index] .= I_model ./ (1 .+ z)
            for i ∈ 1:cube_fitter.n_ssps
                cube_model.stellar[:, index, i] .= comps["SSP_$i"] ./ (1 .+ z)
            end
            cube_model.attenuation_stars[:, index] .= comps["attenuation_stars"]
            cube_model.attenuation_gas[:, index] .= comps["attenuation_gas"]
            if cube_fitter.fit_opt_na_feii
                cube_model.na_feii[:, index] .= comps["na_feii"]
            end
            if cube_fitter.fit_opt_br_feii
                cube_model.br_feii[:, index] .= comps["br_feii"]
            end
            for l ∈ 1:cube_fitter.n_power_law
                cube_model.power_law[:, index, l] .= comps["power_law_$l"]
            end
            for q ∈ 1:cube_fitter.n_templates
                cube_model.templates[:, index, q] .= comps["templates_$q"] 
                if !nuc_temp_fit
                    cube_model.templates[:, index, q] ./= (1 .+ z)
                end
            end

            for j ∈ 1:cube_fitter.n_comps
                for k ∈ 1:cube_fitter.n_lines
                    if !isnothing(cube_fitter.lines.profiles[k, j])
                        cube_model.lines[:, index, k] .+= comps["line_$(k)_$(j)"] ./ (1 .+ z)
                    end
                end
            end
        end

        next!(prog)

    end

    param_maps, cube_model
end


# Optical implementation of the assign_qso3d_outputs function
function assign_qso3d_outputs_opt(param_maps::ParamMaps, cube_model::CubeModel, cube_fitter::CubeFitter,
    psf_norm::Array{<:Real,3})

    param_maps_3d = generate_parammaps(cube_fitter, false)
    cube_model_3d = generate_cubemodel(cube_fitter, false)
    n_pix = nansum(.~cube_fitter.cube.mask, dims=(1,2))
    for k in axes(param_maps_3d.data, 3)
        name = param_maps_3d.names[k]
        if !contains(name, "flux") && !contains(name, "amp")
            param_maps_3d.data[:,:,k] .= param_maps.data[1,1,k]
            continue
        end
        # get the central wavelength of the feature in question
        namelist = split(name, ".")
        if namelist[1] == "lines"
            ln_ind = findfirst(cube_fitter.lines.names .== namelist[2])
            λi = cube_fitter.lines.λ₀[ln_ind]
        else
            # continuum amplitudes
            λi = 10^nanmean(log10.(cube_fitter.cube.λ))
        end
        # convert to an index
        w = argmin(abs.(cube_fitter.cube.λ .- λi))
        dont_log = (namelist[1] == "lines") && cube_fitter.lines_allow_negative
        amp_factor = contains(name, "amp") ? n_pix[k] : 1.
        # intensities need an extra conversion factor
        # (because the intensities recorded in param_maps are over the whole FOV
        #  whereas these intensities are over single spaxels: area_FOV / area_spaxel)
        if dont_log
            param_maps_3d.data[:,:,k] .= param_maps.data[1,1,k] .* psf_norm[:,:,w] .* amp_factor
        else
            new = 10 .^ param_maps.data[1,1,k] .* psf_norm[:,:,w] .* amp_factor
            new[new .< 0] .= 0.
            param_maps_3d.data[:,:,k] .= log10.(new)
        end
    end

    psf_norm_p = permutedims(psf_norm .* n_pix, (3,1,2))
    _shape = size(psf_norm[1:2])

    # Remember the wavelength axis is the first axis here to increase efficiency
    cube_model_3d.model .= extendp(cube_model.model[:,1,1], _shape) .* psf_norm_p
    for i ∈ 1:cube_fitter.n_ssps
        cube_model_3d.stellar[:,:,:,i] .= extendp(cube_model.stellar[:,1,1,i], _shape) .* psf_norm_p
    end
    cube_model_3d.attenuation_stars .= cube_model.attenuation_stars[:,1,1]
    cube_model_3d.attenuation_gas .= cube_model.attenuation_gas[:,1,1]
    if cube_fitter.fit_opt_na_feii
        cube_model_3d.na_feii .= extendp(cube_model.na_feii[:,1,1], _shape) .* psf_norm_p
    end
    if cube_fitter.fit_opt_br_feii
        cube_model_3d.br_feii .= extendp(cube_model.br_feii[:,1,1], _shape) .* psf_norm_p
    end
    for l ∈ 1:cube_fitter.n_power_law
        cube_model_3d.power_law[:,:,:,l] .= extendp(cube_model.power_law[:,1,1,l], _shape) .* psf_norm_p
    end
    for q ∈ 1:cube_fitter.n_templates
        cube_model_3d.templates[:,:,:,q] .= extendp(cube_model.templates[:,1,1,q], _shape) .* psf_norm_p
    end
    for j ∈ 1:cube_fitter.n_comps
        for k ∈ 1:cube_fitter.n_lines
            if !isnothing(cube_fitter.lines.profiles[k, j])
                cube_model_3d.lines[:,:,:,k] .= extendp(cube_model.lines[:,1,1,k], _shape) .* psf_norm_p
            end
        end
    end

    param_maps_3d, cube_model_3d
end


# Helper function for writing the output for an optical cube model
function write_fits_full_model_opt(cube_fitter::CubeFitter, cube_data::NamedTuple, cube_model::CubeModel,
    hdr::FITSHeader, nuc_temp_fit::Bool; qso3d::Bool=false)

    # Create the 3D intensity model FITS file
    FITS(joinpath("output_$(cube_fitter.name)", 
                  "$(cube_fitter.name)_$(nuc_temp_fit ? "nuc_model" : "full_model")$(qso3d ? "_3d" : "").fits"), 
                  "w") do f

        @debug "Writing 3D model FITS HDUs"
        # Permute the wavelength axis here back to the third axis to be consistent with conventions

        write(f, Vector{Int}())                                                                     # Primary HDU (empty)
        write(f, Float32.(cube_data.I ./ (1 .+ cube_fitter.z)); name="DATA", header=hdr)          # Raw data 
        write(f, Float32.(cube_data.σ ./ (1 .+ cube_fitter.z)); name="ERROR")                       # Error in the raw data
        write(f, permutedims(cube_model.model, (2,3,1)); name="MODEL")                              # Full intensity model
        for i ∈ 1:size(cube_model.stellar, 4)
            write(f, permutedims(cube_model.stellar[:, :, :, i], (2,3,1)); name="CONTINUUM.STELLAR_POPULATIONS.$i")   # Stellar population models
        end
        if cube_fitter.fit_opt_na_feii
            write(f, permutedims(cube_model.na_feii, (2,3,1)); name="CONTINUUM.FEII.NA")                      # Narrow Fe II emission
        end
        if cube_fitter.fit_opt_br_feii
            write(f, permutedims(cube_model.br_feii, (2,3,1)); name="CONTINUUM.FEII.BR")                      # Broad Fe II emission
        end
        for j in 1:size(cube_model.power_law, 4)
            write(f, permutedims(cube_model.power_law[:, :, :, j], (2,3,1)); name="CONTINUUM.POWER_LAW.$j")   # Power laws
        end
        for (q, tp) ∈ enumerate(cube_fitter.template_names)
            tpu = uppercase("$tp")
            write(f, permutedims(cube_model.templates[:, :, :, q], (2,3,1)); name="TEMPLATES.$tpu")   # Template profiles
        end
        for (k, line) ∈ enumerate(cube_fitter.lines.names)
            lnu = uppercase("$line")
            write(f, permutedims(cube_model.lines[:, :, :, k], (2,3,1)); name="LINES.$lnu")   # Emission line profiles
        end
        write(f, permutedims(cube_model.attenuation_stars, (2,3,1)); name="ATTENUATION.STARS")      # Starlight attenuation model
        write(f, permutedims(cube_model.attenuation_gas, (2,3,1)); name="ATTENUATION.GAS")          # Gas attenuation model
        write(f, ["wave"], [cube_data.λ .* (1 .+ cube_fitter.z)],                                   # wavelength vector
            hdutype=TableHDU, name="WAVELENGTH", units=Dict(:wave => "um"))

        # Insert physical units into the headers of each HDU -> erg/s/cm^2/Ang/sr for all except
        # the extinction profile, which is a multiplicative constant
        write_key(f["DATA"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
        write_key(f["ERROR"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
        write_key(f["MODEL"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
        for i ∈ 1:size(cube_model.stellar, 4)
            write_key(f["CONTINUUM.STELLAR_POPULATIONS.$i"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
        end
        if cube_fitter.fit_opt_na_feii
            write_key(f["CONTINUUM.FEII.NA"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
        end
        if cube_fitter.fit_opt_br_feii
            write_key(f["CONTINUUM.FEII.BR"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
        end
        for j ∈ 1:size(cube_model.power_law, 4)
            write_key(f["CONTINUUM.POWER_LAW.$j"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
        end
        for tp ∈ cube_fitter.template_names
            tpu = uppercase("$tp")
            write_key(f["TEMPLATES.$tpu"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
        end
        for line ∈ cube_fitter.lines.names
            lnu = uppercase("$line")
            write_key(f["LINES.$lnu"], "BUNIT", "erg.s-1.cm-2.Angstrom-1.sr-1")
        end
        write_key(f["ATTENUATION.STARS"], "BUNIT", "-")
        write_key(f["ATTENUATION.GAS"], "BUNIT", "-")
    end
end

