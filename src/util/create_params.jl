

function create_dust_features(dust::Dict, λunit::Unitful.Units, Iunit::Unitful.Units, region::SpectralRegion; 
    do_absorption::Bool=false)

    # Dust feature central wavelengths and FWHMs
    cent_vals = Qum[]
    complexes = String[]
    feat_labels = String[]
    fit_profiles = FitProfiles()
    all_df_names = String[]
    all_df_labels = String[]
    key = do_absorption ? "absorption_features" : "dust_features"
    short_key = do_absorption ? "abs_features" : "dust_features"

    msg = "$(do_absorption ? "Absorption" : "Dust") features:"
    if haskey(dust, key)
        for df ∈ keys(dust[key])

            # First things first, check if this feature is within our wavelength range
            cent_val = dust[key][df]["wave"]["val"] * u"μm"
            if !is_valid(cent_val, 0.1u"μm", region)
                continue
            end

            profile = :Drude
            complex = df

            prefix = "$(short_key).$(df)."

            # amplitudes (not in file)
            if do_absorption
                amp = parameter_from_dict(dust[key][df]["tau"])
                t_amp = Transformation[]
            else
                amp = FitParameter(NaN, false, (0., Inf))   # the NaN is a placeholder for now and will be replaced
                t_amp = [RestframeTransform, LogTransform, NormalizeTransform]
            end
            msg *= "\n$(do_absorption ? "Depth" : "Amp") $(amp)"
            mean = parameter_from_dict_wave(dust[key][df]["wave"]; units=u"μm")
            mean = uconvert(λunit, mean)
            t_mean = [RestframeTransform]
            msg *= "\nWave $(mean)"
            fwhm = parameter_from_dict_fwhm(dust[key][df]["fwhm"]; units=u"μm")
            fwhm = uconvert(λunit, fwhm)
            t_fwhm = [RestframeTransform]
            msg *= "\nFWHM $(fwhm)"

            names = prefix .* [do_absorption ? "tau" : "amp", "mean", "fwhm"]
            labels = String[do_absorption ? L"$\tau$" : L"$\log_{10}(I / $ %$(latex(Iunit))$)$", 
                L"$\mu$ (%$(latex(λunit)))", L"FWHM (%$(latex(λunit)))"]
            trans = [t_amp, t_mean, t_fwhm]
            _params = [amp, mean, fwhm]

            if haskey(dust[key][df], "index") && haskey(dust[key][df], "cutoff") && !do_absorption 
                profile = :PearsonIV
                index = parameter_from_dict(dust[key][df]["index"])
                t_index = Transformation[]
                msg *= "\nIndex $(index)"
                cutoff = parameter_from_dict(dust[key][df]["cutoff"])
                t_cutoff = Transformation[]
                msg *= "\nCutoff $(cutoff)"
                append!(names, prefix .* ["index", "cutoff"])
                append!(labels, [L"$m$", L"$\nu$"])
                append!(trans, [t_index, t_cutoff])
                append!(_params, [index, cutoff])
            end
            if profile == :Drude
                push!(names, prefix * "asym")
                push!(labels, L"$a$ (%$(latex(λunit))$^{-1}$)")
                push!(trans, Transformation[])
                if haskey(dust[key][df], "asym")
                    # units are inverse wavelength 
                    asym = parameter_from_dict_fwhm(dust[key][df]["asym"]; units=u"μm^-1")  # (fwhm method so that uncertainties are fractional)
                    asym = uconvert(unit(1.0/λunit), asym)
                    msg *= "\nAsym $(asym)"
                    push!(_params, asym)
                else # only add a default asym parameter if its a Drude profile
                    asym = FitParameter(0.0/λunit, true, (-0.01, 0.01)./λunit)
                    msg *= "\nAsym $(asym)"
                    push!(_params, asym)
                end
            end
            if haskey(dust[key][df], "complex")
                complex = dust[key][df]["complex"]
            end

            fit_parameters = FitParameters(names, labels, trans, _params)

            # non-fit parameters
            if !do_absorption
                npnames = prefix .* ["flux", "eqw", "SNR"]
                funit = u"erg/s/cm^2"
                nplabels = String[L"$\log_{10}(F /$ " * latex(funit) * L")", L"$W_{\rm eq}$ (" * latex(λunit) * ")", L"$S/N$"]
                nptrans = [[LogTransform], [RestframeTransform], Transformation[]]
                nparams = [NonFitParameter{typeof(1.0*funit)}(), NonFitParameter{typeof(1.0*λunit)}(), NonFitParameter{Float64}()]
                nonfit_parameters = NonFitParameters(npnames, nplabels, nptrans, nparams)
            else
                nonfit_parameters = NonFitParameters(String[], String[], Vector{Transformation}[], NonFitParameter[])
            end

            push!(fit_profiles, FitProfile(profile, fit_parameters, nonfit_parameters))
            push!(cent_vals, cent_val) 
            push!(complexes, complex)
            push!(all_df_names, df)

            wl = pah_name_to_float(complex)
            push!(feat_labels, "PAH $wl " * L"${\rm \mu m}$")
            wl = pah_name_to_float(df)
            push!(all_df_labels, "PAH $wl " * L"${\rm \mu m}$")
        end
        @debug msg

        # Sort by cent_vals, then reshape 
        ss = sortperm(cent_vals)
        cent_vals = cent_vals[ss]
        complexes = complexes[ss]
        all_df_names = all_df_names[ss]
        feat_labels = feat_labels[ss]
        all_df_labels = all_df_labels[ss]
        fit_profiles = fit_profiles[ss]

        # sort into vectors based on where the complex is the same
        all_fit_profiles = FitProfiles[]
        u_complexes = unique(complexes)
        u_cent_vals = Qum[]
        u_feat_labels = String[]
        composite = NonFitParameters[]  # composite parameters for multi-profiled features

        for complex in u_complexes
            inds = findall(c -> c == complex, complexes)
            push!(all_fit_profiles, fit_profiles[inds])
            wl = pah_name_to_float(complex)
            push!(u_cent_vals, wl*u"μm")
            push!(u_feat_labels, "PAH $wl " * L"${\rm \mu m}$")

            # composite non-fit parameters
            prefix = "$(short_key).$(complex)."
            funit = u"erg/s/cm^2"
            epnames = prefix .* ["total_flux", "total_eqw", "total_snr"]
            eparams = [NonFitParameter{typeof(1.0*funit)}(), NonFitParameter{typeof(1.0*λunit)}(), NonFitParameter{Float64}()]
            eplabels = String[L"$\log_{10}(F /$ " * latex(funit) * L")", L"$W_{\rm eq}$ (" * latex(λunit) * ")", L"$S/N$"]
            eptrans = [[LogTransform], [RestframeTransform], Transformation[]]
            push!(composite, NonFitParameters(epnames, eplabels, eptrans, eparams))
        end

        cfg = PAHConfig(all_df_names, all_df_labels)

        FitFeatures(u_complexes, u_feat_labels, u_cent_vals, all_fit_profiles, composite, cfg)
    
    else

        FitFeatures(String[], String[], typeof(1.0*λunit)[], FitProfiles[], NonFitParameters[], PAHConfig(String[], String[]))
    end
end


function construct_extinction_params!(params::Vector{FitParameter}, pnames::Vector{String}, plabels::Vector{String},
    ptrans::Vector{Vector{Transformation}}, out::Dict, optical::Dict, infrared::Dict, λunit::Unitful.Units, Iunit::Unitful.Units,
    region::SpectralRegion)

    # Extinction parameters, optical depth and mixing ratio
    msg = "Extinction:"
    prefix = "extinction."

    # Reddening and stellar-to-dust reddening factor
    E_BV = parameter_from_dict(optical["extinction"]["E_BV"])
    E_BV_factor = parameter_from_dict(optical["extinction"]["E_BV_factor"])
    msg *= "\nE(B-V) $E_BV"
    msg *= "\nE(B-V) factor $E_BV_factor"
    append!(params, [E_BV, E_BV_factor])
    append!(pnames, prefix .* ["E_BV", "E_BV_factor"])
    append!(plabels, [L"$E(B-V)_{\rm gas}$", L"$E(B-V)_{\rm stars}/E(B-V)_{\rm gas}$"])
    append!(ptrans, [Transformation[], Transformation[]])
    # Optional UV bump slope and covering fraction
    if out[:fit_uv_bump]
        δ_uv = parameter_from_dict(optical["extinction"]["uv_slope"])
        msg *= "\nδ_uv $δ_uv"
        push!(params, δ_uv)
        push!(pnames, prefix * "delta_UV")
        push!(plabels, L"$\delta_{\rm UV}$")
        push!(ptrans, Transformation[])
    end
    # Silicate absorption profiles
    if out[:silicate_absorption] == "decompose"
        N_oli = parameter_from_dict(infrared["extinction"]["N_oli"]; units=u"g/cm^2")
        msg *= "\nN_oli $N_oli"
        N_pyr = parameter_from_dict(infrared["extinction"]["N_pyr"]; units=u"g/cm^2")
        msg *= "\nN_pyr $N_pyr"
        N_for = parameter_from_dict(infrared["extinction"]["N_for"]; units=u"g/cm^2")
        msg *= "\nN_for $N_for"
        append!(params, [N_oli, N_pyr, N_for])
        append!(pnames, prefix .* ["N_oli", "N_pyr", "N_for"])
        append!(plabels, [L"$\log_{10}(N_{\rm oli} / $ g cm$^{-2}$)", L"$\log_{10}(N_{\rm pyr} / N_{\rm oli}$)",
            L"$\log_{10}(N_{\rm for} / N_{\rm oli}$)"])
        append!(ptrans, [[LogTransform], [LogTransform], [LogTransform]])
    else
        τ_97 = parameter_from_dict(infrared["extinction"]["tau_9_7"])
        msg *= "\nTau_sil $τ_97"
        push!(params, τ_97)
        push!(pnames, prefix * "tau_97")
        push!(plabels, L"$\tau_{9.7}$")
        push!(ptrans, Transformation[])
    end
    β = parameter_from_dict(infrared["extinction"]["beta"])
    msg *= "\nBeta $β"
    push!(params, β)
    push!(pnames, prefix .* "beta")
    push!(plabels, L"$\beta")
    push!(ptrans, Transformation[])
    if out[:fit_ch_abs]
        τ_ice = parameter_from_dict(infrared["extinction"]["tau_ice"])
        msg *= "\nTau_ice $τ_ice"
        τ_ch = parameter_from_dict(infrared["extinction"]["tau_ch"])
        msg *= "\nTau_CH $τ_ch"
        append!(params, [τ_ice, τ_ch])
        append!(pnames, prefix .* ["tau_ice", "tau_ch"])
        append!(plabels, [L"$\tau_{\rm ice}$", L"$\tau_{\rm CH}$"]) 
        append!(ptrans, [Transformation[], Transformation[]])
    end
    @debug msg

    # These will be automatically sorted out by the wavelength range in the region object
    abs_features = create_dust_features(infrared, λunit, Iunit, region; do_absorption=true)
    if !isnothing(abs_features)
        abs_parameters = get_flattened_fit_parameters(abs_features)
        append!(params, abs_parameters._parameters)
        append!(pnames, abs_parameters.names)
    end

    # Covering fraction
    if out[:fit_covering_frac]
        frac = parameter_from_dict(optical["extinction"]["frac"])
        msg *= "\nfrac $frac"
        push!(params, frac)
        push!(pnames, prefix * "frac")
        push!(plabels, L"$C_f$")
        push!(ptrans, Transformation[])
        @debug msg
    end

    abs_features
end


function construct_continuum_params!(params::Vector{FitParameter}, pnames::Vector{String}, plabels::Vector{String},
    ptrans::Vector{Vector{Transformation}}, out::Dict, optical::Dict, infrared::Dict, λunit::Unitful.Units, Iunit::Unitful.Units,
    redshift::Real, region::SpectralRegion)

    # Check wavelength ranges to make sure fitting stellar pops is not unreasonable
    if region.λlim[1] > 20.0u"μm" && out[:fit_stellar_continuum]
        @warn "The minimum wavelength in the input spectrum is > 20 μm! Stellar populations will be disabled."
        out[:fit_stellar_continuum] = false
    end

    if out[:fit_stellar_continuum]
        msg = "Stellar populations:"
        for (i, (age0, metal0)) ∈ enumerate(zip(optical["stellar_population_ages"], optical["stellar_population_metallicities"]))
            prefix = "continuum.stellar_populations.$(i)."
            mass = FitParameter(NaN, false, (0., Inf)) 
            t_mass = [RestframeTransform, LogTransform, NormalizeTransform]
            msg *= "\nMass $mass"
            age_param = parameter_from_dict(age0; units=u"Gyr")
            age_univ = age(u"Gyr", out[:cosmology], redshift)
            # Check the upper limit on age and make sure it's not older than the age of the universe at a given redshift
            if age_univ < age_param.limits[2]
                @warn "The age of the universe at z=$redshift is younger than the upper limit on the SSP age $(age_param.limits[2]). " *
                      "The upper limit will be reduced to match the universe age."
                set_plim!(age_param, (age_param.limits[1], age_univ))
            end
            t_age = Transformation[]
            msg *= "\nAge $age"
            z = parameter_from_dict(metal0)
            t_z = Transformation[]
            msg *= "\nMetallicity $z"
            append!(params, [mass, age_param, z])
            append!(pnames, prefix .* ["mass", "age", "metallicity"])
            append!(plabels, [L"$\log_{10}(M / M_{\odot})$", L"$t_{\rm age}$ (Gyr)", L"$\log_{10}(Z/Z_\odot)$"])
            append!(ptrans, [t_mass, t_age, t_z])
        end
        @debug msg

        msg = "Stellar kinematics:"
        prefix = "continuum.stellar_kinematics."
        stel_vel = parameter_from_dict(optical["stellar_kinematics"]["vel"]; units=u"km/s")
        msg *= "\nVelocity $stel_vel"
        stel_vdisp = parameter_from_dict(optical["stellar_kinematics"]["vdisp"]; units=u"km/s")
        msg *= "\nVdisp $stel_vdisp"
        append!(params, [stel_vel, stel_vdisp])
        append!(pnames, prefix .* ["vel", "vdisp"])
        append!(plabels, [L"$v_*$ (km s$^{-1}$)", L"$\sigma_*$ (km s$^{-1}$)"])
        append!(ptrans, Transformation[], Transformation[])
        @debug msg
    end

    if (region.λlim[1] > 7200.0u"angstrom") || (region.λlim[2] < 3400.0u"angstrom")
        if out[:fit_opt_na_feii] || out[:fit_opt_br_feii]
            @warn "The input spectrum falls completely outside the range of the Fe II templates. They will be disabled"
            out[:fit_opt_na_feii] = false
            out[:fit_opt_br_feii] = false
        end
    end

    prefix = "continuum.feii."
    if out[:fit_opt_na_feii]
        msg = "Narrow Fe II templates:"
        na_feii_A = FitParameter(NaN, false, (0., Inf))
        t_na_A = [RestframeTransform, LogTransform, NormalizeTransform]
        msg *= "\nNA Amp $na_feii_A"
        na_feii_vel = parameter_from_dict(optical["na_feii_kinematics"]["vel"]; units=u"km/s")
        msg *= "\nNA Velocity $na_feii_vel"
        na_feii_vdisp = parameter_from_dict(optical["na_feii_kinematics"]["vdisp"]; units=u"km/s")
        msg *= "\nNA Vdisp $na_feii_vdisp"
        append!(params, [na_feii_A, na_feii_vel, na_feii_vdisp])
        append!(pnames, prefix .* "na." .* ["amp", "vel", "vdisp"])
        append!(plabels, [L"$\log_{10}(I / $ %$(latex(Iunit))$)$", L"$v$ (km s$^{-1}$)", L"$\sigma$ (km s$^{-1}$)"])
        append!(ptrans, [t_na_A, Transformation[], Transformation[]])
        @debug msg
    end
    if out[:fit_opt_br_feii]
        msg = "Broad Fe II templates:"
        br_feii_A = FitParameter(NaN, false, (0., Inf))
        t_br_A = [RestframeTransform, LogTransform, NormalizeTransform]
        msg *= "\nBR Amp $br_feii_A"
        br_feii_vel = parameter_from_dict(optical["br_feii_kinematics"]["vel"]; units=u"km/s")
        msg *= "\nBR Velocity $br_feii_vel"
        br_feii_vdisp = parameter_from_dict(optical["br_feii_kinematics"]["vdisp"]; units=u"km/s")
        msg *= "\nBR Vdisp $br_feii_vdisp"
        append!(params, [br_feii_A, br_feii_vel, br_feii_vdisp])
        append!(pnames, prefix .* "br." .* ["amp", "vel", "vdisp"])
        append!(plabels, [L"$\log_{10}(I / $ %$(latex(Iunit))$)$", L"$v$ (km s$^{-1}$)", L"$\sigma$ (km s$^{-1}$)"])
        append!(ptrans, [t_br_A, Transformation[], Transformation[]])
        @debug msg
    end

    if haskey(optical, "power_law_indices")
        msg = "Power Laws:"
        for (i, power_law_index) in enumerate(optical["power_law_indices"])
            prefix = "continuum.power_law.$(i)."
            A_pl = FitParameter(NaN, false, (0., Inf))
            t_A = [RestframeTransform, LogTransform, NormalizeTransform]
            msg *= "\nAmp $A_pl"
            α_pl = parameter_from_dict(power_law_index)
            msg *= "\nIndex $α_pl"
            append!(params, [A_pl, α_pl])
            append!(pnames, prefix .* ["amp", "index"])
            append!(plabels, [L"$\log_{10}(I $ / %$(latex(Iunit))$)$", L"$\alpha$"])
            append!(ptrans, [t_A, Transformation[]])
        end
        @debug msg
    end

    # Dust continua
    if haskey(infrared, "dust_continuum_temps")
        if region.λlim[2] < 2.0u"μm"
            @warn "The maximum wavelength in the input spectrum is < 2 μm! Thermal dust emission will be disabled."
            infrared["dust_continuum_temps"] = []
        end
        msg = "Dust continuum:"
        for i in eachindex(infrared["dust_continuum_temps"])
            prefix = "continuum.dust.$(i)."
            A_dc = FitParameter(NaN, false, (0., Inf))
            t_A = [RestframeTransform, LogTransform]
            msg *= "\nAmp $A_dc"
            T_dc = parameter_from_dict(infrared["dust_continuum_temps"][i]; units=u"K")
            msg *= "\nTemp $T_dc"
            @debug msg
            append!(params, [A_dc, T_dc])
            append!(pnames, prefix .* ["amp", "temp"])
            append!(plabels, [L"$\log_{10}(A_{\rm dust})$", L"$T$ (K)"])
            append!(ptrans, [t_A, Transformation[]])
        end
    end

    if (region.λlim[1] > 30.0u"μm") || (region.λlim[2] < 0.8u"μm")
        if out[:fit_sil_emission]
            @warn "The input spectrum's wavelength range is outside the expected range of hot silicate emisson. It will be disabled."
            out[:fit_sil_emission] = false
        end
    end

    # Hot dust parameters, temperature, covering fraction, warm tau, and cold tau
    # No real restrictions on the wavelength range here!  Leave it up to the user if they want to fit it.
    msg = "Hot Dust:"
    if out[:fit_sil_emission]
        prefix = "continuum.hot_dust."
        A_hot = FitParameter(NaN, false, (0., Inf))
        t_A = [RestframeTransform, LogTransform]
        msg *= "\nAmp $A_hot"
        T_hot = parameter_from_dict(infrared["hot_dust"]["temp"]; units=u"K")
        msg *= "\nTemp $T_hot"
        hd_Cf = parameter_from_dict(infrared["hot_dust"]["frac"])
        msg *= "\nFrac $hd_Cf"
        τ_warm = parameter_from_dict(infrared["hot_dust"]["tau_warm"])
        msg *= "\nTau_Warm $τ_warm"
        τ_cold = parameter_from_dict(infrared["hot_dust"]["tau_cold"])
        msg *= "\nTau_Cold $τ_cold"
        sil_peak = parameter_from_dict(infrared["hot_dust"]["peak"]; units=u"μm")
        sil_peak = uconvert(λunit, sil_peak)
        msg *= "\nSil_Peak $sil_peak"
        @debug msg
        append!(params, [A_hot, T_hot, hd_Cf, τ_warm, τ_cold, sil_peak])
        append!(pnames, prefix .* ["amp", "temp", "frac", "tau_warm", "tau_cold", "sil_peak"])
        append!(plabels, [L"$\log_{10}(A_{\rm sil})$", L"$T$ (K)", L"$C_f$", L"$\tau_{\rm warm}$", 
            L"$\tau_{\rm cold}$", L"$\mu$ (%$(latex(λunit)))"])
        append!(ptrans, [t_A, Transformation[], Transformation[], Transformation[], Transformation[],
            Transformation[]])
    end

end


# Template amplitudes
function construct_template_params!(params::Vector{FitParameter}, pnames::Vector{String}, plabels::Vector{String},
    ptrans::Vector{Vector{Transformation}}, out::Dict, contin_options::Dict, region::SpectralRegion)

    if haskey(contin_options, "template_amps") && !out[:fit_temp_multexp]
        msg = "Template amplitudes:"
        for i ∈ eachindex(contin_options["template_amps"])
            tname = out[:template_names][i]
            for ni in 1:nchannels(region)
                temp_A = parameter_from_dict(contin_options["template_amps"][i])
                # Check if it should be tied
                if out[:tie_template_amps]
                    tie!(temp_A, Symbol(tname, "_amp"))
                end
                msg *= "\n$temp_A channel $ni"
                push!(params, temp_A)
                push!(pnames, "templates.$(tname).amp_$ni")
                push!(plabels, L"$\log_{10}(A_{\rm template})$")
                push!(ptrans, [LogTransform])
            end
        end
        @debug msg
    elseif haskey(contin_options, "template_inds") && out[:fit_temp_multexp]
        msg = "Template amplitudes:"
        for i ∈ 1:4
            temp_A = FitParameter(NaN, false, (0., Inf))
            msg *= "\nAmp $i $temp_A"
            temp_i = parameter_from_dict(contin_options["template_inds"][i])
            msg *= "\nIndex $i $temp_i"
            append!(params, [temp_A, temp_i])
            append!(pnames, "templates." .* ["amp_$i", "index_$i"])
            append!(plabels, [L"$\log_{10}(A_{\rm template})$", L"$b$"])
            append!(ptrans, [[LogTransform], Transformation[]])
        end
        @debug msg
    end

end


"""
    construct_parameters(out, λunit, Iunit, region)

Reads in the optical and infrared options files and creates parameter vectors for the 
continuum and dust features.
"""
function construct_continuum_parameters(out::Dict, λunit::Unitful.Units, Iunit::Unitful.Units, region::SpectralRegion,
    redshift::Real)

    params = FitParameter[]
    pnames = String[]
    plabels = String[]
    ptrans = Vector{Transformation}[]

    @debug """\n
    Parsing optical file
    #######################################################
    """
    # Read in the optical file
    optical = TOML.parsefile(joinpath(@__DIR__, "..", "options", "optical.toml"))
    validate_optical_file(optical)

    @debug """\n
    Parsing infrared file
    #######################################################
    """
    # Read in the infrared file
    infrared = TOML.parsefile(joinpath(@__DIR__, "..", "options", "infrared.toml"))
    validate_ir_file(infrared)

    # Add parameters iteratively
    abs_features = construct_extinction_params!(params, pnames, plabels, ptrans, out, optical, infrared, λunit, Iunit, region)
    construct_continuum_params!(params, pnames, plabels, ptrans, out, optical, infrared, λunit, Iunit, redshift, region)
    construct_template_params!(params, pnames, plabels, ptrans, out, infrared, region)    

    # Make final continuum object
    continuum = FitParameters(pnames, plabels, ptrans, params)
    # Dust features -- kept separate from the rest of the continuum parameters
    dust_features = create_dust_features(infrared, λunit, Iunit, region)

    # Get the spectral coverage of the region
    λrange = wavelength_range(region)

    # For Optical+IR fitting, let the infrared extinction parameters handle JUST the silicate absorption 
    # (i.e. set beta = 0)
    if λrange == UVOptIR
        @assert out[:silicate_absorption] ∈ ("kvt", "d+") "Only the kvt and d+ silicate absorption profiles are supported for joint IR+optical observations!"
        set_val!(continuum["extinction.beta"], 0.0)
        lock!(continuum["extinction.beta"])
    end
    # For optical only fitting, we cant constrain the silicate absorption, so also lock these parameters
    if λrange == UVOptical
        pwhere = "extinction." .* (out[:silicate_absorption] == "decompose" ? ["N_oli", "N_pyr", "N_for"] : ["tau_97"])
        append!(pwhere, "extinction" .* ["tau_ice", "tau_ch", "beta"])
        for pw in pwhere
            set_val!(continuum[pw], 0.0)
            lock!(continuum[pw])
        end
    end
    # For IR-only fitting, we cant constrain the reddening 
    if λrange == Infrared
        pwhere = "extinction." .* ["E_BV", "E_BV_factor"]
        if out[:fit_uv_bump]
            push!(pwhere, "extinction.delta_UV")
        end
        if out[:fit_covering_frac]
            push!(pwhere, "extinction.frac")
        end
        for pw in pwhere
            set_val!(continuum[pw], 0.0)
            lock!(continuum[pw])
        end
    end

    # Final few non-fit parameters
    sname = "statistics." .* ["chi2", "dof"]
    sparam = [NonFitParameter{Float64}(), NonFitParameter{Int}()]
    slabel = String[L"$\chi^2$", "d.o.f."]
    strans = [Transformation[], Transformation[]]
    statistics = NonFitParameters(sname, slabel, strans, sparam)

    continuum, abs_features, dust_features, statistics
end


# Check if the kinematics should be tied to other lines based on the kinematic groups
function check_tied_kinematics!(lines::Dict, prefix::String, line::String, kinematic_groups::Vector, fit_profiles::FitProfiles)

    for group ∈ kinematic_groups
        for groupmember ∈ lines["kinematic_group_" * group]

            #= Loop through the items in the "kinematic_group_*" list and see if the line name matches any of them.
            It needn't be a perfect match, the line name just has to contain the value in the kinematic group list.
            i.e. if you want to automatically tie all FeII lines together, instead of manually listing out each one,
            you can just include an item "FeII_" and it will automatically catch all the FeII lines. Note the underscore
            is important, otherwise "FeII" would also match "FeIII" lines.
            =#
            if occursin(groupmember, line)

                # Check if amp should  be tied
                params = []
                amp_ratio = nothing
                if haskey(lines, "tie_amp_" * group) && !isnothing(lines["tie_amp_" * group])
                    amp_ratio = lines["tie_amp_" * group][line]
                    push!(params, "amp")
                end
                # Check if voff should be tied
                push!(params, "voff")
                if haskey(lines, "tie_voff_" * group) && !lines["tie_voff_" * group]
                    pop!(params)
                end
                # Check if fwhm should be tied
                push!(params, "fwhm")
                if haskey(lines, "tie_fwhm_" * group) && !lines["tie_fwhm_" * group]
                    pop!(params)
                end

                for param in params

                    ind = prefix * "1." * param
                    @assert isnothing(fit_profiles[1].fit_parameters[ind].tie) "$(line) fulfills criteria" * 
                        " to be in multiple kinematic groups! Please amend your kinematic group filters."
                    @debug "Tying $param for $line to the group: $group"
                    # Use the group label
                    groupname = join([group, "1", param], "_") |> Symbol
                    if !isnothing(amp_ratio) 
                        tie!(fit_profiles[1].fit_parameters[ind], groupname, amp_ratio)
                    else
                        tie!(fit_profiles[1].fit_parameters[ind], groupname)
                    end

                    # Check if we need to change the values of any parameters now that it's tied 
                    if haskey(lines, "parameters") && haskey(lines["parameters"], group)
                        if haskey(lines["parameters"][group], param)
                            if haskey(lines["parameters"][group][param], "plim")
                                @debug "Overriding $param limits for $line in group $group"
                                units = unit(fit_profiles[1].fit_parameters[ind].value)
                                set_plim!(fit_profiles[1].fit_parameters[ind], 
                                          (lines["parameters"][line][param]["plim"]...,) .* units)
                            end
                            if haskey(lines["parameters"][group][param], "locked")
                                @debug "Overriding $param locked value for $line in group $group"
                                if lines["parameters"][group][param]["locked"]
                                    lock!(fit_profiles[1].fit_parameters[ind])
                                else
                                    unlock!(fit_profiles[1].fit_parameters[ind])
                                end
                            end
                            if haskey(lines["parameters"][group][param], "val")
                                @debug "Overriding $param initial value for $line"
                                units = unit(fit_profiles[1].fit_parameters[ind].value)
                                set_val!(fit_profiles[1].fit_parameters[ind], lines["parameters"][line][param]["val"] * units)
                            end 
                        end
                    end       

                end

                break
            end
        end
    end

end


# Same as above but for the additional components
function check_acomp_tied_kinematics!(lines::Dict, prefix::String, line::String, kinematic_groups::Vector, 
    fit_profiles::FitProfiles)

    for j ∈ 1:(length(fit_profiles)-1)
        for group ∈ kinematic_groups
            for groupmember ∈ lines["kinematic_group_" * group]
                if occursin(groupmember, line)

                    params = []
                    # Check if amp should be tied
                    amp_ratio = nothing
                    if haskey(lines, "tie_acomp_$(j)_amp_" * group) && !isnothing(lines["tie_acomp_$(j)_amp_" * group])
                        amp_ratio = lines["tie_acomp_$(j)_amp_" * group][line]
                        push!(params, "amp")
                    end
                    # Check if voff should be tied
                    push!(params, "voff")
                    if haskey(lines, "tie_acomp_$(j)_voff_" * group) && !lines["tie_acomp_$(j)_voff_" * group]
                        pop!(params)
                    end
                    # Check if fwhm should be tied
                    push!(params, "fwhm")
                    if haskey(lines, "tie_acomp_$(j)_fwhm_" * group) && lines["tie_acomp_$(j)_fwhm_" * group]
                        pop!(params)
                    end

                    for param in params
                        ind = prefix * "$(j+1)." * param
                        @assert isnothing(fit_profiles[j+1].fit_parameters[ind].tie) "$(line) acomp $(j) fulfills criteria" * 
                            " to be in multiple kinematic groups! Please amend your kinematic group filters."

                        @debug "Tying $param for $line acomp $j to the group: $group"
                        # Use the group label
                        groupname = join([group, "$(j+1)", param], "_") |> Symbol
                        if !isnothing(amp_ratio)
                            tie!(fit_profiles[j+1].fit_parameters[ind], groupname, amp_ratio) 
                        else
                            tie!(fit_profiles[j+1].fit_parameters[ind], groupname)
                        end

                        # Check if we need to change the values of any parameters now that it's tied 
                        param_key = "acomp_" * param
                        if haskey(lines, "parameters") && haskey(lines["parameters"], group)
                            if haskey(lines["parameters"][group], param_key)
                                if haskey(lines["parameters"][group][param_key][j], "plim")
                                    @debug "Overriding $param_key $j limits for $line in group $group"
                                    units = unit(fit_profiles[1].fit_parameters[ind])
                                    set_plim!(fit_profiles[1].fit_parameters[ind], 
                                              (lines["parameters"][line][param_key][j]["plim"]...,) .* units)
                                end
                                if haskey(lines["parameters"][group][param_key][j], "locked")
                                    @debug "Overriding $param_key $j locked value for $line in group $group"
                                    if lines["parameters"][group][param_key][j]["locked"]
                                        lock!(fit_profiles[1].fit_parameters[ind])
                                    else
                                        unlock!(fit_profiles[1].fit_parameters[ind])
                                    end
                                end
                                if haskey(lines["parameters"][group][param_key][j], "val")
                                    @debug "Overriding $param initial value for $line"
                                    units = unit(fit_profiles[1].fit_parameters[ind])
                                    set_val!(fit_profiles[1].fit_parameters[ind], lines["parameters"][line][param_key][j]["val"] * units)
                                end 
                            end
                        end    
                    end

                    break
                end
            end
        end
    end
end


function check_tied_voigt_mixing!(lines::Dict, prefix::String, line::String, fit_profiles::FitProfiles)
    # Go through each profile, check if it's a Voigt, and if so, tie it
    if lines["tie_voigt_mixing"] 
        group = :voigt_mixing
        for i in eachindex(fit_profiles)
            if fit_profiles[i].profile == :Voigt
                @debug "Tying mixing $i for $line to the group $group"
                tie!(fit_profiles[i].fit_parameters[prefix * "$i." * "mixing"], group)
            end
        end
    end
end

function default_line_parameters(out::Dict, lines::Dict, λunit::Unitful.Units, Iunit::Unitful.Units, prefix::String, profile::Symbol, 
    acomp_profiles::Vector{Union{Symbol,Nothing}})

    n_acomps = lines["n_acomps"]
    profs = FitProfiles([])

    all_profiles = [profile; acomp_profiles]

    # Define the initial values of line parameters given the values in the options file (if present)
    for component in 1:(n_acomps+1)

        # Skip acomp components that have not been specified
        if isnothing(all_profiles[component])
            continue
        end

        msg = "Component $component:"
        amp = component > 1 ? parameter_from_dict(lines["acomp_amp"][component-1]) : FitParameter(NaN, false, (0., Inf))
        t_amp = [RestframeTransform, LogTransform]
        if !out[:lines_allow_negative]
            push!(t_amp, LogTransform)
        end
        msg *= "\nAmp $amp"
        voff = parameter_from_dict(component > 1 ? lines["acomp_voff"][component-1] : lines["voff"]; units=u"km/s")
        msg *= "\nVoff $voff"
        fwhm = parameter_from_dict(component > 1 ? lines["acomp_fwhm"][component-1] : lines["fwhm"]; 
            units=component > 1 && lines["rel_fwhm"] ? NoUnits : u"km/s")
        msg *= "\nFWHM $fwhm"
        params = [amp, voff, fwhm]
        pnames = prefix .* "$component." .* ["amp", "voff", "fwhm"]
        plabels = String[out[:lines_allow_negative] ? L"$I$ (%$(latex(Iunit)))" : L"$\log_{10}(I / $ %$(latex(Iunit))$)$", 
            L"$v_{\rm off}$ (km s$^{-1}$)", L"FWHM (km s$^{-1}$)"]
        ptrans = [t_amp, Transformation[], Transformation[]]
        if all_profiles[component] == :GaussHermite
            h3 = parameter_from_dict(component > 1 ? lines["acomp_h3"][component-1] : lines["h3"])
            msg *= "\nh3 $h3"
            h4 = parameter_from_dict(component > 1 ? lines["acomp_h4"][component-1] : lines["h4"])
            msg *= "\nh4 $h4"
            append!(params, [h3, h4])
            append!(pnames, prefix .* "$component." .* ["h3", "h4"])
            append!(plabels, [L"$h_3$", L"$h_4$"])
            append!(ptrans, [Transformation[], Transformation[]])
        elseif all_profiles[component] == :Voigt
            η = parameter_from_dict(component > 1 ? lines["acomp_eta"][component-1] : lines["eta"])
            msg *= "\neta $η"
            push!(params, η)
            push!(pnames, prefix * "$component.mixing")
            push!(plabels, L"$\eta$")
            push!(ptrans, Transformation[])
        end
        @debug msg
        fit_parameters = FitParameters(pnames, plabels, ptrans, params)

        funit = u"erg/s/cm^2"
        nparams = [NonFitParameter{typeof(1.0*funit)}(), NonFitParameter{typeof(1.0*λunit)}(), NonFitParameter{Float64}()]
        npnames = prefix .* "$component." .* ["flux", "eqw", "SNR"]
        nplabels = String[out[:lines_allow_negative] ? L"$F$ (%$(latex(funit)))" : L"$\log_{10}(F /$ %$(latex(funit))$)$", 
            L"$W_{\rm eq}$ (%$(latex(λunit)))", L"$S/N$"]
        nptrans = [out[:lines_allow_negative] ? [LogTransform] : Transformation[], [RestframeTransform], Transformation[]]
        nonfit_parameters = NonFitParameters(npnames, nplabels, nptrans, nparams)

        push!(profs, FitProfile(profile, fit_parameters, nonfit_parameters))
    end

    funit = u"erg/s/cm^2"
    vunit = u"km/s"
    cpnames = prefix .* ["total_flux", "total_eqw", "total_snr", "n_comps", "w80", "delta_v", "vmed", "vpeak"]
    cparams = [NonFitParameter{typeof(1.0*funit)}(), NonFitParameter{typeof(1.0*λunit)}(), NonFitParameter{Float64}(), 
        NonFitParameter{Int}(), NonFitParameter{typeof(1.0*vunit)}(), NonFitParameter{typeof(1.0*vunit)}(), 
        NonFitParameter{typeof(1.0*vunit)}(), NonFitParameter{typeof(1.0*vunit)}()]
    cplabels = String[out[:lines_allow_negative] ? L"$F$ (%$(latex(funit)))" : L"$\log_{10}(F /$ %$(latex(funit))$)$", 
        L"$W_{\rm eq}$ (" * latex(λunit) * ")", L"$S/N$", L"$n_{\rm comp}$", L"$W_{80}$ (km s$^{-1}$)", 
        L"$\Delta v$ (km s$^{-1}$)", L"$v_{\rm med}$ (km s$^{-1}$)", L"$v_{\rm peak}$ (km s$^{-1}$)"]
    cptrans = [out[:lines_allow_negative] ? [LogTransform] : Transformation[], [RestframeTransform], Transformation[],
        Transformation[], Transformation[], Transformation[], Transformation[], Transformation[]]
    composite = NonFitParameters(cpnames, cplabels, cptrans, cparams)

    profs, composite
end


function override_line_parameters!(lines::Dict, prefix::String, fit_profiles::FitProfiles)

    # Check if there are any specific override values present in the options file,
    # and if so, use them
    if haskey(lines, "parameters") && haskey(lines["parameters"], line)
        for param_key in keys(lines["parameters"][line])  
            # Check normal line parameters (amp, voff, fwhm, h3, h4, eta)
            if !contains(param_key, "acomp")
                units = unit(fit_profiles[1].fit_parameters[prefix * "1." * param_key])
                if haskey(lines["parameters"][line][param_key], "plim")
                    @debug "Overriding $param_key limits for $line"
                    set_plim!(fit_profiles[1].fit_parameters[prefix * "1." * param_key], 
                                (lines["parameters"][line][param_key]["plim"]...,) .* units)
                end
                if haskey(lines["parameters"][line][param_key], "locked")
                    @debug "Overriding $param_key locked value for $line"
                    if lines["parameters"][line][param_key]["locked"]
                        lock!(fit_profiles[1].fit_parameters[prefix * "1." * param_key])
                    else
                        unlock!(fit_profiles[1].fit_parameters[prefix * "1." * param_key])
                    end
                end
                if haskey(lines["parameters"][line][param_key], "val")
                    @debug "Overriding $param_key initial value for $line"
                    set_val!(fit_profiles[1].fit_parameters[prefix * "1." * param_key], 
                                lines["parameters"][line][param_key]["val"] * units)
                end
            # Repeat for acomp parameters
            else
                for i in 1:length(lines["parameters"][line][param_key])
                    units = unit(fit_profiles[i+1].fit_parameters[prefix * "$(i+1)." * param_key])
                    if haskey(lines["parameters"][line][param_key][i], "plim")
                        @debug "Overriding $param_key acomp $i limits for $line"
                        set_plim!(fit_profiles[i+1].fit_parameters[prefix * "$(i+1)." * param_key], 
                                    (lines["parameters"][line][param_key][i]["plim"]...,) .* units)
                    end
                    if haskey(lines["parameters"][line][param_key][i], "locked")
                        @debug "Overriding $param_key acomp $i locked value for $line"
                        if lines["parameters"][line][param_key]["locked"]
                            lock!(fit_profiles[i+1].fit_parameters[prefix * "$(i+1)." * param_key])
                        else
                            unlock!(fit_profiles[i+1].fit_parameters[prefix * "$(i+1)." * param_key])
                        end
                    end
                    if haskey(lines["parameters"][line][param_key][i], "val")
                        @debug "Overriding $param_key acomp $i initial value for $line"
                        set_val!(fit_profiles[i+1].fit_parameters[prefix * "$(i+1)." * param_key], 
                                    lines["parameters"][line][param_key][i]["val"] * units)
                    end 
                end
            end
        end
    end
end



"""
    construct_line_parameters(out, λunit, Iunit, region)

Read in the lines.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary with Parameter objects for line fitting parameters.
This deals purely with emission line options.
"""
function construct_line_parameters(out::Dict, λunit::Unitful.Units, Iunit::Unitful.Units, region::SpectralRegion)

    @debug """\n
    Parsing lines file
    #######################################################
    """

    # Read in the lines file
    lines, cent_vals = parse_lines(region, λunit)

    # Create the kinematic groups
    #   ---> kinematic groups apply to all additional components of lines as well as the main component
    #        (i.e. the additional components in a kinematic group are tied to each other, but not to the main components
    #         or to the other additional components)
    kinematic_groups = []
    for key ∈ keys(lines)
        if occursin("kinematic_group_", key) && !occursin("acomp", key)
            append!(kinematic_groups, [replace(key, "kinematic_group_" => "")])
        end
    end

    # Get the profile types of each line
    profiles = Dict(ln => Symbol(lines["profiles"]["default"]) for ln ∈ keys(lines["lines"]))
    for line ∈ keys(lines["lines"])
        if haskey(lines["profiles"], line)
            profiles[line] = Symbol(lines["profiles"][line])
        end
    end

    # Keep in mind that a line can have multiple acomp profiles
    acomp_profiles = Dict{String, Vector{Union{Symbol,Nothing}}}()
    for line ∈ keys(lines["lines"])
        acomp_profiles[line] = Vector{Union{Symbol,Nothing}}(nothing, lines["n_acomps"])
        # check if the line is in a kinematic group
        group_name = nothing
        for group ∈ kinematic_groups
            for groupmember ∈ lines["kinematic_group_" * group]
                if occursin(groupmember, line) 
                    group_name = group
                end
            end
        end
        if !isnothing(group_name) && haskey(lines, "acomps")
            # if it's in a group, make sure individual profiles aren't being set
            if haskey(lines["acomps"], line) 
                error("Parsing error: $(line) is tied to the kinematic group $group_name, so you cannot add additional " *
                    "line profiles to this line individually.  Instead, add them to the group $group_name by setting " *
                    "\"$group_name = [ <profile types> ]\".")
            end
            if haskey(lines["acomps"], group_name)
                acomp_profiles[line] = vcat(Symbol.(lines["acomps"][group_name]), [nothing for _ in 1:(lines["n_acomps"]-length(lines["acomps"][group_name]))])
            end
        end
    end

    # Create buffers
    names = Symbol[]
    feat_labels = String[]
    annotate = BitVector()
    all_fit_profiles = FitProfiles[]
    sort_order = Int[]
    all_composite = NonFitParameters[]

    # Loop through all the lines
    for line ∈ keys(lines["lines"])

        name = Symbol(line)
        push!(names, name)
        push!(feat_labels, lines["lines"][line]["latex"])
        push!(annotate, lines["lines"][line]["annotate"])
        so = lines["default_sort_order"]
        if haskey(lines["lines"][line], "sort_order")
            so = lines["lines"][line]["sort_order"]
        end
        push!(sort_order, so)

        prof_out = profiles[line]
        acomp_prof_out = acomp_profiles[line]
        prefix = "lines.$(name)."

        @debug """\n
        ################# $line #######################
        # Rest wavelength: $(lines["lines"][line]["wave"]) $(lines["lines"][line]["unit"]) #
        """

        fit_profiles, composite = default_line_parameters(out, lines, λunit, Iunit, prefix, prof_out, acomp_prof_out)
        override_line_parameters!(lines, prefix, fit_profiles)

        @debug "Profile: $(profiles[line])"

        # Check if any of the amplitudes/voffs/fwhms should be tied to kinematic groups
        check_tied_kinematics!(lines, prefix, line, kinematic_groups, fit_profiles)
        # Repeat for the acomps
        check_acomp_tied_kinematics!(lines, prefix, line, kinematic_groups, fit_profiles)
        # Check the voigt mixing parameter
        check_tied_voigt_mixing!(lines, prefix, line, fit_profiles)

        push!(all_fit_profiles, fit_profiles)
        push!(all_composite, composite)
    end

    # sort by cent_vals
    ss = sortperm(cent_vals)

    # Check for any combined lines
    combined = []
    if haskey(lines, "combined_maps")
        combined = lines["combined_maps"]
        combined = [[Symbol(ln) for ln in c] for c in combined]
    end

    rel_amp = lines["rel_amp"]
    rel_voff = lines["rel_voff"]
    rel_fwhm = lines["rel_fwhm"]

    # Make the line profile object
    cfg = LineConfig(
        annotate[ss], 
        sort_order[ss],
        combined,
        rel_amp,
        rel_voff,
        rel_fwhm
    )
    lines_out = FitFeatures(names[ss], feat_labels[ss], cent_vals[ss], all_fit_profiles[ss], all_composite[ss], cfg)

    # Go through each tied group and set the value/limits/lock to match
    flat = get_flattened_fit_parameters(lines_out)
    fnames = flat.names
    tied_pairs, = get_tied_pairs(flat)
    for tp in tied_pairs
        n1 = fnames[tp[1]]
        ln_name_1 = split(n1, ".")[2]
        ln_comp_1 = parse(Int, split(n1, ".")[3])
        i1 = fast_indexin(ln_name_1, lines_out.names)
        n2 = fnames[tp[2]]
        ln_name_2 = split(n2, ".")[2]
        ln_comp_2 = parse(Int, split(n2, ".")[3])
        i2 = fast_indexin(ln_name_2, lines_out.names)
        # set plimits and locked values to match (including potential ratio)
        set_plim!(lines_out.profiles[i2][ln_comp_2].fit_parameters[n2], 
            lines_out.profiles[i1][ln_comp_1].fit_parameters[n1].limits .* tp[3])
        if lines_out.profiles[i1][ln_comp_1].fit_parameters[n1].locked 
            lock!(lines_out.profiles[i2][ln_comp_2].fit_parameters[n2])
        else
            unlock!(lines_out.profiles[i2][ln_comp_2].fit_parameters[n2])
        end
        # set initial values to match (including potential ratio)
        set_val!(lines_out.profiles[i2][ln_comp_2].fit_parameters[n2],
            lines_out.profiles[i1][ln_comp_1].fit_parameters[n1].value * tp[3])
    end

    @debug "#######################################################"
    lines_out
end


"""
    construct_model_parameters(out, λunit, Iunit, region)

The main function for constructing a ModelParameters object, which describes all the parameters of the model,
including those that are being fit and those that are not being fit.
"""
function construct_model_parameters(out::Dict, λunit::Unitful.Units, Iunit::Unitful.Units, region::SpectralRegion,
    redshift::Real)
    continuum, abs_features, dust_features, statistics = construct_continuum_parameters(out, λunit, Iunit, region, redshift)
    lines = construct_line_parameters(out, λunit, Iunit, region)
    ModelParameters(continuum, abs_features, dust_features, lines, statistics)
end
