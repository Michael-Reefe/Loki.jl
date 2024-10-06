#=
This file contains various functions for parsing configuration files and inputs and converting
them into code objects.
=#

############################## OPTIONS/SETUP/PARSING FUNCTIONS ####################################


function pah_name_to_float(name::String)
    # assumed to be formatted such that the wavelength is given to two decimal places,
    # i.e. "PAH_620" is interpreted at 6.20
    wl = split(name, "_")[end]
    parse(Float64, wl[1:end-2] * "." * wl[end-1:end])
end


"""
    validate_options_file(options)

Checks that the parsed options file has all of the keys that it should have.
"""
function validate_options_file(options)
    keylist1 = ["n_bootstrap", "extinction_curve", "extinction_screen", "fit_sil_emission", "fit_opt_na_feii", "fit_opt_br_feii", 
                "fit_all_global", "use_pah_templates", "fit_joint", "fit_uv_bump", "fit_covering_frac", "parallel", "plot_spaxels", 
                "plot_maps", "save_fits", "overwrite", "track_memory", "track_convergence", "save_full_model", "line_test_lines", 
                "line_test_threshold", "plot_line_test", "make_movies", "cosmology", "parallel_strategy", "bootstrap_use", 
                "random_seed", "sys_err", "olivine_y", "pyroxene_x", "grain_size", "fit_stellar_continuum", "fit_temp_multexp", 
                "decompose_lock_column_densities", "linemask_width"]
    keylist2 = ["h", "omega_m", "omega_K", "omega_r"]

    # Loop through the keys that should be in the file and confirm that they are there
    for key ∈ keylist1 
        @assert key ∈ keys(options) "Missing option $key in options file!"
    end
    for key ∈ keylist2
        @assert key ∈ keys(options["cosmology"]) "Missing option $key in cosmology options!"
    end
end


"""
    validate_dust_file(dust)

Checks that the parsed dust file has all of the keys that it should have.
"""
function validate_dust_file(dust::Dict)

    keylist1 = ["stellar_continuum_temp", "dust_features", "extinction", "hot_dust"]
    keylist2 = ["wave", "fwhm"]
    keylist3 = ["tau_9_7", "tau_ice", "tau_ch", "beta"]
    keylist4 = ["temp", "frac", "tau_warm", "tau_cold", "peak"]
    keylist5 = ["val", "plim", "locked"]
    for key ∈ keylist1
        @assert haskey(dust, key) "Missing option $key in dust file!"
    end
    for key ∈ keylist5
        @assert haskey(dust["stellar_continuum_temp"], key) "Missing option $key in stellar continuum temp options!"
        for df_key ∈ keys(dust["dust_features"])
            for df_key2 ∈ keylist2
                @assert haskey(dust["dust_features"][df_key], df_key2) "Missing option $df_key2 in dust feature $df_key options!"
                @assert haskey(dust["dust_features"][df_key][df_key2], key) "Missing option $key in dust features $df_key, $df_key2 options!"
            end
        end
        if haskey(dust, "absorption_features")
            for abs_key ∈ keys(dust["absorption_features"])
                for abs_key2 ∈ ["tau"; keylist2]
                    @assert haskey(dust["absorption_features"][abs_key], abs_key2) "Missing option $abs_key2 in absorption feature $abs_key options!"
                    @assert haskey(dust["absorption_features"][abs_key][abs_key2], key) "Missing option $key in absorption features $abs_key, $abs_key2 options!"
                end
            end
        end
        for ex_key ∈ keylist3
            @assert haskey(dust["extinction"], ex_key) "Missing option $ex_key in extinction options!"
            @assert haskey(dust["extinction"][ex_key], key) "Missing option $key in $ex_key options!"
        end
        for hd_key ∈ keylist4
            @assert haskey(dust["hot_dust"], hd_key) "Missing option $hd_key in hot dust options!"
            @assert haskey(dust["hot_dust"][hd_key], key) "Missing option $key in $hd_key options!"
        end
    end
end


"""
    validate_optical_file(optical)

Checks that the parsed optical file has all of the keys that it should have.
"""
function validate_optical_file(optical::Dict)

    keylist1 = ["attenuation", "stellar_population_ages", "stellar_population_metallicities", "stellar_kinematics", 
        "na_feii_kinematics", "br_feii_kinematics"]
    keylist2 = ["E_BV", "E_BV_factor", "uv_slope", "frac"]
    keylist3 = ["vel", "vdisp"]
    keylist4 = ["val", "plim", "locked"]

    for key ∈ keylist1
        @assert haskey(optical, key) "Missing option $key in optical file!"
    end
    for key ∈ keylist2
        @assert haskey(optical["attenuation"], key) "Missing option $key in attenuation options!"
    end
    for key ∈ keylist3
        @assert haskey(optical["stellar_kinematics"], key) "Missing option $key in stellar_kinematics options!"
        @assert haskey(optical["na_feii_kinematics"], key) "Missing option $key in na_feii_kinematics options!"
        @assert haskey(optical["br_feii_kinematics"], key) "Missing option $key in br_feii_kinematics options!"
    end
    for key4 ∈ keylist4
        for key1 ∈ keylist1
            for i in eachindex(optical[key1])
                @assert haskey(optical[key1][i], key4) "Missing option $key4 in $key1 options!"
            end
        end
        for key2 ∈ keylist2
            @assert haskey(optical["attenuation"][key2], key4) "Missing option $key4 in $key2 options!"
        end
        for key3 ∈ keylist3
            @assert haskey(optical["stellar_kinematics"][key3], key4) "Missing option $key4 in $key3 options!"
        end
    end
end


"""
    validate_lines_file(lines)

Checks that the parsed lines file has all of the keys that it should have, and
sorts the profiles and acomp_profiles keys into dictionaries.
"""
function validate_lines_file(lines)

    keylist1 = ["default_sort_order", "tie_voigt_mixing", "lines", "profiles", "acomps", "n_acomps", 
        "rel_amp", "rel_fwhm", "rel_voff"]
    keylist2 = ["wave", "latex", "annotate", "unit"]

    keylist3 = ["voff", "fwhm", "h3", "h4", "acomp_amp", "acomp_voff", "acomp_fwhm"]
    keylist4 = ["val", "plim", "locked"]

    # Loop through all the required keys that should be in the file and confirm that they are there
    for key ∈ keylist1
        @assert haskey(lines, key) "$key not found in line options!"
    end
    for key ∈ keys(lines["lines"])
        for key2 ∈ keylist2
            @assert haskey(lines["lines"][key], key2) "$key2 not found in $key line options!"
        end
    end
    for key ∈ keylist3
        @assert haskey(lines, key) "$key not found in line options!"
        for key2 ∈ keylist4
            @assert haskey(lines[key], key2) "$key2 not found in $key line options!"
        end
    end

    @assert haskey(lines["profiles"], "default") "default not found in line profile options!"
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
        if haskey(lines, "acomps")
            if haskey(lines["acomps"], line)
                # Pad to match the length of n_acomps
                acomp_profiles[line] = vcat(Symbol.(lines["acomps"][line]), [nothing for _ in 1:(lines["n_acomps"]-length(lines["acomps"][line]))])
            end
        end
    end
    
    profiles, acomp_profiles
end


"""
    parse_options()

Read in the options.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary.  This deals with general/top-level code configurations.
"""
function parse_options()

    @debug """\n
    Parsing options file
    #######################################################
    """

    # Read in the options file
    options = TOML.parsefile(joinpath(@__DIR__, "..", "options", "options.toml"))
    validate_options_file(options)

    # Convert keys to symbols
    options = Dict(Symbol(k) => v for (k, v) ∈ options)

    # Convert cosmology keys into a proper cosmology object
    options[:cosmology] = cosmology(h=options[:cosmology]["h"], 
                                    OmegaM=options[:cosmology]["omega_m"],
                                    OmegaK=options[:cosmology]["omega_K"],
                                    OmegaR=options[:cosmology]["omega_r"])
    
    # logging messages 
    @debug "Options: $options"

    options
end


# Template amplitudes
function construct_template_params!(params::Vector{FitParameters}, pnames::Vector{String}, out::Dict, 
    contin_options::Dict, region::SpectralRegion)

    if haskey(contin_options, "template_amps") && !out[:fit_temp_multexp]
        msg = "Template amplitudes:"
        for i ∈ eachindex(contin_options["template_amps"])
            tname = out[:template_names][i]
            for ni in 1:nchannels(region)
                temp_A = parameter_from_dict(contin_options["template_amps"][i])
                msg *= "\n$temp_A channel $ni"
                push!(params, temp_A)
                push!(pnames, "templates.$(tname).amp_$ni")
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
        end
        @debug msg
    end

end


"""
    create_dust_features(dust, λlim, user_mask)

Uses the dust options file to create a DustFeatures object for the
PAH features.
"""
function create_dust_features(dust::Dict, region::SpectralRegion; do_absorption::Bool=false)

    # Dust feature central wavelengths and FWHMs
    cent_vals = Float64[]
    complexes = String[]
    latex = String[]
    fit_profiles = FitProfiles()
    key = do_absorption ? "absorption_features" : "dust_features"
    short_key = do_absorption ? "abs_features" : "dust_features"

    msg = "$(do_absorption ? "Absorption" : "Dust") features:"
    for df ∈ keys(dust[key])

        # First things first, check if this feature is within our wavelength range
        cent_val = dust[key][df]["wave"]["val"]
        if !is_valid(cent_val, 0.1, region)
            continue
        end

        profile = :Drude
        complex = df

        prefix = "$(short_key).$(df)."

        # amplitudes (not in file)
        if do_absorption
            amp = parameter_from_dict(dust[key][df]["tau"])
        else
            amp = FitParameter(NaN, false, (0., Inf))   # the NaN is a placeholder for now and will be replaced
        end
        msg *= "\n$(do_absorption ? "Depth" : "Amp") $(amp)"
        mean = parameter_from_dict_wave(dust[key][df]["wave"])
        msg *= "\nWave $(mean)"
        fwhm = parameter_from_dict_fwhm(dust[key][df]["fwhm"])
        msg *= "\nFWHM $(fwhm)"

        names = prefix .* [do_absorption ? "tau" : "amp", "mean", "fwhm"]
        _params = [amp, mean, fwhm]

        if haskey(dust[key][df], "index") && haskey(dust[key][df], "cutoff") && !do_absorption 
            profile = :PearsonIV
            index = parameter_from_dict(dust[key][df]["index"])
            msg *= "\nIndex $(index)"
            cutoff = parameter_from_dict(dust[key][df]["cutoff"])
            msg *= "\nCutoff $(cutoff)"
            append!(names, prefix .* ["index", "cutoff"])
            append!(_params, [index, cutoff])
        end
        if profile == :Drude
            push!(names, prefix * "asym")
            if haskey(dust[key][df], "asym")
                asym = parameter_from_dict_fwhm(dust[key][df]["asym"])  # (fwhm method so that uncertainties are fractional)
                msg *= "\nAsym $(asym)"
                push!(_params, asym)
            else # only add a default asym parameter if its a Drude profile
                asym = FitParameter(0., true, (-0.01, 0.01))
                msg *= "\nAsym $(asym)"
                push!(_params, asym)
            end
        end
        if haskey(dust[key][df], "complex")
            complex = dust[key][df]["complex"]
        end

        fit_parameters = FitParameters(names, _params)

        # non-fit parameters
        if !do_absorption
            npnames = prefix .* ["flux", "eqw", "SNR"]
            nparams = [NonFitParameter() for _ in 1:3]
            nonfit_parameters = NonFitParameters(npnames, nparams)
        else
            nonfit_parameters = NonFitParameters(String[], NonFitParameter[])
        end

        push!(fit_profiles, FitProfile(profile, fit_parameters, nonfit_parameters))
        push!(cent_vals, cent_val) 
        push!(complexes, complex)

        wl = pah_name_to_float(complex)
        push!(latex, "PAH $wl " * L"${\rm \mu m}$")
    end
    @debug msg

    # Sort by cent_vals, then reshape 
    ss = sortperm(cent_vals)
    cent_vals = cent_vals[ss]
    complexes = complexes[ss]
    latex = latex[ss]
    fit_profiles = fit_profiles[ss]

    # sort into vectors based on where the complex is the same
    all_fit_profiles = FitProfiles[]
    u_complexes = unique(complexes)
    u_cent_vals = Float64[]
    u_latex = String[]
    for complex in u_complexes
        inds = findall(complexes .== complex)
        push!(all_fit_profiles, length(inds) > 1 ? fit_profiles[inds] : [fit_profiles[inds]])
        wl = pah_name_to_float(complex)
        push!(u_cent_vals, wl)
        push!(u_latex, "PAH $wl " * L"${\rm \mu m}$")
    end

    FitFeatures(u_complexes, u_latex, u_cent_vals, all_fit_profiles, NoConfig())
end


"""
    _construct_parameters_ir(out, region)

Read in the dust.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary with Parameter objects for dust fitting parameters.
This deals with continuum, PAH features, and extinction options.
"""
function _construct_parameters_ir(out::Dict, region::SpectralRegion)

    @debug """\n
    Parsing dust file
    #######################################################
    """

    # Read in the dust file
    dust = TOML.parsefile(joinpath(@__DIR__, "..", "options", "dust.toml"))

    # Loop through all of the required keys that should be in the file and confirm that they are there
    validate_dust_file(dust)

    # Convert the options into Parameter objects, and set them to the output dictionary
    pnames = String[]
    params = FitParameter[]

    # Dust continuum temperatures
    if haskey(dust, "dust_continuum_temps")
        msg = "Dust continuum:"
        for i in eachindex(dust["dust_continuum_temps"])
            prefix = "continuum.dust.$(i)."
            A_dc = FitParameter(NaN, false, (0., Inf))
            msg *= "\nAmp $A_dc"
            T_dc = parameter_from_dict(dust["dust_continuum_temps"][i])
            msg *= "\nTemp $T_dc"
            @debug msg
            append!(params, [A_dc, T_dc])
            append!(pnames, prefix .* ["amp", "temp"])
        end
    end
        
    # Extinction parameters, optical depth and mixing ratio
    msg = "Extinction:"
    prefix = "extinction."
    # Write tau_9_7 value based on the provided guess
    if out[:extinction_curve] == "decompose"
        N_oli = parameter_from_dict(dust["extinction"]["N_oli"])
        msg *= "\nN_oli $N_oli"
        N_pyr = parameter_from_dict(dust["extinction"]["N_pyr"])
        msg *= "\nN_pyr $N_pyr"
        N_for = parameter_from_dict(dust["extinction"]["N_for"])
        msg *= "\nN_for $N_for"
        append!(params, [N_oli, N_pyr, N_for])
        append!(pnames, prefix .* ["N_oli", "N_pyr", "N_for"])
    else
        τ_97 = parameter_from_dict(dust["extinction"]["tau_9_7"])
        msg *= "\nTau_sil $τ_97"
        push!(params, τ_97)
        push!(pnames, prefix * "tau_97")
    end
    τ_ice = parameter_from_dict(dust["extinction"]["tau_ice"])
    msg *= "\nTau_ice $τ_ice"
    τ_ch = parameter_from_dict(dust["extinction"]["tau_ch"])
    msg *= "\nTau_CH $τ_ch"
    β = parameter_from_dict(dust["extinction"]["beta"])
    msg *= "\nBeta $β"
    Cf = parameter_from_dict(dust["extinction"]["frac"])
    msg *= "\nFrac $Cf"
    @debug msg
    append!(params, [τ_ice, τ_ch, β, Cf])
    append!(pnames, prefix .* ["tau_ice", "tau_ch", "beta", "frac"])

    abs_features = create_dust_features(dust, region; do_absorption=true)
    abs_parameters = get_flattened_fit_parameters(abs_features)
    append!(params, abs_parameters._parameters)
    append!(pnames, abs_parameters.names)

    # Hot dust parameters, temperature, covering fraction, warm tau, and cold tau
    msg = "Hot Dust:"
    if out[:fit_sil_emission]
        prefix = "continuum.hot_dust."
        A_hot = FitParameter(NaN, false, (0., Inf))
        msg *= "\nAmp $A_hot"
        T_hot = parameter_from_dict(dust["hot_dust"]["temp"])
        msg *= "\nTemp $T_hot"
        hd_Cf = parameter_from_dict(dust["hot_dust"]["frac"])
        msg *= "\nFrac $hd_Cf"
        τ_warm = parameter_from_dict(dust["hot_dust"]["tau_warm"])
        msg *= "\nTau_Warm $τ_warm"
        τ_cold = parameter_from_dict(dust["hot_dust"]["tau_cold"])
        msg *= "\nTau_Cold $τ_cold"
        sil_peak = parameter_from_dict(dust["hot_dust"]["peak"])
        msg *= "\nSil_Peak $sil_peak"
        @debug msg
        append!(params, [A_hot, T_hot, hd_Cf, τ_warm, τ_cold, sil_peak])
        append!(pnames, prefix .* ["amp", "temp", "frac", "tau_warm", "tau_cold", "sil_peak"])
    end

    # Appends the template parameters and pnames in-place
    construct_template_params!(params, pnames, out, dust, region)

    # All of the continuum and PAH parameters conjoined into neat little objects
    continuum_parameters = FitParameters(pnames, params)
    dust_features = create_dust_features(dust, region)

    continuum_parameters, dust_features
end


"""
    _construct_parameters_opt(out, region)

Read in the optical.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary with Parameter objects for optical fitting parameters.
This deals with the optical continuum options.
"""
function _construct_parameters_opt(out::Dict, region::SpectralRegion)

    @debug """\n
    Parsing optical file
    #######################################################
    """

    # Read in the dust file
    optical = TOML.parsefile(joinpath(@__DIR__, "..", "options", "optical.toml"))

    validate_optical_file(optical)

    params = FitParameter[]
    pnames = String[]    

    if out[:fit_stellar_continuum]

        msg = "Stellar populations:"
        for (i, (age, metal)) ∈ enumerate(zip(optical["stellar_population_ages"], optical["stellar_population_metallicities"]))
            prefix = "continuum.stellar_populations.$(i)."
            mass = FitParameter(NaN, false, (0., Inf)) 
            msg *= "\nMass $mass"
            a = parameter_from_dict(age)
            msg *= "\nAge $a"
            z = parameter_from_dict(metal)
            msg *= "\nMetallicity $z"
            append!(params, [mass, a, z])
            append!(pnames, prefix .* ["mass", "age", "metallicity"])
        end
        @debug msg

        msg = "Stellar kinematics:"
        prefix = "continuum.stellar_kinematics."
        stel_vel = parameter_from_dict(optical["stellar_kinematics"]["vel"])
        msg *= "\nVelocity $stel_vel"
        stel_vdisp = parameter_from_dict(optical["stellar_kinematics"]["vdisp"])
        msg *= "\nVdisp $stel_vdisp"
        append!(params, [stel_vel, stel_vdisp])
        append!(pnames, prefix .* ["vel", "vdisp"])
        @debug msg

    end

    # attenuation parameters
    msg = "Attenuation:"
    prefix = "attenuation."
    E_BV = parameter_from_dict(optical["attenuation"]["E_BV"])
    E_BV_factor = parameter_from_dict(optical["attenuation"]["E_BV_factor"])
    # if wavelength range is infrared, dont try fitting a full attenuation curve
    # (the silicate absorption features are still included)
    is_ir = wavelength_range(region) == Infrared
    if is_ir
        set_val!(E_BV, 0.)
        lock!(E_BV)
        lock!(E_BV_factor)
    end
    msg *= "\nE(B-V) $E_BV"
    msg *= "\nE(B-V) factor $E_BV_factor"
    append!(params, [E_BV, E_BV_factor])
    append!(pnames, prefix .* ["E_BV", "E_BV_factor"])
    if out[:fit_uv_bump]
        δ_uv = parameter_from_dict(optical["attenuation"]["uv_slope"])
        if is_ir
            set_val!(δ_uv, 0.)
            lock!(δ_uv)
        end
        msg *= "\nδ_uv $δ_uv"
        push!(params, δ_uv)
        push!(pnames, prefix * "delta_UV")
    end
    if out[:fit_covering_frac]
        frac = parameter_from_dict(optical["attenuation"]["frac"])
        if is_ir
            set_val!(frac, 0.)
            lock!(frac)
        end
        msg *= "\nfrac $frac"
        push!(params, frac)
        push!(pnames, prefix * "frac")
        @debug msg
    end

    msg = "Fe II kinematics:"
    prefix = "continuum.feii."
    if out[:fit_opt_na_feii]
        na_feii_A = FitParameter(NaN, false, (0., Inf))
        msg *= "\nNA Amp $na_feii_A"
        na_feii_vel = parameter_from_dict(optical["na_feii_kinematics"]["vel"])
        msg *= "\nNA Velocity $na_feii_vel"
        na_feii_vdisp = parameter_from_dict(optical["na_feii_kinematics"]["vdisp"])
        msg *= "\nNA Vdisp $na_feii_vdisp"
        append!(params, [na_feii_A, na_feii_vel, na_feii_vdisp])
        append!(pnames, prefix .* "na." .* ["amp", "vel", "vdisp"])
    end
    if out[:fit_opt_br_feii]
        br_feii_A = FitParameter(NaN, false, (0., Inf))
        msg *= "\nBR Amp $br_feii_A"
        br_feii_vel = parameter_from_dict(optical["br_feii_kinematics"]["vel"])
        msg *= "\nBR Velocity $br_feii_vel"
        br_feii_vdisp = parameter_from_dict(optical["br_feii_kinematics"]["vdisp"])
        msg *= "\nBR Vdisp $br_feii_vdisp"
        append!(params, [br_feii_A, br_feii_vel, br_feii_vdisp])
        append!(pnames, prefix .* "br." .* ["amp", "vel", "vdisp"])
    end
    @debug msg

    if haskey(optical, "power_law_indices")
        msg = "Power Laws:"
        for (i, power_law_index) in enumerate(optical["power_law_indices"])
            prefix = "continuum.power_law.$(i)"
            A_pl = FitParameter(NaN, false, (0., Inf))
            msg *= "\nAmp $A_pl"
            α_pl = parameter_from_dict(power_law_index)
            msg *= "\nIndex $α_pl"
            append!(params, [A_pl, α_pl])
            append!(pnames, prefix .* ["amp", "index"])
        end
        @debug msg
    end

    # Appends the template parameters and pnames in-place
    construct_template_params!(params, pnames, out, optical, 1)

    # Pack everything up into a FitParameters object
    continuum_parameters = FitParameters(pnames, params)
 
    continuum_parameters
end


function construct_parameters(out::Dict, region::SpectralRegion)

    continuum_ir, dust_features = construct_parameters(out, region)
    continuum = construct_parameters(out, region)

    append!(continuum, continuum_ir)

    continuum, dust_features
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
                if haskey(lines, "tie_voff_" * group) && lines["tie_voff_" * group]
                    push!(params, "voff")
                end
                # Check if fwhm should be tied
                if haskey(lines, "tie_fwhm_" * group) && lines["tie_fwhm_" * group]
                    push!(params, "fwhm")
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
                                set_plim!(fit_profiles[1].fit_parameters[ind], (lines["parameters"][line][param]["plim"]...,))
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
                                set_val!(fit_profiles[1].fit_parameters[ind], lines["parameters"][line][param]["val"])
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
function check_acomp_tied_kinematics!(lines::Dict, prefix::String, line::String, acomp_kinematic_groups::Vector, 
    fit_profiles::FitProfiles)

    for j ∈ 1:(length(fit_profiles)-1)
        for group ∈ acomp_kinematic_groups[j]
            for groupmember ∈ lines["acomp_$(j)_kinematic_group_" * group]
                if occursin(groupmember, line)

                    params = []
                    # Check if amp should be tied
                    amp_ratio = nothing
                    if haskey(lines, "tie_acomp_$(j)_amp_" * group) && !isnothing(lines["tie_acomp_$(j)_amp_" * group])
                        amp_ratio = lines["tie_acomp_$(j)_amp_" * group][line]
                        push!(params, "amp")
                    end
                    # Check if voff should be tied
                    if haskey(lines, "tie_acomp_$(j)_voff_" * group) && lines["tie_acomp_$(j)_voff_" * group]
                        push!(params, "voff")
                    end
                    # Check if fwhm should be tied
                    if haskey(lines, "tie_acomp_$(j)_fwhm_" * group) && lines["tie_acomp_$(j)_fwhm_" * group]
                        push!(params, "fwhm")
                    end

                    for param in params
                        ind = prefix * "$(j+1)." * param
                        @assert isnothing(fit_profiles[j+1].fit_parameters[ind].tied) "$(line) acomp $(j) fulfills criteria" * 
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
                                    set_plim!(fit_profiles[1].fit_parameters[ind], (lines["parameters"][line][param_key][j]["plim"]...,))
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
                                    set_val!(fit_profiles[1].fit_parameters[ind], lines["parameters"][line][param_key][j]["val"])
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

function default_line_parameters(lines::Dict, prefix::String, profile::Symbol, acomp_profiles::Vector{Union{Symbol,Nothing}})

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
        msg *= "\nAmp $amp"
        voff = parameter_from_dict(component > 1 ? lines["acomp_voff"][component-1] : lines["voff"])
        msg *= "\nVoff $voff"
        fwhm = parameter_from_dict(component > 1 ? lines["acomp_fwhm"][component-1] : lines["fwhm"])
        msg *= "\nFWHM $fwhm"
        params = [amp, voff, fwhm]
        pnames = prefix .* "$component." .* ["amp", "voff", "fwhm"]
        if all_profiles[component] == :GaussHermite
            h3 = parameter_from_dict(component > 1 ? lines["acomp_h3"][component-1] : lines["h3"])
            msg *= "\nh3 $h3"
            h4 = parameter_from_dict(component > 1 ? lines["acomp_h4"][component-1] : lines["h4"])
            msg *= "\nh4 $h4"
            append!(params, [h3, h4])
            append!(pnames, prefix .* "$component." .* ["h3", "h4"])
        elseif all_profiles[component] == :Voigt
            η = parameter_from_dict(component > 1 ? lines["acomp_eta"][component-1] : lines["eta"])
            msg *= "\neta $η"
            push!(params, η)
            push!(pnames, prefix * "$component.mixing")
        end
        @debug msg
        fit_parameters = FitParameters(pnames, params)

        nparams = [NonFitParameter() for _ in 1:3]
        npnames = prefix .* "$component." .* ["flux", "eqw", "SNR"]
        nonfit_parameters = NonFitParameters(npnames, nparams)

        push!(profs, FitProfile(profile, fit_parameters, nonfit_parameters))
    end

    profs
end


function override_line_parameters!(lines::Dict, prefix::String, fit_profiles::FitProfiles)

    # Check if there are any specific override values present in the options file,
    # and if so, use them
    if haskey(lines, "parameters") && haskey(lines["parameters"], line)
        for param_key in keys(lines["parameters"][line])  
            # Check normal line parameters (amp, voff, fwhm, h3, h4, eta)
            if !contains(param_key, "acomp")
                if haskey(lines["parameters"][line][param_key], "plim")
                    @debug "Overriding $param_key limits for $line"
                    set_plim!(fit_profiles[1].fit_parameters[prefix * "1." * param_key], 
                                (lines["parameters"][line][param_key]["plim"]...,))
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
                                lines["parameters"][line][param_key]["val"])
                end
            # Repeat for acomp parameters
            else
                for i in 1:length(lines["parameters"][line][param_key])
                    if haskey(lines["parameters"][line][param_key][i], "plim")
                        @debug "Overriding $param_key acomp $i limits for $line"
                        set_plim!(fit_profiles[i+1].fit_parameters[prefix * "$(i+1)." * param_key], 
                                    (lines["parameters"][line][param_key][i]["plim"]...,))
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
                                    lines["parameters"][line][param_key][i]["val"])
                    end 
                end
            end
        end
    end
end


function parse_lines(region::SpectralRegion)

    lines = TOML.parsefile(joinpath(@__DIR__, "..", "options", "lines.toml"))
    profiles, acomp_profiles = validate_lines_file(lines)

    cent_vals = Vector{typeof(1.0u"μm")}()    # provide all the line wavelengths in microns
    for line in keys(lines["lines"])
        cent_unit = uparse(replace(lines["lines"][line]["unit"], 'u' => 'μ'), unit_context=UnitfulAstro)
        cent_val = lines["lines"][line]["wave"] * cent_unit
        if !is_valid(cent_val, 0, region)
            # remove this line from the dictionary
            delete!(lines["lines"], line)
            delete!(profiles, line)
            delete!(acomp_profiles, line)
            continue
        end
        push!(cent_vals, cent_val)
    end

    lines, profiles, acomp_profiles, cent_vals
end


"""
    construct_line_parameters(out, λlim)

Read in the lines.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary with Parameter objects for line fitting parameters.
This deals purely with emission line options.
"""
function construct_line_parameters(out::Dict, region::SpectralRegion)

    @debug """\n
    Parsing lines file
    #######################################################
    """

    # Read in the lines file
    lines, profiles, acomp_profiles, cent_vals = parse_lines(region)

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
    acomp_kinematic_groups = []
    for j ∈ 1:lines["n_acomps"]
        acomp_kinematic_group_j = []
        for key ∈ keys(lines)
            if occursin("acomp_$(j)_kinematic_group_", key)
                append!(acomp_kinematic_group_j, [replace(key, "acomp_$(j)_kinematic_group_" => "")])
            end
        end
        append!(acomp_kinematic_groups, [acomp_kinematic_group_j])
    end

    # Create buffers
    names = Symbol[]
    latex = String[]
    annotate = BitVector()
    all_fit_profiles = FitProfiles[]
    sort_order = Int[]

    # Loop through all the lines
    for line ∈ keys(lines["lines"])

        name = Symbol(line)
        push!(names, name)
        push!(latex, lines["lines"][line]["latex"])
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

        fit_profiles = default_line_parameters(lines, prefix, prof_out, acomp_prof_out)
        override_line_parameters!(lines, prefix, fit_profiles)

        @debug "Profile: $(profiles[line])"

        # Check if any of the amplitudes/voffs/fwhms should be tied to kinematic groups
        check_tied_kinematics!(lines, prefix, line, kinematic_groups, fit_profiles)
        # Repeat for the acomps
        check_acomp_tied_kinematics!(lines, prefix, line, acomp_kinematic_groups, fit_profiles)
        # Check the voigt mixing parameter
        check_tied_voigt_mixing!(lines, prefix, line, fit_profiles)

        push!(all_fit_profiles, fit_profiles)
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
    lines_out = FitFeatures(names[ss], latex[ss], cent_vals[ss], all_fit_profiles[ss], cfg)

    @debug "#######################################################"

    lines_out
end


"""
    read_smith_temps()

Setup function for reading in the PAH templates from Smith et al. 2006 used in 
Questfit (https://github.com/drupke/questfit/tree/v1.0)
"""
function read_smith_temps()
    
    path3 = joinpath(@__DIR__, "..", "templates", "smith_nftemp3.pah.ext.dat")
    path4 = joinpath(@__DIR__, "..", "templates", "smith_nftemp4.pah.next.dat")
    @debug "Reading in Smith et al. 2006 PAH templates from QUESTFIT: $path3 and $path4"

    temp3 = CSV.read(path3, DataFrame, delim=' ', ignorerepeated=true, stripwhitespace=true,
        header=["rest_wave", "flux"])
    temp4 = CSV.read(path4, DataFrame, delim=' ', ignorerepeated=true, stripwhitespace=true,
        header=["rest_wave", "flux"])

    temp3[!, "rest_wave"], temp3[!, "flux"], temp4[!, "rest_wave"], temp4[!, "flux"]
end


"""
    read_ice_ch_temp()

Setup function for reading in the Ice+CH absorption templates from Donnan's 
PAHDecomp (https://github.com/FergusDonnan/PAHDecomp/tree/main/Ice%20Templates)
"""
function read_ice_ch_temps()
    path1 = joinpath(@__DIR__, "..", "templates", "IceExt.txt")
    path2 = joinpath(@__DIR__, "..", "templates", "CHExt.txt")
    @debug "Reading in Ice+CH templates from: $path1 and $path2"

    temp1 = CSV.read(path1, DataFrame, delim=' ', comment="#", ignorerepeated=true, stripwhitespace=true,
        header=["rest_wave", "tau"])
    temp2 = CSV.read(path2, DataFrame, delim=' ', comment="#", ignorerepeated=true, stripwhitespace=true,
        header=["rest_wave", "tau"])
    
    temp1[!, "rest_wave"], temp1[!, "tau"], temp2[!, "rest_wave"], temp2[!, "tau"]
end


"""
    read_irs_data(path)

Setup function for reading in the configuration IRS spectrum of IRS 08572+3915

# Arguments
- `path::String`: The file path pointing to the IRS 08572+3915 spectrum 
"""
function read_irs_data(path::String)

    @debug "Reading in IRS data from: $path"

    datatable = CSV.read(path, DataFrame, comment="#", delim=' ', ignorerepeated=true, stripwhitespace=true,
        header=["rest_wave", "flux", "e_flux", "enod", "order", "module", "nod1flux", "nod2flux", "e_nod1flux", "e_nod2flux"])

    datatable[!, "rest_wave"], datatable[!, "flux"], datatable[!, "e_flux"]
end


"""
    silicate_dp()

Setup function for creating a silicate extinction profile based on Donnan et al. (2022)
"""
function silicate_dp()

    @debug "Creating Donnan+2022 optical depth profile..."

    # Read in IRS 08572+3915 data from 00000003_0.ideos.mrt
    λ_irs, F_irs, σ_irs = read_irs_data(joinpath(@__DIR__, "..", "templates", "00000003_0.ideos.mrt"))
    # Get flux values at anchor points + endpoints
    anchors = [4.9, 5.5, 7.8, 13.0, 14.5, 26.5, λ_irs[end]]
    values = zeros(length(anchors))
    for (i, anchor) ∈ enumerate(anchors)
        _, ki = findmin(k -> abs(k - anchor), λ_irs)
        values[i] = F_irs[ki]
    end

    # Cubic spline fit with specific anchor points
    cubic_spline_irs = Spline1D(anchors, values; k=3)

    # Get optical depth
    τ_DS = log10.(cubic_spline_irs.(λ_irs) ./ F_irs)
    # Smooth data and remove features < ~7.5 um
    τ_smooth = movmean(τ_DS, 5)
    v1, p1 = findmin(τ_DS[λ_irs .< 6])
    v2, p2 = findmin(τ_DS[7 .< λ_irs .< 8])
    slope_beg = (v2 - v1) / (λ_irs[7 .< λ_irs .< 8][p2] - λ_irs[λ_irs .< 6][p1])
    beg_filt = λ_irs .< λ_irs[7 .< λ_irs .< 8][p2]
    τ_smooth[beg_filt] .= v1 .+ slope_beg .* (λ_irs[beg_filt] .- λ_irs[1])
    mid_filt = (λ_irs[7 .< λ_irs .< 8][p2] .- 0.5) .< λ_irs .< (λ_irs[7 .< λ_irs .< 8][p2] .+ 0.5)  
    τ_smooth[mid_filt] = movmean(τ_smooth, 5)[mid_filt]

    # Normalize to value at 9.7
    τ_98 = τ_smooth[findmin(abs.(λ_irs .- 9.8))[2]]
    τ_λ = τ_smooth ./ τ_98

    λ_irs, τ_λ
end


# Setup function for creating the extinction profile from Chiar+Tielens 2006
function silicate_ct()
    data = CSV.read(joinpath(@__DIR__, "..", "templates", "chiar+tielens_2005.dat"), DataFrame, skipto=15, delim=' ', 
        ignorerepeated=true, header=["wave", "a_galcen", "a_local"], select=[1, 2])
    data[!, "wave"], data[!, "a_galcen"]
end


# Setup function for creating the extinction profile from OHM+92
function silicate_ohm()
    data = CSV.read(joinpath(@__DIR__, "..", "templates", "ohmc.txt"), DataFrame, delim=' ', ignorerepeated=true,
        header=["wave", "ext"])
    data[!, "ext"] ./= 0.4
    data[!, "wave"], data[!, "ext"]
end


function read_dust_κ(x::Real, y::Real, a::Real)

    # Get wavelength, x, y, and a arrays
    λ = readdlm(joinpath(@__DIR__, "..", "templates", "dorschner_wave.txt"), ' ', Float64, '\n')[:,1]
    ρ_oli = 3.71    # g/cm^3
    ρ_pyr = 3.20    # g/cm^3
    a_cm = a / 1e4  # convert a from μm to cm

    # Read in the Qabs/Qsca arrays
    q_abs_oli = readdlm(joinpath(@__DIR__, "..", "templates", "dorschner_qabs_oli_$(y)_$(a).txt"), ' ', Float64, '\n')[:,1]
    q_abs_pyr = readdlm(joinpath(@__DIR__, "..", "templates", "dorschner_qabs_pyr_$(x)_$(a).txt"), ' ', Float64, '\n')[:,1]

    # Convert absorption efficiencies into mass absorption coefficients (cm^2/g)
    κ_abs_oli = @. 3 * q_abs_oli / (4 * a_cm * ρ_oli)
    κ_abs_pyr = @. 3 * q_abs_pyr / (4 * a_cm * ρ_pyr)

    # Create interpolating functions over wavelength
    κ_abs_pyr = Spline1D(λ, κ_abs_pyr, k=3)
    κ_abs_oli = Spline1D(λ, κ_abs_oli, k=3)

    # Read in the mass absorption coefficiencts for crystalline forsterite
    for_data = readdlm(joinpath(@__DIR__, "..", "templates", "tamani_crystalline_forsterite_k.txt"), ' ', Float64, '\n', comments=true)

    # Create interpolating function 
    # extend edges to 0
    λ_for = [for_data[1,1]-0.2; for_data[1,1]-0.1; for_data[:, 1]; for_data[end, 1]+0.1; for_data[end, 1]+0.2]
    κ_abs_for = [0.; 0.; for_data[:, 2]; 0.; 0.]
    κ_abs_for = Spline1D(λ_for, κ_abs_for, k=3)

    κ_abs_oli, κ_abs_pyr, κ_abs_for
end


"""
    generate_stellar_populations(λ, lsf, z, Ω, cosmo, name)

Prepare a 3D grid of Simple Stellar Population (SSP) templates over age, metallicity, and wavelength.
Each template will be cropped around the region of interest given by `λ` (in microns), and degraded 
to a spectral resolution given by `lsf` (in km/s). The redshift `z` and cosmology `cosmo` are used to
calculate a luminosity distance so that the templates can be normalized into units of erg/s/cm^2/ang/Msun.

The templates are generated using the Python version of Charlie Conroy's Flexible Stellar Population Synthesis (FSPS)
package, converted to Python by Dan Foreman-Mackey (and callable within Julia using PyCall).

Returns 1D arrays for the wavelengths, ages, and metals that the templates are evaluated at, as well as the
3D array of templates. Note that some of the templates may be filled with 0s if the grid space is not fully
occupied.
"""
function generate_stellar_populations(λ::Vector{<:Real}, lsf::Vector{<:Real}, z::Real, cosmo::Cosmology.AbstractCosmology,
    name::String)

    # Make sure λ is logarithmically binned
    @assert isapprox((λ[2]/λ[1]), (λ[end]/λ[end-1]), rtol=1e-6) "Input spectrum must be logarithmically binned to fit optical data!"

    # Test to see if templates have already been generated
    if isfile(joinpath("output_$name", "stellar_templates.loki"))
        @info "Loading pre-generated stellar templates from binary file"
        out = deserialize(joinpath("output_$name", "stellar_templates.loki"))
        return out.λ, out.age, out.logz, out.templates
    end

    # Cut off the templates a little bit before/after the input spectrum
    λleft, λright = minimum(λ)*0.98, maximum(λ)*1.02

    # Dummy population
    ssp0 = py_fsps.StellarPopulation()
    # Get the wavelength grid that FSPS uses
    ssp_λ = ssp0.wavelengths
    # Convert from angstroms to microns
    ssp_λ ./= 1e4
    @assert (λleft ≥ minimum(ssp_λ)) && (λright ≤ maximum(ssp_λ)) "The extended input spectrum range of ($λleft, $λright) um " * 
        "is outside the FSPS template range of ($(minimum(ssp_λ)), $(maximum(ssp_λ))) um. Please adjust the input spectrum accordingly."

    # Mask to a range around the input spectrum
    mask = λleft .< ssp_λ .< λright
    ssp_λ = ssp_λ[mask]
    # Resample onto a linear wavelength grid
    Δλ = (λright - λleft) / length(ssp_λ)
    ssp_λlin = collect(λleft:Δλ:λright)
    
    # LSF FWHM of the input spectrum in microns, interpolated at the locations of the SSP templates
    inp_fwhm = Spline1D(λ, lsf ./ C_KMS .* λ, k=1, bc="nearest")(ssp_λlin)
    # FWHM resolution of the FSPS templates in um
    ssp_lsf = Spline1D(ssp_λ, abs.(ssp0.resolutions[mask]), k=1, bc="nearest")(ssp_λlin)
    ssp_fwhm = ssp_lsf ./ C_KMS .* ssp_λlin .* 2√(2log(2))
    # Difference in resolutions between the input spectrum and SSP templates, in pixels
    dfwhm = sqrt.(clamp.(inp_fwhm.^2 .- ssp_fwhm.^2, 0., Inf)) ./ Δλ 

    # Logarithmically rebinned wavelengths
    logscale = log(λ[2]/λ[1])
    ssp_lnλ = get_logarithmic_λ(ssp_λlin, logscale)

    # Calculate luminosity distance in cm in preparation to convert luminosity to flux
    dL = luminosity_dist(u"cm", cosmo, z).val

    # Generate templates over a range of ages and metallicities
    ages = exp.(range(log(0.001), log(13.7), 50))        # logarithmically spaced from 1 Myr to 15 Gyr
    logzs = range(-2.3, 0.4, 10)                         # linearly spaced from log(Z/Zsun) = [M/H] = -2.3 to 0.4
    ssp_templates = zeros(length(ages), length(logzs), length(ssp_lnλ))
    n_temp = size(ssp_templates, 1) * size(ssp_templates, 2)

    @info "Generating $n_temp simple stellar population templates with FSPS with " * 
        "ages ∈ ($(minimum(ages)), $(maximum(ages))) Gyr, [M/H] ∈ ($(minimum(logzs)), $(maximum(logzs)))"

    prog = Progress(n_temp; showspeed=true)
    for (z_ind, logz) in enumerate(logzs)
        # Create a simple stellar population (delta function SFH) with a Salpeter IMF and no attenuation
        ssp = py_fsps.StellarPopulation(zcontinuous=1, logzsol=logz, imf_type=0, sfh=0)
        for (age_ind, age) in enumerate(ages)
            # Evaluate the stellar population at the given age
            _, ssp_LperA = ssp.get_spectrum(tage=age, peraa=true)
            # Convert Lsun/Msun/Ang to erg/s/cm^2/Ang/Msun
            ssp_flux = ssp_LperA[mask] .* 3.846e33 ./ (4π .* dL.^2)
            # Resample onto the linear wavelength grid
            ssp_flux = Spline1D(ssp_λ, ssp_flux, k=1, bc="nearest")(ssp_λlin)
            # Convolve with gaussian kernels to degrade the spectrum to match the input spectrum's resolution
            ssp_flux = convolveGaussian1D(ssp_flux, dfwhm)
            # Resample again, this time onto the logarithmic wavelength grid
            ssp_flux = Spline1D(ssp_λlin, ssp_flux, k=1, bc="nearest")(ssp_lnλ)
            # Add to the templates array
            ssp_templates[age_ind, z_ind, :] .= ssp_flux
            next!(prog)
        end
    end

    # save for later
    serialize(joinpath("output_$name", "stellar_templates.loki"), (λ=ssp_lnλ, age=ages, logz=logzs, templates=ssp_templates))

    ssp_lnλ, ages, logzs, ssp_templates
end


"""
    generate_feii_templates(λ, lsf)

Prepare two Fe II emission templates from Veron-Cetty et al. (2004): https://www.aanda.org/articles/aa/pdf/2004/14/aa0714.pdf
derived from the spectrum of I Zw 1 (Seyfert 1). This function loads the templates from the files, converts to vacuum wavelengths,
interpolates the flux onto a logarithmically spaced grid, and convolves the templates to match the spectral resolution of the
input spectrum (given by the `lsf` argument, in km/s).

The returned values are the length of the templates in wavelength space, the wavelength grid, and the Fourier Transforms of the
templates themselves (this is done to speed up the convolution with a LOSVD during the actual fitting -- we cannot return the 
FFTs of the stellar templates because they may have to be interpolated between age/metallicity, but since the Fe II templates are
fixed, we are free to pre-compute the FFTs).
"""
function generate_feii_templates(λ::Vector{<:Real}, lsf::Vector{<:Real})

    # Make sure λ is logarithmically binned
    @assert isapprox((λ[2]/λ[1]), (λ[end]/λ[end-1]), rtol=1e-6) "Input spectrum must be logarithmically binned to fit optical data!"

    # Read the templates in from the specified directory
    template_path = joinpath(@__DIR__, "..", "templates", "veron-cetty_2004")
    na_feii_temp, _ = readdlm(joinpath(template_path, "VC04_na_feii_template.csv"), ',', Float64, '\n', header=true)
    br_feii_temp, _ = readdlm(joinpath(template_path, "VC04_br_feii_template.csv"), ',', Float64, '\n', header=true)
    feii_λ = na_feii_temp[:, 1]
    na_feii_temp = na_feii_temp[:, 2]
    br_feii_temp = br_feii_temp[:, 2]

    # Convert to vacuum wavelengths
    feii_λ = airtovac.(feii_λ)
    # Convert to microns
    feii_λ ./= 1e4
    # Linear spacing 
    Δλ = (maximum(feii_λ) - minimum(feii_λ)) / length(feii_λ)

    # Cut off the templates a little bit before/after the input spectrum
    λleft, λright = minimum(λ)*0.98, maximum(λ)*1.02 
    inrange = (λleft ≥ minimum(feii_λ)) && (λright ≤ maximum(feii_λ))
    if !inrange
        @debug "The input spectrum falls outside the range covered by the Fe II templates. The templates will be padded with 0s."
        λpad_l = collect(λleft:Δλ:minimum(feii_λ))
        λpad_r = collect(maximum(feii_λ):Δλ:λright)
        feii_λ = [λpad_l; feii_λ; λpad_r]
        Fpad_l = zeros(length(λpad_l))
        Fpad_r = zeros(length(λpad_r))
        na_feii_temp = [Fpad_l; na_feii_temp; Fpad_r]
        br_feii_temp = [Fpad_l; br_feii_temp; Fpad_r]
    else
        mask = λleft .< feii_λ .< λright
        feii_λ = feii_λ[mask]
        na_feii_temp = na_feii_temp[mask]
        br_feii_temp = br_feii_temp[mask]
    end

    # Resample to a linear wavelength grid
    feii_λlin = collect(λleft:Δλ:λright)
    na_feii_temp = Spline1D(feii_λ, na_feii_temp, k=1, bc="nearest")(feii_λlin)
    br_feii_temp = Spline1D(feii_λ, br_feii_temp, k=1, bc="nearest")(feii_λlin)

    # LSF FWHM of the input spectrum in microns, interpolated at the locations of the SSP templates
    inp_fwhm = Spline1D(λ, lsf ./ C_KMS .* λ, k=1, bc="nearest")(feii_λlin)
    # FWHM resolution of the Fe II templates in um
    feii_fwhm = 1.0/1e4
    # Difference in resolutions between the input spectrum and SSP templates, in pixels
    dfwhm = sqrt.(clamp.(inp_fwhm.^2 .- feii_fwhm.^2, 0., Inf)) ./ Δλ

    # Convolve the templates to match the resolution of the input spectrum
    na_feii_temp = convolveGaussian1D(na_feii_temp, dfwhm)
    br_feii_temp = convolveGaussian1D(br_feii_temp, dfwhm)

    # Logarithmically rebinned wavelengths
    logscale = log(λ[2]/λ[1])
    feii_lnλ = get_logarithmic_λ(feii_λlin, logscale)
    na_feii_temp = Spline1D(feii_λlin, na_feii_temp, k=1, bc="nearest")(feii_lnλ)
    br_feii_temp = Spline1D(feii_λlin, br_feii_temp, k=1, bc="nearest")(feii_lnλ)

    # Pad with 0s up to the next product of small prime factors -> to make the FFT more efficient
    npad = nextprod([2,3,5], length(feii_lnλ))
    na_feii_temp = [na_feii_temp; zeros(npad - length(na_feii_temp))]
    br_feii_temp = [br_feii_temp; zeros(npad - length(br_feii_temp))]

    # Re-normalize the spectra so the maximum is 1
    na_feii_temp ./= maximum(na_feii_temp)
    br_feii_temp ./= maximum(br_feii_temp)

    # Pre-compute the Fourier Transforms of the templates to save time during fitting
    na_feii_temp_rfft = rfft(na_feii_temp)
    br_feii_temp_rfft = rfft(br_feii_temp)

    npad, feii_lnλ, na_feii_temp_rfft, br_feii_temp_rfft

end


"""
    update_global_convergence_log(cube_fitter, spaxel, res)

Updates the convergence log file with results from a global simulated annealing fit with Optim.
"""
function update_global_convergence_log(cube_fitter::CubeFitter, spaxel::CartesianIndex, res)
    global file_lock
    # use the ReentrantLock to prevent multiple processes from trying to write to the same file at once
    lock(file_lock) do 
    open(joinpath("output_$(cube_fitter.name)", "loki.convergence.log"), "a") do conv
    redirect_stdout(conv) do
        label = isone(length(spaxel)) ? "Voronoi bin $(spaxel[1])" : "Spaxel ($(spaxel[1]),$(spaxel[2]))"
        println("$label on worker $(myid()):")
        println(res)
        println("-------------------------------------------------------")
    end
    end
    end
end


"""
    write_memory_log(cube_fitter, fname)

Writes the log file that contains information on memory usage for a particular spaxel fit.
"""
function write_memory_log(cube_fitter::CubeFitter, fname::String)
    open(joinpath("output_$(cube_fitter.name)", "logs", "mem.$fname.log"), "w") do f

        print(f, """
        ### PROCESS ID: $(getpid()) ###
        Memory usage stats:
        CubeFitter - $(Base.summarysize(cube_fitter) ÷ 10^6) MB
            Cube - $(Base.summarysize(cube_fitter.cube) ÷ 10^6) MB 
        """)

        print(f, """
        $(InteractiveUtils.varinfo(all=true, imported=true, recursive=true))
        """)
    end
end


"""
    write_fit_results_csv(cube_fitter, fname, p_out, p_err)

Writes the CSV file that contains the fit results for a particular spaxel
(best fit values and errors).
"""
function write_fit_results_csv(cube_fitter::CubeFitter, fname::String, 
    p_out::Vector{<:Real}, p_err::Matrix{<:Real})

    open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.csv"), "w") do f 
        writedlm(f, [p_out p_err], ',')
    end

end


"""
    read_fit_results_csv(cube_fitter, fname)

Reads the CSV file that contains the file results for a particular spaxel
and returns the best fit values and errors as separate vectors (errors is a 
2D matrix with the lower/upper errors)
"""
function read_fit_results_csv(cube_fitter::CubeFitter, fname::String)
    results = readdlm(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.csv"), ',', Float64, '\n')
    p_out = results[:, 1]
    p_err = results[:, 2:3]
    p_out, p_err
end


# Helper function for saving the outputs of the initial integrated fit to both the CubeFitter object
# and to csv files
function save_init_fit_outputs!(cube_fitter::CubeFitter, popt::Vector{<:Real}, pah_amp::Vector{<:Real})
    # Save the results to a file and to the CubeFitter object
    # save running best fit parameters in case the fitting is interrupted
    cube_fitter.p_init_cont[:] .= popt
    open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_cont.csv"), "w") do f
        writedlm(f, cube_fitter.p_init_cont, ',')
    end
    cube_fitter.p_init_pahtemp[:] .= pah_amp
    open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_pahtemp.csv"), "w") do f
        writedlm(f, cube_fitter.p_init_pahtemp, ',')
    end
end


# Alternative dispatch for the line fit (doesnt include the pah_amp argument)
function save_init_fit_outputs!(cube_fitter::CubeFitter, popt::Vector{<:Real})
    # Save results to file
    cube_fitter.p_init_line[:] .= copy(popt)
    open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "init_fit_line.csv"), "w") do f
        writedlm(f, cube_fitter.p_init_line, ',')
    end
end
