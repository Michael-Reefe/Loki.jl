#=
This file contains various functions for parsing configuration files and inputs and converting
them into code objects.
=#

############################## OPTIONS/SETUP/PARSING FUNCTIONS ####################################

"""
    parse_resolving(z, channel)

Read in the resolving_mrs.csv configuration file to create a cubic spline interpolation of the
MIRI MRS resolving power as a function of wavelength.  The function that is returned provides the
FWHM of the line-spread function as a function of (observed-frame) wavelength.

# Arguments
- `channel::String`: The channel of the fit
"""
function parse_resolving(channel::String)

    @debug "Parsing MRS resoling power from resolving_mrs.csv for channel $channel"

    # Read in the resolving power data
    resolve = readdlm(joinpath(@__DIR__, "..", "templates", "resolving_mrs.csv"), ',', Float64, '\n', header=true)
    wave = resolve[1][:, 1]
    R = resolve[1][:, 2]

    # Find points where wavelength jumps down (b/w channels)
    jumps = diff(wave) .< 0
    indices = eachindex(wave)
    ind_left = indices[BitVector([0; jumps])]
    ind_right = indices[BitVector([jumps; 0])]

    # Channel 1: everything before jump 3
    if channel == "1"
        edge_left = 1
        edge_right = ind_right[3]
    # Channel 2: between jumps 3 & 6
    elseif channel == "2"
        edge_left = ind_left[3]
        edge_right = ind_right[6]
    # Channel 3: between jumps 6 & 9
    elseif channel == "3"
        edge_left = ind_left[6]
        edge_right = ind_right[9]
    # Channel 4: everything above jump 9
    elseif channel == "4"
        edge_left = ind_left[9]
        edge_right = length(wave)
    elseif channel == "MULTIPLE"
        edge_left = 1
        edge_right = length(wave)
    else
        error("Unrecognized channel: $(channel)")
    end

    # Filter down to the channel we want
    wave = wave[edge_left:edge_right]
    R = R[edge_left:edge_right]

    # Now get the jumps within the individual channel we're interested in
    jumps = diff(wave) .< 0

    # Define regions of overlapping wavelength space
    wave_left = wave[BitVector([0; jumps])]
    wave_right = wave[BitVector([jumps; 0])]

    # Sort the data to be monotonically increasing in wavelength
    ss = sortperm(wave)
    wave = wave[ss]
    R = R[ss]

    # Smooth the data in overlapping regions
    for i ∈ 1:sum(jumps)
        region = wave_left[i] .≤ wave .≤ wave_right[i]
        R[region] .= movmean(R, 10)[region]
    end

    # Create a linear interpolation function so we can evaluate it at the points of interest for our data,
    # taking an input wi in the OBSERVED frame
    interp_R = Spline1D(wave, R, k=1)

    # The line-spread function in km/s - reduce by 25% from the pre-flight data
    lsf = wi -> C_KMS / interp_R(wi)
    
    lsf
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
    keylist1 = ["n_bootstrap", "extinction_curve", "extinction_screen", "fit_sil_emission", "fit_opt_na_feii", "fit_opt_br_feii", 
                "fit_all_global", "use_pah_templates", "fit_joint", "fit_uv_bump", "fit_covering_frac", "parallel", "plot_spaxels", 
                "plot_maps", "save_fits", "overwrite", "track_memory", "track_convergence", "save_full_model", "line_test_lines", 
                "line_test_threshold", "plot_line_test", "make_movies", "cosmology"]
    keylist2 = ["h", "omega_m", "omega_K", "omega_r"]

    # Loop through the keys that should be in the file and confirm that they are there
    for key ∈ keylist1 
        @assert key ∈ keys(options) "Missing option $key in options file!"
    end
    for key ∈ keylist2
        @assert key ∈ keys(options["cosmology"]) "Missing option $key in cosmology options!"
    end

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


"""
    parse_dust()

Read in the dust.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary with Parameter objects for dust fitting parameters.
This deals with continuum, PAH features, and extinction options.
"""
function parse_dust()

    @debug """\n
    Parsing dust file
    #######################################################
    """

    # Read in the dust file
    dust = TOML.parsefile(joinpath(@__DIR__, "..", "options", "dust.toml"))
    keylist1 = ["stellar_continuum_temp", "dust_features", "extinction", "hot_dust"]
    keylist2 = ["wave", "fwhm"]
    keylist3 = ["tau_9_7", "tau_ice", "tau_ch", "beta"]
    keylist4 = ["temp", "frac", "tau_warm", "tau_cold", "peak"]
    keylist5 = ["val", "plim", "locked"]

    # Loop through all of the required keys that should be in the file and confirm that they are there
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

    # Convert the options into Parameter objects, and set them to the output dictionary

    # Stellar continuum temperature
    T_s = from_dict(dust["stellar_continuum_temp"])
    @debug "Stellar continuum:\nTemp $T_s"

    # Dust continuum temperatures
    if haskey(dust, "dust_continuum_temps")
        T_dc = [from_dict(dust["dust_continuum_temps"][i]) for i ∈ eachindex(dust["dust_continuum_temps"])]
        msg = "Dust continuum:"
        for dci ∈ T_dc
            msg *= "\nTemp $dci"
        end
        @debug msg
    else
        T_dc = []
    end
        
    # Power law indices
    if haskey(dust, "power_law_indices")
        α = [from_dict(dust["power_law_indices"][i]) for i ∈ eachindex(dust["power_law_indices"])]
        msg = "Power laws:"
        for αi ∈ α
            msg *= "\nAlpha $αi"
        end
        @debug msg
    else
        α = []
    end

    # Template amplitudes
    if haskey(dust, "template_amps")
        temp_A = [from_dict(dust["template_amps"][i]) for i ∈ eachindex(dust["template_amps"])]
        msg = "Template amplitudes:"
        for Ai ∈ temp_A
            msg *= "\n$Ai"
        end
        @debug msg
    else
        temp_A = []
    end

    # Dust feature central wavelengths and FWHMs
    cent_vals = zeros(length(dust["dust_features"]))
    name = Vector{String}(undef, length(dust["dust_features"]))
    mean = Vector{Parameter}(undef, length(dust["dust_features"]))
    fwhm = Vector{Parameter}(undef, length(dust["dust_features"]))
    index = Vector{Union{Parameter,Nothing}}(nothing, length(dust["dust_features"]))
    cutoff = Vector{Union{Parameter,Nothing}}(nothing, length(dust["dust_features"]))
    complexes = Vector{Union{String,Nothing}}(nothing, length(dust["dust_features"]))
    _local = falses(length(dust["dust_features"]))
    profiles = [:Drude for _ in 1:length(name)]

    msg = "Dust features:"
    for (i, df) ∈ enumerate(keys(dust["dust_features"]))
        name[i] = df
        mean[i] = from_dict_wave(dust["dust_features"][df]["wave"])
        msg *= "\nWave $(mean[i])"
        fwhm[i] = from_dict_fwhm(dust["dust_features"][df]["fwhm"])
        msg *= "\nFWHM $(fwhm[i])"
        if haskey(dust["dust_features"][df], "index")
            index[i] = from_dict(dust["dust_features"][df]["index"])
            profiles[i] = :PearsonIV
            msg *= "\nIndex $(index[i])"
        end
        if haskey(dust["dust_features"][df], "cutoff")
            cutoff[i] = from_dict(dust["dust_features"][df]["cutoff"])
            profiles[i] = :PearsonIV
            msg *= "\nCutoff $(cutoff[i])"
        end
        if haskey(dust["dust_features"][df], "complex")
            complexes[i] = dust["dust_features"][df]["complex"]
        end
        cent_vals[i] = mean[i].value
    end
    @debug msg

    # Sort by cent_vals
    ss = sortperm(cent_vals)
    dust_features = DustFeatures(name[ss], profiles[ss], mean[ss], fwhm[ss], index[ss], cutoff[ss], complexes[ss], _local[ss])

    # Repeat for absorption features
    if haskey(dust, "absorption_features")
        cent_vals = zeros(length(dust["absorption_features"]))
        name = Vector{String}(undef, length(dust["absorption_features"]))
        depth = Vector{Parameter}(undef, length(dust["absorption_features"]))
        mean = Vector{Parameter}(undef, length(dust["absorption_features"]))
        fwhm = Vector{Parameter}(undef, length(dust["absorption_features"]))
        complexes = Vector{Union{String,Nothing}}(nothing, length(dust["absorption_features"]))
        _local = falses(length(dust["absorption_features"]))

        msg = "Absorption features:"
        for (i, ab) ∈ enumerate(keys(dust["absorption_features"]))
            name[i] = ab
            depth[i] = from_dict(dust["absorption_features"][ab]["tau"])
            msg *= "\nTau $(depth[i])"
            mean[i] = from_dict_wave(dust["absorption_features"][ab]["wave"])
            msg *= "\nWave $(mean[i])"
            fwhm[i] = from_dict_fwhm(dust["absorption_features"][ab]["fwhm"])
            msg *= "\nFWHM $(fwhm[i])"
            cent_vals[i] = mean[i].value

            if haskey(dust["absorption_features"][ab], "local")
                _local[i] = dust["absorption_features"][ab]["local"]
            end
        end
        @debug msg

        # Sort by cent_vals
        ss = sortperm(cent_vals)
        abs_features = DustFeatures(name[ss], [:Drude for _ in 1:length(name)], mean[ss], fwhm[ss],
            Union{Parameter,Nothing}[nothing for _ in 1:length(name)], 
            Union{Parameter,Nothing}[nothing for _ in 1:length(name)],
            complexes[ss], _local[ss])
        abs_taus = depth[ss]
    else
        abs_features = DustFeatures(String[], Symbol[], Parameter[], Parameter[], Vector{Union{Parameter,Nothing}}(),
            Vector{Union{Parameter,Nothing}}(), Vector{Union{String,Nothing}}(), BitVector[])
        abs_taus = Vector{Parameter}()
    end

    # Extinction parameters, optical depth and mixing ratio
    msg = "Extinction:"
    # Write tau_9_7 value based on the provided guess
    # dust["extinction"]["tau_9_7"]["val"] = τ_guess
    τ_97 = from_dict(dust["extinction"]["tau_9_7"])
    msg *= "\nTau_sil $τ_97"
    τ_pah = from_dict(dust["extinction"]["tau_pah"])
    msg *= "\nTau_PAH $τ_pah"
    N_oli = from_dict(dust["extinction"]["N_oli"])
    msg *= "\nN_oli $N_oli"
    N_pyr = from_dict(dust["extinction"]["N_pyr"])
    msg *= "\nN_pyr $N_pyr"
    N_for = from_dict(dust["extinction"]["N_for"])
    msg *= "\nN_for $N_for"
    τ_ice = from_dict(dust["extinction"]["tau_ice"])
    msg *= "\nTau_ice $τ_ice"
    τ_ch = from_dict(dust["extinction"]["tau_ch"])
    msg *= "\nTau_CH $τ_ch"
    β = from_dict(dust["extinction"]["beta"])
    msg *= "\nBeta $β"
    Cf = from_dict(dust["extinction"]["frac"])
    msg *= "\nFrac $Cf"
    @debug msg

    # Hot dust parameters, temperature, covering fraction, warm tau, and cold tau
    msg = "Hot Dust:"
    # Write warm_tau and col_tau values based on the provided guess
    # dust["hot_dust"]["tau_warm"]["val"] = τ_guess
    # dust["hot_dust"]["tau_cold"]["val"] = τ_guess
    T_hot = from_dict(dust["hot_dust"]["temp"])
    msg *= "\nTemp $T_hot"
    hd_Cf = from_dict(dust["hot_dust"]["frac"])
    msg *= "\nFrac $hd_Cf"
    τ_warm = from_dict(dust["hot_dust"]["tau_warm"])
    msg *= "\nTau_Warm $τ_warm"
    τ_cold = from_dict(dust["hot_dust"]["tau_cold"])
    msg *= "\nTau_Cold $τ_cold"
    sil_peak = from_dict(dust["hot_dust"]["peak"])
    msg *= "\nSil_Peak $sil_peak"
    @debug msg

    # Create continuum object
    continuum = MIRContinuum(T_s, T_dc, α, τ_97, τ_pah, N_oli, N_pyr, N_for, τ_ice, τ_ch, β, Cf, 
        T_hot, hd_Cf, τ_warm, τ_cold, sil_peak, temp_A)

    continuum, dust_features, abs_features, abs_taus
end


"""
    parse_optical()

Read in the optical.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary with Parameter objects for optical fitting parameters.
This deals with the optical continuum options.
"""
function parse_optical()

    @debug """\n
    Parsing optical file
    #######################################################
    """

    # Read in the dust file
    optical = TOML.parsefile(joinpath(@__DIR__, "..", "options", "optical.toml"))
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

    msg = "Stellar populations:"
    ssp_ages = Parameter[]
    ssp_metallicities = Parameter[]
    for (age, metal) ∈ zip(optical["stellar_population_ages"], optical["stellar_population_metallicities"])
        a = from_dict(age)
        push!(ssp_ages, a)
        msg *= "\nAge $a"
        z = from_dict(metal)
        msg *= "\nMetallicity $z"
        push!(ssp_metallicities, z)
    end
    @debug msg

    msg = "Stellar kinematics:"
    stel_vel = from_dict(optical["stellar_kinematics"]["vel"])
    msg *= "\nVelocity $stel_vel"
    stel_vdisp = from_dict(optical["stellar_kinematics"]["vdisp"])
    msg *= "\nVdisp $stel_vdisp"
    @debug msg

    msg = "Fe II kinematics:"
    na_feii_vel = from_dict(optical["na_feii_kinematics"]["vel"])
    msg *= "\nNA Velocity $na_feii_vel"
    na_feii_vdisp = from_dict(optical["na_feii_kinematics"]["vdisp"])
    msg *= "\nNA Vdisp $na_feii_vdisp"
    br_feii_vel = from_dict(optical["br_feii_kinematics"]["vel"])
    msg *= "\nBR Velocity $br_feii_vel"
    br_feii_vdisp = from_dict(optical["br_feii_kinematics"]["vdisp"])
    msg *= "\nBR Vdisp $br_feii_vdisp"
    @debug msg

    α = Parameter[]
    if haskey(optical, "power_law_indices")
        msg *= "Power Laws:"
        α = [from_dict(optical["power_law_indices"][i]) for i in eachindex(optical["power_law_indices"])]
        for αi in α
            msg *= "\nIndex $αi"
        end
        @debug msg
    end
 
    # attenuation parameters
    msg = "Attenuation:"
    E_BV = from_dict(optical["attenuation"]["E_BV"])
    msg *= "\nE(B-V) $E_BV"
    E_BV_factor = from_dict(optical["attenuation"]["E_BV_factor"])
    msg *= "\nE(B-V) factor $E_BV_factor"
    δ_uv = from_dict(optical["attenuation"]["uv_slope"])
    msg *= "\nδ_uv $δ_uv"
    frac = from_dict(optical["attenuation"]["frac"])
    msg *= "\nfrac $frac"
    @debug msg

    continuum = OpticalContinuum(ssp_ages, ssp_metallicities, stel_vel, stel_vdisp, na_feii_vel, na_feii_vdisp, 
        br_feii_vel, br_feii_vdisp, α, E_BV, E_BV_factor, δ_uv, frac)

    continuum
end


"""
    parse_lines()

Read in the lines.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary with Parameter objects for line fitting parameters.
This deals purely with emission line options.
"""
function parse_lines()

    @debug """\n
    Parsing lines file
    #######################################################
    """

    # Read in the lines file
    lines = TOML.parsefile(joinpath(@__DIR__, "..", "options", "lines.toml"))

    keylist1 = ["default_sort_order", "tie_voigt_mixing", "voff_plim", "fwhm_plim", "h3_plim", "h4_plim", "acomp_voff_plim", 
        "acomp_fwhm_plim", "flexible_wavesol", "wavesol_unc", "lines", "profiles", "acomps", "n_acomps"]
    keylist2 = ["wave", "latex", "annotate"]

    # Loop through all the required keys that should be in the file and confirm that they are there
    for key ∈ keylist1
        @assert haskey(lines, key) "$key not found in line options!"
    end
    for key ∈ keys(lines["lines"])
        for key2 in keylist2
            @assert haskey(lines["lines"][key], key2) "$key2 not found in $key line options!"
        end
    end
    @assert haskey(lines["profiles"], "default") "default not found in line profile options!"
    profiles = Dict(ln => lines["profiles"]["default"] for ln ∈ keys(lines["lines"]))
    if haskey(lines, "profiles")
        for line ∈ keys(lines["lines"])
            if haskey(lines["profiles"], line)
                profiles[line] = lines["profiles"][line]
            end
        end
    end
    # Keep in mind that a line can have multiple acomp profiles
    acomp_profiles = Dict{String, Vector{Union{String,Nothing}}}()
    for line ∈ keys(lines["lines"])
        acomp_profiles[line] = Vector{Union{String,Nothing}}(nothing, lines["n_acomps"])
        if haskey(lines, "acomps")
            if haskey(lines["acomps"], line)
                # Pad to match the length of n_acomps
                acomp_profiles[line] = vcat(lines["acomps"][line], [nothing for _ in 1:(lines["n_acomps"]-length(lines["acomps"][line]))])
            end
        end
    end

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

    # Initialize undefined vectors for each TransitionLine attribute that will be filled in
    names = Vector{Symbol}(undef, length(lines["lines"]))
    latex = Vector{String}(undef, length(lines["lines"]))
    annotate = BitVector(undef, length(lines["lines"]))
    cent_vals = zeros(length(lines["lines"]))
    voffs = Vector{Parameter}(undef, length(lines["lines"]))
    fwhms = Vector{Parameter}(undef, length(lines["lines"]))
    h3s = Vector{Union{Parameter,Nothing}}(nothing, length(lines["lines"]))
    h4s = Vector{Union{Parameter,Nothing}}(nothing, length(lines["lines"]))
    ηs = Vector{Union{Parameter,Nothing}}(nothing, length(lines["lines"]))
    tied_amp = Vector{Union{Symbol,Nothing}}(nothing, length(lines["lines"]))
    tied_voff = Vector{Union{Symbol,Nothing}}(nothing, length(lines["lines"]))
    tied_fwhm = Vector{Union{Symbol,Nothing}}(nothing, length(lines["lines"]))
    prof_out = Vector{Union{Symbol,Nothing}}(nothing, length(lines["lines"]))
    sort_order = ones(Int, length(lines["lines"])) .* lines["default_sort_order"]

    # Additional components
    acomp_amps = Matrix{Union{Parameter,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_voffs = Matrix{Union{Parameter,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_fwhms = Matrix{Union{Parameter,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_h3s = Matrix{Union{Parameter,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_h4s = Matrix{Union{Parameter,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_ηs = Matrix{Union{Parameter,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_tied_amp = Matrix{Union{Symbol,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_tied_voff = Matrix{Union{Symbol,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_tied_fwhm = Matrix{Union{Symbol,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_prof_out = Matrix{Union{Symbol,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])

    # Loop through all the lines
    for (i, line) ∈ enumerate(keys(lines["lines"]))

        # Define the initial values of line parameters given the values in the options file (if present)
        fwhm_init = haskey(lines, "fwhm_init") ? lines["fwhm_init"] : 100.0
        voff_init = haskey(lines, "voff_init") ? lines["voff_init"] : 0.0
        h3_init = haskey(lines, "h3_init") ? lines["h3_init"] : 0.0        # gauss-hermite series start fully gaussian,
        h4_init = haskey(lines, "h4_init") ? lines["h4_init"] : 0.0        # with both h3 and h4 moments starting at 0
        η_init = haskey(lines, "eta_init") ? lines["eta_init"] : 0.5       # Voigts start half gaussian, half lorentzian

        names[i] = Symbol(line)
        latex[i] = lines["lines"][line]["latex"]
        cent_vals[i] = lines["lines"][line]["wave"]
        annotate[i] = lines["lines"][line]["annotate"]
        prof_out[i] = isnothing(profiles[line]) ? nothing : Symbol(profiles[line])
        if haskey(lines["lines"][line], "sort_order")
            sort_order[i] = lines["lines"][line]["sort_order"]
        end

        mask = isnothing.(acomp_profiles[line])
        acomp_prof_out[i, mask] .= nothing
        acomp_prof_out[i, .!mask] .= Symbol.(acomp_profiles[line][.!mask])

        @debug """\n
        ################# $line #######################
        # Rest wavelength: $(lines["lines"][line]["wave"]) um #
        """

        # Set the parameter limits for FWHM, voff, h3, h4, and eta based on the values in the options file
        voff_plim = (lines["voff_plim"]...,)
        voff_locked = false
        fwhm_plim = (lines["fwhm_plim"]...,)
        fwhm_locked = false
        h3_plim = h3_locked = h4_plim = h4_locked = η_plim = η_locked = nothing
        if profiles[line] == "GaussHermite"
            h3_plim = (lines["h3_plim"]...,)
            h3_locked = false
            h4_plim = (lines["h4_plim"]...,)
            h4_locked = false
        elseif profiles[line] == "Voigt"
            η_plim = (lines["eta_plim"]...,)
            η_locked = false
        end

        # Determine the initial values for the additional components
        acomp_amp_init = haskey(lines, "acomp_amp_init") ? lines["acomp_amp_init"] : repeat([0.1], lines["n_acomps"])
        acomp_fwhm_init = haskey(lines, "acomp_fwhm_init") ? lines["acomp_fwhm_init"] : repeat([fwhm_init], lines["n_acomps"])
        acomp_voff_init = haskey(lines, "acomp_voff_init") ? lines["acomp_voff_init"] : repeat([voff_init], lines["n_acomps"])
        acomp_h3_init = haskey(lines, "acomp_h3_init") ? lines["acomp_h3_init"] : repeat([h3_init], lines["n_acomps"])
        acomp_h4_init = haskey(lines, "acomp_h4_init") ? lines["acomp_h4_init"] : repeat([h4_init], lines["n_acomps"])
        acomp_η_init = haskey(lines, "acomp_eta_init") ? lines["acomp_eta_init"] : repeat([η_init], lines["n_acomps"])

        # Set the parameter limits for additional component FWHM, voff, h3, h4, and eta based on the values in the options file
        acomp_amp_plims = Vector{Union{Tuple,Nothing}}(nothing, lines["n_acomps"])
        acomp_amp_locked = falses(lines["n_acomps"])
        acomp_voff_plims = Vector{Union{Tuple,Nothing}}(nothing, lines["n_acomps"])
        acomp_voff_locked = falses(lines["n_acomps"])
        acomp_fwhm_plims = Vector{Union{Tuple,Nothing}}(nothing, lines["n_acomps"])
        acomp_fwhm_locked = falses(lines["n_acomps"]) 
        acomp_h3_plims = Vector{Union{Tuple,Nothing}}(nothing, lines["n_acomps"]) 
        acomp_h3_locked = falses(lines["n_acomps"])
        acomp_h4_plims = Vector{Union{Tuple,Nothing}}(nothing, lines["n_acomps"])
        acomp_h4_locked = falses(lines["n_acomps"])
        acomp_η_plims = Vector{Union{Tuple,Nothing}}(nothing, lines["n_acomps"])
        acomp_η_locked = falses(lines["n_acomps"])
        for j ∈ 1:lines["n_acomps"]
            if !isnothing(acomp_profiles[line][j])
                acomp_amp_plims[j] = (lines["acomp_amp_plim"][j]...,)
                acomp_voff_plims[j] = (lines["acomp_voff_plim"][j]...,)
                acomp_fwhm_plims[j] = (lines["acomp_fwhm_plim"][j]...,)
                if acomp_profiles[line][j] == "GaussHermite"
                    acomp_h3_plims[j] = (lines["h3_plim"]...,)
                    acomp_h4_plims[j] = (lines["h4_plim"]...,)
                elseif acomp_profiles[line][j] == "Voigt"
                    acomp_η_plims[j] = (0.0, 1.0)
                end
            end
        end

        # Check if there are any specific override values present in the options file,
        # and if so, use them
        if haskey(lines, "parameters")
            if haskey(lines["parameters"], line)
                # Dictionary mapping parameter strings to their limits/locked values
                paramvars = Dict("voff" => [voff_plim, voff_locked, voff_init],
                                 "fwhm" => [fwhm_plim, fwhm_locked, fwhm_init],
                                 "h3" => [h3_plim, h3_locked, h3_init],
                                 "h4" => [h4_plim, h4_locked, h4_init],
                                 "eta" => [η_plim, η_locked, η_init])
                # Iterate over parameters and check if they should be overwritten by the contents of the lines file
                for param_str ∈ ["voff", "fwhm", "h3", "h4", "eta"]
                    if haskey(lines["parameters"][line], "$(param_str)_plim")
                        @debug "Overriding $param_str limits for $line"
                        paramvars[param_str][1] = (lines["parameters"][line]["$(param_str)_plim"]...,)
                    end
                    if haskey(lines["parameters"][line], "$(param_str)_locked")
                        @debug "Overriding $param_str locked value for $line"
                        paramvars[param_str][2] = lines["parameters"][line]["$(param_str)_locked"]
                    end
                    if haskey(lines["parameters"][line], "$(param_str)_init")
                        @debug "Overriding $param_str initial value for $line"
                        paramvars[param_str][3] = lines["parameters"][line]["$(param_str)_init"]
                    end
                end
                # Unpack paramvars back into the appropriate variables
                voff_plim, voff_locked, voff_init = paramvars["voff"]
                fwhm_plim, fwhm_locked, fwhm_init = paramvars["fwhm"]
                h3_plim, h3_locked, h3_init = paramvars["h3"]
                h4_plim, h4_locked, h4_init = paramvars["h4"]
                η_plim, η_locked, η_init = paramvars["eta"]

                # Repeat for acomp parameters
                acomp_paramvars = Dict("acomp_amp" => [acomp_amp_plims, acomp_amp_locked, acomp_amp_init],
                                       "acomp_voff" => [acomp_voff_plims, acomp_voff_locked, acomp_voff_init],
                                       "acomp_fwhm" => [acomp_fwhm_plims, acomp_fwhm_locked, acomp_fwhm_init],
                                       "acomp_h3" => [acomp_h3_plims, acomp_h3_locked, acomp_h3_init],
                                       "acomp_h4" => [acomp_h4_plims, acomp_h4_locked, acomp_h4_init],
                                       "acomp_eta" => [acomp_η_plims, acomp_η_locked, acomp_η_init])
                for param_str ∈ ["acomp_amp", "acomp_voff", "acomp_fwhm", "acomp_h3", "acomp_h4", "acomp_eta"]
                    if haskey(lines["parameters"][line], "$(param_str)_plim")
                        @debug "Overriding $param_str limits for $line"
                        for j ∈ 1:lines["n_acomps"]
                            acomp_paramvars[param_str][1][j] = (lines["parameters"][line]["$(param_str)_plim"][j]...,)
                        end
                    end
                    if haskey(lines["parameters"][line], "$(param_str)_locked")
                        @debug "Overriding $param_str locked value for $line"
                        for j ∈ 1:lines["n_acomps"]
                            acomp_paramvars[param_str][2][j] = lines["parameters"][line]["$(param_str)_locked"][j]
                        end
                    end
                    if haskey(lines["parameters"][line], "$(param_str)_init")
                        @debug "Overriding $param_str initial value for $line"
                        for j ∈ 1:lines["n_acomps"]
                            acomp_paramvars[param_str][3][j] = lines["parameters"][line]["$(param_str)_init"][j]
                        end
                    end
                end
                # Dont need to unpack acomp_paramvars since they are vectors (modified in-place) and not individual variables

            end
        end

        # Check if the kinematics should be tied to other lines based on the kinematic groups
        tied_amp[i] = tied_voff[i] = tied_fwhm[i] = nothing
        for group ∈ kinematic_groups
            for groupmember ∈ lines["kinematic_group_" * group]
                #= Loop through the items in the "kinematic_group_*" list and see if the line name matches any of them.
                 It needn't be a perfect match, the line name just has to contain the value in the kinematic group list.
                 i.e. if you want to automatically tie all FeII lines together, instead of manually listing out each one,
                 you can just include an item "FeII" and it will automatically catch all the FeII lines
                =#
                if occursin(groupmember, line)

                    # Check if amp should  be tied
                    tie_amp_group = false
                    if haskey(lines, "tie_amp_" * group)
                        tie_amp_group = true
                    end
                    # Check if voff should be tied
                    tie_voff_group = true
                    if haskey(lines, "tie_voff_" * group)
                        tie_voff_group = lines["tie_voff_" * group]
                    end
                    # Check if fwhm should be tied
                    tie_fwhm_group = true
                    if haskey(lines, "tie_fwhm_" * group)
                        tie_fwhm_group = lines["tie_fwhm_" * group]
                    end

                    if tie_amp_group
                        @assert isnothing(tied_amp[i]) "Line $(line[i]) is already part of the kinematic group $(tied_amp[i]), but it also passed filtering criteria" *
                            "to be included in the group $group. Make sure your filters are not too lenient!"
                        @debug "Tying amplitudes for $line to the group: $group"
                        # Use the group label
                        tied_amp[i] = Symbol(group)
                    end
                    if tie_voff_group
                        # Make sure line is not already a member of another group
                        @assert isnothing(tied_voff[i]) "Line $(line[i]) is already part of the kinematic group $(tied_voff[i]), but it also passed filtering criteria" * 
                            "to be included in the group $group. Make sure your filters are not too lenient!"
                        @debug "Tying kinematics for $line to the group: $group"
                        # Use the group label (which can be anything you want) to categorize what lines are tied together
                        tied_voff[i] = Symbol(group)
                        # If the wavelength solution is bad, allow the kinematics to still be flexible based on its accuracy
                        if lines["flexible_wavesol"]
                            δv = lines["wavesol_unc"]
                            voff_plim = (-δv, δv)
                            @debug "Using flexible tied voff with lenience of +/-$δv km/s"
                        end
                    end
                    if tie_fwhm_group
                        # Make sure line is not already a member of another group
                        @assert isnothing(tied_fwhm[i]) "Line $(line[i]) is already part of the kinematic group $(tied_fwhm[i]), but it also passed filtering criteria" * 
                        "to be included in the group $group. Make sure your filters are not too lenient!"
                        @debug "Tying kinematics for $line to the group: $group"
                        # Use the group label (which can be anything you want) to categorize what lines are tied together
                        tied_fwhm[i] = Symbol(group)
                    end

                    break
                end
            end
        end

        # Repeat for the acomps
        acomp_tied_amp[i, :] .= nothing
        acomp_tied_voff[i, :] .= nothing
        acomp_tied_fwhm[i, :] .= nothing
        for j ∈ 1:lines["n_acomps"]
            for group ∈ acomp_kinematic_groups[j]
                for groupmember ∈ lines["acomp_$(j)_kinematic_group_" * group]
                    if occursin(groupmember, line)

                        # Check if amp should be tied
                        tie_acomp_amp_group = false
                        if haskey(lines, "tie_acomp_$(j)_amp_" * group)
                            tie_acomp_amp_group = true
                        end

                        # Check if voff should be tied
                        tie_acomp_voff_group = true
                        if haskey(lines, "tie_acomp_$(j)_voff_" * group)
                            tie_acomp_voff_group = lines["tie_acomp_$(j)_voff_" * group]
                        end
                        # Check if fwhm should be tied
                        tie_acomp_fwhm_group = true
                        if haskey(lines, "tie_acomp_$(j)_fwhm_" * group)
                            tie_acomp_fwhm_group = lines["tie_acomp_$(j)_fwhm_" * group]
                        end

                        if !isnothing(acomp_profiles[line][j]) && tie_acomp_amp_group
                            # Make sure line is not already a member of another group
                            @assert isnothing(acomp_tied_amp[i, j]) "Line $(line[i]) acomp $(j) is already part of the kinematic group $(acomp_tied_amp[i, j]), " *
                                "but it also passed filtering criteria to be included in the group $group. Make sure your filters are not too lenient!"
                            @debug "Tying amplitudes for $line acomp $(j) to the group: $group"

                            # only set acomp_tied if the line actually *has* an acomp
                            acomp_tied_amp[i,j] = Symbol(group)
                        end

                        if !isnothing(acomp_profiles[line][j]) && tie_acomp_voff_group
                            # Make sure line is not already a member of another group
                            @assert isnothing(acomp_tied_voff[i, j]) "Line $(line[i]) acomp $(j) is already part of the kinematic group $(acomp_tied_voff[i, j]), " *
                                "but it also passed filtering criteria to be included in the group $group. Make sure your filters are not too lenient!"
                            @debug "Tying kinematics for $line acomp $(j) to the group: $group"

                            # Only set acomp_tied if the line actually *has* an acomp
                            acomp_tied_voff[i,j] = Symbol(group)
                        end

                        if !isnothing(acomp_profiles[line][j]) && tie_acomp_fwhm_group
                            # Make sure line is not already a member of another group
                            @assert isnothing(acomp_tied_fwhm[i, j]) "Line $(line[i]) acomp $(j) is already part of the kinematic group $(acomp_tied_fwhm[i, j]), " *
                                "but it also passed filtering criteria to be included in the group $group. Make sure your filters are not too lenient!"
                            @debug "Tying kinematics for $line acomp $(j) to the group: $group"
                        
                            # Only set acomp_tied if the line actually *has* an acomp
                            acomp_tied_fwhm[i,j] = Symbol(group)
                        end

                        break
                    end
                end
            end
        end

        @debug "Profile: $(profiles[line])"

        # Create parameter objects using the parameter limits
        voffs[i] = Parameter(voff_init, voff_locked, voff_plim)
        @debug "Voff $(voffs[i])"
        fwhms[i] = Parameter(fwhm_init, fwhm_locked, fwhm_plim)
        @debug "FWHM $(fwhms[i])"
        if profiles[line] == "GaussHermite"
            h3s[i] = Parameter(h3_init, h3_locked, h3_plim)
            @debug "h3 $(h3s[i])"
            h4s[i] = Parameter(h4_init, h4_locked, h4_plim)
            @debug "h4 $(h4s[i])"
        elseif profiles[line] == "Voigt" 
            ηs[i] = Parameter(η_init, η_locked, η_plim)
            @debug "eta $(ηs[i])"
        end

        # Do the same for the additional component parameters, but only if the line has an additional component
        for j ∈ 1:lines["n_acomps"]
            if !isnothing(acomp_profiles[line][j])
                @debug "acomp profile: $(acomp_profiles[line][j])"
                acomp_amps[i,j] = Parameter(acomp_amp_init[j], acomp_amp_locked[j], acomp_amp_plims[j])
                if !(acomp_voff_plims[j][1] ≤ acomp_voff_init[j] ≤ acomp_voff_plims[j][2])
                    if abs(acomp_voff_plims[j][1]) < abs(acomp_voff_plims[j][2])
                        acomp_voff_init[j] = acomp_voff_plims[j][1]
                    else
                        acomp_voff_init[j] = acomp_voff_plims[j][2]
                    end
                end
                acomp_voffs[i,j] = Parameter(acomp_voff_init[j], acomp_voff_locked[j], acomp_voff_plims[j])
                @debug "Voff $(acomp_voffs[i,j])"
                if !(acomp_fwhm_plims[j][1] ≤ acomp_fwhm_init[j] ≤ acomp_fwhm_plims[j][2])
                    acomp_fwhm_init[j] = acomp_fwhm_plims[j][1]
                end
                acomp_fwhms[i,j] = Parameter(acomp_fwhm_init[j], acomp_fwhm_locked[j], acomp_fwhm_plims[j])
                @debug "FWHM $(acomp_fwhms[i,j])"
                if acomp_profiles[line][j] == "GaussHermite"
                    acomp_h3s[i,j] = Parameter(acomp_h3_init[j], acomp_h3_locked[j], acomp_h3_plims[j])
                    @debug "h3 $(acomp_h3s[i,j])"
                    acomp_h4s[i,j] = Parameter(acomp_h4_init[j], acomp_h4_locked[j], acomp_h4_plims[j])
                    @debug "h4 $(acomp_h4s[i,j])"
                elseif acomp_profiles[line][j] == "Voigt" 
                    acomp_ηs[i,j] = Parameter(acomp_η_init[j], acomp_η_locked[j], acomp_η_plims[j])
                    @debug "eta $(acomp_ηs[i,j])"
                end
            end
        end

    end

    # sort by cent_vals
    ss = sortperm(cent_vals)

    # Check for any combined lines
    combined = []
    if haskey(lines, "combined_maps")
        combined = lines["combined_maps"]
        combined = [[Symbol(ln) for ln in c] for c in combined]
    end

    rel_amp = rel_voff = rel_fwhm = false
    if haskey(lines, "rel_amp")
        rel_amp = lines["rel_amp"]
    end
    if haskey(lines, "rel_voff")
        rel_voff = lines["rel_voff"]
    end
    if haskey(lines, "rel_fwhm")
        rel_fwhm = lines["rel_fwhm"]
    end

    # create vectorized object for all the line data
    lines_out = TransitionLines(names[ss], latex[ss], annotate[ss], cent_vals[ss], sort_order[ss], hcat(prof_out[ss], acomp_prof_out[ss, :]), 
        hcat(tied_amp[ss], acomp_tied_amp[ss, :]), hcat(tied_voff[ss], acomp_tied_voff[ss, :]), hcat(tied_fwhm[ss], acomp_tied_fwhm[ss, :]), 
        acomp_amps[ss, :], hcat(voffs[ss], acomp_voffs[ss, :]), hcat(fwhms[ss], acomp_fwhms[ss, :]), hcat(h3s[ss], acomp_h3s[ss, :]), 
        hcat(h4s[ss], acomp_h4s[ss, :]), hcat(ηs[ss], acomp_ηs[ss, :]), combined, rel_amp, rel_voff, rel_fwhm)

    @debug "#######################################################"

    # Create a dictionary containing all of the unique `tie` keys, and the tied parameters 
    # corresponding to that tied key
    kin_tied_key_amp = [unique(lines_out.tied_amp[:, j]) for j ∈ 1:size(lines_out.tied_amp, 2)]
    kin_tied_key_amp = [kin_tied_key_amp[i][.!isnothing.(kin_tied_key_amp[i])] for i in eachindex(kin_tied_key_amp)]
    kin_tied_key_voff = [unique(lines_out.tied_voff[:, j]) for j ∈ 1:size(lines_out.tied_voff, 2)]
    kin_tied_key_voff = [kin_tied_key_voff[i][.!isnothing.(kin_tied_key_voff[i])] for i in eachindex(kin_tied_key_voff)]
    kin_tied_key_fwhm = [unique(lines_out.tied_fwhm[:, j]) for j ∈ 1:size(lines_out.tied_fwhm, 2)]
    kin_tied_key_fwhm = [kin_tied_key_fwhm[i][.!isnothing.(kin_tied_key_fwhm[i])] for i in eachindex(kin_tied_key_fwhm)]
    @debug "kin_tied_key_amp: $kin_tied_key_amp"
    @debug "kin_tied_key_voff: $kin_tied_key_voff"
    @debug "kin_tied_key_fwhm: $kin_tied_key_fwhm"
    
    amp_tied = [Vector{Dict{Symbol, Float64}}(undef, length(kin_tied_key_amp[j])) for j ∈ 1:size(lines_out.tied_amp, 2)]
    voff_tied = [Vector{Parameter}(undef, length(kin_tied_key_voff[j])) for j ∈ 1:size(lines_out.tied_voff, 2)]
    fwhm_tied = [Vector{Parameter}(undef, length(kin_tied_key_fwhm[j])) for j ∈ 1:size(lines_out.tied_fwhm, 2)]
    msg = ""
    # Iterate and create the tied amplitude parameters
    for j ∈ 1:size(lines_out.tied_amp, 2)
        for (i, kin_tie) ∈ enumerate(kin_tied_key_amp[j])
            a_ratio = isone(j) ? lines["tie_amp_$kin_tie"] : lines["tie_acomp_$(j-1)_amp_$kin_tie"]
            lines_in_group = lines_out.names[lines_out.tied_amp[:, j] .== kin_tie]
            amp_tied[j][i] = Dict(ln => ai for (ln, ai) in zip(lines_in_group, a_ratio))
            msg *= "\namp_tied_$(kin_tie)_$(j) $(amp_tied[j][i])"
        end
    end
    # Iterate and create the tied voff parameters
    for j ∈ 1:size(lines_out.tied_voff, 2)
        for (i, kin_tie) ∈ enumerate(kin_tied_key_voff[j])
            v_plim = isone(j) ? (lines["voff_plim"]...,) : (lines["acomp_voff_plim"][j-1]...,)
            v_locked = false
            v_init = isone(j) && haskey(lines, "voff_init") ? lines["voff_init"] : 0.0
            # Check if there is an overwrite option in the lines file
            if haskey(lines, "parameters")
                if haskey(lines["parameters"], string(kin_tie))
                    param_str = isone(j) ? "voff" : "acomp_voff"
                    if haskey(lines["parameters"][string(kin_tie)], "$(param_str)_plim")
                        v_plim = isone(j) ? (lines["parameters"][string(kin_tie)]["$(param_str)_plim"]...,) :
                                            (lines["parameters"][string(kin_tie)]["$(param_str)_plim"][j-1]...,)
                    end
                    if haskey(lines["parameters"][string(kin_tie)], "$(param_str)_locked")
                        v_locked = isone(j) ? lines["parameters"][string(kin_tie)]["$(param_str)_locked"] :
                                              lines["parameters"][string(kin_tie)]["$(param_str)_locked"][j-1]
                    end
                    if haskey(lines["parameters"][string(kin_tie)], "$(param_str)_init")
                        v_init = isone(j) ? lines["parameters"][string(kin_tie)]["$(param_str)_init"] :
                                            lines["parameters"][string(kin_tie)]["$(param_str)_init"][j-1]
                    end
                end
            end
            if !(v_plim[1] ≤ v_init ≤ v_plim[2])
                if abs(v_plim[1]) < abs(v_plim[2])
                    v_init = v_plim[1]
                else
                    v_init = v_plim[2]
                end
            end
            voff_tied[j][i] = Parameter(v_init, v_locked, v_plim)
            msg *= "\nvoff_tied_$(kin_tie)_$(j) $(voff_tied[j][i])"
        end
    end
    # Iterate and create the tied fwhm parameters
    for j ∈ 1:size(lines_out.tied_fwhm, 2)
        for (i, kin_tie) ∈ enumerate(kin_tied_key_fwhm[j])
            f_plim = isone(j) ? (lines["fwhm_plim"]...,) : (lines["acomp_fwhm_plim"][j-1]...,)
            f_locked = false
            f_init = isone(j) && haskey(lines, "fwhm_init") ? lines["fwhm_init"] : 100.0
            # Check if there is an overwrite option in the lines file
            if haskey(lines, "parameters")
                if haskey(lines["parameters"], string(kin_tie))
                    param_str = isone(j) ? "fwhm" : "acomp_fwhm"
                    if haskey(lines["parameters"][string(kin_tie)], "$(param_str)_plim")
                        f_plim = isone(j) ? (lines["parameters"][string(kin_tie)]["$(param_str)_plim"]...,) :
                                            (lines["parameters"][string(kin_tie)]["$(param_str)_plim"][j-1]...,)
                    end
                    if haskey(lines["parameters"][string(kin_tie)], "$(param_str)_locked")
                        f_locked = isone(j) ? lines["parameters"][string(kin_tie)]["$(param_str)_locked"] :
                                              lines["parameters"][string(kin_tie)]["$(param_str)_locked"][j-1]
                    end
                    if haskey(lines["parameters"][string(kin_tie)], "$(param_str)_init")
                        f_init = isone(j) ?   lines["parameters"][string(kin_tie)]["$(param_str)_init"] :
                                              lines["parameters"][string(kin_tie)]["$(param_str)_init"][j-1]
                    end
                end
            end
            if !(f_plim[1] ≤ f_init ≤ f_plim[2])
                f_init = f_plim[1]
            end
            fwhm_tied[j][i] = Parameter(f_init, f_locked, f_plim)
            msg *= "\nfwhm_tied_$(kin_tie)_$(j) $(fwhm_tied[j][i])"
        end
    end

    @debug msg
    tied_kinematics = TiedKinematics(kin_tied_key_amp, amp_tied, kin_tied_key_voff, voff_tied, kin_tied_key_fwhm, fwhm_tied)

    # If tie_voigt_mixing is set, all Voigt profiles have the same tied mixing parameter eta
    if lines["tie_voigt_mixing"]
        η_init = haskey(lines, "eta_init") ? lines["eta_init"] : 0.5       # Voigts start half gaussian, half lorentzian
        η_plim = (lines["eta_plim"]...,)
        η_locked = haskey(lines, "eta_locked") ? lines["eta_locked"] : false
        voigt_mix_tied = Parameter(η_init, η_locked, η_plim)
        @debug "voigt_mix_tied $voigt_mix_tied (tied)"
    else
        voigt_mix_tied = nothing
        @debug "voigt_mix_tied is $voigt_mix_tied (untied)"
    end

    lines_out, tied_kinematics, lines["flexible_wavesol"], lines["tie_voigt_mixing"], voigt_mix_tied
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
        ignorerepeated=true, header=["wave", "a_galcen", "a_local"])
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
    @assert (λ[2]/λ[1]) ≈ (λ[end]/λ[end-1]) "Input spectrum must be logarithmically binned to fit optical data!"

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
    @assert (λ[2]/λ[1]) ≈ (λ[end]/λ[end-1]) "Input spectrum must be logarithmically binned to fit optical data!"

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

