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
function parse_resolving(channel::String)::Function

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

    # The line-spread function in km/s
    lsf = wi -> C_KMS / interp_R(wi)
    
    lsf
end


"""
    parse_options()

Read in the options.toml configuration file, checking that it is formatted correctly,
and convert it into a julia dictionary.  This deals with general/top-level code configurations.
"""
function parse_options()::Dict

    @debug """\n
    Parsing options file
    #######################################################
    """

    # Read in the options file
    options = TOML.parsefile(joinpath(@__DIR__, "..", "options", "options.toml"))
    keylist1 = ["extinction_curve", "extinction_screen", "fit_sil_emission", "fit_all_samin", "subtract_cubic", 
                "overwrite", "track_memory", "track_convergence", "save_full_model", "make_movies", "cosmology"]
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
    dust_out = Dict()
    keylist1 = ["stellar_continuum_temp", "dust_continuum_temps", "dust_features", "extinction", "hot_dust"]
    keylist2 = ["wave", "fwhm"]
    keylist3 = ["tau_9_7", "tau_ice", "tau_ch", "beta"]
    keylist4 = ["temp", "frac", "tau_warm", "tau_cold"]
    keylist5 = ["val", "plim", "locked"]

    # Loop through all of the required keys that should be in the file and confirm that they are there
    for key ∈ keylist1
        @assert haskey(dust, key) "Missing option $key in dust file!"
    end
    for key ∈ keylist5
        @assert haskey(dust["stellar_continuum_temp"], key) "Missing option $key in stellar continuum temp options!"
        for dc ∈ dust["dust_continuum_temps"]
            @assert haskey(dc, key) "Missing option $key in dust continuum temp options!"
        end
        for df_key ∈ keys(dust["dust_features"])
            for df_key2 ∈ keylist2
                @assert haskey(dust["dust_features"][df_key], df_key2) "Missing option $df_key2 in dust feature $df_key options!"
                @assert haskey(dust["dust_features"][df_key][df_key2], key) "Missing option $key in dust features $df_key, $df_key2 options!"
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
    T_dc = [from_dict(dust["dust_continuum_temps"][i]) for i ∈ eachindex(dust["dust_continuum_temps"])]
    msg = "Dust continuum:"
    for dci ∈ T_dc
        msg *= "\nTemp $dci"
    end
    @debug msg
    
    # Dust feature central wavelengths and FWHMs
    cent_vals = zeros(length(dust["dust_features"]))
    name = Vector{String}(undef, length(dust["dust_features"]))
    mean = Vector{Parameter}(undef, length(dust["dust_features"]))
    fwhm = Vector{Parameter}(undef, length(dust["dust_features"]))

    msg = "Dust features:"
    for (i, df) ∈ enumerate(keys(dust["dust_features"]))
        name[i] = df
        mean[i] = from_dict_wave(dust["dust_features"][df]["wave"])
        msg *= "\nWave $(mean[i])"
        fwhm[i] = from_dict_fwhm(dust["dust_features"][df]["fwhm"])
        msg *= "\nFWHM $(fwhm[i])"
        cent_vals[i] = mean[i].value
    end
    @debug msg

    # Sort by cent_vals
    ss = sortperm(cent_vals)
    dust_features = DustFeatures(name[ss], [:Drude for _ in 1:length(name)], mean[ss], fwhm[ss])

    # Extinction parameters, optical depth and mixing ratio
    msg = "Extinction:"
    # Write tau_9_7 value based on the provided guess
    # dust["extinction"]["tau_9_7"]["val"] = τ_guess
    τ_97 = from_dict(dust["extinction"]["tau_9_7"])
    msg *= "\nTau_sil $τ_97"
    τ_ice = from_dict(dust["extinction"]["tau_ice"])
    msg *= "\nTau_ice $τ_ice"
    τ_ch = from_dict(dust["extinction"]["tau_ch"])
    msg *= "\nTau_CH $τ_ch"
    β = from_dict(dust["extinction"]["beta"])
    msg *= "\nBeta $β"
    @debug msg

    # Hot dust parameters, temperature, covering fraction, warm tau, and cold tau
    msg = "Hot Dust:"
    # Write warm_tau and col_tau values based on the provided guess
    # dust["hot_dust"]["tau_warm"]["val"] = τ_guess
    # dust["hot_dust"]["tau_cold"]["val"] = τ_guess
    T_hot = from_dict(dust["hot_dust"]["temp"])
    msg *= "\nTemp $T_hot"
    Cf = from_dict(dust["hot_dust"]["frac"])
    msg *= "\nFrac $Cf"
    τ_warm = from_dict(dust["hot_dust"]["tau_warm"])
    msg *= "\nTau_Warm $τ_warm"
    τ_cold = from_dict(dust["hot_dust"]["tau_cold"])
    msg *= "\nTau_Cold $τ_cold"
    @debug msg

    # Create continuum object
    continuum = Continuum(T_s, T_dc, τ_97, τ_ice, τ_ch, β, T_hot, Cf, τ_warm, τ_cold)

    continuum, dust_features
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

    keylist1 = ["tie_voigt_mixing", "voff_plim", "fwhm_plim", "h3_plim", "h4_plim", "acomp_voff_plim", 
        "acomp_fwhm_plim", "flexible_wavesol", "wavesol_unc", "lines", "profiles", "acomps", "n_acomps"]

    # Loop through all the required keys that should be in the file and confirm that they are there
    for key ∈ keylist1
        @assert haskey(lines, key) "$key not found in line options!"
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

    # Define the initial values of line parameters given the values in the options file (if present)
    fwhm_init = haskey(lines, "fwhm_init") ? lines["fwhm_init"] : 100.0
    voff_init = haskey(lines, "voff_init") ? lines["voff_init"] : 0.0
    h3_init = haskey(lines, "h3_init") ? lines["h3_init"] : 0.0        # gauss-hermite series start fully gaussian,
    h4_init = haskey(lines, "h4_init") ? lines["h4_init"] : 0.0        # with both h3 and h4 moments starting at 0
    η_init = haskey(lines, "eta_init") ? lines["eta_init"] : 0.5       # Voigts start half gaussian, half lorentzian

    # Create the kinematic groups
    #   ---> kinematic groups apply to all additional components of lines as well as the main component
    #        (i.e. the additional components in a kinematic group are tied to each other, but not to the main components
    #         or to the other additional components)
    kinematic_groups = []
    for key ∈ keys(lines)
        if occursin("kinematic_group_", key)
            append!(kinematic_groups, [replace(key, "kinematic_group_" => "")])
        end
    end

    # Initialize undefined vectors for each TransitionLine attribute that will be filled in
    names = Vector{Symbol}(undef, length(lines["lines"]))
    cent_vals = zeros(length(lines["lines"]))
    voffs = Vector{Parameter}(undef, length(lines["lines"]))
    fwhms = Vector{Parameter}(undef, length(lines["lines"]))
    h3s = Vector{Union{Parameter,Nothing}}(nothing, length(lines["lines"]))
    h4s = Vector{Union{Parameter,Nothing}}(nothing, length(lines["lines"]))
    ηs = Vector{Union{Parameter,Nothing}}(nothing, length(lines["lines"]))
    tied_voff = Vector{Union{Symbol,Nothing}}(nothing, length(lines["lines"]))
    tied_fwhm = Vector{Union{Symbol,Nothing}}(nothing, length(lines["lines"]))
    prof_out = Vector{Union{Symbol,Nothing}}(nothing, length(lines["lines"]))

    # Additional components
    acomp_voffs = Matrix{Union{Parameter,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_fwhms = Matrix{Union{Parameter,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_h3s = Matrix{Union{Parameter,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_h4s = Matrix{Union{Parameter,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_ηs = Matrix{Union{Parameter,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_tied_voff = Matrix{Union{Symbol,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_tied_fwhm = Matrix{Union{Symbol,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])
    acomp_prof_out = Matrix{Union{Symbol,Nothing}}(nothing, length(lines["lines"]), lines["n_acomps"])

    # Loop through all the lines
    for (i, line) ∈ enumerate(keys(lines["lines"]))

        names[i] = Symbol(line)
        cent_vals[i] = lines["lines"][line]
        prof_out[i] = isnothing(profiles[line]) ? nothing : Symbol(profiles[line])

        mask = isnothing.(acomp_profiles[line])
        acomp_prof_out[i, mask] .= nothing
        acomp_prof_out[i, .!mask] .= Symbol.(acomp_profiles[line][.!mask])

        @debug """\n
        ################# $line #######################
        # Rest wavelength: $(lines["lines"][line]) um #
        """

        # Set the priors for FWHM, voff, h3, h4, and eta based on the values in the options file
        voff_prior = Uniform(lines["voff_plim"]...)
        voff_locked = false
        fwhm_prior = Uniform(lines["fwhm_plim"]...)
        fwhm_locked = false
        h3_prior = h3_locked = h4_prior = h4_locked = η_prior = η_locked = nothing
        if profiles[line] == "GaussHermite"
            h3_prior = truncated(Normal(0.0, 0.1), lines["h3_plim"]... #= normal profile, but truncated with hard limits =#)
            h3_locked = false
            h4_prior = truncated(Normal(0.0, 0.1), lines["h4_plim"]... #= normal profile, but truncated with hard limits =#)
            h4_locked = false
        elseif profiles[line] == "Voigt"
            η_prior = Uniform(0.0, 1.0)
            η_locked = false
        end

        # Set the priors for additional component FWHM, voff, h3, h4, and eta based on the values in the options file
        acomp_voff_priors = Vector{Union{Distribution,Nothing}}(nothing, lines["n_acomps"])
        acomp_voff_locked = falses(lines["n_acomps"])
        acomp_fwhm_priors = Vector{Union{Distribution,Nothing}}(nothing, lines["n_acomps"])
        acomp_fwhm_locked = falses(lines["n_acomps"]) 
        acomp_h3_priors = Vector{Union{Distribution,Nothing}}(nothing, lines["n_acomps"]) 
        acomp_h3_locked = falses(lines["n_acomps"])
        acomp_h4_priors = Vector{Union{Distribution,Nothing}}(nothing, lines["n_acomps"])
        acomp_h4_locked = falses(lines["n_acomps"])
        acomp_η_priors = Vector{Union{Distribution,Nothing}}(nothing, lines["n_acomps"])
        acomp_η_locked = falses(lines["n_acomps"])
        for j ∈ 1:lines["n_acomps"]
            if !isnothing(acomp_profiles[line][j])
                acomp_voff_priors[j] = Uniform(lines["acomp_voff_plim"][j]...)
                acomp_fwhm_priors[j] = Uniform(lines["acomp_fwhm_plim"][j]...)
                if acomp_profiles[line][j] == "GaussHermite"
                    acomp_h3_priors[j] = truncated(Normal(0.0, 0.1), lines["h3_plim"]...)
                    acomp_h4_priors[j] = truncated(Normal(0.0, 0.1), lines["h4_plim"]...)
                elseif acomp_profiles[line][j] == "Voigt"
                    acomp_η_priors[j] = Uniform(0.0, 1.0)
                end
            end
        end

        # Check if there are any specific override values present in the options file,
        # and if so, use them
        if haskey(lines, "priors")
            if haskey(lines["priors"], line)
                # Dictionary mapping parameter strings to their priors/locked values
                paramvars = Dict("voff" => [voff_prior, voff_locked],
                                 "fwhm" => [fwhm_prior, fwhm_locked],
                                 "h3" => [h3_prior, h3_locked],
                                 "h4" => [h4_prior, h4_locked],
                                 "eta" => [η_prior, η_locked])
                # Iterate over parameters and check if they should be overwritten by the contents of the lines file
                for param_str ∈ ["voff", "fwhm", "h3", "h4", "eta"]
                    if haskey(lines["priors"][line], "$(param_str)_plim")
                        @debug "Overriding $param_str limits for $line"
                        paramvars[param_str][1] = Uniform(lines["priors"][line]["$(param_str)_plim"]...)
                    end
                    if haskey(lines["priors"][line], "$(param_str)_pstr")
                        @debug "Overriding $param_str prior for $line"
                        paramvars[param_str][1] = eval(Meta.parse(lines["priors"][line]["$(param_str)_pstr"]))
                    end
                    if haskey(lines["priors"][line], "$(param_str)_locked")
                        @debug "Overriding $param_str locked value for $line"
                        paramvars[param_str][2] = lines["priors"][line]["$(param_str)_locked"]
                    end
                end
                # Repeat for acomp parameters
                acomp_paramvars = Dict("acomp_voff" => [acomp_voff_priors, acomp_voff_locked],
                                       "acomp_fwhm" => [acomp_fwhm_priors, acomp_fwhm_locked],
                                       "acomp_h3" => [acomp_h3_priors, acomp_h3_locked],
                                       "acomp_h4" => [acomp_h4_priors, acomp_h4_locked],
                                       "acomp_eta" => [acomp_η_priors, acomp_η_locked])
                for param_str ∈ ["acomp_voff", "acomp_fwhm", "acomp_h3", "acomp_h4", "acomp_eta"]
                    if haskey(lines["priors"][line], "$(param_str)_plim")
                        @debug "Overriding $param_str limits for $line"
                        for j ∈ 1:lines["n_acomps"]
                            acomp_paramvars[param_str][1][j] = Uniform(lines["priors"][line]["$(param_str)_plim"][j]...)
                        end
                    end
                    if haskey(lines["priors"][line], "$(param_str)_pstr")
                        @debug "Overriding $param_str prior for $line"
                        for j ∈ 1:lines["n_acomps"]
                            acomp_paramvars[param_str][1][j] = eval(Meta.parse(lines["priors"][line]["$(param_str)_pstr"]))
                        end
                    end
                    if haskey(lines["priors"][line], "$(param_str)_locked")
                        @debug "Overriding $param_str locked value for $line"
                        for j ∈ 1:lines["n_acomps"]
                            acomp_paramvars[param_str][2][j] = lines["priors"][line]["$(param_str)_locked"][j]
                        end
                    end
                end
            end
        end

        # Check if the kinematics should be tied to other lines based on the kinematic groups
        tied_voff[i] = tied_fwhm[i] = nothing
        acomp_tied_voff[i, :] .= nothing
        acomp_tied_fwhm[i, :] .= nothing
        for group ∈ kinematic_groups
            for groupmember ∈ lines["kinematic_group_" * group]
                #= Loop through the items in the "kinematic_group_*" list and see if the line name matches any of them.
                 It needn't be a perfect match, the line name just has to contain the value in the kinematic group list.
                 i.e. if you want to automatically tie all FeII lines together, instead of manually listing out each one,
                 you can just include an item "FeII" and it will automatically catch all the FeII lines
                =#
                if occursin(groupmember, line)

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

                    if tie_voff_group
                        # Make sure line is not already a member of another group
                        @assert isnothing(tied_voff[i]) "Line $(line[i]) is already part of the kinematic group $(tied_voff[i]), but it also passed filtering criteria" * 
                            "to be included in the group $group. Make sure your filters are not too lenient!"
                        @debug "Tying kinematics for $line to the group: $group"
                        # Use the group label (which can be anything you want) to categorize what lines are tied together
                        tied_voff[i] = Symbol(group)
                        # Only set acomp_tied if the line actually *has* an acomp
                        for j ∈ 1:lines["n_acomps"]
                            if !isnothing(acomp_profiles[line][j])
                                acomp_tied_voff[i,j] = Symbol(group)
                            end
                        end
                        # If the wavelength solution is bad, allow the kinematics to still be flexible based on its accuracy
                        if lines["flexible_wavesol"]
                            δv = lines["wavesol_unc"]
                            voff_prior = Uniform(-δv, δv)
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
                        # Only set acomp_tied if the line actually *has* an acomp
                        for j ∈ 1:lines["n_acomps"]
                            if !isnothing(acomp_profiles[line][j])
                                acomp_tied_fwhm[i,j] = Symbol(group)
                            end
                        end
                    end

                    break
                end
            end
        end

        @debug "Profile: $(profiles[line])"

        # Create parameter objects using the priors
        voffs[i] = Parameter(voff_init, voff_locked, voff_prior)
        @debug "Voff $(voffs[i])"
        fwhms[i] = Parameter(fwhm_init, fwhm_locked, fwhm_prior)
        @debug "FWHM $(fwhms[i])"
        if profiles[line] == "GaussHermite"
            h3s[i] = Parameter(h3_init, h3_locked, h3_prior)
            @debug "h3 $(h3s[i])"
            h4s[i] = Parameter(h4_init, h4_locked, h4_prior)
            @debug "h4 $(h4s[i])"
        elseif profiles[line] == "Voigt" 
            ηs[i] = Parameter(η_init, η_locked, η_prior)
            @debug "eta $(ηs[i])"
        end
        # Do the same for the additional component parameters, but only if the line has an additional component
        for j ∈ 1:lines["n_acomps"]
            if !isnothing(acomp_profiles[line][j])
                @debug "acomp profile: $(acomp_profiles[line][j])"
                value = 0.
                if !(minimum(acomp_voff_priors[j]) ≤ value ≤ maximum(acomp_voff_priors[j]))
                    if abs(minimum(acomp_voff_priors[j])) < abs(maximum(acomp_voff_priors[j]))
                        value = minimum(acomp_voff_priors[j])
                    else
                        value = maximum(acomp_voff_priors[j])
                    end
                end
                acomp_voffs[i,j] = Parameter(value, acomp_voff_locked[j], acomp_voff_priors[j])
                @debug "Voff $(acomp_voffs[i,j])"
                value = 1.
                if !(minimum(acomp_fwhm_priors[j]) ≤ value ≤ maximum(acomp_fwhm_priors[j]))
                    value = minimum(acomp_fwhm_priors[j])
                end
                acomp_fwhms[i,j] = Parameter(value, acomp_fwhm_locked[j], acomp_fwhm_priors[j])
                @debug "FWHM $(acomp_fwhms[i,j])"
                if acomp_profiles[line][j] == "GaussHermite"
                    acomp_h3s[i,j] = Parameter(h3_init, acomp_h3_locked[j], acomp_h3_priors[j])
                    @debug "h3 $(acomp_h3s[i,j])"
                    acomp_h4s[i,j] = Parameter(h4_init, acomp_h4_locked[j], acomp_h4_priors[j])
                    @debug "h4 $(acomp_h4s[i,j])"
                elseif acomp_profiles[line][j] == "Voigt" 
                    acomp_ηs[i,j] = Parameter(η_init, acomp_η_locked[j], acomp_η_priors[j])
                    @debug "eta $(acomp_ηs[i,j])"
                end
            end
        end

    end

    # sort by cent_vals
    ss = sortperm(cent_vals)

    # create vectorized object for all the line data
    lines_out = TransitionLines(names[ss], cent_vals[ss], hcat(prof_out[ss], acomp_prof_out[ss, :]), 
        hcat(tied_voff[ss], acomp_tied_voff[ss, :]), hcat(tied_fwhm[ss], acomp_tied_fwhm[ss, :]), 
        hcat(voffs[ss], acomp_voffs[ss, :]), hcat(fwhms[ss], acomp_fwhms[ss, :]), hcat(h3s[ss], acomp_h3s[ss, :]), 
        hcat(h4s[ss], acomp_h4s[ss, :]), hcat(ηs[ss], acomp_ηs[ss, :]))

    @debug "#######################################################"

    # Create a dictionary containing all of the unique `tie` keys, and the tied parameters 
    # corresponding to that tied key
    kin_tied_key_voff = unique(lines_out.tied_voff)
    kin_tied_key_voff = kin_tied_key_voff[.!isnothing.(kin_tied_key_voff)]
    @debug "kin_tied_key_voff: $kin_tied_key_voff"
    kin_tied_key_fwhm = unique(lines_out.tied_fwhm)
    kin_tied_key_fwhm = kin_tied_key_fwhm[.!isnothing.(kin_tied_key_fwhm)]
    @debug "kin_tied_key_fwhm: $kin_tied_key_fwhm"

    kin_tied_key_voff = [copy(kin_tied_key_voff) for _ ∈ 1:size(lines_out.tied_voff, 2)]
    kin_tied_key_fwhm = [copy(kin_tied_key_fwhm) for _ ∈ 1:size(lines_out.tied_fwhm, 2)]
    voff_tied = [Vector{Parameter}(undef, length(kin_tied_key_voff[1])) for _ ∈ 1:size(lines_out.tied_voff, 2)]
    fwhm_tied = [Vector{Parameter}(undef, length(kin_tied_key_fwhm[1])) for _ ∈ 1:size(lines_out.tied_fwhm, 2)]
    msg = ""
    # Iterate and create the tied voff parameters
    for (i, kin_tie) ∈ enumerate(kin_tied_key_voff[1]), j ∈ 1:size(lines_out.tied_voff, 2)
        v_prior = isone(j) ? Uniform(lines["voff_plim"]...) : Uniform(lines["acomp_voff_plim"][j-1]...)
        v_locked = false
        # Check if there is an overwrite option in the lines file
        if haskey(lines, "priors")
            if haskey(lines["priors"], string(kin_tie))
                param_str = isone(j) ? "voff" : "acomp_voff"
                if haskey(lines["priors"][string(kin_tie)], "$(param_str)_plim")
                    v_prior = isone(j) ? Uniform(lines["priors"][string(kin_tie)]["$(param_str)_plim"]...) :
                                         Uniform(lines["priors"][string(kin_tie)]["$(param_str)_plim"][j]...)
                end
                if haskey(lines["priors"][string(kin_tie)], "$(param_str)_pstr")
                    v_prior = isone(j) ? eval(Meta.parse(lines["priors"][string(kin_tie)]["$(param_str)_pstr"])) :
                                         eval(Meta.parse(lines["priors"][string(kin_tie)]["$(param_str)_pstr"][j]))
                end
                if haskey(lines["priors"][string(kin_tie)], "$(param_str)_locked")
                    v_locked = isone(j) ? lines["priors"][string(kin_tie)]["$(param_str)_locked"] :
                                          lines["priors"][string(kin_tie)]["$(param_str)_locked"][j]
                end
            end
        end
        value = 0.
        if !(minimum(v_prior) ≤ value ≤ maximum(v_prior))
            if abs(minimum(v_prior)) < abs(maximum(v_prior))
                value = minimum(v_prior)
            else
                value = maximum(v_prior)
            end
        end
        voff_tied[j][i] = isone(j) ? Parameter(voff_init, v_locked, v_prior) : Parameter(value, v_locked, v_prior)
        msg *= "\nvoff_tied_$(kin_tie)_$(j) $(voff_tied[j][i])"
    end
    # Iterate and create the tied fwhm parameters
    for (i, kin_tie) ∈ enumerate(kin_tied_key_fwhm[1]), j ∈ 1:size(lines_out.tied_fwhm, 2)
        f_prior = isone(j) ? Uniform(lines["fwhm_plim"]...) : Uniform(lines["acomp_fwhm_plim"][j-1]...)
        f_locked = false
        # Check if there is an overwrite option in the lines file
        if haskey(lines, "priors")
            if haskey(lines["priors"], string(kin_tie))
                param_str = isone(j) ? "fwhm" : "acomp_fwhm"
                if haskey(lines["priors"][string(kin_tie)], "$(param_str)_plim")
                    f_prior = isone(j) ? Uniform(lines["priors"][string(kin_tie)]["$(param_str)_plim"]...) :
                                         Uniform(lines["priors"][string(kin_tie)]["$(param_str)_plim"][j]...)
                end
                if haskey(lines["priors"][string(kin_tie)], "$(param_str)_pstr")
                    f_prior = isone(j) ? eval(Meta.parse(lines["priors"][string(kin_tie)]["$(param_str)_pstr"])) :
                                         eval(Meta.parse(lines["priors"][string(kin_tie)]["$(param_str)_pstr"][j]))
                end
                if haskey(lines["priors"][string(kin_tie)], "$(param_str)_locked")
                    f_locked = isone(j) ? lines["priors"][string(kin_tie)]["$(param_str)_locked"] :
                                          lines["priors"][string(kin_tie)]["$(param_str)_locked"][j]
                end
            end
        end
        value = 1.
        if !(minimum(f_prior) ≤ value ≤ maximum(f_prior))
            value = minimum(f_prior)
        end
        fwhm_tied[j][i] = isone(j) ? Parameter(fwhm_init, f_locked, f_prior) : Parameter(value, f_locked, f_prior)
        msg *= "\nfwhm_tied_$(kin_tie)_$(j) $(fwhm_tied[j][i])"
    end

    @debug msg
    tied_kinematics = TiedKinematics(kin_tied_key_voff, voff_tied, kin_tied_key_fwhm, fwhm_tied)

    # If tie_voigt_mixing is set, all Voigt profiles have the same tied mixing parameter eta
    if lines["tie_voigt_mixing"]
        voigt_mix_tied = Parameter(η_init, false, Uniform(0.0, 1.0))
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
    τ_smooth = movmean(τ_DS, 10)
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
    data[!, "wave"], data[!, "ext"]
end