#=
This file contains various functions for parsing configuration files and inputs and converting
them into code objects.
=#

############################## OPTIONS/SETUP/PARSING FUNCTIONS ####################################


function pah_name_to_float(name::AbstractString)
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
    keylist1 = ["n_bootstrap", "silicate_absorption", "extinction_curve", "extinction_screen", "fit_sil_emission", "fit_opt_na_feii", 
                "fit_opt_br_feii", "fit_all_global", "use_pah_templates", "fit_joint", "fit_uv_bump", "fit_covering_frac", "parallel", 
                "plot_spaxels", "plot_maps", "save_fits", "overwrite", "track_memory", "track_convergence", "save_full_model", 
                "line_test_lines", "line_test_threshold", "plot_line_test", "make_movies", "cosmology", "parallel_strategy", 
                "random_seed", "sys_err", "olivine_y", "pyroxene_x", "grain_size", "fit_stellar_continuum", 
                "fit_temp_multexp", "decompose_lock_column_densities", "linemask_width", "map_snr_thresh"]
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
    validate_ir_file(dust)

Checks that the parsed dust file has all of the keys that it should have.
"""
function validate_ir_file(dust::Dict)

    keylist1 = ["dust_features", "extinction", "hot_dust"]
    keylist2 = ["wave", "fwhm"]
    keylist3 = ["tau_9_7", "tau_ice", "tau_ch", "beta"]
    keylist4 = ["temp", "frac", "tau_warm", "tau_cold", "peak"]
    keylist5 = ["val", "plim", "locked"]
    for key ∈ keylist1
        @assert haskey(dust, key) "Missing option $key in dust file!"
    end
    for key ∈ keylist5
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

    keylist1 = ["extinction", "stellar_kinematics", "na_feii_kinematics", "br_feii_kinematics"]
    keylist2 = ["E_BV", "E_BV_factor", "uv_slope", "frac"]
    keylist3 = ["vel", "vdisp"]
    keylist4 = ["val", "plim", "locked"]

    for key ∈ keylist1
        @assert haskey(optical, key) "Missing option $key in optical file!"
    end
    for key ∈ keylist2
        @assert haskey(optical["extinction"], key) "Missing option $key in extinction options!"
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
            @assert haskey(optical["extinction"][key2], key4) "Missing option $key4 in $key2 options!"
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

    keylist3 = ["voff", "fwhm", "h3", "h4"]
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

    # Convert SSP keys into a nested NamedTuple object
    options[:ssps] = (age=(min=options[:ssps]["age"]["min"], 
                           max=options[:ssps]["age"]["max"],
                           num=options[:ssps]["age"]["num"]),
                     logz=(min=options[:ssps]["logz"]["min"],
                           max=options[:ssps]["logz"]["max"],
                           num=options[:ssps]["logz"]["num"]))
    
    # Same for single star options 
    options[:stars] = (teff=(min=options[:stars]["teff"]["min"],
                             max=options[:stars]["teff"]["max"]),
                       logg=(min=options[:stars]["logg"]["min"],
                             max=options[:stars]["logg"]["max"]),
                       logz=(min=options[:stars]["logz"]["min"],
                             max=options[:stars]["logz"]["max"]),
                       alpha=(min=options[:stars]["alpha"]["min"],
                              max=options[:stars]["alpha"]["max"]),
                       use_wr=options[:stars]["use_wr"],
                       use_tpagb=options[:stars]["use_tpagb"])

    # Convert cosmology keys into a proper cosmology object
    options[:cosmology] = cosmology(h=options[:cosmology]["h"], 
                                    OmegaM=options[:cosmology]["omega_m"],
                                    OmegaK=options[:cosmology]["omega_K"],
                                    OmegaR=options[:cosmology]["omega_r"])
    
    # logging messages 
    @debug "Options: $options"

    options
end


function parse_lines(region::SpectralRegion, λunit::Unitful.Units)

    lines = TOML.parsefile(joinpath(@__DIR__, "..", "options", "lines.toml"))
    validate_lines_file(lines)

    cent_vals = Vector{typeof(1.0λunit)}()    # provide all the line wavelengths in consistent units
    for line in keys(lines["lines"])
        cent_unit = uparse(replace(lines["lines"][line]["unit"], 'u' => 'μ'))
        cent_val = lines["lines"][line]["wave"] * cent_unit
        cent_val = uconvert(λunit, cent_val)
        if !is_valid(cent_val, 0.0*λunit, region)
            # remove this line from the dictionary
            delete!(lines["lines"], line)
            continue
        end
        push!(cent_vals, cent_val)
    end

    lines, cent_vals
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

    temp3[!, "rest_wave"] .* u"μm", temp3[!, "flux"], temp4[!, "rest_wave"] .* u"μm", temp4[!, "flux"]
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
    
    # normalize the templates
    temp1[!, "tau"] ./= maximum(temp1[!, "tau"])
    temp2[!, "tau"] ./= maximum(temp2[!, "tau"])

    temp1[!, "rest_wave"] .* u"μm", temp1[!, "tau"], temp2[!, "rest_wave"] .* u"μm", temp2[!, "tau"]
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
    τ_98 = τ_smooth[findmin(x -> abs(x - 9.8), λ_irs)[2]]
    τ_λ = τ_smooth ./ τ_98

    λ_irs .* u"μm", τ_λ
end


function silicate_kvt()
    data = readdlm(joinpath(@__DIR__, "..", "templates", "kvt.txt"), ' ', Float64, '\n')
    data[:,1] .* u"μm", data[:,2]
end


# Setup function for creating the extinction profile from Chiar+Tielens 2006
function silicate_ct()
    data = CSV.read(joinpath(@__DIR__, "..", "templates", "chiar+tielens_2005.dat"), DataFrame, skipto=15, delim=' ', 
        ignorerepeated=true, header=["wave", "a_galcen", "a_local"], select=[1, 2])
    # normalize to the value at 9.7 um
    _, mn = findmin(x -> abs(x - 9.7), data[!, "wave"])
    data[!, "a_galcen"] ./= data[mn, "a_galcen"]
    data[!, "wave"] .* u"μm", data[!, "a_galcen"]
end


# Setup function for creating the extinction profile from OHM+92
function silicate_ohm()
    data = CSV.read(joinpath(@__DIR__, "..", "templates", "ohmc.txt"), DataFrame, delim=' ', ignorerepeated=true,
        header=["wave", "ext"])
    # normalize to the value at 9.7 um
    _, mn = findmin(x -> abs(x - 9.7), data[!, "wave"])
    data[!, "ext"] ./= data[mn, "ext"]
    data[!, "wave"] .* u"μm", data[!, "ext"]
end


function read_dust_κ(x::Real, y::Real, a::QLength, λunit::Unitful.Units)

    # Get wavelength, x, y, and a arrays
    λ = readdlm(joinpath(@__DIR__, "..", "templates", "dorschner_wave.txt"), ' ', Float64, '\n')[:,1] .* u"μm"
    λ = uconvert.(λunit, λ)
    ρ_oli = 3.71u"g/cm^3"
    ρ_pyr = 3.20u"g/cm^3"
    a_cm = uconvert(u"cm", a)  # convert a from μm to cm

    # Read in the Qabs/Qsca arrays
    q_abs_oli = readdlm(joinpath(@__DIR__, "..", "templates", "dorschner_qabs_oli_$(y)_$(a).txt"), ' ', Float64, '\n')[:,1]
    q_abs_pyr = readdlm(joinpath(@__DIR__, "..", "templates", "dorschner_qabs_pyr_$(x)_$(a).txt"), ' ', Float64, '\n')[:,1]

    # Convert absorption efficiencies into mass absorption coefficients (cm^2/g)
    κ_abs_oli = @. 3 * q_abs_oli / (4 * a_cm * ρ_oli)
    κ_abs_pyr = @. 3 * q_abs_pyr / (4 * a_cm * ρ_pyr)

    # Create interpolating functions over wavelength
    κ_abs_pyr = Spline1D(ustrip.(λ), ustrip.(κ_abs_pyr), k=3)
    κ_abs_oli = Spline1D(ustrip.(λ), ustrip.(κ_abs_oli), k=3)

    # Read in the mass absorption coefficiencts for crystalline forsterite
    for_data = readdlm(joinpath(@__DIR__, "..", "templates", "tamani_crystalline_forsterite_k.txt"), ' ', Float64, '\n', comments=true)

    # Create interpolating function 
    # extend edges to 0
    λ_for = [for_data[1,1]-0.2; for_data[1,1]-0.1; for_data[:, 1]; for_data[end, 1]+0.1; for_data[end, 1]+0.2] .* u"μm"
    λ_for = uconvert.(λunit, λ_for)
    κ_abs_for = [0.; 0.; for_data[:, 2]; 0.; 0.] .* u"cm^2/g"
    κ_abs_for = Spline1D(ustrip.(λ_for), ustrip.(κ_abs_for), k=3)

    κ_abs_oli, κ_abs_pyr, κ_abs_for
end


"""
    generate_stellar_populations(λ, lsf, z, Ω, cosmo, ssp_options, stars_options, name,
        custom_template_wave, custom_templates)

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
function generate_stellar_populations(λ::Vector{<:QWave}, intensity_units::Unitful.Units, lsf::Vector{typeof(1.0u"km/s")}, 
    z::Real, template_type::String, cosmo::Cosmology.AbstractCosmology, ssp_options::NamedTuple, stars_options::NamedTuple, 
    name::String, user_mask::Union{Nothing,Vector{<:Tuple}}, custom_template_wave::Union{Nothing,Vector{<:Real}}, 
    custom_templates::Union{Nothing,Array{<:Real,2}})

    # Make sure λ is logarithmically binned
    @assert isapprox((λ[2]/λ[1]), (λ[end]/λ[end-1]), rtol=1e-6) "Input spectrum must be logarithmically binned to fit with stellar populations!"

    # Test to see if templates have already been generated
    if isfile(joinpath("output_$name", "stellar_templates.loki"))
        @info "Loading pre-generated stellar templates from binary file"
        out = deserialize(joinpath("output_$name", "stellar_templates.loki"))
        return out.λ, out.age, out.logz, out.templates
    end

    @debug "Loading in full resolution stellar templates"
    if template_type == "ssp"

        stellar_templates = FITS(joinpath(@__DIR__, "..", "templates", "ssps", "loki.ssps.fits.gz"))

        # Read in wavelengths
        ssp_λ0 = read(stellar_templates["AXES"], "WAVE") .* u"angstrom"

    elseif template_type == "stars"

        stellar_templates = FITS[]

        # cool stars (~<10,000 K)
        # zcools = ["-4.0", "-3.0", "-2.0", "-1.5", "-1.0", "-0.5", "-0.0", "+0.5", "+1.0"]
        # αcools = ["-0.20", "-0.00", "+0.20", "+0.40", "+0.60", "+0.80", "+1.00", "+1.20"]
        zcools = ["-4.00", "-3.50", "-3.00", "-2.50", "-2.00", "-1.50", "-1.00", "-0.50", "-0.00", "+0.30", "+0.50"]
        αcools = ["-0.00", "+0.20", "+0.40", "+0.60"]
        for zcool in zcools
            for αcool in αcools
                if !((stars_options.logz.min ≤ parse(Float64, zcool) ≤ stars_options.logz.max) && 
                     (stars_options.alpha.min ≤ parse(Float64, αcool) ≤ stars_options.alpha.max))
                     continue
                end
                fpath = joinpath(@__DIR__, "..", "templates", "single_stars", "cool", "loki.single_star_cool_z$(zcool)_alpha$(αcool).fits.gz")
                if isfile(fpath)
                    @debug "Reading $fpath"
                    push!(stellar_templates, FITS(fpath))
                end
            end
        end
        # hot stars (~>10,000 K)
        zhots  = ["-2.0", "-1.0", "+0.0"]
        for zhot in zhots
            if !(stars_options.logz.min ≤ parse(Float64, zhot) ≤ stars_options.logz.max)
                continue
            end
            fpath = joinpath(@__DIR__, "..", "templates", "single_stars", "hot", "loki.single_star_hot_z$(zhot).fits.gz")
            if isfile(fpath)
                @debug "Reading $fpath"
                push!(stellar_templates, FITS(fpath))
            end
        end
        # wolf-rayet stars (C-rich and N-rich)
        zwrs = ["sSMC", "SMC", "LMC", "MW"]
        logzwrs = log10.([1/14, 1/7, 1/2, 1.])
        if stars_options.use_wr
            for (logzwr,zwr) in zip(logzwrs,zwrs)
                if !(stars_options.logz.min ≤ logzwr ≤ stars_options.logz.max)
                    continue 
                end
                fpath = joinpath(@__DIR__, "..", "templates", "single_stars", "wr", "loki.single_star_wc_z$(zwr).fits.gz")
                if isfile(fpath)
                    @debug "Reading $path"
                    push!(stellar_templates, FITS(fpath))
                end
                fpath = joinpath(@__DIR__, "..", "templates", "single_stars", "wr", "loki.single_star_wn_z$(zwr).fits.gz")
                if isfile(fpath)
                    @debug "Reading $fpath"
                    push!(stellar_templates, FITS(fpath))
                end
            end
        end
        # TP-AGB stars
        if stars_options.use_tpagb
            fpath = joinpath(@__DIR__, "..", "templates", "single_stars", "tpagb", "loki.single_star_tpagb.fits.gz")
            if isfile(fpath)
                @debug "Reading $fpath"
                push!(stellar_templates, FITS(fpath))
            end
        end
        @assert length(stellar_templates) > 0 "No stellar templates exist within the specified Teff, logg, logz, and alpha ranges!"

        # Read in wavelengths
        ssp_λ0 = read(stellar_templates[1]["WAVE"], "WAVE") .* u"angstrom"

    elseif template_type == "custom"

        wave_unit = unit(λ[1])
        ssp_λ0 = custom_template_wave .* wave_unit

    end

    # Convert to the input spectrum's wavelength units
    wave_unit = unit(λ[1])
    ssp_λ0 = uconvert.(wave_unit, ssp_λ0)

    # Cut off the templates a little bit before/after the input spectrum
    λleft  = 0.98minimum(λ)
    λright = 1.02maximum(λ)

    if (λleft < minimum(ssp_λ0)) || (λright > maximum(ssp_λ0)) 
        if template_type == "custom"
            @warn "The extended input spectrum range of ($λleft, $λright) " * 
            "is outside the template range of ($(minimum(ssp_λ0)), $(maximum(ssp_λ0))). Unexpected results may occur at the edges!"
        else
            error("The extended input spectrum range of ($λleft, $λright) " * 
            "is outside the template range of ($(minimum(ssp_λ0)), $(maximum(ssp_λ0))). Please adjust the input spectrum, or use" *
            "different templates.")
        end
    end

    # Mask to a range around the input spectrum
    mask = λleft .< ssp_λ0 .< λright
    ssp_λ = ssp_λ0[mask]
    # Resample onto a linear wavelength grid
    Δλ = minimum(diff(λ))
    ssp_λlin = λleft:Δλ:λright
    
    # LSF FWHM of the input spectrum in wavelength units, interpolated at the locations of the SSP templates
    inp_fwhm = Spline1D(ustrip.(λ), ustrip.(lsf ./ C_KMS .* λ), k=1, bc="nearest")(ustrip.(ssp_λlin)) .* wave_unit
    # The FWHM resolution of the stellar templates is essentially negligible, as they are produced from theoretical model atmospheres 
    ssp_fwhm = 0. * wave_unit
    # Difference in resolutions between the input spectrum and SSP templates, in pixels
    dfwhm = sqrt.(clamp.(inp_fwhm.^2 .- ssp_fwhm.^2, 0.0*wave_unit^2, Inf*wave_unit^2)) ./ Δλ 

    # Logarithmically rebinned wavelengths
    logscale = log(λ[2]/λ[1])
    ssp_lnλ = get_logarithmic_λ(ustrip.(ssp_λlin), logscale) .* wave_unit
    # pad up to the next nice product of low prime numbers
    npad = nextprod([2,3,5], length(ssp_lnλ))
    ssp_lnλ = ssp_lnλ[1] .* (ssp_lnλ[2]/ssp_lnλ[1]).^(0:npad-1)
    @assert isapprox((ssp_lnλ[2]/ssp_lnλ[1]), (ssp_lnλ[end]/ssp_lnλ[end-1]), rtol=1e-6)

    # Calculate luminosity distance in cm in preparation to convert luminosity to flux
    dL = luminosity_dist(u"cm", cosmo, z)

    # Generate templates over a range of ages and metallicities
    if template_type == "ssp"
        ages_out = exp.(range(log(ssp_options.age.min), log(ssp_options.age.max), ssp_options.age.num)) .* u"Gyr"  # logarithmically spaced from 1 Myr to 15 Gyr
        logzs_out = range(ssp_options.logz.min, ssp_options.logz.max, ssp_options.logz.num)                        # linearly spaced from log(Z/Zsun) = [M/H] = -2.3 to 0.4

        # Compare this with out default set of templates 
        n_ages = 40
        n_logzs = 10
        ages_in = Float64.(read(stellar_templates["AXES"], "AGE")[1:n_ages]) .* u"Gyr"
        logzs_in = Float64.(read(stellar_templates["AXES"], "LOGZ")[1:n_logzs])
        input_units = u"Lsun/Msun/Hz"
        ssp_templates_in = Float64.(read(stellar_templates["TEMPLATES"])) .* input_units

        output_units = intensity_units*u"sr"/u"Msun"
        ssp_templates_out = zeros(typeof(1.0*output_units), length(ssp_λlin), length(ages_out), length(logzs_out))
        n_temp = size(ssp_templates_out, 2) * size(ssp_templates_out, 3)

        @assert round(ssp_options.age.min, digits=3) ≥ round(ustrip(minimum(ages_in)), digits=3) "Minimum age is $(round(ustrip(minimum(ages_in)), digits=3)) Gyr"
        @assert round(ssp_options.age.max, digits=3) ≤ round(ustrip(maximum(ages_in)), digits=3) "Maximum age is $(round(ustrip(maximum(ages_in)), digits=3)) Gyr"
        @assert round(ssp_options.logz.min, digits=3) ≥ round(minimum(logzs_in), digits=3) "Minimum logz is $(round(minimum(logzs_in), digits=3))"
        @assert round(ssp_options.logz.max, digits=3) ≤ round(maximum(logzs_in), digits=3) "Maximum logz is $(round(maximum(logzs_in), digits=3))"

        @info "Generating $n_temp simple stellar population templates with " * 
            "ages ∈ ($(minimum(ages_out)), $(maximum(ages_out))), log(Z/Zsun) ∈ ($(minimum(logzs_out)), $(maximum(logzs_out)))"
    
    elseif template_type == "stars"

        input_units = u"erg/s/cm^2/Hz/sr"
        output_units = intensity_units*u"sr"/u"Msun" 
        wave_full = read(stellar_templates[1]["WAVE"], "WAVE") .* u"angstrom"
        nwave = length(wave_full)

        # read in the single star templates
        ssp_templates_in = zeros(typeof(1.0input_units), (0,nwave))
        logg_in = zeros(0)
        logt_in = zeros(0)
        for i in eachindex(stellar_templates)
            temp_in = read(stellar_templates[i]["SPEC"], "SPEC") .* input_units
            ssp_templates_in = cat(ssp_templates_in, temp_in, dims=1)
            logg_in = cat(logg_in, read(stellar_templates[i]["AXES"], "LOGG"), dims=1)
            logt_in = cat(logt_in, read(stellar_templates[i]["AXES"], "LOGT"), dims=1)
        end
        # remove all templates that dont fall within the LOGG/LOGT range
        gt_mask = (stars_options.logg.min .≤ logg_in .≤ stars_options.logg.max) .&
                  (stars_options.teff.min .≤ (10 .^ logt_in) .≤ stars_options.teff.max)
        ssp_templates_in = ssp_templates_in[gt_mask,:]
        @assert size(ssp_templates_in, 2) > 0 "No stellar templates exist within the specified Teff, logg, logz, and alpha ranges!"

        ages_in = ages_out = ones(size(ssp_templates_in, 1)) .* NaN .* u"Gyr"
        logzs_in = logzs_out = [NaN]
        ssp_templates_in = reshape(ssp_templates_in, (length(logzs_in), length(ages_in), nwave))

    elseif template_type == "custom"

        input_units = intensity_units 
        output_units = intensity_units*u"sr"/u"Msun"
        ssp_templates_in = custom_templates .* input_units
        ages_in = ages_out = ones(size(ssp_templates_in, 2)) .* NaN .* u"Gyr"
        logzs_in = logzs_out = [NaN]
        ssp_templates_in = reshape(ssp_templates_in, (length(custom_template_wave), length(ages_in), length(logzs_in)))
        ssp_templates_in = permutedims(ssp_templates_in, (3,2,1))

    end

    ssp_templates_in_resamp = zeros(typeof(1.0*input_units), length(ssp_λlin), length(ages_in), length(logzs_in)) 
    ssp_templates_final = zeros(typeof(1.0*output_units), length(ssp_lnλ), length(ages_out), length(logzs_out))
    peraa = typeof(1.0*intensity_units) <: QPerWave

    # First, resample all of the input templates onto the output wavelength grid
    @info "Resampling wavelengths..."
    prog = Progress(length(ages_in)*length(logzs_in); showspeed=true)
    for (z_ind, logz) in enumerate(logzs_in)
        for (age_ind, age) in enumerate(ages_in)
            ssp_templates_in_resamp[:, age_ind, z_ind] = Spline1D(ustrip.(ssp_λ), ustrip.(ssp_templates_in[z_ind, age_ind, mask]), 
                k=1, bc="nearest")(ustrip.(ssp_λlin)) .* input_units
            next!(prog)
        end
    end

    # Then, do a 2D interpolation over age/metallicity (only if using actual SSPs)
    if template_type == "ssp"
        @info "Resampling ages/logzs..."
        prog = Progress(length(ssp_λlin); showspeed=true)
        for (w_ind, wi) in enumerate(ssp_λlin)
            interpolator = Spline2D(ustrip.(ages_in), ustrip.(logzs_in), ustrip.(ssp_templates_in_resamp[w_ind, :, :]), kx=1, ky=1)
            for (z_ind, logz) in enumerate(logzs_out)
                for (age_ind, age) in enumerate(ages_out)
                    # Get the interpolated luminosity
                    ssp_L = interpolator(ustrip(age), ustrip(logz)) * input_units
                    if peraa 
                        # if necessary, convert to per-wavelength 
                        ssp_L = uconvert(input_units*u"Hz"/unit(wi), ssp_L * C_KMS / wi^2)
                    end
                    # Convert Lsun/Msun/[Hz,Ang] to [flux units]/Msun
                    ssp_flux = uconvert(output_units, ssp_L / (4π*dL^2))
                    # Insert into templates array
                    ssp_templates_out[w_ind, age_ind, z_ind] = ssp_flux
                end
            end
            next!(prog)
        end
    else
        temp_out = ssp_templates_in_resamp
        if peraa 
            temp_out = uconvert.(input_units*u"Hz"/unit(ssp_λlin[1]), ssp_templates_in_resamp .* C_KMS ./ ssp_λlin.^2)
        end
        ssp_templates_out = uconvert.(output_units, temp_out.*u"sr"/u"Msun")
        n_temp = size(ssp_templates_out, 2) * size(ssp_templates_out, 3)
    end

    # Finally, do some instrumental broadening and resample onto the logarithmic wavelength grid
    @info "Applying instrumental broadening..."
    prog = Progress(n_temp; showspeed=true)
    for (z_ind, logz) in enumerate(logzs_out)
        for (age_ind, age) in enumerate(ages_out)
            ssp_F = ssp_templates_out[:, age_ind, z_ind]
            ssp_F = convolveGaussian1D(ssp_F, dfwhm)
            ssp_F = Spline1D(ustrip.(ssp_λlin), ustrip.(ssp_F), k=1, bc="nearest")(ustrip.(ssp_lnλ)) .* output_units
            ssp_templates_final[:, age_ind, z_ind] .= ssp_F
            next!(prog)
        end
    end

    # Normalize if not using SSPs 
    if template_type != "ssp"

        @info "Normalizing templates..."

        for (z_ind, logz) in enumerate(logzs_out)
            for (age_ind, age) in enumerate(ages_out)
                temp_i = ssp_templates_final[:, age_ind, z_ind]
                if !peraa 
                    temp_i = temp_i .* uconvert(wave_unit/u"s", C_KMS) ./ ssp_lnλ.^2
                end
                norm = NumericalIntegration.integrate(ustrip.(ssp_lnλ), ustrip.(temp_i), Trapezoidal())
                ssp_templates_final[:, age_ind, z_ind] ./= norm
            end
        end

    end

    if template_type == "ssp"
        close(stellar_templates)
    elseif template_type == "stars"
        for i in eachindex(stellar_templates)
            close(stellar_templates[i])
        end
    # elseif template_type == "custom"
    end

    # notice: we store the ssps in units of flux per solar mass and not intensity per solar mass.
    #         but the rest of the code works in intensities; this makes the normalized fitted amplitude parameter 
    #         for the ssps have units of 1/sr.  
    # why?  : simply because this makes the fitted amplitude parameter be closer to 1 numerically. typical solar masses 
    #         for galaxies range from ~10^7 - 10^12 msun, whereas typical solid angles for observed spectra are 
    #         on the order of <1 arcsec^2 ~ 10^-11 sr.  in other words, these factors conspire to make the fitted 
    #         amplitude close to 1.
    # so?   : because of this, we will have to multiply the solid angle factor back in at the end of the fitting process
    #         to get the result back in units of Msun.

    # save for later
    serialize(joinpath("output_$name", "stellar_templates.loki"), (λ=ssp_lnλ, age=ages_out, logz=logzs_out, templates=ssp_templates_final))

    ssp_lnλ, ages_out, logzs_out, ssp_templates_final
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
function generate_feii_templates(λ::Vector{<:QWave}, intensity_units::Unitful.Units, lsf::Vector{typeof(1.0u"km/s")})

    # Make sure λ is logarithmically binned
    @assert isapprox((λ[2]/λ[1]), (λ[end]/λ[end-1]), rtol=1e-6) "Input spectrum must be logarithmically binned to fit optical data!"

    # Read the templates in from the specified directory
    template_path = joinpath(@__DIR__, "..", "templates", "veron-cetty_2004")
    na_feii_temp, _ = readdlm(joinpath(template_path, "VC04_na_feii_template.csv"), ',', Float64, '\n', header=true)
    br_feii_temp, _ = readdlm(joinpath(template_path, "VC04_br_feii_template.csv"), ',', Float64, '\n', header=true)
    feii_λ = na_feii_temp[:, 1] * u"angstrom"
    na_feii_temp = na_feii_temp[:, 2]
    br_feii_temp = br_feii_temp[:, 2]

    # Convert to vacuum wavelengths
    feii_λ = airtovac.(ustrip.(feii_λ)) .* unit(feii_λ[1])
    # Convert to match the units of the input wavelength vector
    wave_unit = unit(λ[1])
    feii_λ = uconvert.(wave_unit, feii_λ)
    # Linear spacing 
    Δλ = (maximum(feii_λ) - minimum(feii_λ)) / (length(feii_λ)-1)

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
    na_feii_temp = Spline1D(ustrip.(feii_λ), na_feii_temp, k=1, bc="nearest")(ustrip.(feii_λlin))
    br_feii_temp = Spline1D(ustrip.(feii_λ), br_feii_temp, k=1, bc="nearest")(ustrip.(feii_λlin))

    # LSF FWHM of the input spectrum in wavelength units, interpolated at the locations of the SSP templates
    inp_fwhm = Spline1D(ustrip.(λ), ustrip.(lsf ./ C_KMS .* λ), k=1, bc="nearest")(ustrip.(feii_λlin)) .* wave_unit
    # FWHM resolution of the Fe II templates
    feii_fwhm = uconvert(wave_unit, 1.0u"angstrom")
    # Difference in resolutions between the input spectrum and SSP templates, in pixels
    dfwhm = sqrt.(clamp.(inp_fwhm.^2 .- feii_fwhm.^2, 0.0*wave_unit^2, Inf*wave_unit^2)) ./ Δλ

    # Convolve the templates to match the resolution of the input spectrum
    na_feii_temp = convolveGaussian1D(na_feii_temp, dfwhm)
    br_feii_temp = convolveGaussian1D(br_feii_temp, dfwhm)

    # Logarithmically rebinned wavelengths
    logscale = log(λ[2]/λ[1])
    feii_lnλ = get_logarithmic_λ(ustrip.(feii_λlin), logscale) .* wave_unit
    na_feii_temp = Spline1D(ustrip.(feii_λlin), na_feii_temp, k=1, bc="nearest")(ustrip.(feii_lnλ))
    br_feii_temp = Spline1D(ustrip.(feii_λlin), br_feii_temp, k=1, bc="nearest")(ustrip.(feii_lnλ))

    # if the input flux units are not per wavelength, we need to convert the templates
    # (the templates are normalized, but in per-wavelength units, so they still need to be converted)
    if typeof(intensity_units) <: QPerFreq
        na_feii_temp = ustrip.(fluxconvert.(na_feii_temp .* u"erg/s/cm^2/angstrom", feii_lnλ))
        br_feii_temp = ustrip.(fluxconvert.(br_feii_temp .* u"erg/s/cm^2/angstrom", feii_lnλ))
        # (the normalization doesnt matter because it's divided out later)
    end

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


# Create a local logger for an individual spaxel
timestamp_logger(logger) = TransformerLogger(logger) do log
    merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
end
make_spaxel_logger(name::String, fname::String) = TeeLogger(ConsoleLogger(stdout, Logging.Info), 
    timestamp_logger(MinLevelLogger(FileLogger(joinpath("output_$(name)", "logs", "loki.$fname.log"); 
    always_flush=true), Logging.Debug)))


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
    write_fit_results_csv(cube_fitter, fname, result)

Writes the CSV file that contains the fit results for a particular spaxel
(best fit values and errors).
"""
function write_fit_results_csv(cube_fitter::CubeFitter, fname::String, result::SpaxelFitResult)
    pretty = pretty_print_results(result)
    open(joinpath("output_$(cube_fitter.name)", "spaxel_binaries", "$fname.csv"), "w") do f
        write(f, pretty)
    end
end


"""
    read_fit_results_csv(cube_fitter, fname)

Reads the CSV file that contains the file results for a particular spaxel
and returns the best fit values and errors as separate vectors (errors is a 
2D matrix with the lower/upper errors)
"""
function read_fit_results_csv(name::String, fname::String; output_path::Union{String,Nothing}=nothing)

    folder = isnothing(output_path) ? "output_$(name)" : output_path
    # Read in the CSV as a DataFrame
    df = CSV.read(joinpath(folder, "spaxel_binaries", "$fname.csv"), DataFrame, delim='\t', stripwhitespace=true)
    rename!(df, strip.(names(df)))

    # read back in the raw values
    p_out = df[!, "best"]
    p_err_low = df[!, "error_lower"]
    p_err_upp = df[!, "error_upper"]

    # apply the units
    pwh = findfirst(x -> contains(x, "unit"), names(df))
    p_unit = strip.(df[!, pwh])
    p_unit = uparse.([u == "" ? "NoUnits" : u for u in p_unit])
    p_out = p_out .* p_unit
    p_err_low = p_err_low .* p_unit
    p_err_upp = p_err_upp .* p_unit
    p_err = [p_err_low p_err_upp]

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
