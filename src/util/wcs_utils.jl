
function create_wcs_from_header(hdr::FITSHeader)
    @debug "create_wcs_from_header: n_header_keys=$(length(hdr))"
    if haskey(hdr, "NAXIS")
        naxis = hdr["NAXIS"]
    else
        error("Header is missing NAXIS keyword")
    end
    @debug "create_wcs_from_header: naxis=$naxis"
    wcs_out = WCSTransform(naxis)

    # Go through header keywords, checking if they are present, and inserting them into the WCS

    for hdrkey in ("CRVAL", "CRPIX", "CDELT", "CRDER", "CSYER", "CTYPE", "CROTA", "CUNIT", "CNAME")
        null = hdrkey in ("CTYPE", "CUNIT", "CNAME") ? "" : 0.
        param = repeat([null], naxis)
        for i in eachindex(param)
            if haskey(hdr, "$hdrkey$i")
                param[i] = hdr["$hdrkey$i"]
            end
        end
        if any(param .≠ null)
            setproperty!(wcs_out, Symbol(lowercase(hdrkey)), param)
        end
    end

    for hdrkey in ("PC", "CD")
        param = zeros(naxis, naxis)
        for i in axes(param, 1)
            for j in axes(param, 2)
                if haskey(hdr, "$(hdrkey)$(i)_$(j)")
                    param[i,j] = hdr["$(hdrkey)$(i)_$(j)"]
                end
            end
        end
        if any(param .≠ 0.)
            setproperty!(wcs_out, Symbol(lowercase(hdrkey)), param)
        end
    end

    for hdrkey in ("EQUINOX", "LATPOLE", "LONPOLE", "MJD-AVG", "MJD-OBS", "RESTFRQ", "RESTWAV",
                   "VELANGL", "VELOSYS", "ZSOURCE", "COLNUM", "DATEAVG", "DATEOBS", "RADESYS",
                   "SPECSYS", "SSYSOBS", "SSYSSRC", "WCSNAME", "ALT")
        if haskey(hdr, hdrkey)
            setproperty!(wcs_out, Symbol(lowercase(replace(hdrkey, "-" => ""))), hdr[hdrkey])
        end
    end

    # Purposely ignore the 3 other OBSGEO keywords because they are what causes WCS.jl to freak out
    obsgeo = zeros(3)
    for (i, hdrkey) in enumerate(["OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z"])
        if haskey(hdr, hdrkey)
            obsgeo[i] = hdr[hdrkey]
        end
    end
    if any(obsgeo .≠ 0.)
        setproperty!(wcs_out, :obsgeo, obsgeo)
    end

    # PS and PV keywords are ignored because WCS.jl cannot handle them

    @debug "create_wcs_from_header: done — radesys=$(wcs_out.radesys), naxis=$(wcs_out.naxis)"
    wcs_out
end


# A utility function that reduces the dimensionality of a WCS.
# For example, taking a 3D WCS with a wavelength axis to a 2D WCS
# with only sky coordinate axes
function shrink_wcs_dimensions(wcs::WCSTransform; keep_axes=1:2)
    @debug "shrink_wcs_dimensions: naxis=$(wcs.naxis) → $(length(keep_axes)), keep_axes=$keep_axes"
    # Go pretty exhaustively through all WCS properties
    WCSTransform(length(keep_axes);
        crval=wcs.crval[keep_axes],
        crpix=wcs.crpix[keep_axes],
        cdelt=wcs.cdelt[keep_axes],
        crder=wcs.crder[keep_axes],
        csyer=wcs.csyer[keep_axes],
        ctype=wcs.ctype[keep_axes],
        crota=wcs.crota[keep_axes],
        cunit=wcs.cunit[keep_axes],
        cname=wcs.cname[keep_axes],
        pc=wcs.pc[keep_axes,keep_axes],
        cd=wcs.cd[keep_axes,keep_axes],
        equinox=wcs.equinox,
        latpole=wcs.latpole,
        lonpole=wcs.lonpole,
        mjdavg=wcs.mjdavg,
        mjdobs=wcs.mjdobs,
        restfrq=wcs.restfrq,
        restwav=wcs.restwav,
        velangl=wcs.velangl,
        velosys=wcs.velosys,
        zsource=wcs.zsource,
        colnum=wcs.colnum,
        dateavg=wcs.dateavg,
        dateobs=wcs.dateobs,
        radesys=wcs.radesys,
        specsys=wcs.specsys,
        ssysobs=wcs.ssysobs,
        ssyssrc=wcs.ssyssrc,
        wcsname=wcs.wcsname,
        obsgeo=[wcs.obsgeo...],
        alt=wcs.alt
    )

end

# A utility function that expands the dimensionality of a WCS.
# For example, taking a 2D WCS with only sky coordinate axes and
# adding a third wavelength axis
function extend_wcs_dimensions(wcs::WCSTransform, λ::Vector{<:QWave})
    @debug "extend_wcs_dimensions: naxis=$(wcs.naxis) → $(wcs.naxis+1), nλ=$(length(λ)), λ_unit=$(unit(λ[1]))"
    pc = zeros(3, 3)
    cd = zeros(3, 3)
    if any(.!iszero.(wcs.pc))
        pc = [wcs.pc[1,1] wcs.pc[1,2] 0.; wcs.pc[2,1] wcs.pc[2,2] 0.; 0. 0. 1.]
    end
    if any(.!iszero.(wcs.cd))
        cd = [wcs.cd[1,1] wcs.cd[1,2] 0.; wcs.cd[2,1] wcs.cd[2,2] 0.; 0. 0. 1.]
    end

    if unit(λ[1]) == u"μm"
        cunit3 = "um"
    elseif unit(λ[1]) == u"angstrom"
        cunit3 = "Angstrom"
    else
        error("Unrecognized wavelength unit: $(unit(λ[1]))")
    end

    # Decide whether or not we need to use a WAVE-TAB or a regular WAVE for ctype3
    dλ = diff(λ)
    wave_linear = all(dλ[1] .≈ dλ)
    if !wave_linear
        @debug "extend_wcs_dimensions: non-linear wavelength spacing — using WAVE-TAB"
        ctype3 = "WAVE-TAB"
        crval3 = crpix3 = cdelt3 = 1.
    else
        @debug "extend_wcs_dimensions: linear wavelength spacing — using WAVE, crval3=$(ustrip(λ[1])), cdelt3=$(ustrip(dλ[1]))"
        ctype3 = "WAVE"
        crval3 = ustrip(λ[1])
        crpix3 = 1.
        cdelt3 = ustrip(dλ[1])
        if any(wcs.cd .> 0)
            cd[3,3] *= cdelt3
        end
    end

    # Handle this at some other point when loading/saving WCSs from FITS files:
    # # Create PS cards that point to an external HDU with the wavelength data
    # ps = [(3, 1, "WCS-TABLE"), (3, 2, "wavelength")]

    # Go pretty exhaustively through all WCS properties
    wcs_out = WCSTransform(wcs.naxis+1;
        crval=[wcs.crval..., crval3],
        crpix=[wcs.crpix..., crpix3],
        cdelt=[wcs.cdelt..., cdelt3],
        crder=[wcs.crder..., wcs.crder[1]],
        csyer=[wcs.csyer..., wcs.csyer[1]],
        ctype=[wcs.ctype..., ctype3],
        crota=[wcs.crota..., 0.],
        cunit=[wcs.cunit..., cunit3],
        cname=[wcs.cname..., ""],
        pc=pc,
        cd=cd,
        equinox=wcs.equinox,
        latpole=wcs.latpole,
        lonpole=wcs.lonpole,
        mjdavg=wcs.mjdavg,
        mjdobs=wcs.mjdobs,
        restfrq=wcs.restfrq,
        restwav=wcs.restwav,
        velangl=wcs.velangl,
        velosys=wcs.velosys,
        zsource=wcs.zsource,
        colnum=wcs.colnum,
        dateavg=wcs.dateavg,
        dateobs=wcs.dateobs,
        radesys=wcs.radesys,
        specsys=wcs.specsys,
        ssysobs=wcs.ssysobs,
        ssyssrc=wcs.ssyssrc,
        wcsname=wcs.wcsname,
        obsgeo=[wcs.obsgeo...],
        alt=wcs.alt
    )
    # Because WCS.jl is bad, setting ps doesn't work...
    # WCS.@check_prop :ps length ps (<=) getfield(wcs_out, :npsmax)
    # nps = length(ps)
    # for i in 1:nps
    #     psi = (ps[i][1], ps[i][2], WCS.convert_string(NTuple{72,UInt8}, ps[i][3]))
    #     unsafe_store!(getfield(wcs_out, :ps), convert(WCS.PSCard, psi), i)
    # end

    wcs_out
end


# A utility function to make a new WCS where the pixel coordinate reference is shifted 
# by deltaX and deltaY pixels.
function make_offset_wcs_crpix(wcs::WCSTransform, Δc::Integer...)
    @debug "offset_wcs_crpix: Δc = $(Δc)" 

    # Go pretty exhaustively through all WCS properties
    WCSTransform(wcs.naxis;
        crval=wcs.crval,
        crpix=wcs.crpix .+ collect(Δc),
        cdelt=wcs.cdelt,
        crder=wcs.crder,
        csyer=wcs.csyer,
        ctype=wcs.ctype,
        crota=wcs.crota,
        cunit=wcs.cunit,
        cname=wcs.cname,
        pc=wcs.pc,
        cd=wcs.cd,
        equinox=wcs.equinox,
        latpole=wcs.latpole,
        lonpole=wcs.lonpole,
        mjdavg=wcs.mjdavg,
        mjdobs=wcs.mjdobs,
        restfrq=wcs.restfrq,
        restwav=wcs.restwav,
        velangl=wcs.velangl,
        velosys=wcs.velosys,
        zsource=wcs.zsource,
        colnum=wcs.colnum,
        dateavg=wcs.dateavg,
        dateobs=wcs.dateobs,
        radesys=wcs.radesys,
        specsys=wcs.specsys,
        ssysobs=wcs.ssysobs,
        ssyssrc=wcs.ssyssrc,
        wcsname=wcs.wcsname,
        obsgeo=[wcs.obsgeo...],
        alt=wcs.alt
    )

end


# A little helper function that takes the FITS header keyword RADESYS and 
# converts it into a type of coordinate object that SkyCoords.jl can understand
function string_to_coordframe(sframe::String)
    @debug "string_to_coordframe: sframe=$sframe"
    if sframe == "ICRS"
        @debug "string_to_coordframe: using ICRSCoords"
        ICRSCoords
    elseif sframe == "FK5"
        @debug "string_to_coordframe: using FK5Coords{2000}"
        # FK5 uses an equinox of 2000
        FK5Coords{2000}
    elseif sframe == "FK4"
        @debug "string_to_coordframe: using FK5Coords{1950}"
        # use FK5 with an equinox of 1950
        FK5Coords{1950}
    else
        @warn "Could not recognize RADESYS of FITS header: $(sframe).  Assuming ICRS."
        ICRSCoords
    end
end



# calculate the rotation angle of the data relative to the sky coordinates, given a WCS
function get_sky_rotation_angle(wcs::WCSTransform)
    @debug "get_sky_rotation_angle: has_pc=$(any(wcs.pc .> 0)), has_cd=$(any(wcs.cd .> 0))"
    if any(wcs.pc .> 0)
        # PC is a rotation matrix
        cosθ =  wcs.pc[1,1]    
        sinθ = -wcs.pc[1,2]    # negative because of how the rotation matrix is defined
    elseif any(wcs.cd .> 0)
        # same as the PC case except the rotation matrix is also scaled by cdelt (matrix multiplication)
        cdelt1 = sqrt(wcs.cd[1,1]^2 + wcs.cd[2,1]^2)
        cdelt2 = sqrt(wcs.cd[1,2]^2 + wcs.cd[2,2]^2)
        cosθ =  wcs.cd[1,1]/cdelt1
        sinθ = -wcs.cd[1,2]/cdelt2
    else
        return 0. * u"rad"
    end

    # flipped RA axis (RA increases to the left)
    # sometimes this is handled with a negative CDELT1, in which case no fix is necessary,
    # and other times CDELT1 > 0 and PC1_1 < 0 (or CD1_1 < 0), in which case we need a sign flip
    if occursin("RA", wcs.ctype[1]) && (wcs.cdelt[1] > 0)
        cosθ *= -1
    end

    result = atan(sinθ, cosθ) * u"rad"
    @debug "get_sky_rotation_angle: θ=$(round(ustrip(result), digits=4)) rad"
    result
end


# Utility function to calculate pixel resolutions in each axis from a WCS
function get_pixel_resolutions(wcs::WCSTransform)
    @debug "get_pixel_resolutions: has_pc=$(any(wcs.pc .> 0)), has_cd=$(any(wcs.cd .> 0)), has_cdelt=$(any(wcs.cdelt .> 0))"
    if any(wcs.pc .> 0) && any(wcs.cdelt .> 0)
        pccd = wcs.pc * diagm(wcs.cdelt)   # matrix multiplication!
    elseif any(wcs.cd .> 0)
        pccd = wcs.cd
    elseif any(wcs.cdelt .> 0)
        pccd = diagm(wcs.cdelt)
    else
        error("Could not find CDELT, PC, or CD entries in WCS")
    end

    sqrt.(sum(pccd.^2, dims=1))'
end


# Utility function to parse WCS "CUNIT" keywords into Unitful objects
function parse_cunits(wcs::WCSTransform)
    @debug "parse_cunits: cunit=$(wcs.cunit)"
    s_units = wcs.cunit
    units_i = Unitful.FreeUnits[]
    for i in 1:length(s_units)
        ui = nothing
        s_unitsi = fits_unitstr_to_unitful(s_units[i])
        try
            ui = uparse(s_unitsi, unit_context=[Unitful, UnitfulAstro])
        catch
            if s_unitsi in ("deg", "")
                ui = u"°"
            elseif s_unitsi == "arcsec" 
                ui = u"arcsecond"
            elseif s_unitsi == "arcmin" 
                ui = u"arcminute"
            elseif s_unitsi == "radian"
                ui = u"rad"
            else
                error("Could not parse CUNIT in FITS header: $(s_units[i])")
            end
        end
        push!(units_i, ui)
    end
    units_i
end