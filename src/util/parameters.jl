#=
This file, like utils, is not intended to be directly accessed by the 
user when fitting IFU cubes. Rather, it contains various Parameter structures
that are helpful for containing certain combinations of model parameters and
related quantities.
=#

# aliases for convenience
const QUnitless = typeof(NoUnits)
const QLength = Quantity{<:Real, u"𝐋"}
const QTemp = typeof(1.0u"K")

const Qum = typeof(1.0u"μm")
const QAng = typeof(1.0u"angstrom")
const QWave = Union{Qum,QAng}
const QInvum = typeof(1.0/u"μm")
const QInvAng = typeof(1.0/u"angstrom")
const QInvWave = Union{QInvum,QInvAng}

const QPerFreq = typeof(1.0u"erg/s/cm^2/Hz/sr")
const QPerum = typeof(1.0u"erg/s/cm^2/μm/sr")
const QPerAng = typeof(1.0u"erg/s/cm^2/angstrom/sr")
const QPerWave = Union{QPerum,QPerAng}
const QSIntensity = Union{QPerFreq,QPerum,QPerAng}
const QIntensity = typeof(1.0u"erg/s/cm^2/sr")
const QFlux = typeof(1.0u"erg/s/cm^2")
const QGeneralPerWave = Quantity{<:Real, u"𝐌*𝐋^-1*𝐓^-3"}
const QGeneralPerFreq = Quantity{<:Real, u"𝐌*𝐓^-2"}
const QVelocity = typeof(1.0u"km/s")

# Create some type categories
abstract type Parameter end
abstract type Parameters end
abstract type Tie end
abstract type Config end

# A few simple enum types
@enum WavelengthRange begin
    UVOptical
    Infrared
    UVOptIR
end

@enum Transformation begin
    RestframeTransform
    LogTransform
    NormalizeTransform
end

"""
    FitParameter(value, locked, limits)

A struct for holding information about parameters' intial values and priors

# Fields
- `value::Number`: The initial value of the parameter
- `locked::Bool`: false to allow the parameter to vary based on the prior, true to keep it fixed
- `limits::Tuple`: lower/upper limits on the parameter, if it is allowed to vary
- `tie::Union{Tie,Nothing}`: an optional tie that specifies what other parameters this one is tied to
"""
mutable struct FitParameter{T<:Number} <: Parameter

    value::T
    locked::Bool
    limits::Tuple{T, T}
    tie::Union{Tie,Nothing}
    
    # Constructor function
    function FitParameter(value::T, locked::Bool, limits::Tuple{T, T}, 
        tie::Union{Tie,Nothing}=nothing) where {T<:Number}
        # Make sure the upper limit is strictly greater than the lower limit
        @assert limits[2] > limits[1]
        new{T}(value, locked, limits, tie)
    end

end


# Tie parameters give a group identifier and an optional ratio between the tied parameters
# (i.e. for fitting lines with fixed amplitude ratios)
struct RatioTie{T<:Real} <: Tie
    group::Symbol
    ratio::T
end

struct FlatTie <: Tie
    group::Symbol
end

# A parameter that is not being fit for, but is nevertheless of interest to us
struct NonFitParameter{T<:Number} <: Parameter 
    _type::Type
    function NonFitParameter{T}() where {T}
        new{T}(T)
    end
end

# A collection of FitParameter with names 
# Think of this like a dictionary but with a defined order (couldve just used an OrderedDict but I already wrote all of this so...)
struct FitParameters <: Parameters
    names::Vector{String}                            # names of all the parameters
    labels::Vector{String}                           # latex-formatted labels for all the parameters
    transformations::Vector{Vector{Transformation}}  # transformations to be applied to each parameter after fitting
    _parameters::Vector{FitParameter{<:Number}}      # the internal storage of the parameter objects

    function FitParameters(names::Vector{String}, labels::Vector{String},
        transformations::Vector{Vector{Transformation}}, parameters::Vector{FitParameter{<:Number}})
        @assert names == unique(names) "repeat names are not allowed!"
        new(names, labels, transformations, parameters)
    end
end

# The non-fit equivalent of FitParameters
struct NonFitParameters <: Parameters
    names::Vector{String}
    labels::Vector{String}
    transformations::Vector{Vector{Transformation}}
    _parameters::Vector{NonFitParameter{<:Number}}

    function NonFitParameters(names::Vector{String}, labels::Vector{String},
        transformations::Vector{Vector{Transformation}}, parameters::Vector{NonFitParameter{<:Number}}) 
        @assert names == unique(names) "repeat names are not allowed!"
        new(names, labels, transformations, parameters)
    end
end

# allow both fit and nonfit parameters
struct AllParameters <: Parameters
    names::Vector{String}
    labels::Vector{String}
    transformations::Vector{Vector{Transformation}}
    _parameters::Vector{<:Union{FitParameter{<:Number},NonFitParameter{<:Number}}}

    function AllParameters(names::Vector{String}, labels::Vector{String},
        transformations::Vector{Vector{Transformation}}, 
        parameters::Vector{Union{FitParameter{<:Number},NonFitParameter{<:Number}}}) 
        @assert names == unique(names) "repeat names are not allowed!"
        new(names, labels, transformations, parameters)
    end
end


struct FitProfile{S<:Union{Symbol,String}}
    profile::S
    fit_parameters::FitParameters
    nonfit_parameters::NonFitParameters

    function FitProfile(profile::S, fit_parameters::FitParameters, nonfit_parameters::NonFitParameters) where {S<:Union{Symbol,String}}
        new{S}(profile, fit_parameters, nonfit_parameters)
    end
end

FitProfiles = Vector{FitProfile}


struct FitFeatures{S<:Union{Symbol,String},Q<:QWave}
    names::Vector{S}                       # PAH or emission line names
    labels::Vector{String}                 # latex-formatted labels for each feature 
    λ₀::Vector{Q}                          # central wavelengths of the features
    profiles::Vector{FitProfiles}          # first index = each feature, second index = each profile in the feature
    composite::Vector{NonFitParameters}    # first index = each feature, second index = each extra composite parameter
    config::Config

    function FitFeatures(names::Vector{S}, labels::Vector{String}, λ₀::Vector{Q}, profiles::Vector{FitProfiles},
        composite::Vector{NonFitParameters}, config::Config) where {S<:Union{Symbol,String},Q<:QWave}
        new{S,Q}(names, labels, λ₀, profiles, composite, config)
    end
end


function make_single_line_object(f::FitFeatures, i::Integer, n_prof::Integer)
    cfg = LineConfig(
        BitVector([f.config.annotate[i]]), [f.config.sort_order[i]], f.config.combined, f.config.rel_amp,
        f.config.rel_voff, f.config.rel_fwhm
    )
    FitFeatures([f.names[i]], [f.labels[i]], [f.λ₀[i]], [f.profiles[i][1:n_prof]], [f.composite[i]], cfg)
end


struct NoConfig <: Config end

struct PAHConfig <: Config 
    all_feature_names::Vector{String}    # the names of each individual feature (ignoring complexes)
    all_feature_labels::Vector{String}   # latex-formatted labels
end

struct LineConfig <: Config
    annotate::BitVector                  # whether or not to annotate
    sort_order::Vector{Int}              # defines the sorting sense for the components
    combined::Vector{Vector{Symbol}}     # combined line maps
    # Relative config parameters
    rel_amp::Bool                        #  ...amp relative
    rel_voff::Bool                       #  ...voff relative
    rel_fwhm::Bool                       #  ...fwhm relative
end


# A big composite struct holding all the fit and non-fit parameters for a model
struct ModelParameters
    continuum::FitParameters
    abs_features::FitFeatures
    dust_features::FitFeatures
    lines::FitFeatures
    statistics::NonFitParameters
end


# These will define how CubeFitter objects behave
struct SpectralRegion{T<:Union{typeof(1.0u"μm"),typeof(1.0u"angstrom")}}
    λlim::Tuple{T,T}
    mask::Vector{Tuple{T,T}}
    n_channels::Int
    channel_masks::Vector{BitVector}
    channel_bounds::Vector{T}
    gaps::Vector{Tuple{T,T}}
    wavelength_range::WavelengthRange
end


# A little container for stellar populations 
mutable struct StellarPopulations{Qw<:QWave,Qa<:typeof(1.0u"Gyr"),
    Qv<:typeof(1.0u"km/s"),S<:Real,T<:Quantity}
    λ::Vector{Qw}
    ages::Vector{Qa}
    logzs::Vector{S}
    templates::Matrix{T}
    vsysts::Vector{Qv}
end

# A little container for iron templates
mutable struct FeIITemplates{Qw<:QWave,Qv<:typeof(1.0u"km/s"),C<:Complex}
    λ::Vector{Qw}
    npad::Int
    na_fft::Vector{C}
    br_fft::Vector{C}
    vsysts::Vector{Qv}
end


λlim(region::SpectralRegion) = region.λlim
nchannels(region::SpectralRegion) = region.n_channels
umask(region::SpectralRegion) = region.mask
wavelength_range(region::SpectralRegion) = region.wavelength_range
function is_valid(λ::Quantity{<:Real, u"𝐋"}, tol::Quantity{<:Real, u"𝐋"}, region::SpectralRegion) 
    lim = λlim(region)
    valid = (lim[1]-tol) < λ < (lim[2]+tol)
    for mlim in umask(region)
        valid &= !(mlim[1] < λ < mlim[2])
    end
    valid
end

# lock or unlock the parameter
lock!(p::FitParameter) = (p.locked = true)
unlock!(p::FitParameter) = (p.locked = false)

tie!(p1::FitParameter, group::Symbol) = (p1.tie = FlatTie(group))
tie!(p1::FitParameter, group::Symbol, r::Real) = (p1.tie = RatioTie(group, r))
untie!(p1::FitParameter) = (p1.tie = nothing)

# update the limits or value of the parameter
function set_plim!(p::FitParameter, limits::Tuple) 
    @assert limits[1] < limits[2]
    _, _, l1, l2 = promote(p.limits[1], p.limits[2], limits[1], limits[2])
    p.limits = (l1, l2)
end
function set_val!(p::FitParameter{T}, v::Number) where {T}
    _, vnew = promote(p.value, v)
    p.value = vnew
end

# extend the "uconvert" function from Unitful
function Unitful.uconvert(new_unit::Unitful.Units, p::FitParameter{<:Quantity})
    new_val = uconvert(new_unit, p.value)
    new_lim = (uconvert(new_unit, p.limits[1]), uconvert(new_unit, p.limits[2]))
    FitParameter(new_val, p.locked, new_lim, p.tie)
end

# extend some unitful functions
Unitful.ustrip(p::FitParameter) = FitParameter(ustrip(p.value), p.locked, ustrip.(p.limits), p.tie)
Unitful.unit(p::FitParameter) = unit(p.value)
Unitful.unit(p::NonFitParameter) = unit(p._type)

# check if everything is good
function check_valid(p::FitParameter)
    @assert isfinite(p.value)
    @assert typeof(p.value) == typeof(p.limits[1]) == typeof(p.limits[2])
    @assert p.limits[1] < p.limits[2]
    @assert p.limits[1] ≤ p.value ≤ p.limits[2]
end


# formatting functions
function Base.show(io::IO, p::FitParameter)
    Base.show(io, "Parameter: value = $(p.value) | locked = $(p.locked) | limits = $(p.limits)")
end
function Base.show(io::IO, mp::ModelParameters)
    nc = length(mp.continuum)
    nd = length(get_flattened_fit_parameters(mp.dust_features))
    nl = length(get_flattened_fit_parameters(mp.lines))
    nn1 = length(get_flattened_nonfit_parameters(mp.dust_features))
    nn2 = length(get_flattened_nonfit_parameters(mp.lines))
    nn3 = length(mp.statistics)
    Base.show(io, 
        "ModelParameters | Continuum parameters: $(nc), PAH parameters: $(nd), Line parameters: $(nl), " * 
        "Non-fit parameters: $(nn1+nn2+nn3)"
    )
end


"""
    parameter_from_dict(dict)

A constructor function for Parameter structs given a Dictionary
"""
function parameter_from_dict(dict::Dict; units::Unitful.Units=unit(1.0))

    # Unpack the dictionary into fields of the Parameter
    value = dict["val"] * units
    locked = dict["locked"]
    # plim: absolute upper/lower limits
    lims = (dict["plim"]...,) .* units

    FitParameter(value, locked, lims)
end 


"""
    parameter_from_dict_wave(dict)

A constructor function for Parameter structs given a Dictionary,
using deltas on upper/lower limits, i.e. if val = 5 and plim = [-0.1, 0.1],
then the true limits will be 5 .+ [-0.1, 0.1] = [4.9, 5.1]
"""
function parameter_from_dict_wave(dict::Dict; units::Unitful.Units=unit(1.0))

    # Unpack the dictionary into fields of the Parameter
    value = dict["val"]
    locked = dict["locked"]
    # plim: absolute upper/lower limits
    lims = (dict["plim"].+value...,)
    # hopefully removes floating-point rounding errors
    lims = round.(lims, digits=4)

    value = value * units
    lims = lims .* units

    FitParameter(value, locked, lims)
end


"""
    parameter_from_dict_fwhm(dict)

A constructor function for Parameter structs given a Dictionary,
using fractional values on upper/lower limits, i.e. if val = 5 and
plim = [0.5, 2], then the true limits will be 5 .* [0.5, 2] = [2.5, 10]
"""
function parameter_from_dict_fwhm(dict::Dict; units::Unitful.Units=unit(1.0))

    # Unpack the dictionary into fields of the Parameter
    value = dict["val"]
    locked = dict["locked"]
    # plim: absolute upper/lower limits
    lims = (dict["plim"].*value...,)
    # hopefully removes floating-point rounding errors
    lims = round.(lims, digits=4)

    value = value * units
    lims = lims .* units

    FitParameter(value, locked, lims)
end

# NOTE: this does the exact same thing as the builtin "indexin" function -- indexin(needles, haystack)
# for arrays of strings. But this implementation is faster and allocates less memory
StringOrSymbol = Union{AbstractString,Symbol}
fast_indexin(needles::AbstractVector{<:StringOrSymbol}, haystack::AbstractVector{<:StringOrSymbol}) = 
    [findfirst(straw -> straw == needle, string.(haystack)) for needle in string.(needles)]
fast_indexin(needle::StringOrSymbol, haystack::AbstractVector{<:StringOrSymbol}) = 
    findfirst(straw -> straw == string(needle), string.(haystack))


# methods for obtaining a named parameter
Base.getindex(p::Parameters, ind) = p._parameters[ind]
function Base.getindex(p::Parameters, name::String)
    ind = findfirst(pname -> pname == name, p.names)
    p._parameters[ind]
end
function Base.getindex(p::Parameters, names::AbstractVector{String})
    inds = fast_indexin(names, p.names)
    p._parameters[inds]
end

# methods for adding to a parameter list
function Base.push!(p::Parameters, name::AbstractString, label::AbstractString, trans::Vector{Transformation}, new::Parameter)
    @assert !(name in p.names) "$name already has an entry in $(typeof(p)) object!"
    push!(p.names, name)
    push!(p.labels, label)
    push!(p.transformations, trans)
    push!(p._parameters, new)
end
function Base.append!(p::Parameters, new::Parameters) 
    for new_name in new.names
        @assert !(new_name in p.names) "$new_name has an entry in both $(typeof(p)) objects!"
    end
    append!(p.names, new.names)
    append!(p.labels, new.labels)
    append!(p.transformations, new.transformations)
    append!(p._parameters, new._parameters)
end
Base.length(p::Parameters) = length(p._parameters)

# methods for deleting a parameter from the list
function Base.deleteat!(p::Parameters, ind::Int)
    deleteat!(p.names, ind)
    deleteat!(p.labels, ind)
    deleteat!(p.transformations, ind)
    deleteat!(p._parameters, ind)
end
function Base.deleteat!(p::Parameters, name::String)
    ind = findfirst(pname -> pname == name, p.names)
    deleteat!(p, ind)
end

# methods for obtaining the vector of parameter limits
get_plims(p::FitParameters, ind::Int) = p[ind].limits   
get_plims(p::FitParameters, name::String) = p[name].limits   
function _getproperty(p::FitParameters, ::Val{:limits}) 
    plims = Vector{Tuple{Number,Number}}(undef, length(p.names))
    for (i, param) in enumerate(p._parameters)
        plims[i] = param.limits
    end
    plims
end

# methods for obtaining the lock vector
get_lock(p::FitParameters, ind::Int) = p[ind].locked
get_lock(p::FitParameters, name::String) = p[name].locked
function _getproperty(p::FitParameters, ::Val{:locked}) 
    locks = BitVector(undef, length(p.names))
    for (i, param) in enumerate(p._parameters)
        locks[i] = param.locked
    end
    locks
end

# methods for obtaining the value vector
get_val(p::FitParameters, ind::Int) = p[ind].value
get_val(p::FitParameters, name::String) = p[name].value
function _getproperty(p::FitParameters, ::Val{:values})
    vals = Vector{Number}(undef, length(p.names))
    for (i, param) in enumerate(p._parameters)
        vals[i] = param.value
    end
    vals
end

# methods for obtaining the tied vector (vector of symbols)
get_tie(p::FitParameters, ind::Int) = p[ind].tie
get_tie(p::FitParameters, name::String) = p[name].tie
function _getproperty(p::FitParameters, ::Val{:ties})
    ties = Vector{Union{Tie,Nothing}}(undef, length(p.names))
    for (i, param) in enumerate(p._parameters)
        ties[i] = param.tie
    end
    ties
end

get_units(p::Parameters, ind::Int) = unit(p[ind])
get_units(p::Parameters, name::String) = unit(p[name])
get_units(p::Parameters) = unit.(p._parameters)

_getproperty(p::FitParameters, ::Val{s}) where {s} = getfield(p, s)
Base.getproperty(p::FitParameters, s::Symbol) = _getproperty(p, Val{s}())


# methods for obtaining the tied pair vector (vector of tuples)
function get_tied_pairs(p::FitParameters)
    ties = p.ties
    tie_groups = Vector{Union{Symbol,Nothing}}([!isnothing(tie) ? tie.group : nothing for tie in ties])
    tie_pairs = Vector{Tuple{Int,Int,Float64}}()
    for (i, tie) in enumerate(ties)
        if isnothing(tie) 
            continue
        end
        j = findfirst(tg -> tg == tie.group, tie_groups)
        if j == i 
            continue
        end
        push!(tie_pairs, (j, i, typeof(tie) <: RatioTie ? ties[i].ratio/ties[j].ratio : 1.0))
    end
    # Convert the paired tuples into indices for each tied parameter
    # this doesnt include the first index of each tied pair because these parameters are 
    # the only ones that we will actually allow to vary during the fitting procedure
    tie_indices = Vector{Int}(sort([tp[2] for tp in tie_pairs]))
    tie_pairs, tie_indices
end

# methods for locking a named parameter
lock!(p::FitParameters, ind::Int) = lock!(p[ind])
lock!(p::FitParameters, name::String) = lock!(p[name])
function lock!(p::FitParameters, names::AbstractVector{String})
    for param in p[names]
        lock!(param)
    end
end

# methods for unlocking a named parameter
unlock!(p::FitParameters, ind::Int) = unlock!(p[ind])
unlock!(p::FitParameters, name::String) = unlock!(p[name])
function unlock!(p::FitParameters, names::AbstractVector{String})
    for param in p[names]
        unlock!(param)
    end
end

# methods for updating parameter limits
set_plim!(p::FitParameters, ind::Int, limits::Tuple) = set_plim!(p[ind], limits)
set_plim!(p::FitParameters, name::String, limits::Tuple) = set_plim!(p[name], limits)
function set_plim!(p::FitParameters, names::AbstractVector{String}, limits::AbstractVector{<:Tuple})
    for (param, plimit) in zip(p[names], limits)
        set_plim!(param, plimit)
    end
end

# methods for updating parameter values
set_val!(p::Parameters, ind::Int, v::Number) = set_val!(p[ind], v)
set_val!(p::Parameters, name::String, v::Number) = set_val!(p[name], v)
function set_val!(p::Parameters, names::AbstractVector{String}, vs::AbstractVector{<:Number})
    for (param, v) in zip(p[names], vs)
        set_val!(param, v)
    end
end

function check_valid(p::Parameters)
    for param in p[:]
        check_valid(param)
    end
end

# check if everything is good
function check_valid(p::NonFitParameter)
    @assert isfinite(p)
end

# Count the total number of profiles in a FitFeatures object
function total_num_profiles(p::FitFeatures)
    n_prof = 0
    for profiles_i in p.profiles
        n_prof += length(profiles_i)
    end
    n_prof
end


# get all the fit parameters in a FitProfiles object
# (dont do this until the last step; the object returned 
#  will be a COPY of the fit parameters, so modifying it
#  wont affect the original)
function get_flattened_fit_parameters(p::FitProfiles)
    flat = FitParameters(String[], String[], Vector{Transformation}[], FitParameter[])
    for prof in p
        append!(flat, prof.fit_parameters)
    end
    flat
end
function get_flattened_nonfit_parameters(p::FitProfiles)
    flat = NonFitParameters(String[], String[], Vector{Transformation}[], NonFitParameter[])
    for prof in p
        append!(flat, prof.nonfit_parameters)
    end
    flat
end


# get all the fit parameters in a FitFeatures object
# (dont do this until the last step; the object returned 
#  will be a COPY of the fit parameters, so modifying it
#  wont affect the original)
function get_flattened_fit_parameters(p::FitFeatures)
    flat = FitParameters(String[], String[], Vector{Transformation}[], FitParameter[])
    for profiles_i in p.profiles
        append!(flat, get_flattened_fit_parameters(profiles_i))
    end
    flat
end
function get_flattened_nonfit_parameters(p::FitFeatures)
    flat = NonFitParameters(String[], String[], Vector{Transformation}[], NonFitParameter[])
    for (composite_i, profiles_i) in zip(p.composite, p.profiles)
        append!(flat, get_flattened_nonfit_parameters(profiles_i))
        append!(flat, composite_i)
    end
    flat
end

# get all the fit parameters in a ModelParameters object
function get_flattened_fit_parameters(p::ModelParameters)
    flat = FitParameters(String[], String[], Vector{Transformation}[], FitParameter[])
    append!(flat, p.continuum)
    append!(flat, get_flattened_fit_parameters(p.dust_features))
    append!(flat, get_flattened_fit_parameters(p.lines))
    flat
end
function get_flattened_nonfit_parameters(p::ModelParameters)
    flat = NonFitParameters(String[], String[], Vector{Transformation}[], NonFitParameter[])
    append!(flat, get_flattened_nonfit_parameters(p.dust_features))
    append!(flat, get_flattened_nonfit_parameters(p.lines))
    append!(flat, p.statistics)
    flat
end
function get_flattened_parameters(p::ModelParameters)
    flat = AllParameters(String[], String[], Vector{Transformation}[], Vector{Union{FitParameter,NonFitParameter}}())
    append!(flat, get_flattened_fit_parameters(p))
    append!(flat, get_flattened_nonfit_parameters(p))
    flat
end

# the range parameter mainly determines what to do about extinction curves
# since calzetti and CCM are not defined past ~2-3 um
function get_λrange(λlim::Tuple{QLength,QLength})
    if λlim[1] < 2.2u"μm" && λlim[2] > 2.2u"μm"
        UVOptIR
    elseif λlim[1] > 2.2u"μm"
        Infrared
    else
        UVOptical
    end
end

# get masks for each part of the spectrum separated by the gaps
function get_gap_masks(λ::Vector{T}, gaps::Vector{Tuple{T,T}}) where {T<:Number}
    region_masks = BitVector[]
    if length(gaps) == 0
        push!(region_masks, trues(length(λ)))
    else
        for i in 1:(length(gaps)+1)
            if i == 1
                push!(region_masks, λ .< gaps[i][1])
            elseif i == (length(gaps)+1)
                push!(region_masks, gaps[i-1][2] .< λ)
            else
                push!(region_masks, gaps[i-1][2] .≤ λ .≤ gaps[i][1])
            end
        end
    end
    region_masks
end


# latex string representations of units 
latex(u) = latexify(u)
latex(::Unitful.FreeUnits{(), NoDims, nothing}) = L""
latex(::typeof(unit(QFlux))) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}$"
latex(::typeof(unit(QPerFreq))) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{Hz}^{-1}\,\mathrm{sr}^{-1}$"
latex(::typeof(unit(QPerAng))) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}\,\mathrm{sr}^{-1}$"
latex(::typeof(unit(QPerum))) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\mu{}m}^{-1}\,\mathrm{sr}^{-1}$"
# some more unconventional units too
latex(::typeof(u"erg/s/cm^2/Hz/sr*μm")) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{Hz}^{-1}\,\mathrm{sr}^{-1}\,\mathrm{\mu{}m}$"
latex(::typeof(u"erg/s/cm^2/sr")) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{sr}^{-1}$"
