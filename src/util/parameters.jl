#=
This file, like utils, is not intended to be directly accessed by the 
user when fitting IFU cubes. Rather, it contains various Parameter structures
that are helpful for containing certain combinations of model parameters and
related quantities.
=#

# aliases for convenience
const QLength = Quantity{<:Real, u"𝐋"}

const Qum = typeof(1.0u"μm")
const QAng = typeof(1.0u"angstrom")
const QWave = Union{Qum,QAng}

const QPerFreq = typeof(1.0u"erg/s/cm^2/Hz/sr")
const QPerum = typeof(1.0u"erg/s/cm^2/μm/sr")
const QPerAng = typeof(1.0u"erg/s/cm^2/angstrom/sr")
const QPerWave = Union{QPerum,QPerAng}
const QSIntensity = Union{QPerFreq,QPerum,QPerAng}
const QIntensity = typeof(1.0u"erg/s/cm^2/sr")
const QFlux = typeof(1.0u"erg/s/cm^2")


# Create some type categories
abstract type Parameter end
abstract type Parameters end
abstract type Tie end
abstract type Config end

@enum WavelengthRange begin
    UVOptical
    Infrared
    UVOptIR
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
struct NonFitParameter{T<:Number} <: Parameter end

# A collection of FitParameter with names 
# Think of this like a dictionary but with a defined order (couldve just used an OrderedDict but I already wrote all of this so...)
struct FitParameters{T<:Number} <: Parameters
    names::Vector{String}                  # names of all the parameters
    _parameters::Vector{FitParameter{T}}   # the internal storage of the parameter objects

    function FitParameters(names::Vector{String}, parameters::Vector{FitParameter{T}}) where {T<:Number}
        @assert names == unique(names) "repeat names are not allowed!"
        new{T}(names, parameters)
    end
end

# The non-fit equivalent of FitParameters
struct NonFitParameters{T<:Number} <: Parameters
    names::Vector{String}
    _parameters::Vector{NonFitParameter{T}}

    function NonFitParameters(names::Vector{String}, parameters::Vector{NonFitParameter{T}}) where {T<:Number}
        @assert names == unique(names) "repeat names are not allowed!"
        new{T}(names, parameters)
    end
end


struct FitProfile{T<:Number,S<:Union{Symbol,String}}
    profile::S
    fit_parameters::FitParameters
    nonfit_parameters::NonFitParameters
end

FitProfiles = Vector{FitProfile}


struct FitFeatures{S}
    names::Vector{S}                                             # PAH or emission line names
    latex::Vector{String}                                        # latex-formatted labels
    λ₀::Vector{<:Union{typeof(1.0u"μm"),typeof(1.0u"angstrom")}} # central wavelengths of the features
    profiles::Vector{FitProfiles}  # first index = each feature, second index = each profile in the feature
    config::Config
end

struct NoConfig <: Config end

struct LineConfig <: Config
    annotate::BitVector                  # whether or not to annotate
    sort_order::Vector{Int}              # defines the sorting sense for the components
    combined::Vector{Vector{Symbol}}     # combined line maps
    # Relative config parameters
    rel_amp::Bool                        #  ...amp relative
    rel_voff::Bool                       #  ...voff relative
    rel_fwhm::Bool                       #  ...fwhm relative
end

# These will define how CubeFitter objects behave
struct SpectralRegion{T<:Union{typeof(1.0u"μm"),typeof(1.0u"angstrom")}}
    λlim::Tuple{T,T}
    mask::Vector{Tuple{T,T}}
    n_channels::Int
    wavelength_range::WavelengthRange
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

# check if everything is good
function check_valid(p::FitParameter)
    @assert isfinite(p.value)
    @assert typeof(p.value) == typeof(p.limits[1]) == typeof(p.limits[2])
    @assert p.limits[1] < p.limits[2]
    @assert p.limits[1] ≤ p.value ≤ p.limits[2]
end



"""
    show([io,] p)

Show the parameter object as a nicely formatted string
"""
function Base.show(io::IO, p::FitParameter)
    Base.show(io, "Parameter: value = $(p.value) | locked = $(p.locked) | limits = $(p.limits)")
end


"""
    parameter_from_dict(dict)

A constructor function for Parameter structs given a Dictionary
"""
function parameter_from_dict(dict::Dict)

    # Unpack the dictionary into fields of the Parameter
    value = dict["val"]
    locked = dict["locked"]
    # plim: absolute upper/lower limits
    lims = (dict["plim"]...,)

    FitParameter(value, locked, lims)
end 


"""
    parameter_from_dict_wave(dict)

A constructor function for Parameter structs given a Dictionary,
using deltas on upper/lower limits, i.e. if val = 5 and plim = [-0.1, 0.1],
then the true limits will be 5 .+ [-0.1, 0.1] = [4.9, 5.1]
"""
function parameter_from_dict_wave(dict::Dict)

    # Unpack the dictionary into fields of the Parameter
    value = dict["val"]
    locked = dict["locked"]
    # plim: absolute upper/lower limits
    lims = (dict["plim"].+value...,)

    FitParameter(value, locked, lims)
end


"""
    parameter_from_dict_fwhm(dict)

A constructor function for Parameter structs given a Dictionary,
using fractional values on upper/lower limits, i.e. if val = 5 and
plim = [0.5, 2], then the true limits will be 5 .* [0.5, 2] = [2.5, 10]
"""
function parameter_from_dict_fwhm(dict::Dict)

    # Unpack the dictionary into fields of the Parameter
    value = dict["val"]
    locked = dict["locked"]
    # plim: absolute upper/lower limits
    lims = (dict["plim"].*value...,)

    FitParameter(value, locked, lims)
end


# methods for obtaining a named parameter
Base.getindex(p::Parameters, ind) = p._parameters[ind]
function Base.getindex(p::Parameters, name::String)
    ind = findfirst(p.names .== name)
    p._parameters[ind]
end
function Base.getindex(p::Parameters, names::AbstractVector{String})
    inds = [findfirst(p.names .== name) for name in names]
    p._parameters[inds]
end

# methods for adding to a parameter list
function Base.push!(p::Parameters, name::String, new::Parameter)
    @assert !(name in p.names) "$name already has an entry in $(typeof(p)) object!"
    push!(p.names, name)
    push!(p._parameters, new)
end
function Base.append!(p::Parameters, new::Parameters) 
    for new_name in new.names
        @assert !(new_name in p.names) "$new_name has an entry in both $(typeof(p)) objects!"
    end
    append!(p.names, new.names)
    append!(p._parameters, new._parameters)
end

# methods for deleting a parameter from the list
function Base.deleteat!(p::Parameters, ind::Int)
    deleteat!(p.names, ind)
    deleteat!(p._parameters, ind)
end
function Base.deleteat!(p::Parameters, name::String)
    ind = findfirst(p.names .== name)
    deleteat!(p, ind)
end

# methods for obtaining the vector of parameter limits
get_plims(p::FitParameters, ind::Int) = p[ind].limits   
get_plims(p::FitParameters, name::String) = p[name].limits   
function Base.getproperty(p::FitParameters, ::Val{:limits}) 
    plims = Vector{Tuple}(undef, length(p.names))
    for (i, param) in enumerate(p._parameters)
        plims[i] = param.limits
    end
    plims
end

# methods for obtaining the lock vector
get_lock(p::FitParameters, ind::Int) = p[ind].locked
get_lock(p::FitParameters, name::String) = p[name].locked
function Base.getproperty(p::FitParameters, ::Val{:locked}) 
    locks = BitVector(undef, length(p.names))
    for (i, param) in enumerate(p._parameters)
        locks[i] = param.locked
    end
    locks
end

# methods for obtaining the value vector
get_val(p::FitParameters, ind::Int) = p[ind].value
get_val(p::FitParameters, name::String) = p[name].value
function Base.getproperty(p::FitParameters{T}, ::Val{:values}) where {T}
    vals = Vector{T}(undef, length(p.names))
    for (i, param) in enumerate(p._parameters)
        vals[i] = param.value
    end
    vals
end

# methods for obtaining the tied vector (vector of symbols)
get_tie(p::FitParameters, ind::Int) = p[ind].tie
get_tie(p::FitParameters, name::String) = p[name].tie
function Base.getproperty(p::FitParameters, ::Val{:ties})
    ties = Vector{Union{Tie,Nothing}}(undef, length(p.names))
    for (i, param) in enumerate(p._parameters)
        ties[i] = param.tie
    end
    ties
end

# methods for obtaining the tied pair vector (vector of tuples)
function get_tied_pairs(p::FitParameters)
    ties = p.ties
    tie_groups = Vector{Union{Symbol,Nothing}}([tie.group for tie in ties])
    tie_pairs = Vector{Tuple{Int,Int,Float64}}()
    for (i, tie) in enumerate(ties)
        if !isnothing(tie.group)
            j = findfirst(tie_groups .== tie.group)
            if j == i 
                continue
            end
            push!(tie_pairs, (j, i, typeof(tie) <: RatioTie ? ties[i].ratio/ties[j].ratio : 1.0))
        end
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


# get all the fit parameters in a FitProfiles object
# (dont do this until the last step; the object returned 
#  will be a COPY of the fit parameters, so modifying it
#  wont affect the original)
function get_flattened_fit_parameters(p::FitProfiles)
    flat = FitParameters(String[], FitParameter[])
    for prof in p
        append!(flat, prof.fit_parameters)
    end
    flat
end
function get_flattened_nonfit_parameters(p::FitProfiles)
    flat = NonFitParameters(String[], NonFitParameter[])
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
    flat = FitParameters(String[], FitParameter[])
    for profiles_i in p.profiles
        append!(flat, get_flattened_fit_parameters(profiles_i))
    end
    flat
end
function get_flattened_nonfit_parameters(p::FitFeatures)
    flat = NonFitParameters(String[], FitParameter[])
    for profiles_i in p.profiles
        append!(flat, get_flattened_nonfit_parameters(profiles_i))
    end
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


# latex string representations of units 
latex(u) = latexify(u)
latex(::Unitful.FreeUnits{(), NoDims, nothing}) = L""
latex(::typeof(unit(QFlux))) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}"
latex(::typeof(unit(QPerFreq))) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{Hz}^{-1}\,\mathrm{sr}^{-1}$"
latex(::typeof(unit(QPerAng))) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}\,\mathrm{sr}^{-1}$"
latex(::typeof(unit(QPerum))) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\mu{}m}^{-1}\,\mathrm{sr}^{-1}$"
# some more unconventional units too
latex(::typeof(u"erg/s/cm^2/Hz/sr*μm")) = L"$\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{Hz}^{-1}\,\mathrm{sr}^{-1}\,\mathrm{\mu{}m}$"


# """
#     TransitionLines(names, latex, annotate, λ₀, profiles, tied_amp, tied_voff, tied_fwhm,
#         acomp_amp, voff, fwhm, h3, h4, η, combined)

# A container for ancillary information and modeling parameters relating to transition lines.
# """
# struct TransitionLines

#     # 1st axis: labels each transition line
#     names::Vector{Symbol}
#     latex::Vector{String}
#     annotate::BitVector
#     λ₀::Vector{AbstractFloat}
#     sort_order::Vector{Int}
    
#     # 1st axis: labels each transition line
#     # 2nd axis: labels the components of each line
#     profiles::Matrix{Union{Symbol,Nothing}}
#     tied_amp::Matrix{Union{Symbol,Nothing}}
#     tied_voff::Matrix{Union{Symbol,Nothing}}
#     tied_fwhm::Matrix{Union{Symbol,Nothing}}

#     # Model Parameters
#     acomp_amp::Matrix{Union{Parameter,Nothing}}
#     voff::Matrix{Union{Parameter,Nothing}}
#     fwhm::Matrix{Union{Parameter,Nothing}}
#     h3::Matrix{Union{Parameter,Nothing}}
#     h4::Matrix{Union{Parameter,Nothing}}
#     η::Matrix{Union{Parameter,Nothing}}

#     # Combined lines (for map purposes)
#     combined::Vector{Vector{Symbol}}

#     # Relative flags for additional components
#     rel_amp::Bool
#     rel_voff::Bool
#     rel_fwhm::Bool

# end


# """
#     make_single_transitionline_object(lines, i)

# Takes a TransitionLine object and an index i and creates a new TransitionLine
# object which contains only the single line at index i in the original object.
# """
# function make_single_transitionline_object(lines::TransitionLines, i::Integer)
#     n_comps = size(lines.profiles, 2)

#     TransitionLines(
#         [lines.names[i]], 
#         [lines.latex[i]], 
#         [lines.annotate[i]],
#         [lines.λ₀[i]], 
#         [lines.sort_order[i]],
#         reshape(lines.profiles[i, :], (1, n_comps)), 
#         reshape(lines.tied_amp[i, :], (1, n_comps)),
#         reshape(lines.tied_voff[i, :], (1, n_comps)),
#         reshape(lines.tied_fwhm[i, :], (1, n_comps)), 
#         reshape(lines.acomp_amp[i, :], (1, n_comps-1)), 
#         reshape(lines.voff[i, :], (1, n_comps)), 
#         reshape(lines.fwhm[i, :], (1, n_comps)), 
#         reshape(lines.h3[i, :], (1, n_comps)),
#         reshape(lines.h4[i, :], (1, n_comps)), 
#         reshape(lines.η[i, :], (1, n_comps)),
#         lines.combined, 
#         lines.rel_amp,
#         lines.rel_voff,
#         lines.rel_fwhm
#     )
# end


# """
#     TiedKinematics(key_amp, amp, key_voff, voff, key_fwhm, fwhm)

# A container for tied amplitude and kinematic parameter information.
# """
# struct TiedKinematics

#     # Vectors of vectors, rather than Matrices, since the size of each element may be inhomogeneous 
#     key_amp::Vector{Vector{Symbol}}
#     amp::Vector{Vector{Dict{Symbol, <:Real}}}
#     key_voff::Vector{Vector{Symbol}}
#     voff::Vector{Vector{Parameter}}
#     key_fwhm::Vector{Vector{Symbol}}
#     fwhm::Vector{Vector{Parameter}}

# end