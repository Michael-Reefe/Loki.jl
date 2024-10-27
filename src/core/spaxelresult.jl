
# A container for holding best-fit results
mutable struct SpaxelFitResult
    pnames::Vector{String}
    popt::Vector{Quantity{Float64}}
    perr::Matrix{Quantity{Float64}}
    bounds::Matrix{Quantity{Float64}}
    plock::BitVector
    ptie::Vector{Union{Tie,Nothing}}

    function SpaxelFitResult(pnames, popt, perr, bounds, plock, ptie)
        @assert length(pnames) == length(popt) == size(perr, 1) == size(bounds, 1) == length(plock) == length(ptie) 
        @assert size(perr, 2) == size(bounds, 2) == 2
        new(pnames, popt, perr, bounds, plock, ptie)
    end
end


# A container for holding stellar population results
struct StellarResult
    norm::Quantity
    mtot::typeof(1.0u"Msun")
    mfracs::Matrix{<:Real}
    lfracs::Matrix{<:Real}
    weights::Matrix{<:Real}
    ages::Vector{typeof(1.0u"Gyr")}
    logzs::Vector{<:Real}
end


# combine multiple SpaxelFitResult objects together
function combine!(s1::SpaxelFitResult, s2::SpaxelFitResult)
    s1.pnames = cat(s1.pnames, s2.pnames, dims=1)
    s1.popt = cat(s1.popt, s2.popt, dims=1)
    s1.perr = cat(s1.perr, s2.perr, dims=1)
    s1.bounds = cat(s1.bounds, s2.bounds, dims=1)
    s1.plock = cat(s1.plock, s2.plock, dims=1)
    s1.ptie = cat(s1.ptie, s2.ptie, dims=1)
    s1
end

# Make a copy of a SpaxelFitResult object
function Base.copy(s::SpaxelFitResult)
    SpaxelFitResult(copy(s.pnames), copy(s.popt), copy(s.perr), copy(s.bounds), copy(s.plock), copy(s.ptie))
end


# Function for nicely printing out results
function pretty_print_results(result::SpaxelFitResult; round::Bool=false)

    # prettify locked and tied vectors
    locked = ifelse.(result.plock, "yes", "")
    tie_groups = Vector{String}([!isnothing(tie) ? string(tie.group) : "" for tie in result.ptie])

    # make things have appropriate number of sig figs based on the errors
    _popt = ustrip.(result.popt); _perr_l = ustrip.(result.perr[:,1]); _perr_u = ustrip.(result.perr[:,2])
    _lb = ustrip.(result.bounds[:,1]); _ub = ustrip.(result.bounds[:,2])
    _unit = string.(unit.(result.popt))
    _unit = [_ui == "NoUnits" ? "" : _ui for _ui in _unit]

    round_to_digits(x, y) = y ≤ 0. || !isfinite(y) ? x : round(x, digits=-Int(floor(log10(y))))
    if round
        _perr_l = round.(_perr_l, sigdigits=1)       # rounds the errors to 1 significant figure 
        _perr_u = round.(_perr_u, sigdigits=1)       # rounds the errors to 1 significant figure 
        _popt_temp = round_to_digits.(_popt, _perr_l)     # rounds the values to match the number of significant figures that the errors have
        # dont overwrite if rounding would cause it to go to 0 (can happen if errors blow up b/w degenerate parameters)
        _popt .= ifelse.(iszero.(_popt_temp), _popt, _popt_temp)
        # safeguard - MAXIMUM number of sig figs to prevent really long and ugly floats from getting in there
        mask = (_perr_l .≤ 0.) .| .~isfinite.(_perr_l) .| iszero.(_popt_temp)
        _popt[mask] .= round.(_popt[mask], sigdigits=6)

        # repeat for lower/upper bounds
        _lb_temp = round_to_digits.(_lb, _perr_l)
        _lb .= ifelse.(iszero.(_lb_temp), _lb, _lb_temp)
        mask = (_perr_l .≤ 0.) .| .~isfinite.(_perr_l) .| iszero.(_lb_temp)
        _lb[mask] .= round.(_lb[mask], sigdigits=6)
        _ub_temp = round_to_digits.(_ub, _perr_l)
        _ub .= ifelse.(iszero.(_ub_temp), _ub, _ub_temp)
        mask = (_perr_l .≤ 0.) .| .~isfinite.(_perr_l) .| iszero.(_ub_temp)
        _ub[mask] .= round.(_ub[mask], sigdigits=6)
    end

    # dont print bounds if they are infinite
    _lb = ifelse.(.~isfinite.(_lb), "", string.(_lb))
    _ub = ifelse.(.~isfinite.(_ub), "", string.(_ub))

    data = DataFrame(name=result.pnames, best=_popt, error_lower=_perr_l, error_upper=_perr_u, 
        bound_lower=_lb, bound_upper=_ub, unit=_unit, locked=locked, tied=tie_groups)
    textwidths = [maximum(textwidth.(string.([data[:, i]; names(data)[i]]))) for i in axes(data, 2)]
    msg = ""
    for (i, header) ∈ enumerate(names(data))
        msg *= rpad(header, textwidths[i]) * "\t"
    end
    msg *= "\n"
    for i ∈ axes(data, 1)
        for j ∈ axes(data, 2)
            msg *= rpad(data[i,j], textwidths[j]) * "\t"
        end
        msg *= "\n"
    end
    @debug msg

    msg
end
