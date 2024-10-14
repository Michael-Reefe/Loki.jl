
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


function pretty_print_results(result::SpaxelFitResult)

    # prettify locked and tied vectors
    locked = ifelse.(result.plock, "yes", "")
    tie_groups = Vector{String}([!isnothing(tie) ? string(tie.group) : "" for tie in result.ptie])

    # make things have appropriate number of sig figs based on the errors
    _popt = ustrip.(result.popt); _perr = ustrip.(result.perr); _lb = ustrip.(result.bounds[:,1]); _ub = ustrip.(result.bounds[:,2])
    _unit = string.(unit.(result.popt))
    _unit = [_ui == "NoUnits" ? "" : _ui for _ui in _unit]

    round_to_digits(x, y) = round(x, digits=-Int(floor(log10(y))))
    _perr = round_to_digits.(_perr, _perr)   # rounds the errors to 1 digit
    _popt = round_to_digits.(_popt, _perr)   # rounds the values to match the number of significant figures that the errors have
    _lb = round_to_digits.(_lb, _perr)
    _ub = round_to_digits.(_ub, _perr)

    data = DataFrame(name=result.pnames, best=_popt, error_lower=_perr[:,1], error_upper=_perr[:,2], 
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
