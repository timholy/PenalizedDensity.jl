# The window rule: a local smoothing scale from the largest window around each point in which
# the sample shows no slope or curvature.

"""
    WindowRule(; gamma=0.75, k=0.11, z=4, nmin=30)

Settings of the window candidate in [`select_kappa_adaptive`](@ref)`(x; window)`, whose
default is `WindowRule()`.

The candidate is the scale `κ(x) = c·(1/ĥ(x))^gamma` (a [`WindowScale`](@ref)), where `ĥ(x)` is
the smallest half-width at which a window centered on `x` holds at least `nmin` observations
and its first or second moment departs from that of a uniform spread by at least `z` standard
deviations; `c` is chosen by the same cross-validation score as the other candidates. The
window candidate is chosen only if its score plus the penalty `k·gamma/√N`, for `N`
observations, is below the score of the scale the power family would return. The penalty
offsets the optimism of a held-out score whose scale was estimated from the held-out point
itself; see the extended help of [`select_kappa_adaptive`](@ref).

`gamma` and `z` must be positive, `k` nonnegative, and `nmin` at least 2.
"""
struct WindowRule
    gamma::Float64   # exponent of 1/ĥ
    k::Float64       # penalty coefficient: the window candidate's score is charged k·gamma/√N
    z::Float64       # threshold of the window statistics, in null standard deviations
    nmin::Int        # fewest observations a window must hold before it is tested

    function WindowRule(gamma, k, z, nmin)
        isfinite(gamma) && gamma > 0 ||
            throw(ArgumentError("the window exponent gamma must be positive and finite, got $gamma"))
        k >= 0 || throw(ArgumentError("the penalty coefficient k must be nonnegative, got $k"))
        isfinite(z) && z > 0 ||
            throw(ArgumentError("the window threshold z must be positive and finite, got $z"))
        nmin >= 2 || throw(ArgumentError("nmin must be at least 2, got $nmin"))
        return new(gamma, k, z, nmin)
    end
end

WindowRule(; gamma::Real=0.75, k::Real=0.11, z::Real=4, nmin::Integer=30) =
    WindowRule(gamma, k, z, nmin)

"""
    WindowScale(c, γ, x, logb)

A spatially varying smoothing scale

    κ(t) = c · exp(γ · s(t)),

where `s` interpolates the values `logb` linearly on the sorted grid `x` and is held constant
beyond it. [`select_kappa_adaptive`](@ref) constructs one with `s = ln(1/ĥ)` from the window
rule (see [`WindowRule`](@ref)), shifted to mean zero over the sample, so that `c` is the
scale at a point of typical window size. Like [`AdaptiveScale`](@ref), the result is callable
and is passed straight to [`DensityEstimate`](@ref) as the smoothing scale.

The scale is floored at `1e-6 c`.
"""
struct WindowScale{T<:AbstractFloat}
    c::T
    γ::T
    x::Vector{T}      # sorted grid
    logb::Vector{T}   # s on the grid
    κmin::T           # floor

    function WindowScale{T}(c, γ, x, logb, κmin) where {T<:AbstractFloat}
        axes(x) == axes(logb) ||
            throw(DimensionMismatch("the grid and its values must have the same axes, got $(axes(x)) and $(axes(logb))"))
        isempty(x) && throw(ArgumentError("the grid is empty"))
        issorted(x) || throw(ArgumentError("the grid must be sorted"))
        all(isfinite, logb) || throw(ArgumentError("the shape values must be finite"))
        return new{T}(c, γ, collect(T, x), collect(T, logb), κmin)
    end
end

function WindowScale(c::Real, γ::Real, x::AbstractVector{<:Real}, logb::AbstractVector{<:Real})
    c > 0 || throw(ArgumentError("the scale c must be positive, got $c"))
    T = float(promote_type(typeof(c), typeof(γ), eltype(x), eltype(logb)))
    return WindowScale{T}(c, γ, x, logb, T(_KAPPA_FLOOR) * T(c))
end

function _window_shape_at(s::WindowScale, t::Real)
    x, v = s.x, s.logb
    t <= first(x) && return first(v)
    t >= last(x) && return last(v)
    k = searchsortedlast(x, t)
    w = (t - x[k]) / (x[k+1] - x[k])
    return (1 - w) * v[k] + w * v[k+1]
end

(s::WindowScale)(t::Real) = max(s.c * exp(s.γ * _window_shape_at(s, t)), s.κmin)

Base.show(io::IO, s::WindowScale) =
    print(io, "WindowScale(c=", s.c, ", γ=", s.γ, ") on a grid of ", length(s.x), " points")

# The sorted sample with prefix sums of its deviations from the sample mean and of their
# squares, so that the count and moments of any window cost two binary searches.
struct _WindowMoments{T}
    y::Vector{T}
    S1::Vector{T}
    S2::Vector{T}
    μ::T
end

function _WindowMoments(y::Vector{T}) where {T}
    μ = mean(y)
    return _WindowMoments(y, [zero(T); cumsum(y .- μ)], [zero(T); cumsum((y .- μ) .^ 2)], μ)
end

# Count, and the sums of (t - x) and (t - x)², over the sample points t in [lo, hi].
function _window_sums(W::_WindowMoments{T}, lo, hi, x) where {T}
    i = searchsortedfirst(W.y, lo)
    j = searchsortedlast(W.y, hi)
    N = j - i + 1
    N <= 0 && return (0, zero(T), zero(T))
    xc = x - W.μ
    s1 = W.S1[j+1] - W.S1[i]
    s2 = W.S2[j+1] - W.S2[i]
    return (N, s1 - N * xc, s2 - 2xc * s1 + N * xc^2)
end

# With u = (t - x)/h over the N sample points of the window |u| ≤ 1: the slope statistic
# |Σu| / √(N/3) and the curvature statistic |Σu² - N/3| / √(4N/45), each the moment's
# departure from its value for N points spread uniformly over the window, in units of its
# standard deviation under that spread.
function _window_statistics(W::_WindowMoments{T}, x::T, h::T) where {T}
    N, s1, s2 = _window_sums(W, x - h, x + h, x)
    N <= 0 && return (; N = 0, slope = zero(T), curvature = zero(T))
    return (; N, slope = abs(s1 / h) / sqrt(T(N) / 3),
            curvature = abs(s2 / h^2 - T(N) / 3) / sqrt(4 * T(N) / 45))
end

const _WINDOW_PER_DECADE = 24   # steps per decade of the half-width scan
const _WINDOW_GRID = 200        # grid points spaced evenly over the sample's range (as many
                                # again, plus one, are placed at sample quantiles)
const _WINDOW_HMAX = 10         # the scan stops at this multiple of the sample's range

# The smallest half-width on a logarithmic scan, starting at the distance to the `nmin`-th
# nearest observation, whose window holds at least `nmin` observations and has a slope or
# curvature statistic of at least `z`; `_WINDOW_HMAX` times the sample's range if none does.
function _window_halfwidth(W::_WindowMoments{T}, x::T, z::T, nmin::Int) where {T}
    y = W.y
    hmax = _WINDOW_HMAX * (last(y) - first(y))
    k = searchsortedfirst(y, x)
    near = sort!([abs(y[j] - x) for j in max(firstindex(y), k - nmin):min(lastindex(y), k + nmin)])
    h = max(near[min(nmin, length(near))], eps(abs(x) + 1))
    r = T(10)^(one(T) / _WINDOW_PER_DECADE)
    while h < hmax
        s = _window_statistics(W, x, h)
        s.N >= nmin && max(s.slope, s.curvature) >= z && return h
        h *= r
    end
    return hmax
end

# The window rule's shape on the sorted sample `xs`: `ln(1/ĥ)` on a grid of evenly spaced
# points over the sample's range together with sample quantiles, shifted to mean zero over the
# sample. Returns the grid and the values.
function _window_shape(xs::Vector{T}, z::Real, nmin::Integer) where {T}
    first(xs) < last(xs) ||
        throw(ArgumentError("the window rule needs at least two distinct observations"))
    W = _WindowMoments(xs)
    grid = sort!(unique!([collect(range(first(xs), last(xs); length=_WINDOW_GRID));
                          [quantile(xs, p; sorted=true) for p in range(0, 1; length=_WINDOW_GRID + 1)]]))
    logb = T[-log(_window_halfwidth(W, t, T(z), Int(nmin))) for t in grid]
    s = WindowScale(one(T), one(T), grid, logb)
    m = mean(_window_shape_at(s, t) for t in xs)
    return grid, logb .- m
end

# Locations for the window rule from tallied interval observations (`_tally_intervals`): the
# `counts[j]` observations in interval j sit at its quantile midpoints, evenly spread over it.
# Within a window this gives the sums of u and u² expected for observations spread uniformly
# over their intervals, to O(1/counts[j]), while the counts between intervals keep their
# sampling noise. Exact points (lu[j] == uu[j]) stay where they are; intervals with an infinite
# bound have no location and are left out.
function _spread_intervals(lu::AbstractVector{T}, uu::AbstractVector{T}, counts::AbstractVector) where {T}
    xs = T[]
    for j in eachindex(lu, uu, counts)
        a, b = lu[j], uu[j]
        isfinite(a) && isfinite(b) || continue
        m = Int(counts[j])
        a == b ? append!(xs, Iterators.repeated(a, m)) :
                 append!(xs, (a + (b - a) * ((k - T(1) / 2) / m) for k in 1:m))
    end
    return sort!(xs)
end

# The window rule's choice between the scale `κp` the power family selected (with score `sp`)
# and the window candidate, for the sorted sample `xs`; `score(κfun)` is the KLCV score of a
# scale function and `κ0` the pilot scale, which centers the search for c. An unresolvable
# window candidate never wins; an unresolvable `κp` loses to any resolvable window candidate.
# `N` is the number of observations the scores are taken over.
function _choose_window(score, xs::Vector{T}, κ0::T, κp, sp::T, rule::WindowRule;
                        N::Integer=length(xs)) where {T}
    grid, logb = _window_shape(xs, rule.z, rule.nmin)
    γ = T(rule.gamma)
    res = _select_c_scored(c -> score(WindowScale(c, γ, grid, logb)), κ0; skip_unresolved=true)
    res === nothing && return κp
    c, sw = res
    isfinite(sp) || return WindowScale(c, γ, grid, logb)
    return sw + rule.k * γ / sqrt(T(N)) < sp ? WindowScale(c, γ, grid, logb) : κp
end
