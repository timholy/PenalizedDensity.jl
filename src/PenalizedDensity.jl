module PenalizedDensity

using DensityInterface: DensityInterface, HasDensity
using LinearAlgebra: LinearAlgebra, I, SymTridiagonal, Symmetric, ZeroPivotException, cholesky, dot,
                     issuccess, ldiv!, ldlt!, mul!
using LogExpFunctions: logabssinh, logaddexp, logcosh
using QuadGK: gauss, quadgk
using SpecialFunctions: erfc, erfcinv, erfcx
using Random: Random, AbstractRNG, SamplerTrivial
using Statistics: Statistics, mean, median, quantile, std, var
using StatsAPI: StatsAPI, nobs, pvalue

export AbstractDensityEstimate, DensityEstimate, IntervalDensityEstimate, amplitude, action, select_kappa, select_support
export Adaptive, AdaptiveScale, WindowScale, WindowRule
export chisq, ChisqReference, chisq_pdf, chisq_ccdf, pvalue
export entropy, negentropy
export logdensity, logdensity_eval_gradient, logdensity_node_gradient
export cdf, quantile, median, mean, var, std, nobs
export gaussianize, ungaussianize, gaussianize_logjacobian
# `public` is a syntax error before Julia 1.11.
if VERSION >= v"1.11"
    eval(Expr(:public, :ConvergenceError, :SolveStats, :merge_samples, :roughness_operator,
                       :select_kappa_lscv, :select_kappa_sensitivity, :klcv, :lscv,
                       :action_sensitivity))
end

"""
    AbstractDensityEstimate

Supertype of the fitted densities: [`DensityEstimate`](@ref) (point observations) and
[`IntervalDensityEstimate`](@ref) (interval observations, optionally mixed with points). Every
subtype supports:

- evaluation: `d(x)` (the density), [`amplitude`](@ref), [`logdensity`](@ref),
  [`logdensity_eval_gradient`](@ref), [`cdf`](@ref) and [`quantile`](@ref Statistics.quantile).
  Evaluating any of these at `NaN` throws a `DomainError`;
- the support queries [`minimum`](@ref Base.minimum), `maximum` and `extrema`;
- sampling by inversion through `rand`: `rand(d)`, `rand(rng, d, dims...)` and
  `rand!(rng, A, d)` draw from the fitted density;
- the fit's moments [`mean`](@ref Statistics.mean), [`var`](@ref Statistics.var) and
  [`std`](@ref Statistics.std), its [`median`](@ref Statistics.median), and the observation
  count [`nobs`](@ref);
- [`entropy`](@ref), [`negentropy`](@ref) (in-sample and held-out) and [`action`](@ref);
- the probit transform [`gaussianize`](@ref), its inverse [`ungaussianize`](@ref), and
  [`gaussianize_logjacobian`](@ref).

The pointwise functions take one point at a time, and a fit acts as a scalar in broadcasting:
`cdf.(d, xs)` evaluates the CDF at each element of `xs`. The node cumulatives the mass functions
need are assembled when the fit is constructed, so evaluating at a point costs a binary search
over the nodes plus a constant amount of work.

The goodness-of-fit statistic ([`chisq`](@ref), [`ChisqReference`](@ref)) and
[`logdensity_node_gradient`](@ref) are defined for [`DensityEstimate`](@ref) only.

Every subtype has the fields `d.x` (the sorted, distinct nodes), `d.ψ` (the normalized
amplitude at the nodes) and `d.κ` (the smoothing scale: one number, or a vector with one rate
per cell between consecutive nodes). Other fields are specific to each subtype.

# Interoperability

Every subtype implements [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl)
as an object that *has* a density (`DensityKind(d) == HasDensity()`, as for a Distributions.jl
distribution): `logdensityof(d, x) == logdensity(d, x)` and `densityof(d, x) == d(x)` at a real
`x`.

When Distributions.jl is loaded, `Distributions.cdf`, `Distributions.pdf` and
`Distributions.logpdf` accept a fit and agree with [`cdf`](@ref), `d(x)` and
[`logdensity`](@ref); `quantile` is already shared with Distributions through `Statistics`.
When StatsBase.jl is loaded, `StatsBase.entropy(d)` agrees with [`entropy`](@ref)`(d)`. The
held-out `entropy(d, xeval)` has no StatsBase counterpart, because StatsBase's two-argument
`entropy(p, b)` takes a logarithm base. With both packages loaded, the unqualified names `cdf`
and `entropy` are ambiguous, and either qualified name gives the same result.
"""
abstract type AbstractDensityEstimate end

"""
    ConvergenceError(routine, msg)

Thrown when an iterative routine (a fit's Newton solve, a quantile or probit inversion, or a
selector's search) stops without converging. `routine` names it and `msg` describes the state
at which it stopped. The result such a routine would return is unreliable, so it is not
returned.
"""
struct ConvergenceError <: Exception
    routine::String
    msg::String
end
Base.showerror(io::IO, e::ConvergenceError) = print(io, "ConvergenceError in ", e.routine, ": ", e.msg)

# The evaluators order `x` against the nodes, and NaN has no place in that order.
_check_not_nan(x) = isnan(x) ? throw(DomainError(x, "the evaluation point must not be NaN")) : nothing
# Point observations must be finite in the working type `T`: only interval bounds may be
# infinite (censoring).
function _check_finite_observations(x, ::Type{T}) where {T}
    i = findfirst(xi -> !isfinite(T(xi)), x)
    i === nothing && return nothing
    xi = x[i]
    throw(ArgumentError(isfinite(xi) ? "observation $xi overflows $T" : "observations must be finite, got $xi"))
end

_check_rtol(rtol) = rtol >= 0 ? nothing : throw(ArgumentError("rtol must be nonnegative, got $rtol"))

function _check_support_bounds(support)
    a, b = support
    a < b || throw(DomainError((a, b), "support must satisfy a < b, got support=($a, $b)"))
    return a, b
end

"""
    DensityEstimate(x, κ; support=(-Inf, Inf), rtol=cbrt(eps(float(eltype(x)))))
    DensityEstimate{T}(x, κ; support=(-Inf, Inf), rtol=cbrt(eps(T)))

Estimate a continuous one-dimensional probability density from sample points `x`,
using the scalar-field method of Holy, *Phys. Rev. Lett.* **79**, 3545 (1997).

`κ` is either a positive number, giving one scale everywhere, or a callable `κ(x)` returning
the scale local to `x`; larger `κ` gives a rougher estimate. See [`select_kappa`](@ref) for
choosing it automatically.

The fit's floating-point type `T` is the promotion of the data, `κ`, `rtol` and any finite
`support` bound; an infinite bound does not widen it. `DensityEstimate{T}(x, κ; …)` sets `T`
instead, converting the data and parameters to it: `DensityEstimate{Float32}(x, κ)` fits
`Float64` data in single precision.

`support = (a, b)` fits the density on a finite domain instead of all of `ℝ`; either end may be
`-Inf`/`Inf` for a one-sided or fully unbounded fit (the default). The density `Q` is exactly
zero outside `[a, b]`, [`cdf`](@ref) reaches exactly `0` at `a` and `1` at `b`, and every data
point must lie in `[a, b]` (checked at fit time; a violation, or `a ≥ b`, throws a
`DomainError`). Data points must be finite even when the support is infinite: a NaN or ±Inf in
`x`, or a value that overflows `T`, throws an `ArgumentError`.

Repeated points, and points closer than `rtol / κ(x)` (i.e. within a fraction `rtol` of
the local smoothing length), are merged into one node carrying the count as its integer
weight, so weighted data is handled naturally. Without merging, the resulting
tridiagonal system can be nearly singular.

The returned object is callable: `d(x)` evaluates the density `Q(x)` at any real
`x`, and it can be broadcast over arrays. Use [`amplitude`](@ref) for `ψ(x)`.

# Examples
```jldoctest
julia> d = DensityEstimate([-1.0, 0.0, 0.0, 1.0], 1.0);

julia> d(0.0) > d(2.0)   # denser near the data
true

julia> round(chisq(d, d); digits=8)   # a distribution fits itself perfectly
0.0

julia> a = DensityEstimate([-1.0, 0.0, 0.0, 1.0], x -> 1 + exp(-x^2));  # sharper near 0

julia> a.κ                                # one rate per inter-node interval
2-element Vector{Float64}:
 1.778800783071405
 1.778800783071405

julia> u = DensityEstimate(range(0.05, 0.95; length=50), 10.0; support=(0.0, 1.0));

julia> u(-0.1), u(1.1)              # zero outside the support
(0.0, 0.0)

julia> cdf(u, 0.0), cdf(u, 1.0)     # cdf hits 0 and 1 exactly at the walls
(0.0, 1.0)
```

# Extended help

The density is written as `Q(x) = ψ(x)^2`, where the amplitude `ψ` minimizes the action

    S[ψ] = ∫ (λ/κ(x)²) (ψ')² dx - 2 Σᵢ ln ψ(xᵢ)

subject to `∫ ψ² dx = 1`, with `λ` the normalization multiplier. The smoothing scale `κ` sets
the width of each point's contribution, and the penalty weight `1/κ(x)²` on `(ψ')²` is what
keeps the pressure to normalize spatially uniform.

A callable `κ(x)` is evaluated at the midpoint of each inter-node interval, and at the
outermost nodes for the two tails, so the fit resolves a piecewise-constant scale: `d.κ[k]` is
the rate on `(d.x[k], d.x[k+1])`, and `d.κL`, `d.κR` the tail rates. Making `κ` large where the
density is high and small where it is low buys resolution where the data can pay for it.

Between sorted data points `ψ` solves `ψ'' = κ² ψ`, i.e. it is a sum of rising and
falling exponentials, and decays as `e^{-κ|x|}` in the tails. The nodal amplitudes
`ψ(xᵢ)` satisfy a symmetric tridiagonal system whose solution is the minimizer of a
strictly convex potential; normalization is then a rescaling.

At a finite support endpoint the density is left free rather than pinned to zero (a natural, or
Neumann, boundary condition: `ψ'(a) = 0`) — the wall changes only the outermost interval on
that side, replacing its exponential tail with a `cosh` arc pinned flat at the wall, so a
discontinuous or divergent edge (a "jump edge") is representable directly rather than
approximated by a fast-decaying tail.

The goodness-of-fit machinery ([`ChisqReference`](@ref) and everything built on it) supports a
varying `κ` exactly as it does a constant one, and a finite `support` exactly as it does the
unbounded line.

Passing a [`SolveStats`](@ref) as the `stats` keyword records the Newton-step and backtracking
counts of the fixed-scale solve, for benchmarking; it does not change the fit.
"""
struct DensityEstimate{T<:AbstractFloat,K<:Union{T,Vector{T}}} <: AbstractDensityEstimate
    x::Vector{T}   # sorted, distinct node locations
    w::Vector{T}   # weight (multiplicity) at each node
    ψ::Vector{T}   # normalized amplitude at the nodes
    κ::K           # smoothing scale: one number, or one per inter-node interval
    κL::T          # decay rate of the left tail
    κR::T          # decay rate of the right tail
    lo::T          # left edge of the support; -Inf for an unbounded left tail
    hi::T          # right edge of the support; +Inf for an unbounded right tail
    λ::T           # normalization multiplier (diagnostic)
    F::Vector{T}   # mass below each node (see `_node_cumulatives`)
    G::Vector{T}   # mass above each node
    total::T       # total mass, F[end] + G[end]

    function DensityEstimate{T,K}(x, w, ψ, κ, κL, κR, lo, hi, λ) where {T<:AbstractFloat,K<:Union{T,Vector{T}}}
        length(x) == length(w) == length(ψ) ||
            throw(DimensionMismatch("nodes, weights, and amplitudes must have equal length, " *
                                    "got $(length(x)), $(length(w)), $(length(ψ))"))
        _check_interval_scale(κ, length(x))
        # The node cumulatives are computed from the other fields. During that computation the
        # tables are empty and the total is NaN, so a premature read throws a BoundsError or
        # yields NaN rather than a plausible value.
        d = new{T,K}(x, w, ψ, κ, κL, κR, lo, hi, λ, T[], T[], T(NaN))
        F, G, total = _node_cumulatives(d)
        return new{T,K}(d.x, d.w, d.ψ, d.κ, d.κL, d.κR, d.lo, d.hi, d.λ, F, G, total)
    end
end

# A per-interval scale carries one rate for each of the n-1 gaps between n nodes; a mismatch
# would leave surplus intervals silently unused rather than error at `d.κ[k]`.
_check_interval_scale(::Real, n) = nothing
_check_interval_scale(κ::AbstractVector, n) =
    length(κ) == n - 1 || throw(DimensionMismatch(
        "a per-interval scale needs one rate per inter-node interval: " *
        "got $(length(κ)) rates for $n nodes"))

DensityEstimate{T}(x, w, ψ, κ::Real, κL, κR, lo, hi, λ) where {T} =
    DensityEstimate{T,T}(x, w, ψ, κ, κL, κR, lo, hi, λ)
DensityEstimate{T}(x, w, ψ, κ::AbstractVector, κL, κR, lo, hi, λ) where {T} =
    DensityEstimate{T,Vector{T}}(x, w, ψ, κ, κL, κR, lo, hi, λ)

function DensityEstimate(x::AbstractVector{R}, κ; support::Tuple{Real,Real}=(-Inf, Inf),
                         rtol::Real=cbrt(eps(float(R))), stats=nothing) where R<:Real
    a, b = _check_support_bounds(support)
    T = _estimate_eltype(x, κ, rtol, a, b)
    return DensityEstimate{T}(x, κ; support, rtol, stats)
end

function DensityEstimate{T}(x::AbstractVector{<:Real}, κ; support::Tuple{Real,Real}=(-Inf, Inf),
                            rtol::Real=cbrt(eps(T)), stats=nothing) where {T<:AbstractFloat}
    _check_rtol(rtol)
    isempty(x) && throw(ArgumentError("cannot fit a density to zero points"))
    a, b = _check_support_bounds(support)
    return _estimate(T, x, κ, rtol, a, b; stats)
end

# The working float type of a fit: the promotion of the data, scale, tolerance and finite
# support bounds. A callable scale's precision is seen by sampling it at a data point.
_estimate_eltype(x::AbstractVector{R}, κ::Real, rtol::Real, a::Real, b::Real) where {R<:Real} =
    float(promote_type(R, typeof(κ), typeof(rtol), _support_eltype(a), _support_eltype(b)))
_estimate_eltype(::AbstractVector{<:Real}, ::AbstractVector, ::Real, ::Real, ::Real) = _vector_scale_error()
_estimate_eltype(x::AbstractVector{R}, κfun, rtol::Real, a::Real, b::Real) where {R<:Real} =
    float(promote_type(R, typeof(rtol), isempty(x) ? Bool : typeof(κfun(first(x))),
                       _support_eltype(a), _support_eltype(b)))

# Every data point must lie in the requested support, or the boundary terms below (a cosh arc
# pinned at the wall) would be fit against data outside their own domain.
function _check_support(xs::Vector{T}, lo::T, hi::T) where {T}
    first(xs) >= lo ||
        throw(DomainError(first(xs), "data point $(first(xs)) lies outside the support [$lo, $hi]"))
    last(xs) <= hi ||
        throw(DomainError(last(xs), "data point $(last(xs)) lies outside the support [$lo, $hi]"))
end

# A finite bound competes in the working-type promotion the same way κ or rtol does; an
# infinite one is exactly representable in any AbstractFloat, so the default `support=(-Inf,
# Inf)` (always `Float64`-typed, being a literal) must not force a wider type onto, say, a
# Float32 fit. `Bool` is the promotion lattice's bottom numeric type, so it drops out here.
_support_eltype(a) = isfinite(a) ? typeof(a) : Bool

function _estimate(::Type{T}, x::AbstractVector{<:Real}, κ::Real, rtol::Real, a::Real, b::Real;
                   stats=nothing) where {T}
    κT = _kappa_in(T, κ)
    xs = _sorted_sample(x, T)
    lo, hi = T(a), T(b)
    _check_support(xs, lo, hi)
    nodes, weights = _merge_presorted(xs, T(rtol) / κT)
    return _fit(nodes, weights, κT, lo, hi; stats)
end

# The nodes are not known until the data has been merged, and the merge tolerance is itself
# rtol/κ(x) — so there is no node geometry a caller could have aligned a per-interval vector
# to. The scale has to arrive as a function of position.
_estimate(::Type, ::AbstractVector{<:Real}, ::AbstractVector, ::Real, ::Real, ::Real; stats=nothing) =
    _vector_scale_error()
_vector_scale_error() =
    throw(ArgumentError("the smoothing scale cannot be given as a vector: node merging depends " *
                        "on the local scale, so the nodes it would index do not exist yet. Pass a " *
                        "callable `κ(x)` instead; the fit reports the realized per-interval rates."))

function _estimate(::Type{T}, x::AbstractVector{<:Real}, κfun, rtol::Real, a::Real, b::Real;
                   stats=nothing) where {T}
    xs = _sorted_sample(x, T)
    lo, hi = T(a), T(b)
    _check_support(xs, lo, hi)
    nodes, weights, κs, κL, κR = _merge_and_realize(xs, κfun, T(rtol))
    return _fit(nodes, weights, κs, κL, κR, lo, hi; stats)
end

# The scale on the interval between nodes k and k+1. Constant and piecewise-constant fits
# differ only here, so every evaluation routine below is written once and specializes to
# the constant case at no cost.
_kappa(d::DensityEstimate{T,T}, k::Integer) where {T} = d.κ
_kappa(d::DensityEstimate{T,Vector{T}}, k::Integer) where {T} = d.κ[k]

# The same accessor for a bare scale — one rate, or one per interval — as passed around
# before a `DensityEstimate` exists (the cross-validation scores work on nodes and weights,
# not on a fit).
_kappa(κ::Real, k::Integer) = κ
_kappa(κs::AbstractVector, k::Integer) = κs[k]

_show_kappa(d::DensityEstimate{T,T}) where {T} = "κ=$(d.κ)"
function _show_kappa(d::DensityEstimate{T,Vector{T}}) where {T}
    # A one-node fit has no intervals, only the two tails, so both extrema fold them in.
    lo = min(d.κL, d.κR, minimum(d.κ; init=typemax(T)))
    hi = max(d.κL, d.κR, maximum(d.κ; init=typemin(T)))
    return "κ ∈ [$lo, $hi]"
end

# "" when unbounded, else the support explicitly — appended after λ so a plain `show` of an
# unbounded fit is untouched.
_show_support(d::DensityEstimate) =
    isinf(d.lo) && isinf(d.hi) ? "" : ", support=[$(d.lo), $(d.hi)]"
"""
    nobs(d::AbstractDensityEstimate) -> N

The number of observations underlying the fit `d`, counting each point and each interval
observation once (merged repeats count with their multiplicity). It has `d`'s element type.
Extends `StatsAPI.nobs`.
"""
StatsAPI.nobs(::AbstractDensityEstimate)

StatsAPI.nobs(d::DensityEstimate) = sum(d.w)

Base.show(io::IO, d::DensityEstimate) = print(io, "DensityEstimate with $(length(d.x)) distinct nodes, $(sum(d.w)) total weight, $(_show_kappa(d)), λ=$(d.λ)$(_show_support(d))")

# Fit with an optional natural (Neumann) boundary at `lo`/`hi` (either may be infinite).
function _fit(nodes::Vector{T}, weights::Vector{T}, κ::T, lo::T, hi::T; stats=nothing) where {T}
    ψ = _solve_amplitude(_roughness_operator(nodes, κ, lo, hi), weights; stats)
    Z = _norm_sq(nodes, ψ, κ, κ, κ, lo, hi)
    ψ ./= sqrt(Z)
    λ = κ * Z                       # scaling law: normalized ψ solves Mψ = (κ/λ)/ψ
    return DensityEstimate{T}(nodes, weights, ψ, κ, κ, κ, lo, hi, λ)
end

# Piecewise-constant scale with an optional natural boundary at `lo`/`hi`. The assembled
# operator carries an arbitrary overall factor κ̄ (see `_roughness_operator`), which cancels from
# the normalized amplitude and leaves the multiplier λ = κ̄ Z well defined: the stationarity
# condition of the unscaled operator is Mψ = (1/λ) w ⊘ ψ, whose constant-κ specialization is the
# scaling law `_fit(nodes, weights, κ, lo, hi)` above uses.
function _fit(nodes::Vector{T}, weights::Vector{T}, κs::Vector{T}, κL::T, κR::T, lo::T, hi::T; stats=nothing) where {T}
    κ̄ = _reference_scale(κs, κL, κR)
    ψ = _solve_amplitude(_roughness_operator(nodes, κs, κL, κR, κ̄, lo, hi), weights; stats)
    Z = _norm_sq(nodes, ψ, κs, κL, κR, lo, hi)
    ψ ./= sqrt(Z)
    return DensityEstimate{T}(nodes, weights, ψ, κs, κL, κR, lo, hi, κ̄ * Z)
end

# Reject scale values a fit cannot use: a scalar κ, or a scale function's value κ(x).
_check_kappa(κ) =
    isfinite(κ) && κ > 0 ? κ : throw(ArgumentError("κ must be finite and positive, got $κ"))
_check_kappa(κ, x) =
    isfinite(κ) && κ > 0 ? κ :
    throw(ArgumentError("the smoothing scale must be finite and positive, got κ($x) = $κ"))

# A scalar κ converted to the working type `T`, where it must still be finite and positive.
function _kappa_in(::Type{T}, κ::Real) where {T}
    _check_kappa(κ)
    κT = T(κ)
    isfinite(κT) && κT > 0 ||
        throw(ArgumentError("κ = $κ is not a finite positive $T (it converts to $κT)"))
    return κT
end

# Evaluate a user-supplied scale function at one point.
_checked_kappa(κfun, x, ::Type{T}) where {T} = _check_kappa(T(κfun(x)), x)

# The scale at each position of the *sorted* vector `ts`. A general callable is asked
# pointwise; `AdaptiveScale` overrides this with a single walk of its pilot (see below).
_kappa_sorted(κfun, ts::AbstractVector, ::Type{T}) where {T} =
    T[_checked_kappa(κfun, t, T) for t in ts]

# Realize `κfun` on the node geometry: one rate per inter-node interval (from its
# midpoint), and one per tail (from the outermost node it decays away from). The midpoints
# inherit the nodes' order, so they are realized as a sorted batch.
function _kappa_profile(nodes::Vector{T}, κfun, ::Type{T}) where {T}
    n = length(nodes)
    mids = T[(nodes[k] + nodes[k+1]) / 2 for k in 1:n-1]
    return _kappa_sorted(κfun, mids, T),
           _checked_kappa(κfun, first(nodes), T), _checked_kappa(κfun, last(nodes), T)
end

# Geometric mean of the interval rates: the overall scale the operator is expressed in.
# A constant κ is its own reference scale.
_reference_scale(κ::T, ::T, ::T) where {T} = κ
_reference_scale(κs::Vector{T}, κL::T, κR::T) where {T} =
    isempty(κs) ? sqrt(κL * κR) : exp(sum(log, κs) / length(κs))

# _sorted_sample(x, T) -> xs::Vector{T}
#
# A sorted, one-based working copy of the sample `x`, whatever its axes. Every index the fit
# takes afterwards — into the sample, into a scale realized on it, into the merged nodes —
# is an index into this copy, and none of them escape, so the caller's axes have nothing to
# propagate to.
#
# A value of `x` that is not finite in `T` throws an `ArgumentError`.
function _sorted_sample(x::AbstractVector, ::Type{T}) where {T}
    _check_finite_observations(x, T)
    xs = Vector{T}(undef, length(x))
    i = firstindex(xs)
    for xi in x                     # iterate, rather than index, to stay axis-agnostic
        xs[i] = xi
        i += 1
    end
    return sort!(xs)
end

# Collapse runs of an already-sorted sequence `xs` within `atol` of the run's first member.
# Factored out so a selector can merge one sorted copy at many tolerances. If `nodeof`
# is a vector, `nodeof[i]` receives the index of the node that `xs[i]` joined.
function _merge_presorted(xs, atol::T, nodeof=nothing) where {T}
    nodes = T[]
    weights = T[]
    for i in eachindex(xs)
        xk = T(xs[i])
        if !isempty(nodes) && xk - nodes[end] <= atol
            weights[end] += oneunit(T)
        else
            push!(nodes, xk)
            push!(weights, oneunit(T))
        end
        nodeof === nothing || (nodeof[i] = lastindex(nodes))
    end
    return nodes, weights
end

# As above, but with a tolerance `rtol / κ` local to the run's first member, whose scale is
# `κx[i]` for the point `xs[i]`: the merge threshold is a fraction `rtol` of the smoothing
# length there. The scales come in already realized on `xs` because the merge threshold is
# what *produces* the nodes — a caller has no node geometry to align a per-node vector to.
function _merge_presorted(xs, rtol::T, κx::AbstractVector{T}, nodeof=nothing) where {T}
    nodes = T[]
    weights = T[]
    κrun = zero(T)                  # scale at the run's first member, which sets its tolerance
    for i in eachindex(xs, κx)
        xk = T(xs[i])
        if !isempty(nodes) && κrun * (xk - nodes[end]) <= rtol
            weights[end] += oneunit(T)
        else
            push!(nodes, xk)
            push!(weights, oneunit(T))
            κrun = κx[i]
        end
        nodeof === nothing || (nodeof[i] = lastindex(nodes))
    end
    return nodes, weights
end

"""
    merge_samples(x, κ; rtol=cbrt(eps(float(eltype(x))))) -> (; nodes, weights, nodeindex)

Merge the sample `x` into the nodes and weights that `DensityEstimate(x, κ; rtol)` fits, and
report which node each sample joined.

The sorted sample is split into runs, each starting at the smallest sample not yet assigned.
A run collects every following sample within `rtol/κ` of its first member, where `κ` is the
scale at that first member; `κ` may be a positive number or a callable `κ(x)`, as for
[`DensityEstimate`](@ref). Each run becomes one node located at its first member, with
weight equal to the number of samples in the run.

- `nodes`: the sorted, distinct node locations, as `d.x` of the fit.
- `weights`: the number of samples merged into each node, as `d.w` of the fit.
- `nodeindex`: an integer array with the axes of `x`; sample `x[j]` was merged into node
  `nodes[nodeindex[j]]`. A node lies at the value of its run's first member, so the samples
  with `x[j] == nodes[nodeindex[j]]` (that member and any exact ties to it) are the ones
  whose values set the node's position.

`nodes` and `weights` have the float type that `DensityEstimate(x, κ; rtol)` uses on the
unbounded line. A NaN or ±Inf in `x` throws an `ArgumentError`.

# Examples
```jldoctest
julia> m = PenalizedDensity.merge_samples([2.0, 0.0, 1e-9, 2.0], 1.0);

julia> m.nodes, m.weights
([0.0, 2.0], [2.0, 2.0])

julia> m.nodeindex
4-element Vector{Int64}:
 2
 1
 1
 2
```
"""
function merge_samples(x::AbstractVector{R}, κ; rtol::Real=cbrt(eps(float(R)))) where {R<:Real}
    _check_rtol(rtol)
    isempty(x) && throw(ArgumentError("cannot merge zero points"))
    return _merge_samples(x, κ, rtol)
end

function _merge_samples(x::AbstractVector{R}, κ::Real, rtol::Real) where {R<:Real}
    _check_kappa(κ)
    T = float(promote_type(R, typeof(κ), typeof(rtol)))
    perm, xs = _sortperm_sample(x, T)
    nodeof = similar(perm)
    nodes, weights = _merge_presorted(xs, T(rtol) / T(κ), nodeof)
    return (; nodes, weights, nodeindex = _unsort_nodeindex(x, perm, nodeof))
end

_merge_samples(::AbstractVector{<:Real}, ::AbstractVector, ::Real) = _vector_scale_error()

function _merge_samples(x::AbstractVector{R}, κfun, rtol::Real) where {R<:Real}
    T = float(promote_type(R, typeof(rtol), typeof(κfun(first(x)))))
    perm, xs = _sortperm_sample(x, T)
    nodeof = similar(perm)
    nodes, weights = _merge_presorted(xs, T(rtol), _kappa_sorted(κfun, xs, T), nodeof)
    return (; nodes, weights, nodeindex = _unsort_nodeindex(x, perm, nodeof))
end

# `_sorted_sample` together with the permutation that sorts `x`: `xs[i] == x[perm[i]]`.
function _sortperm_sample(x::AbstractVector, ::Type{T}) where {T}
    _check_finite_observations(x, T)
    perm = sortperm(x)
    xs = T[x[p] for p in perm]
    return perm, xs
end

# Carry node indices from sorted positions back to the positions (and axes) of `x`.
function _unsort_nodeindex(x::AbstractVector, perm::AbstractVector, nodeof::AbstractVector)
    nodeindex = similar(x, Int)
    for i in eachindex(perm, nodeof)
        nodeindex[perm[i]] = nodeof[i]
    end
    return nodeindex
end

# Merge the sample at the tolerance a scale implies, then realize that scale on the nodes the
# merge produced. This pairing is the whole entry into a piecewise-constant fit: the merge
# needs the scale at the sample points, and the fit needs it on the nodes and tails.
function _merge_and_realize(xs::Vector{T}, κfun, rtol::T) where {T}
    nodes, weights = _merge_presorted(xs, rtol, _kappa_sorted(κfun, xs, T))
    κs, κL, κR = _kappa_profile(nodes, κfun, T)
    return nodes, weights, κs, κL, κR
end

"""
    roughness_operator(x, κ, lo=-Inf, hi=Inf) -> SymTridiagonal

The symmetric positive-definite tridiagonal matrix `M` of the roughness penalty on the
amplitude `ψ` at sorted, distinct nodes `x` with constant scale `κ`: the penalty
`∫ (ψ'(t)² + κ² ψ(t)²) dt`, minimized between and beyond the nodes for given nodal amplitudes
`ψ`, is `κ * dot(ψ, M * ψ)`. A finite `lo` or `hi` is a support wall with a natural (zero-slope) boundary
condition; an infinite one is an exponentially decaying tail.

For the gap `hₖ = x[k+1] - x[k]`, the off-diagonal entry is `-csch(κ hₖ)`, and the diagonal
entry at each node sums `coth(κ hₖ)` over its adjacent gaps, plus `tanh(κ Δ)` for the
distance `Δ` from an end node to its wall (`1` for an unbounded side).

`x` must be 1-based, because `M[i, j]` couples `x[i]` and `x[j]`. The arguments are promoted
to a common floating-point type; an infinite `lo` or `hi` does not take part in the promotion,
so `Float32` nodes and scale give a `Float32` operator.
"""
function roughness_operator(x::AbstractVector{<:Real}, κ::Real, lo::Real=-Inf, hi::Real=Inf)
    T = float(promote_type(eltype(x), typeof(κ), _support_eltype(lo), _support_eltype(hi)))
    return _roughness_operator(convert(AbstractVector{T}, x)::AbstractVector{T}, T(κ), T(lo), T(hi))
end

function _roughness_operator(x::AbstractVector{T}, κ::T, lo::T, hi::T) where {T<:AbstractFloat}
    Base.require_one_based_indexing(x)
    n = length(x)
    n >= 1 || throw(ArgumentError("need at least one node to build the roughness operator"))
    d = zeros(T, n)
    e = zeros(T, n - 1)
    d[1] += _tail_diag(κ, x[1] - lo)   # left tail
    d[n] += _tail_diag(κ, hi - x[n])   # right tail
    for k in 1:n-1
        θ = κ * (x[k+1] - x[k])
        d[k]   += coth(θ)
        d[k+1] += coth(θ)
        e[k]    = -csch(θ)          # coth/csch stay finite as θ → ∞ (isolated points)
    end
    return SymTridiagonal(d, e)     # M
end

# tanh(u), overflow-free through e^{-2u} (accurate and finite up to u ≈ 1e300, well past where
# cosh/sinh alone would overflow around u ≈ 710).
_tanh_stable(u::T) where {T} = (e = exp(-2u); (oneunit(T) - e) / (oneunit(T) + e))

# sech(u)² = 1/cosh(u)², overflow-free through e^{-2u}.
_sech2_stable(u::T) where {T} = (e = exp(-2u); 4 * e / (oneunit(T) + e)^2)

# u·sech(u)², the companion term in the boundary tail mass below.
_usech2_stable(u::T) where {T} = u * _sech2_stable(u)

# Tail diagonal contribution to the roughness operator at a boundary gap Δ = |edge - boundary|:
# tanh(κΔ) for a natural (Neumann) boundary, or 1 in the unbounded limit Δ = ∞. Both forms agree
# as Δ → ∞ (tanh → 1); the branch only avoids evaluating tanh at an infinite argument.
_tail_diag(κ::T, Δ::T) where {T} = isfinite(Δ) ? _tanh_stable(κ * Δ) : oneunit(T)

# Tail mass ∫ψ² over a boundary segment of gap Δ: ψ₁²(tanh u + u·sech²u)/(2κ) at u = κΔ finite,
# or the unbounded ψ₁²/(2κ) as Δ → ∞ (both terms of the finite form → 0 and 1 respectively).
function _tail_mass(ψ1::T, κ::T, Δ::T) where {T}
    isfinite(Δ) || return ψ1^2 / (2κ)
    u = κ * Δ
    return ψ1^2 * (_tanh_stable(u) + _usech2_stable(u)) / (2κ)
end

# The same operator for a piecewise-constant scale, with an optional natural boundary at
# `lo`/`hi`: interval k (rate κs[k], θ = κs[k]·hₖ) contributes coth(θ)/κs[k] to each adjacent
# diagonal entry and -csch(θ)/κs[k] off-diagonal, and each tail contributes
# `_tail_diag(κ_edge, Δ)/κ_edge` — `1/κ_edge` as Δ → ∞ (an unbounded side), or
# `tanh(κ_edge Δ)/κ_edge` at a finite gap. Dividing through by one κ no longer cancels the
# entries, so the rates survive explicitly.
#
# Everything is scaled by the reference rate κ̄. That factor is arbitrary — it rescales the
# unnormalized amplitude by κ̄^{-1/2} and drops out of both the normalized fit and λ = κ̄ Z —
# but it fixes the magnitude the Newton solve sees. Taking κ̄ to be the typical rate keeps the
# entries O(1), and at a constant κ (where κ̄ = κ) reproduces `roughness_operator(x, κ, lo, hi)`
# entry for entry.
function _roughness_operator(x::Vector{T}, κs::Vector{T}, κL::T, κR::T, κ̄::T, lo::T, hi::T) where {T<:AbstractFloat}
    n = length(x)
    n >= 1 || throw(ArgumentError("need at least one node to build the roughness operator"))
    length(κs) == n - 1 ||
        throw(DimensionMismatch("$n nodes bound $(n-1) intervals, but got $(length(κs)) scales"))
    d = zeros(T, n)
    e = zeros(T, n - 1)
    d[1] += κ̄ * _tail_diag(κL, x[1] - lo) / κL   # left tail
    d[n] += κ̄ * _tail_diag(κR, hi - x[n]) / κR   # right tail
    for k in 1:n-1
        θ = κs[k] * (x[k+1] - x[k])
        u = κ̄ / κs[k]
        d[k]   += u * coth(θ)
        d[k+1] += u * coth(θ)
        e[k]    = -u * csch(θ)      # coth/csch stay finite as θ → ∞ (isolated points)
    end
    return SymTridiagonal(d, e)
end

# M for a bare scale, whichever form it takes, with an optional natural boundary at `lo`/`hi`.
# A constant κ is its own reference scale, so this reduces to `roughness_operator(x, κ, lo, hi)`
# entry for entry; a per-interval κ is assembled in units of the geometric-mean rate, as the
# fit does.
_operator(x::Vector{T}, κ::T, κL::T, κR::T, lo::T, hi::T) where {T} = _roughness_operator(x, κ, lo, hi)
_operator(x::Vector{T}, κs::Vector{T}, κL::T, κR::T, lo::T, hi::T) where {T} =
    _roughness_operator(x, κs, κL, κR, _reference_scale(κs, κL, κR), lo, hi)

# F(ψ) = ½ ψ'Mψ - Σ wᵢ ln ψᵢ, the potential minimized by _solve_amplitude.
function _objective(M::SymTridiagonal{T}, w::Vector{T}, ψ::Vector{T}) where {T<:AbstractFloat}
    s = zero(T)
    for i in eachindex(w, ψ)
        s += w[i] * log(ψ[i])       # requires ψ > 0, which is enforced by the caller
    end
    return dot(ψ, M, ψ) / 2 - s
end

# The magnitude of the terms summed to form F(ψ), before their cancellation: ∑ Mᵢᵢψᵢ² bounds
# the quadratic form's terms (M is diagonally dominant, so each |Mᵢ,ᵢ₊₁|ψᵢψᵢ₊₁ is covered by the
# adjacent diagonal terms), and ∑ wᵢ|ln ψᵢ| the likelihood's. Rounding leaves F uncertain by a
# multiple of eps times this, not of |F|: on nearly coincident nodes the coth entries reach
# 1e5 or more while the quadratic form they enter nearly cancels, so |F| understates the
# roundoff by that factor.
function _objective_scale(M::SymTridiagonal{T}, w::Vector{T}, ψ::Vector{T}) where {T<:AbstractFloat}
    s = zero(T)
    for i in eachindex(w, ψ)
        s += M.dv[i] * ψ[i]^2 + w[i] * abs(log(ψ[i]))
    end
    return s
end

"""
    SolveStats()

Mutable collector for diagnostics of the fixed-scale Newton solve. Pass one as the
`stats` keyword to [`DensityEstimate`](@ref) or [`IntervalDensityEstimate`](@ref) to record
how the solve behaved:

- `iterations`: the number of Newton steps taken (each a Hessian factorization that
  updated the amplitudes);
- `backtracks`: the total number of Armijo step-halvings across those steps;
- `reason`: why the iteration stopped —
  - `:tolerance`, the Newton correction fell to `eps(T)^(3/4)` relative to `ψ` (the
    healthy exit);
  - `:floor`, the correction stopped falling before reaching that tolerance, held up
    by the roundoff floor a near-singular Hessian imposes (a conditioning-limited
    solve);
  - `:steplength`, backtracking found no positive step moving `ψ` by more than
    rounding, so `ψ` is stationary to the available precision;
  - `:none`, left unset (the solve threw before terminating);
- `final_step`: the last Newton correction magnitude, `maxᵢ |Δᵢ|/ψᵢ` — at or below
  the tolerance on a `:tolerance` exit, above it on a `:floor` exit;
- `final_alpha`: the damping `α` of the last accepted step, `1` when it was undamped.

Construct a fresh collector per fit. The field equation is solved identically
whether or not one is present.
"""
mutable struct SolveStats
    iterations::Int
    backtracks::Int
    reason::Symbol
    final_step::Float64
    final_alpha::Float64
end
SolveStats() = SolveStats(0, 0, :none, NaN, NaN)

function Base.show(io::IO, s::SolveStats)
    s.reason === :none && return print(io, "SolveStats (no solve recorded)")
    print(io, "SolveStats: ", s.iterations, " Newton steps, ", s.backtracks, " backtracks, stopped by :",
          s.reason, " (final step ", s.final_step, ", α = ", s.final_alpha, ")")
end

# Stopping rule of the damped-Newton solvers, given the relative Newton correction `step` =
# maxᵢ |Δᵢ|/xᵢ, its value `prevstep` at the previous iteration, and whether a step has been
# taken without the Armijo test (`unguarded`). Returns `:tolerance`, `:floor` or `:none`.
#
# The correction Δ is what the next step would subtract, so in the quadratic regime it is the
# remaining error in x. Testing the objective instead, or the Newton decrement, which predicts
# the same difference, constrains a quantity that is stationary at the minimizer, and is
# satisfied while x is still wrong by the square root of the tolerance; that error would carry
# into the normalization and the density.
#
# Roundoff floors the attainable correction at a level set by the conditioning of the Hessian,
# and on a widely spread sample that floor can sit well above the tolerance; stopping when the
# correction ceases to fall covers that case. The floor is only meaningful once the objective
# has stopped resolving the decrease the step predicts, which is what `unguarded` records:
# while the Armijo test is still informative the correction may plateau for a step or two
# without the iteration being finished.
function _newton_stop(step::T, prevstep::T, unguarded::Bool) where {T}
    step <= eps(T)^(3 // 4) && return :tolerance
    unguarded && step >= prevstep && return :floor
    return :none
end

# Largest α ≤ 1 keeping x - αΔ strictly positive, shortened by 1% when that bound binds so the
# iterate stays off the boundary of the orthant.
function _positivity_cap(x::AbstractVector{T}, Δ::AbstractVector{T}) where {T}
    α = one(T)
    for i in eachindex(x, Δ)
        Δ[i] > 0 && (α = min(α, x[i] / Δ[i]))
    end
    return α < one(T) ? α * oftype(α, 0.99) : α
end

# _solve_amplitude(M, w) -> ψ
#
# Minimize the strictly convex potential `F(ψ) = ½ ψ'Mψ - Σ wᵢ ln ψᵢ` over `ψ > 0`
# by a damped Newton iteration with an SPD tridiagonal Hessian. The minimizer solves
# `Mψ = w ./ ψ`, i.e. the field equation at unit multiplier; the caller rescales it
# to impose normalization.
#
# Each step factorizes the tridiagonal Hessian in place (`ldlt!`/`ldiv!`) and backtracks
# along the Newton direction to keep `ψ > 0` with Armijo decrease. Iteration stops when the
# Newton correction reaches `eps(T)^(3/4)` relative to `ψ`, componentwise, or the floor
# roundoff imposes on it, so the returned amplitudes — not merely `F` — are accurate to that
# tolerance. Reaching `maxiter` throws a `ConvergenceError`.
#
# A `SolveStats` passed as `stats` accumulates the Newton-step and backtracking counts.
function _solve_amplitude(M::SymTridiagonal{T}, w::Vector{T}; maxiter::Int=100,
                          stats::Union{Nothing,SolveStats}=nothing) where {T<:AbstractFloat}
    n = length(w)
    ψ = fill(oneunit(T), n)             # strictly positive start
    g = similar(ψ); Δ = similar(ψ); ψnew = similar(ψ)
    Hdv = similar(ψ); Hev = similar(M.ev)   # Hessian factorization scratch
    W = sum(w)                          # total multiplicity, the scale of F's two terms
    Fψ = _objective(M, w, ψ)
    prevstep = T(Inf)
    unguarded = false                   # has a step been taken without the Armijo test?
    converged = false
    iters = 0; backs = 0                # Newton steps taken; Armijo halvings across them
    reason = :none
    finalstep = T(Inf); finalα = T(NaN) # last correction magnitude and accepted damping
    for _ in 1:maxiter
        mul!(g, M, ψ)
        @. g -= w / ψ                    # ∇F = Mψ - w./ψ
        @. Hdv = M.dv + w / ψ^2          # diagonal of ∇²F; off-diagonal equals M.ev
        Hev .= M.ev                      # ldlt! overwrites its arguments; refill each step
        Δ .= g
        ldiv!(ldlt!(SymTridiagonal(Hdv, Hev)), Δ)   # Δ = (∇²F)⁻¹ ∇F
        decrement = dot(g, Δ)               # Newton decrement λ² = ∇Fᵀ(∇²F)⁻¹∇F ≥ 0
        step = zero(T)
        for i in eachindex(Δ, ψ)
            step = max(step, abs(Δ[i]) / ψ[i])   # ψ > 0 throughout
        end
        reason = _newton_stop(step, prevstep, unguarded)
        if reason !== :none
            converged = true; finalstep = step
            break
        end
        iters += 1
        prevstep = step
        α = _positivity_cap(ψ, Δ)         # then Armijo backtracking
        # Armijo compares two values of F differing by α·decrement/4. F is a
        # difference of terms accumulated over n nodes, so rounding leaves it
        # uncertain by roughly √n·eps times the size of those terms
        # (`_objective_scale`, which can exceed |F| by orders of magnitude when
        # nearly coincident nodes make the operator's entries large); a predicted
        # decrease below that carries no information, and rejecting the step on
        # it stalls the iteration into halving α when it should be squaring the
        # error. Such a step is deep enough into the quadratic regime to take
        # unguarded.
        if α * decrement / 4 <= 4 * sqrt(T(n)) * eps(T) * (_objective_scale(M, w, ψ) + W)
            unguarded = true
            @. ψ -= α * Δ
            Fψ = _objective(M, w, ψ)
            finalα = α
            continue
        end
        armijo = false
        local Fnew
        while α * step >= eps(T)        # below this ψ - αΔ rounds back to ψ
            @. ψnew = ψ - α * Δ
            Fnew = _objective(M, w, ψnew)
            if Fnew <= Fψ - α * decrement / 4
                armijo = true
                break
            end
            α /= 2
            backs += 1
        end
        # Backtracking ran out of room: no positive step both keeps ψ inside the
        # orthant and moves it by more than rounding, so ψ is stationary to the
        # precision available. That is a converged fit, not a failed one.
        if !armijo
            converged = true; reason = :steplength; finalstep = step
            break
        end
        copyto!(ψ, ψnew)
        Fψ = Fnew
        finalα = α
    end
    converged || throw(ConvergenceError("the amplitude solve", "Newton did not converge in $maxiter iterations"))
    if stats !== nothing
        stats.iterations = iters
        stats.backtracks = backs
        stats.reason = reason
        stats.final_step = Float64(finalstep)
        stats.final_alpha = Float64(finalα)
    end
    return ψ
end

# ∫ ψ² dx for the hyperbolic interpolant with exponential tails, as a tridiagonal quadratic
# form evaluated at the nodal amplitudes. The scale κ is a scalar or one value per interval,
# and either side may have a natural boundary at `lo`/`hi`. The interpolant on interval k and
# the tail decays are set by the rates themselves, not by the operator's overall factor, so
# this is the physical mass whatever κ̄ the amplitude was solved in. Each tail is
# `_tail_mass(ψ_edge, κ_edge, Δ)` — ψ_edge²/(2κ_edge) as Δ → ∞ (an unbounded side), or
# ψ_edge²(tanh u + u·sech²u)/(2κ_edge) at a finite gap.
function _norm_sq(x::Vector{T}, ψ::Vector{T}, κ, κL::T, κR::T, lo::T, hi::T) where {T}
    n = length(x)
    Z = _tail_mass(ψ[1], κL, x[1] - lo) + _tail_mass(ψ[n], κR, hi - x[n])
    for k in 1:n-1
        κk = _kappa(κ, k)
        Z += _interval_mass(ψ[k], ψ[k+1], κk * (x[k+1] - x[k])) / κk
    end
    return Z
end

# Z = ∫ψ² and Gψ = ½ ∂Z/∂ψ, where Z = ψᵀGψ, with an optional natural boundary at `lo`/`hi`: the
# mass and the action of its Gram operator, from one pass over the per-interval
# `_mass_coeffs`. The leave-one-out expansion needs both. Each tail decays at its own rate and
# contributes `_tail_mass(ψ_edge, κ_edge, Δ)` to `Z`; `Gψᵢ = tail-mass(ψᵢ)/ψᵢ` at a boundary
# node reduces to `ψᵢ/(2κ_edge)` as Δ → ∞ (an unbounded side) since the tail mass is homogeneous
# degree 2 in ψᵢ.
function _norm_sq_gram(x::Vector{T}, ψ::Vector{T}, κ, κL::T, κR::T, lo::T, hi::T) where {T}
    n = length(x)
    Gψ = zeros(T, n)
    tl = _tail_mass(ψ[1], κL, x[1] - lo)
    tr = _tail_mass(ψ[n], κR, hi - x[n])
    Z = tl + tr
    Gψ[1] += tl / ψ[1]
    Gψ[n] += tr / ψ[n]
    for k in 1:n-1
        κk = _kappa(κ, k)
        fdiag, fcross = _mass_coeffs(κk * (x[k+1] - x[k])) ./ κk
        Z += fdiag * (ψ[k]^2 + ψ[k+1]^2) + 2 * fcross * ψ[k] * ψ[k+1]
        Gψ[k]   += fdiag * ψ[k]   + fcross * ψ[k+1]
        Gψ[k+1] += fdiag * ψ[k+1] + fcross * ψ[k]
    end
    return Z, Gψ
end

# (μ, σ²) of the fitted density Q = ψ², i.e. ∫x Q dx / Z and ∫x² Q dx / Z - μ² with Z = ∫Q dx.
# Interior cell k contributes `_cell_moments` in its local coordinate t = x - x[k] ∈ [0, h], and
# picks up the node offset x[k] as in ∫x Q dx = ∫(x[k] + t) Q dt. The tails contribute
# `_tail_moments` at their decay rates and boundary gaps. Z is accumulated in the same pass rather
# than assumed to be 1, so the result does not depend on how precisely ψ was normalized.
#
# Node positions are measured relative to x[1] throughout (not the caller's absolute
# coordinate) before accumulating M1, M2: since μ, σ² are translation-covariant/-invariant,
# this doesn't change the result, but it keeps M1, M2 (and hence M2 - M1²) of order the data's
# spread rather than of order its absolute location, which is what M2 - M1² needs to stay
# well-conditioned. Without it, data far from the origin (e.g. x .+ 1e8) can drive σ² negative.
function _moments(d::AbstractDensityEstimate)
    x, ψ = d.x, d.ψ
    n = length(x)
    x0 = x[1]
    xn = x[n] - x0
    rL, rR = _tail_rates(d)
    L0, L1, L2 = _tail_moments(rL, x[1] - d.lo)     # left tail, r = x[1] - x ≥ 0
    R0, R1, R2 = _tail_moments(rR, d.hi - x[n])     # right tail, r = x - x[n] ≥ 0
    Z  = ψ[1]^2 * L0 + ψ[n]^2 * R0
    M1 = -ψ[1]^2 * L1 + ψ[n]^2 * (xn * R0 + R1)
    M2 = ψ[1]^2 * L2 + ψ[n]^2 * (xn^2 * R0 + 2 * xn * R1 + R2)
    for k in 1:n-1
        Iseg, Jseg, Kseg = _cell_moments(_cell_k2(d, k), x[k+1] - x[k], ψ[k], ψ[k+1])
        xk = x[k] - x0
        Z  += Iseg
        M1 += xk * Iseg + Jseg
        M2 += xk^2 * Iseg + 2 * xk * Jseg + Kseg
    end
    μ = M1 / Z
    return μ + x0, M2 / Z - μ^2
end

# The decay rates of the two tails beyond the outermost nodes, and k² = ψ''/ψ on interior cell k.
_tail_rates(d::DensityEstimate) = (d.κL, d.κR)
_cell_k2(d::DensityEstimate, k::Integer) = _kappa(d, k)^2

# (∫₀ʰ Q dt, ∫₀ʰ t Q dt, ∫₀ʰ t² Q dt) for Q = ψ² on a cell of length h on which ψ'' = k2·ψ, with
# ψ(0) = p and ψ(h) = q. For k2·h² ≥ 1 these are quadratic forms in (p, q) with the coth/csch
# coefficients fdiag, fcross (weight 1), gdiag1, gdiag2, gcross (weight t) and hdiag1, hdiag2,
# hcross (weight t²). Those coefficients cancel catastrophically as κh → 0 (a relative error of
# 2e-4 at κh = 1e-3), so narrower cells, and trigonometric ones (k2 < 0, |k|h < π), use
# Gauss–Legendre quadrature, which is exact to roundoff there: 8 points for |k2|h² < 1, 16 beyond.
function _cell_moments(k2::T, h::T, p::T, q::T) where {T}
    z = k2 * h^2
    if z >= 1
        κ = sqrt(k2)
        θ = κ * h; ct, cs = coth(θ), csch(θ)
        fdiag  = (ct - θ * cs^2) / (2κ)
        fcross = cs * (θ * ct - oneunit(T)) / (2κ)
        gdiag1 = (1 - θ^2 * cs^2) / (4κ^2)
        gdiag2 = (2θ * ct - 1 - θ^2 * cs^2) / (4κ^2)
        gcross = θ * cs * (θ * ct - 1) / (4κ^2)
        hdiag1 = (3ct - 3θ * cs^2 - 2θ^3 * cs^2) / (12κ^3)
        hdiag2 = (3ct + 6θ^2 * ct - 6θ - 3θ * cs^2 - 2θ^3 * cs^2) / (12κ^3)
        hcross = cs * (2θ^3 * ct - 3θ^2 + 3θ * ct - 3) / (12κ^3)
        return (fdiag * (p^2 + q^2) + 2 * fcross * p * q,
                gdiag1 * p^2 + gdiag2 * q^2 + 2 * gcross * p * q,
                hdiag1 * p^2 + hdiag2 * q^2 + 2 * hcross * p * q)
    end
    nodes, weights = z > -1 ? _gauss8(T) : _gauss16(T)
    half = h / 2
    Sh = _Sfun(k2, h)
    I = J = K = zero(T)
    for i in eachindex(nodes, weights)
        t = half * (1 + nodes[i])
        wQ = weights[i] * ((p * _Sfun(k2, h - t) + q * _Sfun(k2, t)) / Sh)^2
        I += wQ
        J += t * wQ
        K += t^2 * wQ
    end
    return half * I, half * J, half * K
end

_gauss8(::Type{T}) where {T} = gauss(T, 8)
const _GAUSS8_F64 = gauss(Float64, 8)
_gauss8(::Type{Float64}) = _GAUSS8_F64

# Moments (J₀, J₁, J₂) = ∫ rᵐ ψ̂(r)² dr, m = 0, 1, 2, of a tail field with unit amplitude at the
# outermost node, r ≥ 0 being the distance from that node outward. On an unbounded side
# ψ̂ = e^{-κr}, giving 1/(2κ), 1/(4κ²), 1/(4κ³). At a finite gap Δ the Neumann arc
# ψ̂ = cosh(κ(Δ - r))/cosh(κΔ) on r ∈ [0, Δ] gives, with u = κΔ,
#     J₀ = (tanh u + u sech²u) / (2κ)
#     J₁ = (tanh²u + u² sech²u) / (4κ²)
#     J₂ = (3 tanh u - 3u sech²u + 2u³ sech²u) / (12κ³),
# each reducing to the unbounded value as u → ∞. J₀ is `_tail_mass` at unit amplitude.
function _tail_moments(κ::T, Δ::T) where {T}
    isfinite(Δ) || return (one(T) / (2κ), one(T) / (4κ^2), one(T) / (4κ^3))
    u = κ * Δ
    th, s2 = _tanh_stable(u), _sech2_stable(u)
    J0 = (th + u * s2) / (2κ)
    J1 = (th^2 + u^2 * s2) / (4κ^2)
    J2 = (3th - 3u * s2 + 2u^3 * s2) / (12κ^3)
    return J0, J1, J2
end

"""
    mean(d::AbstractDensityEstimate)

Mean `∫ x Q̂(x) dx` of the fitted density, computed from the nodal amplitudes cell by cell (in
closed form, or by quadrature that is exact to roundoff) plus the exponential or bounded tails. See also
[`var`](@ref Statistics.var) and [`std`](@ref Statistics.std).

# Examples

```jldoctest
julia> d = DensityEstimate([-1.0, 1.0], 1.0);

julia> abs(mean(d)) < 1e-12
true
```
"""
Statistics.mean(d::AbstractDensityEstimate) = first(_moments(d))

"""
    var(d::AbstractDensityEstimate)

Variance `∫ (x - μ)² Q̂(x) dx` of the fitted density, computed like
[`mean`](@ref Statistics.mean). This is the variance of the fitted density itself, not of the
sample it was fitted to.

# Examples

```jldoctest
julia> d = DensityEstimate([0.0], 1.0);   # one point: a Laplace density of rate 2κ

julia> var(d) ≈ 2 / (2 * 1.0)^2
true
```
"""
Statistics.var(d::AbstractDensityEstimate) = last(_moments(d))

"""
    std(d::AbstractDensityEstimate)

Standard deviation `√var(d)` of the fitted density; see [`var`](@ref Statistics.var).
"""
Statistics.std(d::AbstractDensityEstimate) = sqrt(var(d))

"""
    median(d::AbstractDensityEstimate)

Median of the fitted density, `quantile(d, 1/2)`; see [`quantile`](@ref Statistics.quantile).

# Examples

```jldoctest
julia> d = DensityEstimate([0.0], 0.5);   # a Laplace density centered at 0

julia> median(d)
0.0
```
"""
Statistics.median(d::AbstractDensityEstimate) = quantile(d, 1/2)

# Z = ∫ψ² together with its κ-derivative at fixed ψ and Gψ = ½ ∂Z/∂ψ, where Z = ψᵀGψ. The
# three share the per-interval coth/csch coefficients, so one pass returns all of them.
# Differentiating in κ presupposes a single rate: this serves the scalar-κ sensitivity
# `_action_and_slope`, not the piecewise fit.
function _norm_sq_grad(x::Vector{T}, ψ::Vector{T}, κ::T) where {T}
    n = length(x)
    Gψ = zeros(T, n)
    t = one(T) / (2κ)               # tail coefficient
    Z  = t * (ψ[1]^2 + ψ[n]^2)
    dZ = -(ψ[1]^2 + ψ[n]^2) / (2κ^2)
    Gψ[1] += t * ψ[1]
    Gψ[n] += t * ψ[n]
    for k in 1:n-1
        θ = κ * (x[k+1] - x[k])
        fdiag, fcross = _mass_coeffs(θ) ./ κ
        Z += fdiag * (ψ[k]^2 + ψ[k+1]^2) + 2 * fcross * ψ[k] * ψ[k+1]
        dfdiag, dfcross = _mass_coeffs_dκ(θ) ./ κ^2
        dZ += dfdiag * (ψ[k]^2 + ψ[k+1]^2) + 2 * dfcross * ψ[k] * ψ[k+1]
        Gψ[k]   += fdiag * ψ[k]   + fcross * ψ[k+1]
        Gψ[k+1] += fdiag * ψ[k+1] + fcross * ψ[k]
    end
    return Z, dZ, Gψ
end

# ∫ψ⁴ dx = ∫Q² for the hyperbolic interpolant with exponential tails, as a sum of per-interval
# closed forms, with an optional natural boundary at `lo`/`hi`. On each interval ψ solves
# ψ'' = κ²ψ, so u'² - κ²u² = E is constant and d/dx(u³u') = 3u²u'² + κ²u⁴; integrating gives
# ∫u⁴ = ([u³u']ₖ^{k+1} - 3E ∫u²)/(4κ²). The boundary and energy terms are written through
# coshθ - 1 = 2 sinh²(θ/2) and the endpoint difference q - p, keeping them accurate for
# near-coincident points (θ → 0, where the naive csch⁴ forms lose all precision) while staying
# finite for isolated points (θ → ∞). Used by select_kappa_lscv for the ∫Q² term.
#
# The derivation is local to one interval, so a piecewise-constant scale changes nothing but
# which κ each term carries. Each tail is `_tail_quartic(ψ_edge, κ_edge, Δ)` — ψ_edge⁴/(4κ_edge)
# as Δ → ∞ (an unbounded side), or the boundary-segment quartic at a finite gap; the interior
# sum is untouched by a boundary.
function _int_quartic(x::Vector{T}, ψ::Vector{T}, κ, κL::T, κR::T, lo::T, hi::T) where {T}
    n = length(x)
    Q2 = _tail_quartic(ψ[1], κL, x[1] - lo) + _tail_quartic(ψ[n], κR, hi - x[n])
    for k in 1:n-1
        κk = _kappa(κ, k)
        p, q = ψ[k], ψ[k+1]
        θ = κk * (x[k+1] - x[k])
        cs = csch(θ)
        Δ = q - p
        cm1 = 2 * sinh(θ / 2)^2                              # coshθ - 1
        boundary = κk * cs * (cm1 * (p^4 + q^4) + Δ^2 * (p^2 + p*q + q^2))  # [u³u']ₖ^{k+1}
        E = κk^2 * cs^2 * (Δ^2 - 2 * p * q * cm1)            # u'² - κ²u²
        Iseg = _interval_mass(p, q, θ) / κk                  # ∫u² over the interval
        Q2 += (boundary - 3 * E * Iseg) / (4κk^2)
    end
    return Q2
end

# ∫ψ̂⁴ over a boundary segment of gap Δ: ψ₁⁴(3θ + 2sinh 2θ + sinh(4θ)/4)/(8κ cosh⁴θ) at θ = κΔ,
# the unbounded tail's ψ₁⁴/(4κ) being its θ → ∞ limit. Rewritten in p = e^{-2θ}, cosh⁴θ =
# (1+p)⁴/(16p²), and the near-1 differences as expm1(-4θ) = p²-1, expm1(-8θ) = p⁴-1, this stays
# accurate as θ → 0 (each expm1 term individually cancellation-free, and their sum has no
# cross-term cancellation — all three contributions are non-negative) and finite well past where
# raw cosh/sinh would overflow (θ ~ 500).
function _tail_quartic(ψ1::T, κ::T, Δ::T) where {T}
    isfinite(Δ) || return ψ1^4 / (4κ)
    θ = κ * Δ
    p = exp(-2θ)
    num = 6θ * p^2 - 2p * expm1(-4θ) - expm1(-8θ) / 4
    return ψ1^4 * num / (κ * (oneunit(T) + p)^4)
end

# (dM/dκ) ψ: the κ-derivative of roughness_operator's coth/csch entries, applied to ψ. The tails are
# κ-independent and drop out.
function _dM_dκ_mul(x::Vector{T}, κ::T, ψ::Vector{T}) where {T}
    n = length(x)
    r = zeros(T, n)
    for k in 1:n-1
        h = x[k+1] - x[k]; θ = κ * h
        # d/dκ coth(θ) = -h csch²θ and d/dκ (-csch(θ)) = h csch θ coth θ nearly cancel against
        # each other as θ → 0. With csch θ coth θ - csch²θ = 1/(1 + cosh θ), the cancellation
        # is carried by the amplitude difference instead.
        a = h * csch(θ)^2
        b = h / (1 + cosh(θ))
        r[k]   += a * (ψ[k+1] - ψ[k]) + b * ψ[k+1]
        r[k+1] += a * (ψ[k] - ψ[k+1]) + b * ψ[k]
    end
    return r
end

# S(κ) = action of the fit, and dS/dln κ. ψ minimizes the potential, but S also depends on κ
# through the normalization, so the sensitivity ψ′ = dψ/dκ contributes; it solves the same
# SPD Newton system as the fit, `∇²F ψ′ = -(dM/dκ) ψ`.
function _action_and_slope(nodes::Vector{T}, w::Vector{T}, κ::T) where {T<:AbstractFloat}
    A = roughness_operator(nodes, κ)
    ψ = _solve_amplitude(A, w)
    Z, dZdκ, Gψ = _norm_sq_grad(nodes, ψ, κ)
    W = sum(w)
    S = W - κ * Z + W * log(Z)
    for i in eachindex(w, ψ)
        S -= 2 * w[i] * log(ψ[i])
    end
    H = SymTridiagonal(A.dv .+ w ./ ψ.^2, copy(A.ev))
    ψ′ = ldiv!(ldlt!(H), _dM_dκ_mul(nodes, κ, ψ))
    ψ′ .= .-ψ′                          # ψ′ = -H⁻¹ (dM/dκ) ψ
    c = W / Z - κ
    dSdκ = -Z + c * dZdκ + 2 * c * dot(Gψ, ψ′) - 2 * dot(w ./ ψ, ψ′)   # w./ψ = Mψ
    return S, κ * dSdκ
end

"""
    amplitude(d::DensityEstimate, x)

Evaluate the amplitude `ψ(x)` (so that the density is `d(x) == ψ(x)^2`) at real `x`. Zero
outside a finite `support` (see [`DensityEstimate`](@ref)); the fitted density is exactly zero
there, not merely small.
"""
amplitude(d::DensityEstimate, x::Real) = _amplitude(d, x)

# Pointwise evaluation computes in `promote_type(T, typeof(x))`, so a Float64 point on a
# Float32 fit gets a Float64 result. The mass functions (`cdf`, `quantile`, `gaussianize`)
# instead convert their argument to `T`.
function _amplitude(d::DensityEstimate{T}, x::Real) where {T}
    _check_not_nan(x)
    R = promote_type(T, typeof(x))
    xs = d.x
    n = length(xs)
    if x <= xs[1]
        x < d.lo && return zero(R)
        return _left_tail_amplitude(d.ψ[1], d.κL, x, xs[1], d.lo)
    elseif x >= xs[n]
        x > d.hi && return zero(R)
        return _right_tail_amplitude(d.ψ[n], d.κR, x, xs[n], d.hi)
    end
    return _amplitude(d, searchsortedlast(xs, x), x)    # xs[k] <= x < xs[k+1]
end

# ψ(x) in the left tail (x ≤ xs[1], lo ≤ x): the exponential decay ψ₁e^{κ(x-xs[1])} when
# unbounded, or the Neumann cosh arc ψ₁cosh(κ(x-lo))/cosh(κ(xs[1]-lo)) at a finite boundary.
# Both are ψ evaluated relative to its value at xs[1]; the finite form is exactly the unbounded
# one with the exponential's single decaying branch replaced by the cosh arc it limits to as
# lo → -∞.
_left_tail_amplitude(ψ1::T, κ::T, x::Real, x1::T, lo::T) where {T} =
    isfinite(lo) ? ψ1 * _cosh_ratio2(κ * (x - lo), κ * (x1 - lo)) : ψ1 * exp(κ * (x - x1))

# Mirror of `_left_tail_amplitude` for the right tail.
_right_tail_amplitude(ψn::T, κ::T, x::Real, xn::T, hi::T) where {T} =
    isfinite(hi) ? ψn * _cosh_ratio2(κ * (hi - x), κ * (hi - xn)) : ψn * exp(-κ * (x - xn))

# ψ(x) inside interval k, i.e. for xs[k] ≤ x ≤ xs[k+1]. Split out so a caller that already
# knows which interval x falls in — a sorted sweep — need not search for it.
function _amplitude(d::DensityEstimate{T}, k::Integer, x::Real) where {T}
    xs, ψ = d.x, d.ψ
    κ = _kappa(d, k)
    a = κ * (xs[k+1] - x)           # a, b ≥ 0 and a + b = θ
    b = κ * (x - xs[k])
    return ψ[k] * _sinh_ratio(a, a + b) + ψ[k+1] * _sinh_ratio(b, a + b)
end

"""
    logdensity(d::DensityEstimate, x)

Evaluate the log fitted density `ln Q̂(x) == 2 ln ψ(x)` at real `x`. `-Inf` outside a
finite `support` (see [`DensityEstimate`](@ref)), where the density is exactly zero.

Prefer this to `log(d(x))` or `2 * log(amplitude(d, x))`, as `logdensity` avoids
the underflow that would occur when evaluating the density directly. The result
is finite and linear in `x`. See [`logdensity_eval_gradient`](@ref) for
its derivative.

# Examples

```jldoctest
julia> d = DensityEstimate([-1.0, 0.0, 1.0], 1.0);

julia> logdensity(d, 0.0) ≈ 2 * log(amplitude(d, 0.0))
true

julia> amplitude(d, -800.0), isfinite(logdensity(d, -800.0))
(0.0, true)
```
"""
logdensity(d::DensityEstimate, x::Real) = _logdensity(d, x)

# ln Q(x) = 2 ln ψ(x), branch for branch the mirror of `_amplitude`, and the two must be kept
# so: a caller reaching either one for the same quantity may not get a different answer.
function _logdensity(d::DensityEstimate{T}, x::Real) where {T}
    _check_not_nan(x)
    R = promote_type(T, typeof(x))
    xs = d.x
    n = length(xs)
    if x <= xs[1]
        x < d.lo && return R(-Inf)
        return 2 * _log_left_tail_amplitude(d.ψ[1], d.κL, x, xs[1], d.lo)
    elseif x >= xs[n]
        x > d.hi && return R(-Inf)
        return 2 * _log_right_tail_amplitude(d.ψ[n], d.κR, x, xs[n], d.hi)
    end
    return 2 * _log_amplitude(d, searchsortedlast(xs, x), x)
end

# ln Q(t) at every position of the sorted vector `ts`, advancing through the nodes alongside
# `ts` in one pass. Evaluating pointwise would binary-search for each position, and a plug-in
# scale is realized on O(N) positions for every candidate it scores. Same branches as
# `_logdensity`, differing only in how the enclosing interval is found.
function _logdensity_sorted(d::DensityEstimate{T}, ts::AbstractVector) where {T}
    xs = d.x
    n = length(xs)
    out = Vector{T}(undef, length(ts))
    k = 1
    for i in eachindex(out, ts)
        t = T(ts[i])
        while k < n - 1 && xs[k+1] <= t
            k += 1
        end
        if t <= xs[1]
            out[i] = t < d.lo ? T(-Inf) : 2 * _log_left_tail_amplitude(d.ψ[1], d.κL, t, xs[1], d.lo)
        elseif t >= xs[n]
            out[i] = t > d.hi ? T(-Inf) : 2 * _log_right_tail_amplitude(d.ψ[n], d.κR, t, xs[n], d.hi)
        else
            out[i] = 2 * _log_amplitude(d, k, t)
        end
    end
    return out
end

# ln ψ(x) inside interval k, the log-domain mirror of `_amplitude(d, k, x)`. Both sinh arcs
# are formed in the log domain and combined with `logaddexp`, so the result holds wherever
# ψ itself underflows: at κ(xs[k+1] - xs[k]) ≳ 1490 the arcs vanish in the middle of the
# interval, which is where a log density is most worth having. ψ is strictly positive by
# construction (`_solve_amplitude` keeps it so), hence `log(ψ[k])` is always real.
function _log_amplitude(d::DensityEstimate, k::Integer, x::Real)
    xs, ψ = d.x, d.ψ
    κ = _kappa(d, k)
    a = κ * (xs[k+1] - x)           # a, b ≥ 0 and a + b = θ
    b = κ * (x - xs[k])
    return logaddexp(log(ψ[k]) + logabssinh(a), log(ψ[k+1]) + logabssinh(b)) - logabssinh(a + b)
end

# ln ψ(t) in the left tail. Every branch stays in the log domain: the unbounded arm as a
# sum rather than the logarithm of a product, the bounded arm through `logcosh`. The tails
# are where a log-density is worth having precisely because ψ itself has underflowed to
# zero, so forming the amplitude first would return -Inf over the whole region the
# logarithm exists to describe.
_log_left_tail_amplitude(ψ1::T, κ::T, x::Real, x1::T, lo::T) where {T} =
    isfinite(lo) ? log(ψ1) + logcosh(κ * (x - lo)) - logcosh(κ * (x1 - lo)) :
                   log(ψ1) + κ * (x - x1)

# Mirror of `_log_left_tail_amplitude` for the right tail.
_log_right_tail_amplitude(ψn::T, κ::T, x::Real, xn::T, hi::T) where {T} =
    isfinite(hi) ? log(ψn) + logcosh(κ * (hi - x)) - logcosh(κ * (hi - xn)) :
                   log(ψn) - κ * (x - xn)

# sinh(u)/sinh(θ) for 0 ≤ u ≤ θ, evaluated without overflow at large θ.
_sinh_ratio(u::T, θ::T) where {T} = exp(u - θ) * expm1(-2u) / expm1(-2θ)

# cosh(u)/sinh(θ) for 0 ≤ u ≤ θ, evaluated without overflow at large θ (companion to
# _sinh_ratio). With u = θ it is coth θ, also overflow-safe.
_cosh_ratio(u::T, θ::T) where {T} = -exp(u - θ) * (1 + exp(-2u)) / expm1(-2θ)

# cosh(v)/cosh(u) for 0 ≤ v ≤ u, evaluated without overflow at large u (a cosh-denominator
# companion to _sinh_ratio/_cosh_ratio, used by the boundary-segment amplitude).
_cosh_ratio2(v::T, u::T) where {T} = exp(v - u) * (oneunit(T) + exp(-2v)) / (oneunit(T) + exp(-2u))
_cosh_ratio2(v, u) = _cosh_ratio2(promote(v, u)...)

# sinh(v)/cosh(u) for 0 ≤ v ≤ u, evaluated without overflow at large u and accurate as v → 0
# (via expm1, the same treatment _sinh_ratio gives its numerator).
_sinh_ratio2(v::T, u::T) where {T} = exp(v - u) * (-expm1(-2v)) / (oneunit(T) + exp(-2u))

# ψ'(x)/ψ(x) on a sinh arc ψ ∝ ψk sinh(a) + ψk1 sinh(b), with a = κ(x_{k+1} - x) and
# b = κ(x - x_k), in units of κ. Every hyperbolic function is scaled by e^{-max(a, b)}, so
# the larger term of each sum is O(1); ψ and ψ' themselves underflow in the middle of a
# wide interval, where the ratio is still finite.
function _sinh_arc_slope(ψk::T, ψk1::T, a::T, b::T) where {T}
    m = max(a, b)
    ea, eb = exp(a - m), exp(b - m)
    num = ψk1 * eb * (1 + exp(-2b)) - ψk * ea * (1 + exp(-2a))
    den = ψk * ea * (-expm1(-2a)) + ψk1 * eb * (-expm1(-2b))
    return num / den
end
_sinh_arc_slope(ψk, ψk1, a, b) = _sinh_arc_slope(promote(ψk, ψk1, a, b)...)

# ψ'(x)/ψ(x) inside interval k.
function _amplitude_slope(d::DensityEstimate, k::Integer, x::Real)
    xs, ψ = d.x, d.ψ
    κ = _kappa(d, k)
    return κ * _sinh_arc_slope(ψ[k], ψ[k+1], κ * (xs[k+1] - x), κ * (x - xs[k]))
end

# ψ'(x)/ψ(x) in the left tail (lo ≤ x ≤ x1) with decay rate κ: the constant κ when
# unbounded, and κ tanh(κ(x - lo)) on the Neumann cosh arc of a finite boundary. Neither
# depends on ψ, which underflows deep in a tail.
_left_tail_slope(κ::T, x::T, lo::T) where {T} = isfinite(lo) ? κ * tanh(κ * (x - lo)) : κ
_left_tail_slope(κ, x, lo) = _left_tail_slope(promote(κ, x, lo)...)

# Mirror of `_left_tail_slope` for the right tail.
_right_tail_slope(κ::T, x::T, hi::T) where {T} = isfinite(hi) ? -κ * tanh(κ * (hi - x)) : -κ
_right_tail_slope(κ, x, hi) = _right_tail_slope(promote(κ, x, hi)...)

(d::DensityEstimate)(x::Real) = _amplitude(d, x)^2

# sinh(v) - v, accurate for small v where the direct difference loses all precision.
function _sinhm(v::T) where {T<:AbstractFloat}
    abs(v) >= 1 && return sinh(v) - v
    v2 = v * v
    term = v * v2 / 6
    s = term
    k = 2
    while true
        term *= v2 / ((2k) * (2k + 1))
        snew = s + term
        snew == s && return snew
        s = snew
        k += 1
    end
end

# cosh(v) - 1, accurate for small v.
_coshm(v) = 2 * sinh(v / 2)^2

# The coefficients (fdiag, fcross) of κ ∫ ψ² dt = fdiag (ψₖ² + ψₖ₊₁²) + 2 fcross ψₖψₖ₊₁ over an
# interval of width θ = κh; the physical coefficients are these divided by κ. The coth/csch
# forms stay finite as θ → ∞ but cancel catastrophically for θ < 1 (relative error ~eps/θ²),
# where the algebraically identical _sinhm/_coshm forms are used instead.
function _mass_coeffs(θ::T) where {T}
    if θ < 1
        s2 = 2 * sinh(θ)^2
        return _sinhm(2θ) / (2 * s2), (θ * _coshm(θ) - _sinhm(θ)) / s2
    end
    ct, cs = coth(θ), csch(θ)
    return (ct - θ * cs^2) / 2, cs * (θ * ct - oneunit(T)) / 2
end

# κ ∫ ψ² dt over one full inter-node interval of width θ = κh.
function _interval_mass(ψk::T, ψk1::T, θ::T) where {T}
    fdiag, fcross = _mass_coeffs(θ)
    return fdiag * (ψk^2 + ψk1^2) + 2 * fcross * ψk * ψk1
end

# The θ-derivatives (fd′, fc′) of `_mass_coeffs`. For θ < 1 the coth/csch forms cancel
# (relative error ~eps/θ²); the _sinhm/_coshm numerators are algebraically identical.
function _mass_coeffs_dθ(θ::T) where {T}
    if θ < 1
        s3 = sinh(θ)^3
        return (θ * _coshm(θ) - _sinhm(θ)) / s3, (_sinhm(2θ) - θ / 2 * _coshm(2θ)) / (2 * s3)
    end
    ct, cs = coth(θ), csch(θ)
    return cs^2 * (θ * ct - 1), cs * (2ct - θ * (ct^2 + cs^2)) / 2
end

# (θfd′ - fd, θfc′ - fc), so that the κ-derivatives of the physical coefficients at fixed width
# h are d(f/κ)/dκ = (θf′ - f)/κ². These are O(θ³) while θf′ and f are O(θ), so for θ < 1 they
# come from `_mass_slope_series`.
function _mass_coeffs_dκ(θ::T) where {T}
    if θ < 1
        Nd, Nc = _mass_slope_series(θ)
        c = θ^3 / (2 * (sinh(θ) / θ)^3)
        return c * Nd, -c * Nc
    end
    fd, fc = _mass_coeffs(θ)
    fd′, fc′ = _mass_coeffs_dθ(θ)
    return θ * fd′ - fd, θ * fc′ - fc
end

# θ⁻⁶ N_d and θ⁻⁶ N_c, where θfd′ - fd = N_d/(2 sinh³θ) and θfc′ - fc = -N_c/(2 sinh³θ) with
#     N_d = 2θ² cosh θ - θ sinh θ - sinh²θ cosh θ,
#     N_c = 2θ² + θ² sinh²θ - θ sinh θ cosh θ - sinh²θ.
# Both are O(θ⁶), and from θ⁶ on their Taylor coefficients each have one sign:
#     N_d = -Σ_{m≥3} ((9^m - 1)/4 - 8m² + 6m) θ^{2m}/(2m)!,
#     N_c =  Σ_{m≥3} (2m + 1)(m - 2)/4 · (2θ)^{2m}/(2m)!,
# so summing them loses nothing to cancellation.
function _mass_slope_series(θ::T) where {T}
    θ2 = θ^2
    t = one(T) / 720                    # θ^{2(m-3)}/(2m)! at m = 3
    p4, p9 = T(64), T(729)              # 4^m, 9^m
    Nd = Nc = zero(T)
    m = 3
    while true
        Ndnew = Nd - ((p9 - 1) / 4 - 8m^2 + 6m) * t
        Ncnew = Nc + (2m + 1) * (m - 2) * p4 / 4 * t
        Ndnew == Nd && Ncnew == Nc && return Nd, Nc
        Nd, Nc = Ndnew, Ncnew
        t *= θ2 / ((2m + 1) * (2m + 2))
        p4 *= 4
        p9 *= 9
        m += 1
    end
end

# Mass of the tail, or of the boundary segment on a finite support, beyond each outermost node.
_left_end_mass(d::AbstractDensityEstimate) = _tail_mass(d.ψ[1], first(_tail_rates(d)), d.x[1] - d.lo)
_right_end_mass(d::AbstractDensityEstimate) = _tail_mass(d.ψ[end], last(_tail_rates(d)), d.hi - d.x[end])

# Mass of the interior interval k, [x[k], x[k+1]].
function _cell_mass(d::DensityEstimate, k::Integer)
    κ = _kappa(d, k)
    return _interval_mass(d.ψ[k], d.ψ[k+1], κ * (d.x[k+1] - d.x[k])) / κ
end

# ψ̂(v)² integrated from the wall (v = 0) out to v, for the boundary field ψ̂(s) = cosh(κs)/cosh(u)
# on a segment of width u = κΔ (Neumann at the wall, node value ψ_node at s = Δ); v = κs ∈ [0, u].
# Both terms are non-negative for v ≥ 0, so — unlike the interior `_segmass` — this needs no
# small-u cancellation treatment; it reduces to `_tail_mass` at v = u.
function _boundary_mass_from_wall(ψ_node::T, κ::T, v::T, u::T) where {T}
    return ψ_node^2 * (v * _sech2_stable(u) + _cosh_ratio2(v, u) * _sinh_ratio2(v, u)) / (2κ)
end

# The complementary piece of `_boundary_mass_from_wall`: ψ̂² integrated from v out to the node
# (v = u). Written through the identity sinh(2u) - sinh(2v) = 2cosh(u+v)sinh(u-v) so it stays
# cancellation-free as v → u, unlike computing it as `_tail_mass - _boundary_mass_from_wall`
# (a difference of two nearly equal quantities there). Expanding cosh(u+v)sinh(u-v)/cosh(u)² in
# p = e^{-2u} and δ = u - v ≥ 0 collapses both e^{2(v-u)} - 1 and e^{-2(v+u)} - e^{-4u} to the
# same factor `nA` = 1 - e^{-2δ}, evaluated through expm1 for a δ of any size (no cancellation
# as δ → 0, no overflow as u → ∞ — every exponent stays ≤ 0).
function _boundary_mass_from_node(ψ_node::T, κ::T, v::T, u::T) where {T}
    p = exp(-2u)
    nA = -expm1(-2 * (u - v))              # 1 - exp(-2(u-v)), δ = u - v ≥ 0 keeps this safe
    q = p * exp(-2v)                       # exp(-2(u+v))
    R = nA * (oneunit(T) + q) / (oneunit(T) + p)^2   # cosh(u+v)sinh(u-v)/cosh(u)²
    return ψ_node^2 * ((u - v) * _sech2_stable(u) + R) / (2κ)
end

# Node cumulatives from both ends: F[k] = mass below x[k], G[k] = mass above x[k], and the
# grand total. Each is accumulated from its own end, so both carry full relative precision;
# forming total - F[k] instead would cancel wherever F[k] ≈ total, and it is the smaller of
# the two sides that Φ⁻¹ needs at full precision (see _gaussianize). ψ is normalized so the
# total is 1 up to roundoff; the mass functions divide by this total rather than assuming 1,
# which pins cdf(d, ±Inf) to exactly 0 and 1 and keeps the CDF monotone across the last node.
# The inner constructors store the result as `d.F`, `d.G` and `d.total`, so every evaluation
# reads the tables instead of reassembling them at O(length(d.x)) cost.
function _node_cumulatives(d::AbstractDensityEstimate)
    n = length(d.x)
    m = eltype(d.x)[_cell_mass(d, k) for k in 1:n-1]
    F = similar(d.x)
    F[1] = _left_end_mass(d)
    for k in 1:n-1
        F[k+1] = F[k] + m[k]
    end
    G = similar(d.x)
    G[n] = _right_end_mass(d)
    for k in n-1:-1:1
        G[k] = G[k+1] + m[k]
    end
    return F, G, F[n] + G[n]
end

# Unnormalized cumulative mass ∫_{lo}^{x} ψ² dt, from the node cumulatives `d.F`: zero at or below
# `lo` (an unreachable comparison when `lo = -Inf`) and the grand total at or above `hi`. The
# tails are elementary exponential integrals when unbounded. A finite boundary segment is
# integrated from whichever end (wall or node) bounds the smaller part of the segment's mass,
# so the partial mass is never a difference of nearly equal quantities. Choosing by mass rather
# than by distance matters in a steep segment, where nearly all the mass sits near the node and
# a point nearer the node can still have only a sliver of mass between it and the wall. The
# CDF stays continuous through the boundary node, as `_cdf_mass_interior` keeps it at interior
# nodes. `x` is converted to the fit's element type.
function _cdf_mass(d::AbstractDensityEstimate, xin::Real)
    xs, ψ, F, G = d.x, d.ψ, d.F, d.G
    T = eltype(F)
    n = length(xs)
    κL, κR = _tail_rates(d)
    _check_not_nan(xin)
    x = T(xin)
    if x <= xs[1]
        isfinite(d.lo) || return ψ[1]^2 / (2κL) * exp(2κL * (x - xs[1]))
        x <= d.lo && return zero(T)
        v, u = κL * (x - d.lo), κL * (xs[1] - d.lo)
        mw = _boundary_mass_from_wall(ψ[1], κL, v, u)   # F[1] is the segment's mass
        return mw <= F[1] / 2 ? mw : F[1] - _boundary_mass_from_node(ψ[1], κL, v, u)
    elseif x >= xs[n]
        isfinite(d.hi) || return F[n] + ψ[n]^2 / (2κR) * (-expm1(-2κR * (x - xs[n])))
        x >= d.hi && return F[n] + G[n]
        vp, u = κR * (d.hi - x), κR * (d.hi - xs[n])
        mn = _boundary_mass_from_node(ψ[n], κR, vp, u)  # G[n] is the segment's mass
        return mn <= G[n] / 2 ? F[n] + mn : F[n] + (G[n] - _boundary_mass_from_wall(ψ[n], κR, vp, u))
    end
    k = searchsortedlast(xs, x)         # xs[k] ≤ x < xs[k+1]
    return _cdf_mass_interior(d, k, x)
end

# Unnormalized cumulative mass at x within interval k (xs[k] ≤ x ≤ xs[k+1]). The partial
# mass is integrated from the nearer node — subtracting from F[k+1] when x lies in the
# right half — so its absolute error vanishes toward both nodes and the CDF stays
# continuous and monotone through every node. The subtraction keeps full relative
# precision only while its result is at least F[k+1]/2. Since the result is at least F[k],
# that is guaranteed when F[k] ≥ F[k+1]/2; otherwise (a cell holding more mass than lies
# below it, typically wide and steep next to a light node) a result below F[k+1]/2 is
# recomputed as F[k] plus the integral from the left node across the whole of [xs[k], x].
function _cdf_mass_interior(d::DensityEstimate, k::Int, x::Real)
    xs, ψ, F = d.x, d.ψ, d.F
    κ = _kappa(d, k)
    a = κ * (xs[k+1] - x)               # a, b ≥ 0 and a + b = θ
    b = κ * (x - xs[k])
    θ = a + b
    b <= θ / 2 && return F[k] + _segmass(ψ[k], ψ[k+1], a, b, θ) / κ
    m = F[k+1] - _segmass(ψ[k+1], ψ[k], b, a, θ) / κ
    (2 * F[k] >= F[k+1] || 2 * m >= F[k+1]) && return m
    return F[k] + _segmass(ψ[k], ψ[k+1], a, b, θ) / κ
end

# Unnormalized mass above x within interval k (xs[k] ≤ x ≤ xs[k+1]): the complement-side
# mirror of _cdf_mass_interior, with the same guard against cancellation in the subtraction
# from G[k]. `d.G` holds the mass above each node (_node_cumulatives).
function _ccdf_mass_interior(d::DensityEstimate, k::Int, x::Real)
    xs, ψ, G = d.x, d.ψ, d.G
    κ = _kappa(d, k)
    a = κ * (xs[k+1] - x)               # a, b ≥ 0 and a + b = θ
    b = κ * (x - xs[k])
    θ = a + b
    a <= θ / 2 && return G[k+1] + _segmass(ψ[k+1], ψ[k], b, a, θ) / κ
    m = G[k] - _segmass(ψ[k], ψ[k+1], a, b, θ) / κ
    (2 * G[k+1] >= G[k] || 2 * m >= G[k]) && return m
    return G[k+1] + _segmass(ψ[k+1], ψ[k], b, a, θ) / κ
end

# Unnormalized mass above x, ∫_x^{hi} ψ² dt: the complement of `_cdf_mass`, with a boundary
# segment integrated from whichever end bounds the smaller part of its mass, as there. The
# transport feeds Φ⁻¹ the smaller of `_cdf_mass`/`_ccdf_mass`, and `_quantile` inverts the
# right segment through this function, so within a boundary segment the near-wall side —
# computed here from the wall rather than as `total - _cdf_mass` — keeps full relative
# precision where that difference would cancel.
function _ccdf_mass(d::AbstractDensityEstimate, xin::Real)
    xs, ψ, F, G = d.x, d.ψ, d.F, d.G
    T = eltype(G)
    n = length(xs)
    κL, κR = _tail_rates(d)
    _check_not_nan(xin)
    x = T(xin)
    if x >= xs[n]
        isfinite(d.hi) || return ψ[n]^2 / (2κR) * exp(-2κR * (x - xs[n]))
        x >= d.hi && return zero(T)
        vp, u = κR * (d.hi - x), κR * (d.hi - xs[n])
        mw = _boundary_mass_from_wall(ψ[n], κR, vp, u)  # G[n] is the segment's mass
        return mw <= G[n] / 2 ? mw : G[n] - _boundary_mass_from_node(ψ[n], κR, vp, u)
    elseif x <= xs[1]
        isfinite(d.lo) || return G[1] + ψ[1]^2 / (2κL) * (-expm1(2κL * (x - xs[1])))
        x <= d.lo && return G[1] + F[1]
        v, u = κL * (x - d.lo), κL * (xs[1] - d.lo)
        mn = _boundary_mass_from_node(ψ[1], κL, v, u)   # F[1] is the segment's mass
        return mn <= F[1] / 2 ? G[1] + mn : G[1] + (F[1] - _boundary_mass_from_wall(ψ[1], κL, v, u))
    end
    k = searchsortedlast(xs, x)         # xs[k] ≤ x < xs[k+1]
    return _ccdf_mass_interior(d, k, x)
end

# ∫₀ʷ ψ̂(u)² du for the unit-coordinate interval field ψ̂(u) = (p sinh(θ-u) + q sinh(u))/sinh(θ),
# with 0 ≤ w ≤ θ and arem = θ - w; the physical mass over [x[k], x[k]+w/κ] is _segmass/κ.
# Two algebraically identical forms of the exact antiderivative:
# - θ < 1: expanded per-power integrals ∫sinh²(θ-u), ∫sinh(θ-u)sinh(u), ∫sinh²(u), written
#   through _sinhm/_coshm so the small-θ cancellation (relative error ~eps/θ² in the naive
#   coth/csch forms) never occurs;
# - θ ≥ 1: ψ̂'' = ψ̂ makes C = ψ̂'² - ψ̂² constant and (ψ̂ψ̂')' = 2ψ̂² + C, so
#   ∫ψ̂² du = (Δ(ψ̂ψ̂') - C·w)/2, with C and ψ̂' written through coth/csch/_sinh_ratio/
#   _cosh_ratio so everything stays finite for isolated points (large θ).
function _segmass(p, q, arem, w, θ)
    if θ < 1
        Ipp = 2 * sinh((θ + arem) / 2)^2 * sinh(w) + _sinhm(w)      # ∫₀ʷ sinh²(θ-u) du
        Ipq = w * _coshm(θ) - _coshm(arem) * sinh(w) - _sinhm(w)    # ∫₀ʷ sinh(θ-u) sinh(u) du
        Iqq = _sinhm(2w) / 2                                        # ∫₀ʷ sinh²(u) du
        return (p^2 * Ipp + 2 * p * q * Ipq + q^2 * Iqq) / (2 * sinh(θ)^2)
    end
    ct, cs = coth(θ), csch(θ)
    C = cs^2 * (p^2 + q^2) - 2 * p * q * cs * ct    # ψ̂'² - ψ̂², constant on the interval
    ψ0′ = q * cs - p * ct                           # ψ̂'(0)
    ψw  = p * _sinh_ratio(arem, θ) + q * _sinh_ratio(w, θ)
    ψw′ = q * _cosh_ratio(w, θ) - p * _cosh_ratio(arem, θ)
    return (ψw * ψw′ - p * ψ0′ - C * w) / 2
end

cdf(d::AbstractDensityEstimate, x::Real) = _cdf_mass(d, x) / d.total

"""
    cdf(d::DensityEstimate, x)

Cumulative distribution function of the fitted density, `F(x) = ∫_a^x Q(t) dt` with `Q = ψ²`
and `a` the fit's left support endpoint (`-Inf` unless [`DensityEstimate`](@ref) was given a
finite `support`), evaluated in closed form (no quadrature). `F` is nondecreasing with
`F(a) == 0` and `F(b) == 1` exactly, and outside a finite `[a, b]` support is exactly `0` at or
below `a` and exactly `1` at or above `b`.

See [`quantile`](@ref Statistics.quantile) for the inverse.

# Examples
```jldoctest
julia> d = DensityEstimate([0.0], 0.5);   # a Laplace density centered at 0

julia> cdf(d, 0.0)
0.5

julia> quantile(d, 0.5)
0.0
```

# Extended help

The tails (or, on a finite support, the two boundary segments) are pure exponentials or `cosh`
arcs; on each inter-node interval `ψ'' = κ²ψ` makes `ψ'² - κ²ψ²` constant, which yields the
exact antiderivative of the hyperbolic interpolant. The `F(a) == 0`/`F(b) == 1` endpoints are
exact because the few ulps of normalization roundoff in the fitted amplitudes are absorbed by a
global rescaling.
"""
cdf(d::DensityEstimate, x)

Statistics.quantile(d::AbstractDensityEstimate, q::Real) = _quantile(d, q)

"""
    quantile(d::DensityEstimate, q)

Quantile function of the fitted density, the inverse of [`cdf`](@ref):
`cdf(d, quantile(d, q)) ≈ q` for `q ∈ [0, 1]`, with `quantile(d, 0) == a` and
`quantile(d, 1) == b` (the fit's support endpoints, `-Inf`/`Inf` unless
[`DensityEstimate`](@ref) was given a finite `support`); `q` outside `[0, 1]` (including `NaN`)
throws a `DomainError`. Extends `Statistics.quantile`.

In an unbounded tail the inversion is in closed form; on interior intervals, and on a finite
boundary segment (transcendental there), a Newton iteration on the closed-form CDF, bracketed
by the enclosing nodes or by the wall and the outermost node, converges to floating-point
accuracy. The right side is solved through `1 - q`, so upper quantiles lose no more precision
than `q` itself carries.
"""
Statistics.quantile(d::DensityEstimate, q)

# Sampling by inversion, `quantile(d, q)` at uniform `q`, for any fit; each draw costs one
# inversion. `q = 0` (probability 2⁻⁵³ in Float64) is redrawn: on an unbounded left side its
# quantile is `-Inf`, which is not a sample.
Base.eltype(::Type{<:DensityEstimate{T}}) where {T} = T
function Random.rand(rng::AbstractRNG, sp::SamplerTrivial{<:AbstractDensityEstimate})
    d = sp[]
    T = eltype(d)
    while true
        q = rand(rng, T)
        q > 0 && return _quantile(d, q)
    end
end

"""
    minimum(d::AbstractDensityEstimate)
    maximum(d::AbstractDensityEstimate)
    extrema(d::AbstractDensityEstimate)

Lower and upper ends of the support of the fitted density: the `support` the fit was given,
`-Inf`/`Inf` on an unbounded side.
"""
Base.minimum(d::AbstractDensityEstimate) = d.lo
Base.maximum(d::AbstractDensityEstimate) = d.hi
Base.extrema(d::AbstractDensityEstimate) = (minimum(d), maximum(d))

# A fit is a scalar in broadcasting, so `cdf.(d, xs)` evaluates `d` at each element of `xs`.
Base.broadcastable(d::AbstractDensityEstimate) = Ref(d)

DensityInterface.DensityKind(::AbstractDensityEstimate) = HasDensity()
DensityInterface.logdensityof(d::AbstractDensityEstimate, x::Real) = logdensity(d, x)
DensityInterface.densityof(d::AbstractDensityEstimate, x::Real) = d(x)

# Closed-form quantile in an unbounded exponential tail: target = ψ₁²/(2κ) e^{2κ(x-x₁)}.
_left_tail_quantile(ψ1::T, κ::T, x1::T, target::T) where {T} = x1 + log(2κ * target / ψ1^2) / (2κ)

# Mirror of `_left_tail_quantile` for the right tail, solved through the upper-tail
# probability cq = 1 - q so upper quantiles keep the precision cq carries.
_right_tail_quantile(ψn::T, κ::T, xn::T, total::T, cq::T) where {T} =
    xn - log(2κ * (total * cq) / ψn^2) / (2κ)

# Safeguarded Newton (bisection fallback) for the `y` solving `massfun(y) == target` on
# `[lo, hi]`, where `massfun` is increasing with derivative `ψ(y)²`.
function _invert_mass(d::AbstractDensityEstimate, massfun, lo::T, hi::T, y::T, target::T) where {T}
    for _ in 1:200
        r = massfun(y) - target
        r == 0 && return y
        r < 0 ? (lo = y) : (hi = y)
        ynew = y - r / _amplitude(d, y)^2       # Newton: the CDF's derivative is ψ²
        # A Newton step that rounds to zero means y is converged. Test it before the bracket:
        # y is itself an endpoint of the updated bracket, so the strict test would reject it.
        ynew == y && return y
        lo < ynew < hi || (ynew = (lo + hi) / 2)  # bisect when Newton leaves the bracket
        ynew == y && return y                     # the bracket has closed to adjacent floats
        y = ynew
    end
    throw(ConvergenceError("quantile", "safeguarded Newton did not converge in 200 iterations at target = $target; please report this"))
end

# `cq` is the upper-tail probability 1 - q. Callers that know it more precisely than `1 - q`
# (`_ungaussianize`, from ln(1 - Φ(y))) pass it; for q ≥ 1/2 the default is exact. The right
# side is solved through it: the unbounded tail in closed form, and the boundary segment and
# any interior cell with q > cq by inverting the mass above y (`_ccdf_mass`,
# `_ccdf_mass_interior`) against the target cq·total. That mass is accumulated from the right,
# so it keeps full relative precision where `total - _cdf_mass` would cancel to the
# resolution of `total`. It is negated so that `_invert_mass` sees an increasing function.
function _quantile(d::AbstractDensityEstimate, qin::Real, cqin::Real = 1 - qin)
    0 <= qin <= 1 || throw(DomainError(qin, "quantile is defined only for probabilities 0 ≤ q ≤ 1"))
    xs, ψ, F, G, total = d.x, d.ψ, d.F, d.G, d.total
    T = eltype(F)
    n = length(xs)
    κL, κR = _tail_rates(d)
    q, cq = T(qin), T(cqin)
    target = q * total
    if target <= F[1]
        isfinite(d.lo) || return _left_tail_quantile(ψ[1], κL, xs[1], target)
        # F[1] == 0 only at a zero-width boundary segment (xs[1] == d.lo), where target == 0
        # too (target ≤ F[1] and target ≥ 0); the linear start is meaningless there, but any
        # start converges immediately since `_cdf_mass(d, d.lo) == 0 == target` exactly.
        y = F[1] > 0 ? d.lo + (target / F[1]) * (xs[1] - d.lo) : d.lo
        return _invert_mass(d, t -> _cdf_mass(d, t), d.lo, xs[1], y, target)
    elseif target >= F[n]
        isfinite(d.hi) || return _right_tail_quantile(ψ[n], κR, xs[n], total, cq)
        ctarget = cq * total                # = total - target, precise as q → 1
        y = G[n] > 0 ? d.hi - (ctarget / G[n]) * (d.hi - xs[n]) : d.hi
        return _invert_mass(d, t -> -_ccdf_mass(d, t), xs[n], d.hi, y, -ctarget)
    end
    q <= cq ? _quantile_interior_lower(d, target) : _quantile_interior_upper(d, cq * total)
end

# The interior cases of `_quantile`, kept in separate functions so that each closure captures a
# cell index that is assigned once (a reassigned captured variable is boxed).
function _quantile_interior_lower(d::AbstractDensityEstimate, target)
    xs, F = d.x, d.F
    k = searchsortedlast(F, target)     # F[k] ≤ target < F[k+1], so 1 ≤ k < n
    lok, hik = xs[k], xs[k+1]
    y = lok + (target - F[k]) / (F[k+1] - F[k]) * (hik - lok)  # linear-in-mass start
    return _invert_mass(d, t -> _cdf_mass_interior(d, k, t), lok, hik, y, target)
end

# Upper half: solved against ctarget = cq·total through the mass above y, as for the right
# segment.
function _quantile_interior_upper(d::AbstractDensityEstimate, ctarget)
    xs, G = d.x, d.G
    # G[k] ≥ ctarget > G[k+1]
    k = clamp(searchsortedlast(G, ctarget, Base.Order.Reverse), 1, length(xs) - 1)
    lok, hik = xs[k], xs[k+1]
    y = hik - (ctarget - G[k+1]) / (G[k] - G[k+1]) * (hik - lok)  # linear-in-mass start
    return _invert_mass(d, t -> -_ccdf_mass_interior(d, k, t), lok, hik, y, -ctarget)
end

# ln Φ(t), through erfcx so it stays exact into the deep lower tail (Φ itself underflows
# below t ≈ -38.6): ln Φ(t) = ln(erfcx(-t/√2)/2) - t²/2. Use only for t < ∞ (the two
# infinite terms collide at t = +∞); callers needing the upper tail pass -t.
_logΦ(t::T) where {T} = log(erfcx(-t / sqrt(T(2))) / 2) - t^2 / 2

# Φ⁻¹(p) from ℓ = ln p (the upper half enters by symmetry: Φ⁻¹(p) = -Φ⁻¹-from-ln(1-p)).
# Wherever p = e^ℓ is comfortably a normal float, erfcinv delivers full precision directly.
# Deeper, solve ln Φ(-t) = ℓ for t > 0 by Newton in log space: _logΦ is exact there, its
# derivative is -φ(t)/Φ(-t) = -√(2/π)/erfcx(t/√2), and the asymptotic seed
# t² ≈ -2ℓ - ln(2π·(-2ℓ)) puts the quadratically-convergent iteration within a few steps
# of machine precision. Non-convergence indicates a bug, not a data property: fail fast.
function _probit_from_log(ℓ::T) where {T<:AbstractFloat}
    ℓ == T(-Inf) && return T(-Inf)
    ℓ > log(floatmin(T)) / 2 && return -sqrt(T(2)) * erfcinv(2 * exp(ℓ))
    u = -2 * ℓ
    t = sqrt(u - log(2 * T(π) * u))
    for _ in 1:12
        f = log(erfcx(t / sqrt(T(2)))) - t^2 / 2 - log(T(2)) - ℓ
        Δ = f * erfcx(t / sqrt(T(2))) * sqrt(T(π) / 2)
        t += Δ
        abs(Δ) <= 2 * eps(t) && return -t
    end
    throw(ConvergenceError("gaussianize", "the probit Newton iteration did not converge in 12 iterations at ln(p) = $ℓ; please report this"))
end

# Φ⁻¹ fed from whichever tail mass is smaller, so it always receives a probability at full
# relative precision.
_probit_from_masses(mlo::T, mhi::T, total::T) where {T} =
    mlo <= mhi ? -sqrt(T(2)) * erfcinv(2 * (mlo / total)) :
                  sqrt(T(2)) * erfcinv(2 * (mhi / total))

# y = Φ⁻¹(F(x)), assembled from whichever side of the distribution is smaller so Φ⁻¹ is
# always fed a probability at full relative precision. Beyond the extreme nodes on an unbounded
# side ln F and ln(1-F) are exactly linear in x (the tail is a pure exponential with the rates
# `_tail_rates(d)`), so the composition runs in log space and never saturates for finite x.
# Beyond a finite wall the "tail" is the bounded cosh-arc boundary segment: its mass is finite,
# so the same node-cumulative assembly as the interior applies, and past the wall F is 0 (or 1),
# giving the honest y = ∓Inf for a compact-support density.
function _gaussianize(d::AbstractDensityEstimate, xin::Real)
    xs, ψ, total = d.x, d.ψ, d.total
    T = typeof(total)
    n = length(xs)
    κL, κR = _tail_rates(d)
    _check_not_nan(xin)
    x = T(xin)
    if x <= xs[1]
        isinf(d.lo) &&
            return _probit_from_log(2 * κL * (x - xs[1]) + log(ψ[1]^2 / (2 * κL * total)))
        x <= d.lo && return T(-Inf)
    elseif x >= xs[n]
        isinf(d.hi) &&
            return -_probit_from_log(-2 * κR * (x - xs[n]) + log(ψ[n]^2 / (2 * κR * total)))
        x >= d.hi && return T(Inf)
    end
    return _probit_from_masses(_cdf_mass(d, x), _ccdf_mass(d, x), total)
end

# x = F⁻¹(Φ(y)), branched on the sign of y so the smaller of Φ(y), 1-Φ(y) is the one
# computed (each exactly, via _logΦ). On an unbounded side the deep-tail inversion is the
# closed form of _quantile written in log space, so deep |y| maps to the finite x whose
# tail mass matches e^{lnΦ} rather than saturating to ±Inf. A finite wall needs no such
# guard — the support is bounded, so `_quantile` inverts directly and maps y = ∓Inf (Φ = 0
# or 1 underflowed) to the wall itself.
function _ungaussianize(d::AbstractDensityEstimate, yin::Real)
    xs, ψ, F, G, total = d.x, d.ψ, d.F, d.G, d.total
    T = typeof(total)
    n = length(xs)
    κL, κR = _tail_rates(d)
    _check_not_nan(yin)
    y = T(yin)
    if y <= 0
        ℓ = _logΦ(y)
        if isinf(d.lo) && ℓ <= log(F[1] / total)    # left tail: ln F is linear in x
            return xs[1] + (ℓ + log(2 * κL * total / ψ[1]^2)) / (2 * κL)
        end
        return _quantile(d, exp(ℓ))
    else
        r = _logΦ(-y)                               # ln(1 - Φ(y)), exact
        if isinf(d.hi) && r <= log(G[n] / total)    # right tail: ln(1-F) is linear in x
            return xs[n] - (r + log(2 * κR * total / ψ[n]^2)) / (2 * κR)
        end
        return _quantile(d, 1 - exp(r), exp(r))
    end
end

# (y, ln y′): the map together with the log-Jacobian ln Q̂(x) - ln φ(y), Q̂ = ψ²/total.
# Interior, both terms are moderate (|y| stays small wherever a node cumulative bounds the
# probability away from 0 and 1) and the direct formula is accurate. On an unbounded side the
# direct form cancels — ln Q̂ and -y²/2 grow together — but there F = Q̂/(2κ_edge) exactly, and
# eliminating ln Q̂ via ln φ(y) = ln Φ(y) + ln(φ/Φ) collapses the log-Jacobian to
# ln(2κ_edge) + ½ln(π/2) + ln erfcx(∓y/√2), cancellation-free. Inside a finite boundary segment
# the density stays finite at the wall (a natural boundary leaves ψ free), so ln Q̂ does not
# diverge and no cancellation arises — the direct formula is used. Past the wall the density is
# zero: the log-Jacobian is -Inf (the map is constant there), the compact-support convention.
function _gaussianize_logjac(d::AbstractDensityEstimate, xin::Real)
    xs = d.x
    T = typeof(d.total)
    n = length(xs)
    κL, κR = _tail_rates(d)
    y = _gaussianize(d, xin)
    x = T(xin)
    if x <= xs[1]
        isinf(d.lo) &&
            return (; y, logjac = log(2 * κL) + log(T(π) / 2) / 2 + log(erfcx(-y / sqrt(T(2)))))
        x <= d.lo && return (; y, logjac = T(-Inf))
    elseif x >= xs[n]
        isinf(d.hi) &&
            return (; y, logjac = log(2 * κR) + log(T(π) / 2) / 2 + log(erfcx(y / sqrt(T(2)))))
        x >= d.hi && return (; y, logjac = T(-Inf))
    end
    logq = _logdensity(d, x) - log(d.total)
    return (; y, logjac = logq + y^2 / 2 + log(2 * T(π)) / 2)
end

"""
    gaussianize(d::AbstractDensityEstimate, x)

Gaussianizing transport of the fitted density: `y = Φ⁻¹(F(x))`, where `F = `[`cdf`](@ref)`(d, ·)`
and `Φ` is the standard normal CDF. If `X` is distributed with density `d`, then
`gaussianize(d, X)` is distributed `N(0, 1)`; the map is strictly increasing wherever the
density is positive.

On an unbounded side the composition is evaluated in log space beyond the extreme node
(`ln F` is exactly linear in `x` in the exponential tail), so `y` remains finite and fully
accurate arbitrarily far into the tail instead of saturating to `±Inf` where `F` would round
to 0 or 1. `gaussianize(d, ±Inf) == ±Inf`, and `NaN` throws a `DomainError`.

A finite `support` (see [`DensityEstimate`](@ref)) and a per-interval `κ` are both handled.
Because a compact-support density has no bijection onto all of `ℝ`, the walls map to infinity
and points beyond them are given the honest values: `gaussianize(d, x) == -Inf` for `x` at or
below a finite left endpoint and `+Inf` at or above a finite right endpoint (there
`cdf(d, x)` is exactly `0` or `1`). [`ungaussianize`](@ref) inverts this — a saturated `y`
maps back to the wall.

See [`ungaussianize`](@ref) for the inverse, [`gaussianize_logjacobian`](@ref) for the map
together with its log-Jacobian, whose exponential is the map's derivative.

# Examples
```jldoctest
julia> d = DensityEstimate([0.0], 0.5);   # a Laplace density centered at 0

julia> y = gaussianize(d, 1.0);

julia> ungaussianize(d, y) ≈ 1.0
true
```
"""
gaussianize(d::AbstractDensityEstimate, x::Real) = _gaussianize(d, x)

"""
    ungaussianize(d::AbstractDensityEstimate, y)

Inverse of [`gaussianize`](@ref): `x = F⁻¹(Φ(y))` with `F⁻¹ = `[`quantile`](@ref Statistics.quantile)`(d, ·)`,
so that `ungaussianize(d, gaussianize(d, x)) ≈ x`. If `Y ~ N(0, 1)`, then
`ungaussianize(d, Y)` is distributed with density `d` — drawing standard normal variates
and mapping them through this function samples from the fit.

The probability `Φ(y)` is never formed on its losing side: for `y > 0` the complement
`1 - Φ(y)` is computed directly, and deep tails run in log space through the closed-form
exponential-tail inverse, so large `|y|` maps to the corresponding finite `x` rather than
saturating to `±Inf`. `ungaussianize(d, ±Inf) == ±Inf`, and `NaN` throws a `DomainError`.
"""
ungaussianize(d::AbstractDensityEstimate, y::Real) = _ungaussianize(d, y)

"""
    gaussianize_logjacobian(d::AbstractDensityEstimate, x) -> (; y, logjac)

The Gaussianizing map `y = `[`gaussianize`](@ref)`(d, x)` together with the log-Jacobian
of the transport,

    logjac = ln Q̂(x) - ln φ(y),

where `Q̂` is the fitted density and `φ` the standard normal density — i.e.
`logjac = ln(dy/dx)`, the change-of-variables term that makes
`ln φ(y) + logjac` the exact log-likelihood `ln Q̂(x)`. In an unbounded tail the difference is
evaluated in a cancellation-free closed form (both terms grow like `y²/2`), so `logjac`
stays accurate arbitrarily far out. Past a finite `support` wall the density is zero and
`logjac == -Inf` (the map is constant there); inside a boundary segment the density is finite,
so `logjac` is finite and grows toward the wall as the map stretches it to infinity.

Broadcasting, `r = gaussianize_logjacobian.(d, xs)`, returns an array of `(; y, logjac)`
pairs; `first.(r)` and `last.(r)` unpack it. The plain derivative `dy/dx` is `exp(logjac)`.
"""
gaussianize_logjacobian(d::AbstractDensityEstimate, x::Real) = _gaussianize_logjac(d, x)

"""
    action(d::DensityEstimate) -> S

Classical action `S[ψ_cl] = N - λ - Σᵢ wᵢ ln Q(xᵢ)` (Eq. 10) of the fitted density,
where `N = Σ wᵢ`. Used by [`select_kappa_sensitivity`](@ref).
"""
function action(d::DensityEstimate)
    N = sum(d.w)
    return N - d.λ - sum(d.w .* log.(d.ψ.^2))
end

"""
    entropy(d::DensityEstimate) -> Ĥ

Plug-in estimate of the differential entropy `H(Q) = -∫ Q ln Q dx` of the fitted density,

    Ĥ = -(1/W) Σᵢ wᵢ ln Q(xᵢ) = -(2/W) Σᵢ wᵢ ln ψ(xᵢ),

where `W = Σᵢ wᵢ`. Evaluating `ln Q` at the data rather than integrating it exactly avoids a
second quadrature pass and is consistent (`Ĥ → H(Q)` as the sample grows), but it is biased at
small `W`: for the one-point fit (a Laplace density with rate `2κ`), `Ĥ = -ln κ` against the
exact `H = 1 - ln κ`. An [`IntervalDensityEstimate`](@ref) extends the sum to interval
observations. See [`negentropy`](@ref).
"""
function entropy(d::DensityEstimate)
    W = sum(d.w)
    return -2 * sum(d.w .* log.(d.ψ)) / W
end

"""
    negentropy(d::AbstractDensityEstimate) -> J

Negentropy of the fitted density: the entropy deficit relative to the Gaussian with the same
mean and variance,

    J = ½ ln(2π e σ²) - Ĥ,

where `σ²` is the fit's variance [`var`](@ref Statistics.var)`(d)` (computed from the nodal
amplitudes, for any scale profile and support) and `Ĥ` is [`entropy`](@ref)`(d)`. `J ≥ 0` in the large-sample limit (the Gaussian maximizes entropy at
fixed variance), but the plug-in `Ĥ` can push a finite-sample estimate slightly off zero even
when the underlying density is Gaussian. `J` is invariant under `x ↦ a·x + b` (with `κ ↦ κ/|a|`
for the corresponding fit, per the package's scale equivariance): both terms shift by `ln|a|`
under the rescaling and cancel.
"""
function negentropy(d::AbstractDensityEstimate)
    T = eltype(d)
    σ² = var(d)
    return log(2 * T(π) * exp(one(T)) * σ²) / 2 - entropy(d)
end

"""
    entropy(d::AbstractDensityEstimate, xeval::AbstractVector) -> Ĥ

Held-out plug-in estimate of the differential entropy of the fitted density `d`, scored at
points `xeval` that did *not* enter the fit `d`:

    Ĥ = -(1/M) Σⱼ ln Q̂(xevalⱼ),   M = length(xeval),

with `ln Q̂` evaluated by [`logdensity`](@ref), so it stays finite far into an unbounded tail
where `Q̂` itself underflows. A point outside a finite `support` has `ln Q̂ = -Inf` and makes
`Ĥ = Inf`.

Unlike the one-argument [`entropy`](@ref)`(d)`, which evaluates `ln Q̂` at the fit's own
nodes, this evaluates it at independent points and so does not reward collapsing the density
onto the fit sample. It is the entropy term of [`negentropy`](@ref)`(d, xeval)`.
"""
function entropy(d::AbstractDensityEstimate, xeval::AbstractVector)
    T = eltype(d)
    M = length(xeval)
    M > 0 || throw(ArgumentError("need at least one evaluation point"))
    s = zero(T)
    for y in xeval
        s += _logdensity(d, T(y))
    end
    return -s / M
end

# Empirical mean and population (maximum-likelihood) variance of `xeval`. Moments are
# accumulated relative to the first point so that the two terms of E[x²] - E[x]² stay of
# order the data's spread rather than its absolute location; without this, points far from
# the origin lose precision and can drive the variance negative through cancellation.
function _mean_var(xeval, ::Type{T}) where {T}
    M = length(xeval)
    M > 0 || throw(ArgumentError("need at least one evaluation point"))
    x0 = T(first(xeval))
    s1 = zero(T)
    s2 = zero(T)
    for y in xeval
        t = T(y) - x0
        s1 += t
        s2 += t * t
    end
    m = s1 / M
    return m + x0, s2 / M - m * m
end

"""
    negentropy(d::AbstractDensityEstimate, xeval::AbstractVector) -> Ĵ

Held-out negentropy of the fitted density `d`, scored at independent points `xeval`:

    Ĵ = ½ ln(2π e s²) - Ĥ,

where `Ĥ =` [`entropy`](@ref)`(d, xeval)` and `s²` is the *empirical* (maximum-likelihood,
i.e. population) variance of `xeval`. The Gaussian reference thus uses the evaluation batch's
own moments, not the fitted density's; with that choice `½ ln(2π e s²) = -(1/M) Σⱼ ln 𝒩(xevalⱼ)`
is itself a held-out expectation, so `Ĵ` is the mean held-out log-likelihood ratio of the fit
against its matched Gaussian. It is invariant under `xeval ↦ a·xeval + b` together with the
`κ ↦ κ/|a|` rescaling of the fit (both terms shift by `ln|a|` and cancel). `xeval` must
contain at least two distinct points, or the Gaussian reference is undefined and an
`ArgumentError` is thrown.
"""
function negentropy(d::AbstractDensityEstimate, xeval::AbstractVector)
    T = eltype(d)
    H = entropy(d, xeval)
    _, s² = _mean_var(xeval, T)
    s² > 0 || throw(ArgumentError("negentropy(d, xeval) needs at least two distinct evaluation points"))
    return log(2 * T(π) * exp(one(T)) * s²) / 2 - H
end

"""
    logdensity_eval_gradient(d::AbstractDensityEstimate, x) -> ∂ln Q̂(x)/∂x

Derivative of the log fitted density with respect to the evaluation coordinate,

    ∂ ln Q̂(x) / ∂x = 2 ψ'(x) / ψ(x),

in closed form on the cell enclosing `x`, so the cost is `O(1)` per point with no linear
solve. The ratio is formed without evaluating `ψ`, so it stays finite where the density
underflows: on an unbounded tail with decay rate `k` it is exactly `±2k`. Outside a finite
support, where `ln Q̂ ≡ -Inf`, the result is `NaN`. See [`logdensity_node_gradient`](@ref)
for the sensitivity to the node positions instead. The log density has a kink at each node
`d.x[k]` (`ψ'` jumps there), so at a node the value returned is the one-sided derivative
approaching from within the enclosing interval.
"""
function logdensity_eval_gradient(d::AbstractDensityEstimate, x::Real)
    _check_not_nan(x)
    R = promote_type(eltype(d), typeof(x))
    xs = d.x
    n = length(xs)
    κL, κR = _tail_rates(d)
    if x <= xs[1]
        x < d.lo && return R(NaN)
        return 2 * _left_tail_slope(κL, x, d.lo)
    elseif x >= xs[n]
        x > d.hi && return R(NaN)
        return 2 * _right_tail_slope(κR, x, d.hi)
    end
    return 2 * _amplitude_slope(d, searchsortedlast(xs, x), x)
end

"""
    logdensity_node_gradient(d::DensityEstimate, xeval::AbstractVector, weights=nothing) -> g

Gradient, with respect to the node positions `d.x`, of the weighted sum of log densities at
the held-out points `xeval`:

    g[i] = ∂/∂x_i  Σⱼ weightsⱼ · ln Q̂(xevalⱼ),

computed by the implicit-function adjoint of the fit. Moving a node perturbs the fitted
amplitude everywhere through the field equation `M ψ = w ⊘ ψ`; differentiating it reuses the
fit's factored Hessian `∇²F = M + diag(wᵢ/ψᵢ²)`, and the adjoint aggregates the whole
evaluation batch into a *single* extra tridiagonal solve. The cost is therefore `O(N + M)`
for `N` nodes and `M` evaluation points, not one solve per point.

`weights` defaults to all ones (the plain sum `Σⱼ ln Q̂(xevalⱼ)`); pass `fill(1/M, M)` for the
mean. Given weights must have the axes of `xeval` (otherwise a `DimensionMismatch` is thrown),
and a `NaN` in `xeval` throws a `DomainError`. `xeval` should be disjoint from the fit's nodes — an evaluation point coinciding with a
node sits on the log density's kink, where the gradient is one-sided. The returned `g` is
indexed like `d.x`; a caller that moved the nodes through some parameter contracts `g` with
the node-position Jacobian to obtain the parameter gradient. See
[`logdensity_eval_gradient`](@ref) for the sensitivity to the evaluation point instead.

Defined for point data only: the nodes of an [`IntervalDensityEstimate`](@ref) are fixed
interval endpoints, not movable sample points.
"""
function logdensity_node_gradient(d::DensityEstimate{T,T}, xeval::AbstractVector,
                                  weights=nothing) where {T}
    (isinf(d.lo) && isinf(d.hi)) ||
        throw(ArgumentError("logdensity_node_gradient supports only unbounded fits; " *
                            "got support=[$(d.lo), $(d.hi)]"))
    weights === nothing || axes(weights) == axes(xeval) ||
        throw(DimensionMismatch("weights must have the axes of xeval: $(axes(weights)) vs $(axes(xeval))"))
    x, w, κ = d.x, d.w, d.κ
    n = length(x)
    Z = d.λ / d.κ                       # Z⋆ = ∫φ² for the unnormalized field φ = ψ√Z
    φ = d.ψ .* sqrt(Z)
    M = roughness_operator(x, κ)
    H = SymTridiagonal(M.dv .+ w ./ φ.^2, copy(M.ev))   # ∇²F, the fit's Hessian
    _, _, Gφ = _norm_sq_grad(x, φ, κ)                   # ½ ∂Z/∂φ
    src = zeros(T, n)                   # ∂L/∂φ, the adjoint source (L = Σ wⱼ ln Q̂(yⱼ))
    g = zeros(T, n)                     # explicit ∂L/∂xᵢ at fixed φ accumulates here
    C = zero(T)                         # Σ weightsⱼ, the coefficient of -ln Z in L
    wts = weights === nothing ? Iterators.repeated(one(T)) : weights
    for (yj, cj_) in zip(xeval, wts)
        _check_not_nan(yj)
        cj = T(cj_)
        C += cj
        y = T(yj)
        if y <= x[1]                    # left tail: ln φ = ln φ₁ + κ(y - x₁)
            src[1] += 2 * cj / φ[1]
            g[1]   -= 2 * cj * κ
        elseif y >= x[n]                # right tail: ln φ = ln φₙ - κ(y - xₙ)
            src[n] += 2 * cj / φ[n]
            g[n]   += 2 * cj * κ
        else
            k = searchsortedlast(x, y)
            a = κ * (x[k+1] - y); b = κ * (y - x[k]); θ = a + b
            SA = _sinh_ratio(a, θ); SB = _sinh_ratio(b, θ)   # interpolation weights of φₖ, φₖ₊₁
            CA = _cosh_ratio(a, θ); CB = _cosh_ratio(b, θ)
            ct = _cosh_ratio(θ, θ)                           # coth θ
            φy = φ[k] * SA + φ[k+1] * SB
            src[k]   += 2 * cj * SA / φy
            src[k+1] += 2 * cj * SB / φy
            g[k]   += 2 * cj * κ * (ct - φ[k+1] * CB / φy)   # ∂ln φ(y)/∂xₖ at fixed nodes
            g[k+1] += 2 * cj * κ * (φ[k] * CA / φy - ct)
        end
    end
    @. src -= 2 * C / Z * Gφ            # -C ∂ln Z/∂φ
    μ = ldiv!(ldlt!(H), src)            # adjoint μ = H⁻¹ ∂L/∂φ (H and src are consumed)
    # Add the implicit response -μᵀ(∂M/∂xᵢ)φ and the explicit -C ∂ln Z/∂xᵢ|_φ. Both act only
    # through θₖ = κ hₖ, with ∂θₖ/∂xᵢ = κ(δ_{i,k+1} - δ_{i,k}), so interval k contributes
    # ±(κ Bₖ + (C/Z) κ Dₖ) to nodes k and k+1. Bₖ is written as in `_dM_dκ_mul`, so that its two
    # O(1/θ²) terms do not cancel as θ → 0, and stays finite as θ → ∞.
    for k in 1:n-1
        θ = κ * (x[k+1] - x[k])
        Bk = (μ[k] * φ[k+1] + μ[k+1] * φ[k]) / (1 + cosh(θ)) -
             csch(θ)^2 * (μ[k+1] - μ[k]) * (φ[k+1] - φ[k])
        dfdiag, dfcross = _mass_coeffs_dθ(θ) ./ κ
        Dk = dfdiag * (φ[k]^2 + φ[k+1]^2) + 2 * dfcross * φ[k] * φ[k+1]
        contrib = κ * Bk + (C / Z) * κ * Dk
        g[k]   += contrib
        g[k+1] -= contrib
    end
    return g
end

# The adjoint above differentiates the field equation of a single-rate fit; a per-interval
# scale would need the κₖ-dependence of every roughness term.
logdensity_node_gradient(::DensityEstimate{T,Vector{T}}, ::AbstractVector, weights=nothing) where {T} =
    throw(ArgumentError("logdensity_node_gradient supports only a single-rate (scalar κ) fit"))

"""
    chisq(d::DensityEstimate, Q) -> χ²

Goodness-of-fit statistic between a trial density `Q` and the data underlying the
fit `d`, the robust field-theoretic analogue of Pearson's χ² (Eqs. 13–14 of the
paper):

    χ² = 4 Σᵢ wᵢ (√Q(xᵢ) / ψ_cl(xᵢ) - 1)²,

summed over the data nodes `xᵢ` with multiplicities `wᵢ`, where `ψ_cl = √(d(·))`
is the fitted amplitude. `Q` is any callable returning density values; it should be
a normalized density (`∫Q dx = 1`). `chisq(d, d) == 0`. Small χ² means `Q` is close
to the data in the (squared Hellinger) sense. For its significance, build the reference law
once with [`ChisqReference`](@ref) and pass the statistic to [`pvalue`](@ref).

Defined for point data only; there is no form for an [`IntervalDensityEstimate`](@ref).
"""
function chisq(d::DensityEstimate{T}, Q) where {T}
    s = zero(T)
    ψ = d.ψ
    for i in eachindex(d.x, d.w, ψ)
        qi = Q(d.x[i])
        qi >= 0 || throw(ArgumentError("trial density Q must be nonnegative; got Q($(d.x[i]))=$qi"))
        r = sqrt(qi) / ψ[i] - 1
        s += d.w[i] * r^2
    end
    return 4 * s
end

# Standard normal CDF, Φ(t) = ½ erfc(-t/√2).
_Φ(t::T) where {T} = erfc(-t / sqrt(T(2))) / 2

# ── Finite-N reference distribution of χ² (Holy 1997, Eqs. 16–18) ────────────
#
# χ²(δψ) = 4 Σᵢ wᵢ (δψ(xᵢ)/ψ_cl(xᵢ))² is a quadratic form in the Gaussian
# fluctuation field of Eq. 16 (precision L = -ℓ²∂² + 2λ + 2Σ wₖδ(x-xₖ)/ψₖ²,
# constrained by ∫ ψ_cl δψ = 0). Its law is therefore a generalized chi-squared,
# χ² = Σₖ eₖ Zₖ², the eₖ being the eigenvalues of D^½ C D^½ with D = diag(4wᵢ/ψᵢ²)
# and C the covariance of the field values at the nodes. Equivalently its Laplace
# transform is det(I + 2a·DC)^{-1/2}, exactly Eq. 18.
#
# Everything is tridiagonal. The unconstrained node covariance obeys C₀⁻¹ = G₀⁻¹ + S with
# S = diag(2wₖ/ψₖ²), and the free part of the precision is L₀ = 2λ𝒜, 𝒜u = -(κ(x)⁻²u′)′ + u,
# so G₀⁻¹ = 2λ M̂ with M̂ the `roughness_operator` at unit reference scale. That identity is
# not the Gauss-Markov one: M̂ maps the nodal values of an 𝒜-harmonic interpolant to the
# jumps of its flux v = κ⁻²ψ′, and a Green's-function column Ĝ(·,xⱼ) is precisely the
# 𝒜-harmonic field whose only flux jump is a unit one at xⱼ, so M̂ Ĝ|nodes = I. It needs the
# breakpoints of κ to be the nodes — with a jump strictly inside an interval, Ĝ(·,xⱼ) would
# not be a single hyperbolic arc there — which is why the fit realizes one rate per interval.
# At constant κ it reduces to G₀⁻¹ = (2λ/κ)M, with (G₀)ᵢⱼ = κ e^{-κ|xᵢ-xⱼ|}/(4λ).
#
# The ∫ψ_cl δψ = 0 constraint contributes one rank-one term, C = C₀ - b bᵀ/Vφ (Eq. 18's T(g)
# factor). Tail probabilities come from Imhof's inversion, whose integrand needs only
# det(I + iuA) per node — an O(N) tridiagonal determinant plus a rank-one correction — so no
# eigenvalues are formed.

"""
    ChisqReference(d::DensityEstimate)

Reference distribution of the goodness-of-fit statistic [`chisq`](@ref) for the fit `d`,
following Holy 1997 (Eqs. 16–18). The statistic is a quadratic form in the Gaussian
fluctuation field, so its law is a generalized chi-squared, evaluated here at finite `N`
with no large-`N` limit. Construction costs `O(N)`, and the object stores the `O(N)` data —
a symmetric tridiagonal matrix and a rank-one constraint vector — that its density and tail
probabilities are computed from.

Build one per fit and reuse it across many evaluations of [`pvalue`](@ref),
[`chisq_ccdf`](@ref) and [`chisq_pdf`](@ref): one reference serves any number of trial
densities tested against the fit. Its exact mean is [`mean`](@ref Statistics.mean)`(ref)`.
A spatially varying `κ` and a finite `support` (see [`DensityEstimate`](@ref)) are both
supported, and the law stays `O(N)` in either case.

The law is exact for the Gaussian fluctuation field of the Laplace approximation about
the fit, and it is evaluated at finite `N`. The functions that evaluate it take
`method = :exact` (this finite-`N` law, the default) or `method = :largeN` (its large-`N`
limit).

It is not, by itself, a finite-sample frequentist null distribution for data drawn from
the fitted density; it is a Bayesian measure of likelihood among distributions sharing
the same roughness penalty. Its calibration as a frequentist null distribution need not
be distribution-free.

Defined for point data only; there is no form for an [`IntervalDensityEstimate`](@ref).

# Extended help

With a spatially varying `κ` the nodal precision of the fluctuation field is `2λ` times the
same tridiagonal operator the fit assembles, whatever the scale. With a finite `support` the
fluctuation field's natural (Neumann) boundary condition makes `M̂` the Dirichlet-to-Neumann
map of the boundary segments as well as the interior, and the same identity `G₀⁻¹ = 2λM̂` holds
with `M̂` the bounded operator the fit already assembles.
"""
struct ChisqReference{T<:AbstractFloat}
    tri::SymTridiagonal{T,Vector{T}}    # D^{-1/2} C₀⁻¹ D^{-1/2}; A = tri⁻¹ - g gᵀ
    g::Vector{T}                        # rank-one constraint direction, D^{1/2} b / √Vφ
    tg::Vector{T}                       # tri·g, the Imhof rank-one RHS (constant in u)
    mean::T                             # exact ⟨χ²⟩ = tr(A)

    function ChisqReference{T}(tri, g) where {T<:AbstractFloat}
        tri = convert(SymTridiagonal{T,Vector{T}}, tri)::SymTridiagonal{T,Vector{T}}
        g = convert(Vector{T}, g)::Vector{T}
        size(tri, 1) == length(g) ||
            throw(DimensionMismatch("tri is $(size(tri, 1))×$(size(tri, 2)) but g has length $(length(g))"))
        return new{T}(tri, g, tri * g, _tridiag_invdiag(tri) - sum(abs2, g))
    end
end

Base.show(io::IO, r::ChisqReference) =
    print(io, "ChisqReference($(length(r.g)) nodes, ⟨χ²⟩=$(r.mean))")
Base.broadcastable(r::ChisqReference) = Ref(r)

# Coefficients of the per-interval accumulation the Green's-function sweeps run on. Both
# solutions of 𝒜u = 0 have the form u(s) = u(xₖ)cosh(κₖs) + κₖ v(xₖ)sinh(κₖs) on interval k,
# v = κ⁻²u′ being the flux, so with ψ the hyperbolic interpolant
#   e^{-θ}∫₀ʰ u(s)ψ(s)ds = u(xₖ)(ψₖc₁ + ψₖ₊₁c₂) + κₖv(xₖ)(ψₖc₃ + ψₖ₊₁c₄).
# The e^{-θ} keeps every coefficient bounded as θ → ∞ (isolated points). Below θ = 1 the
# coth/csch forms of c₃ and c₄ cancel catastrophically (relative error ~eps/θ²); the
# _sinhm/_coshm forms are algebraically identical and cancellation-free.
function _sweep_coeffs(κ::T, h::T) where {T}
    θ = κ * h
    e = exp(-θ); t = e * e; m = -expm1(-2θ)         # e^{-θ}, e^{-2θ}, 1 - e^{-2θ}
    c₁ = θ * e / (2κ)
    c₂ = m / (4κ)
    if θ < 1
        sh = m / (2e)                               # sinh θ
        c₃ = e * (θ * _coshm(θ) - _sinhm(θ)) / (2κ * sh)
        c₄ = e * _sinhm(2θ) / (4κ * sh)
    else
        c₃ = e * (θ * (1 + t) / m - 1) / (2κ)       # e^{-θ}(θ coth θ - 1)/(2κ)
        c₄ = ((1 + t) / 2 - 2θ * t / m) / (2κ)      # e^{-θ}(cosh θ - θ csch θ)/(2κ)
    end
    return c₁, c₂, c₃, c₄
end

# α = L₀⁻¹ψ_cl at the nodes, mᵢ = α(xᵢ), with an optional natural boundary at `lo`/`hi`. With u∓
# the solutions of 𝒜u = 0 decaying at ∓∞ (or, at a finite boundary, the Dirichlet-to-Neumann
# solution rooted at the wall) and C = v₋u₊ - u₋v₊ their flux Wronskian (constant, by Abel),
# Ĝ(x,y) = u₋(x∧y)u₊(x∨y)/C, so
#   α(x) = [u₊(x)∫_{lo}^x u₋ψ_cl + u₋(x)∫_x^{hi} u₊ψ_cl] / (2λC).
# Each tail fixes one solution: u₋ = e^{κL(x-x₁)} to the left of x₁ when unbounded (normalized to
# 1 there, whence v₋ = 1/κL) or the boundary segment's cosh arc when finite (v₋ = the
# Dirichlet-to-Neumann flux `_tail_diag(κL, ΔL)/κL`), and its mirror to the right. Since u∓ grow
# like e^{±∫κ}, they are propagated — along with their accumulations — scaled by e^{∓∫κ}, which
# is what keeps the recursions bounded; the scale factors cancel identically in α, so it is
# assembled from the scaled quantities alone. `Â[1] = ∫_{lo}^{x₁} u₋ψ_cl / ψ₁` is
# `_tail_mass(ψ₁, κL, ΔL)/ψ₁` at a finite boundary (the same integral `_norm_sq` needs, since u₋
# and ψ_cl are the same cosh arc up to normalization) or `ψ₁/(2κL)` unbounded; mirror on the
# right. The Wronskian `Ĉ = û₊[1]·v̂₋[1] + v̂₊[1]` at a finite boundary specializes to
# `û₊[1]/κL + v̂₊[1]` unbounded (v̂₋[1] = 1/κL there); the specialization is written explicitly
# rather than folded into the product so the unbounded value picks up only the one rounding a
# direct division does.
function _node_alpha(x::Vector{T}, ψ::Vector{T}, κ, κL::T, κR::T, λ::T, lo::T, hi::T) where {T}
    n = length(x)
    û₋ = similar(ψ); v̂₋ = similar(ψ); Â = similar(ψ)   # u₋, v₋, ∫_{lo}^x u₋ψ_cl
    û₊ = similar(ψ); v̂₊ = similar(ψ); B̂ = similar(ψ)   # u₊, -v₊, ∫_x^{hi} u₊ψ_cl
    û₋[1] = one(T)
    v̂₋[1] = isfinite(lo) ? _tail_diag(κL, x[1] - lo) / κL : inv(κL)
    Â[1]  = isfinite(lo) ? _tail_mass(ψ[1], κL, x[1] - lo) / ψ[1] : ψ[1] / (2κL)
    for k in 1:n-1
        κk = _kappa(κ, k); h = x[k+1] - x[k]; θ = κk * h
        c₁, c₂, c₃, c₄ = _sweep_coeffs(κk, h)
        e = exp(-θ); ch = (1 + e * e) / 2; sh = -expm1(-2θ) / 2      # e^{-θ}cosh θ, e^{-θ}sinh θ
        Â[k+1] = e * Â[k] + û₋[k] * (ψ[k] * c₁ + ψ[k+1] * c₂) +
                            κk * v̂₋[k] * (ψ[k] * c₃ + ψ[k+1] * c₄)
        û₋[k+1] = û₋[k] * ch + κk * v̂₋[k] * sh
        v̂₋[k+1] = û₋[k] * sh / κk + v̂₋[k] * ch
    end
    û₊[n] = one(T)
    v̂₊[n] = isfinite(hi) ? _tail_diag(κR, hi - x[n]) / κR : inv(κR)
    B̂[n]  = isfinite(hi) ? _tail_mass(ψ[n], κR, hi - x[n]) / ψ[n] : ψ[n] / (2κR)
    for k in n-1:-1:1
        κk = _kappa(κ, k); h = x[k+1] - x[k]; θ = κk * h
        c₁, c₂, c₃, c₄ = _sweep_coeffs(κk, h)
        e = exp(-θ); ch = (1 + e * e) / 2; sh = -expm1(-2θ) / 2
        B̂[k] = e * B̂[k+1] + û₊[k+1] * (ψ[k+1] * c₁ + ψ[k] * c₂) +
                            κk * v̂₊[k+1] * (ψ[k+1] * c₃ + ψ[k] * c₄)
        û₊[k] = û₊[k+1] * ch + κk * v̂₊[k+1] * sh
        v̂₊[k] = û₊[k+1] * sh / κk + v̂₊[k+1] * ch
    end
    Ĉ = isfinite(lo) ? û₊[1] * v̂₋[1] + v̂₊[1] : û₊[1] / κL + v̂₊[1]   # the Wronskian
    return (û₊ .* Â .+ û₋ .* B̂) ./ (2λ * Ĉ)
end

# ∬ψ_cl G₀ ψ_cl = ∫ψ_cl α, with an optional natural boundary at `lo`/`hi`. On each interval α
# solves 𝒜α = ψ_cl/(2λ) at constant κ against a hyperbolic source, so it is the interpolant of
# its own nodal values mₖ plus the resonant particular solution s·cosh(κs) that the source
# forces; the interior sum is untouched by a boundary. Each tail is `_tail_psi_alpha` — the same
# computation with ψ_cl ∝ e^{∓κ(x-x_edge)} and α acquiring the same resonant factor when
# unbounded, or the boundary segment's closed form at a finite gap.
function _int_psi_alpha(x::Vector{T}, ψ::Vector{T}, m::Vector{T}, κ, κL::T, κR::T, λ::T,
                        lo::T, hi::T) where {T}
    n = length(x)
    acc = _tail_psi_alpha(ψ[1], m[1], κL, λ, x[1] - lo) + _tail_psi_alpha(ψ[n], m[n], κR, λ, hi - x[n])
    for k in 1:n-1
        κk = _kappa(κ, k); h = x[k+1] - x[k]; θ = κk * h
        f = κk / (4λ)
        β = f * h * _cosh_ratio(θ, θ)               # (κ h coth θ)/(4λ)
        a₁ = m[k] + β * ψ[k]; a₂ = m[k+1] + β * ψ[k+1]
        function ψα(s)
            r = h - s
            pr = _sinh_ratio(κk * r, θ); ps = _sinh_ratio(κk * s, θ)
            α = a₁ * pr + a₂ * ps -
                f * (ψ[k] * r * _cosh_ratio(κk * r, θ) + ψ[k+1] * s * _cosh_ratio(κk * s, θ))
            return (ψ[k] * pr + ψ[k+1] * ps) * α
        end
        acc += quadgk(ψα, zero(h), h; rtol = sqrt(eps(T)))[1]
    end
    return acc
end

# ∫₀^Δ ψ(s)α(s) ds over a boundary segment (Neumann wall at s=0, node at s=Δ), or the unbounded
# tail's closed form ψ₁m₁/(2κ) + ψ₁²/(16λκ) as Δ → ∞. On the segment ψ(s) = ψ₁cosh(κs)/cosh(θ)
# (θ = κΔ) and α solves 𝒜α = ψ/(2λ) with a vanishing flux at s=0: since 𝒜(s·sinh(κs)) =
# -(2/κ)cosh(κs) and s·sinh(κs) already has zero flux at s=0, the particular solution
# Ã·s·sinh(κs)/cosh(θ) (Ã = -κψ₁/(4λ)) needs only a cosh(κs)/cosh(θ) term added to match
# α(Δ) = m₁: α(s) = [B̃·cosh(κs) + Ã·s·sinh(κs)]/cosh(θ), B̃ = m₁ - Ã·Δ·tanh(θ). Writing ψ and α
# through `_cosh_ratio2`/`_sinh_ratio2` keeps every term O(1) at θ up to where `_tanh_stable`
# itself stays accurate (θ ~ 500 and beyond), never evaluating a raw cosh/sinh of θ or κs.
function _tail_psi_alpha(ψ1::T, m1::T, κ::T, λ::T, Δ::T) where {T}
    isfinite(Δ) || return ψ1 * m1 / (2κ) + ψ1^2 / (16λ * κ)
    θ = κ * Δ
    Ã = -κ * ψ1 / (4λ)
    B̃ = m1 - Ã * Δ * _tanh_stable(θ)
    function ψα(s)
        cr = _cosh_ratio2(κ * s, θ); sr = _sinh_ratio2(κ * s, θ)
        return ψ1 * (B̃ * cr^2 + Ã * s * cr * sr)
    end
    return quadgk(ψα, zero(T), Δ; rtol = sqrt(eps(T)))[1]
end

# Diagonal of the inverse of a symmetric tridiagonal, O(N), from its top-down and
# bottom-up LDLᵀ pivots.
function _tridiag_invdiag(tri::SymTridiagonal{T}) where {T}
    a, β = tri.dv, tri.ev; n = length(a)
    p = similar(a); q = similar(a)
    p[1] = a[1]
    for i in 2:n; p[i] = a[i] - β[i-1]^2 / p[i-1]; end
    q[n] = a[n]
    for i in n-1:-1:1; q[i] = a[i] - β[i]^2 / q[i+1]; end
    return sum(inv(p[i] + q[i] - a[i]) for i in 1:n)
end

ChisqReference(d::DensityEstimate{T}) where {T} = ChisqReference{T}(d)

function ChisqReference{T}(d::DensityEstimate) where {T}
    x, ψ, w, λ = d.x, d.ψ, d.w, d.λ
    κ, κL, κR, lo, hi = d.κ, d.κL, d.κR, d.lo, d.hi
    n = length(x)
    m = _node_alpha(x, ψ, κ, κL, κR, λ, lo, hi)        # mₖ = ∫ψ_cl(x) G₀(xₖ,x) dx
    # C₀⁻¹ = G₀⁻¹ + S = 2λM̂ + diag(2wᵢ/ψᵢ²);  b = (I + G₀S)⁻¹m solves C₀⁻¹b = G₀⁻¹m. The
    # assembly carries the reference scale κ̄, which G₀⁻¹ = 2λM̂ does not admit: divide it out.
    M = _operator(x, κ, κL, κR, lo, hi)
    f = 2λ / _reference_scale(κ, κL, κR)
    S = 2 .* w ./ ψ.^2
    C0inv = SymTridiagonal(f .* M.dv .+ S, f .* M.ev)
    b = C0inv \ (f .* (M * m))
    Vφ = _int_psi_alpha(x, ψ, m, κ, κL, κR, λ, lo, hi) - sum(m .* S .* b)  # Var(∫ψ_cl δψ)
    # Reduced tridiagonal tri = D^{-1/2} C₀⁻¹ D^{-1/2} and rank-one direction g.
    D = 2 .* S; sq = sqrt.(D)                          # D = 4wᵢ/ψᵢ²
    tri = SymTridiagonal(C0inv.dv ./ D, C0inv.ev ./ (sq[1:n-1] .* sq[2:n]))
    g = sq .* b ./ sqrt(Vφ)
    return ChisqReference{T}(tri, g)
end

"""
    mean(ref::ChisqReference) -> ⟨χ²⟩

Exact mean of the reference distribution of [`chisq`](@ref): the finite-`N` generalized-χ²
law of the quadratic fluctuation approximation (Holy 1997, Eqs. 16–18), whose standing as a
null distribution [`ChisqReference`](@ref) sets out. Defined at any scale, constant or
spatially varying.
"""
Statistics.mean(r::ChisqReference) = r.mean

# Scratch for one sweep of `_logΦ!`: pivots and RHS/solution, both length N. Allocated once
# per tail-probability integral and reused across every integrand evaluation within it, which
# keeps the reference itself immutable and safe to share.
_logΦ_scratch(r::ChisqReference{T}) where {T} =
    (Vector{Complex{T}}(undef, length(r.g)), Vector{Complex{T}}(undef, length(r.g)))

# (unwrapped arg, modulus) of Φ(u) = det(I + iuA), A = tri⁻¹ - g gᵀ. The determinant of
# I+iu·tri⁻¹ is a ratio of tridiagonal determinants (continuant recurrence, accumulated in
# log space so the phase unwraps past π); the rank-one term is one complex tridiagonal solve.
# Both O(N) and, given the scratch buffers `piv`/`rhs` (length N), allocation-free.
#
# The continuant pivots rrₖ of `tri + iuI` are exactly the Thomas pivots of that system, so a
# single forward sweep computes the log-determinant and eliminates the RHS `tg = tri·g`; a
# back-substitution then yields y = (tri+iuI)⁻¹ tg. `piv` holds the pivots for the back sweep,
# `rhs` the eliminated RHS overwritten in place with y.
function _logΦ!(piv::Vector{Complex{T}}, rhs::Vector{Complex{T}},
                r::ChisqReference{T}, u::Real) where {T}
    a, β, tg = r.tri.dv, r.tri.ev, r.tg
    n = length(a)
    r0 = complex(a[1])
    rr = complex(a[1], u)                       # a[1] + iu
    s = log(rr) - log(r0)
    piv[1] = rr
    rhs[1] = tg[1] / rr
    for k in 2:n
        r0 = a[k] - β[k-1]^2 / r0
        rr = complex(a[k], u) - β[k-1]^2 / rr
        s += log(rr) - log(r0)
        piv[k] = rr
        rhs[k] = (tg[k] - β[k-1] * rhs[k-1]) / rr
    end
    for k in n-1:-1:1
        rhs[k] -= (β[k] / piv[k]) * rhs[k+1]     # y_k = d'_k - (β_k/rr_k) y_{k+1}
    end
    gy = zero(Complex{T})
    for k in 1:n
        gy += r.g[k] * rhs[k]                     # g·y, with g real
    end
    rank1 = 1 - complex(zero(T), u) * gy
    return imag(s) + angle(rank1), exp(real(s)) * abs(rank1)
end

# Inverse-Gaussian (Wald) survival at mean μ and shape μ²: the large-`N` shape of the
# generalized-χ² law (paper Eq. 26). Parameterized by the exact mean μ = tr A it is a
# closed-form surrogate for the Imhof inversion, defined at every scale.
function _wald_ccdf(μ::T, z::Real) where {T}
    zT = T(z)
    zT > 0 || return one(T)
    λ = μ^2
    s = sqrt(λ / zT)
    a = s * (zT / μ - 1)
    b = s * (zT / μ + 1)
    # Survival = Φ(-a) - e^{2λ/μ} Φ(-b); the second term uses erfcx so the large
    # positive exponent 2λ/μ cancels against -b²/2 without overflow.
    return _Φ(-a) - erfcx(b / sqrt(T(2))) * exp(2λ / μ - b^2 / 2) / 2
end

# The reference χ² law is supported on z > 0, so both methods return the same limits outside
# it: survival 1 and density 0 for z ≤ 0, survival and density 0 at z = Inf.
function _check_chisq_args(z::Real, method::Symbol)
    method === :exact || method === :largeN ||
        throw(ArgumentError("method must be :exact or :largeN, got :$method"))
    isnan(z) && throw(DomainError(z, "the χ² value must not be NaN"))
    return nothing
end

"""
    chisq_ccdf(ref::ChisqReference, z; method=:exact, rtol, atol) -> P(χ² ≥ z)

Upper-tail (survival) probability of the reference χ² distribution at `z`. Evaluated at an
observed statistic it is a diagnostic significance; see [`pvalue`](@ref), and
[`ChisqReference`](@ref) for the sense in which it is and is not a p-value.

`method=:exact` (default) uses the finite-`N` generalized-χ² law of `ref` via Imhof
inversion. `method=:largeN` uses the inverse-Gaussian (Wald) shape of the large-`N` limit
(Eq. 26), parameterized by the exact mean [`mean`](@ref Statistics.mean)`(ref)`; it is a
closed form, far cheaper per call, and, like the exact law, defined at every scale. `rtol`
and `atol` (default `sqrt(eps(T))`) are the tolerances of the `:exact` quadrature. The law is
supported on `z > 0`: `z ≤ 0` gives `1`, `z = Inf` gives `0`, and `NaN` throws a `DomainError`.
"""
function chisq_ccdf(r::ChisqReference{T}, z::Real; method::Symbol=:exact, rtol=sqrt(eps(T)), atol=sqrt(eps(T))) where {T}
    _check_chisq_args(z, method)
    z > 0 || return one(T)
    isfinite(z) || return zero(T)
    method === :largeN && return _wald_ccdf(r.mean, z)
    # atol is required: at the median of the law the integral is ~0, and a purely relative
    # tolerance would subdivide without end. It bounds the error in the probability by atol/π.
    zT = T(z)
    piv, rhs = _logΦ_scratch(r)
    f(u) = u == 0 ? (r.mean - zT) / 2 :
        (θ = _logΦ!(piv, rhs, r, u); sin((θ[1] - zT * u) / 2) / (u * sqrt(θ[2])))
    I, _ = quadgk(f, zero(T), T(Inf); rtol, atol)  # I ∈ [-π/2, π/2]
    return clamp(one(T)/2 + I / T(π), zero(T), one(T))
end

# Inverse-Gaussian (Wald) density, companion to `_wald_ccdf`.
_wald_pdf(μ::T, z::Real) where {T} =
    z > 0 ? μ / sqrt(2 * T(π) * T(z)^3) * exp(μ - T(z) / 2 - μ^2 / (2 * T(z))) : zero(T)

"""
    chisq_pdf(ref::ChisqReference, z; method=:exact, rtol, atol) -> P(z)

Density of the reference χ² distribution at `z`; it is `0` for `z ≤ 0` and at `z = Inf`, and
`NaN` throws a `DomainError`. `method=:exact` (default) is the finite-`N` generalized-χ² law
of `ref`. `method=:largeN` is the
inverse-Gaussian (Wald) density of the large-`N` limit (Eq. 26),

    P(z) = ⟨χ²⟩ / √(2π z³) · exp[⟨χ²⟩ - z/2 - ⟨χ²⟩²/(2z)],

with `⟨χ²⟩ =` [`mean`](@ref Statistics.mean)`(ref)` the exact mean: a closed form, defined at
every scale. `rtol` and `atol` are as in [`chisq_ccdf`](@ref).
"""
function chisq_pdf(r::ChisqReference{T}, z::Real; method::Symbol=:exact, rtol=sqrt(eps(T)), atol=sqrt(eps(T))) where {T}
    _check_chisq_args(z, method)
    z > 0 && isfinite(z) || return zero(T)
    method === :largeN && return _wald_pdf(r.mean, z)
    # atol floors the density: deep in the tail the true value underflows to ~0, and a purely
    # relative tolerance would otherwise subdivide the oscillatory integrand without end.
    zT = T(z)
    piv, rhs = _logΦ_scratch(r)
    f(u) = (θ = _logΦ!(piv, rhs, r, u); cos((θ[1] - zT * u) / 2) / sqrt(θ[2]))
    I, _ = quadgk(f, zero(T), T(Inf); rtol, atol, maxevals=10^4)
    return max(I / (2 * T(π)), zero(T))
end

"""
    pvalue(ref::ChisqReference, χ²; method=:exact, rtol, atol) -> p

Significance of an observed goodness-of-fit statistic `χ² = `[`chisq`](@ref)`(d, Q)` for a
trial density `Q`, where `ref = `[`ChisqReference`](@ref)`(d)`: the probability that the
reference χ² distribution exceeds it, `chisq_ccdf(ref, χ²; kwargs...)`. The reference law is
that of the quadratic fluctuation approximation, so this is a diagnostic significance whose
calibration is density-dependent; see [`ChisqReference`](@ref).

The keywords are passed to [`chisq_ccdf`](@ref). Extends `StatsAPI.pvalue`.

# Examples
```julia
ref = ChisqReference(d)
pvalue(ref, chisq(d, Q))
```
"""
StatsAPI.pvalue(r::ChisqReference, χ²::Real; kwargs...) = chisq_ccdf(r, χ²; kwargs...)

# Golden-section minimisation of a unimodal `f` on `[a, b]` in `ln κ`; returns the minimizer.
function _golden_min(f, a::T, b::T; iters::Int=60) where {T}
    invφ = (sqrt(T(5)) - 1) / 2      # 1/golden ≈ 0.618
    c = b - invφ * (b - a); fc = f(c)
    d = a + invφ * (b - a); fd = f(d)
    for _ in 1:iters
        if fc < fd
            b, d, fd = d, c, fc
            c = b - invφ * (b - a); fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + invφ * (b - a); fd = f(d)
        end
    end
    return (a + b) / 2
end

# `score(exp(l))` as a function of l = ln c, with non-finite values mapped to +∞
# (`typemax(T)`) so that a search steps over scales it cannot resolve.
_lnscore(score, ::Type{T}) where {T} = l -> (v = score(exp(l)); isfinite(v) ? v : typemax(T))

# Evaluate `f` (from `_lnscore`) at every point of the ln-scale grid `lncs`, visiting the
# points from last to first when `reverse` is set. Returns the scores and the index of the
# smallest, or `nothing` in place of the index when no grid point is resolvable.
function _grid_argmin(f, lncs, ::Type{T}; reverse::Bool=false) where {T}
    scores = similar(lncs, T)
    for j in (reverse ? Iterators.reverse(eachindex(lncs)) : eachindex(lncs))
        scores[j] = f(lncs[j])
    end
    all(==(typemax(T)), scores) && return scores, nothing
    return scores, argmin(scores)
end

# Geometric κ grid from coarse (≈ one blob over the data) to fine (≈ individual points),
# scaled to the data's extent, wide enough to bracket the minimum-sensitivity scale.
function _default_κs(x::AbstractVector{<:Real})
    isempty(x) && throw(ArgumentError("cannot select κ for zero observations"))
    T = float(eltype(x))
    _check_finite_observations(x, T)
    lo, hi = extrema(x)
    span = T(hi) - T(lo)
    span > 0 || throw(ArgumentError("need at least two distinct points to select κ"))
    return exp.(range(log(T(0.5) / span), log(5 * length(x) / span); length = 40))
end

# Whether the selectors' `resolution` asks for the interval likelihood: `nothing` (exact points,
# checked for ties) and `0` (exact points, unchecked) do not; a positive lattice spacing does.
function _is_rounded(resolution)
    (resolution === nothing || resolution == 0) && return false
    isfinite(resolution) && resolution > 0 ||
        throw(ArgumentError("resolution must be finite and nonnegative, got $resolution"))
    return true
end

"""
    select_kappa_sensitivity(x; kappas=nothing, rtol=cbrt(eps(T))) -> κ

Choose the smoothing scale by the principle of minimum sensitivity: return the `κ` at which
the classical action [`action`](@ref) `S` is least sensitive to the scale, i.e. the `κ`
minimizing [`action_sensitivity`](@ref)`(x, κ) = |dS/d ln κ|` (Fig. 1 of the paper). `kappas`
is the grid that brackets the search, as for [`select_kappa`](@ref).

This criterion resolves *information* and over-resolves smooth densities; to target estimation error instead, use [`select_kappa`](@ref) (or
[`select_kappa_lscv`](@ref)); for rounded data, pass the rounding increment to `select_kappa` as
`resolution`.

This selector takes no `support` keyword: the entropy asymptotics behind minimum sensitivity
are derived for the unbounded line and do not generalize to a finite domain, so it always
fits (and returns a scale for) the unbounded problem. It also has no form for interval data.

# Extended help

The derivative `dS/d ln κ` is evaluated analytically and minimized over `κ` by a golden-section
search, bracketed by the grid `kappas`. This is a principled convention rather than a unique
optimum: `S` has no exact stationary point in `κ`, so the flattest point depends on measuring
sensitivity in `ln κ`.
"""
function select_kappa_sensitivity(x::AbstractVector{<:Real};
                                  kappas::Union{Nothing,AbstractVector{<:Real}}=nothing,
                                  rtol::Real=cbrt(eps(float(eltype(x)))))
    κs = kappas === nothing ? _default_κs(x) : _check_kappa_grid(kappas)
    xs, r = _sensitivity_sample(x, rtol)
    return _minimize_on_grid(κ -> _action_sensitivity(xs, r, κ), κs, eltype(xs);
                             extend=kappas === nothing)
end

"""
    action_sensitivity(x, κ; rtol=cbrt(eps(T))) -> s::T

The minimum-sensitivity score `s = |dS/d ln κ|` of the scale `κ` for data `x`, where `S` is the
classical [`action`](@ref) of the unbounded fit [`DensityEstimate`](@ref)`(x, κ; rtol)`.
[`select_kappa_sensitivity`](@ref) returns the `κ` minimizing it; evaluating it over a range
of `κ` shows how sharply that minimum is defined. The derivative is computed analytically.
Like `select_kappa_sensitivity`, it takes point data only.
"""
function action_sensitivity(x::AbstractVector{<:Real}, κ::Real; rtol::Real=cbrt(eps(float(eltype(x)))))
    xs, r = _sensitivity_sample(x, rtol)
    return _action_sensitivity(xs, r, _score_scale(κ, eltype(xs)))
end

function _sensitivity_sample(x::AbstractVector{<:Real}, rtol::Real)
    _check_rtol(rtol)
    T = float(promote_type(eltype(x), typeof(rtol)))
    return _sorted_sample(x, T), T(rtol)
end

_action_sensitivity(xs::Vector{T}, r::T, κ::T) where {T} =
    abs(last(_action_and_slope(_merge_presorted(xs, r / κ)..., κ)))

# Diagonal and first off-diagonal of the inverse of an SPD symmetric tridiagonal, in O(n), from
# the top-down and bottom-up LDLᵀ pivots dᵢ, δᵢ: (H⁻¹)ᵢᵢ = 1/(dᵢ + δᵢ - aᵢ), with aᵢ the original
# diagonal, and (H⁻¹)ᵢ,ᵢ₊₁ = -bᵢ (H⁻¹)ᵢ₊₁,ᵢ₊₁ / dᵢ.
function _inv_band(H::SymTridiagonal{T}) where {T}
    a, b = H.dv, H.ev
    n = length(a)
    d = similar(a); δ = similar(a)
    d[1] = a[1]
    for i in 2:n
        d[i] = a[i] - b[i-1]^2 / d[i-1]
    end
    δ[n] = a[n]
    for i in n-1:-1:1
        δ[i] = a[i] - b[i]^2 / δ[i+1]
    end
    g = inv.(d .+ δ .- a)
    return g, [-b[i] * g[i+1] / d[i] for i in 1:n-1]
end

# ln(-Mₖ,ₖ₊₁) for the operator `_operator` assembles with reference scale κ̄, formed directly so
# that it stays finite where csch(κh) underflows.
_log_coupling(x::Vector{T}, κ::T, κ̄::T, k::Int) where {T} = -logabssinh(κ * (x[k+1] - x[k]))
_log_coupling(x::Vector{T}, κs::Vector{T}, κ̄::T, k::Int) where {T} =
    log(κ̄ / κs[k]) - logabssinh(κs[k] * (x[k+1] - x[k]))

# Normalised amplitude ψ and the leave-one-out densities Q̂₋ᵢ(xᵢ) at every node, in O(N). Dropping
# one observation at node i decrements wᵢ; the deleted field is a Newton step of that problem from
# the full solution, along the direction H⁻¹eᵢ (H the fit's SPD Hessian ∇²F = M + diag(w/φ²)). A
# unit step is the first-order approximation; it under-penalizes over-fitting, because where the node
# leverage h = (H⁻¹)ᵢᵢ/φᵢ² is high the removed point's spike collapses super-linearly and a linear
# step cannot see it. Instead the step length t is set so node i's own nonlinear stationarity holds
# exactly: t² h(1 - wᵢh) - t + 1 = 0, whose root t = 2/(1 + √(1 - 4h(1 - wᵢh))) (real because
# h(1 - wᵢh) ≤ 1/4) is 1 as h → 0, recovering first order, and grows to shrink the overshoot as h
# rises. The node amplitude becomes φᵢ(1 - t h), and the normalization is carried linearly,
# Zₜ = Z - 2t vᵢ/φᵢ with v = H⁻¹Gφ (Gφ = ½ ∂Z/∂φ, Z = ∫φ² = φᵀGφ), so Q̂₋ᵢ(xᵢ) = (φᵢ(1 - t h))² / Zₜ.
#
# A node isolated at unit weight sits at the double root: its own stationarity gives Mᵢᵢφᵢ² = wᵢ, so
# h = 1/2, the discriminant vanishes and t = 2. Rounding can carry the discriminant just below zero
# there, hence the clamp in `_deletion_step`.
#
# t and its inputs are dimensionless or scale like Z, so — as for the linear step — nothing depends
# on M's entries beyond its being the fixed SPD operator with mass functional Z: it holds unchanged
# for a piecewise-constant scale, and for a natural boundary via the bounded `_operator`/`_norm_sq_gram`.
#
# The step tracks a literal leave-one-out refit (wᵢ decremented, a singleton node dropped) closely
# across the scale range on continuous data.
#
# `holdout = :location` instead deletes all wᵢ observations at node i, the held-out value then
# lying away from every atom of the remaining sample. Deleting c observations (c = 1 or wᵢ) takes
# the step τ = c·t along the same direction, with node i's stationarity at weight wᵢ - c fixing
# τ²h(1 - wᵢh) - τ + c = 0; the root continuous with the first-order step is
# τ = 2c/(1 + √(1 - 4ch(1 - wᵢh))), the node amplitude becomes φᵢ(1 - τh), and Zₜ = Z - 2τvᵢ/φᵢ.
# For c = wᵢ the discriminant is (1 - 2wᵢh)² and the node collapses at wᵢh = 1/2, as a single
# isolated point does at h = 1/2. At wᵢ = 1 the two modes coincide. On tied data, leaving out a
# single copy of a repeated value rewards spikes at the atoms, so `:location` is the default.
#
# When all of node i's observations are deleted (c = wᵢ), node i's stationarity at weight zero is
# source-free, (Mφ')ᵢ = 0, so the deleted amplitude is also φ'ᵢ = Σⱼ (-Mᵢⱼ) φ'ⱼ / Mᵢᵢ over the
# neighbors j = i ± 1, which move by the same step, φ'ⱼ = φⱼ - τ (H⁻¹)ⱼᵢ/φᵢ. The two forms agree
# exactly, but φᵢ(1 - τh) cancels to roundoff at an isolated node (h → 1/(2wᵢ)), where the held-out
# density is of order e^{-2κ·gap}; the neighbor sum has positive terms and is formed in the log
# domain, so it holds its relative accuracy there. That form is used whenever c = wᵢ.
#
# Returns ψ, the log of each squared deleted amplitude (NaN where a neighbor's linear update is
# not positive), and each deleted normalization Zₜ; see `_loo_density` and `_loo_logdensity`.
function _loo_terms(nodes::Vector{T}, w::Vector{T}, κ, κL::T, κR::T, lo::T, hi::T;
                    holdout::Symbol=:location) where {T}
    holdout in (:location, :observation) ||
        throw(ArgumentError("holdout must be :location or :observation, got :$holdout"))
    M = _operator(nodes, κ, κL, κR, lo, hi)
    φ = _solve_amplitude(M, w)
    Z, Gφ = _norm_sq_gram(nodes, φ, κ, κL, κR, lo, hi)
    H = SymTridiagonal(M.dv .+ w ./ φ.^2, M.ev)
    gii, gnext = _inv_band(H)
    v = ldiv!(ldlt!(H), Gφ)             # H⁻¹Gφ; H is consumed, its inverse band already extracted
    κ̄ = _reference_scale(κ, κL, κR)
    n = length(φ)
    la2 = similar(φ)
    Zt = similar(φ)
    for i in eachindex(φ, w)
        h = gii[i] / φ[i]^2                             # single-observation leverage
        c = holdout === :location ? w[i] : one(T)       # observations deleted
        τ = _deletion_step(h, w[i], c)
        Zt[i] = Z - 2 * τ * v[i] / φ[i]
        if c == w[i]
            l = T(-Inf)
            if i > 1
                φj = φ[i-1] - τ * gnext[i-1] / φ[i]
                l = φj > 0 ? logaddexp(l, _log_coupling(nodes, κ, κ̄, i - 1) + log(φj)) : T(NaN)
            end
            if i < n
                φj = φ[i+1] - τ * gnext[i] / φ[i]
                l = φj > 0 ? logaddexp(l, _log_coupling(nodes, κ, κ̄, i) + log(φj)) : T(NaN)
            end
            la2[i] = 2 * (l - log(M.dv[i]))
        else
            la2[i] = log((φ[i] * (1 - τ * h))^2)
        end
    end
    return φ ./ sqrt(Z), la2, Zt
end

# Normalized amplitude ψ and the leave-one-out densities Q̂₋ᵢ(xᵢ) at every node (see `_loo_terms`).
function _loo_density(nodes::Vector{T}, w::Vector{T}, κ, κL::T, κR::T, lo::T, hi::T;
                      kwargs...) where {T}
    ψ, la2, Zt = _loo_terms(nodes, w, κ, κL, κR, lo, hi; kwargs...)
    return ψ, exp.(la2) ./ Zt
end

# ψ and the leave-one-out log densities ln Q̂₋ᵢ(xᵢ), NaN where the deleted fit is undefined (a
# non-positive normalization or neighbor update).
function _loo_logdensity(nodes::Vector{T}, w::Vector{T}, κ, κL::T, κR::T, lo::T, hi::T;
                         kwargs...) where {T}
    ψ, la2, Zt = _loo_terms(nodes, w, κ, κL, κR, lo, hi; kwargs...)
    return ψ, [Zt[i] > 0 ? la2[i] - log(Zt[i]) : T(NaN) for i in eachindex(la2, Zt)]
end

# The step τ for deleting c of the w observations at a node of leverage h (see `_loo_density`).
# The discriminant is ≥ 0 for c ∈ {1, w}, up to the rounding the clamp absorbs.
function _deletion_step(h::T, w::T, c::T) where {T}
    disc = 1 - 4 * c * h * (1 - w * h)
    return 2c / (1 + sqrt(max(disc, zero(disc))))
end

# Least-squares cross-validation score LSCV(κ) = ∫Q̂² - (2/N) Σᵢ wᵢ Q̂₋ᵢ(xᵢ), with an optional
# natural boundary at `lo`/`hi`: an unbiased estimate, up to the κ-independent ∫Q², of the
# integrated squared error ∫(Q̂-Q)².
function _lscv(nodes::Vector{T}, w::Vector{T}, κ, κL::T, κR::T, lo::T, hi::T; kwargs...) where {T}
    ψ, looi = _loo_density(nodes, w, κ, κL, κR, lo, hi; kwargs...)
    N = sum(w)
    cross = zero(T)
    for i in eachindex(w, looi)
        cross += w[i] * looi[i]
    end
    return _int_quartic(nodes, ψ, κ, κL, κR, lo, hi) - 2 * cross / N
end

# Kullback–Leibler cross-validation score, the mean negative leave-one-out log-likelihood
# -(1/N) Σᵢ wᵢ ln Q̂₋ᵢ(xᵢ), with an optional natural boundary at `lo`/`hi`: an estimate, up to a
# κ-independent constant, of KL(Q ‖ Q̂_κ). Reuses the same leave-one-out densities as _lscv, in
# the log domain. An undefined held-out log density returns NaN so the search rejects κ.
function _klcv(nodes::Vector{T}, w::Vector{T}, κ, κL::T, κR::T, lo::T, hi::T; kwargs...) where {T}
    _, ℓ = _loo_logdensity(nodes, w, κ, κL, κR, lo, hi; kwargs...)
    s = zero(T)
    for i in eachindex(w, ℓ)
        isfinite(ℓ[i]) || return T(NaN)
        s += w[i] * ℓ[i]
    end
    return -s / sum(w)
end

"""
    select_kappa_lscv(x; kappas=nothing, rtol=cbrt(eps(T)), support=(-Inf, Inf),
                      resolution=nothing, holdout=:location) -> κ

Choose the smoothing scale by least-squares cross-validation: return the `κ` minimizing
[`lscv`](@ref)`(x, κ)`,

    LSCV(κ) = ∫ Q̂_κ(x)² dx - (2/N) Σᵢ Q̂_{κ,-i}(xᵢ),

an unbiased estimate — up to the `κ`-independent `∫Q²` — of the integrated squared error
`∫(Q̂_κ - Q)²`, where `Q̂_{κ,-i}` is the density fitted with the `i`-th point left out. Its
minimizer therefore targets minimum mean integrated squared error (MISE). This generally
selects a coarser scale than [`select_kappa_sensitivity`](@ref) (minimum sensitivity), which
resolves information rather than squared error and tends to over-resolve smooth densities.

`support`, `kappas`, `rtol` and `holdout` mean what they do for [`select_kappa`](@ref).
`resolution` may be `nothing` (check for ties) or `0` (accept the values as exact points);
this criterion has no interval form, because with only the interval of each observation known,
the cross term has no unbiased estimate, so rounded data should go to `select_kappa` with
`resolution` set to the rounding increment.

# Extended help

Both terms are evaluated analytically in `O(N)`: `∫Q̂²` in closed form over the exponential
segments, and each leave-one-out density `Q̂_{-i}(xᵢ)` from the same leave-one-out expansion
[`select_kappa`](@ref) uses, so no per-point refitting is needed. The score is minimized by a
golden-section search over `ln κ`, bracketed by the grid `kappas`.
"""
function select_kappa_lscv(x::AbstractVector{<:Real}; kappas::Union{Nothing,AbstractVector{<:Real}}=nothing,
                           rtol::Real=cbrt(eps(float(eltype(x)))), support::Tuple{Real,Real}=(-Inf, Inf),
                           resolution::Union{Real,Nothing}=nothing, holdout::Symbol=:location)
    _check_lscv_resolution(resolution, "select_kappa_lscv")
    return first(_select_by_score(_lscv, x, kappas, rtol, support; holdout, checkties=resolution === nothing))
end

"""
    lscv(x, κ; support=(-Inf, Inf), rtol=cbrt(eps(T)), resolution=nothing,
         holdout=:location) -> LSCV::T

The least-squares cross-validation score of the constant scale `κ` for data `x`, the quantity
[`select_kappa_lscv`](@ref) minimizes; the keywords mean what they do there. It is `NaN` when a
leave-one-out density is undefined at `κ`. Like `select_kappa_lscv`, it has no interval-data
form.
"""
function lscv(x::AbstractVector{<:Real}, κ::Real; support::Tuple{Real,Real}=(-Inf, Inf),
              rtol::Real=cbrt(eps(float(eltype(x)))), resolution::Union{Real,Nothing}=nothing,
              holdout::Symbol=:location)
    _check_lscv_resolution(resolution, "lscv")
    xs, r, slo, shi = _scoring_sample(x, rtol, support, resolution === nothing)
    return _score_at(_lscv, xs, r, slo, shi, _score_scale(κ, eltype(xs)); holdout)
end

_check_lscv_resolution(resolution, fname) = _is_rounded(resolution) &&
    throw(ArgumentError("$fname has no form for rounded (interval) data; use " *
                        "select_kappa(x; resolution=$resolution), or resolution=0 to treat " *
                        "the values as exact points"))

"""
    select_kappa(x; support=(-Inf, Inf), rtol=cbrt(eps(T)), resolution=nothing,
                 holdout=:location, kappas=nothing) -> κ::T
    select_kappa(lower, upper; support=(-Inf, Inf), rtol=cbrt(eps(T)), holdout=:location,
                 kappas=nothing) -> κ::T

Choose a constant smoothing scale for [`DensityEstimate`](@ref)`(x, κ; support, rtol)` (or, for
interval data, [`IntervalDensityEstimate`](@ref)) by Kullback–Leibler (likelihood)
cross-validation: return the `κ` minimizing the mean negative leave-one-out log-likelihood

    KLCV(κ) = -(1/N) Σᵢ wᵢ ln Q̂_{κ,-i}(xᵢ),

where `Q̂_{κ,-i}` is the density fitted with the `i`-th point left out. `κ` has the data's
floating-point type `T`. For a spatially varying scale, see
[`select_kappa(x, ::Adaptive)`](@ref select_kappa(::AbstractVector{<:Real}, ::Adaptive)); to
choose `support` as well, see [`select_support`](@ref).

`support`, `rtol` and `resolution` describe the model and should match the later fit:

- `support = (a, b)` (default `(-Inf, Inf)`) fits and cross-validates on a finite domain, as
  [`DensityEstimate`](@ref)'s `support` does. It is held fixed across every candidate `κ`. Data
  outside `[a, b]`, or `a ≥ b`, throws a `DomainError`.
- `rtol` is the node-merging tolerance, as a fraction of the local smoothing length.
- Cross-validation of the point likelihood assumes the data are draws from a continuous
  density. If more than 1% of the observations repeat another value, the data are probably
  rounded, and an `ArgumentError` explains the options unless `resolution` is given.
  `resolution = δ > 0` declares that each `x[i]` was rounded to a lattice of spacing `δ`; the
  scale is then chosen for [`IntervalDensityEstimate`](@ref)`(x, κ; resolution=δ)`, and each
  observation contributes the held-out log-probability of its rounding interval.
  `resolution = 0` declares the values exact points, fitted by [`DensityEstimate`](@ref).

The second form takes the bounds of interval observations directly, for
[`IntervalDensityEstimate`](@ref)`(lower, upper, κ)`, including censored values and a mix of
points and intervals.

`holdout` chooses what each held-out term for a point observation leaves out: `:location` (the
default) all observations at the held-out value, `:observation` just one of them. The two agree
when every value is distinct. On tied data leaving out a single copy rewards density spikes at
the repeated values. Interval observations are always left out one at a time.

`kappas` is the grid that brackets the search. By default (`kappas = nothing`) it is a
geometric range scaled to the data's extent, extended past either end when the minimum falls
there. A grid passed as `kappas` must be sorted and positive, with at least three values, and is
searched as given: a minimum on its first or last value throws a `ConvergenceError`. For
interval data the default grid never extends past the `κ` at which `κ` times the widest
interval reaches 4096, beyond which the fit cannot be computed accurately; a minimum there is
returned as the selected scale, since the fit is then at its histogram limit.

[`klcv`](@ref) evaluates the score at a given `κ`. See the "Comparing selection criteria" section
of the manual for the alternative criteria [`select_kappa_lscv`](@ref) and
[`select_kappa_sensitivity`](@ref).

# Extended help

`KLCV` estimates, up to a `κ`-independent constant, the Kullback–Leibler divergence
`KL(Q ‖ Q̂_κ)`; minimizing it is maximum-likelihood cross-validation. It is the criterion native
to the estimator, whose action `-Σ ln Q̂(xᵢ)` is itself the (in-sample) log-likelihood. To
leading order it selects the same error-optimal scale as least-squares cross-validation
([`select_kappa_lscv`](@ref)), and it is cheaper, because the `∫Q̂²` roughness term is not needed.
For interval data it estimates the divergence between the true and fitted probabilities of the
recorded intervals, which is all such data determine.

Each leave-one-out density `Q̂_{-i}(xᵢ)` comes from a one-step leave-one-out expansion of the fit
in the dropped point's weight — a Newton step whose length is set so the deleted node's own
stationarity holds exactly, which tracks a literal refit closely without a linear step's blindness
to over-fitting — so no per-point refitting is needed and the score costs `O(N)`. For an interval
observation, the refit is approximated by minimizing over the interval's own nodal values with
its term in the likelihood exact and the rest of the problem replaced by its quadratic model,
which stays accurate when the deletion empties the interval. The score is minimized by a
golden-section search over `ln κ`, bracketed by the grid `kappas`.
"""
function select_kappa(x::AbstractVector{<:Real}; support::Tuple{Real,Real}=(-Inf, Inf),
                      rtol::Real=cbrt(eps(float(eltype(x)))), resolution::Union{Real,Nothing}=nothing,
                      holdout::Symbol=:location,
                      kappas::Union{Nothing,AbstractVector{<:Real}}=nothing)
    if !_is_rounded(resolution)
        return first(_select_by_score(_klcv, x, kappas, rtol, support; holdout, checkties=resolution === nothing))
    end
    isempty(x) && throw(ArgumentError("cannot select κ for zero observations"))
    T = float(eltype(x))
    lower, upper = _lattice_bounds(x, T(resolution), support)
    return select_kappa(lower, upper; kappas, rtol, support, holdout)
end

"""
    klcv(x, κ; support=(-Inf, Inf), rtol=cbrt(eps(T)), resolution=nothing,
         holdout=:location) -> KLCV::T
    klcv(lower, upper, κ; support=(-Inf, Inf), rtol=cbrt(eps(T)), holdout=:location) -> KLCV::T

The Kullback–Leibler cross-validation score `KLCV(κ) = -(1/N) Σᵢ wᵢ ln Q̂_{κ,-i}(xᵢ)` of the
constant scale `κ`, the quantity [`select_kappa`](@ref) minimizes; the arguments and keywords
mean what they do there, and with `resolution = δ > 0` or interval bounds each observation
contributes the held-out log-probability of its interval. It is `NaN` when a held-out log
density or probability is undefined at `κ`.
"""
function klcv(x::AbstractVector{<:Real}, κ::Real; support::Tuple{Real,Real}=(-Inf, Inf),
              rtol::Real=cbrt(eps(float(eltype(x)))), resolution::Union{Real,Nothing}=nothing,
              holdout::Symbol=:location)
    if !_is_rounded(resolution)
        xs, r, slo, shi = _scoring_sample(x, rtol, support, resolution === nothing)
        return _score_at(_klcv, xs, r, slo, shi, _score_scale(κ, eltype(xs)); holdout)
    end
    isempty(x) && throw(ArgumentError("cannot score κ for zero observations"))
    T = float(eltype(x))
    lower, upper = _lattice_bounds(x, T(resolution), support)
    return klcv(lower, upper, κ; rtol, support, holdout)
end

# Minimize a per-κ score over ln κ, bracketed by the grid `kappas`, on a domain fixed for the
# whole search. `scorefun(nodes, w, κ, κ, κ, lo, hi; holdout)` returns the score for the merged
# nodes/weights at scale κ. A near-coincident pair left unmerged at very large κ can drive the fit
# to a non-finite score, which `_search_grid` steps over. `kappas === nothing` searches the
# extensible default grid. Returns `_search_grid`'s `(κ, κlo, κhi)`.
function _select_by_score(scorefun, x::AbstractVector{<:Real}, kappas, rtol::Real,
                          support::Tuple{Real,Real}; holdout::Symbol=:location, checkties::Bool=false)
    κs = kappas === nothing ? _default_κs(x) : _check_kappa_grid(kappas)
    xs, r, slo, shi = _scoring_sample(x, rtol, support, checkties)
    T = eltype(xs)
    return _search_grid(κ -> _score_at(scorefun, xs, r, slo, shi, κ; holdout), κs, T;
                        extend=kappas === nothing)
end

# The validated, sorted sample a constant-scale score is evaluated on, in the working float type
# of the data, `rtol` and finite support bounds, together with `rtol` and the bounds in that type.
function _scoring_sample(x::AbstractVector{<:Real}, rtol::Real, support::Tuple{Real,Real}, checkties::Bool)
    _check_rtol(rtol)
    a, b = _check_support_bounds(support)
    T = float(promote_type(eltype(x), typeof(rtol), _support_eltype(a), _support_eltype(b)))
    xs = _sorted_sample(x, T)
    slo, shi = T(a), T(b)
    _check_support(xs, slo, shi)
    checkties && _check_ties(xs)
    return xs, T(rtol), slo, shi
end

# `scorefun` at the constant scale κ, on the nodes merged at tolerance `r/κ`.
_score_at(scorefun, xs, r, slo, shi, κ; holdout::Symbol=:location) =
    scorefun(_merge_presorted(xs, r / κ)..., κ, κ, κ, slo, shi; holdout)

# A scale passed to a score function, converted to the working type `T`.
_score_scale(κ::Real, ::Type{T}) where {T} = _kappa_in(T, κ)

function _check_kappa_grid(κs)
    issorted(κs) && all(>(0), κs) || throw(ArgumentError("kappas must be sorted and positive"))
    length(κs) >= 3 || throw(ArgumentError("need at least 3 values in kappas to bracket the minimum"))
    return κs
end

const _KGRID_EXT = 10   # grid steps added each time a default κ grid is extended

# Bracket the minimum of `score(κ)` on the grid `κs`, then refine by golden section in ln κ.
# Non-finite scores count as +∞; a grid with no finite score is an error. A minimum on the
# first or last grid point is an error, unless `extend` is set (a default grid): the grid then
# grows by `_KGRID_EXT` steps of its own spacing past that end, up to `_CSHIFTS` times, but
# never above `κmax`, where a minimum on the top point is accepted. Returns `(κ, κlo, κhi)`,
# with `[κlo, κhi]` the range of the grid searched.
function _search_grid(score, κs, ::Type{T}; extend::Bool=false, κmax::Real=T(Inf)) where {T}
    f = _lnscore(score, T)
    lnκ = T[log(T(κ)) for κ in κs]
    scores, i = _grid_argmin(f, lnκ, T)
    i === nothing && throw(ConvergenceError("the smoothing scale search",
        "no κ in kappas (from $(first(κs)) to $(last(κs))) gives a finite score"))
    lnmax = log(T(κmax))
    for nshifts in 0:_CSHIFTS
        atlo, athi = i == firstindex(lnκ), i == lastindex(lnκ)
        (atlo || athi) || break
        athi && lnκ[i] >= lnmax && break
        extend || throw(ConvergenceError("the smoothing scale search",
            "the minimum is at the $(atlo ? "lower" : "upper") end of kappas (κ = $(atlo ? first(κs) : last(κs))); " *
            "extend kappas past it, or omit kappas to use the default grid"))
        nshifts < _CSHIFTS || throw(ConvergenceError("the smoothing scale search",
            "the minimum kept running off the default κ grid after $_CSHIFTS extensions " *
            "(κ from $(exp(first(lnκ))) to $(exp(last(lnκ))))"))
        h = (last(lnκ) - first(lnκ)) / (length(lnκ) - 1)
        new = atlo ? first(lnκ) .- h .* (_KGRID_EXT:-1:1) :
                     unique!(min.(last(lnκ) .+ h .* (1:_KGRID_EXT), lnmax))
        newscores = map(f, new)
        lnκ = atlo ? vcat(new, lnκ) : vcat(lnκ, new)
        scores = atlo ? vcat(newscores, scores) : vcat(scores, newscores)
        i = argmin(scores)
    end
    loκ = lnκ[max(i - 1, firstindex(lnκ))]
    hiκ = lnκ[min(i + 1, lastindex(lnκ))]
    return exp(_golden_min(f, loκ, hiκ)), exp(first(lnκ)), exp(last(lnκ))
end

_minimize_on_grid(score, κs, ::Type{T}; kwargs...) where {T} = first(_search_grid(score, κs, T; kwargs...))

# Fraction of repeated values above which the point-likelihood selectors refuse the sample.
const _TIE_FRACTION = 0.01

# Throw if more than `_TIE_FRACTION` of the sorted sample `xs` repeats an earlier value, naming
# the lattice spacing when the distinct values lie on one.
function _check_ties(xs::Vector{T}) where {T}
    n = length(xs)
    ntied = count(i -> xs[i] == xs[i-1], 2:n)
    ntied > _TIE_FRACTION * n || return nothing
    pct = round(100 * ntied / n; digits=1)
    msg = "$ntied of the $n observations ($pct%) repeat another value, as rounded data do; the " *
          "point likelihood treats them as exact. "
    d = T[xs[i] - xs[i-1] for i in 2:n if xs[i] != xs[i-1]]    # gaps between distinct values
    δ = isempty(d) ? zero(T) : minimum(d)
    if length(d) >= 2 && all(di -> abs(di / δ - round(di / δ)) <= 1e-3, d)
        msg *= "The distinct values lie on a lattice of spacing δ ≈ $(round(δ; sigdigits=6)): pass " *
               "`resolution=δ` to `select_kappa` to fit them as rounded (interval) data, or " *
               "`resolution=0` to treat them as exact points."
    else
        msg *= "Pass `resolution=0` to treat them as exact points, or give each observation's bounds " *
               "with `select_kappa(lower, upper)`."
    end
    throw(ArgumentError(msg))
end

"""
    AdaptiveScale(c, α, pilot)

A spatially varying smoothing scale of the plug-in form

    κ(x) = c · (p̂(x) / ḡ)^α,

where `p̂` is the `pilot` density estimate and `ḡ` the geometric mean of `p̂` over the
sample it was fitted to. [`select_kappa`](@ref)`(x, Adaptive())` constructs one by choosing
`c` and `α`; the result is callable, and is passed straight to [`DensityEstimate`](@ref) as
the smoothing scale.

Dividing by `ḡ` puts `c` on the same footing as a constant scale: where `p̂` equals its
geometric mean, `κ(x) = c`. The exponent `α > 0` sets how strongly the scale follows the
density — larger `α` resolves the peaks more finely and smooths the tails more heavily.

The scale is floored at `1e-6 c`, which intercepts underflow of `(p̂/ḡ)^α` at points where
the pilot density is negligible; the floor sits far below any scale the rule would
otherwise choose, so it never shapes the fit.
"""
struct AdaptiveScale{T<:AbstractFloat,D<:AbstractDensityEstimate}
    c::T           # scale where the pilot density equals its geometric mean
    α::T           # exponent coupling the scale to the pilot density
    pilot::D       # the pilot density estimate p̂
    loggbar::T     # ln ḡ, the mean of ln p̂ over the pilot's sample
    κmin::T        # underflow floor

    # `loggbar` must equal `_log_geomean(pilot)`. It is an argument only so that a search over
    # many `c` for one pilot computes it once; for an interval pilot it costs an entropy integral.
    function AdaptiveScale{T,D}(c, α, pilot, loggbar=_log_geomean(pilot)) where {T<:AbstractFloat,
                                                                                D<:AbstractDensityEstimate}
        α > 0 || throw(ArgumentError("the exponent α must be positive, got $α"))
        c > 0 || throw(ArgumentError("the scale c must be positive, got $c"))
        return new{T,D}(c, α, pilot, loggbar, T(_KAPPA_FLOOR) * T(c))
    end
end

AdaptiveScale{T}(c, α, pilot::D, args...) where {T,D} = AdaptiveScale{T,D}(c, α, pilot, args...)
AdaptiveScale(c::Real, α::Real, pilot::AbstractDensityEstimate) = AdaptiveScale{eltype(pilot)}(c, α, pilot)

# The pilot density underflows to zero between two far-separated tail nodes, sending
# (p̂/ḡ)^α there to zero; the floor keeps the assembled operator's coth(θ)/κ entries finite.
const _KAPPA_FLOOR = 1e-6

# ln ḡ = (1/N) Σᵢ ln p̂(xᵢ) over the pilot's sample. Merged points share their node's density,
# so the node weights carry the sample's multiplicities.
function _log_geomean(d::DensityEstimate{T}) where {T}
    s = zero(T)
    for i in eachindex(d.x, d.w)
        s += d.w[i] * 2 * log(d.ψ[i])
    end
    return s / sum(d.w)
end

# The rule itself, from ln p̂(x): κ = c·(p̂/ḡ)^α, floored.
_scale_from_logdensity(k::AdaptiveScale, lnp) = max(k.c * exp(k.α * (lnp - k.loggbar)), k.κmin)

# ln p̂ rather than p̂: the pilot density underflows to zero beyond and between far-separated
# tail nodes, where its logarithm is still perfectly finite. `_logdensity` is what makes that
# true; `2 log(_amplitude(...))` would reintroduce the underflow this rule exists to survive,
# and would disagree with the batch path below.
(k::AdaptiveScale)(x::Real) = _scale_from_logdensity(k, _logdensity(k.pilot, x))

# One walk of the pilot for the whole sorted batch, instead of a binary search per position.
function _kappa_sorted(k::AdaptiveScale{T}, ts::AbstractVector, ::Type{T}) where {T}
    κ = _logdensity_sorted(k.pilot, ts)
    for i in eachindex(κ, ts)
        κ[i] = _check_kappa(_scale_from_logdensity(k, κ[i]), ts[i])
    end
    return κ
end

Base.show(io::IO, k::AdaptiveScale) =
    print(io, "AdaptiveScale(c=", k.c, ", α=", k.α, ") over a pilot with ",
          length(k.pilot.x), " nodes")

# Score a candidate scale end to end: merge at the local tolerance it implies, realize it on
# the resulting nodes, and cross-validate. A κ profile spanning many orders of magnitude can
# drive the LDLᵀ factorization of the assembled tridiagonal to an exact zero pivot; that
# candidate is unresolvable, which is what a non-finite score already means to the searches
# here, so it reports NaN rather than aborting the whole selection. `scorefun` takes the support
# bounds `lo`/`hi` as its last two arguments; either may be infinite.
function _score_kappa(scorefun, xs::Vector{T}, κfun, rtol::T, lo::T, hi::T) where {T}
    nodes, w, κs, κL, κR = _merge_and_realize(xs, κfun, rtol)
    length(nodes) >= 2 || return T(NaN)
    try
        return scorefun(nodes, w, κs, κL, κR, lo, hi)
    catch e
        e isa ZeroPivotException && return T(NaN)
        rethrow()
    end
end

# Leave-one-out log densities ln Q̂₋ⱼ(xⱼ), one per observation of the sorted sample `xs`, for the
# candidate scale `κfun`: a constant (a `Real`, merged and scored exactly as the constant
# candidate in the adaptive `select_kappa`) or any scale function (merged and scored as
# `_score_kappa` does). Each observation takes the value at the node it merged into; merging
# collapses contiguous runs of `xs`, so node i covers the next `w[i]` observations in order.
# Hence `-mean` of the result is `_klcv` on the same nodes, up to rounding. Entries whose node has
# an undefined leave-one-out log density are NaN, and an unresolvable candidate (fewer than two
# nodes, or an exact zero pivot) returns all NaN, matching the NaN score `_klcv`/`_score_kappa`
# report in those cases.
function _loo_logdensities(xs::Vector{T}, κ::Real, rtol::T, lo::T, hi::T) where {T}
    κT = T(κ)
    nodes, w = _merge_presorted(xs, rtol / κT)
    return _loo_logdensities(xs, nodes, w, κT, κT, κT, lo, hi)
end

function _loo_logdensities(xs::Vector{T}, κfun, rtol::T, lo::T, hi::T) where {T}
    return _loo_logdensities(xs, _merge_and_realize(xs, κfun, rtol)..., lo, hi)
end

function _loo_logdensities(xs::Vector{T}, nodes::Vector{T}, w::Vector{T}, κ, κL::T, κR::T,
                           lo::T, hi::T) where {T}
    ℓ = fill(T(NaN), length(xs))
    length(nodes) >= 2 || return ℓ
    ℓnode = try
        _loo_logdensity(nodes, w, κ, κL, κR, lo, hi)[2]
    catch e
        e isa ZeroPivotException && return ℓ
        rethrow()
    end
    sum(w) == length(xs) ||
        error("merged weights sum to $(sum(w)), but the sample has $(length(xs)) observations")
    j = firstindex(ℓ)
    for i in eachindex(w, ℓnode)
        v = isfinite(ℓnode[i]) ? ℓnode[i] : T(NaN)
        m = Int(w[i])               # an integer count: merging only ever adds oneunit(T)
        ℓ[j:j+m-1] .= v
        j += m
    end
    return ℓ
end

# Per-observation gain of the adaptive scale `κa` over the constant scale `κ0`,
# dⱼ = ln Q̂₋ⱼ(xⱼ; κa) - ln Q̂₋ⱼ(xⱼ; κ0), returned as its mean and the standard error of that
# mean, std(d)/√n. The mean equals the difference of the two candidates' KLCV scores up to
# rounding. With a single observation the standard error is undefined and is returned as 0.
function _adaptive_gain(xs::Vector{T}, κ0::Real, κa, rtol::T, lo::T, hi::T) where {T}
    return _mean_se(_loo_logdensities(xs, κa, rtol, lo, hi) .- _loo_logdensities(xs, κ0, rtol, lo, hi))
end

function _mean_se(d::Vector{T}) where {T}
    n = length(d)
    gain = mean(d)
    se = n >= 2 ? std(d; corrected=true) / sqrt(T(n)) : zero(T)
    return gain, se
end

# As above, for a sample in which `d[i]` occurs `counts[i]` times.
function _mean_se(d::Vector{T}, counts::Vector{T}) where {T}
    n = sum(counts)
    gain = sum(counts .* d) / n
    se = n >= 2 ? sqrt(sum(counts .* (d .- gain) .^ 2) / (n - 1) / n) : zero(T)
    return gain, se
end

const _CSPAN = 20.0   # the c bracket runs ×/÷ this factor about its center
const _CGRID = 13      # grid points across it, to bracket the minimum
const _CITERS = 20     # golden-section refinements: pins ln c to ~1e-4 of the bracket, far
                       # below the scale on which the score itself varies
const _CSHIFTS = 6     # times the bracket may recenter on an edge minimum before giving up

# Minimize `score(c)` over ln c, bracketed by a geometric grid about `c0` and recentered on
# the best edge until the minimum falls strictly inside. Unresolvable candidates score
# non-finite; treating those as +∞ lets the search step over them.
#
# `span`/`ngrid`/`iters`/`maxshifts` default to the plug-in-scale search's own constants but
# are independently tunable — `select_support` reuses this same coarse-grid-then-golden-with-
# recentering pattern for both its finite-gap search (a wide span, since the gap's effect on
# the score is gentle over a broad range) and its chained inner κ search (a narrow span about
# the previous gap candidate's optimum). `reverse` visits the coarse grid from the wide end of
# the bracket to the narrow end instead of the default low-to-high; `select_support`'s gap
# search needs this so a stateful κ warm start (threaded through `score`) tracks gaps in the
# wide-to-narrow order the coupling between gap and κ assumes. `label` names the candidate in
# the two error messages.
# `bounds`, when given, is an absolute `(lo, hi)` the search may never recenter past: an edge
# minimum that coincides with a clamped bound is accepted outright rather than triggering
# another shift, since there is nowhere sane left to look. `select_support`'s chained κ search
# uses this to stay inside the data-scaled range `select_kappa` itself would ever consider —
# without it, a handful of gap candidates can warm-start each other up an unbounded chain into
# a regime where the first-order LOO expansion is no longer trustworthy and spuriously reports
# ever-improving scores (observed directly: KLCV score turning unboundedly negative for
# κ ≳ 1e6 on data where every sane candidate sits below 1e4).
#
# The returned c never scores non-finite: if the golden-section refinement ends on an
# unresolvable c, the best grid point is returned instead.
_select_c(score, c0; kwargs...) = first(_select_c_scored(score, c0; kwargs...))

# `_select_c`, returning `(c, score(c))`, with `score(c)` finite. With `skip_unresolved=true`, a
# bracket with no resolvable candidate returns `nothing` instead of throwing.
function _select_c_scored(score, c0::T; span::Real=_CSPAN, ngrid::Int=_CGRID, iters::Int=_CITERS,
                          maxshifts::Int=_CSHIFTS, reverse::Bool=false, label::String="smoothing scale",
                          bounds::Union{Nothing,Tuple{Real,Real}}=nothing,
                          skip_unresolved::Bool=false) where {T}
    f = _lnscore(score, T)
    lnspan = log(T(span))
    lnbounds = bounds === nothing ? nothing : (log(T(bounds[1])), log(T(bounds[2])))
    for _ in 0:maxshifts
        loln, hiln = log(c0) - lnspan, log(c0) + lnspan
        if lnbounds !== nothing
            loln, hiln = max(loln, lnbounds[1]), min(hiln, lnbounds[2])
            if !(loln < hiln)
                l = clamp(log(c0), lnbounds[1], lnbounds[2])
                fl = f(l)
                fl < typemax(T) && return exp(l), fl
                skip_unresolved && return nothing
                throw(ConvergenceError("the $label search", "no resolvable $label at the clamped bound c = $(exp(l))"))
            end
        end
        lncs = range(loln, hiln; length=ngrid)
        scores, i = _grid_argmin(f, lncs, T; reverse)
        if i === nothing
            skip_unresolved && return nothing
            throw(ConvergenceError("the $label search", "no resolvable $label anywhere in the search bracket around c = $c0"))
        end
        if i != firstindex(lncs) && i != lastindex(lncs)
            l = _golden_min(f, lncs[i-1], lncs[i+1]; iters)
            fl = f(l)
            return fl < typemax(T) ? (exp(l), fl) : (exp(lncs[i]), scores[i])
        end
        if lnbounds !== nothing
            i == firstindex(lncs) && lncs[i] <= lnbounds[1] && return exp(lnbounds[1]), scores[i]
            i == lastindex(lncs) && lncs[i] >= lnbounds[2] && return exp(lnbounds[2]), scores[i]
        end
        c0 = exp(lncs[i])                       # recenter on the winning edge and search on
    end
    throw(ConvergenceError("the $label search", "the minimum kept running off the search bracket after $maxshifts expansions"))
end

include("window.jl")

const _DEFAULT_ALPHAS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5)

"""
    Adaptive(; alphas=(0.125, 0.25, …, 1.0, 1.25, 1.5), window=WindowRule(),
             nse=(window === nothing ? 1 : 0), pilot=nothing)

The search settings of [`select_kappa`](@ref)`(x, Adaptive(; …))`, which chooses a spatially
varying smoothing scale.

- `alphas`: the exponents of the power family `c·(p̂/ḡ)^α`. They must be positive, because
  `α = 0` is the constant scale, which is always in the comparison. They are stored, and
  searched, in increasing order. The default is
  `(0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5)`.
- `window`: the window rule's settings (a [`WindowRule`](@ref)), or `nothing` to leave out the
  window candidate.
- `nse ≥ 0`: how many standard errors the power family's gain over the constant scale must
  exceed. `nse = 0` returns whichever has the smaller score. The default is `0` with a window
  candidate and `1` without, so that the power family alone must beat the constant scale
  clearly.
- `pilot`: a callable returning the positive constant scale of the pilot density from the
  sample, or from `(lower, upper)` for interval data. The default `nothing` uses the
  constant-scale [`select_kappa`](@ref).
"""
struct Adaptive{P}
    alphas::Vector{Float64}             # exponents of the power family, increasing
    window::Union{WindowRule,Nothing}
    nse::Float64
    pilot::P

    function Adaptive{P}(alphas, window, nse, pilot) where {P}
        isempty(alphas) && throw(ArgumentError("need at least one exponent in alphas"))
        all(>(0), alphas) ||
            throw(ArgumentError("the exponents in alphas must be positive; the constant scale " *
                                "(α = 0) is always compared against them"))
        nse >= 0 || throw(ArgumentError("nse must be nonnegative, got $nse"))
        return new{P}(sort!(collect(Float64, alphas)), window, nse, pilot)
    end
end

Adaptive(alphas, window, nse, pilot::P) where {P} = Adaptive{P}(alphas, window, nse, pilot)
Adaptive(; alphas=_DEFAULT_ALPHAS, window::Union{WindowRule,Nothing}=WindowRule(),
         nse::Real=window === nothing ? 1 : 0, pilot=nothing) =
    Adaptive(alphas, window, nse, pilot)

"""
    select_kappa(x, adaptive::Adaptive; support=(-Inf, Inf), rtol=cbrt(eps(T)),
                 resolution=nothing) -> κ
    select_kappa(lower, upper, adaptive::Adaptive; support=(-Inf, Inf), rtol=cbrt(eps(T))) -> κ

Choose a *spatially varying* smoothing scale by Kullback–Leibler cross-validation, and
return it ready to pass to [`DensityEstimate`](@ref) (or, for interval data,
[`IntervalDensityEstimate`](@ref)). `κ` is a `T`, an `AdaptiveScale{T}` or a `WindowScale{T}`,
with `T` the data's floating-point type.

Three kinds of scale compete on the same score: the constant scale, returned as a number
(so the fit takes the constant-`κ` path and its goodness-of-fit machinery stays available);
the power family `c·(p̂/ḡ)^α` over a pilot density `p̂`, returned as an
[`AdaptiveScale`](@ref); and the window rule, returned as a [`WindowScale`](@ref). The power
family refines the scale where the density is high; the window rule refines it where a small
window around `x` already shows a slope or curvature, and so resolves narrow features whatever
their height. The window candidate is returned when its score plus a penalty (see
[`WindowRule`](@ref)) is below that of the power family's choice. Within the power family, an
exponent `α` is returned when it beats the constant scale by more than `nse` standard errors
of the score. The settings `alphas`, `window`, `nse` and `pilot` are those of `adaptive` (see
[`Adaptive`](@ref)). Selection costs a small multiple of one constant-scale
[`select_kappa`](@ref) call per exponent plus one for the window candidate; shorten `alphas` to
trade capture for speed.

`support`, `rtol` and `resolution` work as for the constant-scale [`select_kappa`](@ref).
There is no `holdout` keyword: every score, including the pilot's, uses `holdout = :location`.
`support` is fixed across every candidate `α`/`c`, and both the pilot density and every
candidate are fitted and cross-validated on it. With `resolution = δ > 0` the scale is selected
for [`IntervalDensityEstimate`](@ref)`(x, κ; resolution=δ)`, with the pilot an interval fit
and every candidate scored by the held-out log-probabilities of the observations' intervals.
With `resolution = δ > 0` or the `(lower, upper)` form, the windows are computed from the
observations spread evenly over their intervals (see the extended help).

To choose `support` as well, call [`select_support`](@ref) first, pass its `support` here, and
fit `DensityEstimate(x, κ; support)` with the scale this returns.

# Examples
```jldoctest
julia> x = -log.(1 .- (0.5:999.5) ./ 1000);   # exponential: a jump at the left edge

julia> κ = select_kappa(x, Adaptive());      # the window rule wins here

julia> κ isa WindowScale
true

julia> d = DensityEstimate(x, κ);

julia> extrema(d.κ)[2] / extrema(d.κ)[1] > 10    # far finer at the edge than in the tail
true

julia> select_kappa(x, Adaptive(window=nothing)).α   # the power family alone
0.625

julia> select_kappa(range(0, 1; length=1000), Adaptive(window=nothing)) isa Real   # nothing to buy
true
```

# Extended help

A single scale must trade resolution in the bulk against noise in the tails. Letting `κ`
follow the density lifts that trade-off, and buys the most where a constant scale is limited
not by noise but by the density's own irregularity: a divergent or discontinuous edge, a
kink, or heavy tails, or narrow features beside broad ones. On smooth densities there is
little to buy, and the selector returns a plain number, the constant scale, when it scores
best; with `nse > 0` the power family must also beat it by that many standard errors.

The rule is a plug-in: fit a pilot density `p̂` at the constant scale `pilot(x)` (by default
[`select_kappa`](@ref)), then consider the family

    κ(x; c, α) = c · (p̂(x) / ḡ)^α,     ḡ = geometric mean of p̂ over the sample

(an [`AdaptiveScale`](@ref)). For each exponent `α` in `alphas`, `c` is chosen by
golden-section search on the leave-one-out score `KLCV(κ) = -(1/N) Σᵢ wᵢ ln Q̂₋ᵢ(xᵢ)`
generalized to a varying scale — the same criterion [`select_kappa`](@ref) minimizes,
and, like it, evaluated in closed form and `O(N)`, with no refitting. The constant scale
competes as the `α = 0` member of the same family and on the same score.

The exponent with the smallest score is the adaptive candidate; it is then compared with the
constant scale by a one-standard-error rule. Each observation `xⱼ` contributes a gain
`dⱼ = ln Q̂₋ⱼ(xⱼ; adaptive) - ln Q̂₋ⱼ(xⱼ; constant)`, where each observation takes the
leave-one-out density of the node it was merged into under that scale (the two scales can
merge the sample into different nodes). The mean of `dⱼ` is the difference of the two scores,
`gain = KLCV(constant) - KLCV(adaptive)`, and `se = std(d)/√N` is its standard error. The
adaptive scale is returned only if `gain > nse · se`. If `se == 0` (including `N = 1`), it is
returned iff `gain > 0`. A candidate whose score is not finite (an unresolvable fit) never
wins, and an exponent with no resolvable `c` near the previous exponent's is skipped; if the
constant scale is the unresolvable one, the best finite adaptive candidate is
returned regardless of `nse`. The rule applies only to this final choice, not among the
exponents.

With `window = WindowRule(; gamma, k, z, nmin)`, the window half-width `ĥ(x)` is the smallest
on a logarithmic scan (24 steps per decade, starting at the distance to the `nmin`-th
nearest observation) at which the window `|t - x| ≤ ĥ` holds at least `nmin` observations
and either of its statistics

    |Σu| / √(N/3)      or      |Σu² - N/3| / √(4N/45),      u = (t - x)/ĥ,

over its `N` observations `t` reaches `z`. Each is a moment's departure from its value for
points spread uniformly over the window, in units of its standard deviation under that
spread: a slope and a curvature test of "no structure within the window". `ĥ` is computed on
a grid of 200 evenly spaced points over the sample's range plus 201 sample quantiles (prefix
sums make each window cost two binary searches) and interpolated linearly between them. The
candidate is `κ(x) = c·exp(gamma·s(x))` with `s = ln(1/ĥ)` shifted to mean zero over the
sample, `c` chosen by the same golden-section search on KLCV as for the power family, starting
from the pilot scale. It is returned if

    KLCV(window) + k·gamma/√N  <  KLCV(power),

where `KLCV(power)` is the score of the scale the power family would return (the constant
scale or an [`AdaptiveScale`](@ref), chosen with `nse` as above); a window candidate with no
resolvable `c` never wins. The penalty accounts for `ĥ` depending on the held-out point
itself: KLCV treats the shape as fixed, and so rates the window candidate too well, by an
amount that grows with `gamma` and shrinks like `1/√N`. The default `k = 0.11` matches the
optimism measured by exact leave-one-out refits (with `ĥ` re-estimated without the held-out
point) on smooth and bimodal densities at `N = 1000` and `4000`. The threshold `z = 4` keeps
the scan above the noise of the window statistics, which have unit standard deviation in any
window without structure. `gamma` is fixed rather than selected per sample, because a
per-sample choice of `gamma` is dominated by sampling noise in the score.

For rounded or interval data, the observations recorded in an interval are placed at its
quantile midpoints, `lower + (upper - lower)·(j - ½)/m` for the `m` observations of that
interval, and the window statistics are computed from these locations; exact observations stay
where they are, and observations with an infinite bound (censored) enter the score but not the
shape. Within a window this gives the sums of `u` and `u²` expected if each observation were
spread uniformly over its interval, while the counts in different intervals keep their sampling
noise. The scores are the interval-likelihood KLCV scores, and `N` counts all observations.

`pilot` is a scale-selection method, and is called on the sample alone with no notion of
`support`; the pilot density it scales is what is fitted on `support`. So a selector with no
notion of a boundary, like [`select_kappa_sensitivity`](@ref), remains usable as `pilot` on a
bounded domain. Because `support` is fixed throughout the `α` search, composing with
[`select_support`](@ref) re-runs that search on each boundary arm's own domain, so `α` gets to
respond to whatever regularity a boundary already bought, rather than being reused from an
unbounded selection that saw a different edge.
"""
function select_kappa(x::AbstractVector{<:Real}, adaptive::Adaptive; support::Tuple{Real,Real}=(-Inf, Inf),
                      rtol::Real=cbrt(eps(float(eltype(x)))), resolution::Union{Real,Nothing}=nothing)
    if _is_rounded(resolution)
        isempty(x) && throw(ArgumentError("cannot select κ for zero observations"))
        TR = float(eltype(x))
        lower, upper = _lattice_bounds(x, TR(resolution), support)
        return select_kappa(lower, upper, adaptive; rtol, support)
    end
    _check_rtol(rtol)
    a, b = _check_support_bounds(support)
    T = float(promote_type(eltype(x), typeof(rtol), _support_eltype(a), _support_eltype(b)))
    xs = _sorted_sample(x, T)
    slo, shi = T(a), T(b)
    _check_support(xs, slo, shi)
    resolution === nothing && _check_ties(xs)
    r = T(rtol)

    # `pilot` chooses only a scalar starting scale, so it is called support-oblivious (any
    # callable returning a positive scale from the sample, including ones with no notion of a
    # boundary at all); the pilot density p̂ below is what actually carries the support.
    κ0 = T(adaptive.pilot === nothing ? select_kappa(xs; resolution=0) : adaptive.pilot(xs))
    κ0 > 0 || throw(ArgumentError("pilot must return a positive scale, got $κ0"))
    p = DensityEstimate(xs, κ0; rtol=r, support=(slo, shi))

    # The constant scale, scored on the same footing as the adaptive candidates. The 7-arg
    # `_klcv` reduces to the unbounded arithmetic exactly when `slo, shi = -Inf, Inf`.
    const_score = _klcv(_merge_presorted(xs, r / κ0)..., κ0, κ0, κ0, slo, shi)
    score(κfun) = _score_kappa(_klcv, xs, κfun, r, slo, shi)
    gain_se(κa) = _adaptive_gain(xs, κ0, κa, r, slo, shi)
    κp, sp = _select_power(score, gain_se, p, κ0, const_score, adaptive.alphas, adaptive.nse)
    adaptive.window === nothing && return κp
    return _choose_window(score, xs, κ0, κp, sp, adaptive.window)
end

# The search shared by point and interval data. `score(κfun)` is the KLCV score of the scale
# function `κfun` (NaN when unresolvable), `const_score` that of the pilot's constant scale `κ0`,
# and `gain_se(κa)` the per-observation gain of `κa` over `κ0` as `(mean, standard error)`.
# Returns `(κ, score)`: the chosen scale and its KLCV score.
function _select_power(score, gain_se, p, κ0::T, const_score::T, alphas, nse) where {T}
    loggbar = _log_geomean(p)

    # The exponents are searched in increasing order, each bracket centered on the previous
    # resolved exponent's optimum. The optimal c climbs steeply with α — a scale falling off as
    # p̂^α needs a larger c to keep the same resolution where the data actually are — and by
    # α = 1 it can sit well outside a bracket centered on the pilot scale. Walking α upward keeps
    # every optimum comfortably inside its bracket. An exponent with no resolvable c in its
    # bracket is skipped and leaves the center where it was.
    best_c, best_α, best_score = κ0, zero(T), typemax(T)
    c0 = κ0
    for α in sort!(collect(T, alphas))
        scale(c) = AdaptiveScale{T}(c, α, p, loggbar)
        res = _select_c_scored(c -> score(scale(c)), c0; skip_unresolved=true)
        res === nothing && continue
        c0, s = res
        if s < best_score
            best_score, best_c, best_α = s, c0, α
        end
    end
    best_α == 0 && return (κ0, const_score)            # no adaptive candidate resolved
    κa = AdaptiveScale(best_c, best_α, p)
    isfinite(const_score) || return (κa, best_score)   # the constant scale is unresolvable

    # Constant versus adaptive: keep the adaptive scale only if its gain in score exceeds `nse`
    # standard errors. The gain is taken as the score difference itself, so `nse = 0` is exactly
    # the comparison of scores.
    gain = const_score - best_score
    gain > 0 || return (κ0, const_score)
    nse == 0 && return (κa, best_score)
    _, se = gain_se(κa)
    isfinite(se) || error("nonfinite standard error of the adaptive gain, although both " *
                          "candidates have finite scores")
    return se == 0 || gain > nse * se ? (κa, best_score) : (κ0, const_score)  # se == 0: the gain alone decides
end

# Mean spacing of the `k` points nearest one edge of the sorted sample `xs`: the local scale a
# finite-boundary search brackets around. A true edge's optimal boundary sits within a few such
# spacings, and the score degrades only gently out to several tens.
function _edge_spacing(xs::Vector{T}, side::Symbol; k::Int=10) where {T}
    n = length(xs)
    m = min(k, n)
    m >= 2 || throw(ArgumentError("need at least two distinct points to seed a boundary search"))
    spacing = side === :left ? (xs[m] - xs[1]) / (m - 1) : (xs[n] - xs[n-m+1]) / (m - 1)
    spacing > 0 ||
        throw(ArgumentError("the $m points nearest the $side edge coincide; cannot seed a " *
                            "boundary search from a zero spacing"))
    return spacing
end

# KLCV score at scale κ on the fixed support (lo, hi), merging at the tolerance κ implies. An
# unresolvable candidate (too few surviving nodes, or a factorization that hits an exact zero
# pivot) scores NaN, which `_select_c` treats as +∞ and steps over.
function _support_klcv(xs::Vector{T}, rtol::T, κ::T, lo::T, hi::T) where {T}
    nodes, w = _merge_presorted(xs, rtol / κ)
    length(nodes) >= 2 || return T(NaN)
    try
        return _klcv(nodes, w, κ, κ, κ, lo, hi)
    catch e
        e isa ZeroPivotException && return T(NaN)
        rethrow()
    end
end

const _CHAIN_SPAN = 4.0    # the chained κ search's window, ×/÷ this factor about the warm start
const _CHAIN_GRID = 9      # its coarse grid: smaller than the plug-in-scale search's, since the
                           # window is narrow and it runs once per gap candidate
const _CHAIN_ITERS = 12
const _GAP_LO_MULT = 5.0   # the gap bracket's lower end, × the edge spacing — a hard floor,
                           # never crossed even by recentering (see `_select_gap`)
const _GAP_HI_MULT = 100.0 # the bracket's upper end, extensible outward
const _GAP_GRID = 9
const _GAP_ITERS = 12

# Best κ at the fixed support (lo, hi): golden-section on ln κ in a window ×/÷`_CHAIN_SPAN`
# about the warm start `κ0`, recentering at the window edge (the `_select_c` discipline,
# reused directly) up to a bounded number of times before erring. `κ_bounds` caps the absolute
# range (see `_select_c`'s note): without it, a chain of gap candidates can warm-start each
# other beyond where the LOO expansion stays trustworthy.
_select_kappa_at_support(xs::Vector{T}, rtol::T, lo::T, hi::T, κ0::T, κ_bounds::Tuple{T,T}) where {T} =
    _select_c(κ -> _support_klcv(xs, rtol, κ, lo, hi), κ0;
             span=_CHAIN_SPAN, ngrid=_CHAIN_GRID, iters=_CHAIN_ITERS, bounds=κ_bounds)

# Golden-section-refined grid search for one side's boundary gap, over ln(gap) starting from
# the bracket [`_GAP_LO_MULT`, `_GAP_HI_MULT`] × `spacing` and extending outward — never
# inward, past the floor — when the grid's minimum sits at the high edge: a more distant wall
# is always a safe direction to keep searching, since it converges to the unbounded fit as the
# gap grows. The low edge is a hard floor, the tightest gap ever tried: within a few edge
# spacings of the extreme data point, a natural (Neumann) boundary reflects the nearest
# interior points back onto it and inflates that point's leave-one-out likelihood on *any*
# sample, genuine edge or not — checked directly against a brute-force leave-one-out refit, so
# this is a property of the reflecting boundary condition itself, not an artifact of the
# package's first-order LOO expansion. Unlike `_select_c`'s two-sided recentering (which would
# chase this reflection effect down to an arbitrarily tight, spurious gap on every sample,
# hard-edge or smooth alike), only the outward direction is treated as informative.
function _select_gap(score, spacing::T) where {T}
    f = _lnscore(score, T)
    lo0 = log(spacing) + log(T(_GAP_LO_MULT))
    hi0 = log(spacing) + log(T(_GAP_HI_MULT))
    for _ in 0:_CSHIFTS
        lncs = range(lo0, hi0; length=_GAP_GRID)
        _, i = _grid_argmin(f, lncs, T; reverse=true)    # visit wide (large gap) to narrow
        i === nothing &&
            throw(ConvergenceError("the boundary-gap search", "no resolvable boundary gap anywhere in the search bracket above $(exp(lo0))"))
        i == firstindex(lncs) && return exp(lo0)     # hit the floor: accept it, don't recenter
        i == lastindex(lncs) || return exp(_golden_min(f, lncs[i-1], lncs[i+1]; iters=_GAP_ITERS))
        lo0, hi0 = hi0, hi0 + (hi0 - lo0)            # extend outward and search on
    end
    throw(ConvergenceError("the boundary-gap search", "the minimum kept running off the search bracket after $_CSHIFTS expansions"))
end

# Finite-boundary search for one side (:left or :right), the opposite side fixed at
# `other_lo`/`other_hi`. `_select_gap` finds the gap; κ is re-optimized at every candidate by
# `_select_kappa_at_support` in a window warm-started from the *previous* (wider) candidate's
# optimum, so consecutive candidates' optima — which move continuously with the gap — stay
# inside their narrow search window rather than triggering its recentering fallback. `κ0`
# warm-starts the widest candidate. Returns the winning `(gap, κ, score)`; `score` is directly
# comparable to the ∞ arm's (the same support with this side left unbounded).
function _search_boundary(xs::Vector{T}, rtol::T, κ0::T, side::Symbol,
                          other_lo::T, other_hi::T, κ_bounds::Tuple{T,T}) where {T}
    spacing = _edge_spacing(xs, side)
    κstate = Ref(κ0)
    function score_gap(gap::T)
        lo, hi = side === :left ? (xs[1] - gap, other_hi) : (other_lo, xs[end] + gap)
        κ = _select_kappa_at_support(xs, rtol, lo, hi, κstate[], κ_bounds)
        κstate[] = κ
        return _support_klcv(xs, rtol, κ, lo, hi)
    end
    gap = _select_gap(score_gap, spacing)
    lo, hi = side === :left ? (xs[1] - gap, other_hi) : (other_lo, xs[end] + gap)
    κ = _select_kappa_at_support(xs, rtol, lo, hi, κstate[], κ_bounds)
    return gap, κ, _support_klcv(xs, rtol, κ, lo, hi)
end

# Whether `challenger` beats `incumbent` by more than floating-point/golden-section noise: a
# relative margin, not a bare `<`. A KLCV score carries ~1e-10-level noise from golden-section
# refinement and summation order, and — per `_select_gap`'s note — a boundary at the gap floor
# can match the unbounded score to within that noise on *any* sample; a genuine edge's gain is
# orders of magnitude larger (percent-level), so the margin only screens out noise.
const _SUPPORT_MARGIN = 1e-8
_beats(challenger::T, incumbent::T) where {T} =
    challenger + _SUPPORT_MARGIN * max(abs(incumbent), oneunit(T)) < incumbent

"""
    select_support(x; rtol=cbrt(eps(T)), resolution=nothing, kappas=nothing)
        -> (; κ, support)

Choose a domain `support = (a, b)` — either side possibly infinite — together with the
smoothing scale `κ`, jointly, by the same Kullback–Leibler cross-validation score
[`select_kappa`](@ref) minimizes. `κ` has the data's floating-point type. Pass the result
straight to [`DensityEstimate`](@ref):

    r = select_support(x)
    d = DensityEstimate(x, r.κ; support = r.support)

A boundary is imposed on a side only when it wins that cross-validation, never assumed from
the fact that one side of the data has an edge; a side that does not win stays `±Inf`. A
finite boundary is always placed outward of the extreme data point on its side, and never
closer to it than five times the mean spacing of the data near that edge. When neither side
wins, the support is `(-Inf, Inf)` and the returned `κ` equals `select_kappa(x; rtol, kappas)`
exactly.

`kappas` and `rtol` mean what they do for [`select_kappa`](@ref): the grid bracketing the κ
search (by default extended as needed), and the node-merging tolerance (a fraction of the local
smoothing length). The κ searched at each finite-boundary candidate stays within the range of
the grid the unbounded search used.

More than 1% repeated values throw an `ArgumentError`, as for [`select_kappa`](@ref), unless
`resolution = 0` is given, which treats the values as exact points. `select_support` has no
form for rounded (interval) data, so `resolution > 0` throws an `ArgumentError`.

# Examples
```jldoctest
julia> x = -log.(1 .- (0.5:499.5) ./ 500);   # exponential draw: a jump edge at the left

julia> r = select_support(x);

julia> r.support[1] <= minimum(x) && r.support[2] == Inf   # never inward of the data
true

julia> d = DensityEstimate(x, r.κ; support = r.support);

julia> d.lo == r.support[1] && d.hi == r.support[2]
true
```

# Extended help

Each side is searched independently and sequentially — the left boundary first (with the
right side unbounded), then the right boundary against the left side's winner — and on each
side the unbounded (`±Inf`) candidate always competes: that side gets a finite boundary only
if the best finite candidate's KLCV beats the score of leaving it unbounded by more than a
small margin (screening out golden-section/floating-point noise, not a real effect size). A
wall is not always safe to add: placed too far past the data it can raise the KLCV score
rather than lower it (a flat field props mass into an empty margin where a decaying tail would
not).

A finite candidate on one side is a gap `Δ > 0`, the distance *outward* from the extreme data
point on that side (`a = x₁ - Δ` on the left, `b = x_N + Δ` on the right), searched by
golden-section on `ln Δ` over a bracket of `[5, 100]` times the mean spacing of the ten data
points nearest that edge (extensible further outward, never inward). The lower end is a hard
floor, not merely a starting guess: closer than a few edge spacings, a natural boundary
reflects the nearest interior points back onto the extreme point and inflates its leave-one-out
likelihood on *any* sample, edge or not, so gaps tighter than the floor are excluded rather
than searched (this is a property of the reflecting boundary condition itself — confirmed
against a brute-force leave-one-out refit — not a search artifact). `κ` is re-selected at every
gap candidate rather than held fixed, because the two are coupled at a hard edge (the optimal
`κ` can move to a fraction of its unbounded value once a wall is added); candidates are
searched from the widest gap to the narrowest, and each candidate's `κ` search is warm-started
in a narrow window about the *previous* candidate's optimum rather than repeating a full search
from scratch, since `κ*` moves continuously with the gap.

The full constant-scale search of [`select_kappa`](@ref) runs at two points only: once at the
start, to seed the unbounded arm's competing score and the first (and widest) gap candidate's
`κ` warm start; and once at the end, to refine `κ` at the winning support over the full
`kappas` grid (the chained inner searches above use a narrower window, for speed). When
neither side wins, no refinement is made and the returned `κ` *is* the first search's result.
"""
function select_support(x::AbstractVector{<:Real}; rtol::Real=cbrt(eps(float(eltype(x)))),
                        resolution::Union{Real,Nothing}=nothing,
                        kappas::Union{Nothing,AbstractVector{<:Real}}=nothing)
    _is_rounded(resolution) &&
        throw(ArgumentError("select_support has no form for rounded (interval) data; use " *
                            "select_kappa(x; resolution=$resolution), or resolution=0 to treat " *
                            "the values as exact points"))
    kappas === nothing || _check_kappa_grid(kappas)
    _check_rtol(rtol)
    T = float(promote_type(eltype(x), typeof(rtol)))
    xs = _sorted_sample(x, T)
    resolution === nothing && _check_ties(xs)
    # Both boundary searches need a nonzero edge spacing. Checking it before any κ search
    # reports coincident edge points as such, rather than as a κ grid with no finite score.
    _edge_spacing(xs, :left); _edge_spacing(xs, :right)
    r = T(rtol)
    full_search(support) = _select_by_score(_klcv, xs, kappas, r, support)

    # The chained κ search never leaves the range of the grid the unbounded search used — see
    # `_select_c`'s note on why an unbounded chain of warm starts is unsafe.
    κ_inf, κlo, κhi = full_search((T(-Inf), T(Inf)))
    κ_bounds = (T(κlo), T(κhi))
    score_cur = _support_klcv(xs, r, κ_inf, T(-Inf), T(Inf))
    lo, hi, κcur = T(-Inf), T(Inf), κ_inf

    gapL, κL, scoreL = _search_boundary(xs, r, κcur, :left, T(-Inf), hi, κ_bounds)
    if _beats(scoreL, score_cur)
        lo, κcur, score_cur = xs[1] - gapL, κL, scoreL
    end

    gapR, κR, scoreR = _search_boundary(xs, r, κcur, :right, lo, T(Inf), κ_bounds)
    if _beats(scoreR, score_cur)
        hi, κcur = xs[end] + gapR, κR
    end

    κ = isinf(lo) && isinf(hi) ? κ_inf : first(full_search((lo, hi)))
    return (; κ, support=(lo, hi))
end

include("interval.jl")

end # module
