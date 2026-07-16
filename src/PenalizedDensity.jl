module PenalizedDensity

using LinearAlgebra: LinearAlgebra, I, SymTridiagonal, ZeroPivotException, dot, ldiv!, ldlt!, mul!
using QuadGK: quadgk
using SpecialFunctions: erfc, erfcx
using Statistics: Statistics, quantile

export DensityEstimate, amplitude, action, select_kappa_ms, select_kappa_cv, select_kappa_kl, kappa_interval
export AdaptiveScale, select_kappa_adaptive
export chisq, expected_chisq, chisq_reference, ChisqReference, chisq_pdf, chisq_ccdf, pvalue
export cdf, quantile

"""
    DensityEstimate(x::AbstractVector{T}, κ; rtol=cbrt(eps(T)))

Estimate a continuous one-dimensional probability density from sample points `x`,
using the scalar-field method of Holy, *Phys. Rev. Lett.* **79**, 3545 (1997).

The density is written as `Q(x) = ψ(x)^2`, where the amplitude `ψ` minimizes the
action

    S[ψ] = ∫ (λ/κ(x)²) (ψ')² dx - 2 Σᵢ ln ψ(xᵢ)

subject to `∫ ψ² dx = 1`, with `λ` the normalization multiplier. The smoothing scale
`κ` sets the width of each point's contribution; larger `κ` gives a rougher estimate.
See [`select_kappa_kl`](@ref) for choosing it automatically (the recommended default;
[`select_kappa_cv`](@ref), [`select_kappa_ms`](@ref), and [`kappa_interval`](@ref) are
alternatives).

`κ` is either a positive number, giving one scale everywhere, or a callable `κ(x)`
returning the scale local to `x`. A callable is evaluated at the midpoint of each
inter-node interval, and at the outermost nodes for the two tails, so the fit resolves
a piecewise-constant scale: `d.κ[k]` is the rate on `(d.x[k], d.x[k+1])`, and `d.κL`,
`d.κR` the tail rates. Making `κ` large where the density is high and small where it is
low buys resolution where the data can pay for it. The penalty weight is `1/κ(x)²` on
`(ψ')²`, which keeps the pressure to normalize spatially uniform. The exact goodness-of-fit
machinery ([`chisq_reference`](@ref) and everything built on it) supports a varying `κ`;
only the large-`N` approximation ([`expected_chisq`](@ref)`(d)`, `method=:largeN`) needs a
constant one, and throws otherwise.

Between sorted data points `ψ` solves `ψ'' = κ² ψ`, i.e. it is a sum of rising and
falling exponentials, and decays as `e^{-κ|x|}` in the tails. The nodal amplitudes
`ψ(xᵢ)` satisfy a symmetric tridiagonal system whose solution is the minimizer of a
strictly convex potential; normalization is then a rescaling.

The returned object is callable: `d(x)` evaluates the density `Q(x)` at any real
`x`, and it can be broadcast over arrays. Use [`amplitude`](@ref) for `ψ(x)`.

Repeated points, and points closer than `rtol / κ(x)` (i.e. within a fraction `rtol` of
the local smoothing length), are merged into one node carrying the count as its integer
weight, so weighted data is handled naturally. Without merging, the resulting
tridiagonal system can be nearly singular.

Passing `κ` as a keyword, `DensityEstimate(x; κ)`, is deprecated in favor of the
positional form.

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
```
"""
struct DensityEstimate{T<:AbstractFloat,K}
    x::Vector{T}   # sorted, distinct node locations
    w::Vector{T}   # weight (multiplicity) at each node
    ψ::Vector{T}   # normalized amplitude at the nodes
    κ::K           # smoothing scale: one number, or one per inter-node interval
    κL::T          # decay rate of the left tail
    κR::T          # decay rate of the right tail
    λ::T           # normalization multiplier (diagnostic)

    function DensityEstimate{T,K}(x, w, ψ, κ, κL, κR, λ) where {T<:AbstractFloat,K}
        length(x) == length(w) == length(ψ) ||
            throw(DimensionMismatch("nodes, weights, and amplitudes must have equal length, " *
                                    "got $(length(x)), $(length(w)), $(length(ψ))"))
        _check_interval_scale(κ, length(x))
        return new{T,K}(x, w, ψ, κ, κL, κR, λ)
    end
end

# A per-interval scale carries one rate for each of the n-1 gaps between n nodes; a mismatch
# would leave surplus intervals silently unused rather than error at `d.κ[k]`.
_check_interval_scale(::Real, n) = nothing
_check_interval_scale(κ::AbstractVector, n) =
    length(κ) == n - 1 || throw(DimensionMismatch(
        "a per-interval scale needs one rate per inter-node interval: " *
        "got $(length(κ)) rates for $n nodes"))

DensityEstimate{T}(x, w, ψ, κ::Real, κL, κR, λ) where {T} =
    DensityEstimate{T,T}(x, w, ψ, κ, κL, κR, λ)
DensityEstimate{T}(x, w, ψ, κ::AbstractVector, κL, κR, λ) where {T} =
    DensityEstimate{T,Vector{T}}(x, w, ψ, κ, κL, κR, λ)

function DensityEstimate(x::AbstractVector{R}, κ; rtol::Real=cbrt(eps(R))) where R<:Real
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    isempty(x) && throw(ArgumentError("cannot fit a density to zero points"))
    return _estimate(x, κ, rtol)
end

function DensityEstimate(x::AbstractVector{R}; κ, rtol::Real=cbrt(eps(R))) where R<:Real
    κ isa Real || throw(ArgumentError("a callable smoothing scale must be passed positionally: " *
                                      "`DensityEstimate(x, κ)`"))
    Base.depwarn("`DensityEstimate(x; κ)` is deprecated, pass the scale positionally as " *
                 "`DensityEstimate(x, κ)`.", :DensityEstimate)
    return DensityEstimate(x, κ; rtol)
end

function _estimate(x::AbstractVector{R}, κ::Real, rtol::Real) where {R<:Real}
    κ > 0 || throw(ArgumentError("κ must be positive, got $κ"))
    T = float(promote_type(R, typeof(κ), typeof(rtol)))
    nodes, weights = _merge_presorted(_sorted_sample(x, T), T(rtol) / T(κ))
    return _fit(nodes, weights, T(κ))
end

# The nodes are not known until the data has been merged, and the merge tolerance is itself
# rtol/κ(x) — so there is no node geometry a caller could have aligned a per-interval vector
# to. The scale has to arrive as a function of position.
_estimate(::AbstractVector{<:Real}, ::AbstractVector, ::Real) =
    throw(ArgumentError("the smoothing scale cannot be given as a vector: node merging depends " *
                        "on the local scale, so the nodes it would index do not exist yet. Pass a " *
                        "callable `κ(x)` instead; the fit reports the realized per-interval rates."))

function _estimate(x::AbstractVector{R}, κfun, rtol::Real) where {R<:Real}
    # The scale's own precision joins the promotion, as a scalar κ's would; sampling κfun at a
    # data point is the only way to see it.
    T = float(promote_type(R, typeof(rtol), typeof(κfun(first(x)))))
    return _fit(_merge_and_realize(_sorted_sample(x, T), κfun, T(rtol))...)
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
Base.show(io::IO, d::DensityEstimate) = print(io, "DensityEstimate with $(length(d.x)) distinct nodes, $(sum(d.w)) total weight, $(_show_kappa(d)), λ=$(d.λ)")

# Fit from already-merged distinct nodes and their weights.
function _fit(nodes::Vector{T}, weights::Vector{T}, κ::T) where {T}
    ψ = _solve_amplitude(nodes, weights, κ)
    Z = _norm_sq(nodes, ψ, κ)
    ψ ./= sqrt(Z)
    λ = κ * Z                       # scaling law: normalized ψ solves Mψ = (κ/λ)/ψ
    return DensityEstimate{T}(nodes, weights, ψ, κ, κ, κ, λ)
end

# Piecewise-constant scale. The assembled operator carries an arbitrary overall factor κ̄
# (see `roughness_operator`), which cancels from the normalized amplitude and leaves the
# multiplier λ = κ̄ Z well defined: the stationarity condition of the unscaled operator is
# Mψ = (1/λ) w ⊘ ψ, whose constant-κ specialization is the scaling law above.
function _fit(nodes::Vector{T}, weights::Vector{T}, κs::Vector{T}, κL::T, κR::T) where {T}
    κ̄ = _reference_scale(κs, κL, κR)
    ψ = _solve_amplitude(roughness_operator(nodes, κs, κL, κR, κ̄), weights)
    Z = _norm_sq(nodes, ψ, κs, κL, κR)
    ψ ./= sqrt(Z)
    return DensityEstimate{T}(nodes, weights, ψ, κs, κL, κR, κ̄ * Z)
end

# Reject scale values a fit cannot use.
_check_kappa(κ, x) =
    isfinite(κ) && κ > 0 ? κ :
    throw(ArgumentError("the smoothing scale must be finite and positive, got κ($x) = $κ"))

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

"""
    _sorted_sample(x, T) -> xs::Vector{T}

A sorted, one-based working copy of the sample `x`, whatever its axes. Every index the fit
takes afterwards — into the sample, into a scale realized on it, into the merged nodes —
is an index into this copy, and none of them escape, so the caller's axes have nothing to
propagate to.
"""
function _sorted_sample(x::AbstractVector, ::Type{T}) where {T}
    xs = Vector{T}(undef, length(x))
    i = firstindex(xs)
    for xi in x                     # iterate, rather than index, to stay axis-agnostic
        xs[i] = xi
        i += 1
    end
    return sort!(xs)
end

# Collapse runs of an already-sorted sequence `xs` within `atol` of the run's first member.
# Factored out so kappa_interval can reMerge one sorted copy at many tolerances.
function _merge_presorted(xs, atol::T) where {T}
    nodes = T[]
    weights = T[]
    for xi in xs
        xk = T(xi)
        if !isempty(nodes) && xk - nodes[end] <= atol
            weights[end] += oneunit(T)
        else
            push!(nodes, xk)
            push!(weights, oneunit(T))
        end
    end
    return nodes, weights
end

# As above, but with a tolerance `rtol / κ` local to the run's first member, whose scale is
# `κx[i]` for the point `xs[i]`: the merge threshold is a fraction `rtol` of the smoothing
# length there. The scales come in already realized on `xs` because the merge threshold is
# what *produces* the nodes — a caller has no node geometry to align a per-node vector to.
function _merge_presorted(xs, rtol::T, κx::AbstractVector{T}) where {T}
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
    end
    return nodes, weights
end

# Merge the sample at the tolerance a scale implies, then realize that scale on the nodes the
# merge produced. This pairing is the whole entry into a piecewise-constant fit: the merge
# needs the scale at the sample points, and the fit needs it on the nodes and tails.
function _merge_and_realize(xs::Vector{T}, κfun, rtol::T) where {T}
    nodes, weights = _merge_presorted(xs, rtol, _kappa_sorted(κfun, xs, T))
    κs, κL, κR = _kappa_profile(nodes, κfun, T)
    return nodes, weights, κs, κL, κR
end

# Tridiagonal operator M (SPD) coupling the nodal amplitudes.
# Off-diagonal e[k] = -csch(κ hₖ); diagonal d[i] accumulates coth(κ hₖ) from each
# adjacent interval and +1 from each adjacent tail.
function roughness_operator(x::Vector{T}, κ::T) where {T<:AbstractFloat}
    n = length(x)
    n >= 1 || throw(ArgumentError("need at least one node to build the roughness operator"))
    d = zeros(T, n)
    e = zeros(T, n - 1)
    d[1] += oneunit(T)              # left tail
    d[n] += oneunit(T)              # right tail
    for k in 1:n-1
        θ = κ * (x[k+1] - x[k])
        d[k]   += coth(θ)
        d[k+1] += coth(θ)
        e[k]    = -csch(θ)          # coth/csch stay finite as θ → ∞ (isolated points)
    end
    return SymTridiagonal(d, e)     # M
end

# The same operator for a piecewise-constant scale: interval k (rate κs[k], θ = κs[k]·hₖ)
# contributes coth(θ)/κs[k] to each adjacent diagonal entry and -csch(θ)/κs[k] off-diagonal,
# and each tail 1/κ_edge to its own. Dividing through by one κ no longer cancels the entries,
# so the rates survive explicitly.
#
# Everything is scaled by the reference rate κ̄. That factor is arbitrary — it rescales the
# unnormalized amplitude by κ̄^{-1/2} and drops out of both the normalized fit and λ = κ̄ Z —
# but it fixes the magnitude the Newton solve sees. Taking κ̄ to be the typical rate keeps the
# entries O(1), and at a constant κ (where κ̄ = κ) reproduces `roughness_operator(x, κ)` entry
# for entry.
function roughness_operator(x::Vector{T}, κs::Vector{T}, κL::T, κR::T, κ̄::T) where {T<:AbstractFloat}
    n = length(x)
    n >= 1 || throw(ArgumentError("need at least one node to build the roughness operator"))
    length(κs) == n - 1 ||
        throw(DimensionMismatch("$n nodes bound $(n-1) intervals, but got $(length(κs)) scales"))
    d = zeros(T, n)
    e = zeros(T, n - 1)
    d[1] += κ̄ / κL                  # left tail
    d[n] += κ̄ / κR                  # right tail
    for k in 1:n-1
        θ = κs[k] * (x[k+1] - x[k])
        u = κ̄ / κs[k]
        d[k]   += u * coth(θ)
        d[k+1] += u * coth(θ)
        e[k]    = -u * csch(θ)      # coth/csch stay finite as θ → ∞ (isolated points)
    end
    return SymTridiagonal(d, e)
end

# M for a bare scale, whichever form it takes. A constant κ is its own reference scale, so
# this reduces to `roughness_operator(x, κ)` entry for entry; a per-interval κ is assembled
# in units of the geometric-mean rate, as the fit does.
_operator(x::Vector{T}, κ::T, κL::T, κR::T) where {T} = roughness_operator(x, κ)
_operator(x::Vector{T}, κs::Vector{T}, κL::T, κR::T) where {T} =
    roughness_operator(x, κs, κL, κR, _reference_scale(κs, κL, κR))

# F(ψ) = ½ ψ'Mψ - Σ wᵢ ln ψᵢ, the potential minimized by _solve_amplitude.
function _objective(M::SymTridiagonal{T}, w::Vector{T}, ψ::Vector{T}) where {T<:AbstractFloat}
    s = zero(T)
    for i in eachindex(w, ψ)
        s += w[i] * log(ψ[i])       # requires ψ > 0, which is enforced by the caller
    end
    return dot(ψ, M, ψ) / 2 - s
end

"""
    _solve_amplitude(M, w)    -> ψ
    _solve_amplitude(x, w, κ) -> ψ

Minimize the strictly convex potential `F(ψ) = ½ ψ'Mψ - Σ wᵢ ln ψᵢ` over `ψ > 0`
by a damped Newton iteration with an SPD tridiagonal Hessian. The minimizer solves
`Mψ = w ./ ψ`, i.e. the field equation at unit multiplier; the caller rescales it
to impose normalization.

Each step factorizes the tridiagonal Hessian in place (`ldlt!`/`ldiv!`) and backtracks
along the Newton direction to keep `ψ > 0` with Armijo decrease. Iteration stops when the
Newton decrement `λ² = ∇FᵀΔ` drops below a relative tolerance, or when the line search can
no longer decrease `F`.
"""
function _solve_amplitude(M::SymTridiagonal{T}, w::Vector{T}; maxiter::Int=100) where {T<:AbstractFloat}
    n = length(w)
    ψ = fill(oneunit(T), n)             # strictly positive start
    g = similar(ψ); Δ = similar(ψ); ψnew = similar(ψ)
    Hdv = similar(ψ); Hev = similar(M.ev)   # Hessian factorization scratch
    ctol = cbrt(eps(T))^2               # relative Newton-decrement tolerance
    Fψ = _objective(M, w, ψ)
    for _ in 1:maxiter
        mul!(g, M, ψ)
        @. g -= w / ψ                    # ∇F = Mψ - w./ψ
        @. Hdv = M.dv + w / ψ^2          # diagonal of ∇²F; off-diagonal equals M.ev
        Hev .= M.ev                      # ldlt! overwrites its arguments; refill each step
        Δ .= g
        ldiv!(ldlt!(SymTridiagonal(Hdv, Hev)), Δ)   # Δ = (∇²F)⁻¹ ∇F
        decrement = dot(g, Δ)               # Newton decrement λ² = ∇Fᵀ(∇²F)⁻¹∇F ≥ 0
        decrement <= ctol * max(oneunit(T), abs(Fψ)) && break
        # Largest α ≤ 1 keeping ψ - αΔ strictly positive, then Armijo backtracking.
        α = one(T)
        for i in eachindex(ψ, Δ)
            Δ[i] > 0 && (α = min(α, ψ[i] / Δ[i]))
        end
        α < one(T) && (α *= oftype(α, 0.99))
        armijo = false
        local Fnew
        while α >= eps(T)
            @. ψnew = ψ - α * Δ
            Fnew = _objective(M, w, ψnew)
            if Fnew <= Fψ - α * decrement / 4
                armijo = true
                break
            end
            α /= 2
        end
        armijo || break                     # no decrease available ⇒ converged to rounding
        copyto!(ψ, ψnew)
        Fψ = Fnew
    end
    return ψ
end
_solve_amplitude(x::Vector{T}, w::Vector{T}, κ::T; kwargs...) where {T<:AbstractFloat} =
    _solve_amplitude(roughness_operator(x, κ), w; kwargs...)

# ∫ ψ² dx for the hyperbolic interpolant with exponential tails, as a tridiagonal
# quadratic form evaluated at the nodal amplitudes.
function _norm_sq(x::Vector{T}, ψ::Vector{T}, κ::T) where {T}
    n = length(x)
    Z = (ψ[1]^2 + ψ[n]^2) / (2κ)    # tails
    for k in 1:n-1
        θ = κ * (x[k+1] - x[k])
        ct, cs = coth(θ), csch(θ)
        # Endpoint and cross contributions of ∫ψ² over the interval, written with
        # coth/csch so they stay finite as θ → ∞ rather than overflowing via sinh.
        fdiag  = (ct - θ * cs^2) / (2κ)
        fcross = cs * (θ * ct - oneunit(T)) / (2κ)
        Z += fdiag * (ψ[k]^2 + ψ[k+1]^2) + 2 * fcross * ψ[k] * ψ[k+1]
    end
    return Z
end

# ∫ ψ² dx for a piecewise-constant scale. The interpolant on interval k and the tail decays
# are set by the rates themselves, not by the operator's overall factor, so this is the
# physical mass whatever κ̄ the amplitude was solved in.
function _norm_sq(x::Vector{T}, ψ::Vector{T}, κs::Vector{T}, κL::T, κR::T) where {T}
    n = length(x)
    Z = ψ[1]^2 / (2κL) + ψ[n]^2 / (2κR)     # tails
    for k in 1:n-1
        κ = κs[k]
        θ = κ * (x[k+1] - x[k])
        ct, cs = coth(θ), csch(θ)
        fdiag  = (ct - θ * cs^2) / (2κ)
        fcross = cs * (θ * ct - oneunit(T)) / (2κ)
        Z += fdiag * (ψ[k]^2 + ψ[k+1]^2) + 2 * fcross * ψ[k] * ψ[k+1]
    end
    return Z
end

# Z = ∫ψ² and Gψ = ½ ∂Z/∂ψ, where Z = ψᵀGψ: the mass and the action of its Gram operator,
# from one pass over the per-interval coth/csch coefficients. The leave-one-out expansion
# needs both. Each tail decays at its own rate, and each interval integrates at its own.
function _norm_sq_gram(x::Vector{T}, ψ::Vector{T}, κ, κL::T, κR::T) where {T}
    n = length(x)
    Gψ = zeros(T, n)
    Z = ψ[1]^2 / (2κL) + ψ[n]^2 / (2κR)     # tails
    Gψ[1] += ψ[1] / (2κL)
    Gψ[n] += ψ[n] / (2κR)
    for k in 1:n-1
        κk = _kappa(κ, k)
        θ = κk * (x[k+1] - x[k])
        ct, cs = coth(θ), csch(θ)
        fdiag  = (ct - θ * cs^2) / (2κk)
        fcross = cs * (θ * ct - oneunit(T)) / (2κk)
        Z += fdiag * (ψ[k]^2 + ψ[k+1]^2) + 2 * fcross * ψ[k] * ψ[k+1]
        Gψ[k]   += fdiag * ψ[k]   + fcross * ψ[k+1]
        Gψ[k+1] += fdiag * ψ[k+1] + fcross * ψ[k]
    end
    return Z, Gψ
end

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
        h = x[k+1] - x[k]; θ = κ * h; ct = coth(θ); cs = csch(θ)
        fdiag  = (ct - θ * cs^2) / (2κ)
        fcross = cs * (θ * ct - oneunit(T)) / (2κ)
        Z += fdiag * (ψ[k]^2 + ψ[k+1]^2) + 2 * fcross * ψ[k] * ψ[k+1]
        dfdiag  = h * cs^2 * (θ * ct - oneunit(T)) / κ - (ct - θ * cs^2) / (2κ^2)
        dfcross = h * cs * (2ct - θ * (ct^2 + cs^2)) / (2κ) - cs * (θ * ct - oneunit(T)) / (2κ^2)
        dZ += dfdiag * (ψ[k]^2 + ψ[k+1]^2) + 2 * dfcross * ψ[k] * ψ[k+1]
        Gψ[k]   += fdiag * ψ[k]   + fcross * ψ[k+1]
        Gψ[k+1] += fdiag * ψ[k+1] + fcross * ψ[k]
    end
    return Z, dZ, Gψ
end

# ∫ψ⁴ dx = ∫Q² for the hyperbolic interpolant with exponential tails, as a sum of per-interval
# closed forms. On each interval ψ solves ψ'' = κ²ψ, so u'² - κ²u² = E is constant and
# d/dx(u³u') = 3u²u'² + κ²u⁴; integrating gives ∫u⁴ = ([u³u']ₖ^{k+1} - 3E ∫u²)/(4κ²). The
# boundary and energy terms are written through coshθ - 1 = 2 sinh²(θ/2) and the endpoint
# difference q - p, keeping them accurate for near-coincident points (θ → 0, where the naive
# csch⁴ forms lose all precision) while staying finite for isolated points (θ → ∞). Used by
# select_kappa_cv for the ∫Q² term.
#
# The derivation is local to one interval, so a piecewise-constant scale changes nothing but
# which κ each term carries.
function _int_quartic(x::Vector{T}, ψ::Vector{T}, κ, κL::T, κR::T) where {T}
    n = length(x)
    Q2 = ψ[1]^4 / (4κL) + ψ[n]^4 / (4κR)    # tails: ∫ψ₁⁴ e^{4κL(x-x₁)} dx and its mirror
    for k in 1:n-1
        κk = _kappa(κ, k)
        p, q = ψ[k], ψ[k+1]
        θ = κk * (x[k+1] - x[k])
        ct, cs = coth(θ), csch(θ)
        Δ = q - p
        cm1 = 2 * sinh(θ / 2)^2                              # coshθ - 1
        boundary = κk * cs * (cm1 * (p^4 + q^4) + Δ^2 * (p^2 + p*q + q^2))  # [u³u']ₖ^{k+1}
        E = κk^2 * cs^2 * (Δ^2 - 2 * p * q * cm1)            # u'² - κ²u²
        fdiag  = (ct - θ * cs^2) / (2κk)
        fcross = cs * (θ * ct - one(T)) / (2κk)
        Iseg = fdiag * (p^2 + q^2) + 2 * fcross * p * q      # ∫u² over the interval
        Q2 += (boundary - 3 * E * Iseg) / (4κk^2)
    end
    return Q2
end
_int_quartic(x::Vector{T}, ψ::Vector{T}, κ::T) where {T} = _int_quartic(x, ψ, κ, κ, κ)

# (dM/dκ) ψ: the κ-derivative of roughness_operator's coth/csch entries, applied to ψ. The tails are
# κ-independent and drop out.
function _dM_dκ_mul(x::Vector{T}, κ::T, ψ::Vector{T}) where {T}
    n = length(x)
    r = zeros(T, n)
    for k in 1:n-1
        h = x[k+1] - x[k]; θ = κ * h; cs = csch(θ); ct = coth(θ)
        dd = -h * cs^2                  # d/dκ coth(θ)
        de =  h * cs * ct               # d/dκ (-csch(θ))
        r[k]   += dd * ψ[k]   + de * ψ[k+1]
        r[k+1] += dd * ψ[k+1] + de * ψ[k]
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

Evaluate the amplitude `ψ(x)` (so that the density is `d(x) == ψ(x)^2`) at real `x`,
which may be a scalar or an array.
"""
amplitude(d::DensityEstimate, x::Real) = _amplitude(d, x)
amplitude(d::DensityEstimate, x::AbstractArray) = map(xi -> _amplitude(d, xi), x)

function _amplitude(d::DensityEstimate{T}, x::Real) where {T}
    xs = d.x
    n = length(xs)
    x <= xs[1] && return d.ψ[1] * exp(d.κL * (x - xs[1]))
    x >= xs[n] && return d.ψ[n] * exp(-d.κR * (x - xs[n]))
    return _amplitude(d, searchsortedlast(xs, x), x)    # xs[k] <= x < xs[k+1]
end

# ψ(x) inside interval k, i.e. for xs[k] ≤ x ≤ xs[k+1]. Split out so a caller that already
# knows which interval x falls in — a sorted sweep — need not search for it.
function _amplitude(d::DensityEstimate{T}, k::Integer, x::Real) where {T}
    xs, ψ = d.x, d.ψ
    κ = _kappa(d, k)
    a = κ * (xs[k+1] - x)           # a, b ≥ 0 and a + b = θ
    b = κ * (x - xs[k])
    return ψ[k] * _sinh_ratio(a, a + b) + ψ[k+1] * _sinh_ratio(b, a + b)
end

# ln Q(t) = 2 ln ψ(t) at every position of the sorted vector `ts`, advancing through the
# nodes alongside `ts` in one pass. Evaluating pointwise would binary-search for each
# position, and a plug-in scale is realized on O(N) positions for every candidate it scores.
# The logarithm keeps the far tails informative where Q itself underflows to zero.
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
        ψt = t <= xs[1] ? d.ψ[1] * exp(d.κL * (t - xs[1])) :
             t >= xs[n] ? d.ψ[n] * exp(-d.κR * (t - xs[n])) :
             _amplitude(d, k, t)
        out[i] = 2 * log(ψt)
    end
    return out
end

# sinh(u)/sinh(θ) for 0 ≤ u ≤ θ, evaluated without overflow at large θ.
_sinh_ratio(u::T, θ::T) where {T} = exp(u - θ) * expm1(-2u) / expm1(-2θ)

# cosh(u)/sinh(θ) for 0 ≤ u ≤ θ, evaluated without overflow at large θ (companion to
# _sinh_ratio). With u = θ it is coth θ, also overflow-safe.
_cosh_ratio(u::T, θ::T) where {T} = -exp(u - θ) * (1 + exp(-2u)) / expm1(-2θ)

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

# Cumulative masses at the nodes, F[k] = ∫_{-∞}^{x[k]} ψ² dt, accumulated from the
# per-interval closed forms (the same integrals as _norm_sq), together with the grand
# total F[n] + right-tail mass. ψ is normalized so the total is 1 up to roundoff; cdf and
# quantile divide by the recomputed total rather than assuming 1, which pins
# cdf(d, ±Inf) to exactly 0 and 1 and keeps the CDF monotone across the last node.
# For θ < 1 the coth/csch coefficient forms cancel catastrophically (relative error
# ~eps/θ²); the _sinhm/_coshm forms are algebraically identical and cancellation-free.
function _node_cdf(d::DensityEstimate{T}) where {T}
    x, ψ = d.x, d.ψ
    n = length(x)
    F = Vector{T}(undef, n)
    F[1] = ψ[1]^2 / (2 * d.κL)          # left tail
    for k in 1:n-1
        κ = _kappa(d, k)
        θ = κ * (x[k+1] - x[k])
        if θ < 1
            s2 = 2 * sinh(θ)^2
            fdiag  = _sinhm(2θ) / (2 * s2)
            fcross = (θ * _coshm(θ) - _sinhm(θ)) / s2
        else
            ct, cs = coth(θ), csch(θ)
            fdiag  = (ct - θ * cs^2) / 2
            fcross = cs * (θ * ct - oneunit(T)) / 2
        end
        F[k+1] = F[k] + (fdiag * (ψ[k]^2 + ψ[k+1]^2) + 2 * fcross * ψ[k] * ψ[k+1]) / κ
    end
    return F, F[n] + ψ[n]^2 / (2 * d.κR)
end

# Unnormalized cumulative mass ∫_{-∞}^{x} ψ² dt, given the node cumulatives F. The tails
# are elementary exponential integrals; interior intervals use _cdf_mass_interior.
function _cdf_mass(d::DensityEstimate{T}, F::Vector{T}, x::Real) where {T}
    xs, ψ = d.x, d.ψ
    n = length(xs)
    isnan(x) && return T(NaN) * one(x)
    if x <= xs[1]
        κ = d.κL
        return ψ[1]^2 / (2κ) * exp(2κ * (x - xs[1]))
    elseif x >= xs[n]
        κ = d.κR
        return F[n] + ψ[n]^2 / (2κ) * (-expm1(-2κ * (x - xs[n])))
    end
    k = searchsortedlast(xs, x)         # xs[k] ≤ x < xs[k+1]
    return _cdf_mass_interior(d, F, k, x)
end

# Unnormalized cumulative mass at x within interval k (xs[k] ≤ x ≤ xs[k+1]). The partial
# mass is integrated from the nearer node — subtracting from F[k+1] when x lies in the
# right half — so its absolute error vanishes toward both nodes and the CDF stays
# continuous and monotone through every node.
function _cdf_mass_interior(d::DensityEstimate{T}, F::Vector{T}, k::Int, x::Real) where {T}
    xs, ψ = d.x, d.ψ
    κ = _kappa(d, k)
    a = κ * (xs[k+1] - x)               # a, b ≥ 0 and a + b = θ
    b = κ * (x - xs[k])
    θ = a + b
    if b <= θ / 2
        return F[k] + _segmass(ψ[k], ψ[k+1], a, b, θ) / κ
    else
        return F[k+1] - _segmass(ψ[k+1], ψ[k], b, a, θ) / κ
    end
end

# ∫₀ʷ ψ̂(u)² du for the unit-coordinate interval field ψ̂(u) = (p sinh(θ-u) + q sinh(u))/sinh(θ),
# with 0 ≤ w ≤ θ/2 and arem = θ - w; the physical mass over [x[k], x[k]+w/κ] is _segmass/κ.
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

"""
    cdf(d::DensityEstimate, x)

Cumulative distribution function of the fitted density: `F(x) = ∫_{-∞}^x Q(t) dt` with
`Q = ψ²`, evaluated in closed form (no quadrature). The tails are pure exponentials;
on each inter-node interval `ψ'' = κ²ψ` makes `ψ'² - κ²ψ²` constant, which yields the
exact antiderivative of the hyperbolic interpolant.

`F` is nondecreasing with `F(-Inf) == 0` and `F(Inf) == 1` exactly: the few ulps of
normalization roundoff in the fitted amplitudes are absorbed by a global rescaling.

`x` may be a scalar or an array. Each call assembles the per-node cumulative masses at
`O(length(d.x))` cost; the array method assembles them once and shares them across all
evaluations. See [`quantile`](@ref Statistics.quantile) for the inverse.

# Examples
```jldoctest
julia> d = DensityEstimate([0.0], 0.5);   # a Laplace density centered at 0

julia> cdf(d, 0.0)
0.5

julia> quantile(d, 0.5)
0.0
```
"""
function cdf(d::DensityEstimate, x::Real)
    F, total = _node_cdf(d)
    return _cdf_mass(d, F, x) / total
end
function cdf(d::DensityEstimate, x::AbstractArray)
    F, total = _node_cdf(d)
    return map(xi -> _cdf_mass(d, F, xi) / total, x)
end

"""
    quantile(d::DensityEstimate, q)

Quantile function of the fitted density, the inverse of [`cdf`](@ref):
`cdf(d, quantile(d, q)) ≈ q` for `q ∈ [0, 1]`, with `quantile(d, 0) == -Inf` and
`quantile(d, 1) == Inf`; `q` outside `[0, 1]` (including `NaN`) throws a
`DomainError`. In the exponential tails the inversion is in closed form; on interior
intervals a Newton iteration on the closed-form CDF, bracketed by the enclosing nodes,
converges to floating-point accuracy. The right tail is solved through `1 - q`, so
upper quantiles lose no more precision than `q` itself carries.

`q` may be a scalar or an array; the array method assembles the per-node cumulative
masses once and shares them across all evaluations.

Extends `Statistics.quantile`.
"""
function Statistics.quantile(d::DensityEstimate, q::Real)
    F, total = _node_cdf(d)
    return _quantile(d, F, total, q)
end
function Statistics.quantile(d::DensityEstimate, q::AbstractArray)
    F, total = _node_cdf(d)
    return map(qi -> _quantile(d, F, total, qi), q)
end

function _quantile(d::DensityEstimate{T}, F::Vector{T}, total::T, q::Real) where {T}
    0 <= q <= 1 || throw(DomainError(q, "quantile is defined only for probabilities 0 ≤ q ≤ 1"))
    xs, ψ = d.x, d.ψ
    n = length(xs)
    target = q * total
    if target <= F[1]                   # left tail: target = ψ₁²/(2κ) e^{2κ(x-x₁)}
        κ = d.κL
        return xs[1] + log(2κ * target / ψ[1]^2) / (2κ)
    elseif target >= F[n]               # right tail, through the complement 1 - q
        κ = d.κR
        return xs[n] - log(2κ * (total * (1 - q)) / ψ[n]^2) / (2κ)
    end
    k = searchsortedlast(F, target)     # F[k] ≤ target < F[k+1], so 1 ≤ k < n
    lo, hi = xs[k], xs[k+1]
    y = lo + (target - F[k]) / (F[k+1] - F[k]) * (hi - lo)  # linear-in-mass start
    for _ in 1:200
        r = _cdf_mass_interior(d, F, k, y) - target
        r == 0 && return y
        r < 0 ? (lo = y) : (hi = y)
        ynew = y - r / _amplitude(d, y)^2       # Newton: the CDF's derivative is ψ²
        lo < ynew < hi || (ynew = (lo + hi) / 2)  # bisect when Newton leaves the bracket
        ynew == y && return y
        y = ynew
    end
    error("quantile: safeguarded Newton failed to converge at q = $q — please report this")
end

"""
    action(d::DensityEstimate) -> S

Classical action `S[ψ_cl] = N - λ - Σᵢ wᵢ ln Q(xᵢ)` (Eq. 10) of the fitted density,
where `N = Σ wᵢ`. Used by [`select_kappa_ms`](@ref).
"""
function action(d::DensityEstimate)
    N = sum(d.w)
    return N - d.λ - sum(d.w .* log.(d.ψ.^2))
end

"""
    chisq(d::DensityEstimate, Q) -> χ²

Goodness-of-fit statistic between a trial density `Q` and the data underlying the
fit `d`, the robust field-theoretic analogue of Pearson's χ² (Eqs. 13–14 of the
paper):

    χ² = 4 Σᵢ wᵢ (√Q(xᵢ) / ψ_cl(xᵢ) - 1)²,

summed over the data nodes `xᵢ` with multiplicities `wᵢ`, where `ψ_cl = √(d(·))`
is the fitted amplitude. `Q` is any callable returning density values; it should be
a normalized density (`∫Q dx = 1`). `chisq(d, d) == 0`. Small χ² means `Q` is close
to the data in the (squared Hellinger) sense; see [`pvalue`](@ref) and
[`chisq_ccdf`](@ref) for significance.
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

"""
    expected_chisq(d::DensityEstimate) -> ⟨χ²⟩
    expected_chisq(ref::ChisqReference) -> ⟨χ²⟩

Mean of the reference distribution of [`chisq`](@ref), in the exact finite-`N` theory
(Holy 1997, Eqs. 16–18). Defined at any scale, constant or spatially varying.

Given a `DensityEstimate`, [`chisq_reference`](@ref) is assembled internally; to draw
several quantities from one fit, build the reference once and pass it here and to
[`chisq_ccdf`](@ref)/[`chisq_pdf`](@ref)/[`pvalue`](@ref) rather than reassembling it.
"""
expected_chisq(d::DensityEstimate) = chisq_reference(d).mean

# Standard normal CDF, Φ(t) = ½ erfc(-t/√2).
_Φ(t::T) where {T} = erfc(-t / sqrt(T(2))) / 2

# ── Exact reference distribution of χ² (Holy 1997, Eqs. 16–18) ───────────────
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
    ChisqReference

Precomputed reference distribution of the goodness-of-fit statistic [`chisq`](@ref)
for one fit, in the exact finite-`N` theory (Holy 1997, Eqs. 16–18). The statistic is
a quadratic form in the Gaussian fluctuation field, so its law is a generalized
chi-squared; this object stores the `O(N)` data — a symmetric tridiagonal matrix and a
rank-one constraint vector — that its density and tail probabilities are computed from.

Build one with [`chisq_reference`](@ref) and reuse it across many evaluations of
[`chisq_ccdf`](@ref), [`chisq_pdf`](@ref), and [`pvalue`](@ref); its exact mean is
[`expected_chisq`](@ref)`(ref)`.
"""
struct ChisqReference{T<:AbstractFloat}
    tri::SymTridiagonal{T,Vector{T}}    # D^{-1/2} C₀⁻¹ D^{-1/2}; A = tri⁻¹ - g gᵀ
    g::Vector{T}                        # rank-one constraint direction, D^{1/2} b / √Vφ
    tg::Vector{T}                       # tri·g, the Imhof rank-one RHS (constant in u)
    mean::T                             # exact ⟨χ²⟩ = tr(A)
end

Base.show(io::IO, r::ChisqReference) =
    print(io, "ChisqReference($(length(r.g)) nodes, ⟨χ²⟩=$(r.mean))")

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

# α = L₀⁻¹ψ_cl at the nodes, mᵢ = α(xᵢ). With u∓ the solutions of 𝒜u = 0 decaying at ∓∞ and
# C = v₋u₊ - u₋v₊ their flux Wronskian (constant, by Abel), Ĝ(x,y) = u₋(x∧y)u₊(x∨y)/C, so
#   α(x) = [u₊(x)∫_{-∞}^x u₋ψ_cl + u₋(x)∫_x^∞ u₊ψ_cl] / (2λC).
# Each tail fixes one solution: u₋ = e^{κL(x-x₁)} to the left of x₁ (normalized to 1 there,
# whence v₋ = u₋/κL), and its mirror to the right. Since u∓ grow like e^{±∫κ}, they are
# propagated — along with their accumulations — scaled by e^{∓∫κ}, which is what keeps the
# recursions bounded; the scale factors cancel identically in α, so it is assembled from the
# scaled quantities alone.
function _node_alpha(x::Vector{T}, ψ::Vector{T}, κ, κL::T, κR::T, λ::T) where {T}
    n = length(x)
    û₋ = similar(ψ); v̂₋ = similar(ψ); Â = similar(ψ)   # u₋, v₋, ∫_{-∞}^x u₋ψ_cl
    û₊ = similar(ψ); v̂₊ = similar(ψ); B̂ = similar(ψ)   # u₊, -v₊, ∫_x^∞ u₊ψ_cl
    û₋[1] = one(T); v̂₋[1] = inv(κL); Â[1] = ψ[1] / (2κL)
    for k in 1:n-1
        κk = _kappa(κ, k); h = x[k+1] - x[k]; θ = κk * h
        c₁, c₂, c₃, c₄ = _sweep_coeffs(κk, h)
        e = exp(-θ); ch = (1 + e * e) / 2; sh = -expm1(-2θ) / 2      # e^{-θ}cosh θ, e^{-θ}sinh θ
        Â[k+1] = e * Â[k] + û₋[k] * (ψ[k] * c₁ + ψ[k+1] * c₂) +
                            κk * v̂₋[k] * (ψ[k] * c₃ + ψ[k+1] * c₄)
        û₋[k+1] = û₋[k] * ch + κk * v̂₋[k] * sh
        v̂₋[k+1] = û₋[k] * sh / κk + v̂₋[k] * ch
    end
    û₊[n] = one(T); v̂₊[n] = inv(κR); B̂[n] = ψ[n] / (2κR)
    for k in n-1:-1:1
        κk = _kappa(κ, k); h = x[k+1] - x[k]; θ = κk * h
        c₁, c₂, c₃, c₄ = _sweep_coeffs(κk, h)
        e = exp(-θ); ch = (1 + e * e) / 2; sh = -expm1(-2θ) / 2
        B̂[k] = e * B̂[k+1] + û₊[k+1] * (ψ[k+1] * c₁ + ψ[k] * c₂) +
                            κk * v̂₊[k+1] * (ψ[k+1] * c₃ + ψ[k] * c₄)
        û₊[k] = û₊[k+1] * ch + κk * v̂₊[k+1] * sh
        v̂₊[k] = û₊[k+1] * sh / κk + v̂₊[k+1] * ch
    end
    Ĉ = û₊[1] / κL + v̂₊[1]              # the Wronskian, in the scaled variables
    return (û₊ .* Â .+ û₋ .* B̂) ./ (2λ * Ĉ)
end

# ∬ψ_cl G₀ ψ_cl = ∫ψ_cl α. On each interval α solves 𝒜α = ψ_cl/(2λ) at constant κ against a
# hyperbolic source, so it is the interpolant of its own nodal values mₖ plus the resonant
# particular solution s·cosh(κs) that the source forces; the tails are the same computation
# with ψ_cl ∝ e^{∓κ(x-x_edge)}, where α acquires the same resonant factor.
function _int_psi_alpha(x::Vector{T}, ψ::Vector{T}, m::Vector{T}, κ, κL::T, κR::T, λ::T) where {T}
    n = length(x)
    acc = ψ[1] * m[1] / (2κL) + ψ[1]^2 / (16λ * κL) +
          ψ[n] * m[n] / (2κR) + ψ[n]^2 / (16λ * κR)
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

"""
    chisq_reference(d::DensityEstimate) -> ChisqReference

Assemble the exact reference distribution of [`chisq`](@ref) for the fit `d`, following
Holy 1997 (Eqs. 16–18). Costs `O(N)`; reuse the result across many calls to
[`chisq_ccdf`](@ref)/[`chisq_pdf`](@ref)/[`pvalue`](@ref) rather than rebuilding it.

A spatially varying `κ` (see [`DensityEstimate`](@ref)) is supported: the nodal precision of
the fluctuation field is `2λ` times the same tridiagonal operator the fit assembles, whatever
the scale, so the law stays exact and `O(N)`.
"""
function chisq_reference(d::DensityEstimate{T}) where {T}
    x, ψ, w, λ = d.x, d.ψ, d.w, d.λ
    κ, κL, κR = d.κ, d.κL, d.κR
    n = length(x)
    m = _node_alpha(x, ψ, κ, κL, κR, λ)                # mₖ = ∫ψ_cl(x) G₀(xₖ,x) dx
    # C₀⁻¹ = G₀⁻¹ + S = 2λM̂ + diag(2wᵢ/ψᵢ²);  b = (I + G₀S)⁻¹m solves C₀⁻¹b = G₀⁻¹m. The
    # assembly carries the reference scale κ̄, which G₀⁻¹ = 2λM̂ does not admit: divide it out.
    M = _operator(x, κ, κL, κR)
    f = 2λ / _reference_scale(κ, κL, κR)
    S = 2 .* w ./ ψ.^2
    C0inv = SymTridiagonal(f .* M.dv .+ S, f .* M.ev)
    b = C0inv \ (f .* (M * m))
    Vφ = _int_psi_alpha(x, ψ, m, κ, κL, κR, λ) - sum(m .* S .* b)         # Var(∫ψ_cl δψ)
    # Reduced tridiagonal tri = D^{-1/2} C₀⁻¹ D^{-1/2} and rank-one direction g.
    D = 2 .* S; sq = sqrt.(D)                          # D = 4wᵢ/ψᵢ²
    tri = SymTridiagonal(C0inv.dv ./ D, C0inv.ev ./ (sq[1:n-1] .* sq[2:n]))
    g = sq .* b ./ sqrt(Vφ)
    return ChisqReference{T}(tri, g, tri * g, _tridiag_invdiag(tri) - sum(abs2, g))
end

expected_chisq(r::ChisqReference) = r.mean

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

"""
    chisq_ccdf(d::DensityEstimate, z; method=:exact)   -> P(χ² ≥ z)
    chisq_ccdf(ref::ChisqReference, z; method=:exact)  -> P(χ² ≥ z)

Upper-tail (survival) probability of the reference χ² distribution at `z`. Evaluated at an
observed statistic it is a p-value; see [`pvalue`](@ref).

`method=:exact` (default) uses the finite-`N` generalized-χ² law via Imhof inversion of
[`chisq_reference`](@ref)`(d)`. `method=:largeN` uses the inverse-Gaussian (Wald) shape of
the large-`N` limit (Eq. 26), parameterized by the exact mean [`expected_chisq`](@ref); it
is a closed form, far cheaper per call, and — like the exact law — defined at every scale.
Pass a prebuilt [`ChisqReference`](@ref) to avoid reassembling it across calls.
"""
function chisq_ccdf(r::ChisqReference{T}, z::Real; method::Symbol=:exact, rtol=sqrt(eps(T))) where {T}
    method === :largeN && return _wald_ccdf(r.mean, z)
    method === :exact || throw(ArgumentError("method must be :exact or :largeN, got :$method"))
    zT = T(z)
    piv, rhs = _logΦ_scratch(r)
    f(u) = u == 0 ? (r.mean - zT) / 2 :
        (θ = _logΦ!(piv, rhs, r, u); sin((θ[1] - zT * u) / 2) / (u * sqrt(θ[2])))
    I, _ = quadgk(f, zero(T), T(Inf); rtol)      # I ∈ [-π/2, π/2]; no tiny-value churn
    return clamp(one(T)/2 + I / T(π), zero(T), one(T))
end
chisq_ccdf(d::DensityEstimate, z::Real; method::Symbol=:exact) =
    chisq_ccdf(chisq_reference(d), z; method)

# Inverse-Gaussian (Wald) density, companion to `_wald_ccdf`.
_wald_pdf(μ::T, z::Real) where {T} =
    z > 0 ? μ / sqrt(2 * T(π) * T(z)^3) * exp(μ - T(z) / 2 - μ^2 / (2 * T(z))) : zero(T)

"""
    chisq_pdf(d::DensityEstimate, z; method=:exact)   -> P(z)
    chisq_pdf(ref::ChisqReference, z; method=:exact)  -> P(z)

Density of the reference χ² distribution at `z ≥ 0`. `method=:exact` (default) is the
finite-`N` generalized-χ² law from [`chisq_reference`](@ref)`(d)`. `method=:largeN` is the
inverse-Gaussian (Wald) density of the large-`N` limit (Eq. 26),

    P(z) = ⟨χ²⟩ / √(2π z³) · exp[⟨χ²⟩ - z/2 - ⟨χ²⟩²/(2z)],

with `⟨χ²⟩ =` [`expected_chisq`](@ref) the exact mean: a closed form, defined at every scale.
Pass a prebuilt [`ChisqReference`](@ref) to reuse it.
"""
function chisq_pdf(r::ChisqReference{T}, z::Real; method::Symbol=:exact, rtol=sqrt(eps(T)), atol=sqrt(eps(T))) where {T}
    method === :largeN && return _wald_pdf(r.mean, z)
    method === :exact || throw(ArgumentError("method must be :exact or :largeN, got :$method"))
    # atol floors the density: deep in the tail the true value underflows to ~0, and a purely
    # relative tolerance would otherwise subdivide the oscillatory integrand without end.
    zT = T(z)
    piv, rhs = _logΦ_scratch(r)
    f(u) = (θ = _logΦ!(piv, rhs, r, u); cos((θ[1] - zT * u) / 2) / sqrt(θ[2]))
    I, _ = quadgk(f, zero(T), T(Inf); rtol, atol, maxevals=10^4)
    return max(I / (2 * T(π)), zero(T))
end
chisq_pdf(d::DensityEstimate, z::Real; method::Symbol=:exact) =
    chisq_pdf(chisq_reference(d), z; method)

"""
    pvalue(d::DensityEstimate, Q; method=:exact)    -> p
    pvalue(ref::ChisqReference, χ²; method=:exact)  -> p

Significance of the fit of a trial density `Q`: the probability that the reference χ²
distribution exceeds the observed [`chisq`](@ref)`(d, Q)`, i.e. `chisq_ccdf(d, chisq(d, Q))`.

`method` is as in [`chisq_ccdf`](@ref). To test several trial densities against one fit,
build the reference once with [`chisq_reference`](@ref) and call `pvalue(ref, chisq(d, Q))`.
"""
pvalue(r::ChisqReference, χ²::Real; method::Symbol=:exact) = chisq_ccdf(r, χ²; method)
pvalue(d::DensityEstimate, Q; method::Symbol=:exact) =
    pvalue(chisq_reference(d), chisq(d, Q); method)

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

# Geometric κ grid from coarse (≈ one blob over the data) to fine (≈ individual points),
# scaled to the data's extent, wide enough to bracket the minimum-sensitivity scale.
function _default_κs(x::AbstractVector{<:Real})
    lo, hi = extrema(x)
    span = hi - lo
    span > 0 || throw(ArgumentError("need at least two distinct points to select κ"))
    return exp.(range(log(0.5 / span), log(5 * length(x) / span); length = 40))
end

"""
    select_kappa_ms(x; κs=<data-scaled grid>, rtol=1e-6) -> κ

Choose the smoothing scale by the principle of minimum sensitivity: return the
`κ` at which the classical action [`action`](@ref) `S` is least sensitive to the
scale, i.e. `|dS/d ln κ|` is smallest (Fig. 1 of the paper). The derivative
`dS/d ln κ` is evaluated analytically and minimized over `κ` by a golden-section
search, bracketed by the grid `κs` (which defaults to a geometric range scaled
to the data's extent).

This is a principled convention rather than a unique optimum: `S` has no exact
stationary point in `κ`, so the flattest point depends on measuring sensitivity
in `ln κ`. It generally selects a different scale than the entropy-based
[`kappa_interval`](@ref); both resolve *information* and over-resolve smooth
densities. To target estimation error instead, prefer [`select_kappa_kl`](@ref)
(the recommended default) or [`select_kappa_cv`](@ref). The information-resolving
scales here and in `kappa_interval` are the better choice only for heavily tied or
discrete data, where the cross-validation scores are unbounded.

`κs` must be sorted and positive, with at least three values to bracket the
minimum.
"""
function select_kappa_ms(x::AbstractVector{<:Real}; κs::AbstractVector{<:Real}=_default_κs(x), rtol::Real=1e-6)
    issorted(κs) && all(>(0), κs) || throw(ArgumentError("κs must be sorted and positive"))
    length(κs) >= 3 || throw(ArgumentError("need at least 3 values in κs to bracket the minimum"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    T = float(promote_type(eltype(x), eltype(κs), typeof(rtol)))
    xs = _sorted_sample(x, T)
    r = T(rtol)
    absslope(κ) = abs(last(_action_and_slope(_merge_presorted(xs, r / κ)..., κ)))
    lnκ = log.(T.(κs))
    i = argmin(absslope.(exp.(lnκ)))            # coarse bracket on the grid
    lo = lnκ[max(i - 1, firstindex(lnκ))]
    hi = lnκ[min(i + 1, lastindex(lnκ))]
    return exp(_golden_min(l -> absslope(exp(l)), lo, hi))
end

"""
    kappa_interval(x; level=0.2, rtol=1e-6) -> (; κ, lo, hi)

Principled smoothing-scale selection with an interval of plausible values.

As `κ` sweeps from `0` to `∞` the classical action's reduced form `g(κ) = S(κ) + W ln κ`
(with `W = Σ wᵢ` the total count) rises monotonically between two exact limits:
`g → W/2` as `κ → 0` (all points merge into one lump) and `g → W/2 + W H` as `κ → ∞`
(the `N` points become isolated), where `H = -Σᵢ (wᵢ/W) ln(wᵢ/W)` is the Shannon entropy
of the multiplicities (`ln N` for distinct points). The normalized quantity

    h(κ) = (g(κ) - W/2) / (W H) ∈ [0, 1]

is therefore the fraction of the data's entropy that scale `κ` resolves, and its half-point
`h = 1/2` is returned as `κ`. This entropy criterion is distinct from the minimum-sensitivity
scale of [`select_kappa_ms`](@ref); one advantage of this function is that it doesn't require
computing a noisy numerical derivative.

Returns the half-entropy scale `κ` (`h = 1/2`) together with the interval `[lo, hi]`
bracketing `h ∈ [(1-level)/2, (1+level)/2]`; the default `level=0.2` spans `h ∈ [0.4, 0.6]`.
Requires at least two distinct points.
"""
function kappa_interval(x::AbstractVector{<:Real}; level::Real=0.2, rtol::Real=1e-6)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    T = float(promote_type(eltype(x), typeof(level), typeof(rtol)))
    xs = _sorted_sample(x, T)
    nodes0, w0 = _merge_presorted(xs, zero(T))      # exact duplicates fix the entropy baseline
    length(nodes0) >= 2 || throw(ArgumentError("need at least two distinct points to select κ"))
    W = sum(w0)
    Hent = -sum(wi / W * log(wi / W) for wi in w0)  # entropy of the multiplicities
    r = T(rtol)
    # h(κ): fraction of the entropy resolved, monotone from 0 (κ→0) to 1 (κ→∞). Points closer
    # than rtol/κ are merged before each fit; as κ→∞ this reduces to the distinct nodes, so h
    # still approaches 1 against the same entropy baseline W, H.
    function h(κ)
        nodes, w = _merge_presorted(xs, r / κ)
        return (action(_fit(nodes, w, κ)) + W * log(κ) - W / 2) / (W * Hent)
    end
    lvl = T(level)
    lo = _invert_monotone(h, (1 - lvl) / 2)
    κ = _invert_monotone(h, oneunit(T) / 2)
    hi = _invert_monotone(h, (1 + lvl) / 2)
    return (; κ, lo, hi)
end

# Solve h(κ) = target for a function h that increases monotonically in κ, by bracketing
# in ln κ and bisecting. Used by kappa_interval.
function _invert_monotone(h, target::T) where {T}
    # At very large κ the points become numerically isolated and h(κ) can overflow to a
    # non-finite value; since h → 1 there, treat non-finite as "above target".
    above(κ) = (v = h(κ); !isfinite(v) || v >= target)
    lnlo, lnhi = log(T(1e-6)), log(T(1e6))
    for _ in 1:40                     # expand outward until the root is bracketed
        above(exp(lnlo)) || break
        lnlo -= log(T(10))
    end
    for _ in 1:40
        above(exp(lnhi)) && break
        lnhi += log(T(10))
    end
    for _ in 1:60                     # bisection
        m = (lnlo + lnhi) / 2
        above(exp(m)) ? (lnhi = m) : (lnlo = m)
    end
    return exp((lnlo + lnhi) / 2)
end

# Diagonal of the inverse of an SPD symmetric tridiagonal, in O(n), from the top-down and
# bottom-up LDLᵀ pivots dᵢ, δᵢ: (H⁻¹)ᵢᵢ = 1/(dᵢ + δᵢ - aᵢ), with aᵢ the original diagonal.
function _inv_diag(H::SymTridiagonal{T}) where {T}
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
    return inv.(d .+ δ .- a)
end

# Normalised amplitude ψ and the leave-one-out densities Q̂₋ᵢ(xᵢ) at every node, in O(N). The
# leave-one-out density is analytic to first order — dropping one observation at node i decrements
# wᵢ, perturbing the unnormalised field φ by δφ = -H⁻¹eᵢ/φᵢ (H the fit's SPD Hessian
# ∇²F = M + diag(w/φ²)). Carrying δφ through the normalization ψ = φ/√Z, with Z = ∫φ² = φᵀGφ
# and v = H⁻¹Gφ (Gφ = ½ ∂Z/∂φ), gives Q̂₋ᵢ(xᵢ) ≈ ψᵢ² (1 - 2(H⁻¹)ᵢᵢ/φᵢ² + 2vᵢ/(φᵢ Z)).
#
# Nothing in that expansion uses M's entries, only that it is the fixed SPD operator whose mass
# functional is Z — so it holds for a piecewise-constant scale unchanged. The overall factor the
# adaptive operator carries (see `roughness_operator`) leaves ψ and the leave-one-out densities
# invariant: under M → cM the pieces move as φ → φ/√c, Z → Z/c, H → cH, (H⁻¹)ᵢᵢ → (H⁻¹)ᵢᵢ/c,
# Gφ → Gφ/√c and v → v/c^{3/2}, and every term above is a ratio in which c cancels.
function _loo_density(nodes::Vector{T}, w::Vector{T}, κ, κL::T, κR::T) where {T}
    M = _operator(nodes, κ, κL, κR)
    φ = _solve_amplitude(M, w)
    Z, Gφ = _norm_sq_gram(nodes, φ, κ, κL, κR)
    H = SymTridiagonal(M.dv .+ w ./ φ.^2, M.ev)
    gii = _inv_diag(H)
    v = ldiv!(ldlt!(H), Gφ)             # H⁻¹Gφ; H is consumed, gii already extracted
    ψ = φ ./ sqrt(Z)
    looi = @. ψ^2 * (1 - 2 * gii / φ^2 + 2 * v / (φ * Z))
    return ψ, looi
end

# Least-squares cross-validation score LSCV(κ) = ∫Q̂² - (2/N) Σᵢ wᵢ Q̂₋ᵢ(xᵢ): an unbiased
# estimate, up to the κ-independent ∫Q², of the integrated squared error ∫(Q̂-Q)².
function _lscv(nodes::Vector{T}, w::Vector{T}, κ, κL::T, κR::T) where {T}
    ψ, looi = _loo_density(nodes, w, κ, κL, κR)
    N = sum(w)
    cross = zero(T)
    for i in eachindex(w, looi)
        cross += w[i] * looi[i]
    end
    return _int_quartic(nodes, ψ, κ, κL, κR) - 2 * cross / N
end
_lscv(nodes::Vector{T}, w::Vector{T}, κ::T) where {T} = _lscv(nodes, w, κ, κ, κ)

# Kullback–Leibler cross-validation score, the mean negative leave-one-out log-likelihood
# -(1/N) Σᵢ wᵢ ln Q̂₋ᵢ(xᵢ): an estimate, up to a κ-independent constant, of KL(Q ‖ Q̂_κ). Reuses
# the same first-order leave-one-out densities as _lscv. A non-positive Q̂₋ᵢ (possible where the
# first-order expansion overshoots) makes the log undefined; return NaN so the search rejects κ.
function _klcv(nodes::Vector{T}, w::Vector{T}, κ, κL::T, κR::T) where {T}
    _, looi = _loo_density(nodes, w, κ, κL, κR)
    s = zero(T)
    for i in eachindex(w, looi)
        looi[i] > 0 || return T(NaN)
        s += w[i] * log(looi[i])
    end
    return -s / sum(w)
end
_klcv(nodes::Vector{T}, w::Vector{T}, κ::T) where {T} = _klcv(nodes, w, κ, κ, κ)

"""
    select_kappa_cv(x; κs=<data-scaled grid>, rtol=1e-6) -> κ

Choose the smoothing scale by least-squares cross-validation: return the `κ` minimizing

    LSCV(κ) = ∫ Q̂_κ(x)² dx - (2/N) Σᵢ Q̂_{κ,-i}(xᵢ),

an unbiased estimate — up to the `κ`-independent `∫Q²` — of the integrated squared error
`∫(Q̂_κ - Q)²`, where `Q̂_{κ,-i}` is the density fitted with the `i`-th point left out. Its
minimizer therefore targets minimum mean integrated squared error (MISE). This generally
selects a finer scale than [`select_kappa_ms`](@ref) (minimum sensitivity) and
[`kappa_interval`](@ref) (half-entropy), which resolve information rather than squared error
and tend to over-resolve smooth densities.

Both terms are evaluated analytically in `O(N)`: `∫Q̂²` in closed form over the exponential
segments, and each leave-one-out density `Q̂_{-i}(xᵢ)` from a first-order expansion of the fit
in the dropped point's weight, so no per-point refitting is needed. The score is minimized by a
golden-section search over `ln κ`, bracketed by the grid `κs` (a geometric range scaled to the
data's extent by default).

Cross-validation assumes the data are draws from a continuous density. Heavily tied or coarsely
rounded data instead resemble a discrete distribution, for which `LSCV` decreases without bound
as `κ → ∞` (finer scales keep resolving the atoms); `select_kappa_cv` then returns a large `κ`.
Prefer [`select_kappa_ms`](@ref) or [`kappa_interval`](@ref), which stay bounded, in that regime.

`κs` must be sorted and positive, with at least three values to bracket the minimum.
"""
select_kappa_cv(x::AbstractVector{<:Real}; κs::AbstractVector{<:Real}=_default_κs(x), rtol::Real=1e-6) =
    _select_by_score(_lscv, x, κs, rtol)

"""
    select_kappa_kl(x; κs=<data-scaled grid>, rtol=1e-6) -> κ

Choose the smoothing scale by Kullback–Leibler (likelihood) cross-validation: return the `κ`
minimizing the mean negative leave-one-out log-likelihood

    KLCV(κ) = -(1/N) Σᵢ wᵢ ln Q̂_{κ,-i}(xᵢ),

where `Q̂_{κ,-i}` is the density fitted with the `i`-th point left out. This estimates, up to a
`κ`-independent constant, the Kullback–Leibler divergence `KL(Q ‖ Q̂_κ)`; minimizing it is
maximum-likelihood cross-validation. It is the criterion native to the estimator, whose action
`-Σ ln Q̂(xᵢ)` is itself the (in-sample) log-likelihood, and to leading order it selects the same
error-optimal scale as [`select_kappa_cv`](@ref) while being cheaper: the `∫Q̂²` roughness term is
not needed. Like `select_kappa_cv` it generally selects a finer scale than [`select_kappa_ms`](@ref)
and [`kappa_interval`](@ref), which resolve information rather than divergence.

This is the **recommended default** selector: on a range of test densities it tracks the
error-optimal scale most closely of the four (see `benchmarks/`), and it is the cheapest of the
cross-validation scores to evaluate.

Each leave-one-out density `Q̂_{-i}(xᵢ)` comes from a first-order expansion of the fit in the
dropped point's weight, so no per-point refitting is needed and the score costs `O(N)`. The score
is minimized by a golden-section search over `ln κ`, bracketed by the grid `κs` (a geometric
range scaled to the data's extent by default).

Cross-validation assumes the data are draws from a continuous density. Heavily tied or coarsely
rounded data instead resemble a discrete distribution, for which the leave-one-out log-likelihood
increases without bound as `κ → ∞` (leaving out one of many coincident copies barely perturbs the
fit); `select_kappa_kl` then returns a large `κ`. Prefer [`select_kappa_ms`](@ref) or
[`kappa_interval`](@ref), which stay bounded, in that regime.

`κs` must be sorted and positive, with at least three values to bracket the minimum.
"""
select_kappa_kl(x::AbstractVector{<:Real}; κs::AbstractVector{<:Real}=_default_κs(x), rtol::Real=1e-6) =
    _select_by_score(_klcv, x, κs, rtol)

# Minimize a per-κ score over ln κ, bracketed by the grid κs. `scorefun(nodes, w, κ)` returns the
# score for the merged nodes/weights at scale κ. A near-coincident pair left unmerged at very large
# κ can drive the fit to a non-finite score; those are treated as +∞ so the search never selects a
# degenerate scale.
function _select_by_score(scorefun, x::AbstractVector{<:Real}, κs::AbstractVector{<:Real}, rtol::Real)
    issorted(κs) && all(>(0), κs) || throw(ArgumentError("κs must be sorted and positive"))
    length(κs) >= 3 || throw(ArgumentError("need at least 3 values in κs to bracket the minimum"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    T = float(promote_type(eltype(x), eltype(κs), typeof(rtol)))
    xs = _sorted_sample(x, T)
    r = T(rtol)
    score(κ) = (v = scorefun(_merge_presorted(xs, r / κ)..., κ); isfinite(v) ? v : typemax(T))
    lnκ = log.(T.(κs))
    i = argmin(score.(exp.(lnκ)))               # coarse bracket on the grid
    lo = lnκ[max(i - 1, firstindex(lnκ))]
    hi = lnκ[min(i + 1, lastindex(lnκ))]
    return exp(_golden_min(l -> score(exp(l)), lo, hi))
end

"""
    AdaptiveScale(c, α, pilot)

A spatially varying smoothing scale of the plug-in form

    κ(x) = c · (p̂(x) / ḡ)^α,

where `p̂` is the `pilot` density estimate and `ḡ` the geometric mean of `p̂` over the
sample it was fitted to. [`select_kappa_adaptive`](@ref) constructs one by choosing `c`
and `α`; the result is callable, and is passed straight to [`DensityEstimate`](@ref) as
the smoothing scale.

Dividing by `ḡ` puts `c` on the same footing as a constant scale: where `p̂` equals its
geometric mean, `κ(x) = c`. The exponent `α > 0` sets how strongly the scale follows the
density — larger `α` resolves the peaks more finely and smooths the tails more heavily.

The scale is floored at `1e-6 c`, which intercepts underflow of `(p̂/ḡ)^α` at points where
the pilot density is negligible; the floor sits far below any scale the rule would
otherwise choose, so it never shapes the fit.
"""
struct AdaptiveScale{T<:AbstractFloat,D}
    c::T           # scale where the pilot density equals its geometric mean
    α::T           # exponent coupling the scale to the pilot density
    pilot::D       # the pilot density estimate p̂
    loggbar::T     # ln ḡ, the mean of ln p̂ over the pilot's sample
    κmin::T        # underflow floor

    function AdaptiveScale{T,D}(c, α, pilot, loggbar, κmin) where {T<:AbstractFloat,D}
        return new{T,D}(c, α, pilot, loggbar, κmin)
    end
end

AdaptiveScale{T}(c, α, pilot::D, loggbar, κmin) where {T,D} =
    AdaptiveScale{T,D}(c, α, pilot, loggbar, κmin)

# The pilot density underflows to zero between two far-separated tail nodes, sending
# (p̂/ḡ)^α there to zero; the floor keeps the assembled operator's coth(θ)/κ entries finite.
const _KAPPA_FLOOR = 1e-6

function AdaptiveScale(c::Real, α::Real, pilot::DensityEstimate{T}) where {T}
    α > 0 || throw(ArgumentError("the exponent α must be positive, got $α"))
    c > 0 || throw(ArgumentError("the scale c must be positive, got $c"))
    return AdaptiveScale{T}(T(c), T(α), pilot, _log_geomean(pilot), T(_KAPPA_FLOOR) * T(c))
end

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

# ln p̂ rather than p̂: the pilot density underflows to zero between far-separated tail nodes,
# where its logarithm is still perfectly finite.
(k::AdaptiveScale)(x::Real) = _scale_from_logdensity(k, 2 * log(_amplitude(k.pilot, x)))

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
# here, so it reports NaN rather than aborting the whole selection.
function _score_kappa(scorefun, xs::Vector{T}, κfun, rtol::T) where {T}
    nodes, w, κs, κL, κR = _merge_and_realize(xs, κfun, rtol)
    length(nodes) >= 2 || return T(NaN)
    try
        return scorefun(nodes, w, κs, κL, κR)
    catch e
        e isa ZeroPivotException && return T(NaN)
        rethrow()
    end
end

const _CSPAN = 20.0    # the c bracket runs ×/÷ this factor about its center
const _CGRID = 13      # grid points across it, to bracket the minimum
const _CITERS = 20     # golden-section refinements: pins ln c to ~1e-4 of the bracket, far
                       # below the scale on which the score itself varies
const _CSHIFTS = 6     # times the bracket may recenter on an edge minimum before giving up

# Minimize `score(c)` over ln c, bracketed by a geometric grid about `c0` and recentered on
# the best edge until the minimum falls strictly inside. Unresolvable candidates score
# non-finite; treating those as +∞ lets the search step over them.
function _select_c(score, c0::T) where {T}
    f(l) = (v = score(exp(l)); isfinite(v) ? v : typemax(T))
    lnspan = log(T(_CSPAN))
    for _ in 0:_CSHIFTS
        lncs = range(log(c0) - lnspan, log(c0) + lnspan; length=_CGRID)
        scores = f.(lncs)
        all(==(typemax(T)), scores) &&
            error("no resolvable smoothing scale anywhere in the search bracket around c = $c0")
        i = argmin(scores)
        if i != firstindex(lncs) && i != lastindex(lncs)
            return exp(_golden_min(f, lncs[i-1], lncs[i+1]; iters=_CITERS))
        end
        c0 = exp(lncs[i])                       # recenter on the winning edge and search on
    end
    error("the smoothing scale kept running off its search bracket after $_CSHIFTS expansions")
end

"""
    select_kappa_adaptive(x; alphas=(0.25, 0.5, 0.75, 1.0), pilot=select_kappa_kl, rtol=cbrt(eps(T)))
        -> κ

Choose a *spatially varying* smoothing scale by Kullback–Leibler cross-validation, and
return it ready to pass to [`DensityEstimate`](@ref).

A single scale must trade resolution in the bulk against noise in the tails. Letting `κ`
follow the density lifts that trade-off, and buys the most where a constant scale is limited
not by noise but by the density's own irregularity: a divergent or discontinuous edge, a
kink, or heavy tails. On smooth densities there is nothing to buy, and this selector says so
— it returns a plain number, the constant scale, whenever adaptivity does not earn its
keep by the same cross-validation score that chose it.

The rule is a plug-in: fit a pilot density `p̂` at the constant scale `pilot(x)` (by default
[`select_kappa_kl`](@ref)), then consider the family

    κ(x; c, α) = c · (p̂(x) / ḡ)^α,     ḡ = geometric mean of p̂ over the sample

(an [`AdaptiveScale`](@ref)). For each exponent `α` in `alphas`, `c` is chosen by
golden-section search on the leave-one-out score `KLCV(κ) = -(1/N) Σᵢ wᵢ ln Q̂₋ᵢ(xᵢ)`
generalized to a varying scale — the same criterion [`select_kappa_kl`](@ref) minimizes,
and, like it, evaluated in closed form and `O(N)`, with no refitting. The constant scale
competes as the `α = 0` member of the same family and on the same score, so the returned
scale is adaptive only if adaptivity wins.

Returns an [`AdaptiveScale`](@ref) when some `α` in `alphas` beats the constant scale, and
the constant scale itself (a number, so the fit takes the constant-`κ` path and its
goodness-of-fit machinery stays available) otherwise. Selection costs a small multiple of
one [`select_kappa_kl`](@ref) call; shorten `alphas` to trade capture for speed.

The `alphas` must be positive: `α = 0` is the constant scale, which is always in the
comparison. They are searched in increasing order, whatever order they are given in. `rtol`
is the node-merging tolerance, as a fraction of the local smoothing length, matching
[`DensityEstimate`](@ref)'s.

# Examples
```jldoctest
julia> x = -log.(1 .- (0.5:999.5) ./ 1000);   # exponential: a jump at the left edge

julia> κ = select_kappa_adaptive(x);          # adaptivity wins here

julia> κ.α                                    # the selected exponent
0.5

julia> d = DensityEstimate(x, κ);

julia> extrema(d.κ)[2] / extrema(d.κ)[1] > 100   # far finer at the edge than in the tail
true

julia> select_kappa_adaptive(range(0, 1; length=1000)) isa Real   # uniform: nothing to buy
true
```
"""
function select_kappa_adaptive(x::AbstractVector{<:Real};
                               alphas=(0.25, 0.5, 0.75, 1.0),
                               pilot=select_kappa_kl,
                               rtol::Real=cbrt(eps(float(eltype(x)))))
    isempty(alphas) && throw(ArgumentError("need at least one exponent in alphas"))
    all(>(0), alphas) ||
        throw(ArgumentError("the exponents in alphas must be positive; the constant scale " *
                            "(α = 0) is always compared against them"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    T = float(promote_type(eltype(x), eltype(alphas), typeof(rtol)))
    xs = _sorted_sample(x, T)
    r = T(rtol)

    κ0 = T(pilot(xs))
    κ0 > 0 || throw(ArgumentError("the pilot must return a positive scale, got $κ0"))
    p = DensityEstimate(xs, κ0; rtol=r)
    loggbar = _log_geomean(p)

    # The constant scale, scored on the same footing as the adaptive candidates.
    best_c, best_α = κ0, zero(T)
    best_score = _klcv(_merge_presorted(xs, r / κ0)..., κ0)
    isfinite(best_score) || (best_score = typemax(T))

    # The exponents are searched in increasing order, each bracket centered on the previous
    # exponent's optimum. The optimal c climbs steeply with α — a scale falling off as p̂^α
    # needs a larger c to keep the same resolution where the data actually are — and by α = 1
    # it can sit well outside a bracket centered on the pilot scale. Walking α upward keeps
    # every optimum comfortably inside its bracket.
    c0 = κ0
    for α in sort!(collect(T, alphas))
        scale(c) = AdaptiveScale{T}(c, α, p, loggbar, T(_KAPPA_FLOOR) * c)
        c0 = _select_c(c -> _score_kappa(_klcv, xs, scale(c), r), c0)
        s = _score_kappa(_klcv, xs, scale(c0), r)
        if isfinite(s) && s < best_score
            best_score, best_c, best_α = s, c0, α
        end
    end
    return best_α == 0 ? best_c : AdaptiveScale(best_c, best_α, p)
end

end # module
