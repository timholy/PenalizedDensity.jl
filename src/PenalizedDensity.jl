module PenalizedDensity

using LinearAlgebra
using SpecialFunctions: erfc, erfcx

export PenalizedDensityEstimate, amplitude, action, select_kappa, kappa_interval
export chisq, expected_chisq, chisq_pdf, chisq_ccdf, pvalue

"""
    PenalizedDensityEstimate(x; κ, rtol=1e-6)

Estimate a continuous one-dimensional probability density from sample points `x`,
using the scalar-field method of Holy, *Phys. Rev. Lett.* **79**, 3545 (1997).

The density is written as `Q(x) = ψ(x)^2`, where the amplitude `ψ` minimises the
action

    S[ψ] = ∫ (ℓ²/2) (ψ')² dx − 2 Σᵢ ln ψ(xᵢ)

subject to `∫ ψ² dx = 1`. The smoothing scale `κ = √(2λ)/ℓ` (with `λ` the
normalisation multiplier) sets the width of each point's contribution; larger `κ`
gives a rougher estimate. See [`select_kappa`](@ref) for choosing it automatically.

Between sorted data points `ψ` solves `ψ'' = κ² ψ`, i.e. it is a sum of rising and
falling exponentials, and decays as `e^{-κ|x|}` in the tails. The nodal amplitudes
`ψ(xᵢ)` satisfy a symmetric tridiagonal system whose solution is the minimiser of a
strictly convex potential; normalisation is then a rescaling.

The returned object is callable: `d(x)` evaluates the density `Q(x)` at any real
`x` (scalar or array). Use [`amplitude`](@ref) for `ψ(x)`.

Repeated points, and points closer than `rtol / κ` (i.e. within a fraction `rtol` of the
smoothing length `1 / κ`), are merged into one node carrying the count as its integer
weight, so weighted data is handled naturally. Points that close carry no independent
information at resolution `1 / κ`, and a near-coincident pair drives an entry of the
tridiagonal system toward a singularity; merging bounds both the node count and the
conditioning. The default `rtol = 1e-6` merges only points far below the resolution, which
keeps the solve well conditioned without visibly changing the estimate; increase it (e.g.
`rtol = 1e-3`) to cap the node count and speed up fits on large, densely packed samples.

# Examples
```jldoctest
julia> d = PenalizedDensityEstimate([-1.0, 0.0, 0.0, 1.0]; κ=1.0);

julia> d(0.0) > d(2.0)   # denser near the data
true

julia> round(chisq(d, d); digits=8)   # a distribution fits itself perfectly
0.0
```
"""
struct PenalizedDensityEstimate{T<:AbstractFloat}
    x::Vector{T}   # sorted, distinct node locations
    w::Vector{T}   # weight (multiplicity) at each node
    ψ::Vector{T}   # normalised amplitude at the nodes
    κ::T           # smoothing scale
    λ::T           # normalisation multiplier (diagnostic)
end

function PenalizedDensityEstimate(x::AbstractVector{<:Real}; κ::Real, rtol::Real=1e-6)
    κ > 0 || throw(ArgumentError("κ must be positive, got $κ"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    isempty(x) && throw(ArgumentError("cannot fit a density to zero points"))
    T = float(promote_type(eltype(x), typeof(κ), typeof(rtol)))
    nodes, weights = _merge_sorted(x, T(rtol) / T(κ), T)
    return _fit(nodes, weights, T(κ))
end

# Fit from already-merged distinct nodes and their weights.
function _fit(nodes::Vector{T}, weights::Vector{T}, κ::T) where {T}
    ψ = _solve_amplitude(nodes, weights, κ)
    Z = _norm_sq(nodes, ψ, κ)
    ψ ./= sqrt(Z)
    λ = κ * Z                       # scaling law: normalised ψ solves (−M)ψ = (κ/λ)/ψ
    return PenalizedDensityEstimate{T}(nodes, weights, ψ, κ, λ)
end

"""
    _merge_sorted(x, atol, T) -> (nodes, weights)

Sort `x` and collapse runs of points within `atol` of the run's first member into a
single node carrying the count as its weight. Returns distinct, strictly increasing
`nodes::Vector{T}` and matching `weights`, regardless of the axes of `x`.
"""
_merge_sorted(x::AbstractVector, atol::T, ::Type{T}) where {T} =
    _merge_presorted(sort!(T[xi for xi in x]), atol)

# Collapse runs of an already-sorted sequence `xs` within `atol` of the run's first member.
# Factored out so kappa_interval can re-merge one sorted copy at many tolerances.
function _merge_presorted(xs, atol::T) where {T}
    nodes = T[]
    weights = T[]
    for xi in xs
        xk = T(xi)
        if !isempty(nodes) && xk - nodes[end] <= atol
            weights[end] += one(T)
        else
            push!(nodes, xk)
            push!(weights, one(T))
        end
    end
    return nodes, weights
end

# Tridiagonal operator −M (SPD) coupling the nodal amplitudes.
# Off-diagonal e[k] = −csch(κ hₖ); diagonal d[i] accumulates coth(κ hₖ) from each
# adjacent interval and +1 from each adjacent tail.
function _neg_M(x::Vector{T}, κ::T) where {T}
    n = length(x)
    d = zeros(T, n)
    e = zeros(T, n - 1)
    d[1] += one(T)                  # left tail
    d[n] += one(T)                  # right tail
    for k in 1:n-1
        θ = κ * (x[k+1] - x[k])
        d[k]   += coth(θ)
        d[k+1] += coth(θ)
        e[k]    = -csch(θ)          # coth/csch stay finite as θ → ∞ (isolated points)
    end
    return SymTridiagonal(d, e)
end

# F(ψ) = ½ ψ'(−M)ψ − Σ wᵢ ln ψᵢ, the potential minimised by _solve_amplitude.
function _objective(negM::SymTridiagonal{T}, w::Vector{T}, ψ::Vector{T}) where {T}
    s = zero(T)
    for i in eachindex(w, ψ)
        s += w[i] * log(ψ[i])
    end
    return dot(ψ, negM, ψ) / 2 - s
end

"""
    _solve_amplitude(x, w, κ) -> ψ

Minimise the strictly convex potential `F(ψ) = ½ ψ'(−M)ψ − Σ wᵢ ln ψᵢ` over `ψ > 0`
by a damped Newton iteration with an SPD tridiagonal Hessian. The minimiser solves
`(−M)ψ = w ./ ψ`, i.e. the field equation at unit multiplier; the caller rescales it
to impose normalisation.

Each step factorises the tridiagonal Hessian in place (`ldlt!`/`ldiv!`) and backtracks
along the Newton direction to keep `ψ > 0` with Armijo decrease. Iteration stops when the
Newton decrement `λ² = ∇FᵀΔ` drops below a relative tolerance, or when the line search can
no longer decrease `F` — the point where rounding, not the algorithm, limits progress.
Chasing the decrement below that floor would spin uselessly, so the stalled line search is
itself the convergence signal. All scratch is allocated once; the iterations do not allocate.
"""
function _solve_amplitude(x::Vector{T}, w::Vector{T}, κ::T) where {T}
    negM = _neg_M(x, κ)
    n = length(w)
    ψ = fill(one(T), n)             # strictly positive start
    g = similar(ψ); Δ = similar(ψ); ψnew = similar(ψ)
    Hdv = similar(ψ); Hev = similar(negM.ev)   # Hessian factorisation scratch
    ctol = eps(T)^(2 // 3)          # relative Newton-decrement tolerance
    Fψ = _objective(negM, w, ψ)
    for _ in 1:100
        mul!(g, negM, ψ)
        @. g -= w / ψ                       # ∇F = (−M)ψ − w./ψ
        @. Hdv = negM.dv + w / ψ^2          # diagonal of ∇²F; off-diagonal equals negM.ev
        Hev .= negM.ev                      # ldlt! overwrites its arguments; refill each step
        Δ .= g
        ldiv!(ldlt!(SymTridiagonal(Hdv, Hev)), Δ)   # Δ = (∇²F)⁻¹ ∇F
        decrement = dot(g, Δ)               # Newton decrement λ² = ∇Fᵀ(∇²F)⁻¹∇F ≥ 0
        decrement <= ctol * max(one(T), abs(Fψ)) && break
        # Largest α ≤ 1 keeping ψ − αΔ strictly positive, then Armijo backtracking.
        α = one(T)
        for i in eachindex(ψ, Δ)
            Δ[i] > 0 && (α = min(α, ψ[i] / Δ[i]))
        end
        α < one(T) && (α *= oftype(α, 0.99))
        armijo = false
        local Fnew
        while α >= eps(T)
            @. ψnew = ψ - α * Δ
            Fnew = _objective(negM, w, ψnew)
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
        fcross = cs * (θ * ct - one(T)) / (2κ)
        Z += fdiag * (ψ[k]^2 + ψ[k+1]^2) + 2 * fcross * ψ[k] * ψ[k+1]
    end
    return Z
end

"""
    amplitude(d::PenalizedDensityEstimate, x)

Evaluate the amplitude `ψ(x)` (so that the density is `d(x) == ψ(x)^2`) at real `x`,
which may be a scalar or an array.
"""
amplitude(d::PenalizedDensityEstimate, x::Real) = _amplitude(d, x)
amplitude(d::PenalizedDensityEstimate, x::AbstractArray) = map(xi -> _amplitude(d, xi), x)

function _amplitude(d::PenalizedDensityEstimate{T}, x::Real) where {T}
    xs, ψ, κ = d.x, d.ψ, d.κ
    n = length(xs)
    if x <= xs[1]
        return ψ[1] * exp(κ * (x - xs[1]))
    elseif x >= xs[n]
        return ψ[n] * exp(-κ * (x - xs[n]))
    end
    k = searchsortedlast(xs, x)     # xs[k] <= x < xs[k+1]
    a = κ * (xs[k+1] - x)           # a, b ≥ 0 and a + b = θ
    b = κ * (x - xs[k])
    return ψ[k] * _sinh_ratio(a, a + b) + ψ[k+1] * _sinh_ratio(b, a + b)
end

# sinh(u)/sinh(θ) for 0 ≤ u ≤ θ, evaluated without overflow at large θ.
_sinh_ratio(u::T, θ::T) where {T} = exp(u - θ) * expm1(-2u) / expm1(-2θ)

(d::PenalizedDensityEstimate)(x::Real) = _amplitude(d, x)^2
(d::PenalizedDensityEstimate)(x::AbstractArray) = map(d, x)

"""
    action(d::PenalizedDensityEstimate) -> S

Classical action `S[ψ_cl] = N − λ − Σᵢ wᵢ ln Q(xᵢ)` (Eq. 10) of the fitted density,
where `N = Σ wᵢ`. Used by [`select_kappa`](@ref).
"""
function action(d::PenalizedDensityEstimate)
    N = sum(d.w)
    return N - d.λ - sum(d.w .* log.(d.ψ.^2))
end

"""
    chisq(d::PenalizedDensityEstimate, Q) -> χ²

Goodness-of-fit statistic between a trial density `Q` and the data underlying the
fit `d`, the robust field-theoretic analogue of Pearson's χ² (Eqs. 13–14 of the
paper):

    χ² = 4 Σᵢ wᵢ (√Q(xᵢ) / ψ_cl(xᵢ) − 1)²,

summed over the data nodes `xᵢ` with multiplicities `wᵢ`, where `ψ_cl = √(d(·))`
is the fitted amplitude. `Q` is any callable returning density values; it should be
a normalised density (`∫Q dx = 1`). `chisq(d, d) == 0`. Small χ² means `Q` is close
to the data in the (squared Hellinger) sense; see [`pvalue`](@ref) and
[`chisq_ccdf`](@ref) for significance.
"""
function chisq(d::PenalizedDensityEstimate{T}, Q) where {T}
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
    expected_chisq(d::PenalizedDensityEstimate) -> ⟨χ²⟩

Mean of the reference χ² distribution in the large-`N` limit (Eq. 25),
`⟨χ²⟩ = κ X / √2`, where `X = (1/N) Σᵢ wᵢ / Q_cl(xᵢ)` estimates the size of the
region occupied by the data. With `1/κ` read as an effective bin width this is
about `1/√2 ≈ 0.7` per degree of freedom.
"""
function expected_chisq(d::PenalizedDensityEstimate{T}) where {T}
    N = sum(d.w)
    X = sum(d.w ./ d.ψ.^2) / N          # (1/N) Σ wᵢ / Q_cl(xᵢ),  Q_cl = ψ²
    return d.κ * X / sqrt(T(2))
end

# Standard normal CDF, Φ(t) = ½ erfc(−t/√2).
_Φ(t::T) where {T} = erfc(-t / sqrt(T(2))) / 2

"""
    chisq_pdf(d::PenalizedDensityEstimate, z) -> P(z)

Density of the reference χ² distribution at `z ≥ 0` in the large-`N` limit
(Eq. 26): the inverse-Gaussian (Wald) law with mean `⟨χ²⟩ =` [`expected_chisq`](@ref)`(d)`
and shape `⟨χ²⟩²`,

    P(z) = ⟨χ²⟩ / √(2π z³) · exp[⟨χ²⟩ − z/2 − ⟨χ²⟩²/(2z)].
"""
function chisq_pdf(d::PenalizedDensityEstimate{T}, z::Real) where {T}
    zT = T(z)
    zT > 0 || return zero(T)
    μ = expected_chisq(d)
    return μ / sqrt(2 * T(π) * zT^3) * exp(μ - zT / 2 - μ^2 / (2 * zT))
end

"""
    chisq_ccdf(d::PenalizedDensityEstimate, z) -> P(χ² ≥ z)

Upper-tail (complementary CDF, i.e. survival) probability of the reference χ²
distribution [`chisq_pdf`](@ref). Evaluating it at an observed statistic gives a
p-value; see [`pvalue`](@ref).
"""
function chisq_ccdf(d::PenalizedDensityEstimate{T}, z::Real) where {T}
    zT = T(z)
    zT > 0 || return one(T)
    μ = expected_chisq(d)
    λ = μ^2
    r = sqrt(λ / zT)
    a = r * (zT / μ - 1)
    b = r * (zT / μ + 1)
    # Survival = Φ(−a) − e^{2λ/μ} Φ(−b); the second term uses erfcx so the large
    # positive exponent 2λ/μ cancels against −b²/2 without overflow.
    term2 = erfcx(b / sqrt(T(2))) * exp(2λ / μ - b^2 / 2) / 2
    return _Φ(-a) - term2
end

"""
    pvalue(d::PenalizedDensityEstimate, Q) -> p

Significance of the fit of a trial density `Q`: the probability that the reference
χ² distribution exceeds the observed [`chisq`](@ref)`(d, Q)`, i.e.
`chisq_ccdf(d, chisq(d, Q))`. Valid in the large-`N` limit.
"""
pvalue(d::PenalizedDensityEstimate, Q) = chisq_ccdf(d, chisq(d, Q))

"""
    select_kappa(x; κs, rtol=1e-6) -> κ

Choose the smoothing scale by Stevenson's principle of minimum sensitivity: return
the `κ` in the grid `κs` at which the classical action [`action`](@ref) is least
sensitive to `κ`, i.e. `|dS/d ln κ|` is smallest (Eq. and Fig. 1 of the paper). Over
a plateau of width `~N`, `S` is insensitive to the precise `κ`.

`κs` must be sorted and positive.
"""
function select_kappa(x::AbstractVector{<:Real}; κs::AbstractVector{<:Real}, rtol::Real=1e-6)
    issorted(κs) && all(>(0), κs) || throw(ArgumentError("κs must be sorted and positive"))
    length(κs) >= 3 || throw(ArgumentError("need at least 3 values in κs to estimate sensitivity"))
    lnκ = log.(κs)
    S = [action(PenalizedDensityEstimate(x; κ, rtol)) for κ in κs]
    best_i, best_slope = 0, Inf
    for i in 2:length(κs)-1
        slope = abs((S[i+1] - S[i-1]) / (lnκ[i+1] - lnκ[i-1]))
        if slope < best_slope
            best_slope, best_i = slope, i
        end
    end
    return oftype(float(first(κs)), κs[best_i])
end

"""
    kappa_interval(x; level=0.2, rtol=1e-6) -> (; κ, lo, hi)

Principled smoothing-scale selection with an interval of plausible values.

As `κ` sweeps from `0` to `∞` the classical action's reduced form `g(κ) = S(κ) + W ln κ`
(with `W = Σ wᵢ` the total count) rises monotonically between two exact limits:
`g → W/2` as `κ → 0` (all points merge into one lump) and `g → W/2 + W H` as `κ → ∞`
(the `N` points become isolated), where `H = −Σᵢ (wᵢ/W) ln(wᵢ/W)` is the Shannon entropy
of the multiplicities (`ln N` for distinct points). The normalised quantity

    h(κ) = (g(κ) − W/2) / (W H) ∈ [0, 1]

is therefore the fraction of the data's entropy that scale `κ` resolves. Its half-point
`h = 1/2` coincides with the point of minimum sensitivity of `S` used by
[`select_kappa`](@ref), but is located against exact bounds rather than a discrete
derivative.

Returns the half-entropy scale `κ` (`h = 1/2`) together with the interval `[lo, hi]`
bracketing `h ∈ [(1−level)/2, (1+level)/2]`; the default `level=0.2` spans `h ∈ [0.4, 0.6]`.
Requires at least two distinct points.
"""
function kappa_interval(x::AbstractVector{<:Real}; level::Real=0.2, rtol::Real=1e-6)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    T = float(promote_type(eltype(x), typeof(level), typeof(rtol)))
    xs = sort!(T[xi for xi in x])
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
    κ = _invert_monotone(h, one(T) / 2)
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

end # module
