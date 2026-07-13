module PenalizedDensity

using LinearAlgebra: LinearAlgebra, I, SymTridiagonal, dot, ldiv!, ldlt!, mul!
using QuadGK: quadgk
using SpecialFunctions: erfc, erfcx
using Statistics: Statistics, quantile

export DensityEstimate, amplitude, action, select_kappa_ms, select_kappa_cv, select_kappa_kl, kappa_interval
export chisq, expected_chisq, chisq_reference, ChisqReference, chisq_pdf, chisq_ccdf, pvalue
export cdf, quantile

"""
    DensityEstimate(x::AbstractVector{T}; κ, rtol=cbrt(eps(T)))

Estimate a continuous one-dimensional probability density from sample points `x`,
using the scalar-field method of Holy, *Phys. Rev. Lett.* **79**, 3545 (1997).

The density is written as `Q(x) = ψ(x)^2`, where the amplitude `ψ` minimizes the
action

    S[ψ] = ∫ (ℓ²/2) (ψ')² dx - 2 Σᵢ ln ψ(xᵢ)

subject to `∫ ψ² dx = 1`. The smoothing scale `κ = √(2λ)/ℓ` (with `λ` the
normalization multiplier) sets the width of each point's contribution; larger `κ`
gives a rougher estimate. See [`select_kappa_kl`](@ref) for choosing it automatically
(the recommended default; [`select_kappa_cv`](@ref), [`select_kappa_ms`](@ref), and
[`kappa_interval`](@ref) are alternatives).

Between sorted data points `ψ` solves `ψ'' = κ² ψ`, i.e. it is a sum of rising and
falling exponentials, and decays as `e^{-κ|x|}` in the tails. The nodal amplitudes
`ψ(xᵢ)` satisfy a symmetric tridiagonal system whose solution is the minimizer of a
strictly convex potential; normalization is then a rescaling.

The returned object is callable: `d(x)` evaluates the density `Q(x)` at any real
`x`, and it can be broadcast over arrays. Use [`amplitude`](@ref) for `ψ(x)`.

Repeated points, and points closer than `rtol / κ` (i.e. within a fraction `rtol` of the
smoothing length `1 / κ`), are merged into one node carrying the count as its integer
weight, so weighted data is handled naturally. Without merging, the resulting
tridiagonal system can be nearly singular.

# Examples
```jldoctest
julia> d = DensityEstimate([-1.0, 0.0, 0.0, 1.0]; κ=1.0);

julia> d(0.0) > d(2.0)   # denser near the data
true

julia> round(chisq(d, d); digits=8)   # a distribution fits itself perfectly
0.0
```
"""
struct DensityEstimate{T<:AbstractFloat}
    x::Vector{T}   # sorted, distinct node locations
    w::Vector{T}   # weight (multiplicity) at each node
    ψ::Vector{T}   # normalized amplitude at the nodes
    κ::T           # smoothing scale
    λ::T           # normalization multiplier (diagnostic)
end

function DensityEstimate(x::AbstractVector{R}; κ::Real, rtol::Real=cbrt(eps(R))) where R<:Real
    κ > 0 || throw(ArgumentError("κ must be positive, got $κ"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    isempty(x) && throw(ArgumentError("cannot fit a density to zero points"))
    T = float(promote_type(R, typeof(κ), typeof(rtol)))
    nodes, weights = _merge_sorted(x, T(rtol) / T(κ), T)
    return _fit(nodes, weights, T(κ))
end

Base.show(io::IO, d::DensityEstimate) = print(io, "DensityEstimate with $(length(d.x)) distinct nodes, $(sum(d.w)) total weight, κ=$(d.κ), λ=$(d.λ)")

# Fit from alreadyMerged distinct nodes and their weights.
function _fit(nodes::Vector{T}, weights::Vector{T}, κ::T) where {T}
    ψ = _solve_amplitude(nodes, weights, κ)
    Z = _norm_sq(nodes, ψ, κ)
    ψ ./= sqrt(Z)
    λ = κ * Z                       # scaling law: normalized ψ solves Mψ = (κ/λ)/ψ
    return DensityEstimate{T}(nodes, weights, ψ, κ, λ)
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

# Z = ∫ψ² together with its κ-derivative at fixed ψ and Gψ = ½ ∂Z/∂ψ, where Z = ψᵀGψ. The
# three share the per-interval coth/csch coefficients, so one pass returns all of them.
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
function _int_quartic(x::Vector{T}, ψ::Vector{T}, κ::T) where {T}
    n = length(x)
    Q2 = (ψ[1]^4 + ψ[n]^4) / (4κ)       # tails: ∫ψ₁⁴ e^{4κ(x-x₁)} dx and its mirror
    for k in 1:n-1
        p, q = ψ[k], ψ[k+1]
        θ = κ * (x[k+1] - x[k])
        ct, cs = coth(θ), csch(θ)
        Δ = q - p
        cm1 = 2 * sinh(θ / 2)^2                              # coshθ - 1
        boundary = κ * cs * (cm1 * (p^4 + q^4) + Δ^2 * (p^2 + p*q + q^2))   # [u³u']ₖ^{k+1}
        E = κ^2 * cs^2 * (Δ^2 - 2 * p * q * cm1)             # u'² - κ²u²
        fdiag  = (ct - θ * cs^2) / (2κ)
        fcross = cs * (θ * ct - one(T)) / (2κ)
        Iseg = fdiag * (p^2 + q^2) + 2 * fcross * p * q      # ∫u² over the interval
        Q2 += (boundary - 3 * E * Iseg) / (4κ^2)
    end
    return Q2
end

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
    x, ψ, κ = d.x, d.ψ, d.κ
    n = length(x)
    F = Vector{T}(undef, n)
    F[1] = ψ[1]^2 / (2κ)                # left tail
    for k in 1:n-1
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
    return F, F[n] + ψ[n]^2 / (2κ)
end

# Unnormalized cumulative mass ∫_{-∞}^{x} ψ² dt, given the node cumulatives F. The tails
# are elementary exponential integrals; interior intervals use _cdf_mass_interior.
function _cdf_mass(d::DensityEstimate{T}, F::Vector{T}, x::Real) where {T}
    xs, ψ, κ = d.x, d.ψ, d.κ
    n = length(xs)
    isnan(x) && return T(NaN) * one(x)
    if x <= xs[1]
        return ψ[1]^2 / (2κ) * exp(2κ * (x - xs[1]))
    elseif x >= xs[n]
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
    xs, ψ, κ = d.x, d.ψ, d.κ
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
julia> d = DensityEstimate([0.0]; κ=0.5);   # a Laplace density centered at 0

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
    xs, ψ, κ = d.x, d.ψ, d.κ
    n = length(xs)
    target = q * total
    if target <= F[1]                   # left tail: target = ψ₁²/(2κ) e^{2κ(x-x₁)}
        return xs[1] + log(2κ * target / ψ[1]^2) / (2κ)
    elseif target >= F[n]               # right tail, through the complement 1 - q
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

Mean of the reference χ² distribution in the large-`N` limit (Eq. 25),
`⟨χ²⟩ = κ X / √2`, where `X = (1/N) Σᵢ wᵢ / Q_cl(xᵢ)` estimates the size of the
region occupied by the data. With `1/κ` read as an effective bin width this is
about `1/√2 ≈ 0.7` per degree of freedom.

For the exact finite-`N` mean, use `expected_chisq(`[`chisq_reference`](@ref)`(d))`.
"""
function expected_chisq(d::DensityEstimate{T}) where {T}
    N = sum(d.w)
    X = sum(d.w ./ d.ψ.^2) / N          # (1/N) Σ wᵢ / Q_cl(xᵢ),  Q_cl = ψ²
    return d.κ * X / sqrt(T(2))
end

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
# Everything is tridiagonal. The unconstrained node covariance obeys
#   C₀⁻¹ = G₀⁻¹ + S,  (G₀)ᵢⱼ = κ e^{-κ|xᵢ-xⱼ|}/(4λ),  S = diag(2wₖ/ψₖ²),
# and the exponential kernel is Markov, so G₀⁻¹ = (2λ/κ)·M with M the
# `roughness_operator`. The ∫ψ_cl δψ = 0 constraint contributes one rank-one term,
# C = C₀ - b bᵀ/Vφ (Eq. 18's T(g) factor). Tail probabilities come from Imhof's
# inversion, whose integrand needs only det(I + iuA) per node — an O(N)
# tridiagonal determinant plus a rank-one correction — so no eigenvalues are formed.

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
    mean::T                             # exact ⟨χ²⟩ = tr(A)
end

Base.show(io::IO, r::ChisqReference) =
    print(io, "ChisqReference($(length(r.g)) nodes, ⟨χ²⟩=$(r.mean))")

# ∫ over one interval of the amplitude against a unit exponential decaying toward each
# endpoint: p̃ = ∫ ψ e^{-κ(x_{k+1}-x)}dx, q̃ = ∫ ψ e^{-κ(x-x_k)}dx. The coth/csch forms
# stay finite as θ → ∞ (isolated points); expm1 keeps S1 accurate as θ → 0.
function _pq_tilde(ψk::T, ψk1::T, θ::T, κ::T) where {T}
    cs = csch(θ)
    S1 = (θ + expm1(-2θ) / 2) / (2κ)                # ∫₀ʰ sinh(κu) e^{-κu} du
    S2 = (sinh(θ) - θ * exp(-θ)) / (2κ)             # ∫₀ʰ sinh(κ(h-u)) e^{-κu} du
    return (ψk * S1 + ψk1 * S2) * cs, (ψk * S2 + ψk1 * S1) * cs
end

# A = (κ²/2λ) ∫ α² dx, α = L₀⁻¹ψ_cl with α(xₖ)=ãₖ; closed-form per interval plus tails.
function _constraint_A(x, ψ, ã, κ::T, λ) where {T}
    n = length(x)
    acc = ψ[1]^2 / (8κ^3)                                        # left tail
    acc += ã[n]^2 / (2κ) + ã[n] * ψ[n] / (2κ^2) + ψ[n]^2 / (4κ^3) # right tail
    for k in 1:n-1
        h = x[k+1] - x[k]; θ = κ * h
        D = ψ[k+1] * csch(θ) - ψ[k] * coth(θ)
        Ak = (ψ[k] + D) / 2; Bk = (ψ[k] - D) / 2; ãk = ã[k]
        α(s) = exp(-κ*s) * ãk + Ak * sinh(κ*s) / κ + Bk * s * exp(-κ*s)
        acc += quadgk(s -> α(s)^2, zero(h), h; rtol = sqrt(eps(T)))[1]
    end
    return (κ^2 / (2λ)) * acc
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
"""
function chisq_reference(d::DensityEstimate{T}) where {T}
    x, ψ, w, κ, λ = d.x, d.ψ, d.w, d.κ, d.λ
    n = length(x)
    # m_k = ∫ ψ_cl(x) G₀(xₖ,x) dx via forward/backward exponential accumulations.
    ã = similar(ψ); b̃ = similar(ψ)
    ã[1] = ψ[1] / (2κ)
    for k in 1:n-1
        θ = κ * (x[k+1] - x[k])
        p, _ = _pq_tilde(ψ[k], ψ[k+1], θ, κ)
        ã[k+1] = exp(-θ) * ã[k] + p
    end
    b̃[n] = ψ[n] / (2κ)
    for k in n-1:-1:1
        θ = κ * (x[k+1] - x[k])
        _, q = _pq_tilde(ψ[k], ψ[k+1], θ, κ)
        b̃[k] = q + exp(-θ) * b̃[k+1]
    end
    m = (κ / (4λ)) .* (ã .+ b̃)
    # C₀⁻¹ = (2λ/κ) M + diag(2 wᵢ/ψᵢ²);  b = (I + G₀S)⁻¹ m solves C₀⁻¹ b = G₀⁻¹ m.
    M = roughness_operator(x, κ)
    C0inv = SymTridiagonal((2λ/κ) .* M.dv .+ 2 .* w ./ ψ.^2, (2λ/κ) .* M.ev)
    b = C0inv \ ((2λ/κ) .* (M * m))
    Vφ = _constraint_A(x, ψ, ã, κ, λ) - sum(m .* (2 .* w ./ ψ.^2) .* b)   # Var(∫ψ_cl δψ)
    # Reduced tridiagonal tri = D^{-1/2} C₀⁻¹ D^{-1/2} and rank-one direction g.
    D = 4 .* w ./ ψ.^2; sq = sqrt.(D)
    tri = SymTridiagonal(C0inv.dv ./ D, C0inv.ev ./ (sq[1:n-1] .* sq[2:n]))
    g = sq .* b ./ sqrt(Vφ)
    return ChisqReference{T}(tri, g, _tridiag_invdiag(tri) - sum(abs2, g))
end

expected_chisq(r::ChisqReference) = r.mean

# (unwrapped arg, modulus) of Φ(u) = det(I + iuA), A = tri⁻¹ - g gᵀ. The determinant of
# I+iu·tri⁻¹ is a ratio of tridiagonal determinants (continuant recurrence, accumulated in
# log space so the phase unwraps past π); the rank-one term is one complex tridiagonal
# solve. Both O(N).
function _logΦ(r::ChisqReference, u::Real)
    a, β = r.tri.dv, r.tri.ev; n = length(a); iu = im * u
    r0 = complex(a[1]); rr = a[1] + iu
    s = log(rr) - log(r0)
    for k in 2:n
        r0 = a[k] - β[k-1]^2 / r0
        rr = (a[k] + iu) - β[k-1]^2 / rr
        s += log(rr) - log(r0)
    end
    y = SymTridiagonal(complex.(a) .+ iu, complex.(β)) \ (r.tri * r.g)
    rank1 = 1 - iu * dot(r.g, y)
    return imag(s) + angle(rank1), exp(real(s)) * abs(rank1)
end

"""
    chisq_ccdf(d::DensityEstimate, z; method=:exact) -> P(χ² ≥ z)
    chisq_ccdf(ref::ChisqReference, z)               -> P(χ² ≥ z)

Upper-tail (survival) probability of the reference χ² distribution at `z`. Evaluated at an
observed statistic it is a p-value; see [`pvalue`](@ref).

`method=:exact` (default) uses the finite-`N` generalized-χ² law via Imhof inversion of
[`chisq_reference`](@ref)`(d)`; pass a prebuilt [`ChisqReference`](@ref) to avoid
reassembling it. `method=:largeN` uses the large-`N` inverse-Gaussian (Wald) approximation
of Eq. 26, with mean `⟨χ²⟩ =` [`expected_chisq`](@ref)`(d)`; it is cheap but overstates tail
probabilities at the scales `select_kappa_ms` and `select_kappa_cv` typically choose.
"""
function chisq_ccdf(r::ChisqReference{T}, z::Real; rtol=sqrt(eps(T))) where {T}
    zT = T(z)
    f(u) = u == 0 ? (r.mean - zT) / 2 :
        (θ = _logΦ(r, u); sin((θ[1] - zT * u) / 2) / (u * sqrt(θ[2])))
    I, _ = quadgk(f, zero(T), T(Inf); rtol)      # I ∈ [-π/2, π/2]; no tiny-value churn
    return clamp(one(T)/2 + I / T(π), zero(T), one(T))
end
chisq_ccdf(d::DensityEstimate, z::Real; method::Symbol=:exact) =
    _dispatch_chisq(chisq_ccdf, _chisq_ccdf_largeN, d, z, method)

function _chisq_ccdf_largeN(d::DensityEstimate{T}, z::Real) where {T}
    zT = T(z)
    zT > 0 || return one(T)
    μ = expected_chisq(d)
    λ = μ^2
    r = sqrt(λ / zT)
    a = r * (zT / μ - 1)
    b = r * (zT / μ + 1)
    # Survival = Φ(-a) - e^{2λ/μ} Φ(-b); the second term uses erfcx so the large
    # positive exponent 2λ/μ cancels against -b²/2 without overflow.
    term2 = erfcx(b / sqrt(T(2))) * exp(2λ / μ - b^2 / 2) / 2
    return _Φ(-a) - term2
end

"""
    chisq_pdf(d::DensityEstimate, z; method=:exact) -> P(z)
    chisq_pdf(ref::ChisqReference, z)               -> P(z)

Density of the reference χ² distribution at `z ≥ 0`. `method=:exact` (default) is the
finite-`N` generalized-χ² law from [`chisq_reference`](@ref)`(d)` (pass a prebuilt
[`ChisqReference`](@ref) to reuse it). `method=:largeN` is the large-`N` inverse-Gaussian
(Wald) density of Eq. 26,

    P(z) = ⟨χ²⟩ / √(2π z³) · exp[⟨χ²⟩ - z/2 - ⟨χ²⟩²/(2z)],

with `⟨χ²⟩ =` [`expected_chisq`](@ref)`(d)`.
"""
function chisq_pdf(r::ChisqReference{T}, z::Real; rtol=sqrt(eps(T)), atol=sqrt(eps(T))) where {T}
    # atol floors the density: deep in the tail the true value underflows to ~0, and a purely
    # relative tolerance would otherwise subdivide the oscillatory integrand without end.
    zT = T(z)
    f(u) = (θ = _logΦ(r, u); cos((θ[1] - zT * u) / 2) / sqrt(θ[2]))
    I, _ = quadgk(f, zero(T), T(Inf); rtol, atol, maxevals=10^4)
    return max(I / (2 * T(π)), zero(T))
end
chisq_pdf(d::DensityEstimate, z::Real; method::Symbol=:exact) =
    _dispatch_chisq(chisq_pdf, _chisq_pdf_largeN, d, z, method)

function _chisq_pdf_largeN(d::DensityEstimate{T}, z::Real) where {T}
    zT = T(z)
    zT > 0 || return zero(T)
    μ = expected_chisq(d)
    return μ / sqrt(2 * T(π) * zT^3) * exp(μ - zT / 2 - μ^2 / (2 * zT))
end

function _dispatch_chisq(exact, largeN, d::DensityEstimate, z, method::Symbol)
    method === :exact  && return exact(chisq_reference(d), z)
    method === :largeN && return largeN(d, z)
    throw(ArgumentError("method must be :exact or :largeN, got :$method"))
end

"""
    pvalue(d::DensityEstimate, Q; method=:exact) -> p
    pvalue(ref::ChisqReference, χ²)              -> p

Significance of the fit of a trial density `Q`: the probability that the reference χ²
distribution exceeds the observed [`chisq`](@ref)`(d, Q)`, i.e. `chisq_ccdf(d, chisq(d, Q))`.

`method=:exact` (default) uses the finite-`N` generalized-χ² law; `method=:largeN` uses the
Eq. 26 approximation. To test several trial densities against one fit, build the reference
once with [`chisq_reference`](@ref) and call `pvalue(ref, chisq(d, Q))`.
"""
pvalue(r::ChisqReference, χ²::Real) = chisq_ccdf(r, χ²)
function pvalue(d::DensityEstimate, Q; method::Symbol=:exact)
    method === :exact  && return chisq_ccdf(chisq_reference(d), chisq(d, Q))
    method === :largeN && return _chisq_ccdf_largeN(d, chisq(d, Q))
    throw(ArgumentError("method must be :exact or :largeN, got :$method"))
end

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
    xs = sort!(T[xi for xi in x])
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
function _loo_density(nodes::Vector{T}, w::Vector{T}, κ::T) where {T}
    φ = _solve_amplitude(nodes, w, κ)
    Z, _, Gφ = _norm_sq_grad(nodes, φ, κ)
    A = roughness_operator(nodes, κ)
    H = SymTridiagonal(A.dv .+ w ./ φ.^2, A.ev)
    gii = _inv_diag(H)
    v = ldiv!(ldlt!(H), Gφ)             # H⁻¹Gφ; H is consumed, gii already extracted
    ψ = φ ./ sqrt(Z)
    looi = @. ψ^2 * (1 - 2 * gii / φ^2 + 2 * v / (φ * Z))
    return ψ, looi
end

# Least-squares cross-validation score LSCV(κ) = ∫Q̂² - (2/N) Σᵢ wᵢ Q̂₋ᵢ(xᵢ): an unbiased
# estimate, up to the κ-independent ∫Q², of the integrated squared error ∫(Q̂-Q)².
function _lscv(nodes::Vector{T}, w::Vector{T}, κ::T) where {T}
    ψ, looi = _loo_density(nodes, w, κ)
    N = sum(w)
    cross = zero(T)
    for i in eachindex(w, looi)
        cross += w[i] * looi[i]
    end
    return _int_quartic(nodes, ψ, κ) - 2 * cross / N
end

# Kullback–Leibler cross-validation score, the mean negative leave-one-out log-likelihood
# -(1/N) Σᵢ wᵢ ln Q̂₋ᵢ(xᵢ): an estimate, up to a κ-independent constant, of KL(Q ‖ Q̂_κ). Reuses
# the same first-order leave-one-out densities as _lscv. A non-positive Q̂₋ᵢ (possible where the
# first-order expansion overshoots) makes the log undefined; return NaN so the search rejects κ.
function _klcv(nodes::Vector{T}, w::Vector{T}, κ::T) where {T}
    _, looi = _loo_density(nodes, w, κ)
    s = zero(T)
    for i in eachindex(w, looi)
        looi[i] > 0 || return T(NaN)
        s += w[i] * log(looi[i])
    end
    return -s / sum(w)
end

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
    xs = sort!(T[xi for xi in x])
    r = T(rtol)
    score(κ) = (v = scorefun(_merge_presorted(xs, r / κ)..., κ); isfinite(v) ? v : typemax(T))
    lnκ = log.(T.(κs))
    i = argmin(score.(exp.(lnκ)))               # coarse bracket on the grid
    lo = lnκ[max(i - 1, firstindex(lnκ))]
    hi = lnκ[min(i + 1, lastindex(lnκ))]
    return exp(_golden_min(l -> score(exp(l)), lo, hi))
end

end # module
