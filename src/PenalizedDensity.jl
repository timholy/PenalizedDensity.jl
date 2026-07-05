module PenalizedDensity

using LinearAlgebra
using SpecialFunctions: erfc, erfcx

export PenalizedDensityEstimate, amplitude, action, select_kappa
export chisq, expected_chisq, chisq_pdf, chisq_ccdf, pvalue

"""
    PenalizedDensityEstimate(x; κ, atol=0.0)

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

Repeated (or near-repeated within `atol`) points are merged and enter with integer
weight, so weighted data is handled naturally.

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

function PenalizedDensityEstimate(x::AbstractVector{<:Real}; κ::Real, atol::Real=0.0)
    κ > 0 || throw(ArgumentError("κ must be positive, got $κ"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative, got $atol"))
    isempty(x) && throw(ArgumentError("cannot fit a density to zero points"))
    T = float(promote_type(eltype(x), typeof(κ), typeof(atol)))
    nodes, weights = _merge_sorted(x, T(atol), T)
    κT = T(κ)
    ψ = _solve_amplitude(nodes, weights, κT)
    Z = _norm_sq(nodes, ψ, κT)
    ψ ./= sqrt(Z)
    λ = κT * Z                      # scaling law: normalised ψ solves (−M)ψ = (κ/λ)/ψ
    return PenalizedDensityEstimate{T}(nodes, weights, ψ, κT, λ)
end

"""
    _merge_sorted(x, atol, T) -> (nodes, weights)

Sort `x` and collapse runs of points within `atol` of the run's first member into a
single node carrying the count as its weight. Returns distinct, strictly increasing
`nodes::Vector{T}` and matching `weights`, regardless of the axes of `x`.
"""
function _merge_sorted(x::AbstractVector, atol::T, ::Type{T}) where {T}
    p = sortperm(x)
    nodes = T[]
    weights = T[]
    for k in p
        xk = T(x[k])
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
        s = sinh(θ)
        d[k]   += cosh(θ) / s       # coth θ
        d[k+1] += cosh(θ) / s
        e[k]    = -one(T) / s       # −csch θ
    end
    return SymTridiagonal(d, e)
end

"""
    _solve_amplitude(x, w, κ) -> ψ

Minimise the strictly convex potential `F(ψ) = ½ ψ'(−M)ψ − Σ wᵢ ln ψᵢ` over `ψ > 0`
by a damped Newton iteration with an SPD tridiagonal Hessian. The minimiser solves
`(−M)ψ = w ./ ψ`, i.e. the field equation at unit multiplier; the caller rescales it
to impose normalisation.
"""
function _solve_amplitude(x::Vector{T}, w::Vector{T}, κ::T) where {T}
    n = length(x)
    negM = _neg_M(x, κ)
    ψ = fill(one(T), n)             # strictly positive start
    F(ψ) = (dot(ψ, negM, ψ) - 2 * dot(w, log.(ψ))) / 2
    Fψ = F(ψ)
    for _ in 1:100
        g = negM * ψ .- w ./ ψ                      # ∇F
        H = SymTridiagonal(negM.dv .+ w ./ ψ.^2, negM.ev)   # ∇²F, SPD tridiagonal
        Δ = H \ g
        # Backtracking that also keeps ψ + αΔ strictly positive.
        α = one(T)
        while any(<=(0), ψ .- α .* Δ)
            α /= 2
        end
        local ψnew, Fnew
        while true
            ψnew = ψ .- α .* Δ
            Fnew = F(ψnew)
            (Fnew <= Fψ - α * dot(g, Δ) / 4 || α < eps(T)) && break
            α /= 2
        end
        decrement = dot(g, Δ)       # Newton decrement λ² = gᵀH⁻¹g = gᵀΔ
        ψ, Fψ = ψnew, Fnew
        decrement <= 2 * eps(T) * max(one(T), abs(Fψ)) && break
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
        s = sinh(θ)
        Ia = sinh(2θ) / 4 - θ / 2               # ∫ sinh² over the interval (both ends)
        Ic = (θ * cosh(θ) - sinh(θ)) / 2        # cross term
        f = 1 / (κ * s^2)
        Z += f * (Ia * (ψ[k]^2 + ψ[k+1]^2) + 2 * Ic * ψ[k] * ψ[k+1])
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
    θ = κ * (xs[k+1] - xs[k])
    return (ψ[k] * sinh(κ * (xs[k+1] - x)) + ψ[k+1] * sinh(κ * (x - xs[k]))) / sinh(θ)
end

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
    select_kappa(x; κs, atol=0.0) -> κ

Choose the smoothing scale by Stevenson's principle of minimum sensitivity: return
the `κ` in the grid `κs` at which the classical action [`action`](@ref) is least
sensitive to `κ`, i.e. `|dS/d ln κ|` is smallest (Eq. and Fig. 1 of the paper). Over
a plateau of width `~N`, `S` is insensitive to the precise `κ`.

`κs` must be sorted and positive.
"""
function select_kappa(x::AbstractVector{<:Real}; κs::AbstractVector{<:Real}, atol::Real=0.0)
    issorted(κs) && all(>(0), κs) || throw(ArgumentError("κs must be sorted and positive"))
    length(κs) >= 3 || throw(ArgumentError("need at least 3 values in κs to estimate sensitivity"))
    lnκ = log.(κs)
    S = [action(PenalizedDensityEstimate(x; κ, atol)) for κ in κs]
    best_i, best_slope = 0, Inf
    for i in 2:length(κs)-1
        slope = abs((S[i+1] - S[i-1]) / (lnκ[i+1] - lnκ[i-1]))
        if slope < best_slope
            best_slope, best_i = slope, i
        end
    end
    return oftype(float(first(κs)), κs[best_i])
end

end # module
