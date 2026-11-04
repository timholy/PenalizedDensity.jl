# Derivative of the selected scale κ̂ with respect to the sample.
#
# κ̂ minimizes the KLCV score S(κ, x), so it solves g(κ, x) = ∂S/∂κ = 0 and, where that minimum
# is interior and nondegenerate, the implicit function theorem gives dκ̂/dx = -∇ₓg / ∂g/∂κ.
# `_klcv_adjoint` is the reverse pass of S: it returns ∇ₓS and ∂S/∂κ at a cost of O(N).
# Evaluating it on dual numbers carrying d/dκ (`_klcv_kappa_derivatives`) returns g, ∂g/∂κ and
# ∇ₓg in one more O(N) pass.

# Tag of the dual numbers that carry d/dκ through the reverse pass.
struct _KappaTag end

# The deletion step τ of `_deletion_step(h, w, 1)` and its slope dτ/dh = τ²(1 - 2wh)/√D, with
# D = 1 - 4h(1 - wh) the discriminant. D ≥ 1 - 1/w > 0 for w ≥ 2. For w = 1, D = (1 - 2h)², which
# an isolated node drives to zero (h → 1/2); τ = 2/(1 + |1 - 2h|) then avoids the square root,
# whose derivative is infinite at D = 0, and (1 - 2wh)/√D is a sign.
function _deletion_step_slope(h::T, w::T) where {T}
    if w == 1
        u = 1 - 2 * h
        τ = 2 / (1 + abs(u))
        return τ, τ^2 * (u >= 0 ? one(T) : -one(T))
    end
    τ = _deletion_step(h, w, one(T))
    return τ, τ^2 * (1 - 2 * w * h) / sqrt(1 - 4 * h * (1 - w * h))
end

# Reverse pass of `_inv_band_pivots`: given the adjoints ḡ of the diagonal g and ḡn of the
# off-diagonal gn of H⁻¹, add the adjoints of H's diagonal a and off-diagonal b to ā and b̄.
function _inv_band_adjoint!(ā::Vector{T}, b̄::Vector{T}, H::SymTridiagonal{T}, g, gn, d, δ,
                            ḡ_in::Vector{T}, ḡn::Vector{T}) where {T}
    a, b = H.dv, H.ev
    n = length(a)
    ḡ = copy(ḡ_in)
    d̄ = zeros(T, n); δ̄ = zeros(T, n)
    for i in 1:n-1                      # gnᵢ = -bᵢ gᵢ₊₁ / dᵢ
        b̄[i]   -= ḡn[i] * g[i+1] / d[i]
        ḡ[i+1] -= ḡn[i] * b[i] / d[i]
        d̄[i]   += ḡn[i] * b[i] * g[i+1] / d[i]^2
    end
    for i in 1:n                        # gᵢ = 1/(dᵢ + δᵢ - aᵢ)
        s̄ = -ḡ[i] * g[i]^2
        d̄[i] += s̄; δ̄[i] += s̄; ā[i] -= s̄
    end
    for i in 1:n-1                      # δᵢ = aᵢ - bᵢ²/δᵢ₊₁, reversed: δ̄ passes from i to i+1
        ā[i] += δ̄[i]
        b̄[i] -= 2 * b[i] * δ̄[i] / δ[i+1]
        δ̄[i+1] += b[i]^2 * δ̄[i] / δ[i+1]^2
    end
    ā[n] += δ̄[n]
    for i in n:-1:2                     # dᵢ = aᵢ - bᵢ₋₁²/dᵢ₋₁, reversed: d̄ passes from i to i-1
        ā[i] += d̄[i]
        b̄[i-1] -= 2 * b[i-1] * d̄[i] / d[i-1]
        d̄[i-1] += b[i-1]^2 * d̄[i] / d[i-1]^2
    end
    ā[1] += d̄[1]
    return ā, b̄
end

# KLCV score S of distinct point observations on the unbounded line at the constant scale κ,
# with its gradient x̄ = ∂S/∂x with respect to the node positions `x` and κ̄ = ∂S/∂κ, at fixed
# weights `w` and merge pattern. `φ` is the unnormalized amplitude (Mφ = w ⊘ φ), and node
# `locnode[k]` holds the k-th of the N observations; each is held out alone, by the deletion step
# of `_loo_terms` (none is large enough to be refitted). Everything depends on x and κ through
# the interval widths θₖ = κ(xₖ₊₁ - xₖ) and the 1/κ factors of the mass, so θ̄ is accumulated
# first and distributed to x̄ and κ̄ at the end.
function _klcv_adjoint(x::Vector{T}, w::Vector{T}, κ::T, φ::Vector{T},
                       locnode::AbstractVector{Int}) where {T}
    n = length(x)
    N = sum(w)
    lo, hi = T(-Inf), T(Inf)
    M = _roughness_operator(x, κ, lo, hi)
    Z, Gφ = _norm_sq_gram(x, φ, κ, κ, κ, lo, hi)
    H = SymTridiagonal(M.dv .+ w ./ φ.^2, copy(M.ev))
    gii, gnext, dpiv, δpiv = _inv_band_pivots(H)
    F = ldlt(H)
    v = F \ Gφ
    θ = T[κ * (x[k+1] - x[k]) for k in 1:n-1]

    φ̄ = zeros(T, n); v̄ = zeros(T, n); ḡii = zeros(T, n); ḡnext = zeros(T, n - 1)
    dv̄ = zeros(T, n)                   # adjoint of M's diagonal, apart from its effect on φ
    θ̄ = zeros(T, n - 1)
    Z̄ = zero(T); κ̄ = zero(T); S = zero(T)
    ℓ̄ = -one(T) / N                    # S = -(1/N) Σ ℓ
    for i in locnode
        h = gii[i] / φ[i]^2
        τ, τh = _deletion_step_slope(h, w[i])
        Zt = Z - 2 * τ * v[i] / φ[i]
        Zt > 0 || throw(ArgumentError("the leave-one-out normalization is not positive at κ = $κ"))
        τ̄ = zero(T); h̄ = zero(T)
        la2̄ = ℓ̄
        if w[i] == 1
            # Source-free deleted node: la2 = 2(ln Σⱼ e^{-ln sinh θ} φ'ⱼ - ln Mᵢᵢ) over the
            # neighbors j, φ'ⱼ = φⱼ - τ gnext/φᵢ, as in `_loo_terms`.
            lA = lB = T(-Inf)
            if i > 1
                pA = φ[i-1] - τ * gnext[i-1] / φ[i]
                pA > 0 || throw(ArgumentError("a leave-one-out neighbor amplitude is not positive at κ = $κ"))
                lA = log(pA) - logabssinh(θ[i-1])
            end
            if i < n
                pB = φ[i+1] - τ * gnext[i] / φ[i]
                pB > 0 || throw(ArgumentError("a leave-one-out neighbor amplitude is not positive at κ = $κ"))
                lB = log(pB) - logabssinh(θ[i])
            end
            l = logaddexp(lA, lB)
            la2 = 2 * (l - log(M.dv[i]))
            dv̄[i] -= 2 * la2̄ / M.dv[i]
            l̄ = 2 * la2̄
            if i > 1
                Ā = l̄ * exp(lA - l)
                θ̄[i-1] -= Ā * coth(θ[i-1])
                p̄ = Ā / pA
                φ̄[i-1] += p̄
                τ̄ -= p̄ * gnext[i-1] / φ[i]
                ḡnext[i-1] -= p̄ * τ / φ[i]
                φ̄[i] += p̄ * τ * gnext[i-1] / φ[i]^2
            end
            if i < n
                B̄ = l̄ * exp(lB - l)
                θ̄[i] -= B̄ * coth(θ[i])
                p̄ = B̄ / pB
                φ̄[i+1] += p̄
                τ̄ -= p̄ * gnext[i] / φ[i]
                ḡnext[i] -= p̄ * τ / φ[i]
                φ̄[i] += p̄ * τ * gnext[i] / φ[i]^2
            end
        else
            r = 1 - τ * h               # la2 = 2 ln|φᵢ| + 2 ln|1 - τh|
            la2 = 2 * log(abs(φ[i] * r))
            φ̄[i] += 2 * la2̄ / φ[i]
            τ̄ -= 2 * h * la2̄ / r
            h̄ -= 2 * τ * la2̄ / r
        end
        S += ℓ̄ * (la2 - log(Zt))
        Zt̄ = -ℓ̄ / Zt                   # Zt = Z - 2τvᵢ/φᵢ
        Z̄ += Zt̄
        τ̄ -= 2 * v[i] / φ[i] * Zt̄
        v̄[i] -= 2 * τ / φ[i] * Zt̄
        φ̄[i] += 2 * τ * v[i] / φ[i]^2 * Zt̄
        h̄ += τ̄ * τh                    # h = gᵢᵢ/φᵢ²
        ḡii[i] += h̄ / φ[i]^2
        φ̄[i] -= 2 * h̄ * gii[i] / φ[i]^3
    end

    # v = H⁻¹Gφ: Ḡφ = H⁻¹v̄ and H̄ = -(H⁻¹v̄)vᵀ, of which only the tridiagonal band matters.
    Ḡφ = F \ v̄
    H̄dv = -Ḡφ .* v
    H̄ev = T[-(Ḡφ[k] * v[k+1] + Ḡφ[k+1] * v[k]) for k in 1:n-1]
    _inv_band_adjoint!(H̄dv, H̄ev, H, gii, gnext, dpiv, δpiv, ḡii, ḡnext)
    # H = M + diag(w ⊘ φ²)
    dv̄ .+= H̄dv
    ēv = H̄ev
    @. φ̄ -= 2 * w * H̄dv / φ^3
    # Z and Gφ: tails φ²/(2κ), and per interval the coefficients (fdiag, fcross) = c(θ)/κ.
    t̄ = Z̄ * (φ[1]^2 + φ[n]^2) / 2 + (Ḡφ[1] * φ[1] + Ḡφ[n] * φ[n]) / 2   # adjoint of 1/κ
    φ̄[1] += (Z̄ * φ[1] + Ḡφ[1] / 2) / κ
    φ̄[n] += (Z̄ * φ[n] + Ḡφ[n] / 2) / κ
    κ̄ -= t̄ / κ^2
    for k in 1:n-1
        p, q = φ[k], φ[k+1]
        cd, cc = _mass_coeffs(θ[k])
        cd′, cc′ = _mass_coeffs_dθ(θ[k])
        fd, fc = cd / κ, cc / κ
        f̄d = Z̄ * (p^2 + q^2) + Ḡφ[k] * p + Ḡφ[k+1] * q
        f̄c = 2 * Z̄ * p * q + Ḡφ[k] * q + Ḡφ[k+1] * p
        φ̄[k]   += Z̄ * (2fd * p + 2fc * q) + Ḡφ[k] * fd + Ḡφ[k+1] * fc
        φ̄[k+1] += Z̄ * (2fd * q + 2fc * p) + Ḡφ[k+1] * fd + Ḡφ[k] * fc
        θ̄[k] += (f̄d * cd′ + f̄c * cc′) / κ
        κ̄ -= (f̄d * cd + f̄c * cc) / κ^2
    end
    # φ solves Mφ = w ⊘ φ, so its response to M is dφ = -H⁻¹(dM)φ and contributes -μᵀ(∂M/∂θₖ)φ
    # with μ = H⁻¹φ̄. That term is written as in `logdensity_node_gradient` (Bₖ), so that its two
    # O(1/θ²) parts do not cancel as θ → 0. The rest of M's adjoint (dv̄, ēv) enters through
    # ∂Mₖₖ/∂θₖ = ∂Mₖ₊₁,ₖ₊₁/∂θₖ = -csch²θₖ and ∂Mₖ,ₖ₊₁/∂θₖ = csch θₖ coth θₖ.
    μ = F \ φ̄
    for k in 1:n-1
        θk = θ[k]
        cs = csch(θk)
        Bk = (μ[k] * φ[k+1] + μ[k+1] * φ[k]) / (1 + cosh(θk)) - cs^2 * (μ[k+1] - μ[k]) * (φ[k+1] - φ[k])
        θ̄[k] += -Bk - cs^2 * (dv̄[k] + dv̄[k+1]) + cs * coth(θk) * ēv[k]
    end
    x̄ = zeros(T, n)
    for k in 1:n-1                      # θₖ = κ(xₖ₊₁ - xₖ)
        x̄[k+1] += κ * θ̄[k]
        x̄[k]   -= κ * θ̄[k]
        κ̄ += (x[k+1] - x[k]) * θ̄[k]
    end
    return S, x̄, κ̄
end

# The score S, g = ∂S/∂κ, ∂g/∂κ and ∇ₓg for `_klcv_adjoint`, from its reverse pass on dual
# numbers carrying d/dκ. The amplitude enters with its exact κ-tangent φ̇ = -H⁻¹(∂M/∂κ)φ.
function _klcv_kappa_derivatives(x::Vector{T}, w::Vector{T}, κ::T, locnode) where {T}
    M = _roughness_operator(x, κ, T(-Inf), T(Inf))
    φ = _solve_amplitude(M, w)
    H = SymTridiagonal(M.dv .+ w ./ φ.^2, copy(M.ev))
    mφ̇ = ldiv!(ldlt!(H), _dM_dκ_mul(x, κ, φ))  # -φ̇
    D(v, dv) = Dual{_KappaTag}(v, dv)
    S, x̄, κ̄ = _klcv_adjoint(D.(x, zero(T)), D.(w, zero(T)), D(κ, one(T)), D.(φ, .-mφ̇), locnode)
    return value(S), value(κ̄), partials(κ̄, 1), T[partials(xi, 1) for xi in x̄]
end

"""
    select_kappa_gradient(x, κ; rtol=cbrt(eps(T))) -> ∇κ

Gradient of the scale chosen by [`select_kappa`](@ref)`(x; rtol)` with respect to the
observations: `∇κ[j] = ∂κ̂/∂x[j]`, with the axes of `x`. `κ` must be the selected scale
`select_kappa(x; rtol)`, a minimum of the score [`klcv`](@ref); the gradient is that of the
stationary point, computed in `O(N)` by implicit differentiation,

    ∂κ̂/∂x = -(∂²KLCV/∂κ∂x) / (∂²KLCV/∂κ²).

The observations must be distinct, on the unbounded line (the default `support`), and scored
as exact points. A `κ` farther from the score's minimum than the search in `select_kappa`
leaves it throws an `ArgumentError`, as do tied observations, at which `κ̂` is not
differentiable.

To differentiate a quantity that depends on the data both directly and through `κ̂`, such as
`L(x) = Σⱼ ln Q̂(yⱼ)` for a fit at `κ̂(x)`, combine the partial derivatives at fixed `κ` with
this gradient: `dL/dx = ∂L/∂x + (∂L/∂κ) ∇κ`. [`logdensity_node_gradient`](@ref) and
[`logdensity_kappa_derivative`](@ref) give the two partial derivatives of `L`.

# Extended help

`κ̂` is a piecewise-smooth function of the data, and the gradient is that of the smooth piece
containing `x`. Its pieces are separated by changes in which observations merge into one node
(at a separation of `rtol/κ`) and, if the score has more than one local minimum, by jumps of
`κ̂` between them. Within a merged node only the first (smallest) observation sets the node's
position, so the others have zero gradient.

`select_kappa` locates the minimum by comparing values of the score, so rounding limits it to
a relative precision of roughly `√(N eps(T))` in `κ`. The gradient is evaluated at the
stationary point itself, which is refined from `κ` by Newton's method on `∂KLCV/∂κ`.

The score's derivatives with respect to the node positions and to `κ` come from a reverse
(adjoint) pass through the leave-one-out expansion, at the cost of a few tridiagonal solves.
The mixed second derivatives come from evaluating that pass on dual numbers that carry the
derivative with respect to `κ`.
"""
function select_kappa_gradient(x::AbstractVector{<:Real}, κ::Real; rtol::Real=cbrt(eps(float(eltype(x)))))
    xs, r, _, _ = _scoring_sample(x, rtol, (-Inf, Inf), false)
    T = eltype(xs)
    for i in firstindex(xs)+1:lastindex(xs)
        xs[i] == xs[i-1] && throw(ArgumentError(
            "select_kappa_gradient requires distinct observations; the value $(xs[i]) is repeated, " *
            "and the selected scale is not differentiable where observations coincide"))
    end
    κT = _score_scale(κ, T)
    perm = copyto!(Vector{Int}(undef, length(xs)), sortperm(x))   # one-based, as xs[i] == x[perm[i]]
    nodeof = similar(perm)
    nodes, w = _merge_presorted(xs, r / κT, nodeof)
    S, g, gκ, gx = _klcv_kappa_derivatives(nodes, w, κT, nodeof)
    # The search in `select_kappa` stops where differences of S are lost to rounding, which
    # grows with N; g²/(2 ∂g/∂κ) is the decrease of S that remains to its minimum.
    gκ > 0 && g^2 / (2gκ) <= _KAPPA_STATIONARITY * length(xs) * eps(T) * abs(S) ||
        throw(ArgumentError("κ = $κ is not a minimum of the KLCV score (∂S/∂κ = $g, ∂²S/∂κ² = $gκ); " *
                            "pass κ = select_kappa(x; rtol) with the same rtol"))
    # Newton's method on g, which is computed without that cancellation, refines κ to the
    # stationary point at the same merge pattern.
    κs, prevstep = κT, T(Inf)
    for iter in 1:_KAPPA_POLISH_ITERS
        step = g / gκ
        (abs(step) <= eps(T)^(3 // 4) * κs || abs(step) >= prevstep) && break
        iter == _KAPPA_POLISH_ITERS && throw(ConvergenceError("the Newton refinement of κ",
            "the step was still $step after $iter iterations"))
        κs -= step
        prevstep = abs(step)
        S, g, gκ, gx = _klcv_kappa_derivatives(nodes, w, κs, nodeof)
        gκ > 0 || throw(ArgumentError("the KLCV score is not convex near κ = $κ (∂²S/∂κ² = $gκ at κ = $κs)"))
    end
    ∇κ = similar(x, T)
    fill!(∇κ, zero(T))
    for i in eachindex(perm, nodeof)
        k = nodeof[i]
        if i == firstindex(nodeof) || nodeof[i-1] != k   # first member: it sets the node's position
            ∇κ[perm[i]] = -gx[k] / gκ
        end
    end
    return ∇κ
end

const _KAPPA_STATIONARITY = 64   # accepted g²/(2 ∂g/∂κ), in units of N eps(T) |S|
const _KAPPA_POLISH_ITERS = 8
