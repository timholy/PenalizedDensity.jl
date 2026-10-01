# Density estimation from interval observations (rounded, binned, or censored values),
# optionally mixed with point observations.
#
# Notation. The unnormalized amplitude u minimizes
#
#     F(u) = ½∫(u'²/κ(x)² + u²) dx - Σⱼ rⱼ ln Pⱼ(u) - Σᵢ wᵢ ln u(yᵢ)²,   Pⱼ(u) = ∫_{Bⱼ} u²,
#
# and ψ = u/√Z with Z = ∫u². Every finite interval endpoint and every point is a node
# y₁ < … < yₘ; the cells are the m-1 gaps between nodes plus the two tails, numbered
# 0 (left tail), 1, …, m-1 (interior), m (right tail). On a cell inside interval j the
# field equation is u'' = k²u with k² = κ²(1 - ωⱼ), ωⱼ = 2rⱼ/Pⱼ; on an empty cell k² = κ².
# Occupied cells usually have k² < 0 (trigonometric), so every cell formula below is written
# for both signs of k².
#
# Minimizing F over each cell's interior at fixed nodal values leaves a reduced objective
# Φ(s) of the nodal densities sᵢ = u(yᵢ)², which is convex; it is minimized by damped Newton
# in s. Each evaluation of Φ solves one scalar equation per interval for its ωⱼ. The Hessian is
# tridiagonal plus one rank-one term per interval spanning several cells, and is factored as a
# banded matrix.

# ---------------------------------------------------------------------------------------------
# Cell coefficients

# Power series of the entire functions of z = k²h² used by the cell coefficients:
#   c(z) = cosh√z, s(z) = sinh√z/√z, p(z) = (cs - 1)/z, q(z) = (c - s)/z,
# and the derivatives p'(z), q'(z). Accurate for |z| ≤ 1, where the closed forms cancel.
function _cell_series(z::T) where {T}
    c = s = p = q = dp = dq = zero(T)
    zj = one(T)             # z^j
    zjm = zero(T)           # z^(j-1), zero at j = 0
    f2 = one(T)             # (2j)!
    for j in 0:60
        f3 = f2 * (2j + 1)                  # (2j+1)!
        f5 = f3 * (2j + 2) * (2j + 3)       # (2j+3)!
        pj = 4^(j + 1) / f5
        qj = (2j + 2) / f5
        tc = zj / f2
        c += tc
        s += zj / f3
        p += pj * zj
        q += qj * zj
        dp += j * pj * zjm
        dq += j * qj * zjm
        abs(tc) <= eps(T) * abs(c) / 16 && j >= 2 && break
        zjm = zj
        zj *= z
        f2 = f3 * (2j + 2)
    end
    return c, s, p, q, dp, dq
end

"""
    _cell_coeffs(k2, h) -> (m11, m12, n11, n12, d11, d12)

Coefficients of one interior cell of length `h` on which `u'' = k2 * u`, as functions of the
endpoint values `v = (v₀, v₁)`:

- `κ²·[uu']₀ʰ = vᵀ [m11 m12; m12 m11] v` (the boundary flux, times κ²);
- `∫₀ʰ u² = vᵀ [n11 n12; n12 n11] v` (the cell mass);
- `[d11 d12; d12 d11]`, the derivative of the mass matrix with respect to `k2`.

The flux matrix's derivative with respect to `k2` is the mass matrix. `k2` may take either
sign; the cell is admissible for `k2 * h² > -π²`. Every entry stays finite for large positive
`k2 * h²`.
"""
function _cell_coeffs(k2::T, h::T) where {T}
    z = k2 * h^2
    if abs(z) <= 1
        c, s, p, q, dp, dq = _cell_series(z)
        invs, cos_s = 1 / s, c / s
        P2, Q2, qos = p / s^2, q / s^2, q / s
        dP2, dQ2 = dp / s^2, dq / s^2
    elseif z > 0
        # Written through coth and csch so nothing overflows as z → ∞.
        t = sqrt(z)
        ct, cs = coth(t), csch(t)
        invs, cos_s = t * cs, t * ct
        P2 = ct / t - cs^2
        Q2 = cs * (ct - 1 / t)
        qos = (t * ct - 1) / z
        dP2 = (1 + ct * (ct - 1 / t)) / (2z) - P2 / z
        dQ2 = (invs - 3Q2) / (2z)
    else
        z > -T(π)^2 || throw(DomainError(z, "cell has k²h² ≤ -π²: the interpolant is not positive"))
        t = sqrt(-z)
        c, s = cos(t), sin(t) / t
        p, q = (c * s - 1) / z, (c - s) / z
        dp = (s^2 / 2 + c * q / 2 - p) / z
        dq = (s - 3q) / (2z)
        invs, cos_s = 1 / s, c / s
        P2, Q2, qos = p / s^2, q / s^2, q / s
        dP2, dQ2 = dp / s^2, dq / s^2
    end
    return cos_s / h, -invs / h, h * P2 / 2, h * Q2 / 2,
           h^3 * (dP2 - P2 * qos) / 2, h^3 * (dQ2 - Q2 * qos) / 2
end

# The same for an unbounded tail with decay rate k = √k2 (k2 > 0): flux·κ² = k v², mass v²/(2k).
function _tail_coeffs(k2::T) where {T}
    k = sqrt(k2)
    return k, 1 / (2k), -1 / (4 * k^3)
end

# sinh(√z)/√z for z ≤ 0 (sin t / t) and small positive z.
function _sfun(z::T) where {T}
    abs(z) <= 1 && return _cell_series(z)[2]
    z > 0 && return sinh(sqrt(z)) / sqrt(z)
    t = sqrt(-z)
    return sin(t) / t
end

# S(k², x) = sinh(kx)/k, entire in k²; used where k² ≤ 0.
_Sfun(k2::T, x::T) where {T} = x * _sfun(k2 * x^2)

# ---------------------------------------------------------------------------------------------
# Problem layout

# A distinct interval with multiplicity r, covering cells `first:last` (0 and m are the tails).
struct _IntervalGroup{T}
    first::Int
    last::Int
    r::T
end

_group_nodes(g::_IntervalGroup, m::Int) = max(g.first, 1):min(g.last + 1, m)

# The observations laid out on the node geometry.
struct _IntervalLayout{T}
    y::Vector{T}                     # sorted distinct nodes
    w::Vector{T}                     # point multiplicity at each node
    groups::Vector{_IntervalGroup{T}}
    cellgroup::Vector{Int}           # cellgroup[c+1]: the group containing cell c, or 0
    κs::Vector{T}                    # scale on each interior cell
    κL::T
    κR::T
    lo::T
    hi::T
end

function _check_observation(a, b, lo, hi)
    (isnan(a) || isnan(b)) && throw(ArgumentError("observation bounds must not be NaN, got [$a, $b]"))
    a <= b || throw(ArgumentError("an interval needs lower ≤ upper, got [$a, $b]"))
    a == b && !isfinite(a) && throw(ArgumentError("a point observation must be finite, got $a"))
    (a >= lo && b <= hi) ||
        throw(DomainError((a, b), "observation [$a, $b] lies outside the support [$lo, $hi]"))
    (a == lo && b == hi) &&
        throw(ArgumentError("the interval [$a, $b] covers the whole support and carries no information"))
    return nothing
end

# Scale on each candidate node: one value, or the callable realized on the sorted candidates.
_candidate_kappa(κ::Real, cand::Vector{T}) where {T} = fill(T(κ), length(cand))
_candidate_kappa(κfun, cand::Vector{T}) where {T} = _kappa_sorted(κfun, cand, T)

_layout_kappa(κ::Real, y::Vector{T}) where {T} = (fill(T(κ), length(y) - 1), T(κ), T(κ))
_layout_kappa(κfun, y::Vector{T}) where {T} = _kappa_profile(y, κfun, T)

# `counts[i]`, when given, is the number of observations sharing the bounds `lower[i]`,
# `upper[i]` (see `_tally_intervals`).
function _interval_layout(lower::AbstractVector, upper::AbstractVector, κ, rtol::T, lo::T, hi::T;
                          counts::Union{AbstractVector,Nothing}=nothing) where {T}
    axes(lower) == axes(upper) ||
        throw(DimensionMismatch("lower and upper bounds must have the same axes, got $(axes(lower)) and $(axes(upper))"))
    counts === nothing || axes(counts) == axes(lower) ||
        throw(DimensionMismatch("counts must have the axes of the bounds, got $(axes(counts)) and $(axes(lower))"))
    isempty(lower) && throw(ArgumentError("cannot fit a density to zero observations"))
    for i in eachindex(lower, upper)
        _check_observation(T(lower[i]), T(upper[i]), lo, hi)
    end
    # Nodes: every finite bound, merged within a fraction `rtol` of the local smoothing length.
    # Each run of merged candidates is represented by its first member, so any candidate maps to
    # its node by `searchsortedlast`.
    cand = T[]
    for i in eachindex(lower, upper)
        a, b = T(lower[i]), T(upper[i])
        isfinite(a) && push!(cand, a)
        isfinite(b) && a != b && push!(cand, b)
    end
    sort!(cand)
    y, _ = _merge_presorted(cand, rtol, _candidate_kappa(κ, cand))
    m = length(y)
    w = zeros(T, m)
    keys = Tuple{Int,Int,T,T,T}[]    # (lower node or 0, upper node or m+1, lower, upper, count)
    for i in eachindex(lower, upper)
        a, b = T(lower[i]), T(upper[i])
        cnt = counts === nothing ? one(T) : T(counts[i])
        ia = isfinite(a) ? searchsortedlast(y, a) : 0
        ib = isfinite(b) ? searchsortedlast(y, b) : m + 1
        if ia == ib             # a point, or an interval narrower than the merge tolerance
            w[ia] += cnt
        else
            push!(keys, (ia, ib, a, b, cnt))
        end
    end
    sort!(keys; by=k -> (k[1], k[2]))
    groups = _IntervalGroup{T}[]
    reps = Tuple{T,T}[]
    for (ia, ib, a, b, cnt) in keys
        if !isempty(groups) && groups[end].first == ia && groups[end].last == ib - 1
            g = groups[end]
            groups[end] = _IntervalGroup{T}(g.first, g.last, g.r + cnt)
        else
            if !isempty(groups) && groups[end].last >= ia
                a0, b0 = reps[end]
                throw(ArgumentError("intervals [$a0, $b0] and [$a, $b] overlap; overlapping intervals are not supported"))
            end
            push!(groups, _IntervalGroup{T}(ia, ib - 1, cnt))
            push!(reps, (a, b))
        end
    end
    cellgroup = zeros(Int, m + 1)
    for (j, g) in pairs(groups)
        cellgroup[g.first+1:g.last+1] .= j
    end
    κs, κL, κR = _layout_kappa(κ, y)
    _check_resolvable(y, κs, cellgroup, reps)
    return _IntervalLayout{T}(y, w, groups, cellgroup, κs, κL, κR, lo, hi)
end

# When the smoothing length 1/κ is far below an interval's width, the coupling between
# neighboring intervals (flux terms of order 1/(κ²h)) falls below the roundoff in the
# log-probability terms and the nodal solve loses accuracy; by then the estimate has long since
# reached its limit as κ → ∞, the histogram on the recorded intervals.
function _check_resolvable(y::Vector{T}, κs::Vector{T}, cellgroup::Vector{Int}, reps) where {T}
    limit = eps(T)^(-1 // 4)
    for c in eachindex(κs)
        j = cellgroup[c+1]
        j == 0 && continue
        θ = κs[c] * (y[c+1] - y[c])
        θ <= limit || throw(ArgumentError(
            "κ·width = $θ on the interval $(reps[j]) exceeds $limit: the smoothing length is too " *
            "far below the interval width to fit accurately, and the estimate is already at its " *
            "histogram limit; use a smaller κ"))
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Reduced objective

# Flux, mass, and mass-derivative coefficients of cell c at k², with the cell's scale κ.
# Interior cells return 2×2 symmetric forms; tail cells a scalar form in the outer node
# (with its value repeated so the two cases share one signature).
function _cell_forms(L::_IntervalLayout{T}, c::Int, k2::T) where {T}
    m = length(L.y)
    if 1 <= c <= m - 1
        return _cell_coeffs(k2, L.y[c+1] - L.y[c])
    end
    k, n, d = _tail_coeffs(k2)
    return k, zero(T), n, zero(T), d, zero(T)
end

_cell_kappa(L::_IntervalLayout, c::Int) = c == 0 ? L.κL : c == length(L.y) ? L.κR : L.κs[c]

# The group's potential is carried as τ = 1 - ω, so that k² = κ²τ has full relative precision
# even when ω ≈ 1 (the regime of a smoothing length far below the interval width).
# Smallest admissible τ for a group: interior cells need k²h² > -π², unbounded tails k² > 0.
function _tau_min(L::_IntervalLayout{T}, g::_IntervalGroup{T}) where {T}
    m = length(L.y)
    τmin = T(-Inf)
    for c in g.first:g.last
        if 1 <= c <= m - 1
            τmin = max(τmin, -(T(π) / (L.κs[c] * (L.y[c+1] - L.y[c])))^2)
        else
            τmin = max(τmin, zero(T))
        end
    end
    return τmin
end

# vᵀ·form·v for cell c (interior: nodes c, c+1; left tail: node 1; right tail: node m).
function _cell_quad(L::_IntervalLayout, v, c::Int, a, b)
    m = length(L.y)
    c == 0 && return a * v[1]^2
    c == m && return a * v[m]^2
    return a * (v[c]^2 + v[c+1]^2) + 2b * v[c] * v[c+1]
end

# Group mass Q(τ) = Σ_c vᵀN_c v and dQ/dτ (negative: the mass falls as k² rises).
function _group_mass(L::_IntervalLayout{T}, g::_IntervalGroup{T}, v, τ::T) where {T}
    Q = dQ = zero(T)
    for c in g.first:g.last
        κ = _cell_kappa(L, c)
        _, _, n11, n12, d11, d12 = _cell_forms(L, c, κ^2 * τ)
        Q += _cell_quad(L, v, c, n11, n12)
        dQ += κ^2 * _cell_quad(L, v, c, d11, d12)
    end
    return Q, dQ
end

# The group's τ = 1 - ω: the root of (1 - τ)·Q(τ) = 2r on (τmin, 1), unique because the left
# side decreases from ∞ to 0 there. Safeguarded Newton from `τ0` (NaN: no warm start).
function _solve_tau(L::_IntervalLayout{T}, g::_IntervalGroup{T}, v, τ0::T) where {T}
    Q1, _ = _group_mass(L, g, v, one(T))
    lo, hi = max(_tau_min(L, g), 1 - 2g.r / Q1), one(T)   # Q falls with τ, so 1 - τ ≤ 2r/Q(1)
    τ = isfinite(τ0) && lo < τ0 < hi ? τ0 : (isfinite(lo) ? (lo + hi) / 2 : zero(T))
    for _ in 1:200
        Q, dQ = _group_mass(L, g, v, τ)
        r = (1 - τ) * Q - 2g.r
        r == 0 && return τ
        r > 0 ? (lo = τ) : (hi = τ)
        τn = τ - r / ((1 - τ) * dQ - Q)
        lo < τn < hi || (τn = (lo + hi) / 2)
        (τn == τ || abs(τn - τ) <= 4 * eps(T) * abs(τ)) && return τn
        τ = τn
    end
    error("interval potential did not converge; please report this")
end

# Workspace and results of one evaluation of Φ.
mutable struct _PhiEval{T}
    Φ::T
    scale::T                # magnitude of the terms summed into Φ (for roundoff tests)
    Z::T                    # ∫u²
    gs::Vector{T}           # ∇ₛΦ
    band::Matrix{T}         # ∇²ₛΦ, lower band: band[1+i-j, j] = H[i, j]
    τ::Vector{T}            # group values of τ = 1 - ω
end

function _PhiEval(L::_IntervalLayout{T}, bw::Int) where {T}
    m = length(L.y)
    return _PhiEval{T}(zero(T), zero(T), zero(T), zeros(T, m), zeros(T, bw + 1, m),
                       fill(T(NaN), length(L.groups)))
end

# Bandwidth of ∇²ₛΦ: tridiagonal, widened by intervals spanning several cells.
function _bandwidth(L::_IntervalLayout)
    m = length(L.y)
    bw = min(1, m - 1)
    for g in L.groups
        bw = max(bw, length(_group_nodes(g, m)) - 1)
    end
    return bw
end

# Add a cell's flux form (divided by κ²) to the objective, the v-gradient, and the v-Hessian.
function _add_flux!(E::_PhiEval{T}, L, v, gv, hd, he, c::Int, a::T, b::T) where {T}
    m = length(L.y)
    if c == 0 || c == m
        i = c == 0 ? 1 : m
        E.Φ += a * v[i]^2 / 2
        E.scale += abs(a) * v[i]^2 / 2
        gv[i] += a * v[i]
        hd[i] += a
    else
        E.Φ += (a * (v[c]^2 + v[c+1]^2) + 2b * v[c] * v[c+1]) / 2
        E.scale += abs(a) * (v[c]^2 + v[c+1]^2) / 2
        gv[c] += a * v[c] + b * v[c+1]
        gv[c+1] += b * v[c] + a * v[c+1]
        hd[c] += a
        hd[c+1] += a
        he[c] += b
    end
    return E
end

# Evaluate Φ(s), its gradient and banded Hessian in s, and Z. Group values in E.τ seed
# the scalar solves and are overwritten with the new roots.
function _eval_phi!(E::_PhiEval{T}, L::_IntervalLayout{T}, s::Vector{T}) where {T}
    y = L.y
    m = length(y)
    v = sqrt.(s)
    gv = zeros(T, m); hd = zeros(T, m); he = zeros(T, max(m - 1, 0))
    E.Φ = E.scale = E.Z = zero(T)
    # Empty cells.
    for c in 0:m
        L.cellgroup[c+1] == 0 || continue
        if c == 0 || c == m
            i, κ = c == 0 ? (1, L.κL) : (m, L.κR)
            Δ = c == 0 ? y[1] - L.lo : L.hi - y[m]
            _add_flux!(E, L, v, gv, hd, he, c, _tail_diag(κ, Δ) / κ, zero(T))
            E.Z += _tail_mass(v[i], κ, Δ)
        else
            κ = L.κs[c]
            m11, m12, n11, n12, _, _ = _cell_coeffs(κ^2, y[c+1] - y[c])
            _add_flux!(E, L, v, gv, hd, he, c, m11 / κ^2, m12 / κ^2)
            E.Z += _cell_quad(L, v, c, n11, n12)
        end
    end
    fill!(E.band, zero(T))
    # Intervals.
    for (j, g) in pairs(L.groups)
        τ = _solve_tau(L, g, v, E.τ[j])
        E.τ[j] = τ
        nodes = _group_nodes(g, m)
        nvec = zeros(T, length(nodes))
        P = dQ = zero(T)
        for c in g.first:g.last
            κ = _cell_kappa(L, c)
            m11, m12, n11, n12, d11, d12 = _cell_forms(L, c, κ^2 * τ)
            _add_flux!(E, L, v, gv, hd, he, c, m11 / κ^2, m12 / κ^2)
            P += _cell_quad(L, v, c, n11, n12)
            dQ += κ^2 * _cell_quad(L, v, c, d11, d12)
            if c == 0
                nvec[1] += n11 * v[1]
            elseif c == m
                nvec[end] += n11 * v[m]
            else
                i = c - first(nodes) + 1
                nvec[i] += n11 * v[c] + n12 * v[c+1]
                nvec[i+1] += n12 * v[c] + n11 * v[c+1]
            end
        end
        E.Z += P
        E.Φ += g.r - g.r * log(P)
        E.scale += g.r * (1 + abs(log(P)))
        D = 1 - 2g.r * dQ / P^2
        γ = 4g.r / (P^2 * D)
        # Rank-one term in s: γ (n ./ 2v)(n ./ 2v)ᵀ, lower band only.
        for (jj, nj) in pairs(nodes), (ii, ni) in pairs(nodes)
            ni >= nj || continue
            E.band[1+ni-nj, nj] += γ * (nvec[ii] / (2v[ni])) * (nvec[jj] / (2v[nj]))
        end
    end
    # Points.
    for i in 1:m
        L.w[i] == 0 && continue
        E.Φ -= L.w[i] * log(s[i])
        E.scale += L.w[i] * abs(log(s[i]))
        gv[i] -= 2L.w[i] / v[i]
        hd[i] += 2L.w[i] / v[i]^2
    end
    # Change of variables v = √s.
    for i in 1:m
        E.gs[i] = gv[i] / (2v[i])
        E.band[1, i] += hd[i] / (4 * s[i]) - gv[i] / (4 * v[i] * s[i])
    end
    for i in 1:m-1
        E.band[2, i] += he[i] / (4 * v[i] * v[i+1])
    end
    return E
end

# Solve H x = b in place for the SPD banded H stored as its lower band (band[1+i-j, j]),
# overwriting the band with its Cholesky factor.
_banded_cholesky_solve!(band::Matrix{T}, b::Vector{T}) where {T} =
    _banded_solve!(_banded_cholesky!(band), b)

# Overwrite the lower band of an SPD banded matrix with its Cholesky factor L (H = LLᵀ).
function _banded_cholesky!(band::Matrix{T}) where {T}
    bw = size(band, 1) - 1
    m = size(band, 2)
    for j in 1:m
        d = band[1, j]
        for k in max(1, j - bw):j-1
            d -= band[1+j-k, k]^2
        end
        d > 0 || throw(LinearAlgebra.PosDefException(j))
        d = sqrt(d)
        band[1, j] = d
        for i in j+1:min(m, j + bw)
            t = band[1+i-j, j]
            for k in max(1, i - bw):j-1
                t -= band[1+i-k, k] * band[1+j-k, k]
            end
            band[1+i-j, j] = t / d
        end
    end
    return band
end

# Solve LLᵀx = b in place, given the banded Cholesky factor from `_banded_cholesky!`.
function _banded_solve!(band::Matrix{T}, b::Vector{T}) where {T}
    bw = size(band, 1) - 1
    m = size(band, 2)
    for j in 1:m                    # forward: L z = b
        t = b[j]
        for k in max(1, j - bw):j-1
            t -= band[1+j-k, k] * b[k]
        end
        b[j] = t / band[1, j]
    end
    for j in m:-1:1                 # backward: Lᵀ x = z
        t = b[j]
        for i in j+1:min(m, j + bw)
            t -= band[1+i-j, j] * b[i]
        end
        b[j] = t / band[1, j]
    end
    return b
end

# The entries of H⁻¹ within the band of H, in the same lower-band layout, from the banded Cholesky
# factor L. Since LᵀH⁻¹ = L⁻¹ is lower triangular with diagonal 1/Lⱼⱼ, the band of H⁻¹ fills in
# from the last column backwards (Takahashi's recurrence), in O(m·bw²).
function _banded_inverse_band(chol::Matrix{T}) where {T}
    bw = size(chol, 1) - 1
    m = size(chol, 2)
    K = zeros(T, bw + 1, m)
    Kij(i, j) = i >= j ? K[1+i-j, j] : K[1+j-i, i]
    for j in m:-1:1
        kmax = min(m, j + bw)
        for i in kmax:-1:j+1
            t = zero(T)
            for k in j+1:kmax
                t += chol[1+k-j, j] * Kij(k, i)
            end
            K[1+i-j, j] = -t / chol[1, j]
        end
        t = zero(T)
        for k in j+1:kmax
            t += chol[1+k-j, j] * K[1+k-j, j]
        end
        K[1, j] = (1 / chol[1, j] - t) / chol[1, j]
    end
    return K
end

# The dense block of H⁻¹ on the contiguous index range `r`, from the band `K` of H⁻¹ and the
# Cholesky factor L. Entries beyond the band follow from (LᵀH⁻¹)ᵢⱼ = 0 for i < j, which gives
# (H⁻¹)ᵢⱼ through (H⁻¹)ₖⱼ for k = i+1, …, i+bw: filling each column upward, those lie either
# lower in the same column of the block or within the band.
function _inverse_block(chol::Matrix{T}, K::Matrix{T}, r::UnitRange{Int}) where {T}
    bw = size(K, 1) - 1
    m = size(K, 2)
    B = zeros(T, length(r), length(r))
    f = first(r) - 1
    for j in r
        for i in j:-1:first(r)
            if j - i <= bw
                B[i-f, j-f] = K[1+j-i, i]
                continue
            end
            t = zero(T)
            for k in i+1:min(m, i + bw)
                t += chol[1+k-i, i] * (k <= j ? B[k-f, j-f] : K[1+k-j, j])
            end
            B[i-f, j-f] = -t / chol[1, i]
        end
    end
    return Matrix(Symmetric(B, :U))
end

# Starting nodal densities: a point fit to one representative location per observation, at a
# scale no finer than the typical interval width, evaluated at the nodes.
function _interval_start(L::_IntervalLayout{T}) where {T}
    y = L.y
    m = length(y)
    xs = T[]
    widths = T[]
    for i in 1:m
        for _ in 1:round(Int, L.w[i])
            push!(xs, y[i])
        end
    end
    for g in L.groups
        nodes = _group_nodes(g, m)
        a = g.first == 0 ? y[1] - 1 / L.κL : y[first(nodes)]
        b = g.last == m ? y[m] + 1 / L.κR : y[last(nodes)]
        g.first > 0 && g.last < m && push!(widths, b - a)
        for _ in 1:round(Int, g.r)
            push!(xs, (a + b) / 2)
        end
    end
    κ̄ = _reference_scale(L.κs, L.κL, L.κR)
    κ0 = isempty(widths) ? κ̄ : min(κ̄, 1 / Statistics.median(widths))
    lo = min(L.lo, minimum(xs))
    hi = max(L.hi, maximum(xs))
    d0 = DensityEstimate(xs, κ0; support=(lo, hi))
    s = d0.λ .* d0.(y)
    smax = maximum(s)
    return max.(s, eps(T) * smax)
end

# A scale at which the nodal densities cannot be represented: some density has underflowed so far
# that the gradient or Hessian of Φ in s overflows. Selectors treat such a scale as unresolvable.
struct _UnresolvableScale <: Exception
    msg::String
end
Base.showerror(io::IO, e::_UnresolvableScale) = print(io, e.msg)

# Run `f()`, returning `fallback` if the scale it fits is unresolvable.
function _unless_unresolvable(f, fallback)
    try
        return f()
    catch err
        err isa _UnresolvableScale || rethrow()
        return fallback
    end
end

# Throw `_UnresolvableScale` unless the evaluation `E` at `s` is finite. A density far below a
# neighboring one (an interval edge far from the points inside it, at large κ) makes the terms of
# order 1/s in the s-Hessian overflow, and Newton steps built from it are meaningless.
function _check_finite_eval(E::_PhiEval, L::_IntervalLayout, s)
    all(isfinite, E.gs) && all(isfinite, E.band) && return nothing
    i = argmin(s)
    throw(_UnresolvableScale(
        "the fitted density at x = $(L.y[i]) underflows to $(s[i]) relative to the largest nodal " *
        "density $(maximum(s)), beyond what the solve can represent; use a smaller κ"))
end

"""
    _solve_interval(L; maxiter=200, stats=nothing) -> (s, E)

Minimize the reduced objective Φ(s) over positive nodal densities by damped Newton with a
banded Hessian and Armijo backtracking, stopping when the Newton correction reaches
`eps(T)^(3/4)` relative to `s` or the floor roundoff imposes on it (the same criteria as
`_solve_amplitude`). Returns the minimizer and the evaluation at it. Throws `_UnresolvableScale`
if an accepted iterate has a non-finite gradient or Hessian.
"""
function _solve_interval(L::_IntervalLayout{T}; maxiter::Int=200, stats=nothing) where {T}
    m = length(L.y)
    bw = _bandwidth(L)
    s = _interval_start(L)
    E = _eval_phi!(_PhiEval(L, bw), L, s)
    _check_finite_eval(E, L, s)
    Et = _PhiEval(L, bw)
    Δ = similar(s); snew = similar(s)
    stol = eps(T)^(3 // 4)
    prevstep = T(Inf)
    unguarded = false
    iters = 0; backs = 0; reason = :none
    finalstep = T(Inf); finalα = T(NaN)
    converged = false
    for _ in 1:maxiter
        Δ .= E.gs
        _banded_cholesky_solve!(E.band, Δ)          # Δ = H⁻¹∇Φ; the step is -Δ
        decrement = dot(E.gs, Δ)
        step = maximum(i -> abs(Δ[i]) / s[i], eachindex(s, Δ))
        if step <= stol
            converged = true; reason = :tolerance; finalstep = step
            break
        elseif unguarded && step >= prevstep
            converged = true; reason = :floor; finalstep = step
            break
        end
        iters += 1
        prevstep = step
        α = one(T)
        for i in eachindex(s, Δ)
            Δ[i] > 0 && (α = min(α, s[i] / Δ[i]))
        end
        α < 1 && (α *= T(0.99))
        Et.τ .= E.τ
        if α * decrement / 4 <= 4 * sqrt(T(m)) * eps(T) * E.scale
            unguarded = true
            @. s -= α * Δ
            E, Et = _eval_phi!(Et, L, s), E
            _check_finite_eval(E, L, s)
            finalα = α
            continue
        end
        armijo = false
        while α * step >= eps(T)
            @. snew = s - α * Δ
            _eval_phi!(Et, L, snew)
            if Et.Φ <= E.Φ - α * decrement / 4
                armijo = true
                break
            end
            α /= 2
            backs += 1
            Et.τ .= E.τ
        end
        if !armijo
            converged = true; reason = :steplength; finalstep = step
            break
        end
        copyto!(s, snew)
        E, Et = Et, E
        _check_finite_eval(E, L, s)
        finalα = α
    end
    converged || error("Newton did not converge in $maxiter iterations; the fit is unreliable")
    if stats !== nothing
        stats.iterations = iters
        stats.backtracks = backs
        stats.reason = reason
        stats.final_step = Float64(finalstep)
        stats.final_alpha = Float64(finalα)
    end
    # The band holds a factorization; re-evaluate so E describes s exactly.
    _eval_phi!(E, L, s)
    _check_finite_eval(E, L, s)
    return s, E
end

# ---------------------------------------------------------------------------------------------
# Deletion scores

# The reduced term φ of group `g` carrying multiplicity `r`, as a function of the amplitudes `vl`
# at its nodes (`_group_nodes(g, m)`, the full-length `v` holding them): its value, the group
# mass P, the root τ, the gradient, the Hessian `SymTridiagonal(hd, he) + γ·n nᵀ`, and the mass
# gradient ∇ᵥP = 2n/D in those amplitudes. With r = 0 the group's cells are empty (τ = 1, γ = 0)
# and φ is their flux term alone.
function _group_local(L::_IntervalLayout{T}, g::_IntervalGroup{T}, r::T, v::Vector{T}, τ0::T) where {T}
    m = length(L.y)
    nodes = _group_nodes(g, m)
    k = length(nodes)
    f = first(nodes)
    τ = r > 0 ? _solve_tau(L, _IntervalGroup{T}(g.first, g.last, r), v, τ0) : one(T)
    φ = P = dQ = scale = zero(T)
    gv = zeros(T, k); hd = zeros(T, k); he = zeros(T, max(k - 1, 0)); nvec = zeros(T, k)
    for c in g.first:g.last
        κ = _cell_kappa(L, c)
        m11, m12, n11, n12, d11, d12 = _cell_forms(L, c, κ^2 * τ)
        a, b = m11 / κ^2, m12 / κ^2
        P += _cell_quad(L, v, c, n11, n12)
        dQ += κ^2 * _cell_quad(L, v, c, d11, d12)
        if c == 0 || c == m
            i = c == 0 ? 1 : m
            ii = i - f + 1
            φ += a * v[i]^2 / 2
            scale += abs(a) * v[i]^2 / 2
            gv[ii] += a * v[i]
            hd[ii] += a
            nvec[ii] += n11 * v[i]
        else
            i = c - f + 1
            φ += (a * (v[c]^2 + v[c+1]^2) + 2b * v[c] * v[c+1]) / 2
            scale += (abs(a) * (v[c]^2 + v[c+1]^2) + 2abs(b * v[c] * v[c+1])) / 2
            gv[i] += a * v[c] + b * v[c+1]
            gv[i+1] += b * v[c] + a * v[c+1]
            hd[i] += a; hd[i+1] += a
            he[i] += b
            nvec[i] += n11 * v[c] + n12 * v[c+1]
            nvec[i+1] += n12 * v[c] + n11 * v[c+1]
        end
    end
    D = one(T)
    γ = zero(T)
    if r > 0
        φ += r - r * log(P)
        scale += r * (1 + abs(log(P)))
        D = 1 - 2r * dQ / P^2
        γ = 4r / (P^2 * D)
    end
    return (; φ, scale, P, τ, gv, hd, he, γ, nvec, gP = 2 .* nvec ./ D)
end

# The symmetric matrix  SymTridiagonal(d, e) + Σ_q γ[q]·n[q] n[q]ᵀ,  each n[q] supported on the
# index range rq[q]: the Hessian, in the nodal amplitudes, of the exact terms of a local deletion
# problem. Cell flux forms and point terms are tridiagonal; each occupied group adds one rank-one
# term through its mass.
struct _LocalHessian{T}
    d::Vector{T}
    e::Vector{T}
    γ::Vector{T}
    rq::Vector{UnitRange{Int}}
    n::Vector{Vector{T}}
end

_plus_tri(H::_LocalHessian, S::SymTridiagonal) = _LocalHessian(H.d .+ S.dv, H.e .+ S.ev, H.γ, H.rq, H.n)

function _dense(H::_LocalHessian{T}) where {T}
    A = Matrix{T}(SymTridiagonal(H.d, H.e))
    for q in eachindex(H.γ, H.rq, H.n)
        r = H.rq[q]
        A[r, r] .+= H.γ[q] .* H.n[q] .* H.n[q]'
    end
    return A
end

# The diagonal and first superdiagonal of H, rank-one terms included.
function _tri_entries(H::_LocalHessian)
    d, e = copy(H.d), copy(H.e)
    for q in eachindex(H.γ, H.rq, H.n)
        r, n, γ = H.rq[q], H.n[q], H.γ[q]
        for (p, i) in pairs(r)
            d[i] += γ * n[p]^2
            p < length(r) && (e[i] += γ * n[p] * n[p+1])
        end
    end
    return d, e
end

function _mul(H::_LocalHessian, x::AbstractVector)
    y = SymTridiagonal(H.d, H.e) * x
    for q in eachindex(H.γ, H.rq, H.n)
        r = H.rq[q]
        y[r] .+= (H.γ[q] * dot(H.n[q], view(x, r))) .* H.n[q]
    end
    return y
end

# H \ b by the Woodbury identity on the tridiagonal part, in O(k) per rank-one term. Returns
# `nothing` unless the tridiagonal part is positive definite (its LDLᵀ pivots all positive);
# the rank-one terms are positive semidefinite, so H is then positive definite too.
function _solve_pd(H::_LocalHessian{T}, b::AbstractVector) where {T}
    k = length(H.d)
    piv = similar(H.d); l = similar(H.e)        # LDLᵀ of the tridiagonal part
    piv[1] = H.d[1]
    piv[1] > 0 || return nothing
    for i in 2:k
        l[i-1] = H.e[i-1] / piv[i-1]
        piv[i] = H.d[i] - l[i-1] * H.e[i-1]
        piv[i] > 0 || return nothing
    end
    function tsolve(rhs)
        z = collect(T, rhs)
        for i in 2:k
            z[i] -= l[i-1] * z[i-1]
        end
        z ./= piv
        for i in k-1:-1:1
            z[i] -= l[i] * z[i+1]
        end
        return z
    end
    x = tsolve(b)
    qs = [q for q in eachindex(H.γ) if H.γ[q] != 0]
    isempty(qs) && return x
    p = length(qs)
    W = Matrix{T}(undef, k, p)                  # T⁻¹U
    C = zeros(T, p, p)                          # Γ⁻¹ + UᵀT⁻¹U
    for (c, q) in pairs(qs)
        u = zeros(T, k)
        u[H.rq[q]] .= H.n[q]
        W[:, c] = tsolve(u)
        C[c, c] += 1 / H.γ[q]
    end
    Utx = zeros(T, p)
    for (c, q) in pairs(qs)
        r = H.rq[q]
        Utx[c] = dot(H.n[q], view(x, r))
        for (c2, q2) in pairs(qs)
            C[c2, c] += dot(H.n[q2], view(W, H.rq[q2], c))
        end
    end
    return x .- W * (cholesky(Symmetric(C)) \ Utx)
end

# ∇ᵥZ, the gradient of the unnormalized mass in the nodal amplitudes, given each group's ∇ᵥP.
function _grad_Z(L::_IntervalLayout{T}, v::Vector{T}, gPs) where {T}
    y = L.y
    m = length(y)
    gZ = zeros(T, m)
    for c in 0:m
        L.cellgroup[c+1] == 0 || continue
        if c == 0 || c == m
            i, κ = c == 0 ? (1, L.κL) : (m, L.κR)
            Δ = c == 0 ? y[1] - L.lo : L.hi - y[m]
            gZ[i] += 2 * _tail_mass(v[i], κ, Δ) / v[i]
        else
            _, _, n11, n12, _, _ = _cell_coeffs(L.κs[c]^2, y[c+1] - y[c])
            gZ[c] += 2 * (n11 * v[c] + n12 * v[c+1])
            gZ[c+1] += 2 * (n12 * v[c] + n11 * v[c+1])
        end
    end
    for (g, gP) in zip(L.groups, gPs)
        gZ[_group_nodes(g, m)] .+= gP
    end
    return gZ
end

"""
    _interval_loo(L; holdout=:location) -> (lp, ld)

Held-out log-probabilities for interval data: `lp[j]` is `ln(P₋/Z₋)` for group `j` refitted
with one of its observations deleted, and `ld[i]` the held-out log-density at node `i`
(`NaN` where the node carries no point observation), with `holdout` choosing whether one point
or the whole location is deleted, as for `_loo_density`. Entries are `NaN` where the
deleted fit collapses (a non-positive held-out probability, density, or normalization).

Both come from the full fit's Hessian without refitting. Each deletion is minimized over a
local set of nodal amplitudes: those of the group losing the observation (or, for a point, of
the groups occupying the cells beside it), widened by the groups occupying the adjacent cells.
The terms of those groups and any point terms on those nodes are kept exact, with the
multiplicity or point weight reduced; the rest of the problem enters through its quadratic
model in the amplitudes, reduced onto the local nodes by the Schur complement `(K_ll)⁻¹ - A`
(`K` the inverse Hessian in amplitudes, `A` the Hessian of the exact terms). The remaining
nodes follow the quadratic model, which carries the normalization to first order. The exact
terms matter where the smoothing length is well below the interval width: deleting an
observation then reshapes the field across its own and the neighboring intervals. The
quadratic model is taken in amplitudes because empty cells are exactly quadratic there. A
point bordered only by empty cells uses the nonlinear step of the point-data score. Where a
deletion leaves a stretch with no observations (a singleton interval, or a location deleted
whole, bordered only by empty cells), the held-out value comes from the source-free
stationarity there in the log domain, as in `_loo_terms`, so that it keeps its relative accuracy
when it is of order e^{-2κ·gap}. Cost is `O(m·bw²)` for the band of the inverse plus a few
small solves per group and per point beside an occupied cell.
"""
function _interval_loo(L::_IntervalLayout{T}; holdout::Symbol=:location) where {T}
    holdout in (:location, :observation) ||
        throw(ArgumentError("holdout must be :location or :observation, got :$holdout"))
    m = length(L.y)
    s, E = _solve_interval(L)
    Z = E.Z
    v = sqrt.(s)
    chol = _banded_cholesky!(copy(E.band))
    dl = chol[1, :]
    dr = _reverse_cholesky_diag(E.band)
    Ks = _banded_inverse_band(chol)             # band of the inverse Hessian in s
    Kv(i, j) = (i >= j ? Ks[1+i-j, j] : Ks[1+j-i, i]) / (4 * v[i] * v[j])   # ∇²ᵥΦ = 4V(∇²ₛΦ)V
    Kvblock(r) = _inverse_block(chol, Ks, r) ./ (4 .* v[r] .* v[r]')
    locals = [_group_local(L, g, g.r, v, E.τ[j]) for (j, g) in pairs(L.groups)]
    gZ = _grad_Z(L, v, (loc.gP for loc in locals))
    KgZ = _banded_solve!(chol, gZ ./ (2 .* v)) ./ (2 .* v)                  # K_v ∇ᵥZ
    # With (K_ll)⁻¹ = S + H, the first-order change in Z is (K∇Z)ₗᵀ(K_ll)⁻¹δ - ∇Pᵀδ.
    dZ(nodes, loc, Kinv, δ) = dot(KgZ[nodes], _mul(Kinv, δ)) - dot(loc.gP, δ)
    lp = fill(T(NaN), length(L.groups))
    for (j, g) in pairs(L.groups)
        js = _with_neighbors(L, [j])            # js[1] == j
        gs = L.groups[js]
        nodes = _span_nodes(gs, m)
        pw = L.w[nodes]
        rs = [gq.r for gq in gs]
        loc = _local_terms(L, gs, rs, E.τ[js], nodes, pw, v)
        S = _rest_hessian(E.band, dl, dr, v, nodes, loc.H)
        Kinv = _plus_tri(loc.H, S)              # (K_ll)⁻¹
        rs[1] -= 1
        vl, new, resolved = _delete_local(L, gs, rs, pw, v, nodes, loc, S, E.τ[js])
        δ = vl .- v[nodes]
        # The group's mass needs at least one of its own amplitudes resolved.
        own = _group_nodes(g, m) .- (first(nodes) - 1)
        lP = any(resolved[own]) ? log(new.Ps[1]) : T(NaN)
        Pm = new.P
        if _empties_isolated(L, g)
            lP, δ = _emptied_group(L, g, v, nodes, _mul(Kinv, δ), Kvblock)
            Pm = exp(lP)
        end
        Zm = Z - loc.P + Pm + dZ(nodes, loc, Kinv, δ)
        isfinite(lP) && Zm > 0 && (lp[j] = lP - log(Zm))
    end
    ld = fill(T(NaN), m)
    for i in 1:m
        w = L.w[i]
        w == 0 && continue
        h = 2 * Kv(i, i) / v[i]^2               # leverage, in the convention of `_loo_density`
        c = holdout === :location ? w : one(T)
        τ = _deletion_step(h, w, c)
        Zm = Z - 2 * τ * KgZ[i] / v[i]
        touching = unique!(filter(!=(0), [L.cellgroup[i], L.cellgroup[i+1]]))
        if !isempty(touching)
            # Inside or beside an occupied interval: deleting the point reshapes that interval's
            # field, so its term (and its neighbors') is kept exact alongside the point's, as for
            # a group deletion.
            js = _with_neighbors(L, touching)
            gs = L.groups[js]
            nodes = _span_nodes(gs, m)
            pw = L.w[nodes]
            rs = [g.r for g in gs]
            loc = _local_terms(L, gs, rs, E.τ[js], nodes, pw, v)
            pw[i-first(nodes)+1] -= c
            S = _rest_hessian(E.band, dl, dr, v, nodes, loc.H)
            vl, new, resolved = _delete_local(L, gs, rs, pw, v, nodes, loc, S, E.τ[js])
            Zm = Z - loc.P + new.P + dZ(nodes, loc, _plus_tri(loc.H, S), vl .- v[nodes])
            p = i - first(nodes) + 1
            lv2 = resolved[p] ? 2 * log(vl[p]) : T(NaN)
        elseif c == w
            # Source-free after deletion: the neighbor sum of `_loo_terms`, over the empty cells.
            a = zero(T)
            l = T(-Inf)
            for (cell, nb) in ((i - 1, i - 1), (i, i + 1))
                acell, lb = _empty_flux_log(L, cell)
                a += acell
                1 <= cell <= m - 1 || continue
                vnb = v[nb] - 2τ * Kv(nb, i) / v[i]
                l = vnb > 0 ? logaddexp(l, lb + log(vnb)) : T(NaN)
            end
            lv2 = 2 * (l - log(a))
        else
            lv2 = log((v[i] * (1 - τ * h))^2)
        end
        isfinite(lv2) && Zm > 0 && (ld[i] = lv2 - log(Zm))
    end
    return lp, ld
end

# The flux form of empty cell c divided by κ², as `_eval_phi!` adds it: its diagonal entry a and
# ln(-b) for its off-diagonal b (-Inf for a tail), formed directly so that it stays finite where
# csch(κh) underflows.
function _empty_flux_log(L::_IntervalLayout{T}, c::Int) where {T}
    m = length(L.y)
    if c == 0 || c == m
        κ, Δ = c == 0 ? (L.κL, L.y[1] - L.lo) : (L.κR, L.hi - L.y[m])
        return _tail_diag(κ, Δ) / κ, T(-Inf)
    end
    κ = L.κs[c]
    θ = κ * (L.y[c+1] - L.y[c])
    return coth(θ) / κ, -log(κ) - logabssinh(θ)
end

# Whether deleting group g's single observation leaves its nodes source-free: no point
# observations on them and no occupied cell beside the group.
function _empties_isolated(L::_IntervalLayout, g::_IntervalGroup)
    m = length(L.y)
    g.r == 1 || return false
    all(i -> L.w[i] == 0, _group_nodes(g, m)) || return false
    g.first >= 1 && L.cellgroup[g.first] != 0 && return false       # cell g.first - 1
    g.last + 1 <= m && L.cellgroup[g.last+2] != 0 && return false   # cell g.last + 1
    return true
end

# Held-out log-probability ln P₋ of a group that `_empties_isolated`, and the change δ in its
# nodal amplitudes. With every cell touching the group's nodes empty, the deleted model's
# stationarity there is linear: C vₗ = -Bᵀv'ₒ, where C is the empty-cell flux form on the group's
# nodes, B couples them to the outer neighbors o, and v'ₒ = vₒ + Kₒₗ Kₗₗ⁻¹δ are those neighbors
# under the quadratic model (`y = Kₗₗ⁻¹δ` from the model's minimizer). C is an M-matrix and the
# right side is positive, so the solve has no cancellation; it is scaled by the largest coupling
# so that vₗ, of order e^{-κ·gap}, never underflows. `Kvblock(r)` gives the block of the inverse
# Hessian in amplitudes on the index range r.
function _emptied_group(L::_IntervalLayout{T}, g::_IntervalGroup{T}, v::Vector{T}, nodes,
                        y::Vector{T}, Kvblock) where {T}
    m = length(L.y)
    k = length(nodes)
    C = let loc = _group_local(L, g, zero(T), v, one(T))
        Matrix{T}(SymTridiagonal(loc.hd, loc.he))
    end
    ext = max(first(nodes) - 1, 1):min(last(nodes) + 1, m)
    Kext = Kvblock(ext)
    Kvfar(i, j) = Kext[i-first(ext)+1, j-first(ext)+1]
    lr = fill(T(-Inf), k)
    for (c, o, q) in ((g.first - 1, first(nodes) - 1, 1), (g.last + 1, last(nodes) + 1, k))
        0 <= c <= m || continue
        a, lb = _empty_flux_log(L, c)
        C[q, q] += a
        1 <= c <= m - 1 || continue
        vo = v[o] + sum(p -> Kvfar(o, nodes[p]) * y[p], 1:k)
        vo > 0 || return T(NaN), fill(T(NaN), k)
        lr[q] = logaddexp(lr[q], lb + log(vo))
    end
    mx = maximum(lr)
    isfinite(mx) || return T(NaN), fill(T(NaN), k)
    x = C \ exp.(lr .- mx)
    all(>(0), x) || return T(NaN), fill(T(NaN), k)
    vv = copy(v)
    vv[nodes] .= x
    return 2mx + log(_group_mass(L, g, vv, one(T))[1]), exp(mx) .* x .- v[nodes]
end

# The exact terms of a local deletion problem on the contiguous `nodes`, summed: each group
# `gs[q]` at multiplicity `rs[q]` (seeded by `τs[q]`), and point terms -2pw[p] ln v at
# nodes[p]. Returns their value, scale, total mass P, the groups' roots τ, and the gradient,
# Hessian (a `_LocalHessian`), and mass gradient in the amplitudes at `nodes`.
function _local_terms(L::_IntervalLayout{T}, gs, rs, τs, nodes, pw::Vector{T}, v::Vector{T}) where {T}
    m = length(L.y)
    k = length(nodes)
    φ = scale = P = zero(T)
    gv = zeros(T, k); hd = zeros(T, k); he = zeros(T, k - 1); gP = zeros(T, k)
    γ = zeros(T, length(gs)); rq = Vector{UnitRange{Int}}(undef, length(gs)); ns = Vector{Vector{T}}(undef, length(gs))
    τ = collect(T, τs)
    Ps = similar(τ)
    for q in eachindex(gs, rs, τ)
        loc = _group_local(L, gs[q], rs[q], v, τ[q])
        idx = _group_nodes(gs[q], m) .- (first(nodes) - 1)
        φ += loc.φ; scale += loc.scale; P += loc.P; Ps[q] = loc.P; τ[q] = loc.τ
        gv[idx] .+= loc.gv; gP[idx] .+= loc.gP
        hd[idx] .+= loc.hd; he[first(idx):last(idx)-1] .+= loc.he
        γ[q] = loc.γ; rq[q] = idx; ns[q] = loc.nvec
    end
    for (p, i) in pairs(nodes)
        pw[p] == 0 && continue
        φ -= 2pw[p] * log(v[i])
        scale += 2pw[p] * abs(log(v[i]))
        gv[p] -= 2pw[p] / v[i]
        hd[p] += 2pw[p] / v[i]^2
    end
    return (; φ, scale, P, Ps, τ, gv, H=_LocalHessian(hd, he, γ, rq, ns), gP)
end

# The Hessian of the rest of the problem (all but the exact local terms with Hessian `H`) reduced
# onto the contiguous `nodes`, (K_ll)⁻¹ - H, in the amplitudes. `band` is ∇²ₛΦ at the fit `v`;
# `dl[i]²` and `dr[i]²` are the Schur complements at node i of the nodes left and right of it (the
# squared Cholesky pivots of ∇²ₛΦ in forward and reverse order). The nodes outside `nodes` couple
# only to its two end nodes, because no group straddles them, so (K_ll)⁻¹ is ∇²ᵥΦ on the block
# with the end diagonals replaced by those Schur complements; the exact terms' rank-one parts
# cancel, and the result is tridiagonal.
function _rest_hessian(band::Matrix{T}, dl::Vector{T}, dr::Vector{T}, v::Vector{T}, nodes,
                       H::_LocalHessian{T}) where {T}
    d, e = _tri_entries(H)
    a, b = first(nodes), last(nodes)
    sd = [4 * v[i]^2 * (i == a ? dl[a]^2 : i == b ? dr[b]^2 : band[1, i]) for i in nodes] .- d
    se = [4 * v[i] * v[i+1] * band[2, i] for i in a:b-1] .- e
    return SymTridiagonal(sd, se)
end

# Diagonal of the Cholesky factor of the banded matrix `band` (lower-band layout) taken in
# reverse node order, indexed by the original nodes.
function _reverse_cholesky_diag(band::Matrix{T}) where {T}
    nb, m = size(band)
    R = zeros(T, nb, m)
    for j in 1:m, d in 0:nb-1
        j + d <= m && (R[1+d, j] = band[1+d, m+1-j-d])
    end
    _banded_cholesky!(R)
    return reverse(R[1, :])
end

# Indices of the groups in `js` together with the groups occupying the cells just outside each.
# A deletion reshapes the field in its own cells; where a neighboring cell is occupied, that
# group's term responds nonlinearly through the shared node and is kept exact too.
function _with_neighbors(L::_IntervalLayout, js::Vector{Int})
    m = length(L.y)
    out = copy(js)
    for j in js
        g = L.groups[j]
        for c in (g.first - 1, g.last + 1)
            0 <= c <= m || continue
            q = L.cellgroup[c+1]
            q == 0 || q in out || push!(out, q)
        end
    end
    return out
end

# The contiguous node range spanned by groups `gs`.
_span_nodes(gs, m::Int) = minimum(g -> first(_group_nodes(g, m)), gs):maximum(g -> last(_group_nodes(g, m)), gs)

# Minimize the deleted local model  φ'(vl) - ∇φ(v)ᵀΔ + ½ΔᵀSΔ,  Δ = vl - v[nodes],  by damped
# Newton over positive amplitudes, where φ = `loc` holds the local terms of the full fit and φ'
# the same terms (`_local_terms`) with multiplicities `rs` and point weights `pw`. Returns the
# minimizer, the local terms there, and which amplitudes are resolved.
#
# A node that the deletion leaves source-free next to an emptied cell falls by a factor of order
# e^{-κh}, which the positivity cap on the step approaches only geometrically. Once an amplitude
# is below √eps of its starting value it is negligible: its effect on the other nodes and on the
# masses is below roundoff. Such nodes are excluded from the step cap and the convergence test and
# held positive by dividing them by 100 whenever the step would overshoot; they are reported as
# unresolved, and a caller that needs one of their values must not use it.
function _delete_local(L::_IntervalLayout{T}, gs, rs, pw::Vector{T}, v::Vector{T}, nodes,
                       loc, S::SymTridiagonal{T}, τ0) where {T}
    v0 = v[nodes]
    w = copy(v)
    resolved = trues(length(v0))
    function advance(vl, α, step)
        vn = vl .- α .* step
        for i in eachindex(vn, vl)
            resolved[i] || vn[i] > 0 || (vn[i] = vl[i] / 100)
        end
        return vn
    end
    function model!(vl, τ)                      # τ seeds the groups' root solves
        w[nodes] .= vl
        new = _local_terms(L, gs, rs, τ, nodes, pw, w)
        Δ = vl .- v0
        return new.φ - dot(loc.gv, Δ) + dot(Δ, S * Δ) / 2, new
    end
    vl = copy(v0)
    f, cur = model!(vl, τ0)
    tol = eps(T)^(3 // 4)
    converged = unguarded = false
    prevstep = T(Inf)
    for _ in 1:100
        τ = cur.τ
        Δ = vl .- v0
        grad = cur.gv .- loc.gv .+ S * Δ
        Hm = _plus_tri(cur.H, S)
        step = _solve_pd(Hm, grad)
        if step === nothing
            Hd = Symmetric(_dense(Hm))
            F = cholesky(Hd; check=false)
            μ = zero(T)
            while !issuccess(F)                 # far from the minimizer: regularize to descend
                μ = max(2μ, sqrt(eps(T)) * maximum(abs, Hd))
                F = cholesky(Hd + μ * I; check=false)
            end
            step = F \ grad
        end
        relstep = zero(T)
        for i in eachindex(vl, step)
            resolved[i] && (relstep = max(relstep, abs(step[i]) / vl[i]))
        end
        # Converged, or at the floor roundoff imposes (unguarded steps no longer shrinking).
        if relstep <= tol || (unguarded && relstep >= prevstep)
            converged = true
            break
        end
        prevstep = relstep
        α = one(T)
        for i in eachindex(vl, step)
            resolved[i] && step[i] > 0 && (α = min(α, T(0.99) * vl[i] / step[i]))
        end
        # Once the predicted decrease is below the roundoff in the model's value (measured by the
        # magnitude of the terms summed into it), the value can no longer guard the step; take it
        # as Newton gives it.
        roundoff = 16 * eps(T) * (cur.scale + abs(dot(loc.gv, Δ)) + abs(dot(Δ, S * Δ)))
        guarded = α * dot(grad, step) > roundoff
        unguarded |= !guarded
        if !guarded
            vl = advance(vl, α, step)
            f, cur = model!(vl, τ)
        end
        while guarded
            vn = advance(vl, α, step)
            fn, new = model!(vn, τ)
            if fn <= f || α * relstep <= eps(T)
                vl = vn
                f, cur = fn, new
                break
            end
            α /= 2
        end
        for i in eachindex(vl, v0)
            vl[i] <= sqrt(eps(T)) * v0[i] && (resolved[i] = false)
        end
    end
    converged || error("deleting an observation in [$(L.y[first(nodes)]), $(L.y[last(nodes)])] " *
                       "did not converge; please report this")
    return vl, cur, resolved
end

# Kullback–Leibler cross-validation score for interval data: the mean negative held-out
# log-likelihood, each interval observation contributing its held-out log-probability and each
# point its held-out log-density. NaN if any held-out value is undefined, so the search rejects κ.
function _interval_klcv(L::_IntervalLayout{T}; holdout::Symbol=:location) where {T}
    lp, ld = _interval_loo(L; holdout)
    s = zero(T)
    n = zero(T)
    for (g, l) in zip(L.groups, lp)
        isfinite(l) || return T(NaN)
        s += g.r * l
        n += g.r
    end
    for i in eachindex(L.w, ld)
        L.w[i] == 0 && continue
        isfinite(ld[i]) || return T(NaN)
        s += L.w[i] * ld[i]
        n += L.w[i]
    end
    return -s / n
end

function select_kappa_kl(lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real};
                         κs::Union{AbstractVector{<:Real},Nothing}=nothing,
                         rtol::Real=cbrt(eps(float(promote_type(eltype(lower), eltype(upper))))),
                         support::Tuple{Real,Real}=(-Inf, Inf), holdout::Symbol=:location)
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    a, b = support
    a < b || throw(DomainError((a, b), "support must satisfy a < b, got support=($a, $b)"))
    axes(lower) == axes(upper) ||
        throw(DimensionMismatch("lower and upper bounds must have the same axes, got $(axes(lower)) and $(axes(upper))"))
    grid = κs === nothing ? _default_interval_κs(lower, upper) : κs
    _check_kappa_grid(grid)
    T = float(promote_type(eltype(lower), eltype(upper), eltype(grid), typeof(rtol),
                           _support_eltype(a), _support_eltype(b)))
    r, slo, shi = T(rtol), T(a), T(b)
    lu, uu, counts = _tally_intervals(lower, upper, T)
    function score(κ)
        L = _interval_layout(lu, uu, κ, r, slo, shi; counts)
        v = _unless_unresolvable(() -> _interval_klcv(L; holdout), T(NaN))
        return isfinite(v) ? v : typemax(T)
    end
    return _minimize_on_grid(score, grid, T)
end

# The distinct (lower, upper) pairs, sorted, with the number of observations sharing each.
# Rounded data have far fewer distinct intervals than observations, and a selector lays out the
# data once per candidate scale.
function _tally_intervals(lower::AbstractVector, upper::AbstractVector, ::Type{T}) where {T}
    ps = sort!([(T(lower[i]), T(upper[i])) for i in eachindex(lower, upper)])
    lu, uu, counts = T[], T[], T[]
    for p in ps
        if !isempty(counts) && p == (lu[end], uu[end])
            counts[end] += 1
        else
            push!(lu, p[1]); push!(uu, p[2]); push!(counts, one(T))
        end
    end
    return lu, uu, counts
end

# Geometric κ grid for interval data: from one blob over the finite bounds up to one scale per
# observation, stopping where κ times the widest finite interval reaches half the fit's accuracy
# limit (`_check_resolvable`).
function _default_interval_κs(lower::AbstractVector, upper::AbstractVector)
    T = float(promote_type(eltype(lower), eltype(upper)))
    lo, hi = T(Inf), T(-Inf)
    wmax = zero(T)
    for i in eachindex(lower, upper)
        a, b = T(lower[i]), T(upper[i])
        for t in (a, b)
            isfinite(t) && (lo = min(lo, t); hi = max(hi, t))
        end
        isfinite(a) && isfinite(b) && (wmax = max(wmax, b - a))
    end
    span = hi - lo
    span > 0 || throw(ArgumentError("need at least two distinct finite bounds to select κ"))
    κhi = 5 * length(lower) / span
    wmax > 0 && (κhi = min(κhi, eps(T)^(-1 // 4) / (2wmax)))
    return exp.(range(log(T(0.5) / span), log(κhi); length = 40))
end

# ---------------------------------------------------------------------------------------------
# The fit

"""
    IntervalDensityEstimate(lower, upper, κ; support=(-Inf, Inf), rtol=cbrt(eps(T)))
    IntervalDensityEstimate(x, κ; resolution, support=(-Inf, Inf), rtol=cbrt(eps(T)))

Estimate a continuous one-dimensional density from interval observations: observation `i` is
known only to lie in `[lower[i], upper[i]]`. An observation with `lower[i] == upper[i]` is an
exact point, so a sample may mix exact and interval-valued observations; a bound of `-Inf` or
`Inf` describes a censored value. Intervals may share endpoints but must not otherwise
overlap (overlapping intervals throw an `ArgumentError`).

The second form is for data recorded on a lattice of spacing `resolution` (rounded values):
each `x[i]` stands for the interval `[x[i] - resolution/2, x[i] + resolution/2]`, clipped to the
support. Every `x[i]` must lie on one lattice `x₀ + k·resolution` (within `resolution/1000`),
or an `ArgumentError` names the first value that does not.

`κ` and `support` have the same meaning as for [`DensityEstimate`](@ref): a positive number or
a callable `κ(x)`, and a domain `(a, b)` outside of which the density is zero. Every
observation must lie within the support. Interval endpoints (and points) closer than
`rtol / κ(x)` are merged.

The estimate maximizes the penalized likelihood in which an interval observation contributes
the log-probability of its interval rather than a log-density, so a density spike inside an
interval earns nothing. Rounded data are therefore never pulled toward point masses at the
recorded values; the finest structure they can support is the histogram at the rounding
increment.

The result is callable, `d(x)` giving the density, and supports [`amplitude`](@ref),
[`logdensity`](@ref), [`cdf`](@ref), and [`quantile`](@ref Statistics.quantile). With point
observations only, the fit equals `DensityEstimate(x, κ)`.

# Examples
```jldoctest
julia> x = round.([-1.23, -0.52, -0.48, 0.07, 0.11, 0.13, 0.66, 1.41]; digits=1);

julia> d = IntervalDensityEstimate(x, 2.0; resolution=0.1);

julia> cdf(d, -Inf), cdf(d, Inf)
(0.0, 1.0)

julia> c = IntervalDensityEstimate([0.0, 0.5, 1.0], [0.0, 0.5, Inf], 1.0);  # 1.0 is right-censored
```

# Extended help

The amplitude `ψ = √Q` minimizes

    S[ψ] = ∫ (λ/κ(x)²) (ψ')² dx - 2 Σᵢ wᵢ ln ψ(xᵢ) - Σⱼ rⱼ ln ∫_{Bⱼ} ψ² dx

subject to `∫ ψ² dx = 1`, where `wᵢ` counts the point observations at `xᵢ` and `rⱼ` the
observations in interval `Bⱼ`. Between consecutive endpoints, `ψ'' = k²ψ` with
`k² = κ²(1 - rⱼ/(λPⱼ))` inside interval `j` (`Pⱼ` its fitted probability) and `k² = κ²`
outside every interval. Inside a well-populated interval `k² < 0` and `ψ` is a
trigonometric arc, so the density is a smooth bump across the interval. The problem is
convex in the density, and the fit is found by Newton's method on the nodal densities at
`O(m)` cost per step for `m` distinct endpoints, provided each interval contains few interior
points.

The fields `d.x` (nodes), `d.ψ` (normalized amplitudes there), `d.k2` (the realized `k²` on
each cell between nodes), and `d.kL`, `d.kR` (the tails' decay rates) describe the fitted
curve.
"""
struct IntervalDensityEstimate{T<:AbstractFloat,K} <: AbstractDensityEstimate
    x::Vector{T}        # sorted distinct nodes: interval endpoints and points
    w::Vector{T}        # point multiplicity at each node
    ψ::Vector{T}        # normalized amplitude at the nodes
    k2::Vector{T}       # realized k² on each interior cell
    kL::T               # decay rate of the left tail (its scale on a finite support)
    kR::T               # decay rate of the right tail
    κ::K                # smoothing scale: one number, or one per interior cell
    κL::T
    κR::T
    lo::T
    hi::T
    groups::Vector{_IntervalGroup{T}}
    λ::T                # normalization multiplier (diagnostic)
end

function IntervalDensityEstimate(lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real}, κ;
                                 support::Tuple{Real,Real}=(-Inf, Inf),
                                 rtol::Real=cbrt(eps(float(promote_type(eltype(lower), eltype(upper))))),
                                 stats=nothing)
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative, got $rtol"))
    a, b = support
    a < b || throw(DomainError((a, b), "support must satisfy a < b, got support=($a, $b)"))
    T = float(promote_type(eltype(lower), eltype(upper), typeof(rtol), _kappa_eltype(κ, lower, upper),
                           _support_eltype(a), _support_eltype(b)))
    κ isa Real && !(κ > 0 && isfinite(κ)) && throw(ArgumentError("κ must be finite and positive, got $κ"))
    L = _interval_layout(lower, upper, κ, T(rtol), T(a), T(b))
    return _interval_fit(L, κ isa Real ? T(κ) : L.κs; stats)
end

function IntervalDensityEstimate(x::AbstractVector{<:Real}, κ; resolution::Union{Real,Nothing}=nothing,
                                 support::Tuple{Real,Real}=(-Inf, Inf),
                                 rtol::Real=cbrt(eps(float(eltype(x)))), stats=nothing)
    resolution === nothing &&
        throw(ArgumentError("pass the lattice spacing as `resolution`, or give the bounds of each " *
                            "observation as `IntervalDensityEstimate(lower, upper, κ)`"))
    isfinite(resolution) && resolution > 0 ||
        throw(ArgumentError("resolution must be finite and positive, got $resolution"))
    isempty(x) && throw(ArgumentError("cannot fit a density to zero observations"))
    T = float(promote_type(eltype(x), typeof(resolution)))
    lower, upper = _lattice_bounds(x, T(resolution), support)
    return IntervalDensityEstimate(lower, upper, κ; support, rtol, stats)
end

# The rounding interval of each lattice value. Bounds are formed from integer lattice indices so
# that adjacent intervals share their endpoint exactly.
function _lattice_bounds(x::AbstractVector, δ::T, support) where {T}
    x0 = T(minimum(x))
    lo, hi = support
    lower = similar(x, T)
    upper = similar(x, T)
    for i in eachindex(x, lower, upper)
        xi = T(x[i])
        k = round((xi - x0) / δ)
        abs(xi - (x0 + k * δ)) <= δ / 1000 ||
            throw(ArgumentError("value $xi is not on the lattice $x0 + k·$δ; check `resolution`"))
        lower[i] = max(x0 + (k - T(1) / 2) * δ, T(lo))
        upper[i] = min(x0 + (k + T(1) / 2) * δ, T(hi))
    end
    return lower, upper
end

_kappa_eltype(κ::Real, lower, upper) = typeof(κ)
function _kappa_eltype(κfun, lower, upper)
    for (a, b) in zip(lower, upper)
        isfinite(a) && return typeof(κfun(a))
        isfinite(b) && return typeof(κfun(b))
    end
    return Bool
end

function _interval_fit(L::_IntervalLayout{T}, κ; stats=nothing) where {T}
    s, E = try
        _solve_interval(L; stats)
    catch err
        err isa _UnresolvableScale && throw(ArgumentError(err.msg))
        rethrow()
    end
    y = L.y
    m = length(y)
    Z = E.Z
    ψ = sqrt.(s ./ Z)
    τcell(c) = (j = L.cellgroup[c+1]; j == 0 ? one(T) : E.τ[j])
    k2 = T[L.κs[c]^2 * τcell(c) for c in 1:m-1]
    kL = L.κL * sqrt(τcell(0))
    kR = L.κR * sqrt(τcell(m))
    return IntervalDensityEstimate{T,typeof(κ)}(y, L.w, ψ, k2, kL, kR, κ, L.κL, L.κR, L.lo, L.hi,
                                                 L.groups, Z / 2)
end

function Base.show(io::IO, d::IntervalDensityEstimate)
    nint = isempty(d.groups) ? 0 : sum(g -> g.r, d.groups)
    κstr = d.κ isa Real ? "κ=$(d.κ)" :
           "κ ∈ [$(min(d.κL, d.κR, minimum(d.κ; init=Inf))), $(max(d.κL, d.κR, maximum(d.κ; init=-Inf)))]"
    print(io, "IntervalDensityEstimate with $(length(d.x)) nodes, $(nint) interval and ",
          "$(sum(d.w)) point observations, $κstr, λ=$(d.λ)", _show_support(d))
end

_show_support(d::IntervalDensityEstimate) =
    isinf(d.lo) && isinf(d.hi) ? "" : ", support=[$(d.lo), $(d.hi)]"

# ---------------------------------------------------------------------------------------------
# Evaluation

# ψ on interior cell c at x ∈ [x_c, x_{c+1}].
function _cell_amplitude(d::IntervalDensityEstimate{T}, c::Int, x::T) where {T}
    x0, x1 = d.x[c], d.x[c+1]
    k2 = d.k2[c]
    if k2 > 0
        k = sqrt(k2)
        a, b = k * (x1 - x), k * (x - x0)
        return d.ψ[c] * _sinh_ratio(a, a + b) + d.ψ[c+1] * _sinh_ratio(b, a + b)
    end
    return (d.ψ[c] * _Sfun(k2, x1 - x) + d.ψ[c+1] * _Sfun(k2, x - x0)) / _Sfun(k2, x1 - x0)
end

function _cell_log_amplitude(d::IntervalDensityEstimate{T}, c::Int, x::T) where {T}
    x0, x1 = d.x[c], d.x[c+1]
    k2 = d.k2[c]
    if k2 > 0
        k = sqrt(k2)
        a, b = k * (x1 - x), k * (x - x0)
        return logaddexp(log(d.ψ[c]) + logabssinh(a), log(d.ψ[c+1]) + logabssinh(b)) - logabssinh(a + b)
    end
    return logaddexp(log(d.ψ[c]) + log(_Sfun(k2, x1 - x)), log(d.ψ[c+1]) + log(_Sfun(k2, x - x0))) -
           log(_Sfun(k2, x1 - x0))
end

function _amplitude(d::IntervalDensityEstimate{T}, x::Real) where {T}
    xs = d.x
    m = length(xs)
    xT = T(x)
    if xT <= xs[1]
        xT < d.lo && return zero(T)
        return _left_tail_amplitude(d.ψ[1], d.kL, xT, xs[1], d.lo)
    elseif xT >= xs[m]
        xT > d.hi && return zero(T)
        return _right_tail_amplitude(d.ψ[m], d.kR, xT, xs[m], d.hi)
    end
    return _cell_amplitude(d, searchsortedlast(xs, xT), xT)
end

function _logdensity(d::IntervalDensityEstimate{T}, x::Real) where {T}
    xs = d.x
    m = length(xs)
    xT = T(x)
    if xT <= xs[1]
        xT < d.lo && return T(-Inf)
        return 2 * _log_left_tail_amplitude(d.ψ[1], d.kL, xT, xs[1], d.lo)
    elseif xT >= xs[m]
        xT > d.hi && return T(-Inf)
        return 2 * _log_right_tail_amplitude(d.ψ[m], d.kR, xT, xs[m], d.hi)
    end
    return 2 * _cell_log_amplitude(d, searchsortedlast(xs, xT), xT)
end

"""
    amplitude(d::IntervalDensityEstimate, x)

Evaluate the amplitude `ψ(x)`, so that `d(x) == ψ(x)^2`, at a scalar or array `x`. Zero
outside a finite support.
"""
amplitude(d::IntervalDensityEstimate, x::Real) = _amplitude(d, x)
amplitude(d::IntervalDensityEstimate, x::AbstractArray) = map(xi -> _amplitude(d, xi), x)

"""
    logdensity(d::IntervalDensityEstimate, x)

Evaluate `ln Q̂(x)` at a scalar or array `x` without underflow in the tails; `-Inf` outside a
finite support.
"""
logdensity(d::IntervalDensityEstimate, x::Real) = _logdensity(d, x)
logdensity(d::IntervalDensityEstimate, x::AbstractArray) = map(xi -> _logdensity(d, xi), x)

(d::IntervalDensityEstimate)(x::Real) = _amplitude(d, x)^2

# Mass of interior cell c.
function _cell_mass(d::IntervalDensityEstimate{T}, c::Int) where {T}
    _, _, n11, n12, _, _ = _cell_coeffs(d.k2[c], d.x[c+1] - d.x[c])
    return n11 * (d.ψ[c]^2 + d.ψ[c+1]^2) + 2n12 * d.ψ[c] * d.ψ[c+1]
end

_left_end_mass(d::IntervalDensityEstimate) = _tail_mass(d.ψ[1], d.kL, d.x[1] - d.lo)
_right_end_mass(d::IntervalDensityEstimate) = _tail_mass(d.ψ[end], d.kR, d.hi - d.x[end])

# Node cumulatives F[i] = ∫_{lo}^{xᵢ} ψ² and the total mass.
function _node_cdf(d::IntervalDensityEstimate{T}) where {T}
    m = length(d.x)
    F = Vector{T}(undef, m)
    F[1] = _left_end_mass(d)
    for c in 1:m-1
        F[c+1] = F[c] + _cell_mass(d, c)
    end
    return F, F[m] + _right_end_mass(d)
end

# ∫ψ² over [x_c, x] (fromleft) or [x, x_{c+1}] on interior cell c, meant for the half of the cell
# nearer the node integrated from. Hyperbolic cells use the exact antiderivative; trigonometric
# ones Gauss–Legendre quadrature, which is exact to roundoff because ψ² there is a
# trigonometric polynomial of frequency below 2π over the cell.
function _cell_partial_mass(d::IntervalDensityEstimate{T}, c::Int, x::T, fromleft::Bool) where {T}
    x0, x1 = d.x[c], d.x[c+1]
    k2 = d.k2[c]
    if k2 > 0
        k = sqrt(k2)
        a, b = k * (x1 - x), k * (x - x0)
        return fromleft ? _segmass(d.ψ[c], d.ψ[c+1], a, b, a + b) / k :
                          _segmass(d.ψ[c+1], d.ψ[c], b, a, a + b) / k
    end
    ta, tb = fromleft ? (x0, x) : (x, x1)
    nodes, weights = _gauss16(T)
    half, mid = (tb - ta) / 2, (tb + ta) / 2
    acc = zero(T)
    for i in eachindex(nodes, weights)
        acc += weights[i] * _cell_amplitude(d, c, mid + half * nodes[i])^2
    end
    return half * acc
end

_gauss16(::Type{T}) where {T} = gauss(T, 16)
const _GAUSS16_F64 = gauss(Float64, 16)
_gauss16(::Type{Float64}) = _GAUSS16_F64

# Unnormalized cumulative mass ∫_{lo}^{x} ψ², integrated from the nearer end of its cell.
function _cdf_mass(d::IntervalDensityEstimate{T}, F::Vector{T}, x::Real) where {T}
    xs, ψ = d.x, d.ψ
    m = length(xs)
    isnan(x) && return T(NaN)
    xT = T(x)
    if xT <= xs[1]
        isfinite(d.lo) || return ψ[1]^2 / (2d.kL) * exp(2d.kL * (xT - xs[1]))
        xT <= d.lo && return zero(T)
        v, u = d.kL * (xT - d.lo), d.kL * (xs[1] - d.lo)
        return v <= u / 2 ? _boundary_mass_from_wall(ψ[1], d.kL, v, u) :
                            F[1] - _boundary_mass_from_node(ψ[1], d.kL, v, u)
    elseif xT >= xs[m]
        isfinite(d.hi) || return F[m] + ψ[m]^2 / (2d.kR) * (-expm1(-2d.kR * (xT - xs[m])))
        xT >= d.hi && return F[m] + _right_end_mass(d)
        vp, u = d.kR * (d.hi - xT), d.kR * (d.hi - xs[m])
        return vp >= u / 2 ? F[m] + _boundary_mass_from_node(ψ[m], d.kR, vp, u) :
                             F[m] + _right_end_mass(d) - _boundary_mass_from_wall(ψ[m], d.kR, vp, u)
    end
    c = searchsortedlast(xs, xT)
    return xT - xs[c] <= xs[c+1] - xT ? F[c] + _cell_partial_mass(d, c, xT, true) :
                                        F[c+1] - _cell_partial_mass(d, c, xT, false)
end

"""
    cdf(d::IntervalDensityEstimate, x)

Cumulative distribution function of the fitted density at a scalar or array `x`; exactly `0`
and `1` at the support endpoints.
"""
function cdf(d::IntervalDensityEstimate, x::Real)
    F, total = _node_cdf(d)
    return _cdf_mass(d, F, x) / total
end
function cdf(d::IntervalDensityEstimate, x::AbstractArray)
    F, total = _node_cdf(d)
    return map(xi -> _cdf_mass(d, F, xi) / total, x)
end

"""
    quantile(d::IntervalDensityEstimate, q)

Quantile function of the fitted density, the inverse of [`cdf`](@ref), for `q ∈ [0, 1]`
(scalar or array); `q` outside `[0, 1]` throws a `DomainError`.
"""
function Statistics.quantile(d::IntervalDensityEstimate, q::Real)
    F, total = _node_cdf(d)
    return _quantile(d, F, total, q)
end
function Statistics.quantile(d::IntervalDensityEstimate, q::AbstractArray)
    F, total = _node_cdf(d)
    return map(qi -> _quantile(d, F, total, qi), q)
end

# Safeguarded Newton for massfun(y) == target on [lo, hi], massfun increasing with slope d(y).
function _invert_mass(d::IntervalDensityEstimate{T}, massfun, lo::T, hi::T, y::T, target::T) where {T}
    for _ in 1:200
        r = massfun(y) - target
        r == 0 && return y
        r < 0 ? (lo = y) : (hi = y)
        ynew = y - r / _amplitude(d, y)^2
        lo < ynew < hi || (ynew = (lo + hi) / 2)
        ynew == y && return y
        y = ynew
    end
    error("quantile: safeguarded Newton failed to converge at target = $target — please report this")
end

function _quantile(d::IntervalDensityEstimate{T}, F::Vector{T}, total::T, q::Real) where {T}
    0 <= q <= 1 || throw(DomainError(q, "quantile is defined only for probabilities 0 ≤ q ≤ 1"))
    xs, ψ = d.x, d.ψ
    m = length(xs)
    target = T(q) * total
    if target <= F[1]
        isfinite(d.lo) || return _left_tail_quantile(ψ[1], d.kL, xs[1], target)
        y = F[1] > 0 ? d.lo + (target / F[1]) * (xs[1] - d.lo) : d.lo
        return _invert_mass(d, t -> _cdf_mass(d, F, t), d.lo, xs[1], y, target)
    elseif target >= F[m]
        isfinite(d.hi) || return _right_tail_quantile(ψ[m], d.kR, xs[m], total, q)
        y = total > F[m] ? xs[m] + (target - F[m]) / (total - F[m]) * (d.hi - xs[m]) : d.hi
        return _invert_mass(d, t -> _cdf_mass(d, F, t), xs[m], d.hi, y, target)
    end
    c = searchsortedlast(F, target)
    y = xs[c] + (target - F[c]) / (F[c+1] - F[c]) * (xs[c+1] - xs[c])
    return _invert_mass(d, t -> _cdf_mass(d, F, t), xs[c], xs[c+1], y, target)
end
