# Tests of IntervalDensityEstimate. Included from runtests.jl.

using LinearAlgebra: diag, dot
using SparseArrays: sparse

# Finite-difference reference for the interval likelihood, independent of the package: minimize
# the discretized objective in f = u² on a uniform grid over [L, R] with trapezoid weights,
#     ½Σ τᵢfᵢ + Σ (f_i + f_{i+1} - 2√(f_i f_{i+1}))/(2δκ²) - Σⱼ rⱼ ln Σ_{Bⱼ} τ f - Σ wᵢ ln fᵢ,
# which is convex in f. Every finite bound must be a grid point; the grid ends are natural
# boundaries, so an unbounded side needs padding past which the tail mass is negligible. `κ`
# is a number or a function evaluated at each grid cell's midpoint. Returns the normalized
# density at `ys`. Converges at O(δ²).
function fd_interval_fit(lower, upper, κ, L, R, δ, ys)
    N = round(Int, (R - L) / δ)
    xg = range(L, R; length=N + 1)
    function idx(z)
        i = round(Int, (z - L) / δ) + 1
        abs(xg[i] - z) < 1e-9 * max(1, abs(z)) || error("$z is not a grid point")
        return i
    end
    tw = fill(δ, N + 1); tw[1] = tw[end] = δ / 2
    bins = Dict{Tuple{Int,Int},Int}()
    pts = Dict{Int,Int}()
    for (a, b) in zip(lower, upper)
        if a == b
            pts[idx(a)] = get(pts, idx(a), 0) + 1
        else
            key = (isfinite(a) ? idx(a) : 1, isfinite(b) ? idx(b) : N + 1)
            bins[key] = get(bins, key, 0) + 1
        end
    end
    binw(r) = (w = fill(δ, length(r)); w[1] = w[end] = δ / 2; w)
    binl = [(r[1]:r[2], cnt, binw(r[1]:r[2])) for (r, cnt) in bins]
    ptl = collect(pts)
    n = length(lower)
    κcell = κ isa Real ? fill(κ, N) : [κ(xg[i] + δ / 2) for i in 1:N]
    avec = @. 1 / (2δ * κcell^2)
    function obj(f)
        F = dot(tw, f) / 2
        for i in 1:N
            F += avec[i] * (f[i] + f[i+1] - 2sqrt(f[i] * f[i+1]))
        end
        for (r, cnt, w) in binl
            F -= cnt * log(dot(w, view(f, r)))
        end
        for (i, cnt) in ptl
            F -= cnt * log(f[i])
        end
        return F
    end
    function gradhess(f)
        g = tw ./ 2
        I = Int[]; J = Int[]; V = Float64[]
        for i in 1:N
            a = avec[i]
            q = sqrt(f[i] * f[i+1])
            g[i] += a * (1 - sqrt(f[i+1] / f[i])); g[i+1] += a * (1 - sqrt(f[i] / f[i+1]))
            append!(I, (i, i + 1, i, i + 1)); append!(J, (i, i + 1, i + 1, i))
            append!(V, (a * sqrt(f[i+1]) / (2f[i]^1.5), a * sqrt(f[i]) / (2f[i+1]^1.5), -a / (2q), -a / (2q)))
        end
        for (r, cnt, w) in binl
            P = dot(w, view(f, r))
            g[r] .-= cnt .* w ./ P
            for (ii, i) in enumerate(r), (jj, j) in enumerate(r)
                push!(I, i); push!(J, j); push!(V, cnt * w[ii] * w[jj] / P^2)
            end
        end
        for (i, cnt) in ptl
            g[i] -= cnt / f[i]; push!(I, i); push!(J, i); push!(V, cnt / f[i]^2)
        end
        return g, sparse(I, J, V, N + 1, N + 1)
    end
    c, s = (L + R) / 2, (R - L) / 4
    f = [n * exp(-((x - c) / s)^2 / 2) + 1e-3 for x in xg]
    for _ in 1:300
        F = obj(f); g, H = gradhess(f)
        Δ = -(H \ g)
        dec = -dot(g, Δ)
        dec < 1e-22 * n && break
        t = 1.0
        while any(f .+ t .* Δ .<= 0) || obj(f .+ t .* Δ) > F - 1e-4 * t * dec
            t /= 2
        end
        f .+= t .* Δ
    end
    return [f[idx(y)] for y in ys] ./ dot(tw, f)
end

# Rounded standard-normal sample as intervals of width δr on the lattice δr·ℤ, restricted to the
# open support and clipped to it; optionally right-censored at c (r extra observations), plus
# exact points `extra`.
function rounded_intervals(; n=200, δr=0.25, extra=Float64[], support=(-Inf, Inf), cens=nothing, seed=1)
    z = round.(randn(Xoshiro(seed), n) ./ δr) .* δr
    lo, hi = support
    z = filter(t -> lo < t < hi, z)
    lower = max.(z .- δr / 2, lo); upper = min.(z .+ δr / 2, hi)
    if cens !== nothing
        c, r = cens
        keep = upper .<= c
        lower, upper = lower[keep], upper[keep]
        append!(lower, fill(c, r)); append!(upper, fill(Inf, r))
    end
    append!(lower, extra); append!(upper, extra)
    return lower, upper
end

# Maximum relative error of the fit's nodal density against the FD reference at two grids.
# Returns the two errors; a correct fit shows them falling by 4 per halving.
function fd_errors(d, lower, upper, κ; support=(-Inf, Inf), pad=14 / (κ isa Real ? κ : 2), δr=0.25)
    L = isfinite(support[1]) ? support[1] : floor((minimum(filter(isfinite, lower)) - pad) / δr) * δr - δr / 2
    R = isfinite(support[2]) ? support[2] : ceil((maximum(filter(isfinite, upper)) + pad) / δr) * δr + δr / 2
    return map((δr / 8, δr / 16)) do δ
        maximum(abs.(fd_interval_fit(lower, upper, κ, L, R, δ, d.x) ./ d.(d.x) .- 1))
    end
end

# Second-order agreement: the error at the finer grid is a quarter of the coarser one (within
# 10%), so the FD solution converges to the nodal fit rather than to something nearby.
second_order(errs) = 3.6 < errs[1] / errs[2] < 4.4

@testset "IntervalDensityEstimate" begin
    @testset "cell coefficients: derivative identities in every branch" begin
        # ∂(flux)/∂k² = mass and ∂(mass)/∂k² = the returned derivative, across the series
        # (|k²h²| ≤ 1), hyperbolic, and trigonometric branches and near their boundaries.
        for (k2, h) in ((0.3, 1.0), (0.99, 1.0), (1.01, 1.0), (-0.99, 1.0), (-1.01, 1.0),
                        (-9.0, 1.0), (25.0, 1.0), (400.0, 1.0), (-5.0, 0.3))
            c = PenalizedDensity._cell_coeffs(k2, h)
            ε = 1e-6 * max(1, abs(k2))
            cp = PenalizedDensity._cell_coeffs(k2 + ε, h)
            cm = PenalizedDensity._cell_coeffs(k2 - ε, h)
            @test all(isapprox.((cp[1:2] .- cm[1:2]) ./ 2ε, c[3:4]; rtol=1e-7))
            @test all(isapprox.((cp[3:4] .- cm[3:4]) ./ 2ε, c[5:6]; rtol=1e-7))
        end
        # Continuity across the branch switches at k²h² = ±1.
        for z in (-1.0, 1.0)
            a = PenalizedDensity._cell_coeffs(prevfloat(z), 1.0)
            b = PenalizedDensity._cell_coeffs(nextfloat(z), 1.0)
            @test all(isapprox.(a, b; rtol=1e-12))
        end
        # Mass matrix of an empty cell matches the point-data interval mass.
        κ, h, v0, v1 = 3.0, 0.7, 1.3, 0.4
        c = PenalizedDensity._cell_coeffs(κ^2, h)
        @test c[3] * (v0^2 + v1^2) + 2c[4] * v0 * v1 ≈ PenalizedDensity._interval_mass(v0, v1, κ * h) / κ
        @test_throws DomainError PenalizedDensity._cell_coeffs(-10.0, 1.0)
    end

    @testset "points only: identical to DensityEstimate" begin
        x = sort(randn(Xoshiro(3), 25))
        for (κ, support) in ((2.0, (-Inf, Inf)), (t -> 1 + exp(-t^2), (-4, 4)), (5.0, (-3, Inf)))
            d = DensityEstimate(x, κ; support)
            di = IntervalDensityEstimate(x, x, κ; support)
            @test di.x == d.x
            @test di.ψ ≈ d.ψ rtol = 1e-12
            @test di.λ ≈ d.λ rtol = 1e-12
            ts = range(-5, 5; length=101)
            @test di.(ts) ≈ d.(ts) rtol = 1e-11
            @test cdf(di, ts) ≈ cdf(d, ts) rtol = 1e-11
        end
    end

    @testset "narrow intervals converge to the point fit at O(ε)" begin
        x = sort(randn(Xoshiro(3), 25))
        d = DensityEstimate(x, 2.0)
        @test minimum(diff(x)) > 1e-3       # the intervals below do not overlap
        errs = map((1e-3, 1e-4)) do ε
            di = IntervalDensityEstimate(x .- ε / 2, x .+ ε / 2, 2.0)
            maximum(abs.(di.(x) ./ d.(x) .- 1))
        end
        @test 8 < errs[1] / errs[2] < 12
        @test errs[2] < 1e-2
    end

    @testset "finite-difference reference" begin
        # Rounded data with points inside bins (intervals spanning several cells), at two scales.
        lower, upper = rounded_intervals(extra=[0.0625, -0.03125, 0.3125, 3.0])
        layout = PenalizedDensity._interval_layout(lower, upper, 3.0, cbrt(eps()), -Inf, Inf)
        @test PenalizedDensity._bandwidth(layout) == 3
        for κ in (3.0, 12.0)
            d = IntervalDensityEstimate(lower, upper, κ)
            @test minimum(d.k2) < 0 < maximum(d.k2)     # both signs of k² occur
            errs = fd_errors(d, lower, upper, κ)
            @test second_order(errs)
            @test errs[2] < 0.01
        end
        # A varying scale, compared at the fit's realized piecewise-constant κ.
        d = IntervalDensityEstimate(lower, upper, t -> 2 + 2exp(-t^2))
        κpc(t) = t <= d.x[1] ? d.κL : t >= d.x[end] ? d.κR : d.κ[searchsortedlast(d.x, t)]
        errs = fd_errors(d, lower, upper, κpc; pad=7.0)
        @test second_order(errs)
        # Bounded support, with bins clipped at the walls.
        support = (-1.625, 2.125)
        lower, upper = rounded_intervals(extra=[0.0625, -1.5]; support)
        d = IntervalDensityEstimate(lower, upper, 3.0; support)
        @test second_order(fd_errors(d, lower, upper, 3.0; support))
        # Right censoring: an occupied unbounded tail decays more slowly than κ.
        lower, upper = rounded_intervals(extra=[0.0625], cens=(1.125, 20))
        d = IntervalDensityEstimate(lower, upper, 3.0)
        @test 0 < d.kR < d.κR
        @test second_order(fd_errors(d, lower, upper, 3.0; pad=10.0))
        # Left censoring: the mirror image of the right-censored fit, also for a varying scale
        # (with the left-censored observations listed first).
        κf(t) = 2 + 2exp(-(t - 0.3)^2)
        p = sortperm(-upper)
        ts = range(-4, 4; length=81)
        for (κ, κm) in ((3.0, 3.0), (κf, t -> κf(-t)))
            d = IntervalDensityEstimate(lower, upper, κ)
            dm = IntervalDensityEstimate(-upper[p], -lower[p], κm)
            @test dm.kL ≈ d.kR rtol = 1e-12
            @test dm.(-ts) ≈ d.(ts) rtol = 1e-12
        end
    end

    @testset "normalization, cdf, quantile, logdensity" begin
        cases = [(rounded_intervals(extra=[0.0625, 3.0])..., 3.0, (-Inf, Inf)),
                 (rounded_intervals(extra=[0.0625, 3.0])..., 40.0, (-Inf, Inf)),
                 (rounded_intervals(extra=[0.0625, -1.5]; support=(-1.625, 2.125))..., 3.0, (-1.625, 2.125)),
                 (rounded_intervals(cens=(1.125, 20))..., 3.0, (-Inf, Inf))]
        for (lower, upper, κ, support) in cases
            d = IntervalDensityEstimate(lower, upper, κ; support)
            lo, hi = support
            breaks = sort(vcat(d.x, filter(isfinite, [lo, hi])))
            edges = [lo; breaks; hi]
            @test sum(quadgk(d, edges[i], edges[i+1]; rtol=1e-12)[1] for i in 1:length(edges)-1 if edges[i] < edges[i+1]) ≈ 1 rtol = 1e-10
            xs = range(first(d.x) - 1, last(d.x) + 1; length=301)
            F = cdf(d, xs)
            @test issorted(F)
            @test cdf(d, lo) == 0 && cdf(d, hi) == 1
            for x in xs[1:50:end]
                @test F[findfirst(==(x), xs)] ≈ quadgk(d, lo, x; rtol=1e-12)[1] atol = 1e-9
            end
            qs = [1e-8, 0.01, 0.3, 0.5, 0.77, 0.999, 1 - 1e-9]
            @test cdf(d, quantile(d, qs)) ≈ qs rtol = 1e-9
            @test quantile(d, 0.0) == lo && quantile(d, 1.0) == hi
            inside = filter(x -> lo < x < hi, xs)
            @test logdensity(d, inside) ≈ log.(d.(inside))
            @test amplitude(d, inside) .^ 2 ≈ d.(inside)
            @test amplitude(d, inside[1])^2 ≈ d(inside[1])
        end
        d = IntervalDensityEstimate(rounded_intervals(; support=(-1.625, 2.125))..., 3.0; support=(-1.625, 2.125))
        @test d(-2.0) == 0 && logdensity(d, 3.0) == -Inf
        @test_throws DomainError quantile(d, 1.5)
    end

    @testset "lattice data (resolution)" begin
        x = round.(randn(Xoshiro(1), 1000); digits=1)
        d = IntervalDensityEstimate(x, 3.0; resolution=0.1)
        @test isempty(d.w) || all(iszero, d.w)
        @test sum(g -> g.r, d.groups) == 1000
        @test length(d.groups) == length(unique(x .+ 0.0))   # one interval per recorded value (-0.0 == 0.0)
        # The same as passing the bounds directly.
        di = IntervalDensityEstimate(x .- 0.05, x .+ 0.05, 3.0)
        @test di.ψ ≈ d.ψ rtol = 1e-8
        # Clipped to a finite support.
        dp = IntervalDensityEstimate(abs.(x), 3.0; resolution=0.1, support=(0, Inf))
        @test dp.x[1] == 0 && cdf(dp, 0.0) == 0
        # Large κ approaches the histogram limit, and the solve stays converged there.
        maxd = map((1e3, 1e4)) do κ
            st = PenalizedDensity.SolveStats()
            dκ = IntervalDensityEstimate(x, κ; resolution=0.1, stats=st)
            @test st.reason === :tolerance
            maximum(dκ.(range(-3, 3; length=6001)))
        end
        @test maxd[1] ≈ maxd[2] rtol = 1e-4
        # Rounded data are never pulled toward point masses at the recorded values.
        @test maxd[2] < 1
        @test_throws "not on the lattice" IntervalDensityEstimate([0.0, 0.13], 1.0; resolution=0.1)
        @test_throws "pass the lattice spacing as `resolution`" IntervalDensityEstimate([0.0, 0.1], 1.0)
        @test_throws "resolution must be finite and positive" IntervalDensityEstimate([0.0, 0.1], 1.0; resolution=0)
        @test_throws "histogram limit" IntervalDensityEstimate(x, 1e6; resolution=0.1)
        # Large samples cost O(number of distinct bins).
        xbig = round.(randn(Xoshiro(2), 10^6); digits=1)
        @test IntervalDensityEstimate(xbig, 5.0; resolution=0.1) isa IntervalDensityEstimate{Float64,Float64}
    end

    @testset "element types and generic indexing" begin
        x = round.(randn(Xoshiro(1), 1000); digits=1)
        d64 = IntervalDensityEstimate(x, 3.0; resolution=0.1)
        d32 = IntervalDensityEstimate(Float32.(x), 3.0f0; resolution=0.1f0)
        @test d32 isa IntervalDensityEstimate{Float32,Float32}
        ts = range(-3, 3; length=61)
        @test d32.(ts) ≈ d64.(ts) rtol = 1e-4
        # Partial cell masses inside occupied cells with k² < 0 use Gauss–Legendre in Float32.
        @test any(<(0), d32.k2)
        @test cdf(d32, Float32.(ts)) ≈ cdf(d64, ts) rtol = 1e-5
        lower, upper = rounded_intervals(extra=[0.0625, 3.0])
        d = IntervalDensityEstimate(lower, upper, 3.0)
        @test IntervalDensityEstimate(OffsetArray(lower, -7), OffsetArray(upper, -7), 3.0).ψ == d.ψ
        @test IntervalDensityEstimate(view(lower, :), view(upper, :), 3.0).ψ == d.ψ
        @test IntervalDensityEstimate(OffsetArray(x, 5), 3.0; resolution=0.1).ψ == d64.ψ
        q = OffsetArray([0.1, 0.2], 3)
        @test axes(cdf(d, q)) == axes(q)
        @test axes(quantile(d, q)) == axes(q)
        @test axes(logdensity(d, q)) == axes(q)
        @test_throws DimensionMismatch IntervalDensityEstimate([0.0, 1.0], [1.0], 1.0)
    end

    @testset "input validation" begin
        @test_throws "overlapping intervals are not supported" IntervalDensityEstimate([0.0, 0.5], [1.0, 1.5], 1.0)
        @test_throws "lower ≤ upper" IntervalDensityEstimate([1.0], [0.0], 1.0)
        @test_throws DomainError IntervalDensityEstimate([0.0], [2.0], 1.0; support=(0, 1))
        @test_throws "carries no information" IntervalDensityEstimate([-Inf], [Inf], 1.0)
        @test_throws "must be finite" IntervalDensityEstimate([Inf], [Inf], 1.0)
        @test_throws "NaN" IntervalDensityEstimate([NaN], [1.0], 1.0)
        @test_throws "zero observations" IntervalDensityEstimate(Float64[], Float64[], 1.0)
        @test_throws "zero observations" IntervalDensityEstimate(Float64[], Float64[], t -> 1.0)
        @test_throws "κ must be finite and positive" IntervalDensityEstimate([0.0], [1.0], -1.0)
        # A point near one edge of an occupied interval: at large κ the density at the far edge
        # underflows beyond what the solve can represent.
        @test_throws "use a smaller κ" IntervalDensityEstimate([0.0, 1.0, 2.0, 0.9, 1.5, 2.5],
                                                               [1.0, 2.0, 3.0, 0.9, 1.5, 2.5], 200.0)
        # Intervals may share an endpoint.
        @test IntervalDensityEstimate([0.0, 1.0], [1.0, 2.0], 1.0) isa IntervalDensityEstimate
    end

    @testset "show and the abstract supertype" begin
        d = IntervalDensityEstimate([0.0, 0.5, 1.0], [0.0, 0.5, Inf], 1.0)
        @test d isa AbstractDensityEstimate
        @test DensityEstimate([0.0], 1.0) isa AbstractDensityEstimate
        s = sprint(show, d)
        @test occursin("1.0 interval and 2.0 point observations", s)
        @test occursin("support", sprint(show, IntervalDensityEstimate([0.0], [0.5], 1.0; support=(0, 1))))
    end
end

# Literal deletion refits, the estimand of `_interval_loo`: for each interval group of the layout,
# the log-probability of its interval under the fit with one of its observations removed, and for
# each point node the log-density there with all of its observations removed.
function held_out_refit(lower, upper, κ; support=(-Inf, Inf))
    L = PenalizedDensity._interval_layout(lower, upper, κ, cbrt(eps()), float.(support)...)
    m = length(L.y)
    lp = map(L.groups) do g
        a = g.first == 0 ? -Inf : L.y[g.first]
        b = g.last == m ? Inf : L.y[g.last+1]
        # Endpoints merged into one node may differ from it by rounding.
        near(s, t) = s == t || abs(s - t) <= 1e-9 * max(1, abs(t))
        k = findfirst(i -> near(lower[i], a) && near(upper[i], b), eachindex(lower, upper))
        keep = setdiff(eachindex(lower), k)
        d = IntervalDensityEstimate(lower[keep], upper[keep], κ; support)
        ia, ib = searchsortedfirst(d.x, a), searchsortedfirst(d.x, b)
        if ib <= length(d.x) && d.x[ia] == a && d.x[ib] == b     # interval still bounded by nodes
            F, total = PenalizedDensity._node_cdf(d)
            log(sum(c -> PenalizedDensity._cell_mass(d, c), ia:ib-1) / total)
        else
            log(cdf(d, b) - cdf(d, a))
        end
    end
    pts = [i for i in 1:m if L.w[i] > 0]
    ld = map(pts) do i
        keep = [j for j in eachindex(lower, upper) if !(lower[j] == upper[j] == L.y[i])]
        logdensity(IntervalDensityEstimate(lower[keep], upper[keep], κ; support), L.y[i])
    end
    return L, lp, pts, ld
end

@testset "interval scale selection" begin
    @testset "band of the inverse of a banded SPD matrix" begin
        rng = Xoshiro(3)
        for (bw, m) in ((0, 5), (1, 1), (1, 9), (3, 12))
            H = zeros(m, m)
            for j in 1:m, i in j:min(m, j + bw)
                H[i, j] = H[j, i] = i == j ? 4 + bw + rand(rng) : randn(rng) / 2
            end
            band = zeros(bw + 1, m)
            for j in 1:m, i in j:min(m, j + bw)
                band[1+i-j, j] = H[i, j]
            end
            K = PenalizedDensity._banded_inverse_band(PenalizedDensity._banded_cholesky!(band))
            Hinv = inv(H)
            @test all(K[1+i-j, j] ≈ Hinv[i, j] for j in 1:m for i in j:min(m, j + bw))
        end
    end

    @testset "held-out values match literal deletion refits" begin
        cases = (
            (rounded_intervals(), (-Inf, Inf)),                                   # rounded
            (rounded_intervals(; cens=(1.375, 6)), (-Inf, Inf)),                  # right-censored
            (rounded_intervals(; δr=0.2, support=(0.0, Inf), seed=2), (0.0, Inf)),# bounded
            (rounded_intervals(; n=150, extra=randn(Xoshiro(4), 20), seed=4), (-Inf, Inf)),  # mixed
        )
        for ((lower, upper), support) in cases, κ in (1.0, 3.0)
            L, lp_ref, pts, ld_ref = held_out_refit(lower, upper, κ; support)
            lp, ld = PenalizedDensity._interval_loo(L)
            @test maximum(abs.(lp .- lp_ref)) < 0.02
            @test isempty(pts) || maximum(abs.(ld[pts] .- ld_ref)) < 0.02
            rs = [g.r for g in L.groups]
            score_ref = -(sum(rs .* lp_ref) + sum(L.w[pts] .* ld_ref)) / length(lower)
            @test PenalizedDensity._interval_klcv(L) ≈ score_ref atol = 2e-3
        end
        # Smoothing lengths far below the bin width: deleting an observation reshapes its own and
        # the neighboring bins, and points inside or on the edge of occupied bins reshape them.
        grid = [repeat(-1.0:0.25:1.0, 3); 6.0]
        cases = (
            ("mixed", rounded_intervals(; n=150, extra=randn(Xoshiro(4), 20), seed=4)),
            ("point inside a bin", ([grid; -4.0; 0.1], [grid .+ 0.25; -4.0; 0.1])),
            ("points on a shared edge", ([grid; 0.25; 0.25], [grid .+ 0.25; 0.25; 0.25])),
            ("rounded", rounded_intervals()),
        )
        for (_, (lower, upper)) in cases, κ in (30.0, 200.0)
            L, lp_ref, pts, ld_ref = held_out_refit(lower, upper, κ)
            lp, ld = PenalizedDensity._interval_loo(L)
            fin = isfinite.(lp_ref)       # the refit's cdf difference underflows for far bins
            @test maximum(abs.(lp[fin] .- lp_ref[fin])) < 0.02
            @test all(isfinite, lp)
            @test isempty(pts) || maximum(abs.(ld[pts] .- ld_ref)) < 0.02
        end
        # The mixed case includes intervals spanning several cells (a point inside a bin).
        L = PenalizedDensity._interval_layout(rounded_intervals(; n=150, extra=randn(Xoshiro(4), 20), seed=4)..., 1.0,
                                cbrt(eps()), -Inf, Inf)
        @test any(g -> g.last > g.first, L.groups)
    end

    @testset "point observations reproduce the point-data held-out densities" begin
        x = sort(randn(Xoshiro(5), 80))
        x = [x; x[1:10]]                          # ten locations with two copies
        for κ in (1.0, 4.0), holdout in (:location, :observation)
            L = PenalizedDensity._interval_layout(x, x, κ, cbrt(eps()), -Inf, Inf)
            _, ld = PenalizedDensity._interval_loo(L; holdout)
            _, looi = PenalizedDensity._loo_density(L.y, L.w, κ, κ, κ; holdout)
            @test ld ≈ log.(looi) rtol = 1e-10
        end
    end

    @testset "blocks of the inverse beyond the band" begin
        rng = Xoshiro(6)
        for (bw, m) in ((1, 9), (2, 10))
            H = zeros(m, m)
            for j in 1:m, i in j:min(m, j + bw)
                H[i, j] = H[j, i] = i == j ? 4 + bw + rand(rng) : randn(rng) / 2
            end
            band = zeros(bw + 1, m)
            for j in 1:m, i in j:min(m, j + bw)
                band[1+i-j, j] = H[i, j]
            end
            chol = PenalizedDensity._banded_cholesky!(band)
            K = PenalizedDensity._banded_inverse_band(chol)
            Hinv = inv(H)
            for r in (1:m, 2:6, 4:4, m-bw-1:m)
                @test PenalizedDensity._inverse_block(chol, K, r) ≈ Hinv[r, r]
            end
        end
    end

    @testset "tridiagonal-plus-rank-one local Hessians" begin
        rng = Xoshiro(8)
        k = 12
        d = 3 .+ rand(rng, k); e = randn(rng, k - 1) / 2
        # Groups share their end nodes, as adjacent groups in a local set do.
        H = PenalizedDensity._LocalHessian(d, e, [0.7, 0.0, 2.5], [1:5, 5:9, 9:12],
                                           [randn(rng, 5), randn(rng, 5), randn(rng, 4)])
        A = PenalizedDensity._dense(H)
        @test A ≈ A'
        @test A[1:4, 1:4] ≈ Matrix(SymTridiagonal(d, e))[1:4, 1:4] + 0.7 * H.n[1][1:4] * H.n[1][1:4]'
        x = randn(rng, k)
        @test PenalizedDensity._mul(H, x) ≈ A * x
        @test PenalizedDensity._solve_pd(H, x) ≈ A \ x
        dd, ee = PenalizedDensity._tri_entries(H)
        @test dd ≈ diag(A) && ee ≈ diag(A, 1)
        # An indefinite tridiagonal part is declined, even where the rank-one terms would repair it.
        Hneg = PenalizedDensity._LocalHessian([d[1:end-1]; -0.1], e, H.γ, H.rq, H.n)
        @test PenalizedDensity._solve_pd(Hneg, x) === nothing
    end

    @testset "the rest's Hessian on a local set is tridiagonal" begin
        # Exact points inside wide intervals give local sets of many nodes. The reduced Hessian
        # (K_ll)⁻¹ - H, formed from Schur complements at the two end nodes, must equal the inverse
        # of the dense block of the inverse Hessian.
        rng = Xoshiro(9)
        x = randn(rng, 200)
        exact = rand(rng, 200) .< 0.5
        lower = ifelse.(exact, x, 0.5 .* floor.(x ./ 0.5)); upper = ifelse.(exact, x, lower .+ 0.5)
        for κ in (1.0, 20.0)
            L = PenalizedDensity._interval_layout(lower, upper, κ, cbrt(eps()), -Inf, Inf)
            m = length(L.y)
            s, E = PenalizedDensity._solve_interval(L)
            v = sqrt.(s)
            chol = PenalizedDensity._banded_cholesky!(copy(E.band))
            K = PenalizedDensity._banded_inverse_band(chol)
            dr = PenalizedDensity._reverse_cholesky_diag(E.band)
            for j in eachindex(L.groups)
                js = PenalizedDensity._with_neighbors(L, [j])
                gs = L.groups[js]
                nodes = PenalizedDensity._span_nodes(gs, m)
                loc = PenalizedDensity._local_terms(L, gs, [g.r for g in gs], E.τ[js], nodes, L.w[nodes], v)
                Kll = PenalizedDensity._inverse_block(chol, K, nodes) ./ (4 .* v[nodes] .* v[nodes]')
                S = PenalizedDensity._rest_hessian(E.band, chol[1, :], dr, v, nodes, loc.H)
                ref = inv(Kll) - PenalizedDensity._dense(loc.H)
                @test Matrix(S) ≈ ref atol = 1e-6 * maximum(abs, ref)
            end
        end
    end

    @testset "held-out values of isolated observations far in the tail" begin
        # A singleton interval and a point far out: their held-out values are of order
        # e^{-2κ·gap}, far below roundoff relative to the fit, and must still track the refit.
        # The reference is the refit's right-tail mass, formed in the log domain.
        lower = [repeat(-1.0:0.25:1.0, 3); 6.0]; upper = lower .+ 0.25
        for κ in (3.0, 30.0, 200.0)
            L = PenalizedDensity._interval_layout(lower, upper, κ, cbrt(eps()), -Inf, Inf)
            lp, _ = PenalizedDensity._interval_loo(L)
            d = IntervalDensityEstimate(lower[1:end-1], upper[1:end-1], κ)
            _, total = PenalizedDensity._node_cdf(d)
            ref = 2log(d.ψ[end]) - log(2d.kR) - 2d.kR * (6.0 - d.x[end]) +
                  log(-expm1(-2d.kR * 0.25)) - log(total)
            @test lp[end] ≈ ref atol = 0.02
            @test isfinite(PenalizedDensity._interval_klcv(L))
            lo2 = [lower[1:end-1]; -4.0]
            up2 = [upper[1:end-1]; -4.0]
            L2 = PenalizedDensity._interval_layout(lo2, up2, κ, cbrt(eps()), -Inf, Inf)
            _, ld = PenalizedDensity._interval_loo(L2)
            @test ld[1] ≈ logdensity(IntervalDensityEstimate(lower[1:end-1], upper[1:end-1], κ), -4.0) atol = 0.03
        end
    end

    @testset "select_kappa_kl on rounded data" begin
        δr = 0.25
        z = round.(randn(Xoshiro(2), 200) ./ δr) .* δr
        lower, upper = PenalizedDensity._lattice_bounds(z, δr, (-Inf, Inf))
        κ = select_kappa_kl(z; resolution=δr)
        @test κ == select_kappa_kl(lower, upper)
        @test κ == select_kappa_kl(OffsetArray(z, -7); resolution=δr)
        # The minimizer of the literal-refit score on a grid.
        grid = exp.(range(log(0.5), log(30); length=25))
        refit_score(κ0) = (r = held_out_refit(lower, upper, κ0);
                           -sum(g.r * l for (g, l) in zip(r[1].groups, r[2])) / length(lower))
        κgrid = grid[argmin(refit_score.(grid))]
        @test κgrid / 1.2 < κ < κgrid * 1.2
        # The default grid stays below the fit's accuracy limit.
        @test maximum(PenalizedDensity._default_interval_κs(lower, upper)) * δr <= eps()^(-1 / 4) / 2
        @test PenalizedDensity._default_interval_κs([0.0, 0.0, 1.0], [0.0, 1.0, Inf]) isa AbstractVector
        # Censored and mixed observations, through the bounds form.
        lc, uc = rounded_intervals(; n=150, extra=randn(Xoshiro(4), 20), cens=(1.375, 6), seed=4)
        @test 0 < select_kappa_kl(lc, uc) < Inf
    end

    @testset "tie check and holdout" begin
        z = round.(randn(Xoshiro(2), 200); digits=1)
        for f in (select_kappa_kl, select_kappa_cv)
            @test_throws ArgumentError f(z)
            @test_throws "lattice of spacing δ ≈ 0.1" f(z)
            @test_throws "resolution=0" f(z)
            @test 0 < f(z; resolution=0) < Inf
        end
        @test_throws "pass `resolution=δ`" select_kappa_kl(z)
        @test_throws "select_kappa_kl(x; resolution=δ)" select_kappa_cv(z)
        @test_throws "no form for rounded (interval) data" select_kappa_cv(z; resolution=0.1)
        @test_throws "resolution must be finite and nonnegative" select_kappa_kl(z; resolution=-0.1)
        # Ties off a lattice: the message points to the bounds form.
        zt = [randn(Xoshiro(1), 100); fill(0.123, 5)]
        @test_throws "select_kappa_kl(lower, upper)" select_kappa_kl(zt)
        # At most 1% repeated values is accepted as point data.
        zs = [randn(Xoshiro(1), 200); 0.5; 0.5]
        @test 0 < select_kappa_kl(zs) < Inf
        # Distinct values: both holdouts give the same scale.
        x = randn(Xoshiro(6), 300)
        @test select_kappa_kl(x) == select_kappa_kl(x; holdout=:observation)
        @test_throws "holdout must be :location or :observation" select_kappa_kl(x; holdout=:point)
        @test_throws "holdout must be :location or :observation" select_kappa_kl(x, x; holdout=:point)
    end

    @testset "an underflowing interval edge is rejected, not an error" begin
        # A point near one edge of an occupied interval: as κ grows the density at the far edge
        # falls toward e^{-κ·distance} and the Newton solve in nodal densities overflows, well
        # inside the κ·width limit of the layout. The selector skips those scales.
        lower = [0.0, 1.0, 2.0, 0.9, 1.5, 2.5]
        upper = [1.0, 2.0, 3.0, 0.9, 1.5, 2.5]
        κs = exp.(range(log(0.5), log(400.0); length=12))
        κ = select_kappa_kl(lower, upper; κs)
        @test κ < 50
        @test κ ≈ select_kappa_kl(lower, upper; κs=filter(<(50), κs)) rtol = 1e-6
    end
end
