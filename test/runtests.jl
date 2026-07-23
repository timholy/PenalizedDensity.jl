using PenalizedDensity
using LinearAlgebra: SymTridiagonal, ZeroPivotException
using LogExpFunctions: logaddexp
using OffsetArrays
using QuadGK: quadgk
using Random, Statistics
using Random: randn!
using Test
using Aqua
using ExplicitImports

# Bytes allocated by one `_logΦ!` sweep, measured behind a function barrier: called with
# untyped arguments the sweep's `Tuple{Float64,Float64}` return is boxed at the call site,
# which would be charged to the callee. The first call compiles.
function sweep_allocated(piv, rhs, r, u)
    PenalizedDensity._logΦ!(piv, rhs, r, u)
    return @allocated PenalizedDensity._logΦ!(piv, rhs, r, u)
end

# Trapezoidal integral of a callable over a wide, fine grid.
function integrate(f, a, b; n=2_000_001)
    xs = range(a, b; length=n)
    ys = f.(xs)
    return (sum(ys) - (ys[1] + ys[end]) / 2) * step(xs)
end

# Direct Monte-Carlo of the field-theoretic χ² (Holy 1997, Eq. 16): sample the fluctuation
# field δψ from the constrained Gaussian and evaluate χ²(δψ). This is the ground truth the
# exact reference distribution must reproduce, at a constant or a varying scale alike, on ℝ or
# on a bounded support.
#
# The field's precision is L = 2λ𝒜 + 2Σ(wᵢ/ψᵢ²)δ(x-xᵢ), 𝒜u = -(κ(x)⁻²u′)′ + u, discretized
# conservatively: each cell of the grid contributes (2λ/κ²)(Δδψ)²/h to the gradient energy
# and 2λ·h to the mass. The grid contains every node, so κ is single-valued on each cell and
# the point sources land on grid points exactly. A finite `d.lo`/`d.hi` terminates the grid
# there instead of padding into a decaying tail; the conservative stencil already has no term
# coupling a grid edge to anything past it, so ending the grid exactly at the wall *is* the
# natural (Neumann) boundary condition, not an approximation to it. Nothing here shares an
# assembly, a Green's function, or a recursion with the package. Returns the χ² samples.
function fieldmc_chisq(d; nsamp=40_000, per_len=50, pad=12.0, seed=1)
    x, ψ, λ = d.x, d.ψ, d.λ
    κ(k) = k == 0 ? d.κL : k > length(d.x) - 1 ? d.κR : PenalizedDensity._kappa(d.κ, k)
    bnds = vcat(isfinite(d.lo) ? d.lo : first(x) - pad / d.κL, x,
                isfinite(d.hi) ? d.hi : last(x) + pad / d.κR)
    y = Float64[]
    for k in 1:length(bnds)-1                       # ≈ per_len points per local length 1/κ
        npts = max(2, ceil(Int, (bnds[k+1] - bnds[k]) * κ(k - 1) * per_len))
        append!(y, range(bnds[k], bnds[k+1]; length=npts + 1)[1:end-1])
    end
    push!(y, last(bnds)); m = length(y); δ = diff(y)
    ψc = amplitude(d, y)
    gid = [searchsortedfirst(y, xi) for xi in x]
    cell = [searchsortedlast(bnds, (y[j] + y[j+1]) / 2) - 1 for j in 1:m-1]
    mass = [(j > 1 ? δ[j-1] : 0.0) / 2 + (j < m ? δ[j] : 0.0) / 2 for j in 1:m]
    p = 2λ .* mass; q = zeros(m - 1)
    for j in 1:m-1
        c = 2λ / (κ(cell[j])^2 * δ[j]); p[j] += c; p[j+1] += c; q[j] = -c
    end
    for (gi, s) in zip(gid, 2 .* d.w ./ ψ.^2); p[gi] += s; end
    # Cholesky P = RᵀR of the tridiagonal precision, R upper-bidiagonal (c diag, b super).
    c = similar(p); b = similar(q); c[1] = sqrt(p[1])
    for j in 2:m; b[j-1] = q[j-1] / c[j-1]; c[j] = sqrt(p[j] - b[j-1]^2); end
    solveR!(v, ξ) = (v[m] = ξ[m] / c[m]; for j in m-1:-1:1; v[j] = (ξ[j] - b[j] * v[j+1]) / c[j]; end; v)
    solveP!(v, a) = (u = similar(a); u[1] = a[1] / c[1];
        for j in 2:m; u[j] = (a[j] - b[j-1] * u[j-1]) / c[j]; end;
        v[m] = u[m] / c[m]; for j in m-1:-1:1; v[j] = (u[j] - b[j] * v[j+1]) / c[j]; end; v)
    mψ = mass .* ψc                                  # the ∫ψ_cl δψ = 0 constraint
    v = solveP!(similar(ψc), mψ); aPa = sum(mψ .* v)
    rng = MersenneTwister(seed)
    z = similar(ψc); ξ = similar(ψc); h = similar(ψc); iψ2 = d.w ./ ψ.^2
    chis = Vector{Float64}(undef, nsamp)
    for s in 1:nsamp
        randn!(rng, ξ); solveR!(z, ξ)
        proj = sum(mψ .* z) / aPa; @. h = z - proj * v
        chis[s] = 4 * sum(iψ2[k] * h[gid[k]]^2 for k in eachindex(gid))
    end
    return chis
end

# 4 Σᵢ wᵢ bᵢ/ψᵢ, which Green's identity pins to 1 for any scale, bounded or not (it needs only
# Lψ_cl = 4Σ(wᵢ/ψᵢ)δ(x-xᵢ) and ∫ψ_cl² = 1). A wrong nodal precision or a wrong α breaks it.
function green_identity(d)
    m = PenalizedDensity._node_alpha(d.x, d.ψ, d.κ, d.κL, d.κR, d.λ, d.lo, d.hi)
    M = PenalizedDensity._operator(d.x, d.κ, d.κL, d.κR, d.lo, d.hi)
    f = 2d.λ / PenalizedDensity._reference_scale(d.κ, d.κL, d.κR)
    S = 2 .* d.w ./ d.ψ.^2
    b = SymTridiagonal(f .* M.dv .+ S, f .* M.ev) \ (f .* (M * m))
    return 4 * sum(d.w .* b ./ d.ψ)
end

@testset "PenalizedDensity.jl" begin
    @testset "single point is a Laplace density" begin
        κ = 1.5
        d = DensityEstimate([2.0], κ)
        @test d.ψ[1] ≈ sqrt(κ)
        @test d(2.0) ≈ κ                       # Q(x₀) = κ
        @test d(3.0) ≈ κ * exp(-2κ)            # Q(x) = κ e^{-2κ|x-x₀|}
        @test d(1.0) ≈ κ * exp(-2κ)            # symmetric
        @test integrate(d, -30, 30) ≈ 1 atol = 1e-8
    end

    @testset "normalization, shape, and `show`" begin
        d = DensityEstimate([-1.0, 0.0, 0.0, 1.0], 1.0)
        @test integrate(d, -30, 30) ≈ 1 atol = 1e-8
        @test d(0.0) > d(2.0)                  # denser near the data
        @test all(≥(0), d.(range(-5, 5; length=101)))   # Q = ψ² ≥ 0
        @test startswith(sprint(show, d), "DensityEstimate with 3 distinct nodes, 4.0 total weight, κ=1.0, λ=2.8163")
    end

    @testset "continuity at the nodes" begin
        d = DensityEstimate([0.0, 1.3, 2.0, 5.1], 0.7)
        for xi in d.x
            @test amplitude(d, xi - 1e-9) ≈ amplitude(d, xi + 1e-9) atol = 1e-6
        end
    end

    @testset "repeated points equal integer weights" begin
        # Merging identical points must reproduce the weighted problem.
        d1 = DensityEstimate([0.0, 0.0, 0.0, 4.0], 1.2)
        @test d1.x == [0.0, 4.0]
        @test d1.w == [3.0, 1.0]
        @test integrate(d1, -30, 40) ≈ 1 atol = 1e-8
        # Heavier weight ⇒ larger amplitude at that node.
        @test d1.ψ[1] > d1.ψ[2]
    end

    @testset "rtol merges points within rtol/κ" begin
        d = DensityEstimate([0.0, 1e-10, 1.0], 1.0; rtol=1e-6)
        @test length(d.x) == 2
        @test d.w == [2.0, 1.0]
        # The threshold is rtol/κ: here 1e-3/2 = 5e-4.
        @test length(DensityEstimate([0.0, 1e-4, 1.0], 2.0; rtol=1e-3).x) == 2
        @test length(DensityEstimate([0.0, 1e-3, 1.0], 2.0; rtol=1e-3).x) == 3
        # The default rtol > 0 merges points far below the resolution (numerical hygiene).
        @test length(DensityEstimate([0.0, 1e-9, 1.0], 1.0).x) == 2
        # Merging points closer than the resolution is lossless.
        Random.seed!(5)
        x = randn(5_000)
        da = DensityEstimate(x, 3.0)
        db = DensityEstimate(x, 3.0; rtol=1e-3)
        g = range(-4, 4; length=101)
        @test maximum(abs.(da.(g) .- db.(g))) < 1e-3
    end

    @testset "solver is robust and allocates O(N), not O(N·iterations)" begin
        # Dense samples drive adjacent-node spacings toward zero, which sharply
        # ill-conditions the tridiagonal Newton system; the fit must still converge
        # and normalise, with memory proportional to N rather than N × iterations.
        # rtol=0 exercises the raw solver without the near-coincident-point merge.
        Random.seed!(4)
        x = randn(20_000)
        d = DensityEstimate(x, 3.0; rtol=0.0)
        @test all(isfinite, d.ψ) && all(>(0), d.ψ)
        @test integrate(d, -40, 40) ≈ 1 atol = 1e-6
        # Stationarity of the normalized amplitude: Mψ = (κ/λ) w ./ ψ (Eq. field equation).
        M = PenalizedDensity.roughness_operator(d.x, d.κ)
        resid = M * d.ψ .- (d.κ / d.λ) .* d.w ./ d.ψ
        @test maximum(abs, resid) < 1e-6 * maximum(abs, M * d.ψ)
        DensityEstimate(x, 3.0; rtol=0.0)           # compile before measuring
        @test (@allocated DensityEstimate(x, 3.0; rtol=0.0)) < 40 * length(x) * sizeof(Float64)
    end

    @testset "scale equivariance" begin
        # Q is scale-equivariant: rescaling x → s·x with κ → κ/s gives Q_s(s·x) = Q(x)/s.
        # M, the Newton solve, and the convergence test depend on x, κ only through κ·Δx,
        # so the unnormalized fit and its stopping criterion are invariant under this scaling.
        Random.seed!(11)
        x = randn(2_000)
        d = DensityEstimate(x, 3.0)
        xt = range(-3, 3; length=25)
        Q = d.(xt)
        for s in (1e-15, 1e20)
            ds = DensityEstimate(s .* x, 3.0 / s)
            @test maximum(abs.(ds.(s .* xt) .- Q ./ s) ./ (Q ./ s)) < 1e-8
        end
    end

    @testset "λ ≈ N near the optimal κ (Eq. 10 asymptotics)" begin
        Random.seed!(1)
        x = randn(400)
        κ = select_kappa_ms(x; κs=exp.(range(log(0.05), log(20); length=40)))
        d = DensityEstimate(x, κ)
        @test 0.75 < d.λ / length(x) < 1.25    # paper: λ ≈ N
        @test integrate(d, -40, 40) ≈ 1 atol = 1e-6
        φ(t) = exp(-t^2 / 2) / sqrt(2π)         # recovers a standard normal
        @test mean(abs.(d.(range(-4, 4; length=161)) .- φ.(range(-4, 4; length=161)))) < 0.05
    end

    @testset "analytic dS/dln κ matches finite differences" begin
        Random.seed!(6)
        nodes, w = PenalizedDensity._merge_presorted(sort(randn(1000)), 0.0)
        Sof(κ) = action(PenalizedDensity._fit(nodes, w, κ))
        for κ in (1.5, 5.0, 15.0)
            S, slope = PenalizedDensity._action_and_slope(nodes, w, κ)
            @test S ≈ Sof(κ)
            δ = 1e-5
            fd = (Sof(κ * exp(δ)) - Sof(κ * exp(-δ))) / (2δ)     # dS/dln κ
            @test slope ≈ fd rtol = 1e-3
        end
    end

    @testset "select_kappa_ms: analytic, grid-independent" begin
        Random.seed!(2)
        x = randn(1500)
        κ = select_kappa_ms(x)                                       # data-scaled default grid
        @test κ > 0
        # Golden-section refinement makes the result independent of the bracketing grid.
        κ10  = select_kappa_ms(x; κs = exp.(range(log(0.5), log(60); length = 10)))
        κ200 = select_kappa_ms(x; κs = exp.(range(log(0.5), log(60); length = 200)))
        @test κ10 ≈ κ200 rtol = 1e-3
        @test κ ≈ κ200 rtol = 1e-3
        # The returned κ is a local minimum of |dS/dln κ|.
        slope(κ) = abs(last(PenalizedDensity._action_and_slope(sort(x), ones(1500), κ)))
        @test slope(κ) < slope(κ * 1.3) && slope(κ) < slope(κ / 1.3)
    end

    @testset "select_kappa_cv: cross-validated (MISE) scale" begin
        Random.seed!(11)
        x = randn(1500)
        κ = select_kappa_cv(x)
        @test κ > 0
        # Golden-section refinement ⇒ independent of the bracketing grid.
        κ10  = select_kappa_cv(x; κs = exp.(range(log(0.3), log(40); length = 10)))
        κ200 = select_kappa_cv(x; κs = exp.(range(log(0.3), log(40); length = 200)))
        @test κ10 ≈ κ200 rtol = 1e-3

        # ∫Q̂² term: the analytic closed form matches numerical quadrature, including the
        # near-coincident points (small θ) that defeat a naive csch⁴ form.
        d = DensityEstimate(x, 4.0)
        ∫Q2 = PenalizedDensity._int_quartic(d.x, d.ψ, d.κ)
        g = range(d.x[1] - 15 / d.κ, d.x[end] + 15 / d.κ; length = 400_001)
        @test ∫Q2 ≈ sum(d(t)^2 for t in g) * step(g) rtol = 1e-4

        # LSCV score: the first-order analytic leave-one-out matches brute-force refitting.
        xs = sort(x); w = ones(length(xs))
        function lscv_refit(nodes, weights, κ)
            cross = 0.0
            for i in eachindex(nodes)
                if weights[i] > 1
                    w2 = copy(weights); w2[i] -= 1; n2 = nodes
                else
                    keep = [j for j in eachindex(nodes) if j != i]
                    n2 = nodes[keep]; w2 = weights[keep]
                end
                cross += weights[i] * PenalizedDensity._fit(collect(n2), collect(w2), κ)(nodes[i])
            end
            di = PenalizedDensity._fit(copy(nodes), copy(weights), κ)
            return PenalizedDensity._int_quartic(di.x, di.ψ, di.κ) - 2cross / sum(weights)
        end
        for κ0 in (1.5, 4.0, 12.0)
            @test PenalizedDensity._lscv(xs, w, κ0) ≈ lscv_refit(xs, w, κ0) rtol = 5e-3
        end

        # MISE targeting: on smooth data the cross-validated scale is finer than the
        # minimum-sensitivity scale and gives a smaller integrated squared error.
        Q(t) = exp(-t^2 / 2) / sqrt(2π)
        ise(κ0) = (gg = range(-8, 8; length = 6001);
                   dd = DensityEstimate(x, κ0);
                   step(gg) * sum((dd(t) - Q(t))^2 for t in gg))
        @test κ < select_kappa_ms(x)
        @test ise(κ) < ise(select_kappa_ms(x))

        # Generic indexing: OffsetArray input gives an identical result.
        @test select_kappa_cv(OffsetArray(x, -750)) == κ
    end

    @testset "select_kappa_kl: likelihood (KL) scale" begin
        Random.seed!(11)
        x = randn(1500)
        κ = select_kappa_kl(x)
        @test κ > 0
        # Golden-section refinement ⇒ independent of the bracketing grid.
        κ10  = select_kappa_kl(x; κs = exp.(range(log(0.3), log(40); length = 10)))
        κ200 = select_kappa_kl(x; κs = exp.(range(log(0.3), log(40); length = 200)))
        @test κ10 ≈ κ200 rtol = 1e-3

        # KLCV score: the mean negative leave-one-out log-likelihood built from the first-order
        # analytic leave-one-out densities matches one built by brute-force refitting. The
        # tolerance is looser than the LSCV analog above: taking the log gives every node equal
        # weight, including tail nodes where the first-order expansion is least accurate, whereas
        # LSCV's cross term downweights them by the density value.
        xs = sort(x); w = ones(length(xs))
        function klcv_refit(nodes, weights, κ)
            s = 0.0
            for i in eachindex(nodes)
                keep = [j for j in eachindex(nodes) if j != i]
                s += weights[i] * log(PenalizedDensity._fit(nodes[keep], weights[keep], κ)(nodes[i]))
            end
            return -s / sum(weights)
        end
        for κ0 in (1.5, 4.0, 12.0)
            @test PenalizedDensity._klcv(xs, w, κ0) ≈ klcv_refit(xs, w, κ0) rtol = 2e-2
        end

        # Divergence targeting: on smooth data the KL scale is finer than the minimum-sensitivity
        # scale and gives a smaller Kullback–Leibler divergence to the truth.
        Q(t) = exp(-t^2 / 2) / sqrt(2π)
        kl(κ0) = (gg = range(-8, 8; length = 6001);
                  dd = DensityEstimate(x, κ0);
                  step(gg) * sum(Q(t) * log(Q(t) / dd(t)) for t in gg))
        @test κ < select_kappa_ms(x)
        @test kl(κ) < kl(select_kappa_ms(x))

        # Tracks the least-squares (MISE) scale to leading order.
        @test select_kappa_kl(x) ≈ select_kappa_cv(x) rtol = 0.5

        # Generic indexing: OffsetArray input gives an identical result.
        @test select_kappa_kl(OffsetArray(x, -750)) == κ
    end

    @testset "amplitude² == density" begin
        d = DensityEstimate([-2.0, 0.5, 3.0], 0.9)
        g = range(-4, 5; length=25)
        @test amplitude(d, collect(g)) .^ 2 ≈ d.(g)
    end

    @testset "cdf and quantile" begin
        # Single point: exactly Laplace(x₀, rate 2κ), where cdf and quantile are elementary.
        κ = 1.5; x0 = 2.0
        d = DensityEstimate([x0], κ)
        lap_cdf(x) = x <= x0 ? exp(2κ * (x - x0)) / 2 : 1 - exp(-2κ * (x - x0)) / 2
        lap_q(q) = q <= 1/2 ? x0 + log(2q) / (2κ) : x0 - log(2 * (1 - q)) / (2κ)
        for x in (-3.0, 0.0, 2.0, 2.5, 7.0)
            @test cdf(d, x) ≈ lap_cdf(x) rtol = 1e-14
        end
        for q in (1e-9, 0.1, 0.5, 0.9, 1 - 1e-9)
            @test quantile(d, q) ≈ lap_q(q) atol = 1e-13
        end

        # Limits, NaN, and domain errors.
        @test cdf(d, -Inf) == 0 && cdf(d, Inf) == 1
        @test quantile(d, 0) == -Inf && quantile(d, 1) == Inf
        @test isnan(cdf(d, NaN))
        @test_throws DomainError quantile(d, -0.1)
        @test_throws DomainError quantile(d, 1.1)
        @test_throws DomainError quantile(d, NaN)

        # Multi-node fits: the closed-form cdf matches quadrature of the density (nodes
        # passed as breakpoints so the quadrature resolves every narrow feature), across
        # smoothing regimes from smooth (θ ≪ 1) to isolated points (θ ≫ 1).
        Random.seed!(21)
        x = randn(60)
        for κ in (0.5, 3.0, 40.0)
            d = DensityEstimate(x, κ)
            lo, hi = d.x[1], d.x[end]
            for t in range(lo - 8 / κ, hi + 8 / κ; length = 41)
                ref = quadgk(d, -Inf, filter(<(t), d.x)..., t; rtol = 1e-13)[1]
                @test cdf(d, t) ≈ ref atol = 1e-12
            end
            # Probability round trip: cdf ∘ quantile is the identity to roundoff, with
            # relative accuracy maintained deep into both tails.
            for q in vcat(exp10.(-14:2:-2), 0.1:0.1:0.9, 1 .- exp10.(-14:2:-2))
                @test cdf(d, quantile(d, q)) ≈ q rtol = 1e-12
            end
            # x round trip wherever the density is not vanishingly small (where the CDF
            # is numerically flat, x cannot be recovered from it).
            for t in range(lo - 5 / κ, hi + 5 / κ; length = 101)
                d(t) > 1e-8 || continue
                @test quantile(d, cdf(d, t)) ≈ t atol = 1e-9
            end
            # Monotone: nondecreasing cdf on a fine grid, nondecreasing quantiles.
            g = collect(range(lo - 10 / κ, hi + 10 / κ; length = 2001))
            @test issorted(cdf(d, g))
            @test issorted(quantile(d, collect(0.0:0.0005:1.0)))
            # Continuity across the nodes.
            for xk in d.x
                @test cdf(d, prevfloat(xk)) ≈ cdf(d, xk) atol = 32eps()
                @test cdf(d, nextfloat(xk)) ≈ cdf(d, xk) atol = 32eps()
            end
        end

        # An isolated far node (θ = κ·Δx = 240): every branch stays finite and the
        # probability round trip survives the numerically flat valley.
        di = DensityEstimate([0.0, 1.0, 2.0, 50.0], 5.0)
        @test all(isfinite, quantile(di, collect(0.001:0.001:0.999)))
        for q in (1e-9, 0.01, 0.3, 0.7, 0.99, 1 - 1e-9)
            @test cdf(di, quantile(di, q)) ≈ q rtol = 1e-12
        end

        # Near-coincident nodes (θ ≈ 1e-3, kept unmerged with rtol = 0).
        dn = DensityEstimate([0.0, 1e-3, 1.0, 2.0], 1.0; rtol = 0.0)
        for t in range(-2, 4; length = 41)
            ref = quadgk(dn, -Inf, filter(<(t), dn.x)..., t; rtol = 1e-13)[1]
            @test cdf(dn, t) ≈ ref atol = 1e-12
        end

        # Affine equivariance: x → s·x + b with κ → κ/s maps the fit onto itself, so cdf
        # values are preserved and quantiles transform affinely.
        Random.seed!(33)
        xa = randn(500)
        d0 = DensityEstimate(xa, 2.5)
        for (s, b) in ((1e-8, 3.0), (1e6, -20.0))
            ds = DensityEstimate(s .* xa .+ b, 2.5 / s)
            for t in range(-3, 3; length = 21)
                @test cdf(ds, s * t + b) ≈ cdf(d0, t) atol = 1e-7
            end
            for q in (0.01, 0.2, 0.5, 0.8, 0.99)
                @test quantile(ds, q) ≈ s * quantile(d0, q) + b rtol = 1e-9
            end
        end

        # Float32 fits give Float32 results that agree with the Float64 fit.
        d32 = DensityEstimate(Float32[-1, 0, 1], 1.0f0)
        d64 = DensityEstimate([-1.0, 0.0, 1.0], 1.0)
        @test @inferred(cdf(d32, 0.5f0)) isa Float32
        @test @inferred(quantile(d32, 0.5f0)) isa Float32
        @test cdf(d32, 0.5f0) ≈ cdf(d64, 0.5) atol = 1e-5
        @test quantile(d32, 0.25f0) ≈ quantile(d64, 0.25) atol = 1e-4

        # Generic indexing: array methods preserve the argument's axes.
        xo = OffsetArray([-0.5, 0.0, 0.5], -2)
        @test axes(cdf(d64, xo)) == axes(xo)
        @test parent(cdf(d64, xo)) == cdf(d64, [-0.5, 0.0, 0.5])
        qo = OffsetArray([0.1, 0.5, 0.9], 7)
        @test axes(quantile(d64, qo)) == axes(qo)
        @test parent(quantile(d64, qo)) == quantile(d64, [0.1, 0.5, 0.9])
    end

    @testset "gaussianize: transport, inverse, and log-Jacobian" begin
        Φ(t) = PenalizedDensity._Φ(t)
        φ(t) = exp(-t^2 / 2) / sqrt(2π)

        # Single point: exactly Laplace(x₀, rate 2κ), whose Gaussianizing map has an
        # elementary reference through Φ.
        κ = 1.5; x0 = 2.0
        d = DensityEstimate([x0], κ)
        lap_cdf(x) = x <= x0 ? exp(2κ * (x - x0)) / 2 : 1 - exp(-2κ * (x - x0)) / 2
        for x in (-3.0, 0.0, 2.5, 7.0)
            @test Φ(gaussianize(d, x)) ≈ lap_cdf(x) rtol = 1e-13
        end
        @test abs(gaussianize(d, x0)) < 1e-8          # median maps to ≈ 0
        # The map's derivative at the median: Q(x₀)/φ(0) = κ√(2π).
        @test gaussianize_gradient(d, x0) ≈ κ * sqrt(2π) rtol = 1e-8

        # Limits, NaN.
        @test gaussianize(d, -Inf) == -Inf && gaussianize(d, Inf) == Inf
        @test ungaussianize(d, -Inf) == -Inf && ungaussianize(d, Inf) == Inf
        @test isnan(gaussianize(d, NaN)) && isnan(ungaussianize(d, NaN))
        nanlj = gaussianize_logjacobian(d, NaN)
        @test isnan(nanlj.y) && isnan(nanlj.logjac)
        @test gaussianize_logjacobian(d, Inf).logjac == -Inf   # dy/dx → 0 in the tails
        @test gaussianize_gradient(d, Inf) == 0

        # Multi-node fits across smoothing regimes, from smooth (θ ≪ 1) to isolated
        # points (θ ≫ 1).
        Random.seed!(21)
        x = randn(60)
        for κ in (0.5, 3.0, 40.0)
            d = DensityEstimate(x, κ)
            lo, hi = d.x[1], d.x[end]
            # Definition: Φ(gaussianize) reproduces the cdf at full relative precision.
            for t in range(lo - 8 / κ, hi + 8 / κ; length = 41)
                c = cdf(d, t)
                1e-12 < c < 1 - 1e-12 || continue
                @test Φ(gaussianize(d, t)) ≈ c rtol = 1e-12
            end
            # Inverse consistency with quantile ∘ Φ where Φ(y) carries full precision.
            for y in (-2.0, -0.5, 0.0, 1.0, 2.0)
                @test ungaussianize(d, y) ≈ quantile(d, Φ(y)) atol = 1e-10
            end
            # y-side round trip, from the bulk to far beyond linear-space Φ (|y| > 38.6
            # underflows Φ; the log-space tail forms carry it without saturation).
            for y in vcat(-37.5:2.5:37.5, [-300.0, -50.0, 50.0, 300.0])
                @test gaussianize(d, ungaussianize(d, y)) ≈ y rtol = 1e-11 atol = 1e-11
            end
            # x-side round trip wherever the density is not vanishingly small (where the
            # CDF is numerically flat, x cannot be recovered from it) …
            for t in range(lo - 5 / κ, hi + 5 / κ; length = 101)
                d(t) > 1e-8 || continue
                @test ungaussianize(d, gaussianize(d, t)) ≈ t atol = 1e-9
            end
            # … and deep into both exponential tails, where it is closed-form.
            for t in (lo - 500 / κ, lo - 50 / κ, hi + 50 / κ, hi + 500 / κ)
                @test ungaussianize(d, gaussianize(d, t)) ≈ t rtol = 1e-13
            end
            # Monotone on a fine grid spanning both tails.
            g = collect(range(lo - 10 / κ, hi + 10 / κ; length = 2001))
            @test issorted(gaussianize(d, g))
            # Derivative against central finite differences, interior and tails.
            for t in vcat(collect(range(lo - 3 / κ, hi + 3 / κ; length = 41)),
                          lo - 50 / κ, hi + 50 / κ)
                h = 1e-6 / κ
                fd = (gaussianize(d, t + h) - gaussianize(d, t - h)) / (2h)
                @test gaussianize_gradient(d, t) ≈ fd rtol = 1e-4
            end
            # The log-Jacobian is the log-derivative, and it makes the change of
            # variables exact: ln φ(y) + logjac = ln Q̂(x).
            for t in range(lo - 3 / κ, hi + 3 / κ; length = 41)
                (; y, logjac) = gaussianize_logjacobian(d, t)
                @test y == gaussianize(d, t)
                @test exp(logjac) ≈ gaussianize_gradient(d, t) rtol = 1e-13
                @test log(φ(y)) + logjac ≈ log(d(t)) rtol = 1e-9
            end
            # Deep-tail log-Jacobian: finite, and consistent with the map's slope.
            for t in (lo - 200 / κ, hi + 200 / κ)
                lj = gaussianize_logjacobian(d, t).logjac
                @test isfinite(lj)
                h = 1e-5 / κ
                fd = (gaussianize(d, t + h) - gaussianize(d, t - h)) / (2h)
                @test lj ≈ log(fd) rtol = 1e-4
            end
        end

        # Distributional: draws from the fit (inverse-CDF sampling) Gaussianize to
        # N(0, 1) — moments and an equiprobable-bin χ² against the standard normal.
        Random.seed!(77)
        xtr = randn(1000)
        df = DensityEstimate(xtr, select_kappa_kl(xtr))
        u = rand(5000)
        y = gaussianize(df, quantile(df, u))
        m = mean(y); v = var(y)
        @test abs(m) < 0.05 && abs(v - 1) < 0.07
        counts = [sum(0.1 * (b - 1) .<= Φ.(y) .< 0.1 * b) for b in 1:10]
        @test sum((counts .- 500) .^ 2 ./ 500) < 30     # χ²₉; multinomial noise only
        # Fresh draws from the underlying truth Gaussianize approximately (fit error).
        yfresh = gaussianize(df, randn(5000))
        @test abs(mean(yfresh)) < 0.1 && abs(var(yfresh) - 1) < 0.15

        # ungaussianize(d, Z) with Z ~ N(0,1) samples the fit: compare tail masses.
        z = randn(5000)
        xs = ungaussianize(df, z)
        @test mean(xs .< quantile(df, 0.25)) ≈ 0.25 atol = 0.03

        # An isolated far node (θ = κ·Δx = 240): every branch stays finite, monotone,
        # and invertible across the numerically flat valley.
        di = DensityEstimate([0.0, 1.0, 2.0, 50.0], 5.0)
        yi = gaussianize(di, collect(range(-3.0, 53.0; length = 501)))
        @test all(isfinite, yi) && issorted(yi)
        for y in (-40.0, -5.0, 0.0, 5.0, 40.0)
            @test gaussianize(di, ungaussianize(di, y)) ≈ y atol = 1e-11
        end

        # Affine equivariance: x → s·x + b with κ → κ/s preserves y exactly and shifts
        # the log-Jacobian by -ln s.
        Random.seed!(33)
        xa = randn(500)
        d0 = DensityEstimate(xa, 2.5)
        for (s, b) in ((1e-8, 3.0), (1e6, -20.0))
            ds = DensityEstimate(s .* xa .+ b, 2.5 / s)
            for t in range(-3, 3; length = 21)
                @test gaussianize(ds, s * t + b) ≈ gaussianize(d0, t) atol = 1e-7
                @test gaussianize_logjacobian(ds, s * t + b).logjac ≈
                      gaussianize_logjacobian(d0, t).logjac - log(s) atol = 1e-6
            end
        end

        # Float32 fits give Float32 results that agree with the Float64 fit, including
        # the deep-tail branch (Φ(y) underflows Float32 below y ≈ -9.3).
        d32 = DensityEstimate(Float32[-1, 0, 1], 1.0f0)
        d64 = DensityEstimate([-1.0, 0.0, 1.0], 1.0)
        @test @inferred(gaussianize(d32, 0.5f0)) isa Float32
        @test @inferred(ungaussianize(d32, 0.5f0)) isa Float32
        @test @inferred(gaussianize_gradient(d32, 0.5f0)) isa Float32
        @test @inferred(gaussianize_logjacobian(d32, 0.5f0)).logjac isa Float32
        @test gaussianize(d32, 0.5f0) ≈ gaussianize(d64, 0.5) atol = 1e-5
        y32 = gaussianize(d32, -30.0f0)
        @test y32 isa Float32 && isfinite(y32)
        @test ungaussianize(d32, y32) ≈ -30.0f0 rtol = 1e-6

        # Generic indexing: array methods preserve the argument's axes.
        xo = OffsetArray([-0.5, 0.0, 0.5], -2)
        @test axes(gaussianize(d64, xo)) == axes(xo)
        @test parent(gaussianize(d64, xo)) == gaussianize(d64, [-0.5, 0.0, 0.5])
        yo = OffsetArray([-1.0, 0.0, 1.0], 7)
        @test axes(ungaussianize(d64, yo)) == axes(yo)
        @test parent(ungaussianize(d64, yo)) == ungaussianize(d64, [-1.0, 0.0, 1.0])
        ljo = gaussianize_logjacobian(d64, xo)
        @test axes(ljo.y) == axes(xo) && axes(ljo.logjac) == axes(xo)
        @test parent(ljo.y) == gaussianize(d64, [-0.5, 0.0, 0.5])
        @test axes(gaussianize_gradient(d64, xo)) == axes(xo)
        @test parent(gaussianize_gradient(d64, xo)) == gaussianize_gradient(d64, [-0.5, 0.0, 0.5])
    end

    @testset "gaussianize: bounded support and per-interval κ" begin
        Φ(t) = PenalizedDensity._Φ(t)
        φ(t) = exp(-t^2 / 2) / sqrt(2π)

        # A fit is Gaussianized correctly when Φ(gaussianize) reproduces its own cdf, the map
        # inverts, and ln φ(y) + logjac = ln Q̂(x) — the change of variables. These held only for
        # the unbounded, scalar-κ fit; they must now hold for a finite support and a
        # per-interval κ as well, over the interior *and* the boundary segments.
        Random.seed!(202)
        xb = clamp.(0.5 .+ 0.18 .* randn(400), 1e-3, 1 - 1e-3)
        adaptive = DensityEstimate(randn(400) .^ 2 .+ 0.1 .* randn(400),
                                   select_kappa_adaptive(randn(400) .^ 2))
        fits = (("bounded [0,1]",     DensityEstimate(xb, 8.0; support = (0.0, 1.0))),
                ("one-sided [0,Inf)", DensityEstimate(xb, 8.0; support = (0.0, Inf))),
                ("one-sided (-Inf,1]",DensityEstimate(xb, 8.0; support = (-Inf, 1.0))),
                ("per-interval κ",    adaptive))
        for (label, d) in fits
            @testset "$label" begin
                lo = isfinite(d.lo) ? d.lo : d.x[1] - 3
                hi = isfinite(d.hi) ? d.hi : d.x[end] + 3
                # cdf identity, sampled across the whole support including both boundary segments.
                for t in range(lo, hi; length = 121)
                    c = cdf(d, t)
                    1e-12 < c < 1 - 1e-12 || continue
                    @test Φ(gaussianize(d, t)) ≈ c rtol = 1e-11
                end
                # Round trip wherever the density is resolvable.
                for t in range(lo, hi; length = 151)
                    d(t) > 1e-8 || continue
                    @test ungaussianize(d, gaussianize(d, t)) ≈ t atol = 1e-9
                end
                # Change of variables and the log-derivative, inset from the walls where y is finite.
                for t in range(lo + 0.03 * (hi - lo), hi - 0.03 * (hi - lo); length = 41)
                    (; y, logjac) = gaussianize_logjacobian(d, t)
                    @test y == gaussianize(d, t)
                    # The cdf behind y is a quadrature at rtol = √eps, so the identity
                    # cannot hold tighter than that; most points land near 1e-10, a few
                    # near the quadrature tolerance itself.
                    @test log(φ(y)) + logjac ≈ log(d(t)) rtol = 1e-7
                end
                # Monotone across the support.
                @test issorted(gaussianize(d, collect(range(lo, hi; length = 1001))))
            end
        end

        # Out-of-support convention for a compact support: honest ±Inf, and the inverse maps a
        # saturated Gaussian back to the wall.
        db = DensityEstimate(xb, 8.0; support = (0.0, 1.0))
        @test gaussianize(db, -0.05) == -Inf && gaussianize(db, 1.05) == Inf
        @test gaussianize_logjacobian(db, -0.05).logjac == -Inf
        @test gaussianize_logjacobian(db, 1.05).logjac == -Inf
        @test ungaussianize(db, -50.0) == 0.0 && ungaussianize(db, 50.0) == 1.0
        # A one-sided support keeps the honest exponential tail on the unbounded side.
        dr = DensityEstimate(xb, 8.0; support = (0.0, Inf))
        @test gaussianize(dr, -0.05) == -Inf
        @test isfinite(gaussianize(dr, 5.0)) && gaussianize(dr, Inf) == Inf

        # Distributional: draws from a bounded fit Gaussianize to N(0, 1).
        Random.seed!(303)
        u = rand(4000)
        du = DensityEstimate(u, select_kappa_kl(u); support = (0.0, 1.0))
        y = gaussianize(du, u)
        @test abs(mean(y)) < 0.05 && abs(var(y) - 1) < 0.07
        counts = [sum(0.1 * (b - 1) .<= Φ.(y) .< 0.1 * b) for b in 1:10]
        @test sum((counts .- 400) .^ 2 ./ 400) < 30     # χ²₉; multinomial noise only

        # Generic indexing on a bounded fit: array methods preserve axes.
        xo = OffsetArray([0.2, 0.5, 0.8], -2)
        @test axes(gaussianize(du, xo)) == axes(xo)
        @test parent(gaussianize(du, xo)) == gaussianize(du, [0.2, 0.5, 0.8])
    end

    @testset "generic indexing: OffsetArray input" begin
        x = [-1.5, 0.2, 0.2, 1.1, 3.4]
        d = DensityEstimate(x, 1.1)
        do_ = DensityEstimate(OffsetArray(x, -3), 1.1)   # 0-based-ish axes
        @test do_.x == d.x
        @test do_.ψ ≈ d.ψ
        @test do_(0.7) ≈ d(0.7)

        κfun(t) = 1.1 * (1 + exp(-t^2))
        a = DensityEstimate(x, κfun)
        ao = DensityEstimate(OffsetArray(x, -3), κfun)
        @test ao.x == a.x
        @test ao.κ == a.κ
        @test ao.ψ ≈ a.ψ
        @test ao(0.7) ≈ a(0.7)
    end

    @testset "adaptive κ: a callable scale" begin
        x = sort!(randn(Xoshiro(7), 2000) .* 1.5)

        @testset "a constant callable reproduces the scalar fit" begin
            for κ in (0.5, 5.0, 25.0, 200.0)
                d = DensityEstimate(x, κ)
                a = DensityEstimate(x, _ -> κ)
                @test a.x == d.x && a.w == d.w
                @test a.κ == fill(κ, length(d.x) - 1)   # one rate per interval
                @test a.κL == κ && a.κR == κ
                # Both solve the same convex program, in operators that differ by an overall
                # factor: they agree to the Newton solver's own convergence floor.
                @test a.ψ ≈ d.ψ rtol=1e-9
                @test a.λ ≈ d.λ rtol=1e-9
                ts = range(-6, 6; length=101)
                @test a.(ts) ≈ d.(ts) rtol=1e-9
                @test cdf(a, ts) ≈ cdf(d, ts) rtol=1e-8
                @test quantile(a, 0.01:0.01:0.99) ≈ quantile(d, 0.01:0.01:0.99) rtol=1e-8
                @test action(a) ≈ action(d) rtol=1e-8
            end
        end

        @testset "the fit follows the requested scale" begin
            # A scale that varies by two orders of magnitude across the sample.
            κfun(t) = 2.0 * exp(-t)
            a = DensityEstimate(x, κfun)
            n = length(a.x)
            @test length(a.κ) == n - 1
            @test a.κ == [κfun((a.x[k] + a.x[k+1]) / 2) for k in 1:n-1]  # interval midpoints
            @test a.κL == κfun(first(a.x)) && a.κR == κfun(last(a.x))    # tails at the edge nodes
            @test extrema(a.κ)[2] / extrema(a.κ)[1] > 100
            # ψ² integrates to 1: exactly through the closed-form cdf, and to quadrature
            # accuracy against an independent integration.
            @test cdf(a, -Inf) == 0 && cdf(a, Inf) == 1
            @test quadgk(a, -Inf, a.x[1], 0.0, a.x[end], Inf; rtol=1e-10)[1] ≈ 1 atol=1e-8
            @test amplitude(a, 0.3)^2 == a(0.3)
            # cdf and quantile still invert each other on a varying scale.
            qs = 0.001:0.017:0.999
            @test cdf(a, quantile(a, collect(qs))) ≈ qs atol=1e-12
        end

        @testset "cross-validation on a varying scale" begin
            xs = sort!(randn(Xoshiro(23), 500))
            κfun(t) = 3.0 + 2.0 * exp(-t^2 / 2)
            nodes, w, κs, κL, κR = PenalizedDensity._merge_and_realize(xs, κfun, cbrt(eps(Float64)))
            W = sum(w)

            # A constant profile scores what the scalar path scores.
            for κ0 in (1.5, 4.0, 12.0)
                cn, cw = PenalizedDensity._merge_presorted(xs, cbrt(eps(Float64)) / κ0)
                flat = fill(κ0, length(cn) - 1)
                @test PenalizedDensity._klcv(cn, cw, flat, κ0, κ0) ≈
                      PenalizedDensity._klcv(cn, cw, κ0) rtol=1e-11
                @test PenalizedDensity._lscv(cn, cw, flat, κ0, κ0) ≈
                      PenalizedDensity._lscv(cn, cw, κ0) rtol=1e-11
            end

            # ∫Q̂², the LSCV roughness term, against quadrature over the segments the varying
            # scale defines.
            a = PenalizedDensity._fit(nodes, w, κs, κL, κR)
            ∫Q2 = PenalizedDensity._int_quartic(a.x, a.ψ, a.κ, a.κL, a.κR)
            num = quadgk(t -> a(t)^2, -Inf, first(a.x); rtol=1e-10)[1] +
                  quadgk(t -> a(t)^2, last(a.x), Inf; rtol=1e-10)[1] +
                  sum(quadgk(t -> a(t)^2, a.x[k], a.x[k+1]; rtol=1e-10)[1] for k in 1:length(a.x)-1)
            @test ∫Q2 ≈ num rtol=1e-8

            # The first-order leave-one-out densities match brute-force refitting, which drops
            # the node and re-realizes the scale on the reduced node set. Tolerances as in the
            # constant-κ scores above: KLCV's log weights the tail nodes, where the expansion is
            # least accurate, as heavily as the bulk.
            loo = map(eachindex(nodes)) do i
                keep = [j for j in eachindex(nodes) if j != i]
                n2, w2 = nodes[keep], w[keep]
                κs2, κL2, κR2 = PenalizedDensity._kappa_profile(n2, κfun, Float64)
                PenalizedDensity._fit(n2, w2, κs2, κL2, κR2)(nodes[i])
            end
            @test PenalizedDensity._klcv(nodes, w, κs, κL, κR) ≈
                  -sum(w .* log.(loo)) / W rtol=2e-2
            @test PenalizedDensity._lscv(nodes, w, κs, κL, κR) ≈
                  ∫Q2 - 2 * sum(w .* loo) / W rtol=5e-3
        end

        @testset "a one-node fit has only tails" begin
            a = DensityEstimate([0.3], t -> 0.5 + t)
            @test isempty(a.κ)
            @test a.κL == a.κR == 0.8
            @test cdf(a, 0.3) ≈ 0.5           # symmetric Laplace about the single point
            @test cdf(a, Inf) == 1
            # `show` reports the range over the tails alone, the intervals being empty.
            @test occursin("κ ∈ [0.8, 0.8]", sprint(show, a))
        end

        @testset "`show` reports the range of a varying scale" begin
            a = DensityEstimate([0.0, 1.0, 2.0], t -> 1 + t)
            @test occursin("κ ∈ [$(a.κL), $(a.κR)]", sprint(show, a))
            @test minimum(a.κ) > a.κL && maximum(a.κ) < a.κR   # the tails are the extremes
        end

        @testset "both χ² methods are defined at a varying scale" begin
            a = DensityEstimate(x, t -> 2.0 * exp(-t))
            @test chisq(a, a) == 0            # the statistic itself is scale-free
            r = chisq_reference(a)
            @test r isa ChisqReference
            @test expected_chisq(a) == expected_chisq(r) > 0
            # :largeN is the Wald shape at the exact mean, so it too works at any scale.
            @test chisq_ccdf(a, 1.0; method=:largeN) == chisq_ccdf(r, 1.0; method=:largeN)
            @test 0 ≤ chisq_ccdf(a, 1.0; method=:largeN) ≤ 1
            @test chisq_pdf(a, 1.0; method=:largeN) ≥ 0
            @test 0 ≤ pvalue(a, a; method=:largeN) ≤ 1
            @test 0 ≤ pvalue(a, a) ≤ 1
            @test_throws "method must be :exact or :largeN" chisq_ccdf(a, 1.0; method=:bogus)
        end

        @testset "input validation" begin
            for bad in (t -> -1.0, t -> 0.0, t -> NaN, t -> Inf)
                @test_throws "smoothing scale must be finite and positive" DensityEstimate(x, bad)
            end
            @test_throws "cannot be given as a vector" DensityEstimate(x, fill(2.0, length(x) - 1))
        end
    end

    @testset "adaptive κ: the plug-in selector" begin
        # χ²₁: a divergent edge at 0, the regularity limit a constant scale cannot resolve.
        chisq1 = sort!(randn(Xoshiro(19), 3000).^2)
        rtol = cbrt(eps(Float64))
        klcv_const(x, κ) = PenalizedDensity._klcv(PenalizedDensity._merge_presorted(x, rtol / κ)..., κ)
        klcv_scale(x, k) = PenalizedDensity._score_kappa(PenalizedDensity._klcv, x, k, rtol)

        @testset "AdaptiveScale evaluates c·(p̂/ḡ)^α" begin
            κ0 = select_kappa_kl(chisq1)
            p = DensityEstimate(chisq1, κ0)
            k = AdaptiveScale(3.0, 0.5, p)
            lgbar = mean(log(p(xi)) for xi in chisq1)     # ln of the geometric mean of p̂
            for t in (0.05, 0.3, 1.0, 2.5, 6.0)
                @test k(t) ≈ 3.0 * (p(t) / exp(lgbar))^0.5
            end
            # Where the scale equals c, the pilot density equals its geometric mean.
            @test k(chisq1[argmin(abs.(log.(p.(chisq1)) .- lgbar))]) ≈ 3.0 rtol=1e-3

            # The batch path the selector uses must agree with the scalar rule exactly: it
            # walks the pilot's nodes once instead of searching for each position.
            ts = sort!(vcat(chisq1[1:50:end], range(1e-4, 9.0; length=97)))
            @test PenalizedDensity._kappa_sorted(k, ts, Float64) == k.(ts)

            # Far out in the tail the pilot density underflows to zero; the rule floors there
            # rather than producing a zero (and an infinite) smoothing length.
            far = last(chisq1) + 1e4
            @test p(far) == 0
            @test k(far) == k.κmin == 1e-6 * k.c

            # A shallow exponent keeps the rule off its floor even where the pilot density
            # has underflowed, which is the only regime that can tell the scalar and batch
            # paths apart: both must read the log-density, not the density.
            ks = AdaptiveScale(3.0, 5e-3, p)
            deep = last(chisq1) .+ [10.0, 15.0, 20.0]
            @test all(t -> p(t) == 0, deep)
            @test all(t -> ks(t) > ks.κmin, deep)
            @test PenalizedDensity._kappa_sorted(ks, deep, Float64) == ks.(deep)

            @test sprint(show, k) ==
                  "AdaptiveScale(c=3.0, α=0.5) over a pilot with $(length(p.x)) nodes"
        end

        @testset "adaptivity is used only when it wins" begin
            κ = select_kappa_adaptive(chisq1)
            @test κ isa AdaptiveScale
            @test κ.α > 0
            # The selector's own guarantee: the chosen scale beats the constant one on the
            # score that chose it.
            @test klcv_scale(chisq1, κ) < klcv_const(chisq1, select_kappa_kl(chisq1))

            d = DensityEstimate(chisq1, κ)
            @test length(d.κ) == length(d.x) - 1
            # The scale follows the density: finest at the divergent edge, coarsest in the tail.
            @test argmax(d.κ) < length(d.κ) ÷ 10
            @test extrema(d.κ)[2] / extrema(d.κ)[1] > 100
            @test cdf(d, Inf) == 1

            # Uniform data has no density contrast to exploit — κ ∝ p̂^α is already constant —
            # so the α = 0 candidate wins and a plain scalar comes back, keeping the fast path
            # and the goodness-of-fit machinery.
            u = sort!(rand(Xoshiro(5), 3000))
            κu = select_kappa_adaptive(u)
            @test κu isa Real
            @test DensityEstimate(u, κu).κ isa Real
            @test chisq_reference(DensityEstimate(u, κu)) isa ChisqReference
        end

        @testset "alphas and pilot are honored" begin
            κ = select_kappa_adaptive(chisq1; alphas=(1.0,))
            @test κ isa AdaptiveScale && κ.α == 1.0
            # pilot_selector: any callable returning a positive scale from the sample.
            κms = select_kappa_adaptive(chisq1; alphas=(0.5,), pilot_selector=select_kappa_ms)
            @test κms isa AdaptiveScale
            @test κms.pilot.κ == select_kappa_ms(chisq1)
            # Offset input is merged and sorted like any other vector.
            @test select_kappa_adaptive(OffsetVector(chisq1, -1500)) isa AdaptiveScale
        end

        @testset "the c search brackets its minimum" begin
            # Driven by synthetic scores: the searches take the objective as an argument, so
            # their geometry is testable without a density pathological enough to force it.
            sel(score, c0) = PenalizedDensity._select_c(score, c0)

            # A minimum already inside the opening bracket is refined in place.
            @test sel(c -> (log(c) - log(2.0))^2, 2.5) ≈ 2.0 rtol=1e-3

            # One centered on an edge recenters until the minimum falls strictly inside. The
            # bracket spans ×/÷20, so a target 1e4 away from c0 needs several shifts.
            @test sel(c -> (log(c) - log(1e4))^2, 1.0) ≈ 1e4 rtol=1e-3
            @test sel(c -> (log(c) - log(1e-4))^2, 1.0) ≈ 1e-4 rtol=1e-3

            # Non-finite scores are unresolvable candidates, stepped over rather than chosen.
            # Some of the opening bracket must resolve; masking all of it is the error below.
            masked(c) = c < 0.5 ? NaN : (log(c) - log(5.0))^2
            @test sel(masked, 1.0) ≈ 5.0 rtol=1e-3

            # Nothing resolvable anywhere in the opening bracket.
            @test_throws "no resolvable smoothing scale" sel(_ -> NaN, 1.0)
            # A score with no interior minimum runs off the bracket until it gives up.
            @test_throws "kept running off its search bracket" sel(c -> -log(c), 1.0)
        end

        @testset "an unresolvable candidate scores NaN" begin
            # A κ profile spanning many orders of magnitude can drive the factorization to an
            # exact zero pivot. That candidate is unresolvable, not a failure of the search.
            zero_pivot(args...) = throw(ZeroPivotException(3))
            @test isnan(PenalizedDensity._score_kappa(zero_pivot, chisq1, _ -> 5.0, rtol))
            # Any other failure is a bug and must not be swallowed.
            other(args...) = throw(DomainError(1.0, "something else went wrong"))
            @test_throws DomainError PenalizedDensity._score_kappa(other, chisq1, _ -> 5.0, rtol)
            # Too few nodes to fit: scored NaN before the scorefun is ever reached.
            @test isnan(PenalizedDensity._score_kappa(zero_pivot, [1.0], _ -> 5.0, rtol))
        end

        @testset "input validation" begin
            @test_throws "at least one exponent" select_kappa_adaptive(chisq1; alphas=())
            @test_throws "must be positive" select_kappa_adaptive(chisq1; alphas=(0.0, 0.5))
            @test_throws "must be positive" select_kappa_adaptive(chisq1; alphas=(-1.0,))
            @test_throws "rtol must be nonnegative" select_kappa_adaptive(chisq1; rtol=-1.0)
            @test_throws "pilot_selector must return a positive scale" select_kappa_adaptive(chisq1; pilot_selector=_ -> 0.0)
            p = DensityEstimate(chisq1, 10.0)
            @test_throws "exponent α must be positive" AdaptiveScale(1.0, 0.0, p)
            @test_throws "scale c must be positive" AdaptiveScale(0.0, 1.0, p)
        end
    end

    @testset "deprecated keyword κ" begin
        x = [-1.0, 0.0, 0.0, 1.0]
        d = @test_deprecated DensityEstimate(x; κ=1.0)
        @test d.ψ == DensityEstimate(x, 1.0).ψ
        dr = @test_deprecated DensityEstimate(x; κ=1.0, rtol=1e-3)
        @test dr.ψ == DensityEstimate(x, 1.0; rtol=1e-3).ψ
        # A callable scale is reachable only through the positional form, so the keyword
        # never has to grow a second meaning.
        @test_throws "must be passed positionally" DensityEstimate(x; κ = t -> 1.0)
    end

    @testset "goodness of fit: statistic" begin
        d = DensityEstimate([-1.0, 0.0, 0.0, 1.0], 1.0)
        @test chisq(d, d) == 0                 # a distribution vs itself
        @test chisq(d, x -> 0.9 * d(x)) > 0    # a mismatched (here unnormalized) trial
        # matches the defining sum 4 Σ wᵢ (√Q(xᵢ)/ψ_cl(xᵢ) − 1)²
        Q(x) = exp(-x^2 / 2) / sqrt(2π)
        manual = 4 * sum(d.w[i] * (sqrt(Q(d.x[i])) / d.ψ[i] - 1)^2 for i in eachindex(d.x))
        @test chisq(d, Q) ≈ manual
        @test_throws ArgumentError chisq(d, x -> -1.0)   # negative trial density
    end

    @testset "goodness of fit: large-N (Eq. 26) reference" begin
        d = DensityEstimate([-1.0, 0.0, 0.0, 1.0], 1.0)
        # :largeN is the Wald inverse-Gaussian(mean μ, shape μ²) at the exact mean μ = tr A.
        μ = expected_chisq(d)
        @test μ ≈ expected_chisq(chisq_reference(d)) > 0
        # method=:largeN is the inverse-Gaussian(mean μ, shape μ²): normalized, with mean μ.
        zs = range(1e-5, 40μ; length=4_000_001)
        p = chisq_pdf.(Ref(d), zs; method=:largeN)
        trap(f) = (sum(f) - (f[1] + f[end]) / 2) * step(zs)
        @test trap(p) ≈ 1 atol = 1e-4
        @test trap(zs .* p) ≈ μ rtol = 1e-4
        @test chisq_pdf(d, -1.0; method=:largeN) == 0 && chisq_pdf(d, 0.0; method=:largeN) == 0
        # ccdf is the pdf's upper tail and its negative derivative is the pdf.
        z0 = 1.3μ
        @test chisq_ccdf(d, z0; method=:largeN) ≈ trap(p .* (zs .≥ z0)) atol = 1e-4
        @test chisq_ccdf(d, 0.0; method=:largeN) == 1
        h = 1e-5
        @test -(chisq_ccdf(d, z0 + h; method=:largeN) - chisq_ccdf(d, z0 - h; method=:largeN)) / 2h ≈
              chisq_pdf(d, z0; method=:largeN) rtol = 1e-4
        # pvalue(...; method=:largeN) is the large-N ccdf at the observed statistic.
        Q(x) = exp(-x^2 / 2) / sqrt(2π)
        @test pvalue(d, Q; method=:largeN) == chisq_ccdf(d, chisq(d, Q); method=:largeN)
    end

    @testset "goodness of fit: exact reference distribution" begin
        # A single point: χ² is exactly a scaled χ²₁ (one generalized-χ² weight = the mean).
        d1 = DensityEstimate([0.7], 1.5); r1 = chisq_reference(d1)
        e1 = expected_chisq(r1)
        @test chisq_ccdf(r1, e1) ≈ 0.3173105 rtol = 1e-3       # P(Z² ≥ 1)
        @test chisq_ccdf(r1, 4 * e1) ≈ 0.0455003 rtol = 1e-3   # P(Z² ≥ 4)
        @test chisq_ccdf(r1, 0.0) ≈ 1 atol = 1e-6

        # The exact distribution reproduces a direct Monte-Carlo of the fluctuation field.
        Random.seed!(1)
        x = randn(150); d = DensityEstimate(x, select_kappa_ms(x))
        r = chisq_reference(d)
        chis = fieldmc_chisq(d; nsamp=60_000, seed=7)
        @test expected_chisq(r) ≈ mean(chis) rtol = 0.02
        for z in quantile(chis, (0.3, 0.6, 0.9, 0.99))
            @test chisq_ccdf(r, z) ≈ mean(>(z), chis) atol = 0.012
        end
        # The Wald shape at the exact mean tracks the exact tail closely.
        for frac in (0.8, 1.2, 1.6)
            z = frac * expected_chisq(r)
            @test chisq_ccdf(d, z; method=:largeN) ≈ chisq_ccdf(r, z) rtol = 0.1
        end

        # ccdf ∈ [0,1] and monotone; pdf ≥ 0, integrates to ~1, and is −d(ccdf)/dz.
        μ = expected_chisq(r)
        grid = range(0.05μ, 4μ; length=60)
        c = chisq_ccdf.(Ref(r), grid)
        @test all(0 .≤ c .≤ 1) && issorted(c; rev=true)
        @test chisq_ccdf(r, 0.0) ≈ 1 atol = 1e-4    # full mass ⇒ normalized
        @test all(≥(0), chisq_pdf.(Ref(r), grid))
        h = 1e-3
        @test -(chisq_ccdf(r, μ + h) - chisq_ccdf(r, μ - h)) / 2h ≈ chisq_pdf(r, μ) rtol = 1e-4

        # Reference reuse and default-exact wiring.
        Q(x) = exp(-x^2 / 2) / sqrt(2π)
        @test pvalue(d, Q) == chisq_ccdf(chisq_reference(d), chisq(d, Q))
        @test pvalue(r, chisq(d, Q)) == chisq_ccdf(r, chisq(d, Q))
        @test chisq_ccdf(d, μ) ≈ chisq_ccdf(chisq_reference(d), μ)   # :exact is the default

        # Method validation.
        @test_throws ArgumentError chisq_ccdf(d, 1.0; method=:bogus)
        @test_throws ArgumentError chisq_pdf(d, 1.0; method=:bogus)
        @test_throws ArgumentError pvalue(d, Q; method=:bogus)

        @test sprint(show, r) == "ChisqReference($(length(r.g)) nodes, ⟨χ²⟩=$(r.mean))"

        # The Imhof integrand sweep is allocation-free given its scratch buffers.
        @test r.tg ≈ r.tri * r.g
        piv, rhs = PenalizedDensity._logΦ_scratch(r)
        @test sweep_allocated(piv, rhs, r, 0.7) == 0

        # Generic indexing: OffsetArray input gives an identical reference.
        ro = chisq_reference(DensityEstimate(OffsetArray(x, -75), d.κ))
        @test ro.tri.dv ≈ r.tri.dv && ro.g ≈ r.g && expected_chisq(ro) ≈ μ

        # Green's identity, the reference's internal consistency check.
        @test green_identity(d) ≈ 1 rtol = 1e-5
    end

    @testset "goodness of fit: exact reference at a varying scale" begin
        # A constant callable takes the piecewise path and lands on the scalar path's answer.
        xg = randn(Xoshiro(11), 500)
        κ = select_kappa_kl(xg)
        rs = chisq_reference(DensityEstimate(xg, κ))
        rc = chisq_reference(DensityEstimate(xg, t -> κ))
        @test rc.tri.dv ≈ rs.tri.dv rtol = 1e-10
        @test rc.g ≈ rs.g rtol = 1e-10
        @test expected_chisq(rc) ≈ expected_chisq(rs) rtol = 1e-10
        @test chisq_ccdf(rc, expected_chisq(rc)) ≈ chisq_ccdf(rs, expected_chisq(rs)) rtol = 1e-10

        # The decisive test: the exact law reproduces a direct Monte-Carlo of the fluctuation
        # field. A hard-edge family under the plug-in selector, whose realized rates span five
        # orders of magnitude — every quantity the reference is built from is propagated in
        # scaled form precisely so this does not overflow.
        xa = sort!(randn(Xoshiro(3), 300).^2)                 # χ²₁: divergent edge at 0
        da = DensityEstimate(xa, select_kappa_adaptive(xa))
        @test maximum(da.κ) / minimum(da.κ) > 1e4
        ra = chisq_reference(da)
        chis = fieldmc_chisq(da; nsamp=40_000, seed=3)
        @test expected_chisq(ra) ≈ mean(chis) rtol = 0.02
        for z in quantile(chis, (0.3, 0.6, 0.9, 0.99))
            @test chisq_ccdf(ra, z) ≈ mean(>(z), chis) atol = 0.012
        end
        @test green_identity(da) ≈ 1 rtol = 1e-5

        # And a smooth family at small N, under a hand-chosen scale.
        xs = sort!(randn(Xoshiro(2), 60))
        ds = DensityEstimate(xs, t -> 2exp(-t / 2))
        rsm = chisq_reference(ds)
        chs = fieldmc_chisq(ds; nsamp=40_000, seed=3)
        @test expected_chisq(rsm) ≈ mean(chs) rtol = 0.02
        for z in quantile(chs, (0.3, 0.9, 0.99))
            @test chisq_ccdf(rsm, z) ≈ mean(>(z), chs) atol = 0.012
        end
        @test green_identity(ds) ≈ 1 rtol = 1e-5

        # ∬ψ_cl G₀ ψ_cl and the nodal α agree with a conservative finite-difference solve of
        # 𝒜α = ψ_cl/(2λ) on a node-aligned grid: an independent route, with no Green's
        # function in it. It converges to the closed forms at the discretization's O(δ²).
        errs = map((100, 400)) do per
            x, ψ, λ = ds.x, ds.ψ, ds.λ
            κ(k) = k == 0 ? ds.κL : k > length(x) - 1 ? ds.κR : PenalizedDensity._kappa(ds.κ, k)
            bnds = vcat(first(x) - 15 / ds.κL, x, last(x) + 15 / ds.κR)
            y = Float64[]
            for k in 1:length(bnds)-1
                npts = max(2, ceil(Int, (bnds[k+1] - bnds[k]) * κ(k - 1) * per))
                append!(y, range(bnds[k], bnds[k+1]; length=npts + 1)[1:end-1])
            end
            push!(y, last(bnds)); m = length(y); δ = diff(y)
            cell = [searchsortedlast(bnds, (y[j] + y[j+1]) / 2) - 1 for j in 1:m-1]
            mass = [(j > 1 ? δ[j-1] : 0.0) / 2 + (j < m ? δ[j] : 0.0) / 2 for j in 1:m]
            p = copy(mass); q = zeros(m - 1)
            for j in 1:m-1
                c = 1 / (κ(cell[j])^2 * δ[j]); p[j] += c; p[j+1] += c; q[j] = -c
            end
            ψy = amplitude(ds, y)
            αfd = SymTridiagonal(p, q) \ (mass .* ψy ./ 2λ)
            mfd = [αfd[searchsortedfirst(y, xi)] for xi in x]
            mex = PenalizedDensity._node_alpha(x, ψ, ds.κ, ds.κL, ds.κR, λ)
            (maximum(abs.(mex .- mfd) ./ abs.(mfd)),
             abs(PenalizedDensity._int_psi_alpha(x, ψ, mex, ds.κ, ds.κL, ds.κR, λ) -
                 sum(mass .* ψy .* αfd)) / sum(mass .* ψy .* αfd))
        end
        @test all(<(2e-4), errs[1]) && all(<(2e-5), errs[2])   # and 16× finer ⇒ 16× closer
        @test all(first(errs) ./ last(errs) .> 10)
    end

    @testset "kappa_interval: principled range" begin
        Random.seed!(7)
        x = randn(600)
        ki = kappa_interval(x)
        @test ki.lo < ki.κ < ki.hi
        # h = ½ point agrees with the minimum-sensitivity scale within a small factor.
        κms = select_kappa_ms(x; κs = exp.(range(log(0.05), log(50); length = 60)))
        @test 0.5 < ki.κ / κms < 2.0
        # A wider band brackets a narrower one around the same central κ.
        wide = kappa_interval(x; level = 0.6)
        @test wide.κ ≈ ki.κ rtol = 1e-3
        @test wide.lo < ki.lo && wide.hi > ki.hi
        # Analytic asymptotes of g(κ) = S(κ) + W ln κ: W/2 (κ→0) and W/2 + W·H (κ→∞).
        # Use widely-separated points so isolation is reached at a moderate κ (avoiding
        # sinh overflow), where H = ln 3 exactly.
        xs = [0.0, 5.0, 10.0]; W3 = 3; H3 = log(3)
        g(κ) = action(DensityEstimate(xs, κ)) + W3 * log(κ)
        @test g(1e-5) ≈ W3 / 2 rtol = 1e-3            # one lump
        @test g(3.0) ≈ W3 / 2 + W3 * H3 rtol = 1e-4   # three isolated points
        # Repeated points enter through the multiplicity entropy.
        @test kappa_interval([0.0, 0.0, 1.0, 5.0]; level = 0.4).κ > 0
    end

    @testset "entropy and negentropy" begin
        # Single-point fit is an exact Laplace(rate 2κ) density, variance σ² = 1/(2κ²): the
        # plug-in Ĥ = -ln κ (biased relative to the exact H = 1 - ln κ at this extreme
        # small-sample size), and negentropy is the gap to the matched Gaussian's entropy.
        κ = 1.5
        d = DensityEstimate([2.0], κ)
        @test entropy(d) ≈ -log(κ)
        @test negentropy(d) ≈ log(2π * ℯ / (2κ^2)) / 2 - entropy(d)

        # Brute-force reference: quadrature entropy and negentropy from the fitted density
        # itself (not the plug-in estimator), to isolate the closed-form mean/variance and
        # the -∫Q ln Q dx integral from any small-sample plug-in bias.
        function quad_entropy(d)
            lo, hi = d.x[1] - 20 / d.κ, d.x[end] + 20 / d.κ
            integrate(t -> (q = d(t); q > 0 ? -q * log(q) : 0.0), lo, hi)
        end
        quad_negentropy(d, σ²) = log(2π * ℯ * σ²) / 2 - quad_entropy(d)

        Random.seed!(42)
        N = 5_000
        families = (
            laplace = rand(N) .- rand(N),
            uniform = 4 .* rand(N) .- 2,
            bimodal = vcat(randn(N ÷ 2) .- 3, randn(N ÷ 2) .+ 3),
        )
        for (name, x) in pairs(families)
            κ = select_kappa_ms(x)
            d = DensityEstimate(x, κ)
            @test negentropy(d) > 0
            @test negentropy(d) ≈ quad_negentropy(d, var(x)) rtol = 0.1
        end

        # Gaussian samples: negentropy fluctuates near 0 and shrinks with N.
        negs = map((200, 2_000, 20_000)) do n
            Random.seed!(1)
            xn = randn(n)
            dn = DensityEstimate(xn, select_kappa_ms(xn))
            abs(negentropy(dn))
        end
        @test issorted(negs; rev=true)

        # Affine invariance: x ↦ a·x + b with κ ↦ κ/|a| is the corresponding fit (package's
        # scale equivariance), and negentropy is invariant under it, for a > 0 and a < 0.
        Random.seed!(9)
        x = 1.7 .* randn(2_000) .+ 0.3
        κ0 = select_kappa_ms(x)
        d0 = DensityEstimate(x, κ0)
        for a in (4.5, -3.0)
            b = -2.1
            da = DensityEstimate(a .* x .+ b, κ0 / abs(a))
            @test negentropy(da) ≈ negentropy(d0) rtol = 1e-6
        end

        # Generic indexing: OffsetArray input gives an identical result.
        d_off = DensityEstimate(OffsetArray(x, -750), κ0)
        @test negentropy(d_off) == negentropy(d0)

        # Translation far from the origin must not corrupt the variance via cancellation
        # in M2 - M1²: node positions are large (~1e8) while the spread stays O(1).
        dfar = DensityEstimate(x .+ 1e8, κ0)
        @test negentropy(dfar) ≈ negentropy(d0) rtol = 1e-6
    end

    @testset "held-out entropy and negentropy" begin
        Random.seed!(17)
        x = 1.2 .* randn(400) .+ 0.5
        κ = select_kappa_kl(x)
        d = DensityEstimate(x, κ)
        ye = 1.2 .* randn(150) .+ 0.5          # independent evaluation batch

        # entropy(d, ye) is the plug-in -(2/M) Σ ln ψ(yⱼ) at the held-out points.
        M = length(ye)
        @test entropy(d, ye) ≈ -2 * sum(log(amplitude(d, y)) for y in ye) / M

        # negentropy(d, ye) = ½ln(2πe s²) - entropy, with s² the empirical (population)
        # variance of ye. That Gaussian term equals -(1/M) Σ ln 𝒩(yⱼ) at ye's own MLE
        # moments, so the reference is itself a held-out expectation.
        m = sum(ye) / M
        s² = sum((y - m)^2 for y in ye) / M
        gaussref = -sum(log(exp(-(y - m)^2 / (2s²)) / sqrt(2π * s²)) for y in ye) / M
        @test log(2π * ℯ * s²) / 2 ≈ gaussref
        @test negentropy(d, ye) ≈ log(2π * ℯ * s²) / 2 - entropy(d, ye)

        # Affine invariance: entropy shifts by ln|a|, negentropy is invariant (a > 0, a < 0).
        for a in (2.5, -0.7)
            b = 1.3
            da = DensityEstimate(a .* x .+ b, κ / abs(a))
            yea = a .* ye .+ b
            @test entropy(da, yea) - entropy(d, ye) ≈ log(abs(a)) atol = 1e-8
            @test negentropy(da, yea) ≈ negentropy(d, ye) atol = 1e-8
        end

        # Generic indexing: OffsetArray evaluation points give identical results.
        yeo = OffsetArray(ye, -75)
        @test entropy(d, yeo) == entropy(d, ye)
        @test negentropy(d, yeo) == negentropy(d, ye)

        @test_throws ArgumentError entropy(d, Float64[])
        @test_throws ArgumentError negentropy(d, Float64[])
    end

    @testset "evaluation-point log-density gradient" begin
        # ∂ln Q̂(y)/∂y = 2ψ'(y)/ψ(y), against central finite differences away from the
        # nodes (where ψ' — and hence the log-density slope — has a kink), across κ regimes.
        nodes = [-1.3, -0.4, 0.2, 0.5, 1.1, 2.0]
        for κ in (0.3, 1.5, 8.0, 40.0)
            d = DensityEstimate(nodes, κ)
            h = 1e-6
            for y in range(-3.0, 4.0; length = 121)
                minimum(abs.(y .- d.x)) < 0.05 && continue
                fd = (log(d(y + h)) - log(d(y - h))) / (2h)
                @test logdensity_eval_gradient(d, y) ≈ fd rtol = 1e-4 atol = 1e-6
            end
            # In the tails ψ ∝ e^{∓κ|·|}, so the slope is exactly ±2κ.
            @test logdensity_eval_gradient(d, nodes[1] - 3.0) ≈ 2κ
            @test logdensity_eval_gradient(d, nodes[end] + 3.0) ≈ -2κ
        end
    end

    @testset "node-position log-density gradient" begin
        # gᵢ = ∂/∂xᵢ Σⱼ cⱼ ln Q̂(yⱼ), by the implicit-function adjoint, against central
        # finite differences that refit at perturbed node positions. Fits are built through
        # _fit on distinct nodes (no merging) so the perturbation stays within one cell.
        function L_of_nodes(xn, w, κ, ye, c)
            dd = PenalizedDensity._fit(collect(float.(xn)), collect(float.(w)), float(κ))
            sum(cj * log(dd(y)) for (y, cj) in zip(ye, c))
        end
        Random.seed!(23)
        for κ in (0.3, 1.5, 8.0, 40.0)
            xn = sort(randn(7)) .* 1.3
            w = ones(7)
            d = PenalizedDensity._fit(copy(xn), copy(w), Float64(κ))
            # evaluation points span both tails and the interior, none at a node
            ye = Float64[minimum(xn) - 0.7, minimum(xn) - 0.1, -0.33, 0.07, 0.44, 0.9,
                         maximum(xn) + 0.15, maximum(xn) + 0.8]
            c = collect(range(0.5, 1.5; length = length(ye)))   # nontrivial weights
            an = logdensity_node_gradient(d, ye, c)
            h = 1e-6
            fd = map(eachindex(xn)) do i
                xp = copy(xn); xp[i] += h
                xm = copy(xn); xm[i] -= h
                (L_of_nodes(xp, w, κ, ye, c) - L_of_nodes(xm, w, κ, ye, c)) / (2h)
            end
            @test an ≈ fd rtol = 1e-4
        end

        # Near-coincident node pair (θ ≈ 0.05) exercises the small-θ hyperbolic forms.
        xn = [-1.0, -0.02, 0.03, 1.0, 2.0]; w = ones(5); κ = 1.0
        d = PenalizedDensity._fit(copy(xn), copy(w), κ)
        ye = Float64[-1.7, -0.4, 0.5, 1.4, 2.6]; c = fill(0.2, 5)
        an = logdensity_node_gradient(d, ye, c)
        h = 1e-7
        fd = map(eachindex(xn)) do i
            xp = copy(xn); xp[i] += h
            xm = copy(xn); xm[i] -= h
            (L_of_nodes(xp, w, κ, ye, c) - L_of_nodes(xm, w, κ, ye, c)) / (2h)
        end
        @test an ≈ fd rtol = 1e-4

        # Weighted multiplicities (w > 1) are handled like distinct points.
        xn = sort(randn(5)); w = [1.0, 3.0, 1.0, 2.0, 1.0]; κ = 1.2
        d = PenalizedDensity._fit(copy(xn), copy(w), κ)
        ye = Float64[xn[1] - 0.5, -0.2, 0.3, xn[end] + 0.4]; c = [0.7, 1.1, 0.9, 1.3]
        an = logdensity_node_gradient(d, ye, c)
        h = 1e-6
        fd = map(eachindex(xn)) do i
            xp = copy(xn); xp[i] += h
            xm = copy(xn); xm[i] -= h
            (L_of_nodes(xp, w, κ, ye, c) - L_of_nodes(xm, w, κ, ye, c)) / (2h)
        end
        @test an ≈ fd rtol = 1e-4

        # Default weights are all ones (the plain sum of log densities).
        d = PenalizedDensity._fit(sort(randn(6)), ones(6), 1.0)
        ye = [-0.5, 0.2, 0.8]
        @test logdensity_node_gradient(d, ye) ≈ logdensity_node_gradient(d, ye, ones(3))

        # The two halves compose into ∂entropy(d, ye)/∂(node positions): since the Gaussian
        # reference of negentropy(d, ye) depends only on ye, the node gradient of entropy is
        # -logdensity_node_gradient with uniform 1/M weights.
        Random.seed!(11)
        xn = sort(randn(6)); w = ones(6); κ = 1.7
        d = PenalizedDensity._fit(copy(xn), copy(w), κ)
        ye = Float64[-1.4, -0.3, 0.25, 0.9, 2.1]; M = length(ye)
        an = .-logdensity_node_gradient(d, ye, fill(1 / M, M))
        h = 1e-6
        fd = map(eachindex(xn)) do i
            xp = copy(xn); xp[i] += h
            xm = copy(xn); xm[i] -= h
            (entropy(PenalizedDensity._fit(xp, copy(w), κ), ye) -
             entropy(PenalizedDensity._fit(xm, copy(w), κ), ye)) / (2h)
        end
        @test an ≈ fd rtol = 1e-4

        # One adjoint solve regardless of batch size: allocations depend on the node count N,
        # not the evaluation count M.
        xn = sort(randn(500)); w = ones(500); d = PenalizedDensity._fit(copy(xn), copy(w), 2.0)
        span(nb) = collect(range(minimum(xn) - 1, maximum(xn) + 1; length = nb))
        ye_small, ye_big = span(200), span(4000)
        c_small, c_big = fill(1 / 200, 200), fill(1 / 4000, 4000)
        logdensity_node_gradient(d, ye_small, c_small)    # compile
        logdensity_node_gradient(d, ye_big, c_big)
        a_small = @allocated logdensity_node_gradient(d, ye_small, c_small)
        a_big = @allocated logdensity_node_gradient(d, ye_big, c_big)
        @test a_small == a_big
    end

    @testset "large κ stays finite (no sinh overflow)" begin
        Random.seed!(3)
        x = randn(300)
        d = DensityEstimate(x, 5000.0)   # kernels far narrower than spacings
        @test all(isfinite, d.ψ) && all(>(0), d.ψ)
        @test isfinite(d.λ) && d.λ > 0
        @test isfinite(action(d))
        @test all(isfinite, amplitude(d, range(-4, 4; length = 200)))  # incl. inter-point gaps
    end

    @testset "log-density stays finite where the density underflows" begin
        Random.seed!(11)
        x = randn(200)
        κ = 1.5
        d = DensityEstimate(x, κ)
        # Far enough out that ψ itself is zero in double precision, which is the
        # regime a log-density exists to serve.
        far = [d.x[1] - 1500.0, d.x[1] - 700.0, d.x[end] + 700.0, d.x[end] + 1500.0]
        ℓ = PenalizedDensity._logdensity_sorted(d, sort(far))
        @test all(isfinite, ℓ)
        @test any(t -> amplitude(d, t) == 0, far)

        # Beyond the outermost nodes the log-density is exactly linear with slope
        # ±2κ, so the closed form is available independently of the recurrence.
        for t in (d.x[1] - 1500.0, d.x[1] - 700.0)
            @test PenalizedDensity._logdensity_sorted(d, [t])[1] ≈
                  2 * (log(d.ψ[1]) + d.κL * (t - d.x[1]))
        end
        for t in (d.x[end] + 700.0, d.x[end] + 1500.0)
            @test PenalizedDensity._logdensity_sorted(d, [t])[1] ≈
                  2 * (log(d.ψ[end]) - d.κR * (t - d.x[end]))
        end

        # Where the density has not underflowed the two routes must agree.
        near = sort(d.x[1] .- [0.5, 2.0, 5.0])
        @test PenalizedDensity._logdensity_sorted(d, near) ≈ 2 .* log.(amplitude(d, near))
    end

    @testset "log-density stays finite inside a wide gap" begin
        # κ·gap ≈ 3000: both sinh arcs underflow in the middle of the interval, so the
        # amplitude is zero over a region where the log-density is finite.
        xs = [-1000.0, -999.0, 0.0, 999.0, 1000.0]
        κ = 3.0
        d = DensityEstimate(xs, κ)
        mid = [-750.0, -500.0, -250.0, 250.0, 500.0, 750.0]
        @test all(t -> amplitude(d, t) == 0, mid)
        @test all(isfinite, logdensity(d, mid))

        # Deep inside the gap each arc is a pure exponential decay from its own node,
        # ψ ≈ ψ_k e^{-κ(x - x_k)} + ψ_{k+1} e^{-κ(x_{k+1} - x)}, which pins the value
        # without reference to the sinh recurrence.
        for t in mid
            k = searchsortedlast(xs, t)
            @test logdensity(d, t) ≈ 2 * logaddexp(log(d.ψ[k]) - κ * (t - xs[k]),
                                                   log(d.ψ[k+1]) - κ * (xs[k+1] - t))
        end
    end

    @testset "logdensity" begin
        Random.seed!(5)
        d = DensityEstimate(randn(150), 1.2)

        # Agrees with the amplitude wherever the amplitude has not underflowed.
        ts = collect(range(d.x[1] - 3, d.x[end] + 3; length = 500))
        @test logdensity(d, ts) ≈ 2 .* log.(amplitude(d, ts))
        @test logdensity(d, ts[17]) == logdensity(d, ts)[17]      # scalar matches array

        # The scalar path and the sorted-batch sweep are two routes to one quantity.
        far = sort(vcat(ts, d.x[1] .- [1e4, 1500.0], d.x[end] .+ [1500.0, 1e4]))
        @test logdensity(d, far) == PenalizedDensity._logdensity_sorted(d, far)

        # Array shape and axes are preserved; matches `map` over the same points.
        m = reshape(ts[1:12], 3, 4)
        @test size(logdensity(d, m)) == (3, 4)
        @test logdensity(d, m) == map(t -> logdensity(d, t), m)
        to = OffsetArray(ts[1:20], -7)
        @test axes(logdensity(d, to)) == axes(to)
        @test logdensity(d, to) == OffsetArray(logdensity(d, ts[1:20]), -7)

        # Outside a finite support the density is exactly zero, so ln Q = -Inf.
        db = DensityEstimate(clamp.(randn(150), -1.9, 1.9), 1.2; support = (-2.0, 2.0))
        @test logdensity(db, -2.5) == -Inf
        @test logdensity(db, 2.5) == -Inf
        @test isfinite(logdensity(db, 0.0))
        # The bounded tails run through `logcosh` and stay finite to the boundary.
        @test all(isfinite, logdensity(db, range(-2, 2; length = 200)))
    end

    @testset "input validation" begin
        @test_throws ArgumentError DensityEstimate(Float64[], 1.0)
        @test_throws "cannot fit a density to zero points" DensityEstimate(Float64[], 1.0)
        @test_throws ArgumentError DensityEstimate([1.0], 0.0)
        @test_throws "κ must be positive" DensityEstimate([1.0], 0.0)
        @test_throws ArgumentError DensityEstimate([1.0], -1.0)
        @test_throws "κ must be positive" DensityEstimate([1.0], -1.0)
        @test_throws ArgumentError DensityEstimate([1.0], 1.0; rtol=-1.0)
        @test_throws "rtol must be nonnegative" DensityEstimate([1.0], 1.0; rtol=-1.0)
        @test_throws ArgumentError select_kappa_ms([1.0, 2.0]; κs=[1.0, -1.0, 2.0])
        @test_throws "κs must be sorted and positive" select_kappa_ms([1.0, 2.0]; κs=[1.0, -1.0, 2.0])
        @test_throws ArgumentError select_kappa_ms([1.0, 2.0]; κs=[1.0, 2.0])  # need ≥ 3
        @test_throws "need at least 3 values in κs to bracket the minimum" select_kappa_ms([1.0, 2.0]; κs=[1.0, 2.0])
        @test_throws ArgumentError kappa_interval([1.0, 2.0]; level=0.0)
        @test_throws "level must be in (0, 1)" kappa_interval([1.0, 2.0]; level=0.0)
        @test_throws ArgumentError kappa_interval([1.0, 2.0]; level=1.0)
        @test_throws "level must be in (0, 1)" kappa_interval([1.0, 2.0]; level=1.0)
        @test_throws ArgumentError kappa_interval([3.0])              # need ≥ 2 distinct
        @test_throws "need at least two distinct points to select κ" kappa_interval([3.0])
        @test_throws ArgumentError kappa_interval([3.0, 3.0, 3.0])    # all coincident
        @test_throws "need at least two distinct points to select κ" kappa_interval([3.0, 3.0, 3.0])

        @test_throws ArgumentError select_kappa_cv([1.0, 2.0]; κs=[1.0, -1.0, 2.0])
        @test_throws "κs must be sorted and positive" select_kappa_cv([1.0, 2.0]; κs=[1.0, -1.0, 2.0])
        @test_throws ArgumentError select_kappa_cv([1.0, 2.0]; κs=[1.0, 2.0])   # need ≥ 3
        @test_throws "need at least 3 values in κs to bracket the minimum" select_kappa_cv([1.0, 2.0]; κs=[1.0, 2.0])
        @test_throws ArgumentError select_kappa_cv([3.0])                        # need ≥ 2 distinct
        @test_throws "need at least two distinct points to select κ" select_kappa_cv([3.0])

        @test_throws ArgumentError select_kappa_kl([1.0, 2.0]; κs=[1.0, -1.0, 2.0])
        @test_throws "κs must be sorted and positive" select_kappa_kl([1.0, 2.0]; κs=[1.0, -1.0, 2.0])
        @test_throws ArgumentError select_kappa_kl([1.0, 2.0]; κs=[1.0, 2.0])   # need ≥ 3
        @test_throws "need at least 3 values in κs to bracket the minimum" select_kappa_kl([1.0, 2.0]; κs=[1.0, 2.0])
        @test_throws ArgumentError select_kappa_kl([3.0])                        # need ≥ 2 distinct
        @test_throws "need at least two distinct points to select κ" select_kappa_kl([3.0])
    end
end

@testset "bounded support (natural/Neumann boundary)" begin
    # Independent reference fit: assembled from scratch with `tanh`/`sech` and `coth`/`csch`
    # (Base's hyperbolic functions, not the package's e^{-2θ}-based stable forms), sharing no
    # arithmetic with the bounded-support code under test. Mirrors the standalone derivation
    # the closed forms below were checked against.
    _tail_diag_ref(κ, Δ) = isfinite(Δ) ? tanh(κ * Δ) : 1.0
    function _tail_mass_ref(ψ1, κ, Δ)
        isfinite(Δ) || return ψ1^2 / (2κ)
        u = κ * Δ
        return ψ1^2 * (tanh(u) + u * sech(u)^2) / (2κ)
    end
    function _bounded_operator_ref(x::Vector{Float64}, κ::Float64, a::Float64, b::Float64)
        n = length(x)
        d = zeros(n); e = zeros(n - 1)
        d[1] += _tail_diag_ref(κ, x[1] - a)
        d[n] += _tail_diag_ref(κ, b - x[n])
        for k in 1:n-1
            θ = κ * (x[k+1] - x[k])
            d[k] += coth(θ); d[k+1] += coth(θ); e[k] = -csch(θ)
        end
        return SymTridiagonal(d, e)
    end
    function _bounded_norm_sq_ref(x::Vector{Float64}, ψ::Vector{Float64}, κ::Float64, a::Float64, b::Float64)
        n = length(x)
        Z = _tail_mass_ref(ψ[1], κ, x[1] - a) + _tail_mass_ref(ψ[n], κ, b - x[n])
        for k in 1:n-1
            θ = κ * (x[k+1] - x[k]); ct, cs = coth(θ), csch(θ)
            fdiag = (ct - θ * cs^2) / (2κ); fcross = cs * (θ * ct - 1) / (2κ)
            Z += fdiag * (ψ[k]^2 + ψ[k+1]^2) + 2 * fcross * ψ[k] * ψ[k+1]
        end
        return Z
    end
    function _bounded_fit_ref(xs_sorted::Vector{Float64}, κ::Float64, a::Float64, b::Float64;
                              rtol::Float64 = cbrt(eps(Float64)))
        nodes, weights = PenalizedDensity._merge_presorted(xs_sorted, rtol / κ)
        M = _bounded_operator_ref(nodes, κ, a, b)
        ψ = PenalizedDensity._solve_amplitude(M, weights)
        Z = _bounded_norm_sq_ref(nodes, ψ, κ, a, b)
        ψ ./= sqrt(Z)
        return (; x = nodes, w = weights, ψ, κ, a, b)
    end

    @testset "unbounded regression" begin
        Random.seed!(41)
        x = randn(1500)
        for κ in (0.7, 6.0)
            d1 = DensityEstimate(x, κ)
            d2 = DensityEstimate(x, κ; support=(-Inf, Inf))
            @test d1.x == d2.x && d1.w == d2.w && d1.ψ == d2.ψ && d1.κ == d2.κ &&
                  d1.κL == d2.κL && d1.κR == d2.κR && d1.λ == d2.λ
            @test d1.lo == -Inf && d1.hi == Inf
            far = 1e4 / κ
            dfar = DensityEstimate(x, κ; support=(minimum(x) - far, maximum(x) + far))
            @test maximum(abs.(dfar.ψ .- d1.ψ)) / maximum(d1.ψ) < 1e-12
        end
    end

    @testset "reproduces the reference fit node-for-node" begin
        for (name, xgen, κ, support) in (
                ("exponential", rng -> -log.(1 .- rand(rng, 800)), 12.0, (0.0, Inf)),
                ("uniform",     rng -> rand(rng, 800),              18.0, (0.0, 1.0)),
                ("chisq1",      rng -> randn(rng, 800) .^ 2,        25.0, (0.0, Inf)))
            rng = Xoshiro(hash((:boundedref, name)))
            x = sort(xgen(rng))
            ref = _bounded_fit_ref(x, κ, support...)
            d = DensityEstimate(x, κ; support)
            @test d.x == ref.x
            @test maximum(abs.(d.ψ .- ref.ψ) ./ ref.ψ) < 1e-9
        end
    end

    @testset "_norm_sq_gram matches _norm_sq on a bounded fit" begin
        # No caller yet threads a finite support into `_norm_sq_gram` (its only consumer,
        # `_loo_density`, stays unbounded-only), but its generalized tail terms must already be
        # self-consistent: Z = ψᵀGψ, and Z itself must agree with `_norm_sq`.
        Random.seed!(47)
        x = sort(rand(30) .* 0.6 .+ 0.2)
        d = DensityEstimate(x, 7.0; support=(0.0, 1.0))
        Zgram, Gψ = PenalizedDensity._norm_sq_gram(d.x, d.ψ, d.κ, d.κL, d.κR, d.lo, d.hi)
        Z = PenalizedDensity._norm_sq(d.x, d.ψ, d.κ, d.lo, d.hi)
        @test Zgram ≈ Z rtol = 1e-13
        @test sum(d.ψ .* Gψ) ≈ Z rtol = 1e-13
    end

    @testset "mass" begin
        Random.seed!(42)
        x = sort(rand(400))
        d = DensityEstimate(x, 30.0; support=(0.0, 1.0))
        @test cdf(d, 1.0) == 1
        @test cdf(d, 0.0) == 0
        left, El = quadgk(d, 0.0, d.x[1]; rtol=1e-10)
        interior, Ei = quadgk(d, d.x...; rtol=1e-10)
        right, Er = quadgk(d, d.x[end], 1.0; rtol=1e-10)
        @test max(El, Ei, Er) < 1e-8
        @test abs(left + interior + right - 1) < 1e-9
    end

    @testset "cdf ∘ quantile round trip; monotonicity across the boundary" begin
        Random.seed!(43)
        x = sort(rand(200) .* 0.6 .+ 0.2)
        d = DensityEstimate(x, 10.0; support=(0.0, 1.0))
        for q in vcat(0.0, exp10.(-12:2:-2), 0.1:0.1:0.9, 1 .- exp10.(-12:2:-2), 1.0)
            @test cdf(d, quantile(d, q)) ≈ q atol = 1e-9
        end
        @test quantile(d, 0.0) == 0.0
        @test quantile(d, 1.0) == 1.0
        g = collect(range(-0.05, 1.05; length=3000))
        @test issorted(cdf(d, g))
    end

    @testset "evaluation is exactly zero outside the support" begin
        Random.seed!(44)
        x = sort(rand(100) .* 0.5 .+ 0.25)
        d = DensityEstimate(x, 12.0; support=(0.0, 1.0))
        @test d(-0.1) == 0 && d(1.1) == 0
        @test amplitude(d, -0.1) == 0 && amplitude(d, 1.1) == 0
        @test d(0.0) > 0 && d(1.0) > 0     # jump edge: nonzero right up to the wall
        @test amplitude(d, prevfloat(0.0)) == 0
        @test amplitude(d, nextfloat(1.0)) == 0
    end

    @testset "input validation" begin
        @test_throws DomainError DensityEstimate([0.5, 1.5], 1.0; support=(0.0, 1.0))
        @test_throws "lies outside the support" DensityEstimate([0.5, 1.5], 1.0; support=(0.0, 1.0))
        @test_throws DomainError DensityEstimate([-0.5, 0.5], 1.0; support=(0.0, 1.0))
        @test_throws "lies outside the support" DensityEstimate([-0.5, 0.5], 1.0; support=(0.0, 1.0))
        @test_throws DomainError DensityEstimate([0.5], 1.0; support=(1.0, 0.0))
        @test_throws "support must satisfy a < b" DensityEstimate([0.5], 1.0; support=(1.0, 0.0))
        @test_throws DomainError DensityEstimate([0.5], 1.0; support=(0.5, 0.5))
        @test_throws "support must satisfy a < b" DensityEstimate([0.5], 1.0; support=(0.5, 0.5))
    end

    @testset "chisq family runs on a bounded fit" begin
        d = DensityEstimate(sort(rand(Xoshiro(45), 50)), 10.0; support=(0.0, 1.0))
        r = chisq_reference(d)
        @test r isa ChisqReference
        @test expected_chisq(d) == r.mean > 0
        @test 0 <= chisq_ccdf(d, 1.0) <= 1
        @test chisq_pdf(d, 1.0) >= 0
        @test 0 <= pvalue(d, t -> 1.0) <= 1
    end

    @testset "cross-validation on a bounded fit" begin
        @testset "unbounded regression" begin
            Random.seed!(51)
            x = randn(800)
            xs = sort(x); w = ones(length(xs))
            for κ0 in (1.2, 5.0, 15.0)
                @test PenalizedDensity._klcv(xs, w, κ0, κ0, κ0, -Inf, Inf) === PenalizedDensity._klcv(xs, w, κ0)
                @test PenalizedDensity._lscv(xs, w, κ0, κ0, κ0, -Inf, Inf) === PenalizedDensity._lscv(xs, w, κ0)
            end
            @test select_kappa_kl(x) == select_kappa_kl(x; support=(-Inf, Inf))
            @test select_kappa_cv(x) == select_kappa_cv(x; support=(-Inf, Inf))
        end

        @testset "_int_quartic: boundary segment vs scale-matched quadrature" begin
            Random.seed!(52)
            x = sort(rand(300))
            d = DensityEstimate(x, 25.0; support=(0.0, 1.0))
            Q2 = PenalizedDensity._int_quartic(d.x, d.ψ, d.κ, d.κL, d.κR, d.lo, d.hi)
            # Sum scale-matched sub-segments (node-to-node plus the two boundary segments)
            # rather than one quadgk spanning the whole support, per the harness's spike-forest
            # discipline: a single call can converge to a misleadingly small error estimate.
            total = quadgk(t -> d(t)^2, d.lo, d.x[1]; rtol=1e-13)[1]
            for k in 1:length(d.x)-1
                total += quadgk(t -> d(t)^2, d.x[k], d.x[k+1]; rtol=1e-13)[1]
            end
            total += quadgk(t -> d(t)^2, d.x[end], d.hi; rtol=1e-13)[1]
            @test Q2 ≈ total rtol = 1e-10
        end

        @testset "brute-force validation of the leave-one-out expansion" begin
            function klcv_refit_b(nodes, weights, κ, lo, hi)
                s = 0.0
                for i in eachindex(nodes)
                    keep = [j for j in eachindex(nodes) if j != i]
                    s += weights[i] * log(DensityEstimate(nodes[keep], κ; support=(lo, hi))(nodes[i]))
                end
                return -s / sum(weights)
            end
            function lscv_refit_b(nodes, weights, κ, lo, hi)
                cross = 0.0
                for i in eachindex(nodes)
                    keep = [j for j in eachindex(nodes) if j != i]
                    cross += weights[i] * DensityEstimate(nodes[keep], κ; support=(lo, hi))(nodes[i])
                end
                di = DensityEstimate(nodes, κ; support=(lo, hi))
                Q2 = PenalizedDensity._int_quartic(di.x, di.ψ, di.κ, di.κL, di.κR, di.lo, di.hi)
                return Q2 - 2cross / sum(weights)
            end
            # Exponential (a hard left edge) and uniform (both edges hard), N a few hundred, at
            # several κ including each family's own KLCV-selected scale. Tolerances match the
            # unbounded suite's (`select_kappa_kl`/`select_kappa_cv` testsets above): KLCV's
            # equal per-node weighting under the log is most sensitive to sparse tail nodes,
            # where the first-order expansion is least accurate, so it gets the looser bound.
            for (name, xgen, support) in (
                    ("exponential", rng -> -log.(1 .- rand(rng, 300)), (0.0, Inf)),
                    ("uniform",     rng -> rand(rng, 300),              (0.0, 1.0)))
                rng = Xoshiro(hash((:boundedcv, name)))
                x = sort(xgen(rng))
                lo, hi = support
                nodes, w = PenalizedDensity._merge_presorted(x, 0.0)
                κsel = select_kappa_kl(x; support)
                for κ0 in (κsel * 0.5, κsel, κsel * 1.5)
                    a_kl = PenalizedDensity._klcv(nodes, w, κ0, κ0, κ0, lo, hi)
                    b_kl = klcv_refit_b(nodes, w, κ0, lo, hi)
                    @test a_kl ≈ b_kl rtol = 2e-2
                    a_ls = PenalizedDensity._lscv(nodes, w, κ0, κ0, κ0, lo, hi)
                    b_ls = lscv_refit_b(nodes, w, κ0, lo, hi)
                    @test a_ls ≈ b_ls rtol = 5e-3
                end
            end
        end

        @testset "selector: bounded vs unbounded κ on a hard-edge family" begin
            x = -log.(1 .- (0.5:1999.5) ./ 2000)   # exponential, a jump edge at the left
            κ_unb = select_kappa_kl(x)
            κ_bnd = select_kappa_kl(x; support=(0.0, Inf))
            @test κ_bnd != κ_unb
            @test κ_bnd < κ_unb    # the boundary itself represents the edge, so cross-validation
                                    # asks for less compensating smoothing, not more
            nodes, w = PenalizedDensity._merge_presorted(sort(x), 0.0)
            klcv_bnd = PenalizedDensity._klcv(nodes, w, κ_bnd, κ_bnd, κ_bnd, 0.0, Inf)
            klcv_unb = PenalizedDensity._klcv(nodes, w, κ_unb, κ_unb, κ_unb, 0.0, Inf)
            @test klcv_bnd < klcv_unb   # the bounded fit at its own selection beats it at the
                                        # unbounded selection, on the bounded fit's own score
        end

        @testset "select_kappa_kl / select_kappa_cv: input validation" begin
            @test_throws DomainError select_kappa_kl([0.2, 0.5, 0.8]; support=(0.5, 0.5))
            @test_throws "support must satisfy a < b" select_kappa_kl([0.2, 0.5, 0.8]; support=(0.5, 0.5))
            @test_throws DomainError select_kappa_kl([0.2, 0.5, 1.5]; support=(0.0, 1.0))
            @test_throws "lies outside the support" select_kappa_kl([0.2, 0.5, 1.5]; support=(0.0, 1.0))
            @test_throws DomainError select_kappa_cv([0.2, 0.5, 0.8]; support=(0.5, 0.5))
            @test_throws DomainError select_kappa_cv([-0.5, 0.5, 0.8]; support=(0.0, 1.0))
            @test_throws "lies outside the support" select_kappa_cv([-0.5, 0.5, 0.8]; support=(0.0, 1.0))
        end

        @testset "generic indexing: OffsetVector through select_kappa_kl(; support)" begin
            Random.seed!(53)
            x = rand(300)
            κ = select_kappa_kl(x; support=(0.0, 1.0))
            κo = select_kappa_kl(OffsetArray(x, -150); support=(0.0, 1.0))
            @test κo == κ
        end
    end

    @testset "quantile precision near q = 1 (complement inversion)" begin
        Random.seed!(54)
        x = sort(rand(200) .* 0.6 .+ 0.2)
        d = DensityEstimate(x, 10.0; support=(0.0, 1.0))
        for q in (1 - 1e-12, 1 - 1e-10, 1 - 1e-8, 1 - 1e-4)
            y = quantile(d, q)
            @test cdf(d, y) ≈ q atol = 4 * eps(1.0)
        end
    end

    @testset "_boundary_mass_from_node stays cancellation-free as v → u" begin
        Random.seed!(55)
        for _ in 1:50
            κ = exp(6 * rand() - 3)
            u = exp(8 * rand() - 2)
            ψ = 1.0 + rand()
            for ε in (1e-3, 1e-9, 1e-15)
                v = u * (1 - ε)
                m = PenalizedDensity._boundary_mass_from_node(ψ, κ, v, u)
                # Reference from the raw (unstabilized) closed form, in BigFloat so its own
                # cancellation is below the comparison tolerance.
                ub, vb, κb, ψb = big(u), big(v), big(κ), big(ψ)
                mref = Float64(ψb^2 * ((ub - vb) * sech(ub)^2 +
                                       cosh(ub + vb) * sinh(ub - vb) / cosh(ub)^2) / (2κb))
                @test m ≈ mref rtol = 1e-12
            end
        end
    end

    @testset "AdaptiveScale composition" begin
        x = -log.(1 .- (0.5:999.5) ./ 1000)
        κ = select_kappa_adaptive(x)
        d = DensityEstimate(x, κ; support=(0.0, Inf))
        @test d.lo == 0.0 && d.hi == Inf
        left, El = quadgk(d, 0.0, d.x[1]; rtol=1e-8)
        interior, Ei = quadgk(d, d.x...; rtol=1e-8)
        right, Er = quadgk(d, d.x[end], d.x[end] + 60; rtol=1e-8)
        @test max(El, Ei, Er) < 1e-6
        @test abs(left + interior + right - 1) < 1e-6
    end

    @testset "generic indexing: OffsetVector with support" begin
        x = [-1.5, 0.2, 0.2, 1.1, 3.4]
        d = DensityEstimate(x, 1.1; support=(-2.0, 4.0))
        do_ = DensityEstimate(OffsetArray(x, -3), 1.1; support=(-2.0, 4.0))
        @test do_.x == d.x && do_.ψ ≈ d.ψ && do_.lo == d.lo && do_.hi == d.hi
        @test do_(0.7) ≈ d(0.7)
    end

    @testset "stress: extreme and vanishing boundary gaps" begin
        # θ_L ≈ 500: far past where raw cosh/sinh would overflow.
        d = DensityEstimate([0.0, 1.0], 1.0; support=(-500.0, 501.0))
        @test all(isfinite, d.ψ) && isfinite(d.λ)
        @test d(-500.0) >= 0 && isfinite(d(-500.0))
        @test d(-500.1) == 0

        # θ_L ≈ 1e-8: the boundary sits essentially at the node.
        dn = DensityEstimate([0.0, 1.0], 1.0; support=(-1e-8, 1.0 + 1e-8))
        @test all(isfinite, dn.ψ)
        @test cdf(dn, -1e-8) == 0
        @test cdf(dn, 1.0 + 1e-8) == 1

        # Single-node fits, one and both walls finite: the analytic solution
        # ψ = √(w / (tanh(κΔl) + tanh(κΔr))), since M is the 1×1 matrix
        # tanh(κΔl) + tanh(κΔr) with no interior terms.
        Random.seed!(46)
        for _ in 1:20
            κ = exp(6 * rand() - 3)
            Δl = exp(8 * rand() - 2)
            Δr = exp(8 * rand() - 2)
            w = 1.0 + 3 * rand()
            M = PenalizedDensity.roughness_operator([0.0], κ, -Δl, Δr)
            φ = PenalizedDensity._solve_amplitude(M, [w])
            φexact = sqrt(w / (tanh(κ * Δl) + tanh(κ * Δr)))
            @test only(φ) ≈ φexact rtol = 1e-5

            # One wall finite, the other unbounded: the unbounded side's tail entry is 1,
            # the finite side's Δr = Inf limit of the same analytic solution.
            Mhalf = PenalizedDensity.roughness_operator([0.0], κ, -Δl, Inf)
            φhalf = PenalizedDensity._solve_amplitude(Mhalf, [w])
            φhalf_exact = sqrt(w / (tanh(κ * Δl) + 1))
            @test only(φhalf) ≈ φhalf_exact rtol = 1e-5
        end
    end
end

@testset "goodness of fit: exact reference on a bounded fit" begin
    @testset "unbounded regression" begin
        Random.seed!(60)
        x = randn(150)
        d = DensityEstimate(x, select_kappa_ms(x))
        r = chisq_reference(d)
        d2 = DensityEstimate(x, d.κ; support=(-Inf, Inf))
        r2 = chisq_reference(d2)
        @test r2.tri.dv == r.tri.dv && r2.tri.ev == r.tri.ev && r2.g == r.g && r2.mean == r.mean

        # A far wall (1e4 smoothing lengths out) reproduces the unbounded reference closely.
        far = 1e4 / d.κ
        dfar = DensityEstimate(x, d.κ; support=(minimum(x) - far, maximum(x) + far))
        rfar = chisq_reference(dfar)
        @test maximum(abs.(rfar.tri.dv .- r.tri.dv) ./ abs.(r.tri.dv)) < 1e-10
        @test maximum(abs.(rfar.g .- r.g) ./ abs.(r.g)) < 1e-10
        @test abs(rfar.mean - r.mean) / r.mean < 1e-10
        @test abs(chisq_ccdf(rfar, r.mean) - chisq_ccdf(r, r.mean)) < 1e-8
    end

    @testset "Green's identity on bounded fits" begin
        # 4Σᵢwᵢbᵢ/ψᵢ = 1 needs only Lψ_cl = 4Σ(wᵢ/ψᵢ)δ(x-xᵢ) and ∫ψ_cl² = 1 (see `green_identity`
        # above), so it exercises the bounded `_node_alpha`/`_operator` together on constant and
        # one- and two-sided supports.
        for (name, xgen, κ, support) in (
                ("uniform, both walls",   rng -> rand(rng, 300),                    15.0, (0.0, 1.0)),
                ("exponential, one wall", rng -> -log.(1 .- rand(rng, 300)),        12.0, (0.0, Inf)),
                ("triangular, both walls", rng -> rand(rng, 300) .+ rand(rng, 300), 10.0, (0.0, 2.0)))
            rng = Xoshiro(hash((:b4green, name)))
            x = sort(xgen(rng))
            d = DensityEstimate(x, κ; support)
            @test green_identity(d) ≈ 1 rtol = 1e-5
        end
        # Composed with adaptive κ.
        xa = sort!(randn(Xoshiro(9), 300) .^ 2)
        da = DensityEstimate(xa, select_kappa_adaptive(xa; support=(0.0, Inf)); support=(0.0, Inf))
        @test green_identity(da) ≈ 1 rtol = 1e-5
    end

    @testset "FD ground truth: m and ∫ψα converge to the closed forms at O(δ²)" begin
        # As the unbounded suite's analogous check above, but the grid now runs exactly from
        # `d.lo` to `d.hi` instead of a padded tail: the conservative stencil's Neumann condition
        # falls out of ending the grid there, per `fieldmc_chisq`'s comment.
        Random.seed!(62)
        x = sort(rand(60) .* 0.6 .+ 0.2)
        d = DensityEstimate(x, 6.0; support=(0.0, 1.0))
        errs = map((100, 400)) do per
            xn, ψ, λ = d.x, d.ψ, d.λ
            κ(k) = k == 0 ? d.κL : k > length(xn) - 1 ? d.κR : PenalizedDensity._kappa(d.κ, k)
            bnds = vcat(d.lo, xn, d.hi)
            y = Float64[]
            for k in 1:length(bnds)-1
                npts = max(2, ceil(Int, (bnds[k+1] - bnds[k]) * κ(k - 1) * per))
                append!(y, range(bnds[k], bnds[k+1]; length=npts + 1)[1:end-1])
            end
            push!(y, last(bnds)); m = length(y); δ = diff(y)
            cell = [searchsortedlast(bnds, (y[j] + y[j+1]) / 2) - 1 for j in 1:m-1]
            mass = [(j > 1 ? δ[j-1] : 0.0) / 2 + (j < m ? δ[j] : 0.0) / 2 for j in 1:m]
            p = copy(mass); q = zeros(m - 1)
            for j in 1:m-1
                c = 1 / (κ(cell[j])^2 * δ[j]); p[j] += c; p[j+1] += c; q[j] = -c
            end
            ψy = amplitude(d, y)
            αfd = SymTridiagonal(p, q) \ (mass .* ψy ./ 2λ)
            mfd = [αfd[searchsortedfirst(y, xi)] for xi in xn]
            mex = PenalizedDensity._node_alpha(xn, ψ, d.κ, d.κL, d.κR, λ, d.lo, d.hi)
            (maximum(abs.(mex .- mfd) ./ abs.(mfd)),
             abs(PenalizedDensity._int_psi_alpha(xn, ψ, mex, d.κ, d.κL, d.κR, λ, d.lo, d.hi) -
                 sum(mass .* ψy .* αfd)) / sum(mass .* ψy .* αfd))
        end
        @test all(<(2e-4), errs[1]) && all(<(2e-5), errs[2])
        @test all(first(errs) ./ last(errs) .> 10)
    end

    @testset "field Monte Carlo: bounded uniform (both walls, flagship)" begin
        # The flagship case: a natural boundary makes the flat field exactly representable,
        # which no unbounded or spatially-varying-κ fit could do.
        Random.seed!(63)
        x = sort(rand(400) .* 0.6 .+ 0.2)
        d = DensityEstimate(x, select_kappa_kl(x; support=(0.0, 1.0)); support=(0.0, 1.0))
        r = chisq_reference(d)
        chis = fieldmc_chisq(d; nsamp=60_000, seed=11)
        @test expected_chisq(r) ≈ mean(chis) rtol = 0.01
        for z in quantile(chis, (0.3, 0.6, 0.9))
            @test chisq_ccdf(r, z) ≈ mean(>(z), chis) atol = 0.012
        end
        @test green_identity(d) ≈ 1 rtol = 1e-5
    end

    @testset "field Monte Carlo: bounded exponential (one wall)" begin
        Random.seed!(64)
        x = sort(-log.(1 .- rand(400)))
        d = DensityEstimate(x, select_kappa_kl(x; support=(0.0, Inf)); support=(0.0, Inf))
        r = chisq_reference(d)
        chis = fieldmc_chisq(d; nsamp=60_000, seed=12)
        @test expected_chisq(r) ≈ mean(chis) rtol = 0.01
        for z in quantile(chis, (0.3, 0.6, 0.9))
            @test chisq_ccdf(r, z) ≈ mean(>(z), chis) atol = 0.012
        end
        @test green_identity(d) ≈ 1 rtol = 1e-5
    end

    @testset "field Monte Carlo: bounded fit with adaptive κ" begin
        Random.seed!(65)
        x = sort!(randn(Xoshiro(66), 300) .^ 2)
        κ = select_kappa_adaptive(x; support=(0.0, Inf))
        d = DensityEstimate(x, κ; support=(0.0, Inf))
        @test maximum(d.κ) / minimum(d.κ) > 1e4
        r = chisq_reference(d)
        chis = fieldmc_chisq(d; nsamp=60_000, seed=13)
        @test expected_chisq(r) ≈ mean(chis) rtol = 0.01
        for z in quantile(chis, (0.3, 0.6, 0.9))
            @test chisq_ccdf(r, z) ≈ mean(>(z), chis) atol = 0.012
        end
        @test green_identity(d) ≈ 1 rtol = 1e-5
    end
end

@testset "select_kappa_adaptive: fixed support" begin
    @testset "runs on a bounded domain and composes into a normalized fit" begin
        # Exponential on its natural support: runs, and the returned scale composes into a
        # normalized bounded fit.
        x = -log.(1 .- rand(Xoshiro(70), 500))
        κ = select_kappa_adaptive(x; support=(0.0, Inf))
        d = DensityEstimate(x, κ; support=(0.0, Inf))
        @test d.lo == 0.0 && d.hi == Inf
        left, El = quadgk(d, 0.0, d.x[1]; rtol=1e-8)
        interior, Ei = quadgk(d, d.x...; rtol=1e-8)
        right, Er = quadgk(d, d.x[end], d.x[end] + 60; rtol=1e-8)
        @test max(El, Ei, Er) < 1e-6
        @test abs(left + interior + right - 1) < 1e-6

        # Uniform on its true (0, 1) support: the machinery must run and the composed fit must
        # normalize; which α wins (possibly 0, the boundary already having fixed the edge) is
        # not asserted.
        xu = rand(Xoshiro(71), 500)
        κu = select_kappa_adaptive(xu; support=(0.0, 1.0))
        @test κu isa Real || κu isa AdaptiveScale
        du = DensityEstimate(xu, κu; support=(0.0, 1.0))
        @test cdf(du, 0.0) == 0.0 && cdf(du, 1.0) == 1.0
        mass, _ = quadgk(du, 0.0, 1.0; rtol=1e-8)
        @test mass ≈ 1 atol=1e-4
    end

    @testset "input validation" begin
        @test_throws DomainError select_kappa_adaptive([0.2, 0.5, 0.8]; support=(0.5, 0.5))
        @test_throws "support must satisfy a < b" select_kappa_adaptive([0.2, 0.5, 0.8]; support=(0.5, 0.5))
        @test_throws DomainError select_kappa_adaptive([-0.5, 0.5, 0.8]; support=(0.0, 1.0))
        @test_throws "lies outside the support" select_kappa_adaptive([-0.5, 0.5, 0.8]; support=(0.0, 1.0))
    end
end

@testset "select_support: joint boundary and κ selection" begin
    # A Student-t(5)-like heavy-tailed draw and a two-component mixture, built from `randn`
    # alone (no extra test dependency): t5 as a normal over the root-mean-square of five more
    # normals, the classical variance-mixture construction.
    _t5(rng) = randn(rng) / sqrt(sum(abs2, randn(rng, 5)) / 5)

    @testset "smooth families decline in every replicate" begin
        families = (
            ("gaussian", rng -> randn(rng, 500)),
            ("t5-like heavy tail", rng -> [_t5(rng) for _ in 1:500]),
            ("two-component mixture", rng -> vcat(randn(rng, 250) .- 2, randn(rng, 250) .+ 2)),
        )
        for (name, gen) in families, seed in 1:3
            x = gen(Xoshiro(hash((:smoothsupport, name, seed))))
            r = select_support(x)
            @test r.support == (-Inf, Inf)
            # Structural, not coincidental: when neither side wins, the returned κ *is* the
            # plain `select_kappa_kl(x)` call, not merely close to it.
            @test r.κ === select_kappa_kl(x)
        end
    end

    @testset "exponential: finite left, infinite right" begin
        for seed in 1:3
            x = -log.(1 .- rand(Xoshiro(hash((:expsupport, seed))), 1000))
            r = select_support(x)
            @test isfinite(r.support[1]) && r.support[2] == Inf
            xs = sort(x)
            spacing = PenalizedDensity._edge_spacing(xs, :left)
            gap = xs[1] - r.support[1]
            @test 0 < gap < 20 * spacing
            # The selected (κ, support) fit beats the plain unbounded selection on the same
            # KLCV score that chooses both.
            klsel = PenalizedDensity._support_klcv(xs, 1e-6, r.κ, r.support...)
            κunb = select_kappa_kl(x)
            klunb = PenalizedDensity._support_klcv(xs, 1e-6, κunb, -Inf, Inf)
            @test klsel < klunb
        end
    end

    @testset "uniform: finite on both sides" begin
        for seed in 1:3
            x = rand(Xoshiro(hash((:unifsupport, seed))), 1000)
            r = select_support(x)
            @test isfinite(r.support[1]) && isfinite(r.support[2])
            xs = sort(x)
            spL = PenalizedDensity._edge_spacing(xs, :left)
            spR = PenalizedDensity._edge_spacing(xs, :right)
            @test 0 < xs[1] - r.support[1] < 20 * spL
            @test 0 < r.support[2] - xs[end] < 20 * spR
            klsel = PenalizedDensity._support_klcv(xs, 1e-6, r.κ, r.support...)
            κunb = select_kappa_kl(x)
            klunb = PenalizedDensity._support_klcv(xs, 1e-6, κunb, -Inf, Inf)
            @test klsel < klunb
        end
    end

    @testset "χ²₁: finite left boundary (divergent edge)" begin
        for seed in 1:3
            x = randn(Xoshiro(hash((:chisqsupport, seed))), 500) .^ 2
            r = select_support(x)
            @test isfinite(r.support[1]) && r.support[2] == Inf
        end
    end

    @testset "input validation" begin
        @test_throws ArgumentError select_support([1.0, 2.0, 3.0]; κs=[1.0, -1.0, 2.0])
        @test_throws "κs must be sorted and positive" select_support([1.0, 2.0, 3.0]; κs=[1.0, -1.0, 2.0])
        @test_throws ArgumentError select_support([1.0, 2.0, 3.0]; κs=[1.0, 2.0])
        @test_throws "need at least 3 values in κs to bracket the minimum" select_support([1.0, 2.0, 3.0]; κs=[1.0, 2.0])
        @test_throws ArgumentError select_support([1.0, 2.0, 3.0]; rtol=-1.0)
        @test_throws "rtol must be nonnegative" select_support([1.0, 2.0, 3.0]; rtol=-1.0)
    end

    @testset "_edge_spacing and _select_gap failure paths" begin
        # Fewer than two points near the edge: unreachable through select_support itself (its
        # upstream κ selection already demands at least two distinct points), so exercised
        # directly, as the `_select_c` failure paths above are.
        @test_throws ArgumentError PenalizedDensity._edge_spacing([5.0], :left)
        @test_throws "need at least two distinct points to seed a boundary search" PenalizedDensity._edge_spacing([5.0], :left)

        # Ten points coincide at the left edge: zero spacing to seed a search from. Reachable
        # through the public API when many duplicates sit at one edge.
        x = vcat(fill(0.0, 10), [100.0])
        @test_throws ArgumentError select_support(x)
        @test_throws "the 10 points nearest the left edge coincide" select_support(x)

        # `_select_gap`'s own two failure paths, driven by synthetic scores as `_select_c`'s are.
        @test_throws ErrorException PenalizedDensity._select_gap(gap -> NaN, 1.0)
        @test_throws "no resolvable boundary gap" PenalizedDensity._select_gap(gap -> NaN, 1.0)
        @test_throws ErrorException PenalizedDensity._select_gap(gap -> -gap, 1.0)
        @test_throws "kept running off its search bracket" PenalizedDensity._select_gap(gap -> -gap, 1.0)
    end

    @testset "generic indexing: OffsetVector input" begin
        x = -log.(1 .- rand(Xoshiro(72), 500))
        r1 = select_support(x)
        r2 = select_support(OffsetVector(x, -250))
        @test r1.κ == r2.κ && r1.support == r2.support
    end

    @testset "seeded reproducibility" begin
        x = -log.(1 .- rand(Xoshiro(73), 500))
        r1 = select_support(x)
        r2 = select_support(x)
        @test r1.κ === r2.κ && r1.support === r2.support
    end

    @testset "efficiency (reported, not asserted — see the test log)" begin
        xbig = randn(Xoshiro(74), 50_000)
        t1 = @elapsed select_kappa_kl(xbig)
        t2 = @elapsed select_support(xbig)
        @info "select_support efficiency at N=50000" t_select_kappa_kl=t1 t_select_support=t2 ratio=t2 / t1
        @test isfinite(t2)   # the search completes; the timing itself is reported, not gated
    end
end

@testset "code quality (Aqua)" begin
    Aqua.test_all(PenalizedDensity)
end

@testset "explicit imports" begin
    @test ExplicitImports.check_no_implicit_imports(PenalizedDensity) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(PenalizedDensity) === nothing
    @test ExplicitImports.check_all_explicit_imports_via_owners(PenalizedDensity) === nothing
    @test ExplicitImports.check_no_self_qualified_accesses(PenalizedDensity) === nothing
end
