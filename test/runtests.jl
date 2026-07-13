using PenalizedDensity
using OffsetArrays
using QuadGK: quadgk
using Random, Statistics
using Random: randn!
using Test
using Aqua
using ExplicitImports

# Trapezoidal integral of a callable over a wide, fine grid.
function integrate(f, a, b; n=2_000_001)
    xs = range(a, b; length=n)
    ys = f.(xs)
    return (sum(ys) - (ys[1] + ys[end]) / 2) * step(xs)
end

# Direct Monte-Carlo of the field-theoretic χ² (Holy 1997, Eq. 16): sample the fluctuation
# field δψ from the constrained Gaussian on a fine grid and evaluate χ²(δψ). This is the
# ground truth the exact reference distribution must reproduce. Returns the χ² samples.
function fieldmc_chisq(d; nsamp=40_000, per_len=50, pad=12.0, seed=1)
    κ, λ = d.κ, d.λ; ℓ2 = 2λ / κ^2
    xlo = first(d.x) - pad / κ; xhi = last(d.x) + pad / κ
    m = ceil(Int, (xhi - xlo) / ((1 / κ) / per_len)) + 1
    y = range(xlo, xhi; length=m); δ = step(y)
    ψc = amplitude(d, collect(y))
    nearest(xi) = (i = searchsortedfirst(y, xi); i == 1 ? 1 : i > m ? m : (xi - y[i-1] < y[i] - xi ? i - 1 : i))
    gid = nearest.(d.x)
    p = fill(2λ * δ, m); q = fill(-ℓ2 / δ, m - 1)
    p[1] += ℓ2 / δ; p[m] += ℓ2 / δ; for j in 2:m-1; p[j] += 2ℓ2 / δ; end
    for (gi, s) in zip(gid, 2 .* d.w ./ d.ψ.^2); p[gi] += s; end
    # Cholesky P = RᵀR of the tridiagonal precision, R upper-bidiagonal (c diag, b super).
    c = similar(p); b = similar(q); c[1] = sqrt(p[1])
    for j in 2:m; b[j-1] = q[j-1] / c[j-1]; c[j] = sqrt(p[j] - b[j-1]^2); end
    solveR!(v, ξ) = (v[m] = ξ[m] / c[m]; for j in m-1:-1:1; v[j] = (ξ[j] - b[j] * v[j+1]) / c[j]; end; v)
    solveP!(v, a) = (u = similar(a); u[1] = a[1] / c[1];
        for j in 2:m; u[j] = (a[j] - b[j-1] * u[j-1]) / c[j]; end;
        v[m] = u[m] / c[m]; for j in m-1:-1:1; v[j] = (u[j] - b[j] * v[j+1]) / c[j]; end; v)
    v = solveP!(similar(ψc), ψc); aPa = sum(ψc .* v)     # for the ∫ψ_cl δψ = 0 constraint
    rng = MersenneTwister(seed)
    z = similar(ψc); ξ = similar(ψc); h = similar(ψc); iψ2 = d.w ./ d.ψ.^2
    chis = Vector{Float64}(undef, nsamp)
    for s in 1:nsamp
        randn!(rng, ξ); solveR!(z, ξ)
        proj = sum(ψc .* z) / aPa; @. h = z - proj * v
        chis[s] = 4 * sum(iψ2[k] * h[gid[k]]^2 for k in eachindex(gid))
    end
    return chis
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

        @testset "a one-node fit has only tails" begin
            a = DensityEstimate([0.3], t -> 0.5 + t)
            @test isempty(a.κ)
            @test a.κL == a.κR == 0.8
            @test cdf(a, 0.3) ≈ 0.5           # symmetric Laplace about the single point
            @test cdf(a, Inf) == 1
        end

        @testset "the χ² machinery refuses a varying scale" begin
            a = DensityEstimate(x, t -> 2.0 * exp(-t))
            msg = "defined only for a constant smoothing scale"
            @test_throws msg chisq_reference(a)
            @test_throws msg expected_chisq(a)
            @test_throws msg chisq_ccdf(a, 1.0)
            @test_throws msg chisq_pdf(a, 1.0)
            @test_throws msg pvalue(a, a)
            @test_throws msg chisq_ccdf(a, 1.0; method=:largeN)
            @test chisq(a, a) == 0            # the statistic itself is scale-free
        end

        @testset "input validation" begin
            for bad in (t -> -1.0, t -> 0.0, t -> NaN, t -> Inf)
                @test_throws "smoothing scale must be finite and positive" DensityEstimate(x, bad)
            end
            @test_throws "cannot be given as a vector" DensityEstimate(x, fill(2.0, length(x) - 1))
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
        μ = expected_chisq(d)
        @test μ > 0
        # Eq. 25: ⟨χ²⟩ is 1/√2 per effective bin (κX bins).
        N = sum(d.w); X = sum(d.w ./ d.ψ.^2) / N
        @test μ / (d.κ * X) ≈ 1 / sqrt(2)
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
        # The exact tail differs substantially from the large-N approximation.
        zt = quantile(chis, 0.99)
        @test chisq_ccdf(d, zt; method=:largeN) > 1.3 * chisq_ccdf(r, zt)

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

        # Generic indexing: OffsetArray input gives an identical reference.
        ro = chisq_reference(DensityEstimate(OffsetArray(x, -75), d.κ))
        @test ro.tri.dv ≈ r.tri.dv && ro.g ≈ r.g && expected_chisq(ro) ≈ μ
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

    @testset "large κ stays finite (no sinh overflow)" begin
        Random.seed!(3)
        x = randn(300)
        d = DensityEstimate(x, 5000.0)   # kernels far narrower than spacings
        @test all(isfinite, d.ψ) && all(>(0), d.ψ)
        @test isfinite(d.λ) && d.λ > 0
        @test isfinite(action(d))
        @test all(isfinite, amplitude(d, range(-4, 4; length = 200)))  # incl. inter-point gaps
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

@testset "code quality (Aqua)" begin
    Aqua.test_all(PenalizedDensity)
end

@testset "explicit imports" begin
    @test ExplicitImports.check_no_implicit_imports(PenalizedDensity) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(PenalizedDensity) === nothing
    @test ExplicitImports.check_all_explicit_imports_via_owners(PenalizedDensity) === nothing
    @test ExplicitImports.check_no_self_qualified_accesses(PenalizedDensity) === nothing
end
