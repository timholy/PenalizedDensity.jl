using PenalizedDensity
using PenalizedDensity: select_kappa_lscv, select_kappa_sensitivity, klcv, lscv, action_sensitivity
using DensityInterface: DensityInterface, DensityKind, HasDensity
import Distributions, StatsBase
using LinearAlgebra: SymTridiagonal, ZeroPivotException, ldlt!
using LogExpFunctions: logaddexp
using OffsetArrays
using QuadGK: quadgk
using Random, Statistics
using Random: randn!
using Test
using Aqua
using ExplicitImports

# Bytes allocated by one `_logΦ!` sweep, measured behind a function barrier: called with
# untyped arguments the sweep's `ComplexF64` return is boxed at the call site,
# which would be charged to the callee. The first call compiles.
function sweep_allocated(piv, rhs, r, u)
    PenalizedDensity._logΦ!(piv, rhs, r, u)
    return @allocated PenalizedDensity._logΦ!(piv, rhs, r, u)
end

# Bytes allocated by one pointwise evaluation `f(d, t)`, behind a function barrier for the
# same reason. The first call compiles.
function eval_allocated(f, d, t)
    f(d, t)
    return @allocated f(d, t)
end

# Trapezoidal integral of a callable over a wide, fine grid.
function integrate(f, a, b; n=2_000_001)
    xs = range(a, b; length=n)
    ys = f.(xs)
    return (sum(ys) - (ys[1] + ys[end]) / 2) * step(xs)
end

# The private fit and score routines on the unbounded line, with a scalar κ or a per-interval
# profile and its two tail rates.
fit_unbounded(n, w, κ::Real) = PenalizedDensity._fit(n, w, κ, -Inf, Inf)
fit_unbounded(n, w, κs::AbstractVector, κL, κR) = PenalizedDensity._fit(n, w, κs, κL, κR, -Inf, Inf)
klcv_unbounded(n, w, κ::Real; kw...) = PenalizedDensity._klcv(n, w, κ, κ, κ, -Inf, Inf; kw...)
klcv_unbounded(n, w, κs, κL, κR; kw...) = PenalizedDensity._klcv(n, w, κs, κL, κR, -Inf, Inf; kw...)
lscv_unbounded(n, w, κ::Real; kw...) = PenalizedDensity._lscv(n, w, κ, κ, κ, -Inf, Inf; kw...)
lscv_unbounded(n, w, κs, κL, κR; kw...) = PenalizedDensity._lscv(n, w, κs, κL, κR, -Inf, Inf; kw...)
# ∫Q̂² of a fit, at its own scale and support.
int_quartic(d) = PenalizedDensity._int_quartic(d.x, d.ψ, d.κ, d.κL, d.κR, d.lo, d.hi)

# Direct Monte-Carlo of the field-theoretic χ² (Holy 1997, Eq. 16): sample the fluctuation
# field δψ from the constrained Gaussian and evaluate χ²(δψ). The exact reference distribution
# must reproduce these samples at a constant or a varying scale, on ℝ or on a bounded support.
#
# The field's precision is L = 2λ𝒜 + 2Σ(wᵢ/ψᵢ²)δ(x-xᵢ), 𝒜u = -(κ(x)⁻²u′)′ + u, discretized
# conservatively: each cell of the grid contributes (2λ/κ²)(Δδψ)²/h to the gradient energy
# and 2λ·h to the mass. The grid contains every node, so κ is single-valued on each cell and
# the point sources land on grid points exactly. A finite `d.lo`/`d.hi` ends the grid there
# instead of padding into a decaying tail; the conservative stencil has no term coupling a grid
# edge to anything past it, so ending the grid at the wall imposes the natural (Neumann)
# boundary condition exactly. The discretization shares no assembly, Green's function, or
# recursion with the package. Returns the χ² samples.
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
    ψc = amplitude.(d, y)
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

# 4 Σᵢ wᵢ bᵢ/ψᵢ, which Green's identity makes equal to 1 for any scale, bounded or not (it needs only
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
        @test nobs(d) === 4.0
    end

    @testset "inner constructor validates field shapes" begin
        d = DensityEstimate([-1.0, 0.0, 0.5, 1.0], x -> 1 + x^2)
        # The stored node cumulatives (the last three fields) are assembled, not passed.
        f = Any[getfield(d, n) for n in fieldnames(typeof(d))[1:end-3]]
        DE = typeof(d)
        d2 = DE(f...)
        @test d2.ψ == d.ψ
        @test d2.F == d.F && d2.G == d.G && d2.total == d.total
        @test d.F == PenalizedDensity._node_cumulatives(d)[1]
        bad = copy(f); bad[2] = d.w[1:end-1]
        @test_throws "equal length" DE(bad...)
        bad = copy(f); bad[4] = [d.κ; 1.0]
        @test_throws "one rate per inter-node interval" DE(bad...)
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
        # The default rtol > 0 merges points far below the resolution.
        @test length(DensityEstimate([0.0, 1e-9, 1.0], 1.0).x) == 2
        # Merging points closer than the resolution is lossless.
        Random.seed!(5)
        x = randn(5_000)
        da = DensityEstimate(x, 3.0)
        db = DensityEstimate(x, 3.0; rtol=1e-3)
        g = range(-4, 4; length=101)
        @test maximum(abs.(da.(g) .- db.(g))) < 1e-3
    end

    @testset "merge_samples: nodes, weights, and the sample→node map" begin
        merge_samples = PenalizedDensity.merge_samples
        # Unsorted, with exact ties and near-ties on both sides of the rtol/κ boundary `b`; a
        # power of two keeps `2.0 + b - 2.0 == b` exact.
        b = 2.0^-10
        x = [5.0, b, 0.0, nextfloat(b), 5.0, 0.0, 2.0, 2.0 + b, 2.0 + 2.5b]
        m = merge_samples(x, 1.0; rtol = b)
        @test m.nodes == [0.0, nextfloat(b), 2.0, 2.0 + 2.5b, 5.0]
        @test m.weights == [3.0, 1.0, 2.0, 1.0, 2.0]
        @test m.nodeindex == [5, 1, 1, 2, 5, 1, 3, 3, 4]
        d = DensityEstimate(x, 1.0; rtol = b)
        @test m.nodes == d.x && m.weights == d.w
        # Each node's weight counts its samples; each sample lies within its run's tolerance.
        @test m.weights == [count(==(k), m.nodeindex) for k in eachindex(m.nodes)]
        @test all(j -> 0 <= x[j] - m.nodes[m.nodeindex[j]] <= b, eachindex(x))

        # Random data with many ties: agreement with the fit, at a constant and a callable κ.
        rng = Random.Xoshiro(11)
        xr = round.(randn(rng, 2_000); digits = 2)
        for κ in (3.0, x -> 2.0 + abs(x))
            mr = merge_samples(xr, κ; rtol = 1e-2)
            dr = DensityEstimate(xr, κ; rtol = 1e-2)
            @test mr.nodes == dr.x && mr.weights == dr.w
            @test mr.weights == [count(==(k), mr.nodeindex) for k in eachindex(mr.nodes)]
            @test all(k -> any(j -> mr.nodeindex[j] == k && xr[j] == mr.nodes[k], eachindex(xr)),
                      eachindex(mr.nodes))
        end

        # The map follows the argument's axes; the element type follows the promotion.
        xo = OffsetArray(x, -4)
        mo = merge_samples(xo, 1.0; rtol = b)
        @test axes(mo.nodeindex) == axes(xo) && parent(mo.nodeindex) == m.nodeindex
        @test mo.nodes == m.nodes
        @test eltype(merge_samples(Float32[1, 0, 2], 1.0f0).nodes) == Float32

        @test_throws "κ must be finite and positive, got 0.0" merge_samples(x, 0.0)
        @test_throws "κ must be finite and positive, got Inf" merge_samples(x, Inf)
        @test_throws "cannot merge zero points" merge_samples(Float64[], 1.0)
        @test_throws "rtol must be nonnegative" merge_samples(x, 1.0; rtol = -1.0)
        @test_throws "cannot be given as a vector" merge_samples(x, ones(length(x)))
        @test_throws ArgumentError merge_samples([0.0, NaN], 1.0)
    end

    @testset "solver converges on dense samples and allocates O(N)" begin
        # Dense samples make adjacent-node spacings small and ill-condition the tridiagonal
        # Newton system. rtol=0 disables merging of near-coincident points.
        Random.seed!(4)
        x = randn(20_000)
        d = DensityEstimate(x, 3.0; rtol=0.0)
        @test all(isfinite, d.ψ) && all(>(0), d.ψ)
        @test integrate(d, -40, 40) ≈ 1 atol = 1e-6
        # Stationarity of the normalized amplitude: Mψ = (κ/λ) w ./ ψ.
        M = PenalizedDensity.roughness_operator(d.x, d.κ)
        resid = M * d.ψ .- (d.κ / d.λ) .* d.w ./ d.ψ
        @test maximum(abs, resid) < 1e-6 * maximum(abs, M * d.ψ)
        DensityEstimate(x, 3.0; rtol=0.0)           # compile before measuring
        @test (@allocated DensityEstimate(x, 3.0; rtol=0.0)) < 40 * length(x) * sizeof(Float64)
    end

    @testset "SolveStats records solver effort without changing the fit" begin
        fresh = PenalizedDensity.SolveStats()
        @test (fresh.iterations, fresh.backtracks) == (0, 0)
        @test fresh.reason === :none
        @test isnan(fresh.final_step) && isnan(fresh.final_alpha)

        st = PenalizedDensity.SolveStats()
        x = randn(MersenneTwister(11), 300)
        d = DensityEstimate(x, 4.0; stats=st)
        # A well-conditioned fit stops on the correction tolerance, and collecting stats
        # does not change the fit.
        @test st.iterations >= 1
        @test st.reason === :tolerance
        @test st.final_step <= eps(Float64)^(3//4)
        @test 0 < st.final_alpha <= 1
        @test d.ψ == DensityEstimate(x, 4.0).ψ

        # Near-coincident nodes at rtol=0 ill-condition the solve, and roundoff stops it short
        # of the correction tolerance.
        st0 = PenalizedDensity.SolveStats()
        DensityEstimate(randn(MersenneTwister(7), 4_000), 3.0; rtol=0.0, stats=st0)
        @test st0.iterations >= 1
        @test st0.reason in (:floor, :steplength)
        @test st0.final_step > eps(Float64)^(3//4)

        # Nearly coincident nodes give operator entries (up to ~2e5 here) far larger than the
        # objective they nearly cancel to, so F's rounding noise far exceeds eps·|F|. The Armijo
        # test must scale its noise allowance by the size of F's terms; scaled by |F|, it
        # rejects every step and leaves the amplitudes wrong by ~1e-6.
        x = -log.(1 .- rand(Xoshiro(0x9a6a4425a4e3843c), 1000))
        xs = sort(x)
        κ, lo = 2.0855457048506127, xs[1] - 0.004565539700564811
        nodes, w = PenalizedDensity._merge_presorted(xs, cbrt(eps(Float64)) / κ)
        M = PenalizedDensity.roughness_operator(nodes, κ, lo, Inf)
        @test maximum(M.dv) > 1e5
        st2 = PenalizedDensity.SolveStats()
        ψ = PenalizedDensity._solve_amplitude(M, w; stats=st2)
        @test st2.reason in (:floor, :tolerance)
        @test st2.final_step < 1e-9
        # The returned amplitudes are stationary: a further Newton correction is at roundoff.
        g = M * ψ .- w ./ ψ
        Δ = ldlt!(SymTridiagonal(M.dv .+ w ./ ψ .^ 2, copy(M.ev))) \ g
        @test maximum(abs.(Δ) ./ ψ) < 1e-9

        @test sprint(show, PenalizedDensity.SolveStats()) == "SolveStats (no solve recorded)"
        @test startswith(sprint(show, st2), "SolveStats: $(st2.iterations) Newton steps, $(st2.backtracks) backtracks, stopped by :$(st2.reason)")
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
        κ = select_kappa_sensitivity(x; kappas=exp.(range(log(0.05), log(20); length=40)))
        d = DensityEstimate(x, κ)
        @test 0.75 < d.λ / length(x) < 1.25    # paper: λ ≈ N
        @test integrate(d, -40, 40) ≈ 1 atol = 1e-6
        φ(t) = exp(-t^2 / 2) / sqrt(2π)         # recovers a standard normal
        @test mean(abs.(d.(range(-4, 4; length=161)) .- φ.(range(-4, 4; length=161)))) < 0.05
    end

    @testset "analytic dS/dln κ matches finite differences" begin
        Random.seed!(6)
        nodes, w = PenalizedDensity._merge_presorted(sort(randn(1000)), 0.0)
        Sof(κ) = action(fit_unbounded(nodes, w, κ))
        for κ in (1.5, 5.0, 15.0)
            S, slope = PenalizedDensity._action_and_slope(nodes, w, κ)
            @test S ≈ Sof(κ)
            δ = 1e-5
            fd = (Sof(κ * exp(δ)) - Sof(κ * exp(-δ))) / (2δ)     # dS/dln κ
            @test slope ≈ fd rtol = 1e-3
        end
    end

    @testset "select_kappa_sensitivity: analytic, grid-independent" begin
        Random.seed!(2)
        x = randn(1500)
        κ = select_kappa_sensitivity(x)       # data-scaled default grid
        @test κ > 0
        # Golden-section refinement makes the result independent of the bracketing grid.
        κ10  = select_kappa_sensitivity(x; kappas = exp.(range(log(0.5), log(60); length = 10)))
        κ200 = select_kappa_sensitivity(x; kappas = exp.(range(log(0.5), log(60); length = 200)))
        @test κ10 ≈ κ200 rtol = 1e-3
        @test κ ≈ κ200 rtol = 1e-3
        # The returned κ is a local minimum of |dS/dln κ|.
        slope(κ) = abs(last(PenalizedDensity._action_and_slope(sort(x), ones(1500), κ)))
        @test slope(κ) < slope(κ * 1.3) && slope(κ) < slope(κ / 1.3)
    end

    @testset "held-out densities of isolated points track literal refits" begin
        # An outlier's held-out density is of order e^{-2κ·gap}: far below roundoff relative to
        # the fit, where a deletion step formed as a difference cancels. Compare every node with
        # a literal leave-one-out refit, through κ large enough to isolate most tail points.
        x = sort([randn(Xoshiro(1), 60); 5.0])
        w = ones(length(x))
        for κ in (0.5, 8.0, 50.0, 200.0)
            _, ℓ = PenalizedDensity._loo_logdensity(x, w, κ, κ, κ, -Inf, Inf)
            lit = [logdensity(DensityEstimate(deleteat!(copy(x), i), κ), x[i]) for i in eachindex(x)]
            @test maximum(abs, ℓ .- lit) < 0.1
        end
        # With a piecewise-constant scale and a natural boundary, the neighbor sum equals the
        # deleted amplitude φᵢ(1 - τh) wherever that difference does not cancel (h well below ½).
        PD = PenalizedDensity
        κs = collect(range(2.0, 40.0; length=length(x) - 1))
        for (lo, hi) in ((-Inf, Inf), (-3.0, 5.5))
            _, ℓ = PD._loo_logdensity(x, w, κs, 2.0, 40.0, lo, hi)
            @test all(isfinite, ℓ)
            @test PD._klcv(x, w, κs, 2.0, 40.0, lo, hi) ≈ -sum(ℓ) / length(x)
            M = PD._operator(x, κs, 2.0, 40.0, lo, hi)
            φ = PD._solve_amplitude(M, w)
            Z, Gφ = PD._norm_sq_gram(x, φ, κs, 2.0, 40.0, lo, hi)
            H = SymTridiagonal(M.dv .+ w ./ φ .^ 2, M.ev)
            g, _ = PD._inv_band(H)
            v = H \ Gφ
            h = g ./ φ .^ 2
            τ = PD._deletion_step.(h, w, w)
            ℓdiff = 2 .* log.(φ .* (1 .- τ .* h)) .- log.(Z .- 2 .* τ .* v ./ φ)
            ok = h .< 0.4
            @test count(ok) > 40
            @test ℓ[ok] ≈ ℓdiff[ok] rtol = 1e-10
        end
    end

    @testset "held-out locations: merged values and large tie groups" begin
        PD = PenalizedDensity
        # At small κ the merge tolerance rtol/κ spans much of the sample, so a node holds many
        # distinct values. Each is still held out alone: the two holdouts agree, and since the
        # fit is flat at 1 on (0, 1), every held-out log density is close to 0.
        xu = rand(Xoshiro(71), 500)
        for κ in (3e-3, 3e-5, 1e-5)
            nodes, _ = PD._merge_presorted(sort(xu), cbrt(eps()) / κ)
            s = klcv(xu, κ; support=(0.0, 1.0))
            @test s == klcv(xu, κ; support=(0.0, 1.0), holdout=:observation)
            @test abs(s) < 1e-4
            κ == 1e-5 && @test length(nodes) <= 5
        end
        @test PD._locations([1.0, 2.0, 2.0, 3.0, 3.0, 3.0], [3.0, 3.0]) == ([1, 1, 2], [1.0, 2.0, 3.0])
        # A tie group holding a large share of the sample is deleted exactly.
        for (support, base) in (((-Inf, Inf), randn(Xoshiro(1), 400)), ((0.0, 1.0), rand(Xoshiro(1), 400))),
                κ in (0.3, 30.0)
            xs = sort([base; fill(0.5, 100)])
            nodes, w = PD._merge_presorted(xs, cbrt(eps()) / κ)
            locs = PD._locations(xs, w)
            _, ℓ = PD._loo_logdensity(nodes, w, κ, κ, κ, support...; locations=locs)
            k = findfirst(==(100.0), locs[2])
            @test nodes[locs[1][k]] == 0.5
            @test ℓ[k] ≈ logdensity(DensityEstimate(base, κ; support), 0.5) rtol = 1e-8
            # The same with a distinct value merged into the tie group's node, which keeps it.
            xs = sort([base; fill(0.5, 100); 0.5 + 1e-9])
            nodes, w = PD._merge_presorted(xs, cbrt(eps()) / κ)
            locs = PD._locations(xs, w)
            _, ℓ = PD._loo_logdensity(nodes, w, κ, κ, κ, support...; locations=locs)
            k = findfirst(==(100.0), locs[2])
            @test count(==(locs[1][k]), locs[1]) == 2
            @test ℓ[k] ≈ logdensity(DensityEstimate([base; 0.5 + 1e-9], κ; support), 0.5) atol = 1e-7
        end
        # Ten coincident points and one other: leaving out the ten copies refits on the single
        # remaining point, and stays finite where that held-out density underflows (κ·gap ≫ 1).
        y = vcat(fill(0.0, 10), [100.0])
        for κ in (0.01, 1.0, 10.0)
            ℓ = PD._loo_logdensities(y, κ, cbrt(eps()), -Inf, Inf)
            @test ℓ[1:10] ≈ fill(logdensity(DensityEstimate([100.0], κ), 0.0), 10) rtol = 1e-8
            @test isfinite(klcv(y, κ; resolution=0))
        end
    end

    @testset "score functions" begin
        Random.seed!(12)
        x = randn(300)
        # Each selector returns the minimizer of its score.
        for (sel, score) in ((select_kappa, klcv), (select_kappa_lscv, lscv),
                             (select_kappa_sensitivity, action_sensitivity))
            κ = sel(x)
            s = score(x, κ)
            @test isfinite(s)
            @test s <= score(x, 0.9κ) && s <= score(x, 1.1κ)
        end
        @test klcv(x, 2.0; support = (-Inf, Inf)) == klcv(x, 2.0)
        @test isfinite(klcv(x, 2.0; support = (-10.0, 10.0)))
        @test isfinite(lscv(x, 2.0; support = (-10.0, 10.0)))
        @test klcv(x, 2.0; holdout = :observation) == klcv(x, 2.0)   # distinct values
        # Float type of the data.
        @test klcv(Float32.(x), 2) isa Float32
        @test lscv(Float32.(x), 2) isa Float32
        @test action_sensitivity(Float32.(x), 2) isa Float32
        # Generic axes.
        xo = OffsetArray(x, -50)
        @test klcv(xo, 2.0) == klcv(x, 2.0)
        @test lscv(xo, 2.0) == lscv(x, 2.0)
        @test action_sensitivity(xo, 2.0) == action_sensitivity(x, 2.0)
        # Rounded data: the `resolution` form is the interval form on the rounding intervals.
        xr = round.(x; digits = 1)
        @test klcv(xr, 3.0; resolution = 0.1) ≈ klcv(xr .- 0.05, xr .+ 0.05, 3.0)
        @test_throws "repeat another value, as rounded data do" klcv(xr, 3.0)
        @test isfinite(klcv(xr, 3.0; resolution = 0))
        @test_throws "lscv has no form for rounded (interval) data" lscv(xr, 3.0; resolution = 0.1)
        # Validation.
        for score in (klcv, lscv, action_sensitivity)
            @test_throws "κ must be finite and positive, got 0.0" score(x, 0.0)
            @test_throws "κ must be finite and positive, got Inf" score(x, Inf)
            @test_throws "is not a finite positive Float32" score(Float32.(x), 1e-50)
        end
        @test_throws "κ must be finite and positive, got -1" klcv(xr .- 0.05, xr .+ 0.05, -1)
        @test_throws "lies outside the support" klcv(x, 2.0; support = (0.0, 1.0))
    end

    @testset "select_kappa_lscv: cross-validated (MISE) scale" begin
        Random.seed!(11)
        x = randn(1500)
        κ = select_kappa_lscv(x)
        @test κ > 0
        # Golden-section refinement ⇒ independent of the bracketing grid.
        κ10  = select_kappa_lscv(x; kappas = exp.(range(log(0.3), log(40); length = 10)))
        κ200 = select_kappa_lscv(x; kappas = exp.(range(log(0.3), log(40); length = 200)))
        @test κ10 ≈ κ200 rtol = 1e-3

        # ∫Q̂² term: the closed form matches quadrature, including near-coincident points
        # (small θ), where a direct csch⁴ form loses accuracy.
        d = DensityEstimate(x, 4.0)
        ∫Q2 = int_quartic(d)
        g = range(d.x[1] - 15 / d.κ, d.x[end] + 15 / d.κ; length = 400_001)
        @test ∫Q2 ≈ sum(d(t)^2 for t in g) * step(g) rtol = 1e-4

        # LSCV score: the analytic one-step leave-one-out matches the exact leave-one-out refit
        # (wᵢ decremented, or a singleton node dropped).
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
                cross += weights[i] * fit_unbounded(collect(n2), collect(w2), κ)(nodes[i])
            end
            di = fit_unbounded(copy(nodes), copy(weights), κ)
            return int_quartic(di) - 2cross / sum(weights)
        end
        for κ0 in (1.5, 4.0, 12.0)
            @test lscv_unbounded(xs, w, κ0) ≈ lscv_refit(xs, w, κ0) rtol = 5e-4
        end

        # Tied branch (wᵢ > 1): rounding to 0.05σ gives high-multiplicity nodes, and the
        # analytic score matches the decrement refit.
        nt, wt = PenalizedDensity._merge_presorted(sort(round.(x ./ 0.05) .* 0.05), 1e-8)
        @test maximum(wt) > 5
        for κ0 in (1.5, 4.0)
            @test lscv_unbounded(nt, wt, κ0; holdout=:observation) ≈ lscv_refit(nt, wt, κ0) rtol = 1e-3
        end
        # The default leaves out every copy at a location; it matches the refit that drops the node.
        function lscv_refit_location(nodes, weights, κ)
            cross = 0.0
            for i in eachindex(nodes)
                keep = [j for j in eachindex(nodes) if j != i]
                cross += weights[i] * fit_unbounded(nodes[keep], weights[keep], κ)(nodes[i])
            end
            di = fit_unbounded(copy(nodes), copy(weights), κ)
            return int_quartic(di) - 2cross / sum(weights)
        end
        for κ0 in (1.5, 4.0)
            @test lscv_unbounded(nt, wt, κ0) ≈ lscv_refit_location(nt, wt, κ0) rtol = 5e-3
        end
        # On distinct points the two holdouts are the same score.
        @test lscv_unbounded(xs, w, 4.0) == lscv_unbounded(xs, w, 4.0; holdout=:observation)

        # MISE targeting: on smooth data the cross-validated scale is finer than the
        # minimum-sensitivity scale and gives a smaller integrated squared error.
        Q(t) = exp(-t^2 / 2) / sqrt(2π)
        ise(κ0) = (gg = range(-8, 8; length = 6001);
                   dd = DensityEstimate(x, κ0);
                   step(gg) * sum((dd(t) - Q(t))^2 for t in gg))
        @test κ < select_kappa_sensitivity(x)
        @test ise(κ) < ise(select_kappa_sensitivity(x))

        # Generic indexing: OffsetArray input gives an identical result.
        @test select_kappa_lscv(OffsetArray(x, -750)) == κ
    end

    @testset "select_kappa: likelihood (KL) scale" begin
        Random.seed!(11)
        x = randn(1500)
        κ = select_kappa(x)
        @test κ > 0
        # Golden-section refinement ⇒ independent of the bracketing grid.
        κ10  = select_kappa(x; kappas = exp.(range(log(0.3), log(40); length = 10)))
        κ200 = select_kappa(x; kappas = exp.(range(log(0.3), log(40); length = 200)))
        @test κ10 ≈ κ200 rtol = 1e-3

        # KLCV score from the analytic one-step leave-one-out densities matches the refit. The
        # tolerance is looser than for LSCV: the log weights every node equally, including tail
        # nodes where the expansion is least accurate, while LSCV weights them by the density.
        xs = sort(x); w = ones(length(xs))
        function klcv_refit(nodes, weights, κ)
            s = 0.0
            for i in eachindex(nodes)
                keep = [j for j in eachindex(nodes) if j != i]
                s += weights[i] * log(fit_unbounded(nodes[keep], weights[keep], κ)(nodes[i]))
            end
            return -s / sum(weights)
        end
        for κ0 in (1.5, 4.0, 12.0)
            @test klcv_unbounded(xs, w, κ0) ≈ klcv_refit(xs, w, κ0) rtol = 1e-3
        end

        # On a small, skewed sample the analytic KLCV minimizes near the scale an exact
        # leave-one-out grid search picks, and matches the refit near that optimum; a first-order
        # expansion would select a far too rough scale here. (At much larger κ, where one
        # collapsing node dominates the log-score, the two scores diverge.)
        xh = sort(exp.(randn(MersenneTwister(3), 25))); wh = ones(length(xh))
        span = xh[end] - xh[1]
        gridh = exp.(range(log(0.5 / span), log(5 * length(xh) / span); length = 41))
        κ_exact = gridh[argmin([klcv_refit(xh, wh, κ) for κ in gridh])]
        @test select_kappa(xh) ≈ κ_exact rtol = 0.2
        for κ0 in (0.5, 1.0)
            @test klcv_unbounded(xh, wh, κ0) ≈ klcv_refit(xh, wh, κ0) rtol = 5e-3
        end

        # Divergence targeting: on smooth data the KL scale is finer than the minimum-sensitivity
        # scale and gives a smaller Kullback–Leibler divergence to the truth.
        Q(t) = exp(-t^2 / 2) / sqrt(2π)
        kl(κ0) = (gg = range(-8, 8; length = 6001);
                  dd = DensityEstimate(x, κ0);
                  step(gg) * sum(Q(t) * log(Q(t) / dd(t)) for t in gg))
        @test κ < select_kappa_sensitivity(x)
        @test kl(κ) < kl(select_kappa_sensitivity(x))

        # Tracks the least-squares (MISE) scale to leading order.
        @test select_kappa(x) ≈ select_kappa_lscv(x) rtol = 0.5

        # Generic indexing: OffsetArray input gives an identical result.
        @test select_kappa(OffsetArray(x, -750)) == κ
    end

    @testset "amplitude² == density" begin
        d = DensityEstimate([-2.0, 0.5, 3.0], 0.9)
        g = range(-4, 5; length=25)
        @test amplitude.(d, collect(g)) .^ 2 ≈ d.(g)
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
        @test_throws DomainError cdf(d, NaN)
        @test_throws "the evaluation point must not be NaN" cdf(d, NaN)
        @test_throws DomainError quantile(d, -0.1)
        for q in (-0.1, 1.1, NaN)
            @test_throws "quantile is defined only for probabilities 0 ≤ q ≤ 1" quantile(d, q)
        end

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
            @test issorted(cdf.(d, g))
            @test issorted(quantile.(d, collect(0.0:0.0005:1.0)))
            # Continuity across the nodes.
            for xk in d.x
                @test cdf(d, prevfloat(xk)) ≈ cdf(d, xk) atol = 32eps()
                @test cdf(d, nextfloat(xk)) ≈ cdf(d, xk) atol = 32eps()
            end
        end

        # An isolated far node (θ = κ·Δx = 240): every branch stays finite and the
        # probability round trip survives the numerically flat valley.
        di = DensityEstimate([0.0, 1.0, 2.0, 50.0], 5.0)
        @test all(isfinite, quantile.(di, collect(0.001:0.001:0.999)))
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
        # A Float64 argument is evaluated in the fit's type.
        @test @inferred(cdf(d32, 0.5)) isa Float32
        @test @inferred(quantile(d32, 0.25)) isa Float32
        @test @inferred(gaussianize(d32, 0.5)) isa Float32
        @test @inferred(ungaussianize(d32, 0.5)) isa Float32
        @test @inferred(gaussianize_logjacobian(d32, 0.5)) isa NamedTuple{(:y, :logjac),Tuple{Float32,Float32}}
        # Pointwise evaluation promotes with the argument, including in a bounded boundary
        # segment and outside the support.
        d32b = DensityEstimate(Float32[0.2, 0.5, 1], 1.0f0; support=(0.0f0, Inf32))
        for dd in (d32, d32b), t in (-2.0, 0.1, 0.5, 3.0)
            @test @inferred(logdensity(dd, t)) isa Float64
            @test @inferred(dd(t)) isa Float64
            @test @inferred(amplitude(dd, t)) isa Float64
            @test @inferred(logdensity_eval_gradient(dd, t)) isa Float64
            @test @inferred(logdensity(dd, Float32(t))) isa Float32
            @test @inferred(logdensity_eval_gradient(dd, Float32(t))) isa Float32
            @test dd(t) ≈ dd(Float32(t)) rtol = 1e-5
        end
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
        @test exp(gaussianize_logjacobian(d, x0).logjac) ≈ κ * sqrt(2π) rtol = 1e-8

        # Limits, NaN.
        @test gaussianize(d, -Inf) == -Inf && gaussianize(d, Inf) == Inf
        @test ungaussianize(d, -Inf) == -Inf && ungaussianize(d, Inf) == Inf
        @test_throws "the evaluation point must not be NaN" gaussianize(d, NaN)
        @test_throws "the evaluation point must not be NaN" ungaussianize(d, NaN)
        @test_throws "the evaluation point must not be NaN" gaussianize_logjacobian(d, NaN)
        @test gaussianize_logjacobian(d, Inf).logjac == -Inf   # dy/dx → 0 in the tails

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
            # y-side round trip, from the bulk to far beyond where Φ underflows (|y| > 38.6),
            # through the log-space tail forms.
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
            @test issorted(gaussianize.(d, g))
            # Derivative against central finite differences, interior and tails.
            for t in vcat(collect(range(lo - 3 / κ, hi + 3 / κ; length = 41)),
                          lo - 50 / κ, hi + 50 / κ)
                h = 1e-6 / κ
                fd = (gaussianize(d, t + h) - gaussianize(d, t - h)) / (2h)
                @test exp(gaussianize_logjacobian(d, t).logjac) ≈ fd rtol = 1e-4
            end
            # The log-Jacobian is the log-derivative, and it makes the change of
            # variables exact: ln φ(y) + logjac = ln Q̂(x).
            for t in range(lo - 3 / κ, hi + 3 / κ; length = 41)
                (; y, logjac) = gaussianize_logjacobian(d, t)
                @test y == gaussianize(d, t)
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
        df = DensityEstimate(xtr, select_kappa(xtr))
        u = rand(5000)
        y = gaussianize.(df, quantile.(df, u))
        m = mean(y); v = var(y)
        @test abs(m) < 0.05 && abs(v - 1) < 0.07
        counts = [sum(0.1 * (b - 1) .<= Φ.(y) .< 0.1 * b) for b in 1:10]
        @test sum((counts .- 500) .^ 2 ./ 500) < 30     # χ²₉; multinomial noise only
        # Fresh draws from the underlying truth Gaussianize approximately (fit error).
        yfresh = gaussianize.(df, randn(5000))
        @test abs(mean(yfresh)) < 0.1 && abs(var(yfresh) - 1) < 0.15

        # ungaussianize(d, Z) with Z ~ N(0,1) samples the fit: compare tail masses.
        z = randn(5000)
        xs = ungaussianize.(df, z)
        @test mean(xs .< quantile(df, 0.25)) ≈ 0.25 atol = 0.03

        # An isolated far node (θ = κ·Δx = 240): every branch stays finite, monotone,
        # and invertible across the numerically flat valley.
        di = DensityEstimate([0.0, 1.0, 2.0, 50.0], 5.0)
        yi = gaussianize.(di, collect(range(-3.0, 53.0; length = 501)))
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
        @test @inferred(gaussianize_logjacobian(d32, 0.5f0)).logjac isa Float32
        @test gaussianize(d32, 0.5f0) ≈ gaussianize(d64, 0.5) atol = 1e-5
        y32 = gaussianize(d32, -30.0f0)
        @test y32 isa Float32 && isfinite(y32)
        @test ungaussianize(d32, y32) ≈ -30.0f0 rtol = 1e-6
    end

    @testset "gaussianize: bounded support and per-interval κ" begin
        Φ(t) = PenalizedDensity._Φ(t)
        φ(t) = exp(-t^2 / 2) / sqrt(2π)

        # Φ(gaussianize) reproduces the cdf, the map inverts, and ln φ(y) + logjac = ln Q̂(x),
        # for a finite support and a per-interval κ, over the interior and the boundary segments.
        Random.seed!(202)
        xb = clamp.(0.5 .+ 0.18 .* randn(400), 1e-3, 1 - 1e-3)
        adaptive = DensityEstimate(randn(400) .^ 2 .+ 0.1 .* randn(400),
                                   select_kappa(randn(400) .^ 2, Adaptive()))
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
                    # The cdf behind y is a quadrature at rtol = √eps, which bounds the
                    # accuracy of this identity.
                    @test log(φ(y)) + logjac ≈ log(d(t)) rtol = 1e-7
                end
                # Monotone across the support.
                @test issorted(gaussianize.(d, collect(range(lo, hi; length = 1001))))
            end
        end

        # Outside a compact support the map is ±Inf, and the inverse maps a saturated Gaussian
        # back to the wall.
        db = DensityEstimate(xb, 8.0; support = (0.0, 1.0))
        @test gaussianize(db, -0.05) == -Inf && gaussianize(db, 1.05) == Inf
        @test gaussianize_logjacobian(db, -0.05).logjac == -Inf
        @test gaussianize_logjacobian(db, 1.05).logjac == -Inf
        @test ungaussianize(db, -50.0) == 0.0 && ungaussianize(db, 50.0) == 1.0
        # A one-sided support keeps the exponential tail on the unbounded side.
        dr = DensityEstimate(xb, 8.0; support = (0.0, Inf))
        @test gaussianize(dr, -0.05) == -Inf
        @test isfinite(gaussianize(dr, 5.0)) && gaussianize(dr, Inf) == Inf

        # Distributional: draws from a bounded fit Gaussianize to N(0, 1).
        Random.seed!(303)
        u = rand(4000)
        du = DensityEstimate(u, select_kappa(u); support = (0.0, 1.0))
        y = gaussianize.(du, u)
        @test abs(mean(y)) < 0.05 && abs(var(y) - 1) < 0.07
        counts = [sum(0.1 * (b - 1) .<= Φ.(y) .< 0.1 * b) for b in 1:10]
        @test sum((counts .- 400) .^ 2 ./ 400) < 30     # χ²₉; multinomial noise only
    end

    @testset "generic indexing: OffsetArray input" begin
        x = [-1.5, 0.2, 0.2, 1.1, 3.4]
        d = DensityEstimate(x, 1.1)
        do_ = DensityEstimate(OffsetArray(x, -3), 1.1)   # offset axes
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

    @testset "integer data matches the same data as Float64" begin
        xi = [-3, -1, 0, 0, 1, 2, 4, 7, 8, 12]
        xf = float(xi)
        d, df = DensityEstimate(xi, 0.5), DensityEstimate(xf, 0.5)
        @test d.x == df.x && d.w == df.w && d.ψ == df.ψ
        @test PenalizedDensity.merge_samples(xi, 0.5) == PenalizedDensity.merge_samples(xf, 0.5)
        di, dif = IntervalDensityEstimate(xi, xi .+ 1, 0.5), IntervalDensityEstimate(xf, xf .+ 1, 0.5)
        @test di.x == dif.x && di.ψ == dif.ψ
        @test IntervalDensityEstimate(xi, 0.5; resolution=1).ψ == IntervalDensityEstimate(xf, 0.5; resolution=1).ψ
        @test select_kappa(xi; resolution=0) == select_kappa(xf; resolution=0)
        @test select_kappa(xi; resolution=1) == select_kappa(xf; resolution=1)
        @test select_kappa(xi, xi .+ 1) == select_kappa(xf, xf .+ 1)
        @test select_support(xi; resolution=0) == select_support(xf; resolution=0)
        @test PenalizedDensity.klcv(xi, 0.5; resolution=0) == PenalizedDensity.klcv(xf, 0.5; resolution=0)
        @test PenalizedDensity.klcv(xi, xi .+ 1, 0.5) == PenalizedDensity.klcv(xf, xf .+ 1, 0.5)
        @test PenalizedDensity.lscv(xi, 0.5; resolution=0) == PenalizedDensity.lscv(xf, 0.5; resolution=0)
        @test PenalizedDensity.select_kappa_lscv(xi; resolution=0) == PenalizedDensity.select_kappa_lscv(xf; resolution=0)
        @test PenalizedDensity.select_kappa_sensitivity(xi) == PenalizedDensity.select_kappa_sensitivity(xf)
        @test PenalizedDensity.action_sensitivity(xi, 0.5) == PenalizedDensity.action_sensitivity(xf, 0.5)
    end

    @testset "DensityEstimate{T}: every call form reaches one constructor" begin
        x = randn(MersenneTwister(2), 300)
        κfun(t) = 1.5 * (1 + exp(-t^2))
        d = DensityEstimate(x, 1.5)
        @test DensityEstimate{Float64}(x, 1.5).ψ == d.ψ
        @test DensityEstimate{Float64}(x, κfun).ψ == DensityEstimate(x, κfun).ψ
        @test DensityEstimate{Float64}(x, 1.5; support=(-5, 5)).ψ == DensityEstimate(x, 1.5; support=(-5, 5)).ψ
        # An explicit T converts the data and parameters, including a callable's values.
        d32 = DensityEstimate{Float32}(x, 1.5)
        @test d32 isa DensityEstimate{Float32,Float32}
        @test d32.ψ == DensityEstimate(Float32.(x), 1.5f0).ψ
        a32 = DensityEstimate{Float32}(x, κfun)
        @test a32 isa DensityEstimate{Float32,Vector{Float32}}
        @test a32.ψ == DensityEstimate(Float32.(x), t -> Float32(κfun(t))).ψ
        @test_throws "cannot be given as a vector" DensityEstimate{Float64}(x, [1.0, 2.0])
        @test_throws "zero points" DensityEstimate{Float32}(Float64[], 1.0)
        @test_throws "zero points" DensityEstimate(Float64[], κfun)
        # K is tied to T: a scale of another float type cannot be stored.
        @test_throws TypeError DensityEstimate{Float64,Vector{Float32}}(d.x, d.w, d.ψ, Float32[1], 1.0, 1.0,
                                                                        -Inf, Inf, 1.0)
    end

    @testset "broadcasting treats a fit as a scalar" begin
        x = [-1.5, 0.2, 0.2, 1.1, 3.4]
        fits = (DensityEstimate(x, 1.1),
                DensityEstimate(x, 1.1; support=(-2.0, 4.0)),
                IntervalDensityEstimate(x .- 0.1, x .+ 0.1, 1.1))
        xs = OffsetArray([-2.5, -0.5, 0.2, 0.9, 5.0], -3)
        qs = OffsetArray([0.05, 0.3, 0.5, 0.95], 2)
        ys = OffsetArray([-1.5, 0.0, 0.7], -1)
        for d in fits
            for f in (amplitude, logdensity, cdf, gaussianize, gaussianize_logjacobian,
                      logdensity_eval_gradient)
                b = f.(d, xs)
                @test axes(b) == axes(xs)
                @test isequal(b, map(x -> f(d, x), xs))   # gradient is NaN past a wall
            end
            @test d.(xs) ≈ map(d, xs)
            @test axes(quantile.(d, qs)) == axes(qs)
            @test quantile.(d, qs) == map(q -> quantile(d, q), qs)
            @test axes(ungaussianize.(d, ys)) == axes(ys)
            @test ungaussianize.(d, ys) == map(y -> ungaussianize(d, y), ys)
            bl = gaussianize_logjacobian.(d, xs)
            @test first.(bl) == gaussianize.(d, xs)
            # Elementwise evaluation is by broadcasting; there are no array methods, and the
            # MethodError hint says so.
            @test_throws MethodError cdf(d, xs)
            @test_throws "broadcast: `cdf.(d, xs)`" cdf(d, xs)
            @test_throws "broadcast: `logdensity.(d, xs)`" logdensity(d, xs)
            @test_throws "broadcast: `d.(xs)`" d(xs)
            @test_throws "`quantile.(d, ps)`" quantile(d, qs)
            @test_throws "not a collection" collect(d)
        end
        ref = ChisqReference(DensityEstimate(randn(MersenneTwister(1), 40), 1.0))
        zs = [0.5, 2.0, 8.0]
        @test pvalue.(ref, zs) == [pvalue(ref, z) for z in zs]
        @test chisq_ccdf.(ref, zs) == [chisq_ccdf(ref, z) for z in zs]
        @test chisq_pdf.(ref, zs) == [chisq_pdf(ref, z) for z in zs]
        @test_throws "broadcast: `pvalue.(ref, xs)`" pvalue(ref, zs)
        @test_throws "`mean(ref)` is its only summary statistic" std(ref)
        # The hint is silent for MethodErrors that broadcasting would not fix.
        d = fits[1]
        msg(f) = try f(); "" catch e; sprint(showerror, e) end
        @test !occursin("broadcast", msg(() -> entropy(d, [1.0 2.0])))
        @test !occursin("broadcast", msg(() -> negentropy(d, "x")))
        @test !occursin("collection", msg(() -> length(1 + 2im, 3)))
    end

    @testset "pointwise evaluation reads the stored node cumulatives" begin
        x = randn(MersenneTwister(3), 2000)
        fits = (DensityEstimate(x, 3.0),
                DensityEstimate(x, 3.0; support=(-6.0, 6.0)),
                DensityEstimate(x, t -> 2 + t^2),
                IntervalDensityEstimate(round.(x; digits=1), 3.0; resolution=0.1))
        for d in fits
            F, G, total = PenalizedDensity._node_cumulatives(d)
            @test d.F == F && d.G == G && d.total == total
            # No per-call O(length(d.x)) assembly: each evaluation allocates nothing.
            for (f, t) in ((cdf, 0.3), (quantile, 0.3), (gaussianize, 0.3),
                           (ungaussianize, 0.3), (gaussianize_logjacobian, 0.3))
                @test eval_allocated(f, d, t) == 0
            end
        end
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
                @test cdf.(a, ts) ≈ cdf.(d, ts) rtol=1e-8
                @test quantile.(a, 0.01:0.01:0.99) ≈ quantile.(d, 0.01:0.01:0.99) rtol=1e-8
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
            # cdf and quantile invert each other on a varying scale.
            qs = 0.001:0.017:0.999
            @test cdf.(a, quantile.(a, collect(qs))) ≈ qs atol=1e-12
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
                @test klcv_unbounded(cn, cw, flat, κ0, κ0) ≈ klcv_unbounded(cn, cw, κ0) rtol=1e-11
                @test lscv_unbounded(cn, cw, flat, κ0, κ0) ≈ lscv_unbounded(cn, cw, κ0) rtol=1e-11
            end

            # ∫Q̂², the LSCV roughness term, against quadrature over the segments the varying
            # scale defines.
            a = fit_unbounded(nodes, w, κs, κL, κR)
            ∫Q2 = int_quartic(a)
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
                fit_unbounded(n2, w2, κs2, κL2, κR2)(nodes[i])
            end
            @test klcv_unbounded(nodes, w, κs, κL, κR) ≈
                  -sum(w .* log.(loo)) / W rtol=2e-2
            @test lscv_unbounded(nodes, w, κs, κL, κR) ≈
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
            r = ChisqReference(a)
            @test r isa ChisqReference
            @test mean(r) == r.mean > 0
            # :largeN is the Wald shape at the exact mean, so it too works at any scale.
            @test 0 ≤ chisq_ccdf(r, 1.0; method=:largeN) ≤ 1
            @test chisq_pdf(r, 1.0; method=:largeN) ≥ 0
            @test 0 < pvalue(r, mean(r); method=:largeN) < 1
            @test 0 < pvalue(r, mean(r)) < 1
            @test_throws "method must be :exact or :largeN" chisq_ccdf(r, 1.0; method=:bogus)
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
        klcv_const(x, κ) = PenalizedDensity._score_at(PenalizedDensity._klcv, x, rtol, -Inf, Inf, κ)
        klcv_scale(x, k) = PenalizedDensity._score_kappa(PenalizedDensity._klcv, x, k, rtol, -Inf, Inf)

        @testset "AdaptiveScale evaluates c·(p̂/ḡ)^α" begin
            κ0 = select_kappa(chisq1)
            p = DensityEstimate(chisq1, κ0)
            k = AdaptiveScale(3.0, 0.5, p)
            lgbar = mean(log(p(xi)) for xi in chisq1)     # ln of the geometric mean of p̂
            for t in (0.05, 0.3, 1.0, 2.5, 6.0)
                @test k(t) ≈ 3.0 * (p(t) / exp(lgbar))^0.5
            end
            # Where the scale equals c, the pilot density equals its geometric mean.
            @test k(chisq1[argmin(abs.(log.(p.(chisq1)) .- lgbar))]) ≈ 3.0 rtol=1e-3

            # The selector's batch path, one walk over the pilot's nodes, agrees exactly with
            # the scalar rule.
            ts = sort!(vcat(chisq1[1:50:end], range(1e-4, 9.0; length=97)))
            @test PenalizedDensity._kappa_sorted(k, ts, Float64) == k.(ts)

            # Far out in the tail the pilot density underflows to zero; the rule floors there
            # rather than producing a zero (and an infinite) smoothing length.
            far = last(chisq1) + 1e4
            @test p(far) == 0
            @test k(far) == k.κmin == 1e-6 * k.c

            # A shallow exponent keeps the rule off its floor where the pilot density has
            # underflowed; there the scalar and batch paths agree only if both use the
            # log-density.
            ks = AdaptiveScale(3.0, 5e-3, p)
            deep = last(chisq1) .+ [40.0, 45.0, 50.0]
            @test all(t -> p(t) == 0, deep)
            @test all(t -> ks(t) > ks.κmin, deep)
            @test PenalizedDensity._kappa_sorted(ks, deep, Float64) == ks.(deep)

            @test sprint(show, k) ==
                  "AdaptiveScale(c=3.0, α=0.5) over a pilot with $(length(p.x)) nodes"
        end

        @testset "an adaptive scale is selected only when it scores better" begin
            κ = select_kappa(chisq1, Adaptive(; window=nothing))
            @test κ isa AdaptiveScale
            @test κ.α > 0
            # The chosen scale scores better than the constant one.
            @test klcv_scale(chisq1, κ) < klcv_const(chisq1, select_kappa(chisq1))

            d = DensityEstimate(chisq1, κ)
            @test length(d.κ) == length(d.x) - 1
            # The scale follows the density: finest near the divergent edge (where the pilot
            # peaks), coarsest in the tail.
            @test d.x[argmax(d.κ)] < 0.1
            @test d.x[argmin(d.κ)] > 5
            @test extrema(d.κ)[2] / extrema(d.κ)[1] > 100
            @test cdf(d, Inf) == 1

            # For uniform data κ ∝ p̂^α is already constant, so the α = 0 candidate is selected
            # and a scalar is returned.
            u = sort!(rand(Xoshiro(5), 3000))
            κu = select_kappa(u, Adaptive(; window=nothing))
            @test κu isa Real
            @test DensityEstimate(u, κu).κ isa Real
            @test ChisqReference(DensityEstimate(u, κu)) isa ChisqReference
        end

        @testset "alphas and pilot are honored" begin
            κ = select_kappa(chisq1, Adaptive(; window=nothing, alphas=(1.0,)))
            @test κ isa AdaptiveScale && κ.α == 1.0
            # pilot: any callable returning a positive scale from the sample.
            κms = select_kappa(chisq1, Adaptive(; window=nothing, alphas=(0.5,), pilot=select_kappa_sensitivity))
            @test κms isa AdaptiveScale
            @test κms.pilot.κ == select_kappa_sensitivity(chisq1)
            # Offset input is merged and sorted like any other vector.
            @test select_kappa(OffsetVector(chisq1, -1500), Adaptive(; window=nothing)) isa AdaptiveScale
        end

        @testset "per-observation leave-one-out log densities" begin
            loo(x, k) = PenalizedDensity._loo_logdensities(x, k, rtol, -Inf, Inf)
            κ0 = select_kappa(chisq1)
            p = DensityEstimate(chisq1, κ0)
            ka = AdaptiveScale(8.0, 0.75, p)
            tied = round.(chisq1; digits=2)          # sorted, and heavily tied
            @test length(unique(tied)) < length(tied) ÷ 2
            for x in (chisq1, tied)
                ℓ0 = loo(x, κ0)
                @test length(ℓ0) == length(x)
                @test -mean(ℓ0) ≈ klcv_const(x, κ0) rtol=1e-12
                ℓa = loo(x, ka)
                @test length(ℓa) == length(x)
                @test -mean(ℓa) ≈ klcv_scale(x, ka) rtol=1e-12
                # Tied observations share their node's value.
                for j in 2:length(x)
                    x[j] == x[j-1] && @test ℓa[j] == ℓa[j-1]
                end
            end
            # On a finite support, through the 7-argument scoring.
            lo, hi = 0.0, Inf
            ℓs = PenalizedDensity._loo_logdensities(chisq1, ka, rtol, lo, hi)
            @test -mean(ℓs) ≈ PenalizedDensity._score_kappa(PenalizedDensity._klcv, chisq1, ka, rtol, lo, hi) rtol=1e-12
            # An unresolvable candidate: all NaN.
            @test all(isnan, PenalizedDensity._loo_logdensities([1.0, 1.0], 5.0, rtol, -Inf, Inf))
        end

        @testset "one-standard-error rule" begin
            # nse = 0 is the minimum-score choice, and the gain is the score difference.
            κa = select_kappa(chisq1, Adaptive(; window=nothing, nse=0))
            @test κa isa AdaptiveScale
            κ0 = select_kappa(chisq1)
            gain, se = PenalizedDensity._adaptive_gain(chisq1, κ0, κa, rtol, -Inf, Inf)
            @test gain ≈ klcv_const(chisq1, κ0) - klcv_scale(chisq1, κa) atol=1e-12
            @test gain > 0 && se > 0

            # A smooth density where the best exponent edges out the constant scale by far less
            # than one standard error: nse = 0 takes the adaptive scale, nse = 1 the constant.
            xn = sort!(randn(Xoshiro(3), 500))
            κn0 = select_kappa(xn, Adaptive(; window=nothing, nse=0))
            @test κn0 isa AdaptiveScale
            κn = select_kappa(xn, Adaptive(; window=nothing))
            @test κn isa Real && κn == select_kappa(xn)
            gain, se = PenalizedDensity._adaptive_gain(xn, κn, κn0, rtol, -Inf, Inf)
            @test 0 < gain < se
            @test gain ≈ klcv_const(xn, κn) - klcv_scale(xn, κn0) atol=1e-12

            # A divergent edge: adaptivity wins by several standard errors.
            xc = sort!(randn(Xoshiro(2), 250) .^ 2)
            κc = select_kappa(xc, Adaptive(; window=nothing))
            @test κc isa AdaptiveScale
            gain, se = PenalizedDensity._adaptive_gain(xc, select_kappa(xc), κc, rtol, -Inf, Inf)
            @test gain > 3se
            @test select_kappa(xc, Adaptive(; window=nothing, nse=Inf)) isa Real

            @test_throws "nse must be nonnegative" select_kappa(xc, Adaptive(; nse=-1))
        end

        @testset "the c search brackets its minimum" begin
            # Synthetic scores, passed as the objective, exercise the bracketing directly.
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

            # The returned c always resolves: a refinement that ends on an unresolvable c falls
            # back to the best grid point. Here only the opening grid resolves.
            lngrid = range(log(2.0) - log(PenalizedDensity._CSPAN), log(2.0) + log(PenalizedDensity._CSPAN);
                           length=PenalizedDensity._CGRID)
            gridonly(c) = any(g -> abs(log(c) - g) < 1e-9, lngrid) ? (log(c) - log(2.1))^2 : NaN
            c, s = PenalizedDensity._select_c_scored(gridonly, 2.0)
            @test c ≈ 2.0 && s == gridonly(c)

            # Nothing resolvable anywhere in the opening bracket.
            @test_throws PenalizedDensity.ConvergenceError sel(_ -> NaN, 1.0)
            @test_throws "no resolvable smoothing scale" sel(_ -> NaN, 1.0)
            @test PenalizedDensity._select_c_scored(_ -> NaN, 1.0; skip_unresolved=true) === nothing
            # A score with no interior minimum runs off the bracket until it gives up.
            @test_throws "kept running off the search bracket" sel(c -> -log(c), 1.0)
        end

        @testset "the fixed-grid κ search" begin
            mog(score, κs) = PenalizedDensity._minimize_on_grid(score, κs, Float64)
            κs = exp.(range(log(0.1), log(100); length=13))
            @test mog(κ -> (log(κ) - log(3.0))^2, κs) ≈ 3.0 rtol=1e-6
            # Non-finite scores are stepped over, never chosen, whichever sign they carry.
            @test mog(κ -> κ < 1 ? NaN : (log(κ) - log(3.0))^2, κs) ≈ 3.0 rtol=1e-6
            @test mog(κ -> κ > 10 ? -Inf : (log(κ) - log(3.0))^2, κs) ≈ 3.0 rtol=1e-6
            # A grid with no finite score is an error.
            @test_throws PenalizedDensity.ConvergenceError mog(_ -> NaN, κs)
            @test_throws "no κ in kappas" mog(_ -> Inf, κs)
            # A minimum at either end of a given grid is an error.
            @test_throws "the minimum is at the upper end of kappas (κ = 100.0" mog(κ -> (log(κ) - log(500.0))^2, κs)
            @test_throws "the minimum is at the lower end of kappas (κ = 0.1" mog(κ -> (log(κ) - log(0.01))^2, κs)
            # An extensible (default) grid grows past either end until the minimum is inside,
            # and reports the range it searched.
            sg(score; kw...) = PenalizedDensity._search_grid(score, κs, Float64; extend=true, kw...)
            κ, κlo, κhi = sg(κ -> (log(κ) - log(500.0))^2)
            @test κ ≈ 500 rtol=1e-6
            @test κlo ≈ first(κs) && 500 < κhi < 1e5
            κ, κlo, κhi = sg(κ -> (log(κ) - log(0.01))^2)
            @test κ ≈ 0.01 rtol=1e-6
            @test 1e-4 < κlo < 0.01 && κhi ≈ last(κs)
            @test all(sg(κ -> (log(κ) - log(3.0))^2)[2:3] .≈ (first(κs), last(κs)))  # interior: no growth
            # ... but not past κmax, where a minimum at the top is accepted.
            κ, _, κhi = sg(κ -> (log(κ) - log(1e4))^2; κmax=300.0)
            @test κhi ≈ 300 && 200 < κ <= 300
            # ... and not without limit.
            @test_throws "kept running off the default κ grid" sg(κ -> -log(κ))
        end

        @testset "edge minima through the selectors" begin
            x = randn(Xoshiro(1), 200)
            tiny = [0.01, 0.02, 0.03]
            for sel in (select_kappa, select_kappa_lscv, select_kappa_sensitivity)
                @test_throws PenalizedDensity.ConvergenceError sel(x; kappas=tiny)
                @test_throws "the minimum is at the upper end of kappas (κ = 0.03)" sel(x; kappas=tiny)
                @test_throws "at the lower end of kappas (κ = 100.0)" sel(x; kappas=[100.0, 200.0, 300.0])
            end
            @test_throws "at the upper end of kappas" select_support(x; kappas=tiny)
            z = round.(x ./ 0.25) .* 0.25
            @test_throws "at the upper end of kappas" select_kappa(z; resolution=0.25, kappas=tiny)
            # A default grid extends: Cauchy data spread the default grid far below the bulk's scale.
            xc = tan.(π .* (rand(Xoshiro(2), 1000) .- 0.5))
            κs = PenalizedDensity._default_κs(xc)
            κ = select_kappa_sensitivity(xc)
            @test κ > last(κs)
            @test κ ≈ select_kappa_sensitivity(xc; kappas=exp.(range(log(first(κs)), log(10last(κs)); length=60))) rtol=1e-4
        end

        @testset "an unresolvable candidate scores NaN" begin
            # A κ profile spanning many orders of magnitude can give the factorization an exact
            # zero pivot; that candidate scores NaN.
            zero_pivot(args...; kw...) = throw(ZeroPivotException(3))
            @test isnan(PenalizedDensity._score_kappa(zero_pivot, chisq1, _ -> 5.0, rtol, -Inf, Inf))
            # Any other exception propagates.
            other(args...; kw...) = throw(DomainError(1.0, "something else went wrong"))
            @test_throws DomainError PenalizedDensity._score_kappa(other, chisq1, _ -> 5.0, rtol, -Inf, Inf)
            # Too few nodes to fit: scored NaN before the scorefun is ever reached.
            @test isnan(PenalizedDensity._score_kappa(zero_pivot, [1.0], _ -> 5.0, rtol, -Inf, Inf))
        end

        @testset "input validation" begin
            @test_throws "at least one exponent" select_kappa(chisq1, Adaptive(; alphas=()))
            @test_throws "must be positive" select_kappa(chisq1, Adaptive(; alphas=(0.0, 0.5)))
            @test_throws "must be positive" select_kappa(chisq1, Adaptive(; alphas=(-1.0,)))
            @test_throws "rtol must be nonnegative" select_kappa(chisq1, Adaptive(); rtol=-1.0)
            @test_throws "pilot must return a positive scale" select_kappa(chisq1, Adaptive(; pilot=_ -> 0.0))
            p = DensityEstimate(chisq1, 10.0)
            @test_throws "exponent α must be positive" AdaptiveScale(1.0, 0.0, p)
            @test_throws "scale c must be positive" AdaptiveScale(0.0, 1.0, p)
            # Every call form validates and derives ln ḡ and the floor the same way.
            D = typeof(p)
            for make in ((c, α) -> AdaptiveScale(c, α, p), (c, α) -> AdaptiveScale{Float64}(c, α, p),
                         (c, α) -> AdaptiveScale{Float64,D}(c, α, p))
                @test_throws "exponent α must be positive" make(1.0, 0.0)
                @test_throws "scale c must be positive" make(-1.0, 1.0)
                k = make(3, 1 // 2)
                @test k isa AdaptiveScale{Float64,D}
                @test (k.c, k.α, k.loggbar, k.κmin) ==
                      (3.0, 0.5, PenalizedDensity._log_geomean(p), 3e-6)
            end
            k32 = AdaptiveScale{Float32}(3.0, 0.5, p)
            @test k32 isa AdaptiveScale{Float32} && k32.c === 3.0f0
            @test_throws TypeError AdaptiveScale{Float64,Vector{Float64}}(3.0, 0.5, [1.0])
        end
    end

    @testset "adaptive κ: the window rule" begin
        PD = PenalizedDensity
        rtol = cbrt(eps(Float64))
        klcv_scale(x, k) = PD._score_kappa(PD._klcv, x, k, rtol, -Inf, Inf)
        klcv_const(x, κ) = PD._score_at(PD._klcv, x, rtol, -Inf, Inf, κ)

        @testset "window statistics match their definition" begin
            y = sort!(randn(Xoshiro(1), 500))
            W = PD._WindowMoments(y)
            for (x0, h) in ((0.3, 0.7), (-1.2, 0.25), (2.0, 3.0))
                u = [(t - x0) / h for t in y if abs(t - x0) <= h]
                N = length(u)
                s = PD._window_statistics(W, x0, h)
                @test s.N == N
                @test s.slope ≈ abs(sum(u)) / sqrt(N / 3)
                @test s.curvature ≈ abs(sum(u .^ 2) - N / 3) / sqrt(4N / 45)
            end
            @test PD._window_statistics(W, 100.0, 1.0).N == 0
        end

        @testset "on a lattice, windows grow until they reach an edge" begin
            # Evenly spaced points show no slope or curvature in any window inside the range,
            # so the first crossing comes only once the window runs past the nearest edge.
            xl = collect(range(0, 1; length=2001))
            W = PD._WindowMoments(xl)
            ts = (0.1, 0.25, 0.5, 0.75, 0.9)
            hs = [PD._window_halfwidth(W, t, 4.0, 30) for t in ts]
            @test all(hs .> min.(ts, 1 .- ts))
            @test argmax(hs) == 3
            @test hs ≈ reverse(hs) rtol=1e-8
        end

        @testset "WindowScale evaluates c·exp(γ·s)" begin
            k = WindowScale(2.0, 0.5, [0.0, 1.0, 3.0], [1.0, -1.0, 0.0])
            @test k(0.5) ≈ 2.0                         # s interpolates linearly: s(0.5) = 0
            @test k(2.0) ≈ 2.0 * exp(0.5 * -0.5)
            @test k(-10.0) ≈ 2.0 * exp(0.5)            # held constant beyond the grid
            @test k(10.0) ≈ 2.0
            @test WindowScale(2.0, 1e3, [0.0, 1.0], [0.0, -1.0])(1.0) == 2e-6   # the floor
            @test sprint(show, k) == "WindowScale(c=2.0, γ=0.5) on a grid of 3 points"
            @test WindowScale(2.0f0, 0.5f0, Float32[0, 1], Float32[0, 1]) isa WindowScale{Float32}
            @test_throws "scale c must be positive" WindowScale(0.0, 1.0, [0.0, 1.0], [0.0, 0.0])
            @test_throws "scale c must be positive" WindowScale{Float64}(0.0, 1.0, [0.0, 1.0], [0.0, 0.0])
            k64 = WindowScale{Float64}(2, 1 // 2, [0, 1, 3], [1, -1, 0])
            @test (k64.c, k64.γ, k64.x, k64.logb, k64.κmin) == (k.c, k.γ, k.x, k.logb, k.κmin)
            @test_throws "grid must be sorted" WindowScale(1.0, 1.0, [1.0, 0.0], [0.0, 0.0])
            @test_throws "must be finite" WindowScale(1.0, 1.0, [0.0, 1.0], [0.0, Inf])
            @test_throws DimensionMismatch WindowScale(1.0, 1.0, [0.0, 1.0], [0.0])
            @test_throws "the grid is empty" WindowScale(1.0, 1.0, Float64[], Float64[])
        end

        # A narrow bump beside a broad mode, with its peak above the mode's.
        rng = Xoshiro(7)
        xb = [rand(rng) < 0.8 ? randn(rng) : 3 + 0.15randn(rng) for _ in 1:1000]
        xbs = sort(xb)

        @testset "the window shape" begin
            grid, logb = PD._window_shape(xbs, 4.0, 30)
            s = WindowScale(1.0, 1.0, grid, logb)
            @test issorted(grid) && allunique(grid)
            @test first(grid) == first(xbs) && last(grid) == last(xbs)
            @test mean(PD._window_shape_at(s, t) for t in xbs) ≈ 0 atol=1e-12
            # Smaller windows (larger ln 1/ĥ) at the narrow bump than at the broad mode.
            @test s(3.0) > 2 * s(0.0)
            @test_throws "at least two distinct observations" PD._window_shape([1.0, 1.0], 4.0, 30)
        end

        @testset "selection" begin
            κ = select_kappa(xb, Adaptive(; window=WindowRule()))
            @test κ isa WindowScale
            @test κ.γ == 0.75
            @test κ(3.0) > κ(0.0)
            # The window candidate's penalized score is below the score of the scale the power
            # family returns, with nse = 0 by default.
            κp = select_kappa(xb, Adaptive(; window=nothing, nse=0))
            sp = κp isa Real ? klcv_const(xbs, κp) : klcv_scale(xbs, κp)
            @test klcv_scale(xbs, κ) + 0.11 * 0.75 / sqrt(1000) < sp
            d = DensityEstimate(xb, κ)
            @test length(d.κ) == length(d.x) - 1
            @test cdf(d, Inf) == 1

            # An infinite penalty leaves the power family's choice, made with nse = 0.
            κinf = select_kappa(xb, Adaptive(; window=WindowRule(k=Inf)))
            @test typeof(κinf) == typeof(κp)
            @test κp isa Real ? κinf == κp : (κinf.c, κinf.α) == (κp.c, κp.α)
            κinf1 = select_kappa(xb, Adaptive(; window=WindowRule(k=Inf), nse=1))
            κp1 = select_kappa(xb, Adaptive(; window=nothing))
            @test typeof(κinf1) == typeof(κp1)

            # The window rule is the default, for point and interval data alike.
            κd = select_kappa(xb, Adaptive())
            @test κd isa WindowScale && (κd.c, κd.x, κd.logb) == (κ.c, κ.x, κ.logb)
            xr = round.(xb; digits=1)
            κrd = select_kappa(xr, Adaptive(); resolution=0.1)
            κrw = select_kappa(xr, Adaptive(; window=WindowRule()); resolution=0.1)
            @test κrd isa WindowScale && (κrd.c, κrd.x, κrd.logb) == (κrw.c, κrw.x, κrw.logb)

            # Other settings are honored.
            @test select_kappa(xb, Adaptive(; window=WindowRule(gamma=0.5, k=0))).γ == 0.5

            # Generic input: offset axes and views select the same scale.
            for xg in (OffsetVector(xb, -400), view([xb; xb], 1:1000))
                κg = select_kappa(xg, Adaptive(; window=WindowRule()))
                @test κg isa WindowScale && κg.c == κ.c && κg.x == κ.x && κg.logb == κ.logb
            end

            # A bounded domain: the shape comes from the sample, the score from the domain.
            xe = -log.(1 .- (0.5:999.5) ./ 1000)
            κe = select_kappa(xe, Adaptive(; window=WindowRule()); support=(0.0, Inf))
            de = DensityEstimate(xe, κe; support=(0.0, Inf))
            @test cdf(de, Inf) ≈ 1
        end

        @testset "input validation" begin
            @test_throws "gamma must be positive" WindowRule(gamma=0)
            @test_throws "gamma must be positive" WindowRule(gamma=Inf)
            @test_throws "k must be nonnegative" WindowRule(k=-0.1)
            @test_throws "threshold z must be positive" WindowRule(z=0)
            @test_throws "nmin must be at least 2" WindowRule(nmin=1)
            @test WindowRule() == WindowRule(0.75, 0.11, 4, 30)
        end

        @testset "rounded and interval data" begin
            # Each interval's observations sit at its quantile midpoints; exact points stay;
            # intervals with an infinite bound are left out.
            @test PD._spread_intervals([0.0, 2.0, 5.0], [1.0, 2.0, Inf], [4.0, 2.0, 3.0]) ==
                  [0.125, 0.375, 0.625, 0.875, 2.0, 2.0]
            κpt = select_kappa(xb, Adaptive(; window=WindowRule()))
            # Rounding far below every window's size leaves the rule as on the exact sample.
            δ = 0.001
            κf = select_kappa(round.(xb ./ δ) .* δ, Adaptive(; window=WindowRule()); resolution=δ)
            @test κf isa WindowScale
            @test κf.c ≈ κpt.c rtol=0.01
            ts = range(-2.5, 3.5; length=301)
            @test mean(abs(log(κf(t) / κf.c) - log(κpt(t) / κpt.c)) for t in ts) < 0.02
            # At a coarser increment the window candidate still resolves the narrow bump.
            δ = 0.1
            xr = round.(xb ./ δ) .* δ
            κr = select_kappa(xr, Adaptive(; window=WindowRule()); resolution=δ)
            @test κr isa WindowScale && κr(3.0) > 2 * κr(0.0)
            # (`resolution` builds the bounds by its own arithmetic; they agree to roundoff.)
            @test select_kappa(xr .- δ / 2, xr .+ δ / 2, Adaptive(; window=WindowRule())).c ≈ κr.c rtol=1e-8
            d = IntervalDensityEstimate(xr .- δ / 2, xr .+ δ / 2, κr)
            @test cdf(d, Inf) ≈ 1
            # An infinite penalty leaves the power family's choice, made with nse = 0.
            κi = select_kappa(xr, Adaptive(; window=WindowRule(k=Inf)); resolution=δ)
            κp = select_kappa(xr, Adaptive(; window=nothing, nse=0); resolution=δ)
            @test typeof(κi) == typeof(κp) && κi.c == κp.c
            # Censored observations enter the score but not the window shape.
            κc = select_kappa([xr .- δ / 2; 5.0; 5.0], [xr .+ δ / 2; Inf; Inf], Adaptive(; window=WindowRule()))
            @test κc isa WindowScale && last(κc.x) < maximum(xr) + δ / 2 < 5
        end
    end

    @testset "goodness of fit: statistic" begin
        d = DensityEstimate([-1.0, 0.0, 0.0, 1.0], 1.0)
        @test chisq(d, d) == 0                 # a distribution vs itself
        @test chisq(d, x -> 0.9 * d(x)) > 0    # a mismatched (here unnormalized) trial
        # matches the defining sum 4 Σ wᵢ (√Q(xᵢ)/ψ_cl(xᵢ) − 1)²
        Q(x) = exp(-x^2 / 2) / sqrt(2π)
        manual = 4 * sum(d.w[i] * (sqrt(Q(d.x[i])) / d.ψ[i] - 1)^2 for i in eachindex(d.x))
        @test chisq(d, Q) ≈ manual
        @test_throws ArgumentError chisq(d, x -> -1.0)
        @test_throws "trial density Q must be nonnegative" chisq(d, x -> -1.0)
    end

    @testset "goodness of fit: large-N (Eq. 26) reference" begin
        d = DensityEstimate([-1.0, 0.0, 0.0, 1.0], 1.0)
        r = ChisqReference(d)
        μ = mean(r)
        @test μ > 0
        # method=:largeN is the Wald inverse-Gaussian(mean μ, shape μ²) at the exact mean
        # μ = tr A: normalized, with mean μ.
        zs = range(1e-5, 40μ; length=4_000_001)
        p = chisq_pdf.(Ref(r), zs; method=:largeN)
        trap(f) = (sum(f) - (f[1] + f[end]) / 2) * step(zs)
        @test trap(p) ≈ 1 atol = 1e-4
        @test trap(zs .* p) ≈ μ rtol = 1e-4
        @test chisq_pdf(r, -1.0; method=:largeN) == 0 && chisq_pdf(r, 0.0; method=:largeN) == 0
        # ccdf is the pdf's upper tail and its negative derivative is the pdf.
        z0 = 1.3μ
        @test chisq_ccdf(r, z0; method=:largeN) ≈ trap(p .* (zs .≥ z0)) atol = 1e-4
        @test chisq_ccdf(r, 0.0; method=:largeN) == 1
        h = 1e-5
        @test -(chisq_ccdf(r, z0 + h; method=:largeN) - chisq_ccdf(r, z0 - h; method=:largeN)) / 2h ≈
              chisq_pdf(r, z0; method=:largeN) rtol = 1e-4
        # pvalue(...; method=:largeN) is the large-N ccdf at the observed statistic.
        Q(x) = exp(-x^2 / 2) / sqrt(2π)
        @test pvalue(r, chisq(d, Q); method=:largeN) == chisq_ccdf(r, chisq(d, Q); method=:largeN)
    end

    @testset "goodness of fit: exact reference distribution" begin
        # A single point: χ² is exactly a scaled χ²₁ (one generalized-χ² weight = the mean).
        d1 = DensityEstimate([0.7], 1.5); r1 = ChisqReference(d1)
        e1 = mean(r1)
        @test chisq_ccdf(r1, e1) ≈ 0.31731050786291404 rtol = 1e-8      # P(Z² ≥ 1)
        @test chisq_ccdf(r1, 4 * e1) ≈ 0.04550026389635842 rtol = 1e-8  # P(Z² ≥ 4)
        @test chisq_ccdf(r1, 0.0) ≈ 1 atol = 1e-6

        # The exact distribution reproduces a direct Monte-Carlo of the fluctuation field.
        Random.seed!(1)
        x = randn(150); d = DensityEstimate(x, select_kappa_sensitivity(x))
        r = ChisqReference(d)
        chis = fieldmc_chisq(d; nsamp=60_000, seed=7)
        @test mean(r) ≈ mean(chis) rtol = 0.02
        for z in quantile(chis, (0.3, 0.6, 0.9, 0.99))
            @test chisq_ccdf(r, z) ≈ mean(>(z), chis) atol = 0.012
        end
        # The Wald shape at the exact mean tracks the exact tail closely.
        for frac in (0.8, 1.2, 1.6)
            z = frac * mean(r)
            @test chisq_ccdf(r, z; method=:largeN) ≈ chisq_ccdf(r, z) rtol = 0.1
        end

        # ccdf ∈ [0,1] and monotone; pdf ≥ 0, integrates to ~1, and is −d(ccdf)/dz.
        μ = mean(r)
        grid = range(0.05μ, 4μ; length=60)
        c = chisq_ccdf.(Ref(r), grid)
        @test all(0 .≤ c .≤ 1) && issorted(c; rev=true)
        @test chisq_ccdf(r, 0.0) ≈ 1 atol = 1e-4    # full mass ⇒ normalized
        @test all(≥(0), chisq_pdf.(Ref(r), grid))
        h = 1e-3
        @test -(chisq_ccdf(r, μ + h) - chisq_ccdf(r, μ - h)) / 2h ≈ chisq_pdf(r, μ) rtol = 1e-4
        # A five-node fit with a small κ, where the real-axis density integrand decays only
        # algebraically: the density matches -d(ccdf)/dz in the bulk and both tails.
        x5 = -log.(rand(Xoshiro(5), 5))
        r5 = ChisqReference(DensityEstimate(x5, 0.2select_kappa(x5); support=(minimum(x5) - 0.5, maximum(x5) + 0.5)))
        μ5 = mean(r5)
        for z in (0.1μ5, μ5, 2μ5, 6μ5)
            h5 = 1e-4z
            fd = -(chisq_ccdf(r5, z + h5; rtol=1e-10, atol=1e-12) - chisq_ccdf(r5, z - h5; rtol=1e-10, atol=1e-12)) / 2h5
            @test chisq_pdf(r5, z) ≈ fd rtol = 1e-4 atol = 1e-8
        end
        # An unconverged inversion integral throws.
        @test_throws PenalizedDensity.ConvergenceError chisq_pdf(r5, 2μ5; rtol=0, atol=1e-300)
        @test_throws "chisq_pdf: the inversion integral at z = " chisq_pdf(r5, 2μ5; rtol=0, atol=1e-300)
        @test_throws "chisq_ccdf: the inversion integral at z = " chisq_ccdf(r5, 2μ5; rtol=0, atol=1e-300)
        @test_throws "chisq_ccdf: the inversion integral at z = 2.0 did not converge (error estimate 0.001, tolerance 1.0e-5)" PenalizedDensity._check_quadrature("chisq_ccdf", 2.0, 0.5, 1e-3, 0.0, 1e-5)
        @test PenalizedDensity._check_quadrature("chisq_ccdf", 2.0, 0.5, 1e-4, 0.0, 1e-5) === nothing
        # At the median the tail integral is ~0; evaluation terminates.
        a, b = 0.5μ, 2μ
        for _ in 1:60
            m = (a + b) / 2
            chisq_ccdf(r, m) > 0.5 ? (a = m) : (b = m)
        end
        @test chisq_ccdf(r, (a + b) / 2) ≈ 0.5 atol = 1e-6

        # pvalue is StatsAPI's function, and :exact is the default method.
        Q(x) = exp(-x^2 / 2) / sqrt(2π)
        @test pvalue === PenalizedDensity.StatsAPI.pvalue
        @test pvalue(r, chisq(d, Q)) == chisq_ccdf(r, chisq(d, Q); method=:exact)
        @test chisq_ccdf(r, μ) == chisq_ccdf(r, μ; method=:exact)
        # pvalue passes its quadrature tolerances through to chisq_ccdf.
        z = chisq(d, Q)
        @test pvalue(r, z; rtol=1e-3, atol=1e-3) == chisq_ccdf(r, z; rtol=1e-3, atol=1e-3)
        # The reference is built once with ChisqReference; nothing takes a fit in its place.
        @test !hasmethod(pvalue, Tuple{DensityEstimate,Any})
        @test !hasmethod(chisq_ccdf, Tuple{DensityEstimate,Real})
        @test !hasmethod(chisq_pdf, Tuple{DensityEstimate,Real})

        # The inner constructor derives tg and the mean from (tri, g), and converts eltype.
        @test mean(r) == PenalizedDensity._tridiag_invdiag(r.tri) - sum(abs2, r.g)
        rr = ChisqReference{Float64}(r.tri, r.g)
        @test rr.tg == r.tg && mean(rr) == mean(r)
        r32 = ChisqReference{Float32}(d)
        @test r32 isa ChisqReference{Float32}
        @test mean(r32) ≈ μ rtol = 1e-5
        @test_throws "but g has length" ChisqReference{Float64}(r.tri, r.g[1:end-1])

        # Method validation.
        @test_throws ArgumentError chisq_ccdf(r, 1.0; method=:bogus)
        @test_throws ArgumentError chisq_pdf(r, 1.0; method=:bogus)
        @test_throws ArgumentError pvalue(r, chisq(d, Q); method=:bogus)

        @test sprint(show, r) == "ChisqReference($(length(r.g)) nodes, ⟨χ²⟩=$(r.mean))"

        # The Imhof integrand sweep is allocation-free given its scratch buffers.
        @test r.tg ≈ r.tri * r.g
        piv, rhs = PenalizedDensity._logΦ_scratch(r)
        @test sweep_allocated(piv, rhs, r, 0.7) == 0
        @test sweep_allocated(piv, rhs, r, 0.7 * cis(0.3)) == 0

        # Generic indexing: OffsetArray input gives an identical reference.
        ro = ChisqReference(DensityEstimate(OffsetArray(x, -75), d.κ))
        @test ro.tri.dv ≈ r.tri.dv && ro.g ≈ r.g && mean(ro) ≈ μ

        # Green's identity, the reference's internal consistency check.
        @test green_identity(d) ≈ 1 rtol = 1e-5
    end

    @testset "goodness of fit: exact reference at a varying scale" begin
        # A constant callable takes the piecewise path and lands on the scalar path's answer.
        xg = randn(Xoshiro(11), 500)
        κ = select_kappa(xg)
        rs = ChisqReference(DensityEstimate(xg, κ))
        rc = ChisqReference(DensityEstimate(xg, t -> κ))
        @test rc.tri.dv ≈ rs.tri.dv rtol = 1e-10
        @test rc.g ≈ rs.g rtol = 1e-10
        @test mean(rc) ≈ mean(rs) rtol = 1e-10
        @test chisq_ccdf(rc, mean(rc)) ≈ chisq_ccdf(rs, mean(rs)) rtol = 1e-10

        # The exact law reproduces a direct Monte-Carlo of the fluctuation field for a hard-edge
        # family under the plug-in selector. Its realized rates span five orders of magnitude,
        # which the reference handles by building every quantity in scaled form.
        xa = sort!(randn(Xoshiro(3), 300).^2)                 # χ²₁: divergent edge at 0
        da = DensityEstimate(xa, select_kappa(xa, Adaptive(; window=nothing)))
        @test maximum(da.κ) / minimum(da.κ) > 1e4
        ra = ChisqReference(da)
        chis = fieldmc_chisq(da; nsamp=40_000, seed=3)
        @test mean(ra) ≈ mean(chis) rtol = 0.02
        for z in quantile(chis, (0.3, 0.6, 0.9, 0.99))
            @test chisq_ccdf(ra, z) ≈ mean(>(z), chis) atol = 0.012
        end
        @test green_identity(da) ≈ 1 rtol = 1e-5

        # The default selector's window scale on a narrow bump beside a broad mode.
        rng = Xoshiro(7)
        xw = [rand(rng) < 0.8 ? randn(rng) : 3 + 0.15randn(rng) for _ in 1:400]
        dw = DensityEstimate(xw, select_kappa(xw, Adaptive()))
        @test dw.κ isa AbstractVector
        rw = ChisqReference(dw)
        chw = fieldmc_chisq(dw; nsamp=40_000, seed=3)
        @test mean(rw) ≈ mean(chw) rtol = 0.02
        for z in quantile(chw, (0.3, 0.9, 0.99))
            @test chisq_ccdf(rw, z) ≈ mean(>(z), chw) atol = 0.012
        end

        # And a smooth family at small N, under a hand-chosen scale.
        xs = sort!(randn(Xoshiro(2), 60))
        ds = DensityEstimate(xs, t -> 2exp(-t / 2))
        rsm = ChisqReference(ds)
        chs = fieldmc_chisq(ds; nsamp=40_000, seed=3)
        @test mean(rsm) ≈ mean(chs) rtol = 0.02
        for z in quantile(chs, (0.3, 0.9, 0.99))
            @test chisq_ccdf(rsm, z) ≈ mean(>(z), chs) atol = 0.012
        end
        @test green_identity(ds) ≈ 1 rtol = 1e-5

        # ∬ψ_cl G₀ ψ_cl and the nodal α agree with a conservative finite-difference solve of
        # 𝒜α = ψ_cl/(2λ) on a node-aligned grid, which uses no Green's function. It converges
        # to the closed forms at the discretization's O(δ²).
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
            ψy = amplitude.(ds, y)
            αfd = SymTridiagonal(p, q) \ (mass .* ψy ./ 2λ)
            mfd = [αfd[searchsortedfirst(y, xi)] for xi in x]
            mex = PenalizedDensity._node_alpha(x, ψ, ds.κ, ds.κL, ds.κR, λ, ds.lo, ds.hi)
            (maximum(abs.(mex .- mfd) ./ abs.(mfd)),
             abs(PenalizedDensity._int_psi_alpha(x, ψ, mex, ds.κ, ds.κL, ds.κR, λ, ds.lo, ds.hi) -
                 sum(mass .* ψy .* αfd)) / sum(mass .* ψy .* αfd))
        end
        @test all(<(2e-4), errs[1]) && all(<(2e-5), errs[2])   # and 16× finer ⇒ 16× closer
        @test all(first(errs) ./ last(errs) .> 10)
    end

    @testset "action asymptotes" begin
        # g(κ) = S(κ) + W ln κ tends to W/2 (κ→0) and W/2 + W·H (κ→∞). Widely separated points
        # reach isolation at a moderate κ (avoiding sinh overflow), where H = ln 3 exactly.
        xs = [0.0, 5.0, 10.0]; W3 = 3; H3 = log(3)
        g(κ) = action(DensityEstimate(xs, κ)) + W3 * log(κ)
        @test g(1e-5) ≈ W3 / 2 rtol = 1e-3            # one lump
        @test g(3.0) ≈ W3 / 2 + W3 * H3 rtol = 1e-4   # three isolated points
    end

    @testset "entropy and negentropy" begin
        # Single-point fit is an exact Laplace(rate 2κ) density, variance σ² = 1/(2κ²): the
        # plug-in Ĥ = -ln κ (biased relative to the exact H = 1 - ln κ at this extreme
        # small-sample size), and negentropy is the gap to the matched Gaussian's entropy.
        κ = 1.5
        d = DensityEstimate([2.0], κ)
        @test entropy(d) ≈ -log(κ)
        @test negentropy(d) ≈ log(2π * ℯ / (2κ^2)) / 2 - entropy(d)

        # Reference: quadrature entropy and negentropy of the fitted density rather than the
        # plug-in estimator, which separates the closed-form mean/variance and -∫Q ln Q dx
        # from small-sample plug-in bias.
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
            κ = select_kappa_sensitivity(x)
            d = DensityEstimate(x, κ)
            @test negentropy(d) > 0
            @test negentropy(d) ≈ quad_negentropy(d, var(x)) rtol = 0.1
        end

        # Gaussian samples: negentropy fluctuates near 0 and shrinks with N.
        negs = map((200, 2_000, 20_000)) do n
            Random.seed!(1)
            xn = randn(n)
            dn = DensityEstimate(xn, select_kappa_sensitivity(xn))
            abs(negentropy(dn))
        end
        @test issorted(negs; rev=true)

        # Affine invariance: x ↦ a·x + b with κ ↦ κ/|a| maps the fit onto itself, and
        # negentropy is invariant under it, for a > 0 and a < 0.
        Random.seed!(9)
        x = 1.7 .* randn(2_000) .+ 0.3
        κ0 = select_kappa_sensitivity(x)
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

    @testset "fit moments, sampling, and support" begin
        # Reference moments by quadrature, split at every node and wall so each piece is smooth.
        function quad_moments(d)
            pts = [d.lo; d.x; d.hi]
            seg(f) = sum(quadgk(f, pts[i], pts[i+1]; rtol=1e-13)[1] for i in 1:length(pts)-1)
            Z = seg(d)
            m = seg(t -> t * d(t)) / Z
            return Z, m, seg(t -> (t - m)^2 * d(t)) / Z
        end
        rng = Xoshiro(7)
        x = randn(rng, 40) .+ 3
        a, b = minimum(x), maximum(x)
        κfun = t -> 1 + 0.5 * abs(t - 3)
        fits = (
            unbounded       = DensityEstimate(x, 2.0),
            left_wall       = DensityEstimate(x, 2.0; support=(a - 0.05, Inf)),
            both_walls      = DensityEstimate(x, 2.0; support=(a - 0.05, b + 0.3)),
            touching_wall   = DensityEstimate(x, 2.0; support=(a, b)),
            varying         = DensityEstimate(x, κfun),
            varying_bounded = DensityEstimate(x, κfun; support=(a - 0.01, 10.0)),
        )
        for (name, d) in pairs(fits)
            Z, m, v = quad_moments(d)
            @test Z ≈ 1 rtol = 1e-10
            @test mean(d) ≈ m atol = 1e-10
            @test var(d) ≈ v rtol = 1e-10
            @test std(d) == sqrt(var(d))
            @test median(d) == quantile(d, 0.5)
            @test cdf(d, median(d)) ≈ 0.5 rtol = 1e-12
            @test negentropy(d) ≈ log(2π * ℯ * v) / 2 - entropy(d) rtol = 1e-10
            @test extrema(d) === (minimum(d), maximum(d)) === (d.lo, d.hi)
        end
        @test extrema(fits.unbounded) === (-Inf, Inf)
        # A wide gap makes the Neumann arc indistinguishable from the exponential tail.
        far = DensityEstimate(x, 2.0; support=(a - 40, b + 40))
        @test var(far) ≈ var(fits.unbounded) rtol = 1e-12
        # Translation far from the origin keeps the variance well-conditioned.
        shifted = DensityEstimate(x .+ 1e8, κfun ∘ (t -> t - 1e8); support=(a - 0.01 + 1e8, 10.0 + 1e8))
        @test var(shifted) ≈ var(fits.varying_bounded) rtol = 1e-6
        d32 = DensityEstimate(Float32.(x), 2.0f0; support=(Float32(a) - 0.05f0, Float32(b) + 0.3f0))
        @test @inferred(mean(d32)) isa Float32
        @test @inferred(var(d32)) ≈ var(fits.both_walls) rtol = 1e-4
        @test @inferred(median(d32)) isa Float32

        # Sampling by inversion: reproducible, inside the support, and distributed as the fit
        # (the probability-integral transform of the draws is uniform).
        for d in (fits.unbounded, fits.both_walls, fits.varying)
            ys = rand(Xoshiro(3), d, 20_000)
            @test ys == rand(Xoshiro(3), d, 20_000)
            @test all(y -> d.lo <= y <= d.hi, ys)
            u = sort!(cdf.(d, ys))
            @test maximum(abs.(u .- (1:length(u)) ./ length(u))) < 0.015   # KS distance
            @test mean(ys) ≈ mean(d) atol = 4 * std(d) / sqrt(length(ys))
        end
        @test rand(Xoshiro(3), fits.both_walls) isa Float64
        @test eltype(rand(d32, 3)) === Float32
        @test rand!(Xoshiro(1), OffsetArray(zeros(5), -2), fits.unbounded) isa OffsetArray
        di = IntervalDensityEstimate(round.(x; digits=1), 2.0; resolution=0.1)
        ys = rand(Xoshiro(5), di, 20_000)
        @test eltype(ys) === Float64 && all(isfinite, ys)
        u = sort!(cdf.(di, ys))
        @test maximum(abs.(u .- (1:length(u)) ./ length(u))) < 0.015   # KS distance
        @test extrema(di) === (-Inf, Inf)

        # Held-out entropy is computed in the log domain: finite far into an unbounded tail
        # where the density underflows, infinite only outside a finite support.
        d = fits.unbounded
        @test d(-800.0) == 0 && isfinite(entropy(d, [-800.0, 3.0]))
        @test entropy(fits.both_walls, [a - 1, 3.0]) == Inf
    end

    @testset "held-out entropy and negentropy" begin
        Random.seed!(17)
        x = 1.2 .* randn(400) .+ 0.5
        κ = select_kappa(x)
        d = DensityEstimate(x, κ)
        ye = 1.2 .* randn(150) .+ 0.5          # independent evaluation batch

        # The Gaussian reference needs a positive variance.
        msg = "negentropy(d, xeval) needs at least two distinct evaluation points"
        @test_throws ArgumentError negentropy(d, [0.3])
        @test_throws msg negentropy(d, [0.3])
        @test_throws msg negentropy(d, fill(0.3, 4))

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
        # nodes (where ψ', and hence the log-density slope, has a kink), across κ regimes.
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
            # Far enough out ψ underflows to zero, and the slope is still exactly ±2κ.
            @test logdensity_eval_gradient(d, nodes[1] - 1000.0) == 2κ
            @test logdensity_eval_gradient(d, nodes[end] + 1000.0) == -2κ
        end
        # ψ underflows in the middle of a wide interval; with equal node amplitudes the slope
        # at signed distance t from the midpoint is 2κ tanh(κt).
        d = DensityEstimate([0.0, 2000.0], 1.0)
        @test iszero(d(999.0))
        @test logdensity_eval_gradient(d, 1000.0) == 0
        @test logdensity_eval_gradient(d, 999.0) ≈ -2 * tanh(1.0)
        @test logdensity_eval_gradient(d, 1001.0) ≈ 2 * tanh(1.0)
        # At a finite wall the Neumann condition makes the slope zero; beyond it ln Q̂ ≡ -Inf.
        d = DensityEstimate([0.2, 0.5, 0.6], 2.0; support=(0, 1))
        @test logdensity_eval_gradient(d, 0.0) == 0
        @test logdensity_eval_gradient(d, 1.0) == 0
        @test logdensity_eval_gradient(d, 0.1) ≈ 2 * 2 * tanh(2 * 0.1)
        @test isnan(logdensity_eval_gradient(d, 1.5))
    end

    @testset "node-position log-density gradient" begin
        # gᵢ = ∂/∂xᵢ Σⱼ cⱼ ln Q̂(yⱼ), by the implicit-function adjoint, against central
        # finite differences that refit at perturbed node positions. Fits are built through
        # _fit on distinct nodes (no merging) so the perturbation stays within one cell.
        function L_of_nodes(xn, w, κ, ye, c)
            dd = fit_unbounded(collect(float.(xn)), collect(float.(w)), float(κ))
            sum(cj * log(dd(y)) for (y, cj) in zip(ye, c))
        end
        Random.seed!(23)
        for κ in (0.3, 1.5, 8.0, 40.0)
            xn = sort(randn(7)) .* 1.3
            w = ones(7)
            d = fit_unbounded(copy(xn), copy(w), Float64(κ))
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
        d = fit_unbounded(copy(xn), copy(w), κ)
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
        d = fit_unbounded(copy(xn), copy(w), κ)
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
        d = fit_unbounded(sort(randn(6)), ones(6), 1.0)
        ye = [-0.5, 0.2, 0.8]
        @test logdensity_node_gradient(d, ye) ≈ logdensity_node_gradient(d, ye, ones(3))

        # The two halves compose into ∂entropy(d, ye)/∂(node positions): since the Gaussian
        # reference of negentropy(d, ye) depends only on ye, the node gradient of entropy is
        # -logdensity_node_gradient with uniform 1/M weights.
        Random.seed!(11)
        xn = sort(randn(6)); w = ones(6); κ = 1.7
        d = fit_unbounded(copy(xn), copy(w), κ)
        ye = Float64[-1.4, -0.3, 0.25, 0.9, 2.1]; M = length(ye)
        an = .-logdensity_node_gradient(d, ye, fill(1 / M, M))
        h = 1e-6
        fd = map(eachindex(xn)) do i
            xp = copy(xn); xp[i] += h
            xm = copy(xn); xm[i] -= h
            (entropy(fit_unbounded(xp, copy(w), κ), ye) -
             entropy(fit_unbounded(xm, copy(w), κ), ye)) / (2h)
        end
        @test an ≈ fd rtol = 1e-4

        # One adjoint solve regardless of batch size: allocations depend on the node count N,
        # not the evaluation count M.
        xn = sort(randn(500)); w = ones(500); d = fit_unbounded(copy(xn), copy(w), 2.0)
        span(nb) = collect(range(minimum(xn) - 1, maximum(xn) + 1; length = nb))
        ye_small, ye_big = span(200), span(4000)
        c_small, c_big = fill(1 / 200, 200), fill(1 / 4000, 4000)
        logdensity_node_gradient(d, ye_small, c_small)    # compile
        logdensity_node_gradient(d, ye_big, c_big)
        a_small = @allocated logdensity_node_gradient(d, ye_small, c_small)
        a_big = @allocated logdensity_node_gradient(d, ye_big, c_big)
        @test a_small == a_big

        # Input checks.
        d = fit_unbounded(sort(randn(6)), ones(6), 1.0)
        ye = [-0.5, 0.2, 0.8]
        @test_throws DimensionMismatch logdensity_node_gradient(d, ye, ones(2))
        @test_throws "weights must have the axes of xeval" logdensity_node_gradient(d, ye, ones(4))
        @test_throws "weights must have the axes of xeval" logdensity_node_gradient(d, ye, OffsetArray(ones(3), -1))
        @test_throws DomainError logdensity_node_gradient(d, [-0.5, NaN, 0.8])
        @test_throws "the evaluation point must not be NaN" logdensity_node_gradient(d, [-0.5, NaN, 0.8], ones(3))
        xb = [-1.3, -0.2, 0.4, 0.9, 2.1]
        @test_throws "supports only unbounded fits" logdensity_node_gradient(DensityEstimate(xb, 1.0; support=(-2.0, 3.0)), ye)
        @test_throws "supports only a single-rate (scalar κ) fit" logdensity_node_gradient(DensityEstimate(xb, t -> 1 + t^2 / 4), ye)

        # Offset axes: matching OffsetArray points and weights give the same gradient, indexed
        # like d.x.
        c = [0.7, 1.1, 0.9]
        g = logdensity_node_gradient(d, ye, c)
        @test logdensity_node_gradient(d, OffsetArray(ye, -2), OffsetArray(c, -2)) == g
        @test logdensity_node_gradient(d, OffsetArray(ye, 5)) == logdensity_node_gradient(d, ye)
    end

    @testset "large κ stays finite (no sinh overflow)" begin
        Random.seed!(3)
        x = randn(300)
        d = DensityEstimate(x, 5000.0)   # kernels far narrower than spacings
        @test all(isfinite, d.ψ) && all(>(0), d.ψ)
        @test isfinite(d.λ) && d.λ > 0
        @test isfinite(action(d))
        @test all(isfinite, amplitude.(d, range(-4, 4; length = 200)))  # incl. inter-point gaps
    end

    @testset "log-density stays finite where the density underflows" begin
        Random.seed!(11)
        x = randn(200)
        κ = 1.5
        d = DensityEstimate(x, κ)
        # Far enough out that ψ is zero in Float64.
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

        # Where the density has not underflowed the two routes agree.
        near = sort(d.x[1] .- [0.5, 2.0, 5.0])
        @test PenalizedDensity._logdensity_sorted(d, near) ≈ 2 .* log.(amplitude.(d, near))
    end

    @testset "log-density stays finite inside a wide gap" begin
        # κ·gap ≈ 3000: both sinh arcs underflow in the middle of the interval, so the
        # amplitude is zero over a region where the log-density is finite.
        xs = [-1000.0, -999.0, 0.0, 999.0, 1000.0]
        κ = 3.0
        d = DensityEstimate(xs, κ)
        mid = [-750.0, -500.0, -250.0, 250.0, 500.0, 750.0]
        @test all(t -> amplitude(d, t) == 0, mid)
        @test all(isfinite, logdensity.(d, mid))

        # Deep inside the gap each arc is a pure exponential decay from its own node,
        # ψ ≈ ψ_k e^{-κ(x - x_k)} + ψ_{k+1} e^{-κ(x_{k+1} - x)}, which gives the value
        # independently of the sinh recurrence.
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
        @test logdensity.(d, ts) ≈ 2 .* log.(amplitude.(d, ts))

        # The scalar path and the sorted-batch sweep agree exactly.
        far = sort(vcat(ts, d.x[1] .- [1e4, 1500.0], d.x[end] .+ [1500.0, 1e4]))
        @test logdensity.(d, far) == PenalizedDensity._logdensity_sorted(d, far)

        # Outside a finite support the density is exactly zero, so ln Q = -Inf.
        db = DensityEstimate(clamp.(randn(150), -1.9, 1.9), 1.2; support = (-2.0, 2.0))
        @test logdensity(db, -2.5) == -Inf
        @test logdensity(db, 2.5) == -Inf
        @test isfinite(logdensity(db, 0.0))
        # The bounded tails run through `logcosh` and stay finite to the boundary.
        @test all(isfinite, logdensity.(db, range(-2, 2; length = 200)))
    end

    @testset "input validation" begin
        @test_throws ArgumentError DensityEstimate(Float64[], 1.0)
        @test_throws "cannot fit a density to zero points" DensityEstimate(Float64[], 1.0)
        @test_throws ArgumentError DensityEstimate([1.0], 0.0)
        @test_throws "κ must be finite and positive, got 0.0" DensityEstimate([1.0], 0.0)
        @test_throws ArgumentError DensityEstimate([1.0], -1.0)
        @test_throws "κ must be finite and positive, got -1.0" DensityEstimate([1.0], -1.0)
        @test_throws "κ must be finite and positive, got Inf" DensityEstimate([1.0, 2.0], Inf)
        @test_throws "κ must be finite and positive, got NaN" DensityEstimate([1.0, 2.0], NaN)
        # κ is checked in the fit's float type, after conversion.
        @test_throws ArgumentError DensityEstimate{Float32}([1.0, 2.0, 4.0], 1e-50)
        @test_throws "κ = 1.0e-50 is not a finite positive Float32 (it converts to 0.0)" DensityEstimate{Float32}([1.0, 2.0, 4.0], 1e-50)
        @test_throws "κ = 1.0e50 is not a finite positive Float32 (it converts to Inf)" DensityEstimate{Float32}([1.0, 2.0, 4.0], 1e50)
        @test_throws ArgumentError DensityEstimate([1.0], 1.0; rtol=-1.0)
        @test_throws "rtol must be nonnegative" DensityEstimate([1.0], 1.0; rtol=-1.0)
        @test_throws ArgumentError select_kappa_sensitivity([1.0, 2.0]; kappas=[1.0, -1.0, 2.0])
        @test_throws "kappas must be sorted and positive" select_kappa_sensitivity([1.0, 2.0]; kappas=[1.0, -1.0, 2.0])
        @test_throws ArgumentError select_kappa_sensitivity([1.0, 2.0]; kappas=[1.0, 2.0])  # need ≥ 3
        @test_throws "need at least 3 values in kappas to bracket the minimum" select_kappa_sensitivity([1.0, 2.0]; kappas=[1.0, 2.0])

        @test_throws ArgumentError select_kappa_lscv([1.0, 2.0]; kappas=[1.0, -1.0, 2.0])
        @test_throws "kappas must be sorted and positive" select_kappa_lscv([1.0, 2.0]; kappas=[1.0, -1.0, 2.0])
        @test_throws ArgumentError select_kappa_lscv([1.0, 2.0]; kappas=[1.0, 2.0])   # need ≥ 3
        @test_throws "need at least 3 values in kappas to bracket the minimum" select_kappa_lscv([1.0, 2.0]; kappas=[1.0, 2.0])
        @test_throws ArgumentError select_kappa_lscv([3.0])                        # need ≥ 2 distinct
        @test_throws "need at least two distinct points to select κ" select_kappa_lscv([3.0])

        @test_throws ArgumentError select_kappa([1.0, 2.0]; kappas=[1.0, -1.0, 2.0])
        @test_throws "kappas must be sorted and positive" select_kappa([1.0, 2.0]; kappas=[1.0, -1.0, 2.0])
        @test_throws ArgumentError select_kappa([1.0, 2.0]; kappas=[1.0, 2.0])   # need ≥ 3
        @test_throws "need at least 3 values in kappas to bracket the minimum" select_kappa([1.0, 2.0]; kappas=[1.0, 2.0])
        @test_throws ArgumentError select_kappa([3.0])                        # need ≥ 2 distinct
        @test_throws "need at least two distinct points to select κ" select_kappa([3.0])
    end

    @testset "select_kappa on Float32 data" begin
        x = randn(Xoshiro(3), 1000)
        κ32 = select_kappa(Float32.(x))
        @test κ32 isa Float32
        # The default rtol, cbrt(eps(T)), is looser for Float32; at a common rtol the two
        # types agree. The score is flat near its minimum, so Float32 rounding in the score
        # moves the minimizer by about 0.6% (one standard deviation over samples), and by
        # about 2.6% at the tighter Float64 rtol.
        @test κ32 ≈ select_kappa(x; rtol=cbrt(eps(Float32))) rtol = 0.03
        @test select_kappa(Float32.(x); rtol=Float32(cbrt(eps()))) ≈ select_kappa(x) rtol = 0.1
    end

    @testset "error model: NaN throws, non-convergence throws ConvergenceError" begin
        # NaN observations, in every form that takes data.
        @test_throws ArgumentError DensityEstimate([0.0, NaN, 1.0], 1.0)
        @test_throws "observations must be finite, got NaN" DensityEstimate([0.0, NaN, 1.0], 1.0)
        @test_throws "observations must be finite, got NaN" select_kappa([0.0, NaN, 1.0, 2.0])
        @test_throws "observations must be finite, got NaN" IntervalDensityEstimate([0.0, NaN, 1.0], 1.0; resolution=1.0)

        # ±Inf point observations, in every form that takes points; ±Inf interval bounds are
        # censoring and stay legal.
        xinf = [-1.3, -0.2, 0.4, 0.9, 2.1, Inf]
        msg = "observations must be finite, got Inf"
        @test_throws ArgumentError DensityEstimate(xinf, 1.0)
        @test_throws msg DensityEstimate(xinf, 1.0)
        @test_throws "observations must be finite, got -Inf" DensityEstimate(-xinf, 1.0)
        @test_throws msg DensityEstimate(xinf, 1.0; support=(-2.0, Inf))
        @test_throws msg DensityEstimate(xinf, t -> 1 + t^2 / 4)
        @test_throws msg DensityEstimate{Float32}(xinf, 1.0)
        @test_throws "observation 1.0e300 overflows Float32" DensityEstimate{Float32}([0.0, 1.0, 1e300], 1.0)
        @test_throws msg PenalizedDensity.merge_samples(xinf, 1.0)
        @test_throws msg PenalizedDensity.merge_samples(xinf, t -> 1.0)
        @test_throws msg select_kappa(xinf)
        @test_throws msg select_kappa(xinf; kappas=[0.5, 1.0, 2.0])
        @test_throws msg select_kappa(xinf, Adaptive())
        @test_throws msg select_kappa([0.0, 1.0, 2.0, Inf], Adaptive(); resolution=1.0)
        @test_throws msg PenalizedDensity.select_kappa_lscv(xinf)
        @test_throws msg PenalizedDensity.select_kappa_sensitivity(xinf)
        @test_throws msg PenalizedDensity.klcv(xinf, 1.0)
        @test_throws msg PenalizedDensity.lscv(xinf, 1.0)
        @test_throws msg select_support(xinf)
        @test_throws msg IntervalDensityEstimate([0.0, 1.0, Inf], 1.0; resolution=1.0)
        @test_throws msg select_kappa([0.0, 1.0, 2.0, Inf]; resolution=1.0)
        @test_throws msg PenalizedDensity.klcv([0.0, 1.0, 2.0, Inf], 1.0; resolution=1.0)
        @test_throws "point observation" IntervalDensityEstimate([0.0, Inf], [1.0, Inf], 1.0)
        @test IntervalDensityEstimate([-Inf, 0.0, 1.0], [0.0, 1.0, Inf], 1.0) isa IntervalDensityEstimate

        # NaN evaluation points, for every evaluator and every kind of fit; ±Inf keep their
        # limits.
        x = [-1.3, -0.2, 0.4, 0.9, 2.1]
        fits = (DensityEstimate(x, 1.0),
                DensityEstimate(x, 1.0; support=(-2.0, 3.0)),
                DensityEstimate(x, t -> 1 + t^2 / 4),
                IntervalDensityEstimate([-1.0, 0.0, 1.5], [-0.5, 0.5, 2.0], 1.0))
        for d in fits
            @test_throws DomainError d(NaN)
            @test_throws "the evaluation point must not be NaN" d(NaN)
            @test_throws "the evaluation point must not be NaN" amplitude(d, NaN)
            @test_throws "the evaluation point must not be NaN" logdensity(d, NaN)
            @test_throws "the evaluation point must not be NaN" cdf(d, NaN)
            @test d(-Inf) == 0 && logdensity(d, -Inf) == -Inf
            @test cdf(d, -Inf) == 0 && cdf(d, Inf) == 1
        end
        for d in fits[1:3]
            @test_throws "the evaluation point must not be NaN" gaussianize(d, NaN)
            @test_throws "the evaluation point must not be NaN" ungaussianize(d, NaN)
            @test_throws "the evaluation point must not be NaN" gaussianize_logjacobian(d, NaN)
            @test gaussianize(d, -Inf) == -Inf && gaussianize(d, Inf) == Inf
        end
        @test_throws "the evaluation point must not be NaN" logdensity_eval_gradient(fits[1], NaN)
        # Past a finite wall the compact-support conventions apply, without throwing.
        db = fits[2]
        @test db(-2.5) == 0 && logdensity(db, -2.5) == -Inf
        @test gaussianize(db, -2.5) == -Inf && gaussianize(db, 3.5) == Inf
        @test gaussianize_logjacobian(db, 3.5).logjac == -Inf

        # The χ² reference: both methods treat z identically outside its support.
        r = ChisqReference(fits[1])
        for method in (:exact, :largeN)
            @test_throws DomainError chisq_ccdf(r, NaN; method)
            @test_throws "the χ² value must not be NaN" chisq_ccdf(r, NaN; method)
            @test_throws "the χ² value must not be NaN" chisq_pdf(r, NaN; method)
            @test chisq_ccdf(r, -1.0; method) == 1 && chisq_ccdf(r, 0.0; method) == 1
            @test chisq_ccdf(r, Inf; method) == 0
            @test chisq_pdf(r, -1.0; method) == 0 && chisq_pdf(r, 0.0; method) == 0
            @test chisq_pdf(r, Inf; method) == 0
        end
        @test_throws "method must be :exact or :largeN" chisq_ccdf(r, NaN; method=:bogus)

        # Non-convergence: a Newton solve cut off before it converges.
        nodes, w = [-1.0, 0.0, 0.5, 2.0], [1.0, 2.0, 1.0, 1.0]
        M = PenalizedDensity.roughness_operator(nodes, 3.0)
        @test_throws PenalizedDensity.ConvergenceError PenalizedDensity._solve_amplitude(M, w; maxiter=1)
        err = try
            PenalizedDensity._solve_amplitude(M, w; maxiter=1)
        catch e
            e
        end
        @test sprint(showerror, err) ==
              "ConvergenceError in the amplitude solve: Newton did not converge in 1 iterations"
    end

    @testset "roughness_operator promotes its arguments" begin
        ref = PenalizedDensity.roughness_operator([0.0, 1.0, 2.0], 1.0)
        @test PenalizedDensity.roughness_operator(Float32[0, 1, 2], 1.0) == ref
        @test PenalizedDensity.roughness_operator([0, 1, 2], 1) == ref
        @test PenalizedDensity.roughness_operator(0:2, 1.0) == ref
        # Infinite walls do not widen the type; a finite wall takes part in the promotion.
        M32 = PenalizedDensity.roughness_operator(Float32[0, 1, 2], 1.0f0)
        @test eltype(M32) == Float32 && M32 ≈ ref
        @test eltype(PenalizedDensity.roughness_operator(Float32[0, 1, 2], 1.0f0, -1.0, Inf)) == Float64
        @test PenalizedDensity.roughness_operator([0, 1, 2], 1, -1, 3) ==
              PenalizedDensity.roughness_operator([0.0, 1.0, 2.0], 1.0, -1.0, 3.0)
        @test_throws ArgumentError PenalizedDensity.roughness_operator(OffsetArray([0.0, 1.0], -1), 1.0)
    end

    @testset "interoperability: DensityInterface, Distributions, StatsBase" begin
        x = [-1.3, -0.2, 0.4, 0.9, 2.1]
        fits = (DensityEstimate(x, 1.0),
                DensityEstimate(x, 1.0; support=(-2.0, 3.0)),
                DensityEstimate(x, t -> 1 + t^2 / 4),
                IntervalDensityEstimate([-1.0, 0.0, 1.5], [-0.5, 0.5, 2.0], 1.0))
        for d in fits
            @test DensityKind(d) === HasDensity()
            for t in (-1.0, 0.3, 2.5)
                DensityInterface.test_density_interface(d, t, logdensity(d, t))
                @test DensityInterface.densityof(d, t) == d(t)
                @test Distributions.cdf(d, t) == cdf(d, t)
                @test Distributions.pdf(d, t) == d(t)
                @test Distributions.logpdf(d, t) == logdensity(d, t)
            end
            @test Distributions.quantile(d, 0.3) == quantile(d, 0.3)
        end
        for d in fits
            @test StatsBase.entropy(d) == entropy(d)
        end
        # StatsBase's two-argument entropy takes a logarithm base, not evaluation points.
        @test !hasmethod(StatsBase.entropy, Tuple{DensityEstimate,AbstractVector})

        # With both packages loaded, the unqualified name is ambiguous; qualified names work.
        m = Module()
        Core.eval(m, :(using Distributions, PenalizedDensity))
        Core.eval(m, :(d = DensityEstimate([0.0, 1.0], 1.0)))
        @test Core.eval(m, :(Distributions.cdf(d, 0.5))) == Core.eval(m, :(PenalizedDensity.cdf(d, 0.5)))
        @test_throws UndefVarError Core.eval(m, :(cdf(d, 0.5)))
    end
end

@testset "bounded support (natural/Neumann boundary)" begin
    # Independent reference fit, assembled with Base's `tanh`/`sech` and `coth`/`csch` rather
    # than the package's e^{-2θ}-based stable forms, sharing no arithmetic with the
    # bounded-support code under test.
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

    @testset "an infinite support reproduces the unbounded fit" begin
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
        # With a finite support, Z = ψᵀGψ and Z agrees with `_norm_sq`.
        Random.seed!(47)
        x = sort(rand(30) .* 0.6 .+ 0.2)
        d = DensityEstimate(x, 7.0; support=(0.0, 1.0))
        Zgram, Gψ = PenalizedDensity._norm_sq_gram(d.x, d.ψ, d.κ, d.κL, d.κR, d.lo, d.hi)
        Z = PenalizedDensity._norm_sq(d.x, d.ψ, d.κ, d.κL, d.κR, d.lo, d.hi)
        @test Zgram ≈ Z rtol = 1e-13
        @test sum(d.ψ .* Gψ) ≈ Z rtol = 1e-13
    end

    @testset "interval masses are accurate for near-coincident nodes" begin
        # Pairs 1e-6 apart give θ = κh near 1e-8, where the coth/csch coefficient forms lose
        # about half the digits. Compare with the same computation in BigFloat.
        xs = sort(randn(Xoshiro(5), 40))
        xs = sort(vcat(xs, xs[1:2:end] .+ 1e-6 .* rand(Xoshiro(6), 20)))
        d = DensityEstimate(xs, 5.0; rtol=0)
        @test minimum(diff(d.x)) * d.κ < 1e-6
        x, ψ, κ = d.x, d.ψ, d.κ
        X, Ψ, K = big.(x), big.(ψ), big(κ)
        Zb = PenalizedDensity._norm_sq(X, Ψ, K, K, K, big(-Inf), big(Inf))
        @test PenalizedDensity._norm_sq(x, ψ, κ, κ, κ, -Inf, Inf) ≈ Zb rtol = 4eps()
        @test PenalizedDensity._norm_sq_gram(x, ψ, κ, κ, κ, -Inf, Inf)[1] ≈ Zb rtol = 4eps()
        @test PenalizedDensity._norm_sq_grad(x, ψ, κ)[1] ≈ Zb rtol = 4eps()
        @test d.total ≈ Zb rtol = 4eps()
        @test int_quartic(d) ≈ PenalizedDensity._int_quartic(X, Ψ, K, K, K, big(-Inf), big(Inf)) rtol = 4eps()
        # κ-derivatives of the mass and of the roughness operator, on the same amplitudes.
        @test PenalizedDensity._norm_sq_grad(x, ψ, κ)[2] ≈ PenalizedDensity._norm_sq_grad(X, Ψ, K)[2] rtol = 4eps()
        r, rb = PenalizedDensity._dM_dκ_mul(x, κ, ψ), PenalizedDensity._dM_dκ_mul(X, K, Ψ)
        @test all(isapprox.(r, rb; rtol=1e-13))
        # The node gradient, from the same fit evaluated in BigFloat. What remains is the
        # conditioning of the adjoint solve, whose entries grow like 1/θ.
        db = DensityEstimate{BigFloat}(X, big.(d.w), Ψ, K, K, K, big(d.lo), big(d.hi), big(d.λ))
        yev = randn(Xoshiro(7), 30)
        g, gb = logdensity_node_gradient(d, yev), logdensity_node_gradient(db, big.(yev))
        @test all(isapprox.(g, gb; rtol=1e-8))
    end

    @testset "derivatives of the interval mass coefficients" begin
        # Against the closed coth/csch forms evaluated in BigFloat, where cancellation does not
        # limit accuracy, over widths from near-coincident nodes to isolated points.
        dθref(θ) = (cs = csch(θ); ct = coth(θ); (cs^2 * (θ * ct - 1), cs * (2ct - θ * (ct^2 + cs^2)) / 2))
        function dκref(θ)
            cs, ct = csch(θ), coth(θ)
            fd, fc = (ct - θ * cs^2) / 2, cs * (θ * ct - 1) / 2
            fd′, fc′ = dθref(θ)
            return θ * fd′ - fd, θ * fc′ - fc
        end
        setprecision(BigFloat, 512) do
            for θ in exp.(range(log(1e-9), log(40); length=101))
                for (f, ref) in ((PenalizedDensity._mass_coeffs_dθ, dθref), (PenalizedDensity._mass_coeffs_dκ, dκref))
                    a, b = f(θ), ref(big(θ))
                    # Measured on the scale of the pair: fc′ passes through zero near θ = 1.5.
                    @test maximum(abs.(a .- b)) <= 8eps() * sum(abs.(b))
                end
            end
            # The series converges to the working precision.
            a, b = PenalizedDensity._mass_coeffs_dκ(big"1e-3"), dκref(big"1e-3")
            @test all(abs.(a .- b) .<= 1e-140 .* abs.(b))
        end
        @test @inferred(PenalizedDensity._mass_coeffs_dκ(0.01f0)) isa Tuple{Float32,Float32}
        @test @inferred(PenalizedDensity._mass_coeffs_dθ(0.01f0)) isa Tuple{Float32,Float32}
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
        @test issorted(cdf.(d, g))
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
        r = ChisqReference(d)
        @test r isa ChisqReference
        @test mean(r) == r.mean > 0
        @test 0 <= chisq_ccdf(r, 1.0) <= 1
        @test chisq_pdf(r, 1.0) >= 0
        @test 0 <= pvalue(r, chisq(d, t -> 1.0)) <= 1
    end

    @testset "cross-validation on a bounded fit" begin
        @testset "an infinite support reproduces the unbounded selection" begin
            Random.seed!(51)
            x = randn(800)
            @test select_kappa(x) == select_kappa(x; support=(-Inf, Inf))
            @test select_kappa_lscv(x) == select_kappa_lscv(x; support=(-Inf, Inf))
        end

        @testset "_int_quartic: boundary segment vs scale-matched quadrature" begin
            Random.seed!(52)
            x = sort(rand(300))
            d = DensityEstimate(x, 25.0; support=(0.0, 1.0))
            Q2 = PenalizedDensity._int_quartic(d.x, d.ψ, d.κ, d.κL, d.κR, d.lo, d.hi)
            # Quadrature over each node-to-node and boundary segment: a single quadgk call over
            # the whole support can return a misleadingly small error estimate.
            total = quadgk(t -> d(t)^2, d.lo, d.x[1]; rtol=1e-13)[1]
            for k in 1:length(d.x)-1
                total += quadgk(t -> d(t)^2, d.x[k], d.x[k+1]; rtol=1e-13)[1]
            end
            total += quadgk(t -> d(t)^2, d.x[end], d.hi; rtol=1e-13)[1]
            @test Q2 ≈ total rtol = 1e-10
        end

        @testset "leave-one-out expansion matches refits" begin
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
            # several κ including each family's own KLCV-selected scale. KLCV's equal per-node
            # weighting under the log is most sensitive to sparse tail nodes, so it gets the looser
            # bound; on the uniform fit the score itself is ≈ 0 (Q ≈ 1, so ln Q̂ ≈ 0), so it gets
            # an absolute tolerance.
            for (name, xgen, support) in (
                    ("exponential", rng -> -log.(1 .- rand(rng, 300)), (0.0, Inf)),
                    ("uniform",     rng -> rand(rng, 300),              (0.0, 1.0)))
                rng = Xoshiro(hash((:boundedcv, name)))
                x = sort(xgen(rng))
                lo, hi = support
                nodes, w = PenalizedDensity._merge_presorted(x, 0.0)
                κsel = select_kappa(x; support)
                for κ0 in (κsel * 0.5, κsel, κsel * 1.5)
                    a_kl = PenalizedDensity._klcv(nodes, w, κ0, κ0, κ0, lo, hi)
                    b_kl = klcv_refit_b(nodes, w, κ0, lo, hi)
                    @test a_kl ≈ b_kl rtol = 1e-3 atol = 1e-4
                    a_ls = PenalizedDensity._lscv(nodes, w, κ0, κ0, κ0, lo, hi)
                    b_ls = lscv_refit_b(nodes, w, κ0, lo, hi)
                    @test a_ls ≈ b_ls rtol = 5e-3
                end
            end
        end

        @testset "selector: flat limit on a finite support" begin
            # A uniform sample whose cross-validation score falls monotonically as κ → 0: the
            # default grid stops at κ·(hi - lo) = 0.1 and returns a scale near there, rather than
            # extending to where the leave-one-out expansions' error exceeds the score differences.
            u = rand(Xoshiro(0xdd467046e2850655), 300)
            for (x, support) in ((u, (0.0, 1.0)), (3 .* u .- 1, (-1.0, 2.0)))
                κfloor = 0.1 / (support[2] - support[1])
                for sel in (select_kappa, PenalizedDensity.select_kappa_lscv)
                    κ = sel(x; support)
                    @test κfloor <= κ < 2κfloor
                end
            end
            # The rounded-data (interval) selector shares the floor
            κ = select_kappa(round.(u; digits=3); resolution=0.001, support=(0.0, 1.0))
            @test 0.1 <= κ < 0.2
        end

        @testset "selector: bounded vs unbounded κ on a hard-edge family" begin
            x = -log.(1 .- (0.5:1999.5) ./ 2000)   # exponential, a jump edge at the left
            κ_unb = select_kappa(x)
            κ_bnd = select_kappa(x; support=(0.0, Inf))
            @test κ_bnd != κ_unb
            @test κ_bnd < κ_unb    # the wall represents the edge, so no large κ is needed to sharpen it
            nodes, w = PenalizedDensity._merge_presorted(sort(x), 0.0)
            klcv_bnd = PenalizedDensity._klcv(nodes, w, κ_bnd, κ_bnd, κ_bnd, 0.0, Inf)
            klcv_unb = PenalizedDensity._klcv(nodes, w, κ_unb, κ_unb, κ_unb, 0.0, Inf)
            @test klcv_bnd < klcv_unb   # on the bounded score, the bounded selection scores better
        end

        @testset "select_kappa / select_kappa_lscv: input validation" begin
            @test_throws DomainError select_kappa([0.2, 0.5, 0.8]; support=(0.5, 0.5))
            @test_throws "support must satisfy a < b" select_kappa([0.2, 0.5, 0.8]; support=(0.5, 0.5))
            @test_throws DomainError select_kappa([0.2, 0.5, 1.5]; support=(0.0, 1.0))
            @test_throws "lies outside the support" select_kappa([0.2, 0.5, 1.5]; support=(0.0, 1.0))
            @test_throws DomainError select_kappa_lscv([0.2, 0.5, 0.8]; support=(0.5, 0.5))
            @test_throws DomainError select_kappa_lscv([-0.5, 0.5, 0.8]; support=(0.0, 1.0))
            @test_throws "lies outside the support" select_kappa_lscv([-0.5, 0.5, 0.8]; support=(0.0, 1.0))
        end

        @testset "generic indexing: OffsetVector through select_kappa(; support)" begin
            Random.seed!(53)
            x = rand(300)
            κ = select_kappa(x; support=(0.0, 1.0))
            κo = select_kappa(OffsetArray(x, -150); support=(0.0, 1.0))
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

    @testset "boundary-segment inversion converges and round-trips" begin
        # Steep boundary segments (κ·width up to ~25) carry little mass, most of it near the
        # node: a partial mass formed as a difference of nearly equal masses there loses the
        # precision the inversion needs.
        ys = range(-8, 8; length=1601)
        qs = [range(0, 1e-3; length=501); range(1 - 1e-3, 1; length=501)]
        for seed in 1:3, N in (100, 500), κ in (3.0, 10.0), sup in ((-5.0, 5.0), (-4.0, 6.0))
            x = clamp.(randn(Xoshiro(seed), N), sup[1] + 0.01, sup[2] - 0.01)
            for d in (DensityEstimate(x, κ; support=sup),
                      IntervalDensityEstimate(round.(x; digits=1), κ; resolution=0.1, support=sup),
                      DensityEstimate(Float32.(x), Float32(κ); support=Float32.(sup)))
                xs = ungaussianize.(d, ys)
                @test issorted(xs) && all(t -> d.lo <= t <= d.hi, xs)
                @test issorted(quantile.(d, qs))
                # x → y → x across both boundary segments. (The y → x → y direction is
                # limited near a wall by the spacing of floats in x, not by the inversion.)
                xb = [range(d.lo, d.x[1]; length=201); range(d.x[end], d.hi; length=201)]
                @test ungaussianize.(d, gaussianize.(d, xb)) ≈ xb atol = 200 * eps(eltype(d)(5))
            end
        end
    end

    @testset "interior-cell masses keep relative precision" begin
        # A heavy node, then a wide, steep cell, then a light far node: the mass beyond a point
        # in the cell is a tiny fraction of the mass beyond the heavy node, so forming it as a
        # difference from the heavy node's cumulative would cancel. The reference is the same
        # fit evaluated in BigFloat.
        bigfit(d::DensityEstimate) =
            DensityEstimate{BigFloat}(big.(d.x), big.(d.w), big.(d.ψ), big(d.κ), big(d.κL),
                                      big(d.κR), big(d.lo), big(d.hi), big(d.λ))
        bigfit(d::IntervalDensityEstimate) =
            IntervalDensityEstimate{BigFloat}(big.(d.x), big.(d.w), big.(d.ψ), big.(d.k2),
                big(d.kL), big(d.kR), big(d.κ), big(d.κL), big(d.κR), big(d.lo), big(d.hi),
                [PenalizedDensity._IntervalGroup{BigFloat}(g.first, g.last, big(g.r)) for g in d.groups],
                big(d.λ))
        cdfm, ccdfm = PenalizedDensity._cdf_mass, PenalizedDensity._ccdf_mass
        for sgn in (1, -1), κ in (3.0, 10.0), interval in (false, true)
            data = sgn .* vcat(0.3 .* randn(Xoshiro(2), 200), fill(1.0, 100_000), 5.0)
            d = interval ? IntervalDensityEstimate(round.(data; digits=1), κ; resolution=0.1) :
                           DensityEstimate(data, κ)
            db = bigfit(d)
            k = argmax(diff(d.x))           # the wide cell between the heavy and far nodes
            xs = range(d.x[k], d.x[k+1]; length=102)[2:end-1]
            # gaussianize passes the smaller tail mass to Φ⁻¹, at full relative precision.
            for x in xs
                mlo, mhi, mlob, mhib = cdfm(d, x), ccdfm(d, x), cdfm(db, big(x)), ccdfm(db, big(x))
                m, mb = mlo <= mhi ? (mlo, mlob) : (mhi, mhib)
                @test m ≈ mb rtol = 64 * eps()
            end
            # ungaussianize: the tail probability at the returned x matches the target to within
            # what the float spacing of x allows.
            ys = range(extrema(gaussianize.(d, xs))...; length=201)
            xq = ungaussianize.(d, ys)
            @test issorted(xq)
            for (y, x) in zip(ys, xq)
                p = exp(PenalizedDensity._logΦ(-abs(y)))
                pb = (y <= 0 ? cdfm(db, big(x)) : ccdfm(db, big(x))) / db.total
                @test abs(pb - p) <= 64 * eps() * p + 4 * d(x) * eps(x)
            end
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
        κ = select_kappa(x, Adaptive())
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

    @testset "extreme and vanishing boundary gaps" begin
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
    @testset "an infinite support reproduces the unbounded reference" begin
        Random.seed!(60)
        x = randn(150)
        d = DensityEstimate(x, select_kappa_sensitivity(x))
        r = ChisqReference(d)
        d2 = DensityEstimate(x, d.κ; support=(-Inf, Inf))
        r2 = ChisqReference(d2)
        @test r2.tri.dv == r.tri.dv && r2.tri.ev == r.tri.ev && r2.g == r.g && r2.mean == r.mean

        # A far wall (1e4 smoothing lengths out) reproduces the unbounded reference closely.
        far = 1e4 / d.κ
        dfar = DensityEstimate(x, d.κ; support=(minimum(x) - far, maximum(x) + far))
        rfar = ChisqReference(dfar)
        @test maximum(abs.(rfar.tri.dv .- r.tri.dv) ./ abs.(r.tri.dv)) < 1e-10
        @test maximum(abs.(rfar.g .- r.g) ./ abs.(r.g)) < 1e-10
        @test abs(rfar.mean - r.mean) / r.mean < 1e-10
        @test abs(chisq_ccdf(rfar, r.mean) - chisq_ccdf(r, r.mean)) < 1e-8
    end

    @testset "Green's identity on bounded fits" begin
        # 4Σᵢwᵢbᵢ/ψᵢ = 1 (see `green_identity`) checks the bounded `_node_alpha` and `_operator`
        # together on one- and two-sided supports.
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
        da = DensityEstimate(xa, select_kappa(xa, Adaptive(); support=(0.0, Inf)); support=(0.0, Inf))
        @test green_identity(da) ≈ 1 rtol = 1e-5
    end

    @testset "finite differences: m and ∫ψα converge to the closed forms at O(δ²)" begin
        # As in the unbounded check above, but the grid runs from `d.lo` to `d.hi`; ending the
        # conservative stencil there imposes the Neumann condition (see `fieldmc_chisq`).
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
            ψy = amplitude.(d, y)
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

    @testset "field Monte Carlo: bounded uniform (both walls)" begin
        # With natural boundaries a flat density is exactly representable; on an unbounded
        # support it is not, for any constant or varying scale.
        Random.seed!(63)
        x = sort(rand(400) .* 0.6 .+ 0.2)
        d = DensityEstimate(x, select_kappa(x; support=(0.0, 1.0)); support=(0.0, 1.0))
        r = ChisqReference(d)
        chis = fieldmc_chisq(d; nsamp=60_000, seed=11)
        @test mean(r) ≈ mean(chis) rtol = 0.01
        for z in quantile(chis, (0.3, 0.6, 0.9))
            @test chisq_ccdf(r, z) ≈ mean(>(z), chis) atol = 0.012
        end
        @test green_identity(d) ≈ 1 rtol = 1e-5
    end

    @testset "field Monte Carlo: bounded exponential (one wall)" begin
        Random.seed!(64)
        x = sort(-log.(1 .- rand(400)))
        d = DensityEstimate(x, select_kappa(x; support=(0.0, Inf)); support=(0.0, Inf))
        r = ChisqReference(d)
        chis = fieldmc_chisq(d; nsamp=60_000, seed=12)
        @test mean(r) ≈ mean(chis) rtol = 0.01
        for z in quantile(chis, (0.3, 0.6, 0.9))
            @test chisq_ccdf(r, z) ≈ mean(>(z), chis) atol = 0.012
        end
        @test green_identity(d) ≈ 1 rtol = 1e-5
    end

    @testset "field Monte Carlo: bounded fit with adaptive κ" begin
        Random.seed!(65)
        x = sort!(randn(Xoshiro(66), 300) .^ 2)
        κ = select_kappa(x, Adaptive(; window=nothing); support=(0.0, Inf))
        d = DensityEstimate(x, κ; support=(0.0, Inf))
        @test maximum(d.κ) / minimum(d.κ) > 1e4
        r = ChisqReference(d)
        chis = fieldmc_chisq(d; nsamp=60_000, seed=13)
        @test mean(r) ≈ mean(chis) rtol = 0.01
        for z in quantile(chis, (0.3, 0.6, 0.9))
            @test chisq_ccdf(r, z) ≈ mean(>(z), chis) atol = 0.012
        end
        @test green_identity(d) ≈ 1 rtol = 1e-5
    end
end

@testset "select_kappa(x, Adaptive()): fixed support" begin
    @testset "on a bounded domain the selected scale gives a normalized fit" begin
        # Exponential on its natural support.
        x = -log.(1 .- rand(Xoshiro(70), 500))
        κ = select_kappa(x, Adaptive(); support=(0.0, Inf))
        d = DensityEstimate(x, κ; support=(0.0, Inf))
        @test d.lo == 0.0 && d.hi == Inf
        left, El = quadgk(d, 0.0, d.x[1]; rtol=1e-8)
        interior, Ei = quadgk(d, d.x...; rtol=1e-8)
        right, Er = quadgk(d, d.x[end], d.x[end] + 60; rtol=1e-8)
        @test max(El, Ei, Er) < 1e-6
        @test abs(left + interior + right - 1) < 1e-6

        # Uniform on its true (0, 1) support: the selected fit is flat. The power and window
        # candidates both reach a flat fit, with scores that differ by far less than the window
        # rule's penalty k·γ/√N, so the power family is selected.
        xu = rand(Xoshiro(71), 500)
        κu = select_kappa(xu, Adaptive(); support=(0.0, 1.0))
        @test κu isa AdaptiveScale
        du = DensityEstimate(xu, κu; support=(0.0, 1.0))
        @test maximum(t -> abs(du(t) - 1), 0.0:0.01:1.0) < 1e-6
        @test cdf(du, 0.0) == 0.0 && cdf(du, 1.0) == 1.0
        mass, _ = quadgk(du, 0.0, 1.0; rtol=1e-8)
        @test mass ≈ 1 atol=1e-4
    end

    @testset "input validation" begin
        @test_throws DomainError select_kappa([0.2, 0.5, 0.8], Adaptive(); support=(0.5, 0.5))
        @test_throws "support must satisfy a < b" select_kappa([0.2, 0.5, 0.8], Adaptive(); support=(0.5, 0.5))
        @test_throws DomainError select_kappa([-0.5, 0.5, 0.8], Adaptive(); support=(0.0, 1.0))
        @test_throws "lies outside the support" select_kappa([-0.5, 0.5, 0.8], Adaptive(); support=(0.0, 1.0))
    end
end

@testset "select_support: joint boundary and κ selection" begin
    # A Student-t(5) draw built from `randn`: a normal over the root-mean-square of five more
    # normals.
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
            # When neither side gets a wall, the returned κ is identical to `select_kappa(x)`.
            @test r.κ === select_kappa(x)
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
            κunb = select_kappa(x)
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
            κunb = select_kappa(x)
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
        @test_throws ArgumentError select_support([1.0, 2.0, 3.0]; kappas=[1.0, -1.0, 2.0])
        @test_throws "kappas must be sorted and positive" select_support([1.0, 2.0, 3.0]; kappas=[1.0, -1.0, 2.0])
        @test_throws ArgumentError select_support([1.0, 2.0, 3.0]; kappas=[1.0, 2.0])
        @test_throws "need at least 3 values in kappas to bracket the minimum" select_support([1.0, 2.0, 3.0]; kappas=[1.0, 2.0])
        @test_throws ArgumentError select_support([1.0, 2.0, 3.0]; rtol=-1.0)
        @test_throws "rtol must be nonnegative" select_support([1.0, 2.0, 3.0]; rtol=-1.0)
    end

    @testset "tied data and resolution" begin
        y = -log.(1 .- (0.5:499.5) ./ 500)
        x = [y; y[200:219]]          # 4% ties, none near the edge
        @test_throws "repeat another value" select_support(x)
        @test_throws "resolution=0" select_support(x)
        @test_throws ArgumentError select_support(x; resolution=0.1)
        @test_throws "select_support has no form for rounded (interval) data" select_support(x; resolution=0.1)
        r = select_support(x; resolution=0)
        @test r.support[1] <= minimum(x) && isfinite(r.support[1]) && r.support[2] == Inf
        @test r.κ == select_kappa(x; resolution=0, support=r.support)
    end

    @testset "_edge_spacing and _select_gap failure paths" begin
        # Fewer than two points near the edge, tested directly: through select_support the κ
        # selection rejects such data first.
        @test_throws ArgumentError PenalizedDensity._edge_spacing([5.0], :left)
        @test_throws "need at least two distinct points to seed a boundary search" PenalizedDensity._edge_spacing([5.0], :left)

        # Ten points coincide at the left edge, leaving zero spacing to seed a search. The tie
        # check refuses such data first; with resolution=0 they reach the edge check.
        x = vcat(fill(0.0, 10), [100.0])
        @test_throws "repeat another value" select_support(x)
        @test_throws ArgumentError select_support(x; resolution=0)
        @test_throws "the 10 points nearest the left edge coincide" select_support(x; resolution=0)

        # `_select_gap`'s two failure paths, with synthetic scores.
        @test_throws PenalizedDensity.ConvergenceError PenalizedDensity._select_gap(gap -> NaN, 1.0)
        @test_throws "no resolvable boundary gap" PenalizedDensity._select_gap(gap -> NaN, 1.0)
        @test_throws PenalizedDensity.ConvergenceError PenalizedDensity._select_gap(gap -> -gap, 1.0)
        @test_throws "kept running off the search bracket" PenalizedDensity._select_gap(gap -> -gap, 1.0)
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

    @testset "select_support timing (logged, not tested)" begin
        xbig = randn(Xoshiro(74), 50_000)
        t1 = @elapsed select_kappa(xbig)
        t2 = @elapsed select_support(xbig)
        @info "select_support efficiency at N=50000" t_select_kappa=t1 t_select_support=t2 ratio=t2 / t1
        @test isfinite(t2)   # the search completes
    end
end

include("interval.jl")

@testset "code quality (Aqua)" begin
    Aqua.test_all(PenalizedDensity)
end

@testset "explicit imports" begin
    @test ExplicitImports.check_no_implicit_imports(PenalizedDensity) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(PenalizedDensity) === nothing
    @test ExplicitImports.check_all_explicit_imports_via_owners(PenalizedDensity) === nothing
    @test ExplicitImports.check_no_self_qualified_accesses(PenalizedDensity) === nothing
end

# Unexported names that are part of the API; `public` is visible from Julia 1.11 on.
if VERSION >= v"1.11"
    @testset "public API" begin
        for n in (:ConvergenceError, :SolveStats, :merge_samples, :roughness_operator,
                  :select_kappa_lscv, :select_kappa_sensitivity, :klcv, :lscv, :action_sensitivity)
            @test Base.ispublic(PenalizedDensity, n) && !Base.isexported(PenalizedDensity, n)
        end
    end
end
