using PenalizedDensity
using OffsetArrays
using Random, Statistics
using Test

# Trapezoidal integral of a callable over a wide, fine grid.
function integrate(f, a, b; n=2_000_001)
    xs = range(a, b; length=n)
    ys = f.(xs)
    return (sum(ys) - (ys[1] + ys[end]) / 2) * step(xs)
end

@testset "PenalizedDensity.jl" begin
    @testset "single point is a Laplace density" begin
        κ = 1.5
        d = PenalizedDensityEstimate([2.0]; κ)
        @test d.ψ[1] ≈ sqrt(κ)
        @test d(2.0) ≈ κ                       # Q(x₀) = κ
        @test d(3.0) ≈ κ * exp(-2κ)            # Q(x) = κ e^{-2κ|x-x₀|}
        @test d(1.0) ≈ κ * exp(-2κ)            # symmetric
        @test integrate(d, -30, 30) ≈ 1 atol = 1e-8
    end

    @testset "normalization and shape" begin
        d = PenalizedDensityEstimate([-1.0, 0.0, 0.0, 1.0]; κ=1.0)
        @test integrate(d, -30, 30) ≈ 1 atol = 1e-8
        @test d(0.0) > d(2.0)                  # denser near the data
        @test all(≥(0), d.(range(-5, 5; length=101)))   # Q = ψ² ≥ 0
    end

    @testset "continuity at the nodes" begin
        d = PenalizedDensityEstimate([0.0, 1.3, 2.0, 5.1]; κ=0.7)
        for xi in d.x
            @test amplitude(d, xi - 1e-9) ≈ amplitude(d, xi + 1e-9) atol = 1e-6
        end
    end

    @testset "repeated points equal integer weights" begin
        # Merging identical points must reproduce the weighted problem.
        d1 = PenalizedDensityEstimate([0.0, 0.0, 0.0, 4.0]; κ=1.2)
        @test d1.x == [0.0, 4.0]
        @test d1.w == [3.0, 1.0]
        @test integrate(d1, -30, 40) ≈ 1 atol = 1e-8
        # Heavier weight ⇒ larger amplitude at that node.
        @test d1.ψ[1] > d1.ψ[2]
    end

    @testset "rtol merges points within rtol/κ" begin
        d = PenalizedDensityEstimate([0.0, 1e-10, 1.0]; κ=1.0, rtol=1e-6)
        @test length(d.x) == 2
        @test d.w == [2.0, 1.0]
        # The threshold is rtol/κ: here 1e-3/2 = 5e-4.
        @test length(PenalizedDensityEstimate([0.0, 1e-4, 1.0]; κ=2.0, rtol=1e-3).x) == 2
        @test length(PenalizedDensityEstimate([0.0, 1e-3, 1.0]; κ=2.0, rtol=1e-3).x) == 3
        # The default rtol > 0 merges points far below the resolution (numerical hygiene).
        @test length(PenalizedDensityEstimate([0.0, 1e-9, 1.0]; κ=1.0).x) == 2
        # Merging points closer than the resolution is lossless.
        Random.seed!(5)
        x = randn(5_000)
        da = PenalizedDensityEstimate(x; κ=3.0)
        db = PenalizedDensityEstimate(x; κ=3.0, rtol=1e-3)
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
        d = PenalizedDensityEstimate(x; κ=3.0, rtol=0.0)
        @test all(isfinite, d.ψ) && all(>(0), d.ψ)
        @test integrate(d, -40, 40) ≈ 1 atol = 1e-6
        # Stationarity of the normalised amplitude: (−M)ψ = (κ/λ) w ./ ψ (Eq. field equation).
        negM = PenalizedDensity._neg_M(d.x, d.κ)
        resid = negM * d.ψ .- (d.κ / d.λ) .* d.w ./ d.ψ
        @test maximum(abs, resid) < 1e-6 * maximum(abs, negM * d.ψ)
        PenalizedDensityEstimate(x; κ=3.0, rtol=0.0)           # compile before measuring
        @test (@allocated PenalizedDensityEstimate(x; κ=3.0, rtol=0.0)) < 40 * length(x) * sizeof(Float64)
    end

    @testset "scale equivariance" begin
        # Q is scale-equivariant: rescaling x → s·x with κ → κ/s gives Q_s(s·x) = Q(x)/s.
        # −M, the Newton solve, and the convergence test depend on x, κ only through κ·Δx,
        # so the unnormalised fit and its stopping criterion are invariant under this scaling.
        Random.seed!(11)
        x = randn(2_000)
        d = PenalizedDensityEstimate(x; κ=3.0)
        xt = range(-3, 3; length=25)
        Q = d.(xt)
        for s in (1e-15, 1e20)
            ds = PenalizedDensityEstimate(s .* x; κ=3.0 / s)
            @test maximum(abs.(ds.(s .* xt) .- Q ./ s) ./ (Q ./ s)) < 1e-8
        end
    end

    @testset "λ ≈ N near the optimal κ (Eq. 10 asymptotics)" begin
        Random.seed!(1)
        x = randn(400)
        κ = select_kappa(x; κs=exp.(range(log(0.05), log(20); length=40)))
        d = PenalizedDensityEstimate(x; κ)
        @test 0.75 < d.λ / length(x) < 1.25    # paper: λ ≈ N
        @test integrate(d, -40, 40) ≈ 1 atol = 1e-6
        φ(t) = exp(-t^2 / 2) / sqrt(2π)         # recovers a standard normal
        @test mean(abs.(d.(range(-4, 4; length=161)) .- φ.(range(-4, 4; length=161)))) < 0.05
    end

    @testset "callable on arrays; amplitude² == density" begin
        d = PenalizedDensityEstimate([-2.0, 0.5, 3.0]; κ=0.9)
        g = range(-4, 5; length=25)
        @test d(collect(g)) ≈ d.(g)
        @test amplitude(d, collect(g)) .^ 2 ≈ d.(g)
    end

    @testset "generic indexing: OffsetArray input" begin
        x = [-1.5, 0.2, 0.2, 1.1, 3.4]
        d = PenalizedDensityEstimate(x; κ=1.1)
        do_ = PenalizedDensityEstimate(OffsetArray(x, -3); κ=1.1)   # 0-based-ish axes
        @test do_.x == d.x
        @test do_.ψ ≈ d.ψ
        @test do_(0.7) ≈ d(0.7)
    end

    @testset "goodness of fit: statistic" begin
        d = PenalizedDensityEstimate([-1.0, 0.0, 0.0, 1.0]; κ=1.0)
        @test chisq(d, d) == 0                 # a distribution vs itself
        @test chisq(d, x -> 0.9 * d(x)) > 0    # a mismatched (here unnormalised) trial
        # matches the defining sum 4 Σ wᵢ (√Q(xᵢ)/ψ_cl(xᵢ) − 1)²
        Q(x) = exp(-x^2 / 2) / sqrt(2π)
        manual = 4 * sum(d.w[i] * (sqrt(Q(d.x[i])) / d.ψ[i] - 1)^2 for i in eachindex(d.x))
        @test chisq(d, Q) ≈ manual
        @test_throws ArgumentError chisq(d, x -> -1.0)   # negative trial density
    end

    @testset "goodness of fit: reference distribution" begin
        d = PenalizedDensityEstimate([-1.0, 0.0, 0.0, 1.0]; κ=1.0)
        μ = expected_chisq(d)
        @test μ > 0
        # Eq. 25: ⟨χ²⟩ is 1/√2 per effective bin (κX bins).
        N = sum(d.w); X = sum(d.w ./ d.ψ.^2) / N
        @test μ / (d.κ * X) ≈ 1 / sqrt(2)
        # Inverse-Gaussian(mean μ, shape μ²): normalised, with mean μ.
        zs = range(1e-5, 40μ; length=4_000_001)
        p = chisq_pdf.(Ref(d), zs)
        trap(f) = (sum(f) - (f[1] + f[end]) / 2) * step(zs)
        @test trap(p) ≈ 1 atol = 1e-4
        @test trap(zs .* p) ≈ μ rtol = 1e-4
        @test chisq_pdf(d, -1.0) == 0 && chisq_pdf(d, 0.0) == 0
        # ccdf is the pdf's upper tail and its negative derivative is the pdf.
        z0 = 1.3μ
        @test chisq_ccdf(d, z0) ≈ trap(p .* (zs .≥ z0)) atol = 1e-4
        @test chisq_ccdf(d, 0.0) == 1
        h = 1e-5
        @test -(chisq_ccdf(d, z0 + h) - chisq_ccdf(d, z0 - h)) / 2h ≈ chisq_pdf(d, z0) rtol = 1e-4
        # ccdf stays in [0,1] and monotone decreasing.
        grid = range(0.01μ, 10μ; length=200)
        c = chisq_ccdf.(Ref(d), grid)
        @test all(0 .≤ c .≤ 1) && issorted(c; rev=true)
        # pvalue is the ccdf at the observed statistic.
        Q(x) = exp(-x^2 / 2) / sqrt(2π)
        @test pvalue(d, Q) == chisq_ccdf(d, chisq(d, Q))
    end

    @testset "kappa_interval: principled range" begin
        Random.seed!(7)
        x = randn(600)
        ki = kappa_interval(x)
        @test ki.lo < ki.κ < ki.hi
        # h = ½ point agrees with the minimum-sensitivity scale within a small factor.
        κms = select_kappa(x; κs = exp.(range(log(0.05), log(50); length = 60)))
        @test 0.5 < ki.κ / κms < 2.0
        # A wider band brackets a narrower one around the same central κ.
        wide = kappa_interval(x; level = 0.6)
        @test wide.κ ≈ ki.κ rtol = 1e-3
        @test wide.lo < ki.lo && wide.hi > ki.hi
        # Analytic asymptotes of g(κ) = S(κ) + W ln κ: W/2 (κ→0) and W/2 + W·H (κ→∞).
        # Use widely-separated points so isolation is reached at a moderate κ (avoiding
        # sinh overflow), where H = ln 3 exactly.
        xs = [0.0, 5.0, 10.0]; W3 = 3; H3 = log(3)
        g(κ) = action(PenalizedDensityEstimate(xs; κ)) + W3 * log(κ)
        @test g(1e-5) ≈ W3 / 2 rtol = 1e-3            # one lump
        @test g(3.0) ≈ W3 / 2 + W3 * H3 rtol = 1e-4   # three isolated points
        # Repeated points enter through the multiplicity entropy.
        @test kappa_interval([0.0, 0.0, 1.0, 5.0]; level = 0.4).κ > 0
    end

    @testset "large κ stays finite (no sinh overflow)" begin
        Random.seed!(3)
        x = randn(300)
        d = PenalizedDensityEstimate(x; κ = 5000.0)   # kernels far narrower than spacings
        @test all(isfinite, d.ψ) && all(>(0), d.ψ)
        @test isfinite(d.λ) && d.λ > 0
        @test isfinite(action(d))
        @test all(isfinite, amplitude(d, range(-4, 4; length = 200)))  # incl. inter-point gaps
    end

    @testset "input validation (fail fast)" begin
        @test_throws ArgumentError PenalizedDensityEstimate(Float64[]; κ=1.0)
        @test_throws ArgumentError PenalizedDensityEstimate([1.0]; κ=0.0)
        @test_throws ArgumentError PenalizedDensityEstimate([1.0]; κ=-1.0)
        @test_throws ArgumentError PenalizedDensityEstimate([1.0]; κ=1.0, rtol=-1.0)
        @test_throws ArgumentError select_kappa([1.0, 2.0]; κs=[1.0, -1.0, 2.0])
        @test_throws ArgumentError select_kappa([1.0, 2.0]; κs=[1.0, 2.0])  # need ≥ 3
        @test_throws ArgumentError kappa_interval([1.0, 2.0]; level=0.0)
        @test_throws ArgumentError kappa_interval([1.0, 2.0]; level=1.0)
        @test_throws ArgumentError kappa_interval([3.0])              # need ≥ 2 distinct
        @test_throws ArgumentError kappa_interval([3.0, 3.0, 3.0])    # all coincident
    end
end
