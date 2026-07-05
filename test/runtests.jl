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

    @testset "atol merges near-coincident points" begin
        d = PenalizedDensityEstimate([0.0, 1e-10, 1.0]; κ=1.0, atol=1e-6)
        @test length(d.x) == 2
        @test d.w == [2.0, 1.0]
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

    @testset "input validation (fail fast)" begin
        @test_throws ArgumentError PenalizedDensityEstimate(Float64[]; κ=1.0)
        @test_throws ArgumentError PenalizedDensityEstimate([1.0]; κ=0.0)
        @test_throws ArgumentError PenalizedDensityEstimate([1.0]; κ=-1.0)
        @test_throws ArgumentError PenalizedDensityEstimate([1.0]; κ=1.0, atol=-1.0)
        @test_throws ArgumentError select_kappa([1.0, 2.0]; κs=[1.0, -1.0, 2.0])
        @test_throws ArgumentError select_kappa([1.0, 2.0]; κs=[1.0, 2.0])  # need ≥ 3
    end
end
