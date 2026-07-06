# Compare PenalizedDensity against other one-dimensional Julia density estimators on both
# runtime and accuracy. Run from the repository root with
#
#     julia benchmarks/benchmarks.jl
#
# The script activates its own environment (developing the parent package) so it does not
# touch the package's own dependencies.
#
# Two comparisons are reported:
#   * Runtime scaling at a fixed smoothing scale, which isolates each method's algorithmic
#     cost as the sample size N grows.
#   * Accuracy with each method's own automatic bandwidth selection, i.e. the quality a user
#     gets out of the box, as mean integrated absolute error and mean integrated squared
#     error against a known density.

import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = dirname(@__DIR__))     # bind PenalizedDensity to this checkout
Pkg.instantiate()

using PenalizedDensity
using KernelDensity: kde, pdf, InterpKDE
using KernelDensitySJ: bwsj, density
using KernelDensityEstimate: kde!
using BenchmarkTools: @belapsed
using Random, Statistics, Printf

# ----------------------------------------------------------------------------------------
# Estimators. Each fixed-scale variant fits and evaluates on `grid` at a prescribed
# bandwidth; each auto variant selects its own bandwidth first.
# ----------------------------------------------------------------------------------------

const Κ = 3.0      # PenalizedDensity smoothing scale for the runtime comparison
const H = 0.3      # kernel bandwidth for the runtime comparison (comparable resolution)

pd_fixed(x, grid; rtol = 0.0) = DensityEstimate(x; κ = Κ, rtol).(grid)
kd_fixed(x, grid) = pdf(InterpKDE(kde(x; bandwidth = H)), grid)
sj_fixed(x, grid) = density(x, H, grid)
ke_fixed(x, grid) = vec(kde!(x, [H])(reshape(collect(grid), 1, :)))

pd_auto(x, grid) = (κ = kappa_interval(x; rtol = 1e-3).κ;
                    DensityEstimate(x; κ, rtol = 1e-3).(grid))
kd_auto(x, grid) = pdf(InterpKDE(kde(x)), grid)                 # default (Silverman) bandwidth
sj_auto(x, grid) = density(x, bwsj(x), grid)                    # Sheather–Jones bandwidth
ke_auto(x, grid) = vec(kde!(x)(reshape(collect(grid), 1, :)))   # leave-one-out likelihood

# ----------------------------------------------------------------------------------------
# Test densities: sampler, true pdf, and an integration window wide enough to hold the mass.
# ----------------------------------------------------------------------------------------

const φ = x -> exp(-x^2 / 2) / sqrt(2π)

struct TestCase
    name::String
    sample::Function
    pdf::Function
    lo::Float64
    hi::Float64
end

const CASES = [
    TestCase("standard normal", N -> randn(N), φ, -8.0, 8.0),
    TestCase("bimodal mixture",
             N -> randn(N) .+ 2.5 .* rand((-1.0, 1.0), N),
             x -> (φ(x - 2.5) + φ(x + 2.5)) / 2, -8.0, 8.0),
    TestCase("log-normal (skewed)",
             N -> exp.(randn(N)),
             x -> x > 0 ? exp(-log(x)^2 / 2) / (x * sqrt(2π)) : 0.0, 1e-4, 25.0),
]

trapz(y, grid) = (sum(y) - (y[1] + y[end]) / 2) * step(grid)

# ----------------------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------------------

# Minimum elapsed time over `reps` runs after one warm-up (compilation) call.
function bestof(f, reps)
    @belapsed $f()
end

function runtime_table(; sizes = (1_000, 10_000, 100_000, 1_000_000), ke_cap = 100_000)
    grid = collect(range(-5, 5; length = 512))
    methods = [
        ("PenalizedDensity rtol=0",    (x, g) -> pd_fixed(x, g),               Inf),
        ("PenalizedDensity rtol=1e-3", (x, g) -> pd_fixed(x, g; rtol = 1e-3),  Inf),
        ("KernelDensity",              kd_fixed,                               Inf),
        ("KernelDensitySJ",            sj_fixed,                               Inf),
        ("KernelDensityEstimate",      ke_fixed,                               ke_cap),
    ]
    println("\nRuntime: fit + evaluate on a 512-point grid, fixed bandwidth [ms]")
    @printf("%-28s", "N")
    for N in sizes
        @printf("%12d", N)
    end
    println()
    Random.seed!(1)
    data = Dict(N => randn(N) for N in sizes)
    for (name, f, cap) in methods
        @printf("%-28s", name)
        for N in sizes
            if N > cap
                @printf("%12s", "-")
            else
                reps = N <= 10_000 ? 20 : (N <= 100_000 ? 5 : 2)
                @printf("%12.2f", 1e3 * bestof(() -> f(data[N], grid), reps))
            end
        end
        println()
    end
end

# ----------------------------------------------------------------------------------------
# Accuracy
# ----------------------------------------------------------------------------------------

# Mean L1 = ∫|Q̂ − Q| dx and mean ISE = ∫(Q̂ − Q)² dx over `trials` independent samples.
function accuracy(estimate, case::TestCase, N, trials; grid_n = 4000)
    grid = range(case.lo, case.hi; length = grid_n)
    gridv = collect(grid)
    ptrue = case.pdf.(grid)
    l1 = 0.0
    ise = 0.0
    for _ in 1:trials
        q = estimate(case.sample(N), gridv)
        l1 += trapz(abs.(q .- ptrue), grid)
        ise += trapz((q .- ptrue) .^ 2, grid)
    end
    return l1 / trials, ise / trials
end

function accuracy_table(; N = 2_000, trials = 25)
    methods = [
        ("PenalizedDensity",      pd_auto),
        ("KernelDensity",         kd_auto),
        ("KernelDensitySJ",       sj_auto),
        ("KernelDensityEstimate", ke_auto),
    ]
    println("\nAccuracy: automatic bandwidth, N = $N, $trials trials")
    println("mean integrated |error| (L1)  and  mean integrated squared error (ISE)")
    for case in CASES
        println("\n  ", case.name)
        @printf("    %-24s %14s %14s\n", "method", "L1", "ISE")
        for (name, est) in methods
            Random.seed!(2)                        # same samples for every method
            l1, ise = accuracy(est, case, N, trials)
            @printf("    %-24s %14.5f %14.6f\n", name, l1, ise)
        end
    end
end

# ----------------------------------------------------------------------------------------
# Scale-selection diagnostic
# ----------------------------------------------------------------------------------------

# Separate the estimator's accuracy from its automatic scale choice: compare the κ that
# kappa_interval selects against the κ on a scan that minimises L1 against the true density.
# A large gap means the method could be more accurate than its out-of-the-box selection is.
function selection_diagnostic(; N = 2_000, trials = 15, κscan = range(0.5, 40; length = 80))
    println("\nPenalizedDensity scale selection: kappa_interval vs L1-optimal κ, N = $N")
    @printf("  %-22s %10s %10s %10s %10s\n", "density", "auto κ", "auto L1", "best κ", "best L1")
    for case in CASES
        grid = range(case.lo, case.hi; length = 4000)
        gridv = collect(grid)
        ptrue = case.pdf.(grid)
        aκ = 0.0; aL = 0.0; bκ = 0.0; bL = 0.0
        Random.seed!(3)
        for _ in 1:trials
            x = case.sample(N)
            aκ += (κ = kappa_interval(x; rtol = 1e-3).κ)
            aL += trapz(abs.(DensityEstimate(x; κ, rtol = 1e-3).(gridv) .- ptrue), grid)
            bestκ = first(κscan); bestL = Inf
            for κ in κscan
                L = trapz(abs.(DensityEstimate(x; κ, rtol = 1e-3).(gridv) .- ptrue), grid)
                L < bestL && (bestL = L; bestκ = κ)
            end
            bκ += bestκ; bL += bestL
        end
        @printf("  %-22s %10.2f %10.5f %10.2f %10.5f\n",
                case.name, aκ / trials, aL / trials, bκ / trials, bL / trials)
    end
end

# ----------------------------------------------------------------------------------------

runtime_table()
accuracy_table()
selection_diagnostic()
