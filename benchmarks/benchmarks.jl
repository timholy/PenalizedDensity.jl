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

pd_fixed(x, grid; rtol = 0.0) = DensityEstimate(x, Κ; rtol).(grid)
kd_fixed(x, grid) = pdf(InterpKDE(kde(x; bandwidth = H)), grid)
sj_fixed(x, grid) = density(x, H, grid)
ke_fixed(x, grid) = vec(kde!(x, [H])(reshape(collect(grid), 1, :)))

pd_auto(x, grid) = (κ = select_kappa_kl(x; rtol = 1e-3);      # KL cross-validation (see shootout)
                    DensityEstimate(x, κ; rtol = 1e-3).(grid))
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
# Bandwidth-selector shootout (within PenalizedDensity)
# ----------------------------------------------------------------------------------------

# PenalizedDensity offers several automatic scale selectors. Hold the estimator fixed and pit
# them head-to-head. The action-based selectors (minimum sensitivity, half-entropy) resolve
# information and scale like √N; the cross-validation selectors (LSCV, KL) target error and
# scale like N^{1/5}. For each selector report the mean selected κ and the resulting L1/ISE,
# alongside the oracle κ that minimises L1 on a scan — the best the estimator can do at any
# fixed scale. The selector whose error sits closest to the oracle wins.
const SELECTORS = [
    ("select_kappa_ms (min-sensitivity)", x -> select_kappa_ms(x; rtol = 1e-3)),
    ("kappa_interval (half-entropy)",  x -> kappa_interval(x; rtol = 1e-3).κ),
    ("select_kappa_cv (LSCV/MISE)",    x -> select_kappa_cv(x; rtol = 1e-3)),
    ("select_kappa_kl (KL)",           x -> select_kappa_kl(x; rtol = 1e-3)),
]

function selector_shootout(; N = 2_000, trials = 25, κscan = range(0.5, 40; length = 80))
    println("\nSelector shootout: PenalizedDensity automatic scales head-to-head, N = $N, $trials trials")
    nsel = length(SELECTORS)
    winsL1 = zeros(Int, nsel)
    for case in CASES
        grid = range(case.lo, case.hi; length = 4000)
        gridv = collect(grid)
        ptrue = case.pdf.(grid)
        sumκ = zeros(nsel); sumL1 = zeros(nsel); sumISE = zeros(nsel)
        oκ = 0.0; oL1 = 0.0; oISE = 0.0
        Random.seed!(4)
        for _ in 1:trials
            x = case.sample(N)
            for (j, (_, sel)) in enumerate(SELECTORS)
                κ = sel(x)
                q = DensityEstimate(x, κ; rtol = 1e-3).(gridv)
                sumκ[j] += κ
                sumL1[j] += trapz(abs.(q .- ptrue), grid)
                sumISE[j] += trapz((q .- ptrue) .^ 2, grid)
            end
            bestκ = first(κscan); bestL1 = Inf; bestISE = 0.0
            for κ in κscan
                q = DensityEstimate(x, κ; rtol = 1e-3).(gridv)
                L = trapz(abs.(q .- ptrue), grid)
                L < bestL1 && (bestL1 = L; bestκ = κ; bestISE = trapz((q .- ptrue) .^ 2, grid))
            end
            oκ += bestκ; oL1 += bestL1; oISE += bestISE
        end
        winsL1[argmin(sumL1)] += 1
        println("\n  ", case.name)
        @printf("    %-32s %10s %12s %12s\n", "selector", "mean κ", "L1", "ISE")
        for (j, (name, _)) in enumerate(SELECTORS)
            @printf("    %-32s %10.2f %12.5f %12.6f\n",
                    name, sumκ[j] / trials, sumL1[j] / trials, sumISE[j] / trials)
        end
        @printf("    %-32s %10.2f %12.5f %12.6f\n",
                "oracle (L1-optimal κ)", oκ / trials, oL1 / trials, oISE / trials)
    end
    println("\n  L1 wins across the ", length(CASES), " densities:")
    for (j, (name, _)) in enumerate(SELECTORS)
        @printf("    %-32s %d\n", name, winsL1[j])
    end
end

# ----------------------------------------------------------------------------------------

runtime_table()
accuracy_table()
selector_shootout()
