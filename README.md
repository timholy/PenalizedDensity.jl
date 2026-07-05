# PenalizedDensity

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://timholy.github.io/PenalizedDensity.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://timholy.github.io/PenalizedDensity.jl/dev/)
[![Build Status](https://github.com/timholy/PenalizedDensity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/timholy/PenalizedDensity.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/timholy/PenalizedDensity.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/timholy/PenalizedDensity.jl)

Estimate a continuous one-dimensional probability density from sample points, without
binning, using the penalized maximum-likelihood scalar-field method of

> T. E. Holy, "Analysis of Data from Continuous Probability Distributions,"
> *Phys. Rev. Lett.* **79**, 3545 (1997).

The density is written `Q(x) = ψ(x)²`, and the amplitude `ψ` is chosen to maximize the
likelihood of the data subject to a smoothness penalty and normalization `∫Q dx = 1`.
Each data point contributes a peak of width `~1/κ`; the single smoothing parameter `κ`
trades resolution against smoothness and can be selected automatically.

Sorting the data and writing `ψ` as rising and falling exponentials between adjacent
points turns the estimator into a symmetric tridiagonal problem, solved by a convex
Newton iteration in `O(N)` time with normalization imposed by a single rescaling.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/timholy/PenalizedDensity.jl")
```

## Usage

```julia
using PenalizedDensity

x = randn(500)                       # sample from an unknown distribution

# Choose the smoothing scale by Stevenson's minimum-sensitivity criterion...
κ = select_kappa(x; κs = exp.(range(log(0.05), log(20); length = 40)))

# ...and fit. `d` is callable: d(x) is the estimated density Q(x).
d = PenalizedDensityEstimate(x; κ)

d(0.0)                               # density at a point
d(-3:0.1:3)                          # ...or on a grid
amplitude(d, 0.0)                    # the amplitude ψ(x) = √Q(x)
```

If you already know the scale you want, pass it directly:

```julia
d = PenalizedDensityEstimate(x; κ = 2.5)
```

Repeated (or, with `atol`, near-coincident) points are merged and enter with integer
weight, so weighted data is handled naturally.

### Goodness of fit

The method provides a robust, binning-free χ² — the squared Hellinger distance between
a trial density and the data — together with its reference distribution, so a candidate
model can be tested for significance:

```julia
model(x) = exp(-x^2 / 2) / sqrt(2π)  # e.g. a standard normal

χ² = chisq(d, model)                 # goodness of fit of `model` to the data
p  = pvalue(d, model)                # significance in the large-N limit
```

`expected_chisq(d)` returns the mean of the reference distribution (about `0.7` per
effective bin), and `chisq_pdf` / `chisq_ccdf` give its density and upper-tail
probability (an inverse-Gaussian law).

## API

| Function | Purpose |
|---|---|
| `PenalizedDensityEstimate(x; κ, atol=0)` | Fit the density; the result is callable, `d(x) == Q(x)`. |
| `amplitude(d, x)` | The amplitude `ψ(x) = √Q(x)`. |
| `select_kappa(x; κs, atol=0)` | Choose `κ` by minimum sensitivity of the action. |
| `action(d)` | Classical action of the fit (Eq. 10). |
| `chisq(d, Q)` | Goodness-of-fit statistic for a trial density `Q`. |
| `expected_chisq(d)` | Mean `⟨χ²⟩` of the reference distribution. |
| `chisq_pdf(d, z)`, `chisq_ccdf(d, z)` | Reference χ² density and upper-tail probability. |
| `pvalue(d, Q)` | Significance of `Q`, `chisq_ccdf(d, chisq(d, Q))`. |
