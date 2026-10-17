# PenalizedDensity

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://timholy.github.io/PenalizedDensity.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://timholy.github.io/PenalizedDensity.jl/dev/)
[![Build Status](https://github.com/timholy/PenalizedDensity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/timholy/PenalizedDensity.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/timholy/PenalizedDensity.jl/graph/badge.svg?token=ELSYIUfH6g)](https://codecov.io/gh/timholy/PenalizedDensity.jl)

Estimate a continuous one-dimensional probability density from sample points, without
binning, using the penalized maximum-likelihood scalar-field method of

> T. E. Holy, "Analysis of Data from Continuous Probability Distributions,"
> *Phys. Rev. Lett.* **79**, 3545 (1997).

The estimate is written `Q̂(x) = ψ(x)²`, and the amplitude `ψ` maximizes the likelihood of
the data subject to a smoothness penalty and normalization `∫Q̂ dx = 1`. A scale
`κ` sets the resolution and can be chosen automatically; the recommended default is
likelihood (KL) cross-validation, with least-squares cross-validation and two action-based
criteria as alternatives. Sorting the data and writing `ψ` as rising and falling exponentials
between adjacent points reduces the estimator to a symmetric tridiagonal problem solved in
`O(N)` time. A robust, binning-free χ² and its reference distribution let you test
candidate models for significance.

```julia
using PenalizedDensity

x = randn(500)                       # samples from an unknown distribution
κ = select_kappa(x)                  # recommended smoothing scale (KL cross-validation)
d = DensityEstimate(x, κ)            # callable: d(x) is the estimate Q̂(x)

d(0.0)                               # density at a point
ref = ChisqReference(d)              # reference law for goodness-of-fit tests
pvalue(ref, chisq(d, x -> exp(-x^2/2)/√(2π)))   # test a model (here a standard normal)
```

`κ` may also vary across the data, which resolves features that a single scale cannot: a
divergent or discontinuous edge, a kink, a heavy tail, or a narrow bump beside a broad mode.
`select_kappa(x, Adaptive())` chooses such a scale by the same cross-validation, and returns a
constant when that scores best:

```julia
z = randn(4000).^2                   # χ²₁: the density diverges at x = 0
d = DensityEstimate(z, select_kappa(z, Adaptive()))
```

The domain may also be bounded: with `support = (a, b)` the density is zero outside `[a, b]`,
so a hard edge is fit exactly rather than approximated by a fast-decaying tail.
`select_support` chooses `a` and `b` by the same cross-validation:

```julia
w = -log.(1 .- rand(1500))           # exponential: a jump edge at x = 0
r = select_support(w)
d = DensityEstimate(w, r.κ; support = r.support)
```

See the [documentation](https://timholy.github.io/PenalizedDensity.jl/dev/) for a full
tutorial and the API reference.

## Related packages

- [KernelDensity](https://github.com/JuliaStats/KernelDensity.jl)
- [KernelDensityEstimate](https://github.com/JuliaRobotics/KernelDensityEstimate.jl)
- [KernelDensitySJ](https://github.com/rasmushenningsson/KernelDensitySJ.jl)

PenalizedDensity is typically about as fast as the fastest of these. It differs from them in
two ways: its model is fully differentiable, so many quantities can be computed analytically,
and it does not bin the data.
