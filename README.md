# PenalizedDensity

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://timholy.github.io/PenalizedDensity.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://timholy.github.io/PenalizedDensity.jl/dev/)
[![Build Status](https://github.com/timholy/PenalizedDensity.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/timholy/PenalizedDensity.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/timholy/PenalizedDensity.jl/graph/badge.svg?token=ELSYIUfH6g)](https://codecov.io/gh/timholy/PenalizedDensity.jl)

Estimate a continuous one-dimensional probability density from sample points, without
binning, using the penalized maximum-likelihood scalar-field method of

> T. E. Holy, "Analysis of Data from Continuous Probability Distributions,"
> *Phys. Rev. Lett.* **79**, 3545 (1997).

The density is written `Q(x) = ψ(x)²`, and the amplitude `ψ` maximizes the likelihood of
the data subject to a smoothness penalty and normalization `∫Q dx = 1`. A single scale
`κ` sets the resolution and can be chosen automatically as the scale of minimum sensitivity
of the fit. Sorting the data and writing `ψ` as rising and falling exponentials
between adjacent points reduces the estimator to a symmetric tridiagonal problem solved in
`O(N)` time. A robust, binning-free χ² and its reference distribution let you test
candidate models for significance.

```julia
using PenalizedDensity

x = randn(500)                       # samples from an unknown distribution
κ = select_kappa(x)                  # principled smoothing scale (minimum sensitivity)
                                     # or select_kappa_cv(x) for the MISE-optimal scale
d = PenalizedDensityEstimate(x; κ)   # callable: d(x) is the density Q(x)

d(0.0)                               # density at a point
pvalue(d, x -> exp(-x^2/2)/√(2π))    # test a model (here a standard normal)
```

See the [documentation](https://timholy.github.io/PenalizedDensity.jl/dev/) for a full
tutorial and the API reference.

## Related packages

- [KernelDensity](https://github.com/JuliaStats/KernelDensity.jl)
- [KernelDensityEstimate](https://github.com/JuliaRobotics/KernelDensityEstimate.jl)
- [KerndelDensitySJ](https://github.com/rasmushenningsson/KernelDensitySJ.jl)

Performance-wise, PenalizedDensity is typically on par with the fastest of
these, and has two other strengths: it is based on a fully-differentiable model
for which much can be computed analytically, and it does not require binning.
