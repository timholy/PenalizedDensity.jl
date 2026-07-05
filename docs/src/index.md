```@meta
CurrentModule = PenalizedDensity
```

# PenalizedDensity

[PenalizedDensity](https://github.com/timholy/PenalizedDensity.jl) estimates a
continuous one-dimensional probability density from sample points, without binning,
using the penalized maximum-likelihood scalar-field method of

> T. E. Holy, "Analysis of Data from Continuous Probability Distributions,"
> *Phys. Rev. Lett.* **79**, 3545 (1997).

The density is written ``Q(x) = \psi(x)^2``, and the amplitude ``\psi`` maximizes the
likelihood of the data subject to a smoothness penalty and normalization
``\int Q\,dx = 1``. Each data point contributes a peak of width ``\sim 1/\kappa``; the
single smoothing scale ``\kappa`` trades resolution against smoothness. Sorting the data
and writing ``\psi`` as rising and falling exponentials between adjacent points reduces
the estimator to a symmetric tridiagonal problem, solved by a convex Newton iteration in
``O(N)`` time with normalization imposed by a single rescaling.

## Getting started

```@example
using PenalizedDensity

x = [-2.1, -0.4, -0.4, 0.3, 1.2, 1.9]   # samples from an unknown distribution

d = PenalizedDensityEstimate(x; κ = 1.0)  # callable: d(x) is the density Q(x)
(d(0.0), d(2.0))
```

The fit is callable and returns the estimated density; [`amplitude`](@ref) gives
``\psi(x) = \sqrt{Q(x)}``. Use [`select_kappa`](@ref) to choose the smoothing scale by
Stevenson's principle of minimum sensitivity, and [`chisq`](@ref) / [`pvalue`](@ref) to
test a candidate model against the data with a robust, binning-free ``\chi^2`` whose
reference distribution is known ([`chisq_pdf`](@ref), [`chisq_ccdf`](@ref)).

## API reference

```@index
```

```@autodocs
Modules = [PenalizedDensity]
```
