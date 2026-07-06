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
smoothing scale ``\kappa`` trades resolution against smoothness, and may either be a single
number or vary across the data, ``\kappa(x)``, to resolve an irregular density where a single
scale cannot. The domain itself may also be bounded, fitting a hard edge exactly rather than
approximating it with a decaying tail. Sorting the data and writing ``\psi`` as rising and
falling exponentials between adjacent points reduces the estimator to a symmetric tridiagonal
problem, solved by a convex Newton iteration in ``O(N)`` time with normalization imposed by a
single rescaling.

## Installation

```julia
using Pkg
Pkg.add("PenalizedDensity")
```

## Quick start

```@example quickstart
using PenalizedDensity

xs = [-2.1, -0.4, -0.4, 0.3, 1.2, 1.9]     # samples from an unknown distribution

d = DensityEstimate(xs, 1.0)  # d is Q, and callable: d(x) returns the density estimate at x 
(d(0.0), d(2.0))
```

The result is callable and returns the estimated density; [`amplitude`](@ref) gives
``\psi(x) = \sqrt{Q(x)}``.

The [Tutorial](@ref) works through choosing the smoothing scale, testing a model
against the data, and measuring the fit's [`entropy`](@ref) and [`negentropy`](@ref);
the [API reference](@ref) documents every function.
