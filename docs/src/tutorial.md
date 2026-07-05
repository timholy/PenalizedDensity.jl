```@meta
CurrentModule = PenalizedDensity
```

# Tutorial

This walkthrough fits a bimodal distribution, chooses the smoothing scale, and tests
candidate models against the data.

## A worked example

Consider a mixture of two Gaussians that differ in **both** location and width — a narrow
component at ``x=-2`` (``\sigma=0.4``) and a broad one at ``x=3`` (``\sigma=1.2``):

```@example tutorial
using PenalizedDensity, Random

Random.seed!(42)
comps = [(w=0.5, μ=-2.0, σ=0.4), (w=0.5, μ=3.0, σ=1.2)]
truepdf(x) = sum(c.w * exp(-((x - c.μ) / c.σ)^2 / 2) / (c.σ * sqrt(2π)) for c in comps)

N = 2000
x = [rand() < comps[1].w ? comps[1].μ + comps[1].σ * randn() :
                           comps[2].μ + comps[2].σ * randn() for _ in 1:N]

ki = kappa_interval(x)                          # a principled scale + range (see below)
d = PenalizedDensityEstimate(x; κ = ki.κ)       # fit at the half-entropy scale
nothing # hide
```

`d` is a callable density: `d(x)` returns ``Q(x)``, and it accepts arrays too. Plotting the
estimate against the truth — here at both the half-entropy scale ``\kappa`` and the
most-smoothing end ``\kappa_\mathrm{lo}`` of the returned range (the plotting code uses
[Makie](https://docs.makie.org/), which is not a dependency of the package):

```julia
using CairoMakie
d_smooth = PenalizedDensityEstimate(x; κ = ki.lo)
g = range(-4.5, 7.5; length = 800)
lines(g, truepdf.(g); linestyle = :dash, label = "true density")
lines!(g, d_smooth.(g); label = "κ = $(round(ki.lo, digits=1)) (widest in range)")
lines!(g, d.(g); label = "κ = $(round(ki.κ, digits=1)) (half-entropy)")
axislegend()
```

![Two-Gaussian mixture recovered with a single κ](assets/mixture_example.png)

Both peaks are recovered across the whole plausible range of ``\kappa``: the smaller
``\kappa_\mathrm{lo}`` is visibly smoother, the half-entropy ``\kappa`` sharper, but both
resolve the narrow and the broad component at once. This is the key point: **`κ` is a
resolution scale, not a component width.** Its reciprocal ``1/\kappa`` is smaller than
either component's ``\sigma``, so the estimator resolves features of any larger width; the
local width of ``Q`` is set by the data, not by ``\kappa``. A method that tied the kernel
width to a single bandwidth would over-smooth the narrow peak or under-smooth the broad
one. [`kappa_interval`](@ref), introduced next, produced `ki`.

## Choosing the smoothing scale

To pick ``\kappa`` automatically, [`kappa_interval`](@ref) returns a principled scale with
a plausible range. Its basis is that the reduced action ``g(\kappa) = S(\kappa) + W\ln\kappa``
(``W`` = total count) rises monotonically between two *exact* limits — ``W/2`` as
``\kappa\to0`` (all points merge into one lump) and ``W/2 + W H`` as ``\kappa\to\infty``
(the ``N`` points become isolated), where ``H`` is the Shannon entropy of the data. The
normalized quantity ``h(\kappa)\in[0,1]`` is thus the *fraction of the data's entropy that
``\kappa`` resolves*, and its half-point is the returned scale:

```@example tutorial
ki = kappa_interval(x)          # (; κ, lo, hi): the half-entropy scale and a band
(κ = round(ki.κ, digits=1), lo = round(ki.lo, digits=1), hi = round(ki.hi, digits=1))
```

The picture below makes this concrete. The reduced action (red) climbs from its
``\kappa\to0`` floor ``W/2`` (nothing resolved, ``h=0``) to its ``\kappa\to\infty`` ceiling
``W/2 + W H`` (every point resolved, ``h=1``); both dashed limits are exact and depend only
on the counts. The returned scale is where the curve is halfway up — half the entropy
resolved — and the shaded band is the plausible range:

![Reduced action rising between its two entropy limits](assets/action_entropy.png)

The half-entropy point coincides with the classical point of minimum sensitivity — the
scale returned by [`select_kappa`](@ref) — but is located against exact bounds rather than
a discrete derivative, and it comes with an interval (widen it with the `level` keyword).
The band is broad because the action is genuinely flat over a wide range of ``\kappa``.

## Goodness of fit

Because the estimate is a genuine likelihood fit, you can ask how well a *specific* model
distribution describes the data. Using the half-entropy fit `d`, [`chisq`](@ref) is a
robust, binning-free ``\chi^2`` (the squared Hellinger distance between a trial density and
the data):

```@example tutorial
wrong(x) = exp(-((x - 0.5) / 2.5)^2 / 2) / (2.5 * sqrt(2π))   # a single broad Gaussian

(correct = round(chisq(d, truepdf); digits = 1),
 incorrect = round(chisq(d, wrong); digits = 1))
```

The true mixture yields a far smaller ``\chi^2`` than the mismatched single Gaussian.
[`pvalue`](@ref) turns the statistic into a significance under the known large-``N``
reference distribution:

```@example tutorial
(p_correct = round(pvalue(d, truepdf); digits = 3),
 p_incorrect = round(pvalue(d, wrong); digits = 6))
```

The single Gaussian is decisively rejected; the true mixture is not. (The large-``N``
p-value is conservative for an accepted model, so the headline discriminator is the
``\chi^2`` value itself.) [`expected_chisq`](@ref) gives the mean of the reference
distribution (about ``0.7`` per effective bin), and [`chisq_pdf`](@ref) /
[`chisq_ccdf`](@ref) give its full density and upper-tail probability — an inverse-Gaussian
law.
