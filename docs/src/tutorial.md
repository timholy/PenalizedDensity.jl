```@meta
CurrentModule = PenalizedDensity
```

# Tutorial

This walkthrough fits a bimodal distribution, chooses the smoothing scale, and tests
candidate models against the data.

## A worked example

Consider a mixture of two Gaussians with equal weight, a narrow one (``\sigma=0.4``) centered at ``x=-2`` and a broad one (``\sigma=1.2``) at ``x=3``:

```@example tutorial
using PenalizedDensity, Random

Random.seed!(42)
comps = [(w=0.5, μ=-2.0, σ=0.4), (w=0.5, μ=3.0, σ=1.2)]
truepdf(x) = sum(c.w * exp(-((x - c.μ) / c.σ)^2 / 2) / (c.σ * sqrt(2π)) for c in comps)

N = 2000
x = [rand() < comps[1].w ? comps[1].μ + comps[1].σ * randn() :
                           comps[2].μ + comps[2].σ * randn() for _ in 1:N]

κ = select_kappa_kl(x)                          # recommended scale (KL cross-validation)
d = DensityEstimate(x, κ)              # fit at the recommended scale
nothing # hide
```

`d` is a callable density: `d(x)` returns ``Q(x)``. Plotting the
estimate against the truth — here at the recommended KL scale (see [`select_kappa_kl`](@ref)
below) and, for contrast, the sharper half-entropy scale ``\kappa`` (the plotting code uses
[Makie](https://docs.makie.org/), which is not a dependency of the package):

```julia
using CairoMakie
ki = kappa_interval(x)
d_half = DensityEstimate(x, ki.κ)
g = range(-4.5, 7.5; length = 800)
lines(g, truepdf.(g); linestyle = :dash, label = "true density")
lines!(g, d.(g); label = "κ = $(round(κ, digits=1)) (KL, recommended)")
lines!(g, d_half.(g); label = "κ = $(round(ki.κ, digits=1)) (half-entropy)")
axislegend()
```

![Two-Gaussian mixture recovered with a single κ](assets/mixture_example.png)

Both scales recover both peaks: the recommended KL ``\kappa`` is visibly smoother, the
half-entropy ``\kappa`` sharper, but each resolves the narrow and the broad component at
once. This is the key point: **`κ` is a resolution scale, not a component width.** Its
reciprocal ``1/\kappa`` is smaller than either component's ``\sigma``, so the estimator
resolves features of any larger width; the local width of ``Q`` is set by the data, not by
``\kappa``. The selectors that produced these two scales — [`select_kappa_kl`](@ref) and
[`kappa_interval`](@ref) — are introduced next.

## Choosing the smoothing scale

PenalizedDensity offers four automatic selectors. They split into two families: two that
target *estimation error* (recommended for smooth data) and two that resolve *information*
in the data (the right choice for heavily tied or discrete data). The **recommended
default** is [`select_kappa_kl`](@ref), which minimizes a likelihood (Kullback–Leibler)
cross-validation score:

```@example tutorial
κ_kl = select_kappa_kl(x)       # recommended: error-optimal, likelihood cross-validation
```

Its score is the leave-one-out log-likelihood ``-\tfrac1N\sum_i \ln \hat Q_{-i}(x_i)``, an
estimate of ``\mathrm{KL}(Q \,\|\, \hat Q_\kappa)`` up to a constant; it is the criterion
native to the estimator, whose action is itself a log-likelihood. A close relative,
[`select_kappa_cv`](@ref), instead minimizes a least-squares (MISE) cross-validation score:

```@example tutorial
κ_cv = select_kappa_cv(x)       # least-squares cross-validation (MISE)
```

Both are evaluated analytically — each leave-one-out density comes from a first-order
expansion of the fit, so no point-by-point refitting is needed — and to leading order they
select the same ``\kappa \propto N^{1/5}``. Across a range of test densities `select_kappa_kl`
tracks the error-optimal scale most closely (see `benchmarks/`) and is the cheaper of the
two, which is why it is the default. Both assume a *continuous* underlying density; on
heavily tied or coarsely rounded data their scores are unbounded as ``\kappa\to\infty``, and
the two information-resolving selectors below are the better choice.

The first information-resolving selector, [`kappa_interval`](@ref), returns a principled
scale with a plausible range. Its basis is that the reduced action ``g(\kappa) = S(\kappa) + W\ln\kappa``
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

The second information-resolving selector, [`select_kappa_ms`](@ref), returns a related but
distinct scale: the point of *minimum sensitivity*, where `|dS/d ln κ|` is smallest. Its
derivative is computed analytically, so the result is free of the noise that
finite-differencing the action curve would introduce.

```@example tutorial
κ_ms = select_kappa_ms(x)       # minimum-sensitivity scale
```

`select_kappa_ms` and `kappa_interval` generally select different scales, but both resolve
*information* in the data rather than minimizing error, and on smooth densities they tend to
over-resolve — which is why the cross-validation selectors are recommended by default. Their
value is on heavily tied or discrete data, where they stay bounded and the cross-validation
scores do not.

## Letting the scale vary across the data

Every selector above returns one ``\kappa`` for the whole line, and a single scale has to
compromise: fine enough to resolve the peaks, coarse enough to stay quiet in the tails.
On a smooth density that compromise costs little. But when the density is *irregular* — a
divergent or discontinuous edge, a kink, a heavy tail — a constant ``\kappa`` is limited not
by noise but by the density's own shape, and no choice of it is good everywhere.

[`select_kappa_adaptive`](@ref) lifts the compromise by letting the scale follow the density,
``\kappa(x)`` large where the density is high and small where it is low. Take a ``\chi^2_1``
sample, whose density diverges as ``x^{-1/2}`` at the origin:

```@example adaptive
using PenalizedDensity, Random, Statistics

Random.seed!(7)
z = randn(4000) .^ 2                  # χ²₁: the density diverges at x = 0

κ_const = select_kappa_kl(z)          # one scale everywhere
κ_var = select_kappa_adaptive(z)      # a scale that follows the density
```

The adaptive selector returns an [`AdaptiveScale`](@ref) — a callable ``\kappa(x)`` — which
[`DensityEstimate`](@ref) takes exactly where a number would go:

```@example adaptive
d_const = DensityEstimate(z, κ_const)
d_var = DensityEstimate(z, κ_var)

# d.κ holds the realized rates, one per interval between nodes
(edge = round(Int, d_var.κ[1]),           # at the divergent edge
 median = round(Int, median(d_var.κ)),
 coarsest_tenth = round(Int, quantile(d_var.κ, 0.1)),   # out in the tail
 constant = round(Int, κ_const))
```

`d.κ` reports the scale the fit actually used on each interval between nodes. Here it is
roughly ten times finer than the constant scale at the divergent edge, and several times
coarser than it in the tail — spending resolution where the data can pay for it:

![A varying κ against a divergent edge](assets/adaptive_kappa.png)

Neither fit can truly diverge, but the adaptive one tracks the power law about a decade
further in before rolling over. The payoff is measurable on held-out data — the mean
log-likelihood of a fresh sample, whose gap is the reduction in KL divergence, in nats per
sample:

```@example adaptive
ztest = randn(4000) .^ 2
loglik(d) = mean(log.(d.(ztest)))

(constant = round(loglik(d_const); digits = 3),
 varying = round(loglik(d_var); digits = 3),
 gain = round(loglik(d_var) - loglik(d_const); digits = 3))
```

The rule is a plug-in. A pilot fit ``\hat p`` at the constant scale supplies the shape, and
the scale is drawn from the family ``\kappa(x) = c\,(\hat p(x)/\bar g)^{\alpha}`` (``\bar g``
the geometric mean of ``\hat p`` over the sample). Both the overall scale ``c`` and the
exponent ``\alpha`` — how strongly ``\kappa`` follows the density — are chosen by the *same*
leave-one-out KL score that [`select_kappa_kl`](@ref) minimizes, generalized to a varying
scale and still evaluated in closed form and in ``O(N)``:

```@example adaptive
(c = round(κ_var.c; digits = 1), α = κ_var.α)   # the selected scale and exponent
```

Crucially, the constant scale competes in that same comparison as the ``\alpha = 0`` member
of the family, so **adaptivity is used only when it wins**. When it does not, the selector
says so by returning a plain number rather than an `AdaptiveScale` — as on uniform data,
where ``\kappa \propto \hat p^\alpha`` has no contrast to exploit:

```@example adaptive
select_kappa_adaptive(rand(2000)) isa Real   # nothing to buy: the constant scale wins
```

That return type is not a formality: a fit at a constant scale keeps the full large-``N``
goodness-of-fit machinery described next, which a varying scale does not have.

## Goodness of fit

Because the estimate is a genuine likelihood fit, you can ask how well a *specific* model
distribution describes the data. Using the fit `d` from above, [`chisq`](@ref) is a
robust, binning-free ``\chi^2`` (the squared Hellinger distance between a trial density and
the data):

```@example tutorial
wrong(x) = exp(-((x - 0.5) / 2.5)^2 / 2) / (2.5 * sqrt(2π))   # a single broad Gaussian

(correct = round(chisq(d, truepdf); digits = 1),
 incorrect = round(chisq(d, wrong); digits = 1))
```

The true mixture yields a far smaller ``\chi^2`` than the mismatched single Gaussian.
[`pvalue`](@ref) turns the statistic into a significance under the reference distribution of
``\chi^2`` — the exact finite-``N`` law (a generalized chi-squared), computed by default:

```@example tutorial
(p_correct = round(pvalue(d, truepdf); digits = 3),
 p_incorrect = round(pvalue(d, wrong); digits = 6))
```

The single Gaussian is decisively rejected; the true mixture is not. [`chisq_pdf`](@ref) /
[`chisq_ccdf`](@ref) give the reference density and upper-tail probability, and
[`expected_chisq`](@ref) its mean. To test many trial densities against one fit, build the
reference once with [`chisq_reference`](@ref) and reuse it:

```@example tutorial
ref = chisq_reference(d)
(mean = round(expected_chisq(ref); digits = 2),
 p_correct = round(pvalue(ref, chisq(d, truepdf)); digits = 3))
```

Passing `method = :largeN` selects instead the closed-form large-``N`` approximation of the
original paper (an inverse-Gaussian law with mean ``\kappa X/\sqrt2 \approx 0.7`` per
effective bin); it is cheaper but overstates tail probabilities at the scales the selectors
above typically choose.

All of this works unchanged on a fit at a varying scale: the exact finite-``N`` reference is
computed from the same operator the fit assembles, in the same ``O(N)``. The one exception is
the large-``N`` approximation, whose mean ``\kappa X/\sqrt2`` reads ``1/\kappa`` as a single
effective bin width — a quantity a varying scale does not have. So [`expected_chisq`](@ref)`(d)`
and `method = :largeN` throw on an adaptive fit rather than quietly using the wrong reference;
the exact mean, `expected_chisq(chisq_reference(d))`, is available at every scale.
