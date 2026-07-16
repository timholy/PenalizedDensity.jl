```@meta
CurrentModule = PenalizedDensity
```

# Tutorial

This walkthrough fits a bimodal distribution, chooses the smoothing scale, and tests
candidate models against the data.

## A worked example

Consider a mixture of two Gaussians with equal weight, a narrow one (``\sigma=0.4``) centered at ``x=-2`` and a broad one (``\sigma=1.2``) at ``x=3``. First let's draw some samples from this distribution:

```@example tutorial
using Random

Random.seed!(42)
comps = [(w=0.5, μ=-2.0, σ=0.4), (w=0.5, μ=3.0, σ=1.2)]
truepdf(x) = sum(c.w * exp(-((x - c.μ) / c.σ)^2 / 2) / (c.σ * sqrt(2π)) for c in comps)

N = 2000
xs = [rand() < comps[1].w ? comps[1].μ + comps[1].σ * randn() :
                            comps[2].μ + comps[2].σ * randn() for _ in 1:N]
nothing # hide
```

To construct an estimate of the density from the points `xs`, we first select a *smoothing length scale* and then construct the density:

```@example tutorial
using PenalizedDensity

κ = select_kappa_kl(xs)                 # recommended scale (KL cross-validation)
d = DensityEstimate(xs, κ)              # fit at the recommended scale
nothing # hide
```

`κ` determines the amount of smoothing: the amplitude of the density estimate will decay over a length scale `1/κ`.
`d` is a callable density: `d(x)` returns the estimated density at a single point `x`.
Let's plot this estimate (constructed using the recommended [`select_kappa_kl`](@ref) to compute the scale),
along with the result for a different ``\kappa`` estimator, the half-entropy as computed by
[`kappa_interval`](@ref): 

```julia
using CairoMakie                              # for plotting
ki = kappa_interval(x)                        # alternative estimator for scale
d_half = DensityEstimate(x, ki.κ)             # construct the alternative estimate
g = range(-4.5, 7.5; length = 800)
lines(g, truepdf.(g); linestyle = :dash, label = "true density")
lines!(g, d.(g); label = "κ = $(round(κ, digits=1)) (KL, recommended)")
lines!(g, d_half.(g); label = "κ = $(round(ki.κ, digits=1)) (half-entropy)")
axislegend()
```

![Two-Gaussian mixture recovered with a single κ](assets/mixture_example.png)

Both scales recover both peaks: the recommended KL ``\kappa`` is visibly
smoother, the half-entropy ``\kappa`` sharper, but each resolves the narrow and
the broad component at once. Note that **`κ` is a resolution scale, not a
component width.** Its reciprocal ``1/\kappa`` is smaller than either
component's ``\sigma``, so the estimator resolves features of any larger width,
which is set by the underlying distribution and the data obtained by sampling
from it.

## Choosing the smoothing scale

PenalizedDensity offers four automatic selectors. They split into two families: two that
target *estimation error* (recommended for smooth data) and two that resolve *information*
in the data (the right choice for heavily tied or discrete data). The **recommended
default** is [`select_kappa_kl`](@ref), which minimizes a likelihood (Kullback–Leibler)
cross-validation score:

```@example tutorial
κ_kl = select_kappa_kl(xs)       # recommended: error-optimal, likelihood cross-validation
```

Its score is the leave-one-out log-likelihood ``-\tfrac1N\sum_i \ln \hat
Q_{-i}(x_i)``, where ``\hat Q_{-i}(x_i)`` is an estimate at the point ``x_i``
formed from all points *except* ``x_i`` itself. This score is an estimate of
``\mathrm{KL}(Q \,\|\, \hat Q_\kappa)`` up to a constant; it is the criterion
native to the estimator, whose action is itself a log-likelihood. A close
relative, [`select_kappa_cv`](@ref), instead minimizes a least-squares (MISE)
cross-validation score:

```@example tutorial
κ_cv = select_kappa_cv(xs)       # least-squares cross-validation (MISE)
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
ki = kappa_interval(xs)          # (; κ, lo, hi): the half-entropy scale and a band
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
κ_ms = select_kappa_ms(xs)       # minimum-sensitivity scale
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
``\kappa(x)`` large where the density is high, and small where it is low.
In other words, the smoothing length scale will grow with the size of the expected gap between adjacent sampled points.

As an example, take a ``\chi^2_1`` sample, whose density diverges as ``x^{-1/2}`` at the origin:

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
```

Below, the left panel shows true underlying density (dashed) and the two estimates `d_const` and `d_var`;
the right panel shows how `κ_const` and `κ_var` depend on position.

![A varying κ against a divergent edge](assets/adaptive_kappa.png)

Neither density estimate can track much below ``x \approx 10^{-3}`` (typically, fewer than one hundred points land within `[0, 1e-3]`),
but the adaptive one tracks the power law to about tenfold-smaller `x` than the one with constant `κ`.
The payoff is measurable on held-out data — the mean
log-likelihood of a fresh sample, whose gap is the reduction in KL divergence, in nats per
sample:

```@example adaptive
ztest = randn(4000) .^ 2
loglik(d) = mean(log.(d.(ztest)))

(constant = round(loglik(d_const); digits = 3),
 varying = round(loglik(d_var); digits = 3),
 gain = round(loglik(d_var) - loglik(d_const); digits = 3))
```

Concretely, how is ``\kappa(x)`` determined? The rule is a *plug-in* of the variable-bandwidth kind; Abramson's square-root law,
[*Ann. Statist.* **10**, 1217 (1982)](https://doi.org/10.1214/aos/1176345986), is the
``\alpha = 1/2`` member. A pilot fit ``\hat p`` at the constant scale supplies the shape, and
the scale is drawn from the family ``\kappa(x) = c\,(\hat p(x)/\bar g)^{\alpha}`` (``\bar g``
the geometric mean of ``\hat p`` over the sample). Both the overall scale ``c`` and the
exponent ``\alpha`` — how strongly ``\kappa`` follows the density — are chosen by the *same*
leave-one-out KL score that [`select_kappa_kl`](@ref) minimizes, generalized to a varying
scale and still evaluated in closed form and in ``O(N)``:

```@example adaptive
(c = round(κ_var.c; digits = 1), α = κ_var.α)   # the selected scale and exponent
```

Crucially, the constant scale competes in that same comparison: it's the ``\alpha = 0`` member
of the family, ``\kappa(x) = c``, so **adaptivity is used only when it wins**. When it does not, the selector
says so by returning a plain number rather than an `AdaptiveScale` — as on uniform data,
where ``\kappa \propto \hat p^\alpha`` has no contrast to exploit:

```@example adaptive
select_kappa_adaptive(rand(2000)) isa Real   # nothing to buy: the constant scale wins
```

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

The exact law is a per-call integral (Imhof inversion): accurate, but tens of milliseconds
apiece. Passing `method = :largeN` selects instead the closed-form inverse-Gaussian (Wald)
shape of the original paper's large-``N`` limit, parameterized by the exact mean
[`expected_chisq`](@ref); it costs microseconds, so a large batch of trial densities against
one fit is far cheaper. Anchored to the exact mean, it tracks the exact tail closely rather
than overstating it:

```@example tutorial
(exact = round(pvalue(d, truepdf); digits = 3),
 largeN = round(pvalue(d, truepdf; method = :largeN); digits = 3))
```

All of this — the exact law and the `:largeN` shape alike — works unchanged on a fit at a
varying scale: both are read from the same reference the fit assembles, in the same ``O(N)``.
