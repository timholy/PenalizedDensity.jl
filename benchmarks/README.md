# Benchmarks

Compares `PenalizedDensity` against other one-dimensional Julia density estimators
([KernelDensity](https://github.com/JuliaStats/KernelDensity.jl),
[KernelDensitySJ](https://github.com/tpapp/KernelDensitySJ.jl), and
[KernelDensityEstimate](https://github.com/JuliaRobotics/KernelDensityEstimate.jl)) on
runtime and accuracy.

## Running

From the repository root:

```
julia benchmarks/benchmarks.jl
```

The script activates `benchmarks/Project.toml` and `develop`s the parent package into it,
so it does not touch `PenalizedDensity`'s own dependencies. The `Manifest.toml` is not
checked in; the first run resolves and installs the comparison packages.

## What it reports

1. **Runtime scaling** — fit and evaluate on a 512-point grid at a *fixed* bandwidth, as the
   sample size `N` grows. Fixing the bandwidth isolates each method's algorithmic cost from
   its bandwidth-selection cost. Both `rtol = 0` and `rtol = 1e-3` are shown for
   `PenalizedDensity`.

2. **Accuracy** — each method fits with its *own* automatic bandwidth selection (the quality
   a user gets out of the box) on samples from known densities. Reported as mean integrated
   absolute error (`L1 = ∫|Q̂ − Q|`) and mean integrated squared error (`ISE = ∫(Q̂ − Q)²`)
   over many trials. `PenalizedDensity` uses `select_kappa_kl`, the winner of the selector
   shootout below.

3. **Selector shootout** — `PenalizedDensity` offers several automatic scale selectors; this
   holds the estimator fixed and pits them head-to-head. The action-based selectors
   (`select_kappa_ms`, minimum sensitivity; `kappa_interval`, half-entropy) resolve information
   and their κ grows like `√N`; the cross-validation selectors (`select_kappa_cv`, LSCV;
   `select_kappa_kl`, KL) target error and grow like `N^{1/5}`. Each row's `L1`/`ISE` is
   compared against the `oracle` κ that minimises `L1` on a scan — the best the estimator can
   do at any fixed scale.

## Interpreting what you will see

- **`rtol = 0` runtime is not robust at large `N`.** The tridiagonal solve depends on the
  gaps between adjacent points through `κ · Δx`. With `rtol = 0` nothing is merged, so a
  single pathologically close pair (increasingly likely as `N` grows) drives `κ · Δx → 0`,
  where `coth`/`csch` blow up and the Newton system's condition number explodes; the fit
  then converges slowly. `rtol = 1e-3` merges points closer than `rtol / κ` — a fraction of
  the smoothing length that carries no independent information — which bounds the
  conditioning and keeps the fit fast. Use `rtol > 0` on large, densely packed samples.

- **The action-based selectors over-resolve smooth data.** The minimum-sensitivity and
  half-entropy scales grow like `√N`, faster than the mean-integrated-error-optimal scale
  (`~N^{1/5}`), so they choose a finer κ than minimises `L1`/`ISE`. They are the right choice
  only for heavily tied or discrete data, where the cross-validation scores are unbounded as
  `κ → ∞` (see the docstrings).

- **`select_kappa_kl` is the recommended default.** Across the test densities it sits closest
  to the oracle in `L1` — tracking the oracle κ tightly even on the skewed log-normal, where
  `select_kappa_cv` over-resolves — and it is the cheapest cross-validation score to compute
  (it reuses the leave-one-out densities and omits the `∫Q̂²` roughness term). The shootout's
  `oracle` column also shows the estimator is competitive with, and often better than, the
  kernel methods once κ is well chosen.
