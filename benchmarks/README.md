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
   over many trials.

3. **Scale-selection diagnostic** — for `PenalizedDensity`, the κ that `kappa_interval`
   selects versus the κ (on a scan) that minimises `L1` against the true density. This
   separates the estimator's accuracy from its automatic scale choice.

## Interpreting two things you will see

- **`rtol = 0` runtime is not robust at large `N`.** The tridiagonal solve depends on the
  gaps between adjacent points through `κ · Δx`. With `rtol = 0` nothing is merged, so a
  single pathologically close pair (increasingly likely as `N` grows) drives `κ · Δx → 0`,
  where `coth`/`csch` blow up and the Newton system's condition number explodes; the fit
  then converges slowly. `rtol = 1e-3` merges points closer than `rtol / κ` — a fraction of
  the smoothing length that carries no independent information — which bounds the
  conditioning and keeps the fit fast. Use `rtol > 0` on large, densely packed samples.

- **`kappa_interval` over-resolves smooth data.** The half-entropy scale it selects grows
  roughly like `√N`, faster than the mean-integrated-error-optimal scale (`~N^{1/5}`), so it
  chooses a finer κ than minimises `L1`/`ISE` for smooth unimodal densities. The
  `best κ` column shows the estimator itself is competitive with — often better than — the
  kernel methods once κ is well chosen; the gap is in the automatic selection, not the
  estimator.
