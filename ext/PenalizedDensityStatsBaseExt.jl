module PenalizedDensityStatsBaseExt

using PenalizedDensity: PenalizedDensity, DensityEstimate
using StatsBase: StatsBase

# Only the in-sample form: StatsBase's two-argument `entropy(p, b)` takes a logarithm base,
# which the held-out `PenalizedDensity.entropy(d, xeval)` does not.
StatsBase.entropy(d::DensityEstimate) = PenalizedDensity.entropy(d)

end
