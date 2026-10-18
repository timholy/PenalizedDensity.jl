module PenalizedDensityStatsBaseExt

using PenalizedDensity: PenalizedDensity, AbstractDensityEstimate
using StatsBase: StatsBase

# Only the in-sample form: StatsBase's two-argument `entropy(p, b)` takes a logarithm base,
# which the held-out `PenalizedDensity.entropy(d, xeval)` does not.
StatsBase.entropy(d::AbstractDensityEstimate) = PenalizedDensity.entropy(d)

end
