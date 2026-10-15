module PenalizedDensityDistributionsExt

using PenalizedDensity: PenalizedDensity, AbstractDensityEstimate, logdensity
using Distributions: Distributions

# `quantile` needs no method here: Distributions extends `Statistics.quantile`, which the fits
# already extend.
Distributions.cdf(d::AbstractDensityEstimate, x::Real) = PenalizedDensity.cdf(d, x)
Distributions.pdf(d::AbstractDensityEstimate, x::Real) = d(x)
Distributions.logpdf(d::AbstractDensityEstimate, x::Real) = logdensity(d, x)

end
