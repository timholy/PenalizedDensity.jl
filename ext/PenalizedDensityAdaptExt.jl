module PenalizedDensityAdaptExt

using PenalizedDensity: DensityEstimate
using Adapt: Adapt, adapt

# Moves the node arrays (and a per-interval κ) to the storage `to` selects, such as a GPU
# array type, keeping the node cumulatives already computed. `adapt(CuArray, d)` makes a fit
# that broadcasts over `CuArray`s; CUDA's kernel conversion reaches this same method.
function Adapt.adapt_structure(to, d::DensityEstimate{T}) where {T}
    x, w, ψ, F, G = adapt(to, d.x), adapt(to, d.w), adapt(to, d.ψ), adapt(to, d.F), adapt(to, d.G)
    κ = adapt(to, d.κ)
    return DensityEstimate{T,typeof(κ),typeof(x)}(x, w, ψ, κ, d.κL, d.κR, d.lo, d.hi, d.λ, F, G, d.total)
end

end
