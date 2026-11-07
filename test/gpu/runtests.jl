# Evaluation of fits stored on a CUDA GPU. These tests need a CUDA-capable GPU, so they are
# not part of `Pkg.test()`; run them from the package root with
#
#     julia --project=test/gpu -e 'using Pkg; Pkg.instantiate()'
#     julia --project=test/gpu test/gpu/runtests.jl
#
# (Julia 1.10 ignores `[sources]`: first run `Pkg.develop(path=".")` in this environment.)
# `test/runtests.jl` covers the same evaluators through JLArrays, which runs the GPUArrays
# broadcast machinery on the CPU but cannot detect code that fails to compile for a GPU,
# device math that disagrees with the CPU, or Float64 arithmetic in a Float32 kernel.

using PenalizedDensity
using Adapt: adapt
using CUDA
using Random: MersenneTwister
using Statistics: median, quantile
using Test

CUDA.functional() || error("CUDA is not functional on this machine; these tests need a CUDA GPU")

# The pointwise evaluators, the domain each takes (points or probabilities), and the internal
# function each reaches, whose presence in the compiled module shows that the module dump
# includes the evaluator's code.
const EVALUATORS = (
    (name = "d(x)",                    f = (d, s) -> d(s),                                    dom = :x, impl = "_amplitude"),
    (name = "amplitude",               f = amplitude,                                         dom = :x, impl = "_amplitude"),
    (name = "logdensity",              f = logdensity,                                        dom = :x, impl = "_logdensity"),
    (name = "cdf",                     f = cdf,                                               dom = :x, impl = "_cdf_mass"),
    (name = "quantile",                f = quantile,                                          dom = :p, impl = "_quantile"),
    (name = "gaussianize",             f = gaussianize,                                       dom = :x, impl = "_gaussianize"),
    (name = "ungaussianize",           f = ungaussianize,                                     dom = :x, impl = "_ungaussianize"),
    (name = "gaussianize_logjacobian", f = (d, s) -> gaussianize_logjacobian(d, s).logjac,    dom = :x, impl = "_gaussianize_logjac"),
)

# GPU and CPU differ in their elementary functions (CUDA's libdevice versus Julia's), so results
# agree to roundoff rather than bit for bit. The bound is relative to max(|value|, 1): the
# interior log-Jacobian adds ln Q̂ and y²/2, which are much larger than their sum, and its
# measured worst case is 160 eps(T). Algorithmic errors, such as a wrong branch, are far larger.
const RTOL_EPS = 1024

function check_agreement(g, c, ::Type{T}) where {T}
    fin = isfinite.(c)
    isequal(g[.!fin], c[.!fin]) || return false
    return all(abs.(g[fin] .- c[fin]) .<= RTOL_EPS * eps(T) .* max.(abs.(c[fin]), 1))
end

function eval_kernel!(Y, d, X, f)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(X)
        Y[i] = f(d, X[i])
    end
    return nothing
end

# The full LLVM module of a kernel applying `f` to a GPU-stored fit, including every function
# it calls.
device_module(f, dg, X) =
    sprint(io -> CUDA.@device_code_llvm io=io dump_module=true @cuda launch=false eval_kernel!(similar(X), dg, X, f))

# Evaluation points: random draws plus every special case: the nodes, midpoints between them,
# the median, the walls of a finite support and points beyond them, deep tails, and ±Inf.
function eval_points(d, rng, n)
    T = eltype(d)
    xs = d.x
    walls = filter(isfinite, T[d.lo, d.hi])
    special = T[-Inf; -40; xs[1:max(1, length(xs) ÷ 50):end]; (xs[1:end-1] .+ xs[2:end])[1:max(1, length(xs) ÷ 50):end] ./ 2;
                median(d); walls; walls .- 1; walls .+ 1; xs[end]; 40; Inf]
    return [special; 3 .* randn(rng, T, n)]
end
eval_probabilities(::Type{T}, rng, n) where {T} = T[0; 1; floatmin(T); 1e-6; 0.5; 1 - eps(T); rand(rng, T, n)]

@testset "GPU evaluation ($(CUDA.name(CUDA.device())))" begin
    rng = MersenneTwister(23)
    z = randn(rng, 5000)
    zb = abs.(z)
    sel = select_support(Float32.(zb))
    fits = [
        "Float32, unbounded"        => DensityEstimate{Float32}(z, select_kappa(Float32.(z))),
        "Float32, bounded"          => DensityEstimate(Float32.(zb), sel.κ; support = sel.support),
        "Float32, per-interval κ"   => DensityEstimate(Float32.(z .^ 2), select_kappa(Float32.(z .^ 2), Adaptive())),
        "Float64, unbounded"        => DensityEstimate(z, select_kappa(z)),
    ]
    for (label, d) in fits
        T = eltype(d)
        @testset "$label" begin
            dg = adapt(CuArray, d)
            @test dg.x isa CuArray{T,1} && dg.F isa CuArray{T,1}
            @test d.κ isa Real ? dg.κ === d.κ : dg.κ isa CuArray{T,1}
            dv = adapt(Array, dg)
            @test dv.x == d.x && dv.F == d.F && dv.G == d.G && dv.total === d.total

            x = eval_points(d, rng, 10^5)
            p = eval_probabilities(T, rng, 10^5)
            X, P = CuArray(x), CuArray(p)
            for ev in EVALUATORS
                @testset "$(ev.name)" begin
                    xs, Xs = ev.dom === :x ? (x, X) : (p, P)
                    G = ev.f.(dg, Xs)
                    @test G isa CuArray{T,1}
                    @test check_agreement(Array(G), ev.f.(d, xs), T)
                    if T === Float32
                        ll = device_module(ev.f, dg, Xs)
                        @test occursin("@julia_" * ev.impl * "_", ll)
                        @test !occursin("double", ll)       # no Float64 arithmetic
                    end
                end
            end

            # An error inside a kernel still fails, without the message the CPU call gives.
            Xn = copy(X)
            CUDA.@allowscalar Xn[1] = T(NaN)
            @test_throws "exception thrown during kernel execution" Array(gaussianize.(dg, Xn))
            @test check_agreement(Array(gaussianize.(dg, X)), gaussianize.(d, x), T)   # device still usable
        end
    end
end
