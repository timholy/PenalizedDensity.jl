# Regenerate the figures committed under docs/src/assets/.
#
# These are static PNGs so the Documenter build needs no plotting stack. Run this
# script by hand after changing an example. It needs CairoMakie in the active
# environment (not a dependency of the package or the docs build):
#
#     julia --project=@figures docs/make_figures.jl
#
# where the @figures shared environment has CairoMakie and PenalizedDensity dev'd.

using CairoMakie, PenalizedDensity, Random
CairoMakie.activate!(type="png")

const ASSETS = joinpath(@__DIR__, "src", "assets")
mkpath(ASSETS)

# Mixture of two Gaussians differing in both centroid and width. A single κ whose
# kernel width 1/κ is below the narrower component resolves both.
Random.seed!(42)
comps = [(w=0.5, μ=-2.0, σ=0.4), (w=0.5, μ=3.0, σ=1.2)]
truepdf(x) = sum(c.w * exp(-((x - c.μ) / c.σ)^2 / 2) / (c.σ * sqrt(2π)) for c in comps)

N = 2000
x = [rand() < comps[1].w ? comps[1].μ + comps[1].σ * randn() :
                           comps[2].μ + comps[2].σ * randn() for _ in 1:N]

# Fit at the recommended KL cross-validation scale and at the half-entropy scale.
ki = kappa_interval(x)
κkl = select_kappa_kl(x)
d_half = DensityEstimate(x; κ=ki.κ)
d_kl = DensityEstimate(x; κ=κkl)

g = range(-4.5, 7.5; length=800)
fig = Figure(size=(760, 420), fontsize=15)
ax = Axis(fig[1, 1]; xlabel="x", ylabel="probability density",
          title="Two Gaussians, σ = 0.4 and 1.2, recovered with a single κ")
hist!(ax, x; bins=60, normalization=:pdf, color=(:gray, 0.22), strokewidth=0, label="data")
lines!(ax, g, truepdf.(g); color=:black, linestyle=:dash, linewidth=2, label="true density")
lines!(ax, g, d_kl.(g); color=:steelblue, linewidth=2.5,
       label="κ = $(round(κkl; digits=1)) (KL, recommended)")
lines!(ax, g, d_half.(g); color=:crimson, linewidth=2.5,
       label="κ = $(round(ki.κ; digits=1)) (half-entropy)")
axislegend(ax; position=:rt, framevisible=false)
xlims!(ax, -4.5, 7.5); ylims!(ax, -0.005, nothing)

save(joinpath(ASSETS, "mixture_example.png"), fig; px_per_unit=2)

# --- Second figure: the reduced action and the half-entropy scale ---------------
# g(κ) = S(κ) + W ln κ rises from W/2 (one lump) to W/2 + W H (isolated points); the
# right axis reads this as h = fraction of the entropy resolved.
W = N; Hent = log(N)                    # distinct samples ⇒ H = ln N
κg = exp.(range(log(0.02), log(5000); length=80))
Svals = [action(DensityEstimate(x; κ)) for κ in κg]
gvals = Svals .+ W .* log.(κg)
glo, ghi = W / 2, W / 2 + W * Hent

fig2 = Figure(size=(770, 440), fontsize=15)
ax = Axis(fig2[1, 1]; xscale=log10, xlabel="κ  (smoothing scale)",
          ylabel="reduced action  g = S + W ln κ",
          title="How much of the data's entropy each κ resolves")
xlims!(ax, κg[1], κg[end]); ylims!(ax, glo - 0.06 * W * Hent, ghi + 0.06 * W * Hent)
vspan!(ax, ki.lo, ki.hi; color=(:orange, 0.13))
vlines!(ax, ki.κ; color=:seagreen, linestyle=:dot, linewidth=2)
hlines!(ax, [glo, ghi]; color=:gray40, linestyle=:dash, linewidth=1.5)
lines!(ax, κg, gvals; color=:crimson, linewidth=3)
text!(ax, κg[end], glo; text="W/2  —  one lump (κ→0)", align=(:right, :bottom), fontsize=12, color=:gray40)
text!(ax, 0.03, ghi; text="W/2 + W·H  —  N isolated points (κ→∞)", align=(:left, :top), fontsize=12, color=:gray40)
text!(ax, ki.κ, ghi - 0.02 * W * Hent; text="half-entropy κ", align=(:center, :top), fontsize=12, color=:seagreen)
text!(ax, sqrt(ki.lo * ki.hi), glo + 0.16 * W * Hent; text="plausible range",
      align=(:center, :center), fontsize=11, color=:darkorange3)

axr = Axis(fig2[1, 1]; xscale=log10, yaxisposition=:right,
           ylabel="h  =  fraction of entropy resolved")
hidespines!(axr); hidexdecorations!(axr)
xlims!(axr, κg[1], κg[end]); ylims!(axr, -0.06, 1.06)

save(joinpath(ASSETS, "action_entropy.png"), fig2; px_per_unit=2)
