using PenalizedDensity
using Documenter

DocMeta.setdocmeta!(PenalizedDensity, :DocTestSetup, :(using PenalizedDensity); recursive=true)

makedocs(;
    modules=[PenalizedDensity],
    authors="Tim Holy <tim.holy@gmail.com> and contributors",
    sitename="PenalizedDensity.jl",
    format=Documenter.HTML(;
        canonical="https://timholy.github.io/PenalizedDensity.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "API reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/timholy/PenalizedDensity.jl",
    devbranch="main",
)
