using Documenter, ExtendableSparse, AlgebraicMultigrid, IncompleteLU, Sparspak, LinearAlgebra

function mkdocs()
    makedocs(; sitename = "ExtendableSparse.jl",
             modules = [ExtendableSparse],
             doctest = false,
             warnonly = true,
             clean = false,
             authors = "J. Fuhrmann",
             repo = "https://github.com/WIAS-PDELib/ExtendableSparse.jl",
             pages = [
                 "Home" => "index.md",
                 "example.md",
                 "extsparse.md",
                 "linearsolve.md",
                 "internal.md",
                 "iter.md",
                 "changes.md",
             ])
end

mkdocs()

deploydocs(; repo = "github.com/WIAS-PDELib/ExtendableSparse.jl.git")
