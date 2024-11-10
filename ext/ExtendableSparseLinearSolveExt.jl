module ExtendableSparseLinearSolveExt
using LinearSolve
import ExtendableSparse: LinearSolvePreconBuilder
import LinearAlgebra
using SparseArrays: AbstractSparseMatrixCSC

# Harrr!
# Avoid type piracy by adding a wrapper struct
function (method::LinearSolve.AbstractFactorization)(A)
    pr = LinearProblem(A, zeros(eltype(A), size(A, 1)))
    init(pr, method)
end

function LinearAlgebra.ldiv!(u, cache::LinearSolve.LinearCache, b)
    cache.b = b
    sol = solve!(cache)
    copyto!(u, sol.u)
end

(b::LinearSolvePreconBuilder)(A::AbstractSparseMatrixCSC,p) = (b.method(A), LinearAlgebra.I)

end

