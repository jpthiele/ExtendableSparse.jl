"""
    UMFPACKPreconBuilder()

Return callable object constructing a formal left preconditioner from an LU factorization using UMFPACK to be passed
as the `precs` parameter to iterative methods wrapped by LinearSolve.jl.
"""
struct UMFPACKPreconBuilder end
@static if USE_GPL_LIBS
(::UMFPACKPreconBuilder)(A::AbstractSparseMatrixCSC,p)=(SparseArrays.UMFPACK.UmfpackLU(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),nonzeros(A))),LinearAlgebra.I)
end

"""
    SparspakPreconBuilder()

Return callable object constructing a formal left preconditioner from an LU factorization using Sparspak to be passed
as the `precs` parameter to iterative methods wrapped by LinearSolve.jl.
"""
struct SparspakPreconBuilder end
(::SparspakPreconBuilder)(A::AbstractSparseMatrixCSC,p)=(sparspaklu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),nonzeros(A))),LinearAlgebra.I)

"""
    JacobiPreconBuilder()

Return callable object constructing a left Jacobi preconditioner
to be passed as the `precs` parameter to iterative methods wrapped by LinearSolve.jl.
"""
struct JacobiPreconBuilder end
(::JacobiPreconBuilder)(A::AbstractSparseMatrixCSC,p)=(JacobiPreconditioner(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),nonzeros(A))),LinearAlgebra.I)


"""
    ILUZeroPreconBuilder()

Return callable object constructing a left zero fill-in ILU preconditioner 
using [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl)
"""
struct ILUZeroPreconBuilder end
(::ILUZeroPreconBuilder)(A,p)=(ilu0(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),nonzeros(A))),LinearAlgebra.I)

"""
    ILUTPreconBuilder(; droptol=0.1)

Return callable object constructing a left ILUT preconditioner 
using [IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl)
"""
Base.@kwdef struct ILUTPreconBuilder
    droptol::Float64=0.1
end
(::ILUTPreconBuilder)(A,p)= error("import IncompleteLU.jl in order to use ILUTBuilder")



"""
    SmoothedAggregationPreconBuilder(;blocksize=1, kwargs...)

Return callable object constructing a left smoothed aggregation algebraic multigrid preconditioner
using [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl).

Needs `import AlgebraicMultigrid` to trigger the corresponding extension.
"""
struct SmoothedAggregationPreconBuilder{Tk}
    blocksize::Int
    kwargs::Tk
end

function SmoothedAggregationPreconBuilder(;blocksize=1, kwargs...)
    SmoothedAggregationPreconBuilder(blocksize,kwargs)
end

(::SmoothedAggregationPreconBuilder)(A,p)= error("import AlgebraicMultigrid in order to use SmoothedAggregationPreconBuilder")

"""
   RugeStubenPreconBuilder(;blocksize=1, kwargs...)

Return callable object constructing a  left algebraic multigrid preconditioner after Ruge & Stüben
using [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl).

Needs `import AlgebraicMultigrid` to trigger the corresponding extension.
"""
struct RugeStubenPreconBuilder{Tk}
    blocksize::Int
    kwargs::Tk
end

function RugeStubenPreconBuilder(;blocksize=1, kwargs...)
    SmoothedAggregationPreconBuilder(blocksize,kwargs)
end

(::RugeStubenPreconBuilder)(A,p)= error("import AlgebraicMultigrid in order to use RugeStubenAMGBuilder")
