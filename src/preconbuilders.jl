"""
        LinearSolvePreconBuilder(; method=UMFPACKFactorization())

Return callable object constructing a formal left preconditioner from a sparse LU factorization using LinearSolve
as the `precs` parameter for a  [`BlockPreconBuilder`](@ref)  or  iterative methods wrapped by LinearSolve.jl.
"""
Base.@kwdef struct LinearSolvePreconBuilder
    method=UMFPACKFactorization()
end
(::LinearSolvePreconBuilder)(A,p)= error("import LinearSolve in order to use LinearSolvePreconBuilder")


"""
    JacobiPreconBuilder()

Return callable object constructing a left Jacobi preconditioner
to be passed as the `precs` parameter to iterative methods wrapped by LinearSolve.jl.
"""
struct JacobiPreconBuilder end
(::JacobiPreconBuilder)(A::AbstractSparseMatrixCSC,p)=(JacobiPreconditioner(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),nonzeros(A))),LinearAlgebra.I)


"""
    ILUZeroPreconBuilder(;blocksize=1)

Return callable object constructing a left zero fill-in ILU preconditioner 
using [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl)
"""
Base.@kwdef struct ILUZeroPreconBuilder
    blocksize::Int = 1
end

struct ILUBlockPrecon{N,NN,Tv,Ti}
    ilu0::ILUZero.ILU0Precon{SMatrix{N, N, Tv, NN}, Ti, SVector{N, Tv}}
end

function LinearAlgebra.ldiv!(Y::Vector{Tv},
                             A::ILUBlockPrecon{N,NN,Tv,Ti},
                             B::Vector{Tv}) where {N,NN,Tv,Ti}
    BY=reinterpret(SVector{N,Tv},Y)
    BB=reinterpret(SVector{N,Tv},B)
    ldiv!(BY,A.ilu0,BB)
    Y
end

function (b::ILUZeroPreconBuilder)(A0,p)
    A=SparseMatrixCSC(size(A0)..., getcolptr(A0), rowvals(A0),nonzeros(A0))
    if b.blocksize==1
        (ILUZero.ilu0(A),LinearAlgebra.I)
    else
        (ILUBlockPrecon(ILUZero.ilu0(pointblock(A,b.blocksize),SVector{b.blocksize,eltype(A)})),LinearAlgebra.I)
    end
end


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
