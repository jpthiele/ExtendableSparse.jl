module ExtendableSparseIncompleteLUExt
using ExtendableSparse
using IncompleteLU 
using LinearAlgebra: I
using SparseArrays: AbstractSparseMatrixCSC, SparseMatrixCSC, getcolptr, rowvals, nonzeros

import ExtendableSparse: ILUTPreconBuilder

(b::ILUTPreconBuilder)(A::AbstractSparseMatrixCSC,p)=(IncompleteLU.ilu(SparseMatrixCSC(A); τ = b.droptol),I)


import ExtendableSparse: @makefrommatrix, AbstractPreconditioner, update!


# Deprecated from here
warned=false
mutable struct ILUTPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::IncompleteLU.ILUFactorization
    droptol::Float64
    function ExtendableSparse.ILUTPreconditioner(; droptol = 1.0e-3)
        global warned
        if !warned
            @warn "ILUTPreconditioner is deprecated. Use LinearSolve with `precs=ILUTPreconBuilder()` instead"
            warned=true
        end
        p = new()
        p.droptol = droptol
        p
    end
end


@eval begin
    @makefrommatrix ExtendableSparse.ILUTPreconditioner
end

function update!(precon::ILUTPreconditioner)
    A = precon.A
    @inbounds flush!(A)
    precon.factorization = IncompleteLU.ilu(A.cscmatrix; τ = precon.droptol)
end

end

