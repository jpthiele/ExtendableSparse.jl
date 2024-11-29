mutable struct _ILU0Preconditioner{Tv, Ti}
    cscmatrix::SparseMatrixCSC{Tv, Ti}
    xdiag::Vector{Tv}
    idiag::Vector{Ti}
end


function ilu0(cscmatrix::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    colptr = cscmatrix.colptr
    rowval = cscmatrix.rowval
    nzval = cscmatrix.nzval
    n = cscmatrix.n

    # Find main diagonal index and
    # copy main diagonal values
    idiag = Vector{Ti}(undef, n)
    @inbounds for j in 1:n
        @inbounds for k in colptr[j]:(colptr[j + 1] - 1)
            i = rowval[k]
            if i == j
                idiag[j] = k
                break
            end
        end
    end

    xdiag = Vector{Tv}(undef, n)
    @inbounds for j in 1:n
        xdiag[j] = one(Tv) / nzval[idiag[j]]
        @inbounds for k in (idiag[j] + 1):(colptr[j + 1] - 1)
            i = rowval[k]
            for l in colptr[i]:(colptr[i + 1] - 1)
                if rowval[l] == j
                    xdiag[i] -= nzval[l] * xdiag[j] * nzval[k]
                    break
                end
            end
        end
    end
    return _ILU0Preconditioner(cscmatrix, xdiag, idiag)
end

function ilu0!(p::_ILU0Preconditioner{Tv, Ti}, cscmatrix::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    colptr = cscmatrix.colptr
    rowval = cscmatrix.rowval
    nzval = cscmatrix.nzval
    n = cscmatrix.n
    idiag = p.idiag
    xdiag = p.xdiag

    @inbounds for j in 1:n
        xdiag[j] = one(Tv) / nzval[idiag[j]]
        @inbounds for k in (idiag[j] + 1):(colptr[j + 1] - 1)
            i = rowval[k]
            for l in colptr[i]:(colptr[i + 1] - 1)
                if rowval[l] == j
                    xdiag[i] -= nzval[l] * xdiag[j] * nzval[k]
                    break
                end
            end
        end
    end
    p.cscmatrix = cscmatrix
    return p
end


function LinearAlgebra.ldiv!(u, p::_ILU0Preconditioner{Tv, Ti}, v) where {Tv, Ti}
    colptr = p.cscmatrix.colptr
    rowval = p.cscmatrix.rowval
    nzval = p.cscmatrix.nzval
    n = p.cscmatrix.n
    idiag = p.idiag
    xdiag = p.xdiag

    for j in 1:n
        u[j] = xdiag[j] * v[j]
    end

    for j in n:-1:1
        for k in (idiag[j] + 1):(colptr[j + 1] - 1)
            i = rowval[k]
            u[i] -= xdiag[i] * nzval[k] * u[j]
        end
    end

    for j in 1:n
        for k in colptr[j]:(idiag[j] - 1)
            i = rowval[k]
            u[i] -= xdiag[i] * nzval[k] * u[j]
        end
    end
    return u
end

function LinearAlgebra.ldiv!(p::_ILU0Preconditioner{Tv, Ti}, v) where {Tv, Ti}
    return ldiv!(copy(v), p, v)
end


mutable struct ILU0Preconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::_ILU0Preconditioner
    phash::UInt64
    function ILU0Preconditioner()
        p = new()
        p.phash = 0
        return p
    end
end

"""
```
ILU0Preconditioner()
ILU0Preconditioner(matrix)
```

Incomplete LU preconditioner with zero fill-in, without modification of off-diagonal entries, so it delivers
slower convergende than  [`ILUZeroPreconditioner`](@ref).
"""
function ILU0Preconditioner end


function update!(p::ILU0Preconditioner)
    flush!(p.A)
    Tv = eltype(p.A)
    if p.A.phash != p.phash
        p.factorization = ilu0(p.A.cscmatrix)
        p.phash = p.A.phash
    else
        ilu0!(p.factorization, p.A.cscmatrix)
    end
    return p
end

allow_views(::ILU0Preconditioner) = true
allow_views(::Type{ILU0Preconditioner}) = true
