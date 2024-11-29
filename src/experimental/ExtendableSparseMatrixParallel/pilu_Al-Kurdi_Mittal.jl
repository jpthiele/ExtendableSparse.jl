#module PILUAM
#using Base.Threads
#using LinearAlgebra, SparseArrays

import LinearAlgebra.ldiv!, LinearAlgebra.\, SparseArrays.nnz

#@info "PILUAM"

mutable struct PILUAMPrecon{T, N}

    diag::AbstractVector
    nzval::AbstractVector
    A::AbstractMatrix
    start::AbstractVector
    nt::Integer
    depth::Integer

end

function use_vector_par(n, nt, Ti)
    point = [Vector{Ti}(undef, n) for tid in 1:nt]
    @threads for tid in 1:nt
        point[tid] = zeros(Ti, n)
    end
    return point
end

function compute_lu!(nzval, point, j0, j1, tid, rowval, colptr, diag, Ti)
    for j in j0:(j1 - 1)
        for v in colptr[j]:(colptr[j + 1] - 1)
            point[tid][rowval[v]] = v
        end

        for v in colptr[j]:(diag[j] - 1)
            i = rowval[v]
            for w in (diag[i] + 1):(colptr[i + 1] - 1)
                k = point[tid][rowval[w]]
                if k > 0
                    nzval[k] -= nzval[v] * nzval[w]
                end
            end
        end

        for v in (diag[j] + 1):(colptr[j + 1] - 1)
            nzval[v] /= nzval[diag[j]]
        end

        for v in colptr[j]:(colptr[j + 1] - 1)
            point[tid][rowval[v]] = zero(Ti)
        end
    end
    return
end

function piluAM!(ILU::PILUAMPrecon{Tv, Ti}, A::ExtendableSparseMatrixParallel{Tv, Ti}) where {Tv, Ti <: Integer}
    #@info "piluAM!"
    diag = ILU.diag
    nzval = ILU.nzval
    ILU.A = A
    start = ILU.start

    ILU.nt = A.nt
    nt = A.nt

    ILU.depth = A.depth
    depth = A.depth


    colptr = A.cscmatrix.colptr
    rowval = A.cscmatrix.rowval
    n = A.cscmatrix.n # number of columns
    diag = Vector{Ti}(undef, n)
    nzval = Vector{Tv}(undef, length(rowval)) #copy(A.nzval)
    point = use_vector_par(n, A.nt, Int32)


    @threads for tid in 1:(depth * nt + 1)
        for j in start[tid]:(start[tid + 1] - 1)
            for v in colptr[j]:(colptr[j + 1] - 1)
                nzval[v] = A.cscmatrix.nzval[v]
                if rowval[v] == j
                    diag[j] = v
                end
                #elseif rowval[v]
            end
        end
    end

    for level in 1:depth
        @threads for tid in 1:nt
            for j in start[(level - 1) * nt + tid]:(start[(level - 1) * nt + tid + 1] - 1)
                for v in colptr[j]:(colptr[j + 1] - 1)
                    point[tid][rowval[v]] = v
                end

                for v in colptr[j]:(diag[j] - 1)
                    i = rowval[v]
                    for w in (diag[i] + 1):(colptr[i + 1] - 1)
                        k = point[tid][rowval[w]]
                        if k > 0
                            nzval[k] -= nzval[v] * nzval[w]
                        end
                    end
                end

                for v in (diag[j] + 1):(colptr[j + 1] - 1)
                    nzval[v] /= nzval[diag[j]]
                end

                for v in colptr[j]:(colptr[j + 1] - 1)
                    point[tid][rowval[v]] = zero(Ti)
                end
            end
        end
    end

    #point = zeros(Ti, n) #Vector{Ti}(undef, n)
    for j in start[depth * nt + 1]:(start[depth * nt + 2] - 1)
        for v in colptr[j]:(colptr[j + 1] - 1)
            point[1][rowval[v]] = v
        end

        for v in colptr[j]:(diag[j] - 1)
            i = rowval[v]
            for w in (diag[i] + 1):(colptr[i + 1] - 1)
                k = point[1][rowval[w]]
                if k > 0
                    nzval[k] -= nzval[v] * nzval[w]
                end
            end
        end

        for v in (diag[j] + 1):(colptr[j + 1] - 1)
            nzval[v] /= nzval[diag[j]]
        end

        for v in colptr[j]:(colptr[j + 1] - 1)
            point[1][rowval[v]] = zero(Ti)
        end
    end

    return
end

function piluAM(A::ExtendableSparseMatrixParallel{Tv, Ti}) where {Tv, Ti <: Integer}
    start = A.start
    nt = A.nt
    depth = A.depth

    colptr = A.cscmatrix.colptr
    rowval = A.cscmatrix.rowval
    nzval = Vector{Tv}(undef, length(rowval)) #copy(A.nzval)
    n = A.cscmatrix.n # number of columns
    diag = Vector{Ti}(undef, n)
    point = use_vector_par(n, A.nt, Int32)

    # find diagonal entries
    #

    #=
	for j=1:n
		for v=colptr[j]:colptr[j+1]-1
			nzval[v] = A.cscmatrix.nzval[v]
			if rowval[v] == j
				diag[j] = v
				#break
			end
			#elseif rowval[v] 
		end
	end
	=#


    @threads for tid in 1:(depth * nt + 1)
        for j in start[tid]:(start[tid + 1] - 1)
            for v in colptr[j]:(colptr[j + 1] - 1)
                nzval[v] = A.cscmatrix.nzval[v]
                if rowval[v] == j
                    diag[j] = v
                end
                #elseif rowval[v]
            end
        end
    end


    #@info diag[1:20]'
    #@info diag[end-20:end]'

    for level in 1:depth
        @threads for tid in 1:nt
            for j in start[(level - 1) * nt + tid]:(start[(level - 1) * nt + tid + 1] - 1)
                for v in colptr[j]:(colptr[j + 1] - 1)
                    point[tid][rowval[v]] = v
                end

                for v in colptr[j]:(diag[j] - 1)
                    i = rowval[v]
                    for w in (diag[i] + 1):(colptr[i + 1] - 1)
                        k = point[tid][rowval[w]]
                        if k > 0
                            nzval[k] -= nzval[v] * nzval[w]
                        end
                    end
                end

                for v in (diag[j] + 1):(colptr[j + 1] - 1)
                    nzval[v] /= nzval[diag[j]]
                end

                for v in colptr[j]:(colptr[j + 1] - 1)
                    point[tid][rowval[v]] = zero(Ti)
                end
            end
        end
    end

    #point = zeros(Ti, n) #Vector{Ti}(undef, n)
    for j in start[depth * nt + 1]:(start[depth * nt + 2] - 1)
        for v in colptr[j]:(colptr[j + 1] - 1)
            point[1][rowval[v]] = v
        end

        for v in colptr[j]:(diag[j] - 1)
            i = rowval[v]
            for w in (diag[i] + 1):(colptr[i + 1] - 1)
                k = point[1][rowval[w]]
                if k > 0
                    nzval[k] -= nzval[v] * nzval[w]
                end
            end
        end

        for v in (diag[j] + 1):(colptr[j + 1] - 1)
            nzval[v] /= nzval[diag[j]]
        end

        for v in colptr[j]:(colptr[j + 1] - 1)
            point[1][rowval[v]] = zero(Ti)
        end
    end

    #nzval, diag
    return PILUAMPrecon{Tv, Ti}(diag, nzval, A.cscmatrix, start, nt, depth)
end

function forward_subst_old!(y, v, nzval, diag, start, nt, depth, A)
    #@info "pfso, $(sum(nzval)), $(sum(nzval.^2)), $(sum(diag)), $(A[1,1])"
    #@info "fwo"
    n = A.n
    colptr = A.colptr
    rowval = A.rowval

    y .= 0

    for level in 1:depth
        @threads for tid in 1:nt
            @inbounds for j in start[(level - 1) * nt + tid]:(start[(level - 1) * nt + tid + 1] - 1)
                y[j] += v[j]
                for v in (diag[j] + 1):(colptr[j + 1] - 1)
                    y[rowval[v]] -= nzval[v] * y[j]
                end
            end
        end
    end

    return @inbounds for j in start[depth * nt + 1]:(start[depth * nt + 2] - 1)
        y[j] += v[j]
        for v in (diag[j] + 1):(colptr[j + 1] - 1)
            y[rowval[v]] -= nzval[v] * y[j]
        end
    end

end


function backward_subst_old!(x, y, nzval, diag, start, nt, depth, A)
    #@info "pbso, $(sum(nzval)), $(sum(nzval.^2)), $(sum(diag)), $(A[1,1])"

    #@info "bwo"
    n = A.n
    colptr = A.colptr
    rowval = A.rowval
    #wrk = copy(y)


    @inbounds for j in (start[depth * nt + 2] - 1):-1:start[depth * nt + 1]
        x[j] = y[j] / nzval[diag[j]]

        for i in colptr[j]:(diag[j] - 1)
            y[rowval[i]] -= nzval[i] * x[j]
        end

    end

    for level in depth:-1:1
        @threads for tid in 1:nt
            @inbounds for j in (start[(level - 1) * nt + tid + 1] - 1):-1:start[(level - 1) * nt + tid]
                x[j] = y[j] / nzval[diag[j]]
                for i in colptr[j]:(diag[j] - 1)
                    y[rowval[i]] -= nzval[i] * x[j]
                end
            end
        end
    end

    return
end


function ldiv!(x, ILU::PILUAMPrecon, b)
    #@info "piluam ldiv 1"
    nzval = ILU.nzval
    diag = ILU.diag
    A = ILU.A
    start = ILU.start
    nt = ILU.nt
    depth = ILU.depth
    y = copy(b)
    #forward_subst!(y, b, ILU)
    forward_subst_old!(y, b, nzval, diag, start, nt, depth, A)
    backward_subst_old!(x, y, nzval, diag, start, nt, depth, A)
    #@info "PILUAM:", b[1], y[1], x[1], maximum(abs.(b-A*x)), nnz(A) #, A[10,10]
    #@info "PILUAM:", maximum(abs.(b-A*x)), b[1], x[1], maximum(abs.(b)), maximum(abs.(x))
    return x
end

function ldiv!(ILU::PILUAMPrecon, b)
    #@info "piluam ldiv 2"
    nzval = ILU.nzval
    diag = ILU.diag
    A = ILU.A
    start = ILU.start
    nt = ILU.nt
    depth = ILU.depth
    y = copy(b)
    #forward_subst!(y, b, ILU)
    forward_subst_old!(y, b, nzval, diag, start, nt, depth, A)
    backward_subst_old!(b, y, nzval, diag, start, nt, depth, A)
    return b
end

function \(ilu::PILUAMPrecon{T, N}, b) where {T, N <: Integer}
    x = copy(b)
    ldiv!(x, ilu, b)
    return x
end

function nnz(ilu::PILUAMPrecon{T, N}) where {T, N <: Integer}
    return length(ilu.nzval)
end

#end
