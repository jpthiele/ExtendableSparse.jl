struct UMFPACKPrecs end
(::UMFPACKPrecs)(A::AbstractSparseMatrixCSC,p)=(SparseArrays.UMFPACK.UmfpackLU(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),nonzeros(A))),I)

struct SparspakPrecs end
(::SparspakPrecs)(A::AbstractSparseMatrixCSC,p)=(sparspaklu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),nonzeros(A))),I)


