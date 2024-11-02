module test_linearsolve

using Test
using ExtendableSparse
using SparseArrays
using LinearAlgebra
using LinearSolve
using ForwardDiff
using MultiFloats

using AMGCLWrap
import Pardiso

f64(x::ForwardDiff.Dual{T}) where {T} = Float64(ForwardDiff.value(x))
f64(x::Number) = Float64(x)
const Dual64 = ForwardDiff.Dual{Float64, Float64, 1}

function test_ls1(T, k, l, m; linsolver = SparspakFactorization())
    A = fdrand(k, l, m; matrixtype = ExtendableSparseMatrix)
    b = A*ones(k * l * m)
    x0 = A \ b
    p = LinearProblem(T.(A), T.(b))
    x1 = solve(p, linsolver, abstol=1.0e-12)
    x0 ≈ x1
end

function test_ls2(T, k, l, m; linsolver = SparspakFactorization())
    A = fdrand(T, k, l, m; rand = () -> 1, matrixtype = ExtendableSparseMatrix)
    b = T.(rand(k * l * m))
    p = LinearProblem(A, b)
    x0 = solve(p, linsolver)
    cache = x0.cache
    x0=copy(x0)
    nonzeros(A).-=1.0e-4
    for i = 1:k*l*m
        A[i, i] += 1.0e-4
    end

    reinit!(cache; A, reuse_precs=true)
    x1 = solve!(cache, linsolver)
    all(x0 .< x1)
end



for T in [Float32, Float64, Float64x1, Float64x2, Dual64]
    println("$T:")
    @test test_ls1(T, 10, 10, 10, linsolver = SparspakFactorization())
    @test test_ls1(T, 25, 40, 1, linsolver = SparspakFactorization())
    @test test_ls1(T, 100, 1, 1, linsolver = SparspakFactorization())

    @test test_ls2(T, 10, 10, 10, linsolver = SparspakFactorization())
    @test test_ls2(T, 25, 40, 1, linsolver = SparspakFactorization())
    @test test_ls2(T, 100, 1, 1, linsolver = SparspakFactorization())

end


for factorization in [UMFPACKFactorization(),
                      KLUFactorization(reuse_symbolic=false),
                      MKLPardisoFactorize()]
    println("$factorization:")
    @test test_ls1(Float64, 10, 10, 10, linsolver = factorization)
    @test test_ls1(Float64, 25, 40, 1, linsolver = factorization)
    @test test_ls1(Float64, 100, 1, 1, linsolver = factorization)

    @test test_ls2(Float64, 10, 10, 10, linsolver = factorization)
    @test test_ls2(Float64, 25, 40, 1, linsolver = factorization)
    @test test_ls2(Float64, 100, 1, 1, linsolver = factorization)
end



for iteration in [
    KrylovJL_GMRES(precs=AMGCLWrap.AMGPreconBuilder()),
    KrylovJL_GMRES(precs=AMGCLWrap.RLXPreconBuilder())

                  ]
    println("$iteration:")
    @test test_ls1(Float64, 10, 10, 10, linsolver = iteration)
    @test test_ls1(Float64, 25, 40, 1, linsolver = iteration)
    @test test_ls1(Float64, 100, 1, 1, linsolver = iteration)

    @test test_ls2(Float64, 10, 10, 10, linsolver = iteration)
    @test test_ls2(Float64, 25, 40, 1, linsolver = iteration)
    @test test_ls2(Float64, 100, 1, 1, linsolver = iteration)
end



function mainprecs(;n=100)
    A=fdrand(n,n)
    partitioning=A->[1:2:size(A,1), 2:2:size(A,1)]
    sol0=ones(n^2)
    b=A*ones(n^2);

    precs=EquationBlockPrecs(;precs=UMFPACKPrecs(), partitioning)
    @info precs
    p=LinearProblem(A,b)
    sol=solve(p, KrylovJL_CG(;precs))
    @test sol≈sol0

    precs=EquationBlockPrecs(;precs=SparspakPrecs(), partitioning)
    @info precs
    p=LinearProblem(A,b)
    sol=solve(p, KrylovJL_CG(;precs))
    @test sol≈sol0

    precs=EquationBlockPrecs(;precs=AMGCLWrap.AMGPreconBuilder(), partitioning)
    @info precs
    p=LinearProblem(A,b)
    sol=solve(p, KrylovJL_CG(;precs))
    @test sol≈sol0

    precs=EquationBlockPrecs(;precs=AMGCLWrap.RLXPreconBuilder(), partitioning)
    @info precs
    p=LinearProblem(A,b)
    sol=solve(p, KrylovJL_CG(;precs))
    @test sol≈sol0

end
mainprecs()


end
