# Integration with LinearSolve.jl

Starting with version 0.9.6, ExtendableSparse is compatible
with [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl).
Since version 0.9.7, this is facilitated via the
AbstractSparseMatrixCSC interface. Since version 1.6, 
preconditioner constructors can be passed to iterative solvers via the [`precs`
keyword argument](https://docs.sciml.ai/LinearSolve/stable/basics/Preconditioners/#prec).

We can create a test problem and solve it with the `\` operator.

```@example
using ExtendableSparse # hide
A = fdrand(10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(1000)
b = A * x
y = A \ b
sum(y)
```

The same problem can be solved via `LinearSolve.jl`:

```@example
using ExtendableSparse # hide
using LinearSolve # hide
A = fdrand(10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(1000)
b = A * x
y = solve(LinearProblem(A, b), SparspakFactorization()).u
sum(y)
```


## Preconditioning

LinearSolve allows to pass preconditioner constructors via the `precs` keyword
to the iterative solver specification.

```@example
using ExtendableSparse # hide
using LinearSolve # hide
using ExtendableSparse: ILUZeroPreconBuilder
A = fdrand(10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(1000)
b = A * x
y = LinearSolve.solve(LinearProblem(A, b), 
                      KrylovJL_CG(; precs=ILUZeroPreconBuilder())).u
sum(y)
```

ExtendableSparse provides constructors for preconditioners wich can be used as `precs`.
These generally return a tuple `(Pl,I)` of a left preconditioner and a trivial right preconditioner.

ExtendableSparse has a number of package extensions which construct preconditioners
from some other packages. In the future, these packages may provide this functionality on their own.

```@docs
ExtendableSparse.ILUZeroPreconBuilder
ExtendableSparse.ILUTPreconBuilder
ExtendableSparse.SmoothedAggregationPreconBuilder
ExtendableSparse.RugeStubenPreconBuilder
```

In addition, ExtendableSparse implements some preconditioners:

```@docs
ExtendableSparse.JacobiPreconBuilder
```

LU factorizations of  matrices from previous iteration steps may be good
preconditioners for Krylov solvers called during a nonlinear solve via
Newton's method. For this purpose, ExtendableSparse provides preconditioner constructors
which wrap sparse LU factorizations.

```@docs
ExtendableSparse.UMFPACKPreconBuilder
ExtendableSparse.SparspakPreconBuilder
```

Block preconditioner constructors are provided as well

```@docs;  canonical=false
ExtendableSparse.BlockPreconBuilder
```

## Deprecated API
Passing a preconditioner via the `Pl` or `Pr` keyword arguments
will be deprecated in LinearSolve. ExtendableSparse used to
export a number of wrappers for preconditioners from other packages
for this purpose. This approach is deprecated as of v1.6 and will be removed
with v2.0.

```@example
using ExtendableSparse # hide
using LinearSolve # hide
using SparseArrays # hide
using ILUZero
A = fdrand(10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(1000)
b = A * x
y = LinearSolve.solve(LinearProblem(A, b), KrylovJL_CG();
                      Pl = ILUZero.ilu0(SparseMatrixCSC(A))).u
sum(y)
```

