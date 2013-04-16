% Mathematically Informed Automated Linear Algebra
% Matthew Rocklin
% April 21st, 2013

Who am I?
---------

digraph
{
    "Applied Math" -> Computation [label="High Level Compilers"];
    "Hardware" -> Computation [label="Distributed Systems"];
}


Uncertainty Propagation via Derivatives
---------------------------------------

Uncertainty Propagation via Derivatives
---------------------------------------

Uncertainty Propagation via Derivatives
---------------------------------------

Argument for High Level Languages 
---------------------------------

Physical process, Time stepping method, Derivatives, Matrices

Run on CPU, GPU, ....


Argument for High Level Compilers
---------------------------------

MatLab Ordering Problem


SciComp example
---------------

[http://scicomp.stackexchange.com/questions/74/symbolic-software-packages-for-matrix-expressions/](http://scicomp.stackexchange.com/questions/74/symbolic-software-packages-for-matrix-expressions/)


Background and Related Work
---------------------------

*   BLAS/LAPACK
*   ATLAS - Autotunes for architecture (algorithm selection, blocksizes, ...)
*   FLAME - Formal Linear Algebra Methods Environment
*   Tensor Contraction Engine for Chemistry
*   Bientinesi and Fabregat
*   Theano - Tensor compiler Python -> Python/C/CUDA

*   Trillinos - shared ideals - high-level, separable scientific software 


Linear Regression - Math
------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$


\begin{figure}[htbp]
\centering
\includegraphics[width=.4\textwidth]{images/linregress-xy}
\end{figure}

Linear Regression - Python/MatLab
---------------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$

Python

    beta = (X.T*X).I * X.T*y

MatLab

    beta = inv(X'*X) * X'*y


Linear Regression - Python/MatLab
---------------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$

Python

    beta = solve(X.T*X, X.T*y)

MatLab

    beta = X'*X \ X'*y


Linear Regression - Python/MatLab
---------------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$

Python

    beta = spd_solve(X.T*X, X.T*y)

MatLab

    beta = (X'*X) \ (X'*y)


BLAS/LAPACK
-----------

Numeric libraries for dense linear algebra

>*  `DGEMM` - **D**ouble precision **GE**neral **M**atrix **M**ultiply -- $\alpha A B + \beta C$
    *   `SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)`

>*  `DSYMM` - **D**ouble precision **SY**mmetric **M**atrix **M**ultiply -- $\alpha A B + \beta C$
    *   `SUBROUTINE DSYMM(SIDE,UPLO,M,N,ALPHA,A,LDA,B,LDB,BETA,C,LDC)`

>*  ...

>*  `DPOSV` - **D**ouble symmetric **PO**sitive definite matrix **S**ol**V**e  -- $A^{-1}y$
    *   `SUBROUTINE DPOSV( UPLO, N, NRHS, A, LDA, B, LDB, INFO )`


Computation
-----------

**Given**:

    (X.T*X).I*X.T*y
    full_rank(X)

**Produce**:

![](images/hat-comp.pdf)


Necessary Definitions
---------------------

**Language**: Multiply, addition, inverse, transpose, trace, determinant, blocks, etc...

    X = MatrixSymbol('X', n, n)
    y = MatrixSymbol('y', n, 1)
    beta = (X.T*X).I * X.T*y              X.I*X -> Identity

**Predicates**: symmetric, positive_definite, full_rank, orthogonal, triangular, etc....

    fullrank(X)                           fullrank(X) -> positive_definite(X.T*X)

**Computations**:

    class SYMM(BLAS):
        inputs    = [alpha, A, B, beta, C]
        outputs   = [alpha*A*B + beta*C]
        condition = symmetric(A) or symmetric(B)
        inplace   = {0: 4}
        fortran   = ....

Necessary Definitions
---------------------

**Language**: Multiply, addition, inverse, transpose, trace, determinant, blocks, etc...

    X = MatrixSymbol('X', n, n)
    y = MatrixSymbol('y', n, 1)
    beta = (X.T*X).I * X.T*y              X.I*X -> Identity

**Predicates**: symmetric, positive_definite, full_rank, orthogonal, triangular, etc....

    fullrank(X)                           fullrank(X) -> positive_definite(X.T*X)

**Computations**:

\begin{figure}[htbp]
\centering
\includegraphics[width=.5\textwidth]{images/symm}
\end{figure}

Computation
-----------

\begin{figure}[htbp]
\centering
\includegraphics<1->[width=.24\textwidth]{images/hat0}
\includegraphics<2->[width=.24\textwidth]{images/hat1}
\includegraphics<3->[width=.24\textwidth]{images/hat2}
\includegraphics<4->[width=.24\textwidth]{images/hat3}
\end{figure}


User Experience
---------------

~~~~~~~~Python
X = MatrixSymbol('X', n, m)
y = MatrixSymbol('y', n, 1)

inputs  = [X, y]
outputs = [(X.T*X).I*X.T*y]
facts   = Q.fullrank(X)

f = f2py(next(compile(inputs, outputs, facts)))
~~~~~~~~~

\hrule

~~~~~~~~Fortran
subroutine f(X, y, var_7, m, n)
implicit none

integer, intent(in) :: m
integer, intent(in) :: n
real*8, intent(in) :: y(n)          !  y
real*8, intent(in) :: X(n, m)       !  X
real*8, intent(out) :: var_7(m)     !  0 -> X'*y -> (X'*X)^-1*X'*y
real*8 :: var_8(m, m)               !  0 -> X'*X
integer :: INFO                     !  INFO

call dgemm('N', 'N', m, 1, n, 1.0, X, n, y, n, 0.0, var_7, m)
call dgemm('N', 'N', m, m, n, 1.0, X, n, X, n, 0.0, var_8, m)
call dposv('U', m, 1, var_8, m, var_7, m, INFO)

RETURN
END
~~~~~~~~~

Multiple Results
----------------

\begin{figure}[htbp]
\centering
\includegraphics<1->[width=.24\textwidth]{images/hat0}
\includegraphics<1->[width=.24\textwidth]{images/hat1}
\includegraphics<1->[width=.24\textwidth]{images/hat2}
\includegraphics<1->[width=.24\textwidth]{images/hat3}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics<2>[width=.23\textwidth]{images/hat_gesv1}
\includegraphics<2>[width=.23\textwidth]{images/hat_gesv2}
\includegraphics<2>[width=.53\textwidth]{images/hat_gesv3}
\end{figure}


Kalman Filter
-------------

~~~~~~~~~~Python
newmu   = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
newSigma= Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma

assumptions = [positive_definite(Sigma), symmetric(Sigma), 
               positive_definite(R), symmetric(R), fullrank(H)]
f = fortran_function([mu, Sigma, H, R, data], [newmu, newSigma], *assumptions)
~~~~~~~~~~

\begin{figure}[htbp]
\centering
\includegraphics[width=.7\textwidth]{images/kalman-math}
\end{figure}


Picture
-------

\begin{tikzpicture}
    [scale=.8,auto=right,every node/.style={rectangle,fill=white!20}]

    \node (math) at (10,10) {Mathematics \\ (Inverse, transpose, positive-definite)};
    \node (computation) at (10,1)  {Computation \\ (GEMM, POSV)};
    \node (pl) at (5,5)  {Programming Languages \\ (Graph covering)};
    \node (connection) at (10,5)  {};

    \foreach \from/\to in
    {math/connection, pl/connection, computation/connection}
    \draw (\from) -- (\to);

\end{tikzpicture}


End
---

Thanks!
