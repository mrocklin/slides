% Matrix Expressions and BLAS/LAPACK
% Matthew Rocklin
% June 27th, 2013

Need For High Level Compilers
=============================

Math is Important
-----------------

~~~~~~~~~C
// A: Naive             // B: Programmer    // C: Mathematician
int fact(int n){        int fact(int n){    int fact(int n){
    if (n == 0)             int prod = n;       // n! = Gamma(n+1)   
        return 1;           while(n--)          return lround(exp(lgamma(n+1)));
    return n*fact(n-1);         prod *= n;  }
}                           return prod;
                        }
~~~~~~~~~

\vspace{10em}
Evan Miller [http://www.evanmiller.org/mathematical-hacker.html](http://www.evanmiller.org/mathematical-hacker.html)

---------------------------------

    x = matrix(ones(10000, 1))

    x*x.T*x              Elapsed time is    ?     seconds.
    (x*x.T)*x            Elapsed time is 0.337711 seconds.
    x*(x.T*x)            Elapsed time is 0.000956 seconds.

\begin{figure}[htbp]
\centering
\includegraphics[width=.6\textwidth]{images/xxtrans}
\end{figure}


---------------------------------

If $\mathbf A$ is symmetric positive-definite and $\mathbf B$ is orthogonal:

**Question**: is $\mathbf B \cdot\mathbf A \cdot\mathbf B^\top$ symmetric and
positive-definite? 

**Answer**: Yes.

**Question**: Could a computer have told us this?

**Answer**: Probably.

\vspace{5em}

[http://scicomp.stackexchange.com/questions/74/symbolic-software-packages-for-matrix-expressions/](http://scicomp.stackexchange.com/questions/74/symbolic-software-packages-for-matrix-expressions/)

---------------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$

Python/NumPy

    beta = (X.T*X).I * X.T*y

MatLab

    beta = inv(X'*X) * X'*y


---------------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$

Python/NumPy

    beta = solve(X.T*X, X.T*y)

MatLab

    beta = X'*X \ X'*y


---------------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$

Python/NumPy

    beta = solve(X.T*X, X.T*y, sym_pos=True)

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


-------------------------------

    Naive  :     (X.T*X).I * X.T*y         

    Expert :     solve(X.T*X, X.T*y, sym_pos=True)


\begin{figure}[htbp]
\centering
\includegraphics[width=.8\textwidth]{images/hat-comp}
\end{figure}

~~~~~~~~~~~Fortran
subroutine least_squares(X, y)
 ...
endsubroutine
~~~~~~~~~~~


Matrix Expressions and SymPy
============================

SymPy Expressions
-----------------

Operators (`Add, log, exp, sin, integral, derivative`, ...) are Python classes

Terms ( `3, x, log(3*x), integral(x**2)`, ...) are Python objects


\begin{columns}
  \begin{column}{0.5\textwidth}
    \lstinputlisting{expr.py}
  \end{column}

  \begin{column}{0.5\textwidth}
    \begin{figure}[htbp]
    \centering
    \includegraphics<1>[width=.7\textwidth, totalheight=.6\textheight, keepaspectratio]{images/expr}
    \includegraphics<2>[width=.7\textwidth, totalheight=.6\textheight, keepaspectratio]{images/sexpr2}
    \end{figure}
  \end{column}
\end{columns}


Inference
---------

~~~~~~~~~~~~Python
>>> x = Symbol('x')
>>> y = Symbol('y')

>>> facts = Q.real(x) & Q.positive(y)
>>> query = Q.positive(x**2 + y)

>>> ask(query, facts)
True
~~~~~~~~~~~~


Matrix Expressions
------------------

~~~~~~~~~~~~Python
>>> X = MatrixSymbol('X', n, m)
>>> y = MatrixSymbol('y', n, 1)

>>> beta = (X.T*X).I * X.T*y
~~~~~~~~~~~~

\begin{figure}[htbp]
\centering
\includegraphics[width=.4\textwidth]{images/matrixexpr}
\end{figure}


Matrix Inference
----------------

If $\mathbf A$ is symmetric positive-definite and $\mathbf B$ is orthogonal:

**Question**: is $\mathbf B \cdot\mathbf A \cdot\mathbf B^\top$ symmetric and
positive-definite? 

**Answer**: Yes.

**Question**: Could a computer have told us this?

**Answer**: Probably.


\vspace{1em}
\hrule

    sympy.matrices.expressions

~~~~~~~~Python
>>> A = MatrixSymbol('A', n, n)
>>> B = MatrixSymbol('B', n, n)

>>> facts = Q.symmetric(A) & Q.positive_definite(A) & Q.orthogonal(B)
>>> query = Q.symmetric(B*A*B.T) & Q.positive_definite(B*A*B.T)

>>> ask(query, facts)
True
~~~~~~~~


Computations
============

-----------

Numeric libraries for dense linear algebra

*  `DGEMM` - **D**ouble precision **GE**neral **M**atrix **M**ultiply -- $\alpha A B + \beta C$
    *   `SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)`

*  `DSYMM` - **D**ouble precision **SY**mmetric **M**atrix **M**ultiply -- $\alpha A B + \beta C$
    *   `SUBROUTINE DSYMM(SIDE,UPLO,M,N,ALPHA,A,LDA,B,LDB,BETA,C,LDC)`

*  ...

*  `DPOSV` - **D**ouble symmetric **PO**sitive definite matrix **S**ol**V**e  -- $A^{-1}y$
    *   `SUBROUTINE DPOSV( UPLO, N, NRHS, A, LDA, B, LDB, INFO )`


-----------------------

[http://github.com/mrocklin/computations](http://github.com/mrocklin/computations) 
\vspace{-2em}
\begin{figure}[htbp]
\centering
\includegraphics<1>[width=.8\textwidth]{images/linregress}
\includegraphics<2>[width=\textwidth]{images/ilinregress}
\end{figure}
\vspace{-4em}

~~~~~~~~~~~~~~~Python
comp = (GEMM(1, X.T, X, 0, 0) 
      + GEMM(1, X.T, y, 0, 0)
      + POSV(X.T*X, X.T*y))

class GEMM(Computation):
    """ Genreral Matrix Multiply """
    inputs    = [alpha, A, B, beta, C]
    outputs   = [alpha*A*B + beta*C]
    inplace   = {0: 4}
    fortran   = ....

class POSV(Computation):
    """ Symmetric Positive Definite Matrix Solve """
    inputs    = [A, y]
    outputs   = [UofCholesky(A), A.I*y]
    inplace   = {0: 0, 1: 1}
    condition = Q.symmetric(A) & Q.positive_definite(A)
    fortran   = ....
~~~~~~~~~~~~~~~

---------------

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

Logic Programming
=================

---------

                                      TODO

---------

*   `http://github.com/mrocklin/term` \newline
    An interface for terms  
    Composable with legacy code via Monkey patching \newline
    Supports pattern matching via Unification

*   `http://github.com/logpy/logpy` \newline
    Implements miniKanren, a logic programming language

*   `http://github.com/logpy/strategies` \newline
    Partially implements Stratego, a control flow programming language
    

Automation
==========

-------


**Have**

    (X.T*X).I*X.T*y
    Q.fullrank(X)

**Want**

    comp = (GEMM(1, X.T, X, 0, 0) 
          + GEMM(1, X.T, y, 0, 0)
          + POSV(X.T*X, X.T*y))

**Accomplish Using Pattern Matching**

~~~~~~~~~~~Python
# Source Expression,  Target Computation,        Condition
(alpha*A*B + beta*C, SYMM(alpha, A, B, beta, C), Q.symmetric(A) | Q.symmetric(B)),
(alpha*A*B + beta*C, GEMM(alpha, A, B, beta, C), True),
(A.I*B,              POSV(A, B),        Q.symmetric(A) & Q.positive_definite(A)),
(A.I*B,              GESV(A, B),        True),
(alpha*A + B,        AXPY(alpha, A, B), True),
~~~~~~~~~~~

Pattern Matching done with LogPy, a composable Logic Programming library

---------

\begin{figure}[htbp]
\centering
\includegraphics<1->[width=.24\textwidth]{images/hat0}
\includegraphics<2->[width=.24\textwidth]{images/hat1}
\includegraphics<3->[width=.24\textwidth]{images/hat2}
\includegraphics<4->[width=.24\textwidth]{images/hat3}
\end{figure}

~~~~~~~~~~~Python
# Source Expression,  Target Computation,        Condition
(alpha*A*B + beta*C, SYMM(alpha, A, B, beta, C), Q.symmetric(A) | Q.symmetric(B)),
(alpha*A*B + beta*C, GEMM(alpha, A, B, beta, C), True),
(A.I*B,              POSV(A, B),        Q.symmetric(A) & Q.positive_definite(A)),
(A.I*B,              GESV(A, B),        True),
(alpha*A + B,        AXPY(alpha, A, B), True),
~~~~~~~~~~~

Kalman Filter
-------------

~~~~~~~~~~Python
newmu       = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
newSigma    = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma

assumptions = [Q.positive_definite(Sigma), Q.symmetric(Sigma), 
               Q.positive_definite(R), Q.symmetric(R), Q.fullrank(H)]

f = fortran_function([mu, Sigma, H, R, data], [newmu, newSigma], *assumptions)
~~~~~~~~~~
\vspace{-1em}

\begin{figure}[htbp]
\centering
\includegraphics[width=.9\textwidth]{images/kalman-math}
\end{figure}


Kalman Filter
-------------

~~~~~~~~~~~Fortran
include [Kalman](kalman.f90)
~~~~~~~~~~~


Software Design
===============

----------

\begin{figure}[htbp]
\centering
\includegraphics<1>[width=\textwidth]{images/not-separation}
\includegraphics<2>[width=\textwidth]{images/separation}
\includegraphics<3>[width=\textwidth]{images/separation-2}
\end{figure}
   

----

~~~~~~~~Python
X = MatrixSymbol('X', n, m)
y = MatrixSymbol('y', n, 1)

inputs  = [X, y]
outputs = [(X.T*X).I*X.T*y]
facts   = fullrank(X)
~~~~~~~~~

\begin{figure}[htbp]
\centering
\includegraphics<1>[width=.9\textwidth]{images/hat-comp}
\includegraphics<2>[width=.9\textwidth]{images/hat-comp-syrk}
\end{figure}


----

\begin{figure}[htbp]
\centering
\includegraphics[width=.9\textwidth]{images/hat-comp-syrk}
\end{figure}

~~~~~~~~~~~Python
class SYRK(Computation):
    """ Symmetric Rank-K Update `alpha X' X + beta Y' """
    inputs  = (alpha, A, beta, D)
    outputs = (alpha * A * A.T + beta * D,)
    inplace  = {0: 3}
    fortran_template = ("call %(fn)s('%(UPLO)s', '%(TRANS)s', %(N)s, %(K)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, "
                        "%(beta)s, %(D)s, %(LDD)s)")
    ...

  (alpha*A*A.T + beta*D, SYRK(alpha, A, beta, D), True),
  (A*A.T,                SYRK(1.0, A, 0.0, 0),    True),
~~~~~~~~~~~~


Separation promotes Comparison and Experimentation
--------------------------------------------------

\begin{figure}[htbp]
\centering
\includegraphics<1>[width=\textwidth]{images/separation-2}
\includegraphics<2>[width=\textwidth]{images/separation-theano}
\end{figure}


Kalman Filter - Theano v. Fortran
---------------------------------

~~~~~~~~~~Python
newmu       = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
newSigma    = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma

assumptions = [Q.positive_definite(Sigma), Q.symmetric(Sigma), 
               Q.positive_definite(R), Q.symmetric(R), Q.fullrank(H)]

f = fortran_function([mu, Sigma, H, R, data], [newmu, newSigma], *assumptions)
~~~~~~~~~~
\vspace{-1em}

\begin{figure}[htbp]
\centering
\includegraphics[width=.9\textwidth]{images/kalman-math}
\end{figure}


Kalman Filter - Theano v. Fortran
---------------------------------

\vspace{-1.5em}
~~~~~~~~~~Python
newmu       = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
newSigma    = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma




f = theano_function( [mu, Sigma, H, R, data], [newmu, newSigma])
~~~~~~~~~~
\vspace{+5em}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{images/kalman-theano}
\end{figure}


Thoughts
--------

*   Modularity is good 
    *   Cater to single-field experts
    *   Eases comparison and evolution 
    *   This project might die but the parts will survive

*   Intermediate Representations are Good
    *   Fortran code doesn't depend on Python
    *   Readability encourages development
    *   Extensibility (lets generate CUDA)

*   Read more! [http://matthewrocklin.com/blog](http://matthewrocklin.com/blog)


\begin{figure}[htbp]
\centering
\includegraphics[width=.7\textwidth]{images/kalman-math}
\end{figure}



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

