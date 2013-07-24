% Modular Generation of Scientific Software
% Matthew Rocklin
% July 26th, 2013


Sales Pitch
===========


Most Talks
----------

You want your programs to run faster

Accelerate computation through sophisticated hardware

This is hard and it's going to get harder

This is actually really hard for most users

Discussion of automating some problem

This Talk
------------------------

You want your programs to run faster

~~Accelerate~~ Reduce computation through sophisticated ~~hardware~~ methods 

This is hard ~~and it's going to get harder~~

This is actually really hard for most users

Discussion of automating some problem 

------------------------

Slide on integration comparing trapezoid rule to simpsons rule to something
exotic

------------------------

~~~~~~~~~C
// A: Naive             // B: Programmer    // C: Mathematician
int fact(int n){        int fact(int n){    int fact(int n){
    if (n == 0)             int prod = 1;       // n! = Gamma(n+1)   
        return 1;           while(n > 1)        return lround(exp(lgamma(n+1)));
    return n*fact(n-1);         prod *= n--;
                            return prod;     
}                       }                   }
~~~~~~~~~

---------------------------------

Most science is done by naive developers at moderate scale

Support mainstream work by automation

---------------------------------

\begin{figure}[htbp]
\centering
\includegraphics<1>[height=\textheight]{images/parallel-programmers}
\includegraphics<2>[height=\textheight]{images/gamma-knowers}
\includegraphics<3>[width=\textwidth]{images/venn-capable-1b}
\includegraphics<4>[width=\textwidth]{images/venn-capable-3b}
\includegraphics<5>[width=\textwidth]{images/venn-capable-4b}
\end{figure}

---------------------------------


*   The distribution of deep expertise is both concentrated and separate \newline
    Demographics of specialist communities

*   Per-application collaboration doesn't scale

*   Software modularity expands the development pool and \newline
    enables reuse across applications

----------------------------------


\begin{figure}[htbp]
\centering
\includegraphics<1>[width=\textwidth]{images/modularity-1}
\includegraphics<2>[width=\textwidth]{images/modularity-2}
\includegraphics<3>[width=\textwidth]{images/modularity-3}
\end{figure}


Automated Linear Algebra
========================

Need for Arrray Compilers
-------------------------

    x = ones(10000, 1)

    x*x'*x              Elapsed time is    ?     seconds.
    (x*x')*x            Elapsed time is 0.337711 seconds.
    x*(x'*x)            Elapsed time is 0.000956 seconds.

\begin{figure}[htbp]
\centering
\includegraphics[width=.6\textwidth]{images/xxtrans}
\end{figure}

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
  call dgemm('N', 'N', m, 1, n, 1.0, X, n, y, n, 0.0, var_7, m)
  call dgemm('N', 'N', m, m, n, 1.0, X, n, X, n, 0.0, var_8, m)
  call dposv('U', m, 1, var_8, m, var_7, m, INFO)
  ...
endsubroutine
~~~~~~~~~~~


Matrix Expressions and Computer Algebra
=======================================

Mathematical Terms
------------------

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

Automation
==========

-------


**Have**

    expr  = (X.T*X).I*X.T*y
    facts = Q.fullrank(X)

**Want**

    comp  = (GEMM(1, X.T, X, 0, 0) 
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
 ...
~~~~~~~~~~~


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


Terms Rewrite Systems
=====================


Pattern Matching
----------------

Pattern:

    a + b

Expression:

    x + 5*y


Pattern Matching
----------------

Pattern:

    a + b

Expression:

    x + 5*y + z

Pattern Matching
----------------

Pattern:

    a + b

Expression:

    (x + 5*y) + z


Pattern Matching
----------------

Pattern:

    a + b

Expression:

    x + (5*y + z)

Pattern Matching
----------------

Pattern:

    a + b

Expression:

    (x + z) + 5*y


Rewrite Rule
------------

$$ |x| \rightarrow x  \;\; \textrm{if} \;\; x \ge 0 $$

                              Abs(x), x ,  x >= 0 

    Source:         Abs(x)

    Target:         x

    Condition:      x >= 0


Before:

~~~~~~~~~~~~Python
def simplify_abs_of_positive(expr):
    if isinstance(expr, Abs) and expr.args[0] >= 0:
        return expr.args[0]
~~~~~~~~~~~~

After:

    patterns = [(Abs(x), x, x >= 0), 
                (sin(x**2) + cos(x**2), 1, True),
                ...
                ]

Rule Coordination
-----------------

Rules:

$$ tan(a) \rightarrow sin(a) / cos(a) $$
$$ sin(a) / cos(a) \rightarrow tan(a) $$
$$ sin^2(a) + cos^2(a) \rightarrow 1 $$
$$ sin^2(a) \rightarrow \frac{1-cos(2a)}{2}$$
$$ cos^2(a) \rightarrow \frac{1+cos(2a)}{2}$$
$$ sin(a) + sin(b) \rightarrow 2 sin(\frac{a+b}{2}) cos(\frac{a+b}{2})$$
$$ ... $$

Input:
    
$$ sin^2(y) + \frac{sin(z)}{cos(z)} + cos^2(y)  $$

Rule Coordination
-----------------

\begin{figure}[htbp]
\centering
\includegraphics<1>[width=\textwidth]{images/search}
\includegraphics<2>[width=\textwidth]{images/search-left}
\includegraphics<3>[width=\textwidth]{images/search-dumb}
\includegraphics<4>[width=\textwidth]{images/search-greedy}
\includegraphics<5>[width=\textwidth]{images/search-continue}
\end{figure}


-----------------

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

-----------------

\begin{figure}[htbp]
\centering
\includegraphics[width=.24\textwidth]{images/hat0}
\includegraphics[width=.24\textwidth]{images/hat1}
\includegraphics[width=.24\textwidth]{images/hat2}
\includegraphics[width=.24\textwidth]{images/hat3}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=.23\textwidth]{images/hat_gesv1}
\includegraphics[width=.23\textwidth]{images/hat_gesv2}
\includegraphics[width=.53\textwidth]{images/hat_gesv3}
\end{figure}


Summary of Project
------------------

    Have:   expr  = (X.T*X).I*X.T*y
            facts =  Q.fullrank(X)

    Make:   comp  = (GEMM(1, X.T, X, 0, 0)
                   + GEMM(1, X.T, y, 0, 0)
                   + POSV(X.T*X, X.T*y))

With modular development by field

*   Pure Mathematics
*   Numerics / Low-level code
*   Term/Graph manipulations


Development
===========

---------------------------------------------------------------

\begin{figure}[htbp]
\centering
\includegraphics<1>[width=\textwidth]{images/not-separation}
\includegraphics<2>[width=\textwidth]{images/separation}
\includegraphics<3>[width=\textwidth]{images/separation-2}
\end{figure}
   

---------------------------------------------------------------

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


---------------------------------------------------------------

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


---------------------------------------------------------------

\begin{figure}[htbp]
\centering
\includegraphics<1>[width=\textwidth]{images/separation-2}
\includegraphics<2>[width=\textwidth]{images/separation-theano}
\end{figure}


Kalman Filter
---------------------------------------------------------------

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
---------------------------------------------------------------

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

Blocking
---------------------------------------------------------------

$$ \begin{bmatrix} A & B \\\\ C & D \end{bmatrix} 
   \begin{bmatrix} E & F \\\\ G & K \end{bmatrix}
   \rightarrow
   \begin{bmatrix} A E + B G & A F + B K \\\\ 
                   C E + D G & C F + D K\end{bmatrix} $$

$$ \begin{bmatrix} A & B \\\\ C & D \end{bmatrix}^{-1}
   \rightarrow
   \begin{bmatrix} 
\left(- B D^{-1} C + A\right)^{-1} & - A^{-1} B \left(- C A^{-1} B + D\right)^{-1} \\\\ 
- \left(- C A^{-1} B + D\right)^{-1} C A^{-1} & \left(- C A^{-1} B + D\right)^{-1}
\end{bmatrix} $$

Blocking
--------------------------------------------------------------

~~~~~~~~~~Python
newmu       = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
newSigma    = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma

assumptions = [Q.positive_definite(Sigma), Q.symmetric(Sigma), 
               Q.positive_definite(R), Q.symmetric(R), Q.fullrank(H)]

blocksizes = {
        Sigma: [(n/2, n/2), (n/2, n/2)],
        H:     [(k/2, k/2), (n/2, n/2)],
        R:     [(k/2, k/2), (k/2, k/2)],
        mu:    [(n/2, n/2), (1,)],
        data:  [(k/2, k/2), (1,)]
        }
blockinputs = [blockcut(i, *blocksizes[i]) for i in inputs]
blockoutputs = [o.subs(dict(zip(inputs, blockinputs))) for o in outputs]
collapsed_outputs = map(block_collapse, blockoutputs)
~~~~~~~~~~

Blocking
---------------------------------------------------------------

\begin{figure}
\centering
\includegraphics[width=\textwidth]{images/fblocked}
\end{figure}


Static Scheduling
---------------------------------------------------------------

Parallelize a simple computation

$$(AB)^{-1} \times (CD)$$

Onto a simple architecture 

\begin{tikzpicture}[every text node part/.style={align=center, circle}, node
distance=2.4cm, semithick]

    \tikzstyle{every node}=[thick,draw=blue!75,fill=blue!5,minimum size=6mm]

    \node  (A) at (5, 5) {CPU-1};
    \node  (B) [right of=A] {CPU-2};

    \foreach \from/\to in
    {A/B}
    \draw (\from) -- (\to);

\end{tikzpicture}

Static Scheduling
---------------------------------------------------------------

Array computation times and bulk communication times are predictable,
particularly on HPC hardware.

\begin{figure}[htbp]
\centering
\includegraphics[width=.45\textwidth]{images/gemm-profile-fortran}
\includegraphics[width=.45\textwidth]{images/communication-time}
\end{figure}


Static Scheduling
---------------------------------------------------------------

Build computation from matrix expression

\begin{figure}[htbp]
\centering
\includegraphics[width=.5\textwidth]{images/ABiCD}
\end{figure}


Static Scheduling
---------------------------------------------------------------

Schedule computation with static schedulers and inject MPI operations:

\begin{figure}[htbp]
\centering
\includegraphics[width=.55\textwidth]{images/ABiCD_0}
\includegraphics[width=.35\textwidth]{images/ABiCD_1}
\end{figure}


Static Scheduling
---------------------------------------------------------------

Generate code:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Fortran
! Machine 0 excerpt

  call MPI_IRecv( var_11, 4000000, MPI_DOUBLE_PRECISION, &
                  1, 1000, MPI_COMM_WORLD, request_1, ierr_1)
  if (ierr_1 .ne. MPI_SUCCESS) print *, 'MPI_IRecv Failed'
  call Dgemm('N', 'N', 2000, 2000, 2000, 1.0d+0, A, 2000, B, &
             2000, 0.0d+0, var_10, 2000)
  call MPI_WAIT( request_1, status_1, ierr_2)
  if (ierr_2 .ne. MPI_SUCCESS) print *, 'MPI_WAIT Failed'
  call Dgesv(2000, 2000, var_10, 2000, var_7, var_11, 2000, INFO)
  call Dlaswp(2000, var_11, 2000, 1, 2000, var_7, 1)
  
! Machine 1 excerpt

  call Dgemm('N', 'N', 2000, 2000, 2000, 1.0d+0, C, 2000, D, &
             2000, 0.0d+0, var_8, 2000)
  call MPI_ISend( var_8, 4000000, MPI_DOUBLE_PRECISION, 0, 1000,  &
                  MPI_COMM_WORLD, request_2, ierr_3)
  if (ierr_3 .ne. MPI_SUCCESS) print *, 'MPI_ISend Failed'
  call MPI_WAIT( request_2, status_2, ierr_4)
  if (ierr_4 .ne. MPI_SUCCESS) print *, 'MPI_WAIT Failed'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conclusion
==========

Related Work
------------

Numerical Linear Algebra

*   FLAME
*   Plasma/Magma

Notable for software engineering

*   Trilinos
*   Sprial

Program Generation

*   Maude
*   Stratego/XT
