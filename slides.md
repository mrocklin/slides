% Mathematically Informed Automated Linear Algebra
% Matthew Rocklin
% April 22nd, 2013

Challenges of Scientific Computation
------------------------------------

\begin{figure}[htbp]
\centering
\includegraphics<1>[width=\textwidth]{images/math-hardware-computation}
\includegraphics<2>[width=\textwidth]{images/math-hardware-computation-automation}
\end{figure}

Slides 

    http://github.com/mrocklin/slides
    git clone git@github.com:mrocklin/slides  
    git checkout sandia
    make pdf

Motivating Problem
==================


Uncertainty Propagation via Derivatives
---------------------------------------

\begin{figure}[htbp]
\centering
\includegraphics<1>[width=\textwidth]{images/uq-1}
\includegraphics<2>[width=\textwidth]{images/uq-2}
\includegraphics<3>[width=\textwidth]{images/uq-3}
\includegraphics<4>[width=.8\textwidth]{images/uq-4}
\end{figure}



Argument for High Level Languages 
---------------------------------

**Want**:  Physical processes, derivatives, matrix computations, statistics, and time stepping methods all on array expressions

**Don't want**:  Multiple implementations across hardware (CPU, GPU, ....)

\begin{figure}[htbp]
\centering
\includegraphics[width=.6\textwidth]{images/venn-uq-cuda}
\end{figure}

**Have**:  Plenty of static libraries (BLAS/LAPACK/PETSc/Trillinos) 

**Have**:  Plenty of high level scripting environments (Matlab/Python/R)

**Don't Have**:  Ability to express high-level transformations on high-level code


Today
-----

**Computer Algebra**: Define Linear Algebra in a Computer Algebra System (`SymPy`)

**Backends**: Connect to separate computational backends 
            (Theano and BLAS/LAPACK)

**Performance**: Improvements through algorithm selection and blocking

**Development**: Demographic challenges behind scientific software development

**If time**: Static scheduling (briefly)


Argument for High Level Compilers - Optimizations
-------------------------------------------------

    x = ones(10000, 1)

    x*x'*x              Elapsed time is    ?     seconds.
    (x*x')*x            Elapsed time is 0.337711 seconds.
    x*(x'*x)            Elapsed time is 0.000956 seconds.

\begin{figure}[htbp]
\centering
\includegraphics[width=.6\textwidth]{images/xxtrans}
\end{figure}


Argument for High Level Compilers - Inference
---------------------------------------------

For all matrices $\mathbf{A, B}$ such that $\mathbf A$ is symmetric positive-definite and $\mathbf B$ is orthogonal:

**Question**: is $\mathbf B \cdot\mathbf A \cdot\mathbf B^\top$ symmetric and
positive-definite? 

**Answer**: Yes.

**Question**: Could a computer have told us this?

**Answer**: Probably.

Are there any symbolic algebra systems (like Mathematica) that handle and propagate known facts about matrices?


Argument for High Level Compilers - Inference
---------------------------------------------

For all matrices $\mathbf{A, B}$ such that $\mathbf A$ is symmetric positive-definite and $\mathbf B$ is orthogonal:

**Question**: is $\mathbf B \cdot\mathbf A \cdot\mathbf B^\top$ symmetric and
positive-definite? 

**Answer**: Yes.

**Question**: Could a computer have told us this?

**Answer**: Probably.

Are there any symbolic algebra systems (like Mathematica) that handle and propagate known facts about matrices?

\vspace{1em}
\hrule

    sympy.matrices.expressions

~~~~~~~~Python
>>> A = MatrixSymbol('A', n, n)
>>> B = MatrixSymbol('B', n, n)
>>> context = symmetric(A) & positive_definite(A) & orthogonal(B)
>>> query   = symmetric(B*A*B.T) & positive_definite(B*A*B.T)
>>> ask(query, context)
True
~~~~~~~~


Matrix Expressions and Computations
===================================


Linear Regression - Math
------------------------

$$ X \beta \cong y $$
$$ \beta = (X^TX)^{-1}X^Ty $$


\begin{figure}[htbp]
\centering
\includegraphics[width=.4\textwidth]{images/linregress-xy}
\end{figure}

Linear Regression - Python/MatLab
---------------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$

Python/NumPy

    beta = (X.T*X).I * X.T*y

MatLab

    beta = inv(X'*X) * X'*y


Linear Regression - Python/MatLab
---------------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$

Python/NumPy

    beta = solve(X.T*X, X.T*y)

MatLab

    beta = X'*X \ X'*y


Linear Regression - Python/MatLab
---------------------------------

$$ \beta = (X^TX)^{-1}X^Ty $$

Python/NumPy

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


Connecting Math and Computation
-------------------------------

**Given**:

    (X.T*X).I*X.T*y
    full_rank(X)

**Produce**:

\begin{figure}[htbp]
\centering
\includegraphics[width=.8\textwidth]{images/hat-comp}
\end{figure}


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


Compilation
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
facts   = fullrank(X)

f = fortran_function(inputs, outputs, facts)
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

Kalman Filter
-------------

~~~~~~~~~~Python
newmu       = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
newSigma    = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma

assumptions = [positive_definite(Sigma), symmetric(Sigma), 
               positive_definite(R), symmetric(R), fullrank(H)]

f = fortran_function([mu, Sigma, H, R, data], [newmu, newSigma], *assumptions)
~~~~~~~~~~

\begin{figure}[htbp]
\centering
\includegraphics[width=.7\textwidth]{images/kalman-math}
\end{figure}


Background and Related Work
---------------------------

*   **BLAS/LAPACK** - Excellent libraries for dense linear algebra computations
*   **ATLAS** - Autotunes for architecture (algorithm selection, blocksizes, ...)
*   **Matlab** - Impressive dynamic runtime checks.  E.g. `\` operator
*   **FLAME** - Formal Linear Algebra Methods Environment
*   **TCE** - Tensor Contraction Engine 
*   "A Domain-Specific Compiler for Linear Algebra Operations" Fabregat, Bientinesi, 2012  -- AICES
*   **Spiral** - Hardware specific numeric code generation with internal computation language
*   **Theano** - Tensor compiler Python $\rightarrow$ Python/C/CUDA

*   **Trillinos** - shared ideals - high-level, separable scientific software, 
    some high level transformations through templates


Separation
==========

Separation
----------

\begin{tikzpicture}[every text node part/.style={align=center,rectangle}]

    \node (math) at (10,8) {Mathematics \\ (Inverse, transpose, positive-definite)};
    \node (connection) at (10,5)  {};
    \node (computation) at (10,2)  {Computation/DAG \\ (GEMM, POSV)};
    \node (pl) at (7,5)  {Programming Languages \\ (Graph covering)};
    \node (fortran) at (10,0)  {Fortran};

    \foreach \from/\to in
    {math/connection, pl/connection, computation/connection, computation/fortran}
    \draw (\from) -- (\to);

\end{tikzpicture}


Separation
----------

\begin{tikzpicture}[every text node part/.style={align=center,rectangle}]

    \node (math) at (10,8) {Mathematics \\ (Inverse, transpose, positive-definite)};
    \node (connection) at (10,5)  {};
    \node (computation) at (10,2)  {Computation/DAG \\ (GEMM, POSV)};
    \node (pl) at (3,5)  {Programming Languages \\ (Graph covering)};
    \node (fortran) at (10,0)  {Fortran};

    \foreach \from/\to in
    {math/connection, pl/connection, computation/connection, computation/fortran}
    \draw (\from) -- (\to);

\end{tikzpicture}

Separation promotes Growth
--------------------------

\begin{tikzpicture}[every text node part/.style={align=center,rectangle}]

    \node (math) at (10,8) {Mathematics \\ (Inverse, transpose, positive-definite\\ determinant, block matrices, DFT)};
    \node (connection) at (10,5)  {};
    \node (computation) at (10,2)  {Computation/DAG \\ (GEMM, POSV, FFTW)};
    \node (pl) at (3,5)  {Programming Languages \\ (Graph covering)};
    \node (fortran) at (10,0)  {Fortran};

    \foreach \from/\to in
    {math/connection, pl/connection, computation/connection, computation/fortran}
    \draw (\from) -- (\to);

\end{tikzpicture}


SYRK
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
\includegraphics[width=.9\textwidth]{images/hat-comp}
\end{figure}


SYRK
----

\begin{figure}[htbp]
\centering
\includegraphics[width=.9\textwidth]{images/hat-comp-syrk}
\end{figure}

~~~~~~~~~~~Python
class SYRK(BLAS):
    """ Symmetric Rank-K Update `alpha X' X + beta Y' """
    _inputs  = (alpha, A, beta, D)
    _outputs = (alpha * A * A.T + beta * D,)
    inplace  = {0: 3}
    fortran_template = ("call %(fn)s('%(UPLO)s', '%(TRANS)s', %(N)s, %(K)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, "
                        "%(beta)s, %(D)s, %(LDD)s)")
    ...

  (alpha*A*A.T + beta*D, SYRK(alpha, A, beta, D), True),
  (A*A.T,                SYRK(1.0, A, 0.0, 0),  , True),
~~~~~~~~~~~~

SYRK
----

\begin{figure}[htbp]
\centering
\includegraphics[width=.9\textwidth]{images/hat-comp}
\end{figure}

    Elapsed real time = 0.43399999 
    
\begin{figure}[htbp]
\centering
\includegraphics[width=.9\textwidth]{images/hat-comp-syrk}
\end{figure}

    Elapsed real time = 0.39500001 



Separation promotes Growth
--------------------------

\begin{tikzpicture}[every text node part/.style={align=center,rectangle}]

    \node (math) at (10,8) {Mathematics \\ (Inverse, transpose, positive-definite\\ determinant, block matrices, DFT)};
    \node (connection) at (10,5)  {};
    \node (computation) at (10,2)  {Computation/DAG \\ (GEMM, POSV, FFTW)};
    \node (pl) at (3,5)  {Programming Languages \\ (Graph covering)};
    \node (fortran) at (10,0)  {Fortran};

    \foreach \from/\to in
    {math/connection, pl/connection, computation/connection, computation/fortran}
    \draw (\from) -- (\to);

\end{tikzpicture}


Separation promotes Experimentation and Comparison
--------------------------------------------------

\begin{tikzpicture}[every text node part/.style={align=center,rectangle}]

    \node (math) at (10,8) {Mathematics \\ (Inverse, transpose, positive-definite\\ determinant, block matrices, DFT)};
    \node (connection) at (10,5)  {};
    \node (computation) at (10,2)  {Computation/DAG \\ (GEMM, POSV, FFTW)};
    \node (pl) at (3,5)  {Programming Languages \\ (Graph covering)};
    \node (fortran) at (10,0)  {Fortran};

    \node (Theano) at (6, 2)  {Theano \\ (Tensor computations)};

    \foreach \from/\to in
    {math/connection, pl/connection, computation/connection, computation/fortran, connection/Theano}
    \draw (\from) -- (\to);

\end{tikzpicture}

Kalman Filter - Theano v. Fortran
---------------------------------

~~~~~~~~~~Python
from sympy.computations.matrices import fortran_function

newmu   = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
newSigma= Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma

assumptions = [positive_definite(Sigma), symmetric(Sigma), 
               positive_definite(R), symmetric(R), fullrank(H)]

f = fortran_function([mu, Sigma, H, R, data], [newmu, newSigma], *assumptions)
~~~~~~~~~~

Kalman Filter - Theano v. Fortran
---------------------------------

~~~~~~~~~~Python
from sympy.printing.theanocode import theano_function

newmu   = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
newSigma= Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma




f = theano_function([mu, Sigma, H, R, data], [newmu, newSigma])
~~~~~~~~~~

Blocked Algorithms
==================

Blocked Matrix Multiply Improves Cache Performance
----------------------------------------------------

~~~~~~~~~~Python
from sympy import BlockMatrix, block_collapse
A, B, C, D, E, F, G, K = [MatrixSymbol(a, n, n) for a in 'ABCDEFGK']
X = BlockMatrix([[A, B],
                 [C, D]])
Y = BlockMatrix([[E, F],
                 [G, K]])
latex(X*Y)
~~~~~~~~~~

$$ \begin{bmatrix} A & B \\\\ C & D \end{bmatrix} 
   \begin{bmatrix} E & F \\\\ G & K \end{bmatrix}$$

~~~~~~~~~~Python
latex(block_collapse(X*Y))
~~~~~~~~~~

$$ \begin{bmatrix} A E + B G & A F + B K \\\\ 
                   C E + D G & C F + D K\end{bmatrix} $$


Blocked Matrix Inverse Improves Cache Performance
-------------------------------------------------

~~~~~~~~~~Python
X = BlockMatrix([[A, B],
                 [C, D]])
~~~~~~~~~~

$$ \begin{bmatrix} A & B \\\\ C & D \end{bmatrix}^{-1} $$

~~~~~~~~~~Python
latex(block_collapse(X.I))
~~~~~~~~~~

$$ \begin{bmatrix} 
\left(- B D^{-1} C + A\right)^{-1} & - A^{-1} B \left(- C A^{-1} B + D\right)^{-1} \\\\ 
- \left(- C A^{-1} B + D\right)^{-1} C A^{-1} & \left(- C A^{-1} B + D\right)^{-1}
\end{bmatrix} $$


Kalman Filter
-------------

~~~~~~~~~~Python
newmu       = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
newSigma    = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma

assumptions = [positive_definite(Sigma), symmetric(Sigma), 
               positive_definite(R), symmetric(R), fullrank(H)]
~~~~~~~~~~

\begin{figure}[htbp]
\centering
\includegraphics[width=.7\textwidth]{images/kalman-math}
\end{figure}


Blocked Kalman Filter
---------------------

~~~~~~~~~~Python
from sympy import blockcut, block_collapse
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

fblocked = theano_function(inputs, collapsed_outputs, dtypes=dtypes)
~~~~~~~~~~

\begin{figure}[htbp]
\centering
\includegraphics[width=.9\textwidth]{images/fblocked}
\end{figure}


Blocked Kalman Filter
---------------------

\begin{figure}[htbp]
\centering
\includegraphics[width=.9\textwidth]{images/fblocked}
\end{figure}

~~~~~~~~~~~Python
>>> inputs = [numpy.random.....  ]
>>> timeit f(*inputs)
1 loops, best of 3: 2.69 s per loop

>>> timeit fblocked(*inputs)
1 loops, best of 3: 2.12 s per loop
~~~~~~~~~~~


Separation
----------

\begin{tikzpicture}[every text node part/.style={align=center,rectangle}]

    \node (math) at (10,8) {Mathematics \\ (Inverse, transpose, positive-definite\\ determinant, block matrices, DFT)};
    \node (connection) at (10,5)  {};
    \node (computation) at (10,2)  {Computation/DAG \\ (GEMM, POSV, FFTW)};
    \node (pl) at (3,5)  {Programming Languages \\ (Graph covering)};
    \node (fortran) at (10,0)  {Fortran};

    \node (Theano) at (6, 2)  {Theano \\ (Tensor computations)};

    \foreach \from/\to in
    {math/connection, pl/connection, computation/connection, computation/fortran, connection/Theano}
    \draw (\from) -- (\to);

\end{tikzpicture}


Separation promotes Adaptability
--------------------------------

\begin{tikzpicture}[every text node part/.style={align=center,rectangle}]

    \node (math) at (10,8) {Mathematics \\ (Inverse, transpose, positive-definite\\ determinant, block matrices, DFT)};
    \node (connection) at (10,5)  {};
    \node (computation) at (10,2)  {Computation/DAG \\ (GEMM, POSV, FFTW)};
    \node (pl) at (3,5)  {Programming Languages \\ (Graph covering)};
    \node (fortran) at (10,0)  {Fortran};
    \node (C) at (12,0)  {C};
    \node (CUDA) at (8, 0)  {CUDA};

    \node (Theano) at (6, 2)  {Theano \\ (Tensor computations)};

    \foreach \from/\to in
    {math/connection, pl/connection, computation/connection, computation/fortran, computation/C, computation/CUDA, connection/Theano}
    \draw (\from) -- (\to);

\end{tikzpicture}

Static Scheduling
=================

Static Scheduling
-----------------

**Given**:
\begin{columns}
\column{.5\textwidth}

Computation Graph

\column{.5\textwidth}

\begin{figure}[htbp]
\centering
\includegraphics[width=.8\textwidth]{images/hat-comp}
\end{figure}

\end{columns}

\begin{columns}
\column{.5\textwidth}
Worker network 

\column{.5\textwidth}

\begin{figure}[htbp]
\centering
\includegraphics[width=.8\textwidth]{images/worker}
\end{figure}
\end{columns}


\begin{columns}
\column{.5\textwidth}
Computation times

Communication times 

\column{.5\textwidth}
:: task, worker $\rightarrow$ time

:: variable, source, target $\rightarrow$ time
\end{columns}

**Produce**:

Set of computation subgraphs to minimize total runtime


Related work
------------

*   Heterogeneous Static Scheduling
    *   **HEFT**: H. Topcuoglu, S. Hariri, M. Wu. *Performance-effective and low-complexity task scheduling for heterogeneous computing.* 2002
    *   **ILP**: M. Tompkins. *Optimization Techniques for Task Allocation and Scheduling in Distributed Multi-Agent Operations.* 2003

*   Performance Modeling
    *   E Peise and P Bientinesi. *Performance Modeling for Dense Linear Algebra.* 2012
    *   Roman Iakymchuk. *Performance Modeling and Prediction for Linear Algebra Algorithms* 2012
    *   JJ Dongarra, RA Vandegeijn, and DW Walker. *Scalability issues affecting the design of a dense linear algebra library* 1994

*   Automated Dense Linear Algebra
    *   ScaLAPACK, PlaLAPACK, BLACS
    *   FLAME - Language for blocked matrix algorithms
        -   SuperMatrix - Dynamic shared memory variant
        -   Elemental - Distributed memory variant
    *   Magma - Hybrid LAPACK - Parametrized dynamic/static scheduling


Static Scheduling
-----------------

    newmu    = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
    newSigma = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma

\begin{figure}[htbp]
\centering
\includegraphics[width=.8\textwidth]{images/kalman-math}
\end{figure}

Static Scheduling
-----------------
    
    newmu    = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
    newSigma = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma

\begin{figure}[htbp]
\centering
\includegraphics[width=.48\textwidth]{images/kalman_cpu1}
\includegraphics[width=.48\textwidth]{images/kalman_cpu0}
\end{figure}


Conclusion
==========

Software Links
--------------

+--------------+---------------------------+-------------------------------------------+
|  Project     | Application               | URL                                       |
+==============+===========================+===========================================+
| SymPy        | Computer Algebra          | http://github.com/sympy/sympy/            |
+--------------+---------------------------+-------------------------------------------+
| Computations | BLAS/LAPACK/FFTW/...      | http://github.com/mrocklin/computations/  |
+--------------+---------------------------+-------------------------------------------+
| LogPy        | Logic Programming         | http://github.com/logpy/logpy/            |
+--------------+---------------------------+-------------------------------------------+
| Strategies   | Control Flow Programming  | http://github.com/logpy/strategies/       |
+--------------+---------------------------+-------------------------------------------+
| Theano       | Array Computation         | http://deeplearning.net/software/theano/  |
+--------------+---------------------------+-------------------------------------------+
| HEFT         | Scheduling Heuristic      | http://github.com/mrocklin/heft/          |
+--------------+---------------------------+-------------------------------------------+
| ILP          | Scheduling Algorithm      | http://github.com/mrocklin/tompkins/      |
+--------------+---------------------------+-------------------------------------------+
| Blog         | Numerical Experiments     | http://matthewrocklin.com/blog/           |
+--------------+---------------------------+-------------------------------------------+


Other Work and Collaborations
-----------------------------

*   SymPy - Standard CAS in numeric Python ecosystem
    *   Statistics / Uncertainty modeling
    *   Linear algebra
*   Theano - Array computations
    *   GPU/MPI asynchronous communication
    *   Scheduling
*   SymPy-Theano code generation 
    *   Mechanical engineering - ODEs of complex expressions
*   LogPy - Term rewrite system/Compiler tools
    *   SymPy
    *   Theano
        *   Nengo neural simulator

\begin{footnotesize}

\begin{itemize}
\item   M. Rocklin, A. Pinar \textit{On Clustering on Graphs with Multiple Edge
    Types}, Internet Mathematics, 2012
\item   M. Rocklin, A. Pinar, \textit{Latent Clustering on Graphs with Multiple
    Edge Types} Algorithms and Models for the Web-Graph, 2011
\item   M. Rocklin, A. Pinar, \textit{Computing an Aggregate Edge-Weight Function
    for Clustering Graphs with Multiple Edge
    Types} Algorithms and Models for the Web-Graph, 2010
\item   E. Constantinescu,V. Zavala, M. Rocklin, S. Lee, and M. Anitescu,
    \textit{A Computational Framework for Uncertainty Quantification and
    Stochastic Optimization in Unit Commitment with Wind Power
    Generation.} IEEE Transactions on Power Systems, 2010.
\item M. Rocklin, \textit{Uncertainty Quantification and Sensitivity Analysis in
    Dynamical Systems} 2011, Masters Thesis
\item   M. Rocklin, \textit{Uncertainty Modeling with SymPy Stats}
    SciPy-2012
\end{itemize}


\end{footnotesize}


End
---

~~~~~~~~~~Python
newmu       = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
newSigma    = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma

assumptions = [positive_definite(Sigma), symmetric(Sigma), 
               positive_definite(R), symmetric(R), fullrank(H)]
~~~~~~~~~~

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

