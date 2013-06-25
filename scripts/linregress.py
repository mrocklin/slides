from sympy.abc import n, m
from sympy import *
from computations.matrices import GEMM, POSV, COPY
from computations.dot import writepdf
from computations.inplace import inplace_compile

X = MatrixSymbol('X', n, m)
y = MatrixSymbol('y', n, 1)

comp = (GEMM(1, X.T, X, 0, 0)
      + GEMM(1, X.T, y, 0, 0)
      + POSV(X.T*X, X.T*y))

icomp = inplace_compile(comp, Copy=COPY)

writepdf(comp,  'images/linregress',  rankdir='LR')
writepdf(icomp, 'images/ilinregress', rankdir='LR')
