from sympy import assuming, Q
from computations.matrices.fortran.core import generate
from computations.matrices.examples.kalman import inputs, outputs, assumptions
from computations.matrices.examples.kalman_comp import c

from computations.inplace import inplace_compile
from computations.matrices.blas import COPY
ic = inplace_compile(c, Copy=COPY)

with assuming(*(assumptions + tuple(map(Q.real_elements, inputs)))):
    s = generate(ic, inputs, outputs)

with open('kalman.f90', 'w') as f:
    f.write(s)
