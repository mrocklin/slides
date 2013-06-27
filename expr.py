>>> x = Symbol('x')
>>> expr = log(3*exp(x + 2))

>>> simplify(expr)
x + 2 + log(3)
