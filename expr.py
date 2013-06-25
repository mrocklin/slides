>>> x = Symbol('x')
>>> expr = log(3*exp(x + 2))
>>> expr = log(Mul(3, exp(Add(x, 2))))

>>> simplify(expr)
x + 2 + log(3)
