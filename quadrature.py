import numpy as np
from sympy import *
from matplotlib import pyplot as plt

x = Symbol('x')

def plot_func(func, left, right, n=100, color=(.5, .5, .5), alpha=0.5):
    from pylab import plot, fill_between, grid, figure
    u = np.linspace(float(left), float(right), int(n))
    v = np.array(func(u), dtype=np.float)

    m = np.amax(v)

    col = list(color)

    fill_between(u, np.zeros(u.shape), v, color=col+[alpha])
    plot(u,v, color=col)

def element(x, expr, left, right, n):
    n = S(n)
    ip = interpolating_poly(n, x)
    xs = symbols('x0:%d'%n)
    ys = symbols('y0:%d'%n)
    nxs = [left + i*(right-left)/(n-1) for i in range(n)]
    f = Lambda(x, expr)
    nys = map(f, nxs)

    return ip.subs(dict(zip(xs+ys, list(nxs)+list(nys))))

def test_element():
    x = Symbol('x')
    assert element(x, x, 0, 1, 2) == x
    assert simplify(element(x, 2*x, 0, 1, 3)) == 2*x
    assert element(x, sin(x), 0, pi/2, 3) == \
            -8*sqrt(2)*x*(x - pi/2)/pi**2 + 8*x*(x - pi/4)/pi**2

def ranges(left, right, n):
    delta = (right - left) / n
    return zip(frange(left, right, n+1)[:-1],
               frange(left+delta, right+delta, n+1)[:-1])


def test_ranges():
    assert ranges(0, 12, 3) == [(0, 4), (4, 8), (8, 12)]


def frange(a, b, n):
    return [a + i*(b-a)/(S(n)-1) for i in range(n)]

def elements(x, expr, left, right, numelements, order):
    return [element(x, expr, a, b, order)
            for a, b in ranges(left, right, numelements)]

def show(x, expr, left, right, numelements, order):
    plt.figure()
    for i, (a, b) in enumerate(ranges(left, right, numelements)):
        e = element(x, sin(x), a, b, order)
        f = lambdify(x, e)
        plot_func(f, a, b, alpha=.5 - .1 * (-1)**i)
        xs = [a + i*(b-a)/(S(order)-1) for i in range(order)]
        ys = map(f, xs)
        plt.plot(xs, ys, 'ko')
    plot_func(lambdify(x, expr, np), left, right, 1000, (0,0,0), alpha=0)

def slide():
    left, right = 0, 2*pi/3
    show(x, sin(x), left, right, 2, 2)
    plt.axis((left, float(right), 0, 1))
    plt.savefig('images/quadrature-22.pdf')
    show(x, sin(x), left, right, 1, 3)
    plt.axis((left, float(right), 0, 1))
    plt.savefig('images/quadrature-13.pdf')
    show(x, sin(x), left, right, 6, 2)
    plt.axis((left, float(right), 0, 1))
    plt.savefig('images/quadrature-62.pdf')
    show(x, sin(x), left, right, 1, 4)
    plt.axis((left, float(right), 0, 1))
    plt.savefig('images/quadrature-14.pdf')

if __name__ == '__main__':
    slide()
