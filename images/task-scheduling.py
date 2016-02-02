from dask.imperative import do, value
import dask.imperative as di
from toolz import sliding_window

do = lambda func: di.do(func, pure=True)


def load(x):
    pass

def read_sql(x):
    pass

def filter(x):
    pass

def join(a, b):
    pass

def store(x):
    pass

def reduce(*args):
    pass

def share(a, b):
    pass


locations = [value(i, name='2016-01-%02d.data' % i) for i in range(1, 7)]
reference = do(read_sql)(value(None, name='reference.data'))


loaded = list(map(do(load), locations))
filtered = list(map(do(filter), loaded))
joined = [do(join)(f, reference) for f in filtered]
shared = [do(share)(a, b) for a, b in sliding_window(2, joined)]
reduced = list(map(do(store), shared))

from dask import visualize
visualize(*reduced, filename='task-scheduling.svg', rankdir='LR')
visualize(*reduced, filename='task-scheduling.pdf', rankdir='LR')
