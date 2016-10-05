import random
from time import sleep

def load(address):
    sleep(random.random() / 2)
    pass

def load_from_sql(address):
    sleep(random.random() / 1 + 1)
    pass

def process(data, reference):
    sleep(random.random() / 2)
    pass

def roll(a, b, c):
    sleep(random.random() / 5)
    pass

def compare(a, b):
    sleep(random.random() / 10)
    pass

def reduction(seq):
    sleep(random.random() / 1)
    pass


from dask import delayed, visualize

load = delayed(load)
load_from_sql = delayed(load_from_sql)
process = delayed(process)
roll = delayed(roll)
compare = delayed(compare)
reduction = delayed(reduction)


filenames = ['mydata-%d.dat' % i for i in range(10)]

data = [load(fn) for fn in filenames]

reference = load_from_sql('sql://mytable')

processed = [process(d, reference) for d in data]

rolled = []
for i in range(len(processed) - 2):
    a = processed[i]
    b = processed[i + 1]
    c = processed[i + 2]
    r = roll(a, b, c)
    rolled.append(r)

compared = []
for i in range(20):
    a = random.choice(rolled)
    b = random.choice(rolled)
    c = compare(a, b)
    compared.append(c)

best = reduction(compared)

kwargs = {'bgcolor': '#00000000',
          'rankdir': 'BT',
          'node_attr': {'color': 'white', 'fontcolor': '#FFFFFF'},
          'edge_attr': {'color': 'white'}}

visualize(*data, filename='custom-etl-1.svg', **kwargs)
visualize(reference, *data, filename='custom-etl-2.svg', **kwargs)
visualize(*processed, filename='custom-etl-3.svg', **kwargs)
visualize(*rolled, filename='custom-etl-4.svg', **kwargs)
visualize(*compared, filename='custom-etl-5.svg', **kwargs)
visualize(best, filename='custom-etl-6.svg', **kwargs)

