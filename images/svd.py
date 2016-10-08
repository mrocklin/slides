import dask.array as da
from dask import visualize

x = da.random.normal(10, 1, size=(1000, 200), chunks=(200, 200))


kwargs = {'bgcolor': '#00000000',
          'rankdir': 'BT',
          'node_attr': {'color': 'white',
                        'fontcolor': '#FFFFFF',
                        'penwidth': '3'},
          'edge_attr': {'color': 'white', 'penwidth': '3'}}

u, s, v = da.linalg.svd(x)
visualize(u, s, v, filename='svd.svg', **kwargs)

x = da.random.normal(10, 1, size=(1000, 1000), chunks=(200, 200))
u, s, v = da.linalg.svd_compressed(x, 1)
visualize(u, s, v, filename='svd-compressed.svg', **kwargs)

x = da.random.normal(10, 1, size=(1600, 1600), chunks=(200, 200))
u, s, v = da.linalg.svd_compressed(x, 300, n_power_iter=2)
visualize(u, s, v, filename='svd-compressed-large.svg', **kwargs)
