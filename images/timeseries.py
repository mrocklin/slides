import dask.dataframe as dd
from toolz import assoc

df = dd.demo.make_timeseries('2010-01-01', '2010-12-31',
                             {'value': float, 'name': str, 'id': int},
                             freq='10s',
                             partition_freq='1M',
                             seed=1)


kwargs = {'bgcolor': '#00000000',
          'rankdir': 'BT',
          'node_attr': {'color': 'white',
                        'fontcolor': '#FFFFFF',
                        'penwidth': '3'},
          'edge_attr': {'color': 'white',
                        'penwidth': '3'}}

df.value.resample('1w').mean().visualize('resample.svg', **kwargs)

df = dd.demo.make_timeseries('2010-01-01', '2010-08-30',
                             {'value': float, 'name': str, 'id': int},
                             freq='10s',
                             partition_freq='1M',
                             seed=1)


df.value.rolling(100).mean().visualize('rolling.svg', **assoc(kwargs,
'rankdir', 'LR'))


