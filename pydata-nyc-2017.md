Streaming Processing with Dask
------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

Anaconda Inc.

(*formerly Continuum Analytics*)


Streaming Processing
--------------------

<img src="images/dask_icon.svg" width=20% style="opacity:0.0">

*Matthew Rocklin*

Anaconda Inc.

(*formerly Continuum Analytics*)


-  **Streaming data processing is ...**:
    -  Unbounded: we might receive data forever
    -  Timely: we care about responding quickly
    -  ...
-  **Used in ...**
    -  Scientific instruments (like the keynote)
    -  Web server logs
    -  Financial time series
    -  Telemetry from devices, cars, IoT, ...
    -  Network data
    -  ...



### Lets start with a quick demonstration

<a href="https://www.youtube.com/watch?v=G981CbrUUwQ">video link</a>



### Solutions already exist for these problems

-  **Big Data Solutions**
    - Apache Flink
    - Apache Spark Streaming
    - Apache Beam
    - Apache Storm
-  **Complex / performant solutions**
    - Akka
    - Message queueing systems (ZeroMQ, RabbitMQ)
    - Custom protocols (science, finance)
    - Custom code (threads, queues, sockets, ...)
-  **Reactive Programming** (UI)
    - ReactiveX / observer pattern


### But we're going to make another one ...

<hr>

### Streamz

1.  Pythonic
2.  Simple in simple cases
3.  Flexible enough for complex cases
4.  Integrates nicely with PyData libraries (Jupyter, Pandas, ...)
5.  Scales nicely

<hr>

*Disclaimer: everything in this talk is experimental.  APIs subject to change
without notice*


What we'll see today
--------------------

1.  **Streamz.core**
    1.  map, accumulate
    2.  time management and back pressure
    3.  Jupyter notebook integration
2.  **Streamz.dataframe**
    1.  Stream of Pandas dataframes
    2.  With pandas API
    3.  Plotting integration with Holoviews/Bokeh
3.  **Streamz.dask**
    1.  Full implementation of Streamz.core on top of Dask
    2.  Adds millisecond overhead and 10-20ms latency
    3.  Scales



<img src="images/streamz-map.svg" width="100%">

<hr>

```python
>>> new_stream = stream.map(func)
```


<img src="images/streamz-accumulate.svg" width="100%">

<hr>

```python
def binop(total, new):
    return total + new

>>> reduce(binop, range(10))      # single final result
45

>>> accumulate(binop, range(10))  # new result for every new element
[0, 1, 3, 6, 10, 15, 21, 28, 36, 45]

>>> new_stream = stream.accumulate(binop, start=0)
```


### Branching

<img src="images/streamz-branch.svg" width="100%">

<hr>

### Joining

<img src="images/streamz-join.svg" width="100%">


### Many other operations exist

<a href="http://streamz.readthedocs.io/en/latest/api.html">Stream API</a>


Time and Back Pressure
----------------------

<img src="images/streamz-time.svg" width="100%">

```python
# Later stages in a pipeline might be slow (like writing to a database)
stream.map(f).combine_latest(other).accumulate(h).map(write_to_database)

for element in data:      # user pushes data into stream
    stream.emit(element)  # needs to be told to slow down
```


Time and Back Pressure
----------------------

<img src="images/streamz-time-1.5.svg" width="100%">

```python
# Later stages in a pipeline might be slow (like writing to a database)
stream.map(f).combine_latest(other).accumulate(h).map(write_to_database)

for element in data:      # user pushes data into stream
    stream.emit(element)  # needs to be told to slow down
```


Time and Back Pressure
----------------------

<img src="images/streamz-time-2.svg" width="100%">

```python
# Later stages in a pipeline might be slow (like writing to a database)
stream.map(f).combine_latest(other).accumulate(h).map(write_to_database)

for element in data:      # user pushes data into stream
    stream.emit(element)  # needs to be told to slow down
```


Time and Back Pressure
----------------------

<img src="images/streamz-time-2.svg" width="100%">

```python
# Later stages in a pipeline might be slow (like writing to a database)
stream.map(f).combine_latest(other).accumulate(h).map(write_to_database)

for element in data:            # user pushes data into stream
    await stream.emit(element)  # needs to be told to slow down
```


Time and Back Pressure
----------------------

<img src="images/streamz-time-3.svg" width="100%">

```python
# Later stages in a pipeline might be slow (like writing to a database)
stream...buffer(100)...rate_limit('5ms')...map(write_to_database)

for element in data:            # user pushes data into stream
    await stream.emit(element)  # needs to be told to slow down
```



Streams are easy to extend
--------------------------

```python
@Stream.register_api()
class map(Stream):
    """ Apply a function to every element in the stream """
    def __init__(self, upstream, func, *args, **kwargs):
        self.func = func
        # this is one of a few stream specific kwargs
        stream_name = kwargs.pop('stream_name', None)
        self.kwargs = kwargs
        self.args = args

        Stream.__init__(self, upstream, stream_name=stream_name)

    def update(self, x, who=None):
        result = self.func(x, *self.args, **self.kwargs)

        return self._emit(result)
```


Streams are easy to extend
--------------------------

```python
@Stream.register_api()
class filter(Stream):
    """ Only pass through elements that satisfy the predicate """
    def __init__(self, upstream, predicate, **kwargs):
        if predicate is None:
            predicate = _truthy
        self.predicate = predicate

        Stream.__init__(self, upstream, **kwargs)

    def update(self, x, who=None):
        if self.predicate(x):
            return self._emit(x)
```


Streams are easy to extend
--------------------------

```python
@Stream.register_api()
class rate_limit(Stream):
    """ Limit the flow of data """
    _graphviz_shape = 'octagon'

    def __init__(self, upstream, interval, **kwargs):
        self.interval = convert_interval(interval)
        self.next = 0

        Stream.__init__(self, upstream, **kwargs)

    @gen.coroutine
    def update(self, x, who=None):
        now = time()
        old_next = self.next
        self.next = max(now, self.next) + self.interval
        if now < old_next:
            yield gen.sleep(old_next - now)
        yield self._emit(x)
```

Use Tornado coroutines for time-dependent operations


Streams are easy to extend
--------------------------

```python
@Stream.register_api()
class rate_limit(Stream):
    """ Limit the flow of data """
    _graphviz_shape = 'octagon'

    def __init__(self, upstream, interval, **kwargs):
        self.interval = convert_interval(interval)
        self.next = 0

        Stream.__init__(self, upstream, **kwargs)

    .
    async def update(self, x, who=None):
        now = time()
        old_next = self.next
        self.next = max(now, self.next) + self.interval
        if now < old_next:
            await gen.sleep(old_next - now)
        await self._emit(x)
```

Or use async-await syntax if you prefer



## DataFrames


DataFrames
----------

-  Passing dataframes through streams is a common case
-  Can map/accumulate Pandas functions on normal Streams
-  Or use streamz.dataframe module for syntactic sugar

```python
from streamz import Stream
stream = Stream()

def query(df):
    return df[df.name == 'Alice']

def add_x(acc, new):
    return acc + new.x.sum()

stream.map(query).accumulate(add_x)  # like df[df.name == 'Alice'].x.sum()
```

<img src="images/streamz-map.svg" width="100%">


DataFrames
----------

-  Passing dataframes through streamz is a common case
-  Can map/accumulate Pandas functions on normal Streams
-  Or use streamz.dataframe module for syntactic sugar

```python
from streamz.dataframe import DataFrame

df = DataFrame(stream=stream,
               example=pd.DataFrame({'name': [], 'x': [], 'y': []}))

df[df.name == 'Alice'].x.sum()

.
.
.
```

<img src="images/streamz-map.svg" width="100%">


DataFrames
----------

-  Passing dataframes through streamz is a common case
-  Can map/accumulate Pandas functions on normal Streams
-  Or use streamz.dataframe module for syntactic sugar

```python
from streamz.dataframe import DataFrame

df = DataFrame(stream=stream,
               example=pd.DataFrame({'name': [], 'x': [], 'y': []}))

df.window('60s').groupby('name').x.var()

.
.
.
```

<img src="images/streamz-map.svg" width="100%">


Dataframe Plotting
------------------

-  Copies Pandas `.plot` interface
-  Currently using Bokeh + Holoviews (thanks [Philipp Rudiger](http://philippjfr.com/)!)

<img src="images/streamz-plot-line.gif" width="80%">

<img src="images/streamz-plot-bar.gif" width="80%">


Jupyter Notebook Integration
----------------------------

-  Uses IPython widgets
-  Pleasant for demonstration and exploration
-  Has some performance penalty
-  Rate limited at 500ms for visual sanity

<img src="images/streamz-ipywidgets.gif" width="100%">



Streaming DataFrams with Dask
-----------------------------

<img src="images/dask_icon.svg" width=20%>


### Dask is known for ...

1.  Parallelizing NumPy
2.  Parallelizing Pandas
3.  Parallelizing parts of Scikit-Learn
4.  Scaling concurrent.futures

### Dask is used for ...

-  Parallelizing custom internal systems

    *Can we parallelize streamz?*

```python
from streamz import Stream
```


### Dask is known for ...

1.  Parallelizing NumPy
2.  Parallelizing Pandas
3.  Parallelizing parts of Scikit-Learn
4.  Scaling concurrent.futures

### Dask is used for ...

-  Parallelizing custom internal systems

    *Can we parallelize streamz?*

```python
from streamz.dask import DaskStream  # drop in replacement (mostly)
```


### DaskStream is a drop-in replacement for Stream

### Scales down to multi-core. Scales up to clusters.

<hr>

### Written in around 200 lines of code


Using streamz.dask
------------------

```python
.
.
.
a = Stream()
b = Stream()

a2 = a.map(parse).rate_limit('10ms')
b2 = b.map(load_from_file).map(process)
c = combine_latest(a2, b2).accumulate(...).map(write).map(log)
```

<img src="images/streamz-dask-map.svg">


Using streamz.dask
------------------

```python
from dask.distributed import Client
client = Client()  # create or connect to Dask cluster

a = Stream()
b = Stream()

a2 = a.map(parse).scatter().rate_limit('10ms')
b2 = b.scatter().map(load_from_file).map(process)
c = combine_latest(a2, b2).accumulate(...).map(write).gather().map(log)
```

<img src="images/streamz-dask-map.svg">


Using streamz.dask
------------------

<a href="images/streamz-dask.gif"><img src="images/streamz-dask.gif" width="100%"></a>

-  Dask scheduler handles 1000's of tasks per second
-  Adds 10-20ms roundtrip latency



Performance
-----------

-  Python iterators: 100ns
-  Streamz: 1-10us
-  Pandas: 100us-1ms (creating a dataframe)
-  Dask: 200us (centralized)


What doesn't work
-----------------

1.  Convenient data source integration
2.  Out-of-order dataframe handling
3.  Benchmarking and profiling
4.  Lineage culling / checkpointing
5.  General use, debugging, user feedback


Questions?
----------

-  [streamz.readthedocs.io](https://streamz.readthedocs.io)
-  [github.com/mrocklin/streamz](https://github.com/mrocklin/streamz)
-  Interactive plots: [holoviews.org](http://holoviews.org/) and [bokeh.pydata.org](https://bokeh.pydata.org/en/latest/)
-  [@mrocklin](https://twitter.com/mrocklin)

<img src="images/streamz-plot-line.gif" width="80%">
