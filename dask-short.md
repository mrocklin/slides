Dask: Parallel Programming in Python
------------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

Continuum Analytics



### Dask enables parallel Python

<hr>

### ... originally designed to parallelize NumPy and Pandas

<hr>

### ... but also used today for arbitrary computations


### Dask.array

<img src="images/dask-array.svg" width="60%">

    import numpy as np
    x = np.random.random(...)
    u, s, v = np.linalg.svg(x.dot(x.T))

    import dask.array as da
    x = da.random.random(..., chunks=(1000, 1000))
    u, s, v = da.linalg.svg(x.dot(x.T))


### Dask.DataFrame

<img src="images/dask-dataframe-inverted.svg" width="30%">

    import pandas as pd
    df = pd.read_csv('myfile.csv', parse_dates=['timestamp'])
    df.groupby(df.timestamp.dt.hour).value.mean()

    import dask.dataframe as dd
    df = dd.read_csv('hdfs://myfile.*.csv', parse_dates=['timestamp'])
    df.groupby(df.timestamp.dt.hour).value.mean()


### Fine Grained Python Code

    .

<hr>

    results = {}

    for a in A:
        for b in B:
            if a < b:
                results[a, b] = f(a, b)
            else:
                results[a, b] = g(a, b)

    .


### Fine Grained Python Code

    from dask import delayed, compute

<hr>

    results = {}

    for a in A:
        for b in B:
            if a < b:
                results[a, b] = delayed(f)(a, b)  # lazily construct graph
            else:
                results[a, b] = delayed(g)(a, b)  # without structure

    results = compute(delayed(results))  # trigger all computation



### NumPy

<img src="images/numpy-inverted.svg">


### Dask.Array

<img src="images/dask-array-inverted.svg">


### Pandas

<img src="images/pandas-inverted.svg">


### Dask.DataFrame

<img src="images/dask-dataframes-inverted.svg" width="70%">


### Many problems don't fit into a

### "big array" or "big dataframe"

<hr>

### Fortunately the system that backs dask.array/dataframes

### can be used for general applications


### Python

<img src="images/python-inverted.svg">


### Dask

<img src="images/dask-arbitrary-inverted.svg">



### Custom Script

    filenames = ['mydata-%d.dat' % i for i in range(10)]
    data = [load(fn) for fn in filenames]

    reference = load_from_sql('sql://mytable')
    processed = [normalize(d, reference) for d in data]

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


### Custom Script

<img src="images/custom-etl-1.svg">

    data = [load(fn) for fn in filenames]


### Custom Script

<img src="images/custom-etl-2.svg">

    reference = load_from_sql('sql://mytable')


### Custom Script

<img src="images/custom-etl-3.svg">

    processed = [normalize(d, reference) for d in data]


### Custom Script

<img src="images/custom-etl-4.svg" width="70%">

    rolled = []
    for i in range(len(processed) - 2):
        a = processed[i]
        b = processed[i + 1]
        c = processed[i + 2]
        r = roll(a, b, c)
        rolled.append(r)


### Custom Script

<img src="images/custom-etl-5.svg">

    compared = []
    for i in range(20):
        a = random.choice(rolled)
        b = random.choice(rolled)
        c = compare(a, b)
        compared.append(c)


### Custom Script

<img src="images/custom-etl-6.svg">

    best = reduction(compared)


### This flexibility is novel and liberating

<hr>

### But it's also tricky to do well



### Contrast with High Level Parallelism

**Spark**

    outputs = collection.filter(predicate)
                        .groupBy(key)
                        .map(function)

<hr>

**SQL**

    SELECT city, sum(population)
    WHERE population > 1000000
    GROUP BY city

<hr>

**Matrices**

    solve(A.dot(A.T), x)


### Map - Shuffle - Reduce

<table>
<tr>
  <td>
    <img src="images/embarrassing.svg">
  </td>
  <td>
    <img src="images/shuffle.svg">
  </td>
  <td>
    <img src="images/reduction.svg">
  </td>
</tr>
</table>

*  Build optimized Map
*  Build optimized Shuffle
*  Build optimized Aggregations
*  Get a decent database-like project


### Many parallel problems don't fit this model


### Custom Pipelines

<img src="images/custom-etl-6.svg">


### ETL: Luigi

<img src="images/luigi.png" width="80%">

http://luigi.readthedocs.io/en/stable/


### ETL: Airflow

<img src="images/airflow.png" width="80%">

https://github.com/apache/incubator-airflow


### Efficient TimeSeries - Resample

<img src="images/resample.svg">

    df.value.resample('1w').mean()


### Efficient TimeSeries - Rolling

<img src="images/rolling.svg">

    df.value.rolling(100).mean()


### ND-Array - Sum

<img src="images/array-sum.svg">

    x = da.ones((15, 15), (5, 5))
    x.sum(axis=0)


### ND-Array - Transpose

<img src="images/array-xxT.svg">

    x = da.ones((15, 15), (5, 5))
    x + x.T


### ND-Array - Matrix Multiply

<img src="images/array-xdotxT.svg">

    x = da.ones((15, 15), (5, 5))
    x.dot(x.T + 1)


### ND-Array - Compound Operations

<img src="images/array-xdotxT-mean.svg">

    x = da.ones((15, 15), (5, 5))
    x.dot(x.T + 1) - x.mean()


### ND-Array - Compound Operations

<img src="images/array-xdotxT-mean-std.svg">

    x = da.ones((15, 15), (5, 5))
    (x.dot(x.T + 1) - x.mean()).std()


### Modern SVD

<img src="images/svd.svg" width="45%">

    u, s, v = da.linalg.svd(x)


### Modern Approximate SVD

<img src="images/svd-compressed.svg">

    u, s, v = da.linalg.svd_compressed(x, k=10)


### Arbitrary graph execution eases developer burden

<hr>

### It enables algorithms and custom applications

<hr>

### But it's hard to do well


### Arbitrary Graph Scheduling is Hard

<img src="images/svd-compressed.svg" width="60%">

<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">


### Optimal Graph Scheduling is NP-Hard

<img src="images/svd-compressed.svg" width="60%">

<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">


### Scalable Scheduling Requires Linear Time Solutions

<img src="images/svd-compressed-large.svg" width="100%">

<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">


### Fortunately we can do pretty well with heuristics



### Dask.array/dataframe/delayed author task graphs

<hr>

<img src="images/grid_search_schedule-0.png" width="100%">

<hr>

### Now we need to run them efficiently


### Dask.array/dataframe/delayed author task graphs

<hr>

<img src="images/grid_search_schedule.gif" width="100%">

<hr>

### Now we need to run them efficiently


### Dask schedulers target different architectures

<hr>

### Easy swapping enables scaling up *and down*


### Single Machine Scheduler

Stable for a year or so.  Optimized for larger-than-memory use.

*   **Parallel CPU**: Uses multiple threads or processes
*   **Minimizes RAM**: Choose tasks to remove intermediates
*   **Low overhead:** ~100us per task
*   **Concise**: ~600 LOC, stable for ~12 months
*   **Real world workloads**: dask.array, xarray, dask.dataframe, dask.bag,
    Custom projects with dask.delayed


### Distributed Scheduler

<img src="images/scheduler-async-1.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-2.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-3.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-4.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-5.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-6.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-7.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-8.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-9.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-10.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-11.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-12.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-13.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-14.svg" width="90%">


### Distributed Scheduler

<img src="images/scheduler-async-15.svg" width="90%">


### Distributed Scheduler

*   **Distributed**: One scheduler coordinates many workers
*   **Data local**: Moves computation to correct worker
*   **Asynchronous**: Continuous non-blocking conversation
*   **Multi-user**: Several users share the same system
*   **HDFS Aware**: Works well with HDFS, S3, YARN, etc..
*   **Solidly supports**: dask.array, dask.dataframe, dask.bag, dask.delayed,
    concurrent.futures, ...
*   **Less Concise**: ~3000 LOC Tornado TCP application

    But all of the logic is hackable Python


### Easy to get started

    $ conda install dask distributed -c conda-forge
    $ pip install dask[complete] distributed --upgrade

<hr>

    >>> from dask.distributed import Executor
    >>> e = Executor()  # sets up local cluster

<hr>

    $ dask-scheduler

    $ dask-worker scheduler-hostname:8786
    $ dask-worker scheduler-hostname:8786



### Dask.array/dataframe/delayed author task graphs

<hr>

<img src="images/grid_search_schedule-0.png" width="100%">

<hr>

### Now we need to run them efficiently


### Dask.array/dataframe/delayed author task graphs

<hr>

<img src="images/grid_search_schedule.gif" width="100%">

<hr>

### Now we need to run them efficiently


### You can safely ignore the rest of this talk


### All decisions are done in-the-small (almost)

<img src="images/svd-compressed.svg" width="60%">
<img src="images/small-simple.svg" width="20%">

### All decisions are done in constant time (almost)


### Task Scheduling

<img src="images/fg-simple.svg">

    x = f(1)
    y = f(2)
    z = g(x, y)

<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">


### Which function to run first?

<img src="images/critical-path-1.svg">


### Prefer Tasks on Critical Path

<img src="images/critical-path-2.svg">


### Which function to run first?

<img src="images/expose-parallelism-1.svg">


### Expose Parallelism

<img src="images/expose-parallelism-2.svg">


### Which function to run first?

<img src="images/garbage-collection-1.svg">


### Release Data

<img src="images/garbage-collection-2.svg">


### Release Data, free Memory

<img src="images/garbage-collection-3.svg">


### Distributed Network

<img src="images/network-inverse.svg">


### Two Workers

<img src="images/scheduling-workers-1.svg">


### Two Workers

<img src="images/scheduling-workers-2.svg">


### Two Workers

<img src="images/scheduling-workers-3.svg">


### Two Workers

<img src="images/scheduling-workers-4.svg">


### Two Workers

<img src="images/scheduling-workers-5.svg">


### Data Locality

<img src="images/scheduling-workers-6.svg">


### Data Locality

<img src="images/scheduling-workers-7.svg">


### Data Locality

<img src="images/scheduling-workers-8.svg">


### .

<img src="images/scheduling-workers-9.svg">


### Minimize Communication

<img src="images/scheduling-workers-10.svg">


### Balance Computation and Communication

<img src="images/scheduling-workers-12.svg">


### .

<img src="images/scheduling-workers-13.svg">


### Work Steal

<img src="images/scheduling-workers-14.svg">


### Work Steal

<img src="images/scheduling-workers-15.svg">


### Intelligent scheduling requires measurement

*  Measure size of outputs in bytes (`__sizeof__`)
*  Measure process reported memory use
*  Measure computation time (EWMA, with restarts)
*  Measure communication time / network distance
*  Measure disk load time
*  ...


### Other Optimizations ...

*  Gracefully scale up or down based on load
*  Optionally compress messages based on small samples
*  Oversubscribe workers with many small tasks
*  Batch many-small-messages in 2ms windows
*  Spill unused data to disk
*  ...

<hr>

#### These optimizations suffice to make dask.array/dataframe fast
#### But they are not specific to arrays/dataframes
#### These optimizations apply to all situations, including novel ones


### Fine-grained scheduling requires constant-time decisions

*  Computational graphs scale out to 100,000s of tasks
*  We spend ~200us per task in the scheduler, 5000 tasks/s
*  Each task is ~1-10kB in RAM

### How do we make this fast?

*  Heavily indexed Pure Python data structures.
*  No classes, just bytestrings and dicts/sets/deques.

<img src="images/dicts-everywhere.jpg">



### Comparison to Spark

*Disclaimer: I am biased and ignorant.*

<hr>

*   Spark is firmly established, Dask is new
*   Language choice: Python (C/Fortran/LLVM)  or Scala (JVM)

<hr>

*   Spark focuses on SQL-like computations

    Dask focuses on generic computations

*   Spark is a monolithic framework

    Dask complements PyData


### Comparison to Spark

#### Spark

<table>
<tr>
  <td>
    <img src="images/embarrassing.svg">
  </td>
  <td>
    <img src="images/shuffle.svg">
  </td>
  <td>
    <img src="images/reduction.svg">
  </td>
</tr>
</table>


#### Dask

<img src="images/svd-compressed.svg" width="50%">


### Comparison to Airflow/Luigi/Celery

*  Dask is optimized for interactive computation
    *  10ms roundtrips
    *  200us overhead
    *  Inter-worker communication
*  Airflow/Luigi/Celery are optimized for ETL cases
    *  Cron functionality
    *  Expressive retry logic
    *  Batteries included for common problems
*  Dask could do this, but hasn't developed these niceties



### Visual Dashboards

<hr>

### Before you optimize performance you must understand performance


### Live Performance Dashboards

<img src="https://raw.githubusercontent.com/dask/dask-org/master/images/daskboard.gif"
     alt="Dask dashboard">


### Worker Statistics

<img src="images/worker-communications-fft.png"
     alt="worker communications"
     width="40%">
<img src="images/worker-state-fft.png"
     alt="worker state"
     width="40%">


### Scheduler Statistics

<img src="images/daskboard-scheduler-stealing.png"
     alt="Scheduler stealing dashboard"
     width="50%">


### Live Profile Plots

<iframe src="https://cdn.rawgit.com/mrocklin/52e1c411878fcdd64e04574877fe265e/raw/
98d9f38c51b250523e9c584779e74156ab14a4fe/task-stream-custom-etl.html"
        width="1000" height="600"></iframe>


### Live Profile Plots

<iframe src="https://cdn.rawgit.com/mrocklin/e09cad939ff7a85a06f3b387f65dc2fc/raw/
fa5e20ca674cf5554aa4cab5141019465ef02ce9/task-stream-image-fft.html"
        width="1000" height="600"></iframe>


*   **What Dask needed:**

    *   Customized / Bespoke Visuals
    *   Responsive real-time streaming updates
    *   Powerful client-side rendering (10k-100k elements)
    *   Easy to develop for non-web developers

*   **Bokeh**

    *   Python library for interactive visualizations on the web
    *   Use in a notebook, embed in static HTML, or use with Bokeh Server...  [example](http://bokeh.pydata.org/en/latest/docs/gallery/periodic.html)

*   **Bokeh Server**

    *   Bokeh Server maintains shared state between the Python server and web
        client

<img src="http://bokeh.pydata.org/en/latest/_static/images/logo.png">


### Setup Data Source

    from bokeh.models import ColumnDataSource
    tasks = ColumnDataSource({'start': [], 'stop': [], 'color': [],
                              'worker': [], 'name': []})

### Construct Plot around Data Source

    from bokeh.plotting import figure
    plot = figure(title='Task Stream')
    plot.rect(source=tasks, x='start', y='stop', color='color', y='worker')
    plot.text(source=tasks, x='start', y='stop', text='name')

### Push to Data Source on Server

    while True:
        collect_diagnostics_data()
        tasks.update({'start': [...], 'stop': [...], 'color': [...],
                      'worker': [...], 'name': [...]})

<img src="http://bokeh.pydata.org/en/latest/_static/images/logo.png">



### Final Slide:  Dask is ...

*   **Familiar:** Pandas and Numpy users find it easy to switch
*   **Flexible:** Handles arbitrary task graphs efficiently
*   **Well Founded:** Builds on the existing Python ecosystem
*   **Community Backed:** Involves core developers of other projects
*   **Accessible:** Just a Python library

<hr>

    $ pip install dask[complete] distributed --upgrade
    $ ipython
    >>> from distributed import Client
    >>> client = Client()  # starts a "cluster" on your local machine

<img src="images/grid_search_schedule.gif" width="100%">


### Acknowledgements

*  Countless open source developers
*  SciPy developer community
*  Continuum Analytics
*  XData Program from DARPA

<img src="images/moore.png">

<hr>

### Questions?

<img src="images/grid_search_schedule.gif" width="100%">
