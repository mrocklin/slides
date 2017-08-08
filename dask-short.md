Dask: Parallel Programming in Python
------------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

Continuum Analytics


### Dask enables parallel computing

-  **Parallelizes libraries** like Pandas, NumPy, and SKLearn
-  **Scales** from 1 to 1000's of computers (Spark-like scaling)
-  **Flexible** backed by a task scheduler (like Airflow, Celery)
-  **Adapts** to custom systems
-  **Pure Python** and built from standard technology
-  **Supported** by community, for/non-profit, and government


### History

1.  Parallel NumPy algorithms
2.  Computational task scheduler (single machine)
3.  Dataframes and Bags
4.  Custom computations (dask.delayed)
5.  Distributed scheduler
6.  Asynchronous workflows (concurrent.futures)
7.  Increased diversity of workloads
    -  Auto-scaling
    -  Multi-client collaboration
    -  Other languages (Julia client exists)
    -  Non-task-based APIs



### Parallelism in Python

Sequential code

    data = [...]
    results = []
    for x in data:
        result = func(x)
        results.append(result)

Map (also sequential)

    results = map(func, data)

Parallel map

    from multiprocessing import Pool
    pool = Pool()

    results = pool.map(func, data)


### Parallelism in Python

Sequential code

    data = [...]
    results = []
    for x in data:
        result = func(x)
        results.append(result)

Map (also sequential)

    results = map(func, data)

Parallel map

    from concurrent.futures import ProcessPoolExecutor
    executor = ProcessPoolExecutor()

    results = executor.map(func, data)


### Parallelism in Python

Sequential code

    data = [...]
    results = []
    for x in data:
        result = func(x)
        results.append(result)

Map (also sequential)

    results = map(func, data)

Parallel map

    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor()

    results = executor.map(func, data)


### Parallelism in Python

Sequential code

    data = [...]
    results = []
    for x in data:
        result = func(x)
        results.append(result)

Map (also sequential)

    results = map(func, data)

Parallel map

    import pyspark
    sc = pyspark.SparkContext()

    results = sc.parallelize(data).map(func).collect()


### More complex code

    .
    .
    .
    .
    results = []
    for x in L1:
        for y in L2:
            if x < y:
                z = f(x, y)
            else:
                z = g(x, y)
            results.append(z)

    .


### More complex code

    import dask
    f = dask.delayed(f)
    g = dask.delayed(g)

    lazy = []
    for x in L1:
        for y in L2:
            if x < y:
                z = f(x, y)
            else:
                z = g(x, y)
            lazy.append(z)

    results = dask.compute(*lazy)


### More complex code

    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor()
    .

    futures = []
    for x in L1:
        for y in L2:
            if x < y:
                z = executor.submit(f, x, y)
            else:
                z = executor.submit(g, x, y)
            futures.append(z)

    results = [future.result() for future in futures]


### Scalability and Flexibility

*  Big data systems are scalable but rarely flexible.  Also inefficient.
*  Single-machine systems are flexible and convenient

<a href="https://pbs.twimg.com/media/C2162quUsAAXvin.jpg"><img src="https://pbs.twimg.com/media/C2162quUsAAXvin.jpg" width="50%"></a>



### Dask enables parallel Python

<hr>

### ... originally designed to parallelize NumPy and Pandas

<hr>

### ... but also used today for arbitrary computations


### Dask.array

<img src="images/dask-array.svg" width="60%">

    import numpy as np
    x = np.random.random(...)
    u, s, v = np.linalg.svd(x.dot(x.T))

    import dask.array as da
    x = da.random.random(..., chunks=(1000, 1000))
    u, s, v = da.linalg.svd(x.dot(x.T))


### Dask.DataFrame

<img src="images/dask-dataframe-inverted.svg" width="30%">

    import pandas as pd
    df = pd.read_csv('myfile.csv', parse_dates=['timestamp'])
    df.groupby(df.timestamp.dt.hour).value.mean()

    import dask.dataframe as dd
    df = dd.read_csv('hdfs://myfiles.*.csv', parse_dates=['timestamp'])
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

    results = compute(results)  # trigger all computation



### Dask APIs Produce Task Graphs

<hr>

### Dask Schedulers Execute Task Graphs

<img src="images/collections-schedulers-inverse.png"
     width="70%">


### 1D-Array

<img src="images/array-1d.svg">

    >>> np.ones((15,))
    array([ 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    >>> x = da.ones((15,), chunks=(5,))


### 1D-Array

<img src="images/array-1d-sum.svg" width="30%">

    x = da.ones((15,), chunks=(5,))
    x.sum()


### ND-Array - Sum

<img src="images/array-sum.svg">

    x = da.ones((15, 15), chunks=(5, 5))
    x.sum(axis=0)


### ND-Array - Transpose

<img src="images/array-xxT.svg">

    x = da.ones((15, 15), chunks=(5, 5))
    x + x.T


### ND-Array - Matrix Multiply

<img src="images/array-xdotxT.svg">

    x = da.ones((15, 15), chunks=(5, 5))
    x.dot(x.T + 1)


### ND-Array - Compound Operations

<img src="images/array-xdotxT-mean.svg">

    x = da.ones((15, 15), chunks=(5, 5))
    x.dot(x.T + 1) - x.mean()


### ND-Array - Compound Operations

<img src="images/array-xdotxT-mean-std.svg">

    import dask.array as da
    x = da.ones((15, 15), chunks=(5, 5))
    y = (x.dot(x.T + 1) - x.mean()).std()


### Dask APIs Produce Task Graphs

<hr>

### Dask Schedulers Execute Task Graphs

<img src="images/collections-schedulers-inverse.png"
     width="70%">


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


<img src="http://dask.pydata.org/en/latest/_images/dask_horizontal_white.svg"
     alt="dask logo"
     width="40%">

<img src="images/grid_search_schedule.gif" width="100%">

-  Dynamic task scheduler for generic applications
-  Handles data locality, resilience, work stealing, etc..
-  With 10ms roundtrip latencies and 200us overheads
-  Native Python library respecting Python protocols
-  Lightweight and well supported


### Single Machine Scheduler

Optimized for larger-than-memory use.

*   **Parallel CPU**: Uses multiple threads or processes
*   **Minimizes RAM**: Choose tasks to remove intermediates
*   **Low overhead:** ~100us per task
*   **Concise**: ~1000 LOC
*   **Real world workloads**: Under heavy load by many different projects


### Distributed Scheduler

*   **Distributed**: One scheduler coordinates many workers
*   **Data local**: Moves computation to correct worker
*   **Asynchronous**: Continuous non-blocking conversation
*   **Multi-user**: Several users share the same system
*   **HDFS Aware**: Works well with HDFS, S3, YARN, etc..
*   **Solidly supports**: dask.array, dask.dataframe, dask.bag, dask.delayed,
    concurrent.futures, ...
*   **Less Concise**: ~5000 LOC Tornado TCP application

    All of the logic is hackable Python, separate from Tornado


### Distributed Network

<img src="images/network-inverse.svg">


### Distributed Network

Set up locally

    from dask.distributed import Client
    client = Client()  # set up local scheduler and workers

Set up on a cluster

    host1$ dask-scheduler
    Starting scheduler at 192.168.0.1:8786

    host2$ dask-worker 192.168.0.1:8786
    host3$ dask-worker 192.168.0.1:8786
    host4$ dask-worker 192.168.0.1:8786



### Brief and Incomplete Summary of Parallelism Options

-  Embarrassingly parallel systems (multiprocessing, joblib)
-  Big Data collections (MapReduce, Spark, Flink, Database)
-  Task schedulers (Airflow, Luigi, Celery, Make)


### map

    # Sequential Code
    data = [...]
    output = map(func, data)

<hr>

    # Parallel Code
    pool = multiprocessing.Pool()
    output = pool.map(func, data)

-   Pros
    -   Easy to install and use in the common case
    -   Lightweight dependency
-   Cons
    -  Data interchange cost
    -  Not able to handle complex computations


### Big Data collections

    from pyspark import SparkContext
    sc = SparkContext('local[4]')

    rdd = sc.parallelize(data)
    rdd.map(json.loads).filter(...).groupBy(...).count()

    df = spark.read_json(...)
    df.groupBy('name').aggregate({'value': 'sum'})

-   Pros
    -   Larger set of operations
    -   Scales nicely on clusters
    -   Well trusted by enterprise
-   Cons
    -  Heavyweight and JVM focused
    -  Not able to handle complex computations


### This is what I mean by complex

<img src="images/array-xdotxT-mean-std.svg">

### Spark does the following well

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


### Task Schedulers (Airflow, Luigi, Celery, ...)

<img src="images/airflow.png" width="40%">
<img src="images/luigi.png" width="40%">

-  Pros
    -  Handle arbitrarily complex task graphs
    -  Python Native
-  Cons
    -  No inter-worker storage or data interchange
    -  Long latencies (relatively)
    -  Not designed for computational loads


### Want a task scheduler (like Airflow, Luigi)

<hr>

### Built for computational loads (like Spark, Flink)



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

    x = da.ones((15, 15), chunks=(5, 5))
    x.sum(axis=0)


### ND-Array - Transpose

<img src="images/array-xxT.svg">

    x = da.ones((15, 15), chunks=(5, 5))
    x + x.T


### ND-Array - Matrix Multiply

<img src="images/array-xdotxT.svg">

    x = da.ones((15, 15), chunks=(5, 5))
    x.dot(x.T + 1)


### ND-Array - Compound Operations

<img src="images/array-xdotxT-mean.svg">

    x = da.ones((15, 15), chunks=(5, 5))
    x.dot(x.T + 1) - x.mean()


### ND-Array - Compound Operations

<img src="images/array-xdotxT-mean-std.svg">

    x = da.ones((15, 15), chunks=(5, 5))
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

<hr>

<img src="images/collections-schedulers.png" width=50%>


### Start with a single machine

    import dask.dataframe as dd
    df = dd.read_csv('/path/to/*.csv')
    df.groupby(df.timestamp.dt.month).value.var().compute()

### Connect to a cluster later

    from dask.distributed import Client
    client = Client('scheduler-address:8786')

    df = dd.read_csv('hdfs:///path/to/*.csv')
    df.groupby(df.timestamp.dt.month).value.var().compute()


### Start with a single machine

    import dask.dataframe as dd
    df = dd.read_csv('/path/to/*.csv')
    df.groupby(df.timestamp.dt.month).value.var().compute()

### Connect to a cluster later

    from dask.distributed import Client
    client = Client('scheduler-address:8786')

    df = dd.read_csv('s3:///path/to/*.csv')
    df.groupby(df.timestamp.dt.month).value.var().compute()


### Start with a single machine

    import dask.dataframe as dd
    df = dd.read_csv('/path/to/*.csv')
    df.groupby(df.timestamp.dt.month).value.var().compute()

### Connect to a cluster later

    from dask.distributed import Client
    client = Client('scheduler-address:8786')

    import dask.dataframe as dd
    df = dd.read_custom('internal://db/project')
    df.groupby(df.timestamp.dt.month).value.var().compute()


### Single Machine Scheduler

Optimized for larger-than-memory use.

*   **Parallel CPU**: Uses multiple threads or processes
*   **Minimizes RAM**: Choose tasks to remove intermediates
*   **Low overhead:** ~50us per task
*   **Concise**: ~600 LOC, stable for ~12 months

### Distributed Scheduler

Optimized for 10-1000 machine clusters

*   **Distributed**: One scheduler coordinates many workers
*   **Data local**: Moves computation to correct worker
*   **Asynchronous**: Continuous non-blocking conversation
*   **Multi-user**: Several users share the same system
*   **Hackable**: all of the logic is hackable Python


### Easy to get started

    $ conda install dask distributed -c conda-forge
    or
    $ pip install dask[complete] distributed --upgrade

<hr>

    computer1:$ dask-scheduler

    computer2:$ dask-worker scheduler-hostname:8786
    computer3:$ dask-worker scheduler-hostname:8786

<hr>

    >>> from dask.distributed import Client
    >>> client = Client('scheduler-hostname:8786')

    >>> dask.compute(...)



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

    >>> from dask.distributed import Client
    >>> client = Client()  # sets up local cluster

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

-  Reasons to prefer Spark
    -   More established
    -   All-in-one framework for clusters
    -   Full SQL support plus extensions
    -   Complements existing JVM infrastructure
-  Reasons to prefer Dask
    -   Grows out of existing Python stack
    -   Familiar to Python users and applications
    -   Supports more complex computations
    -   Integrates nicely into existing systems, lightweight


### Spark

<table>
<tr>
<td>Map</td>
<td>Shuffle</td>
<td>Reduce</td>
</tr>
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

### Dask

<img src="images/array-xxT.svg" width="40%">
<img src="images/array-xdotxT-mean-std.svg" width="50%">


### Comparison to Airflow/Luigi/Celery

*  Airflow/Luigi/Celery are optimized for ETL cases
    *  Cron functionality
    *  Expressive retry logic
    *  Batteries included for common problems
*  Dask is optimized for interactive computation
    *  10ms roundtrips
    *  200us overhead
    *  Inter-worker communication



### Visual Dashboards

<hr>

### Optimizing performance requires understanding performance


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



### Dask enables Machine Learning

1.  Model parallelism with Scikit-Learn

    ```python
    pipe = Pipeline(steps=[('pca', PCA()),
                           ('logistic', LogisticRegression)])
    grid = GridSearchCV(pipe, parameter_grid)
    ```

2.  Implement known algorithms with dask.array

    ```python
    eXbeta = da.exp(X.dot(beta))
    gradient = X.T.dot(eXbeta / (eXbeta + 1) - y)
    ...
    ```

3.  Collaborate with other distributed systems

    -  **Pre-process** with dataframe
    -  **Deploy** other services
    -  **Pass data** from Dask and **train** with other service

4.  Build custom systems with dask.delayed, concurrent.futures



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
