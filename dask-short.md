Dask: Parallel Programming in Python
------------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

NVIDIA


### High Level: Dask scales other Python libraries

-  Pandas + Dask

        df = dask.dataframe.read_csv('s3://path/to/*.csv')
        df.groupby(df.timestamp.dt.hour).value.mean()

-  Numpy + Dask

        X = dask.array.random((100000, 100000), chunks=(1000, 1000))
        (X + X.T) - X.mean(axis=0)

-  Scikit-Learn + Dask + ...

        from dask_ml.linear_models import LogisticRegression

        model = LogisticRegression()
        model.fit(X, y)

-  ... and several other applicaitons throughout PyData


### Low Level: Dask is a task scheduler

Like `make`, but where each task is a short Python function

    (X + X.T) - X.mean(axis=0)  # Dask code turns into task graphs

<img src="images/grid_search_schedule-0.png" width="100%">


### Low Level: Dask is a task scheduler

Like `make`, but where each task is a short Python function

    (X + X.T) - X.mean(axis=0)  # Dask code turns into task graphs

<img src="images/grid_search_schedule.gif" width="100%">



### Ecosystem


### Python has a mature analytics stack (Numpy, Pandas, ...)

<hr>

### But it is restricted to RAM and a single CPU

### How do we parallelize an ecosystem?


<img src="images/scipy-stack/1.png">


<img src="images/scipy-stack/2.png">


<img src="images/scipy-stack/3.png">


<img src="images/scipy-stack/4.png">


<img src="images/scipy-stack/5.png">


### How do we parallelize an ecosystem

<hr>

### Filled with a wide variety of algorithms?

### Written in Python, C/C++, Fortran, CUDA, ...


### Parallelism Options Today

-  **Message Passing Interface (MPI)**
    -  **Good**: Algorithmically flexible, fast, native
    -  **Bad**: Difficult for non-experts, brittle
-  **Spark / MapReduce / Flink / ...**
    -  **Good**: Easy to use, resilient
    -  **Bad**: Not algorithmically flexible, typically Java/Scala based
-  **We want ...**
    -  Some of the ease and automation of Spark
    -  With some of flexibility and native support of MPI



### Parallelism Options Today


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



### Collections


### Dask enables parallel Python

<hr>

### ... originally designed to parallelize NumPy and Pandas

<hr>

### ... but also used today for arbitrary computations


### Dask.array

<img src="images/dask-array.svg" width="60%">

    import numpy as np
    x = np.random.random(...)
    y = x + x.T - x.mean(axis=0)

    import dask.array as da
    x = da.random.random(..., chunks=(1000, 1000))
    y = x + x.T - x.mean(axis=0)


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
    .
    .

    for a in A:
        for b in B:
            if a < b:
                results[a, b] = f(a, b)
            else:
                results[a, b] = g(a, b)

    .


### Fine Grained Python Code

    import dask

<hr>

    results = {}
    f = dask.delayed(f)  # mark functions as lazily evaluated
    g = dask.delayed(g)

    for a in A:
        for b in B:
            if a < b:
                results[a, b] = f(a, b)  # construct task graph
            else:
                results[a, b] = g(a, b)

    results = dask.compute(results)  # trigger computation



### Dask array graphs


### Dask APIs Produce Task Graphs

<hr>

### Dask Schedulers Execute Task Graphs


### 1D-Array

<img src="images/array-1d.svg">

    >>> x = np.ones((15,))
    >>> x
    array([ 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    >>> x = da.ones((15,), chunks=(5,))
    dask.array<ones, shape=(15,), dtype=float64, chunksize=(5,)>


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



### Schedulers


### Dask APIs Produce Task Graphs

<hr>

### Dask Schedulers Execute Task Graphs


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
*   **Overhead:** ~50us per task
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


### Cross-Language Support

-  Scheduler
    -   Coordinates everything

        Most of the difficult logic and code
    -   Language agnostic

        Communicates with msgpack and large bytestrings
        -  To support other languages ([Julia prototype](https://github.com/invenia/DaskDistributedDispatcher.jl) exists)
        -  May rewrite someday in Go/C++/PyPy
-  Workers/Clients
    -  Need to match software environments/language exactly
    -  Relatively simple codebase
        -  User API
        -  Data and function serialization
        -  Networking



### How the Scheudler Works


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

Very Stable.  Optimized for larger-than-memory use.

*   **Parallel CPU**: Uses multiple threads or processes
*   **Minimizes RAM**: Choose tasks to remove intermediates
*   **Overhead:** ~50us per task
*   **Concise**: ~600 LOC
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
*   **Deployable**: Easy to deploy on HPC, Hadoop, Cloud
*   **Less Concise**: ~3000 LOC Tornado TCP application

    But all of the logic is hackable Python


### Easy to get started

    $ conda install dask
    $ pip install dask[complete] --upgrade

<hr>

    >>> from dask.distributed import Client
    >>> client = Client()  # sets up local cluster

<hr>

    $ dask-scheduler

    $ dask-worker scheduler-hostname:8786
    $ dask-worker scheduler-hostname:8786



### Scheduling Heuristics


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



### Apache Spark


### Comparison to Apache Spark

-  Trust
    -  Spark is well established and trusted
    -  Dask is new, but part of the established PyData ecosystem
-  Framework vs Library
    -  Spark is an all-in-one framework
    -  Dask is a small part of the larger PyData ecosystem
-  JVM vs Python
    -  Spark is JVM based, with support for Python and R
    -  Dask is nicer for Python users, but supports no one else
-  High vs Low level
    -  Spark works at a high level of Map / Shuffle / Reduce stages
    -  Dask thinks at a lower level of individual task scheduling
-  Applications
    -  Spark is focused on SQL and BI applications
    -  Dask is less focused, and better at complex situations


### People choose Dask for two reasons

1.  They like Python
2.  Their use case is too complex for Apache Spark


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



### Machine Learning

[dask-ml.readthedocs.io](http://dask-ml.readthedocs.io/)


### Machine Learning: We have a few options ...

1.  Accelerate Scikit-Learn directly

    ```python
    pipe = Pipeline(steps=[('pca', PCA()),
                           ('logistic', LogisticRegression)])
    grid = GridSearchCV(pipe, parameter_grid)
    ```

2.  Build well-known algorithms with Dask.array

    ```python
    eXbeta = da.exp(X.dot(beta))
    gradient = X.T.dot(eXbeta / (eXbeta + 1) - y)
    ...
    ```

3.  Support and deploy other distributed systems

    <img src="images/dask-xgboost-pre.svg" width="40%">
    <img src="images/dask-xgboost-pre.svg" width="40%">

4.  Build custom algorithms with concurrent.futures, dask.delayed, ...


### Machine Learning: We have a few options ...

1.  Accelerate Scikit-Learn directly

    ```python
    pipe = Pipeline(steps=[('pca', PCA()),
                           ('logistic', LogisticRegression)])
    grid = GridSearchCV(pipe, parameter_grid)
    ```

2.  Build well-known algorithms with Dask.array

    ```python
    eXbeta = da.exp(X.dot(beta))
    gradient = X.T.dot(eXbeta / (eXbeta + 1) - y)
    ...
    ```

3.  Support and deploy other distributed systems side-by-side

    <img src="images/dask-xgboost-post.svg" width="40%">
    <img src="images/dask-xgboost-post.svg" width="40%">

4.  Build custom algorithms with concurrent.futures, dask.delayed, ...


### Accelerate Scikit-Learn directly with Joblib

-  Scikit-Learn uses [Joblib](https://pythonhosted.org/joblib/) for parallelism
-  Joblib now supports swapping backends
-  Can replace the normal thread pool with Dask

-  Thread Pool <-- Joblib <-- Scikit Learn

```python
from sklearn.model_selection import GridSearchCV
.
.

est = GridSearchCV(...)  # this could be any joblib-parallelized estimator

est.fit(X, y)  # uses a thread pool
```


### Accelerate Scikit-Learn directly with Joblib

-  Scikit-Learn uses [Joblib](https://pythonhosted.org/joblib/) for parallelism
-  Joblib now supports swapping backends
-  Can replace the normal thread pool with Dask

-  Dask Cluster <-- Joblib <-- Scikit Learn

```python
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import parallel_backend
import dask_ml.joblib

est = GridSearchCV(...)  # this could be any joblib-parallelized estimator
with parallel_backend('dask.distributed', scheduler_host='...'):
    est.fit(X, y)  # uses Dask
```


### Accelerate Scikit-Learn directly with Joblib

-  Good:
    -  model selection (grid search)
    -  embarrassingly parallel computations (random forests)
-  Bad:
    -  Training large data
    -  Still some backends baked into Scikit-Learn
-  Status:
    - Works well today
    - Will extend to new algorithms as Joblib evolves


### Use Dask Array to Build Optimization Algorithms

Implement optimization algorithms with NumPy syntax

<div class="columns">
  <div class="column">
    <pre>
Xbeta = X.dot(beta_hat)
func = ((y - Xbeta)\*\*2).sum()
gradient = 2 \* X.T.dot(Xbeta - y)

beta_hat = beta_hat - step_size \* gradient
new_func = ((y - X.dot(beta_hat)) \*\* 2).sum()
    </pre>

    <p> Dask.array provides scalable algorithms </p>
    <p> Easy for mathematical programmers </p>
  </div>

  <div class="column">
    <img src="images/grad-step-white-on-transparent.svg" width="100%">
  </div>
</div>


### Use Dask Array to Build Optimization Algorithms

```python
>>> from dask_ml.estimators import LogisticRegression
>>> from dask_ml.datasets import make_classification
>>> X, y = make_classification()
>>> lr = LogisticRegression()
>>> lr.fit(X, y)
>>> lr
LogisticRegression(abstol=0.0001, fit_intercept=True, lamduh=1.0,
                   max_iter=100, over_relax=1, regularizer='l2', reltol=0.01,
                                      rho=1, solver='admm', tol=0.0001)
```

-  Combine the following:
    -  Optimization algorithms with Dask.array
    -  Regularizers (L1, L2, ElasticNet, ...)
    -  Generalized Linear Model families
-  Get:
    -  Linear Regression
    -  Logistic Regression
    -  Poisson Regression
    -  ...


### Use Dask Array to Build Optimization Algorithms

-  Good:
    -  Train large datasets
    -  Extensible to new regularization methods, link functions
    -  Supports SKLearn API
-  Bad:
    -  Not as efficient as SKLearn on single machines
-  Status:
    -  Good to go
    -  Needs benchmarking on real problems

<hr>

*Work by Chris White and Tom Augspurger*


### Deploy Other Services with Dask

<div class="columns">
  <div class="column">
  <ul>
    <li>Other distributed machine learning systems exist</li>
    <li>Dask can deploy these and serve data</li>
  <ul>
  <pre>
import dask.dataframe as dd
df = dd.read_parquet('s3://...')

# Split into training and testing data
train, test = df.random_split([0.8, 0.2])

# Separate labels from data
train_labels = train.x > 0
test_labels = test.x > 0

del train['x']  # remove informative column from data
del test['x']  # remove informative column from data

.
.

.
.

.
  </pre>
  </div>

  <div class="column">
    <img src="images/network-inverse.svg" width="100%">
  </div>
</div>


### Deploy Other Services with Dask

<div class="columns">
  <div class="column">
  <ul>
    <li>Other distributed machine learning systems exist</li>
    <li>Dask can deploy these and serve data</li>
  <ul>
  <pre>
import dask.dataframe as dd
df = dd.read_parquet('s3://...')

# Split into training and testing data
train, test = df.random_split([0.8, 0.2])

# Separate labels from data
train_labels = train.x > 0
test_labels = test.x > 0

del train['x']  # remove informative column from data
del test['x']  # remove informative column from data

.
.

.
.

.
  </pre>
  </div>

  <div class="column">
    <img src="images/network-inverse-xgboost.svg" width="100%">
  </div>
</div>


### Deploy Other Services with Dask

<div class="columns">
  <div class="column">
  <ul>
    <li>Other distributed machine learning systems exist</li>
    <li>Dask can deploy these and serve data</li>
  <ul>
  <pre>
import dask.dataframe as dd
df = dd.read_parquet('s3://...')

# Split into training and testing data
train, test = df.random_split([0.8, 0.2])

# Separate labels from data
train_labels = train.x > 0
test_labels = test.x > 0

del train['x']  # remove informative column from data
del test['x']  # remove informative column from data

.
.

.
.

.
  </pre>
  </div>
  <div class="column">
    <img src="images/network-inverse-xgboost-connections.svg" width="100%">
  </div>
</div>


### Deploy Other Services with Dask

<div class="columns">
  <div class="column">
  <ul>
    <li>Other distributed machine learning systems exist</li>
    <li>Dask can deploy these and serve data</li>
  <ul>
  <pre>
import dask.dataframe as dd
df = dd.read_parquet('s3://...')

# Split into training and testing data
train, test = df.random_split([0.8, 0.2])

# Separate labels from data
train_labels = train.x > 0
test_labels = test.x > 0

del train['x']  # remove informative column from data
del test['x']  # remove informative column from data

# from xgboost import XGBRegressor  # change import
from dask_ml.xgboost import XGBRegressor

est = XGBRegressor(...)
est.fit(train, train_labels)

prediction = est.predict(test)
  </pre>
  </div>
  <div class="column">
    <img src="images/network-inverse-xgboost-connections.svg" width="100%">
  </div>
</div>


### Deploy Other Services with Dask

-  Good
    -  Works with XGBoost
    -  Works with TensorFlow
    -  Handles administrative setup
    -  Delivers distributed data
    -  Doesn't reinvent anything unnecessarily
-  Bad
    -  You still need to understand XGBoost
    -  You still need to understand TensorFlow
    -  Requires that the service plays nicely with Python
-  Status
    -  Very small projects
    -  Not heavily used, so expect some friction


### Machine Learning Overview

-  Dask enable parallel machine learning
    -  Uses existing technologies like SKLearn, XGBoost
    -  Implements new algorithms when necessary
-  Highly collaborative
-  Maintain familiar Scikit-Learn APIs

<hr>

-  See blogposts by [Tom Augspurger](https://tomaugspurger.github.io/)
    -  [Overview](https://tomaugspurger.github.io/scalable-ml-01.html)
    -  [Incremental Learning](https://tomaugspurger.github.io/scalable-ml-02.html)
    -  ...
-  And [Jim Crist](http://jcrist.github.io/)
    -  [Grid Search](http://jcrist.github.io/introducing-dask-searchcv.html)
-  And [Chris White](https://github.com/moody-marlin/)
    -  [Convex Optimization](https://matthewrocklin.com/blog/work/2017/03/22/dask-glm-1)
    -  [Asynchronous Algorithms](http://matthewrocklin.com/blog/work/2017/04/19/dask-glm-2)



### Final Slide:  Dask is ...

*   **Familiar:** Pandas and Numpy users find it easy to switch
*   **Flexible:** Handles arbitrary task graphs efficiently
*   **Well Founded:** Builds on the existing Python ecosystem
*   **Community Backed:** Involves core developers of other projects
*   **Accessible:** Just a Python library

<hr>

    $ pip install dask[complete] --upgrade
    $ ipython
    >>> from distributed import Client
    >>> client = Client()  # starts a "cluster" on your local machine

<img src="images/grid_search_schedule.gif" width="100%">


### Thanks!

    $ conda install dask
    $ pip install dask[complete] --upgrade

<img src="images/moore.png" width="20%">
<img src="images/Anaconda_Logo.png" width="20%">
<img src="images/NSF.png" width="10%">
<img src="images/DARPA_Logo.jpg" width="20%">

<hr>

### Questions?

<img src="images/grid_search_schedule.gif" width="100%">



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
