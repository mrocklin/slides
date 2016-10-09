Dask: Fine Grained Task Parallelism
-----------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

Continuum Analytics


### Dask provides parallelism

<hr>

### ... originally designed for NumPy and Pandas

<hr>

### ... but used today for arbitrary computations


### Recent talks

*   SciPy - July, 2016:

    Overview, author custom algorithms, some machine learning

*  PyGotham - August 2016:

    Build DataFrame computations

*  *PyData DC - October 2016 (this talk)*:

    Fine-grained parallelism, scheduling motivation and heuristics



### In the beginning, there was NumPy ...

<hr>

### And it was good

### (except in parallel or out of RAM)


### NumPy

<img src="images/numpy-inverted.svg">


### Dask.Array

<img src="images/dask-array-inverted.svg">


### Pandas

<img src="images/pandas-inverted.svg">


### Dask.DataFrame

<img src="images/dask-dataframes-inverted.svg" width="70%">



### Many problems don't fit

### into a "big array" or "big dataframe"


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


### Spark and friends can compute on clusters efficiently

<hr>

### Airflow/Luigi can execute arbitrary graphs

<hr>

### Dask tries to do both



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
*  Get a decent database-like-thing


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



<img src="images/dask_horizontal_white.svg"
     alt="Dask logo"
     width="50%">

*  Dynamic task scheduler for arbitrary computations
*  Single machines or clusters
*  200 us overhead per task
*  Handles data locality, resilience, collaboration, etc..
*  Arrays, DataFrames, Lists, etc. built on top
*  Accessible Python

        pip install dask[complete] distributed --upgrade
        conda install -c conda-forge dask distributed



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


### Two Workers

<img src="images/scheduling-workers-1.svg">


### Two Workers

<img src="images/scheduling-workers-2.svg">


### Two Workers

<img src="images/scheduling-workers-3.svg">


### Two Workers

<img src="images/scheduling-workers-4.svg">


### Data Locality

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
*  Measure computation time (EWMA, with restarts)
*  Measure communication time / network distance
*  Measure disk load time
*  Measure process reported memory use
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

### Solution

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


### Final Thoughts

*   Dask provides parallelism for Python
    *   Parallel NumPy, Pandas, Scikit-Learn, etc..
    *   Built on an arbitrary computational task scheduler
*   Distributed scheduling of arbitrary graphs is hard
    *   Benefits from on-the-fly measurement
    *   Useful for ad-hoc situations


### Acknowledgements

*  Countless open source developers
*  SciPy developer community
*  Continuum Analytics
*  XData Program from DARPA

<img src="images/moore.png">

<hr>

### Questions?

<img src="images/grid_search_schedule.gif" width="100%">
