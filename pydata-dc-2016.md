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

    Dask.delayed, some machine learning

*  PyGotham - August 2016:

    Build DataFrame computations

*  PyData DC - October 2016:

    Fine-grained parallelism, motivation and performance


### NumPy

<img src="images/numpy-inverted.svg">


### Dask.Array

<img src="images/dask-array-inverted.svg">


### Pandas

<img src="images/pandas-inverted.svg">


### Dask.DataFrame

<img src="images/dask-dataframe-inverted.svg" width="70%">


### Many problems don't fit

### into a "big array" or "big dataframe"


### Python

<img src="images/python-inverted.svg">


### Dask

<img src="images/dask-arbitrary-inverted.svg">


### This flexibility is novel and liberating

### It's also tricky to do well


### High Level Parallelism

**Spark**

    outputs = collection.filter(predicate)
                        .groupby(key)
                        .map(function)

<hr>

**SQL**

    SELECT city, sum(value)
    WHERE value > 0
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


### Other Patterns

<table>
<tr>
  <td>
    <img src="images/structured.svg">
  </td>
  <td>
    <img src="images/unstructured.svg">
  </td>
  <td>
    <img src="images/iterative.svg">
  </td>
</tr>
</table>


### Messy Parallelism

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


### Messy Parallelism

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


### Custom Script

<img src="images/custom-etl-1.svg">

    data = [load(fn) for fn in filenames]


### Custom Script

<img src="images/custom-etl-2.svg">

    reference = load_from_sql('sql://mytable')


### Custom Script

<img src="images/custom-etl-3.svg">

    processed = [process(d, reference) for d in data]


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


### Custom Script

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


### But first!  A flashy demo!



### Dask was designed for NumPy and Pandas

<hr>

### However, the lower-level parts solve messier problems

<hr>

<img src="images/async-comment.png">



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


### Task Scheduling

<img src="images/fg-simple.svg">

    x = f(1)
    y = f(2)
    z = g(x, y)

<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">


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

    $ conda install dask distributed
    $ pip install dask[complete] distributed --upgrade

<hr>

    >>> from dask.distributed import Executor
    >>> e = Executor()  # sets up local cluster

<hr>

    $ dask-scheduler

    $ dask-worker scheduler-hostname:8786
    $ dask-worker scheduler-hostname:8786


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


### Examples



### Concluding thoughts


### Dask provides parallel NumPy and Pandas

<hr>

### ... and it parallelizes custom algorithms

<hr>

### ... on single machines or clusters


### Schedulers are common, but hidden

*   Task scheduling is ubiquitous in parallel computing

    Examples: MapReduce, Spark, SQL, TensorFlow, Plasma

*   But raw task scheduler is rarely exposed

    Exceptions: Make, Luigi, Airflow

<img src="images/switchboard-operator.jpg" width="60%">


### Don't Parallelize if you don't have to

*  But I need speed ...
    *  Profile first
    *  Use C/Cython/Numba/Julia/...
    *  Use better algorithms, sample
*  But I need to scale ...
    *  Profile first
    *  Use better data structures, sample, stream
*  Yes, but I actually really need to ...
    *  Start with your laptop and concurrent.futures
    *  Then, move up to a heavy workstation
    *  Then, very reluctantly, move to a cluster


### Acknowledgements

*  Countless open source developers
*  SciPy developer community
*  Continuum Analytics
*  XData Program from DARPA

<img src="images/moore.png">

<hr>

### Questions?

<img src="images/grid_search_schedule.gif" width="100%">



<img src="https://zekeriyabesiroglu.files.wordpress.com/2015/04/ekran-resmi-2015-04-29-10-53-12.png"
     align="right"
     width="30%">

### Q: How does Dask differ from Spark?

*  Spark is great
    *  ETL + Database operations
    *  SQL-like streaming
    *  Spark 2.0 is decently fast
    *  Integrate with Java infrastructure
*  Dask is great
    *  Tight integration with NumPy, Pandas, Toolz, SKLearn, ...
    *  Ad-hoc parallelism for custom algorithms
    *  Easy deployment on clusters or laptops
    *  Complement the existing SciPy ecosystem (Dask is lean)
*  Both are great
    *  Similar network designs and scalability limits
    *  Decent Python APIs


### Schedulers are common, but hidden

*   Task scheduling is ubiquitous in parallel computing

    Examples: MapReduce, Spark, SQL, TensorFlow, Plasma

*   But raw task scheduler is rarely exposed

    Exceptions: Make, Luigi, Airflow

<img src="images/switchboard-operator.jpg" width="60%">


### Other Parallel Libraries

*  System
    *  threading, multiprocessing, concurrent.futures
    *  mpi4py, socket/zmq
*  MRJob, PySpark, Some SQL Databases, ...
*  Joblib, IPython Parallel,  ...
*  BLAS, Elemental, ...
*  <strike>Asyncio/Tornado</strike>
