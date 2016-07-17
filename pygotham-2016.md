Dask: Parallelism From the Ground Up
------------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

Continuum Analytics


### Dask provides parallel NumPy and Pandas

<hr>

### ... and it parallelizes custom algorithms

<hr>

### ... on single machines or clusters


### But first!  A flashy demo!



### Dask was designed for NumPy and Pandas

<hr>

### However, the lower-level parts solve messier problems


### Dask Stack

<img src="images/dask-stack-0.svg" width="70%">


### Dask Stack

<img src="images/dask-stack-1.svg" width="70%">


### Dask Stack

<img src="images/dask-stack-2.svg" width="70%">


### Dask Stack

<img src="images/dask-stack-3.svg" width="70%">


### Dask Stack

<img src="images/dask-stack-4.svg" width="70%">


### Dask Stack

<img src="images/dask-stack-5.svg" width="70%">


### Messy Parallelism

    .
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
    .

<hr>

    results = {}

    for a in A:
        for b in B:
            if a < b:
                results[a, b] = delayed(f)(a, b)
            else:
                results[a, b] = delayed(g)(a, b)

    results = compute(delayed(results))  # trigger all computation


### dask.delayed

*  Capture a single function evaluation:

        value = add(1, 2)
        lazy_value = delayed(add)(1, 2)

*  Link multiple tasks together:

        x = delayed(f)(1)
        y = delayed(f)(2)
        z = delayed(g)(x, y)

    <img src="images/fg-simple.svg" align="right">

*  Combine with loops:

        x = [delayed(f)(i) for i in range(100)]


### Example



### Dask.delayed authors arbitrary task graphs

<hr>

<img src="images/grid_search_schedule-0.png" width="100%">

<hr>

### Now we need to run them efficiently


### Dask.delayed authors arbitrary task graphs

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


### Distributed Scheduler (new!)

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


### Examples



## Concluding thoughts


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
    *  Move up to a heavy workstation
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
