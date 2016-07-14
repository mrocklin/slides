Dask: Flexible Distributed Computing
------------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

*Jim Crist*

Continuum Analytics


### Dask provides parallel NumPy and Pandas

<hr>

### ... but ad-hoc parallel algorithms are exciting too

<hr>

### ... especially with intelligent scheduling


### DataFrame Example

[http://scipy2016.jupyter.org/](http://scipy2016.jupyter.org/)


### Dask was designed for NumPy and Pandas

<hr>

### However, the lower-level bits end up being quite useful


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


### dask.delayed

- Tool for creating arbitrary task graphs
- Dead simple interface (one function)
- Plays well with existing code (with some caveats)


- delayed(function)(\*args, \*\*kwargs) -> Delayed

- delayed(data) -> Delayed


###Examples


### Caveats

Can't use in control flow.

    # iterable in loop
    for i in delayed_object:
        ...

    # case in if statement
    if delayed_object:
        ...
    else:
        ...



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
    $ pip install dask distributed --upgrade

<hr>

    >>> from dask.distributed import Executor
    >>> e = Executor()  # sets up local cluster

<hr>

    $ dask-scheduler

    $ dask-worker scheduler-hostname:8786
    $ dask-worker scheduler-hostname:8786


### Machine Learning Example



### Dask provides parallel NumPy and Pandas

<hr>

### ... but ad-hoc parallel algorithms are exciting too

<hr>

### ... especially with intelligent scheduling


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


### Internals

*  Tornado web application over TCP sockets with custom protocol
*  Event driven (new worker, task finished, worker died, ...)
*  State is ~30 Python dictionaries indexing each other
*  Processes 1000s of tasks per second
*  Language agnostic (msgpack protocol)



### IT

    $ dask-scheduler
    Running scheduler at scheduler-hostname:8786 ...

    $ dask-worker scheduler-hostname:8786
    $ dask-worker scheduler-hostname:8786

<hr>

### User

    >>> from dask.distributed import Executor
    >>> e = Executor('scheduler-hostname:8786', set_as_default=True)

    >>> import dask.dataframe as dd
    >>> df = dd.read_csv('s3://my-bucket/2015-*.*.csv')
    >>> df.groupby(df.timestamp.dt.hour).value.mean().compute()
    .


### IT

    $ dask-scheduler
    Running scheduler at scheduler-hostname:8786

    $ dask-worker scheduler-hostname:8786
    $ dask-worker scheduler-hostname:8786

<hr>

### User

    >>> from dask.distributed import Executor
    >>> e = Executor('scheduler-hostname:8786', set_as_default=True)

    >>> import dask.array as da
    >>> x = da.from_array(my_distributed_array_store)
    >>> x = x - x.mean(axis=0) / x.std(axis=0)
    .


### IT

    $ dask-scheduler
    Running scheduler at scheduler-hostname:8786

    $ dask-worker scheduler-hostname:8786
    $ dask-worker scheduler-hostname:8786

<hr>

### User

    >>> from dask.distributed import Executor
    >>> e = Executor('scheduler-hostname:8786', set_as_default=True)

    >>> from dask import delayed
    >>> remote_data = e.scatter(sequence)
    >>> values = [delayed(f)(x) for x in remote_data]
    >>> total = delayed(sum)(values)
