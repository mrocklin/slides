Parallelizing Python with Dask
------------------------------

*Matthew Rocklin*

Continuum Analytics


### Python has blazingly fast data science ecosystem

<hr>

### ... restricted to a single core


### How do we parallelize an ecosystem?

*  **NumPy**: arrays
*  **Pandas**: in-memory tables
*  **Scikit Learn**: machine learning
*  ...
*  ...
*  **Statsmodels**: statistics
*  **GeoPandas**: geo-spatial
*  **Scikit-Image**: image analysis
*  **Scikit-Bio**:


### How do we parallelize a complex ecosystem?

<hr>

### ... without rewriting everything?


<img src="images/gridsearch-lr.svg"
     alt="Dask machine learning gridsearch"
     align="right"
     width="25%">

### Dask core

*  Task scheduler (like airflow)
*  Designed for flexible algorithms
*  On a single machine or a cluster

### Dask collections

*  NumPy + dask = dask.array

        x.T - x.mean(axis=0)

*  Pandas + dask = dask.dataframe

        df.groupby(df.name).balance.mean()

*  Lists + dask = dask.bag

        b.map(json.loads).filter(...)


### Dask DataFrame

<img src="images/dask-dataframe-inverted.svg" align="right" width=30%>

*  Borrows heavily from Pandas
    *  Composed of Pandas DataFrames
    *  Matches the Pandas interface
*  Accesses data from HDFS, S3, local disk
*  Fast, low latency
*  Responsive user interface


### From queries to task graphs

<img src="images/df-len.svg" align="right" width=40%>

`len(df) -> `

*   Task scheduling is ubiquitous in parallel computing

    Examples: MapReduce, Spark, SQL, TensorFlow, Plasma


### Wide variety of algorithm types

<table>
<tr>
  <td>
    Map

    <img src="images/embarrassing.svg">
  </td>
  <td>
    Shuffle

    <img src="images/shuffle.svg">
  </td>
  <td>
    Reduce

    <img src="images/reduction.svg">
  </td>
</tr>
<tr>
  <td>
    Nearest Neighbor

    <img src="images/structured.svg">
  </td>
  <td>
    <h2>...</h2>
  </td>
  <td>
    Unstructured

    <img src="images/unstructured.svg">
  </td>
</tr>
</table>


Dask Arrays
-----------

*  Combines NumPy with task scheduling
*  Coordinate many NumPy arrays into single logical Dask array
*  Blocked algorithms implement broad subset of Numpy

<img src="images/dask-array.svg"
     alt="Dask array is built from many numpy arrays"
     width="70%">


### Dask core: dynamic task scheduling


Task Scheduling
---------------

<img src="images/fg-simple.svg">

    x = f(1)
    y = f(2)
    z = g(x, y)

<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">

Where and when do we run tasks?


Task Scheduling
---------------

*   Task scheduling is ubiquitous in parallel computing

    Examples: MapReduce, Spark, SQL, TensorFlow, Plasma

*   But raw task scheduler is rarely exposed

    Exceptions: Make, Luigi, Airflow

<img src="images/switchboard-operator.jpg" width="60%">



### Built flexibile system to parallelize NumPy

<hr>

### But found that analysts value it for general computation


*Demonstration: small run tasks directly on cluster*

    >>> e = Executor('scheduler-address')  # connect to cluster

    >>> x = e.submit(add, 1, 2)            # submit single task to cluster
    >>> x
    <Future: status=pending>

    >>> y = e.submit(mul, x, 10)           # submit dependent tasks
    >>> e.gather(y)                        # Gather data locally
    30

    >>> for i in ...                       # Use in complex loops
    ...     for j in ...
    ...         if ...
    ...             e.submit(func, ...)


*Demonstration: small run tasks directly on cluster*

    # Minimal API
    e = Executor('scheduler-address')    # connect to cluster
    e.submit(function, *args, **kwargs)  # submit single task
    e.submit(function, future, future)   # submit dependent tasks
    e.gather(futures)                    # Gather data locally

    # Full API
    e.map(function, sequence)            # submit many tasks
    e.map(function, queue)               # submit stream of tasks
    e.compute(collection)                # Submit full graph
    e.scatter(data)                      # Scatter data out to cluster
    e.rebalance(futures)                 # Move data around
    e.replicate(futures)                 # Move data around
    e.cancel(futures)                    # Cancel tasks
    e.upload_file('myscript.py')         # Send code or files


### Flexibility enables Sophisticated Algorithms

*  Parametrized machine learning pipeline

<img src="images/pipeline.svg" alt="Dask machine learning pipeline">


### Flexibility enables Sophisticated Algorithms

*  Embarrassingly parallel gridsearch

<img src="images/pipeline.svg" alt="Dask machine learning pipeline">

<img src="images/pipeline.svg" alt="Dask machine learning pipeline">

<img src="images/pipeline.svg" alt="Dask machine learning pipeline">


### Flexibility enables Sophisticated Algorithms

*  Efficient gridsearch

<a href=images/gridsearch-lr-black-on-white.pdf>
<img src="images/gridsearch-lr.svg"
     alt="Dask machine learning gridsearch"
     width="40%">
</a>


### Sophisticated algorithms defy structure

<hr>

### Dask enables algorithms through raw task scheduling


### Wrapping up


Dask is...
----------

*  **Familiar:** Implements parallel NumPy and Pandas objects
*  **Flexible:** for sophisticated and messy algorithms
*  **Fast:** Optimized for demanding algorithms
*  **Scales up:** Runs resiliently on clusters of 100s of machines
*  **Scales down:** Pragmatic in a single process on a laptop
*  **Interactive:** Responsive and fast for interactive computing

<hr>

Dask **complements** the Python ecosystem.  It was developed with NumPy,
Pandas, and Scikit-Learn developers.


Dask is not...
----------------

*  **A Database:**
    *  No query planner (only low-level optimizations)
    *  No shuffle (some groupbys and hash joins a problem)
*  **MPI:**
    *  Central dynamic scheduler
    *  100s of microseconds overhead per task


How dask is used in practice
----------------------------

*  Large arrays for climate and atmospheric science (HDF5 data)
*  Single machine lightweight PySpark clone for logs and JSON
*  Dataframes on piles of CSV data
*  Custom applications

<hr>

*  Roughly equal mix of academic/research and corporate


Lessons learned
---------------

*   Task scheduling complements existing ecosystems well

    Users can handle more control if you give it to them

*   Move quickly by embracing existing projects and communities

*   Wide variety of parallel computing problems out there


Questions?
----------

### [dask.pydata.org](http://dask.pydata.org/en/latest/)

<hr>

Start on a single machine

    $ pip install dask

    >>> import dask.bag as db
    >>> db.read_text('/path/to/*.json.gz').filter(...)

<hr>

Start a cluster on EC2

    $ pip install dask distributed dec2
    $ dec2 --keyname mrocklin
           --keypair ~/.ssh/keypair.pem
           --count 20 --type m4.2xlarge
