Parallelizing Python with Dask
------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

Continuum Analytics



### Python has a fast and pragmatic data science ecosystem

<hr>

### ... restricted to a single core


### How do we parallelize an ecosystem?

*  **NumPy**: arrays
*  **Pandas**: tables
*  **Scikit Learn**: machine learning
*  **Statsmodels**: statistics
*  ...
*  ...
*  **GeoPandas**: geo-spatial
*  **Scikit-Image**: image analysis
*  **Scikit-Bio**: ...


### How do we parallelize a complex ecosystem?

<hr>

### ... without rewriting everything?


<img src="images/dask_horizontal_white.svg"
     alt="Dask logo"
     width="50%">


<img src="images/gridsearch-lr.svg"
     alt="Dask machine learning gridsearch"
     align="right"
     width="25%">

### Dask is a task scheduler

*  Task scheduler like make or airflow
*  Designed for computation (like spark)
*  On a single machine or a cluster

### Dask has "big data" collections

*  Pandas + dask = dask.dataframe

        df.groupby(df.name).balance.mean()

*  NumPy + dask = dask.array

        x.T - x.mean(axis=0)

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

[Example notebook with NYC Taxi
data](https://gist.github.com/mrocklin/86764c5eaba5c23892430975ae3a983a#file-odsc-2016-dataframe-ipynb)


### From queries to task graphs

    >>> len(df)

<img src="images/df-len.svg" width=40%>

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
    Cumulative reductions

    <img src="images/iterative.svg">
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

[Example notebook with weather
data](https://gist.github.com/mrocklin/86764c5eaba5c23892430975ae3a983a#file-odsc-2016-meteorology-ipynb)


Dask bag
--------

*  Combines Python lists with task scheduling
*  Fairly standard approach
*  Builds off of the fast CyToolz library

.

    >>> import dask.bag as db
    >>> import json

    >>> records = db.read_text('path/to/data.*.json.gz').map(json.loads)

    >>> records.filter(...).pluck('name').frequencies().topk(10, ...)



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


### We originally made Dask flexibile to parallelize NumPy

<hr>

### Found that it was surprisingly useful for general work


### Task Scheduling API

    >>> from dask.distributed import Executor
    >>> e = Executor('scheduler-address')  # connect to cluster

    >>> x = e.submit(add, 1, 2)            # submit single task to cluster
    >>> x
    <Future: status=pending>


### Task Scheduling API

Call many times with for loops.

    >>> for i in ...                       # Use in complex loops
    ...     for j in ...
    ...         if ...
    ...             e.submit(func, ...)

Submit tasks on results of other tasks.

    >>> x = e.submit(add, 1, 2)            # submit single task to cluster
    >>> y = e.submit(mul, x, 10)           # some tasks depend on others

[Example notebook submitting custom
tasks](https://gist.github.com/mrocklin/f57bc107a9eb5fe965175d4b507a1bf1#file-odsc-2016-custom-futures-ipynb)


### Build graphs locally, submit all at once

<img src="images/fg-simple.svg" align=right>

    @dask.delayed
    def f(x):
        return ...

    @dask.delayed
    def g(x, y):
        return ...

    x = f(1)
    y = f(2)
    z = g(x, y)

    z.compute()

[Example notebook with scikit
learn](https://gist.github.com/mrocklin/86764c5eaba5c23892430975ae3a983a#file-odsc-2016-sklearn-ipynb)



### Wrapping up


### Flexibility opens up access to messy problems

<hr>

### Dask provides flexibility in parallelism


Things we didn't cover
----------------------

*   Resilience and elasticity
*   Multi-client collaboration
*   Deployment on Yarn, Docker, SGE, SSH, single thread
*   Networking:  Tornado TCP application. Fast protocol.
*   How to avoid parallelism and clusters
*   Single-machine larger-than-memory use


Dask is not...
----------------

*  **A Database:**
    *  No query planner (only low-level optimizations)
    *  No shuffle (some groupbys and hash joins a problem)
*  **MPI:**
    *  Central dynamic scheduler
    *  100s of microseconds overhead per task


<img src="images/gridsearch-lr.svg"
     alt="Dask machine learning gridsearch"
     align="right"
     width="20%">

Dask is...
----------

*  **Familiar:** Implements NumPy/Pandas interfaces
*  **Flexible:** for sophisticated and messy algorithms
*  **Fast:** Optimized for demanding applications
*  **Scales up:** Runs resiliently on clusters
*  **Scales down:** Pragmatic on a laptop
*  **Responsive:** for interactive computing

<hr>

Dask **complements** the Python ecosystem.

It was developed with NumPy, Pandas, and Scikit-Learn developers.


Questions?
----------

*  [dask.pydata.org](http://dask.pydata.org/en/latest/),
   [distributed.readthedocs.org](http://distributed.readthedocs.org/en/latest/)
*  [gitter.im/dask/dask](http://gitter.im/dask/dask/),
   [youtube channel](https://www.youtube.com/playlist?list=PLRtz5iA93T4PQvWuoMnIyEIz1fXiJ5Pri)

[@mrocklin](http://twitter.com/mrocklin)

<hr>

Start on a single machine

    $ conda/pip install dask

    >>> import dask.bag as db
    >>> db.read_text('/path/to/*.json.gz').filter(...)

<hr>

Start a cluster on EC2

    $ conda/pip install dask distributed dec2
    $ dec2 --keyname AWS-KEYNAME
           --keypair ~/.ssh/AWS-KEYPAIR.pem
           --count 10 --type m4.2xlarge
           --notebook



### Extras


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


How dask is used in practice
----------------------------

*  Large arrays for climate and atmospheric science (HDF5 data)
*  Single machine lightweight PySpark clone for logs and JSON
*  Dataframes on piles of CSV data
*  Custom applications

<hr>

*  Roughly equal mix of academic/research and corporate

