Dask at AOSPy
-------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

Continuum Analytics


*   What are Dask and Dask.array?
*   Dask and Dask.array work on clusters


### What is Dask.array?

*  Numpy-like library
*  Converts computations into blocked algorithms on NumPy
*  Backs XArray for larger arrays


### Blocked algorithms

    x = da.ones((15, 15), chunks=(5, 5))
    x.sum(axis=1)

<img src="images/array-sum.svg">


### Blocked algorithms

    x = da.ones((15, 15), chunks=(5, 5))
    x + x.T

<img src="images/array-xxT.svg">


### Blocked algorithms

    x = da.ones((15, 15), chunks=(5, 5))
    x.dot(x.T + 1)

<img src="images/array-xdotxT.svg">


### Blocked algorithms

    x = da.ones((15, 15), chunks=(5, 5))
    (x.dot(x.T + 1) - x.mean(axis=0))

<img src="images/array-xdotxT-mean.svg">


### Blocked algorithms

    x = da.ones((15, 15), chunks=(5, 5))
    (x.dot(x.T + 1) - x.mean(axis=0)).std()

<img src="images/array-xdotxT-mean-std.svg">


### There are other mechanisms to produce graphs

### when arrays aren't flexible enough

<hr>

### Dask.dataframe, Dask.bag, Dask.delayed, concurrent.futures, ...



### What is Dask?

*  Generic task scheduler for computational loads
*  Runs Python functions in parallel with dependencies
*  Knows nothing about arrays, dataframes, etc..

<img src="images/grid_search_schedule-0.png">

*This happens to be a machine learning grid-search-pipeline problem, can you
tell?*


### What is Dask?

*  Generic task scheduler for computational loads
*  Runs Python functions in parallel with dependencies
*  Knows nothing about arrays, dataframes, etc..

<img src="images/grid_search_schedule.gif">

*This happens to be a machine learning grid-search-pipeline problem, can you
tell?*


### Dask now runs on a cluster

<hr>

### Dask.array/dataframe/bag can as well (and XArray?)



### Dask/distributed

*   Python-based distributed computing framework
*   Spark-like scaling and reliability
*   But for more complex computational workloads, like XArray

<hr>

### Setup

    user@host1$ dask-scheduler
    Starting Scheduler at 192.168.1.100:8786

    user@host2$ dask-worker 192.168.1.100:8786
    user@host3$ dask-worker 192.168.1.100:8786
    user@host4$ dask-worker 192.168.1.100:8786
    user@host5$ dask-worker 192.168.1.100:8786

    >>> from dask.distributed import Client
    >>> client = Client('192.168.1.100:8786')  # changes default scheduler

    >>> my_dask_collection.compute()  # now runs on cluster


### ... play time ...


### Final Thoughts

*   How do we make distributed arrays more accessible?
*   Are non-array workloads useful to this community?
*   How else can I help?

### Collaboration Points

*   Deploy distributed dask.arrays on datasets
*   Deploy dask.distributed on job schedulers (SGE, SLURM, LSF, ...)
*   Special case formats
*   ND-locality compression
*   Government money running out, harder to do free work
*   Please publish examples, make me aware of papers
