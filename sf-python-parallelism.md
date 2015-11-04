Python and Parallelism
---------------

*Matthew Rocklin*

Continuum Analytics


### Python has a sophisticated analytics stack

<hr>

### Computers have many cores and fast hard drives


Outline
-------

0.  Don't use parallelism
1.  Multiprocessing and log files
2.  Threading and numpy arrays
3.  Complex analytic workloads
4.  Dynamic task scheduling and Dask
5.  Larger than memory arrays / dataframes
6.  Distributed Computing


0.  Don't use parallelism
-------------------------

*  Explore APIs
*  Write C/Cython/Numba
*  Store your data in nice formats
*  Use smarter algorithms


### 1.  Multiprocessing log files

    def process(filename):
        with open(filename) as f:
            lines = f.readlines()

        output = ...  # do work here

        with open(filename.replace('.log', '.out'), 'w') as f:
            for line in output:
                f.write(line)

    import multiprocessing
    pool = multiprocessing.Pool()

    filenames = glob('2014-*-*.log')
    # [process(fn) for fn in filenames]
    pool.map(process, filenames)


![](images/embarrassing-process.png)


### 1.  Multiprocessing log files

*   Benefits
    *  Dead simple
    *  Handles 80% of all cases
*   Drawbacks
    *  Dead simple
    *  Hard to share data between processes


### 2.  Threading and NumPy Arrays

    timeseries = [np.load(fn) for fn in glob('2014-*.*.npy')]

    pool = multiprocessing.pool.ThreadPool()

    # correlations = [np.correlate(a, b) for a in timeseries_list
    #                                    for b in timeseries_list]

    futures = [pool.apply_async(np.correlate, (a, b)) for a in timeseries_list
                                                      for b in timeseries_list]
    results = [f.get() for f in futures]


![](images/correlation.png)


### 2.  Threading and NumPy Arrays

*   Benefits
    *   Seamlessly share data between threads
    *   Avoid GIL with C/Fortran code
*   Drawbacks
    *   A bit more complex (`apply_async`, `get`)
    *   Doesn't accelerate Pure Python
    *   All data in memory


### 3.  Complex analytic workloads

### Data churn -- Analytics

* Bulk data ingest

    <img src="images/embarrassing-process.png" align=center width=60%>

*  Analysis

    <img src="images/correlation.png" align=center width=60%>


### 3.  Complex analytic workloads

<img src="images/grid-search.png">


### 3.  Complex analytic workloads

<img src="images/dask-svd.png" width=50%>


### 4.  Dynamic task scheduling and Dask

*   Dynamic task scheduler (run graphs)
    *  1 ms latency
    *  keeps a small amount of data in memory
*   Parallel larger-than-memory collections
    *  Arrays
    *  DataFrames
    *  Custom work


### 5.  `dask.array/dataframe`

*   Dask.array supports larger-than-memory arrays
    *  Break large dask.array operations into many small numpy operations

*   Dask.dataframe supports larger-than-memory dataframes
    *  Break large dask.array operations into many small numpy operations
