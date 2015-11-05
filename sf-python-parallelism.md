Python and Parallelism
----------------------

*Matthew Rocklin*

[Continuum Analytics](https://www.continuum.io/)


### Python has a sophisticated analytics stack

<hr>

### Computers have many cores and fast hard drives


Outline
-------

0.  Don't use parallelism
1.  Multiprocessing
2.  Threading
3.  Complex analytic workloads
4.  Dynamic task scheduling (`dask`)
5.  Larger than memory arrays / dataframes
6.  Distributed


### Don't use parallelism

*  Explore APIs
*  Write C/Cython/Numba
*  Store data in nice formats
*  Use smarter algorithms
*  Sample



### Multiprocessing log files

    filenames = glob('2014-*-*.log')            # Collect all filenames

    def process(filename):
        with open(filename) as f:               # Load from file
            lines = f.readlines()

        output = ...                            # do work

        with open(filename.replace('.log', '.out'), 'w') as f:
            for line in output:
                f.write(line)                   # Write to file

    # [process(fn) for fn in filenames]         # Single-core processing

    import multiprocessing
    pool = multiprocessing.Pool()
    pool.map(process, filenames)                # Multi-core processing


### Multiprocessing log files

    filenames = glob('2014-*-*.log')            # Collect all filenames

    def process(filename):
        with open(filename) as f:               # Load from file
            lines = f.readlines()

        output = ...                            # do work

        with open(filename.replace('.log', '.out'), 'w') as f:
            for line in output:
                f.write(line)                   # Write to file

    # [process(fn) for fn in filenames]         # Single-core processing

    from concurrent.futures import ProcessPoolExecutor
    executor = ProcessPoolExecutor()
    executor.map(process, filenames)            # Multi-core processing


![](images/embarrassing-process.png)


### Multiprocessing log files

*   Benefits
    *  Dead simple
    *  Handles 80% of all cases
    *  Many software solutions
*   Drawbacks
    *  Dead simple
    *  Hard to share data between processes
    *  Function serialization


### Threads vs Processes

![](images/threads-procs.png)


### Threading and Numeric Data

    pool = multiprocessing.pool.ThreadPool()
    filenames = glob('2014-*.*.npy')

    timeseries_list = pool.map(np.load, filenames)

    # correlations = [[np.correlate(a, b) for a in timeseries_list]
    #                                     for b in timeseries_list]

    futures = [[pool.apply_async(np.correlate, (a, b)) for a in timeseries_list]
                                                       for b in timeseries_list]
    results = [[f.get() for f in L]
                        for L in futures]


### Threading and Numeric Data

    executor = ThreadPoolExecutor()
    filenames = glob('2014-*.*.npy')

    timeseries_list = executor.map(np.load, filenames)

    # correlations = [[np.correlate(a, b) for a in timeseries_list]
    #                                     for b in timeseries_list]

    futures = [[executor.submit(np.correlate, (a, b)) for a in timeseries_list]
                                                      for b in timeseries_list]
    results = [[f.result() for f in L]
                           for L in futures]


![](images/correlation.png)


### Threading and Numeric Data

*   Benefits
    *   Seamlessly share data between threads
    *   Avoid GIL with NumPy/Pandas/Sci*
*   Drawbacks
    *   A bit more complex (`submit`, `result`)
    *   Doesn't accelerate Pure Python


### The GIL is mostly a non-issue for PyData


### Complex data dependencies (analysis)

<img src="images/correlation.png" align=center width=60%>

### Embarrassingly Parallel (data ingest)

<img src="images/embarrassing-process.png" align=center width=60%>


### Complex workloads -- GridSearch, CV, Pipeline

    pipeline = Pipeline([('cnt', CountVectorizer()), ..., ('svm', LinearSVC())])
    gridsearch = GridSearch(pipeline, {'svm__C': np.logspace(-3, 2, 10), ...})

<img src="images/grid-search.png">


### Complex workloads -- Larger-than-memory SVD

    u, s, v = da.linalg.svd(X)

<img src="images/dask-svd.png" width=50%>



## dask

### Large NumPy/Pandas collections

<hr>

### Dynamic Task Scheduling


### Dask executes task graphs nicely

*   Dynamic task scheduler
    *  Executes task graphs in parallel
    *  Respects data dependencies
    *  1 ms latency per task
    *  Minimizes intermediate data in memory
*   Parallel larger-than-memory collections
    *  Large Arrays
    *  Large DataFrames
    *  Large Python Lists
    *  Custom work


### dask.array

*   Copies the NumPy interface

        >>> x.dot(y.T) - y.mean(axis=0)
*   Supports larger-than-memory data.  Limited by disk size, not RAM.

        >>> x.nbytes
        100000000000
*  Parallel execution, small memory footprint

    ![](images/350percent-cpu-usage-alpha.png)
*  Break large operations into many small numpy operations


### Dask collections build graphs

    (2*x + 1) ** 2

![](images/embarrassing.png)


### Dask Schedulers Execute Graphs

    (2*x + 1) ** 2

![](images/embarrassing.gif)


### Sometimes this fails

(but that's ok)

![](images/fail-case.gif)



### Distributed

*  Hadoop/Spark/Storm/...
    *   Built by data engineers
    *   Java Virtual Machine (JVM) based

![](images/apache-numfocus-goldilocks.png)

*  Native code (C/Fortran/Python/R/Julia)
    *   Built by scientists / academics / analysts
    *   Native code based


### distributed prototype

*   Grew out of dask
    *  Full dynamic task scheduler for data dependencies
    *  1ms overhead per task
*   Concurrent.futures and dask APIs
*   Data local, resilient (mostly), easy to deploy
*   Peer-to-peer communication of data betwee workers



### Questions?

<img src="images/fail-case.gif" width=70%>

* Dask: [dask.pydata.org](http://dask.pydata.org/en/latest/)
* Distributed: [distributed.readthedocs.org](http://distributed.readthedocs.org/en/latest/)
* Blog: [matthewrocklin.com/blog](http://matthewrocklin.com/blog/)
* Blaze Blog: [blaze.pydata.org](http://blaze.pydata.org/)

[@mrocklin](https://twitter.com/mrocklin)
