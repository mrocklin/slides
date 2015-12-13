`distributed`
-------------

*Matthew Rocklin*


### Dynamic distributed task scheduler

<hr>

### with dask and concurrent.futures user interfaces


### Why we care -- as a community

*  Some people have large problems
   (big data, or big computation)
*  Complex analytics (ndarrays, machine learning, statistics) don't easily
   fit existing abstractions (MapReduce/Spark)
*  Few high level distributed computing abstractions exist outside of the JVM
   (but most analytics do!)

<hr>

### Why we care -- as a company

*  Slow energy drain over to JVM stack
*  Customers seem to like distributed computing
*  It's hip!



### Dynamic distributed task scheduler

<hr>

### with dask and concurrent.futures user interfaces


### Task Scheduling

![](images/embarrassing.png)


### Task Scheduling

![](images/dasklearn-small.png)


### Task Scheduling

![](images/dasklearn-large.png)


### Task Scheduling

![](images/fail-case.png)


### Task Scheduling

![](images/fail-case.gif)


### Schedulers execute task graphs

*  Single Machine -- (dask.threaded)
    *  Fit in memory  (hard)
    *  Use all of our cores  (easy)
*  Distributed Machine -- (distributed.scheduler)
    *  Fit in memory  (easy)
    *  Use all of our cores  (medium)
    *  Avoid communication  (hard)


### Distributed adds some other stuff too

*  Fully asynchronous
    *  Can add new tasks during computation
    *  A bit smoother user experience
*  Multi-user support
*  Immediate non-lazy `concurrent.futures` API
*  Base to build other distributed projects



### Distributed: basic architecture and local setup

*  Center (single): Coordination point
*  Workers (many): Store data, perform computations
*  Scheduler (single): Orchestrates workers
*  Executor (few): Mediate between user and Scheduler

<hr>

    $ pip install distributed --upgrade

    $ dcenter
    distributed.dcenter - INFO - Start center at 192.168.1.106:8787

    <New Terminal>
    $ dworker 127.0.0.1:8787   # run this in a few terminals
    distributed.worker - INFO - Start worker at             192.168.1.106:8789
    distributed.worker - INFO - Registered with center at:  127.0.0.1:8787

    <New Terminal>
    $ dworker 127.0.0.1:8787   # run this in a few terminals

<hr>

Verify that the center and workers shake hands in log messages



### Concurrent.futures

    >>> from concurrent.futures import ThreadPoolExecutor
    >>> executor = ThreadPoolExecutor(4)  # four threads

    >>> def inc(x):
            return x + 1

    >>> future = executor.submit(inc, 1)  # run inc(1) in thread
    >>> future
    <Future at 0x7f831983cc50 state=finished returned int>

    >>> future.result()  # block, return result when finished
    2


### Concurrent.futures

**Use `map` for embarrassingly parallel tasks**

    filenames = glob('2015-*-*.log')
    L = executor.map(process_file, filenames)  # Common case

**Use `submit` for more complex analyis**

    filenames = glob('INET-*.csv')
    timeseries_list = executor.map(pd.read_csv, filenames)
    def func(df1, df2):
        return df1.volume.corr(df2.volume)

    correlations = [[executor.submit(func, df1, df2) for df1 in timeseries_list]
                                                     for df2 in timeseries_list]

    correlations = [[future.result() for future in inner_list] # block
                                     for inner_list in correlations]


### Problem with `Future.result()`

We explicitly wait and gather futures

    a = executor.submit(f, x)       # Run two functions in parallel
    b = executor.submit(f, y)       # Run two functions in parallel

    a2 = a.result()                 # Wait until both functions finish
    b2 = b.result()                 # Wait until both functions finish

    c = executor.submit(g, a2, b2)  # Run new function on gathered results
    c.result()

<hr>

We would prefer to ignore this step

    a = executor.submit(f, x)       # Run two functions in parallel
    b = executor.submit(f, y)       # Run two functions in parallel

    c = executor.submit(g, a, b)    # Scheduler determines when safe to run g
    c.result()

<hr>

*  Avoid collecting results mid-computation
*  Trust scheduler to run tasks when ready



### Use Distributed: concurrent.futures

    from distributed import Executor
    executor = Executor('127.0.0.1:8787')

    def inc(x):
        return x + 1

    def add(x, y):
        return x + y

    a = executor.submit(inc, 1)
    b = executor.submit(inc, 100)

    c = executor.submit(add, a, b)
    c.result()


### Use Distributed: dask

    >>> from distributed import Executor
    >>> executor = Executor('127.0.0.1:8787')

    >>> import dask.array as da
    >>> x = da.random.random((10000, 10000), chunks=(1000, 1000))

    >>> y = x.mean()
    >>> y.compute(get=executor.get)      # Immediate execution
    0.50001713553447302

    >>> executor.compute(y)  # Asynchronous execution
    [<Future: status: finished, key: finalize-4fcca230d2098922b342>]

