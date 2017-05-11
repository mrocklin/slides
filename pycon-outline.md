Dask: Parallel Programming in Python
------------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

Continuum Analytics


### We have a strong analytics ecosystem (numpy, pandas, ...)

<hr>

### that is mostly restricted to a single core and RAM


### High quality and useful algorithms (e.g. scikit-image)

    skimage.feature.canny(im, sigma=3)

<img src="http://scikit-image.org/docs/dev/_images/sphx_glr_plot_canny_001.png"
     alt="Canny edge detection from skimage"
     width="50%">

### Broad coverage in niche corners (e.g. scikit-allel)

<img src="http://alimanfoo.github.io/assets/2016-06-10-scikit-allel-tour_files/2016-06-10-scikit-allel-tour_50_0.png" alt="scikit-allel example" width="50%" align="center">

*Example taken from scikit-allel webpage*


### How do we scale an ecosystem?

<hr>

### Focus on flexibility,

### Adhere to existing APIs/standards


### Dask enables scalable computing

-  **Parallelizes libraries** like Pandas, NumPy, and SKLearn
-  **Scales** from 1 to 1000's of computers (Spark-like scaling)
-  **Flexible** backed by a task scheduler (like Airflow, Celery)
-  **Adapts** to serve custom systems
-  **Async real time** with 200us overheads and 10ms latencies
-  **Pure Python** and built from standard technology
-  **Supported** by community, non/for-profit, and government


Outline
-------

-  **Examples**: Dask parallelizes Python APIs, high to low level
-  **Ecosystem**: Python's strengths and weaknesses for parallelism
-  **Internals**: How Dask works



### Dask enables parallel Python

<hr>

### ... originally designed to parallelize NumPy and Pandas

<hr>

### ... but also used today for arbitrary computations


### Dask.array

<img src="images/dask-array.svg" width="60%">

    # NumPy code
    import numpy as np
    x = np.random.random(...)
    u, s, v = np.linalg.svd(x.dot(x.T))

    # Dask.array code
    import dask.array as da
    x = da.random.random(..., chunks=(1000, 1000))
    u, s, v = da.linalg.svd(x.dot(x.T))


### Dask.DataFrame

<img src="images/dask-dataframe-inverted.svg" width="30%">

    import pandas as pd
    df = pd.read_csv('myfile.csv', parse_dates=['timestamp'])
    df.groupby(df.timestamp.dt.hour).value.mean()

    import dask.dataframe as dd
    df = dd.read_csv('hdfs://myfiles.*.csv', parse_dates=['timestamp'])
    df.groupby(df.timestamp.dt.hour).value.mean()


### Fine Grained Python Code

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


### Fine Grained Python Code

    from dask import delayed, compute

<hr>

    results = {}

    for a in A:
        for b in B:
            if a < b:
                results[a, b] = delayed(f)(a, b)  # lazily construct graph
            else:
                results[a, b] = delayed(g)(a, b)  # without structure

    results = compute(results)  # trigger all computation


### Fine Grained Python Code

    e = concurrent.futures.ThreadPoolExecutor()

<hr>

    results = {}

    for a in A:
        for b in B:
            if a < b:
                results[a, b] = e.submit(f, a, b)  # submit work asynchronously
            else:
                results[a, b] = e.submit(g, a, b)  # submit work asynchronously

    results = {k: v.result() for k, v in results.items()} # block until finished



### Dask APIs Produce Task Graphs

<hr>

### Dask Schedulers Execute Task Graphs


### 1D-Array

<img src="images/array-1d.svg">

    >>> np.ones((15,))
    array([ 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    >>> x = da.ones((15,), chunks=(5,))


### 1D-Array

<img src="images/array-1d-sum.svg" width="30%">

    x = da.ones((15,), chunks=(5,))
    x.sum()


### ND-Array - Sum

<img src="images/array-sum.svg">

    x = da.ones((15, 15), chunks=(5, 5))
    x.sum(axis=0)


### ND-Array - Transpose

<img src="images/array-xxT.svg">

    x = da.ones((15, 15), chunks=(5, 5))
    x + x.T


### ND-Array - Matrix Multiply

<img src="images/array-xdotxT.svg">

    x = da.ones((15, 15), chunks=(5, 5))
    x.dot(x.T + 1)


### ND-Array - Compound Operations

<img src="images/array-xdotxT-mean.svg">

    x = da.ones((15, 15), chunks=(5, 5))
    x.dot(x.T + 1) - x.mean()


### ND-Array - Compound Operations

<img src="images/array-xdotxT-mean-std.svg">

    import dask.array as da
    x = da.ones((15, 15), chunks=(5, 5))
    y = (x.dot(x.T + 1) - x.mean()).std()


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



### Dask APIs Produce Task Graphs

<hr>

### Dask Schedulers Execute Task Graphs


### Single Machine Scheduler

Optimized for larger-than-memory use.

*   **Parallel CPU**: Uses multiple threads or processes
*   **Minimizes RAM**: Choose tasks to remove intermediates
*   **Low overhead:** ~100us per task
*   **Concise**: ~1000 LOC
*   **Real world workloads**: Under heavy load by many different projects


### Distributed Scheduler

*   **Distributed**: One scheduler coordinates many workers
*   **Data local**: Moves computation to correct worker
*   **Asynchronous**: Continuous non-blocking conversation
*   **Multi-user**: Several users share the same system
*   **HDFS Aware**: Works well with HDFS, S3, YARN, etc..
*   **Solidly supports**: dask.array, dask.dataframe, dask.bag, dask.delayed,
    concurrent.futures, ...
*   **Less Concise**: ~5000 LOC Tornado TCP application

    All of the logic is hackable Python, separate from Tornado


### Distributed Network

<img src="images/network-inverse.svg">


### Distributed Network

Or use it on your laptop

Install

    $ conda install dask distributed
    or
    $ pip install dask[complete] distributed --upgrade

Set up locally

    from dask.distributed import Client
    client = Client()  # set up local scheduler and workers

Use in lightweight manner

    In [3]: %time client = Client(processes=False)  # use local threads
    CPU times: user 44 ms, sys: 0 ns, total: 44 ms
    Wall time: 43.6 ms



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


### High level overview

-  Dask is a **low level** task scheduler
    -  Determines when and where to call Python functions
    -  Works with any Python functions on any Python objects
    -  Handles data dependencies, locality, data movement, etc..
-  **High level** APIs built on top
    -  Dask.array = dask + numpy
    -  Dask.dataframe = dask + pandas
    -  Other APIs for lists, machine learning, streaming, etc..
    -  Maybe good for your work as well?



## Ecosystem


### The Python ecosystem is ideal for parallel computation

<hr>

### Combines strong analytics with strong networking


### Python's Strengths and Weaknesses

*From a parallel data analytics point of view*

-  **Strengths**
    -  Strong algorithmic tradition
    -  Battle hardened C/Fortran/LLVM/CUDA codes
    -  Strong and active networking and concurrency stack
    -  Standard language in research groups, teaching, etc.
-  **Weaknesses**
    -  Slow?  Interpreted
    -  GIL (fixed in numeric ecosystem)
    -  Packaging (fixed by better pip, conda, docker)
    -  Lack of a distributed computing ecosystem


### Global Interpreter Lock

-   Stops two Python threads from running Python code at the same time
-   Numeric libraries don't call Python code.

    They call C/Fortran/LLVM/CUDA.
-   *The GIL is not a problem for the numeric Python ecosystem*

### Packaging / Deployment

-   Python packaging is better than it used to be.
    -  Pip improvements and Wheels
    -  Conda binaries and environments, dependency resolution
-   Containers becoming standard in deployment.  Also driven by Node, Go, etc.
-   Deployment solutions increasingly not tied to computational systems

    Can leverage existing work with Yarn, Mesos, Kubernetes, etc..


### Strong Algorithmic Tradition

-   High quality implementations of useful algorithms

        skimage.feature.canny(im, sigma=3)

<img src="http://scikit-image.org/docs/dev/_images/sphx_glr_plot_canny_001.png" alt="Canny edge detection from skimage" width="50%" align="center">

-   Broad coverage throughout many corners of science

<img src="http://alimanfoo.github.io/assets/2016-06-10-scikit-allel-tour_files/2016-06-10-scikit-allel-tour_50_0.png" alt="scikit-allel example" width="50%" align="center">

*Example taken from scikit-allel webpage*


### Battle hardened C/Fortran/LLVM/CUDA codes

    In [1]: import numpy as np

    In [2]: x = np.random.random((1000, 1000))

    In [3]: %time _ = x.dot(x)  # roughly a billion calculations
    CPU times: user 76 ms, sys: 0 ns, total: 76 ms
    Wall time: 76.9 ms

    In [4]: %time _ = x.dot(x.T)
    CPU times: user 36 ms, sys: 4 ms, total: 40 ms
    Wall time: 36.9 ms

    In [5]: %time for i in range(1000000000): 1.0 * 2.3
    CPU times: user 37.5 s, sys: 4 ms, total: 37.5 s
    Wall time: 37.7 s

For numeric computations, Python libraries run at bare-metal speeds


### Strong networking and concurrency stack

-  Sensible concurrency model

        @tornado.gen.coroutine
        def execute_task(self, task):
            data = yield gather_data_from_peers(dependencies[task])
            result = yield executor.submit(run, task, data)  # run on thread
            yield self.report_to_scheduler({'operation': 'task-complete',
                                            'task': task})

-  Performance oriented

    <a href="https://magic.io/blog/uvloop-blazing-fast-python-networking/"><img src="images/python-tcp-benchmarks.png" alt="Python tcp benchmarks"></a>

    *From:
    [https://magic.io/blog/uvloop-blazing-fast-python-networking/](https://magic.io/blog/uvloop-blazing-fast-python-networking/)*


### Strong networking and concurrency stack

-  Sensible concurrency model

        .
        async def execute_task(self, task):
            data = await gather_data_from_peers(dependencies[task])
            result = await executor.submit(run, task, data)  # run on thread
            await self.report_to_scheduler({'operation': 'task-complete',
                                            'task': task})

-  Performance oriented

    <a href="https://magic.io/blog/uvloop-blazing-fast-python-networking/"><img src="images/python-tcp-benchmarks.png" alt="Python tcp benchmarks"></a>

    *From:
    [https://magic.io/blog/uvloop-blazing-fast-python-networking/](https://magic.io/blog/uvloop-blazing-fast-python-networking/)*


### Python has an excellent analytics stack and community

<hr>

### Python has an excellent networking stack and community


### Dask is a small and lightweight project

<hr>

### Most of the hard problems were already solved



## Engagement

-  Developers
    -  ~150 contributors total (counting all subprojects)
    -  ~10 part time developers, about half at Continuum
    -  Core developers from Pandas, NumPy, SKLearn, Jupyter, ...
-  Funding agencies
    -  Government (DAPRA, NASA, Army Engineers, UK Met, ...)
    -  Moore Foundation Data Driven Discovery
    -  Companies who fund directly (largely finance)
    -  Institutions who use and contribute code/bug reports (maybe yours?)
    -  Continuum Analytics
    -  Strong developer community (maybe you?)


### Final Slide.  Questions?

-  Dask enables the existing ecosystem to scale
-  Dask leverages existing libraries and APIs to do this cheaply (thanks!)
-  You can set it up right now during questions:

        $ pip/conda install dask distributed
        $ ipython

        >>> from dask.distributed import Client
        >>> client = Client()  # starts a "cluster" on your local machine

        >>> futures = client.map(lambda x: x + 1, range(1000))
        >>> total = client.submit(sum, futures)
        >>> total.result()

<img src="images/grid_search_schedule.gif" width="100%">
