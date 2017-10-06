Dask for Task Scheduling
------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

Anaconda Inc.


### My goal today is to determine

<hr>

### If and how Dask should prioritize task queueing workloads

### similar to Luigi/Celery/Airflow


### Python Parallel Programming Options

-  Simple single-machine
    -  Multiprocessing
    -  concurrent.futures
    -  Joblib
-  Distributed data science
    -  IPyParallel
    -  PySpark
    -  MapReduce
    -  ...
-  Task queuing / ETL systems
    -  Airflow
    -  Celery
    -  Luigi
-  ...



### Usually Dask talks focus on scaling NumPy, Pandas, ML ...


### Dask.array

<img src="images/dask-array.svg" width="60%">

    import numpy as np
    x = np.random.random(...)
    u, s, v = np.linalg.svd(x.dot(x.T))

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


### These generate complex task graphs

<hr>

<img src="images/grid_search_schedule-0.png" width="100%">

<hr>

### .


### These generate complex task graphs

<hr>

<img src="images/grid_search_schedule.gif" width="100%">

<hr>

### Which we need to run efficiently


### This execution looks like other task queuing projects

-  Celery
-  Luigi
-  Airflow
-  Make


### *Question:* Should Dask approach traditional task queuing

<hr>

### If so, what features, and why?


### Dask runs complex graphs efficiently

-  Complex graphs
    -  Arbitrary data dependencies
    -  Any Python function (without global state)
    -  Any (serializable) data
    -  Resource constraints (GPUs, ...)
    -  ...
-  Efficient execution
    -  200us scheduler overhead per task
    -  10ms roundtrip times
    -  Peer-to-peer data communication (no central data broker)
    -  Clever about data locality, load balancing, ...
    -  Scales to 1000's of nodes


### But Dask lacks some popular attributes of Celery/Airflow

-  Retry logic
-  Cron functionality
-  Policy constraints
-  ...



### For the rest of the talk: Examples

1.  Concurrent.futures
2.  Web server (async-await)
3.  Streaming Pipeline



### Concurrent.futures

    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(8)

    # result = func(*args, **kwargs)                 # sequential, blocking
    future = executor.submit(func, *args, **kwargs)  # runs on separate thread

    # do other things concurrently

    result = future.result()

-  Futures provide complete flexibility in parallel execution
-  Common API implemented across many implementations
-  Dask satisfies (and extends) this API


### Concurrent.futures

    from dask.distributed import Client
    client = Client()

    # result = func(*args, **kwargs)                 # sequential, blocking
    future = client.submit(func, *args, **kwargs)    # runs on separate thread

    # do other things concurrently

    result = future.result()

-  Futures provide complete flexibility in parallel execution
-  Common API implemented across many implementations
-  Dask satisfies (and extends) this API


### Concurrent.futures

    .
    .
    .
    results = []
    for x in L1:
        for y in L2:
            if x < y:
                z = f(x, y)
            else:
                z = g(x, y)
            results.append(z)

    .

-  Futures provide complete flexibility in parallel execution
-  Common API implemented across many implementations
-  Dask satisfies (and extends) this API


### Concurrent.futures

    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(8)
    .
    futures = []
    for x in L1:
        for y in L2:
            if x < y:
                future = executor.submit(f, x, y)
            else:
                future = executor.submit(g, x, y)
            futures.append(z)

    results = [future.result() for future in futures]

-  Futures provide complete flexibility in parallel execution
-  Common API implemented across many implementations
-  Dask satisfies (and extends) this API


### Concurrent.futures

    from dask.distributed import Client
    client = Client()  # or optionally provide cluster address
    .
    futures = []
    for x in L1:
        for y in L2:
            if x < y:
                future = client.submit(f, x, y)
            else:
                future = client.submit(g, x, y)
            futures.append(z)

    results = [future.result() for future in futures]

-  Futures provide complete flexibility in parallel execution
-  Common API implemented across many implementations
-  Dask satisfies (and extends) this API



### Web Servers

    .
    .

    def fib(n):
        if n < 2:
            return n
        else:
            return fib(n - 1) + fib(n - 2)

    class FibHandler(tornado.web.RequestHandler):

        def get(self, n):
            n = int(n)
            result = fib(n)  # <<--- this can slow down the server
            self.write('<h1>' + str(result) + '</h1>')

    application = tornado.web.Application([
        (r"/fib/(\d+)", FibHandler),
    ])
    application.listen(8000)
    tornado.ioloop.IOLoop.current().start()


### Web Servers

    from concurrent.futures import ThreadPoolExecutor
    executor = ProcessPoolExecutor(8)

    def fib(n):
        if n < 2:
            return n
        else:
            return fib(n - 1) + fib(n - 2)

    class FibHandler(tornado.web.RequestHandler):
        @gen.coroutine
        def get(self, n):
            n = int(n)
            result = yield executor.submit(fib, n)
            self.write('<h1>' + str(result) + '</h1>')

    application = tornado.web.Application([
        (r"/fib/(\d+)", FibHandler),
    ])
    application.listen(8000)
    tornado.ioloop.IOLoop.current().start()


### Web Servers

    from dask.distributed import Client
    client = Client(asynchronous=True)

    def fib(n):
        if n < 2:
            return n
        else:
            return fib(n - 1) + fib(n - 2)

    class FibHandler(tornado.web.RequestHandler):
        @gen.coroutine
        def get(self, n):
            n = int(n)
            result = yield client.submit(fib, n)
            self.write('<h1>' + str(result) + '</h1>')

    application = tornado.web.Application([
        (r"/fib/(\d+)", FibHandler),
    ])
    application.listen(8000)
    tornado.ioloop.IOLoop.current().start()


### Web Servers

    from dask.distributed import Client
    client = Client(asynchronous=True)

    def fib(n):
        if n < 2:
            return n
        else:
            return fib(n - 1) + fib(n - 2)

    class FibHandler(tornado.web.RequestHandler):

        async def get(self, n):
            n = int(n)
            result = await client.submit(fib, n)
            self.write('<h1>' + str(result) + '</h1>')

    application = tornado.web.Application([
        (r"/fib/(\d+)", FibHandler),
    ])
    application.listen(8000)
    tornado.ioloop.IOLoop.current().start()



## Questions?

<img src="images/grid_search_schedule.gif" width="80%">

-  `pip install dask[complete]`
-  [dask.pydata.org](https://dask.pydata.org)

<img src="images/moore.png" width="20%">
<img src="images/Anaconda_Logo.png" width="20%">
<img src="images/NSF.png" width="10%">
<img src="images/DARPA_Logo.jpg" width="20%">


### Streaming Systems

-  Pipelines for infinite streams of data
    -  Processing clicks, web logs
    -  Machine learning models
    -  Financial time series
    -  Scientific instruments (next talk)
-  Computationally Dask is well suited here
    -  Responsive
    -  Can submit work on the fly
    -  But needs higher level APIs


### Prototype library `streamz`

-  Supports branching / joining
-  Processing time logic
-  Back pressure
-  Event time logic for Pandas

<hr>

### Definitely not ready for public use
