Dask: Fine Grained Task Parallelism
-----------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

Continuum Analytics


### Dask provides parallelism

<hr>

### ... originally designed for NumPy and Pandas

<hr>

### ... but used today for arbitrary computations


### Recent talks

*   SciPy - July, 2016:

    Dask.delayed, some machine learning

*  PyGotham - August 2016:

    Build DataFrame computations

*  *PyData DC - October 2016 (this talk!)*:

    Fine-grained parallelism, motivation and performance



### In the beginning, there was NumPy ...

<hr>

### And it was good (except in parallel or out of RAM)


### NumPy

<img src="images/numpy-inverted.svg">


### Dask.Array

<img src="images/dask-array-inverted.svg">


### Pandas

<img src="images/pandas-inverted.svg">


### Dask.DataFrame

<img src="images/dask-dataframes-inverted.svg" width="70%">


### Many problems don't fit

### into a "big array" or "big dataframe"


### Python

<img src="images/python-inverted.svg">


### Dask

<img src="images/dask-arbitrary-inverted.svg">



### This flexibility is novel and liberating

<hr>

### It's also tricky to do well

<hr>

    results = {}

    for a in A:
        for b in B:
            if a < b:
                results[a, b] = f(a, b)
            else:
                results[a, b] = g(a, b)


### High Level Parallelism

**Spark**

    outputs = collection.filter(predicate)
                        .groupby(key)
                        .map(function)

<hr>

**SQL**

    SELECT city, sum(population)
    WHERE population > 1000000
    GROUP BY city

<hr>

**Matrices**

    solve(A.dot(A.T), x)


### Map - Shuffle - Reduce

<table>
<tr>
  <td>
    <img src="images/embarrassing.svg">
  </td>
  <td>
    <img src="images/shuffle.svg">
  </td>
  <td>
    <img src="images/reduction.svg">
  </td>
</tr>
</table>


### Some Parallel Problems are Messy

    results = {}

    for a in A:
        for b in B:
            if a < b:
                results[a, b] = f(a, b)
            else:
                results[a, b] = g(a, b)


### TimeSeries - Resample

<img src="images/resample.svg">

    df.value.resample('1w').mean()


### TimeSeries - Rolling

<img src="images/rolling.svg">

    df.value.rolling(100).mean()


### Stable SVD

<img src="images/svd.svg" width="45%">

    u, s, v = da.linalg.svd(x)


### Approximate SVD

<img src="images/svd-compressed.svg">

    u, s, v = da.linalg.svd_compressed(x, k=1)


### Dask executes arbitrary graphs in parallel

<hr>

### Flexibility enables custom applications and efficiency


### We've seen graphs like this before: Luigi

<img src="images/luigi.png" width="80%">

http://luigi.readthedocs.io/en/stable/


### We've seen graphs like this before: Airflow

<img src="images/airflow.png" width="80%">

https://github.com/apache/incubator-airflow



### Messy Parallelism

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

<hr>

    results = {}

    for a in A:
        for b in B:
            if a < b:
                results[a, b] = delayed(f)(a, b)  # lazily construct graph
            else:
                results[a, b] = delayed(g)(a, b)  # without structure

    results = compute(delayed(results))  # trigger all computation


### Custom Script

    filenames = ['mydata-%d.dat' % i for i in range(10)]
    data = [load(fn) for fn in filenames]

    reference = load_from_sql('sql://mytable')
    processed = [process(d, reference) for d in data]

    rolled = []
    for i in range(len(processed) - 2):
        a = processed[i]
        b = processed[i + 1]
        c = processed[i + 2]
        r = roll(a, b, c)
        rolled.append(r)

    compared = []
    for i in range(20):
        a = random.choice(rolled)
        b = random.choice(rolled)
        c = compare(a, b)
        compared.append(c)

    best = reduction(compared)


### Custom Script

    from dask import delayed

    load = delayed(load)
    load_from_sql = delayed(load_from_sql)
    process = delayed(process)
    roll = delayed(roll)
    compare = delayed(compare)
    reduction = delayed(reduction)

Dask.delayed converts function calls to lazily executed tasks in a graph.

Inputs to the function become graph dependencies.


### Custom Script

<img src="images/custom-etl-1.svg">

    data = [load(fn) for fn in filenames]


### Custom Script

<img src="images/custom-etl-2.svg">

    reference = load_from_sql('sql://mytable')


### Custom Script

<img src="images/custom-etl-3.svg">

    processed = [process(d, reference) for d in data]


### Custom Script

<img src="images/custom-etl-4.svg" width="70%">

    rolled = []
    for i in range(len(processed) - 2):
        a = processed[i]
        b = processed[i + 1]
        c = processed[i + 2]
        r = roll(a, b, c)
        rolled.append(r)


### Custom Script

<img src="images/custom-etl-5.svg">

    compared = []
    for i in range(20):
        a = random.choice(rolled)
        b = random.choice(rolled)
        c = compare(a, b)
        compared.append(c)


### Custom Script

<img src="images/custom-etl-6.svg">

    best = reduction(compared)


### Custom Script

    filenames = ['mydata-%d.dat' % i for i in range(10)]
    data = [load(fn) for fn in filenames]

    reference = load_from_sql('sql://mytable')
    processed = [process(d, reference) for d in data]

    rolled = []
    for i in range(len(processed) - 2):
        a = processed[i]
        b = processed[i + 1]
        c = processed[i + 2]
        r = roll(a, b, c)
        rolled.append(r)

    compared = []
    for i in range(20):
        a = random.choice(rolled)
        b = random.choice(rolled)
        c = compare(a, b)
        compared.append(c)

    best = reduction(compared)



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


### Task Scheduling

<img src="images/fg-simple.svg">

    x = f(1)
    y = f(2)
    z = g(x, y)

<img src="images/computer-tower.svg" width="15%">
<img src="images/computer-tower.svg" width="15%">


<img src="images/scheduling-workers-1.svg">


<img src="images/scheduling-workers-2.svg">


<img src="images/scheduling-workers-3.svg">


<img src="images/scheduling-workers-4.svg">


<img src="images/scheduling-workers-5.svg">


<img src="images/scheduling-workers-6.svg">


<img src="images/scheduling-workers-7.svg">


<img src="images/scheduling-workers-8.svg">


<img src="images/scheduling-workers-9.svg">


<img src="images/scheduling-workers-10.svg">


<img src="images/scheduling-workers-11.svg">


<img src="images/scheduling-workers-12.svg">


<img src="images/scheduling-workers-13.svg">


<img src="images/scheduling-workers-14.svg">


<img src="images/scheduling-workers-15.svg">



### Intelligent scheduling requires measurement

*  Size of outputs with `__sizeof__` protocol
*  Computation time (EWMA, with restarts)
*  Communication time
*  Disk load time
*  Process reported memory use
*  ...


### Fine-grained scheduling requires constant-time decisions

*  Computational graphs scale out to 100,000s of tasks
*  We spend ~200us per task in the scheduler
*  And ~1-10kB in RAM

### Solution

*  Heavily indexed Pure Python data structures.
*  No classes, just bytestrings and dicts/sets/deques.

<img src="images/dicts-everywhere.jpg">



### How does Dask compare to Airflow/Luigi?

<hr>

### How does Dask compare to Spark?


### <strike>The best</strike> An interesting combination of both worlds!


### Airflow/Luigi

*  Dask communicates between workers, manages data
*  Dask operates at interactive/computational timescales (ms)
*  Dask scales out to large graphs

*  Airflow/Luigi have connectors with Hive, MapReduce, etc..
*  Airflow has cron-like ability "Run this every day"
*  Airflow/Luigi have policies for retry-on-fail, etc..

### Spark

*  Dask computes arbitrary graphs well
*  Dask is optimized for Python experience

*  Spark computes SQL-like computations well
*  Spark is optimized for Scala (though supports Python, R, etc..)
*  Spark is more established in the business community



### Acknowledgements

*  Countless open source developers
*  SciPy developer community
*  Continuum Analytics
*  XData Program from DARPA

<img src="images/moore.png">

<hr>

### Questions?

<img src="images/grid_search_schedule.gif" width="100%">
