<img src="http://dask.pydata.org/en/latest/_images/dask_horizontal_white.svg"
     alt="dask logo"
     width="40%">

-   Open source library to scale Python code

-   Works with existing libraries like Pandas and Scikit-Learn

-   Can also parallelize pre-existing internal codebases


How do people use Dask?
-----------------------

1.  Drop-in API replacements for subsets of Numpy, Pandas, Scikit-Learn

    ```python
    import pandas as pd
    df = pd.read_csv('myfile.csv', parse_dates=['timestamp'])
    df.groupby('name').balance.mean()
    ```

2.  Low-level APIs to parallelize custom code

    ```python
    futures = client.map(train, data)

    async for future in as_completed(futures):
        result = await future
        if not done():
            future = client.submit(train, ...)

        ...
    ```


How do people use Dask?
-----------------------

1.  Drop-in API replacements for subsets of Numpy, Pandas, Scikit-Learn

    ```python
    import dask.dataframe as dd
    df = dd.read_csv('s3://.../*.csv', parse_dates=['timestamp'])
    df.groupby('name').balance.mean().compute()
    ```

2.  Low-level APIs to parallelize custom code

    ```python
    futures = client.map(train, data)

    async for future in as_completed(futures):
        result = await future
        if not done():
            future = client.submit(train, ...)

        ...
    ```


### Dask APIs help users construct task graphs

<hr>

### Dask schedulers execute task graphs on parallel hardware


### Parallel Dask Arrays

Leverage existing Numpy library for in-memory arrays

<img src="images/dask-array.svg" width="60%">

    import numpy as np
    x = np.random.random(size=(1000, 1000))
    y = x + x.T - x.mean(axis=0)

    import dask.array as da
    x = da.random.random(size=(10000, 10000), chunks=(1000, 1000))
    y = x + x.T - x.mean(axis=0)


### Parallel Dask Dataframes

Leverage existing Pandas library for in-memory dataframes

<img src="images/dask-dataframe-inverted.svg" width="30%">

    import pandas as pd
    df = pd.read_csv('myfile.csv', parse_dates=['timestamp'])
    df.groupby(df.timestamp.dt.hour).value.mean()

    import dask.dataframe as dd
    df = dd.read_csv('hdfs://myfiles.*.csv', parse_dates=['timestamp'])
    df.groupby(df.timestamp.dt.hour).value.mean().compute()


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


### Example custom graph built with dask.delayed

<img src="images/credit_models/simple-model.svg" width="100%">


### Machine Learning

```python
pipe = Pipeline(steps=[('pca', PCA()),
                       ...,
                       ('logistic', LogisticRegression)])
grid = GridSearchCV(pipe, parameter_grid)
```

### Translate to task graph of normal Python calls

<img src="images/grid_search_schedule-0.png" width="100%">


### Machine Learning

```python
pipe = Pipeline(steps=[('pca', PCA()),
                       ...,
                       ('logistic', LogisticRegression)])
grid = GridSearchCV(pipe, parameter_grid)
```

### Execute graphs efficiently on parallel hardware

<img src="images/grid_search_schedule.gif" width="100%">


### Dask Network

<img src="images/network-inverse.svg">


### Dask Network

Set up locally

    from dask.distributed import Client
    client = Client()  # set up local scheduler and workers

Set up on a cluster

    host1$ dask-scheduler
    Starting scheduler at 192.168.0.1:8786

    host2$ dask-worker 192.168.0.1:8786
    host3$ dask-worker 192.168.0.1:8786
    host4$ dask-worker 192.168.0.1:8786



### Use Cases


### People who want faster Pandas

<img src="images/ian-ozsvald-3.png" width="60%">


### People who want faster Scikit-Learn

```python
pipe = Pipeline(steps=[('pca', PCA()),
                       ...,
                       ('logistic', LogisticRegression)])
grid = GridSearchCV(pipe, parameter_grid)
```

<img src="images/grid_search_schedule.gif" width="100%">


### Atmospheric and Oceanographic Science

<img src="images/day-vs-night-cropped.png" width="80%">


### Atmospheric and Oceanographic Science

<img src="images/gulf-mexico-680-400.png" width="80%">


### Custom systems in Finance

<img src="images/credit_models/simple-model.svg" width="100%">


### Data processing pipelines

<img src="images/beamline_illustation.jpg" width="80%">

*Image processing pipeline in Brookhaven Synchrotron*


### Realtime optimization algorithms

<img src="images/dask-patternsearch.gif" width="60%">


### Geospatial analysis

<img src="images/nyc-taxi-geo-counts.png" width="60%">



### How does Dask compare to Apache Spark?


-  Trust
    -  Spark is well established and trusted
    -  Dask is new, but part of the established PyData ecosystem
-  Framework vs Library
    -  Spark is an all-in-one framework
    -  Dask is a small part of the larger PyData ecosystem
-  JVM vs Python
    -  Spark is JVM based, with some support for Python and R
    -  Dask is nicer for Python users, but supports no one else
-  High vs Low level
    -  Spark works at a high level of Map / Shuffle / Reduce stages
    -  Dask thinks at a lower level of individual task scheduling
-  Applications
    -  Spark is focused on SQL and BI applications
    -  Dask is less focused, and better at complex situations


### Broadly

-  People choose Spark because ...
    -  Better at SQL-like computations
    -  Integrates with JVM infrastructure
    -  Well-known name
-  People choose Dask because
    -  It's lighter weight to adopt
    -  They like Python
    -  Their problems are too complex to fit into Spark


### Spark

<table>
<tr>
<td>Map</td>
<td>Shuffle</td>
<td>Reduce</td>
</tr>
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

### Dask

<img src="images/array-xxT.svg" width="40%">
<img src="images/array-xdotxT-mean-std.svg" width="50%">
