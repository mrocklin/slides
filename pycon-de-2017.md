Parallelism in Python, and Dask
-------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

Anaconda Inc.


### Python has a mature analytics stack (Numpy, Pandas, ...)

<hr>

### Restricted to in-memory and single-core processing

### How do we parallelize an ecosystem?


### TODO: add Jake's scipy stack slide


### Parallel programming paradigms in Python

-  **Embarrassingly parallel systems:** multiprocessing
-  **Big Data collections:** MapReduce, Flink, Spark, SQL
-  **Task schedulers:**  Airflow, Luigi, Make
-  ... many more paradigms


### map

Apply a function in parallel across a list:

    output = []                   # Sequential
    for x in data:
        y = func(x)
        output.append(y)

<hr>

    output = map(func, data)      # Sequential

<hr>

    pool = multiprocessing.Pool()
    output = pool.map(func, data)  # Parallel

<hr>

-   Pros
    -   Easy to install and use in the common case
    -   Lightweight dependency
-   Cons
    -  Data interchange cost
    -  Not able to handle dependencies


### Big Data collections

    from pyspark import SparkContext
    sc = SparkContext('...')

    rdd = sc.parallelize(data)
    rdd.map(json.loads).filter(...).groupBy(...).count()

<hr>

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master(...).getOrCreate()

    df = spark.read_json(...)
    df.groupBy('name').aggregate({'value': 'sum'})

<hr>

-   Pros
    -   Larger set of operations (map, groupby, join, ...)
    -   Scales nicely on clusters
    -   Mature and well trusted by enterprise
-   Cons
    -  Heavyweight
    -  JVM focused (debugging, performance costs, ...)
    -  Not able to handle complex computations


### This is what I mean by complex

<img src="images/array-xdotxT-mean-std.svg">

```python
(x.dot(x.T + 1) - x.mean()).std()
```

### Spark does the following well

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

*These operations are the common case for database computations*


### Task Schedulers (Airflow, Luigi, Celery, ...)

<img src="images/airflow.png" width="40%">
<img src="images/luigi.png" width="40%">

-  Pros
    -  Handle arbitrarily complex task graphs
    -  Python Native
-  Cons
    -  No inter-worker storage or data interchange
    -  Long latencies (relatively)
    -  Not designed for computational loads
    -  Not designed for user interaction

*These operations are the common case for data pipelines*


### Multiprocessing

```python
pool = multiprocessing.Pool()
output = pool.map(func, data)  # Parallel
```

### Concurrent.futures (simple)

```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()
output = executor.map(func, data)  # Parallel
```


### Concurrent.futures (complex)

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


### Concurrent.futures (complex)

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

-   Pros
    -  Flexible for complex situations
    -  Lightweight (in standard library)
    -  .
-   Cons
    -  Does not scale
    -  Low level


### Concurrent.futures (complex)

    from concurrent.futures import ProcessPoolExecutor
    executor = ProcessPoolExecutor(8)
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

-   Pros
    -  Flexible for complex situations
    -  Lightweight (in standard library)
    -  Mulitple implementations (threads, processes, ...)
-   Cons
    -  Does not scale
    -  Low level


### To parallelize the PyData Stack ...

### what features do we need?

-  Lightweight dependence of multiprocessing
-  Scalability of Spark
-  Airflow/Celery's complex dependency handling



<img src="images/dask_icon.svg" width=20%>

-  Designed to parallelize the Python ecosystem
    -  Flexible task scheduler
    -  Familiar APIs for Python users
    -  Co-developed with Pandas/SKLearn/Jupyter teams
-  Scales
    -  From multicore laptops to 1000-node clusters
    -  Resilient, responsive, and real-time


<img src="images/dask_icon.svg" width=20%>

-  High level: Scalable versions of ...
    -  Numpy
    -  Pandas
    -  Scikit-learn
    -  Concurrent.futures
    -  ...
-  Low Level:
    -  Executes many Python functions in parallel
    -  Tracks dependencies between functions
    -  Handles data movement, worker failure, ...


### Task Graphs

<img src="images/small-simple.svg" width="40%">

-  Circles are Python functions
-  Boxes are Python objects
-  Dask executes tasks in parallel and tracks dependencies


### Task Graphs (SVD)

<img src="images/svd.svg" width="40%">

-  Circles are Python functions
-  Boxes are Python objects
-  Dask executes tasks in parallel and tracks dependencies


### Task Graphs (Pipelined Grid Search)

<img src="images/grid_search_schedule-0.png" width="100%">

-  Circles are Python functions
-  Boxes are Python objects
-  Dask executes tasks in parallel and tracks dependencies


### Task Graphs (Executing)

<img src="images/grid_search_schedule.gif" width="100%">

-  Executes graph on parallel hardware
-  Manages memory and communication between workers
-  Relatively low latencies


### Example with concurrent.futures



### We build high-level libraries on top

-   Dask.array = Dask + NumPy
-   Dask.dataframe = Dask + Pandas
-   ... (you can build your own)


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


### 1D-Array

<img src="images/array-1d.svg">

    >>> x = np.ones((15,))
    >>> x
    array([ 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    >>> x = da.ones((15,), chunks=(5,))
    dask.array<ones, shape=(15,), dtype=float64, chunksize=(5,)>


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


### Example with dask.dataframes



### Python for distributed computing

*  Strengths

    -   Strong single-core analytics stack

        NumPy, SciPy, Pandas, Scikit-*

        Also compression, storage, data access, ...

    -   Strong networking/concurrency stack

        Twisted, Tornado, Asyncio, gevent, async-await, ...

    -   Intuitive, easy to use, and broadly adopted

*   Weaknesses

    -   Deployment, compilation, dependencies (no Java JARs)

        Solvable with Docker, Conda, ...

    -   Global Interpreter Lock

        Solvable with C/C++/Cython, NumPy, Pandas, Numba, ...

    -   Lack of tight integration with existing Big Data systems

        Hard to deploy on YARN, access HDFS, ...


### Dask Interfaces

-  Mature, dependable
    -  Dask.array: Numpy arrays
    -  Dask.bag: Lists
    -  Dask.dataframe: Pandas Dataframes
    -  Concurrent.futures: Futures
    -  Dask.delayed: Decorator syntax for manual task graph construction
-  New work (unstable API)
    -  Dask-GeoPandas: geospatial analytics
    -  Dask-ML: Machine learning
    -  Streamz: real-time continuous processing


### GeoPandas

Pandas sub-project for geographical and spatial data (points, lines, polygons)

```python
geopandas.read_file('taxi_zones.shp')
         .to_crs({'init' :'epsg:4326'})
         .plot(column='borough', categorical=True)
```

<img src="images/nyc-taxi-zones.svg">

### Operations like

-  Select points within a region
-  Group/join points by region
-  Select points within 5 kilometers of this path


### GeoPandas

-  Wraps OSGeo C++ library in Python
-  Currently quite slow

<img src "images/geopandas-shapely-1.svg">
<img src "images/timings_sjoin.png">


### GeoPandas + Cython

-  Wraps OSGeo C++ library in Python
-  Rewriting in Cython

<img src "images/geopandas-shapely-2.svg">
<img src "images/timings_sjoin_all.png">


### GeoPandas + Dask

-  Partition data into geospatial regions
-  Gives an extra 2-3x on a laptop
-  Enables scaling across a cluster

<img src="images/dask-array.svg" width="30%">
<img src="images/dask-dataframe-inverted.svg" width="15%">
<img src="images/nyc-boroughs.svg" width="30%">

### TODO: add bokeh plot


### GeoPandas Status

### TODO: add bokeh plot

-  Cython (current focus)
    -  Fast and efficient now
    -  Decently complete
    -  Requires latest release of Pandas
    -  Needs users to identify issues
-  Dask (waiting until Cython is finished)
    -  Hard algorithms implemented (spatial join)
    -  Easy algorithms still missing
    -  Need to improve distributed serialization, data ingesion, ...



### Machine Learning

-  [dask-ml.readthedocs.io](http://dask-ml.readthedocs.io/)
-  See blogposts by [Tom Augspurger](https://tomaugspurger.github.io/)
    -  [Overview](https://tomaugspurger.github.io/scalable-ml-01.html)
    -  [Incremental Learning](https://tomaugspurger.github.io/scalable-ml-02.html)
    -  ...
-  And [Jim Crist](http://jcrist.github.io/)
    -  [Grid Search](http://jcrist.github.io/introducing-dask-searchcv.html)
-  And [Chris White](https://github.com/moody-marlin/)
    -  [Convex Optimization](https://matthewrocklin.com/blog/work/2017/03/22/dask-glm-1)
    -  [Asynchronous Algorithms](http://matthewrocklin.com/blog/work/2017/04/19/dask-glm-2)


### ML: We have a few options ...

1.  Accelerate Scikit-Learn directly

    Useful for model selection, random forests, ...

    ```python
    pipe = Pipeline(steps=[('pca', PCA()),
                           ('logistic', LogisticRegression)])
    grid = GridSearchCV(pipe, parameter_grid)
    ```

2.  Build well-known algorithms with Dask.array

    Useful for Logistic regression, optimization, ...

    ```python
    eXbeta = da.exp(X.dot(beta))
    gradient = X.T.dot(eXbeta / (eXbeta + 1) - y)
    ...
    ```

3.  Collaborate with other distributed systems

    Useful for XGBoost, Tensorflow, ...

    -  **Pre-process** with Dask.dataframe
    -  **Deploy** other services
    -  **Pass data** from Dask and **train** with other service

4.  Build custom algorithms with concurrent.futures, dask.delayed, ...

    Useful for algorithm researchers


### Accelerate Scikit-Learn directly: Joblib

-  Joblib
    -  Scikit-Learn uses Joblib for parallelism
    -  Joblib now supports swapping backends
    -  Can replace the normal thread pool with Dask
-  Good for ...
    -  model selection (grid search)
    -  embarrassingly parallel computations (random forests)
-  Bad for ...
    -  Large data training
-  Status: works now, will improve as joblib changes

TODO: threadpool <- Joblib <- SKLearn
TODO: Dask <- Joblib <- SKLearn


### Build Algorithms with Dask.array

-  Optimization algorithms can be implemented with NumPy syntax

    Xbeta = X.dot(beta_hat)
    func = ((y - Xbeta)**2).sum()
    gradient = 2 * X.T.dot(Xbeta - y)

    beta_hat = beta_hat - step_size * gradient
    new_func = ((y - X.dot(beta_hat))**2).sum()

<img src="images/grad-step-white-on-transparent.svg">

[Related blogpost](http://matthewrocklin.com/blog/work/2017/03/22/dask-glm-1)


### Build Algorithms with Dask.array

-  Optimization algorithms can be implemented with NumPy syntax
-  Combine with regularizers (L1, L2, ElasticNet, ...)
-  Combine with Generalized Linear Model families
-  Get
    -  Linear Regression
    -  Logistic Regression
    -  Poisson Regression
    -  ...

```python
>>> from dask_ml.estimators import LogisticRegression
>>> from dask_ml.datasets import make_classification
>>> X, y = make_classification()
>>> lr = LogisticRegression()
>>> lr.fit(X, y)
>>> lr
LogisticRegression(abstol=0.0001, fit_intercept=True, lamduh=1.0,
                   max_iter=100, over_relax=1, regularizer='l2', reltol=0.01,
                                      rho=1, solver='admm', tol=0.0001)
```


### Deploy Other Services with Dask

-  We prefer to avoid reinventing algorithms
-  Some systems, like XGBoost, Tensorflow, already provide distributed training
-  Dask can set these up and pass them data

<img src="images/dask-xgboost-pre.svg">

```python
import dask.dataframe as dd
df = dd.read_parquet('s3://...')

# Split into training and testing data
train, test = df.random_split([0.8, 0.2])

# Separate labels from data
train_labels = train.x > 0
test_labels = test.x > 0

del train['x']  # remove informative column from data
del test['x']  # remove informative column from data

# from xgboost import XGBRegressor  # change import
from dask_ml.xgboost import XGBRegressor

est = XGBRegressor(...)
est.fit(train, train_labels)

prediction = est.predict(test)
```


### Deploy Other Services with Dask

-  We prefer to avoid reinventing algorithms
-  Some systems, like XGBoost, Tensorflow, already provide distributed training
-  Dask can set these up and pass them data

<img src="images/dask-xgboost-post.svg">

```python
import dask.dataframe as dd
df = dd.read_parquet('s3://...')

# Split into training and testing data
train, test = df.random_split([0.8, 0.2])

# Separate labels from data
train_labels = train.x > 0
test_labels = test.x > 0

del train['x']  # remove informative column from data
del test['x']  # remove informative column from data

# from xgboost import XGBRegressor  # change import
from dask_ml.xgboost import XGBRegressor

est = XGBRegressor(...)
est.fit(train, train_labels)

prediction = est.predict(test)
```


### Machine Learning Overview

-  Dask works to enable machine learning
    -  Using existing technologies like SKLearn, XGBoost
    -  Or implementing new algorithms when necessary
-  Development depends on collaboration with other groups
-  Continue to maintain familiar Scikit-Learn APIs regardless


### Real-time systems

How do we handle live, continuous, datasets?

TODO: image of data stream with increasingly complex systems


### Real-time systems: Background

-   Iterators / generators
-   JVM Streaming systems like Flink, Akka, Spark Streaming
-   Reactive systems like ReactiveX / RxPy
