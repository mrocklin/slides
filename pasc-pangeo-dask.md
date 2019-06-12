Interactive Analytics with Dask
-------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

NVIDIA



### Request: Transform pile of NetCDF into a Plot

```
$ ls simulation-output/
2001-01-01.nc  2001-02-01.nc  2001-03-01.nc  2001-04-01.nc
2001-01-02.nc  2001-02-02.nc  2001-03-02.nc  2001-04-02.nc
2001-01-03.nc  2001-02-03.nc  2001-03-03.nc  2001-04-03.nc
2001-01-04.nc  2001-02-04.nc  2001-03-04.nc  2001-04-04.nc
...
```

<img
src="https://cdn-images-1.medium.com/max/1600/1*AxT-i3EYClv46DxsOyFUkA.jpeg"
width="70%"
alt="Map of ocean currents and sea-surface temperatures from the ECCO2 ocean
model. Credit: NASA/Goddard Space Flight Center Scientific Visualization
Studio. https://svs.gsfc.nasa.gov/3821">


### Request: Transform pile of NetCDF into another pile

```
$ ls
2001-01-01.nc  2001-02-01.nc  2001-03-01.nc  2001-04-01.nc
2001-01-02.nc  2001-02-02.nc  2001-03-02.nc  2001-04-02.nc
2001-01-03.nc  2001-02-03.nc  2001-03-03.nc  2001-04-03.nc
2001-01-04.nc  2001-02-04.nc  2001-03-04.nc  2001-04-04.nc
...
```

```
$ ls processed/
2001-01-01.nc  2001-02-01.nc  2001-03-01.nc  2001-04-01.nc
2001-01-02.nc  2001-02-02.nc  2001-03-02.nc  2001-04-02.nc
2001-01-03.nc  2001-02-03.nc  2001-03-03.nc  2001-04-03.nc
2001-01-04.nc  2001-02-04.nc  2001-03-04.nc  2001-04-04.nc
...
```


### This is not traditional HPC

-  Loosely coupled communication
-  IO/Memory bound -- Not compute bound
-  Poorly scoped problem.  Requires interactivity.
-  Developers are only modestly technical (Python, not C++/MPI)


### Scaling challenges are cultural, not technical

*How do I run my Jupyter notebook on the super-computer?*


<img src="http://dask.pydata.org/en/latest/_images/dask_horizontal_white.svg"
     alt="dask logo"
     width="40%">

-  Parallel programming library for Python
-  Scales data libraries like Numpy, Pandas, Scikit-Learn
-  Deploys on HPC systems
-  Culturally native to Scientific Computing


-  Jupyter Notebook/Script/...
-  Xarray/NumPy/Pandas/Scikit-Learn/...
-  ...
-  ...
-  ...
-  Cluster of hardware

<hr>

Dask mediates between Python users and distributed hardware


-  Jupyter Notebook/Script/...
-  Xarray/NumPy/Pandas/Scikit-Learn/...
-  ...
-  ... MPI, Spark, Dask, ...
-  ...
-  Cluster of hardware

<hr>

Dask mediates between Python users and distributed hardware


-  Jupyter Notebook/Script/...
-  Xarray/NumPy/Pandas/Scikit-Learn/...
-  **Parallel algorithms**
-  **Distributed execution**
-  **Sensible deployment**
-  Cluster of hardware

<hr>

Dask mediates between Python users and distributed hardware



-  Jupyter Notebook/Script/...
-  Xarray/NumPy/Pandas/Scikit-Learn/...
-  **Parallel algorithms**
-  Distributed execution
-  Sensible deployment
-  Cluster of hardware

<hr>

Dask mediates between Python users and distributed hardware


### High Level: Dask scales other Python libraries

-  Pandas

    ```python
    df = pandas.read_csv('my-file.csv')

    df.groupby(df.timestamp.dt.hour).value.mean()
    ```

-  Numpy

    ```python
    X = numpy.random.random((1000, 1000))

    (X + X.T) - X.mean(axis=0)
    ```

-  Scikit-Learn

    ```python
    from scikit_learn.linear_models import LogisticRegression

    model = LogisticRegression()
    model.fit(X, y)
    ```

-  ... and several other applications throughout PyData


### High Level: Dask scales other Python libraries

-  Pandas + Dask

    ```python
    df = dask.dataframe.read_csv('s3://path/to/*.csv')

    df.groupby(df.timestamp.dt.hour).value.mean()
    ```

-  Numpy + Dask

    ```python
    X = dask.array.random((100000, 100000), chunks="1 GiB")

    (X + X.T) - X.mean(axis=0)
    ```

-  Scikit-Learn + Dask + ...

    ```python
    from dask_ml.linear_models import LogisticRegression

    model = LogisticRegression()
    model.fit(X, y)
    ```

-  ... and several other applications throughout PyData


### Dask.DataFrame

<img src="images/dask-dataframe-inverted.svg" width="30%">

```python
import pandas as pd
df = pd.read_csv('myfile.csv', parse_dates=['timestamp'])
df.groupby(df.timestamp.dt.hour).value.mean()

import dask.dataframe as dd
df = dd.read_csv('s3://myfiles.*.csv', parse_dates=['timestamp'])
df.groupby(df.timestamp.dt.hour).value.mean()
```


### Dask.array

<img src="images/dask-array.svg" width="60%">

```python
import numpy as np
x = np.random.random((1000, 1000))
y = x + x.T - x.mean(axis=0)

import dask.array as da
x = da.random.random((100000, 100000), chunks="1 GiB")
y = x + x.T - x.mean(axis=0)
```


### Fine Grained Python Code

    .

<hr>

```python
results = {}
.
.

for a in A:
    for b in B:
        if a < b:
            results[a, b] = f(a, b)
        else:
            results[a, b] = g(a, b)

.
```


### Fine Grained Python Code

```python
import dask
```

<hr>

```python
results = {}
f = dask.delayed(f)  # mark functions as lazily evaluated
g = dask.delayed(g)

for a in A:
    for b in B:
        if a < b:
            results[a, b] = f(a, b)  # construct task graph
        else:
            results[a, b] = g(a, b)

results = dask.compute(results)  # trigger computation
```


### Dask array turns array computations

### into chunked task graphs


### 1D-Array

<img src="images/array-1d.svg">

```python
>>> x = np.ones((15,))
>>> x
array([ 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

>>> x = da.ones((15,), chunks=(5,))
dask.array<ones, shape=(15,), dtype=float64, chunksize=(5,)>
```


### 1D-Array

<img src="images/array-1d-sum.svg" width="30%">

```python
x = da.ones((15,), chunks=(5,))
x.sum()
```


### ND-Array - Sum

<img src="images/array-sum.svg">

```python
x = da.ones((15, 15), chunks=(5, 5))
x.sum(axis=0)
```


### ND-Array - Transpose

<img src="images/array-xxT.svg">

```python
x = da.ones((15, 15), chunks=(5, 5))
x + x.T
```


### ND-Array - Matrix Multiply

<img src="images/array-xdotxT.svg">

```python
x = da.ones((15, 15), chunks=(5, 5))
x.dot(x.T + 1)


### ND-Array - Compound Operations

<img src="images/array-xdotxT-mean.svg">

```python
x = da.ones((15, 15), chunks=(5, 5))
x.dot(x.T + 1) - x.mean()
```


### ND-Array - Compound Operations

<img src="images/array-xdotxT-mean-std.svg">

```python
import dask.array as da
x = da.ones((15, 15), chunks=(5, 5))
y = (x.dot(x.T + 1) - x.mean()).std()
```



-  Jupyter Notebook/Script/...
-  Xarray/NumPy/Pandas/Scikit-Learn/...
-  **Parallel algorithms**
-  Distributed execution
-  Sensible deployment
-  Cluster of hardware

<hr>

Dask mediates between Python users and distributed hardware


-  Jupyter Notebook/Script/...
-  Xarray/NumPy/Pandas/Scikit-Learn/...
-  Parallel algorithms
-  **Distributed execution**
-  Sensible deployment
-  Cluster of hardware

<hr>

Dask mediates between Python users and distributed hardware


### Dask is a task scheduler

Like `make`, but where each task is a short Python function

```python
(X + X.T) - X.mean(axis=0)  # Dask code turns into task graphs
```

<img src="images/grid_search_schedule-0.png" width="100%">


### Dask is a task scheduler

Like `make`, but where each task is a short Python function

```python
(X + X.T) - X.mean(axis=0)  # Dask code turns into task graphs
```

<img src="images/grid_search_schedule.gif" width="100%">


### Dask thinks about ...

-  Data locality
-  Load balancing
-  Scarce resources
-  Network communication
-  Resilience
-  Scaling up and down
-  Diagnostics
-  Profiling
-  ...



-  Jupyter Notebook/Script/...
-  Xarray/NumPy/Pandas/Scikit-Learn/...
-  Parallel algorithms
-  **Distributed execution**
-  Sensible deployment
-  Cluster of hardware

<hr>

Dask mediates between Python users and distributed hardware


-  Jupyter Notebook/Script/...
-  Xarray/NumPy/Pandas/Scikit-Learn/...
-  Parallel algorithms
-  Distributed execution
-  **Sensible deployment**
-  Cluster of hardware

<hr>

Dask mediates between Python users and distributed hardware


### Deploys on Standard Cluster Hardware

<img src="images/network-inverse.svg" width="50%" align="right">

-  HPC:
    -  SLURM
    -  PBS
    -  SGE
    -  LSF
    -  ...
-   Cloud with Kubernetes
-   Hadoop/Spark with Yarn


### Deploys on Standard Cluster Hardware

```python
>>> from dask_jobqueue import SLURMCluster
>>> cluster = SLURMCluster(cores=24,
                           memory="100GB",
                           project="my-project",
                           queue="regular")

>>> cluster.scale(10)  # ask for ten nodes

>>> cluster.adapt(minimum=0, maximum=100)  # or adapt nodes based on load
```

Integrates with widely deployed HPC job schedulers


### Deploys on Standard Cluster Hardware

```python
>>> from dask_kubenetes import KubeCluster
>>> cluster = KubeCluster(...,
                          ...,
                          ...,
                          ...)

>>> cluster.scale(10)  # ask for ten nodes

>>> cluster.adapt(minimum=0, maximum=100)  # or adapt nodes based on load
```

Or newer cloud technologies


### Deploys on Standard Cluster Hardware

```python
>>> from dask_yarn import YarnCluster
>>> cluster = YarnCluster(...,
                          ...,
                          ...,
                          ...)

>>> cluster.scale(10)  # ask for ten nodes

>>> cluster.adapt(minimum=0, maximum=100)  # or adapt nodes based on load
```

Or Hadoop clusters you may have


### Deploys on Standard Cluster Hardware

```python
>>> from dask.distributed import LocalCluster
>>> cluster = LocalCluster()
.
.
.
.
.
.
.
```

Or, most commonly, a laptop



### Demonstration

<iframe width="1200"
        height="600"
        src="https://www.youtube.com/embed/FXsgmwpRExM?start=1186"
        frameborder="0"
        allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen></iframe>


### Learn More

<img src="images/dask_icon.svg" width=20%>

[pangeo.io](https://pangeo.io)

[dask.org](https://dask.org)

[examples.dask.org](https://examples.dask.org)
