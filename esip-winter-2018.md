Scaling Scientific Python
-------------------------

<img src="images/dask_horizontal_white.svg" width="30%">
<img src="images/xarray.png" width=30%>
<img src="images/jupyterhub.svg" width=30%>

*Matthew Rocklin*

Anaconda Inc.


<img src="http://dask.pydata.org/en/latest/_images/dask_horizontal_white.svg"
     alt="dask logo"
     width="40%">

<img src="images/grid_search_schedule.gif" width="100%">

-  Parallel task scheduler for Python
-  Parallelizes Numpy, Pandas, Scikit-Learn, ...
-  Developed by NumPy, Pandas, Scikit-Learn, Jupyter, ...
-  Light weight, well supported, BSD licensed


### Dask.DataFrame

<img src="images/dask-dataframe-inverted.svg" width="30%">

    import pandas as pd
    df = pd.read_csv('myfile.csv', parse_dates=['timestamp'])
    df.groupby(df.timestamp.dt.hour).value.mean()

    import dask.dataframe as dd
    df = dd.read_csv('hdfs://myfiles.*.csv', parse_dates=['timestamp'])
    df.groupby(df.timestamp.dt.hour).value.mean()


### Dask.array

<img src="images/dask-array.svg" width="60%">

    import numpy as np
    x = np.random.random(...)
    u, s, v = np.linalg.svd(x.dot(x.T))

    import dask.array as da
    x = da.random.random(..., chunks=(1000, 1000))
    u, s, v = da.linalg.svd(x.dot(x.T))


### Dask Array on a Cluster

<iframe width="560" height="315"
src="https://www.youtube.com/embed/cxcq35aruG0?ecver=1" frameborder="0"
gesture="media" allow="encrypted-media" allowfullscreen></iframe>



### Dask array translates Numpy API

<hr>

### into task graphs of many Numpy operations


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



## Deployment


### Pangeo endeavors to scale XArray workflows

-  Algorithm support
-  Deployment on HPC systems
-  Deployment on Cloud systems
    -  Standing up distributeed systems
    -  Accessibility
    -  User management
    -  File formats
-  ...


### Pangeo endeavors to scale XArray workflows

-  Algorithm support
-  Deployment on HPC systems
-  **Deployment on Cloud systems**
    -  Standing up distributeed systems
    -  Accessibility
    -  User management
    -  File formats
-  ...


<iframe width="1024"
        height="600"
        src="https://www.youtube.com/embed/rSOJKbfNBNk?ecver=1"
        frameborder="0"
        gesture="media"
        allow="encrypted-media" allowfullscreen></iframe>


### pangeo.pydata.org

-  JupyterHub-on-Kubernetes for notebooks and user management
-  Dask-on-Kubernetes for computation
-  XArray for user code
-  Data file formats
    -  NetCDF + FUSE
    -  Zarr (experimental cloud-friendly format)
    -  HSDS (see talk later this afternoon)
    -  GRIB, GeoTIFF, images, ... community provides solutions


### Some of the people and organizations responsible

-  Alistair Miles - Oxford - CGGH
-  Jacob Tomlinson - UK Met Informatics Lab
-  Joe Hamman - NCAR - NSF/Pangeo
-  Martin Durant - Anaconda
-  Matthew Pryor - UK Met CEDADev
-  Matthew Rocklin - Anaconda - NSF/Pangeo, Moore
-  Ryan Abernathy - Columbia - NSF/Pangeo
-  Stephan Hoyer - Google
-  Yuvi Panda - UC Berkeley / Jupyter - Moore
-  Dask, XArray, Jupyter, ... communities

<img src="images/moore.png" width="20%">
<img src="images/Anaconda_Logo.png" width="20%">
<img src="images/NSF.png" width="10%">
<img src="images/DARPA_Logo.jpg" width="20%">
<img src="images/mo-logo.svg" width="20%">


### Building this was easy

<hr>

### because we tapped community expertise


### No one person knows enough to build these systems


### No organization knows enough to build these systems


### No one funding source fully encompasses this project


### Encourage Multi-Organization Collaborations

-   Example:  Pangeo NSF Earthcube award
    -  **Columbia:** Scientific user community
    -  **NCAR:** HPC and dataset stewards
    -  **Anaconda Inc:** Open source developers
    -  ...

    Funding arose after constant use by XArray community
-   ...



## Questions?

### [pangeo.pydata.org](http://pangeo.pydata.org)

<hr>

<img src="images/dask_horizontal_white.svg" width="30%">
<img src="images/xarray.png" width=30%>
<img src="images/jupyterhub.svg" width=30%>

<hr>

<img src="images/moore.png" width="20%">
<img src="images/Anaconda_Logo.png" width="20%">
<img src="images/NSF.png" width="10%">
<img src="images/DARPA_Logo.jpg" width="20%">
<img src="images/mo-logo.svg" width="20%">
