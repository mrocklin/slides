Dask: Parallel PyData
---------------------


### PyData is Efficient and Intuitive

<hr>

### Dask extends to multiple cores for speed and scale


### Dask

*  Native Python library for parallel computation
*  Scales up to clusters and down to laptops
*  Supports familiar NumPy/Pandas interfaces
*  Fast with low overhead
*  Flexible for messy situations


### Three Examples

*  Dataframes on a cluster
    *  Demonstrate Pandas at scale
    *  Observe responsive user interface
*  Arrays on a laptop
    *  Visualize complex algorithms
    *  Learn about dask collections and tasks
*  Custom code
    *  Deal with messy situations
    *  Learn about scheduling


### Example 1

<hr>

### Dask.DataFrame on a Cluster with CSV/S3 data

<div>
<ul style="float: left;">
<li> Built from Pandas DataFrames
<li> Match Pandas Interface</li>
<li> Access data from Local/S3/HDFS </li>
<li> Fast, low latency </li>
<li> Responsive user interface </li>
</ul>

<img src="images/dask-dataframe-inverted.svg" style="float: left;" width="40%">
</div>


### Example 2

<hr>

### Dask.Array on a laptop with HDF5 data

<img src="images/dask-array.svg" width="60%">


### Dask.Array on a laptop with HDF5 data

*  Built from NumPy n-dimensional arrays
*  Matches NumPy interface (subset)
*  Solve medium-large problems on a laptop
*  Complex algorithms

<img src="images/dask-array.svg" width="60%">


### Example 3

<hr>

### Custom computations


### Custom computations

*  Manually handle functions to support messy situations
*  Life saver when collections aren't flexible enough
*  Combine futures with collections for best of both worlds
*  Scheduler provides resilient ane elastic execution

<img src="images/gridsearch.svg">



### Final Advice

<hr>

### Don't use Parallelism


### Don't use Parallelism

*  **Consider the following first:**
    1.  Use better algorithms
    2.  Try C/Fortran/Numba
    3.  Store data in efficient formats
    4.  Subsample
*  **If you have to parallelize:**
    1.  Start with your laptop (4 cores, 16GB RAM, 1TB disk)
    2.  Then a large workstation  (24 cores, 1TB RAM)
    3.  Finally scale out to a cluster
