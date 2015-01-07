## Outline

1.  Expression Chunking with Pandas has been successful [notebook
1](http://nbviewer.ipython.org/url/blaze.pydata.org/notebooks/timings-csv.ipynb), [notebook 2](http://nbviewer.ipython.org/url/blaze.pydata.org/notebooks/timings-bcolz.ipynb), but doesn't generalize.
2.  Simple task scheduling with [`dask`](http://dask.readthedocs.org/en/latest/)
3.  Build simple OOC-NumPy operations with dask by hand [notebook](http://nbviewer.ipython.org/github/ContinuumIO/dask/blob/master/notebooks/simple-numpy-sum.ipynb)
4.  Blaze builds these for us
5.  Running schedules efficiently
    [blog 1](http://mrocklin.github.com/blog/work/2014/12/27/Towards-OOC)
    [blog 2](http://mrocklin.github.com/blog/work/2014/12/30/Towards-OOC-Frontend)
    [blog 3](http://mrocklin.github.com/blog/work/2015/01/06/Towards-OOC-Scheduling)
6.  Possible directions



### Expression chunking and simple communication pattern

![](images/chunking.png)



### Blaze currently generates dasks for the following:

1.  Elementwise operations (like `+`, `*`, `exp`, `log`, ...)
2.  Dimension shuffling like `np.transpose`
3.  Tensor contraction like `np.tensordot`
4.  Reductions like `np.mean(..., axis=...)`
5.  All combinations of the above

### Blaze doesn't yet generate the following:

1.  Slicing (though this should be easy to add)
2.  Solve, SVD, QR, or any more complex linear algebra.
3.  Anything that NumPy can't do.


### A useful class of operations:

    top(in_memory_function, 'z', 'ij', 'x', 'ijk', 'y', 'jki')

### Examples

    top(lambda x: x + 1, 'out', 'ij', 'x', 'ij')  # embarassingly parallel

    top(np.transpose, 'out', 'ij', 'x', 'ji')  # transpose

    top(dotmany, 'out', 'ik', 'x', 'ij', 'y', 'jk')  # blocked dot product

    def dotmany(A, B):
        return sum(map(np.dot, A, B))



### Dask inlining and avoiding temporaries

![](images/dask/uninlined.png)


### Dask inlining and avoiding temporaries

![](images/dask/inlined.png)



### Embarrassingly parallel computation

    expr = (((A + 1) * 2) ** 3)

![](images/dask/embarrassing.gif)


### More complex computation

    expr = (B - B.mean(axis=0)) + (B.T / B.std())

![](images/dask/normalized-b.gif)


### Fail case

    expr = (A.T.dot(B) - B.mean(axis=0))

![](images/dask/fail-case.gif)



## Future Directions

### Immediate directions

1.  Make it actually work (find and fix memory leak)
2.  Better on-disk caching (fix chest)
3.  Better concurrent scheduler?  (feedback welcome, I'm out of my element)
4.  Other applications for `dask` than `Array`?

### Bigger questions

1.  Distributed-memory scheduler
2.  Do we care?  Relative value of ndarrays vs tables.  This project comes at a cost.
3.  Who else can work on this?

