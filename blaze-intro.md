## Blaze: Foundations of Array Computing



## NumPy arrays and Pandas DataFrames are *foundational data structures*


## But they are restricted to memory

This is ok 95% of cases<br>
what about the other 5%?


## Computational Projects

Excellent streaming, out-of-core, and distributed alternatives exist

### NumPy like

*   SciDB
*   h5py
*   DistArray
*   Elemental
*   PETCs, Trillinos
*   Biggus
*   ...

Each approach is valid in a particular situation


## Computational Projects

Excellent streaming, out-of-core, and distributed alternatives exist

### Pandas like

*   Postgres/SQLite/MySQL/Oracle
*   PyTables, BColz
*   HDFS
    * Hadoop (Pig, Hive, ...)
    * Spark
    * Impala
*   ...

Each approach is valid in a particular situation


## Data Storage

Analagous variety of data storage techniques
</br>

- CSV - Accessible
- JSON - Pervasive, human/machine readable
- HDF5 - Efficient binary access
- BColz - Efficient columnar access
- Parquet - Efficient columnar access
- HDFS - Big!
- SQL - SQL!

</br>
Each approach is valid in a particular situation


## Spinning up a new technology is expensive


## Keeping up with a changing landscape frustrates developers


## Foundations address these challenges by being adaptable



### Blaze connects familiar interfaces to a variety of backends

Three parts

*   Abstract expression system around Tables, Arrays
*   Dispatch system from these expressions to computational backends
*   Dispatch system between data stored in different backends


Blaze looks and feels like Pandas

```Python
>>> from blaze import *
>>> iris = CSV('examples/data/iris.csv')

>>> t = Table(iris)
>>> t.head(3)
    sepal_length  sepal_width  petal_length  petal_width      species
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa

>>> t.species.distinct()
           species
0      Iris-setosa
1  Iris-versicolor
2   Iris-virginica
```


Blaze operates on various systems, like SQL

```Python
>>> from blaze import *
>>> iris = SQL('sqlite:///examples/data/iris.db', 'iris')

>>> t = Table(iris)
>>> t.head(3)
    sepal_length  sepal_width  petal_length  petal_width      species
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa

>>> t.species.distinct()
           species
0      Iris-setosa
1  Iris-versicolor
2   Iris-virginica
```


... and Spark

```Python
>>> import pyspark
>>> sc = pyspark.SparkContext("local", "blaze-demo")
>>> rdd = into(sc, csv)  # handle data conversion
>>> t = Table(rdd)
>>> t.head(3)
    sepal_length  sepal_width  petal_length  petal_width      species
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa

>>> t.species.distinct()
           species
0      Iris-setosa
1  Iris-versicolor
2   Iris-virginica
```


### Currently supports the following

*   Python -- (through `toolz`)
*   NumPy
*   Pandas
*   SQL -- (through `sqlalchemy`)
*   HDF5 -- (through `h5py`, `pytables`)
*   MongoDB -- (through `pymongo`)
*   Spark -- (through `pyspark`)
*   Impala -- (through `impyla`, `sqlalchemy`)


Blaze organizes other open source projects to achieve a cohesive and flexible data analytics engine

</br></br>
Blaze doesn't do any real work.

It orchestrates functionality already in the Python ecosystem.



## How does Blaze work?


Blaze separates the computations that we want to perform:

```python
>>> accounts = TableSymbol('accounts', '{id: int, name: string, amount: int}')

>>> deadbeats = accounts[accounts['amount'] < 0]['name']
```

from the representation of data:

```python
>>> L = [[1, 'Alice',   100],
...      [2, 'Bob',    -200],
...      [3, 'Charlie', 300],
...      [4, 'Dennis',   400],
...      [5, 'Edith',  -500]]
...
```

and then combines the two explicitly

```python
>>> list(compute(deadbeats, L))  # Iterator in, Iterator out
['Bob', 'Edith']

.
```


Separating expressions from data lets us switch backends

```python
>>> accounts = TableSymbol('accounts', '{id: int, name: string, amount: int}')

>>> deadbeats = accounts[accounts['amount'] < 0]['name']
```

so we can drive Pandas instead

```python
>>> df = DataFrame([[1, 'Alice',   100],
...                 [2, 'Bob',    -200],
...                 [3, 'Charlie', 300],
...                 [4, 'Dennis',   400],
...                 [5, 'Edith',  -500]],
...                 columns=['id', 'name', 'amount'])
```

getting the same result, but through different means

```python
>>> compute(deadbeats, df)  # DataFrame in, DataFrame out
1      Bob
4    Edith
Name: name, dtype: object
```


We now have the freedom to reach out into the ecosystem

```python
>>> accounts = TableSymbol('accounts', '{id: int, name: string, amount: int}')

>>> deadbeats = accounts[accounts['amount'] < 0]['name']
```

and write to newer technologies

```python
>>> import pyspark
>>> sc = pyspark.SparkContext('local', 'Blaze-demo')

>>> rdd = into(sc, L)  # migrate to Resilient Distributed Datastructure(RDD)
>>> rdd
ParallelCollectionRDD[0] at parallelize at PythonRDD.scala:315
```

evolving Blaze with the ecosystem

```python
>>> compute(deadbeats, rdd)  # RDD in, RDD out
PythonRDD[1] at RDD at PythonRDD.scala:43
>>> compute(deadbeats, rdd).collect()
['Bob', 'Edith']
```
