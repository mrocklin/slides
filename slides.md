% Blaze - Foundations for Array Computing
% Mark Wiebe, Matthew Rocklin - Continuum Analytics
% Thursday July 10th, 2014

## Motivation


The NumPy NDArray and Pandas DataFrame are foundational data structures
for the numeric Python ecosystem.


##

But they are restricted to memory.

TODO: expand


## Computational projects

Lots of projects try to correct this

**NumPy-like**

*   Distarray
*   SciDB
*   Biggus
*   ...

**Pandas-like**

*   PyTables
*   SQL (Postgres, SQLite, ...)
*   The HDFS world
    *   Hadoop (Pig, Hive, ...)
    *   Spark
    *   Impala
    *   ...

Each approach is valid in a particular situation


## Data Projects

Data storage has an analagous collection of storage techniques

*   CSV - Accessible
*   JSON - Web transferrable
*   HDF5 - efficient access
*   BLZ - efficient columnar access
*   Parquet - efficient columnar access (HDFS)
*   SQL - Robust
*   HDFS - Big!
*   PyTables HDF5 - HDF5 + indices
*   ...

Each approach is valid in a particular situation


## Challenge

Spinning up a new technology is expensive


## Challenge

Adapting to this changing landscape frustrates data scientists


##  Foundation

Future foundations can't be data structures/projects, must be interfaces to a
variety of projects.


## What is Blaze?

Blaze abstracts array and tablular computation

*   Blaze expressions abstract over compute systems
*   Blaze data descriptors abstract over data storage
*   DataShape abstracts over data type systems

These abstractions enable interactions


## What is Blaze?

Blaze provides a symbolic system for Table computations

~~~~~~~
>>> accounts = TableSymbol('accounts', '{id: int, name: string, balance: int}')

>>> deadbeats = accounts[accounts['balance'] < 0]['name']
>>> deadbeats
accounts[accounts['balance'] < 0]['name']
~~~~~~~

Blaze provides interpreters to computation

~~~~~~~
>>> L = [(1, 'Alice', 100),
         (2, 'Bob', -200),
         (3, 'Charlie', 300),
         (4, 'Denis', 400),
         (5, 'Edith', -500)]

>>> compute(deadbeats, L)
['Bob', 'Edith']
~~~~~~~

## What is Blaze?

Blaze provides a symbolic system for Table computations

~~~~~~~
>>> accounts = TableSymbol('accounts', '{id: int, name: string, balance: int}')

>>> deadbeats = accounts[accounts['balance'] < 0]['name']
>>> deadbeats
accounts[accounts['balance'] < 0]['name']
~~~~~~~

Blaze provides interpreters to computation

~~~~~~~
>>> df = DataFrame([(1, 'Alice', 100),
                    (2, 'Bob', -200),
                    (3, 'Charlie', 300),
                    (4, 'Denis', 400),
                    (5, 'Edith', -500)],
                    columns=['id', 'name', 'balance'])

>>> compute(deadbeats, df)
1      Bob
4    Edith
Name: name, dtype: object
~~~~~~~

## Notebook Demo

## Data

~~~~~~
$ cat accounts.csv
id, name, balance
1, Alice, 100
2, Bob, -200
3, Charlie, 300
4, Denis, 400
5, Edith, -500
~~~~~~

~~~~~~
>>> from blaze import *
>>> csv = CSV('accounts.csv')
>>> csv.columns
['id', 'name', 'balance']
~~~~~~
