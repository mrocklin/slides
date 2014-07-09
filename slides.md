% Blaze - Foundations for Array Computing
% Mark Wiebe, Matthew Rocklin - Continuum Analytics
% Thursday July 10th, 2014

## Motivation


The NumPy NDArray and Pandas DataFrame are *foundational data structures*.

They support an ecosystem


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
>>> csv[:3, ['name', 'balance']]
[('Alice', 100), ('Bob', -200), ('Charlie', 300)]
~~~~~~


## Data

~~~~~~
$ h5dump -H accounts.hdf5
HDF5 "accounts.hdf5" {
GROUP "/" {
   DATASET "accounts" {
      DATATYPE  H5T_COMPOUND {
         H5T_STD_I64LE "id";
         H5T_STRING {
            STRSIZE H5T_VARIABLE;
            STRPAD H5T_STR_NULLTERM;
            CSET unknown_cset;
            CTYPE H5T_C_S1;
         } "name";
         H5T_STD_I64LE "balance";
      }
      DATASPACE  SIMPLE { ( 5 ) / ( H5S_UNLIMITED ) }
   }
}
~~~~~~

~~~~~~
>>> from blaze import *
>>> hdf5 = HDF5('accounts.hdf5', '/accounts')
>>> hdf5.columns
['id', 'name', 'balance']

>>> hdf5.py[:3, ['name', 'balance']]
[('Alice', 100), ('Bob', -200), ('Charlie', 300)]
~~~~~~

## Data

empty space

~~~~~~
>>> from blaze.sql import *
>>> sql = SQL('postgresql://user:pass@hostname/', 'accounts')
>>> sql.columns
['id', 'name', 'balance']

>>> sql.py[:3, ['name', 'balance']]
[('Alice', 100), ('Bob', -200), ('Charlie', 300)]
~~~~~~


## Data API

Data Descriptors support native Python access

*   Iteration: iter(csv)
*   Extension: csv.extend(...)
*   Item access:  csv.py[:, [column, column, column]]

Data Descriptors support chunked access

*   Iteration: csv.chunks()
*   Extension: csv.extend_chunks(...)
*   Item access:  csv.dynd[:, [column, column, column]]


## Data API

Data Descriptors support native Python access

*   Iteration: iter(sql)
*   Extension: sql.extend(...)
*   Item access:  sql.py[:, [column, column, column]]

Data Descriptors support chunked access

*   Iteration: sql.chunks()
*   Extension: sql.extend_chunks(...)
*   Item access:  sql.dynd[:, [column, column, column]]

## Data descriptors enable interaction

**User-Data interaction**

~~~~~
>>> csv.py[:3, ['name', 'balance']]
[('Alice', 100), ('Bob', -200), ('Charlie', 300)]
>>> json.py[:3, ['name', 'balance']]
[('Alice', 100), ('Bob', -200), ('Charlie', 300)]
>>> hdf5.py[:3, ['name', 'balance']]
[('Alice', 100), ('Bob', -200), ('Charlie', 300)]
>>> sql.py[:3, ['name', 'balance']]
[('Alice', 100), ('Bob', -200), ('Charlie', 300)]
~~~~~


**Data-Data interaction**

~~~~~
# Migrate data from CSV file to SQL database
>>> sql.extend(iter(csv))

>>> hdf5.extend_chunks(sql.chunks())
~~~~~

