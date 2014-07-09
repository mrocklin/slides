% Blaze
% Mark Wiebe, Matthew Rocklin
% Thursday July 10th, 2014

## Motivation

The NumPy NDArray and Pandas DataFrame serve as foundational data structures
for the numeric Python ecosystem.

They are restricted to memory.


## Other projects

Lots of projects try to correct this

**NumPy-like**

*   Distarray
*   SciDB
*   Biggus
*   ...

** Pandas-like**

*   PyTables
*   SQL (Postgres, SQLite, ...)
*   The HDFS world
    *   Hadoop (Pig, Hive, ...)
    *   Spark
    *   Impala
    *   ...

Each is valid in a particular situation


## Data Storage

*   CSV
*   JSON
*   HDF5
*   SQL
*   HDFS
*   PyTables HDF5
*   ...


## Challenge

Spinning up a new technology is expensive

## Challenge

Adapting to this changing landscape frustrates data scientists


## What is Blaze?

Blaze provides a standard interface to Array* and DataFrame computation

Blaze provides simple hooks to compute on a variety of computational backends

