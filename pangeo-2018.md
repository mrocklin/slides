Technical Status Update
-----------------------

<img src="images/dask_icon.svg" width=20%>
<img src="images/pangeo/pangeo_card_green.png" width="20%">

*Matthew Rocklin*

Anaconda Inc.



### Process

1.  Connect to users
2.  Get machines
3.  Share machines dynamically among users
4.  Distribute users' libraries
5.  Run Dask on machines
6.  Execute XArray queries quickly
    -  Also for very large computations
7.  Handle permissions for users to save their results


### Process: HPC

1.  **Connect to users**
2.  Get machines
3.  Share machines dynamically among users
4.  Distribute users' libraries
5.  Run Dask on machines
6.  Execute XArray queries quickly
    -  **Also for very large computations**
7.  Handle permissions for users to save their results


### Process: Cloud

1.  Connect to users
2.  Get machines
3.  **Share machines dynamically among users**
4.  **Distribute users' libraries**
5.  Run Dask on machines
6.  Execute XArray queries quickly
    -  **Also for very large computations**
7.  **Handle permissions for users to save their results**



1.  Connect to Users
--------------------


### Success: pangeo.pydata.org demonstrates one-click access

### Challenge: HPC clusters still experts-only

-  SSH in
-  Ask for an interactive node
-  Launch Jupyter server
-  SSH-tunnel from laptop to get Jupyter
-  Launch Dask
-  SSH-tunnel to get Dask dashboard
-  Do work


### Solutions?

-  Socialize JupyterLab + SSH-tunneling to HPC users
-  Socialize JupyterHub to HPC admins
-  Ask JupyterHub community to develop Zero-to-JupyterHub-HPC documentation


2.  Get machines
----------------


3. Share machines dynamically among users
-----------------------------------------


### Success: HPC Systems have queues, projects, allocation policies

### Challenge: Kubernetes does not

-  Anyone can log into pangeo.pydata.org and charge us credits
-  Resources are first come first serve


### Solutions?

-  Let groups manage their own JupyterHub+Dask deployment
-  Use Kubernetes namespaces per user/group
-  Hide Kubernetes from user.  Build Dask-Hub service
-  Wait for Kubernetes to develop these features?
-  ...


4. Distribute users' Libraries
-------------------------------


### Success: On HPC users distribute code through their home directory

### Challenge: On Kubernetes they have to build custom images

-  Different groups have different software requests
-  Need to match their Jupyter Environment with Dask worker environment
-  Hard to build and match docker images


### Solutions?

-  Let groups manage their own JupyterHub+Dask deployment
-  Integrate Binder/repo2docker in some way
-  Use NFS on Kubernetes
-  Link Persisent Volume to workers in a read-only way?
-  Package and ship user's environment to Dask workers


5.  Run Dask on machines
------------------------


### Success: Easy utilities for HPC, Kubernetes, YARN, ...

### Success: Co-developed by many different groups

### Challenge: improve adaptive policies, tune, ...

-  [dask-jobqueue.readthedocs.io](https://dask-jobqueue.readthedocs.io)

    Joe Hamman (NCAR), Loic Esteve (INRIA), Guillaume (CRNS), ...

-  [dask-kubernetes.readthedocs.io](https://dask-kubernetes.readthedocs.io)

    Jacob Tomlinson (UK Met), Yuvi (UC Berkeley, BIDS)


6.  Execute Large XArray queries quickly
----------------------------------------


### Success: we do science on 100TB datasets

### Challenge: we don't do science on 1PB datasets


### Thought Experiment

-  **Problem**
    -  1PB dataset cut into 1GB chunks has 1,000,000 chunks
    -  Our computation has 100 operations, so 100,000,000 tasks total
    -  Each task consumes 1kB on the scheduler
    -  And takes 1ms of centralized overhead

-  **Costs**
    -  100GB of metadata
    -  27 hours of overhead


### Solutions?

-  Some basic tuning, 1ms and 1kB are conservative
-  Fuse operations
-  Fancier solutions

**TODO:** continue providing use cases


7.  Handle permissions for users to save results
------------------------------------------------


### Success: HPC users can write results back to NFS

### Challenge: Kubernetes users don't have permissions to cloud storage

-  Ideally they would log in to a Google/AWS/Azure account
-  Their credentials would propagate to their workers
-  They would be secured from other users, and us


### Process: HPC/Cloud

1.  *Connect to users*
2.  Get machines
3.  **Share machines dynamically among users**
4.  **Distribute users' libraries**
5.  Run Dask on machines
6.  Execute XArray queries quickly
    -  **Also for very large computations**
7.  **Handle permissions for users to save their results**


### Broad problems

1.  Access on HPC

    **Challenge**: Technical and social inertia

2.  User management on Kubernetes

    **Challenge**: Lack of cloud-native skillset


### Personal Suggestions

1.  **HPC**: Push on local groups to adopt JupyterHub

    Write and talk about it

    Connect it to today's pain points among HPC administrators

2.  **Cloud**: Fragment and shrink [pangeo.pydata.org]()

    Collaborate with other communities to produce

    -  [ocean.pangeo.io]()
    -  [esipfed.pangeo.io]()
    -  [nasa-access.pangeo.io]()
    -  [astropy.pangeo.io]()

    Build tooling to shift burden to sub-community maintainers

    We incubate other sub-communities, and require their engagement





### Dask provides

-  Parallel algorithms around Numpy, Pandas, Scikit-Learn
-  Parallel execution on local machines and clusters
-  Deployment solutions for HPC, Kubernetes, Yarn

```python
import dask.array as da
x = da.random.random(..., chunks=(1000, 1000))
y = x + x.T - x.mean(axis=0)
```

<img src="images/grid_search_schedule.gif" width="100%">
