Visualizing Parallel Computations in Dask
-----------------------------------------

<img src="images/dask_icon.svg" width=20%>

*Matthew Rocklin*

Continuum Analytics


### Disclaimer:  I know very little about visualization

<hr>

### Performance optimization in distributed systems is hard

<hr>

### Visualization drives development and elevates conversation


### PyData

*  Fast and intuitive libraries like NumPy, Pandas, and Scikit-Learn
*  Running at near-optimal machine performance
*  On a single core and in RAM

### Dask

*  Parallel and distributed computing library
*  Complements existing PyData ecosystem
*  Acheives performance through flexible algorithms and smart scheduling

### Visualization/interaction tools today

*   Graphviz
*   IPython Widgets
*   Bokeh Server
*   Jupyter Lab


### Visualization ...

*   Builds intuition around parallel algorithms
*   Relaxes user anxiety with feedback
*   Exposes state and history of distributed runtime
*   Elevates level of conversation between users and developers


### We've just seen ...

*   Algorithm visualization builds intuition
*   Realtime feedback from execution system reduces anxiety
*   This was on a single laptop machine

### But distributed computing is more complex

*   Information scattered throughout the cluster
*   Communication costs, serialization, disk reads, etc..
*   Different workers with different capabilities
*   Asynchronous execution

### Progressbars and node-link diagrams are just a start



### Bokeh - web viz from Python

<img src="http://bokeh.pydata.org/en/latest/_static/images/logo.png">

*Bokeh is a Python interactive visualization library that targets modern web
browsers*

Bokeh maintains shared state between JS Clients and my Python server


### Bokeh - web viz from Python

<img src="http://bokeh.pydata.org/en/latest/_static/images/logo.png">

### Setup

    data = bokeh.ColumnDataSource({'start': [], 'stop': [],
                                   'color': [], 'core-id': [],
                                   'name': []})

    plot = bokeh.plotting.figure(title='Task Stream')
    plot.rect(x='start', y='stop', color='color', y='core-id', source=data)
    plot.text(x='start', y='stop', text='name', text_align='center')

### Update

    data.update({'start': [...], 'stop': [...], 'color': [...], 'core-id': [...]})


### Things I needed that Bokeh delivered

*   Write in Python and manage in Python
*   Customized plots without much code (~700 lines)
*   Responsive real-time updates on streaming data
*   Powerful client-side rendering (100,000 rectangles)

### Anti-goals

*   Didn't need trivial API: `plot(x, y)`
*   Only needed web graphics
*   Was willing to put in a few hours of work


### Visualization driven development

When faced with a new performance challenge I now create the plot I need
*before* I begin benchmarking or development.

No longer have to hunt for performance issues.
Problems announce themselves loudly.

Two examples:

*   Work stealing
*   Rolling aggregations in dask.dataframe


### Recent talks

*   [SciPy - July, 2016](https://www.youtube.com/watch?v=PAGjm4BMKlk):

    Overview, author custom algorithms, some machine learning

*   [PyData DC - October 2016](https://www.youtube.com/watch?v=EEfI-11itn0)

    Fine-grained parallelism, scheduling motivation and heuristics

*   Plotcon - November, 2016

    Visualization of distributed systems


### Final Thoughts

*   Dask provides parallelism for Python
    *   Parallel NumPy, Pandas, Scikit-Learn, etc..
    *   Built on an arbitrary computational task scheduler
*   Distributed scheduling of arbitrary graphs is hard
    *   Benefits from on-the-fly measurement
    *   Useful for ad-hoc situations


### Acknowledgements

*  Countless open source developers
*  SciPy developer community
*  Continuum Analytics
*  XData Program from DARPA

<img src="images/moore.png">

<hr>

### Questions?

<img src="images/grid_search_schedule.gif" width="100%">


### Algorithm flexibility


### Map/Shuffle/Reduce (Hadoop/Spark)

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

#### Dask

<img src="images/svd-compressed.svg" width="50%">

