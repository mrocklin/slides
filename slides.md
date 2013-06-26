% GroupBy \newline or \newline Packages Considered Slightly Harmful
% Matthew Rocklin
% June 26th, 2013

---------

\begin{figure}
\includegraphics<1>[width=\textwidth]{images/care}
\includegraphics<2>[width=\textwidth]{images/care2}
\includegraphics<3>[width=\textwidth]{images/care3}
\includegraphics<4>[width=\textwidth]{images/care4}
\includegraphics<5>[width=\textwidth]{images/care4b}
\includegraphics<6>[width=\textwidth]{images/care5}
\end{figure}


-------

GroupBy groups elements of a collection by their value under a function.

~~~~~~~~~~~Python
# https://gist.github.com/mrocklin/5722155
include [test_groupby.py](test_groupby.py)
~~~~~~~~~~~


-------

~~~~~~~~~~~Python
# https://gist.github.com/mrocklin/5722752
include [hist.py](hist.py)
~~~~~~~~~~~

-------

GroupBy groups elements of a collection by their value under a function.

~~~~~~~~~~~Python
# https://gist.github.com/mrocklin/5618992
include [groupby.py](groupby.py)
~~~~~~~~~~~


------------------------------

    mrocklin/workspace$ ack-grep "def groupby" */*/*.py 

    computations/computations/util.py
    80:def groupby(f, coll):

    itertoolz/itertoolz/core.py
    17:def groupby(f, coll):

    logpy/logpy/util.py
    90:def groupby(f, coll):

    megatron/megatron/util.py
    25:def groupby(f, coll):

    term/term/util.py
    90:def groupby(f, coll):

    tompkins/tompkins/util.py
    55:def groupby(f, coll):

-------------------------------------------

In which package should this function live?

\begin{figure}
\centering
\includegraphics<1>[width=\textwidth]{images/groupby0}
\includegraphics<2>[width=\textwidth]{images/groupby1}
\end{figure}

-------------------------------------------

~~~~~~~~~~~~~Python
                    from groupby import groupby
~~~~~~~~~~~~~

-------------------------------------------

*   Packaging general code with specific code is bad.  
    Separating code is good!

*   Dependency managers (e.g. PyPI with `easy_install`, `pip`, `conda`) are cheap!  
    Use them aggressively!

*   Demographic sensitive software engineering

*   When you do this aggressivly odd things happen.
    What are best practices?

