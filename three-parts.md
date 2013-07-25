
-------------------------

### Mathematics 

\begin{columns}
  \begin{column}{0.5\textwidth}

$$ \mu = \mu + \Sigma H' (R + H\Sigma H')^{-1}  (H*\mu - \textrm{data}) $$
$$ \Sigma = \Sigma - \Sigma H' * (R + H \Sigma H')^{-1} H \Sigma $$
  \end{column}
  \begin{column}{0.5\textwidth}
    \begin{figure}[htbp]
    \centering
    \includegraphics[width=.4\textwidth]{images/Linear_subspaces_with_shading}
    \end{figure}
  \end{column}
\end{columns}

\hrule

### Low-level code generation

\begin{columns}
  \begin{column}{0.5\textwidth}
    \lstinputlisting{least_squares.f90}
  \end{column}
  \begin{column}{0.5\textwidth}
    \begin{figure}[htbp]
    \centering
    \includegraphics[width=.4\textwidth]{images/120px-Lapack}
    \end{figure}
  \end{column}
\end{columns}

\hrule

### Automation

\begin{columns}
  \begin{column}{0.5\textwidth}
    \lstinputlisting{automation.foo}
  \end{column}
  \begin{column}{0.5\textwidth}
    \begin{figure}[htbp]
    \centering
    \includegraphics[width=.5\textwidth]{images/search}
    \end{figure}
  \end{column}
\end{columns}
