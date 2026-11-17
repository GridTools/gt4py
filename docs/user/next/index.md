# GT4Py (next)

`gt4py.next` is the current generation of GT4Py, supporting both structured
and unstructured meshes. Computations are written as _field operators_ and
_programs_, regular Python functions working on fields defined over named
dimensions, which can be executed directly in Python (_embedded_ execution) or
compiled to a high-performance implementation by one of the backends.

Code written for <a href="../cartesian/index.html"><code>gt4py.cartesian</code></a>,
the previous generation of GT4Py for Cartesian grids, does not work with
`gt4py.next`.

```{toctree}
---
caption: User guide
maxdepth: 2
---
QuickstartGuide
```

```{toctree}
---
caption: Reference
maxdepth: 1
---
apiref
```
