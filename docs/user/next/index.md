# GT4Py (next)

`gt4py.next` is a Python library to write the stencil computations of weather
and climate models as regular Python functions, and to run them efficiently on
CPUs and GPUs. It supports Cartesian grids and unstructured meshes.

This is how it looks:

```python
import gt4py.next as gtx

I = gtx.Dimension("I")
J = gtx.Dimension("J")

IJField = gtx.Field[gtx.Dims[I, J], gtx.float64]


@gtx.field_operator
def laplacian(phi: IJField) -> IJField:
    return phi(I + 1) + phi(I - 1) + phi(J + 1) + phi(J - 1) - 4.0 * phi


grid = gtx.domain({I: 64, J: 48})
phi = gtx.ones(grid)
result = gtx.zeros(grid)

# Run it in Python, on NumPy arrays ...
laplacian(phi, out=result, domain={I: (1, 63), J: (1, 47)})

# ... or compile it to C++
laplacian_cpu = laplacian.with_backend(gtx.gtfn_cpu)
laplacian_cpu(phi, out=result, domain={I: (1, 63), J: (1, 47)})
```

Three ideas are behind this code:

- **Fields have named dimensions.** A field is an array defined over a domain
  of the grid, and its axes are identified by name instead of by position.
  Neighbors are accessed by shifting a field along a dimension, `phi(I + 1)`, or
  through the connectivities of an unstructured mesh.
- **Computations are operators on whole fields.** There are no loops and no
  indices. The operators are written in a subset of Python, they can be composed
  freely, and they are checked when they are defined.
- **The same code runs in different ways.** By default it is executed directly
  in Python, which is ideal to develop and to debug. For production, a backend
  compiles it to optimized code for CPUs or GPUs.

The [Quickstart](quickstart.md) explains this example step by step.

```{toctree}
---
caption: Getting started
maxdepth: 1
---
quickstart
```

```{toctree}
---
caption: The DSL
maxdepth: 1
---
dsl/field-operators
dsl/shifts
dsl/scan-operators
dsl/programs
```

```{toctree}
---
caption: Using it from Python
maxdepth: 1
---
python/grids-and-fields
python/running
```

```{toctree}
---
caption: Examples
maxdepth: 2
---
examples/index
```

```{note}
Code written for <a href="../cartesian/index.html"><code>gt4py.cartesian</code></a>,
the previous generation of GT4Py, does not work with `gt4py.next`.
```
