---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# GT4Py Next

`gt4py.next` is a Python library to write the stencil computations of weather
and climate models as regular Python functions, and to run them efficiently on
CPUs and GPUs. It supports Cartesian grids and unstructured meshes.

It is the current generation of GT4Py. Code written for
<a href="../cartesian/index.html"><code>gt4py.cartesian</code></a>, the previous
generation, is not compatible with it.

This is what it looks like:

```{code-cell} ipython3
---
tags: [remove-stderr]
---
import gt4py.next as gtx

I = gtx.Dimension("I")
J = gtx.Dimension("J")

IJField = gtx.Field[gtx.Dims[I, J], gtx.float64]


@gtx.field_operator
def laplacian(phi: IJField) -> IJField:
    return phi(I + 1) + phi(I - 1) + phi(J + 1) + phi(J - 1) - 4.0 * phi


if __name__ == "__main__":
    domain = gtx.domain({I: 64, J: 48})
    phi = gtx.ones(domain)
    result = gtx.zeros(domain)

    # Run it in Python, on NumPy arrays ...
    laplacian(phi, out=result, domain={I: (1, 63), J: (1, 47)})

    # ... or compile it to C++
    laplacian_cpu = laplacian.with_backend(gtx.gtfn_cpu)
    laplacian_cpu(phi, out=result, domain={I: (1, 63), J: (1, 47)})
```

Three ideas are behind this code:

- **Fields have named dimensions.** A field is an array defined over a domain
  of the grid, and its axes are identified by name instead of by position.
  Neighbors are accessed by calling a field with an offset, `phi(I + 1)`, or
  through the connectivities of an unstructured mesh.
- **Computations are operators on whole fields.** There are no loops and no
  indices. The operators are written in a subset of Python and can call each
  other. Type errors are reported as soon as an operator is defined.
- **The same code runs in different ways.** By default it is executed directly
  in Python, which is ideal for development and debugging. For production, a
  backend compiles it to optimized code for CPUs or GPUs.

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
caption: Data structures
maxdepth: 1
---
python/grids-and-fields
```

```{toctree}
---
caption: GT4Py DSL
maxdepth: 1
---
dsl/field-operators
dsl/premap
dsl/scan-operators
dsl/programs
```

```{toctree}
---
caption: Execution
maxdepth: 1
---
python/running
```
