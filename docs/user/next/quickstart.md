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

# Quickstart

This page walks through a complete `gt4py.next` computation: a Laplacian on a
two-dimensional Cartesian grid. It takes a few minutes to read and every code
cell is executed when the documentation is built, so what you see is the actual
output.

## Installation

GT4Py is a regular Python package:

```bash
pip install gt4py
```

This is all that is needed to run computations in Python. To compile them with
the `gtfn_cpu` backend, used at the end of this page, a C++ compiler supporting
C++17 must also be available in the system.

## Dimensions

In GT4Py, the axes of the grid have names. A **dimension** is created once, and
it is then used to declare fields, to move between grid points, and to select the
region where a computation runs.

```{code-cell} ipython3
import numpy as np

import gt4py.next as gtx

I = gtx.Dimension("I")
J = gtx.Dimension("J")
```

## Fields

A **field** is an array whose axes are labeled with dimensions. The set of grid
points where it is defined is its **domain**: a range of indices for each
dimension.

`gtx.as_field` creates a field from the data of an existing array, and
`gtx.zeros` allocates a new one:

```{code-cell} ipython3
nx, ny = 64, 48
grid = gtx.domain({I: nx, J: ny})

x, y = np.meshgrid(np.linspace(-2.0, 2.0, nx), np.linspace(-1.5, 1.5, ny), indexing="ij")
phi = gtx.as_field(grid, np.exp(-(x**2 + y**2)))
lap_phi = gtx.zeros(grid)

print(phi.domain)
```

The values are stored in a regular array, which here is a NumPy one:

```{code-cell} ipython3
type(phi.ndarray), phi.ndarray.shape, phi.ndarray.dtype
```

## A field operator

Computations are written as **field operators**: Python functions decorated with
`@gtx.field_operator`, which take fields and return fields. There are no loops
and no indices inside. An expression describes the result at every grid point,
and neighboring points are reached by _shifting_ a field along a dimension:
`phi(I + 1)` is the value of `phi` at the next point in the `I` direction.

This is the five-point Laplacian (with unit grid spacing):

```{code-cell} ipython3
IJField = gtx.Field[gtx.Dims[I, J], gtx.float64]


@gtx.field_operator
def laplacian(phi: IJField) -> IJField:
    return phi(I + 1) + phi(I - 1) + phi(J + 1) + phi(J - 1) - 4.0 * phi
```

The type annotations are part of the definition: they declare the dimensions and
the data type of the arguments and of the result.

## Running it

A field operator is called with its arguments plus the field where the result is
written, `out`. The Laplacian needs one neighbor on each side, so it can only be
computed in the interior of the grid. The `domain` argument selects those points:

```{code-cell} ipython3
interior = {I: (1, nx - 1), J: (1, ny - 1)}

laplacian(phi, out=lap_phi, domain=interior)
```

The points of `lap_phi` outside `interior` have not been touched. `asnumpy()`
returns the data as a NumPy array, for example to plot it:

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
for ax, field, title in zip(axes, (phi, lap_phi), ("phi", "laplacian(phi)")):
    image = ax.imshow(field.asnumpy().T, origin="lower", cmap="RdBu_r")
    ax.set(title=title, xlabel="I", ylabel="J")
    fig.colorbar(image, ax=ax)
```

No backend was selected, so this call was executed directly in Python, with the
operations on fields evaluated by NumPy. This is called **embedded** execution.
It is the default, it needs no compilation, and it can be debugged like any other
Python code.

## Fields outside operators

The expressions used inside a field operator also work on fields in plain Python,
which is handy to explore and to debug:

```{code-cell} ipython3
d_phi = phi(I + 1) - phi

print(d_phi.domain)
```

Note the domain of the result: `phi(I + 1)` is defined only where `phi` has a
neighbor to the right, and combining two fields gives a field on the points they
have in common. Named dimensions make this kind of bookkeeping automatic.

## Compiling with a backend

The same field operator can be compiled to optimized code by a **backend**.
`with_backend` returns a new version of the operator bound to that backend, here
`gtx.gtfn_cpu`, which generates C++ code using the GridTools library:

```{code-cell} ipython3
---
tags: [remove-stderr]
---
laplacian_cpu = laplacian.with_backend(gtx.gtfn_cpu)

lap_phi_cpu = gtx.zeros(grid, allocator=gtx.gtfn_cpu)
laplacian_cpu(phi, out=lap_phi_cpu, domain=interior)

np.allclose(lap_phi_cpu.asnumpy(), lap_phi.asnumpy())
```

The first call generates and compiles the code, which takes some seconds. Later
calls reuse the compiled version. Nothing changes in the definition of the
operator nor in the way it is called.

## Where to go next

- **The DSL**: what can be written inside operators.
  - [Field operators](dsl/field-operators.md): types, expressions, conditionals and built-in functions.
  - [Shifts and neighbors](dsl/shifts.md): Cartesian shifts and unstructured meshes.
  - [Scan operators](dsl/scan-operators.md): computations which advance along the vertical.
  - [Programs](dsl/programs.md): grouping several operator calls.
- **Using it from Python**:
  - [Grids and fields](python/grids-and-fields.md): dimensions, domains, allocation and connectivities.
  - [Running operators](python/running.md): embedded execution, backends and compilation.
- [Examples](examples/index.md): complete, small applications.
