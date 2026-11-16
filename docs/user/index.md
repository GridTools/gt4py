# GT4Py: GridTools for Python

GT4Py is a Python library for generating high performance implementations of
stencil kernels from a high-level definition using regular Python functions.
It is part of the [GridTools](https://github.com/GridTools) framework, a set of
libraries and utilities to develop performance portable applications in the
area of weather and climate modeling.

Computations are written in a domain specific language embedded in Python and
translated to high-performance implementations for CPUs and GPUs. To install
GT4Py and get involved in its development, check the
[project repository](https://github.com/GridTools/gt4py).

## Two libraries in one package

GT4Py ships two embedded domain specific languages as separate subpackages.
They are **not compatible** with each other and each one has its own
documentation:

````{grid} 1 1 2 2
---
gutter: 3
---
```{grid-item-card} gt4py.cartesian
---
link: cartesian/index.html
---
The stable version of GT4Py. Stencil computations on 3-dimensional Cartesian
grids are expressed in the GTScript language: the horizontal axes are always
computed in parallel, while the vertical one can be iterated in sequential,
forward or backward, order.

**gt4py.cartesian documentation →**
```

```{grid-item-card} gt4py.next
---
link: next/index.html
---
The new version of GT4Py, supporting both structured and unstructured meshes.
Computations are expressed as field operators and programs working on fields
defined over named dimensions, and can be executed directly in Python or
compiled by one of the performance backends.

**gt4py.next documentation →**
```
````

## License

GT4Py is licensed under the terms of the
[BSD-3-Clause](https://github.com/GridTools/gt4py/blob/main/LICENSE) license.
