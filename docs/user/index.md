# GT4Py: GridTools for Python

GT4Py is a Python library to write typical weather and climate computations
in Python using a compact, high-level specification, and to run them efficiently
on multiple hardware platforms. It is part of the [GridTools](https://github.com/GridTools)
framework, a set of libraries and utilities to develop performance portable
applications in this area.

Currently, two generations live side by side in the library: `gt4py.next` and
`gt4py.cartesian`. `gt4py.next` is where all new development happens.
`gt4py.cartesian` is the previous generation, which is still maintained for
the models relying on it.

To install GT4Py or get involved in its development, check the
[project repository](https://github.com/GridTools/gt4py).

## GT4Py Next

`gt4py.next` is the current development version of GT4Py and the one to choose
for any new project.

Its language is fully embedded in Python: field operators and programs are
regular Python functions working on fields defined over named dimensions, on
both structured and unstructured meshes. They can be run directly by the
Python interpreter, where GT4Py acts as a thin, domain-specific layer on top
of standard array libraries, so they can be debugged and combined with other
Python code like any other function. The very same definitions can be handed
over to one of the performance backends, which generate optimized code for the
target hardware.

```{card} gt4py.next documentation →
---
link: next/index.html
---
Quickstart, guide to the DSL and examples.
```

## GT4Py Cartesian

`gt4py.cartesian` is the first generation of GT4Py. Stencils are written in
GTScript, a language using Python syntax which is always translated to
generated code, instead of being executed as regular Python, and which is
limited to Cartesian grids. It still delivers excellent performance, and it
is used by several models.

Its current status is:

- **Maintained**: it is part of the `gt4py` package, and the GT4Py team keeps
  fixing bugs, reviewing contributions and taking part in discussions with
  external developers.
- **Feature complete**: the core team does not develop new features for it.
- **Not recommended for new developments**: use `gt4py.next` instead.

```{card} gt4py.cartesian documentation →
---
link: cartesian/index.html
---
Quick start, GTScript language reference and API documentation.
```

`gt4py.next` and `gt4py.cartesian` are independent from each other and each one
has its own documentation. Note that although both can easily interoperate via
buffer sharing, code written for one of them does not work with the other.

## License

GT4Py is licensed under the terms of the
[BSD-3-Clause](https://github.com/GridTools/gt4py/blob/main/LICENSE) license.
