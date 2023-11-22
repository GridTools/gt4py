---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

<img src="logos/cscs_logo.jpeg" alt="cscs" style="width:270px;"/> <img src="logos/c2sm_logo.gif" alt="c2sm" style="width:220px;"/>
<img src="logos/exclaim_logo.png" alt="exclaim" style="width:270px;"/> <img src="logos/mch_logo.svg" alt="mch" style="width:270px;"/>

+++

# GT4Py workshop

+++

## GT4Py: GridTools for Python

GT4Py is a Python library for generating high performance implementations of stencil kernels from a high-level definition using regular Python functions.

GT4Py is part of the GridTools framework: a set of libraries and utilities to develop performance portable applications in the area of weather and climate modeling.

**NOTE:** The `gt4py.next` subpackage contains a new and currently experimental version of GT4Py.

## Description

GT4Py is a Python library for expressing computational motifs as found in weather and climate applications. 

These computations are expressed in a domain specific language (DSL) which is translated to high-performance implementations for CPUs and GPUs.

In addition, GT4Py provides functions to allocate arrays with memory layout suited for a particular backend.

The following backends are supported:
- `None` aka _embedded_: runs the DSL code directly via the Python interpreter (experimental)
- `gtfn_cpu` and `gtfn_gpu`: transpiles the DSL to C++ code using the GridTools library
- `dace`: uses the DaCe library to generate optimized code (experimental)

In this workshop we will mainly use the _embedded_ backend.

## Current efforts

GT4Py is being used to port the ICON model from FORTRAN. Currently the **dycore**, **diffusion**, and **microphysics** are complete. 

The ultimate goal is to have a more flexible and modularized model that can be run on CSCS Alps infrastructure as well as other hardware.

Other models ported using GT4Py are ECMWF's FVM, in global (with `gt4py.next` and local area configuration (with `gt4py.cartesian`) and GFDL's FV3 (with `gt4py.cartesian`; original port by AI2).

+++

## Installation

After cloning the repository to $SCRATCH and setting a symlink to your home-directory

```
cd $SCRATCH
git clone --branch gt4py-workshop https://github.com/GridTools/gt4py.git
cd $HOME
ln -s $SCRATCH/gt4py
```

you can install the library with pip.

Make sure that GT4Py is in the expected location, remove `#` and run the cell)

```{code-cell} ipython3
#! pip install $HOME/gt4py
```

```{code-cell} ipython3
import warnings
warnings.filterwarnings('ignore')
```

```{code-cell} ipython3
import numpy as np
import gt4py.next as gtx
from gt4py.next import float64, neighbor_sum, where
```

## Key concepts and application structure

- [Fields](#Fields),
- [Field operators](#Field-operators), and
- [Programs](#Programs).

+++

### Fields
Fields are **multi-dimensional array** defined over a set of dimensions and a dtype: `gtx.Field[[dimensions], dtype]`.

Fields can be constructed with the following functions, inspired by numpy:

- `zeros`
- `full` to fill with a given value
- `as_field` to convert from numpy or cupy arrays

The first argument is the domain of the field, which can be constructed from a mapping from `Dimension` to range.

Optional we can pass
- `dtype` the description of type of the field
- `allocator` which describes how and where (e.g. GPU) the buffer is allocated.

Note: `as_field` can also take a sequence of Dimensions and infer the shape

```{code-cell} ipython3
Cell = gtx.Dimension("Cell")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)

domain = gtx.domain({Cell: 5, K: 6})

a = gtx.zeros(domain, dtype=float64)
b = gtx.full(domain, fill_value=3.0, dtype=float64)
c = gtx.as_field([Cell, K], np.fromfunction(lambda c, k: c*10+k, shape=domain.shape))

print("a definition: \n {}".format(a))
print("a array: \n {}".format(a.asnumpy()))
print("b array: \n {}".format(b.asnumpy()))
print("c array: \n {}".format(c.asnumpy()))
```

### Field operators

Field operators perform operations on a set of fields, i.e. elementwise addition or reduction along a dimension. 

They are written as Python functions by using the `@field_operator` decorator.

```{code-cell} ipython3
@gtx.field_operator
def add(a: gtx.Field[[Cell, K], float64],
        b: gtx.Field[[Cell, K], float64]) -> gtx.Field[[Cell, K], float64]:
    return a + b
```

Direct calls to field operators require two additional arguments: 
- `out`: a field to write the return value to
- `offset_provider`: empty dict for now, explanation will follow

```{code-cell} ipython3
result = gtx.zeros(domain)
add(a, b, out=result, offset_provider={})

print("result array \n {}".format(result.asnumpy()))
```

### Programs

+++

Programs are used to call field operators to mutate the latter's output arguments.

They are written as Python functions by using the `@program` decorator. 

This example below calls the `add` field operator twice:

```{code-cell} ipython3
@gtx.program
def run_add(a : gtx.Field[[Cell, K], float64],
            b : gtx.Field[[Cell, K], float64],
            result : gtx.Field[[Cell, K], float64]):
    add(a, b, out=result)
    add(b, result, out=result)
```

```{code-cell} ipython3
result = gtx.zeros(domain, dtype=float64)
run_add(a, b, result, offset_provider={})

print("result array: \n {}".format(result.asnumpy()))
```

```{code-cell} ipython3

```
