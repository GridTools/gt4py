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

It allows to express physical computations without the need for extensive parameters handling and manual code optimization.

The DSL expresses computations on a multi-dimensional grid. The horizontal axes are always computed in parallel, while the vertical axes can be iterated in sequential (forward or backward) order.

In addition, GT4Py provides functions to allocate arrays with memory layout suited for a particular backend.

The following backends are supported:
- `embedded`: runs the DSL code directly via the Python interpreter
- `gtfn`: transpiles the DSL to C++ code using the GridTools library
- `roundtrip`: generates a lower-level Python implementation of the DSL
- `dace`: uses the DaCe library to generate high performance machine code

This workshop will use the `embedded` backend.

## Current efforts

GT4Py is being used to port the ICON model from FORTRAN. Currently the **dycore**, **diffusion**, and **microphysics** are complete. 

The ultimate goal is to have a more flexible and modularized model that can be run on CSCS Alps infrastructure as well as other hardware.

Other models ported using Cartesian GT4Py are ECMWF's FVM (global and local area configuration) and GFDL's FV3.

+++

## Installation

You can install the library directly from GitHub using pip:

```{code-cell} ipython3
pip install --upgrade git+https://github.com/gridtools/gt4py.git
```

```{code-cell} ipython3
import warnings
warnings.filterwarnings('ignore')
```

```{code-cell} ipython3
import numpy as np
import gt4py.next as gtx
from gt4py.next import float64, neighbor_sum, where
from gt4py.next.common import DimensionKind
from gt4py.next.program_processors.runners import roundtrip
```

## Key concepts and application structure

- [Fields](#Fields),
- [Field operators](#Field-operators), and
- [Programs](#Programs).

+++

### Fields
Fields are **multi-dimensional array** defined over a set of dimensions and a dtype: `gtx.Field[[dimensions], dtype]`

The `as_field` builtin is used to define fields

```{code-cell} ipython3
CellDim = gtx.Dimension("Cell")
KDim = gtx.Dimension("K", kind=DimensionKind.VERTICAL)
grid_shape = (5, 6)
a = gtx.as_field([CellDim, KDim], np.full(shape=grid_shape, fill_value=2.0, dtype=np.float64))
b = gtx.as_field([CellDim, KDim], np.full(shape=grid_shape, fill_value=3.0, dtype=np.float64))

print("a definition: \n {}".format(a))
print("a array: \n {}".format(a.asnumpy()))
print("b array: \n {}".format(b.asnumpy()))
```

### Field operators

Field operators perform operations on a set of fields, i.e. elementwise addition or reduction along a dimension. 

They are written as Python functions by using the `@field_operator` decorator.

```{code-cell} ipython3
@gtx.field_operator
def add(a: gtx.Field[[CellDim, KDim], float64],
        b: gtx.Field[[CellDim, KDim], float64]) -> gtx.Field[[CellDim, KDim], float64]:
    return a + b
```

Direct calls to field operators require two additional arguments: 
- `out`: a field to write the return value to
- `offset_provider`: empty dict for now, explanation will follow

```{code-cell} ipython3
result = gtx.as_field([CellDim, KDim], np.zeros(shape=grid_shape))
add(a, b, out=result, offset_provider={})

print("result array \n {}".format(result.asnumpy()))
```

### Programs

+++

Programs are used to call field operators to mutate the latter's output arguments.

They are written as Python functions by using the `@program` decorator. 

This example below calls the `add` field operator twice:

```{code-cell} ipython3
# @gtx.field_operator
# def add(a, b):
#    return a + b

@gtx.program
def run_add(a : gtx.Field[[CellDim, KDim], float64],
            b : gtx.Field[[CellDim, KDim], float64],
            result : gtx.Field[[CellDim, KDim], float64]):
    add(a, b, out=result) # 2.0 + 3.0 = 5.0
    add(b, result, out=result) # 5.0 + 3.0 = 8.0
```

```{code-cell} ipython3
result = gtx.as_field([CellDim, KDim], np.zeros(shape=grid_shape))
run_add(a, b, result, offset_provider={})

print("result array: \n {}".format(result.asnumpy()))
```

```{code-cell} ipython3

```
