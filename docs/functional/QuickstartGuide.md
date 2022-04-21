---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# GT4Py User Quickstart Guide

+++

## Info for writing the guide
Goals:
- Read by someone completely new
- Focus on unstructured
They will be able to:
- Install GT4Py
- Write simple code
- Understand basic concepts of GT4Py
Examples could be:
- Laplacian
    - Main concepts: fields, field operators, programs, reductions
- Realistic example: diffusion, advection, solve some simple PDE on a 2D grid

+++

## Installation

GT4Py is distributed as a Python package and can be installed directly from GitHub:

```{code-cell} ipython3
#! pip install git+https://github.com/gridtools/gt4py.git@functional
```

## Concepts

### Basics

We will shortly compute the sum of two 2-dimensional arrays using GT4Py, but let us first import some commonly used tools from GT4Py: 

```{code-cell} ipython3
import numpy as np

from functional.ffront.fbuiltins import Field, float32
from functional.iterator.runtime import CartesianAxis
from functional.ffront.decorator import field_operator, program
from functional.iterator.embedded import np_as_located_field
```

GT4Py operates on *fields*, therefore a 2D array will also have to be represented as a `Field`.
*We declare dims then make field of given size blablabla...*

```{code-cell} ipython3
IDim = CartesianAxis("I")
JDim = CartesianAxis("J")

grid_width = 5
grid_height = 6
grid_shape = (grid_width, grid_height)

a_value = 2.0
b_value = 3.0
a = np_as_located_field(IDim, JDim)(np.full(shape=grid_shape, fill_value=a_value))
b = np_as_located_field(IDim, JDim)(np.full(shape=grid_shape, fill_value=b_value))
```

To define operations involving one or more fields, we will use *field operators*. Field operators are pure functions (i.e. functions without side effects) that take immutable `Field`s as arguments and output another `Field` as a result. Field operators must be declared with the `@field_operator` decorator, and are allowed to use a certain subset of the Python syntax.

```{code-cell} ipython3
@field_operator
def add(a : Field[[IDim, JDim], float32],
        b : Field[[IDim, JDim], float32]) -> Field[[IDim, JDim], float32]:
    return a + b
```

*Programs* are similar to fields operators, but they allow mutability of the arguments. Programs must be declared with the `@program` decorator and are allowed to use a different subset of the Python syntax compared to field operators. Programs are used to call and chain field operators:

```{code-cell} ipython3
@program
def compute_sum(a : Field[[IDim, JDim], float32],
                b : Field[[IDim, JDim], float32],
                out : Field[[IDim, JDim], float32]):
    add(a, b, out=out)
```

Finally, we will call the program we just declared to compute 2 + 3:

```{code-cell} ipython3
result = np_as_located_field(IDim, JDim)(np.zeros(shape=grid_shape))
compute_sum(a, b, result, offset_provider={})

print("{} + {} = {} ± {}".format(a_value, b_value, np.average(np.asarray(result)), np.std(np.asarray(result))))
```

### Unstructured grids and connectivity

***E2V, V2E, reductions***

To get familiar with using fields on explicitly connected grids, we will take the surface of a die as our domain:

![grid_topo](die_grid.png)

The <span style="color: #C02020">faces</span> and the <span style="color: #0080FF">vertices</span> of the die have been numbered with zero-based indices. (The same 3D vertex may show up at multiple locations on the unfolded 2D map of the 3D die, in which case it's labelled with the same index everywhere.)

#### Define cell neighbours of vertices

#### Get number of dots on 0th/1st/2nd cell adjacent to vertex

#### Calculate total number of dots next to a vertex




+++

## Examples

+++

### Space derivatives

```{code-cell} ipython3
@field_operator
def ddx():
    pass

@program
def compute_ddx():
    pass

get_ddx()
```

### Diffusion

```{code-cell} ipython3
@field_operator
def op1():
    pass

@field_operator
def op1():
    pass

@program
def diffuse():
    pass

timestep = 0.01
endtime = 5
for time in range(0:endtime:timestep):
    diffuse()
```
