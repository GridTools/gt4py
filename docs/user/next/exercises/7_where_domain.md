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

```{code-cell} ipython3
import numpy as np

import gt4py.next as gtx
from gt4py.next.iterator.embedded import MutableLocatedField 
from gt4py.next import float64, neighbor_sum, where
from gt4py.next import Dimension, DimensionKind, FieldOffset
```

```{code-cell} ipython3
def random_field(
    sizes, *dims, low: float = -1.0, high: float = 1.0
) -> MutableLocatedField:
    return gtx.as_field([*dims],
        np.random.default_rng().uniform(
            low=low, high=high, size=sizes
        )
    )

def zero_field(
    sizes, *dims: Dimension, dtype=float
) -> MutableLocatedField:
    return gtx.as_field([*dims], 
        np.zeros(shape=sizes, dtype=dtype)
    )
```

```{code-cell} ipython3
CellDim = Dimension("C")
```

## Conditional: where

+++

The `where` builtin works analogously to the numpy version (https://numpy.org/doc/stable/reference/generated/numpy.where.html)

Both require the same 3 input arguments:
- mask: a field of booleans or an expression evaluating to this type
- true branch: a tuple, a field, or a scalar
- false branch: a tuple, a field, of a scalar

+++

Take a simple numpy example, the `mask` here is a condition:

```{code-cell} ipython3
a_np = np.arange(10.0)
b_np = np.where(a_np < 6.0, a_np, a_np*10.0)
print("a_np array: {}".format(a_np))
print("b_np array: {}".format(b_np))
```

### **Task**: replicate this example in gt4py

```{code-cell} ipython3
@gtx.field_operator
def fieldop_where(a: gtx.Field[[CellDim], float64]) -> gtx.Field[[CellDim], float64]:
    return where(a < 6.0, a, a*10.0)

@gtx.program
def program_where(a: gtx.Field[[CellDim], float64],
            b: gtx.Field[[CellDim], float64]):
    fieldop_where(a, out=b) 
```

```{code-cell} ipython3
def test_where():
    a = gtx.as_field([CellDim], np.arange(10.0))
    b = gtx.as_field([CellDim], np.zeros(shape=10))
    program_where(a, b, offset_provider={})
    
    assert np.allclose(b_np, b.asnumpy())
```

```{code-cell} ipython3
test_where()
print("Test successful")
```

## Domain

+++

The same operation can be performed in gt4py by including the `domain` keyowrd argument on `field_operator` call

### **Task**: implement the same operation as above using `domain` instead of `where`

```{code-cell} ipython3
@gtx.field_operator
def fieldop_domain(a: gtx.Field[[CellDim], float64]) -> gtx.Field[[CellDim], float64]:
    return a*10.0

@gtx.program
def program_domain(a: gtx.Field[[CellDim], float64],
            b: gtx.Field[[CellDim], float64]):
    fieldop_domain(a, out=b, domain={CellDim: (6, 10)}) 
```

```{code-cell} ipython3
def test_domain():
    a = gtx.as_field([CellDim], np.arange(10.0))
    b = gtx.as_field([CellDim], np.arange(10.0))
    program_domain(a, b, offset_provider={})

    assert np.allclose(b_np, b.asnumpy())
```

```{code-cell} ipython3
test_domain()
print("Test successful")
```

## where and domain

A combination of `where` and `domain` is useful in cases when a certain domain exceeds the field size

e.g. a field `a: gtx.Field[[CellDim], float64]` with shape (10,) is applied a condition `where(a<13.0, .., ...)`

+++

### **Task**: combine `domain` and `where` to account for extra indices

Edit the code below such that operations on field `a` are performed only up until the 10th index

```{code-cell} ipython3
@gtx.field_operator
def fieldop_domain_where(a: gtx.Field[[CellDim], float64]) -> gtx.Field[[CellDim], float64]:
    return where(a<13.0, a*10.0, a)

@gtx.program
def program_domain_where(a: gtx.Field[[CellDim], float64],
            b: gtx.Field[[CellDim], float64]):
    fieldop_domain_where(a, out=b, domain={CellDim: (0, 10)}) 
```

```{code-cell} ipython3
def test_domain_where():
    a = gtx.as_field([CellDim], np.arange(10.0))
    b = gtx.as_field([CellDim], np.zeros(shape=10))
    program_domain_where(a, b, offset_provider={})

    assert np.allclose(a_np*10, b.asnumpy()[:10])
```

```{code-cell} ipython3
test_domain_where()
print("Test successful")
```

```{code-cell} ipython3

```
