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

# Where, Offset, and domain

+++

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
from helpers import *

import gt4py.next as gtx

backend = None
# backend = gtfn_cpu
# backend = gtfn_gpu
```

```{code-cell} ipython3
a_np = np.arange(10.0)
b_np = np.where(a_np < 6.0, a_np, a_np*10.0)
print("a_np array: {}".format(a_np))
print("b_np array: {}".format(b_np))
```

### **Task**: replicate this example in gt4py

```{code-cell} ipython3
# TODO implement the field_operator


@gtx.program(backend=backend)
def program_where(a: gtx.Field[Dims[K], float], b: gtx.Field[Dims[K], float]):
    fieldop_where(a, out=b)
```

```{code-cell} ipython3
def test_where():
    a = gtx.as_field([K], np.arange(10.0), allocator=backend)
    b = gtx.as_field([K], np.zeros(shape=10), allocator=backend)
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
def fieldop_domain(a: gtx.Field[Dims[K], float]) -> gtx.Field[Dims[K], float]:
    return a * 10.0


@gtx.program(backend=backend)
def program_domain(a: gtx.Field[Dims[K], float], b: gtx.Field[Dims[K], float]):
    ...  # TODO write the call to fieldop_domain
```

```{code-cell} ipython3
def test_domain():
    a = gtx.as_field([K], np.arange(10.0), allocator=backend)
    b = gtx.as_field([K], np.arange(10.0), allocator=backend)
    program_domain(a, b, offset_provider={})

    assert np.allclose(b_np, b.asnumpy())
```

```{code-cell} ipython3
test_domain()
print("Test successful")
```

## where and domain

A combination of `where` and `domain` is useful in cases when an offset is used which exceeds the field size.

e.g. a field `a: gtx.Field[Dims[K], float]` with shape (10,) is applied an offset (`Koff`).

+++

### **Task**: combine `domain` and `where` to account for extra indices

Edit the code below such that:
 1. operations on field `a` are performed only up until the 8th index
 2. the domain is properly set accound for the offset

+++

#### Python reference

```{code-cell} ipython3
a_np_result = np.zeros(shape=10)
for i in range(len(a_np)):
    if a_np[i] < 8.0:
        a_np_result[i] = a_np[i + 1] + a_np[i]
    elif i < 9:
        a_np_result[i] = a_np[i]
print("a_np_result array: {}".format(a_np_result))
```

```{code-cell} ipython3
@gtx.field_operator
def fieldop_domain_where(a: gtx.Field[Dims[K], float]) -> gtx.Field[Dims[K], float]:
    return # TODO

@gtx.program(backend=backend)
def program_domain_where(a: gtx.Field[Dims[K], float], b: gtx.Field[Dims[K], float]):
    ... # TODO 
```

```{code-cell} ipython3
def test_domain_where():  
    a = gtx.as_field([K], np.arange(10.0), allocator=backend)
    b = gtx.as_field([K], np.zeros(shape=10), allocator=backend)
    program_domain_where(a, b, offset_provider={"Koff": K})
    
    assert np.allclose(a_np_result, b.asnumpy())
```

```{code-cell} ipython3
test_domain_where()
print("Test successful")
```

```{code-cell} ipython3

```
