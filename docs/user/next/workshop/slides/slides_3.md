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

<img src="../images/logos/cscs_logo.jpeg" alt="cscs" style="width:270px;"/> <img src="../images/logos/c2sm_logo.gif" alt="c2sm" style="width:220px;"/>
<img src="../images/logos/exclaim_logo.png" alt="exclaim" style="width:270px;"/> <img src="../images/logos/mch_logo.svg" alt="mch" style="width:270px;"/>

```{code-cell} ipython3
import warnings
warnings.filterwarnings('ignore')
```

```{code-cell} ipython3
import numpy as np
import gt4py.next as gtx
from gt4py.next import where, Dims
```

```{code-cell} ipython3
Cell = gtx.Dimension("Cell")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
```

## Using conditionals on Fields

To conditionally compose a Field from two inputs, we borrow the `where` function from numpy. 

This function takes 3 input arguments:
- mask: a Field of booleans
- true branch: a tuple, a Field, or a scalar
- false branch: a tuple, a Field, of a scalar

```{code-cell} ipython3
mask = gtx.as_field([Cell], np.asarray([True, False, True, True, False]))

true_Field = gtx.as_field([Cell], np.asarray([11.0, 12.0, 13.0, 14.0, 15.0]))
false_Field = gtx.as_field([Cell], np.asarray([21.0, 22.0, 23.0, 24.0, 25.0]))

result = gtx.zeros(gtx.domain({Cell:5}))

@gtx.field_operator
def conditional(mask: gtx.Field[Dims[Cell], bool], true_Field: gtx.Field[Dims[Cell], gtx.float64], false_Field: gtx.Field[Dims[Cell], gtx.float64]
) -> gtx.Field[Dims[Cell], gtx.float64]:
    return where(mask, true_Field, false_Field)

conditional(mask, true_Field, false_Field, out=result, offset_provider={})
print("mask array: {}".format(mask.asnumpy()))
print("true_Field array:  {}".format(true_Field.asnumpy()))
print("false_Field array: {}".format(false_Field.asnumpy()))
print("where return:      {}".format(result.asnumpy()))
```

## Using domain on Fields

By default the whole `out` Field is updated. If only a subset should be updated, we can specify the output domain by passing the `domain` keyword argument when calling the Field operator.

```{code-cell} ipython3
@gtx.field_operator
def add(a: gtx.Field[Dims[Cell, K], gtx.float64],
        b: gtx.Field[Dims[Cell, K], gtx.float64]) -> gtx.Field[Dims[Cell, K], gtx.float64]:
    return a + b   # 2.0 + 3.0

@gtx.program
def run_add_domain(a : gtx.Field[Dims[Cell, K], gtx.float64],
            b : gtx.Field[Dims[Cell, K], gtx.float64],
            result : gtx.Field[Dims[Cell, K], gtx.float64]):
    add(a, b, out=result, domain={Cell: (1, 3), K: (1, 4)}) 
```

```{code-cell} ipython3
domain = gtx.domain({Cell: 5, K: 6})

a = gtx.full(domain, fill_value=2.0, dtype=np.float64)
b = gtx.full(domain, fill_value=3.0, dtype=np.float64)
result = gtx.zeros(domain)
run_add_domain(a, b, result, offset_provider={})

print("result array: \n {}".format(result.asnumpy()))
```

```{code-cell} ipython3

```
