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

```{code-cell} ipython3
CellDim = gtx.Dimension("Cell")
KDim = gtx.Dimension("K", kind=DimensionKind.VERTICAL)
grid_shape = (5, 6)
```

## Using conditionals on fields

To filter operations such that they are performed on only certain cells instead of the whole field, the `where` builtin was developed. 

This function takes 3 input arguments:
- mask: a field of booleans or an expression evaluating to this type
- true branch: a tuple, a field, or a scalar
- false branch: a tuple, a field, of a scalar

```{code-cell} ipython3
mask = gtx.as_field([CellDim], np.asarray([True, False, True, True, False]))
result = gtx.as_field([CellDim], np.zeros(shape=grid_shape[0]))
true_field = gtx.as_field([CellDim], np.asarray([11.0, 12.0, 13.0, 14.0, 15.0]))
false_field = gtx.as_field([CellDim], np.asarray([21.0, 22.0, 23.0, 24.0, 25.0]))

@gtx.field_operator
def conditional(mask: gtx.Field[[CellDim], bool], true_field: gtx.Field[[CellDim], float64], false_field: gtx.Field[[CellDim], float64]
) -> gtx.Field[[CellDim], float64]:
    return where(mask, true_field, false_field)

conditional(mask, true_field, false_field, out=result, offset_provider={})
print("mask array: {}".format(mask.asnumpy()))
print("true_field array:  {}".format(true_field.asnumpy()))
print("false_field array: {}".format(false_field.asnumpy()))
print("where return:      {}".format(result.asnumpy()))
```

## Using domain on fields

Another way to filter parts of a field where to perform operations, is to use the `domain` keyword argument when calling the field operator.

Note: domain needs both dimensions to be included with integer tuple values.

```{code-cell} ipython3
@gtx.field_operator
def add(a: gtx.Field[[CellDim, KDim], float64],
        b: gtx.Field[[CellDim, KDim], float64]) -> gtx.Field[[CellDim, KDim], float64]:
    return a + b   # 2.0 + 3.0

@gtx.program
def run_add_domain(a : gtx.Field[[CellDim, KDim], float64],
            b : gtx.Field[[CellDim, KDim], float64],
            result : gtx.Field[[CellDim, KDim], float64]):
    add(a, b, out=result, domain={CellDim: (1, 3), KDim: (1, 4)}) 
```

```{code-cell} ipython3
a = gtx.as_field([CellDim, KDim], np.full(shape=grid_shape, fill_value=2.0, dtype=np.float64))
b = gtx.as_field([CellDim, KDim], np.full(shape=grid_shape, fill_value=3.0, dtype=np.float64))
result = gtx.as_field([CellDim, KDim], np.zeros(shape=grid_shape))
run_add_domain(a, b, result, offset_provider={})

print("result array: \n {}".format(result.asnumpy()))
```

```{code-cell} ipython3

```
