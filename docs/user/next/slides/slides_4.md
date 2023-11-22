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
KDim = gtx.Dimension("K", kind=DimensionKind.VERTICAL)
```

## Scan operators

Scan operators work in a similar fashion to iterations in Python.

```{code-cell} ipython3
x = np.asarray([1.0, 2.0, 4.0, 6.0, 0.0, 2.0, 5.0])
def x_iteration(x):
    for i in range(len(x)):
        if i > 0:
            x[i] = x[i-1] + x[i]
    return x
    
print("result array: \n {}".format(x_iteration(x)))
```

Visually, this is what `x_iteration` is doing: 

| ![scan_operator](../scan_operator.png) |
| :---------------------------------: |
|         _Iterative sum over K_      |

+++

`scan_operators` allow for the same computations and only require a return statement for the operation, for loops and indexing are handled in the background. The return `state` of the previous iteration is provided as its first argument.

This decorator takes 3 input arguments:
- `axis`: vertical axis over which operations have to be performed
- `forward`: True if order of operations is from bottom to top, False if from top to bottom
- `init`: scalar value from which the iteration starts

```{code-cell} ipython3
@gtx.scan_operator(axis=KDim, forward=True, init=0.0)
def add_scan(state: float, k_field: float) -> float:
    return state + k_field
```

```{code-cell} ipython3
k_field = gtx.as_field([KDim], np.asarray([1.0, 2.0, 4.0, 6.0, 0.0, 2.0, 5.0]))
result = gtx.as_field([KDim], np.zeros(shape=(7,)))

add_scan(k_field, out=result, offset_provider={})

print("result array: \n {}".format(result.asnumpy()))
```

Note: `scan_operators` can be called from `field_operators` and `programs`. Likewise, `field_operators` can be called from `scan_operators`

```{code-cell} ipython3

```
