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

# 1. Simple Addition

```{code-cell} ipython3
import gt4py.next as gtx

import numpy as np
from helpers import random_field_new, gtfn_cpu, gtfn_gpu
```

Next we implement the stencil and a numpy reference version, in order to verify them against each other.

```{code-cell} ipython3
C = gtx.Dimension("C")
n_cells = 42
```

```{code-cell} ipython3
def addition_numpy(a: np.array, b: np.array) -> np.array:
    c = a + b
    return c
```

```{code-cell} ipython3
@gtx.field_operator
def addition(
    a: gtx.Field[[C], float], b: gtx.Field[[C], float]
) -> gtx.Field[[C], float]:
    return a + b
```

```{code-cell} ipython3
def test_addition():
    backend = None
    # backend = gtfn_cpu
    # backend = gtfn_gpu

    domain = gtx.domain({C:n_cells})
    
    a = random_field_new(domain, allocator = backend)
    b = random_field_new(domain, allocator = backend)

    c_numpy = addition_numpy(a.asnumpy(), b.asnumpy())

    c = gtx.zeros(domain, allocator = backend)

    addition(a, b, out=c, offset_provider={})

    assert np.allclose(c.asnumpy(), c_numpy)

    print("Test successful!")
```

```{code-cell} ipython3
test_addition()
```
