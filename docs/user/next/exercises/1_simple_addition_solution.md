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

# 1. Simple Addition

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

import gt4py.next as gtx
from gt4py.next import Dims
```

Next we implement the stencil and a numpy reference version, in order to verify them against each other.

```{code-cell} ipython3
I = gtx.Dimension("I")
J = gtx.Dimension("J")
size = 10
```

```{code-cell} ipython3
def addition_numpy(a: np.array, b: np.array) -> np.array:
    c = a + b
    return c
```

```{code-cell} ipython3
@gtx.field_operator
def addition(
    a: gtx.Field[Dims[I, J], float], b: gtx.Field[Dims[I, J], float]
) -> gtx.Field[Dims[I, J], float]:
    return a + b
```

```{code-cell} ipython3
def test_addition(backend=None):
    domain = gtx.domain({I: size, J: size})

    a_data = np.fromfunction(lambda xx, yy: xx, domain.shape, dtype=float)
    a = gtx.as_field(domain, a_data, allocator=backend)
    b_data = np.fromfunction(lambda xx, yy: yy, domain.shape, dtype=float)
    b = gtx.as_field(domain, b_data, allocator=backend)

    c_numpy = addition_numpy(a.asnumpy(), b.asnumpy())

    c = gtx.zeros(domain, allocator=backend)

    addition(a, b, out=c, offset_provider={})

    assert np.allclose(c.asnumpy(), c_numpy)

    print("Result:")
    print(c)
    print(c.asnumpy())
    
    # Plots
    fig, ax = plt.subplot_mosaic([
        ['a', 'b', 'c']
    ])
    ax['a'].imshow(a.asnumpy())
    ax['b'].imshow(b.asnumpy())
    ax['c'].imshow(c.asnumpy())

    print("\nTest successful!")
```

```{code-cell} ipython3
test_addition()
```

```{code-cell} ipython3

```
