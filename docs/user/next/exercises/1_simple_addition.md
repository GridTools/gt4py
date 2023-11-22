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
from helpers import *
```

Next we implement the stencil and a numpy reference version, in order to verify them against each other.

```{code-cell} ipython3
def addition_numpy(
    a: np.array, b: np.array,
) -> np.array:
    c = a + b
    return c
```

```{code-cell} ipython3
@gtx.field_operator
def addition(
    a: gtx.Field[[C], float],
    b: gtx.Field[[C], float],
) -> gtx.Field[[C], float]:
    c = a
    return c
```

```{code-cell} ipython3
def test_mo_nh_diffusion_stencil_06():
    
    a = random_field((n_cells), C)
    b = random_field((n_cells), C)
    
    c_numpy = addition_numpy(
        a.asnumpy(), b.asnumpy()
    )

    c = zero_field((n_cells), C)

    addition(
        a, b, out=c, offset_provider={}
    )
    
    assert np.allclose(c.asnumpy(), c_numpy)

    print("Test successful!")
```

```{code-cell} ipython3
test_mo_nh_diffusion_stencil_06()
```
