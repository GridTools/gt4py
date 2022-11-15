---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Hands on Session of the GT4Py Workshop

## Installation
Please follow the instructions in the README.md

## Import Modules

```{code-cell} ipython3
import numpy as np

from functional.ffront.decorator import program, scan_operator, field_operator
from functional.iterator.embedded import np_as_located_field
from functional.ffront.fbuiltins import Field, Dimension
from functional.common import DimensionKind
```

# Excercises

## 1. point-wise (Christoph)

```{code-cell} ipython3

```

## 2. reduction: gradient or laplace (Christoph)

```{code-cell} ipython3

```

## 3. neighbor access without reduction (dusk weight) (Hannes - diff 5)

```{code-cell} ipython3

```

## 4. Scan Operator

Configuration: Single column

```{code-cell} ipython3
CellDim = Dimension("Cell")
KDim = Dimension("K", kind=DimensionKind.VERTICAL)

num_cells = 1
num_layers = 6
grid_shape = (num_cells, num_layers)
```

Task: Port the following numpy scheme to a `scan_operator` below:

```{code-cell} ipython3
def graupel_toy_numpy(qc, qr, autoconversion_rate=0.1, sedimentaion_constant=0.05):
    """A toy model of a microphysics scheme contaning autoconversion and scavenging"""

    #Init
    sedimentation_flux = 0.0

    for cell, k in np.ndindex(qc.shape):
        
        # Autoconversion: Cloud Drops -> Rain Drops
        
        ## Obtain autoconversion tendency
        autoconv_t = qc[cell, k] * autoconversion_rate
        
        ## Apply tendency in place
        qc[cell, k] -= autoconv_t
        qr[cell, k] += autoconv_t

        # Sedimentaion
        
        ## Apply sedimentation flux from level above
        qr[cell, k] += sedimentation_flux

        ## Scavenging due to strong precipitation flux
        if qr[cell, k - 1] >= 0.1:
            sedimentation_flux = sedimentaion_constant * qr[cell, k]
        else:
            sedimentation_flux = 0.0

        # Remove mass due to sedimentation flux
        qr[cell, k] -= sedimentation_flux
```

### Template

Caveats of the `scan_operator`:
- Optional arguents are not supported
- `If statments` are currently not supported, use `ternary operator` instead

```{code-cell} ipython3
@scan_operator(axis=KDim, forward=True, init=(0.0, 0.0, 0.0))
def _graupel_toy_scan(
    carry: tuple[float, float, float],
    qc_in: float,
    qr_in: float,
) -> tuple[float, float, float]:

    ### Implement here ###

    return qc, qr, sedimentation_flux
```

Embed the `scan_operator` in a `field_operator`, such that the sedimentation flux is treated as a temporary:

```{code-cell} ipython3
@field_operator
def graupel_toy_scan(qc: Field[[CellDim, KDim], float], qr: Field[[CellDim, KDim], float],
    ) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    
    qc, qr, _ = _graupel_toy_scan(qc, qr)

    return qc, qr
```

### Test

You can test your implementaion by executing the following test:

```{code-cell} ipython3
# Initialize GT4Py fields to zero
qc = np_as_located_field(CellDim, KDim)(np.full(shape=grid_shape, fill_value=1.0, dtype=np.float64))
qr = np_as_located_field(CellDim, KDim)(np.full(shape=grid_shape, fill_value=0.0, dtype=np.float64))

#Initialize Numpy fields from GT4Py fields
qc_numpy = np.asarray(qc).copy()
qr_numpy = np.asarray(qr).copy()

#Execute the Numpy version of scheme
graupel_toy_numpy(qc_numpy, qr_numpy)

#Execute the GT4Py version of scheme
graupel_toy_scan(qc, qr, out=(qc, qr), offset_provider={})

# Compare results
assert np.allclose(np.asarray(qc), qc_numpy)
assert np.allclose(np.asarray(qr), qr_numpy)

print("Test successful")
```
