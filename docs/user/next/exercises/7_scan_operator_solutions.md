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

## Scan operator

+++

The unique feature of this operator is that it provides the return state of the previous iteration as its first argument (i.e., the result from the previous grid point). In other words, all the arguments of the current `return` will be available (as a tuple) in the next iteration from the first argument of the defined function.  

Example: A FORTRAN pseudocode for integrating a moisture variable (e.g., cloud water or water vapour) over a column could look as follows:


```FORTRAN
SUBROUTINE column_integral( var_in, rho, dz, var_out, ie, je, ke )
    ! Return the column integral of a moist species.
    INTEGER, INTENT (IN) :: &
      ie, je, ke         ! array dimensions of the I/O-fields (horizontal, horizontal, vertical)

    REAL (KIND=wp), INTENT (OUT) :: &
      q_colsum (ie,je) ! Vertically-integrated mass of water species

    REAL (KIND=wp), INTENT (IN) ::  &
      rho (ie,je,ke),  & 
      dz (ie,je,ke),   & ! height of model half levels
      var_in  (ie,je,ke) ! humidity mass concentration at time-level nnow
    
    !$acc parallel present( iq ) if (lzacc)
    !$acc loop gang
    DO j=1,je
      !$acc loop vector
      DO i=1,ie
        q_sum(i,j) = 0.0
      END DO
    END DO
    !$acc end parallel
    
    
    !$acc parallel present( iq, rho, hhl, q ) if (lzacc)
    DO k = 1, ke ! Vertical loop
      !$acc loop gang
      DO j=1,je
        !$acc loop vector
        DO i=1,ie
          q_colsum(i,j) = q_colsum(i,j) + var_in(i,j,k) * rho(i,j,k)* dz(i,j,k)
        END DO
      END DO
    END DO
    !$acc end parallel
END SUBROUTINE column_integral
```

Where:
- `var_in` is the 3D variable that will be summed up
- `q_colsum` is the resulting 2D variable
- `rho` the air density
- `dz`the thickness of the vertical layers

In the first loop nest, `column_sum` is set to zero for all grid columns. The vertical dependency enters on the RHS of the second loop nest `q_colsum(i,j) = q_colsum(i,j) + ...`

Using the `scan_operator` this operation would be written like this:

```python
@scan_operator(axis=KDim, forward=True, init=0.0)
def column_integral(float: state, float: var, float: rho, float: dz)
    """Return the column integral of a moist species."""
    return var * rho * dz + state
```

Here the vertical dependency is expressed by the first function argument (`state`).  This argument carries the return from the previous k-level and does not need to be specified when the function is called (similar to the `self` argument of Python classes). The argument is intialized to `init=0.0` in the function decorator (first loop nest above) and the dimension of the integral is specified with `axis=KDim`.


```python
q_colsum = column_integral(qv, rho, dz)
```

+++

#### Exercise: port a toy cloud microphysics scheme from python/numpy using the template of a `scan_operator` below

```{code-cell} ipython3
from helpers import *

import gt4py.next as gtx

backend = None
# backend = gtfn_cpu
# backend = gtfn_gpu
```

```{code-cell} ipython3
def toy_microphysics_numpy(qc, qr, autoconversion_rate=0.1, sedimentaion_constant=0.05):
    """A toy model of a microphysics scheme contaning autoconversion and scavenging"""

    sedimentation_flux = 0.0

    for cell, k in np.ndindex(qc.shape):
        # Autoconversion: Cloud Drops -> Rain Drops
        autoconversion_tendency = qc[cell, k] * autoconversion_rate

        qc[cell, k] -= autoconversion_tendency
        qr[cell, k] += autoconversion_tendency

        ## Apply sedimentation flux from level above
        qr[cell, k] += sedimentation_flux

        ## Remove mass due to sedimentation flux from the current cell
        qr[cell, k] -= sedimentation_flux
```

```{code-cell} ipython3
@gtx.scan_operator(axis=K, forward=True, init=(0.0, 0.0, 0.0))
def _graupel_toy_scan(
    state: tuple[float, float, float], qc_in: float, qr_in: float
) -> tuple[float, float, float]:
    autoconversion_rate = 0.1
    sedimentaion_constant = 0.05

    # unpack state of previous iteration
    _, _, sedimentation_flux = state

    # Autoconversion: Cloud Drops -> Rain Drops
    autoconv_t = qc_in * autoconversion_rate
    qc = qc_in - autoconv_t
    qr = qr_in + autoconv_t

    ## Add sedimentation flux from level above
    qr = qr + sedimentation_flux

    # Remove mass due to sedimentation flux
    qr = qr - sedimentation_flux

    return qc, qr, sedimentation_flux
```

```{code-cell} ipython3
@gtx.field_operator(backend=backend)
def graupel_toy_scan(
    qc: gtx.Field[Dims[C, K], float], qr: gtx.Field[Dims[C, K], float]
) -> tuple[
    gtx.Field[Dims[C, K], float],
    gtx.Field[Dims[C, K], float]
]:
    qc, qr, _ = _graupel_toy_scan(qc, qr)

    return qc, qr
```

```{code-cell} ipython3
def test_scan_operator():
    cell_k_domain = gtx.domain({C: n_cells, K: n_levels})
    
    qc = random_field(cell_k_domain, allocator=backend)
    qr = random_field(cell_k_domain, allocator=backend)

    qc_new = gtx.zeros(cell_k_domain, allocator=backend)
    qr_new = gtx.zeros(cell_k_domain, allocator=backend)

    # Initialize Numpy fields from GT4Py fields
    qc_numpy = qc.asnumpy().copy()
    qr_numpy = qr.asnumpy().copy()

    # Execute the Numpy version of scheme
    toy_microphysics_numpy(qc_numpy, qr_numpy)

    # Execute the GT4Py version of scheme
    graupel_toy_scan(qc, qr, out=(qc_new, qr_new), offset_provider={})

    # Compare results
    assert np.allclose(qc_new.asnumpy(), qc_numpy)
    assert np.allclose(qr_new.asnumpy(), qr_numpy)
```

```{code-cell} ipython3
test_scan_operator()
print("Test successful")
```

```{code-cell} ipython3

```
