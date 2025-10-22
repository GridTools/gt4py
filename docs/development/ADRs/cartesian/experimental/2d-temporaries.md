# ⚠️ Two-dimensional Temporaries

In the context of porting physics parametrizations, we have encountered multiple examples of computations requiring a temporary 2D storage within a stencil.

We thus decided to expand the DLS be able to port these stencils. We accept the increase in DSL surface and test it as an [experimental feature](../experimental-features.md).

## Context

Porting physics parametrizations, we found temporary 2D fields used, mostly when computing particular levels (e.g. surface, phase change, etc.). The previous workaround was to define the temporaries outside and pass them as an argument. The goal is to undo this workaround that complexifies calling code.

Another workaround, driven by how GridTools C++ behaves, is to force all temporaries to be 3D. The way to do it would be to create you temporary and then propagate it's value throughout. E.g., from FV3 dynamics code:

```python
with computation(FORWARD):
    with interval(0, 1):
        bet = bb
    with interval(1, -1):
        bet = bet[0, 0, -1]
```

In this example `bet` can be _read_ at every grid point.

Use case in GEOS's shallow convection parametrization: `total_pwo_solid_phase` will be used later in the stencil but not outside of it.

```python
    with computation(FORWARD), interval(...):
        if MELT_GLAC == 1 and cumulus == 1:
            if K <= ktf-1:
                if ierr == 0:
                    dp = 100.*(po_cup-po_cup[0,0,1])\
                    pwo_eff = 0.5*(pwo+pwo[0,0,1])
                    pwo_solid_phase = (1.-p_liq_ice)*pwo_eff
                    total_pwo_solid_phase = total_pwo_solid_phase+pwo_solid_phase*dp/constants.MAPL_GRAV
```

Use case in GEOS's microphysics. Vertical interpolation to a given level requires to compound pressure around it. Here `pb`, `pt` and `boolean_2d_mask` are all temporary 2D fields.

```python
def vertical_interpolation(
    field: FloatField,
    interpolated_field: FloatFieldIJ,
    p_interface_mb: FloatField,
    target_pressure: Float,
    pb: FloatFieldIJ,
    pt: FloatFieldIJ,
    boolean_2d_mask: BoolFieldIJ,
):
    """
    Interpolate to a specific vertical level.

    Only works for non-interface fields. Must be constructed using Z_DIM.

    Arguments:
        field (in): three dimensional field to be interpolated to a specific pressure
        interpolated_field (out): output two dimension field of interpolated values
        p_interface_mb (in): interface pressure in mb
        target_pressure (in): target pressure for interpolation in Pascals
        pb (in): placeholder 2d quantity, can be removed onces 2d temporaries are available
        pt (in): placeholder 2d quantity, can be removed onces 2d temporaries are available
        boolean_2d_mask (in): boolean mask to track when each cell is modified
    """
    # mask tracks which points have been touched. check later on ensures that every point has been touched
    with computation(FORWARD), interval(0, 1):
        boolean_2d_mask = False

    with computation(FORWARD), interval(-1, None):
        pb = 0.5 * (log(p_interface_mb * 100) + log(p_interface_mb[0, 0, 1] * 100))

    with computation(BACKWARD), interval(1, None):
        pt = 0.5 * (log(p_interface_mb[0, 0, -1] * 100) + log(p_interface_mb * 100))
        if log(target_pressure) > pt and log(target_pressure) <= pb and not boolean_2d_mask:
            al = (pb - log(target_pressure)) / (pb - pt)
            interpolated_field = field[0, 0, -1] * al + field * (1.0 - al)
            boolean_2d_mask = True
        pb = pt

    with computation(FORWARD), interval(-1, None):
        pt = 0.5 * (log(p_interface_mb * 100) + log(p_interface_mb[0, 0, -1] * 100))
        pb = 0.5 * (log(p_interface_mb * 100) + log(p_interface_mb[0, 0, 1] * 100))
        if (
            log(target_pressure) > pb
            and log(target_pressure) <= log(p_interface_mb[0, 0, 1] * 100)
            and not boolean_2d_mask
        ):
            interpolated_field = field
            boolean_2d_mask = True

        # ensure every point was actually touched
        if boolean_2d_mask == False:  # noqa
            interpolated_field = field

    # reset masks and temporaries for later use
    with computation(FORWARD), interval(0, 1):
        boolean_2d_mask = False
        pb = 0
        pt = 0
```

2D temporaries would clean those codes and allow for further optimizations (e.g. scalarization) in all performance backends.

## Decision

All the guardrails for 2D temporaries are already in place:

- allocation can only be done under `interval == 1` and `computation` in `FORWARD` or `BACKWARD`,
- use of 2D fields are fully defined, using the above rules again.

The main change is on the frontend of stencils and proper forwarding of dimensions through to code generation, except for `gt:X` backend (see [Consequences](#consequences)).

Type hints for temporaries have been introduced before for mixed precision e.g.:

```python
with computation(PARALLEL), interval(...):
    tmp_3D_as_f32: float32 = 0
```

We propose to extend this type hint to allow specification of the dimensions re-using the `FieldDescriptor`, e.g.

```python
with computation(FORWARD), interval(...):
    tmp_2D_as_f32: Field[IJ, np.float64] = 0
```

We also guard against any definition of other type of temporaries (e.g. 1D temporaries) because they are ill-defined within `gtscript` which pre-supposes that the horizontal dimensions are always computed upon.

The type hint can be a little verbose, so we offer to expand upon the `dtypes` dictionary present in stencil configuration to give a `str: type` pair that can be swapped at parsing time, as long as the type is a derivative of `FieldDescriptor` so the relevant information can be retrieved.

```python
@stencil(backend=..., dtype={"My2DType": Field[IJ, np.float64]})
def the_stencil(...):
    with computation(FORWARD), interval(...):
        tmp_2D: My2DType = 0
        ...
```

## Consequences

Users of GT4Py can now define 2D temporaries inside stencils.

There's one remaining hiccup: GridTools C++ does not natively offer 2D temporaries. GridTools pre-supposes that all computations of temporaries are done on the 3D grid at least - and won't allocate below that dimensionality.
This is a fair limitation since GridTools doesn't provide the guardrails against race condition that GT4Py does. We can circumvent this by creating a pool of buffers passed as arguments for `gt:X` backends - see [issue](https://github.com/GridTools/gt4py/issues/2322).

## References

- [PR](https://github.com/GridTools/gt4py/pull/2314) PR where the proposal and the discussion occurred.
- [Issue](https://github.com/GridTools/gt4py/issues/2322) to extend support to the `gt:X` backends
