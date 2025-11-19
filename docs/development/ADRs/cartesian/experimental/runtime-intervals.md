# ⚠️ Allow for runtime specification of interval bounds

In the context of porting physics parametrization, a repeating pattern in a lot of solver-like codes are column based loops that only affect parts of the column until a certain condition is met.

⚠️ This feature is not yet mature. Expect breaking changes if you use it in the current form.

## Context

In the physics we see patterns like this:

```fortran
integer    kpen        ! Highest layer with positive updraft velocity
do k = 0, k0
  if( wtw .le. 0. ) then
    kpen = k
    go to 45
  end if
end do
do k = kpen - 1, kbup, -1                    
    rhoifc0j = pifc0(k) / ( r * 0.5 * ( thv0bot(k+1) + thv0top(k) ) * exnifc0(k) )
    ...
end do 
```

This pattern could theoretically be implemented with the current tooling in gt4py by using vertical masking and access to the K-index:

```py
with computation(FORWARD), interval(...):
    temporary_mask : Field[IJ, int]
    if wtw < 0:
        temporary_mask = min(temporary_mask,K)
with computation(FORWARD), interval(0, -1):
    if K > temporary_mask: # ignoring the upper bound here for simplicity
        rhoifc0j = pifc0 / ( r * 0.5 * ( thv0bot[K+1] + thv0top ) * exnifc0)
```

The problem with this solution is that:

1. Bloats the user code, especially when these masks have to be carried around for long times.
2. Generates performance issues on the CPU with potentially much more computation and conditionals that might kill caching.

**Similar patterns**

The most prominent related feature that is already in the gt4py syntax are vertical intervals.
These have two different origins:

1. The top or bottom most layer(s) of the atmosphere need a different form of numerical interpolation due to neighbors simply not existing.
2. The vertical grid layout is usually hybrid with `σ`-coordinates near the surface and `p`-coordinates near the top of the atmosphere.
   This allows for a terrain-following grid near the surface and therefore can represent near-surface processes really well while still allowing for better representations of large-scale dynamics in the stratosphere.

With the different expressions in the vertical, is is common to see slightly different numerical patterns supported in either of the intervals. Since the switch of coordinates is a grid-property and known at runtime, this patterns is not really applicable to what we see in the physics codes.

## Decision

We chose to allow for more flexibility in what is allowed in the interval specification. Now it is possible to:

1. Use externals (already supported)
2. Use scalar arguments
3. Use two-dimensional Fields
4. Use two-dimensional temporaries
   to specify the bounds of the interval:

```py
def test_stencil(
    out_field: FloatField,
    input_data: FloatField,
    index_data: Field[gtscript.IJ, np.int64],
    scalar_arg: int,
):
    from __externals__ import external_scalar  # type: ignore

    with computation(FORWARD), interval(0, 1):
        temporary: Field[IJ, np.float64] = 7

    with computation(PARALLEL):
        with interval(0, external_scalar):
             out_field = input_data[0, 0, 0]

        with interval(0, scalar_arg):
            out_field[0, 0, 0] = input_data[0, 0, 0]

        with interval(0, index_data):
            out_field[0, 0, 0] = input_data[0, 0, 0]

        with interval(0, temporary):
            out_field[0, 0, 0] = input_data[0, 0, 0]
```

To bound the interval.

## Consequences

The biggest issue that comes with allowing temporary fields as well as 2d fields is the fact that this enforces an `(i,j)(k)`-loop order. We think it is ok for such a requirement to exist and want to allow the backend to make the decision that is right on a case-by-case basis.

Another consequence is we break the guardrail of `gt4py` that guarantees you won't read or write out of bounds. 
Another feature (absolute indexing in K) is breaking this guardrail, and [we propose a runtime checker to be implemented](https://github.com/GridTools/gt4py/issues/1684).

## References

Physics port of the UW-convection scheme [here](https://github.com/GEOS-ESM/GEOSgcm_GridComp/tree/dsl/develop/GEOSagcm_GridComp/GEOSphysics_GridComp/GEOSmoist_GridComp/pyMoist/pyMoist/UW)
