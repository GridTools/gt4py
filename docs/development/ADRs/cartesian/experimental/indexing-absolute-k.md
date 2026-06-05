# ⚠️ Absolute K indexing

In the context of porting physics parametrizations, some field accesses were at absolute indices in the K-axis. We thus decided to expand the DLS be able to port these stencils. We accept the increase in DSL surface and test it as an [experimental feature](../experimental-features.md).

## Context

Porting physics parametrizations, we found two following two access patterns

```fortran
do k = 1, K
  do j = 1, J
   do i = 1, I
      ...
      Field(i, j, K)  ! access top layer
      Field(i, j, 1)  ! access bottom layer
      ...
```

```fortran
do k = 1, K
  do j = 1, J
   do i = 1, I
      ...
      computed_k_index = FindK(...)  ! call a function or search all (previous) levels
      Field(i, j, computed_k_index)  ! then, access a pre-computed absolute index (e.g. boundary layer)
      ...
```

So far, porting these codes to GT4Py stencils has always been possible by using temporary helper fields and by breaking loops apart. Even cases of looping over previous levels have been done with extra `while` loops. Doing so adds a ton of bloat and makes the resulting code difficult to read.

Absolute indexing in the `K` dimension would allow our users to express the above concepts in a succinct way as GT4Py stencils. We thus think it is worth to allow this feature in a limited context.

## Decision

We decided to allow absolute indexing on the K axis in certain circumstances with the following syntax:

```py
@stencil(backend="dace:cpu", externals={K4: "4"})
def absolute_k_indexing(in_field: Field[np.float64], index_field: Field[IJ, np.int64], out_field: Field[np.float64], index: int) -> None:
    with computation(PARALLEL), interval(...):
        from __externals__ import K4

        # access third level in K (useful for fixed boundary layers)
        out_field = in_field.at(K=2)

        # works with externals
        out_field = in_field.at(K=K4)

        # also works if level is not known at compile time
        out_field = in_field.at(K=index)

        # and even if we read from another field
        out_field = in_field.at(K=index_field)
```

The following two restrictions apply to absolute indexing in the `K` dimension:

1. read-only access
2. reads are "centered" in `I` and `J`, i.e., no (relative) offset in the `I` and `J` dimension

Both rules are enforced by the syntax `field.at(K=...)`, which

1. can't be written to and
2. errors if users try to write something like `field.at(K=42, I=-1)`.

One can use absolute indexing (in `K`) on a field that has data dimension. Assuming one data dimension, the resulting call would like like `field.at(K=2, ddims=[4])` to access the fifth element of the data dimension on the third level in `K`. Future work is needed to unify this access pattern with (global) tables, where the syntax is `global_table.A[4]`.

## Consequences

Users of GT4Py can now succinctly write the patterns shown above in the context section.

One major drawback of absolute indexing is that we loose the opportunity to warn users at compile time if they read out fo bounds. Since we allow absolute K indices that aren't know at compile time, the extent analysis can't warn us anymore. We plan to mitigate that by supporting an option to check every absolute read (at runtime) in debug mode, see references below.

We ship absolute K indexing as an [experimental features](../experimental-features.md). The feature is only available in `debug` and `dace:*` backends. Other backends raise an error at compile time when transpiling from `oir` into the respective backend IR.

## References

- [Issue](https://github.com/GridTools/gt4py/issues/1684) to add an out of bounds check at runtime. The issue is old because already now, with `VariableKOffset`, we can't catch every out of bounds access at compile time.
- [Issue](https://github.com/GridTools/gt4py/issues/2282) about unifying access patterns with global tables.
