# Absolute K indexing

In the context of porting physics parametrizations, some field accesses were at absolute indices in the K-axis. We thus decided to expand the DLS be able to port these stencils. We accept the increase in DSL surface.

## Context

Porting physics parametrizations, we found two following two access patterns

```fortran
do L = 1, LM
  do J = 1, JM
   do I = 1, IM
      ...
      Field(i, j, LM)  ! access top layer
      Field(i, j, 1)   ! access bottom layer
      ...
```

```fortran
do L = 1, LM
  do J = 1, JM
   do I = 1, IM
      ...
      computed_k_index = FindK(...)  ! call a function or search all (previous) levels
      Field(i, j, computed_k_index)  ! then, access a pre-computed absolute index (e.g. boundary layer)
      ...
```

which can't be ported to GT4Py stencils.

## Decision

We decided to allow absolute indexing in the K axis with the following syntax:

```py
@stencil(backend="dace:cpu", externals={K4: "4"})
def absolute_k_indexing(in_field: Field[np.float64], k_field: Field[IJ, np.int64], out_field: Field[np.float64], idx: int) -> None:
    with computation(PARALLEL), interval(...):
        from __externals__ import K4

        # access third level in K (useful for fixed boundary layers)
        out_field = in_field.at(K=2)

        # works with externals
        out_field = in_field.at(K=K4)

        # also works if level is not known at compile time
        out_field = in_field.at(K=idx)

        # and even if we read from another field
        out_field = in_field.at(K=k_field)
```

and the following two restrictions:

1. read only
2. reads are "centered" in I and J, i.e., no (relative) offset in the I and J dimensions

Both rules are enforced by syntax `field.at(K=...)`, which

1. can't be written to and
2. errors if users try to write something like `field.at(K=42, I=-1)`.

## Consequences

Users of GT4Py can now succinctly write the patterns shown above in the context section.

One major drawback of absolute indexing is that we loose the opportunity to warn users at compile time if they read out fo bounds. Since we allow absolute K indices that aren't know at compile time, we extent analysis can't warn us anymore. We plan to mitigate that by supporting an option to check every absolute read (at runtime) in debug mode.

As a starting point, absolute indexing in K will only be available in the `dace:*` backends.
