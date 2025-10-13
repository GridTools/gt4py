# ⚠️ Expose iteration index in `K`

In the context of porting physics parametrizations, we decided to expose the current iteration index in `K` to increase code readability. We are actively looking for more targeted DSL features as a replacement.

⚠️ This feature is not yet mature. Expect breaking changes if you use it in the current form.

## Context

Porting physics code, we came across patterns that are hard to map to a stencil with current DSL features. Specifically, we are seeing things that depend on the current iteration index in `K`. For example variable interval bounds, such as

```py
"""
krel [IntField]: Release layer where buoyancy sorting first occurs
kinv [IntField]: Inversion layer with PBL top interface as lower interface
"""
with computation(), interval():
    # ...
    if iteration_index_K >= kinv - 1 and iteration_index_K <= max(krel - 1, kinv - 1):
        uplus_3D = uplus + PGFc * ssu0 * (pifc0[0, 0, 1] - pifc0)
        vplus_3D = vplus + PGFc * ssv0 * (pifc0[0, 0, 1] - pifc0)
    # ...
```

or in combination with [absolute indexing in K](./indexing-absolute-k.md)

```py
with computation(PARALLEL), interval(...):
    # Flip mid-level variables
    k_inv = k_end - iteration_index_K
    pmid0_in = pmid0_inv.at(K=k_inv)
```

## Decision

We chose to expose the current iteration index in `K` with the following syntax in order to facilitate porting the use-cases outlined above. The syntax, for now, is that the current iteration index is exposes as just `K`, e.g. the examples above would read:

```py
"""
krel [IntField]: Release layer where buoyancy sorting first occurs
kinv [IntField]: Inversion layer with PBL top interface as lower interface
"""
with computation(), interval():
    # ...
    if K >= kinv - 1 and K <= max(krel - 1, kinv - 1):
        uplus_3D = uplus + PGFc * ssu0 * (pifc0[0, 0, 1] - pifc0)
        vplus_3D = vplus + PGFc * ssv0 * (pifc0[0, 0, 1] - pifc0)
    # ...
```

and

```py
with computation(PARALLEL), interval(...):
    # Flip mid-level variables
    k_inv = k_end - K
    pmid0_in = pmid0_inv.at(K=k_inv)
```

While `K` is short, we are aware that the letter is loaded. For now, we chose something short and easy. The plan is to see (with real-world use-cases) if there are

1. ways to avoid exposing the iteration index, e.g. a set of more restrictive DSL features
2. conceptual problems with exposing the iteration index, e.g. ways to break the DSL.

We are also aware of [issue #208](https://github.com/GridTools/gt4py/issues/208) and are using it to find a more suitable/permanent name for the case that we don't find any conceptual problems with exposing the current iteration index.

## Consequences

On the plus side, porting physics code can continue with this experimental feature. It will allows us to gather rapid feedback on real-world use-cases. On the other hand, we already suspect breaking changes (e.g. name change and/or new features) ahead. And we know that we open "pandora's box" by exposing the current iteration index.

From [issue #208](https://github.com/GridTools/gt4py/issues/208) it is clear that the community expects a solution. This experimental feature has to be seen as a first version in an iterative approach to find an expressive DSL.

## Alternatives considered

As mentioned above, this feature is is rather immature, we are actively exploring alternatives in two directions

1. finding a more descriptive / less loaded name than `K`, see discussions in [this issue](https://github.com/GridTools/gt4py/issues/208).
2. looking for alternative DSL features to remove the need for direct iterator access (e.g. `interval`s with non-static bounds).

## References

Issues [#208](https://github.com/GridTools/gt4py/issues/208) and maybe also [#72](https://github.com/GridTools/gt4py/issues/72).
