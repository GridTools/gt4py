---
tags: [embedded, jax, connectivities]
---

# Connectivities Under JAX Tracing

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-11-18
- **Updated**: 2026-11-19

In the context of running embedded `premap` under `jax.jit`, facing a neighbor table that must be a runtime argument for the gather but concrete for domain inference, we decided to pass the concrete table along with its tracer through the pytree aux data of the connectivity, to get the same narrowing as eager execution without compiling tables into the program. We accept a table reference held by jax's jit cache and a silent clamp when the table is only known at runtime (`as_offset`).

## Context

`premap` uses the table twice, and needs it at different times:

- The **output domain** comes from `_inverse_image`, which depends on the table's values. It is computed at trace time, so it needs concrete values: Python `slice`s cannot be staged.
- The **gather** runs in the compiled program, so it should take the table as a run-time argument. A closed-over table is inlined as a constant, which does not scale to large meshes.

## Decision

There is one connectivity class, `NdArrayConnectivityField`, for all array libraries and for all cases below. As for every embedded field, the connectivity does the domain operations, and `ArrayOps` only works on plain arrays. Domain inference reads the table's *static values*: an array with the table's values that can be computed on now, possibly of another library. `ArrayOps.static_values` gives them for a buffer: the buffer itself by default, and `None` if the values exist only at run time. The box of entries that map into an image range is computed on them by one plain function for all libraries. The gather is `ArrayOps.gather`. `JaxOps` overrides `static_values` and `gather`, and everything that knows about tracing is in `jax_ops.py`.

For a concrete jax table, `JaxOps.static_values` is a read-only NumPy view of it, so domain inference is never staged, also inside a trace. For a tracer it is `None`, and the connectivity then assumes that every target lies in range. A tracer that stands for a concrete table, the case of a connectivity passed to `jit`, gets the static values from its creator: the constructor takes them as a keyword-only init-only argument, which the pytree unflatten passes. They are not a dataclass field, so no copy inherits them: a connectivity that got another table, for example through `dataclasses.replace` or `copy`, takes its static values from the new table. `restrict` restricts static values that are not the buffer itself along with the buffer, so a restricted connectivity keeps them.

The connectivity is a pytree node with the table as its child. The aux data holds domain, codomain, skip value and, if the table is concrete or a tracer with static values, the static values. They are compared by their memory, because NumPy returns a new view of a jax array on every call: two connectivities have the same aux data only if their static values are the same read-only memory, which holds for every wrapper of one buffer and for its restrictions with the same slice. Unflattening gives a child that is not a concrete array the static values from the aux data.

Nothing that belongs to a trace may be kept beyond it, so a connectivity caches domains and slices, never tables or restricted connectivities.

How a `premap` behaves under `jit` depends on how its connectivity reaches the compiled function:

- **Passed as a jit argument** (`jax.jit(lambda f, c: f.premap(c))(field, conn)`), the recommended way. The table is a tracer with static values. Domain inference and `restrict` read the static values, and the gather takes the table as a runtime argument. A neighbor reduction takes its table from the offset provider, so the provider has to be built from the traced connectivity inside the jitted function; a provider set outside inlines the table as a constant for the reduction.
- **Closed over** (`jax.jit(lambda f: f.premap(conn))(field)`). The table is concrete, so domain inference reads its static values, also after a `restrict` inside the trace, which restricts them along with the then staged table. The table is inlined into the compiled program as a constant. `as_offset` of a closed-over offset field behaves the same way, but slicing or computing with that field inside the trace makes it traced (next case).
- **Computed from traced values** (`as_offset(Koff, offset_field)` with a traced offset field, or any other table that is a tracer without static values). No concrete values exist at trace time. Domain inference assumes every target lies inside the field, so the output domain is not narrowed. `JaxOps.gather` clamps every index to the field at both ends, so an out-of-range target reads the field's boundary value.

A tracer of any jax transformation counts as traced, not only one of `jit`. In particular a table that depends on a `vmap`ped value has no values even when `vmap` is called eagerly, because it differs per batch element.

### Control flow

The aux data claims that the child has the static values. Where jax rebuilds a pytree around new tracers, that claim stays true:

- At the entry of `jit`, of a nested `jit`, and for the operands of `lax.cond`, the new tracer stands for the flattened table.
- `lax.cond` and `lax.switch` require all branches to return the same aux data, and `lax.scan`, `lax.while_loop` and `lax.fori_loop` require the carry to keep its aux data. Selecting between two tables, changing the table of a carried connectivity, or mixing a table that has static values with one computed in the trace raises jax's `TypeError` about differing pytree structure. The same table on all branches, or an unchanged carry, keeps its static values.

For that check jax rebuilds the trees around placeholders that are not tracers, which is why unflattening attaches the static values to every child that is not a concrete array.

A traced connectivity can also be closed over by the branches of `lax.cond`, the body of `lax.scan` or an inner `jit`, and used again afterwards.

## Consequences

- Domains under `jit` match eager execution when the table is concrete or has static values.
- A table computed from traced values behaves differently from eager execution when a target is out of range. Eagerly, a uniform offset narrows the domain and a non-contiguous one raises. Under `jit` the domain stays as it is and those targets read the field's boundary value, without an error.
- Every gather of a jax field uses `mode="clip", wrap_negative_indices=False`. That clamps both ends with no operation besides the gather, where jax's default indexing clamps past the end but emits a select to wrap negative indices. With a table that has values the clamp never acts.
- Skip entries read the field's first element, and reductions mask the result afterwards. A gradient through an operator that is singular at that value is NaN (`0 * inf`), for example `sqrt` of a first element that is zero. The gather cannot prevent this, because no replacement value is safe for every operator.
- Connectivities reject item assignment, because their domain lookups are cached. Writing into the buffer through `.ndarray` is not detected and leaves those lookups stale, so it is not supported.
- `restrict` builds a new connectivity on every call. It shares the cached domain lookups of the connectivity it was restricted from.
- A traced connectivity that is copied, or rebuilt with `dataclasses.replace`, loses its static values: its domain is not narrowed and the gather clamps.
- The static values of a jax table are NumPy, so on a GPU domain inference reads a host copy of the table.
- Domain inference on static values is cached on the connectivity object, which is new in every trace, so it is repeated per trace but not per call.

Open questions, all caused by the static values in the aux data:

- The aux data is a strong reference to the table, so the jit cache keeps every table it has seen alive until `jax.clear_caches()`. Memory stays bounded when a run uses a fixed set of tables.
- Rebuilding a connectivity around another traced table through the pytree API inside a trace (`tree_map` with a function that changes the leaf, `tree_unflatten`) keeps the static values of the old table, so domain inference silently uses them. This is unsupported, and nothing detects it: with jax's public API such a rebuild cannot be told apart from the entry of a `jit`.
- `shard_map` rejects the aux data when the table is sharded, so distributed execution is not supported yet.

## Alternatives considered

### Table operations on a per-connectivity object

Domain inference and `restrict` were methods of a `ConnectivityOps` object, one per connectivity and registered per library, which held the static values and returned its successor from `restrict`. Its methods took plain arrays, but the object carried state about one buffer and the decision what a table without values means, which is the connectivity's. It needed its own registry next to `ArrayOps`. An earlier variant passed an opaque `state` stored on the connectivity back to `ArrayOps`; every copy inherited it, so each operation had to check that the state described its table.

### Static values in a side table keyed by the connectivity

A weak map from connectivity to static values in `jax_ops.py` needs no constructor argument, but `restrict` would need a hook to hand the restricted values back, and `ArrayOps` would have to take connectivities.

### Static values found by the table

A map from table to static values, inside `jax_ops.py`, needs no state on the connectivity. But unflattening attaches static values to placeholders like `None`, integers and `object()`, which cannot be weakly referenced, and `None` is the same object everywhere.

### `is_traced` and `concrete_eval` on `ArrayOps`

Two small hooks, with the case distinction written out in `NdArrayConnectivityField`. The connectivity would then have to know when to evaluate concretely. `static_values` instead returns an array that can be computed on now, so the connectivity needs no evaluation mode.

### Static values that follow the connectivity instead of the table

Earlier versions kept the concrete connectivity in a field that every copy of the connectivity inherits, and derived the aux data of a connectivity with a concrete leaf from that leaf. `dataclasses.replace` then attached the values to another table, and jax's comparison of branches and carries, which runs on trees with a shared placeholder leaf, found nothing to object to: `lax.cond` between two tables silently used the values of the first one. The static values are not inherited, and a concrete table derives its own.

### Bounds in the aux data instead of the table

Storing only `(min, max)` retraces per bounds rather than per buffer and holds no table reference, but cannot answer narrowing ranges.

### For tables computed in the trace: reject them, or mask with NaN

Accepting only concrete tables rules out dynamic offsets under `jit`. NaN makes an out-of-range read visible, but does not work for integer fields and propagates into gradients.

## References

- [ADR 0019 - Connectivities](0019-Connectivities.md)
