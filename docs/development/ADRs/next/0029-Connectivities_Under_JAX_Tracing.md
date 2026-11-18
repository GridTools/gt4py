---
tags: [embedded, jax, connectivities]
---

# Connectivities Under JAX Tracing

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-11-18
- **Updated**: 2026-11-18

In the context of running embedded `premap` under `jax.jit`, facing a neighbor table that must be a runtime argument for the gather but concrete for domain inference, we decided to pass the concrete table along with its tracer through the pytree aux data of the connectivity, to get the same narrowing as eager execution without compiling tables into the program. We accept a table reference held by jax's jit cache and a silent clamp when the table is only known at runtime (`as_offset`).

## Context

`premap` uses the table twice, and needs it at different times:

- The **output domain** comes from `_inverse_image`, which depends on the table's values. It is computed at trace time, so it needs concrete values: Python `slice`s cannot be staged.
- The **gather** runs in the compiled program, so it should take the table as a run-time argument. A closed-over table is inlined as a constant, which does not scale to large meshes.

## Decision

There is one connectivity class, `NdArrayConnectivityField`, for all array libraries and for all cases below. The operations that read the values of a table, `inverse_image_box` (domain inference) and `restrict`, are methods of `ConnectivityOps`, which builds on the library's `ArrayOps`; a library that needs different table operations registers a subclass for its namespace. The gather is `ArrayOps.gather`, an operation on the field's buffer. The defaults are plain array code. `JaxOps` overrides `gather`, `JaxConnectivityOps` the table operations, and everything that knows about tracing is in `jax_ops.py`.

Each connectivity has its own `ConnectivityOps` instance, which can carry what the library knows about the table beyond the table itself. It is not a dataclass field, so no copy inherits it: a creator that knows it passes it at construction, as `restrict` does with the one returned for the restricted table and the pytree unflatten with the static values, and otherwise `connectivity_ops.of` creates it from the table on first use. For a table that is a tracer, `JaxConnectivityOps` holds the concrete array with the same values, the static values. A connectivity that got another table, for example through `dataclasses.replace` or `copy`, has none.

The connectivity is a pytree node with the table as its child. The aux data holds domain, codomain, skip value and, if the table is concrete or a tracer with static values, the static values. They are compared by identity, so two connectivities have the same aux data only if they have the same static values. Unflattening gives a child that is not a concrete array the static values from the aux data, in its `JaxConnectivityOps`.

Nothing that belongs to a trace may be kept beyond it, so a connectivity caches domains and slices, never tables or restricted connectivities. `JaxConnectivityOps` returns the same array when the same concrete table is restricted again with the same slice. A restriction then has the same aux data in every call, so it does not cause a retrace.

How a `premap` behaves under `jit` depends on how its connectivity reaches the compiled function:

- **Passed as a jit argument** (`jax.jit(lambda f, c: f.premap(c))(field, conn)`), the recommended way. The table is a tracer with static values. Domain inference and `restrict` read the static values, and the gather takes the table as a runtime argument. A neighbor reduction takes its table from the offset provider, so the provider has to be built from the traced connectivity inside the jitted function; a provider set outside inlines the table as a constant for the reduction.
- **Closed over** (`jax.jit(lambda f: f.premap(conn))(field)`). The table stays concrete, also after `restrict`, because the operations on it run under `jax.ensure_compile_time_eval()`. Domain inference reads it directly, but the table is inlined into the compiled program as a constant. `as_offset` of a closed-over offset field behaves the same way, but slicing or computing with that field inside the trace makes it traced (next case).
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
- The restrictions of a concrete jax table are kept for as long as the table is alive.
- Domain inference on static values is cached on the connectivity object, which is new in every trace, so it is repeated per trace but not per call.

Open questions, all caused by the static values in the aux data:

- The aux data is a strong reference to the table, so the jit cache keeps every table it has seen alive until `jax.clear_caches()`. Memory stays bounded when a run uses a fixed set of tables.
- Rebuilding a connectivity around another traced table through the pytree API inside a trace (`tree_map` with a function that changes the leaf, `tree_unflatten`) keeps the static values of the old table, so domain inference silently uses them. This is unsupported, and nothing detects it: with jax's public API such a rebuild cannot be told apart from the entry of a `jit`.
- `shard_map` rejects the aux data when the table is sharded, so distributed execution is not supported yet.

## Alternatives considered

### An opaque state passed back to `ArrayOps`

The table operations were methods of `ArrayOps`, which took a `state` stored on the connectivity. Every copy inherited the state, also with another table, possibly of another library, so each operation had to check that the state described its table, and the static values of restricted tracers were kept in a weak map on the state of the connectivity they were restricted from.

### Static values found by the table

A map from table to static values, inside `jax_ops.py`, needs no state on the connectivity. But unflattening attaches static values to placeholders like `None`, integers and `object()`, which cannot be weakly referenced, and `None` is the same object everywhere.

### `is_traced` and `concrete_eval` on `ArrayOps`

Two small hooks, with the case distinction written out in `NdArrayConnectivityField`. Rejected because tracing then shows in the code shared by all array libraries.

### Static values that follow the connectivity instead of the table

Earlier versions kept the concrete connectivity in a field that every copy of the connectivity inherits, and derived the aux data of a connectivity with a concrete leaf from that leaf. `dataclasses.replace` then attached the values to another table, and jax's comparison of branches and carries, which runs on trees with a shared placeholder leaf, found nothing to object to: `lax.cond` between two tables silently used the values of the first one. `ConnectivityOps` is not inherited, and a concrete table is its own values.

### Bounds in the aux data instead of the table

Storing only `(min, max)` retraces per bounds rather than per buffer and holds no table reference, but cannot answer narrowing ranges.

### For tables computed in the trace: reject them, or mask with NaN

Accepting only concrete tables rules out dynamic offsets under `jit`. NaN makes an out-of-range read visible, but does not work for integer fields and propagates into gradients.

## References

- [ADR 0019 - Connectivities](0019-Connectivities.md)
