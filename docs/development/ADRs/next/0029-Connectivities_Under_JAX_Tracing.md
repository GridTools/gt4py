---
tags: [embedded, jax, connectivities]
---

# Connectivities Under JAX Tracing

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-11-06
- **Updated**: 2026-11-13

In the context of running embedded `premap` under `jax.jit`, facing a neighbor table that must be a runtime argument for the gather but concrete for domain inference, we decided to pass the concrete table along with its tracer through the pytree aux data of the connectivity, to get the same narrowing as eager execution without compiling tables into the program. We accept a table reference held by jax's jit cache and a silent clamp when the table is only known at runtime.

## Context

`premap` uses the table in two ways that pull in opposite directions:

- The **output domain** comes from `_inverse_image`, which depends on the table's values. Under a trace it needs concrete values, and Python `slice`s cannot be staged.
- The **gather** should take the table as an argument of the compiled program. A closed-over table is inlined as a constant, which does not scale to large meshes.

Narrowing is the common case: the field usually covers only part of the connectivity's image, so a solution that handles only covering ranges is not sufficient.

## Decision

There is one connectivity class, `NdArrayConnectivityField`, for all array libraries and for all cases below. The operations on a table are methods of `ArrayOps`: `hyperslice` (domain inference), `restrict_table` and `gather`. Their defaults are plain array code. `JaxOps` overrides them, and everything that knows about tracing is in `jax_ops.py`.

A connectivity stores one opaque value for its `ArrayOps`, `_static`, and hands it to `hyperslice` and `restrict_table`. Only the library creates it, and `restrict` passes it on unchanged. For a table that is a tracer, `JaxOps` keeps there the concrete array with the same values, the static values, for that tracer and for the tracers restricted from it. Tables are looked up by identity, so a connectivity that got another table, for example through `dataclasses.replace`, has none.

`JaxOps.gather` clamps indices to the field at both ends, inside the gather. It does not need to know where an index came from.

The connectivity is a pytree node. The table is the child. The aux data holds domain, codomain, skip value and the static values, compared by the identity of the array: a new wrapper around the same buffer does not retrace, a new buffer does. Flattening emits them if the table is concrete or a tracer with static values. Unflattening attaches them to a child that is not a concrete array; a concrete child has its own values. Because identity only stands for the values of an immutable array, flattening rejects a connectivity with a NumPy or CuPy table: such a connectivity can be closed over, but not passed as a `jit` argument.

A connectivity caches domains and slices, never tables or restricted connectivities, so nothing cached can belong to a trace that has ended. `JaxOps` returns the same array when the same concrete table is restricted again with the same slice, which keeps a restricted connectivity from retracing.

How a `premap` behaves under `jit` depends on how its connectivity reaches the compiled function:

- **Passed as a jit argument** (`jax.jit(lambda f, c: f.premap(c))(field, conn)`), the recommended way. The table is a tracer with static values. Domain inference and `restrict` read the static values, and the gather takes the table as a runtime argument. A neighbor reduction takes its table from the offset provider, so the provider has to be built from the traced connectivity inside the jitted function; a provider set outside inlines the table as a constant for the reduction.
- **Closed over** (`jax.jit(lambda f: f.premap(conn))(field)`). The table stays concrete, also after `restrict`, because the operations on it run under `jax.ensure_compile_time_eval()`. Domain inference reads it directly, but the table is inlined into the compiled program as a constant. `as_offset` of a closed-over offset field behaves the same way, but slicing or computing with that field inside the trace makes it traced (next case).
- **Computed from traced values** (`as_offset(Koff, offset_field)` with a traced offset field, or any other table that is a tracer without static values). No concrete values exist at trace time. Domain inference assumes every target lies inside the field, so the output domain is not narrowed, and the clamp of the gather decides what an out-of-range target reads.

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
- The restrictions of a concrete jax table are kept for as long as the table is alive.
- Domain inference on static values is cached on the connectivity object, which is new in every trace, so it is repeated per trace but not per call.

Open questions, all caused by the static values in the aux data:

- The aux data is a strong reference to the table, so the jit cache keeps every table it has seen alive until `jax.clear_caches()`. Memory stays bounded when a run uses a fixed set of tables.
- Rebuilding a connectivity around another traced table through the pytree API inside a trace (`tree_map` with a function that changes the leaf, `tree_unflatten`) keeps the static values of the old table, so domain inference silently uses them. This is unsupported, and nothing detects it: with jax's public API such a rebuild cannot be told apart from the entry of a `jit`.
- `shard_map` rejects the aux data when the table is sharded, so distributed execution is not supported yet.

## Alternatives considered

### A connectivity subclass per case

One class each for a concrete table, a traced table with static values and a traced table without, chosen when the connectivity is constructed. This keeps `ArrayOps` free of table operations, but brings back per-library field classes and a second registry next to `ArrayOps`, which a connectivity built with `from_array` bypasses.

### `is_traced` and `concrete_eval` on `ArrayOps`

Two small hooks, with the case distinction written out in `NdArrayConnectivityField`. Rejected because tracing then shows in the code shared by all array libraries.

### Static values that follow the connectivity instead of the table

Earlier versions kept the concrete connectivity in a field that every copy of the connectivity inherits, and derived the aux data of a connectivity with a concrete leaf from that leaf. `dataclasses.replace` then attached the values to another table, and jax's comparison of branches and carries, which runs on trees with a shared placeholder leaf, found nothing to object to: `lax.cond` between two tables silently used the values of the first one.

### Bounds in the aux data instead of the table

Storing only `(min, max)` retraces per bounds rather than per buffer and holds no table reference, but cannot answer narrowing ranges.

### For tables computed in the trace: reject them, or mask with NaN

Accepting only concrete tables rules out dynamic offsets under `jit`. NaN makes an out-of-range read visible, but does not work for integer fields and propagates into gradients.

## References

- [Design proposal](https://github.com/GridTools/gt4py_knowledge/pull/25)
- [ADR 0019 - Connectivities](0019-Connectivities.md)
