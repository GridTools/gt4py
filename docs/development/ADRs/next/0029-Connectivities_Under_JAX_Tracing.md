---
tags: [embedded, jax, connectivities]
---

# Connectivities Under JAX Tracing

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-11-19
- **Updated**: 2026-11-19

In the context of running embedded `premap` under `jax.jit`, facing a neighbor table that must be a runtime argument for the gather but concrete for domain inference, we decided to pass the concrete table along with its tracer through the pytree aux data of the connectivity, to get the same narrowing as eager execution without compiling tables into the program. We accept a table reference held by jax's jit cache and a silent clamp when the table is only known at runtime (`as_offset`).

## Context

`premap` uses the table twice, and needs it at different times:

- The **output domain** comes from `_inverse_image`, which depends on the table's values. It is computed at trace time, so it needs concrete values: Python `slice`s cannot be staged.
- The **gather** runs in the compiled program, so it should take the table as a run-time argument. A closed-over table is inlined as a constant, which does not scale to large meshes.

## Decision

There is one connectivity class, `NdArrayConnectivityField`, for all array libraries. As for every embedded field, the connectivity does the domain operations and `ArrayOps` only works on plain arrays; everything that knows about tracing is in `jax_ops.py`. Domain inference reads the connectivity's *static values*: an array with the table's values that can be computed on now, by default the buffer itself.

Under a trace, two connectivities are supported, and their creators give them their static values:

- **Passed as a jit argument** (`jax.jit(lambda f, c: f.premap(c))(field, conn)`). The pytree unflatten rebuilds the connectivity around a tracer, with a read-only NumPy view of the concrete table from the aux data as static values. Domain inference and `restrict` read them, and the gather takes the table as a runtime argument. A neighbor reduction takes its table from the offset provider, so the provider has to be built from the traced connectivity inside the jitted function.
- **Computed by `as_offset`** from a traced offset field. It has no static values: its values exist only at run time. Domain inference assumes every target lies inside the field, so the output domain is not narrowed, and the gather clamps its indices, so an out-of-range target reads some element of the field.

`restrict` restricts static values that are not the buffer itself along with the buffer.

A concrete connectivity closed over by a trace is not supported, nor is a connectivity built from another tracer. It either raises or works; it does not compute a wrong result.

The connectivity is a pytree node with the table as its child and the static values in the aux data, compared by their memory, so every wrapper of one buffer has equal aux data.

Nothing that belongs to a trace may be kept beyond it, so a connectivity caches domains and slices, never tables or restricted connectivities.

### Control flow

The aux data claims that the child has the static values. Where jax rebuilds a pytree around new tracers, that claim stays true:

- At the entry of `jit`, of a nested `jit`, and for the operands of `lax.cond`, the new tracer stands for the flattened table.
- `lax.cond` and `lax.switch` require all branches to return the same aux data, and `lax.scan`, `lax.while_loop` and `lax.fori_loop` require the carry to keep its aux data. Selecting between two tables, changing the table of a carried connectivity, or mixing a table that has static values with one computed in the trace raises jax's `TypeError` about differing pytree structure. The same table on all branches, or an unchanged carry, keeps its static values.

For that check jax rebuilds the trees around placeholders that are not tracers, which is why unflattening attaches the static values to every child that is not a concrete array.

A connectivity passed to `jit` can also be used by the branches of `lax.cond`, the body of `lax.scan` or an inner `jit` that close over it, and again afterwards.

### Under `torch.compile`

Torch connectivities get the same narrowing under `torch.compile`, and everything that knows about it is in `torch_ops.py`. Dynamo does not trace embedded code: a compiled region calls it through a `torch.compiler.nonstrict_trace` function, which is one graph without breaks under `fullgraph=True` and runs the embedded code on fake tensors. Embedded programs and field operators do not do that call themselves yet.

Fields and connectivities are torch pytree nodes. Dynamo flattens them symbolically and accepts only constants in the pytree context, so the domain is in it as plain tuples. A region that gets fields of different domains has to be compiled with `dynamic=False`: otherwise Dynamo makes the bounds of the second domain symbolic, and they are no constants. Dynamo cannot read a table from its fake tensor, so `common._connectivity` gives every connectivity it creates from a concrete tensor its static values for the pytree context, a `torch.utils._pytree.register_constant` object compared by memory, as in the aux data for jax. Torch tensors are mutable, so the memory includes the version counter of the tensor. Dynamo guards it by equality. Unflattening gives a fake child their values as NumPy static values, read when a trace first unflattens the connectivity. A table computed by `as_offset` in the trace has none, as for jax.

A connectivity closed over by the region and passed to the `nonstrict_trace` function is supported; one closed over by that function itself is not. A connectivity without static values for the context, built by `restrict`, `dataclasses.replace`, `copy`, pickling or `NdArrayConnectivityField.from_array`, raises a `TypeError` when it is flattened in the region, which Dynamo reports as `Unsupported`. The `nonstrict_trace` function cannot return a connectivity with static values.

## Consequences

- Domains under `jit` match eager execution when the connectivity is a `jit` argument.
- A compiled program is specific to the buffers of its connectivities: another buffer, even with equal values, retraces, and an ahead-of-time compiled program rejects it. A connectivity restricted eagerly has a new buffer.
- The array of a connectivity must not change after it is used. A jax array that shares its memory with a NumPy array, as `jax.device_put` can on a CPU, must not be refilled through that array: the aux data would take the new table for the old one.
- A table computed by `as_offset` behaves differently from eager execution when a target is out of range. Eagerly, a uniform offset narrows the domain and a non-contiguous one raises. Under `jit` the domain stays as it is and those targets read some element of the field, without an error.
- Skip entries read some element of the field, and reductions mask the result afterwards. A gradient through an operator that is singular at that value is NaN (`0 * inf`), for example `sqrt` of an element that is zero. The gather cannot prevent this, because no replacement value is safe for every operator.
- Connectivities reject item assignment, because their domain lookups are cached. Writing into the buffer through `.ndarray` is not detected and leaves those lookups stale, so it is not supported.
- `restrict` builds a new connectivity on every call. It shares the cached domain lookups of the connectivity it was restricted from.
- On a GPU, every flatten of a concrete connectivity (`jit`, `device_put`, `tree_map`) reads a host copy of the table, which jax keeps as long as the array, and the jit cache keeps alive. The aux data's identity relies on jax keeping that copy, which is not public API. Eagerly, domain inference runs on the table itself.
- Domain inference is cached on the connectivity object. A connectivity rebuilt in a trace is new in every trace, so its domain inference is repeated per trace but not per call.
- Under `torch.compile` a compiled graph is specific to the buffers of its connectivities, as under `jit`, and to the domains of its fields. Every other buffer or domain is a recompile, so Dynamo's recompile limit (`torch._dynamo.config.recompile_limit`) bounds their number, and with `fullgraph=True` the call after the limit raises. Unlike under `jit`, a connectivity restricted eagerly is not supported.
- The static values in the pytree context of a torch connectivity hold its tensor, so the compiled code keeps every table it has seen alive; with torch 2.15 even `torch._dynamo.reset()` does not release them. On a GPU the first trace of a connectivity reads a host copy of its table, which they keep.

Open questions, all caused by the static values in the aux data:

- The aux data is a strong reference to the table, so the jit cache keeps every table it has seen alive until `jax.clear_caches()`. Memory stays bounded when a run uses a fixed set of tables.
- Rebuilding a connectivity around another traced table through the pytree API inside a trace (`tree_map` with a function that changes the leaf, `tree_unflatten`) keeps the static values of the old table, so domain inference silently uses them. This is unsupported, and nothing detects it: with jax's public API such a rebuild cannot be told apart from the entry of a `jit`.
- `shard_map` rejects the aux data when the table is sharded, so distributed execution is not supported yet.

## Alternatives considered

### Table operations on a per-connectivity object

A `ConnectivityOps` object per connectivity, registered per library, could hold the static values and do domain inference and `restrict` on plain arrays. But it carries state about one buffer and the decision what a table without values means, which is the connectivity's, and needs a second registry next to `ArrayOps`.

### Supporting closed-over connectivities

A hook `ArrayOps.static_values` could give the static values of any buffer: for a concrete jax array a NumPy view, so that domain inference on a closed-over table is not staged, and none for a tracer, which would also support connectivities built from other tracers. For now we don't have a use case, but this decision can be re-opened when we find one.

### `is_traced` and `concrete_eval` on `ArrayOps`

Two small hooks, with the case distinction written out in `NdArrayConnectivityField`, which would then have to know when to evaluate concretely. With static values that can be computed on, it needs no evaluation mode.

### Bounds in the aux data instead of the table

Storing only `(min, max)` retraces per bounds rather than per buffer and holds no table reference, but cannot answer narrowing ranges.

### For tables computed in the trace: reject them, or mask with NaN

Accepting only concrete tables rules out dynamic offsets under `jit`. NaN makes an out-of-range read visible, but does not work for integer fields and propagates into gradients.

## References

- [ADR 0019 - Connectivities](0019-Connectivities.md)
