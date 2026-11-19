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

There is one connectivity class, `NdArrayConnectivityField`, for all array libraries and for all cases below. As for every embedded field, the connectivity does the domain operations, and `ArrayOps` only works on plain arrays. Domain inference reads the table's *static values*: an array with the table's values that can be computed on now, possibly of another library. By default they are the buffer itself. The box of entries that map into an image range is computed on them by one plain function for all libraries. The gather is `ArrayOps.take`, and everything that knows about tracing is in `jax_ops.py`.

Under a trace, a connectivity is supported in two cases, and its creator gives it its static values through a keyword-only init-only constructor argument:

- A connectivity passed to `jit` is rebuilt by the pytree unflatten around a tracer, which gets a read-only NumPy view of the concrete table from the aux data.
- A connectivity that `as_offset` computes from traced offsets gets `None`: its values exist only at run time, so the connectivity assumes that every target lies in range.

The static values are not a dataclass field, so no copy inherits them: a connectivity that got another table, for example through `dataclasses.replace` or `copy`, takes its static values from the new table. `restrict` restricts static values that are not the buffer itself along with the buffer, so a restriction inside a trace keeps them.

A concrete connectivity closed over by a trace is not supported, nor is any other connectivity built from a tracer, for example by `dataclasses.replace` or `copy` of a traced one. It either raises, usually jax's `ConcretizationTypeError` when domain inference computes on the traced buffer, or works, for example from domain lookups cached by an eager use; it does not compute a wrong result. Flattening keeps such a connectivity unsupported, so that it raises behind a pytree boundary as well.

The connectivity is a pytree node with the table as its child. The aux data holds domain, codomain, skip value and, if the table is concrete or a tracer with static values, the static values as a read-only NumPy array. They are compared by their memory, because NumPy returns a new view of a jax array on every call: two connectivities have the same aux data only if their static values are the same memory, which holds for every wrapper of one buffer and for its restrictions inside a trace with the same slice. Unflattening gives a child that is not a concrete array the static values from the aux data if it has their shape, and `None` otherwise, as under `vmap`, where the child is one element of the flattened table.

Nothing that belongs to a trace may be kept beyond it, so a connectivity caches domains and slices, never tables or restricted connectivities.

How a `premap` behaves under `jit` in the two supported cases:

- **Passed as a jit argument** (`jax.jit(lambda f, c: f.premap(c))(field, conn)`). The table is a tracer with static values. Domain inference and `restrict` read the static values, and the gather takes the table as a runtime argument. A neighbor reduction takes its table from the offset provider, so the provider has to be built from the traced connectivity inside the jitted function; a provider set outside is closed over.
- **Computed by `as_offset`** in the trace, from a traced offset field or from a closed-over one, whose table the trace computes. No concrete values exist at trace time. Domain inference assumes every target lies inside the field, so the output domain is not narrowed. The gather clamps its indices, so an out-of-range target reads some element of the field.

A tracer of any jax transformation counts as traced, not only one of `jit`. In particular a table that depends on a `vmap`ped value has no values even when `vmap` is called eagerly, because it differs per batch element.

### Control flow

The aux data claims that the child has the static values. Where jax rebuilds a pytree around new tracers, that claim stays true:

- At the entry of `jit`, of a nested `jit`, and for the operands of `lax.cond`, the new tracer stands for the flattened table.
- `lax.cond` and `lax.switch` require all branches to return the same aux data, and `lax.scan`, `lax.while_loop` and `lax.fori_loop` require the carry to keep its aux data. Selecting between two tables, changing the table of a carried connectivity, or mixing a table that has static values with one computed in the trace raises jax's `TypeError` about differing pytree structure. The same table on all branches, or an unchanged carry, keeps its static values.

For that check jax rebuilds the trees around placeholders that are not tracers, which is why unflattening attaches the static values to every child that is not a concrete array of jax, NumPy or CuPy.

A traced connectivity can also be closed over by the branches of `lax.cond`, the body of `lax.scan` or an inner `jit`, and used again afterwards.

### Under `torch.compile`

The same static values give torch connectivities the same narrowing under `torch.compile`, and everything that knows about it is in `torch_ops.py`. Dynamo does not trace the embedded code itself: an embedded program or field operator called in a compiled region is one `torch.compiler.nonstrict_trace` call, so a call with `fullgraph=True` is one graph without breaks. Inside it the embedded code runs on fake tensors; a backend that only runs the Dynamo graph, without AOT tracing, runs it eagerly on the concrete tensors instead.

Fields and connectivities are torch pytree nodes. Dynamo flattens them symbolically and accepts only constants in the pytree context: the domain is in it as plain tuples, and so is the `domain` argument of a field operator call. The concrete table of a connectivity cannot be read from its fake tensor, so `common._connectivity` gives every connectivity it creates from a concrete tensor an identity object of its table, registered with `torch.utils._pytree.register_constant`. Dynamo guards it by equality, which compares the memory of the table as the aux data does for jax. Unflattening gives a fake child the identity's table as NumPy static values, read on first use. `as_offset` in the trace gives `None`, as for jax.

Every tensor a compiled region reads reaches the call as an argument, so a connectivity closed over by the region, e.g. in an offset provider built outside it, is supported. A connectivity that did not get an identity, because `restrict`, `dataclasses.replace`, `copy` or `NdArrayConnectivityField.from_array` created it, raises a `TypeError` when it is flattened in the region.

## Consequences

- Domains under `jit` match eager execution when the connectivity is a `jit` argument.
- A compiled program is specific to the buffers of its connectivities: another buffer, even with equal values, retraces, and an ahead-of-time compiled program rejects it. A connectivity restricted eagerly has a new buffer.
- The tables must not change after they are used. A jax array that shares its memory with a NumPy array, as `jax.device_put` can on a CPU, must not be refilled through that array: the aux data would take the new table for the old one.
- A table computed by `as_offset` behaves differently from eager execution when a target is out of range. Eagerly, a uniform offset narrows the domain and a non-contiguous one raises. Under `jit` the domain stays as it is and those targets read some element of the field, without an error.
- The gather of a jax field is `jnp.take` with `mode="clip"`, which clamps both ends with no operation besides the gather.
- Skip entries read some element of the field, and reductions mask the result afterwards. A gradient through an operator that is singular at that value is NaN (`0 * inf`), for example `sqrt` of an element that is zero. The gather cannot prevent this, because no replacement value is safe for every operator.
- Connectivities reject item assignment, because their domain lookups are cached. Writing into the buffer through `.ndarray` is not detected and leaves those lookups stale, so it is not supported.
- `restrict` builds a new connectivity on every call. It shares the cached domain lookups of the connectivity it was restricted from.
- A traced connectivity that is copied, or rebuilt with `dataclasses.replace`, loses its static values and is not supported, like one closed over.
- The static values in the aux data are NumPy, so on a GPU every flatten of a concrete connectivity, as a `jit` argument, by `device_put` or by `tree_map`, reads a host copy of the table. jax makes that copy once per array and keeps it as long as the array; the jit cache keeps it alive. Eagerly, domain inference runs on the table itself. On a GPU the aux identity relies on jax keeping that copy, which is not public API.
- Domain inference is cached on the connectivity object. A connectivity rebuilt in a trace is new in every trace, so its domain inference is repeated per trace but not per call.
- Under `torch.compile` a compiled graph is specific to the buffers of its connectivities, as under `jit`. A connectivity restricted eagerly is not supported; JAX supports it.
- The identity of a torch connectivity holds its tensor, so a cached graph keeps every table it has seen alive, until `torch._dynamo.reset()`. On a GPU the first trace of a connectivity reads a host copy of its table, which the identity keeps.

Open questions, all caused by the static values in the aux data:

- The aux data is a strong reference to the table, so the jit cache keeps every table it has seen alive until `jax.clear_caches()`. Memory stays bounded when a run uses a fixed set of tables.
- Rebuilding a connectivity around another traced table through the pytree API inside a trace (`tree_map` with a function that changes the leaf, `tree_unflatten`) keeps the static values of the old table, so domain inference silently uses them. This is unsupported, and nothing detects it: with jax's public API such a rebuild cannot be told apart from the entry of a `jit`.
- `shard_map` rejects the aux data when the table is sharded, so distributed execution is not supported yet.

## Alternatives considered

### Table operations on a per-connectivity object

Domain inference and `restrict` were methods of a `ConnectivityOps` object, one per connectivity and registered per library, which held the static values and returned its successor from `restrict`. Its methods took plain arrays, but the object carried state about one buffer and the decision what a table without values means, which is the connectivity's. It needed its own registry next to `ArrayOps`. An earlier variant passed an opaque `state` stored on the connectivity back to `ArrayOps`; every copy inherited it, so each operation had to check that the state described its table.

### Supporting closed-over connectivities

A hook `ArrayOps.static_values` would give the static values of any buffer: for a concrete jax array a NumPy view, so that domain inference on a closed-over table is not staged, and `None` for any tracer. It would also support connectivities built from other tracers, with a clamped gather. icon4py passes its connectivities as `jit` arguments, and a closed-over table is inlined into the program, which is slower; no use case needs the hook.

### Static values in a side table keyed by the connectivity

A weak map from connectivity to static values in `jax_ops.py` needs no constructor argument, but `restrict` would need a hook to hand the restricted values back, and `ArrayOps` would have to take connectivities.

### Static values found by the table

A map from table to static values, inside `jax_ops.py`, needs no state on the connectivity. But unflattening attaches static values to placeholders like `None`, integers and `object()`, which cannot be weakly referenced, and `None` is the same object everywhere.

### `is_traced` and `concrete_eval` on `ArrayOps`

Two small hooks, with the case distinction written out in `NdArrayConnectivityField`. The connectivity would then have to know when to evaluate concretely. Its creators instead give it static values that can be computed on, so the connectivity needs no evaluation mode.

### Static values that follow the connectivity instead of the table

Earlier versions kept the concrete connectivity in a field that every copy of the connectivity inherits, and derived the aux data of a connectivity with a concrete leaf from that leaf. `dataclasses.replace` then attached the values to another table, and jax's comparison of branches and carries, which runs on trees with a shared placeholder leaf, found nothing to object to: `lax.cond` between two tables silently used the values of the first one. The static values are not inherited, and a concrete table derives its own.

### Bounds in the aux data instead of the table

Storing only `(min, max)` retraces per bounds rather than per buffer and holds no table reference, but cannot answer narrowing ranges.

### For tables computed in the trace: reject them, or mask with NaN

Accepting only concrete tables rules out dynamic offsets under `jit`. NaN makes an out-of-range read visible, but does not work for integer fields and propagates into gradients.

## References

- [ADR 0019 - Connectivities](0019-Connectivities.md)
