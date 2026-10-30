---
tags: [embedded, jax, connectivities]
---

# Connectivities Under JAX Tracing

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-10-30
- **Updated**: 2026-10-30

In the context of running embedded `premap` under `jax.jit`, facing a neighbor table that must be a runtime argument for the gather but concrete for domain inference, we decided to register jax connectivities as pytree nodes that carry a handle to the concrete table, to get the same narrowing as eager execution without compiling tables into the program. We accept a table reference held by jax's jit cache and a silent clamp when the table is only known at runtime.

## Context

`premap` uses the table in two ways that pull in opposite directions:

- The **output domain** comes from `inverse_image`, which depends on the table's values. Under a trace it needs concrete values, and Python `slice`s cannot be staged.
- The **gather** should take the table as an argument of the compiled program. A closed-over table is inlined as a constant, which does not scale to large meshes.

Narrowing is the common case: the field usually covers only part of the connectivity's image, so a solution that handles only covering ranges is not sufficient.

## Decision

`JaxArrayConnectivityField` is a pytree node. The table is the child, and the aux data holds domain, codomain, skip value and a `_TableHandle` to the same buffer. The handle compares by buffer identity, so a new wrapper around the same buffer does not retrace, while a new buffer does. Domain inference reads the table through the handle under `jax.ensure_compile_time_eval()`.

How a `premap` behaves under `jit` depends on how its connectivity reaches the compiled function:

- **Passed as a jit argument** (`jax.jit(lambda f, c: f.premap(c))(field, conn)`), the recommended way. The table is a tracer, and flattening attached the handle. Domain inference reads the concrete table through the handle, and the gather takes the table as a runtime argument.
- **Closed over** (`jax.jit(lambda f: f.premap(conn))(field)`). The table stays concrete, also after `restrict`. Domain inference reads it directly, but the table is inlined into the compiled program as a constant.
- **Computed inside the trace** (`as_offset(Koff, offset_field)`, where the offsets are data). The table is a tracer with no handle, because no concrete values exist at trace time. Domain inference assumes every target lies inside the field, so the output domain is not narrowed, and the gather clips its indices to the field.

## Consequences

- Domains under `jit` match eager execution when the table is concrete or has a handle.
- For a table computed in the trace, an out-of-range target raises in eager execution, but under `jit` reads the field's boundary value without an error and the domain is not narrowed. Such values are data, so no concrete bounds exist at trace time.
- Connectivities reject item assignment, because their domain lookups are cached per buffer.

Open questions, all caused by the handle:

- The handle is a strong reference, so the jit cache keeps every table it has seen alive until `jax.clear_caches()`. Memory stays bounded when a run uses a fixed set of tables.
- A pytree transform of a connectivity (`tree_map`, `vmap`) inside a trace changes the child but not the handle, so domain inference silently uses the old values. This is unsupported, and nothing detects it.
- `shard_map` rejects the handle when the table is sharded, so distributed execution is not supported yet.

## Alternatives considered

### Bounds in the aux data instead of the table

Storing only `(min, max)` retraces per bounds rather than per buffer and holds no table reference, but cannot answer narrowing ranges.

### For tables computed in the trace: reject them, or mask with NaN

Accepting only concrete tables rules out dynamic offsets under `jit`. NaN makes an out-of-range read visible, but does not work for integer fields and propagates into gradients.

## References

- [Design proposal](https://github.com/GridTools/gt4py_knowledge/pull/25)
- [ADR 0019 - Connectivities](0019-Connectivities.md)
