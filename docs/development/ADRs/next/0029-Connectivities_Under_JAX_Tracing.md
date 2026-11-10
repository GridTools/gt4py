---
tags: [embedded, jax, connectivities]
---

# Connectivities Under JAX Tracing

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-11-06
- **Updated**: 2026-11-06

In the context of running embedded `premap` under `jax.jit`, facing a neighbor table that must be a runtime argument for the gather but concrete for domain inference, we decided to register jax connectivities as pytree nodes that carry a handle to the concrete table, to get the same narrowing as eager execution without compiling tables into the program. We accept a table reference held by jax's jit cache and a silent clamp when the table is only known at runtime.

## Context

`premap` uses the table in two ways that pull in opposite directions:

- The **output domain** comes from `_inverse_image`, which depends on the table's values. Under a trace it needs concrete values, and Python `slice`s cannot be staged.
- The **gather** should take the table as an argument of the compiled program. A closed-over table is inlined as a constant, which does not scale to large meshes.

Narrowing is the common case: the field usually covers only part of the connectivity's image, so a solution that handles only covering ranges is not sufficient.

## Decision

`JaxArrayConnectivityField` is a pytree node. The table is the child, and the aux data holds domain, codomain, skip value and a handle to a concrete connectivity with the same table, its twin. The handle compares by the identity of the twin's table, so a new wrapper around the same buffer does not retrace, while a new buffer does. While the table is traced, domain inference and `restrict` delegate to the twin, which evaluates under `jax.ensure_compile_time_eval()`. One property, `_concrete`, names which of the cases below a connectivity is in, and every operation that differs between them branches on it.

How a `premap` behaves under `jit` depends on how its connectivity reaches the compiled function:

- **Passed as a jit argument** (`jax.jit(lambda f, c: f.premap(c))(field, conn)`), the recommended way. The table is a tracer, and flattening attached the twin. Domain inference reads the twin's table, and the gather takes the table as a runtime argument.
- **Closed over** (`jax.jit(lambda f: f.premap(conn))(field)`). The table stays concrete, also after `restrict`. Domain inference reads it directly, but the table is inlined into the compiled program as a constant. `as_offset` of a closed-over offset field behaves the same way, but slicing or computing with that field inside the trace makes it traced (next case).
- **Computed from traced values** (`as_offset(Koff, offset_field)` with a traced offset field). The table is a tracer with no twin, because no concrete values exist at trace time. Domain inference assumes every target lies inside the field, so the output domain is not narrowed, and the gather clips its indices to the field.

## Consequences

- Domains under `jit` match eager execution when the table is concrete or has a twin.
- A table computed from traced values behaves differently from eager execution when a target is out of range. Eagerly, a uniform offset narrows the domain and a non-contiguous one raises. Under `jit` the domain stays as it is and those targets read the field's boundary value, without an error.
- The clamp is only partly redundant with jax's own indexing, which clamps indices past the end but wraps negative ones as NumPy does. It compiles to an extra clamp next to the select jax emits for negative indices. A gather with `mode="clip", wrap_negative_indices=False` clamps both ends with no extra operations; whether that makes jax code faster is still to be investigated.
- Skip entries read the field's first element, and reductions mask the result afterwards. A gradient through an operator that is singular at that value is NaN (`0 * inf`), for example `sqrt` of a first element that is zero. The gather cannot prevent this, because no replacement value is safe for every operator.
- Connectivities reject item assignment, because their domain lookups are cached per buffer. Writing into the buffer through `.ndarray` is not detected and leaves those lookups stale, so it is not supported.

Open questions, all caused by the handle:

- The handle is a strong reference to the twin, so the jit cache keeps every table it has seen, and its cached domain lookups, alive until `jax.clear_caches()`. Memory stays bounded when a run uses a fixed set of tables.
- A pytree transform of a connectivity (`tree_map`, `vmap`) inside a trace changes the child but not the twin, so domain inference silently uses the old values. This is unsupported, and nothing detects it.
- `shard_map` rejects the twin when the table is sharded, so distributed execution is not supported yet.

## Alternatives considered

### Bounds in the aux data instead of the table

Storing only `(min, max)` retraces per bounds rather than per buffer and holds no table reference, but cannot answer narrowing ranges.

### For tables computed in the trace: reject them, or mask with NaN

Accepting only concrete tables rules out dynamic offsets under `jit`. NaN makes an out-of-range read visible, but does not work for integer fields and propagates into gradients.

## References

- [Design proposal](https://github.com/GridTools/gt4py_knowledge/pull/25)
- [ADR 0019 - Connectivities](0019-Connectivities.md)
