---
tags: [embedded, backend, jax, torch]
---

# Embedded Jit Backends

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-11-20
- **Updated**: 2026-11-20

In the context of running programs and field operators on JAX arrays or torch tensors with the speed of the library's compiler, facing that embedded execution is plain Python and compiled backends do not take these arrays, we decided to add backends that wrap the embedded execution of an entry point in the library's tracing compiler (`jax.jit`, `torch.compile`), to get compiled programs without writing jax or torch code. We accept that scalar arguments are traced, so Python control flow or a domain bound on one needs `static_params`.

## Context

Embedded execution runs the definition of a program on fields of any array library. Under `jax.jit` or `torch.compile` it traces into a single compiled function, and ADR 0029 makes connectivities runtime arguments of it. The user had to write the jit wrapper by hand: split the offset provider, return the new `out` buffers (JAX arrays are immutable), and keep the wrapper cached.

## Decision

`EmbeddedJitBackend` (in `gt4py.next.backend`) is a `Backend` that does not compile: its `jit(entry_point)` returns the entry point run embedded inside the tracing compiler. There are two instances, `runners.jax_jit.jax_jit` and `runners.torch_compile.torch_compile`. What they share (the embedded execution, its signature and static parameters, the named collections) is in `runners.embedded_jit`; each backend only wraps it in its compiler.

- **Dispatch**: this is a stopgap until embedded execution is represented by backends, so the decorator has a single hook. An entry point with such a backend takes the embedded path and calls `backend.jit(entry_point)`, cached on the entry point. Whether an entry point is compiled is an attribute set at creation, so the call of a compiled backend costs what it did. `.compile()` raises.
- **Arguments**: fields and connectivities (pytrees, ADR 0029) and scalars are traced. The `Dimension`s of the offset provider and the parameters in `static_params` are static; for a field operator called with `out=`, so is `domain`.
- **Named collections** reach the backend as tuples and are rebuilt inside, as the compiled path extracts them: no user type is registered as a pytree.
- **`out`**: under jax the jitted function returns the new buffers of the fields it assigned to, which the caller's fields get; a field passed twice and assigned to is rejected. Under torch the assignment is in place.
- **Embedded changes**: a traced scalar is a 0-d array, so embedded accepts 0-d arrays where it accepts scalars, and the element of a 0-d traced array stays an array (`ArrayOps.scalar`).

torch specifics, all from Dynamo: the embedded code runs through `nonstrict_trace`, which takes neither NumPy scalars nor frozen dataclasses, so scalars become 0-d tensors and the static arguments are closed over by one region per combination of their values. Each region gets its own code object, since Dynamo's recompile limit counts per code object. An exception of the embedded code is re-raised as itself.

## Consequences

- A program on jax or torch fields is compiled by `with_backend(jax_jit)` or `with_backend(torch_compile)`.
- A scalar that the embedded code needs concretely (an `if`, a domain bound, a scan carry in a condition) fails under tracing unless it is in `static_params`; the test matrix marks such tests (`uses_scalar_in_control_flow`, `uses_scalar_in_domain`, `uses_scan_carry_in_control_flow`).
- Retracing: per field domain and per table buffer (ADR 0029), per value of a static parameter. torch allows 8 recompiles per region.
- Embedded scans loop per column and level, so their trace grows with the size of the domain.
- The embedded-program hook wraps the jitted call, so it runs per call, not per trace.

## Alternatives considered

### Jit backends as compiled backends

Entering through the compiled programs pool needs the embedded context set up again, a special case for `domain=` on the compiled path, and skips the embedded hook.

### Registering named collections as pytrees

Mutates the global pytree registry for user types, conflicts with a user's own registration, and misses types that appear only as `out`.

### Specialized scalars under torch

Dynamo's default: every new value of an integer recompiles, and the recompile limit is reached by, e.g., a time-step counter.
