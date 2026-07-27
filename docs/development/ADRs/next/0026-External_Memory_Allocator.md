---
tags: []
---

# External Memory Allocator for DaCe Transients

- **Status**: valid
- **Authors**: Edoardo Paone (@edopao)
- **Created**: 2026-07-27
- **Updated**: 2026-07-27

In the context of `gt4py.next` DaCe backends that run the same compiled SDFG
many times (e.g. inside a time loop), facing per-call GPU transient allocation
overhead and the desire to bound workspace memory to one SDFG's transient
footprint, we decided to add an explicit `EXTERNAL` transient-memory mode
backed by a caller-supplied `ExternalMemoryAllocator` protocol (allocate once
per SDFG storage type, release when the compiled program is closed).

## Context

DaCe transients default to `AllocationLifetime.SDFG` (scoped): the generated
code allocates and frees them around every SDFG call. For workloads that run
the same compiled SDFG many times, this per-call allocation/free is overhead,
and for workloads that run many distinct SDFGs in a loop, the natural goal is a
single reusable workspace buffer per storage type, bounded to one SDFG's
transient footprint.

Two properties are required of such a mode:

- **Allocate once, reuse across calls.** The workspace is sized once (from the
  SDFG's own `get_workspace_sizes()`) and installed via `set_workspace(...)`,
  so no per-call allocation appears in the generated runtime path.
- **Explicit, pool-driven lifetime.** The workspace is released when the
  compiled program is done — not at GC time, and not per call.

The existing `PERSISTENT` mode ties one workspace to exactly one SDFG:
transients are allocated once and retained for that SDFG's lifetime, so the
workspace grows without bound across distinct SDFGs and is owned by the
compiled program, not the caller. `EXTERNAL` can reproduce that behaviour --
allocate once, retain, release at close -- but is more general: it hands
ownership of the workspace memory to the caller. With that ownership the
caller can reuse a single workspace buffer across multiple compiled programs
(sized to the largest, or per storage type), which `PERSISTENT` cannot.
Reuse is the caller's responsibility: `EXTERNAL` issues one workspace per
storage type and the generated runtime path assumes the SDFG runs
sequentially on the default stream, so the caller must ensure no two
programs using the same workspace run concurrently. `POOL` is also not a
fit for this reuse-across-programs goal: it still allocates per SDFG, just
amortized through the mempool.

## Decision

A single explicit `EXTERNAL` transient-memory mode, backed by a caller-supplied
allocator.

- `TransientMemoryMode` enum (`SCOPED`, `PERSISTENT`, `POOL`, `EXTERNAL`),
  mutually exclusive, threaded through `gt_auto_optimize()` and the backend
  factory. `_gt_auto_post_processing` dispatches per mode: `SCOPED` is a
  no-op, `PERSISTENT` sets `AllocationLifetime.Persistent`, `POOL` applies the
  GPU mempool pass, and `EXTERNAL` sets `AllocationLifetime.External` on
  eligible transients via `gt_configure_transient_lifetime`.
- A public `ExternalMemoryAllocator` Protocol
  (`allocate(request: AllocationRequest) -> Buffer` /
  `deallocate(buffer) -> None`), defined in `transformations/auto_optimize.py`
  alongside `AllocationRequest` (`nbytes`, `device`, `alignment=256`) and
  `Buffer` (a `TypeAlias` over the existing `ArrayInterface` /
  `CUDAArrayInterface` from `gt4py.eve.extended_typing`, not a new protocol).
- At runtime, `CompiledDaceProgram.construct_arguments` calls
  `sdfg_program.get_workspace_sizes()`, invokes `allocate` once per storage
  type, validates the returned buffer (array interface, size, alignment) and
  installs it via `sdfg_program.set_workspace(...)`. The buffers are kept on
  the `CompiledDaceProgram` for its lifetime.
- Teardown is explicit and pool-driven, not `__del__`-based:
  `CompiledDaceProgram.close()` calls `deallocate` once per storage type (and
  is idempotent and resilient: a failing `deallocate` is warned, not raised,
  so one bad buffer does not strand the rest). `decorated_program` in
  `workflow/decoration.py` forwards `close()` to the underlying
  `CompiledDaceProgram`, and `CompiledProgramsPool.__post_init__` registers a
  `weakref.finalize` that walks `compiled_programs` and calls `close()` on
  each when the pool is collected. This mirrors the existing
  `metrics_source_key` finalizer and its "avoid id reuse once a pool dies"
  rationale.
- The allocator is part of `DaCeCompilationArtifact` and is therefore
  **picklable**: when the OTF runner offloads compilation to a
  `ProcessPoolExecutor` it pickles the executor chain, which carries the
  allocator. `DaCeCompiler.__post_init__` probes the allocator with
  `pickle.dumps` and raises `AllocatorNotPicklableError` (a `TypeError`) at
  backend construction if it cannot be pickled, rather than letting a closure
  or lambda silently degrade to in-process compilation via the generic runner
  warning. The error chains the original pickle failure and names the
  recommended shape (module-level class or `functools.partial` of picklable
  callables).

## Consequences

- A single reusable workspace buffer per storage type can back many sequential
  SDFG calls, removing per-call transient allocation/free in the generated
  runtime path while keeping memory bounded to one SDFG's transient footprint.
- Workspace lifetime is explicit and tied to the compiled program's lifetime
  through the pool finalizer, not to GC timing. Backends owning external
  resources (this one) get `close()` called at pool teardown.
- The public API has a typed allocator protocol and a single mode enum;
  incompatible combinations (e.g. an allocator with a non-`EXTERNAL` mode)
  are detected and warned at backend construction.
- A non-picklable allocator fails loudly at construction instead of silently
  serializing compilation, at the cost of probing every allocator once with
  `pickle.dumps` (cheap for the common module-level-class shape).
- Multi-stream safety is explicitly **out of scope** for this delivery:
  `EXTERNAL` reuses one workspace across sequential SDFG calls on the default
  stream. Concurrent use across streams is a follow-up (the plan's Phase 6).

## Alternatives considered

### `__del__`-based teardown on `CompiledDaceProgram`

- Good, because it needs no pool integration.
- Bad, because the codebase strongly prefers `weakref.finalize` over `__del__`
  (hostile conditions at interpreter shutdown, partial initialization, and an
  allocator that may already be gone). Rejected in favor of the explicit
  `close()` forwarded through the closure and driven by the pool finalizer.

### A DLPack-consuming `set_workspace`

- Good, because DLPack is the cross-framework standard for zero-copy.
- Bad, because `dace.dtypes.array_interface_ptr()` (what DaCe uses for
  `set_workspace`) duck-types via `__array_interface__` /
  `__cuda_array_interface__` and `hasattr('data_ptr')`, and does not consume
  DLPack for this path. Introducing a DLPack requirement would narrow the set
  of accepted buffers without buying anything on this code path. Rejected;
  `Buffer` is kept as a `TypeAlias` over the two array interfaces.

## References

- `src/gt4py/next/program_processors/runners/dace/transformations/auto_optimize.py`
  (`ExternalMemoryAllocator`, `AllocationRequest`, `Buffer`,
  `TransientMemoryMode`, `_gt_auto_post_processing`).
- `src/gt4py/next/program_processors/runners/dace/workflow/compilation.py`
  (`CompiledDaceProgram.construct_arguments`/`close`,
  `DaCeCompilationArtifact`, `DaCeCompiler`, `AllocatorNotPicklableError`).
- `src/gt4py/next/program_processors/runners/dace/workflow/decoration.py`
  (`decorated_program.close` forwarding).
- `src/gt4py/next/otf/compiled_program.py`
  (`_close_compiled_programs`, `CompiledProgramsPool.__post_init__`
  finalizer).
- [ADR 0023](0023-Fingerprinting.md) and
  [ADR 0025](0025-Crash_Consistent_Build_Caches.md) for the cache and
  build-folder guarantees the external-memory path must not regress.
