---
tags: []
---

# External Workspace Memory for DaCe Transients

- **Status**: valid
- **Authors**: Edoardo Paone (@edopao)
- **Created**: 2026-07-27
- **Updated**: 2026-08-03

In the context of `gt4py.next` DaCe backends that run the same compiled SDFG
many times (e.g. inside a time loop), facing per-call GPU transient allocation
overhead and the desire to bound workspace memory to one SDFG's transient
footprint, we decided to add an explicit `EXTERNAL` transient-memory mode
backed by caller-supplied workspace buffers (a plain `dict` of array-like
objects, one per device) to achieve bounded memory and zero per-call
allocation.

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
- **Explicit lifetime.** The workspace is released when the compiled program
  is done — not at GC time, and not per call.

The existing `PERSISTENT` mode ties one workspace to exactly one SDFG:
transients are allocated once and retained for that SDFG's lifetime, so the
workspace grows without bound across distinct SDFGs and is owned by the
compiled program, not the caller. `EXTERNAL` can reproduce that behaviour —
allocate once, retain — but is more general: it hands ownership of the
workspace memory to the caller. With that ownership the caller can reuse a
single workspace buffer across multiple compiled programs (sized to the
largest, or per storage type), which `PERSISTENT` cannot. Reuse is the
caller's responsibility: `EXTERNAL` issues one workspace per storage type and
the generated runtime path assumes the SDFG runs sequentially on the default
stream, so the caller must ensure no two programs using the same workspace run
concurrently. `POOL` is also not a fit for this reuse-across-programs goal: it
still allocates per SDFG, just amortized through the mempool.

## Decision

A single explicit `EXTERNAL` transient-memory mode, backed by **workspace
buffers supplied directly by the caller** (no allocator object).

- `TransientMemoryMode` enum (`SCOPED`, `PERSISTENT`, `POOL`, `EXTERNAL`),
  mutually exclusive, threaded through `gt_auto_optimize()` and the backend
  factory. `_gt_auto_post_processing` dispatches per mode: `SCOPED` is a
  no-op, `PERSISTENT` sets `AllocationLifetime.Persistent`, `POOL` applies the
  GPU mempool pass, and `EXTERNAL` sets `AllocationLifetime.External` on
  eligible transients via `gt_configure_transient_lifetime`.
- A public `ExternalWorkspace` `TypeAlias` —
  `dict[core_defs.DeviceType, ArrayInterface | CUDAArrayInterface]` — defined in
  `workflow/common.py`. The per-device array-like must be accepted by
  `dace.dtypes.array_interface_ptr()` as a workspace: a host array exposing
  `__array_interface__` or a device array exposing `__cuda_array_interface__`.
  No new protocol is introduced.
- The workspace is carried on a dedicated `DaCeBackend` (a `backend.Backend`
  subclass, frozen dataclass) through a new `external_workspace` attribute.
  `backend.Backend.load_artifact()` is a new overridable hook (defaulting to
  `artifact.load()`); `DaCeBackend.load_artifact()` calls the default and then
  injects the backend-level workspace onto the wrapped program via
  `DaCeDecoratedProgram.set_external_workspace(...)`. `CompiledProgramsPool`
  now calls `self.backend.load_artifact(...)` (in both
  `_finish_compilation_job` and `_compile_variant`) instead of
  `artifact.load()`, so every program loaded through a backend that carries a
  workspace gets it installed.
- At runtime, `CompiledDaceProgram.construct_arguments` calls
  `sdfg_program.get_workspace_sizes()`, looks up the matching per-device buffer
  in `external_workspace` (raising `RuntimeError` if the workspace is unset or
  the device is missing), validates it (array interface present; size at least
  the required bytes) and installs it via `sdfg_program.set_workspace(...)`.
- **Workspace lifetime is the caller's responsibility.** There is no
  `finalize()`, no `__del__`, and no `weakref.finalize` registered on
  `CompiledProgramsPool`. The caller owns the buffers and decides when they are
  freed; the compiled program simply borrows them for as long as it lives.

## Consequences

- A single reusable workspace buffer per storage type can back many sequential
  SDFG calls, removing per-call transient allocation/free in the generated
  runtime path while keeping memory bounded to one SDFG's transient footprint.
- Workspace lifetime is entirely the caller's: GT4Py never allocates or frees
  external workspace, so there is no pool-driven teardown and no resource leak
  from GT4Py's side. The flip side is that the caller must keep the buffers
  alive for the full lifetime of every compiled program that uses them and free
  them when appropriate.
- The public API is a single mode enum plus a plain `dict` of array-like
  buffers, threaded through the `DaCeBackend` instance. Incompatible
  combinations — a workspace without `EXTERNAL` mode (warns), or `EXTERNAL`
  mode without a workspace (raises `ValueError`) — are detected at backend
  construction.
- Because the workspace is no longer part of the compilation artifact, there is
  no picklability requirement on the buffers. Compilation can be offloaded to a
  worker process regardless of the workspace shape; the workspace is injected at
  load time, in the main process.
- Multi-stream safety is explicitly **out of scope** for this delivery:
  `EXTERNAL` reuses one workspace across sequential SDFG calls on the default
  stream. Concurrent use across streams is a follow-up.

## Alternatives considered

### An `ExternalMemoryAllocator` protocol (allocate/deallocate)

- Good, because it gives GT4Py a symmetric allocate/free contract and lets
  GT4Py drive teardown at pool finalization, so a forgotten buffer is reclaimed
  automatically.
- Bad, because it forces an allocator *object* (a `Protocol` with
  `allocate(request) -> wsp` / `deallocate(wsp)` methods, an
  `AllocationRequest` dataclass, and a separate `TypeAlias` for the workspace)
  into the compilation artifact, which is pickled when compilation is offloaded
  to a worker process. That in turn requires a picklability probe
  (`pickle.dumps` at `DaCeCompiler.__post_init__`) and a dedicated
  `AllocatorNotPicklableError` to avoid silently degrading to in-process
  compilation — machinery that exists only to shuttle a buffer from the backend
  to the runtime. The actual need is simpler: the caller already has the buffer
  and only wants to hand it over. Replaced by passing the workspace buffers
  directly, which removes the protocol, the request type, the picklability
  probe, the error, and the pool finalizer in one step. The cost is that the
  caller now owns the lifetime — acceptable because the caller is the party
  that allocated the memory in the first place.

### `__del__`-based teardown on `CompiledDaceProgram`

- Good, because it needs no pool integration.
- Bad, because the codebase strongly prefers `weakref.finalize` over `__del__`
  (hostile conditions at interpreter shutdown, partial initialization, and an
  allocator/buffer that may already be gone). In the direct-buffer design
  teardown is the caller's responsibility, so neither `__del__` nor a finalizer
  is needed. Rejected.

### A DLPack-consuming `set_workspace`

- Good, because DLPack is the cross-framework standard for zero-copy.
- Bad, because `dace.dtypes.array_interface_ptr()` (what DaCe uses for
  `set_workspace`) duck-types via `__array_interface__` /
  `__cuda_array_interface__` and `hasattr('data_ptr')`, and does not consume
  DLPack for this path. Introducing a DLPack requirement would narrow the set
  of accepted buffers without buying anything on this code path. Rejected;
  `ExternalWorkspace` is kept as a `TypeAlias` over the two array interfaces.

## References

- `src/gt4py/next/program_processors/runners/dace/transformations/auto_optimize.py`
  (`TransientMemoryMode`, `_gt_auto_post_processing`).
- `src/gt4py/next/program_processors/runners/dace/workflow/common.py`
  (`ExternalWorkspace` `TypeAlias`).
- `src/gt4py/next/program_processors/runners/dace/workflow/backend.py`
  (`DaCeBackend`, `make_dace_backend`, the `external_workspace` parameter and
  the mode/workspace compatibility checks).
- `src/gt4py/next/program_processors/runners/dace/workflow/compilation.py`
  (`CompiledDaceProgram.construct_arguments`/`_configure_external_workspace`,
  `_validate_external_workspace`).
- `src/gt4py/next/program_processors/runners/dace/workflow/decoration.py`
  (`DaCeDecoratedProgram.set_external_workspace`).
- `src/gt4py/next/backend.py` (`Backend.load_artifact`).
- `src/gt4py/next/otf/compiled_program.py`
  (`CompiledProgramsPool._finish_compilation_job`/`_compile_variant` calling
  `backend.load_artifact`).
- [ADR 0023](0023-Fingerprinting.md) and
  [ADR 0025](0025-Crash_Consistent_Build_Caches.md) for the cache and
  build-folder guarantees the external-memory path must not regress.
