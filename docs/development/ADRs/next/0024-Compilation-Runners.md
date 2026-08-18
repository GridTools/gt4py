---
tags: []
---

# [Compilation Runners]

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-07-08
- **Updated**: 2026-07-08

In the context of ahead-of-time compilation of many program variants
(`CompiledProgramsPool`), facing a thread pool that can neither hide the
CPython-bound cost of translation and optimization (GIL) nor parallelize the
DaCe compile path, we decided to introduce pluggable **compilation runners**
with a **process-pool default**, executing decomposed **compilation tasks** in
`spawn`ed worker processes using only stdlib `pickle`. We considered `fork` /
`forkserver` start methods, `cloudpickle`, and keeping threads as the default,
and accept a picklability contract on backend executors and compilation
artifacts, per-worker interpreter startup cost, and stricter isolation
requirements for GPU builds.

## Context

Programs are precompiled asynchronously: `Program.compile()` submits one job
per variant and execution overlaps with the rest of model setup (see
`wait_for_compilation`). This was previously backed by a module-global
`ThreadPoolExecutor`. Threads parallelize the external compiler invocations but serialize
everything that runs in the interpreter — GTIR transformations, lowering to
SDFG/C++ sources, DaCe auto-optimization — which dominates compile time for
large programs. Measured on ICON (dace GPU backend, clean cache), moving to
worker processes cut startup-to-first-timestep from ~28.5 min to ~10.4 min.

## Decision

### Runner abstraction

`gt4py.next.otf.runners` defines a `Runner` protocol —
`submit(CompilationTask) -> Future[CompilationArtifact]` plus `shutdown()` —
with three implementations selected via `GT4PY_BUILD_JOBS_MODE`
(`serial | thread | process`, default `process`; `GT4PY_BUILD_JOBS` sizes the
pools). Runners produce *artifacts*, never loaded programs: `load()` must run
in the process that will call the program, so loading is the caller's concern
(the pool loads on first use of a variant).

### Task decomposition

`gt4py.next.otf.compilation_tasks.make_compilation_task` prepares the unit a
runner schedules. Frontend lowering runs main-side — the raw user function is
rebound by the decorators and cannot cross a process boundary — producing a
pickle-safe `CompilableProgramDef`. A `CompilationTask` carries exactly two
ingredients, `construct_compilable(with_refs)` and `executor`, and every
runner executes the same pipeline: in-process as
`executor(construct_compilable(False))`, cross-process by shipping the pickled
`executor` together with `construct_compilable(True)`.

### Crossing the process boundary

- Only stdlib `pickle`. Backend executors are picklable by construction;
  the fingerprinting deconstructor singletons carry their own pickle
  identity for this purpose (see `0023`).
- Connectivity tables are not pickled: with `with_refs` they are replaced by
  file references that dump the table once per run (lazily, on first
  shipping) and rehydrate as memory-mapped arrays, shared by all workers
  through the page cache. Compilation only consumes device-independent values
  of the tables (shape and maximum), so main-side and worker-side compilation
  produce equivalent artifacts.
- A per-submit snapshot of `gt4py.next.config` is applied in the worker,
  since `spawn` workers do not see post-import config changes.

### Worker isolation

- `spawn`, not `fork`: the parent may hold running threads, CUDA contexts and
  MPI state, none of which survive `fork` safely — and `fork` would not relax
  the picklability requirement, since `ProcessPoolExecutor` pickles every
  submitted task regardless of start method.
- Workers must never touch the parent's GPU: GPU builds probe the device for
  its architecture (creating a CUDA context). The architecture is therefore
  resolved once main-side (`CUDAARCHS` environment variable takes precedence
  over a device query, see `otf.compilation.common.get_device_arch`) and
  exported to the workers, which run with hidden devices.
- Workers share the main process's session build cache directory, so the
  artifacts they produce outlive them.

### Lifecycle and failure handling

The default runner is created lazily on first use — importing gt4py must not
spin up multiprocessing machinery (e.g. inside mypy loading the gt4py plugin)
— and is shut down at interpreter exit. Worker processes always use a serial
runner to prevent recursive pools. `wait_for_compilation()` is the barrier for
all ongoing compilations and raises the failures it observes; failures are
otherwise raised when the affected program variant is first called.

### Graceful degradation

Backends that customize `Backend.compile` (which cannot be decomposed) or
whose executor is not picklable are compiled in the calling thread, with a
warning stating the reason. A non-conforming backend keeps working — it just
does not offload.

## Alternatives considered

- **`fork` start method**: inherits the parent's memory so pickle-by-reference
  of dynamically created objects would resolve, but is unsafe with CUDA, MPI
  and threads (all typical for gt4py users) and still pickles every submitted
  task.
- **`forkserver` with preload**: would amortize the per-worker import cost;
  left as a possible future optimization, it does not change the contracts.
- **Shipping derived information instead of connectivity tables**: the
  transformations consume only the table shape and the maximum neighbor index
  (besides the offset-provider type), so these scalars could be extracted
  main-side and shipped instead of the tables. This changes the interface of
  the GTIR transformations and of every consumer of compile-time offset
  providers, whereas the memory-mapped file references leave today's
  interfaces intact.
- **`cloudpickle`**: serializes lambdas and closures the stdlib cannot, but
  adds a runtime dependency and hides, rather than fixes, executors whose
  state should not cross a process boundary. An earlier prototype used it; it
  was removed in favor of making the executors properly picklable.
- **Keeping `thread` as the default**: avoids the picklability contract and
  the worker startup cost, but retains the GIL limitation that motivated this
  work; `thread` and `serial` remain available via `GT4PY_BUILD_JOBS_MODE`.

## Consequences

- `Backend.executor` and the `CompilationArtifact` it produces must be
  picklable with stdlib `pickle`; adding a lambda, closure or dynamically
  created callable to an executor silently degrades process mode to
  calling-thread compilation (with a warning).
- Each worker pays an interpreter start and gt4py import on spawn, amortized
  over the session; scripts that compile at module level need the standard
  `if __name__ == "__main__":` guard under `spawn`.
- `GT4PY_BUILD_JOBS` defaults to `min(os.cpu_count(), 32)`; in process mode
  every job is a full interpreter, so memory-constrained environments should
  set it explicitly.
