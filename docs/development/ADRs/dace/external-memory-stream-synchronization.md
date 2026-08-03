---
tags: []
---

# External Memory Stream Synchronization for the DaCe Backend

- **Status**: valid
- **Authors**: Edoardo Paone (@edopao)
- **Created**: 2026-07-28
- **Updated**: 2026-07-28

## Context

The DaCe backend in `gt4py.next` can keep transient workspace memory externally
allocated across program calls by setting
`transient_memory_mode=TransientMemoryMode.EXTERNAL`. This removes per-call GPU
allocation overhead, but it relies on sequential execution on the default stream:
no other work may access the workspace between two consecutive calls.

A concrete use case that breaks this assumption is overlapping GT4Py program
execution with MPI halo exchange. The user runs halo exchange on one CUDA/HIP
stream and the GT4Py program on DaCe's internal streams. Without explicit
ordering, the external workspace can be reused before the previous call's kernels
or the halo exchange has finished, leading to data races.

DaCe itself can schedule GPU kernels on multiple internal streams when
`compiler.cuda.max_concurrent_streams` is set to a positive value. GT4Py has
historically forced this value to `-1` (default stream only) because of DaCe
stream codegen issues. We want to make multi-stream scheduling opt-in and safe
for external workspace mode.

## Decision

We introduce an optional multi-stream safety layer with the following design.

### Configuration

- Add `GT4PY_MAX_CONCURRENT_GPU_STREAMS` to `gt4py.next.config`. It maps to
  DaCe's `compiler.cuda.max_concurrent_streams`. A value of `0` keeps the
  historical default-stream-only behavior (DaCe value `-1`); values `> 0` enable
  DaCe's internal stream pool. Negative values are rejected at import time.
- The setting is ignored on CPU targets.

### External sync stream

- Add an optional `external_sync_stream: cupy.cuda.Stream | None = None`
  argument to `make_dace_backend`.
- The stream is stored on a DaCe-specific `Backend` subclass
  (`DaCeBackend`) instead of the picklable executor workflow, because
  `cupy.cuda.Stream` is not picklable. The stream is injected at compiled-program
  load time via `DaCeCompilationArtifact.load(external_sync_stream=...)`.
- If no stream is provided, the CUDA/HIP default stream (`0`) is used implicitly
  as the synchronization anchor.

### Synchronization mechanism

When `GT4PY_MAX_CONCURRENT_GPU_STREAMS > 0` and the SDFG contains GPU schedules,
GT4Py injects two CUDA/HIP tasklets into the generated SDFG:

- **Entry tasklet** (`external_ws_sync_entry_tlet`): every internal DaCe stream
  waits on a GT4Py-owned cross-call event before touching the external workspace.
- **Exit tasklet** (`external_ws_sync_exit_tlet`):
  1. Record per-stream done events on each internal stream.
  2. Make the external sync stream wait on all those events.
  3. Record the cross-call event on the external sync stream.

The tasklets reference two SDFG scalar symbols:

- `__external_ws_event`: a `cupy.cuda.Event` owned by `CompiledDaceProgram`.
- `__external_sync_stream`: the integer pointer of the external stream (or `0`).

Both are passed to the SDFG as `uint64` scalar arguments.

### Validation

- `external_sync_stream` is validated during compiled-program construction:
  - Rejected on CPU targets.
  - Must be a `cupy.cuda.Stream`.
  - Must be on the current GPU device.
  - Must pass `cudaStreamQuery` (accepting both `cudaSuccess` and
    `cudaErrorNotReady`).
- The combination `transient_memory_mode=POOL` and
  `GT4PY_MAX_CONCURRENT_GPU_STREAMS > 0` is rejected in `make_dace_backend`,
  because the existing GPU memory-pool pass assumes default-stream ordering and
  could hand out overlapping pool regions to concurrent kernels.

### Out of scope

- Per-call stream selection.
- Non-cupy stream support (e.g. torch, raw CUDA, HIP handles).
- Device-wide synchronization.
- Multi-stream safety for `SCOPED`/`POOL` modes.

## Consequences

- External workspace mode becomes safe with concurrent GPU streams, enabling the
  halo-exchange overlap use case.
- Default behavior is unchanged: `GT4PY_MAX_CONCURRENT_GPU_STREAMS=0` produces
  the same SDFG as before.
- The feature introduces additional CUDA/HIP event overhead per call, but only
  when multi-stream scheduling is explicitly requested.
- The `cupy` dependency for stream validation is isolated to GPU execution paths;
  CPU-only installations are unaffected.

## Alternatives considered

- **Device-wide synchronize**: simpler, but serializes the entire device and
  defeats the purpose of multi-stream scheduling.
- **Per-call stream argument**: more flexible, but requires changing the
  decorated program call signature and was not needed for the primary use case.
- **Store events in generated device code**: rejected because events must be
  created and destroyed from host code; a host-side Python object is needed for
  proper lifecycle management.

## References

- `PLAN-External_Memory_For_Dace_Arrays.md`
- `dace_stream_spec.md`
- [DaCe issue #2120](https://github.com/spcl/dace/issues/2120)
