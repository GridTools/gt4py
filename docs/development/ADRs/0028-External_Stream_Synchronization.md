---
tags: []
---

# External Memory Stream Synchronization for the DaCe Backend

- **Status**: valid
- **Authors**: Edoardo Paone (@edopao)
- **Created**: 2026-07-28
- **Updated**: 2026-08-05

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

We introduce an optional synchronization layer with the following design.

### Configuration

- Add `max_concurrent_gpu_streams` as a parameter to `DaCeWorkflowFactory` and
  the `make_dace_backend` helper. It maps to DaCe's
  `compiler.cuda.max_concurrent_streams`. A value of `0` keeps the historical
  default-stream-only behavior (DaCe value `-1`); values `>= 1` enable DaCe's
  internal stream pool. Negative values are rejected at backend construction
  time (the validation lives in `DaCeTranslator.__post_init__`, which is reached
  by both direct translator creation and the backend factory).
- Add `async_sdfg_call` as a parameter to `make_dace_backend`. When `True`
  (default), the SDFG call returns immediately and GPU kernels run concurrently
  with the Python driver; when `False`, the SDFG call blocks until the GPU work
  has completed.
- The settings are ignored on CPU targets.

### External sync stream

- Add an optional `external_sync_stream: cupy.cuda.Stream | None = None`
  argument to `make_dace_backend`.
- The stream is stored on a DaCe-specific `Backend` subclass
  (`DaCeBackend`) instead of the picklable executor workflow, because
  `cupy.cuda.Stream` is not picklable. The stream is injected at compiled-program
  load time via `DaCeCompilationArtifact.load()`.
- If no stream is provided and `max_concurrent_gpu_streams >= 1`, the CUDA/HIP
  default stream (`0`) is used implicitly as the synchronization anchor.
- When `max_concurrent_gpu_streams == 0`, `external_sync_stream` is ignored and
  the SDFG executes asynchronously on the default stream.

### Synchronization mechanism

When `max_concurrent_gpu_streams >= 1` and the SDFG contains GPU schedules,
GT4Py injects two CUDA/HIP tasklets into the generated SDFG:

- **Entry tasklet** (`sync_entry_tlet`): records per-stream done events on each
  internal DaCe stream and makes the external sync stream wait on them before
  touching the external workspace.
- **Exit tasklet** (`sync_exit_tlet`): records per-stream done events on each
  internal stream and makes the external sync stream wait on them before the SDFG
  call returns.

When `async_sdfg_call=False`, both tasklets additionally call
`cudaStreamSynchronize` on the external sync stream, so the SDFG call blocks
until all GPU work has completed.

The tasklets reference the following SDFG scalar symbol:

- `__external_sync_stream`: the integer pointer of the external stream (or `0`).

It is passed to the SDFG as a `uint64` scalar argument.

When `max_concurrent_gpu_streams == 0`, no event tasklets are generated. Instead,
GT4Py emits DaCe init code that forces all internal streams to the default
stream, preserving the historical asynchronous single-stream execution.

### Validation

- `external_sync_stream` is validated during compiled-program construction when
  it is not `None`:
  - Rejected on CPU targets.
  - Must be a `cupy.cuda.Stream`.
  - Must be on the current GPU device.
  - Must pass `cudaStreamQuery` (accepting both `cudaSuccess` and
    `cudaErrorNotReady`).
- When `max_concurrent_gpu_streams == 0`, `external_sync_stream` is ignored even
  if provided, so no validation occurs.
- The combination `transient_memory_mode=POOL` and
  `max_concurrent_gpu_streams >= 1` is rejected in `make_dace_backend`,
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
- Default behavior is unchanged: `max_concurrent_gpu_streams=0` produces the same
  SDFG as before.
- The feature introduces additional CUDA/HIP event overhead per call, but only
  when multi-stream scheduling is explicitly requested.
- The `cupy` dependency for stream validation is isolated to GPU execution paths;
  CPU-only installations are unaffected.

## Alternatives considered

- **Device-wide synchronize**: simpler, but serializes the entire device and
  defeats the purpose of multi-stream scheduling.
- **Per-call stream argument**: more flexible, but requires changing the
  decorated program call signature and was not needed for the primary use case.

## References

- `PLAN-External_Memory_For_Dace_Arrays.md`
- `dace_stream_spec.md`
- [DaCe issue #2120](https://github.com/spcl/dace/issues/2120)
