# Backend `dace:cpu_KJI`

In the context of performance optimizations, we decided to add `dace:cpu_KJI` next to `dace:cpu_kfirst` and `dace:cpu` (j-first layout) to allow a no-copy streaming of memory from Fortran. We considered replacing the others and accept the maintenance overhead of another backend.

## Context

During performance optimization work, we discovered that there's currently no way to stream memory directly (i.e. without copy) from Fortran into a DaCe-based backend. For hybrid Fortran/GT4Py applications where we have no control over the memory layout of the data, this can be a bottleneck.

## Decision

We decided to add a backend that allows directly streaming memory from Fortran into GT4Py stencils (and back) without any copies. To communicate the expected device and memory layout, we named the backend `dace:cpu_KJI`. We chose to keep `cpu` as part of the name because we do not see the potential use-case of using such a poorly-fitting layout to do computation on GPU ever.

## Consequences

Power users of GT4Py have now a way to stream memory directly from (and back to) Fortran without a copy.

To help with the growing number of DaCe-based backends, we introduce an automatic match for DaCe-based backends between layout and schedule. This makes the default scheduling always cache-optimal by construction and supersedes a previous rubber band fix introduced with the `dace:cpu_kfirst` backend.

To be clear, while we add yet another backend, the maintenance overhead of this backend is very low. All dace cpu backends only differ in layout and with the automatic layout/schedule matching described above, neither no extra maintenance overhead occurs. And with the new way of describing backends this also does not add significant cognitive load to have to keep more backends in mind. If anything, we feel that spelling out all cartesian axes in the backend lightens the cognitive load.

## Alternatives considered

### Change one of the existing backends

Both previously existing backends (`dace:cpu` and `dace:cpu_kfirst`) keep their validity and serve different purposes. We thus opted for an additional backend instead of changing an existing one.

## References

- Why we have [DaCe backends](./dace.md) in the first place.
