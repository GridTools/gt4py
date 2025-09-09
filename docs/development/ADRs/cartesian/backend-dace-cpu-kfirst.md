# Backend `dace:cpu_kfirst`

In the context of performance optimizations, comparing with other implementations, we decided to add `dace:cpu_kfirst` next to `dace:cpu` (j-first layout) to achieve the same loop structure in the generated code. We considered replacing the default and accept the maintenance overhead of another backend.

## Context

During performance evaluation of the schedule tree based DaCe backend, we noticed that the direct oir -> schedule tree translation leaves us with a K-I-J loop structure (from outside in). This loops structure (and memory layout) is not 1:1 comparable to other implementations (e.g. the existing GridTools backends).

## Decision

We decided to write a pass to optionally change the loop order to be 1:1 comparable to the `gt:cpu_kfirst` backend. Inspired by the GridTools backends, we named that additional backend `dace:cpu_kfirst`.

## Consequences

Power users of GT4Py now have an easy way to run `dace:cpu` with either J-first (as before) or K-first (new with `dac:cpu_kfirst`) loop structure / memory layout.

On disadvantage of this approach is that we add duplicate the dace cpu backend. Duplication of code is minimal since everything is neatly abstracted. We expect and accept longer CI runtimes for those tests that run with all backends.

## Alternatives considered

### Change the loop structure of `dace:cpu`

We could just change the loop structure / memory layout of the existing `dace:cpu` backend to be K-first. This would allow direct comparison with `gt:cpu_kfirst` and avoid the backend duplication. We see loop reordering as an optimization step and would like to keep the naive direct translations (with the K-I-J loops) or comparison.

### Derive loop structure from memory layout

Currently, memory layout and loop structure are two separate parts of the codebase, only loosely linked through "the backend". We though about a bigger refactor to e.g. derive the loop structure from the memory layout or bind them together in other ways.

We realized this challenges some assumptions built deep into GT4Py. This would need careful evaluation with clear goals and use-cases. We thus postpone this approach for a future discussion and went with the existing approach of duplicating the backend.

## References

- Why we have [DaCe backends](./backend-dace.md) in the first place.
- Where the [schedule tree](./backend-dace-schedule-tree.md) comes from.
