# Cuda backend

Update: g4py version 1.0.9 was the last one to contain the `cuda` backend. It was removed afterwards and the ADR got moved here.

Update: the `cuda` backend was officially deprecated with gt4py version 1.0.4

## Cuda backend: feature freeze

In the context of (backend) feature development, facing maintainability/duplication concerns, we decided to put a feature freeze on the `cuda` backend and focus on the `dace:gpu` backends instead to keep the number of backends manageable.

## Context

The introduction of the [`dace:*`](./dace.md) backends brought up the question of backend redundancy. In particular, it seems that `cuda` and `dace:gpu` backends serve similar purposes.

`dace:gpu` backends not only generate code for different graphics cards, they also share substantial code paths with the `dace:cpu` backend. This simplifies (backend) feature development.

## Decision

We decided to put a feature freeze on the `cuda` backend, focusing on the `dace:*` backends instead. While we don't drop the backend, new DSL features won't be available in the `cuda` backend. New features will error out cleanly and suggest to use the `dace:gpu` backend instead.

## Consequences

While the `cuda` backend only targets NVIDIA cards, the `dace:*` backends allow to generate code for NVIDIA and AMD graphics cards. Furthermore, `dace:cpu` and `dace:gpu` backends share large parts of the transpilation layers because code generation is deferred to DaCe and only depending on the SDFG. This allows us to develop many (backend) features for the `dace:*` backends in one place.
