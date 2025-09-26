# Cartesian ADRs

This folder contains the _Architecture Decision Records_ for the cartesian part of the codebase. The [top-level README](../README.md) explains when and why we write ADRs.

## How to write ADRs

Writing and reading ADRs should be as simple as possible. Use the [template](../Template.md) as a starting point for new ADRs. Simplify as much as possible. Don't add bloat just to get a longer document. It's not worth it.

## Organization

In general, ADRs in this folder are prefixed with either

- `general-` for general topics / concepts,
- `frontend-` for user-facing features (in `gtscript`),
- `backend-` for backend descriptions or details within a particular backend.

### General topics / concepts

- [Experimental features](./general-experimental-features.md): what you can (and can't) expect from experimental features.

### Frontend

- [Literal precision](./frontend-literal-precision.md): How to control literal precision if needed
- [Round functions](./frontend-round-functions.md): How `round()` and `round_away_from_zero()` came to be in `gtscript`

#### ⚠️ Experimental features

Remember: [Experimental features](./general-experimental-features.md) might change at any time without prior warning and generally might not be available in all backends.

- [Absolute indexing in `K`](./frontend-indexing-absolute-k.md): Tradeoffs behind absolute indexation vs. relative offsets

### Backend

Backend ADRs are sorted per backend in the following table:

| Backend name | ADRs                                                                                                                                                                                               |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `debug`      | [Debug backend](./backend-debug.md)                                                                                                                                                                |
| `dace:*`     | [`dace:*` backends](./backend-dace.md) \| [`dace:cpu_kfirst`](./backend-dace-cpu-kfirst.md) <br/> [Schedule tree](./backend-dace-schedule-tree.md) <br/> [DaCe version](./backend-dace-version.md) |
| `cuda`       | [Feature freeze](./backend-cuda-feature-freeze.md)                                                                                                                                                 |
| `numpy`      | (none so far)                                                                                                                                                                                      |
| `gt:*`       | (none so far)                                                                                                                                                                                      |
