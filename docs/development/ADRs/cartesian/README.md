# Cartesian ADRs

This folder contains the _Architecture Decision Records_ for the cartesian part of the codebase. The [top-level README](../README.md) explains when and why we write ADRs.

## How to write ADRs

Writing and reading ADRs should be as simple as possible. Use the [template](../Template.md) as a starting point for new ADRs. Simplify as much as possible. Don't add bloat just to get a longer document. It's not worth it.

## Organization

ADRs about frontend and backend are separated in respective folders. General (e.g. experimental features go in this folder).

```none
.
├── archived/
│   # old ADRs for reference
├── backend/
│   # backend descriptions or details within a particular backend
├── experimental/
│   # experimental features (mostly frontend)
├── frontend/
│   # user-facing features, i.e. `gtscript`
│
└ # ADRs related to general concepts
```

### General topics / concepts

- [Experimental features](./experimental-features.md): what you can (and can't) expect from experimental features.

### Frontend

- [Literal precision](./frontend/literal-precision.md): How to control literal precision if needed
- [Round functions](./frontend/round-functions.md): How `round()` and `round_away_from_zero()` came to be in `gtscript`

#### ⚠️ Experimental features

Remember: [Experimental features](./experimental-features.md) might change at any time without prior warning and generally might not be available in all backends.

- [Absolute indexing in `K`](./frontend/indexing-absolute-k.md): Tradeoffs behind absolute indexation vs. relative offsets

### Backend

Backend ADRs are sorted per backend in the following table:

| Backend name | ADRs                                                                                                                                                                                               |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `debug`      | [Debug backend](./backend/debug.md)                                                                                                                                                                |
| `dace:*`     | [`dace:*` backends](./backend/dace.md) \| [`dace:cpu_kfirst`](./backend/dace-cpu-kfirst.md) <br/> [Schedule tree](./backend/dace-schedule-tree.md) <br/> [DaCe version](./backend/dace-version.md) |
| `cuda`       | [Feature freeze](./backend/cuda-feature-freeze.md)                                                                                                                                                 |
| `numpy`      | (none so far)                                                                                                                                                                                      |
| `gt:*`       | (none so far)                                                                                                                                                                                      |
