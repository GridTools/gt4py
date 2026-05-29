---
tags: []
---

# [Connectivities]

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2024-11-08
- **Updated**: 2026-05-27

The representation of Connectivities (neighbor tables, `NeighborTableOffsetProvider`) and their identifier (offset tag, `FieldOffset`, etc.) was extended and modified based on the needs of different parts of the toolchain. Here we outline the ideas for consolidating the different closely-related concepts.

## History

In the early days of Iterator IR (ITIR), an `offset` was a literal in the IR. Its meaning was only provided at execution time by a mapping from `offset` tag to an entity that we labelled `OffsetProvider`. We had mainly 2 kinds of `OffsetProvider`: a `Dimension` representing a Cartesian shift and a `NeighborTableOffsetProvider` for unstructured shifts. Since the type of `offset` needs to be known for compilation (strided for Cartesian, lookup-table for unstructured), this prevents a clean interface for ahead-of-time compilation.
For the frontend type-checking we later introduce a `FieldOffset` which contained type information of the mapped dimensions.
For (field-view) embedded we introduced a `ConnectivityField` (now `Connectivity`) which could be generated from the OffsetProvider information.

These different concepts had overlap but were not 1-to-1 replacements.

## Decision

We update and introduce the following concepts

### Conceptual definitions

**Connectivity** is a mapping from index (or product of indices) to index. It covers 1-to-1 mappings, e.g. Cartesian shifts, NeighborTables (2D mappings) and dynamic Cartesian shifts.

**GatherConnectivity** is a _Connectivity_ whose `premap` rearranges data via advanced indexing (its index map is materializable as an integer `ndarray`). Cartesian shifts and relocations are _not_ `GatherConnectivity`s: their `premap` only relabels the domain (no data movement). Embedded field-view `premap` dispatches on this distinction.

**NeighborTable** is a _GatherConnectivity_ that is a 2D mapping of the N neighbors of a Location A to a Location B, backed by a buffer.

**ConnectivityType**, **NeighborConnectivityType** contains all information that is needed for compilation.

### Full definitions

See `next.common` module

Note: Currently, the compiled backends support only `NeighborTable`s. If a non-buffer-backed neighbor connectivity (e.g. `StridedNeighborConnectivity`) is ever needed, the `NeighborTable` concept would have to be generalized again.

## Which parts of the toolchain use which concept?

### Embedded

Embedded execution of field-view supports any kind of `Connectivity`.
Embedded execution of iterator (local) view supports only `NeighborTable`s.

### IR transformations and compiled backends

All transformations and code-generation should use `ConnectivityType`, not the `Connectivity` which contains the runtime mapping.

Note, currently the `global_tmps` pass uses runtime information, therefore this is not strictly enforced.

The only supported `Connectivity`s in compiled backends (currently) are `NeighborTable`s.

## Changelog

### 2026-05-27

- Removed the abstract `NeighborConnectivity` concept; `NeighborTable` is now the single neighbor-connectivity concept (there is no non-buffer-backed neighbor connectivity in use).
- Added `GatherConnectivity` (a `Connectivity` whose `premap` rearranges data via a gather), which the embedded field-view `premap` dispatches on. It replaces the former `ConnectivityKind` flag and unifies the previous reshuffling/remapping `premap` implementations into a single advanced-index gather.
