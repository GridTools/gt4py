---
tags: []
---

# [Connectivities]

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2024-11-08
- **Updated**: 2024-11-08

The representation of Connectivities (neighbor tables, NeighborTableOffsetProvider) and their identifier (offset tag, FieldOffset, etc) was extended and modified based on the needs of different parts of the toolchain. Here we outline the ideas for consolidating the different closely-related concepts.

## History

In the early days of itir, a `offset` was a literal in the IR. Its meaning was only provided at execution time by a mapping from `offset` tag to an entity that we labelled `OffsetProvider`. We had mainly 2 kinds of `OffsetProvider`: a `Dimension` representing a Cartesian shift and a `NeighborTableOffsetProvider` for unstructured shifts.
For the frontend type-checking we later introduce a `FieldOffset` which contained type information of the mapped dimensions.
For (fieldview) embedded we introduced a `ConnectivityField` which could be generated from the OffsetProvider information.

These different concepts had overlap but were not 1-to-1 replacements.

## Definitions

### Conceptual definitions

**Connection** is a mapping from index to index, possibly defined on different spaces `m: A -> B`, e.g. mapping a vertex index to a cell index, or an I index to another I index.
**Connectivity** is a mapping from an index (neighbor) to a _Connection_ c: N -> A -> B = N -> m, or alternatively interpretable as a 2-D mapping of a (neighbor, position) to an index.

**NeighborTable** is a _Connectivity_ backed by a buffer.

**ConnectivityField** is a _Connectivity_ with additional operations e.g. addition. Note: this concept seems wrong.

**ConnectivityType**, **NeighborTableType** contains all information that is needed for compilation.

### Full definitions

```python
class Connectivity(Protocol):
    domain: Domain
    codomain: Dimension
    def __getitem__(self, index: tuple[NeighborIndex, Index]) -> OtherIndex: ...
    def inverse_image(...): ...
    def __gt_type__(self) -> ConnectivityType: ...


class ConnectivityType(TypeSpec):
    domain_dims: tuple[LocalDimension, Dimension]
    codomain: Dimension

class NeighborConnectivity(Connectivity, Protocol):
    domain: Domain[LocalDimension, Dimension]

@dataclass
class NeighborConnectivityType(ConnectivityType): # Should be property of the LocalDimension
    has_skip_value: bool
    max_neighbors: int
```

```python
class LocalDimensionType:
    has_skip_value: bool
    max_neighbors: int

@field_operator(meta={E2V: mesh["E2V"]})
def foo(e2v: Connectivity[[Edge, E2V], Vertex]):
    ...
```

```python
class NeighborTable(NeighborConnectivity, NDArrayConnectivityField):
    ...
```

```python
ConnectivityField = Connectivity
```

TODO: where does the inverse image live?

## Which parts of the toolchain use which concept?

## Decision

We keep/introduce the following concepts:

### ConnectivityType

```python
class ConnectivityType(Protocol):
    domain_dims: tuple[Dimension, Dimension]
    codomain: Dimension
    skip_value: Optional[bool]
    ...
```

Note: This type describes only `Connectivity` not `Connection`, note the `common.CartesianConnectivity` is actually a connection, therefore doesn't have a `ConnectivityType`. Further concept cleanup is required here!

### NeighborTable

```python
class NeigborTable:
    type_: ConnectivityType
    ndarray
```

### ConnectivityField

## Future

In the PR with this ADR, we make a step into the following direction.

The IR should contain all information that is required to compile a program. Specifically it should contain the `ConnectivityType`.

## Consequences

[ What becomes easier or more difficult to do because of this change?

Describe both the positive (e.g., improvement of quality attribute satisfaction, follow-up decisions required, ...) and negative (e.g., compromising quality attribute, follow-up decisions required, ...)] outcomes of this decision. ]

## Alternatives Considered

### [option 1]

[example | description | pointer to more information | ...] <!-- optional -->

- Good, because [argument a]
- Bad, because [argument c]
- ...

### [option 2]

[example | description | pointer to more information | ...] <!-- optional -->

- Good, because [argument a]
- Bad, because [argument c]
- ...

## References <!-- optional -->

- [Someone - Something](https://someone.com/a-title-about-something)
- ...
