---
tags: []
---

# [Mapping Dimensions to the C++-Backend]

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2022-06-29
- **Updated**: 2023-11-08

This document proposes a (temporary) solution for mapping domain dimensions to field dimensions.

> [!NOTE]
> This ADR was written before the integration of `gt4py.storage` into `gt4py.next`, so the example is using `np_as_located_field` (now deprecated) instead of `gtx.as_field.partial`. The idea conveyed by the example remains unchanged.

## Context

The Python embedded execution for Iterator IR keeps track of the current location type of an iterator (allows safety checks) while the C++ backend does not.

### Python side

On the Python side, we label dimensions of fields with the location type, e.g. `Edge` or `Vertex`. The domain uses `named_ranges` that uses the same location types to express _where_ to iterate, e.g. `named_range(Vertex, range(0, 100))` is an iteration over the `Vertex` dimension, no order in the domain is required. Additionally, the `Connectivity` describes the mapping between location types.

### C++ side

On the C++ side, the `unstructured_domain` is described by two (size, offset)-pairs, the first is the horizontal/unstructured dimension, the second is the vertical/structured dimension. The fields need to be tagged by `dim::horizontal` (alias for `integral_constant<int, 0>`) and `dim::vertical` (alias for `integral_constant<int, 1>`). If a shift is with a `Connectivity`, it is applied to the dimension tagged as `dim::horizontal`, no checking is performed.
The `cartesian_domain` takes a `hymap` from dimension tags to (size, offset)-pairs, Cartesian shifts assume that the offset tag and the dimension tag are the same.

### Problems

- How can we map from the unordered domain on the Python side to the ordered `unstructured_domain` on the C++ side?
- How can we map from user-defined dimensions labels (`Edge`, `Foo`, etc) to `dim::horizontal` in the unstructured case?
- For Cartesian we need a mapping of offset to dimensions.

## Decision

### Mapping the Domain Arguments

We decide, for `unstructured_domain` order matters.

### Mapping User-Defined Labels to `dim::horizontal`

We introduce tags in the `Dimension` object on the Python-side, the bindings need to consume these tags for remapping.

Example (this is implemented in the current PR, but could be done differently in the future)

```python
class Dimension:
    value: str
    kind: DimensionKind

# the following field doesn't have correct default order, as without remapping we would interpret first dimension as dim::horizontal
np_as_located_field(Dimension("K", DimensionKind.VERTICAL), Dimension("Vertex", DimensionKind.HORIZONTAL))
```

### Mapping of Cartesian Offsets to Dimensions

When lowering from iterator IR to gtfn_ir, we replace Cartesian offset tags by Dimension tags.

## Alternatives Considered

### C++ backend could track locations

This was implemented, but was discarded to simplify the C++ backend.

### Embedded execution could implement the C++ execution strategy

We would give up on several safety features if we don't track locations, therefore we didn't consider this direction.
