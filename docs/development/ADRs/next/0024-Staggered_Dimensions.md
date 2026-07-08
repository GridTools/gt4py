---
tags: []
---

# Staggered Dimensions

- **Status**: valid
- **Authors**: Till Ehrengruber (@tehrengruber)
- **Created**: 2026-07-08
- **Updated**: 2026-07-08

A **staggered dimension** is a dimension sitting at the **half-integer**
positions of a base dimension. For example, in a cell-centered 2D Cartesian grid
with cells located at `I`, `J`, the edges are located at `I − ½`, `J` and
`I`, `J − ½`. Differentiating between staggered and non-staggered dimensions
avoids accidental errors when fields defined on different entities are combined,
and improves readability by staying close to the mathematical formulation.

## Context

Without a distinction between an entity and its staggered counterpart, fields
living on, e.g., cells and on edges share the same dimension and can be combined
silently even though they are geometrically different. We want the type system to
keep them apart, while still letting users move between the two with an intuitive,
mathematically-flavoured syntax.

## Indexing convention

A shift by a **half-integer** offset moves between a dimension and its staggered
counterpart; an integer offset shifts within the (non-staggered) dimension as
usual.

Let `i_field` be a 1D field of cell values defined on `I`, and `IHalf` the
corresponding staggered dimension of the edges between those cells. Then:

- `i_field(IHalf + 0.5)` → maps edges to the cell value above: `(i−½)+½ = i`
- `i_field(IHalf − 0.5)` → maps edges to the cell value below: `(i−½)−½ = i−1`

```
           ┊      ●      ┊       ●      ┊      ●      ┊
 I               -1              0             1
 IHalf    -1½           -½             +½            +1½
```

Symmetrically, let `ihalf_field` be defined on `I − ½` (i.e. on `IHalf`). Then:

- `ihalf_field(I + 0.5)` → maps cells to the edge above: `i+½`
- `ihalf_field(I − 0.5)` → maps cells to the edge below: `i−½`

### Storage

`ihalf_field.ndarray[idx]` holds the value for logical index
`i = domain.start + idx`, which is *interpreted* as sitting at `i − ½`. In the
example above, `IHalf(0)` therefore sits at `0 − ½ = −½`, i.e. it is the index of
the edge just below the cell `I(0)`. The memory layout is identical to any normal
field; the "half" is purely how the index is interpreted geometrically.

## Encoding

A staggered dimension is encoded as its base dimension's name with the internal
`_Staggered` prefix (`common._STAGGERED_PREFIX`), rather than as a new attribute
on `Dimension`. The helpers `is_staggered`, `flip_staggered` and
`as_non_staggered` operate purely on that prefix.

Dimensions are identified by their **name** and appear throughout the toolchain
in more than one form: as a `common.Dimension` instance, but also as an
`itir.AxisLiteral` (and as plain strings in generated backend code). Adding a
"staggered" flag to `Dimension` would have required threading it through every one
of these representations and every place that constructs, compares, or lowers a
dimension — a large, invasive change touching the frontend, the IR, and all
backends. Carrying the marker in the name instead means it round-trips through all
representations for free. The leading underscore marks the prefix as internal: it
is an implementation detail, not part of the user-facing API.
