---
tags: []
---

# Limitations of embedded concat_where

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-03-12
- **Updated**: 2026-03-17

In embedded execution, `concat_where` is, for now, limited to simple but common cases.

We do not support `concat_where` in cases

- where the domain would be infinite and therefore can't be represented as an ndarray, e.g. `concat_where(I < 0, 0.0, somefield)` where the scalar 0.0 would be broadcasted to a field reaching to -infinity;
- with multi-dimensional domains, e.g. `concat_where(I > 0 | J > 0, a, b)`. These cases need to be represented by a nested `concat_where(I > 0, a, concat_where(J > 0, a, b))`;
- with non-consecutive (disjoint) domain conditions, e.g. `concat_where(I != 0, a, b)`. These cases need to be expressed using nested `concat_where`, e.g. `concat_where(I < 0, a, concat_where(I > 0, a, b))`.

## Context

`concat_where` requires expressing conditions like `I != i`, which would produce two disjoint 1D domains (everything before index `i` and everything after). We need a way to represent these non-consecutive domains.

A complete implementation would require designing how to handle fields on non-hypercubic domains. Currently, `Domain` is a Cartesian product of per-dimension `UnitRange`s, which inherently describes hypercubic (rectangular) regions. Supporting arbitrary non-consecutive domains in multiple dimensions would mean fields could live on non-rectangular regions, requiring fundamental changes to field storage, slicing, and iteration.

## Decision

Non-consecutive (disjoint) domains are **not supported** in the domain expression API:

- `Dimension.__ne__(value)` raises `NotImplementedError` when called with an integer value, since it would produce two disjoint domains.
- `Domain.__or__` raises `NotImplementedError` for both multidimensional domains and for 1D domains that are disjoint (non-overlapping and non-adjacent).

The domain expression API only supports operations that result in a single contiguous `Domain`.

## Consequences

- `concat_where` with `I != i` must be rewritten as `concat_where(I < i, ..., concat_where(I > i, ..., ...))`.
- This keeps the domain expression API simple: all supported operations return a single `Domain`.

## Alternatives considered

### General `concat_where` with multi-dimensional domain conditions

Implementation for multi-dimensional domain conditions (e.g. `(I != 2) | (K != 5)`) and full support for domain operations in `concat_where` would require

1. **A `DomainTuple` class** with full algebra: a `tuple` subclass carrying `__and__`, `__or__`, `__rand__`, `__ror__` operators so that expressions like `tuple & Domain`, `Domain & tuple`, and `tuple | tuple` all work.

2. **Normalization of domain tuples**: We need to design `DomainTuple` invariants, e.g.

   - Should all domains be promoted to the same rank (missing dimensions filled with infinite ranges)?
   - Should we reduce overlapping domains to non-overlapping via box subtraction?

Before implementing a complex `DomainTuple`, we should conclude on (if we want) a concept of non-consecutive fields.
