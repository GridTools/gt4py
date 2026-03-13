---
tags: []
---

# Limitations of embedded concat_where

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-03-12
- **Updated**: 2026-03-12

In embedded execution, `concat_where` is, for now, limited to simple but common cases.

We do not support `concat_where` in cases

- where the domain would be infinite and therefore can't be represented as an ndarray, e.g. `concat_where(I < 0, 0.0, somefield)` where the scalar 0.0 would be broadcasted to a field reaching to -infinity;
- with multi-dimensional domains, e.g. `concat_where(I > 0 | J > 0, a, b)`. These cases need to be represented by a nested `concat_where(I > 0, a, concat_where(J > 0, a, b))`;

Additionally, we only support the most basic cases of non-consecutive domains: a tuple of `Domain`s resulting from e.g. `I != 0` or the equivalent `I < 0 | I > 0`. Operations on tuples of `Domain`s are not supported.

## Context

`concat_where` requires expressing conditions like `I != i`, which produces two disjoint 1D domains (everything before index `i` and everything after). We need a way to represent these non-consecutive domains.

A complete implementation would require designing how to handle fields on non-hypercubic domains. Currently, `Domain` is a Cartesian product of per-dimension `UnitRange`s, which inherently describes hypercubic (rectangular) regions. Supporting arbitrary non-consecutive domains in multiple dimensions would mean fields could live on non-rectangular regions, requiring fundamental changes to field storage, slicing, and iteration.

## Decision

We use a simple tuple-of-`Domain` representation for non-consecutive domains, restricted to:

- **1D only**: `Domain.__or__` raises `NotImplementedError` for multidimensional domains.
- **At most 2 domains**: `Dimension.__ne__` produces exactly 2 disjoint domains. No attempt is made to support arbitrary numbers of disjoint regions.

This is sufficient for the most common `concat_where` use case (`I != i` splits a dimension into two parts) without requiring a general solution for non-hypercubic domains/fields.

## Consequences

- `concat_where` works for 1D domain conditions, which covers the primary use case of vertical level exclusion.
- Combining multiple exclusions (e.g. `(I != 2) & (I != 5)`) is not supported because it would require a custom tuple type to implement the intersection/union operations.

## Alternatives considered

### General `concat_where` with multi-dimensional domain conditions

Implementation for multi-dimensional domain conditions (e.g. `(I != 2) | (K != 5)`) and full support for domain operations in `concat_where` would require

1. **A `DomainTuple` class** with full algebra: a `tuple` subclass carrying `__and__`, `__or__`, `__rand__`, `__ror__` operators so that expressions like `tuple & Domain`, `Domain & tuple`, and `tuple | tuple` all work.

2. **Normalization of domain tuples**: We need to design `DomainTuple` invariants, e.g.

   - Should all domains be promoted to the same rank (missing dimensions filled with infinite ranges)?
   - Should we reduce overlapping domains to non-overlapping via box subtraction?

Before implementing a complex `DomainTuple`, we should conclude on (if we want) a concept of non-consecutive fields.
