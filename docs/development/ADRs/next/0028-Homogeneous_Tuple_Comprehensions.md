---
tags: []
---

# Homogeneous Tuple Comprehensions

- **Status**: valid
- **Authors**: Till Ehrengruber (@tehrengruber), Sara Faghih-Naini (@SF-N)
- **Created**: 2026-08-27
- **Updated**: 2026-08-27

In the context of tuple comprehensions in the field-view frontend, facing the constraint that every FOAST node carries exactly one type, we decided to support only homogeneous iterables — all elements of the same type — and reject heterogeneous ones in the type deduction.

## Context

Tuple comprehensions, e.g. `tuple(2.0 * el for el in (a, b))`, are typed and lowered with a single mapper: one target (`el`) and one element expression (`2.0 * el`), shared by all elements of the iterable. In FOAST every node has a single `type` attribute, so the target symbol — and consequently every node in the element expression — can only be typed once. If the iterable's elements had different types, the mapper would need a different type per element, i.e. per-element re-typing (monomorphization) of the element expression, which the FOAST type system does not support.

The same constraint exists at the GTIR level: the ITIR type inference also stores a single type per node (and asserts on conflicting re-assignment), so the single `map_tuple` lambda used to lower comprehensions over variable-length tuples can only have one function type. For variable-length iterables heterogeneity cannot occur in the first place, since `VarArgType` describes all elements with a single element type. Rejecting heterogeneous iterables in the FOAST type deduction just surfaces the error earliest, with a source location.

## Decision

Only homogeneous iterables are supported, both fixed-length and variable-length. Heterogeneous ones are rejected in the type deduction. E.g., with `a`, `b`, `c`, `d` of equal type:

```python
tuple(2.0 * el for el in (a, b))  # supported
tuple(2.0 * el for el in (a(V2E), b(V2E)))  # supported
tuple(local_el + el for local_el, el in ((a(V2E), b), (c(V2E), d)))  # supported
tuple(2.0 * el for el in (a(V2E), b))  # rejected: local vs. non-local element
```

Note that homogeneity applies to the iterable's elements as a whole: in the third example each element is a pair of a local and a non-local field, but all elements share that same tuple type, so each target symbol still has a single consistent type.

## Consequences

- Typing and lowering stay simple: the element expression is visited once, with one type per node.
- Computations over differently-typed elements cannot be written as a comprehension; they must be spelled out per element.
- The restriction could be lifted for fixed-length iterables by typing the mapper generically: the target symbol gets a type variable bounded by the valid element types, so a single type per node still suffices during type deduction. After `map_tuple` expansion the mapper is instantiated once per element, and each instance can then be specialized to its concrete element type. This is possible as a follow-up without breaking existing code, since it only widens the set of accepted programs. For variable-length iterables there is nothing to lift: heterogeneity cannot occur, as `VarArgType` has a single element type by construction.

## References

- PR [#2833](https://github.com/GridTools/gt4py/pull/2833) (tuple comprehension support)
