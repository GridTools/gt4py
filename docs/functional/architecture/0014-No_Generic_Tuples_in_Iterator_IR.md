---
tags: [iterator]
---

# [No Generic Tuples in Iterator IR]

- **Status**: valid
- **Authors**: Enrique G. Pareded (@egparedes), Felix Thaler (@fthaler), Hannes Vogt (@havogt), Till Ehrengruber (@tehrengruber)
- **Created**: 2022-01-18
- **Updated**: 2022-01-18

We decided that generic tuples (tuples of other elements than values), in particular tuples of iterators, are not desirable in Iterator IR.

## Context

When lowering from Field View it seems desirable to have generic tuples in Iterator IR. Consider the following examples

### Scalar If

```python
if some_condition:
    field_v: Field[Vertex, ...] = v_true
    field_e: Field[Edge, ...] = e_true
else:
    field_v: Field[Vertex, ...] = v_false
    field_e: Field[Edge, ...] = e_false
```

In this example a straight-forward lowering would like to produce

```python
if_(deref(some_condition), make_tuple(v_true, e_true), make_tuple(v_false, e_false))
```
where the arguments to `make_tuple` are iterators.

Note that this cannot be expressed with tuples on values TODO why not like this?
```python
if(deref(some_condition), make_tuple(deref(v_true), deref(e_true)), make_tuple(deref(v_false), deref(e_false)))
```
as the `deref`s can never both point to a valid location.


### Expressing computations with different output location types

TODO

### Tuple on same location

Tuples on same location can be expressed as Iterator of tuples

```python
lift(lambda a,b: return make_tuple(deref(a), deref(b)))
```

## Conclusion

The composability of Iterator IR is focused around liftable stencils. Lifting requires that all input iterators current location types match. Passing tuples of iterators pointing to different locations is not a useful feature in Iterator IR, as this makes the resulting function non-liftable and thus non-composable.

TODO improve this blabla

## Decision

Generic tuples are not a desirable feature at Iterator IR level.

