---
tags: [iterator]
---

# Iterator View Tuple Support for Fields

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2022-02-16
- **Updated**: 2022-02-16

This document covers how fields of tuples and tuples of fields are treated at the interface level (closure/apply_stencil).

## Output fields

### Context

A stencil in iterator view has a single return value, which can be a tuple. How do we map a tuple return value to field(s)?

If the return value of a stencil is a tuple there are 2 possible relations between these values:

- they are dimensions of the same kind (e.g. the 3 components of wind velocity),
- they are of different kind, i.e. 2 completely unrelated quantities that just happen to be computed in the same stencil (e.g. temperature and pressure).

The natural representation of the 2 cases are:

- a single field of tuples,
- a tuple of fields.

### Decision

Iterator view does not distinguish tuple of fields or field of tuples as output of stencils.

## Input fields

### Context

Fields of tuples are obvious: they are treated as iterators of tuples.

How should tuples of fields be treated? Unlike for output, we have the option to pass multiple inputs. Therefore, the only reason to combine fields to a tuple is that they are related and should be treated uniformly within the stencil.

### Decision

We don't distinguish between field of tuple and tuple of fields. They are both treated as iterator of tuple.

### Alternatives

We could represent tuple of fields as tuple of iterators, but then we cannot shift the components in a single shift call, instead it would look like

```python
make_tuple(shift(offset)(tuple_get(0,inp)), shift(offset)(tuple_get(1,inp)), ...)
```

This would only make sense if each component needs a different shift, however then most likely the input should be represented as separate fields, not as tuple.
