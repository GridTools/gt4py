---
tags: [iterator]
---

# Lifted Stencils with Tuple Return

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2022-02-16
- **Updated**: 2022-02-16

What's the return type of a lifted stencil that has tuple return type? Is it tuple of iterator or iterator of tuple?

## Context

Example

```python
def stencil():
    return 42, 43

# What is the type of
lift(stencil)()
```

## Decision

We return an iterator of tuples.

There are several reasons why this is attractive:

- At least for simple cases, the values returned from a stencil are most likely uniform and need to be treated uniformly, e.g. in a subsequent shift.
- Implementation-wise, it is not straight-forward to implement tuple of iterators. E.g. on the Python-side the return type (tuple vs scalar) of the function is not known when `lift` is applied, i.e. when a lift-iterator needs to be constructed. The solution could be to execute-once arbitrary input then analyze the result (this would be ok, as our language doesn't have side-effects).
- In general, transformations that remove unused elements are easier to implement than transformations that try to match things. See the following example.

### Example of shifting all elements of the return

Let's assume the following stencil

```python
def foo(inp0, inp1, inp2):
    return deref(inp0), deref(inp1), deref(inp2)
```

### Iterator of tuples

```python
def uniform(inp0,inp1,inp2):
    res = lift(foo)(inp0,inp1,inp2)
    shifted = shift(O)(res)
    return deref(shifted)

def non_uniform(inp0,inp1,inp2):
    res_it = lift(foo)(inp0,inp1,inp2)
    shifted_vals_O = deref(shift(O)(res_it)) # everything is shifted
    shifted_vals_L = deref(shift(L)(res_it)) # everything is shifted
    return tuple_get(1, shifted_vals_O), tuple_get(2, shifted_vals_L) # we throw away element 0
```

In a naive (non-optimized) implementation of the `non_uniform` case we pay for 4 extra shifts (ptr updates) that are not needed.

### Tuple of Iterators

```python
def uniform(inp0, inp1, inp2):
    res = lift(foo)(inp0, inp1, inp2)
    shifted0 = shift(O)(tuple_get(0, res))
    shifted1 = shift(O)(tuple_get(1, res))
    shifted2 = shift(O)(tuple_get(2, res))
    return make_tuple(deref(shifted0), deref(shifted1), deref(shifted2))

def non_uniform(inp0, inp1, inp2):
    res = lift(foo)(inp0, inp1, inp2)
    return deref(shift(O)(tuple_get(1, res))),deref(shift(L)(tuple_get(2, res)))
```

In a naive implementation of the `uniform` case we might do extra stride-computations/lookups for each of the shifts.

### Summary

This example illustrates that the selected option (iterator of tuple) is better suited for uniform usage of the result, while tuple of iterators is better suited if parts of the result are not used or used differently.

In the case of iterator of tuple, after inlining in the `non_uniform` case, we can remove the unused elements before applying the shifts and optimize to the optimal implementation.

In the case of tuple of iterator, we have currently no way to express shifting of multiple elements at once.

## References <!-- optional -->

- Related decision taken in [0003-Iterator_View_Tuple_Support_for_Fields](0003-Iterator_View_Tuple_Support_for_Fields.md)
