---
tags: [frontend]
---

# Scalar vs 0d-fields

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2022-10-03
- **Updated**: 2022-10-03

Do we distinguish scalars (e.g. `float`) and fields with 0 dimensions (e.g. `Field[[], float]`) in the context of field view?

## Context

In the context of multi-dimensional fields, the question arises if we need scalars **and** 0-d fields, only one of them or even make them the same thing. Similar discussions happened within the Numpy community, see e.g. https://mail.python.org/pipermail/numpy-discussion/2006-February/006384.html.

## Decision

We decided to have both scalars and 0d-fields. They are not treated as alias, but have their distinct properties. Python typing semantics makes it difficult to have a nice solution and we want to comply with Python semantics for typing, especially for embedded execution.

### Examples

#### Wrong annotation for return type

The following is forbidden because the return type annotation doesn't match the type of `1.0` (the correct annotation would be `float`).
Note: the annotation (by Python design) is for the function that is being decorated, not for the function after the decorator is applied, see also this discussion https://github.com/python/typing/issues/412.

```python=
@field_operator
def foo() -> Field[[], float]
    return 1.0
```

An explicit broadcast would be required to make this correct.

```python=
@field_operator
def foo() -> Field[[], float]
    return broadcast(1.0, ())
```

#### Implicit broadcast to Field

In the following call an implicit broadcast to Field is performed in the decorator (the annotation influences behavior).

```python=
@field_operator
def foo(scalar: Field[[], float]) -> Field[[], float]:
    return scalar

foo(1.0)
```

Note that the following is also valid as no broadcasting to `Field` is applied.

```python=
@field_operator
def foo(scalar: float) -> float:
    return scalar

foo(1.0)
```

This enables propagating scalars for cases where a scalars are required.

#### 0-d fields cannot be used where a scalar is expected

The following is forbidden because ternaries only work with scalar conditions.

```python=
@field_operator
def foo(cond: Field[[], int]):
    return some_field_op() if cond == 1 else some_other_field_op()
```

If needed, we might add functionality to extract the scalar value from a 0-d field.

### Notes on lowering

This decision doesn't change the current definition of iterator ir or the `fn` C++ backend:

- a stencil (any lowered field_operator) will only accept iterators (i.e. `deref` is required);
- `fn` will accept SIDs, scalars have to we wrapped in a helper, e.g. `global_parameter`.

## Alternatives Considered

### Scalar `T` and 0-d field `Field[[], T]` are aliases for each other

Making scalars and 0-d field the same (aliases for each other) simplifies usage of the language in many places.

Example: A reduction over a dimension of a field returns a field with one dimension less. Reducing over a 1-d field will therefore return a 0-d field, but that value might be needed in a place where a scalar is expected.

However, we would need implement it with either possibly strange side-effects (making builtin types, e.g. `float` a subclass of Field) or create custom types for scalars (which is not desirable from a usage perspective, e.g. `gt4py.float`) to make this properly work with Python typing.

It would allow to make some cases work that are forbidden by the now decided rules, e.g.

```python=
@field_operator
def foo() -> Field[[], float]
    return 1.0
```

would be correct, e.g. if we implement `Field[[], float]` as a protocol that matches `1.0`.

If we implement our own types, we have full freedom, to really make them aliases and get the following semantic work

```python=
scalar: gt4py.float(4.0)
assert isinstance(scalar, gt4py.float)
assert isinstance(scalar, Field[[], gt4py.float])
assert scalar.dtype == gt4py.float
assert scalar.dtype == Field[[], gt4py.float]
assert scalar.dtype.dtype.dtype.dtype == gt4py.float
assert scalar.dtype.dtype.dtype.dtype == Field[[], gt4py.float]
```

### 0-d fields are regular fields implementing scalar conversion operators

In this proposal 0-d fields are regular fields which additionally implement the appropriate scalar conversion operator matching its datatype (`__int__` for integer types, `__float__` for floating point types, `__bool__` for booleans and `__complex__` for complex types). This would allow to use 0-d fields in places where Python require scalars (e.g. `if` conditions, sequence indexing).

```python
a_0d_field = field(2.3)

assert isinstance(a_0d_field, Field)

if a_0d_field > 0.5:
    print("No problem!")
```

This is essentially the same approach used in the [Python array API standard](https://data-apis.org/array-api/latest/API_specification/array_object.html) where all methods dealing with arrays return arrays, and 0-d arrays support the conversion to scalars using standard _dunder_ methods. Additionally, this alternative is fully compatible with the implementation of our own custom scalar types, although it doesn't require it.

In the long-term, this alternative seems to combine the best features from other options, but it requires a bit more discussion and a slightly larger implementation effort, so it has been postponed for now.

## References <!-- optional -->

- [\[Numpy-discussion\] A case for rank-0 arrays](https://mail.python.org/pipermail/numpy-discussion/2006-February/006384.html)
- [Python - annotating the decorated function](https://github.com/python/typing/issues/412)
- [Python array API standard - Array object](https://data-apis.org/array-api/latest/API_specification/array_object.html)
- [GT4Py discussion and notes (HackMD)](https://hackmd.io/@gridtools/SyQ4vJ9Js)
