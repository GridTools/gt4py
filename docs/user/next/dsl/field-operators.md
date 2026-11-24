---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Field operators

A field operator is a Python function decorated with `@gtx.field_operator`. Its
body is written in a subset of Python where values are whole fields: there are
no loops over grid points and no indices. This page describes that subset.

```{code-cell} ipython3
import numpy as np

import gt4py.next as gtx

I = gtx.Dimension("I")
J = gtx.Dimension("J")
```

The examples use these fields:

```{code-cell} ipython3
domain = gtx.domain({I: 4, J: 3})

a = gtx.as_field(domain, np.arange(12.0).reshape(4, 3))
b = gtx.full(domain, 10.0)
result = gtx.zeros(domain)

print(a.asnumpy())
```

## Anatomy of a field operator

A field operator takes fields and scalars, and returns fields:

```{code-cell} ipython3
IJField = gtx.Field[gtx.Dims[I, J], gtx.float64]


@gtx.field_operator
def weighted_mean(a: IJField, b: IJField, weight: gtx.float64) -> IJField:
    return weight * a + (1.0 - weight) * b


weighted_mean(a, b, 0.25, out=result)
print(result.asnumpy())
```

- All the **parameters must have a type annotation**. The annotation of the
  return value is optional, and when present it is checked against the type of
  the returned expression.
- A field operator is a **pure function**: it does not modify its arguments, and
  the result only depends on them.
- The function is analyzed when it is decorated, so mistakes in the definition
  are reported at that point, before running anything.

```{note}
The built-in functions and data types used in the body of an operator must be
**imported by name**, as in `from gt4py.next import where, astype, float32`.
Calling them through the module (`gtx.where(...)`) is not supported inside
operators. We are working on removing this restriction, which will go away
soon. Types in annotations can already be written either way.
```

## Types

### Fields

The type of a field is `gtx.Field[gtx.Dims[...], dtype]`, with its dimensions
and the data type of its values.

```{tip}
Give names to the field types used in a model, as done with `IJField` above.
Signatures become shorter and easier to read.
```

### Scalars

Scalar parameters are annotated with one of the data types:

| Kind            | Types                                                                      |
| --------------- | -------------------------------------------------------------------------- |
| Boolean         | `bool`                                                                     |
| Signed integers | `gtx.int8`, `gtx.int16`, `gtx.int32`, `gtx.int64`, `int` (same as `int64`) |
| Unsigned        | `gtx.uint8`, `gtx.uint16`, `gtx.uint32`, `gtx.uint64`                      |
| Floating point  | `gtx.float32`, `gtx.float64`, `float` (same as `float64`)                  |

A floating-point number written in the code (`2.0`, `1e-3`) has type `float64`.
An integer has type `int32`, or `int64` if it does not fit.

### No implicit conversions

GT4Py never converts between data types on its own. Both sides of an operation
must have the same data type, which means that:

- With `float64` fields, write `2.0 * a`, not `2 * a`.
- With `float32` fields, write `float32(2.0) * a`.
- With `int64` values, including the parameters annotated with `int`, write
  `int64(1)`.
- To combine values of different types, convert one of them with `astype`.

```{code-cell} ipython3
from gt4py.next import astype, float32, float64


@gtx.field_operator
def add_single_precision(
    a: IJField, b: gtx.Field[gtx.Dims[I, J], gtx.float32]
) -> IJField:
    return a + astype(float32(2.0) * b, float64)


b_single = gtx.full(domain, 10.0, dtype=np.float32)
add_single_precision(a, b_single, out=result)
print(result.asnumpy())
```

The conversions are explicit so that the precision of every operation can be read
directly from the code. This matters for models running in single or mixed
precision.

### Tuples

Tuples of fields or scalars can be used as arguments, as local values and as
results. They are annotated with `tuple[...]`:

```{code-cell} ipython3
@gtx.field_operator
def sum_and_difference(a: IJField, b: IJField) -> tuple[IJField, IJField]:
    return a + b, a - b


@gtx.field_operator
def product(a: IJField, b: IJField) -> IJField:
    total, difference = sum_and_difference(a, b)
    return total * difference


product(a, b, out=result)
print(result.asnumpy())
```

This example also shows that a field operator can **call other field
operators**.

A tuple can also be an argument. Its members are unpacked, or accessed with an
index like `vector[0]`, and the operator is called with a regular Python tuple:

```{code-cell} ipython3
@gtx.field_operator
def squared_norm(vector: tuple[IJField, IJField]) -> IJField:
    u, v = vector
    return u * u + v * v


squared_norm((a, b), out=result)
print(result.asnumpy())
```

```{tip}
Build large computations out of small operators. The compiled backends inline
the operators called by another one and optimize the result as a whole.
```

## Expressions

| Operation  | Syntax                                                                  |
| ---------- | ----------------------------------------------------------------------- |
| Arithmetic | `+`, `-`, `*`, `/`, `//`, `%`, `**`, unary `-`                          |
| Comparison | `==`, `!=`, `<`, `<=`, `>`, `>=` (the result is a field of `bool`)      |
| Logical    | `&` (and), <code>\|</code> (or), `^` (xor), `~` (not)                   |
| Premap     | `field(I + 1)`, `field(E2C[0])` (see [Premap and neighbors](premap.md)) |

Operations between fields and scalars apply to every point of the field. When
the fields have different dimensions, the result has all of them: adding a field
over `K` to a field over `I, J` gives a field over `I, J, K`.

`%` is only defined for integers; use `fmod` for floating point values.

```{important}
Use `&` and `|` to combine conditions, and put each comparison between
parentheses: `(a > 0.0) & (a < 1.0)`. The Python keywords `and` and `or`, and
chained comparisons like `0.0 < a < 1.0`, are not supported, not even for
scalars. Use `~` instead of `not` for fields.
```

### Built-in functions

These are the built-in functions, all of them imported from `gt4py.next`:

| Functions                                                            | Description                             |
| -------------------------------------------------------------------- | --------------------------------------- |
| `abs`, `neg`, `sqrt`, `cbrt`, `exp`, `log`, `gamma`, `power`, `fmod` | Elementary functions                    |
| `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`                    | Trigonometric functions                 |
| `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`              | Hyperbolic functions                    |
| `floor`, `ceil`, `trunc`                                             | Rounding                                |
| `minimum`, `maximum`                                                 | Minimum or maximum of two values        |
| `isfinite`, `isinf`, `isnan`                                         | Checks on floating point values         |
| `where(mask, a, b)`                                                  | `a` where `mask` is true, `b` elsewhere |
| `astype(value, dtype)`                                               | Conversion to another data type         |
| `broadcast(value, (I, J))`                                           | Extension of a value to more dimensions |
| `neighbor_sum`, `min_over`, `max_over`                               | [Reductions over neighbors](premap.md)  |

```{code-cell} ipython3
from gt4py.next import maximum, sqrt, where


@gtx.field_operator
def clipped_root(a: IJField, threshold: gtx.float64) -> IJField:
    return where(a > threshold, sqrt(maximum(a - threshold, 0.0)), 0.0)


clipped_root(a, 2.0, out=result)
print(result.asnumpy())
```

## Local variables and conditionals

Intermediate results can be stored in local variables, which helps to write
long expressions in readable steps:

```{code-cell} ipython3
from gt4py.next import abs  # the Python built-in `abs` is not accepted inside operators


@gtx.field_operator
def relative_difference(a: IJField, b: IJField) -> IJField:
    difference = a - b
    scale = maximum(abs(a), abs(b))
    return difference / scale


relative_difference(a, b, out=result)
print(result.asnumpy())
```

There are two ways to choose between values, for two different situations:

- **`where(mask, a, b)`** selects point by point with a field of `bool`. Both
  `a` and `b` are computed at all the points.
- **`if` statements** and the `x if condition else y` expression take a
  **scalar** `bool` condition, so they select the computation for the whole
  field. They are meant for configuration options.

```{code-cell} ipython3
@gtx.field_operator
def limit(a: IJField, upper: gtx.float64, enabled: bool) -> IJField:
    if enabled:
        limited = where(a > upper, upper, a)
    else:
        limited = a
    return limited


limit(a, 6.0, True, out=result)
print(result.asnumpy())
```

A variable assigned in the two branches of an `if` must have the same type in
both of them.

## Values from the enclosing scope

Operators can refer to objects defined outside them: dimensions, other
operators and constants. Scalar constants are grouped in an `Enum` class whose
members also subclass a numeric type, so they behave like plain numbers in
arithmetic:

```{code-cell} ipython3
import enum


class Constants(np.float64, enum.Enum):
    GRAVITY = 9.80665
    EARTH_RADIUS = 6.371e6


@gtx.field_operator
def geopotential(height: IJField) -> IJField:
    return Constants.GRAVITY * height


geopotential(a, out=result)
print(result.asnumpy())
```

```{important}
The compiled backends **inline the values of the constants** in the generated
code when the operator is compiled. This is why they have to be immutable: a
plain global variable (`GRAVITY = 9.80665`) is only accepted in embedded
execution, and the compiled backends reject it.
```

```{tip}
Use this only for true constants. A value which may change during the execution
of the application must be passed as an argument.
```

## Named collections

Models usually handle groups of related fields, like the components of the wind
or the state of the atmosphere. Instead of long lists of arguments or anonymous
tuples, operators can take and return **named collections**: a
`typing.NamedTuple`, or a dataclass, whose members are fields, scalars or other
collections.

```{code-cell} ipython3
import typing


class Wind(typing.NamedTuple):
    u: IJField
    v: IJField


@gtx.field_operator
def kinetic_energy(wind: Wind) -> IJField:
    return 0.5 * (wind.u * wind.u + wind.v * wind.v)


@gtx.field_operator
def scaled(wind: Wind, factor: gtx.float64) -> Wind:
    return Wind(u=factor * wind.u, v=factor * wind.v)


wind = Wind(u=a, v=b)

kinetic_energy(wind, out=result)
print(result.asnumpy())
```

When the result is a collection, `out` is a collection of the same type:

```{code-cell} ipython3
half_wind = Wind(u=gtx.zeros(domain), v=gtx.zeros(domain))

scaled(wind, 0.5, out=half_wind)
print(half_wind.u.asnumpy())
```

Dataclasses work in the same way:

```{code-cell} ipython3
import dataclasses


@dataclasses.dataclass
class State:
    temperature: IJField
    pressure: IJField


@gtx.field_operator
def heated(state: State, increment: gtx.float64) -> State:
    return State(temperature=state.temperature + increment, pressure=state.pressure)


new_state = State(temperature=gtx.zeros(domain), pressure=gtx.zeros(domain))

heated(State(temperature=a, pressure=b), 0.5, out=new_state)
print(new_state.temperature.asnumpy())
```

The rules are:

- The class must be defined at module level, not inside a function.
- Dataclasses are accepted when all their members are set by the constructor and
  none of them has a default value.

```{note}
Support for other kinds of static containers (_pytrees_) will probably be added
in the future.
```

## Unsupported Python, and what to use instead

The body of an operator is not arbitrary Python. These are the most common
constructs which are rejected, and what to use instead:

| Not supported                           | Use instead                                                         |
| --------------------------------------- | ------------------------------------------------------------------- |
| `for` and `while` loops                 | Field expressions, or a [scan operator](scan-operators.md)          |
| `and`, `or`, chained comparisons        | `&`, <code>\|</code> with parentheses                               |
| `if` with a field as condition          | `where` or [`concat_where`](premap.md#boundaries-with-concat-where) |
| Indexing a field, `a[i, j]`             | Premap, `a(I + 1)`                                                  |
| Mixing data types, `2 * a`              | Literals of the right type (`2.0 * a`) or `astype`                  |
| Modifying an argument in place          | Return a new field                                                  |
| `lambda`, comprehensions, `try`, `with` | A separate field operator, or regular Python code outside of it     |
| Calling NumPy or other Python functions | The built-in functions, or compute the value before the call        |

In all these cases the error is reported when the operator is defined, before
anything runs.
