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

# Scan operators

In a field operator, the value at a point is computed only from input values,
never from results at other points, so all the points can be computed in any
order. Some computations do not fit this model: integrating along a column,
solving a tridiagonal system, or precipitation falling from one level to the
next. In these, the value at a level depends on the _result_ of the previous
level.

A **scan operator** expresses this kind of dependency along the vertical
dimension. It replaces a loop over the levels.

```{code-cell} ipython3
import numpy as np

import gt4py.next as gtx

I = gtx.Dimension("I")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)

KField = gtx.Field[gtx.Dims[K], gtx.float64]
IKField = gtx.Field[gtx.Dims[I, K], gtx.float64]
```

## A first scan operator

A scan operator is a function describing **one step** of the iteration, the
computation done at a single level. The following one computes a cumulative sum:

```{code-cell} ipython3
@gtx.scan_operator(axis=K, forward=True, init=0.0)
def cumulative_sum(state: float, value: float) -> float:
    return state + value
```

- The first parameter, the **state**, receives the result of the previous step.
  At the first step of the iteration it receives the value of `init`.
- The other parameters receive the values of the arguments **at the current
  level**. Although the operator is called with fields, inside the function
  they are scalars, annotated here with `float` (which means `float64`).
- The returned value is the result at the current level, and it becomes the
  state of the next step.

Applied to a field, the operator returns a field with the value computed at
each level:

```{code-cell} ipython3
values = gtx.as_field({K: 5}, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
result = gtx.zeros({K: 5})

cumulative_sum(values, out=result)
print(result.asnumpy())
```

The decorator takes three arguments:

| Argument  | Meaning                                                                                                                                                                                |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `axis`    | Dimension to iterate over. It must be a `VERTICAL` dimension                                                                                                                           |
| `forward` | `True` (the default) to go from the first level to the last one, `False` for the opposite                                                                                              |
| `init`    | Value of the state at the first step (`0.0` by default). It must have exactly the type of the state: `0.0` for `float`, `gtx.float32(0.0)` for `gtx.float32`, `gtx.int64(0)` for `int` |

The iteration covers the levels being computed. If the call has a `domain`
argument selecting a range of levels, the first step is the first level of that
range, or the last one with `forward=False`.

With `forward=False` the iteration starts at the last level:

```{code-cell} ipython3
@gtx.scan_operator(axis=K, forward=False, init=0.0)
def reversed_cumulative_sum(state: float, value: float) -> float:
    return state + value


reversed_cumulative_sum(values, out=result)
print(result.asnumpy())
```

## Using scan operators in field operators

A scan operator is called from a field operator like any other operator. Fields
with horizontal dimensions are processed column by column, and every column is
independent from the others. A scalar argument has the same value at every
level.

```{code-cell} ipython3
@gtx.field_operator
def running_mean(values: IKField, weights: IKField) -> IKField:
    return cumulative_sum(weights * values) / cumulative_sum(weights)


values_ik = gtx.as_field({I: 2, K: 5}, np.arange(10.0).reshape(2, 5))
weights_ik = gtx.ones({I: 2, K: 5})
result_ik = gtx.zeros({I: 2, K: 5})

running_mean(values_ik, weights_ik, out=result_ik)
print(result_ik.asnumpy())
```

```{tip}
Keep scan operators small: put in them only the part of the computation that
depends on the previous level, and do everything else in field operators.
```

A scan operator is a function of scalars, not of fields, so it cannot access
the neighbors of a point. If the value of a neighbor is needed, compute it in
the field operator and pass it as one more argument. For example, in a field
operator with `values: IKField`, `some_scan(values, values(I + 1))`.

## States with several values

When several quantities are carried from level to level, the state needs to be
a tuple or, better, a
[named collection](field-operators.md#named-collections) of scalars. `init` and
the returned value must have the same structure:

```{code-cell} ipython3
import typing


class RunningStats(typing.NamedTuple):
    total: float
    largest: float


@gtx.scan_operator(axis=K, forward=True, init=RunningStats(total=0.0, largest=0.0))
def running_stats(state: RunningStats, value: float) -> RunningStats:
    return RunningStats(
        total=state.total + value,
        largest=value if value > state.largest else state.largest,
    )


@gtx.field_operator
def below_running_maximum(values: KField) -> KField:
    stats = running_stats(values)
    return stats.largest - values


levels = gtx.as_field({K: 5}, np.array([1.0, 3.0, 2.0, 5.0, 4.0]))

below_running_maximum(levels, out=result)
print(result.asnumpy())
```

The result is zero at the levels where a new maximum is reached, and the
distance to the largest value seen so far at the others. The values are assumed
to be non-negative, because `largest` starts at `0.0`.

Because the arguments are scalars, the body can use regular `if` statements and
conditional expressions, as in
`value if value > state.largest else state.largest` above. Use them instead of
`where`, which needs fields.

The scan stores the state at every level, so the result of `running_stats` is a
`RunningStats` of two fields. The field operator that calls it picks the
members it needs, like `stats.largest` above.

## Rules

- The first parameter is the state. Its type, the type of `init` and the type of
  the returned value must be the same.
- All the parameters are scalars, or named collections or tuples of scalars,
  and they must be annotated.
- Call a scan operator with at least one field argument, which defines the
  columns it runs on. Calls with only scalar arguments are not supported.
- The body follows the rules of the body of a
  [field operator](field-operators.md), with one difference: the values are
  scalars. Select with `if` or `x if condition else y`, not with `where`, and
  do not use the built-ins that act on fields (`where`, `broadcast`, the
  reductions).
