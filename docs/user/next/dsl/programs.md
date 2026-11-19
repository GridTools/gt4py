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

# Programs

Field operators are pure functions: they compute new fields from their
arguments. A **program** is where results are written into existing fields. It
is a function decorated with `@gtx.program`, containing a sequence of calls to
operators, each one with the field receiving its result and, optionally, the
region where it is computed.

Programs are the natural entry points of an application: a time step of a model,
or one of its components.

```{code-cell} ipython3
import numpy as np

import gt4py.next as gtx

I = gtx.Dimension("I")
J = gtx.Dimension("J")

IJField = gtx.Field[gtx.Dims[I, J], gtx.float64]


@gtx.field_operator
def laplacian(phi: IJField) -> IJField:
    return phi(I + 1) + phi(I - 1) + phi(J + 1) + phi(J - 1) - 4.0 * phi


@gtx.field_operator
def diffuse(phi: IJField, laplacian_of_phi: IJField, coefficient: gtx.float64) -> IJField:
    return phi + coefficient * laplacian_of_phi
```

## Defining and calling a program

```{code-cell} ipython3
@gtx.program
def diffusion_step(
    phi: IJField,
    coefficient: gtx.float64,
    nx: gtx.int32,
    ny: gtx.int32,
    work: IJField,
    new_phi: IJField,
):
    laplacian(phi, out=work, domain={I: (1, nx - 1), J: (1, ny - 1)})
    diffuse(phi, work, coefficient, out=new_phi, domain={I: (1, nx - 1), J: (1, ny - 1)})
```

- The parameters are annotated like in field operators. Input and output fields
  are all regular parameters.
- Every statement is a call to a field operator or a scan operator, with the
  destination of its result in `out`.
- `domain` gives the points to compute, as a range of indices for each
  dimension. The bounds can be expressions with the scalar parameters of the
  program, which avoids fixing the size of the grid in the code.
- A program does not return anything.

A program is called like a function:

```{code-cell} ipython3
nx, ny = 6, 5
grid = gtx.domain({I: nx, J: ny})

data = np.zeros((nx, ny))
data[2:4, 2] = 16.0

phi = gtx.as_field(grid, data)
work = gtx.zeros(grid)
new_phi = gtx.zeros(grid)

diffusion_step(phi, 0.125, nx, ny, work, new_phi)
print(new_phi.asnumpy())
```

The warning says that the program was run in embedded execution, in Python,
because no backend was selected. This is the right choice to develop and to test
a program, and the warning is a reminder for production runs.
[Running operators](../python/running.md) shows how to select a backend. The
warning is not shown in the rest of this page.

```{note}
All the fields used by a program have to be allocated outside, including the
ones for intermediate results like `work`. Operators cannot be nested in the
calls of a program. When an intermediate result is only needed by one operator,
avoid the extra field by writing a field operator which calls the other one, as
in `return phi + coefficient * laplacian(phi)`.
```

## Destination of the results

Without `domain`, the result is computed in all the points of the `out` field.
A part of the field can also be selected by slicing it:

```{code-cell} ipython3
---
tags: [remove-stderr]
---
@gtx.program
def laplacian_in_the_interior(phi: IJField, result: IJField):
    laplacian(phi, out=result[1:-1, 1:-1])


result = gtx.zeros(grid)

laplacian_in_the_interior(phi, result)
print(result.asnumpy())
```

`domain` and slicing cannot be combined in the same call.

When the operator returns a tuple, `out` is a tuple of fields:

```{code-cell} ipython3
---
tags: [remove-stderr]
---
@gtx.field_operator
def differences(phi: IJField) -> tuple[IJField, IJField]:
    return phi(I + 1) - phi, phi(J + 1) - phi


@gtx.program
def compute_differences(phi: IJField, nx: gtx.int32, ny: gtx.int32, dx: IJField, dy: IJField):
    differences(phi, out=(dx, dy), domain={I: (0, nx - 1), J: (0, ny - 1)})


dx, dy = gtx.zeros(grid), gtx.zeros(grid)

compute_differences(phi, nx, ny, dx, dy)
print(dx.asnumpy())
```

The same applies to
[named collections](field-operators.md#named-collections): `out` is then a
collection of the same type.

## Programs or direct calls?

Calling a field operator directly from Python with `out=`, as done in most pages
of this documentation, is equivalent to a program with that single call. It is
convenient for small computations and for tests.

A program is preferable when several operators are always run together: it is
compiled once as a unit, the domains of all the calls are written in one place,
and the application has a single function to call.

## Rules

- The body of a program can only contain calls to operators. Assignments,
  conditionals, loops and `return` are not supported.
- Every call needs the `out` argument.
- The type of `out` must be the type returned by the operator.
- Offsets and connectivities are passed when calling the program, with the
  `offset_provider` argument, and they are used by all the operators in it.
