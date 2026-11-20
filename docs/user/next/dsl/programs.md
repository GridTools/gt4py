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

In the previous pages, operators were called directly from Python with `out=`.
A **program** groups several such calls into one function, decorated with
`@gtx.program`. It is compiled as a unit and it gives the application a single
function to call, for example for a time step of a model.

Programs are the **interface between the DSL and the rest of the application**.
Operators are pure functions returning new fields, but a program returns
nothing and allocates nothing: the results of the operators it calls are
written **in place** into fields received as arguments of the program. The
memory of all the fields therefore belongs to the caller. This makes it
possible to use GT4Py from frameworks and models which need to own the buffers
of their data, instead of relying on GT4Py to allocate them.

A direct call to an operator is equivalent to a program with that single call.
Direct calls are convenient for small computations and for tests, and programs
are preferable when several operators are always run together.

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

This program computes the Laplacian of a field into a work field, and then uses
it to update the field:

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

The warning is a reminder that no backend was selected, so the program was run
in embedded execution. This is the right choice for developing and testing.
[Running operators and programs](../python/running.md) shows how to select a backend. The
later cells of this page hide the warning.

```{tip}
When an intermediate result is only needed by one operator, avoid the extra
field (`work` above) by writing a field operator which calls the other one, as
in `return phi + coefficient * laplacian(phi)`.
```

## Destination of the results

Without `domain`, the result is computed at all the points of the `out` field.
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

## Rules

- The body of a program can only contain calls to operators. Assignments,
  conditionals, loops and `return` are not supported.
- The arguments of a call cannot be calls to other operators.
- All the fields used by a program are allocated outside of it, including the
  ones for intermediate results.
- Every call needs the `out` argument.
- The type of `out` must be the type returned by the operator.
- Offsets and connectivities are passed when calling the program, with the
  `offset_provider` argument, and they are used by all the operators in it.
