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

# Implicit vertical diffusion

Vertical diffusion is usually solved with an implicit scheme, because the
levels of a model can be very thin near the surface. This requires solving a
tridiagonal linear system in every column, which is the typical use of
[scan operators](../dsl/scan-operators.md). The example shows:

- A tridiagonal solver written with a forward and a backward scan.
- [`concat_where`](../dsl/shifts.md#boundaries-with-concat-where) (an experimental
  feature) to give the first and the last level a different treatment.
- The same code running on many columns at once.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

I = gtx.Dimension("I")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)

IKField = gtx.Field[gtx.Dims[I, K], gtx.float64]
```

## The tridiagonal solver

The Thomas algorithm solves the system
$a_k x_{k-1} + b_k x_k + c_k x_{k+1} = d_k$ in two sweeps. The forward sweep
eliminates the lower diagonal, going from the first level to the last one:

$$
c'_k = \frac{c_k}{b_k - a_k c'_{k-1}}, \qquad
d'_k = \frac{d_k - a_k d'_{k-1}}{b_k - a_k c'_{k-1}}
$$

The backward sweep computes the solution from the last level to the first one:

$$
x_k = d'_k - c'_k x_{k+1}
$$

Each sweep is a scan operator. The state of the forward sweep is the pair
$(c'_{k-1}, d'_{k-1})$, and the state of the backward sweep is $x_{k+1}$:

```{code-cell} ipython3
@gtx.scan_operator(axis=K, forward=True, init=(0.0, 0.0))
def forward_sweep(
    state: tuple[float, float], a: float, b: float, c: float, d: float
) -> tuple[float, float]:
    c_prime, d_prime = state
    denominator = b - a * c_prime
    return c / denominator, (d - a * d_prime) / denominator


@gtx.scan_operator(axis=K, forward=False, init=0.0)
def backward_sweep(x_next: float, c_prime: float, d_prime: float) -> float:
    return d_prime - c_prime * x_next


@gtx.field_operator
def solve_tridiagonal(a: IKField, b: IKField, c: IKField, d: IKField) -> IKField:
    c_prime, d_prime = forward_sweep(a, b, c, d)
    return backward_sweep(c_prime, d_prime)
```

`solve_tridiagonal` is a general solver, which can be reused by any operator
needing one.

## The diffusion step

A backward Euler step of the diffusion equation, with no flux through the top
and the bottom of the column, gives the system

$$
-r\,T^{n+1}_{k-1} + (1 + 2r)\,T^{n+1}_k - r\,T^{n+1}_{k+1} = T^n_k
$$

where $r = \kappa \Delta t / \Delta z^2$. At the first and at the last level the
term with the missing neighbor is dropped, which leaves $1 + r$ in the diagonal.
The solver never uses the lower diagonal at the first level nor the upper one
at the last level, so only the diagonal needs a special treatment at both ends.
`concat_where` writes it:

```{code-cell} ipython3
@gtx.field_operator
def diffuse(temperature: IKField, r: IKField, nk: gtx.int32) -> IKField:
    diagonal = concat_where((K > 0) & (K < nk - 1), 1.0 + 2.0 * r, 1.0 + r)
    return solve_tridiagonal(-r, diagonal, -r, temperature)
```

## Running it

The grid has 64 columns with 40 levels. All the columns start with the same
profile, a warm layer in the middle, but the diffusivity grows from the first
column to the last one:

```{code-cell} ipython3
ni, nk = 64, 40
grid = gtx.domain({I: ni, K: nk})

profile = 280.0 + 10.0 * np.exp(-(((np.arange(nk) - 20.0) / 3.0) ** 2))
initial = np.tile(profile, (ni, 1))
r_values = np.tile(np.linspace(0.0, 2.0, ni)[:, np.newaxis], (1, nk))

temperature = gtx.as_field(grid, initial)
new_temperature = gtx.zeros(grid)
r = gtx.as_field(grid, r_values)

for _ in range(10):
    diffuse(temperature, r, nk, out=new_temperature)
    temperature, new_temperature = new_temperature, temperature
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
for ax, data, title in zip(axes, (initial, temperature.asnumpy()), ("initial", "after 10 steps")):
    image = ax.imshow(data.T, origin="lower", aspect="auto", cmap="inferno", vmin=280.0, vmax=290.0)
    ax.set(title=title, xlabel="I (diffusivity increases)", ylabel="K")
fig.colorbar(image, ax=axes, label="temperature")
plt.show()
```

The scheme is unconditionally stable, so the columns with $r > 1/2$, where an
explicit scheme would blow up, are solved without problems.

## Checking the result

With no flux through the boundaries, the sum of every column is conserved:

```{code-cell} ipython3
np.allclose(temperature.asnumpy().sum(axis=1), initial.sum(axis=1))
```

And one step agrees with solving the same systems with NumPy:

```{code-cell} ipython3
def numpy_step(temperature, r):
    result = np.empty_like(temperature)
    for i in range(temperature.shape[0]):
        matrix = np.zeros((nk, nk))
        for k in range(nk):
            lower = -r[i, k] if k > 0 else 0.0
            upper = -r[i, k] if k < nk - 1 else 0.0
            matrix[k, k] = 1.0 - lower - upper
            if k > 0:
                matrix[k, k - 1] = lower
            if k < nk - 1:
                matrix[k, k + 1] = upper
        result[i] = np.linalg.solve(matrix, temperature[i])
    return result


diffuse(gtx.as_field(grid, initial), r, nk, out=new_temperature)

np.allclose(new_temperature.asnumpy(), numpy_step(initial, r_values))
```
