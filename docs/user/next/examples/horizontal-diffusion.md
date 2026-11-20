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

# Horizontal diffusion

This example implements a classic scheme of weather models: a fourth-order horizontal
diffusion with a flux limiter, which removes small-scale noise from a field
without creating new extrema. It is written for a Cartesian grid and shows:

- Small field operators composed into a larger one.
- Point by point selection with `where`.
- Boundary conditions with [`concat_where`](../dsl/premap.md#boundaries-with-concat-where)
  (an experimental feature).
- A time loop in Python calling a GT4Py operator.
- A [program](../dsl/programs.md) compiled with the `gtfn_cpu` backend.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

import gt4py.next as gtx
from gt4py.next import where
from gt4py.next.experimental import concat_where

I = gtx.Dimension("I")
J = gtx.Dimension("J")

IJField = gtx.Field[gtx.Dims[I, J], gtx.float64]
```

## The scheme

The diffusive fluxes are the differences of the Laplacian of the field between
neighboring points. The limiter sets a flux to zero where it would transport
the quantity up its own gradient, which is what creates overshoots:

```{code-cell} ipython3
@gtx.field_operator
def laplacian(phi: IJField) -> IJField:
    return phi(I + 1) + phi(I - 1) + phi(J + 1) + phi(J - 1) - 4.0 * phi


@gtx.field_operator
def limited_fluxes(phi: IJField) -> tuple[IJField, IJField]:
    lap = laplacian(phi)

    # flux between the points i and i + 1
    flux_i = lap(I + 1) - lap
    flux_i = where(flux_i * (phi(I + 1) - phi) > 0.0, 0.0, flux_i)

    # flux between the points j and j + 1
    flux_j = lap(J + 1) - lap
    flux_j = where(flux_j * (phi(J + 1) - phi) > 0.0, 0.0, flux_j)

    return flux_i, flux_j
```

The new value of the field is the old one minus the divergence of the fluxes.
The stencil reaches two points in each direction, so it can only be evaluated
at a distance of two points from the boundary. In the remaining points the
field keeps its value, which is expressed with `concat_where`:

```{code-cell} ipython3
@gtx.field_operator
def diffuse(
    phi: IJField, coefficient: gtx.float64, nx: gtx.int32, ny: gtx.int32
) -> IJField:
    flux_i, flux_j = limited_fluxes(phi)
    diffused = phi - coefficient * (flux_i - flux_i(I - 1) + flux_j - flux_j(J - 1))

    result = concat_where((I >= 2) & (I < nx - 2), diffused, phi)
    return concat_where((J >= 2) & (J < ny - 2), result, phi)
```

Since `concat_where` only evaluates `diffused` in the interior, the operator can
be called on the whole grid.

## Running it

The initial state is a smooth bump with noise added to it:

```{code-cell} ipython3
nx, ny = 96, 96
domain = gtx.domain({I: nx, J: ny})

x, y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny), indexing="ij")
rng = np.random.default_rng(seed=42)
initial = np.exp(-4.0 * (x**2 + y**2)) + 0.1 * rng.standard_normal((nx, ny))

phi = gtx.as_field(domain, initial)
new_phi = gtx.zeros(domain)
```

The time loop is regular Python code. After each step the two fields are
swapped, so no data is copied:

```{code-cell} ipython3
for _ in range(50):
    diffuse(phi, 0.03, nx, ny, out=new_phi)
    phi, new_phi = new_phi, phi
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
for ax, data, title in zip(axes, (initial, phi.asnumpy()), ("initial", "after 50 steps")):
    image = ax.imshow(data.T, origin="lower", cmap="viridis", vmin=-0.2, vmax=1.1)
    ax.set(title=title, xlabel="I", ylabel="J")
fig.colorbar(image, ax=axes)
plt.show()
```

The noise is gone while the bump keeps its shape, and the two outermost rows of
points still have their initial values.

## Checking the result

The same scheme written with NumPy slices gives the same values:

```{code-cell} ipython3
def numpy_step(phi, coefficient):
    lap = np.zeros_like(phi)
    lap[1:-1, 1:-1] = (
        phi[2:, 1:-1] + phi[:-2, 1:-1] + phi[1:-1, 2:] + phi[1:-1, :-2] - 4.0 * phi[1:-1, 1:-1]
    )

    flux_i = np.zeros_like(phi)
    flux_i[:-1, :] = lap[1:, :] - lap[:-1, :]
    flux_i[:-1, :] = np.where(flux_i[:-1, :] * (phi[1:, :] - phi[:-1, :]) > 0.0, 0.0, flux_i[:-1, :])

    flux_j = np.zeros_like(phi)
    flux_j[:, :-1] = lap[:, 1:] - lap[:, :-1]
    flux_j[:, :-1] = np.where(flux_j[:, :-1] * (phi[:, 1:] - phi[:, :-1]) > 0.0, 0.0, flux_j[:, :-1])

    result = phi.copy()
    result[2:-2, 2:-2] = phi[2:-2, 2:-2] - coefficient * (
        flux_i[2:-2, 2:-2] - flux_i[1:-3, 2:-2] + flux_j[2:-2, 2:-2] - flux_j[2:-2, 1:-3]
    )
    return result


expected = initial
for _ in range(50):
    expected = numpy_step(expected, 0.03)

np.allclose(phi.asnumpy(), expected)
```

Compare the two versions: in the NumPy one, every line needs slices which
encode both the offsets of the stencil and the region where each intermediate
value is valid. In the GT4Py one the offsets are written explicitly and the
regions are deduced.

## Using a program and a compiled backend

Nothing in the definitions depends on how they are executed. For a production
run, the time step is wrapped in a [program](../dsl/programs.md), which gives
the model a single function to call, and the program is bound to a backend:

```{code-cell} ipython3
@gtx.program
def diffusion_step(
    phi: IJField, coefficient: gtx.float64, nx: gtx.int32, ny: gtx.int32, new_phi: IJField
):
    diffuse(phi, coefficient, nx, ny, out=new_phi)


diffusion_step_cpu = diffusion_step.with_backend(gtx.gtfn_cpu)
```

The fields are allocated for the backend, and the loop is the same as before.
The first call compiles the program:

```{code-cell} ipython3
---
tags: [remove-stderr]
---
phi_cpu = gtx.as_field(domain, initial, allocator=gtx.gtfn_cpu)
new_phi_cpu = gtx.zeros(domain, allocator=gtx.gtfn_cpu)

for _ in range(50):
    diffusion_step_cpu(phi_cpu, 0.03, nx, ny, new_phi_cpu)
    phi_cpu, new_phi_cpu = new_phi_cpu, phi_cpu

np.allclose(phi_cpu.asnumpy(), phi.asnumpy())
```

The compiled program gives the same result as the embedded execution.
[Running operators and programs](../python/running.md) explains the backends,
and how to compile in advance.
