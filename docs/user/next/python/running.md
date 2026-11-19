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

# Running operators

The same operators and programs can be executed in two ways: directly in Python,
which is the default, or compiled by a backend. This page explains both, and how
to control when the compilation happens.

The examples use this small diffusion program:

```{code-cell} ipython3
import time

import numpy as np

import gt4py.next as gtx
from gt4py.next import where

I = gtx.Dimension("I")
J = gtx.Dimension("J")

IJField = gtx.Field[gtx.Dims[I, J], gtx.float64]


@gtx.field_operator
def laplacian(phi: IJField) -> IJField:
    return phi(I + 1) + phi(I - 1) + phi(J + 1) + phi(J - 1) - 4.0 * phi


@gtx.field_operator
def diffuse(phi: IJField, coefficient: gtx.float64, limited: bool) -> IJField:
    increment = coefficient * laplacian(phi)
    if limited:
        increment = where(increment > 0.5, 0.5, increment)
    return phi + increment


@gtx.program
def diffusion_step(
    phi: IJField, coefficient: gtx.float64, limited: bool, nx: gtx.int32, ny: gtx.int32, out: IJField
):
    diffuse(phi, coefficient, limited, out=out, domain={I: (1, nx - 1), J: (1, ny - 1)})
```

## The calling interface

Programs are called with the arguments of their definition. Field operators and
scan operators are called with their arguments plus `out`.

| Argument          | Used in             | Meaning                                                                        |
| ----------------- | ------------------- | ------------------------------------------------------------------------------ |
| `out`             | Operators           | Field, or tuple of fields, where the result is written. Required               |
| `domain`          | Operators           | Points to compute. By default, all the points of `out`                         |
| `offset_provider` | Operators, programs | Connectivities of the unstructured offsets used. Not needed in Cartesian grids |

An operator called from Python does not return the result: it writes it into
`out`. Scalar arguments accept Python and NumPy numbers.

## Embedded execution

Without a backend, the code is executed by the Python interpreter. Every
operation on fields is evaluated by the array library holding their data, which
is NumPy for fields allocated in the default way.

```{code-cell} ipython3
nx, ny = 256, 256
grid = gtx.domain({I: nx, J: ny})

rng = np.random.default_rng(seed=0)
phi = gtx.as_field(grid, rng.random((nx, ny)))
reference = gtx.zeros(grid)

diffusion_step(phi, 0.1, True, nx, ny, reference)
```

The warning is a reminder that no backend was selected. It is only given for
programs.

Embedded execution is the mode to use while developing:

- There is no compilation, so changes are tested immediately.
- Domains are checked. Reading a field outside of its domain is an error, while
  a compiled backend would silently read wrong values.
- Errors point to the line of the operator which failed, and a debugger can step
  through the code.
- Field expressions also work outside operators, so intermediate values can be
  inspected by typing parts of an operator in a Python prompt:

```{code-cell} ipython3
increment = 0.1 * (phi(I + 1) + phi(I - 1) - 2.0 * phi)

print(increment.domain)
print(gtx.where(increment > 0.05, 1.0, 0.0).asnumpy().sum())
```

Outside operators the built-in functions can be called through the module, as in
`gtx.where`.

## Compiled backends

A backend translates programs and operators into optimized code. The backends
available in `gt4py.next` are:

| Backend        | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| `gtx.gtfn_cpu` | C++ code using the GridTools library, for CPUs              |
| `gtx.gtfn_gpu` | The same for GPUs (see [Running on GPUs](#running-on-gpus)) |

`with_backend` returns a copy of a program or operator bound to a backend. The
original one is not changed:

```{code-cell} ipython3
diffusion_step_cpu = diffusion_step.with_backend(gtx.gtfn_cpu)
```

The backend can also be fixed in the definition, with
`@gtx.program(backend=gtx.gtfn_cpu)` or
`@gtx.field_operator(backend=gtx.gtfn_cpu)`. Selecting it with `with_backend`
in the code which sets up the application has the advantage of keeping the
definitions independent of where they run.

Fields should be allocated for the backend:

```{code-cell} ipython3
phi_cpu = gtx.as_field(grid, phi.asnumpy(), allocator=gtx.gtfn_cpu)
out_cpu = gtx.zeros(grid, allocator=gtx.gtfn_cpu)
```

### Compilation at the first call

By default the compilation is done _just in time_: the first call generates the
code, compiles it and runs it, and the next calls only run it.

```{code-cell} ipython3
---
tags: [remove-stderr]
---
def timed_call():
    start = time.perf_counter()
    diffusion_step_cpu(phi_cpu, 0.1, True, nx, ny, out_cpu)
    return time.perf_counter() - start


print(f"first call:  {timed_call():8.4f} s")
print(f"second call: {timed_call():8.4f} s")
print("same result:", np.allclose(out_cpu.asnumpy(), reference.asnumpy()))
```

```{note}
Compiled programs warn when Python runs with assertions enabled, which is the
default, because the checks done by GT4Py in every call take some time. For
production runs, start Python with the `-O` option or set the `PYTHONOPTIMIZE=1`
environment variable. These warnings are hidden in this page.
```

### The build cache

Generated code and binaries are stored in a build cache. Two environment
variables control it:

| Variable                     | Values                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------ |
| `GT4PY_BUILD_CACHE_LIFETIME` | `session` (default): the cache is deleted when Python exits. `persistent`: it is kept and reused |
| `GT4PY_BUILD_CACHE_DIR`      | Folder containing the persistent cache (`.gt4py_cache`). By default, the current folder          |

With the default settings every run of an application compiles its programs
again. Set `GT4PY_BUILD_CACHE_LIFETIME=persistent` to compile them only once.

## Compiling in advance

In an application it is usually better to compile everything during the
initialization than at the first time step. `compile` starts the compilation of
a program without running it:

```{code-cell} ipython3
---
tags: [remove-stderr]
---
diffusion_step_cpu = diffusion_step.with_backend(gtx.gtfn_cpu)

diffusion_step_cpu.compile(offset_provider={})
gtx.wait_for_compilation()
```

- `compile` needs the offset provider that will be used in the calls, because
  the generated code depends on the connectivities. For Cartesian programs it is
  an empty dictionary.
- The compilation runs in the background and `compile` returns immediately, so
  several programs can be compiled in parallel. `gtx.wait_for_compilation()`
  waits until all of them have finished, and it reports the errors found.
- `compile` also works on field operators.

```{important}
Compilation jobs run in separate processes. In a script, put the code calling
GT4Py programs inside a function called under `if __name__ == "__main__":`, as
required by the Python `multiprocessing` module.
```

## Static arguments

Some scalar arguments are configuration options which never change during a
run, like `limited` in the example. Declaring them as **static** lets the
backend generate a specialized version of the program for each value, where
the `if` on `limited` is already decided when compiling.

`with_static_params` declares the static parameters of a program, and `compile`
receives the list of values to prepare for each of them:

```{code-cell} ipython3
---
tags: [remove-stderr]
---
diffusion_step_static = diffusion_step.with_backend(gtx.gtfn_cpu).with_static_params("limited")

diffusion_step_static.compile(limited=[True, False], offset_provider={})
gtx.wait_for_compilation()
```

Two variants have been compiled. The program is called as usual, and the value
of the argument selects the variant:

```{code-cell} ipython3
diffusion_step_static(phi_cpu, 0.1, True, nx, ny, out_cpu)

print("same result:", np.allclose(out_cpu.asnumpy(), reference.asnumpy()))
```

A call with a value which was not compiled in advance compiles a new variant at
that moment. To forbid it, and get an error instead of an unexpected
compilation in the middle of a run, disable the just-in-time compilation with
`program.with_compilation_options(enable_jit=False)`.

Static parameters can also be declared in the decorator, which is the way to do
it for field operators:

```python
@gtx.field_operator(static_params=["limited"])
def diffuse(phi: IJField, coefficient: gtx.float64, limited: bool) -> IJField: ...
```

```{tip}
Make static only the arguments with a few possible values, like flags and
options. Every combination of values is a different compilation.
```

## Running on GPUs

`gtx.gtfn_gpu` generates code for GPUs. It needs the GPU version of the
dependencies, installed with one of the extras of the package, for example
`pip install gt4py[cuda12]`. The fields must be allocated in the device, which
is done by passing the backend as allocator. Nothing else changes:

```python
diffusion_step_gpu = diffusion_step.with_backend(gtx.gtfn_gpu)

phi_gpu = gtx.as_field(grid, initial_values, allocator=gtx.gtfn_gpu)
out_gpu = gtx.zeros(grid, allocator=gtx.gtfn_gpu)

diffusion_step_gpu(phi_gpu, 0.1, True, nx, ny, out_gpu)

result = out_gpu.asnumpy()  # copies the data back to the main memory
```

The buffer of a field allocated this way, `field.ndarray`, is a CuPy array.
Fields located in a GPU also work in embedded execution, where the operations
are evaluated by CuPy.
