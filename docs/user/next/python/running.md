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

# Executing GT4Py programs

The same program can be executed in two ways: directly in Python, which is the
default, or compiled by a backend. This page explains both, and how to control
when the compilation happens.

```{note}
This page does not distinguish between programs and direct calls to field
operators: both are executed in the same way, and everything said here about
programs also applies to an operator called from Python.
```

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
    phi: IJField,
    coefficient: gtx.float64,
    limited: bool,
    nx: gtx.int32,
    ny: gtx.int32,
    new_phi: IJField,
):
    diffuse(phi, coefficient, limited, out=new_phi, domain={I: (1, nx - 1), J: (1, ny - 1)})
```

## The calling interface

Programs are called with the arguments of their definition. Field operators and
scan operators are called with their arguments plus `out`.

| Argument          | Used in                   | Meaning                                                                        |
| ----------------- | ------------------------- | ------------------------------------------------------------------------------ |
| `out`             | Direct calls to operators | Field, or tuple of fields, where the result is written. Required               |
| `domain`          | Direct calls to operators | Points to compute. By default, all the points of `out`                         |
| `offset_provider` | All calls                 | Connectivities of the unstructured offsets used. Not needed in Cartesian grids |

An operator called from Python does not return the result: it writes it into
`out`. Scalar arguments accept Python and NumPy numbers.

## Embedded execution

Without a backend, the code is executed by the Python interpreter. Every
operation on fields is evaluated by the array library holding their data, which
is NumPy for fields allocated in the default way. Other libraries, such as CuPy
or JAX, are selected when
[allocating the fields](grids-and-fields.md#allocating-for-embedded-execution).

```{code-cell} ipython3
---
tags: [remove-stderr]
---
nx, ny = 256, 256
domain = gtx.domain({I: nx, J: ny})

rng = np.random.default_rng(seed=0)
phi = gtx.as_field(domain, rng.random((nx, ny)))
reference = gtx.zeros(domain)

diffusion_step(phi, 0.1, True, nx, ny, reference)
```

Embedded execution is the mode to use while developing. It is not meant to be
fast, but:

- There is no compilation, so changes are tested immediately.
- Errors raised while evaluating an operator point to its line, and a debugger
  can step through the code.
- On Cartesian grids, field expressions also work outside operators, so
  intermediate values can be inspected by typing parts of an operator at a
  Python prompt:

```{code-cell} ipython3
increment = 0.1 * (phi(I + 1) + phi(I - 1) - 2.0 * phi)

print(increment.domain)
print(gtx.where(increment > 0.05, 1.0, 0.0).asnumpy().sum())
```

Outside operators the built-in functions can be called through the module, as in
`gtx.where`.

```{note}
Outside operators, expressions using the offsets of an unstructured mesh or
scan operators need some additional setup, which is not documented yet. We are
working on making this simpler.
```

## Compiled backends

A backend translates programs and operators into optimized code. These are the
backends exported by `gt4py.next`:

| Backend        | Description                                                                                    |
| -------------- | ---------------------------------------------------------------------------------------------- |
| `gtx.dace_cpu` | Code generated and optimized with the [DaCe](https://github.com/spcl/dace) framework, for CPUs |
| `gtx.dace_gpu` | The same for GPUs (see [Running on GPUs](#running-on-gpus))                                    |
| `gtx.gtfn_cpu` | C++ code using the GridTools library, for CPUs                                                 |
| `gtx.gtfn_gpu` | The same for GPUs                                                                              |

All of them are used in the same way. The examples of this page use
`gtx.gtfn_cpu`, and they work with any other backend by replacing its name.

```{important}
Compilation jobs run in separate processes by default. In a script, put the code calling
compiled programs or operators inside a function called under
`if __name__ == "__main__":`, as required by the Python `multiprocessing`
module.
```

`with_backend` returns a copy of a program or operator bound to a backend. The
original one is not changed:

```{code-cell} ipython3
diffusion_step_cpu = diffusion_step.with_backend(gtx.gtfn_cpu)
```

The backend can also be fixed in the definition, with
`@gtx.program(backend=gtx.gtfn_cpu)` or
`@gtx.field_operator(backend=gtx.gtfn_cpu)`.

The fields are [allocated for the backend](grids-and-fields.md#allocating-for-a-backend):

```{code-cell} ipython3
phi_cpu = gtx.as_field(domain, phi.asnumpy(), allocator=gtx.gtfn_cpu)
new_phi_cpu = gtx.zeros(domain, allocator=gtx.gtfn_cpu)
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
    diffusion_step_cpu(phi_cpu, 0.1, True, nx, ny, new_phi_cpu)
    return time.perf_counter() - start


print(f"first call:  {timed_call():8.4f} s")
print(f"second call: {timed_call():8.4f} s")
print("same result:", np.allclose(new_phi_cpu.asnumpy(), reference.asnumpy()))
```

```{tip}
Compiled programs warn when Python runs with assertions enabled, which is the
default, because the checks done by GT4Py in every call take some time. For
production runs, start Python with the `-O` option or set the `PYTHONOPTIMIZE=1`
environment variable. These warnings are hidden in this page.
```

The same program with the DaCe backend:

```{code-cell} ipython3
---
tags: [remove-stderr]
---
diffusion_step_dace = diffusion_step.with_backend(gtx.dace_cpu)

phi_dace = gtx.as_field(domain, phi.asnumpy(), allocator=gtx.dace_cpu)
new_phi_dace = gtx.zeros(domain, allocator=gtx.dace_cpu)

diffusion_step_dace(phi_dace, 0.1, True, nx, ny, new_phi_dace)

print("same result:", np.allclose(new_phi_dace.asnumpy(), reference.asnumpy()))
```

### The build cache

Generated code and binaries are stored in a build cache. Two environment
variables control it:

| Variable                     | Values                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------ |
| `GT4PY_BUILD_CACHE_LIFETIME` | `session` (default): the cache is deleted when Python exits. `persistent`: it is kept and reused |
| `GT4PY_BUILD_CACHE_DIR`      | Folder containing the persistent cache (`.gt4py_cache`). By default, the current folder          |

```{tip}
With the default settings every run of an application compiles its programs
again. Set `GT4PY_BUILD_CACHE_LIFETIME=persistent` to compile them only once.
```

## Compiling in advance

In an application it is usually better to compile everything during the
initialization than at the first time step. `compile` starts the compilation of
a program without running it, replacing the compilation at the first call shown
above:

```{code-cell} ipython3
---
tags: [remove-stderr]
---
precompiled_step = diffusion_step.with_backend(gtx.gtfn_cpu)

precompiled_step.compile(offset_provider={})
gtx.wait_for_compilation()
```

- `compile` needs the offset provider that will be used in the calls, because
  the generated code depends on the connectivities.
- The compilation runs in the background and `compile` returns immediately, so
  several programs can be compiled in parallel. A program which is called
  before its compilation has finished simply waits for it at that point.
- `gtx.wait_for_compilation()` waits until all the compilations have finished,
  and it reports the errors found. It is only needed to keep compilation and
  execution apart, for instance to measure them, and it is not recommended in
  general.
- `compile` also works on field operators.

## Static parameters

Some scalar parameters are configuration options which never change during a
run, like `limited` in the example. Declaring them as **static** lets the
backend generate a specialized version of the program for each value, where
the `if` on `limited` is already decided when compiling.

`with_static_params` declares the static parameters of a program, and `compile`
receives the list of values to prepare for each of them:

```{code-cell} ipython3
---
tags: [remove-stderr]
---
static_step = (
    diffusion_step.with_backend(gtx.gtfn_cpu)
    .with_static_params("limited")
    .with_compilation_options(enable_jit=False)
)

static_step.compile(limited=[True, False], offset_provider={})
```

Two variants are compiled. The program is called as usual, and the value of the
argument selects the variant:

```{code-cell} ipython3
static_step(phi_cpu, 0.1, True, nx, ny, new_phi_cpu)

print("same result:", np.allclose(new_phi_cpu.asnumpy(), reference.asnumpy()))
```

By default, a call with a value which was not compiled in advance compiles a
new variant at that moment. The `enable_jit=False` option used above forbids
it: such a call is an error, instead of an unexpected compilation in the middle
of a run. Set the options before calling `compile`, because
`with_compilation_options` returns a new program which does not share the
variants already compiled.

Static parameters can also be declared in the decorator, which is the way to do
it for field operators:

```python
@gtx.field_operator(static_params=["limited"])
def diffuse(phi: IJField, coefficient: gtx.float64, limited: bool) -> IJField: ...
```

```{tip}
Make static only the parameters with a few possible values, like flags and
options. Every combination of values is a different compilation.
```

## Static domains

By default the compiled code works for fields of any size, because the domains
of the fields are only known when the program is called. With the
`static_domains` option, the domains of the field arguments are fixed at
compile time, like the values of static parameters, and the backend uses them
to generate better code.

```{code-cell} ipython3
---
tags: [remove-stderr]
---
static_domain_step = diffusion_step.with_backend(gtx.gtfn_cpu).with_compilation_options(
    static_domains=True
)

static_domain_step(phi_cpu, 0.1, True, nx, ny, new_phi_cpu)

print("same result:", np.allclose(new_phi_cpu.asnumpy(), reference.asnumpy()))
```

- The program is compiled at its first call, for the domains of the fields it
  receives. A call with fields over different domains compiles a new variant.
  `compile` does not support programs with static domains yet.
- The option can also be given in the decorator, as in
  `@gtx.program(static_domains=True)`.

```{important}
We currently recommend using static domains: the optimizations for domains
which are only known at run time are not as good.
```

## Running on GPUs

`gtx.dace_gpu` and `gtx.gtfn_gpu` generate code for GPUs. They need CuPy, installed with one of the
extras of the package (for example `pip install gt4py[cuda12]`), and the
compiler toolchain of the GPU vendor on the system. The fields must be on the
device, which is done by passing the backend as allocator. Nothing else changes:

```python
diffusion_step_gpu = diffusion_step.with_backend(gtx.gtfn_gpu)

phi_gpu = gtx.as_field(domain, phi.asnumpy(), allocator=gtx.gtfn_gpu)
new_phi_gpu = gtx.zeros(domain, allocator=gtx.gtfn_gpu)

diffusion_step_gpu(phi_gpu, 0.1, True, nx, ny, new_phi_gpu)

result = new_phi_gpu.asnumpy()  # copies the data back to main memory
```

The data of a field allocated this way is held in a CuPy array. Fields on a
GPU also work in embedded execution, where the operations are evaluated by
CuPy.
