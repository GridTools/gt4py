---
tags: [embedded, dependencies]
---

# PyTorch Tensors in Embedded Execution

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-11-13
- **Updated**: 2026-11-13

In the context of embedded execution of `gt4py.next` field operators and programs, facing users
who hold their data in `torch` tensors on the CPU or on CUDA, we decided to support
`torch.Tensor` as the buffer of `NdArrayField`, next to NumPy, CuPy and JAX. It is an optional
dependency (`gt4py[torch]`), so such users can run embedded programs on their tensors without
copying them to NumPy or CuPy. `torch.compile` is not part of this decision.

## Context

`NdArrayField` implements its operations against the Array API standard, through the namespace
`array_api_compat` returns for the buffer. A library that deviates subclasses `ArrayOps` and
registers it. `array_api_compat.torch` covers most of what embedded fields need. The rest:

- `cbrt` and `gamma` are not in the standard, and torch has neither (it has `lgamma`).
- Several functions reject Python or NumPy scalar operands. `minimum`, `maximum`, `logical_*`,
  `atan2` and `hypot` reject them in `array-api-compat` < 1.16; `fmod` with the scalar first
  rejects them in every version.
- Creating a tensor without `device` allocates on the CPU or on the torch default device, not on
  the current CUDA device as CuPy does. Mixing CPU and CUDA tensors in an operation is an error,
  except for 0-d CPU tensors.
- `uint16`, `uint32` and `uint64` tensors can be allocated, compared, indexed and selected, but
  arithmetic and reductions on them raise `NotImplementedError` in torch.
- A tensor that requires grad cannot be exported through DLPack.

Unlike JAX arrays, tensors are mutable, so `__setitem__` writes in place as for NumPy.

## Decision

- `TorchOps` (`embedded/array_libraries/torch_ops.py`) overrides `array_ns` (the `torch` module
  itself, as `field.array_ns` is for the other libraries), `cbrt`/`gamma`, `operand`, `to_numpy`
  and `buffer_info`.
- Scalar operands become 0-d tensors **on the CPU**. torch takes them next to CUDA tensors
  without a copy to the device. A Python scalar of the field's kind takes the field's dtype, which
  keeps the weak typing of the standard.
- `TorchOps` is registered when the first torch tensor is seen (`array_ops.register_on_first_use`),
  so importing `gt4py.next` does not import torch, which takes about a second.
- Constructors accept `allocator=torch`. Without `device`, tensors go to the torch default device
  (`torch.set_default_device`, or a `with torch.device(...)` block), as `allocator=jax.numpy`
  allocates on the JAX default device. An explicit `device` goes through a device translator in
  `_core.ndarray_utils`: CPU to `cpu`, and CUDA and ROCm to `cuda:<id>` (ROCm builds of torch also
  use the `cuda` device type). The translator matches torch by module name, so importing
  `gt4py._core` does not import torch.
- Internal allocations follow the device of the inputs. The embedded scan allocates its result
  through the compat namespace of its first field argument, on that field's device.
- `__gt_buffer_info__` raises `NotImplementedError`, as for JAX.
- Testing: torch is in the namespace matrix of the embedded unit tests (CPU). The ffront feature
  tests have `EmbeddedTorch` and `EmbeddedTorchCUDA` entries. The CUDA entry allocates through a
  test-only field-buffer allocator on an explicit device, not through `torch.set_default_device`,
  which would hide device bugs. Torch-specific tests run on both devices. The extra is installed in
  the internal codegen test sessions, and the `requires_torch` marker covers tests that import
  torch.

## Consequences

- Features that need unsigned integer arithmetic do not work on torch fields. No embedded feature
  test uses unsigned integers.
- Torch fields cannot be passed to compiled backends yet.
- `as_field(dims, tensor)` without `allocator` copies to the default allocator of the tensor's
  device (NumPy on the CPU, CuPy on CUDA), as for any DLPack array. `allocator=torch` keeps a
  torch tensor.
- Autograd works through embedded operations. A neighbor table entry with a skip value still
  gathers an element before the result is masked, so the gradient is NaN where the operator is
  singular at that element.
- The torch wheel on PyPI is the CUDA build. With its `nvidia-*` dependencies it is about 2.8 GiB
  for Linux x86_64, and the internal test sessions download it.
- Out of scope, possible follow-ups: `torch.compile`, and zero-copy passing of torch fields to
  gtfn/DaCe via DLPack.

## Alternatives considered

- **A torch namespace module with NumPy names and per-device field classes**, written before
  `NdArrayField` was ported to the Array API standard. Rejected: the standard port removed the
  NumPy-isms that needed it.
- **`torch.set_default_device` for the CUDA test entry**: rejected. It hides device bugs that
  users hit without a default device.
- **Registering `TorchOps` on import, as for CuPy and JAX**: rejected because of the import time.
  JAX cannot be registered lazily, because `jax_enable_x64` and the pytree registration must run
  before user arrays exist.
