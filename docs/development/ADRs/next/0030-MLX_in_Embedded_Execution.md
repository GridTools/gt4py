---
tags: [embedded, mlx, array-api, devices]
---

# MLX in Embedded Execution

- **Status**: valid
- **Authors**: Enrique González Paredes (@egparedes)
- **Created**: 2026-11-24
- **Updated**: 2026-11-24

In the context of running embedded `gt4py.next` programs on Apple silicon, facing an array library (MLX) that deviates from the Array API standard in ways `array-api-compat` does not cover, we decided to wrap `mlx.core` in an in-tree namespace and register `ArrayOps` for it, to get MLX fields with the machinery of the other array libraries. We accept an in-tree module that has to be maintained until `array-api-compat` supports MLX, a float32-only GPU, and fields that run only in embedded execution.

## Context

`mlx.core` is close to the Array API standard, but not conforming. What matters for fields:

- Operations take no `device` argument: arrays have no device, and every operation runs on MLX's default device or on a `stream`.
- It has `bool_` instead of `bool`, no `nonzero`, `cbrt`, `fmod` or `gamma`, and its `sum` has no `dtype`. Integer division gives float32.
- It converts float64 NumPy data and Python floats to float32, computes `sin`, `cos`, `exp` and `cbrt` of float64 with float32 accuracy, and does not support float64 on the GPU.
- Dtype promotion is its own (`int64` and `float32` give `float32`).
- Evaluation is lazy: operations build a graph, computed when values are needed.
- Slices are copies, and the NumPy view of an MLX array changes with writes into the array.

`array_api_compat.array_namespace` returns `mlx.core` itself for MLX arrays.

## Decision

- **In-tree namespace.** `_mlx_namespace` is the `ArrayOps.xp` of MLX arrays. It adds what `gt4py.next` uses on top of `mlx.core` (any other name is `mlx.core`'s), and is registered with `array_ops.register(..., namespace=mx)`, since compat gives `mlx.core` as the namespace of the arrays. It is to be deleted when `array-api-compat` supports MLX.
- **float64 policy.** The namespace never downcasts silently: data keeps the dtype NumPy gives it, and creation functions without `dtype` use NumPy's defaults, so the default dtype of fields stays float64. The float64 functions that MLX computes with float32 accuracy are computed by NumPy (noted in the code with the MLX version, 0.32.4, to be removed when MLX fixes them). Integers are divided in float64, as in NumPy, also where the DSL types `int32 / int32` as `int32`. Requesting float64 on the GPU raises a `TypeError`; users pass `dtype=float32` there, also for the `init` of a scan. Dividing integers on the GPU raises a `TypeError` of its own, pointing to `//` or to float32 operands, instead of falling back to float32. Mixed-dtype arithmetic on fields promotes as MLX does.
- **`Device` and `DeviceType.METAL`.** `DeviceType.METAL` (value 8, as in DLPack) stands for `mlx.core.gpu` on Apple silicon; on a CUDA build of MLX the GPU is `DeviceType.CUDA`. A field reports MLX's default device. A `device` argument is accepted only if it names that device, otherwise a `ValueError`. A Metal `device` without `allocator` allocates with MLX, the only library for it; without MLX installed it raises a `ValueError` naming `allocator=mlx.core`. MLX's `__dlpack_device__` is not used to find the device: it reports Metal whenever Metal is available, also for arrays computed on the CPU, and the CPU on CUDA builds. GT4Py never calls `mlx.core.set_default_device`: it is global state of the process, owned by the user.
- **`as_field`.** An MLX array given without `allocator` or `device` stays an MLX array (`ArrayOps.own_allocator`), instead of being copied to NumPy by the default allocator of its device.
- **Evaluation.** A write into a field and the fields returned by a field operator called outside any other one are evaluated (`ArrayOps.evaluate`), so that a time loop does not build a graph over all its steps. Field arithmetic outside operators stays lazy until read.
- **`to_numpy` copies**, because a NumPy view of an MLX array changes with writes into it.
- **No buffer info.** `buffer_info` raises: MLX gives no stable pointer and strides, so MLX fields are not passed to compiled backends.

## Alternatives considered

### Wait for `array-api-compat`

Leaves MLX unusable for now; the wrapper is small and deletable.

### Patch the missing functions at use sites

Spreads library-specific code in `NdArrayField` and the builtins, against the design of `ArrayOps`.

### Follow MLX: float32 default and silent downcast

Results would differ from the other libraries without notice, and field dtypes would no longer match the declared ones.

### Per-array devices in `Device`

MLX has none; faking them would put arrays on a device they do not run on.

### Evaluate after every operation

Gives up the graph fusion that MLX does within an operator.

## Consequences

- MLX fields run only in embedded execution; DaCe and gtfn raise `NotImplementedError` for them.
- Writes through sliced `out` fields are lost, as slices are copies. The corresponding tests are excluded (`MLX_EMBEDDED_SKIP_LIST`), as are those using `gamma`.
- Scans work but are very slow.
- On the GPU, users must pass float32 everywhere a dtype is chosen.
- MLX is tested on Linux with the CPU build (`requires_mlx`), and on Apple GPUs in a dedicated workflow (`requires_mlx_metal`).

## References

- [ADR 0029 - Connectivities in Tracing Embedded Backends](0029-Connectivities_in_Tracing_Embedded_Backends.md)
