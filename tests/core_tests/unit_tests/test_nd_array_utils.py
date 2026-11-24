# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
from typing import Any

import numpy as np
import pytest

from gt4py._core import definitions, dlpack, ndarray_utils


def cuda_device(id_: int) -> Any:
    if ndarray_utils.cupy is not None:
        return ndarray_utils.cupy.cuda.Device(id_)
    return NotImplemented  # something that's not `None`


@pytest.mark.parametrize(
    "array_ns, gt4py_device, expected_device",
    [
        (ndarray_utils.np, None, None),
        (ndarray_utils.np, definitions.Device(definitions.DeviceType.CPU, 0), None),
        pytest.param(ndarray_utils.cupy, None, None, marks=pytest.mark.requires_gpu),
        pytest.param(
            ndarray_utils.cupy,
            definitions.Device(definitions.CUPY_DEVICE_TYPE, 42),
            cuda_device(42),
            marks=pytest.mark.requires_gpu,
        ),
    ],
)
def test_get_device_translator(array_ns, gt4py_device, expected_device):
    translator = ndarray_utils.get_device_translator(array_ns)
    assert translator(gt4py_device) == expected_device


@pytest.mark.parametrize(
    "array_ns",
    [
        ndarray_utils.np,
        pytest.param(ndarray_utils.cupy, marks=pytest.mark.requires_gpu),
    ],
)
def test_is_array_namespace(array_ns):
    assert ndarray_utils.is_array_namespace(array_ns)


@pytest.mark.requires_torch
@pytest.mark.parametrize(
    "gt4py_device, expected_device",
    [
        (None, None),
        (definitions.Device(definitions.DeviceType.CPU, 0), "cpu"),
        (definitions.Device(definitions.DeviceType.CUDA, 1), "cuda:1"),
        (definitions.Device(definitions.DeviceType.ROCM, 0), "cuda:0"),
    ],
)
def test_get_device_translator_torch(gt4py_device, expected_device):
    import torch

    translator = ndarray_utils.get_device_translator(torch)
    expected = None if expected_device is None else torch.device(expected_device)
    assert translator(gt4py_device) == expected


@pytest.mark.requires_torch
def test_is_array_namespace_torch():
    import torch

    assert ndarray_utils.is_array_namespace(torch)


@pytest.mark.requires_jax
@pytest.mark.parametrize("gt4py_device", [None, definitions.Device(definitions.DeviceType.CPU, 0)])
def test_get_device_translator_jax_cpu(gt4py_device):
    import jax
    import jax.numpy as jnp

    translator = ndarray_utils.get_device_translator(jnp)
    expected = None if gt4py_device is None else jax.local_devices(backend="cpu")[0]
    assert translator(gt4py_device) == expected


@pytest.mark.requires_jax
@pytest.mark.requires_gpu
@pytest.mark.parametrize("device_type", [definitions.DeviceType.CUDA, definitions.DeviceType.ROCM])
def test_get_device_translator_jax_gpu(device_type):
    import jax
    import jax.numpy as jnp

    try:
        gpus = jax.local_devices(backend="gpu")
    except RuntimeError:
        pytest.skip("no jax GPU device")
    translator = ndarray_utils.get_device_translator(jnp)
    assert translator(definitions.Device(device_type, 0)) == gpus[0]


@pytest.mark.requires_jax
def test_is_array_namespace_jax():
    import jax.numpy as jnp

    assert ndarray_utils.is_array_namespace(jnp)


@pytest.mark.parametrize("array_ns_name", ["numpy", "array_api_strict"])
@pytest.mark.parametrize(
    "scalar_type", [np.bool_, np.int32, np.int64, np.uint8, np.float32, np.float64]
)
def test_dtype_translation(array_ns_name, scalar_type):
    array_ns = importlib.import_module(array_ns_name)
    name = np.dtype(scalar_type).name
    array = array_ns.empty((2,), dtype=getattr(array_ns, name))

    assert ndarray_utils.namespace_dtype(array_ns, scalar_type) == getattr(array_ns, name)
    assert ndarray_utils.dtype_of(array) == definitions.dtype(scalar_type)


@pytest.mark.parametrize("dtype", [">f8", "<f8", ">i4", "<i4"])
def test_dtype_of_ignores_byte_order(dtype):
    assert ndarray_utils.dtype_of(np.zeros((2,), dtype=dtype)) == definitions.dtype(
        np.dtype(dtype).newbyteorder("=").type
    )


@pytest.mark.requires_jax
def test_dtype_of_jax_without_x64():
    import jax

    # `jax.enable_x64` replaced `jax.experimental.disable_x64` in newer JAX versions
    disable_x64 = (
        jax.enable_x64(False) if hasattr(jax, "enable_x64") else jax.experimental.disable_x64()
    )
    with disable_x64:
        array = jax.numpy.zeros((2,), dtype=jax.numpy.float32)
        assert ndarray_utils.dtype_of(array) == definitions.dtype(np.float32)


@pytest.mark.parametrize(
    "c_type, expected", [(np.longlong, np.int64), (np.ulonglong, np.uint64), (np.intc, np.int32)]
)
def test_dtype_of_c_type_aliases(c_type, expected):
    assert ndarray_utils.dtype_of(np.zeros((2,), dtype=c_type)) == definitions.dtype(expected)


@pytest.mark.parametrize("dtype", [np.float16, np.complex128])
def test_dtype_of_rejects_non_gt4py_types(dtype):
    with pytest.raises(TypeError, match="not a gt4py scalar type"):
        ndarray_utils.dtype_of(np.zeros((2,), dtype=dtype))


@pytest.mark.parametrize("array_ns_name", ["numpy", "array_api_strict"])
@pytest.mark.parametrize(
    "c_type, expected", [(np.longlong, "int64"), (np.ulonglong, "uint64"), (np.intc, "int32")]
)
def test_namespace_dtype_of_c_type_aliases(array_ns_name, c_type, expected):
    array_ns = importlib.import_module(array_ns_name)

    assert ndarray_utils.namespace_dtype(array_ns, c_type) == getattr(array_ns, expected)


_BASE = np.arange(5 * 6 * 7, dtype=np.float64).reshape(5, 6, 7)


@pytest.mark.parametrize("array", [_BASE, _BASE[1:4, 2:, ::-2]], ids=["contiguous", "view"])
def test_buffer_info_from_dlpack(array):
    info = ndarray_utils.BufferInfo.from_dlpack(array)

    assert info.data_ptr == array.__array_interface__["data"][0]
    assert info.shape == array.shape
    assert info.elem_strides == tuple(s // array.itemsize for s in array.strides)
    assert info.dtype == definitions.dtype(np.float64)
    assert info.device == definitions.Device(definitions.DeviceType.CPU, 0)


class _PatchedExport:
    """Exports `array` through DLPack after `patch` modified the capsule's `DLTensor`."""

    def __init__(self, array, patch):
        self.array = array
        self.patch = patch

    def __dlpack__(self, *, stream=None, max_version=None, copy=None):
        capsule = self.array.__dlpack__(max_version=(1, 0), copy=copy)
        address = dlpack._capsule_pointer(capsule, b"dltensor_versioned")
        self.patch(dlpack._DLTensor.from_address(address + dlpack._DL_TENSOR_OFFSET))
        return capsule

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()


def test_buffer_info_adds_byte_offset():
    def move_offset(tensor):
        tensor.data -= 16
        tensor.byte_offset = 16

    array = _BASE[1:4]

    info = ndarray_utils.BufferInfo.from_dlpack(_PatchedExport(array, move_offset))

    assert info.data_ptr == array.__array_interface__["data"][0]


def test_buffer_info_null_strides_are_c_contiguous():
    def null_strides(tensor):
        tensor.strides = type(tensor.strides)()

    info = ndarray_utils.BufferInfo.from_dlpack(_PatchedExport(_BASE, null_strides))

    assert info.elem_strides == (42, 7, 1)


def test_buffer_info_impossible_stride():
    def cupy_before_10228(tensor):
        # Stride -42 as CuPy before cupy#10228 exports it for float64.
        tensor.strides[0] = 2**64 // _BASE.itemsize - 42

    with pytest.raises(BufferError, match="cupy#10228"):
        ndarray_utils.BufferInfo.from_dlpack(_PatchedExport(_BASE, cupy_before_10228))


@pytest.mark.parametrize(
    "dtype",
    [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
    + [np.float32, np.float64],
)
def test_buffer_info_dtype(dtype):
    info = ndarray_utils.BufferInfo.from_dlpack(np.zeros(3, dtype=dtype))

    assert info.dtype == definitions.dtype(dtype)


@pytest.mark.parametrize("dtype", [np.float16, np.complex64])
def test_buffer_info_unsupported_dtype(dtype):
    with pytest.raises(TypeError, match="not a gt4py scalar type"):
        ndarray_utils.BufferInfo.from_dlpack(np.zeros(3, dtype=dtype))


@pytest.mark.parametrize(
    ("dlpack_device", "device"),
    [
        ((1, 0), definitions.Device(definitions.DeviceType.CPU, 0)),
        ((2, 1), definitions.Device(definitions.DeviceType.CUDA, 1)),
        ((10, 0), definitions.Device(definitions.DeviceType.ROCM, 0)),
        ((13, 0), definitions.Device(13, 0)),
    ],
    ids=["cpu", "cuda", "rocm", "unnamed_cuda_managed"],
)
def test_device_from_dlpack(dlpack_device, device):
    result = ndarray_utils.device_from_dlpack(*dlpack_device)

    assert result == device
    assert type(result.device_type) is type(device.device_type)


def test_buffer_info_hash_key_tells_apart_dtypes_at_the_same_address():
    def info(array):
        return ndarray_utils.BufferInfo.from_dlpack(array)

    a = np.zeros(4, dtype=np.float64)
    scalar = np.zeros((), dtype=np.float64)
    scalar_float32 = scalar.reshape(1).view(np.float32)[:1].reshape(())
    assert scalar_float32.__array_interface__["data"][0] == scalar.__array_interface__["data"][0]

    for x, y in [(a, a.view(np.int64)), (scalar, scalar_float32)]:
        assert info(x) != info(y)
        assert info(x).hash_key != info(y).hash_key

    assert info(a[:]) == info(a)
    assert info(a[:]).hash_key == info(a).hash_key
    assert hash(info(a)) == info(a).hash_key
