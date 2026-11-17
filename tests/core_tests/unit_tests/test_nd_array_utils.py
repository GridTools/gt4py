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

    assert info == ndarray_utils.BufferInfo(**dlpack.buffer_attributes(array))
    assert info.data_ptr == array.__array_interface__["data"][0]
    assert info.elem_strides == tuple(s // array.itemsize for s in array.strides)


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
