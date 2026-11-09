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

from gt4py._core import ndarray_utils, definitions


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


def test_dtype_of_unsupported_dtype():
    with pytest.raises(TypeError, match="not a gt4py scalar type"):
        ndarray_utils.dtype_of(np.empty((2,), dtype=np.complex128))


@pytest.mark.parametrize("dtype", [">f8", "<f8", ">i4", "<i4"])
def test_dtype_of_ignores_byte_order(dtype):
    assert ndarray_utils.dtype_of(np.zeros((2,), dtype=dtype)) == definitions.dtype(
        np.dtype(dtype).newbyteorder("=").type
    )


@pytest.mark.requires_jax
def test_dtype_of_jax_without_x64():
    import jax

    with jax.enable_x64(False):
        array = jax.numpy.zeros((2,), dtype=jax.numpy.float32)
        assert ndarray_utils.dtype_of(array) == definitions.dtype(np.float32)
