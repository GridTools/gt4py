# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from gt4py._core import dlpack


_BASE = np.arange(5 * 6 * 7, dtype=np.float64).reshape(5, 6, 7)


class _Wrapper:
    """Producer exporting `array` through `__dlpack__` with only the given keyword arguments."""

    def __init__(self, array, dlpack_fn):
        self.array = array
        self.dlpack_fn = dlpack_fn

    def __dlpack__(self, **kwargs):
        return self.dlpack_fn(self.array, **kwargs)

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()


def test_tensor_fields():
    array = _BASE[1:4, 2:, ::-2]

    # NumPy folds the view offset into `data` and exports `byte_offset == 0`.
    assert dlpack.tensor_fields(array) == {
        "data": array.__array_interface__["data"][0],
        "device": (dlpack.DeviceType.CPU, 0),
        "ndim": 3,
        "dtype": (dlpack.DataTypeCode.FLOAT, 64, 1),
        "shape": array.shape,
        "strides": tuple(s // array.itemsize for s in array.strides),
        "byte_offset": 0,
    }


def test_null_strides():
    def null_strides(array, *, stream=None, max_version=None, copy=None):
        capsule = array.__dlpack__(max_version=(1, 0), copy=copy)
        address = dlpack._capsule_pointer(capsule, b"dltensor_versioned")
        tensor = dlpack._DLTensor.from_address(address + dlpack._DL_TENSOR_OFFSET)
        tensor.strides = type(tensor.strides)()
        return capsule

    assert dlpack.tensor_fields(_Wrapper(_BASE, null_strides))["strides"] is None


@pytest.mark.parametrize(
    ("dtype", "code", "bits"),
    [
        (np.bool_, 6, 8),
        (np.int8, 0, 8),
        (np.uint16, 1, 16),
        (np.int32, 0, 32),
        (np.float16, 2, 16),
        (np.complex64, 5, 64),
    ],
)
def test_dtype_codes(dtype, code, bits):
    assert dlpack.tensor_fields(np.zeros(3, dtype=dtype))["dtype"] == (code, bits, 1)


def test_no_dlpack():
    with pytest.raises(TypeError, match="DLPack"):
        dlpack.tensor_fields(object())


def test_producer_without_max_version():
    def only_stream(array, *, stream=None):
        return array.__dlpack__(stream=stream)

    array = _BASE[1:4]

    assert dlpack.tensor_fields(_Wrapper(array, only_stream)) == dlpack.tensor_fields(array)


def test_unversioned_capsule():
    def unversioned(array, *, stream=None, max_version=None, copy=None):
        return array.__dlpack__(stream=stream)

    array = _BASE[1:4]

    assert dlpack.tensor_fields(_Wrapper(array, unversioned)) == dlpack.tensor_fields(array)


@pytest.mark.parametrize(
    ("device_type", "stream"),
    [(1, None), (2, -1), (10, -1), (3, None), (13, None), (14, None)],
    ids=["cpu", "cuda", "rocm", "cuda_host", "cuda_managed", "oneapi"],
)
def test_stream_argument(device_type, stream):
    streams = []

    class Producer:
        def __dlpack__(self, *, stream=None, max_version=None, copy=None):
            streams.append(stream)
            return _BASE.__dlpack__(max_version=max_version, copy=copy)

        def __dlpack_device__(self):
            return device_type, 0

    dlpack.tensor_fields(Producer())

    assert streams == [stream]
