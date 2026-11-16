# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from gt4py._core import definitions as core_defs, dlpack


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


def test_from_dlpack():
    array = _BASE[1:4, 2:, ::-2]

    info = dlpack.BufferInfo.from_dlpack(array)

    assert isinstance(info, dlpack.BufferInfo)

    assert info.data_ptr == array.__array_interface__["data"][0]
    assert info.shape == array.shape
    assert info.elem_strides == tuple(s // array.itemsize for s in array.strides)
    assert info.dtype == core_defs.dtype(np.float64)
    assert info.device == core_defs.Device(core_defs.DeviceType.CPU, 0)


def test_no_dlpack():
    with pytest.raises(TypeError, match="DLPack"):
        dlpack.BufferInfo.from_dlpack(object())


def test_producer_without_max_version():
    def only_stream(array, *, stream=None):
        return array.__dlpack__(stream=stream)

    array = _BASE[1:4]

    assert dlpack.BufferInfo.from_dlpack(_Wrapper(array, only_stream)) == dlpack.BufferInfo.from_dlpack(array)


def test_unversioned_capsule():
    def unversioned(array, *, stream=None, max_version=None, copy=None):
        return array.__dlpack__(stream=stream)

    array = _BASE[1:4]

    assert dlpack.BufferInfo.from_dlpack(_Wrapper(array, unversioned)) == dlpack.BufferInfo.from_dlpack(array)


def test_unsupported_dtype():
    with pytest.raises(TypeError, match="not a gt4py scalar type"):
        dlpack.BufferInfo.from_dlpack(np.zeros(3, dtype=np.float16))


def test_impossible_stride():
    def cupy_before_10228(array, *, stream=None, max_version=None, copy=None):
        capsule = array.__dlpack__(max_version=(1, 0), copy=copy)
        address = dlpack._capsule_pointer(capsule, b"dltensor_versioned")
        tensor = dlpack._DLTensor.from_address(address + dlpack._DL_TENSOR_OFFSET)
        # Stride -42 as CuPy before cupy#10228 exports it for float64.
        tensor.strides[0] = 2**64 // array.itemsize - 42
        return capsule

    with pytest.raises(BufferError, match="cupy#10228"):
        dlpack.BufferInfo.from_dlpack(_Wrapper(_BASE, cupy_before_10228))


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

    dlpack.BufferInfo.from_dlpack(Producer())

    assert streams == [stream]


def test_hash_key_tells_apart_dtypes_at_the_same_address():
    a = np.zeros(4, dtype=np.float64)
    scalar = np.zeros((), dtype=np.float64)
    scalar_float32 = scalar.reshape(1).view(np.float32)[:1].reshape(())
    assert scalar_float32.__array_interface__["data"][0] == scalar.__array_interface__["data"][0]

    for x, y in [(a, a.view(np.int64)), (scalar, scalar_float32)]:
        assert dlpack.BufferInfo.from_dlpack(x) != dlpack.BufferInfo.from_dlpack(y)
        assert dlpack.BufferInfo.from_dlpack(x).hash_key != dlpack.BufferInfo.from_dlpack(y).hash_key

    assert dlpack.BufferInfo.from_dlpack(a[:]) == dlpack.BufferInfo.from_dlpack(a)
    assert dlpack.BufferInfo.from_dlpack(a[:]).hash_key == dlpack.BufferInfo.from_dlpack(a).hash_key
    assert hash(dlpack.BufferInfo.from_dlpack(a)) == dlpack.BufferInfo.from_dlpack(a).hash_key
