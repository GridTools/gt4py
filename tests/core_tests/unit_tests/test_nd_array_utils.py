# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from gt4py._core import ndarray_utils, definitions
import pytest


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
