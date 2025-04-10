# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py._core import definitions as core_defs
from gt4py.next import common, constructors, field_utils


@pytest.mark.parametrize(
    "device_type",
    [
        core_defs.DeviceType.CPU,
        pytest.param(core_defs.DeviceType.CUDA, marks=pytest.mark.requires_gpu),
        pytest.param(core_defs.DeviceType.ROCM, marks=pytest.mark.requires_gpu),
    ],
)
def test_verify_device_field_type(nd_array_implementation_and_device_type, device_type):
    nd_array_implementation, compatible_device_type = nd_array_implementation_and_device_type

    testee = constructors.as_field([common.Dimension("X")], nd_array_implementation.asarray([42.0]))

    is_correct_device = compatible_device_type == device_type
    assert field_utils.verify_device_field_type(testee, device_type) == is_correct_device
