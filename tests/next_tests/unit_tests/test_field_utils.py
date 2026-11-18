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


def test_first_buffer_skips_fields_without_a_buffer():
    import numpy as np

    from gt4py.next.iterator import embedded as iterator_embedded

    I = common.Dimension("I")
    J = common.Dimension("J")
    field = common._field(np.arange(3.0), domain=common.domain({I: 3}))
    args = (
        iterator_embedded.IndexField(I),
        (iterator_embedded.ConstantField(1.0), 2.0),
        iterator_embedded.StridedConnectivityField([I, J], I, 2),
        field,
    )

    assert field_utils.first_buffer(*args) is field.ndarray


def test_first_buffer_without_a_buffer():
    from gt4py.next.iterator import embedded as iterator_embedded

    assert (
        field_utils.first_buffer(iterator_embedded.IndexField(common.Dimension("I")), 1.0) is None
    )
