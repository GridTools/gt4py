# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.cartesian import backend
from gt4py.storage.cartesian import layout

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable adds the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


@pytest.mark.parametrize(
    ["name", "device"],
    [
        ("dace:cpu", "cpu"),
        ("dace:cpu_kfirst", "cpu"),
        pytest.param("dace:gpu", "gpu", marks=[pytest.mark.requires_gpu]),
    ],
)
def test_dace_backend(name: str, device: str):
    dace_backend = backend.from_name(name)

    assert dace_backend.storage_info["device"] == device
