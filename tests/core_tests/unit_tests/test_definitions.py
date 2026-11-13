# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from gt4py._core import definitions as core_defs


@pytest.mark.parametrize(
    "scalar_type, expected",
    [
        (bool, core_defs.DTypeKind.BOOL),
        (np.bool_, core_defs.DTypeKind.BOOL),
        (np.int32, core_defs.DTypeKind.INT),
        (np.uint16, core_defs.DTypeKind.UINT),
        (np.float32, core_defs.DTypeKind.FLOAT),
    ],
)
def test_dtype_kind(scalar_type, expected):
    assert core_defs.dtype(scalar_type).kind == expected
