# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common
from gt4py.next.embedded import context as embedded_context
from gt4py.next.ffront import fbuiltins


V = gtx.Dimension("V")
E = gtx.Dimension("E")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)

_TABLE = np.array([[0, 1], [1, 2], [2, -1]], dtype=np.int32)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize(
    "reduction, np_reduction, neutral",
    [
        (fbuiltins.neighbor_sum, np.sum, lambda a: 0),
        (fbuiltins.max_over, np.max, np.min),
        (fbuiltins.min_over, np.min, np.max),
    ],
)
@pytest.mark.parametrize("skip_value", [None, -1])
def test_neighbor_reduction(reduction, np_reduction, neutral, dtype, skip_value):
    table = np.where(_TABLE < 0, 0, _TABLE) if skip_value is None else _TABLE
    conn = common._connectivity(table, codomain=V, domain={E: 3, E2VDim: 2}, skip_value=skip_value)
    values = np.arange(12, dtype=dtype).reshape(3, 2, 2) * dtype(3) - dtype(5)
    field = common._field(values, domain={E: 3, E2VDim: 2, K: 2})

    with embedded_context.update(offset_provider={"E2V": conn}):
        result = reduction(field, axis=E2VDim)

    masked = np.where((table != -1)[:, :, None], values, neutral(values))
    expected = np_reduction(masked, axis=1)
    assert result.domain == common.domain({E: 3, K: 2})
    np.testing.assert_array_equal(result.asnumpy(), expected)
