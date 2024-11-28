# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import gt4py.next as gtx


IDim = gtx.Dimension("IDim")
IOff = gtx.FieldOffset("IOff", source=IDim, target=(IDim,))


@gtx.field_operator
def fop(
    a: gtx.Field[[IDim], gtx.float64], b: gtx.Field[[IDim], gtx.float64]
) -> gtx.Field[[IDim], gtx.float64]:
    return a(IOff[1]) + b


@gtx.program
def prog(
    a: gtx.Field[[IDim], gtx.float64],
    b: gtx.Field[[IDim], gtx.float64],
    out: gtx.Field[[IDim], gtx.float64],
):
    fop(a, b, out=out)


def test_basic():
    a = gtx.as_field([(IDim, gtx.common.UnitRange(1, 5))], np.asarray([0.0, 1.0, 2.0, 3.0]))
    b = gtx.as_field([(IDim, gtx.common.UnitRange(0, 4))], np.asarray([0.0, 1.0, 2.0, 3.0]))
    out = gtx.as_field([(IDim, gtx.common.UnitRange(0, 4))], np.asarray([0.0, 0.0, 0.0, 0.0]))

    prog(a, b, out, offset_provider={"IOff": IDim})
    assert out.domain == b.domain
    assert np.allclose(out.ndarray, a.ndarray + b.ndarray)
