# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
