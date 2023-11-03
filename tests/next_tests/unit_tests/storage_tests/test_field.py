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

from gt4py import next as gtx
from gt4py._core import definitions as core_defs
from gt4py.next import allocators as next_allocators, common
from gt4py.next.program_processors.runners import roundtrip


I = gtx.Dimension("I")
J = gtx.Dimension("J")


@gtx.field_operator
def add(
    a: gtx.Field[[I, J], gtx.float32], b: gtx.Field[[I, J], gtx.float32]
) -> gtx.Field[[I, J], gtx.float32]:
    return a + b


@gtx.program(backend=roundtrip.backend)
def prog(
    a: gtx.Field[[I, J], gtx.float32],
    b: gtx.Field[[I, J], gtx.float32],
    out: gtx.Field[[I, J], gtx.float32],
):
    add(a, b, out=out)


a = gtx.constructors.ones(
    common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
    dtype=core_defs.dtype(np.float32),
    allocator=prog,
)


arr = np.full((10, 10), 42.0)
b = gtx.constructors.as_field(
    common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
    arr,
    dtype=core_defs.dtype(np.float32),
    allocator=prog,
)

out = gtx.constructors.empty(
    common.Domain(dims=(I, J), ranges=(common.UnitRange(0, 10), common.UnitRange(0, 10))),
    dtype=core_defs.dtype(np.float32),
    allocator=prog,
)

prog(a, b, out, offset_provider={})

print(out.ndarray)
