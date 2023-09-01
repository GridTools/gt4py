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

import cupy as cp
import numpy as np

from gt4py import next as gtx
from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.program_processors.runners import roundtrip
from gt4py.next.storage import allocators as next_allocators


I = gtx.Dimension("I")
J = gtx.Dimension("J")

a = gtx.field.empty(
    common.Domain(dims=(I, J), ranges=(common.UnitRange(10, 20), common.UnitRange(30, 40))),
    dtype=core_defs.dtype(float),
)


arr = cp.full((10, 10), 42.0)
b = gtx.field.asfield(
    common.Domain(dims=(I, J), ranges=(common.UnitRange(10, 20), common.UnitRange(30, 40))),
    arr,
    dtype=core_defs.dtype(float),
    allocator=next_allocators.DefaultCUDAAllocator(),
)

out = gtx.field.asfield(
    common.Domain(dims=(I, J), ranges=(common.UnitRange(10, 20), common.UnitRange(30, 40))),
    arr,
    dtype=core_defs.dtype(float),
    allocator=next_allocators.DefaultCUDAAllocator(),
)


assert isinstance(a.ndarray, np.ndarray)
assert isinstance(b.ndarray, cp.ndarray)
assert b.ndarray[0, 0] == 42.0, b.ndarray[0, 0]


@gtx.field_operator
def add(
    a: gtx.Field[[I, J], gtx.float32], b: gtx.Field[[I, J], gtx.float32]
) -> gtx.Field[[I, J], gtx.float32]:
    return a + b


@gtx.program(backend=roundtrip.executor)
def prog(
    a: gtx.Field[[I, J], gtx.float32],
    b: gtx.Field[[I, J], gtx.float32],
    out: gtx.Field[[I, J], gtx.float32],
):
    add(a, b, out=out)


prog(a, b, out, offset_provider={})
