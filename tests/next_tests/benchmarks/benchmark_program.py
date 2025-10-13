# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import pytest

from gt4py import next as gtx
from gt4py.next import Dims

from .. import definitions

if TYPE_CHECKING:
    from pytest_benchmark import fixture as ptb_fixture


IDim = gtx.Dimension("IDim")


@gtx.field_operator
def identity_fop(
    in_field: gtx.Field[Dims[IDim], gtx.float64],
) -> gtx.Field[Dims[IDim], gtx.float64]:
    return in_field


@gtx.program
def copy_program(
    in_field: gtx.Field[Dims[IDim], gtx.float64], out: gtx.Field[Dims[IDim], gtx.float64]
):
    identity_fop(in_field, out=out)


def benchmark_program(benchmark: ptb_fixture.BenchmarkFixture, program: Any = copy_program):
    size = 1000
    in_field = gtx.full([(IDim, size)], 1, dtype=gtx.float64)
    out_field = gtx.empty([(IDim, size)], dtype=gtx.float64)

    benchmark(program, in_field, out=out_field)
