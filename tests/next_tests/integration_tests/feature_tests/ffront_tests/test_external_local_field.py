# -*- coding: utf-8 -*-
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
import pytest

import gt4py.next as gtx
from gt4py.next import int64, neighbor_sum
from gt4py.next.program_processors.runners import gtfn_cpu

from next_tests.integration_tests.feature_tests import cases
from next_tests.integration_tests.feature_tests.cases import (
    V2E,
    Edge,
    V2EDim,
    Vertex,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    fieldview_backend,
    reduction_setup,
)


def test_external_local_field(unstructured_case):
    @gtx.field_operator
    def testee(inp: gtx.Field[[Vertex, V2EDim], int64], ones: cases.EField) -> cases.VField:
        return neighbor_sum(
            inp * ones(V2E), axis=V2EDim
        )  # multiplication with shifted `ones` because reduction of only non-shifted field with local dimension is not supported

    inp = gtx.np_as_located_field(Vertex, V2EDim)(unstructured_case.offset_provider["V2E"].table)
    ones = cases.allocate(unstructured_case, testee, "ones").strategy(cases.ConstInitializer(1))()

    cases.verify(
        unstructured_case,
        testee,
        inp,
        ones,
        out=cases.allocate(unstructured_case, testee, cases.RETURN)(),
        ref=np.sum(unstructured_case.offset_provider["V2E"].table, axis=1),
    )


def test_external_local_field_only(unstructured_case):
    if unstructured_case.backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip(
            "Reductions over only a non-shifted field with local dimension is not supported in gtfn."
        )

    @gtx.field_operator
    def testee(inp: gtx.Field[[Vertex, V2EDim], int64]) -> cases.VField:
        return neighbor_sum(inp, axis=V2EDim)

    inp = gtx.np_as_located_field(Vertex, V2EDim)(unstructured_case.offset_provider["V2E"].table)

    cases.verify(
        unstructured_case,
        testee,
        inp,
        out=cases.allocate(unstructured_case, testee, cases.RETURN)(),
        ref=np.sum(unstructured_case.offset_provider["V2E"].table, axis=1),
    )
