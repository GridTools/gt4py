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
from gt4py.next import common, int32, neighbor_sum

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import V2E, Edge, V2EDim, Vertex, unstructured_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


pytestmark = pytest.mark.uses_unstructured_shift


def test_external_local_field(unstructured_case):
    @gtx.field_operator
    def testee(
        inp: gtx.Field[[Vertex, V2EDim], int32], ones: gtx.Field[[Edge], int32]
    ) -> gtx.Field[[Vertex], int32]:
        return neighbor_sum(
            inp * ones(V2E), axis=V2EDim
        )  # multiplication with shifted `ones` because reduction of only non-shifted field with local dimension is not supported

    inp = unstructured_case.as_field(
        [Vertex, V2EDim], unstructured_case.offset_provider["V2E"].table
    )
    ones = cases.allocate(unstructured_case, testee, "ones").strategy(cases.ConstInitializer(1))()

    v2e_table = unstructured_case.offset_provider["V2E"].table
    cases.verify(
        unstructured_case,
        testee,
        inp,
        ones,
        out=cases.allocate(unstructured_case, testee, cases.RETURN)(),
        ref=np.sum(v2e_table, axis=1, initial=0, where=v2e_table != common._DEFAULT_SKIP_VALUE),
    )


@pytest.mark.skip(
    "Reductions over only a non-shifted field with local dimension is not supported"
)  # we keep the test in case we will change that in the future
def test_external_local_field_only(unstructured_case):
    @gtx.field_operator
    def testee(inp: gtx.Field[[Vertex, V2EDim], int32]) -> gtx.Field[[Vertex], int32]:
        return neighbor_sum(inp, axis=V2EDim)

    inp = unstructured_case.as_field(
        [Vertex, V2EDim], unstructured_case.offset_provider["V2E"].table
    )

    cases.verify(
        unstructured_case,
        testee,
        inp,
        out=cases.allocate(unstructured_case, testee, cases.RETURN)(),
        ref=np.sum(unstructured_case.offset_provider["V2E"].table, axis=1),
    )


@pytest.mark.uses_sparse_fields_as_output
def test_write_local_field(unstructured_case):
    @gtx.field_operator
    def testee(inp: gtx.Field[[Edge], int32]) -> gtx.Field[[Vertex, V2EDim], int32]:
        return inp(V2E)

    out = unstructured_case.as_field(
        [Vertex, V2EDim], np.zeros_like(unstructured_case.offset_provider["V2E"].table)
    )
    inp = cases.allocate(unstructured_case, testee, "inp")()
    cases.verify(
        unstructured_case,
        testee,
        inp,
        out=out,
        ref=inp.asnumpy()[unstructured_case.offset_provider["V2E"].table],
    )
