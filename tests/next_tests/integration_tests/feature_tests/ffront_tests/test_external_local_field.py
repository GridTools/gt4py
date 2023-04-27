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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int64, neighbor_sum

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import *


def test_external_local_field(reduction_setup, fieldview_backend):
    V2EDim, V2E = reduction_setup.V2EDim, reduction_setup.V2E
    inp = np_as_located_field(Vertex, V2EDim)(reduction_setup.v2e_table)
    ones = np_as_located_field(Edge)(np.ones(reduction_setup.num_edges, dtype=int64))

    @field_operator(backend=fieldview_backend)
    def testee(
        inp: Field[[Vertex, V2EDim], int64], ones: Field[[Edge], int64]
    ) -> Field[[Vertex], int64]:
        return neighbor_sum(
            inp * ones(V2E), axis=V2EDim
        )  # multiplication with shifted `ones` because reduction of only non-shifted field with local dimension is not supported

    testee(inp, ones, out=reduction_setup.out, offset_provider=reduction_setup.offset_provider)

    ref = np.sum(reduction_setup.v2e_table, axis=1)
    assert np.allclose(ref, reduction_setup.out)


def test_external_local_field_only(reduction_setup, fieldview_backend):
    if fieldview_backend in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        pytest.skip(
            "Reductions over only a non-shifted field with local dimension is not supported in gtfn."
        )

    V2EDim, V2E = reduction_setup.V2EDim, reduction_setup.V2E
    inp = np_as_located_field(Vertex, V2EDim)(reduction_setup.v2e_table)

    @field_operator(backend=fieldview_backend)
    def testee(inp: Field[[Vertex, V2EDim], int64]) -> Field[[Vertex], int64]:
        return neighbor_sum(inp, axis=V2EDim)

    testee(inp, out=reduction_setup.out, offset_provider=reduction_setup.offset_provider)

    ref = np.sum(reduction_setup.v2e_table, axis=1)
    assert np.allclose(ref, reduction_setup.out)
