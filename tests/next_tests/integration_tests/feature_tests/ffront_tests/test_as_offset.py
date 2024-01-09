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

from functools import reduce

import numpy as np
import pytest

import gt4py.next as gtx

from gt4py.next.ffront.experimental import as_offset
from tests.next_tests.integration_tests import cases
from tests.next_tests.integration_tests.cases import (
    IDim,
    Ioff,
    JDim,
    KDim,
    Koff,
    cartesian_case,
    unstructured_case,
)
from tests.next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    fieldview_backend,
    reduction_setup,
)

@pytest.mark.uses_dynamic_offsets
def test_offset_field_no_exntend(cartesian_case):

    @gtx.field_operator
    def testee(
        a: gtx.Field[[IDim, KDim], int], offset_field: gtx.Field[[IDim, KDim], int]
    ) -> gtx.Field[[IDim, KDim], int]:
        return a(as_offset(Ioff, offset_field))

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    a = cases.allocate(cartesian_case, testee, "a")()
    offset_field = cases.allocate(cartesian_case, testee, "offset_field").strategy(
        cases.ConstInitializer(3)
    )()
    ref = a[3:]

    cases.verify(
        cartesian_case,
        testee,
        a,
        offset_field,
        out=out,
        offset_provider={"Ioff": IDim},
        ref=ref,
        comparison=lambda out, ref: np.all(out == ref),
    )

@pytest.mark.uses_dynamic_offsets
def test_offset_field_domain(cartesian_case):

    @gtx.field_operator
    def testee_fo(
        a: gtx.Field[[IDim, KDim], int], offset_field: gtx.Field[[IDim, KDim], int]
    ) -> gtx.Field[[IDim, KDim], int]:
        return a(as_offset(Ioff, offset_field))

    @gtx.program
    def testee(
        a: gtx.Field[[IDim, KDim], int], offset_field: gtx.Field[[IDim, KDim], int], out: gtx.Field[[IDim, KDim], int]
    ):
        testee_fo(a, offset_field, out=out, domain={IDim: (1, 5), KDim: (0, 10)})

    out = cases.allocate(cartesian_case, testee, "out")()
    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 3)})()
    offset_field = cases.allocate(cartesian_case, testee, "offset_field").strategy(
        cases.ConstInitializer(3)
    )()
    ref = a[3:]

    cases.verify(
        cartesian_case,
        testee,
        a,
        offset_field,
        out,
        inout=out,
        offset_provider={"Ioff": IDim},
        ref=ref[1:5],
        comparison=lambda out, ref: np.all(out == ref),
    )

@pytest.mark.parametrize(
    "dims, offset, offset_dim",
    [([IDim, KDim], Ioff, IDim), ([IDim, JDim], Ioff, IDim), ([IDim, KDim], Koff, KDim)],
)
@pytest.mark.uses_dynamic_offsets
def test_offset_field(cartesian_case, dims, offset, offset_dim):

    @gtx.field_operator
    def testee(
        a: gtx.Field[[dims[0], dims[1]], int], offset_field: gtx.Field[[dims[0], dims[1]], int]
    ) -> gtx.Field[[dims[0], dims[1]], int]:
        return a(as_offset(offset, offset_field))

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    a = cases.allocate(cartesian_case, testee, "a").extend({offset_dim: (0, 3)})()
    offset_field = cartesian_case.as_field([dims[0], dims[1]], np.random.randint(0, 4, size=(cartesian_case.default_sizes[dims[0]], cartesian_case.default_sizes[dims[1]])))
    # buffer_field =
    # ref = np.diagonal(a.asnumpy()[offset_field.asnumpy()]).T if offset_dim.kind == "vertical" else np.diagonal(a.asnumpy()[offset_field.asnumpy()].T)
    # offset_field = cases.allocate(cartesian_case, testee, "offset_field").strategy(
    #     cases.ConstInitializer(3)
    # )()
    ref = a[:, 3:] if offset_dim.kind == "vertical" else a[3:]

    cases.verify(
        cartesian_case,
        testee,
        a,
        offset_field,
        out=out,
        offset_provider={"offset": offset_dim, "Ioff": IDim, "Koff": KDim},
        ref=ref,
        comparison=lambda out, ref: np.all(out == ref),
    )


@pytest.mark.uses_dynamic_offsets
def test_offset_field_3d(cartesian_case):

    @gtx.field_operator
    def testee(
        a: cases.IJKField, offset_field: cases.IJKField
    ) -> cases.IJKField:
        return a(as_offset(Ioff, offset_field))

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 2)})()
    offset_field = cases.allocate(cartesian_case, testee, "offset_field").strategy(
        cases.ConstInitializer(2)
    )()

    cases.verify(
        cartesian_case,
        testee,
        a,
        offset_field,
        out=out,
        offset_provider={"Ioff": IDim, "Koff": KDim},
        ref=a[2:],
        comparison=lambda out, ref: np.all(out == ref),
    )