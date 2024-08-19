# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from gt4py import next as gtx
from gt4py.next import common, neighbor_sum

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    C2E,
    E2V,
    V2E,
    E2VDim,
    IDim,
    Ioff,
    V2EDim,
    cartesian_case,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


@pytest.mark.uses_cartesian_shift
def test_cartesian_shift(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        return a(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[1:])


@pytest.mark.uses_unstructured_shift
def test_unstructured_shift(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.VField) -> cases.EField:
        return a(E2V[0])

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: a[unstructured_case.offset_provider["E2V"].table[:, 0]],
    )


@pytest.mark.uses_unstructured_shift
def test_unstructured_shift_offset_symbol(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.VField) -> cases.EField:
        e2v0 = E2V[0]
        return a(e2v0)

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: a[unstructured_case.offset_provider["E2V"].table[:, 0]],
    )


@pytest.mark.uses_unstructured_shift
def test_composed_unstructured_shift(unstructured_case):
    @gtx.field_operator
    def composed_shift_unstructured_flat(inp: cases.VField) -> cases.CField:
        return inp(E2V[0])(C2E[0])

    @gtx.field_operator
    def composed_shift_unstructured_intermediate_result(inp: cases.VField) -> cases.CField:
        tmp = inp(E2V[0])
        return tmp(C2E[0])

    @gtx.field_operator
    def shift_e2v(inp: cases.VField) -> cases.EField:
        return inp(E2V[0])

    @gtx.field_operator
    def composed_shift_unstructured(inp: cases.VField) -> cases.CField:
        return shift_e2v(inp)(C2E[0])

    cases.verify_with_default_data(
        unstructured_case,
        composed_shift_unstructured_flat,
        ref=lambda inp: inp[unstructured_case.offset_provider["E2V"].table[:, 0]][
            unstructured_case.offset_provider["C2E"].table[:, 0]
        ],
    )

    cases.verify_with_default_data(
        unstructured_case,
        composed_shift_unstructured_intermediate_result,
        ref=lambda inp: inp[unstructured_case.offset_provider["E2V"].table[:, 0]][
            unstructured_case.offset_provider["C2E"].table[:, 0]
        ],
        comparison=lambda inp, tmp: np.all(inp == tmp),
    )

    cases.verify_with_default_data(
        unstructured_case,
        composed_shift_unstructured,
        ref=lambda inp: inp[unstructured_case.offset_provider["E2V"].table[:, 0]][
            unstructured_case.offset_provider["C2E"].table[:, 0]
        ],
    )


@pytest.mark.uses_cartesian_shift
def test_fold_shifts(cartesian_case):
    """Shifting the result of an addition should work."""

    @gtx.field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
        tmp = a + b(Ioff[1])
        return tmp(Ioff[1])

    a = cases.allocate(cartesian_case, testee, "a").extend({cases.IDim: (0, 1)})()
    b = cases.allocate(cartesian_case, testee, "b").extend({cases.IDim: (0, 2)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, b, out=out, ref=a.ndarray[1:] + b.ndarray[2:])


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_reduction_over_lift_expressions
def test_nested_reduction(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.VField) -> cases.VField:
        tmp = neighbor_sum(a(E2V), axis=E2VDim)
        tmp_2 = neighbor_sum(tmp(V2E), axis=V2EDim)
        return tmp_2

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.sum(
            np.sum(a[unstructured_case.offset_provider["E2V"].table], axis=1, initial=0)[
                unstructured_case.offset_provider["V2E"].table
            ],
            axis=1,
            where=unstructured_case.offset_provider["V2E"].table != common._DEFAULT_SKIP_VALUE,
        ),
        comparison=lambda a, tmp_2: np.all(a == tmp_2),
    )


@pytest.mark.uses_unstructured_shift
@pytest.mark.xfail(reason="Not yet supported in lowering, requires `map_`ing of inner reduce op.")
def test_nested_reduction_shift_first(unstructured_case):
    @gtx.field_operator
    def testee(inp: cases.EField) -> cases.EField:
        tmp = inp(V2E)
        tmp2 = tmp(E2V)
        return neighbor_sum(neighbor_sum(tmp2, axis=V2EDim), axis=E2VDim)

    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda inp: np.sum(
            np.sum(inp[unstructured_case.offset_provider["V2E"].table], axis=1)[
                unstructured_case.offset_provider["E2V"].table
            ],
            axis=1,
        ),
    )
