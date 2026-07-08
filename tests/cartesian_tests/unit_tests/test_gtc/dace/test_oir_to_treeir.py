# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.cartesian.gtc.dace import oir_to_treeir
from gt4py.cartesian.gtc import common
from gt4py.cartesian.stencil_builder import StencilBuilder
from gt4py.cartesian.gtscript import PARALLEL, Field, computation, interval

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable adds the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


def ignore_me(field: Field[float]):  # type: ignore
    with computation(PARALLEL), interval(...):
        field += 1


@pytest.mark.parametrize(
    "node,axis_start,axis_end,expected",
    [
        (common.AxisBound.from_start(1), "2", "n/a", "(2) + (1)"),
        (common.AxisBound.from_start(0), "2", "n/a", "(2)"),
        (common.AxisBound.from_start(1), "0", "n/a", "(1)"),
        (common.AxisBound.from_start(0), "0", "n/a", "(0)"),
        (common.AxisBound.from_end(1), "n/a", "2", "(2) + (1)"),
        (common.AxisBound.from_end(0), "n/a", "2", "(2)"),
        (common.AxisBound.from_end(1), "n/a", "0", "(1)"),
        (common.AxisBound.from_end(0), "n/a", "0", "(0)"),
    ],
)
def test_visit_AxisBound(
    node: common.AxisBound, axis_start: str, axis_end: str, expected: str
) -> None:
    builder = StencilBuilder(ignore_me)
    visitor = oir_to_treeir.OIRToTreeIR(builder)

    assert visitor.visit_AxisBound(node, axis_start, axis_end) == expected
