# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import textwrap

from gt4py.eve.utils import UIDGenerator
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.infer_domain_ops import InferDomainOps
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding

from next_tests.integration_tests.cases import IDim, JDim, KDim


def test_data():
    return [
        (
            im.less(im.axis_literal(IDim), 1),
            im.domain(common.GridType.CARTESIAN, {IDim: (itir.InfinityLiteral.NEGATIVE, 1)}),
        ),
        (
            im.less_equal(im.axis_literal(IDim), 1),
            im.domain(common.GridType.CARTESIAN, {IDim: (itir.InfinityLiteral.NEGATIVE, 2)}),
        ),
        (
            im.greater(im.axis_literal(IDim), 1),
            im.domain(common.GridType.CARTESIAN, {IDim: (2, itir.InfinityLiteral.POSITIVE)}),
        ),
        (
            im.greater_equal(im.axis_literal(IDim), 1),
            im.domain(common.GridType.CARTESIAN, {IDim: (1, itir.InfinityLiteral.POSITIVE)}),
        ),
        (
            im.less(1, im.axis_literal(IDim)),
            im.domain(common.GridType.CARTESIAN, {IDim: (2, itir.InfinityLiteral.POSITIVE)}),
        ),
        (
            im.less_equal(1, im.axis_literal(IDim)),
            im.domain(common.GridType.CARTESIAN, {IDim: (1, itir.InfinityLiteral.POSITIVE)}),
        ),
        (
            im.greater(1, im.axis_literal(IDim)),
            im.domain(common.GridType.CARTESIAN, {IDim: (itir.InfinityLiteral.NEGATIVE, 1)}),
        ),
        (
            im.greater_equal(1, im.axis_literal(IDim)),
            im.domain(common.GridType.CARTESIAN, {IDim: (itir.InfinityLiteral.NEGATIVE, 2)}),
        ),
        (im.eq(1, im.axis_literal(IDim)), im.domain(common.GridType.CARTESIAN, {IDim: (1, 2)})),
        (
            im.not_eq(1, im.axis_literal(IDim)),
            im.and_(
                im.domain(common.GridType.CARTESIAN, {IDim: (itir.InfinityLiteral.NEGATIVE, 1)}),
                im.domain(common.GridType.CARTESIAN, {IDim: (2, itir.InfinityLiteral.POSITIVE)}),
            ),
        ),
    ]


@pytest.mark.parametrize("testee,expected", test_data())
def test_trivial(testee, expected):
    actual = InferDomainOps(grid_type=common.GridType.CARTESIAN).visit(testee, recurse=True)
    actual = ConstantFolding.apply(actual)  # simplify expr to get simpler expected expressions
    assert actual == expected
