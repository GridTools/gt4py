# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.cartesian.gtc import oir
from gt4py.cartesian.gtc.passes.oir_optimizations.temporaries import (
    LocalTemporariesToScalars,
    WriteBeforeReadTemporariesToScalars,
)

from ...oir_utils import (
    AssignStmtFactory,
    HorizontalExecutionFactory,
    StencilFactory,
    TemporaryFactory,
)


def test_local_temporaries_to_scalars_basic():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions__0__body=[
            AssignStmtFactory(left__name="tmp"),
            AssignStmtFactory(right__name="tmp"),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = LocalTemporariesToScalars().visit(testee)
    hexec = transformed.vertical_loops[0].sections[0].horizontal_executions[0]
    assert isinstance(hexec.body[0].left, oir.ScalarAccess)
    assert isinstance(hexec.body[1].right, oir.ScalarAccess)
    assert not transformed.declarations
    assert len(hexec.declarations) == 1


def test_local_temporaries_to_scalars_multiexec():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="tmp"), AssignStmtFactory(right__name="tmp")]
            ),
            HorizontalExecutionFactory(body=[AssignStmtFactory(right__name="tmp")]),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = LocalTemporariesToScalars().visit(testee)
    assert "tmp" in {d.name for d in transformed.declarations}
    assert not transformed.walk_values().if_isinstance(oir.ScalarAccess).to_list()


def test_write_before_read_temporaries_to_scalars():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(left__name="tmp1"),
                    AssignStmtFactory(left__name="tmp2", right__name="tmp1"),
                    AssignStmtFactory(left__name="tmp3", right__name="tmp2"),
                ]
            ),
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(right__name="tmp2"),
                    AssignStmtFactory(left__name="tmp3"),
                    AssignStmtFactory(left__name="foo", right__name="tmp3"),
                ]
            ),
        ],
        declarations=[
            TemporaryFactory(name="tmp1"),
            TemporaryFactory(name="tmp2"),
            TemporaryFactory(name="tmp3"),
        ],
    )
    transformed = WriteBeforeReadTemporariesToScalars().visit(testee)
    hexec0 = transformed.vertical_loops[0].sections[0].horizontal_executions[0]
    hexec1 = transformed.vertical_loops[0].sections[0].horizontal_executions[1]
    assert len(hexec0.declarations) == 2
    assert len(hexec1.declarations) == 1
    assert len(transformed.declarations) == 1
    assert isinstance(hexec0.body[0].left, oir.ScalarAccess)
    assert not isinstance(hexec0.body[1].left, oir.ScalarAccess)
    assert isinstance(hexec0.body[1].right, oir.ScalarAccess)
    assert isinstance(hexec0.body[2].left, oir.ScalarAccess)
    assert not isinstance(hexec0.body[2].right, oir.ScalarAccess)
    assert not isinstance(hexec1.body[0].right, oir.ScalarAccess)
    assert isinstance(hexec1.body[1].left, oir.ScalarAccess)
    assert isinstance(hexec1.body[2].right, oir.ScalarAccess)
