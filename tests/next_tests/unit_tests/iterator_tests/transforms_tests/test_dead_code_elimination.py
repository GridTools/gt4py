# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next import common
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import dead_code_elimination

TDim = common.Dimension(value="TDim")
int_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
field_type = ts.FieldType(dims=[TDim], dtype=int_type)


def program_factory(expr: itir.Expr) -> itir.Program:
    return itir.Program(
        id="testee",
        function_definitions=[],
        params=[
            im.sym("inp1", field_type),
            im.sym("inp2", field_type),
            im.sym("out", field_type),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=expr,
                target=im.ref("out"),
                domain=im.domain(common.GridType.CARTESIAN, {TDim: (0, 1)}),
            )
        ],
    )


@pytest.mark.parametrize(
    "input,expected",
    [(im.let("val", "inp1")(im.if_(True, "val", "inp2")), im.ref("inp1"))],
)
def test_let_constant_foldable_if(input, expected):
    input_program = program_factory(input)
    inlined = dead_code_elimination.dead_code_elimination(input_program, offset_provider_type={})
    assert inlined == program_factory(expected)
