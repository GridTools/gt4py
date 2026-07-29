# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import ast
import textwrap

import pytest

from gt4py.next import Dimension, DimensionKind, Field, field_operator, int32, int64, scan_operator
from gt4py.next.ffront.ast_passes import single_static_assign as ssa
from gt4py.next.ffront.foast_pretty_printer import pretty_format
from gt4py.next.ffront.func_to_foast import FieldOperatorParser


@pytest.mark.parametrize(
    "test_case",
    [
        "1 + (2 * 3) ** (4 * 2) + 2",
        "(2 ** 3) ** 4",
        "((2 ** 3) ** 3) ** 4",
        "1 if True else 2",
        "(1 if True else 2) if True else 3",
        "()",
        "(1,)",
        "(1, 2)",
        "--1",
        "1 <= 10",
        "not True",
        "~True",
        "True & False",
        "True | False",
    ],
)
def test_one_to_one(test_case: str):
    from gt4py.next.ffront.source_utils import SourceDefinition

    case_source_def = SourceDefinition(test_case)
    foast_node = FieldOperatorParser.apply(case_source_def, {}, {})
    assert pretty_format(foast_node) == test_case


def test_fieldop():
    I = Dimension("I")

    @field_operator
    def foo(inp1: Field[[I], int64], inp2: Field[[I], int64]):
        return inp1 + inp2

    @field_operator
    def bar(inp1: Field[[I], int64], inp2: Field[[I], int64]) -> Field[[I], int64]:
        return foo(inp1, inp2=inp2)

    expected = textwrap.dedent(
        """
        @field_operator
        def bar(inp1: Field[[I], int64], inp2: Field[[I], int64]) -> Field[[I], int64]:
          return foo(inp1, inp2=inp2)
        """
    ).strip()

    assert pretty_format(bar.foast_stage.foast_node) == expected


def test_scanop():
    KDim = Dimension("KDim", kind=DimensionKind.VERTICAL)

    @scan_operator(axis=KDim, forward=False, init=1)
    def scan(inp: int32) -> int32:
        foo = inp
        return inp

    expected = textwrap.dedent(
        f"""
        @scan_operator(axis=KDim[vertical], forward=False, init=1)
        def scan(inp: int32) -> int32:
          {ssa.unique_name("foo", 0)} = inp
          return inp
        """
    ).strip()

    assert pretty_format(scan.foast_stage.foast_node) == expected
