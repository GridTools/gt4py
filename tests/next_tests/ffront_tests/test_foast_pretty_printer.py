import ast
import textwrap

import pytest

from gt4py.next.common import Dimension, DimensionKind, Field
from gt4py.next.ffront.decorator import field_operator, scan_operator
from gt4py.next.ffront.fbuiltins import int64
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
    @field_operator
    def foo(inp: Field[[], int64]):
        return inp

    @field_operator
    def bar(inp: Field[[], int64]) -> Field[[], int64]:
        return foo(inp)

    expected = textwrap.dedent(
        """
        @field_operator
        def bar(inp: Field[[], int64]) -> Field[[], int64]:
          return foo(inp)
        """
    ).strip()

    assert pretty_format(bar.foast_node) == expected


def test_scanop():
    KDim = Dimension("KDim", kind=DimensionKind.VERTICAL)

    @scan_operator(axis=KDim, forward=False, init=1.0)
    def scan(inp: int64) -> int64:
        foo = inp
        return inp

    expected = textwrap.dedent(
        """
        @scan_operator(axis=Dimension(value="KDim", kind=DimensionKind.VERTICAL), forward=False, init=1.0)
        def scan(inp: int64) -> int64:
          foo__0 = inp
          return inp
        """
    ).strip()

    assert pretty_format(scan.foast_node) == expected
