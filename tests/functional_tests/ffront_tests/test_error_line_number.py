import pytest

from functional.common import Dimension, Field
from functional.ffront.func_to_foast import FieldOperatorParser, FieldOperatorSyntaxError


# NOTE: This test is sensitive to filename and the line number of the marked statement

TDim = Dimension("TDim")  # Meaningless dimension, used for tests.


def test_invalid_syntax_error_empty_return():
    """Field operator syntax errors point to the file, line and column."""

    def wrong_syntax(inp: Field[[TDim], float]):
        return  # <-- this line triggers error

    with pytest.raises(
        FieldOperatorSyntaxError,
        match=(
            r"Invalid Field Operator Syntax: "
            r"Empty return not allowed \(test_error_line_number.py, line 16\)"
        ),
    ):
        _ = FieldOperatorParser.apply_to_function(wrong_syntax)
