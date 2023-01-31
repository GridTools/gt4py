# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import inspect
import traceback

import pytest

from gt4py.next import common
from gt4py.next.common import Dimension, Field
from gt4py.next.ffront import func_to_foast as f2f, source_utils as src_utils
from gt4py.next.ffront.foast_passes import type_deduction
from gt4py.next.ffront.func_to_foast import FieldOperatorParser, FieldOperatorSyntaxError


# NOTE: These tests are sensitive to filename and the line number of the marked statement

TDim = Dimension("TDim")  # Meaningless dimension, used for tests.


def test_invalid_syntax_error_empty_return():
    """Field operator syntax errors point to the file, line and column."""

    line = inspect.getframeinfo(inspect.currentframe()).lineno

    def wrong_syntax(inp: common.Field[[TDim], float]):
        return  # <-- this line triggers the syntax error

    with pytest.raises(
        f2f.FieldOperatorSyntaxError,
        match=(
            r"Invalid Field Operator Syntax: "
            r"Empty return not allowed \(test_error_line_number.py, line " + str(line + 3) + r"\)"
        ),
    ) as exc_info:
        _ = f2f.FieldOperatorParser.apply_to_function(wrong_syntax)

    assert traceback.format_exception_only(exc_info.value)[1:3] == [
        "    return  # <-- this line triggers the syntax error\n",
        "    ^^^^^^\n",
    ]


def test_wrong_caret_placement_bug():
    """Field operator syntax errors respect python's carets (`^^^^^`) placement."""

    line = inspect.getframeinfo(inspect.currentframe()).lineno

    def wrong_line_syntax_error(inp: common.Field[[TDim], float]):
        # the next line triggers the syntax error
        inp = inp.this_attribute_surely_doesnt_exist

        return inp

    with pytest.raises(f2f.FieldOperatorSyntaxError) as exc_info:
        _ = f2f.FieldOperatorParser.apply_to_function(wrong_line_syntax_error)

    exc = exc_info.value

    assert (exc.lineno, exc.end_lineno) == (line + 4, line + 4)

    # if `offset` is set, python will display carets (`^^^^`) after printing `text`.
    # So `text` has to be the line where the error occurs (otherwise the carets
    # will be very misleading).

    # See https://github.com/python/cpython/blob/6ad47b41a650a13b4a9214309c10239726331eb8/Lib/traceback.py#L852-L855
    python_printed_text = exc.text.rstrip("\n").lstrip(" \n\f")

    assert python_printed_text == "inp = inp.this_attribute_surely_doesnt_exist"

    # test that `offset` is aligned with `exc.text`
    return_offset = (
        exc.text.find("inp.this_attribute_surely_doesnt_exist") + 1
    )  # offset is 1-based for syntax errors
    assert (exc.offset, exc.end_offset) == (return_offset, return_offset + 38)

    print("".join(traceback.format_exception_only(exc)))

    assert traceback.format_exception_only(exc)[1:3] == [
        "    inp = inp.this_attribute_surely_doesnt_exist\n",
        "          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
    ]


def test_syntax_error_without_function():
    """Dialect parsers report line numbers correctly when applied to `SourceDefinition`."""

    source_definition = src_utils.SourceDefinition(
        starting_line=62,
        source="""
            def invalid_python_syntax():
                # This function contains a python syntax error

                ret%%  # <-- this line triggers the syntax error
        """,
    )

    with pytest.raises(SyntaxError) as exc_info:
        _ = f2f.FieldOperatorParser.apply(source_definition, {}, {})

    exc = exc_info.value

    assert (exc.lineno, exc.end_lineno) == (66, 66)

    assert traceback.format_exception_only(exc)[1:3] == [
        "    ret%%  # <-- this line triggers the syntax error\n",
        "        ^\n",
    ]


def test_fo_type_deduction_error():
    """Field operator type deduction errors report location correctly"""

    line = inspect.getframeinfo(inspect.currentframe()).lineno

    def field_operator_with_undeclared_symbol():
        return undeclared_symbol

    with pytest.raises(type_deduction.FieldOperatorTypeDeductionError) as exc_info:
        _ = f2f.FieldOperatorParser.apply_to_function(field_operator_with_undeclared_symbol)

    exc = exc_info.value

    assert (exc.lineno, exc.end_lineno) == (line + 3, line + 3)

    assert traceback.format_exception_only(exc)[1:3] == [
        "    return undeclared_symbol\n",
        "           ^^^^^^^^^^^^^^^^^\n",
    ]


# TODO: test program type deduction?
