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

import inspect
import traceback

import pytest

import gt4py.next as gtx
from gt4py.next import errors
from gt4py.next.ffront import func_to_foast as f2f, source_utils as src_utils
from gt4py.next.ffront.foast_passes import type_deduction


# NOTE: These tests are sensitive to filename and the line number of the marked statement

TDim = gtx.Dimension("TDim")  # Meaningless dimension, used for tests.


def test_invalid_syntax_error_empty_return():
    """Field operator syntax errors point to the file, line and column."""

    line = inspect.getframeinfo(inspect.currentframe()).lineno

    def wrong_syntax(inp: gtx.Field[[TDim], float]):
        return  # <-- this line triggers the syntax error

    with pytest.raises(
        f2f.errors.DSLError,
        match=(r".*return.*"),
    ) as exc_info:
        _ = f2f.FieldOperatorParser.apply_to_function(wrong_syntax)

    assert exc_info.value.location
    assert exc_info.value.location.filename.find("test_func_to_foast_error_line_number.py")
    assert exc_info.value.location.line == line + 3
    assert exc_info.value.location.end_line == line + 3
    assert exc_info.value.location.column == 9
    assert exc_info.value.location.end_column == 15


def test_syntax_error_without_function():
    """Dialect parsers report line numbers correctly when applied to `SourceDefinition`."""

    source_definition = src_utils.SourceDefinition(
        line_offset=61,
        source="""
            def invalid_python_syntax():
                # This function contains a python syntax error

                ret%%  # <-- this line triggers the syntax error
        """,
    )

    with pytest.raises(errors.DSLError) as exc_info:
        _ = f2f.FieldOperatorParser.apply(source_definition, {}, {})

    assert exc_info.value.location
    assert exc_info.value.location.filename.find("test_func_to_foast_error_line_number.py")
    assert exc_info.value.location.line == 66
    assert exc_info.value.location.end_line == 66
    assert exc_info.value.location.column == 9
    assert exc_info.value.location.end_column == 10


def test_fo_type_deduction_error():
    """Field operator type deduction errors report location correctly"""

    line = inspect.getframeinfo(inspect.currentframe()).lineno

    def field_operator_with_undeclared_symbol():
        return undeclared_symbol  # noqa: F821 [undefined-name]

    with pytest.raises(errors.DSLError) as exc_info:
        _ = f2f.FieldOperatorParser.apply_to_function(field_operator_with_undeclared_symbol)

    exc = exc_info.value

    assert exc_info.value.location
    assert exc_info.value.location.filename.find("test_func_to_foast_error_line_number.py")
    assert exc_info.value.location.line == line + 3
    assert exc_info.value.location.end_line == line + 3
    assert exc_info.value.location.column == 16
    assert exc_info.value.location.end_column == 33


# TODO: test program type deduction?
