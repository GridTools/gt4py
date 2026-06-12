# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import re

import pytest

from gt4py.eve import SourceLocation
from gt4py.next import errors
from gt4py.next.errors.formatting import format_compilation_error, format_diagnostic_parts


@pytest.fixture
def message():
    return "a message"


@pytest.fixture
def location():
    return SourceLocation("/source/file.py", 5, 2, end_line=5, end_column=9)


@pytest.fixture
def tb():
    try:
        raise Exception()
    except Exception as ex:
        return ex.__traceback__


@pytest.fixture
def type_():
    return errors.DSLError


@pytest.fixture
def qualname(type_):
    return f"{type_.__module__}.{type_.__name__}"


def test_format(type_, qualname, message):
    cls_pattern = f"{qualname}: {message}"
    s = "\n".join(format_compilation_error(type_, message, None, None, None))
    assert re.match(cls_pattern, s)


def test_format_loc(type_, qualname, message, location):
    loc_pattern = "Source location.*"
    file_pattern = '  File "/source.*".*'
    cls_pattern = f"{qualname}: {message}"
    pattern = r"\n".join([loc_pattern, file_pattern, cls_pattern])
    s = "".join(format_compilation_error(type_, message, location, None, None))
    assert re.match(pattern, s)


def test_format_traceback(type_, qualname, message, tb):
    tb_pattern = "Traceback.*"
    file_pattern = '  File ".*".*'
    line_pattern = ".*"
    cls_pattern = f"{qualname}: {message}"
    pattern = r"\n".join([tb_pattern, file_pattern, line_pattern, cls_pattern])
    s = "".join(format_compilation_error(type_, message, None, tb, None))
    assert re.match(pattern, s)


@pytest.fixture
def loc_here():
    # location spanning the body of this fixture function, in this file
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    return SourceLocation(
        frameinfo.filename,
        frameinfo.lineno - 1,
        1,
        end_line=frameinfo.lineno + 10,
        end_column=2,
    )


def test_snippet_multiline_is_truncated(message, loc_here):
    s = format_diagnostic_parts(message, loc_here)
    lines = s.splitlines()
    snippet_lines = [line for line in lines if re.match(r"\s+(\d+ )?\|", line)]
    assert snippet_lines[-1].strip().endswith("...")
    # gutter-prefixed source lines are capped, +1 caret row, +1 truncation row
    assert len(snippet_lines) <= 5


def test_notes_and_hints_are_rendered_and_wrapped(message):
    s = format_diagnostic_parts(
        message, None, notes=["A short note."], hints=["A very long hint. " * 10]
    )
    assert "  Note: A short note." in s
    assert "  Hint: A very long hint." in s
    assert all(len(line) <= 88 for line in s.splitlines())


def test_same_line_related_merges_into_snippet(message, loc_here):
    related_loc = SourceLocation(
        loc_here.filename, loc_here.line, 5, end_line=loc_here.line, end_column=14
    )
    s = format_diagnostic_parts(message, loc_here, related=[(related_loc, "relevant")])
    assert "Related:" not in s
    assert re.search(r"\| {5}-{9} relevant", s), s


def test_remote_related_gets_own_block(message, loc_here):
    related_loc = SourceLocation(loc_here.filename, 1, 1, end_line=1, end_column=2)
    s = format_diagnostic_parts(message, loc_here, related=[(related_loc, "relevant")])
    assert "  Related: relevant" in s
    assert re.search(r'    File ".*", line 1', s), s


def test_format_cause(type_, qualname, message):
    cause = ValueError("asd")
    blank_pattern = ""
    cause_pattern = "ValueError: asd"
    bridge_pattern = "The above.*"
    cls_pattern = f"{qualname}: {message}"
    pattern = r"\n".join([cause_pattern, blank_pattern, bridge_pattern, blank_pattern, cls_pattern])
    s = "".join(format_compilation_error(type_, message, None, None, cause))
    assert re.match(pattern, s)
