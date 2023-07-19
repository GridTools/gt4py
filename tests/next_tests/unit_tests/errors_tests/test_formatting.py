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

import re

import pytest

from gt4py.eve import SourceLocation
from gt4py.next import errors
from gt4py.next.errors.formatting import format_compilation_error


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
    pattern = f"{qualname}: {message}"
    s = "\n".join(format_compilation_error(type_, message, None, None, None))
    assert re.match(pattern, s)


def test_format_loc(type_, qualname, message, location):
    pattern = "Source location.*\\n" '  File "/source.*".*\\n' f"{qualname}: {message}"
    s = "".join(format_compilation_error(type_, message, location, None, None))
    assert re.match(pattern, s)


def test_format_traceback(type_, qualname, message, tb):
    pattern = "Traceback.*\\n" '  File ".*".*\\n' ".*\\n" f"{qualname}: {message}"
    s = "".join(format_compilation_error(type_, message, None, tb, None))
    assert re.match(pattern, s)


def test_format_cause(type_, qualname, message):
    cause = ValueError("asd")
    pattern = "ValueError: asd\\n\\n" "The above.*\\n\\n" f"{qualname}: {message}"
    s = "".join(format_compilation_error(type_, message, None, None, cause))
    assert re.match(pattern, s)
