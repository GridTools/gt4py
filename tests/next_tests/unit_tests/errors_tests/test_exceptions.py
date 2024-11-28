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


@pytest.fixture
def loc_snippet():
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    # This very line of comment should be shown in the snippet.
    return SourceLocation(
        frameinfo.filename, frameinfo.lineno + 1, 15, end_line=frameinfo.lineno + 1, end_column=29
    )


@pytest.fixture
def loc_plain():
    return SourceLocation("/source/file.py", 5, 2, end_line=5, end_column=9)


@pytest.fixture
def message():
    return "a message"


def test_message(loc_plain, message):
    assert errors.DSLError(loc_plain, message).message == message


def test_location(loc_plain, message):
    assert errors.DSLError(loc_plain, message).location == loc_plain


def test_with_location(loc_plain, message):
    assert errors.DSLError(None, message).with_location(loc_plain).location == loc_plain


def test_str(loc_plain, message):
    pattern = f'{message}\\n  File ".*", line.*'
    s = str(errors.DSLError(loc_plain, message))
    assert re.match(pattern, s)


def test_str_snippet(loc_snippet, message):
    pattern = r"\n".join(
        [
            f"{message}",
            '  File ".*", line.*',
            "        # This very line of comment should be shown in the snippet.",
            r"                  \^\^\^\^\^\^\^\^\^\^\^\^\^\^",
        ]
    )
    s = str(errors.DSLError(loc_snippet, message))
    assert re.match(pattern, s)
