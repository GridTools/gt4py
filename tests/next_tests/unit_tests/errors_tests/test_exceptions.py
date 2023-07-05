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
import re

from gt4py.eve import SourceLocation
from gt4py.next.errors import CompilerError


frameinfo = inspect.getframeinfo(inspect.currentframe())
loc = SourceLocation("/source/file.py", 5, 2, end_line=5, end_column=9)
loc_snippet = SourceLocation(
    frameinfo.filename, frameinfo.lineno + 2, 15, end_line=frameinfo.lineno + 2, end_column=29
)
msg = "a message"


def test_message():
    assert CompilerError(loc, msg).message == msg


def test_location():
    assert CompilerError(loc, msg).location == loc


def test_with_location():
    assert CompilerError(None, msg).with_location(loc).location == loc


def test_str():
    pattern = f'{msg}\\n  File ".*", line.*'
    s = str(CompilerError(loc, msg))
    assert re.match(pattern, s)


def test_str_snippet():
    pattern = (
        f"{msg}\\n"
        '  File ".*", line.*\\n'
        "    loc_snippet = SourceLocation.*\\n"
        "                  \^\^\^\^\^\^\^\^\^\^\^\^\^\^"
    )
    s = str(CompilerError(loc_snippet, msg))
    assert re.match(pattern, s)
