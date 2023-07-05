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

from gt4py.eve import SourceLocation
from gt4py.next.errors import CompilerError
from gt4py.next.errors.formatting import format_compilation_error
import re
import inspect


frameinfo = inspect.getframeinfo(inspect.currentframe())
loc = SourceLocation("/source/file.py", 5, 2, end_line=5, end_column=9)
msg = "a message"

module = CompilerError.__module__
name = CompilerError.__name__
try:
    raise Exception()
except Exception as ex:
    tb = ex.__traceback__


def test_format():
    pattern = f"{module}.{name}: {msg}"
    s = "\n".join(format_compilation_error(CompilerError, msg, None, None, None))
    assert re.match(pattern, s);


def test_format_loc():
    pattern = \
        "Source location.*\\n" \
        "  File \"/source.*\".*\\n" \
        f"{module}.{name}: {msg}"
    s = "".join(format_compilation_error(CompilerError, msg, loc, None, None))
    assert re.match(pattern, s);


def test_format_traceback():
    pattern = \
        "Traceback.*\\n" \
        "  File \".*\".*\\n" \
        ".*\\n" \
        f"{module}.{name}: {msg}"
    s = "".join(format_compilation_error(CompilerError, msg, None, tb, None))
    assert re.match(pattern, s);


def test_format_cause():
    cause = ValueError("asd")
    pattern = \
        "ValueError: asd\\n\\n" \
        "The above.*\\n\\n" \
        f"{module}.{name}: {msg}"
    s = "".join(format_compilation_error(CompilerError, msg, None, None, cause))
    assert re.match(pattern, s);
