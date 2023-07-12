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

import pathlib
import textwrap
import traceback
import types
from typing import Optional

from gt4py.eve import SourceLocation


def get_source_from_location(location: SourceLocation) -> str:
    if not location.filename:
        raise FileNotFoundError()
    source_file = pathlib.Path(location.filename)
    source_code = source_file.read_text()
    source_lines = source_code.splitlines(False)
    start_line = location.line
    end_line = location.end_line + 1 if location.end_line else start_line + 1
    relevant_lines = source_lines[(start_line - 1) : (end_line - 1)]
    return "\n".join(relevant_lines)


def format_location(loc: SourceLocation, caret: bool = False) -> str:
    filename = loc.filename or "<unknown>"
    lineno = loc.line or "<unknown>"
    loc_str = f'File "{filename}", line {lineno}'

    if caret and loc.column is not None:
        offset = loc.column - 1
        width = loc.end_column - loc.column if loc.end_column is not None else 1
        caret_str = "".join([" "] * offset + ["^"] * width)
    else:
        caret_str = None

    try:
        snippet_str = get_source_from_location(loc)
        if caret_str:
            snippet_str = f"{snippet_str}\n{caret_str}"
        return f"{loc_str}\n{textwrap.indent(snippet_str, '  ')}"
    except Exception:
        return loc_str


def _format_cause(cause: BaseException) -> list[str]:
    """Print the cause of an exception plus the bridging message to STDERR."""
    bridging_message = "The above exception was the direct cause of the following exception:"
    cause_strs = [*traceback.format_exception(cause), "\n", f"{bridging_message}\n\n"]
    return cause_strs


def _format_traceback(tb: types.TracebackType) -> list[str]:
    """Format the traceback of an exception."""
    intro_message = "Traceback (most recent call last):"
    traceback_strs = [
        f"{intro_message}\n",
        *traceback.format_tb(tb),
    ]
    return traceback_strs


def format_compilation_error(
    type_: type[Exception],
    message: str,
    location: Optional[SourceLocation],
    tb: Optional[types.TracebackType] = None,
    cause: Optional[BaseException] = None,
) -> list[str]:
    bits: list[str] = []

    if cause is not None:
        bits = [*bits, *_format_cause(cause)]
    if tb is not None:
        bits = [*bits, *_format_traceback(tb)]
    if location is not None:
        loc_str = format_location(location, caret=True)
        loc_str_all = f"Source location:\n{textwrap.indent(loc_str, '  ')}\n"
        bits = [*bits, loc_str_all]
    msg_str = f"{type_.__module__}.{type_.__name__}: {message}"
    bits = [*bits, msg_str]
    return bits
