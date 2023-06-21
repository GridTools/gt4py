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

from gt4py.eve import SourceLocation


def get_source_from_location(location: SourceLocation):
    if not location.filename:
        raise FileNotFoundError()
    source_file = pathlib.Path(location.filename)
    source_code = source_file.read_text()
    source_lines = source_code.splitlines(False)
    start_line = location.line
    end_line = location.end_line + 1 if location.end_line else start_line + 1
    relevant_lines = source_lines[(start_line - 1) : (end_line - 1)]
    return "\n".join(relevant_lines)


def format_location(loc: SourceLocation, caret: bool = False):
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


def format_compilation_error(
    type_: type[Exception], message: str, location_trace: list[SourceLocation]
):
    msg_str = f"{type_.__module__}.{type_.__name__}: {message}"

    try:
        loc_str = "".join([format_location(loc, caret=True) for loc in location_trace])
        stack_str = f"Source location (most recent call last):\n{textwrap.indent(loc_str, '  ')}\n"
        return [stack_str, msg_str]
    except ValueError:
        return [msg_str]
