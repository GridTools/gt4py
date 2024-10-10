# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for formatting :class:`DSLError` and its subclasses."""

from __future__ import annotations

import linecache
import textwrap
import traceback
import types
from typing import TYPE_CHECKING, Optional

from gt4py.eve import SourceLocation


if TYPE_CHECKING:
    from . import exceptions


def get_source_from_location(location: SourceLocation) -> str:
    if not location.filename:
        raise FileNotFoundError()
    source_lines = linecache.getlines(location.filename)
    if not source_lines:
        raise FileNotFoundError()
    start_line = location.line
    end_line = location.end_line + 1 if location.end_line is not None else start_line + 1
    relevant_lines = source_lines[(start_line - 1) : (end_line - 1)]
    return "\n".join(relevant_lines)


def format_location(loc: SourceLocation, show_caret: bool = False) -> str:
    """
    Format the source file location.

    Args:
        show_caret (bool): Indicate the position within the source line by placing carets underneath.
    """
    filename = loc.filename or "<unknown>"
    lineno = loc.line
    loc_str = f'File "{filename}", line {lineno}'

    if show_caret and loc.column is not None:
        offset = loc.column - 1
        width = loc.end_column - loc.column if loc.end_column is not None else 1
        caret_str = "".join([" "] * offset + ["^"] * width)
    else:
        caret_str = None

    try:
        snippet_str = get_source_from_location(loc)
        if caret_str:
            snippet_str = f"{snippet_str}{caret_str}"
        return f"{loc_str}\n{textwrap.indent(snippet_str, '  ')}"
    except (FileNotFoundError, IndexError):
        return loc_str


def _format_cause(cause: BaseException) -> list[str]:
    """Format the cause of an exception plus the bridging message to the current exception."""
    bridging_message = "The above exception was the direct cause of the following exception:"
    cause_strs = [*traceback.format_exception(cause), "\n", f"{bridging_message}\n\n"]
    return cause_strs


def _format_traceback(tb: types.TracebackType) -> list[str]:
    """Format the traceback of an exception."""
    intro_message = "Traceback (most recent call last):"
    traceback_strs = [f"{intro_message}\n", *traceback.format_tb(tb)]
    return traceback_strs


def format_compilation_error(
    type_: type[exceptions.DSLError],
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
        loc_str = format_location(location, show_caret=True)
        loc_str_all = f"Source location:\n{textwrap.indent(loc_str, '  ')}\n"
        bits = [*bits, loc_str_all]
    msg_str = f"{type_.__module__}.{type_.__name__}: {message}"
    bits = [*bits, msg_str]
    return bits
