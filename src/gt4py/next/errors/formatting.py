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
from typing import TYPE_CHECKING, Optional, Sequence

from gt4py.eve import SourceLocation


if TYPE_CHECKING:
    from gt4py.next.errors import exceptions

#: Maximum number of source lines rendered for a multi-line location.
_MAX_SNIPPET_LINES = 3


#: An underline row below the first snippet line: (column, end_column, label).
_UnderlineRow = tuple[int, Optional[int], str]


def _gutter(width: int, body: str, lineno: Optional[int] = None) -> str:
    """A gutter row: ``"11 | <body>"`` with a right-aligned line number, or blank."""
    head = f"{lineno:>{width}}" if lineno is not None else " " * width
    return f"{head} | {body}"


def _caret_span(location: SourceLocation, first_line: str) -> tuple[int, int]:
    """Start column (0-based) and width of the caret run on the first line of ``location``."""
    start = location.column - 1
    if location.end_line == location.line and location.end_column is not None:
        width = max(location.end_column - location.column, 1)
    else:
        width = max(len(first_line) - start, 1)
    return start, width


def _format_snippet_lines(
    location: SourceLocation,
    *,
    label: Optional[str] = None,
    secondary: Sequence[_UnderlineRow] = (),
) -> Optional[list[str]]:
    """
    Render the source lines of ``location`` with a line-number gutter and carets.

    The caret row carries ``label`` (if any); ``secondary`` spans (which must lie
    on the first line of ``location``) are underlined with ``-`` markers below it.

    Returns ``None`` if the source is not available (e.g. code defined in a REPL).

    Example output (as a list of lines)::

        11 |     return tmp_feild
           |            ^^^^^^^^^ not defined at this point
    """
    if not location.filename:
        return None
    source_lines = linecache.getlines(location.filename)
    if not source_lines or location.line > len(source_lines):
        return None

    start_line = location.line
    end_line = location.end_line if location.end_line is not None else location.line
    snippet_lines = [line.rstrip("\n") for line in source_lines[(start_line - 1) : end_line]]
    if not snippet_lines:
        return None

    truncated = len(snippet_lines) > _MAX_SNIPPET_LINES
    snippet_lines = snippet_lines[:_MAX_SNIPPET_LINES]

    width = len(str(start_line + len(snippet_lines) - 1))
    result: list[str] = []
    for i, line in enumerate(snippet_lines):
        result.append(_gutter(width, line, lineno=start_line + i))
        if i == 0:
            # Carets, the label, and secondary underlines all sit under the first line.
            caret_start, caret_width = _caret_span(location, line)
            carets = " " * caret_start + "^" * caret_width
            result.append(_gutter(width, f"{carets} {label}" if label else carets))
            for column, end_column, sec_label in sorted(secondary, key=lambda s: s[0]):
                sec_width = max(end_column - column, 1) if end_column is not None else 1
                underline = " " * (column - 1) + "-" * sec_width
                result.append(_gutter(width, f"{underline} {sec_label}"))
    if truncated:
        result.append(_gutter(width, "..."))
    return result


def format_location(
    loc: SourceLocation,
    show_caret: bool = False,
    label: Optional[str] = None,
    secondary: Sequence[_UnderlineRow] = (),
) -> str:
    """
    Format the source file location, optionally with a code snippet.

    Args:
        loc: The location to format.
        show_caret: Indicate the position within the source line by placing carets
            underneath, together with a line-number gutter.
        label: Short text appended after the carets explaining the marked code.
        secondary: Further labeled spans on the first line of ``loc``, underlined
            with ``-`` markers.

    Example output::

        File "model/diffusion.py", line 11
          11 |     return tmp_feild
             |            ^^^^^^^^^ undeclared name
    """
    filename = loc.filename or "<unknown>"
    loc_str = f'File "{filename}", line {loc.line}'

    if not show_caret:
        return loc_str

    snippet_lines = _format_snippet_lines(loc, label=label, secondary=secondary)
    if snippet_lines is None:
        return loc_str
    snippet_str = "\n".join(snippet_lines)
    return f"{loc_str}\n{textwrap.indent(snippet_str, '  ')}"


def _split_related(
    location: Optional[SourceLocation],
    related: Sequence[tuple[SourceLocation, str]],
) -> tuple[list[_UnderlineRow], list[tuple[SourceLocation, str]]]:
    """
    Partition related locations into those on the primary location's first line and the rest.

    Same-line spans are rendered inside the primary snippet; the rest get their
    own ``Related:`` snippet block.
    """
    same_line: list[_UnderlineRow] = []
    remote: list[tuple[SourceLocation, str]] = []
    for rel_location, rel_message in related:
        if (
            location is not None
            and rel_location.filename == location.filename
            and rel_location.line == location.line
            and (rel_location.end_line is None or rel_location.end_line == rel_location.line)
        ):
            same_line.append((rel_location.column, rel_location.end_column, rel_message))
        else:
            remote.append((rel_location, rel_message))
    return same_line, remote


#: Wrapping width of notes and hints; source snippet lines are never wrapped.
_TEXT_WIDTH = 88


def _format_extras(
    related: Sequence[tuple[SourceLocation, str]],
    notes: Sequence[str],
    hints: Sequence[str],
) -> list[str]:
    """Render the secondary parts of a diagnostic: related locations, notes and hints."""
    bits: list[str] = []
    for rel_location, rel_message in related:
        rel_str = format_location(rel_location, show_caret=True)
        bits.append(f"  Related: {rel_message}\n{textwrap.indent(rel_str, '    ')}")
    for prefix, items in (("Note", notes), ("Hint", hints)):
        for item in items:
            bits.append(
                textwrap.fill(
                    f"{prefix}: {item}",
                    width=_TEXT_WIDTH,
                    initial_indent="  ",
                    subsequent_indent="    ",
                )
            )
    return bits


def format_diagnostic_parts(
    message: str,
    location: Optional[SourceLocation],
    *,
    label: Optional[str] = None,
    related: Sequence[tuple[SourceLocation, str]] = (),
    notes: Sequence[str] = (),
    hints: Sequence[str] = (),
) -> str:
    """
    Render the full body of a diagnostic: message, source snippet, related locations, notes and hints.

    This is the single place defining the on-screen shape of GT4Py diagnostics;
    both ``DSLError.__str__`` and the excepthook go through it.
    """
    same_line_related, remote_related = _split_related(location, related)
    bits: list[str] = [message]
    if location is not None:
        bits.append(
            textwrap.indent(
                format_location(
                    location, show_caret=True, label=label, secondary=same_line_related
                ),
                "  ",
            )
        )
    bits.extend(_format_extras(remote_related, notes, hints))
    return "\n".join(bits)


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
    *,
    label: Optional[str] = None,
    related: Sequence[tuple[SourceLocation, str]] = (),
    notes: Sequence[str] = (),
    hints: Sequence[str] = (),
) -> list[str]:
    bits: list[str] = []

    same_line_related, remote_related = _split_related(location, related)
    if cause is not None:
        bits = [*bits, *_format_cause(cause)]
    if tb is not None:
        bits = [*bits, *_format_traceback(tb)]
    if location is not None:
        loc_str = format_location(
            location, show_caret=True, label=label, secondary=same_line_related
        )
        loc_str_all = f"Source location:\n{textwrap.indent(loc_str, '  ')}\n"
        bits = [*bits, loc_str_all]
    msg_str = f"{type_.__module__}.{type_.__name__}: {message}"
    bits = [*bits, msg_str]
    if extras := _format_extras(remote_related, notes, hints):
        bits = [*bits, "\n" + "\n".join(extras)]
    return bits
