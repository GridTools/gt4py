# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Text and templating utilities."""

import collections.abc
import contextlib
import re
import textwrap

import black


black_mode = black.FileMode(
    target_versions={
        black.TargetVersion.PY36,  # type: ignore[attr-defined]  # black.TargetVersion
        black.TargetVersion.PY37,  # type: ignore[attr-defined]  # black.TargetVersion
    },
    line_length=120,
)


def format_source(source: str, line_length: int) -> str:
    black_mode.line_length = line_length
    return black.format_str(source, mode=black_mode)


def get_line_number(text, re_query, re_flags=0):
    """Return (0-based) line number of the match or None."""
    prog = re.compile(re_query, re_flags)
    lines = text.splitlines()
    for n, line in enumerate(lines):
        if prog.search(line):
            return n

    return None


class Joiner:
    class Placeholder:
        def __init__(self, joiner, index):
            self.joiner = joiner
            self.index = index

        def __str__(self) -> str:
            return self.joiner.joiner_str if self.index < self.joiner.n_items - 1 else ""

    def __init__(self, joiner_str):
        self.joiner_str = joiner_str
        self.n_items = 0

    def __call__(self):
        self.n_items += 1
        return type(self).Placeholder(self, self.n_items - 1)


class TextBlock:
    def __init__(self, indent_level=0, indent_size=4, indent_char=" ", end_line="\n"):
        self.indent_level = indent_level
        self.indent_size = indent_size
        self.indent_char = indent_char
        self.end_line = end_line
        self.lines = []

    def append(self, source_line, indent_steps=0):
        self.indent(indent_steps)
        if isinstance(source_line, str):
            source_line = [source_line]

        line = [self.indent_char * (self.indent_level * self.indent_size)]
        for item in source_line:
            if isinstance(item, str) and isinstance(line[-1], str):
                line[-1] += item
            else:
                line.append(item)

        self.lines.append(line)

        return self

    def extend(self, source_lines, *, dedent=False):
        assert isinstance(source_lines, (str, collections.abc.Sequence, TextBlock))

        if dedent:
            if isinstance(source_lines, TextBlock):
                source_lines = source_lines.text
            elif not isinstance(source_lines, str):
                source_lines = "\n".join(source_lines)
            source_lines = textwrap.dedent(source_lines)

        if isinstance(source_lines, str):
            source_lines = source_lines.splitlines()
        elif isinstance(source_lines, TextBlock):
            source_lines = source_lines.lines

        for line in source_lines:
            self.append(line)
        return self

    def empty_line(self, steps=1):
        self.lines.append("")
        return self

    def indent(self, steps=1):
        self.indent_level += steps
        return self

    def dedent(self, steps=1):
        assert self.indent_level >= steps
        self.indent_level -= steps
        return self

    @contextlib.contextmanager
    def indented(self, steps=1):
        self.indent(steps)
        yield self
        self.dedent(steps)

    @property
    def text(self):
        lines = ["".join([str(item) for item in line]) for line in self.lines]
        return self.end_line.join(lines)

    def __iadd__(self, source_line):
        return self.append(source_line)

    def __len__(self):
        return len(self.lines)

    def __str__(self) -> str:
        return self.text
