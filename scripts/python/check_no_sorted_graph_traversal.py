#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Check that ``sorted()`` is not used in gt4py's DaCe backend.

Using ``sorted()`` to obtain a "canonical" ordering is usually a sign that the
underlying data structure has non-deterministic iteration order. DaCe graph
methods such as ``in_edges()``, ``out_edges()`` and ``nodes()`` already return
lists, connector names are already stored in dicts, and sets should be replaced
by ``OrderedSet`` when order matters. Wrapping any of these in ``sorted()`` is
therefore unnecessary and can hide non-determinism when the sort key
accidentally depends on object identity, e.g. ``id()`` or ``repr()``.

The only allowed exception is sorting ``free_symbols``, because the set of
symbols is a genuine mathematical set and sorting it yields a stable order.

This script is intended as a pre-commit / CI check for the DaCe backend code.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


# ``sorted()`` may only be used on these expressions.
ALLOWED_SORTED_ARGUMENTS = frozenset({"free_symbols"})


def _is_allowed_sorted_argument(node: ast.AST) -> bool:
    """Return True if ``sorted(node)`` is an allowed use case."""
    arg_src = ast.unparse(node)
    return any(allowed in arg_src for allowed in ALLOWED_SORTED_ARGUMENTS)


def check_file(path: Path) -> list[str]:
    """Return a list of violation messages for ``path``."""
    try:
        source = path.read_text()
    except (OSError, UnicodeDecodeError) as e:
        return [f"{path}: error reading file: {e}"]

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"{path}:{e.lineno}: syntax error: {e.msg}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sorted"
        ):
            continue
        if node.args and not _is_allowed_sorted_argument(node.args[0]):
            arg_src = ast.unparse(node.args[0])
            violations.append(f"{path}:{node.lineno}: sorted({arg_src})")

    return violations


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: check_no_sorted_graph_traversal.py <file>...", file=sys.stderr)
        return 2

    all_violations: list[str] = []
    for arg in argv:
        path = Path(arg)
        if not path.is_file():
            continue
        all_violations.extend(check_file(path))

    if all_violations:
        print(
            "Found sorted() calls in the DaCe backend. "
            "Use OrderedSet or rely on the existing list/dict order instead; "
            "only sorting free_symbols is allowed:",
            file=sys.stderr,
        )
        for violation in all_violations:
            print(violation, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
