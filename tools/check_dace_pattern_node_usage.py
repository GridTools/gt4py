# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Check that PatternNode attributes in DaCe transformations are only referenced in 'apply' or 'can_be_applied'.

In classes that inherit from ``dace_transformation.SingleStateTransformation`` or
``dace_transformation.MultiStateTransformation``, class-level attributes declared as
``dace_transformation.PatternNode(...)`` must only be accessed via ``self.<attr>`` inside
the ``apply`` or ``can_be_applied`` methods. Accessing them from any other method is
forbidden, since the DaCe pattern-matching infrastructure only populates those attributes
during the application phase.

Usage (pre-commit)::

    python tools/check_dace_pattern_node_usage.py <file1.py> [<file2.py> ...]
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers for recognising relevant AST constructs
# ---------------------------------------------------------------------------

_TRANSFORMATION_BASE_NAMES: frozenset[str] = frozenset(
    {"SingleStateTransformation", "MultiStateTransformation"}
)

_ALLOWED_METHODS: frozenset[str] = frozenset({"apply", "can_be_applied"})


def _is_pattern_node_call(node: ast.expr) -> bool:
    """Return True if *node* is a call to ``PatternNode(...)`` (with or without module qualifier)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    # Bare ``PatternNode(...)``
    if isinstance(func, ast.Name) and func.id == "PatternNode":
        return True
    # Qualified ``dace_transformation.PatternNode(...)``
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "PatternNode"
        and isinstance(func.value, ast.Name)
        and func.value.id == "dace_transformation"
    ):
        return True
    return False


def _is_dace_transformation_base(base: ast.expr) -> bool:
    """Return True if *base* is a DaCe Single/MultiStateTransformation base class expression."""
    if isinstance(base, ast.Name) and base.id in _TRANSFORMATION_BASE_NAMES:
        return True
    if (
        isinstance(base, ast.Attribute)
        and base.attr in _TRANSFORMATION_BASE_NAMES
        and isinstance(base.value, ast.Name)
        and base.value.id == "dace_transformation"
    ):
        return True
    return False


def _collect_pattern_node_attrs(class_node: ast.ClassDef) -> frozenset[str]:
    """Return the names of all class-level ``PatternNode`` attributes in *class_node*."""
    attrs: set[str] = set()
    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and _is_pattern_node_call(stmt.value):
                    attrs.add(target.id)
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.value is not None
            and _is_pattern_node_call(stmt.value)
        ):
            attrs.add(stmt.target.id)
    return frozenset(attrs)


# ---------------------------------------------------------------------------
# Per-file checker
# ---------------------------------------------------------------------------


def check_file(filepath: str) -> list[str]:
    """Return a list of error strings found in *filepath*, empty if none."""
    errors: list[str] = []
    source = Path(filepath).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        return [f"{filepath}: SyntaxError: {exc}"]

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_is_dace_transformation_base(base) for base in node.bases):
            continue

        pattern_node_attrs = _collect_pattern_node_attrs(node)
        if not pattern_node_attrs:
            continue

        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if method.name in _ALLOWED_METHODS:
                continue

            for sub in ast.walk(method):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"
                    and sub.attr in pattern_node_attrs
                ):
                    errors.append(
                        f"{filepath}:{sub.lineno}: In class '{node.name}', method"
                        f" '{method.name}' references PatternNode attribute"
                        f" 'self.{sub.attr}'. PatternNode attributes may only be"
                        f" accessed inside 'apply' or 'can_be_applied'."
                    )

    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    files = sys.argv[1:]
    if not files:
        return 0

    all_errors: list[str] = []
    for filepath in files:
        all_errors.extend(check_file(filepath))

    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
