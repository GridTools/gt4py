# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Best-effort formatting of generated source code for human readers."""

from __future__ import annotations

import functools
import os
import subprocess
import sys


def format_python_source(source: str, *, line_length: int = 100) -> str:
    """Format Python source code with `black`.

    The target Python version is pinned to the running interpreter.

    Args:
        source: Python source code.
        line_length: Maximum line length of the formatted code.

    Returns:
        The formatted source code, or `source` unchanged if `black` is not
        installed, does not support the running interpreter, or formatting
        fails.
    """
    try:
        # Lazy import: `black` is an optional dependency and slow to import.
        import black
    except ImportError:
        return source

    try:
        target_version = black.TargetVersion[  # type: ignore[attr-defined]  # .TargetVersion implicitly exported
            f"PY{sys.version_info.major}{sys.version_info.minor}"
        ]
    except KeyError:  # `black` is too old to know the running interpreter
        return source

    try:
        return black.format_str(
            source, mode=black.Mode(line_length=line_length, target_versions={target_version})
        )
    except Exception:  # e.g. `black.InvalidInput` for unparsable source, or internal `black` errors
        return source


def format_cpp_source(source: str) -> str:
    """Format C++ source code with `clang-format` using the LLVM style.

    The executable can be overridden with the `CLANG_FORMAT_EXECUTABLE`
    environment variable. Its availability is checked once, on first use.

    Args:
        source: C++ source code.

    Returns:
        The formatted source code, or `source` unchanged if `clang-format` is
        not available or formatting fails.
    """
    executable = _get_clang_format()
    if executable is None:
        return source

    args = [executable, "--style=LLVM", "--assume-filename=_gt4py_generated_file.cpp"]
    try:
        # use a timeout as clang-format used to deadlock on some sources
        return subprocess.run(
            args, input=source, capture_output=True, encoding="utf-8", check=True, timeout=3
        ).stdout
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return source


@functools.cache
def _get_clang_format() -> str | None:
    """Return the `clang-format` executable, or `None` if it is not available.

    The result is cached, so the executable is probed only once.
    """
    executable = os.getenv("CLANG_FORMAT_EXECUTABLE", "clang-format")
    try:
        if subprocess.run([executable, "--version"], capture_output=True).returncode != 0:
            return None
    except Exception:
        return None

    return executable
