# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Best-effort formatting of generated source code for human readers."""

from __future__ import annotations

import os
import subprocess


def format_python_source(source: str) -> str:
    """Format Python source code with `black`.

    Args:
        source: Python source code.

    Returns:
        The formatted source code, or `source` unchanged if `black` is not
        installed or formatting fails.
    """
    try:
        # Lazy import: `black` is an optional dependency and slow to import.
        import black
    except ImportError:
        return source

    try:
        return black.format_str(source, mode=black.Mode(line_length=100))
    except ValueError:  # `black.InvalidInput` for unparsable source
        return source


def format_cpp_source(source: str) -> str:
    """Format C++ source code with `clang-format` using the LLVM style.

    The executable can be overridden with the `CLANG_FORMAT_EXECUTABLE`
    environment variable.

    Args:
        source: C++ source code.

    Returns:
        The formatted source code, or `source` unchanged if `clang-format` is
        not available or formatting fails.
    """
    args = [
        os.getenv("CLANG_FORMAT_EXECUTABLE", "clang-format"),
        "--style=LLVM",
        "--assume-filename=_gt4py_generated_file.cpp",
    ]
    try:
        # use a timeout as clang-format used to deadlock on some sources
        return subprocess.run(
            args, input=source, capture_output=True, encoding="utf-8", check=True, timeout=3
        ).stdout
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return source
