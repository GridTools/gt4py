# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any


def is_compile_time_integer(expr: Any) -> bool:
    """Check whether an expression is a non-negative integer literal.

    The expression is typically an element of an SDFG array shape or stride:
    either a concrete integer (Python `int`, `numpy` or `sympy` integer) or a
    symbolic expression involving SDFG symbols. Note that negative literals
    (e.g. `-1`) return `False`, which matches the intent of the call sites
    (shapes, strides and iteration counts are non-negative).

    Args:
        expr: The value to check, e.g. an element of an SDFG array shape or stride.

    Returns:
        `True` if `expr` is a non-negative integer literal, `False` if it is
        a symbol or a symbolic expression.

    Examples:
        >>> is_compile_time_integer(3)
        True
        >>> is_compile_time_integer(1 / 3)
        False
        >>> import sympy
        >>> is_compile_time_integer(sympy.Rational(1, 3))
        False
        >>> is_compile_time_integer(sympy.Symbol("N"))
        False
    """
    return str(expr).isdigit()
