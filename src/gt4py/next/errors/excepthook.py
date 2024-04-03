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

"""
Loading this module registers an excepthook that formats :class:`DSLError`.

The excepthook is necessary because the default hook prints :class:`DSLError`s
in an inconvenient way. The previously set excepthook is used to print all
other errors.
"""

import sys
import types
from collections.abc import Callable
from typing import Optional

from gt4py.next import config

from . import exceptions, formatting


def _format_uncaught_error(err: exceptions.DSLError, verbose_exceptions: bool) -> list[str]:
    if verbose_exceptions:
        return formatting.format_compilation_error(
            type(err), err.message, err.location, err.__traceback__, err.__cause__
        )
    else:
        return formatting.format_compilation_error(type(err), err.message, err.location)


def compilation_error_hook(
    fallback: Callable, type_: type, value: BaseException, tb: Optional[types.TracebackType]
) -> None:
    """
    Format `CompilationError`s in a neat way.

    All other Python exceptions are formatted by the `fallback` hook. When
    verbose exceptions are enabled, the stack trace and cause of the error is
    also printed.
    """
    if isinstance(value, exceptions.DSLError):
        exc_strs = _format_uncaught_error(value, config.VERBOSE_EXCEPTIONS)
        print("".join(exc_strs), file=sys.stderr)
    else:
        fallback(type_, value, tb)


_fallback = sys.excepthook
sys.excepthook = lambda ty, val, tb: compilation_error_hook(_fallback, ty, val, tb)
