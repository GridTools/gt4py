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

import os
import sys
import warnings
from typing import Callable

from . import exceptions, formatting


def _get_verbose_exceptions_envvar() -> bool:
    """Detect if the user enabled verbose exceptions in the environment variables."""
    env_var_name = "GT4PY_VERBOSE_EXCEPTIONS"
    if env_var_name in os.environ:
        false_values = ["0", "false", "off"]
        true_values = ["1", "true", "on"]
        value = os.environ[env_var_name].lower()
        if value in false_values:
            return False
        elif value in true_values:
            return True
        else:
            values = ", ".join([*false_values, *true_values])
            msg = f"the 'GT4PY_VERBOSE_EXCEPTIONS' environment variable must be one of {values} (case insensitive)"
            warnings.warn(msg)
    return False


_verbose_exceptions: bool = _get_verbose_exceptions_envvar()


def set_verbose_exceptions(enabled: bool = False) -> None:
    """Programmatically set whether to use verbose printing for uncaught errors."""
    global _verbose_exceptions
    _verbose_exceptions = enabled


def _format_uncaught_error(err: exceptions.DSLError, verbose_exceptions: bool) -> list[str]:
    if verbose_exceptions:
        return formatting.format_compilation_error(
            type(err),
            err.message,
            err.location,
            err.__traceback__,
            err.__cause__,
        )
    else:
        return formatting.format_compilation_error(type(err), err.message, err.location)


def compilation_error_hook(fallback: Callable, type_: type, value: BaseException, tb) -> None:
    """
    Format `CompilationError`s in a neat way.

    All other Python exceptions are formatted by the `fallback` hook. When
    verbose exceptions are enabled, the stack trace and cause of the error is
    also printed.
    """
    if isinstance(value, exceptions.DSLError):
        exc_strs = _format_uncaught_error(value, _verbose_exceptions)
        print("".join(exc_strs), file=sys.stderr)
    else:
        fallback(type_, value, tb)


_fallback = sys.excepthook
sys.excepthook = lambda ty, val, tb: compilation_error_hook(_fallback, ty, val, tb)
