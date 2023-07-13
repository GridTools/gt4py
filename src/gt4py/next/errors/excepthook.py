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
import os
import sys
from typing import Callable

from . import exceptions, formatting


def _get_developer_mode_envvar() -> bool:
    """Detect if the user set developer mode in environment variables."""
    env_var_name = "GT4PY_DEVELOPER_MODE"
    if env_var_name in os.environ:
        try:
            return bool(os.environ[env_var_name])
        except TypeError:
            return False
    return False


_developer_mode: bool = _get_developer_mode_envvar()


def set_developer_mode(enabled: bool = False) -> None:
    """In developer mode, information useful for gt4py developers is also shown."""
    global _developer_mode
    _developer_mode = enabled


def _format_uncaught_error(err: exceptions.CompilerError, developer_mode: bool) -> list[str]:
    if developer_mode:
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

    All other Python exceptions are formatted by the `fallback` hook.
    """
    if isinstance(value, exceptions.CompilerError):
        exc_strs = _format_uncaught_error(value, _developer_mode)
        print("".join(exc_strs), file=sys.stderr)
    else:
        fallback(type_, value, tb)


_fallback = sys.excepthook
sys.excepthook = lambda ty, val, tb: compilation_error_hook(_fallback, ty, val, tb)
