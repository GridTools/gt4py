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
import traceback
from typing import Callable, Optional

import importlib_metadata

from . import exceptions, formatting


def _get_developer_mode_python() -> bool:
    """Guess if the Python environment is used to develop gt4py."""
    # Import gt4py and use its __name__ because hard-coding "gt4py" would fail
    # silently if the module's name changes for whatever reason.
    import gt4py

    package_name = gt4py.__name__

    # Check if any package requires gt4py as a dependency. If not, we are
    # probably developing gt4py itself rather than something else using gt4py.
    dists = importlib_metadata.distributions()
    for dist in dists:
        for req in dist.requires or []:
            if req.startswith(package_name):
                return False
    return True


def _get_developer_mode_os() -> Optional[bool]:
    """Detect if the user set developer mode in environment variables."""
    env_var_name = "GT4PY_DEVELOPER_MODE"
    if env_var_name in os.environ:
        try:
            return bool(os.environ[env_var_name])
        except TypeError:
            return False
    return None


def _guess_developer_mode() -> bool:
    """Guess if gt4py is run by its developers or by third party users."""
    env = _get_developer_mode_os()
    if env is not None:
        return env
    return _get_developer_mode_python()


_developer_mode = _guess_developer_mode()


def set_developer_mode(enabled: bool = False):
    """In developer mode, information useful for gt4py developers is also shown."""
    global _developer_mode
    _developer_mode = enabled


def _print_cause(exc: BaseException):
    """Print the cause of an exception plus the bridging message to STDERR."""
    bridging_message = "The above exception was the direct cause of the following exception:"

    if exc.__cause__ or exc.__context__:
        traceback.print_exception(exc.__cause__ or exc.__context__)
        print(f"\n{bridging_message}\n", file=sys.stderr)


def _print_traceback(exc: BaseException):
    """Print the traceback of an exception to STDERR."""
    intro_message = "Traceback (most recent call last):"
    traceback_strs = [
        f"{intro_message}\n",
        *traceback.format_tb(exc.__traceback__),
    ]
    print("".join(traceback_strs), file=sys.stderr)


def compilation_error_hook(fallback: Callable, type_: type, value: BaseException, tb):
    if isinstance(value, exceptions.CompilerError):
        if _developer_mode:
            _print_cause(value)
            _print_traceback(value)
        exc_strs = formatting.format_compilation_error(
            type(value), value.message, value.location_trace
        )
        print("".join(exc_strs), file=sys.stderr)
    else:
        fallback(type_, value, tb)


_fallback = sys.excepthook
sys.excepthook = lambda ty, val, tb: compilation_error_hook(_fallback, ty, val, tb)
