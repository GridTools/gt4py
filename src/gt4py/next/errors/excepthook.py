# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
from gt4py.next.errors import exceptions, formatting


def _format_uncaught_error(err: exceptions.DSLError, verbose_exceptions: bool) -> list[str]:
    return formatting.format_compilation_error(
        type(err),
        err.message,
        err.location,
        err.__traceback__ if verbose_exceptions else None,
        err.__cause__ if verbose_exceptions else None,
        label=err.label,
        related=err.related,
        notes=err.notes,
        hints=err.hints,
    )


def compilation_error_hook(
    fallback: Callable, type_: type, value: BaseException, tb: Optional[types.TracebackType]
) -> None:
    """
    Format `CompilationError`s in a neat way.

    All other Python exceptions are formatted by the `fallback` hook. When
    verbose exceptions are enabled, the stack trace and cause of the error is
    also printed.
    """
    # in hard crashes of the interpreter, the `exceptions` module might be partially unloaded
    if exceptions.DSLError is not None and isinstance(value, exceptions.DSLError):
        exc_strs = _format_uncaught_error(value, config.VERBOSE_EXCEPTIONS)
        # Our hook replaces the default, so we render PEP 678 '__notes__'
        # ('add_note' breadcrumbs) ourselves instead of the traceback machinery.
        exc_strs += [f"\n{note}" for note in getattr(value, "__notes__", [])]
        print("".join(exc_strs), file=sys.stderr)
    else:
        fallback(type_, value, tb)


_fallback = sys.excepthook
sys.excepthook = lambda ty, val, tb: compilation_error_hook(_fallback, ty, val, tb)
