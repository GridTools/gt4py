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

from __future__ import annotations

import contextlib
import contextvars as cvars
from typing import Any

import gt4py.eve as eve
import gt4py.next.common as common


#: Column range used in column mode (`column_axis != None`) in the current embedded iterator
#: closure execution context.
closure_column_range: cvars.ContextVar[range] = cvars.ContextVar("column_range")

_undefined_offset_provider: common.OffsetProvider = {}

#: Offset provider dict in the current embedded execution context.
offset_provider: cvars.ContextVar[common.OffsetProvider] = cvars.ContextVar(
    "offset_provider", default=_undefined_offset_provider
)


@contextlib.contextmanager
def new_context(
    *,
    closure_column_range: range | eve.NothingType = eve.NOTHING,
    offset_provider: common.OffsetProvider | eve.NothingType = eve.NOTHING,
):
    import gt4py.next.embedded.context as this_module

    updates: list[tuple[cvars.ContextVar[Any], Any]] = []
    if closure_column_range is not eve.NOTHING:
        updates.append((this_module.closure_column_range, closure_column_range))
    if offset_provider is not eve.NOTHING:
        updates.append((this_module.offset_provider, offset_provider))

    # Create new context with provided values
    ctx = cvars.copy_context()

    def ctx_updater(*args):
        for cvar, value in args:
            cvar.set(value)

    ctx.run(ctx_updater, *updates)

    yield ctx


def within_context() -> bool:
    return offset_provider.get() is not _undefined_offset_provider
