# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import contextvars as cvars
from collections.abc import Generator
from typing import Any

import gt4py.eve as eve
import gt4py.next.common as common


#: Column range used in column mode (`column_axis != None`) in the current embedded iterator
#: closure execution context.
closure_column_range: cvars.ContextVar[common.NamedRange] = cvars.ContextVar("column_range")

#: Offset provider dict in the current embedded execution context.
offset_provider: cvars.ContextVar[common.OffsetProvider] = cvars.ContextVar("offset_provider")


@contextlib.contextmanager
def new_context(
    *,
    closure_column_range: common.NamedRange | eve.NothingType = eve.NOTHING,
    offset_provider: common.OffsetProvider | eve.NothingType = eve.NOTHING,
) -> Generator[cvars.Context, None, None]:
    """Create a new context, updating the provided values."""

    import gt4py.next.embedded.context as this_module

    updates: list[tuple[cvars.ContextVar[Any], Any]] = []
    if closure_column_range is not eve.NOTHING:
        updates.append((this_module.closure_column_range, closure_column_range))
    if offset_provider is not eve.NOTHING:
        updates.append((this_module.offset_provider, offset_provider))

    # Create new context with provided values
    ctx = cvars.copy_context()

    def ctx_updater(*args: tuple[cvars.ContextVar[Any], Any]) -> None:
        for cvar, value in args:
            cvar.set(value)

    ctx.run(ctx_updater, *updates)

    yield ctx


def within_valid_context() -> bool:
    return offset_provider.get(eve.NOTHING) is not eve.NOTHING
