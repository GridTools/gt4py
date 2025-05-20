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

import gt4py.eve as eve
import gt4py.next.common as common
import gt4py.next.embedded as gtx_embedded


#: Column range used in column mode (`column_axis != None`) in the current embedded iterator
#: closure execution context.
closure_column_range: cvars.ContextVar[common.NamedRange] = cvars.ContextVar("column_range")

#: Offset provider dict in the current embedded execution context.
offset_provider: cvars.ContextVar[common.OffsetProvider] = cvars.ContextVar("offset_provider")


@contextlib.contextmanager
def update(
    *,
    closure_column_range: common.NamedRange | eve.NothingType = eve.NOTHING,
    offset_provider: common.OffsetProvider | eve.NothingType = eve.NOTHING,
) -> Generator[cvars.Context, None, None]:
    """Create a new context, updating the provided values."""

    closure_token, offset_provider_token = None, None
    if closure_column_range is not eve.NOTHING:
        closure_token = gtx_embedded.context.closure_column_range.set(closure_column_range)
    if offset_provider is not eve.NOTHING:
        offset_provider_token = gtx_embedded.context.offset_provider.set(offset_provider)

    yield

    if closure_token is not None:
        gtx_embedded.context.closure_column_range.reset(closure_token)
    if offset_provider_token is not None:
        gtx_embedded.context.offset_provider.reset(offset_provider_token)


def within_valid_context() -> bool:
    return offset_provider.get(eve.NOTHING) is not eve.NOTHING
