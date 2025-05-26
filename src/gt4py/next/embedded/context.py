# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Generator
from typing import Any, TypeVar, overload

import gt4py.eve as eve
import gt4py.next.common as common
import gt4py.next.embedded as gtx_embedded
import gt4py.next.errors.exceptions as exceptions


_closure_column_range: contextvars.ContextVar[common.NamedRange] = contextvars.ContextVar(
    "_column_range"
)
_offset_provider: contextvars.ContextVar[common.OffsetProvider] = contextvars.ContextVar(
    "_offset_provider"
)


_T = TypeVar("_T")


_NO_DEFAULT_SENTINEL: Any = object()


@overload
def get_closure_column_range() -> common.NamedRange: ...


@overload
def get_closure_column_range(default: _T) -> common.NamedRange | _T: ...


def get_closure_column_range(default: _T = _NO_DEFAULT_SENTINEL) -> common.NamedRange | _T:
    """Column range used in 'column mode' in the current embedded iterator closure execution context."""
    result = _closure_column_range.get(default)
    if result is _NO_DEFAULT_SENTINEL:
        raise exceptions.EmbeddedExecutionError(
            "No column range set in the current embedded iterator closure execution context."
        )
    return result


@overload
def get_offset_provider() -> common.OffsetProvider: ...


@overload
def get_offset_provider(default: _T) -> common.OffsetProvider | _T: ...


def get_offset_provider(default: _T = _NO_DEFAULT_SENTINEL) -> common.OffsetProvider | _T:
    """Offset provider used in the current embedded iterator closure execution context."""
    result = _offset_provider.get(default)
    if result is _NO_DEFAULT_SENTINEL:
        raise exceptions.EmbeddedExecutionError(
            "No offset provider set in the current embedded iterator closure execution context."
        )
    return result


@contextlib.contextmanager
def update(
    *,
    closure_column_range: common.NamedRange | eve.NothingType = eve.NOTHING,
    offset_provider: common.OffsetProvider | eve.NothingType = eve.NOTHING,
) -> Generator[None, None, None]:
    """Context handler updating the current embedded context with the provided values."""

    closure_token, offset_provider_token = None, None
    if closure_column_range is not eve.NOTHING:
        assert not isinstance(closure_column_range, eve.NothingType)
        closure_token = gtx_embedded.context._closure_column_range.set(closure_column_range)
    if offset_provider is not eve.NOTHING:
        assert not isinstance(offset_provider, eve.NothingType)
        offset_provider_token = gtx_embedded.context._offset_provider.set(offset_provider)

    try:
        yield None
    finally:
        if closure_column_range is not eve.NOTHING:
            assert closure_token is not None
            gtx_embedded.context._closure_column_range.reset(closure_token)
        if offset_provider is not eve.NOTHING:
            assert offset_provider_token is not None
            gtx_embedded.context._offset_provider.reset(offset_provider_token)


def within_valid_context() -> bool:
    return _offset_provider.get(SENTINEL := object()) is not SENTINEL
