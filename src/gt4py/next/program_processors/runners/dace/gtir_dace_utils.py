# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from typing import Final, Literal

import dace

from gt4py.next import common as gtx_common
from gt4py.next.type_system import type_specifications as ts


# arrays for connectivity tables use the following prefix
CONNECTIVITY_INDENTIFIER_PREFIX: Final[str] = "connectivity_"
CONNECTIVITY_INDENTIFIER_RE: Final[re.Pattern] = re.compile(r"^connectivity_(.+)$")


# regex to match the symbols for field shape and strides
FIELD_SYMBOL_RE: Final[re.Pattern] = re.compile(r"^__.+_((\d+_range_[01])|((size|stride)_\d+))$")


def as_dace_type(type_: ts.ScalarType) -> dace.typeclass:
    """Converts GT4Py scalar type to corresponding DaCe type."""

    match type_.kind:
        case ts.ScalarKind.BOOL:
            return dace.bool_
        case ts.ScalarKind():
            return getattr(dace, type_.kind.name.lower())
        case _:
            raise ValueError(f"Scalar type '{type_}' not supported.")


def as_itir_type(dtype: dace.typeclass) -> ts.ScalarType:
    """Get GT4Py scalar representation of a DaCe type."""
    type_name = str(dtype.as_numpy_dtype())
    try:
        kind = getattr(ts.ScalarKind, type_name.upper())
    except AttributeError as ex:
        raise ValueError(f"Data type {type_name} not supported.") from ex
    return ts.ScalarType(kind)


def connectivity_identifier(name: str) -> str:
    return f"{CONNECTIVITY_INDENTIFIER_PREFIX}{name}"


def is_connectivity_identifier(
    name: str, offset_provider_type: gtx_common.OffsetProviderType
) -> bool:
    m = CONNECTIVITY_INDENTIFIER_RE.match(name)
    if m is None:
        return False
    return m[1] in offset_provider_type


def field_symbol_name(field_name: str, axis: int, sym: Literal["size", "stride"]) -> str:
    return f"__{field_name}_{sym}_{axis}"


def field_size_symbol_name(field_name: str, axis: int) -> str:
    return field_symbol_name(field_name, axis, "size")


def field_stride_symbol_name(field_name: str, axis: int) -> str:
    return field_symbol_name(field_name, axis, "stride")


def is_field_symbol(name: str) -> bool:
    return FIELD_SYMBOL_RE.match(name) is not None


def filter_connectivity_types(
    offset_provider_type: gtx_common.OffsetProviderType,
) -> dict[str, gtx_common.NeighborConnectivityType]:
    """
    Filter offset provider types of type `NeighborConnectivityType`.

    In other words, filter out the cartesian offset providers.
    """
    return {
        offset: conn
        for offset, conn in offset_provider_type.items()
        if isinstance(conn, gtx_common.NeighborConnectivityType)
    }
