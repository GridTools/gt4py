# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from typing import Final, Literal, Mapping, Union

import dace

from gt4py.next import common as gtx_common
from gt4py.next.type_system import type_specifications as ts


# arrays for connectivity tables use the following prefix
CONNECTIVITY_INDENTIFIER_PREFIX: Final[str] = "gt_conn_"
CONNECTIVITY_INDENTIFIER_RE: Final[re.Pattern] = re.compile(r"^gt_conn_(.+)$")


# regex to match the symbols for field shape and strides
FIELD_SYMBOL_RE: Final[re.Pattern] = re.compile(r"^__(.+)_((\d+_range_[01])|((size|stride)_\d+))$")


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


def is_connectivity_symbol(name: str, offset_provider_type: gtx_common.OffsetProviderType) -> bool:
    m = FIELD_SYMBOL_RE.match(name)
    if m is None:
        return False
    m = CONNECTIVITY_INDENTIFIER_RE.match(m[1])
    if m is None:
        return False
    return m[1] in offset_provider_type


def field_symbol_name(field_name: str, axis: int, sym: Literal["size", "stride"]) -> str:
    return f"__{field_name}_{sym}_{axis}"


def field_size_symbol_name(field_name: str, axis: int) -> str:
    return field_symbol_name(field_name, axis, "size")


def field_stride_symbol_name(field_name: str, axis: int) -> str:
    return field_symbol_name(field_name, axis, "stride")


def range_start_symbol(field_name: str, axis: int) -> str:
    """Format name of start symbol for domain range, as expected by GTIR."""
    return f"__{field_name}_{axis}_range_0"


def range_stop_symbol(field_name: str, axis: int) -> str:
    """Format name of stop symbol for domain range, as expected by GTIR."""
    return f"__{field_name}_{axis}_range_1"


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


def safe_replace_symbolic(
    val: dace.symbolic.SymbolicType,
    symbol_mapping: Mapping[
        Union[dace.symbolic.SymbolicType, str], Union[dace.symbolic.SymbolicType, str]
    ],
) -> dace.symbolic.SymbolicType:
    """
    Replace free symbols in a dace symbolic expression, using `safe_replace()`
    in order to avoid clashes in case the new symbol value is also a free symbol
    in the original exoression.

    Args:
        val: The symbolic expression where to apply the replacement.
        symbol_mapping: The mapping table for symbol replacement.

    Returns:
        A new symbolic expression as result of symbol replacement.
    """
    # The list `x` is needed because `subs()` returns a new object and can not handle
    # replacement dicts of the form `{'x': 'y', 'y': 'x'}`.
    # The utility `safe_replace()` will call `subs()` twice in case of such dicts.
    x = [val]
    dace.symbolic.safe_replace(symbol_mapping, lambda m, xx=x: xx.append(xx[-1].subs(m)))
    return x[-1]
