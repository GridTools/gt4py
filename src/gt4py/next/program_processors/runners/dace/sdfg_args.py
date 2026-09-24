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
from gt4py.next.iterator import builtins as gtir_builtins
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners.dace.lowering import gtir_python_codegen
from gt4py.next.type_system import type_specifications as ts


# arrays for connectivity tables use the following prefix
CONNECTIVITY_INDENTIFIER_PREFIX: Final[str] = "gt_conn_"
CONNECTIVITY_INDENTIFIER_RE: Final[re.Pattern] = re.compile(r"^gt_conn_(\S+)$")

# regex for field size/stride symbol name
FIELD_SYMBOL_RE: Final[re.Pattern] = re.compile(r"^__(\S+)_(\S+)_(size|stride)$")

# element data type for field size/stride symbols
FIELD_SYMBOL_DTYPE: Final[dace.typeclass] = getattr(dace, gtir_builtins.INTEGER_INDEX_BUILTIN)


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
    # NOTE: `name` is an offset-provider key, i.e. a qualified tag, which is not a valid SDFG
    # array name; parse it back with `from_codegen_name` (see `is_connectivity_identifier`).
    return f"{CONNECTIVITY_INDENTIFIER_PREFIX}{gtx_common.codegen_name(name)}"


def is_connectivity_identifier(name: str, table_types: gtx_common.TableTypes | None = None) -> bool:
    if (m := CONNECTIVITY_INDENTIFIER_RE.match(name)) is None:
        return False
    elif table_types is None:
        # If no offset provider type is provided, we assume there is a connectivity identifier
        # that matches the CONNECTIVITY_INDENTIFIER_RE.
        return True
    else:
        return gtx_common.has_offset(table_types, gtx_common.from_codegen_name(m[1]))


def _field_symbol(
    field_name: str,
    dim: gtx_common.Dimension,
    sym: Literal["size", "stride"],
    table_types: gtx_common.TableTypes | None,
) -> dace.symbol:
    if (m := CONNECTIVITY_INDENTIFIER_RE.match(field_name)) is None:
        name = f"__{field_name}_{gtx_common.codegen_name(dim.tag)}_{sym}"
    else:  # a connectivity field
        assert table_types is not None
        offset = gtx_common.from_codegen_name(m[1])
        assert offset in table_types
        conn_type = table_types[offset]
        assert isinstance(conn_type, gtx_common.NeighborTableType)
        if dim == conn_type.domain[0]:
            name = f"__{field_name}_source_{sym}"
        elif dim == conn_type.domain[1]:
            name = f"__{field_name}_neighbor_{sym}"
        else:
            raise ValueError(f"Unexpect dimension '{dim}' for '{offset}' connectivity.")
    return dace.symbol(name, FIELD_SYMBOL_DTYPE)


def field_size_symbol(
    field_name: str,
    dim: gtx_common.Dimension,
    table_types: gtx_common.TableTypes,
) -> dace.symbol:
    return _field_symbol(field_name, dim, "size", table_types)


def field_stride_symbol(
    field_name: str,
    dim: gtx_common.Dimension,
    table_types: gtx_common.TableTypes | None = None,
) -> dace.symbol:
    return _field_symbol(field_name, dim, "stride", table_types)


def local_dimension_size(
    field_name: str,
    dim: gtx_common.Dimension,
    neighbor_table_types: dict[str, gtx_common.NeighborTableType],
) -> int:
    """
    Number of neighbors along the local dimension `dim` of the field or connectivity table.

    A connectivity table has its own neighbor count. Any other field finds it in a table over
    `dim`: normally the one keyed by `dim`'s tag, but a connectivity sharing `dim` with another
    one (see `NeighborConnectivity`) is keyed by its own tag, and may be the only one bound.
    """
    if (m := CONNECTIVITY_INDENTIFIER_RE.match(field_name)) is not None:
        own_type = neighbor_table_types[gtx_common.from_codegen_name(m[1])]
        if own_type.domain[1] == dim:
            return own_type.max_neighbors
    return neighbor_table_types[
        gtx_common.connectivity_key_over(neighbor_table_types, dim)
    ].max_neighbors


def _range_symbol_name(field_name: str, dim: gtx_common.Dimension) -> str:
    """Common part of the name for the range start/stop symbols."""
    field_range = im.call("get_domain_range")(field_name, dim)
    return gtir_python_codegen.get_source(field_range)


def range_start_symbol(field_name: str, dim: gtx_common.Dimension) -> dace.symbol:
    """Format name of the start symbol for domain range."""
    name = f"{_range_symbol_name(field_name, dim)}_0"
    return dace.symbol(name, FIELD_SYMBOL_DTYPE)


def range_stop_symbol(field_name: str, dim: gtx_common.Dimension) -> dace.symbol:
    """Format name of the stop symbol for domain range."""
    name = f"{_range_symbol_name(field_name, dim)}_1"
    return dace.symbol(name, FIELD_SYMBOL_DTYPE)


def filter_connectivity_types(
    table_types: gtx_common.TableTypes,
) -> dict[str, gtx_common.NeighborTableType]:
    """
    Filter offset provider types of type `NeighborTableType`.

    In other words, filter out the cartesian offset providers.
    """
    return {
        offset: conn
        for offset, conn in table_types.items()
        if isinstance(conn, gtx_common.NeighborTableType)
    }
