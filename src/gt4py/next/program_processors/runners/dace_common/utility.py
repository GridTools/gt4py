# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import dace

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.program_processors.runners.dace_common import defs as dace_defs
from gt4py.next.type_system import type_specifications as ts


def as_scalar_type(typestr: str) -> ts.ScalarType:
    """Obtain GT4Py scalar type from generic numpy string representation."""
    try:
        kind = getattr(ts.ScalarKind, typestr.upper())
    except AttributeError as ex:
        raise ValueError(f"Data type {typestr} not supported.") from ex
    return ts.ScalarType(kind)


def connectivity_identifier(name: str) -> str:
    return f"{dace_defs.CONNECTIVITY_PREFIX}_{name}"


def field_size_symbol_name(field_name: str, axis: int) -> str:
    return f"__{field_name}_size_{axis}"


def field_stride_symbol_name(field_name: str, axis: int) -> str:
    return f"__{field_name}_stride_{axis}"


def debug_info(
    node: gtir.Node, *, default: Optional[dace.dtypes.DebugInfo] = None
) -> Optional[dace.dtypes.DebugInfo]:
    """Include the GT4Py node location as debug information in the corresponding SDFG nodes."""
    location = node.location
    if location:
        return dace.dtypes.DebugInfo(
            start_line=location.line,
            start_column=location.column if location.column else 0,
            end_line=location.end_line if location.end_line else -1,
            end_column=location.end_column if location.end_column else 0,
            filename=location.filename,
        )
    return default


def filter_connectivities(offset_provider: Mapping[str, Any]) -> dict[str, gtx_common.Connectivity]:
    """
    Filter offset providers of type `Connectivity`.

    In other words, filter out the cartesian offset providers.
    Returns a new dictionary containing only `Connectivity` values.
    """
    return {
        offset: table
        for offset, table in offset_provider.items()
        if isinstance(table, gtx_common.Connectivity)
    }


def get_sorted_dims(
    dims: Sequence[gtx_common.Dimension],
) -> Sequence[tuple[int, gtx_common.Dimension]]:
    """Sort list of dimensions in alphabetical order."""
    return sorted(enumerate(dims), key=lambda v: v[1].value)
