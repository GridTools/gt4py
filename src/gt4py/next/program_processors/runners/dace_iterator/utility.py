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

from typing import Any

import dace

from gt4py.next.iterator.embedded import NeighborTableOffsetProvider
from gt4py.next.type_system import type_specifications as ts


def type_spec_to_dtype(type_: ts.ScalarType):
    if type_.kind == ts.ScalarKind.BOOL:
        return dace.bool_
    elif type_.kind == ts.ScalarKind.INT32:
        return dace.int32
    elif type_.kind == ts.ScalarKind.INT64:
        return dace.int64
    elif type_.kind == ts.ScalarKind.FLOAT32:
        return dace.float32
    elif type_.kind == ts.ScalarKind.FLOAT64:
        return dace.float64
    raise ValueError(f"scalar type {type_} not supported")


def filter_neighbor_tables(offset_provider: dict[str, Any]):
    return [
        (offset, table)
        for offset, table in offset_provider.items()
        if isinstance(table, NeighborTableOffsetProvider)
    ]


def connectivity_identifier(name: str):
    return f"__connectivity_{name}"


def create_memlet_full(source_identifier: str, source_array: dace.data.Array):
    bounds = [(0, size) for size in source_array.shape]
    subset = ", ".join(f"{lb}:{ub}" for lb, ub in bounds)
    return dace.Memlet(data=source_identifier, subset=subset)


def create_memlet_at(source_identifier: str, index: tuple[str, ...]):
    subset = ", ".join(index)
    return dace.Memlet(data=source_identifier, subset=subset)
