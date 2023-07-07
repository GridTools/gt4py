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


def as_dace_type(type_: ts.ScalarType):
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


def map_nested_sdfg_symbols(
    parent_sdfg: dace.SDFG,
    nested_sdfg: dace.SDFG,
    array_mapping: dict[str, dace.Memlet],
    dim_index: int = None,
) -> dict[str, Any]:
    symbol_mapping: dict[str, Any] = {}
    for param, arg in array_mapping.items():
        arg_array = parent_sdfg.arrays[arg.data]
        param_array = nested_sdfg.arrays[param]
        if isinstance(param_array, dace.data.Scalar):
            assert arg.subset.num_elements() == 1
        elif dim_index:
            assert len(param_array.shape) == 1
            arg_shape = arg.subset.size()[dim_index]
            param_shape = param_array.shape[0]
            if isinstance(param_shape, dace.symbol):
                symbol_mapping[str(param_shape)] = str(arg_shape)
            assert len(param_array.strides) == 1
            arg_stride = arg_array.strides[dim_index]
            param_stride = param_array.strides[0]
            if isinstance(param_stride, dace.symbol):
                symbol_mapping[str(param_stride)] = str(arg_stride)
        else:
            assert len(arg.subset.size()) == len(param_array.shape)
            for arg_shape, param_shape in zip(arg.subset.size(), param_array.shape):
                if isinstance(param_shape, dace.symbol):
                    symbol_mapping[str(param_shape)] = str(arg_shape)
            assert len(arg_array.strides) == len(param_array.strides)
            for arg_stride, param_stride in zip(arg_array.strides, param_array.strides):
                if isinstance(param_stride, dace.symbol):
                    symbol_mapping[str(param_stride)] = str(arg_stride)
    for sym in nested_sdfg.free_symbols:
        if str(sym) not in symbol_mapping:
            symbol_mapping[str(sym)] = str(sym)
    return symbol_mapping


_unique_id = 0


def unique_var_name():
    global _unique_id
    _unique_id += 1
    return f"__var_{_unique_id}"
