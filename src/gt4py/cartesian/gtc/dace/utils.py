# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import re
from functools import lru_cache

import numpy as np
from dace import data, dtypes, symbolic

from gt4py import eve
from gt4py.cartesian.gtc import common


def get_dace_debuginfo(node: common.LocNode) -> dtypes.DebugInfo:
    if node.loc is None:
        return dtypes.DebugInfo(0)

    return dtypes.DebugInfo(
        node.loc.line, node.loc.column, node.loc.line, node.loc.column, node.loc.filename
    )


def array_dimensions(array: data.Array) -> list[bool]:
    return [
        any(
            re.match(f"__.*_{k}_stride", str(sym))
            for st in array.strides
            for sym in symbolic.pystr_to_symbolic(st).free_symbols
        )
        or any(
            re.match(f"__{k}", str(sym))
            for sh in array.shape
            for sym in symbolic.pystr_to_symbolic(sh).free_symbols
        )
        for k in "IJK"
    ]


def replace_strides(arrays: list[data.Array], get_layout_map) -> dict[str, str]:
    symbol_mapping = {}
    for array in arrays:
        dims = array_dimensions(array)
        ndata_dims = len(array.shape) - sum(dims)
        axes = [ax for ax, m in zip("IJK", dims) if m] + [str(i) for i in range(ndata_dims)]
        layout = get_layout_map(axes)
        if array.transient:
            stride = 1
            for idx in reversed(np.argsort(layout)):
                symbol = array.strides[idx]
                if symbol.is_symbol:
                    symbol_mapping[str(symbol)] = symbolic.pystr_to_symbolic(stride)
                stride *= array.shape[idx]
    return symbol_mapping


def data_type_to_dace_typeclass(data_type: common.DataType) -> dtypes.typeclass:
    dtype = np.dtype(common.data_type_to_typestr(data_type))
    return dtypes.typeclass(dtype.type)


@lru_cache(maxsize=None)
def get_dace_symbol(
    name: eve.SymbolRef, dtype: common.DataType = common.DataType.INT32
) -> symbolic.symbol:
    return symbolic.symbol(name, dtype=data_type_to_dace_typeclass(dtype))
