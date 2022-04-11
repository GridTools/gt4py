# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from typing import List, Tuple

import dace
import dace.data
import networkx as nx

from gtc import common, oir
from gtc.dace.nodes import VerticalLoopLibraryNode
from gtc.dace.utils import (
    OIRFieldRenamer,
    array_dimensions,
    dace_dtype_to_typestr,
    get_node_name_mapping,
    internal_symbols,
    validate_oir_sdfg,
)


def sdfg_arrays_to_oir_decls(sdfg: dace.SDFG) -> Tuple[List[oir.Decl], List[oir.Temporary]]:
    params = list()
    decls = list()

    array: dace.data.Data
    for name, array in sdfg.arrays.items():
        dtype = common.typestr_to_data_type(dace_dtype_to_typestr(array.dtype))
        if isinstance(array, dace.data.Array):
            dimensions = array_dimensions(array)
            if not array.transient:
                params.append(
                    oir.FieldDecl(
                        name=name,
                        dtype=dtype,
                        dimensions=dimensions,
                        data_dims=array.shape[sum(dimensions) :],
                    )
                )
            else:
                decls.append(
                    oir.Temporary(
                        name=name,
                        dtype=dtype,
                        dimensions=dimensions,
                        data_dims=array.shape[sum(dimensions) :],
                    )
                )
        else:
            assert isinstance(array, dace.data.Scalar)
            params.append(oir.ScalarDecl(name=name, dtype=dtype))

    reserved_symbols = internal_symbols(sdfg)
    for sym, stype in sdfg.symbols.items():
        if sym not in reserved_symbols:
            params.append(
                oir.ScalarDecl(
                    name=sym, dtype=common.typestr_to_data_type(stype.as_numpy_dtype().str)
                )
            )
    return params, decls


def convert(sdfg: dace.SDFG) -> oir.Stencil:

    validate_oir_sdfg(sdfg)

    params, decls = sdfg_arrays_to_oir_decls(sdfg)
    vertical_loops = []
    for state in sdfg.topological_sort(sdfg.start_state):

        for node in (
            n for n in nx.topological_sort(state.nx) if isinstance(n, VerticalLoopLibraryNode)
        ):

            new_node = OIRFieldRenamer(get_node_name_mapping(state, node)).visit(node.as_oir())
            vertical_loops.append(new_node)

    return oir.Stencil(
        name=sdfg.name, params=params, declarations=decls, vertical_loops=vertical_loops
    )
