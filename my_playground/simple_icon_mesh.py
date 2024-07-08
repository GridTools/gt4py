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

"""Reimplementation of the simple icon grid for testing."""

import numpy as np

from gt4py.next.common import DimensionKind
from gt4py.next.ffront.fbuiltins import Dimension, FieldOffset
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider
from gt4py.next.type_system import type_specifications as ts

IDim = Dimension("IDim")
JDim = Dimension("JDim")

KDim = Dimension("K", kind=DimensionKind.VERTICAL)
EdgeDim = Dimension("Edge")
CellDim = Dimension("Cell")
VertexDim = Dimension("Vertex")
ECVDim = Dimension("ECV")
E2C2VDim = Dimension("E2C2V", DimensionKind.LOCAL)

E2ECV = FieldOffset("E2ECV", source=ECVDim, target=(EdgeDim, E2C2VDim))
E2C2V = FieldOffset("E2C2V", source=VertexDim, target=(EdgeDim, E2C2VDim))

Koff = FieldOffset("Koff", source=KDim, target=(KDim,))

NbCells = 18
NbEdges = 27
NbVertices = 9

SIZE_TYPE = ts.ScalarType(ts.ScalarKind.INT32)

e2c2v_table = np.asarray(
    [
        [0, 1, 4, 6],  # 0
        [0, 4, 1, 3],  # 1
        [0, 3, 4, 2],  # 2
        [1, 2, 5, 7],  # 3
        [1, 5, 2, 4],  # 4
        [1, 4, 5, 0],  # 5
        [2, 0, 3, 8],  # 6
        [2, 3, 5, 0],  # 7
        [2, 5, 1, 3],  # 8
        [3, 4, 0, 7],  # 9
        [3, 7, 4, 6],  # 10
        [3, 6, 7, 5],  # 11
        [4, 5, 8, 1],  # 12
        [4, 8, 7, 5],  # 13
        [4, 7, 3, 8],  # 14
        [5, 3, 6, 2],  # 15
        [6, 5, 3, 8],  # 16
        [8, 5, 6, 4],  # 17
        [6, 7, 3, 1],  # 18
        [6, 1, 7, 0],  # 19
        [6, 0, 1, 8],  # 20
        [7, 8, 2, 4],  # 21
        [7, 2, 8, 1],  # 22
        [7, 1, 2, 6],  # 23
        [8, 6, 0, 5],  # 24
        [8, 0, 6, 2],  # 25
        [8, 2, 0, 6],  # 26
    ]
)

E2C2V_connectivity = NeighborTableOffsetProvider(
    # I do not understand the ordering here? Why is `Edge` the source if you read
    #  it right to left?
    e2c2v_table,
    EdgeDim,
    VertexDim,
    e2c2v_table.shape[1],
)


def _make_E2ECV_connectivity(E2C2V_connectivity: NeighborTableOffsetProvider):
    # Implementation is adapted from icon's `_get_offset_provider_for_sparse_fields()`
    e2c2v_table = E2C2V_connectivity.table
    t = np.arange(e2c2v_table.shape[0] * e2c2v_table.shape[1]).reshape(e2c2v_table.shape)
    return NeighborTableOffsetProvider(t, EdgeDim, ECVDim, t.shape[1])


E2ECV_connectivity = _make_E2ECV_connectivity(E2C2V_connectivity)


def dace_strides(
    array: np.ndarray,
    name: None | str = None,
) -> tuple[int, ...] | dict[str, int]:
    if not hasattr(array, "strides"):
        return {}
    strides = array.strides
    if hasattr(array, "itemsize"):
        strides = tuple(stride // array.itemsize for stride in strides)
    if name is not None:
        strides = {f"__{name}_stride_{i}": stride for i, stride in enumerate(strides)}
    return strides


def dace_shape(
    array: np.ndarray,
    name: str,
) -> dict[str, int]:
    if not hasattr(array, "shape"):
        return {}
    return {f"__{name}_size_{i}": size for i, size in enumerate(array.shape)}


def make_syms(**kwargs: np.ndarray) -> dict[str, int]:
    SYMBS = {}
    for name, array in kwargs.items():
        SYMBS.update(**dace_shape(array, name))
        SYMBS.update(**dace_strides(array, name))
    return SYMBS
