# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import gt4py.next as gtx
from gt4py.next.iterator import builtins, ir as itir


class Vertex(gtx.DimensionIndex): ...


class Edge(gtx.DimensionIndex): ...


class Cell(gtx.DimensionIndex): ...


class V2E(gtx.NeighborConnectivity[Vertex, Edge]):
    class Local(gtx.LocalDimensionIndex): ...


class E2V(gtx.NeighborConnectivity[Edge, Vertex]):
    class Local(gtx.LocalDimensionIndex): ...


class C2E(gtx.NeighborConnectivity[Cell, Edge]):
    class Local(gtx.LocalDimensionIndex): ...


class V2V(gtx.NeighborConnectivity[Vertex, Vertex]):
    class Local(gtx.LocalDimensionIndex): ...


V2EDim = V2E.Local
E2VDim = E2V.Local
C2EDim = C2E.Local
V2VDim = V2V.Local

# 3x3 periodic   edges        cells
# 0 - 1 - 2 -    0 1 2
# |   |   |      9 10 11      0 1 2
# 3 - 4 - 5 -    3 4 5
# |   |   |      12 13 14     3 4 5
# 6 - 7 - 8 -    6 7 8
# |   |   |      15 16 17     6 7 8


c2e_arr = np.array(
    [
        [0, 10, 3, 9],  # 0
        [1, 11, 4, 10],
        [2, 9, 5, 11],
        [3, 13, 6, 12],  # 3
        [4, 14, 7, 13],
        [5, 12, 8, 14],
        [6, 16, 0, 15],  # 6
        [7, 17, 1, 16],
        [8, 15, 2, 17],
    ],
    dtype=np.dtype(builtins.INTEGER_INDEX_BUILTIN),
)

c2e_conn = gtx.as_connectivity(domain={Cell: 9, C2EDim: 4}, codomain=Edge, data=c2e_arr)

v2v_arr = np.array(
    [
        [1, 3, 2, 6],
        [2, 3, 0, 7],
        [0, 5, 1, 8],
        [4, 6, 5, 0],
        [5, 7, 3, 1],
        [3, 8, 4, 2],
        [7, 0, 8, 3],
        [8, 1, 6, 4],
        [6, 2, 7, 5],
    ],
    dtype=np.dtype(builtins.INTEGER_INDEX_BUILTIN),
)

v2v_conn = gtx.as_connectivity(domain={Vertex: 9, V2VDim: 4}, codomain=Vertex, data=v2v_arr)

e2v_arr = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 0],
        [3, 4],
        [4, 5],
        [5, 3],
        [6, 7],
        [7, 8],
        [8, 6],
        [0, 3],
        [1, 4],
        [2, 5],
        [3, 6],
        [4, 7],
        [5, 8],
        [6, 0],
        [7, 1],
        [8, 2],
    ],
    dtype=np.dtype(builtins.INTEGER_INDEX_BUILTIN),
)

e2v_conn = gtx.as_connectivity(domain={Edge: 18, E2VDim: 2}, codomain=Vertex, data=e2v_arr)

# order east, north, west, south (counter-clock wise)
v2e_arr = np.array(
    [
        [0, 15, 2, 9],  # 0
        [1, 16, 0, 10],
        [2, 17, 1, 11],
        [3, 9, 5, 12],  # 3
        [4, 10, 3, 13],
        [5, 11, 4, 14],
        [6, 12, 8, 15],  # 6
        [7, 13, 6, 16],
        [8, 14, 7, 17],
    ],
    dtype=np.dtype(builtins.INTEGER_INDEX_BUILTIN),
)

v2e_conn = gtx.as_connectivity(domain={Vertex: 9, V2EDim: 4}, codomain=Edge, data=v2e_arr)
