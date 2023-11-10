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

import numpy as np

import gt4py.next as gtx
from gt4py.next.iterator import ir as itir


Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
Cell = gtx.Dimension("Cell")
V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
V2VDim = gtx.Dimension("V2V", kind=gtx.DimensionKind.LOCAL)

V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))
V2V = gtx.FieldOffset("V2V", source=Vertex, target=(Vertex, V2VDim))

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
    dtype=np.dtype(itir.INTEGER_INDEX_BUILTIN),
)

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
    dtype=np.dtype(itir.INTEGER_INDEX_BUILTIN),
)

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
    dtype=np.dtype(itir.INTEGER_INDEX_BUILTIN),
)


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
    dtype=np.dtype(itir.INTEGER_INDEX_BUILTIN),
)
