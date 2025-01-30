# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import gt4py.next as gtx
from gt4py.next.iterator.embedded import MutableLocatedField
from gt4py.next import neighbor_sum, where, Dims
from gt4py.next import Dimension, DimensionKind, FieldOffset
from gt4py.next.program_processors.runners import roundtrip
from gt4py.next.program_processors.runners.gtfn import (
    run_gtfn_cached as gtfn_cpu,
    run_gtfn_gpu as gtfn_gpu,
)


def random_mask(sizes, *dims, dtype=None) -> MutableLocatedField:
    arr = np.full(shape=sizes, fill_value=False).flatten()
    arr[: int(arr.size * 0.5)] = True
    np.random.shuffle(arr)
    arr = np.reshape(arr, newshape=sizes)
    if dtype:
        arr = arr.astype(dtype)
    return gtx.as_field([*dims], (arr))


def random_field(
    domain: gtx.Domain, low: float = -1.0, high: float = 1.0, *, allocator=None
) -> MutableLocatedField:
    return gtx.as_field(
        domain,
        np.random.default_rng().uniform(low=low, high=high, size=domain.shape),
        allocator=allocator,
    )


def random_sign(domain: gtx.Domain, allocator=None, dtype=float) -> MutableLocatedField:
    return gtx.as_field(
        domain,
        np.asarray(np.random.randint(0, high=2, size=domain.shape) * 2 - 1, dtype=dtype),
        allocator=allocator,
    )


def ripple_field(domain: gtx.Domain, *, allocator=None) -> MutableLocatedField:
    assert domain.ndim == 2
    nx, ny = domain.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    data = (
        5.0
        + 8.0 * (2.0 + np.cos(np.pi * (xx + 1.5 * yy)) + np.sin(2 * np.pi * (xx + 1.5 * yy))) / 4.0
    )

    return gtx.as_field(domain, data, allocator=allocator)


# For simplicity we use a triangulated donut in the horizontal.

# 0v---0e-- 1v---3e-- 2v---6e-- 0v
# |  \ 0c   |  \ 1c   |  \2c
# |   \1e   |   \4e   |   \7e
# |2e   \   |5e   \   |8e   \
# |  3c   \ |   4c  \ |    5c\
# 3v---9e-- 4v--12e-- 5v--15e-- 3v
# |  \ 6c   |  \ 7c   |  \ 8c
# |   \10e  |   \13e  |   \16e
# |11e  \   |14e  \   |17e  \
# |  9c  \  |  10c \  |  11c \
# 6v--18e-- 7v--21e-- 8v--24e-- 6v
# |  \12c   |  \ 13c  |  \ 14c
# |   \19e  |   \22e  |   \25e
# |20e  \   |23e  \   |26e  \
# |  15c  \ | 16c   \ | 17c  \
# 0v       1v         2v        0v


n_edges = 27
n_vertices = 9
n_cells = 18
n_levels = 10


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

e2c_table = np.asarray(
    [
        [0, 15],
        [0, 3],
        [3, 2],
        [1, 16],
        [1, 4],
        [0, 4],
        [2, 17],
        [2, 5],
        [1, 5],
        [3, 6],
        [6, 9],
        [9, 8],
        [4, 7],
        [7, 10],
        [6, 10],
        [5, 8],
        [8, 11],
        [7, 11],
        [9, 12],
        [12, 15],
        [15, 14],
        [10, 13],
        [13, 16],
        [12, 16],
        [11, 14],
        [14, 17],
        [13, 17],
    ]
)

e2v_table = np.asarray(
    [
        [0, 1],
        [0, 4],
        [0, 3],
        [1, 2],
        [1, 5],
        [1, 4],
        [2, 0],
        [2, 3],
        [2, 5],
        [3, 4],
        [3, 7],
        [3, 6],
        [4, 5],
        [4, 8],
        [4, 7],
        [5, 3],
        [5, 6],
        [5, 8],
        [6, 7],
        [6, 1],
        [6, 0],
        [7, 8],
        [7, 2],
        [7, 1],
        [8, 6],
        [8, 0],
        [8, 2],
    ]
)

e2c2e_table = np.asarray(
    [
        [1, 5, 19, 20],
        [0, 5, 2, 9],
        [1, 9, 6, 7],
        [4, 8, 22, 23],
        [3, 8, 5, 12],
        [0, 1, 4, 12],
        [7, 2, 25, 26],
        [6, 2, 8, 15],
        [3, 4, 7, 15],
        [1, 2, 10, 14],
        [9, 14, 11, 18],
        [10, 18, 15, 16],
        [4, 5, 13, 17],
        [12, 17, 14, 21],
        [9, 10, 13, 21],
        [7, 8, 16, 11],
        [15, 11, 17, 24],
        [12, 13, 16, 24],
        [10, 11, 19, 23],
        [18, 23, 20, 0],
        [19, 0, 24, 25],
        [13, 14, 22, 26],
        [21, 26, 23, 3],
        [18, 19, 22, 3],
        [16, 17, 25, 20],
        [24, 20, 26, 6],
        [25, 6, 21, 22],
    ]
)

e2c2eO_table = np.asarray(
    [
        [0, 1, 5, 19, 20],
        [0, 1, 5, 2, 9],
        [1, 2, 9, 6, 7],
        [3, 4, 8, 22, 23],
        [3, 4, 8, 5, 12],
        [0, 1, 5, 4, 12],
        [6, 7, 2, 25, 26],
        [6, 7, 2, 8, 15],
        [3, 4, 8, 7, 15],
        [1, 2, 9, 10, 14],
        [9, 10, 14, 11, 18],
        [10, 11, 18, 15, 16],
        [4, 5, 12, 13, 17],
        [12, 13, 17, 14, 21],
        [9, 10, 14, 13, 21],
        [7, 8, 15, 16, 11],
        [15, 16, 11, 17, 24],
        [12, 13, 17, 16, 24],
        [10, 11, 18, 19, 23],
        [18, 19, 23, 20, 0],
        [19, 20, 0, 24, 25],
        [13, 14, 21, 22, 26],
        [21, 22, 26, 23, 3],
        [18, 19, 23, 22, 3],
        [16, 17, 24, 25, 20],
        [24, 25, 20, 26, 6],
        [25, 26, 6, 21, 22],
    ]
)

c2e_table = np.asarray(
    [
        [0, 1, 5],  # cell 0
        [3, 4, 8],  # cell 1
        [6, 7, 2],  # cell 2
        [1, 2, 9],  # cell 3
        [4, 5, 12],  # cell 4
        [7, 8, 15],  # cell 5
        [9, 10, 14],  # cell 6
        [12, 13, 17],  # cell 7
        [15, 16, 11],  # cell 8
        [10, 11, 18],  # cell 9
        [13, 14, 21],  # cell 10
        [16, 17, 24],  # cell 11
        [18, 19, 23],  # cell 12
        [21, 22, 26],  # cell 13
        [24, 25, 20],  # cell 14
        [19, 20, 0],  # cell 15
        [22, 23, 3],  # cell 16
        [25, 26, 6],  # cell 17
    ]
)

v2c_table = np.asarray(
    [
        [17, 14, 3, 0, 2, 15],
        [0, 4, 1, 12, 16, 15],
        [1, 5, 2, 16, 13, 17],
        [3, 6, 9, 5, 8, 2],
        [6, 10, 7, 4, 0, 3],
        [7, 11, 8, 5, 1, 4],
        [9, 12, 15, 8, 11, 14],
        [12, 16, 13, 10, 6, 9],
        [13, 17, 14, 11, 7, 10],
    ]
)

v2e_table = np.asarray(
    [
        [0, 1, 2, 6, 25, 20],
        [3, 4, 5, 0, 23, 19],
        [6, 7, 8, 3, 22, 26],
        [9, 10, 11, 15, 7, 2],
        [12, 13, 14, 9, 1, 5],
        [15, 16, 17, 12, 4, 8],
        [18, 19, 20, 24, 16, 11],
        [21, 22, 23, 18, 10, 14],
        [24, 25, 26, 21, 13, 17],
    ]
)

diamond_table = np.asarray(
    [
        [0, 1, 4, 6],  # 0
        [0, 4, 1, 3],
        [0, 3, 4, 2],
        [1, 2, 5, 7],  # 3
        [1, 5, 2, 4],
        [1, 4, 5, 0],
        [2, 0, 3, 8],  # 6
        [2, 3, 0, 5],
        [2, 5, 1, 3],
        [3, 4, 0, 7],  # 9
        [3, 7, 4, 6],
        [3, 6, 5, 7],
        [4, 5, 1, 8],  # 12
        [4, 8, 5, 7],
        [4, 7, 3, 8],
        [5, 3, 2, 6],  # 15
        [5, 6, 3, 8],
        [5, 8, 4, 6],
        [6, 7, 3, 1],  # 18
        [6, 1, 7, 0],
        [6, 0, 1, 8],
        [7, 8, 4, 2],  # 21
        [7, 2, 8, 1],
        [7, 1, 6, 2],
        [8, 6, 5, 0],  # 24
        [8, 0, 6, 2],
        [8, 2, 7, 0],
    ]
)

c2e2cO_table = np.asarray(
    [
        [15, 4, 3, 0],
        [16, 5, 4, 1],
        [17, 3, 5, 2],
        [0, 6, 2, 3],
        [1, 7, 0, 4],
        [2, 8, 1, 5],
        [3, 10, 9, 6],
        [4, 11, 10, 7],
        [5, 9, 11, 8],
        [6, 12, 8, 9],
        [7, 13, 6, 10],
        [8, 14, 7, 11],
        [9, 16, 15, 12],
        [10, 17, 16, 13],
        [11, 15, 17, 14],
        [12, 0, 14, 15],
        [13, 1, 12, 16],
        [14, 2, 13, 17],
    ]
)

c2e2c_table = np.asarray(
    [
        [15, 4, 3],
        [16, 5, 4],
        [17, 3, 5],
        [0, 6, 2],
        [1, 7, 0],
        [2, 8, 1],
        [3, 10, 9],
        [4, 11, 10],
        [5, 9, 11],
        [6, 12, 8],
        [7, 13, 6],
        [8, 14, 7],
        [9, 16, 15],
        [10, 17, 16],
        [11, 15, 17],
        [12, 0, 14],
        [13, 1, 12],
        [14, 2, 13],
    ]
)


C = Dimension("C")
V = Dimension("V")
E = Dimension("E")
K = Dimension("K", kind=gtx.DimensionKind.VERTICAL)

C2EDim = Dimension("C2E", kind=DimensionKind.LOCAL)
C2E = FieldOffset("C2E", source=E, target=(C, C2EDim))
V2EDim = Dimension("V2E", kind=DimensionKind.LOCAL)
V2E = FieldOffset("V2E", source=E, target=(V, V2EDim))
E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
E2V = FieldOffset("E2V", source=V, target=(E, E2VDim))
E2CDim = Dimension("E2C", kind=DimensionKind.LOCAL)
E2C = FieldOffset("E2C", source=C, target=(E, E2CDim))
E2C2VDim = Dimension("E2C2V", kind=DimensionKind.LOCAL)
E2C2V = FieldOffset("E2C2V", source=V, target=(E, E2C2VDim))
Coff = FieldOffset("Coff", source=C, target=(C,))  # delete this?
Koff = FieldOffset("Koff", source=K, target=(K,))
