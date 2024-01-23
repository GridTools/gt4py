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
from gt4py.next import float64, neighbor_sum


CellDim = gtx.Dimension("Cell")
EdgeDim = gtx.Dimension("Edge")
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
E2C = gtx.FieldOffset("E2C", source=CellDim, target=(EdgeDim, E2CDim))

edge_to_cell_table = np.array(
    [
        [0, -1],  # edge 0 (neighbours: cell 0)
        [2, -1],  # edge 1
        [2, -1],  # edge 2
        [3, -1],  # edge 3
        [4, -1],  # edge 4
        [5, -1],  # edge 5
        [0, 5],  # edge 6 (neighbours: cell 0, cell 5)
        [0, 1],  # edge 7
        [1, 2],  # edge 8
        [1, 3],  # edge 9
        [3, 4],  # edge 10
        [4, 5],  # edge 11
    ]
)

E2C_offset_provider = gtx.NeighborTableOffsetProvider(
    edge_to_cell_table, EdgeDim, CellDim, 2, has_skip_values=True
)


@gtx.field_operator
def sum_adjacent_cells(cells: gtx.Field[[CellDim], float64]) -> gtx.Field[[EdgeDim], float64]:
    # type of cells(E2C) is gtx.Field[[CellDim, E2CDim], float64]
    return neighbor_sum(cells(E2C), axis=E2CDim)


@gtx.program
def run_sum_adjacent_cells(
    cells: gtx.Field[[CellDim], float64], out: gtx.Field[[EdgeDim], float64]
):
    sum_adjacent_cells(cells, out=out)


cell_values = gtx.as_field([CellDim], np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0]))
edge_values = gtx.as_field([EdgeDim], np.zeros((12,)))

run_sum_adjacent_cells(cell_values, edge_values, offset_provider={"E2C": E2C_offset_provider})

print(np.clip(edge_to_cell_table, 0, 5))
ref = cell_values.ndarray[np.clip(edge_to_cell_table, 0, 5)]
print(ref)
ref = np.sum(ref, initial=0, where=edge_to_cell_table >= 0, axis=1)


print("sum of adjacent cells: {}".format(edge_values.asnumpy()))
print(ref)
