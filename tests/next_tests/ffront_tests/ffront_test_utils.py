# -*- coding: utf-8 -*-
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

#
from collections import namedtuple
from typing import TypeVar

import numpy as np
import pytest

from gt4py.next.common import DimensionKind
from gt4py.next.ffront.fbuiltins import Dimension, FieldOffset
from gt4py.next.iterator.embedded import (
    NeighborTableOffsetProvider,
    index_field,
    np_as_located_field,
)
from gt4py.next.program_processors.runners import gtfn_cpu, roundtrip


@pytest.fixture(params=[roundtrip.executor, gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative])
def fieldview_backend(request):
    yield request.param


def debug_itir(tree):
    """Compare tree snippets while debugging."""
    from devtools import debug

    from gt4py.eve.codegen import format_python_source
    from gt4py.next.program_processors import EmbeddedDSL

    debug(format_python_source(EmbeddedDSL.apply(tree)))


DimsType = TypeVar("DimsType")
DType = TypeVar("DType")

IDim = Dimension("IDim")
JDim = Dimension("JDim")
KDim = Dimension("KDim", kind=DimensionKind.VERTICAL)
Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))
Joff = FieldOffset("Joff", source=JDim, target=(JDim,))
Koff = FieldOffset("Koff", source=KDim, target=(KDim,))

Vertex = Dimension("Vertex")
Edge = Dimension("Edge")
EdgeOffset = FieldOffset("EdgeOffset", source=Edge, target=(Edge,))

size = 10


@pytest.fixture
def reduction_setup():
    num_vertices = 9
    edge = Dimension("Edge")
    vertex = Dimension("Vertex")
    v2edim = Dimension("V2E", kind=DimensionKind.LOCAL)
    e2vdim = Dimension("E2V", kind=DimensionKind.LOCAL)

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
        dtype=np.int32,
    )

    # create e2v connectivity by inverting v2e
    num_edges = np.max(v2e_arr) + 1
    e2v_arr = [[] for _ in range(0, num_edges)]
    for v in range(0, v2e_arr.shape[0]):
        for e in v2e_arr[v]:
            e2v_arr[e].append(v)
    assert all(len(row) == 2 for row in e2v_arr)
    e2v_arr = np.asarray(e2v_arr, dtype=np.int32)

    inp = index_field(edge, dtype=np.int64)
    # TODO(tehrengruber): use index field
    inp = np_as_located_field(edge)(np.array([inp.field_getitem(i) for i in range(num_edges)]))

    yield namedtuple(
        "ReductionSetup",
        [
            "num_vertices",
            "num_edges",
            "Edge",
            "Vertex",
            "V2EDim",
            "E2VDim",
            "V2E",
            "E2V",
            "inp",
            "out",
            "offset_provider",
            "v2e_table",
            "e2v_table",
        ],
    )(
        num_vertices=num_vertices,
        num_edges=num_edges,
        Edge=edge,
        Vertex=vertex,
        V2EDim=v2edim,
        E2VDim=e2vdim,
        V2E=FieldOffset("V2E", source=edge, target=(vertex, v2edim)),
        E2V=FieldOffset("E2V", source=vertex, target=(edge, e2vdim)),
        inp=inp,
        out=np_as_located_field(vertex)(np.zeros([num_vertices], dtype=np.int64)),
        offset_provider={
            "V2E": NeighborTableOffsetProvider(v2e_arr, vertex, edge, 4),
            "E2V": NeighborTableOffsetProvider(e2v_arr, edge, vertex, 2, has_skip_values=False),
            "V2EDim": v2edim,
            "E2VDim": e2vdim,
        },
        v2e_table=v2e_arr,
        e2v_table=e2v_arr,
    )  # type: ignore
