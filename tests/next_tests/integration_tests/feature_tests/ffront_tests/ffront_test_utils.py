# -*- coding: utf-8 -*-
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

from collections import namedtuple
from typing import Any, TypeVar

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.ffront import decorator
from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.runners import gtfn, roundtrip


try:
    from gt4py.next.program_processors.runners import dace_iterator
except ModuleNotFoundError as e:
    if "dace" in str(e):
        dace_iterator = None
    else:
        raise e

import next_tests
import next_tests.exclusion_matrices as definitions


def no_backend(program: itir.FencilDefinition, *args: Any, **kwargs: Any) -> None:
    """Temporary default backend to not accidentally test the wrong backend."""
    raise ValueError("No backend selected! Backend selection is mandatory in tests.")


OPTIONAL_PROCESSORS = []
if dace_iterator:
    OPTIONAL_PROCESSORS.append(definitions.OptionalProgramBackendId.DACE_CPU)
    OPTIONAL_PROCESSORS.append(
        pytest.param(definitions.OptionalProgramBackendId.DACE_GPU, marks=pytest.mark.requires_gpu)
    ),


@pytest.fixture(
    params=[
        definitions.ProgramBackendId.ROUNDTRIP,
        definitions.ProgramBackendId.GTFN_CPU,
        definitions.ProgramBackendId.GTFN_CPU_IMPERATIVE,
        definitions.ProgramBackendId.GTFN_CPU_WITH_TEMPORARIES,
        pytest.param(definitions.ProgramBackendId.GTFN_GPU, marks=pytest.mark.requires_gpu),
        None,
    ]
    + OPTIONAL_PROCESSORS,
    ids=lambda p: p.short_id() if p is not None else "None",
)
def fieldview_backend(request):
    """
    Fixture creating field-view operator backend on-demand for tests.

    Notes:
        Check ADR 15 for details on the test-exclusion matrices.
    """
    backend_id = request.param
    backend = None if backend_id is None else backend_id.load()

    for marker, skip_mark, msg in next_tests.exclusion_matrices.BACKEND_SKIP_TEST_MATRIX.get(
        backend_id, []
    ):
        if request.node.get_closest_marker(marker):
            skip_mark(msg.format(marker=marker, backend=backend_id))

    backup_backend = decorator.DEFAULT_BACKEND
    decorator.DEFAULT_BACKEND = no_backend
    yield backend
    decorator.DEFAULT_BACKEND = backup_backend


def debug_itir(tree):
    """Compare tree snippets while debugging."""
    from devtools import debug

    from gt4py.eve.codegen import format_python_source
    from gt4py.next.program_processors import EmbeddedDSL

    debug(format_python_source(EmbeddedDSL.apply(tree)))


DimsType = TypeVar("DimsType")
DType = TypeVar("DType")

IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)
Ioff = gtx.FieldOffset("Ioff", source=IDim, target=(IDim,))
Joff = gtx.FieldOffset("Joff", source=JDim, target=(JDim,))
Koff = gtx.FieldOffset("Koff", source=KDim, target=(KDim,))

Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
Cell = gtx.Dimension("Cell")
EdgeOffset = gtx.FieldOffset("EdgeOffset", source=Edge, target=(Edge,))

size = 10


@pytest.fixture
def reduction_setup():
    num_vertices = 9
    num_cells = 8
    k_levels = 10
    v2edim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
    e2vdim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
    c2vdim = gtx.Dimension("C2V", kind=gtx.DimensionKind.LOCAL)
    c2edim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)

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
        dtype=gtx.IndexType,
    )

    c2v_arr = np.array(
        [
            [0, 1, 4, 3],
            [1, 2, 5, 6],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
            [6, 7, 1, 0],
            [7, 8, 2, 1],
            [2, 0, 3, 5],
            [5, 3, 6, 8],
        ],
        dtype=gtx.IndexType,
    )

    c2e_arr = np.array(
        [
            [0, 10, 3, 9],
            [1, 11, 4, 10],
            [3, 13, 6, 12],
            [4, 14, 7, 13],
            [6, 16, 0, 15],
            [7, 17, 1, 16],
            [2, 9, 5, 11],
            [5, 12, 8, 14],
        ],
        dtype=gtx.IndexType,
    )

    # create e2v connectivity by inverting v2e
    num_edges = np.max(v2e_arr) + 1
    e2v_arr = [[] for _ in range(0, num_edges)]
    for v in range(0, v2e_arr.shape[0]):
        for e in v2e_arr[v]:
            e2v_arr[e].append(v)
    assert all(len(row) == 2 for row in e2v_arr)
    e2v_arr = np.asarray(e2v_arr, dtype=gtx.IndexType)

    yield namedtuple(
        "ReductionSetup",
        [
            "num_vertices",
            "num_edges",
            "num_cells",
            "k_levels",
            "V2EDim",
            "E2VDim",
            "C2VDim",
            "C2EDim",
            "V2E",
            "E2V",
            "C2V",
            "C2E",
            "inp",
            "out",
            "offset_provider",
            "v2e_table",
            "e2v_table",
        ],
    )(
        num_vertices=num_vertices,
        num_edges=num_edges,
        num_cells=num_cells,
        k_levels=k_levels,
        V2EDim=v2edim,
        E2VDim=e2vdim,
        C2VDim=c2vdim,
        C2EDim=c2edim,
        V2E=gtx.FieldOffset("V2E", source=Edge, target=(Vertex, v2edim)),
        E2V=gtx.FieldOffset("E2V", source=Vertex, target=(Edge, e2vdim)),
        C2V=gtx.FieldOffset("C2V", source=Vertex, target=(Cell, c2vdim)),
        C2E=gtx.FieldOffset("C2E", source=Edge, target=(Cell, c2edim)),
        # inp=gtx.index_field(edge, dtype=np.int64), # TODO enable once we support gtx.index_fields in bindings
        inp=gtx.as_field([Edge], np.arange(num_edges, dtype=np.int32)),
        out=gtx.as_field([Vertex], np.zeros([num_vertices], dtype=np.int32)),
        offset_provider={
            "V2E": gtx.NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4, has_skip_values=False),
            "E2V": gtx.NeighborTableOffsetProvider(e2v_arr, Edge, Vertex, 2, has_skip_values=False),
            "C2V": gtx.NeighborTableOffsetProvider(c2v_arr, Cell, Vertex, 4, has_skip_values=False),
            "C2E": gtx.NeighborTableOffsetProvider(c2e_arr, Cell, Edge, 4, has_skip_values=False),
        },
        v2e_table=v2e_arr,
        e2v_table=e2v_arr,
    )  # type: ignore


__all__ = [
    "fieldview_backend",
    "reduction_setup",
    "debug_itir",
    "DimsType",
    "DType",
    "IDim",
    "JDim",
    "KDim",
    "Ioff",
    "Joff",
    "Koff",
    "Vertex",
    "Edge",
    "Cell",
    "EdgeOffset",
    "size",
]
