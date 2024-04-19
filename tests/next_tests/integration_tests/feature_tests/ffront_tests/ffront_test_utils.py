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

import types
from collections import namedtuple
from typing import Any, Protocol, TypeVar

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import backend as next_backend, common
from gt4py.next.ffront import decorator
from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors import processor_interface as ppi
from gt4py.next.program_processors.runners import gtfn, roundtrip


try:
    from gt4py.next.program_processors.runners import dace_iterator
except ModuleNotFoundError as e:
    if "dace" in str(e):
        dace_iterator = None
    else:
        raise e

import next_tests


@ppi.program_executor
def no_exec(program: itir.FencilDefinition, *args: Any, **kwargs: Any) -> None:
    """Temporary default backend to not accidentally test the wrong backend."""
    raise ValueError("No backend selected! Backend selection is mandatory in tests.")


class NoBackend(next_backend.Backend):
    def __call__(self, program, *args, **kwargs) -> None:
        raise ValueError("No backend selected! Backend selection is mandatory in tests.")


no_backend = NoBackend(executor=no_exec, transformer=None, allocator=None)


OPTIONAL_PROCESSORS = []
if dace_iterator:
    OPTIONAL_PROCESSORS.append(next_tests.definitions.OptionalProgramBackendId.DACE_CPU)
    (
        OPTIONAL_PROCESSORS.append(
            pytest.param(
                next_tests.definitions.OptionalProgramBackendId.DACE_GPU,
                marks=pytest.mark.requires_gpu,
            )
        ),
    )


@pytest.fixture(
    params=[
        next_tests.definitions.ProgramBackendId.ROUNDTRIP,
        next_tests.definitions.ProgramBackendId.GTFN_CPU,
        next_tests.definitions.ProgramBackendId.GTFN_CPU_IMPERATIVE,
        next_tests.definitions.ProgramBackendId.GTFN_CPU_WITH_TEMPORARIES,
        pytest.param(
            next_tests.definitions.ProgramBackendId.GTFN_GPU, marks=pytest.mark.requires_gpu
        ),
        # will use the default (embedded) execution, but input/output allocated with the provided allocator
        next_tests.definitions.EmbeddedIds.NUMPY_EXECUTION,
        pytest.param(
            next_tests.definitions.EmbeddedIds.CUPY_EXECUTION, marks=pytest.mark.requires_gpu
        ),
    ]
    + OPTIONAL_PROCESSORS,
    ids=lambda p: p.short_id(),
)
def exec_alloc_descriptor(request):
    """
    Fixture creating field-view operator backend on-demand for tests.

    Notes:
        Check ADR 15 for details on the test-exclusion matrices.
    """
    backend_id = request.param
    backend = backend_id.load()

    for marker, skip_mark, msg in next_tests.definitions.BACKEND_SKIP_TEST_MATRIX.get(
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

V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
C2VDim = gtx.Dimension("C2V", kind=gtx.DimensionKind.LOCAL)
V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))
C2V = gtx.FieldOffset("C2V", source=Vertex, target=(Cell, C2VDim))

size = 10


class MeshDescriptor(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def num_vertices(self) -> int: ...

    @property
    def num_cells(self) -> int: ...

    @property
    def num_edges(self) -> int: ...

    @property
    def num_levels(self) -> int: ...

    @property
    def offset_provider(self) -> dict[str, common.Connectivity]: ...


def simple_mesh() -> MeshDescriptor:
    num_vertices = 9
    num_cells = 8

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

    return types.SimpleNamespace(
        name="simple_mesh",
        num_vertices=num_vertices,
        num_edges=np.int32(num_edges),
        num_cells=num_cells,
        offset_provider={
            V2E.value: gtx.NeighborTableOffsetProvider(
                v2e_arr, Vertex, Edge, 4, has_skip_values=False
            ),
            E2V.value: gtx.NeighborTableOffsetProvider(
                e2v_arr, Edge, Vertex, 2, has_skip_values=False
            ),
            C2V.value: gtx.NeighborTableOffsetProvider(
                c2v_arr, Cell, Vertex, 4, has_skip_values=False
            ),
            C2E.value: gtx.NeighborTableOffsetProvider(
                c2e_arr, Cell, Edge, 4, has_skip_values=False
            ),
        },
    )


def skip_value_mesh() -> MeshDescriptor:
    """Mesh with skip values from the GT4Py quickstart guide."""

    num_vertices = 7
    num_cells = 6
    num_edges = 12

    v2e_arr = np.array(
        [
            [1, 8, 7, 0, common._DEFAULT_SKIP_VALUE],
            [2, 8, 1, common._DEFAULT_SKIP_VALUE, common._DEFAULT_SKIP_VALUE],
            [3, 9, 8, 2, common._DEFAULT_SKIP_VALUE],
            [4, 10, 3, common._DEFAULT_SKIP_VALUE, common._DEFAULT_SKIP_VALUE],
            [5, 11, 4, common._DEFAULT_SKIP_VALUE, common._DEFAULT_SKIP_VALUE],
            [0, 6, 4, common._DEFAULT_SKIP_VALUE, common._DEFAULT_SKIP_VALUE],
            [6, 7, 9, 10, 11],
        ],
        dtype=gtx.IndexType,
    )

    e2v_arr = np.array(
        [
            [0, 5],
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 0],
            [0, 2],
            [2, 6],
            [3, 6],
            [4, 6],
        ],
        dtype=gtx.IndexType,
    )

    c2v_arr = np.array(
        [[0, 6, 5], [0, 2, 6], [0, 1, 2], [2, 3, 6], [3, 4, 6], [4, 5, 6]], dtype=gtx.IndexType
    )

    c2e_arr = np.array(
        [
            [0, 6, 7],  # cell 0 (neighbors: edge 0, edge 6, edge 7)
            [7, 8, 9],  # cell 1
            [1, 2, 8],  # cell 2
            [3, 9, 10],  # cell 3
            [4, 10, 11],  # cell 4
            [5, 6, 11],  # cell 5
        ],
        dtype=gtx.IndexType,
    )

    return types.SimpleNamespace(
        name="skip_value_mesh",
        num_vertices=num_vertices,
        num_edges=num_edges,
        num_cells=num_cells,
        offset_provider={
            V2E.value: gtx.NeighborTableOffsetProvider(
                v2e_arr, Vertex, Edge, 5, has_skip_values=True
            ),
            E2V.value: gtx.NeighborTableOffsetProvider(
                e2v_arr, Edge, Vertex, 2, has_skip_values=False
            ),
            C2V.value: gtx.NeighborTableOffsetProvider(
                c2v_arr, Cell, Vertex, 3, has_skip_values=False
            ),
            C2E.value: gtx.NeighborTableOffsetProvider(
                c2e_arr, Cell, Edge, 3, has_skip_values=False
            ),
        },
    )


__all__ = [
    "exec_alloc_descriptor",
    "mesh_descriptor",
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


@pytest.fixture(
    params=[
        simple_mesh(),
        pytest.param(skip_value_mesh(), marks=pytest.mark.uses_mesh_with_skip_values),
    ],
    ids=lambda p: p.name,
)
def mesh_descriptor(request) -> MeshDescriptor:
    yield request.param
