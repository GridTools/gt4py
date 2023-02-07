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

import numpy as np
import pytest

from gt4py.next.common import Dimension
from gt4py.next.iterator import transforms
from gt4py.next.iterator.builtins import deref, lift, plus, reduce, shift
from gt4py.next.iterator.embedded import (
    NeighborTableOffsetProvider,
    index_field,
    np_as_located_field,
)
from gt4py.next.iterator.runtime import fundef, offset
from gt4py.next.program_processors.formatters import gtfn
from gt4py.next.program_processors.runners import gtfn_cpu

from .conftest import run_processor


Vertex = Dimension("Vertex")
Edge = Dimension("Edge")
Cell = Dimension("Cell")


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
    ]
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
    ]
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
    ]
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
    ]
)

V2E = offset("V2E")
E2V = offset("E2V")
C2E = offset("C2E")


@fundef
def sum_edges_to_vertices(in_edges):
    return (
        deref(shift(V2E, 0)(in_edges))
        + deref(shift(V2E, 1)(in_edges))
        + deref(shift(V2E, 2)(in_edges))
        + deref(shift(V2E, 3)(in_edges))
    )


def test_sum_edges_to_vertices(program_processor_no_gtfn_exec, lift_mode):
    program_processor, validate = program_processor_no_gtfn_exec
    inp = index_field(Edge)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(list(sum(row) for row in v2e_arr))

    run_processor(
        sum_edges_to_vertices[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)},
        lift_mode=lift_mode,
    )
    if validate:
        assert np.allclose(out, ref)


@fundef
def sum_edges_to_vertices_reduce(in_edges):
    return reduce(plus, 0)(shift(V2E)(in_edges))


def test_sum_edges_to_vertices_reduce(program_processor_no_gtfn_exec, lift_mode):
    program_processor, validate = program_processor_no_gtfn_exec
    inp = index_field(Edge)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(list(sum(row) for row in v2e_arr))

    run_processor(
        sum_edges_to_vertices_reduce[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)},
        lift_mode=lift_mode,
    )
    if validate:
        assert np.allclose(out, ref)


@fundef
def first_vertex_neigh_of_first_edge_neigh_of_cells(in_vertices):
    return deref(shift(E2V, 0)(shift(C2E, 0)(in_vertices)))


def test_first_vertex_neigh_of_first_edge_neigh_of_cells_fencil(
    program_processor_no_gtfn_exec, lift_mode
):
    program_processor, validate = program_processor_no_gtfn_exec
    inp = index_field(Vertex)
    out = np_as_located_field(Cell)(np.zeros([9]))
    ref = np.asarray(list(v2e_arr[c[0]][0] for c in c2e_arr))

    run_processor(
        first_vertex_neigh_of_first_edge_neigh_of_cells[{Cell: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={
            "E2V": NeighborTableOffsetProvider(e2v_arr, Edge, Vertex, 2),
            "C2E": NeighborTableOffsetProvider(c2e_arr, Cell, Edge, 4),
        },
        lift_mode=lift_mode,
    )
    if validate:
        assert np.allclose(out, ref)


@fundef
def sparse_stencil(non_sparse, inp):
    return reduce(lambda a, b, c: a + c, 0)(shift(V2E)(non_sparse), inp)


def test_sparse_input_field(program_processor_no_gtfn_exec, lift_mode):
    program_processor, validate = program_processor_no_gtfn_exec
    non_sparse = np_as_located_field(Edge)(np.zeros(18))
    inp = np_as_located_field(Vertex, V2E)(np.asarray([[1, 2, 3, 4]] * 9))
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = np.ones([9]) * 10

    run_processor(
        sparse_stencil[{Vertex: range(0, 9)}],
        program_processor,
        non_sparse,
        inp,
        out=out,
        offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out, ref)


V2V = offset("V2V")


def test_sparse_input_field_v2v(program_processor_no_gtfn_exec, lift_mode):
    program_processor, validate = program_processor_no_gtfn_exec
    non_sparse = np_as_located_field(Edge)(np.zeros(18))
    inp = np_as_located_field(Vertex, V2V)(v2v_arr)
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = np.asarray(list(sum(row) for row in v2v_arr))

    run_processor(
        sparse_stencil[{Vertex: range(0, 9)}],
        program_processor,
        non_sparse,
        inp,
        out=out,
        offset_provider={
            "V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4),
            "V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4),
        },
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out, ref)


@fundef
def slice_sparse_stencil(sparse):
    return deref(shift(1)(sparse))


def test_slice_sparse(program_processor_no_gtfn_exec, lift_mode):
    program_processor, validate = program_processor_no_gtfn_exec
    inp = np_as_located_field(Vertex, V2V)(v2v_arr)
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = v2v_arr[:, 1]

    run_processor(
        slice_sparse_stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={
            "V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4),
        },
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out, ref)


@fundef
def slice_twice_sparse_stencil(sparse):
    return deref(shift(2)(shift(1)(sparse)))


def test_slice_twice_sparse(program_processor_no_gtfn_exec, lift_mode):
    program_processor, validate = program_processor_no_gtfn_exec
    inp = np_as_located_field(Vertex, V2V, V2V)(v2v_arr[v2v_arr])
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = v2v_arr[v2v_arr][:, 2, 1]
    run_processor(
        slice_twice_sparse_stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={
            "V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4),
        },
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(np.asarray(out), ref)


@fundef
def shift_sliced_sparse_stencil(sparse):
    return deref(shift(V2V, 0)(shift(1)(sparse)))


def test_shift_sliced_sparse(program_processor_no_gtfn_exec, lift_mode):
    program_processor, validate = program_processor_no_gtfn_exec
    inp = np_as_located_field(Vertex, V2V)(v2v_arr)
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = v2v_arr[:, 1][v2v_arr][:, 0]

    run_processor(
        shift_sliced_sparse_stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={
            "V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4),
        },
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out, ref)


@fundef
def slice_shifted_sparse_stencil(sparse):
    return deref(shift(1)(shift(V2V, 0)(sparse)))


def test_slice_shifted_sparse(program_processor_no_gtfn_exec, lift_mode):
    program_processor, validate = program_processor_no_gtfn_exec
    inp = np_as_located_field(Vertex, V2V)(v2v_arr)
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = v2v_arr[:, 1][v2v_arr][:, 0]

    run_processor(
        slice_shifted_sparse_stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={
            "V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4),
        },
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out, ref)


@fundef
def deref_stencil(inp):
    return deref(shift(V2V, 0)(inp))


@fundef
def lift_stencil(inp):
    return deref(shift(V2V, 2)(lift(deref_stencil)(inp)))


def test_lift(program_processor, lift_mode):
    program_processor, validate = program_processor
    inp = index_field(Vertex)
    if program_processor in [gtfn_cpu.run_gtfn, gtfn_cpu.run_gtfn_imperative]:
        # TODO(tehrengruber): only a temporary solution until index fields are supported in the
        #  gtfn backend.
        inp = np_as_located_field(Vertex)(np.array([inp.field_getitem(i) for i in range(0, 9)]))
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(np.asarray(range(9)))

    run_processor(
        lift_stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
        lift_mode=lift_mode,
    )
    if validate:
        assert np.allclose(out, ref)


@fundef
def sparse_shifted_stencil(inp):
    return deref(shift(0, 2)(shift(V2V)(inp)))


def test_shift_sparse_input_field(program_processor_no_gtfn_exec, lift_mode):
    program_processor, validate = program_processor_no_gtfn_exec
    inp = np_as_located_field(Vertex, V2V)(v2v_arr)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(np.asarray(range(9)))

    run_processor(
        sparse_shifted_stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out, ref)


@fundef
def shift_shift_stencil2(inp):
    return deref(shift(E2V, 1)(shift(V2E, 3)(inp)))


@fundef
def shift_sparse_stencil2(inp):
    return deref(shift(3, 1)(shift(V2E)(inp)))


def test_shift_sparse_input_field2(program_processor_no_gtfn_exec, lift_mode):
    program_processor, validate = program_processor_no_gtfn_exec
    inp = index_field(Vertex)
    inp_sparse = np_as_located_field(Edge, E2V)(e2v_arr)
    out1 = np_as_located_field(Vertex)(np.zeros([9]))
    out2 = np_as_located_field(Vertex)(np.zeros([9]))

    offset_provider = {
        "E2V": NeighborTableOffsetProvider(e2v_arr, Edge, Vertex, 2),
        "V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4),
    }

    domain = {Vertex: range(0, 9)}
    run_processor(
        shift_shift_stencil2[domain],
        program_processor,
        inp,
        out=out1,
        offset_provider=offset_provider,
        lift_mode=lift_mode,
    )
    run_processor(
        shift_sparse_stencil2[domain],
        program_processor,
        inp_sparse,
        out=out2,
        offset_provider=offset_provider,
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out1, out2)


@fundef
def sparse_shifted_stencil_reduce(inp):
    def sum_(a, b):
        return a + b

    # return deref(shift(V2V, 0)(lift(deref)(shift(0)(inp))))
    return reduce(sum_, 0)(shift(V2V)(lift(reduce(sum_, 0))(inp)))


def test_sparse_shifted_stencil_reduce(program_processor_no_gtfn_exec, lift_mode):
    program_processor, validate = program_processor_no_gtfn_exec
    if program_processor == gtfn.format_sourcecode:
        pytest.xfail("We cannot unroll a reduction on a sparse field only.")
        # With our current understanding, this iterator IR program is illegal, however we might want to fix it and therefore keep the test for now.

    if lift_mode != transforms.LiftMode.FORCE_INLINE:
        pytest.xfail("shifted input arguments not supported for lift_mode != LiftMode.FORCE_INLINE")

    inp = np_as_located_field(Vertex, V2V)(v2v_arr)
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = []
    for row in v2v_arr:
        elem_sum = 0
        for neigh in row:
            elem_sum += sum(v2v_arr[neigh])
        ref.append(elem_sum)

    ref = np.asarray(ref)

    domain = {Vertex: range(0, 9)}
    run_processor(
        sparse_shifted_stencil_reduce[domain],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(np.asarray(out), ref)
