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
import pytest

import gt4py.next as gtx
from gt4py.next.iterator import transforms
from gt4py.next.iterator.builtins import (
    deref,
    lift,
    list_get,
    make_const_list,
    map_,
    multiplies,
    neighbors,
    plus,
    reduce,
    shift,
)
from gt4py.next.iterator.runtime import fundef
from gt4py.next.program_processors.runners import gtfn

from next_tests.toy_connectivity import (
    C2E,
    E2V,
    V2E,
    V2V,
    Cell,
    E2VDim,
    Edge,
    V2EDim,
    V2VDim,
    Vertex,
    c2e_arr,
    e2v_arr,
    v2e_arr,
    v2v_arr,
)
from next_tests.unit_tests.conftest import lift_mode, program_processor, run_processor


def edge_index_field():  # TODO replace by gtx.index_field once supported in bindings
    return gtx.as_field([Edge], np.arange(e2v_arr.shape[0], dtype=np.int32))


def vertex_index_field():  # TODO replace by gtx.index_field once supported in bindings
    return gtx.as_field([Vertex], np.arange(v2e_arr.shape[0], dtype=np.int32))


@fundef
def sum_edges_to_vertices(in_edges):
    return (
        deref(shift(V2E, 0)(in_edges))
        + deref(shift(V2E, 1)(in_edges))
        + deref(shift(V2E, 2)(in_edges))
        + deref(shift(V2E, 3)(in_edges))
    )


@fundef
def sum_edges_to_vertices_list_get_neighbors(in_edges):
    neighs = neighbors(V2E, in_edges)
    return list_get(0, neighs) + list_get(1, neighs) + list_get(2, neighs) + list_get(3, neighs)


@fundef
def sum_edges_to_vertices_reduce(in_edges):
    return reduce(plus, 0)(neighbors(V2E, in_edges))


@pytest.mark.parametrize(
    "stencil",
    [sum_edges_to_vertices, sum_edges_to_vertices_list_get_neighbors, sum_edges_to_vertices_reduce],
)
def test_sum_edges_to_vertices(program_processor, lift_mode, stencil):
    program_processor, validate = program_processor
    inp = edge_index_field()
    out = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))
    ref = np.asarray(list(sum(row) for row in v2e_arr))

    run_processor(
        stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2E": gtx.NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)},
        lift_mode=lift_mode,
    )
    if validate:
        assert np.allclose(out.asnumpy(), ref)


@fundef
def map_neighbors(in_edges):
    return reduce(plus, 0)(map_(plus)(neighbors(V2E, in_edges), neighbors(V2E, in_edges)))


def test_map_neighbors(program_processor, lift_mode):
    program_processor, validate = program_processor
    inp = edge_index_field()
    out = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))
    ref = 2 * np.sum(v2e_arr, axis=1)

    run_processor(
        map_neighbors[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2E": gtx.NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)},
        lift_mode=lift_mode,
    )
    if validate:
        assert np.allclose(out.asnumpy(), ref)


@fundef
def map_make_const_list(in_edges):
    return reduce(plus, 0)(map_(multiplies)(neighbors(V2E, in_edges), make_const_list(2)))


@pytest.mark.uses_constant_fields
def test_map_make_const_list(program_processor, lift_mode):
    program_processor, validate = program_processor
    inp = edge_index_field()
    out = gtx.as_field([Vertex], np.zeros([9], inp.dtype))
    ref = 2 * np.sum(v2e_arr, axis=1)

    run_processor(
        map_make_const_list[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2E": gtx.NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)},
        lift_mode=lift_mode,
    )
    if validate:
        assert np.allclose(out.asnumpy(), ref)


@fundef
def first_vertex_neigh_of_first_edge_neigh_of_cells(in_vertices):
    return deref(shift(E2V, 0)(shift(C2E, 0)(in_vertices)))


def test_first_vertex_neigh_of_first_edge_neigh_of_cells_fencil(program_processor, lift_mode):
    program_processor, validate = program_processor
    inp = vertex_index_field()
    out = gtx.as_field([Cell], np.zeros([9], dtype=inp.dtype))
    ref = np.asarray(list(v2e_arr[c[0]][0] for c in c2e_arr))

    run_processor(
        first_vertex_neigh_of_first_edge_neigh_of_cells[{Cell: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={
            "E2V": gtx.NeighborTableOffsetProvider(e2v_arr, Edge, Vertex, 2),
            "C2E": gtx.NeighborTableOffsetProvider(c2e_arr, Cell, Edge, 4),
        },
        lift_mode=lift_mode,
    )
    if validate:
        assert np.allclose(out.asnumpy(), ref)


@fundef
def sparse_stencil(non_sparse, inp):
    return reduce(lambda a, b, c: a + c, 0)(neighbors(V2E, non_sparse), deref(inp))


def test_sparse_input_field(program_processor, lift_mode):
    program_processor, validate = program_processor

    non_sparse = gtx.as_field([Edge], np.zeros(18, dtype=np.int32))
    inp = gtx.as_field([Vertex, V2EDim], np.asarray([[1, 2, 3, 4]] * 9, dtype=np.int32))
    out = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))

    ref = np.ones([9]) * 10

    run_processor(
        sparse_stencil[{Vertex: range(0, 9)}],
        program_processor,
        non_sparse,
        inp,
        out=out,
        offset_provider={"V2E": gtx.NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out.asnumpy(), ref)


def test_sparse_input_field_v2v(program_processor, lift_mode):
    program_processor, validate = program_processor

    non_sparse = gtx.as_field([Edge], np.zeros(18, dtype=np.int32))
    inp = gtx.as_field([Vertex, V2VDim], v2v_arr)
    out = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))

    ref = np.asarray(list(sum(row) for row in v2v_arr))

    run_processor(
        sparse_stencil[{Vertex: range(0, 9)}],
        program_processor,
        non_sparse,
        inp,
        out=out,
        offset_provider={
            "V2V": gtx.NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4),
            "V2E": gtx.NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4),
        },
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out.asnumpy(), ref)


@fundef
def slice_sparse_stencil(sparse):
    return list_get(1, deref(sparse))


@pytest.mark.uses_sparse_fields
def test_slice_sparse(program_processor, lift_mode):
    program_processor, validate = program_processor
    inp = gtx.as_field([Vertex, V2VDim], v2v_arr)
    out = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))

    ref = v2v_arr[:, 1]

    run_processor(
        slice_sparse_stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2V": gtx.NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out.asnumpy(), ref)


@fundef
def slice_twice_sparse_stencil(sparse):
    return deref(shift(2)(shift(1)(sparse)))


@pytest.mark.xfail(reason="Field with more than one sparse dimension is not implemented.")
def test_slice_twice_sparse(program_processor, lift_mode):
    program_processor, validate = program_processor
    inp = gtx.as_field([Vertex, V2VDim, V2VDim], v2v_arr[v2v_arr])
    out = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))

    ref = v2v_arr[v2v_arr][:, 2, 1]
    run_processor(
        slice_twice_sparse_stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2V": gtx.NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(np.asarray(out), ref)


@fundef
def shift_sliced_sparse_stencil(sparse):
    return list_get(1, deref(shift(V2V, 0)(sparse)))


@pytest.mark.uses_sparse_fields
def test_shift_sliced_sparse(program_processor, lift_mode):
    program_processor, validate = program_processor
    inp = gtx.as_field([Vertex, V2VDim], v2v_arr)
    out = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))

    ref = v2v_arr[:, 1][v2v_arr][:, 0]

    run_processor(
        shift_sliced_sparse_stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2V": gtx.NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out.asnumpy(), ref)


@fundef
def slice_shifted_sparse_stencil(sparse):
    return list_get(1, deref(shift(V2V, 0)(sparse)))


@pytest.mark.uses_sparse_fields
def test_slice_shifted_sparse(program_processor, lift_mode):
    program_processor, validate = program_processor
    inp = gtx.as_field([Vertex, V2VDim], v2v_arr)
    out = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))

    ref = v2v_arr[:, 1][v2v_arr][:, 0]

    run_processor(
        slice_shifted_sparse_stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2V": gtx.NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out.asnumpy(), ref)


@fundef
def deref_stencil(inp):
    return deref(shift(V2V, 0)(inp))


@fundef
def lift_stencil(inp):
    return deref(shift(V2V, 2)(lift(deref_stencil)(inp)))


def test_lift(program_processor, lift_mode):
    program_processor, validate = program_processor
    inp = vertex_index_field()
    out = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))
    ref = np.asarray(np.asarray(range(9)))

    run_processor(
        lift_stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2V": gtx.NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
        lift_mode=lift_mode,
    )
    if validate:
        assert np.allclose(out.asnumpy(), ref)


@fundef
def sparse_shifted_stencil(inp):
    return list_get(2, list_get(0, neighbors(V2V, inp)))


@pytest.mark.uses_sparse_fields
def test_shift_sparse_input_field(program_processor, lift_mode):
    program_processor, validate = program_processor
    inp = gtx.as_field([Vertex, V2VDim], v2v_arr)
    out = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))
    ref = np.asarray(np.asarray(range(9)))

    run_processor(
        sparse_shifted_stencil[{Vertex: range(0, 9)}],
        program_processor,
        inp,
        out=out,
        offset_provider={"V2V": gtx.NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out.asnumpy(), ref)


@fundef
def shift_shift_stencil2(inp):
    return deref(shift(E2V, 1)(shift(V2E, 3)(inp)))


@fundef
def shift_sparse_stencil2(inp):
    return list_get(1, list_get(3, neighbors(V2E, inp)))


@pytest.mark.uses_sparse_fields
def test_shift_sparse_input_field2(program_processor, lift_mode):
    program_processor, validate = program_processor
    if program_processor in [
        gtfn.run_gtfn,
        gtfn.run_gtfn_imperative,
        gtfn.run_gtfn_with_temporaries,
    ]:
        pytest.xfail(
            "Bug in bindings/compilation/caching: only the first program seems to be compiled."
        )  # observed in `config.BuildCacheLifetime.PERSISTENT` mode
    inp = vertex_index_field()
    inp_sparse = gtx.as_field([Edge, E2VDim], e2v_arr)
    out1 = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))
    out2 = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))

    offset_provider = {
        "E2V": gtx.NeighborTableOffsetProvider(e2v_arr, Edge, Vertex, 2),
        "V2E": gtx.NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4),
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
        assert np.allclose(out1.asnumpy(), out2.asnumpy())


@fundef
def sparse_shifted_stencil_reduce(inp):
    def sum_(a, b):
        return a + b

    return reduce(sum_, 0)(neighbors(V2V, lift(lambda x: reduce(sum_, 0)(deref(x)))(inp)))


@pytest.mark.uses_sparse_fields
@pytest.mark.uses_reduction_with_only_sparse_fields
def test_sparse_shifted_stencil_reduce(program_processor, lift_mode):
    program_processor, validate = program_processor
    if lift_mode != transforms.LiftMode.FORCE_INLINE:
        pytest.xfail("shifted input arguments not supported for lift_mode != LiftMode.FORCE_INLINE")

    inp = gtx.as_field([Vertex, V2VDim], v2v_arr)
    out = gtx.as_field([Vertex], np.zeros([9], dtype=inp.dtype))

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
        offset_provider={"V2V": gtx.NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(out.asnumpy(), ref)
