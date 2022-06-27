import numpy as np

from functional.common import Dimension
from functional.iterator.builtins import *
from functional.iterator.embedded import (
    NeighborTableOffsetProvider,
    index_field,
    np_as_located_field,
)
from functional.iterator.runtime import fundef, offset


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


def test_sum_edges_to_vertices(backend):
    backend, validate = backend
    inp = index_field(Edge)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(list(sum(row) for row in v2e_arr))

    sum_edges_to_vertices[{Vertex: range(0, 9)}](
        inp,
        out=out,
        backend=backend,
        offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)},
    )
    if validate:
        assert np.allclose(out, ref)


@fundef
def sum_edges_to_vertices_reduce(in_edges):
    return reduce(lambda a, b: a + b, 0)(shift(V2E)(in_edges))


def test_sum_edges_to_vertices_reduce(backend):
    backend, validate = backend
    inp = index_field(Edge)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(list(sum(row) for row in v2e_arr))

    sum_edges_to_vertices_reduce[{Vertex: range(0, 9)}](
        inp,
        out=out,
        backend=backend,
        offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)},
    )
    if validate:
        assert np.allclose(out, ref)


@fundef
def first_vertex_neigh_of_first_edge_neigh_of_cells(in_vertices):
    return deref(shift(E2V, 0)(shift(C2E, 0)(in_vertices)))


def test_first_vertex_neigh_of_first_edge_neigh_of_cells_fencil(backend):
    backend, validate = backend
    inp = index_field(Vertex)
    out = np_as_located_field(Cell)(np.zeros([9]))
    ref = np.asarray(list(v2e_arr[c[0]][0] for c in c2e_arr))

    first_vertex_neigh_of_first_edge_neigh_of_cells[{Cell: range(0, 9)}](
        inp,
        out=out,
        backend=backend,
        offset_provider={
            "E2V": NeighborTableOffsetProvider(e2v_arr, Edge, Vertex, 2),
            "C2E": NeighborTableOffsetProvider(c2e_arr, Cell, Edge, 4),
        },
    )
    if validate:
        assert np.allclose(out, ref)


@fundef
def sparse_stencil(non_sparse, inp):
    return reduce(lambda a, b, c: a + c, 0)(shift(V2E)(non_sparse), inp)


def test_sparse_input_field(backend):
    backend, validate = backend
    non_sparse = np_as_located_field(Edge)(np.zeros(18))
    inp = np_as_located_field(Vertex, V2E)(np.asarray([[1, 2, 3, 4]] * 9))
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = np.ones([9]) * 10

    sparse_stencil[{Vertex: range(0, 9)}](
        non_sparse,
        inp,
        out=out,
        backend=backend,
        offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)},
    )

    if validate:
        assert np.allclose(out, ref)


V2V = offset("V2V")


def test_sparse_input_field_v2v(backend):
    backend, validate = backend
    non_sparse = np_as_located_field(Edge)(np.zeros(9))
    inp = np_as_located_field(Vertex, V2V)(v2v_arr)
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = np.asarray(list(sum(row) for row in v2v_arr))

    sparse_stencil[{Vertex: range(0, 9)}](
        non_sparse,
        inp,
        out=out,
        backend=backend,
        offset_provider={
            "V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4),
            "V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4),
        },
    )

    if validate:
        assert np.allclose(out, ref)


@fundef
def deref_stencil(inp):
    return deref(shift(V2V, 0)(inp))


@fundef
def lift_stencil(inp):
    return deref(shift(V2V, 2)(lift(deref_stencil)(inp)))


def test_lift(backend):
    backend, validate = backend
    inp = index_field(Vertex)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(np.asarray(range(9)))

    lift_stencil[{Vertex: range(0, 9)}](
        inp,
        out=out,
        backend=backend,
        offset_provider={"V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
    )
    if validate:
        assert np.allclose(out, ref)


@fundef
def sparse_shifted_stencil(inp):
    return deref(shift(0, 2)(shift(V2V)(inp)))


def test_shift_sparse_input_field(backend):
    backend, validate = backend
    inp = np_as_located_field(Vertex, V2V)(v2v_arr)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(np.asarray(range(9)))

    sparse_shifted_stencil[{Vertex: range(0, 9)}](
        inp,
        out=out,
        backend=backend,
        offset_provider={"V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
    )

    if validate:
        assert np.allclose(out, ref)


@fundef
def shift_shift_stencil2(inp):
    return deref(shift(E2V, 1)(shift(V2E, 3)(inp)))


@fundef
def shift_sparse_stencil2(inp):
    return deref(shift(3, 1)(shift(V2E)(inp)))


def test_shift_sparse_input_field2(backend):
    backend, validate = backend
    inp = index_field(Vertex)
    inp_sparse = np_as_located_field(Edge, E2V)(e2v_arr)
    out1 = np_as_located_field(Vertex)(np.zeros([9]))
    out2 = np_as_located_field(Vertex)(np.zeros([9]))

    offset_provider = {
        "E2V": NeighborTableOffsetProvider(e2v_arr, Edge, Vertex, 2),
        "V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4),
    }

    domain = {Vertex: range(0, 9)}
    shift_shift_stencil2[domain](inp, out=out1, offset_provider=offset_provider, backend=backend)
    shift_sparse_stencil2[domain](
        inp_sparse, out=out2, offset_provider=offset_provider, backend=backend
    )

    if validate:
        assert np.allclose(out1, out2)


@fundef
def sparse_shifted_stencil_reduce(inp):
    def sum_(a, b):
        return a + b

    # return deref(shift(V2V, 0)(lift(deref)(shift(0)(inp))))
    return reduce(sum_, 0)(shift(V2V)(lift(reduce(sum_, 0))(inp)))


def test_sparse_shifted_stencil_reduce(backend):
    backend, validate = backend
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
    sparse_shifted_stencil_reduce[domain](
        inp,
        out=out,
        backend=backend,
        offset_provider={"V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
    )

    if validate:
        assert np.allclose(np.asarray(out), ref)
