from dataclasses import field
import numpy as np
from numpy.core.numeric import allclose
from iterator.runtime import *
from iterator.builtins import *
from iterator.embedded import (
    NeighborTableOffsetProvider,
    np_as_located_field,
    index_field,
)


Vertex = CartesianAxis("Vertex")
Edge = CartesianAxis("Edge")


# 3x3 periodic   edges
# 0 - 1 - 2 -    0 1 2
# |   |   |      9 10 11
# 3 - 4 - 5 -    3 4 5
# |   |   |      12 13 14
# 6 - 7 - 8 -    6 7 8
# |   |   |      15 16 17


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


@fundef
def sum_edges_to_vertices(in_edges):
    return (
        deref(shift(V2E, 0)(in_edges))
        + deref(shift(V2E, 1)(in_edges))
        + deref(shift(V2E, 2)(in_edges))
        + deref(shift(V2E, 3)(in_edges))
    )


@fendef(offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)})
def e2v_sum_fencil(in_edges, out_vertices):
    closure(
        domain(named_range(Vertex, 0, 9)),
        sum_edges_to_vertices,
        [out_vertices],
        [in_edges],
    )


def test_sum_edges_to_vertices():
    inp = index_field(Edge)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(list(sum(row) for row in v2e_arr))

    e2v_sum_fencil(inp, out, backend="double_roundtrip")
    assert allclose(out, ref)
    e2v_sum_fencil(None, None, backend="cpptoy")


@fundef
def sum_edges_to_vertices_reduce(in_edges):
    return reduce(lambda a, b: a + b, 0)(shift(V2E)(in_edges))


@fendef(offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)})
def e2v_sum_fencil_reduce(in_edges, out_vertices):
    closure(
        domain(named_range(Vertex, 0, 9)),
        sum_edges_to_vertices_reduce,
        [out_vertices],
        [in_edges],
    )


def test_sum_edges_to_vertices_reduce():
    inp = index_field(Edge)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(list(sum(row) for row in v2e_arr))

    e2v_sum_fencil_reduce(None, None, backend="cpptoy")
    e2v_sum_fencil_reduce(inp, out, backend="double_roundtrip")
    assert allclose(out, ref)


@fundef
def sparse_stencil(inp):
    return reduce(lambda a, b: a + b, 0)(inp)


@fendef
def sparse_fencil(inp, out):
    closure(
        domain(named_range(Vertex, 0, 9)),
        sparse_stencil,
        [out],
        [inp],
    )


def test_sparse_input_field():
    inp = np_as_located_field(Vertex, V2E)(np.asarray([[1, 2, 3, 4]] * 9))
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = np.ones([9]) * 10

    sparse_fencil(
        inp,
        out,
        backend="double_roundtrip",
        offset_provider={"V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4)},
    )

    assert allclose(out, ref)


V2V = offset("V2V")


def test_sparse_input_field_v2v():
    inp = np_as_located_field(Vertex, V2V)(v2v_arr)
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = np.asarray(list(sum(row) for row in v2v_arr))

    sparse_fencil(
        inp,
        out,
        backend="double_roundtrip",
        offset_provider={"V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
    )

    assert allclose(out, ref)


@fundef
def deref_stencil(inp):
    return deref(shift(V2V, 0)(inp))


@fundef
def lift_stencil(inp):
    return deref(shift(V2V, 2)(lift(deref_stencil)(inp)))


@fendef
def lift_fencil(inp, out):
    closure(domain(named_range(Vertex, 0, 9)), lift_stencil, [out], [inp])


def test_lift():
    inp = index_field(Vertex)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(np.asarray(range(9)))

    lift_fencil(None, None, backend="cpptoy")
    lift_fencil(
        inp,
        out,
        backend="double_roundtrip",
        offset_provider={"V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
    )
    assert allclose(out, ref)


@fundef
def sparse_shifted_stencil(inp):
    return deref(shift(0, 2)(shift(V2V)(inp)))


@fendef
def sparse_shifted_fencil(inp, out):
    closure(
        domain(named_range(Vertex, 0, 9)),
        sparse_shifted_stencil,
        [out],
        [inp],
    )


def test_shift_sparse_input_field():
    inp = np_as_located_field(Vertex, V2V)(v2v_arr)
    out = np_as_located_field(Vertex)(np.zeros([9]))
    ref = np.asarray(np.asarray(range(9)))

    sparse_shifted_fencil(
        inp,
        out,
        backend="double_roundtrip",
        offset_provider={"V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
    )

    assert allclose(out, ref)


@fundef
def shift_shift_stencil2(inp):
    return deref(shift(V2E, 3)(shift(E2V, 1)(inp)))


@fundef
def shift_sparse_stencil2(inp):
    return deref(shift(1, 3)(shift(V2E)(inp)))


@fendef
def sparse_shifted_fencil2(inp_sparse, inp, out1, out2):
    closure(
        domain(named_range(Vertex, 0, 9)),
        shift_shift_stencil2,
        [out1],
        [inp],
    )
    closure(
        domain(named_range(Vertex, 0, 9)),
        shift_sparse_stencil2,
        [out2],
        [inp_sparse],
    )


def test_shift_sparse_input_field2():
    inp = index_field(Vertex)
    inp_sparse = np_as_located_field(Edge, E2V)(e2v_arr)
    out1 = np_as_located_field(Vertex)(np.zeros([9]))
    out2 = np_as_located_field(Vertex)(np.zeros([9]))

    sparse_shifted_fencil2(
        inp_sparse,
        inp,
        out1,
        out2,
        backend="double_roundtrip",
        offset_provider={
            "E2V": NeighborTableOffsetProvider(e2v_arr, Edge, Vertex, 2),
            "V2E": NeighborTableOffsetProvider(v2e_arr, Vertex, Edge, 4),
        },
    )

    assert allclose(out1, out2)


@fundef
def sparse_shifted_stencil_reduce(inp):
    def sum_(a, b):
        return a + b

    # return deref(shift(V2V, 0)(lift(deref)(shift(0)(inp))))
    return reduce(sum_, 0)(shift(V2V)(lift(reduce(sum_, 0))(inp)))


@fendef
def sparse_shifted_fencil_reduce(inp, out):
    closure(
        domain(named_range(Vertex, 0, 9)),
        sparse_shifted_stencil_reduce,
        [out],
        [inp],
    )


def test_shift_sparse_input_field():
    inp = np_as_located_field(Vertex, V2V)(v2v_arr)
    out = np_as_located_field(Vertex)(np.zeros([9]))

    ref = []
    for row in v2v_arr:
        elem_sum = 0
        for neigh in row:
            elem_sum += sum(v2v_arr[neigh])
        ref.append(elem_sum)

    ref = np.asarray(ref)

    sparse_shifted_fencil_reduce(
        inp,
        out,
        backend="double_roundtrip",
        offset_provider={"V2V": NeighborTableOffsetProvider(v2v_arr, Vertex, Vertex, 4)},
    )

    assert allclose(np.asarray(out), ref)
