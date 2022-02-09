from typing import Optional

import numpy as np

from functional.iterator.builtins import deref, reduce, shift
from functional.iterator.embedded import (
    NeighborTableOffsetProvider,
    index_field,
    np_as_located_field,
)
from functional.iterator.runtime import CartesianAxis, fundef, offset


# Simple connectivity:
# Vertex0 ---Edge0--- Vertex1 ---Edge1--- Vertex2

# TODO C++ supports only skip value at the end, therefore it's implemented here as well.
# If we decide to lift this restriction we have to change it in all places.
v2e_arr = [[0, None], [0, 1], [1, None]]


class ListAsNeighTable:
    def __init__(self, lst) -> None:
        self.lst = lst

    def __getitem__(self, indices) -> Optional[int]:
        return v2e_arr[indices[0]][indices[1]]


V2E = offset("V2E")
Edge = CartesianAxis("Edge")
Vertex = CartesianAxis("Vertex")


def test_normal_field(backend):
    backend, validate = backend

    @fundef
    def sum_edges(inp):
        return reduce(lambda a, b: a + b, 0.0)(shift(V2E)(inp))

    inp = np_as_located_field(Edge)(np.asarray([1, 2]))
    out = np_as_located_field(Vertex)(np.zeros([3]))
    ref = np.asarray([1, 3, 2])

    sum_edges[{Vertex: range(0, 3)}](
        inp,
        out=out,
        backend=backend,
        offset_provider={
            "V2E": NeighborTableOffsetProvider(ListAsNeighTable(v2e_arr), Vertex, Edge, 2)
        },
    )
    if validate:
        assert np.allclose(out, ref)


def test_sparse(backend):
    backend, validate = backend

    @fundef
    def sum_edges(inp):
        return reduce(lambda a, b: a + b, 0.0)(deref(inp))

    inp = np_as_located_field(Vertex, V2E)(np.asarray([[1, -9999], [2, 3], [4, -9999]]))
    out = np_as_located_field(Vertex)(np.zeros([3]))
    ref = np.asarray([1, 5, 4])

    sum_edges[{Vertex: range(0, 3)}](
        inp,
        out=out,
        backend=backend,
        offset_provider={
            "V2E": NeighborTableOffsetProvider(ListAsNeighTable(v2e_arr), Vertex, Edge, 2)
        },
    )
    if validate:
        assert np.allclose(np.asarray(out), ref)
