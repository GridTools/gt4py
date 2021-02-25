# -*- coding: utf-8 -*-
#
# Cell to cell reduction.
# Note that the reduction refers to a LocationRef from outside!
#
# ```python
# for c1 in cells(mesh):
#     field1 = sum(f[c1] * f[c2] for c2 in cells(c1))
# ```

import types

from gtc_unstructured.frontend.gtscript import (
    FORWARD,
    Connectivity,
    Edge,
    Field,
    SparseField,
    Vertex,
    computation,
    location,
)
from gtc_unstructured.irs.common import DataType


E2V = types.new_class("E2V", (Connectivity[Edge, Vertex, 2, False],))

dtype = DataType.FLOAT64


def sten(e2v: E2V, in_field: Field[Vertex, dtype], out_field: SparseField[E2V, dtype]):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            # TODO: maybe support slicing for lhs: out_sparse_field[e,:]
            out_field = (in_field[v] for v in e2v[e])
            # TODO: Fix silently generates invalid code
            # out_sparse_field = in_sparse_field


if __name__ == "__main__":
    import generator

    generator.default_main(sten)
