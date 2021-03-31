# -*- coding: utf-8 -*-
#
# Simple vertex to edge reduction.
#
# ```python
# for e in edges(mesh):
#     out = sum(in[v] for v in vertices(e))
# ```

import types

from gtc_unstructured.frontend.gtscript import (
    FORWARD,
    Connectivity,
    Edge,
    Field,
    Vertex,
    computation,
    location,
)
from gtc_unstructured.irs.common import DataType


E2V = types.new_class("E2V", (Connectivity[Edge, Vertex, 2, False],))
dtype = DataType.FLOAT64


def sten(e2v: E2V, field_in: Field[Vertex, dtype], field_out: Field[Edge, dtype]):
    with computation(FORWARD), location(Edge) as e:
        field_out[e] = sum(field_in[v] for v in e2v[e])


if __name__ == "__main__":
    import generator

    generator.default_main(sten)
