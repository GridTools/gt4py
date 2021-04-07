# -*- coding: utf-8 -*-
#
# Simple vertex to edge reduction.
#
# ```python
# for e in edges(mesh):
#     out = sum(in[v] for v in vertices(e))
# ```

from gtc_unstructured.frontend.gtscript import FORWARD, Edge, Field, computation, location
from gtc_unstructured.irs.common import DataType


dtype = DataType.FLOAT64


def sten(field_in: Field[Edge, dtype], field_out: Field[Edge, dtype]):
    with computation(FORWARD), location(Edge) as e:
        field_out[e] = field_in[e]


if __name__ == "__main__":
    import generator

    generator.default_main(sten)
