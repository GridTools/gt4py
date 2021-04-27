# -*- coding: utf-8 -*-
#
# FVM nabla stencil
#

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
V2E = types.new_class("V2E", (Connectivity[Vertex, Edge, 7, True],))
dtype = DataType.FLOAT64


def sten(
    v2e: V2E,
    e2v: E2V,
    S_MXX: Field[Edge, dtype],
    S_MYY: Field[Edge, dtype],
    pp: Field[Vertex, dtype],
    pnabla_MXX: Field[Vertex, dtype],
    pnabla_MYY: Field[Vertex, dtype],
    vol: Field[Vertex, dtype],
    sign: SparseField[V2E, dtype],
):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            zavg = 0.5 * sum(pp[v] for v in e2v[e])
            zavgS_MXX = S_MXX * zavg
            zavgS_MYY = S_MYY * zavg
        with location(Vertex) as v:
            pnabla_MXX = sum(zavgS_MXX[e] * sign[v, e] for e in v2e[v])
            pnabla_MYY = sum(zavgS_MYY[e] * sign[v, e] for e in v2e[v])
            pnabla_MXX = pnabla_MXX / vol
            pnabla_MYY = pnabla_MYY / vol


if __name__ == "__main__":
    import generator

    generator.default_main(sten)
