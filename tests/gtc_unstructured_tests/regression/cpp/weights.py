# -*- coding: utf-8 -*-
#
# Weighted reduction.

import types

from gtc_unstructured.frontend.gtscript import (
    FORWARD,
    Connectivity,
    Edge,
    Field,
    LocalField,
    Vertex,
    computation,
    interval,
    location,
)
from gtc_unstructured.irs.common import DataType


E2V = types.new_class("E2V", (Connectivity[Edge, Vertex, 2, False],))

dtype = DataType.FLOAT64


def sten(e2v: E2V, in_field: Field[Vertex, dtype], out_field: Field[Edge, dtype]):
    with computation(FORWARD), interval(0, None):
        with location(Edge) as e:
            # TODO: ints don't work right now
            weights = LocalField[E2V, dtype]([-1.0, 1.0])
            out_field = sum(in_field[v] * weights[e, v] for v in e2v[e])
