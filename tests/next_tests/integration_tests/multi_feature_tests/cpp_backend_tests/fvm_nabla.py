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

import sys
from dataclasses import dataclass

import gt4py.next as gtx
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import closure, fundef, offset
from gt4py.next.iterator.tracing import trace_fencil_definition
from gt4py.next.program_processors.codegens.gtfn.gtfn_backend import generate


E2V = offset("E2V")
V2E = offset("V2E")


@fundef
def compute_zavgS(pp, S_M):
    zavg = 0.5 * (deref(shift(E2V, 0)(pp)) + deref(shift(E2V, 1)(pp)))
    return make_tuple(tuple_get(0, deref(S_M)) * zavg, tuple_get(1, deref(S_M)) * zavg)


@fundef
def tuple_dot_fun(acc, zavgS, sign):
    return make_tuple(
        tuple_get(0, acc) + tuple_get(0, zavgS) * sign,
        tuple_get(1, acc) + tuple_get(1, zavgS) * sign,
    )


@fundef
def tuple_dot(a, b):
    return reduce(tuple_dot_fun, make_tuple(0.0, 0.0))(a, b)


@fundef
def compute_pnabla(pp, S_M, sign, vol):
    zavgS = lift(compute_zavgS)(pp, S_M)
    pnabla_M = tuple_dot(neighbors(V2E, zavgS), deref(sign))
    return make_tuple(tuple_get(0, pnabla_M) / deref(vol), tuple_get(1, pnabla_M) / deref(vol))


def zavgS_fencil(edge_domain, out, pp, S_M):
    closure(
        edge_domain,
        compute_zavgS,
        out,
        [pp, S_M],
    )


Vertex = gtx.Dimension("Vertex")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)


def nabla_fencil(n_vertices, n_levels, out, pp, S_M, sign, vol):
    closure(
        unstructured_domain(named_range(Vertex, 0, n_vertices), named_range(K, 0, n_levels)),
        compute_pnabla,
        out,
        [pp, S_M, sign, vol],
    )


@dataclass
class DummyConnectivity:
    max_neighbors: int
    has_skip_values: int
    origin_axis: gtx.Dimension = gtx.Dimension("dummy_origin")
    neighbor_axis: gtx.Dimension = gtx.Dimension("dummy_neighbor")
    index_type: type[int] = int

    def mapped_index(_, __) -> int:
        return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file> <imperative>")
    output_file = sys.argv[1]
    imperative = sys.argv[2].lower() == "true"

    # prog = trace(zavgS_fencil, [None] * 4) # TODO allow generating of 2 fencils
    prog = trace_fencil_definition(nabla_fencil, [None] * 7, use_arg_types=False)
    offset_provider = {
        "V2E": DummyConnectivity(max_neighbors=6, has_skip_values=True),
        "E2V": DummyConnectivity(max_neighbors=2, has_skip_values=False),
    }
    generated_code = generate(prog, offset_provider=offset_provider, imperative=imperative)

    with open(output_file, "w+") as output:
        output.write(generated_code)
