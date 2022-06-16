import sys
from types import SimpleNamespace

from functional.iterator import library
from functional.iterator.backends.gtfn.gtfn_backend import generate
from functional.iterator.builtins import *
from functional.iterator.runtime import closure, fundef, offset
from functional.iterator.tracing import trace


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
    pnabla_M = tuple_dot(shift(V2E)(zavgS), sign)
    return make_tuple(tuple_get(0, pnabla_M) / deref(vol), tuple_get(1, pnabla_M) / deref(vol))


def zavgS_fencil(edge_domain, out, pp, S_M):
    closure(
        edge_domain,
        compute_zavgS,
        out,
        [pp, S_M],
    )


def nabla_fencil(vertex_domain, out, pp, S_M, sign, vol):
    closure(
        vertex_domain,
        compute_pnabla,
        out,
        [pp, S_M, sign, vol],
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    # prog = trace(zavgS_fencil, [None] * 4) # TODO allow generating of 2 fencils
    prog = trace(nabla_fencil, [None] * 6)
    offset_provider = {
        "V2E": SimpleNamespace(max_neighbors=6, has_skip_values=True),
        "E2V": SimpleNamespace(max_neighbors=2, has_skip_values=False),
    }
    generated_code = generate(prog, grid_type="unstructured", offset_provider=offset_provider)

    with open(output_file, "w+") as output:
        output.write(generated_code)
