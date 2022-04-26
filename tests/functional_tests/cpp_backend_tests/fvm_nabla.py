import sys

from functional.iterator.backends.gtfn.backend import generate
from functional.iterator.builtins import *
from functional.iterator.runtime import closure, fundef, offset
from functional.iterator.tracing import trace


E2V = offset("E2V")
V2E = offset("V2E")


@fundef
def compute_zavgS(pp, S_M):
    zavg = 0.5 * (deref(shift(E2V, 0)(pp)) + deref(shift(E2V, 1)(pp)))
    return make_tuple(tuple_get(0, deref(S_M)) * zavg, tuple_get(1, deref(S_M)) * zavg)


def _unroll_reduce(binop, init, n):
    def impl(fun):
        acc = init
        for i in range(n):
            acc = binop(acc, fun(i))
        return acc

    return impl


@fundef
def compute_pnabla(pp, S_M, sign, vol):
    zavgS = lift(compute_zavgS)(pp, S_M)

    def step(zavgS, sign, tuple_index, neigh_index, prev):
        return if_(
            can_deref(shift(V2E, neigh_index)(zavgS)),
            tuple_get(tuple_index, deref(shift(V2E, neigh_index)(zavgS)))
            * tuple_get(neigh_index, deref(sign))
            + prev,
            prev,
        )

    pnabla_M = make_tuple(
        _unroll_reduce(lambda a, b: a + b, 0.0, 6)(lambda i: step(zavgS, sign, 0, i, 0.0)),
        _unroll_reduce(lambda a, b: a + b, 0.0, 6)(lambda i: step(zavgS, sign, 1, i, 0.0)),
    )

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
    generated_code = generate(prog, grid_type="unstructured")

    with open(output_file, "w+") as output:
        output.write(generated_code)
