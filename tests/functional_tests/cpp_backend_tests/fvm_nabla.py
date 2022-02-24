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


@fundef
def compute_pnabla(pp, S_M, sign, vol):
    zavgS = lift(compute_zavgS)(pp, S_M)

    init_0 = if_(
        can_deref(shift(V2E, 0)(zavgS)),
        tuple_get(0, deref(shift(V2E, 0)(zavgS))) * tuple_get(0, deref(sign)),
        0,
    )
    init_1 = if_(
        can_deref(shift(V2E, 0)(zavgS)),
        tuple_get(1, deref(shift(V2E, 0)(zavgS))) * tuple_get(0, deref(sign)),
        0,
    )

    state_0_0 = if_(
        can_deref(shift(V2E, 1)(zavgS)),
        tuple_get(0, deref(shift(V2E, 1)(zavgS))) * tuple_get(1, deref(sign)) + init_0,
        init_0,
    )
    state_0_1 = if_(
        can_deref(shift(V2E, 1)(zavgS)),
        tuple_get(1, deref(shift(V2E, 1)(zavgS))) * tuple_get(1, deref(sign)) + init_1,
        init_1,
    )

    state_1_0 = if_(
        can_deref(shift(V2E, 2)(zavgS)),
        tuple_get(0, deref(shift(V2E, 2)(zavgS))) * tuple_get(2, deref(sign)) + state_0_0,
        state_0_0,
    )
    state_1_1 = if_(
        can_deref(shift(V2E, 2)(zavgS)),
        tuple_get(1, deref(shift(V2E, 2)(zavgS))) * tuple_get(2, deref(sign)) + state_0_1,
        state_0_1,
    )

    state_2_0 = if_(
        can_deref(shift(V2E, 3)(zavgS)),
        tuple_get(0, deref(shift(V2E, 3)(zavgS))) * tuple_get(3, deref(sign)) + state_1_0,
        state_1_0,
    )
    state_2_1 = if_(
        can_deref(shift(V2E, 3)(zavgS)),
        tuple_get(1, deref(shift(V2E, 3)(zavgS))) * tuple_get(3, deref(sign)) + state_1_1,
        state_1_1,
    )

    state_3_0 = if_(
        can_deref(shift(V2E, 4)(zavgS)),
        tuple_get(0, deref(shift(V2E, 4)(zavgS))) * tuple_get(4, deref(sign)) + state_2_0,
        state_2_0,
    )
    state_3_1 = if_(
        can_deref(shift(V2E, 4)(zavgS)),
        tuple_get(1, deref(shift(V2E, 4)(zavgS))) * tuple_get(4, deref(sign)) + state_2_1,
        state_2_1,
    )

    state_4_0 = if_(
        can_deref(shift(V2E, 5)(zavgS)),
        tuple_get(0, deref(shift(V2E, 5)(zavgS))) * tuple_get(5, deref(sign)) + state_3_0,
        state_3_0,
    )
    state_4_1 = if_(
        can_deref(shift(V2E, 5)(zavgS)),
        tuple_get(1, deref(shift(V2E, 5)(zavgS))) * tuple_get(5, deref(sign)) + state_3_1,
        state_3_1,
    )

    return make_tuple(state_4_0 / deref(vol), state_4_1 / deref(vol))


def zavgS_fencil(edge_domain, out, pp, S_M):
    closure(
        edge_domain,
        compute_zavgS,
        [out],
        [pp, S_M],
    )


def nabla_fencil(vertex_domain, out, pp, S_M, sign, vol):
    closure(
        vertex_domain,
        compute_pnabla,
        [out],
        [pp, S_M, sign, vol],
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    prog_0 = trace(zavgS_fencil, [None] * 6)
    prog = trace(nabla_fencil, [None] * 6)
    prog.fencil_definitions.append(prog_0.fencil_definitions[0])
    generated_code = generate(prog)

    with open(output_file, "w+") as output:
        output.write(generated_code)
