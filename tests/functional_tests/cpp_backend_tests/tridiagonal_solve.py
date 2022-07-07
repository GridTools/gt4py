import sys

from functional.fencil_processors.gtfn.gtfn_backend import generate
from functional.iterator.builtins import *
from functional.iterator.runtime import CartesianAxis, closure, fundef
from functional.iterator.tracing import trace
from functional.iterator.transforms import LiftMode


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


@fundef
def tridiag_forward(state, a, b, c, d):
    return make_tuple(
        deref(c) / (deref(b) - deref(a) * tuple_get(0, state)),
        (deref(d) - deref(a) * tuple_get(1, state)) / (deref(b) - deref(a) * tuple_get(0, state)),
    )


@fundef
def tridiag_backward(x_kp1, cpdp):
    cpdpv = deref(cpdp)
    cp = tuple_get(0, cpdpv)
    dp = tuple_get(1, cpdpv)
    return dp - cp * x_kp1


@fundef
def solve_tridiag(a, b, c, d):
    cpdp = lift(scan(tridiag_forward, True, make_tuple(0.0, 0.0)))(a, b, c, d)
    return scan(tridiag_backward, False, 0.0)(cpdp)


def tridiagonal_solve_fencil(dom, a, b, c, d, x):
    closure(dom, solve_tridiag, x, [a, b, c, d])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    prog = trace(tridiagonal_solve_fencil, [None] * 6)
    offset_provider = {"I": CartesianAxis("IDim"), "J": CartesianAxis("JDim")}
    generated_code = generate(
        prog,
        grid_type="Cartesian",
        offset_provider=offset_provider,
        lift_mode=LiftMode.SIMPLE_HEURISTIC,
    )

    with open(output_file, "w+") as output:
        output.write(generated_code)
