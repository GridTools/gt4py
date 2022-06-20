import sys

from functional.fencil_processors.gtfn.gtfn_backend import generate
from functional.iterator.builtins import *
from functional.iterator.runtime import CartesianAxis, closure, fundef, offset
from functional.iterator.tracing import trace


@fundef
def ldif(d):
    return lambda inp: deref(shift(d, -1)(inp)) - deref(inp)


@fundef
def rdif(d):
    return lambda inp: ldif(d)(shift(d, 1)(inp))


@fundef
def dif2(d):
    return lambda inp: ldif(d)(lift(rdif(d))(inp))


i = offset("i")
j = offset("j")


@fundef
def lap(inp):
    return dif2(i)(inp) + dif2(j)(inp)


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


def lap_fencil(dom, out, inp):
    closure(
        dom,
        lap,
        out,
        [inp],
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    prog = trace(lap_fencil, [None] * 3)
    generated_code = generate(prog, grid_type="Cartesian")

    with open(output_file, "w+") as output:
        output.write(generated_code)
