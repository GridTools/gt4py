import sys

from functional.iterator.backends.gtfn.gtfn_backend import generate
from functional.iterator.builtins import *
from functional.iterator.runtime import CartesianAxis, closure, fundef
from functional.iterator.tracing import trace


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


@fundef
def copy_stencil(inp):
    return deref(inp)


def copy_fencil(dom, inp, out):
    closure(
        dom,
        copy_stencil,
        out,
        [inp],
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    prog = trace(copy_fencil, [None] * 3)
    generated_code = generate(prog, grid_type="Cartesian")

    with open(output_file, "w+") as output:
        output.write(generated_code)
