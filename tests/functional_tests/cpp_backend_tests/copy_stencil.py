import sys

from functional.iterator.backends import gtfn
from functional.iterator.builtins import *
from functional.iterator.runtime import CartesianAxis, closure, fundef
from functional.iterator.tracing import trace


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


@fundef
def copy_stencil(inp):
    return deref(inp)


def copy_fencil(x, y, z, inp, out):
    closure(
        domain(named_range(IDim, 0, x), named_range(JDim, 0, y), named_range(KDim, 0, z)),
        copy_stencil,
        [out],
        [inp],
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: copy_stencil.py <output_file>")
    output_file = sys.argv[1]

    prog = trace(copy_fencil, [None] * 5)
    generated_code = gtfn.gtfn.apply(prog)

    with open(output_file, "w+") as output:
        output.write(generated_code)
