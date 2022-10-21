import sys

from functional.iterator.builtins import *
from functional.iterator.runtime import CartesianAxis, closure, fundef
from functional.iterator.tracing import trace
from functional.program_processors.codegens.gtfn.gtfn_backend import generate


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


@fundef
def copy_stencil(inp):
    return deref(inp)


def copy_fencil(isize, jsize, ksize, inp, out):
    closure(
        cartesian_domain(
            named_range(IDim, 0, isize), named_range(JDim, 0, jsize), named_range(KDim, 0, ksize)
        ),
        copy_stencil,
        out,
        [inp],
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    prog = trace(copy_fencil, [None] * 5)
    generated_code = generate(prog)

    with open(output_file, "w+") as output:
        output.write(generated_code)
