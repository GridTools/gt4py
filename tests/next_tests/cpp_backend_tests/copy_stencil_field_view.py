import sys

from numpy import float64

from gt4py.next.common import Field
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.iterator.runtime import CartesianAxis
from gt4py.next.program_processors.codegens.gtfn.gtfn_backend import generate


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


@field_operator
def copy_stencil(inp: Field[[IDim, JDim, KDim], float64]) -> Field[[IDim, JDim, KDim], float64]:
    return inp


@program
def copy_program(
    inp: Field[[IDim, JDim, KDim], float64],
    out: Field[[IDim, JDim, KDim], float64],
    out2: Field[[IDim, JDim, KDim], float64],
):
    copy_stencil(inp, out=out)
    copy_stencil(inp, out=out2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    prog = copy_program.itir
    generated_code = generate(prog, offset_provider={})

    with open(output_file, "w+") as output:
        output.write(generated_code)
