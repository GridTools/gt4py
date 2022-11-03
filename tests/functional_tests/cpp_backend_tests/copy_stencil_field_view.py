import sys

from numpy import float64

from functional.common import Field
from functional.ffront.decorator import field_operator, program
from functional.iterator.runtime import CartesianAxis
from functional.program_processors.codegens.gtfn.gtfn_backend import generate


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
    generated_code = generate(prog)

    with open(output_file, "w+") as output:
        output.write(generated_code)
