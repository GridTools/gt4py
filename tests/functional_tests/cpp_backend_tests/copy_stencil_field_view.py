import sys

from numpy import float64

from functional.common import Field
from functional.fencil_processors.gtfn.gtfn_backend import generate
from functional.ffront.decorator import field_operator, program
from functional.iterator.runtime import CartesianAxis


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


# For use outside of GT4Py we should probably support a domain being used explicitly instead of relative to out field
# someting like:
# @program
# def copy_program(
#     isize, jsize, ksize,
#     inp: Field[[IDim, JDim, KDim], float64],
#     out: Field[[IDim, JDim, KDim], float64],
#     out2: Field[[IDim, JDim, KDim], float64],
# ):
#     copy_stencil(inp, out=out[{IDim:[0,isize], JDim:[...], ...}])
#     copy_stencil(inp, out=out2[...])

# or just
# @program
# def copy_program(
#     domain,
#     inp: Field[[IDim, JDim, KDim], float64],
#     out: Field[[IDim, JDim, KDim], float64],
#     out2: Field[[IDim, JDim, KDim], float64],
# ):
#     copy_stencil(inp, out=out[domain])
#     copy_stencil(inp, out=out2[domain])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    prog = copy_program.itir
    generated_code = generate(prog, grid_type="Cartesian")

    with open(output_file, "w+") as output:
        output.write(generated_code)
