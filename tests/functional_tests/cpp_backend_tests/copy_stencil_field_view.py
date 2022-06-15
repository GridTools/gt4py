import sys

from numpy import float64

from functional.common import Domain, Field
from functional.ffront.decorator import field_operator, program
from functional.iterator.backends.gtfn.gtfn_backend import generate
from functional.iterator.runtime import CartesianAxis


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


@field_operator
def copy_stencil(inp: Field[[IDim, JDim, KDim], float64]) -> Field[[IDim, JDim, KDim], float64]:
    return inp


@program
def copy(
    dom: Domain[IDim, JDim, KDim],
    inp: Field[[IDim, JDim, KDim], float64],
    out: Field[[IDim, JDim, KDim], float64],
):
    copy_stencil(inp, out=out[dom])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    prog = copy.itir
    prog.params = list(
        filter(lambda p: not p.id.startswith("__"), prog.params)
    )  # hack to remove all field sizes which are not used if domain is explicit/absolute
    generated_code = generate(prog, grid_type="Cartesian")

    with open(output_file, "w+") as output:
        output.write(generated_code)
