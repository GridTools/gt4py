import sys

from functional.iterator.backends import gtfn
from functional.iterator.builtins import *
from functional.iterator.runtime import CartesianAxis, closure, fundef, offset
from functional.iterator.tracing import trace
from functional.iterator.transforms.common import apply_common_transforms


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


# @fendef(offset_provider={"i": IDim, "j": JDim})
def lap_fencil(x, y, z, out, inp):
    closure(
        domain(named_range(IDim, 0, x), named_range(JDim, 0, y), named_range(KDim, 0, z)),
        # domain(named_range(IDim, 1, x + 1), named_range(JDim, 1, y + 2), named_range(KDim, 0, z)), TODO allow start != 0
        lap,
        [out],
        [inp],
    )


# def naive_lap(inp):
#     shape = [inp.shape[0] - 2, inp.shape[1] - 2, inp.shape[2]]
#     out = np.zeros(shape)
#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             for k in range(0, shape[2]):
#                 out[i, j, k] = -4 * inp[i, j, k] + (
#                     inp[i + 1, j, k] + inp[i - 1, j, k] + inp[i, j + 1, k] + inp[i, j - 1, k]
#                 )
#     return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <output_file>")
    output_file = sys.argv[1]

    prog = trace(lap_fencil, [None] * 5)
    generated_code = gtfn.gtfn.apply(prog, grid_type="Cartesian")

    with open(output_file, "w+") as output:
        output.write(generated_code)
