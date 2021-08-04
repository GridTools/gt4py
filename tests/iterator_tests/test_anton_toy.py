from iterator.builtins import *
from iterator.embedded import np_as_located_field
from iterator.runtime import *
import numpy as np


@fundef
def ldif(d):
    return lambda inp: deref(shift(d, -1)(inp)) - deref(inp)


@fundef
def rdif(d):
    # return compose(ldif(d), shift(d, 1))
    return lambda inp: ldif(d)(shift(d, 1)(inp))


@fundef
def dif2(d):
    # return compose(ldif(d), lift(rdif(d)))
    return lambda inp: ldif(d)(lift(rdif(d))(inp))


i = offset("i")
j = offset("j")


@fundef
def lap(inp):
    return dif2(i)(inp) + dif2(j)(inp)


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


@fendef
def fencil(x, y, z, output, input):
    closure(
        domain(named_range(IDim, 0, x), named_range(JDim, 0, y), named_range(KDim, 0, z)),
        lap,
        [output],
        [input],
    )


fencil(*([None] * 5), backend="lisp")
fencil(*([None] * 5), backend="cpptoy")


def naive_lap(inp):
    shape = [inp.shape[0] - 2, inp.shape[1] - 2, inp.shape[2]]
    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(0, shape[2]):
                out[i, j, k] = -4 * inp[i, j, k] + (
                    inp[i + 1, j, k] + inp[i - 1, j, k] + inp[i, j + 1, k] + inp[i, j - 1, k]
                )
    return out


def test_anton_toy():
    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp = np_as_located_field(IDim, JDim, KDim, origin={IDim: 1, JDim: 1, KDim: 0})(
        rng.normal(size=(shape[0] + 2, shape[1] + 2, shape[2])),
    )
    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))
    ref = naive_lap(inp)

    fencil(
        shape[0],
        shape[1],
        shape[2],
        out,
        inp,
        backend="double_roundtrip",
        offset_provider={"i": IDim, "j": JDim},
    )

    assert np.allclose(out, ref)
