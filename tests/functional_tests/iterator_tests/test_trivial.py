import numpy as np

from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import *


I = offset("I")
J = offset("J")

IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")


@fundef
def foo(foo_inp):
    return deref(foo_inp)


@fundef
def bar(bar_inp):
    return deref(lift(foo)(bar_inp))


@fundef
def baz(baz_inp):
    return deref(lift(bar)(baz_inp))


@fendef(offset_provider={"I": IDim, "J": JDim})
def foobarbaz(inp, out, x, y):
    closure(
        domain(named_range(IDim, 0, x), named_range(JDim, 0, y)),
        baz,
        [out],
        [inp],
    )


def test_trivial(backend, use_tmps):
    backend, validate = backend
    rng = np.random.default_rng()
    inp = rng.uniform(size=(5, 7, 9))
    out = np.copy(inp)
    shape = (out.shape[0], out.shape[1])

    inp_s = np_as_located_field(IDim, JDim, origin={IDim: 0, JDim: 0})(inp[:, :, 0])
    out_s = np_as_located_field(IDim, JDim)(np.zeros_like(inp[:, :, 0]))

    foobarbaz(inp_s, out_s, inp.shape[0], inp.shape[1], backend=backend, use_tmps=use_tmps)

    if validate:
        assert np.allclose(out[:, :, 0], out_s)
