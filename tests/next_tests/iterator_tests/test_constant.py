import numpy as np

from gt4py.next.common import Dimension
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.iterator.runtime import fundef
from gt4py.next.program_processors.runners import roundtrip


IDim = Dimension("IDim")


def test_constant():
    @fundef
    def add_constant(inp):
        def constant_stencil():  # this is traced as a lambda, TODO directly feed iterator IR nodes
            return 1

        return deref(inp) + deref(lift(constant_stencil)())

    inp = np_as_located_field(IDim)(np.asarray([0, 42]))
    res = np_as_located_field(IDim)(np.zeros_like(inp))

    add_constant[{IDim: range(2)}](inp, out=res, offset_provider={}, backend=roundtrip.executor)

    assert np.allclose(res, np.asarray([1, 43]))
