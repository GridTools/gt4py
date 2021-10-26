# (defun calc (p_vn input_on_cell)
#   (do_some_math
#     (deref
#         ((if (less (deref p_vn) 0)
#             (shift e2c 0)
#             (shift e2c 1)
#          )
#          input_on_cell
#         )
#     )
#   )
# )
import numpy as np
from numpy.core.numeric import allclose

from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import *


I = offset("I")


@fundef
def compute_shift(cond):
    return if_(deref(cond) < 0, shift(I, -1), shift(I, 1))


@fundef
def foo(inp, cond):
    return deref(compute_shift(cond)(inp))


@fendef
def fencil(size, inp, cond, out):
    closure(domain(named_range(IDim, 0, size)), foo, [out], [inp, cond])


IDim = CartesianAxis("IDim")


def test_simple_indirection():
    shape = [8]
    inp = np_as_located_field(IDim, origin={IDim: 1})(np.asarray(range(shape[0] + 2)))
    rng = np.random.default_rng()
    cond = np_as_located_field(IDim)(rng.normal(size=shape))
    out = np_as_located_field(IDim)(np.zeros(shape))

    ref = np.zeros(shape)
    for i in range(shape[0]):
        ref[i] = inp[i - 1] if cond[i] < 0 else inp[i + 1]

    fencil(shape[0], inp, cond, out, offset_provider={"I": IDim})

    assert allclose(ref, out)
