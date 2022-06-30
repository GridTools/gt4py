import numpy as np

from functional.common import Dimension
from functional.fencil_processors import double_roundtrip, roundtrip
from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import closure, fendef, fundef, offset


I = offset("I")
J = offset("J")
I_loc = Dimension("I_loc")
J_loc = Dimension("J_loc")


@fundef
def foo(inp):
    return deref(shift(J, 1)(inp))


@fendef(offset_provider={"I": I_loc, "J": J_loc})
def fencil(output, input):
    closure(
        cartesian_domain(named_range(I_loc, 0, 1), named_range(J_loc, 0, 1)),
        foo,
        output,
        [input],
    )


@fendef(offset_provider={"I": J_loc, "J": I_loc})
def fencil_swapped(output, input):
    closure(
        cartesian_domain(named_range(I_loc, 0, 1), named_range(J_loc, 0, 1)),
        foo,
        output,
        [input],
    )


def test_cartesian_offset_provider():
    inp = np_as_located_field(I_loc, J_loc)(np.asarray([[0, 42], [1, 43]]))
    out = np_as_located_field(I_loc, J_loc)(np.asarray([[-1]]))

    fencil(out, inp)
    assert out[0][0] == 42

    fencil_swapped(out, inp)
    assert out[0][0] == 1

    fencil(out, inp, backend=roundtrip.executor)
    assert out[0][0] == 42

    fencil(out, inp, backend=double_roundtrip.executor)
    assert out[0][0] == 42


@fundef
def delay_complete_shift(inp):
    return deref(shift(I, J, 1, 1)(inp))


@fendef(offset_provider={"I": J_loc, "J": I_loc})
def delay_complete_shift_fencil(output, input):
    closure(
        cartesian_domain(named_range(I_loc, 0, 1), named_range(J_loc, 0, 1)),
        delay_complete_shift,
        output,
        [input],
    )


def test_delay_complete_shift():
    inp = np_as_located_field(I_loc, J_loc)(np.asarray([[0, 42], [1, 43]]))

    out = np_as_located_field(I_loc, J_loc)(np.asarray([[-1]]))
    delay_complete_shift_fencil(out, inp)
    assert out[0, 0] == 43

    out = np_as_located_field(I_loc, J_loc)(np.asarray([[-1]]))
    delay_complete_shift_fencil(out, inp, backend=roundtrip.executor)
    assert out[0, 0] == 43
