import numpy as np

from gt4py.next.common import Dimension, Field, DimensionKind
from gt4py.next.ffront.fbuiltins import FieldOffset
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.iterator.embedded import np_as_located_field, NeighborTableOffsetProvider
from gt4py.next.program_processors.runners.dace_iterator import run_dace_iterator

IDim = Dimension("IDim")
I2IDim = Dimension("I2IDim", kind=DimensionKind.LOCAL)
IOff = FieldOffset("IOff", IDim, (IDim,))
I2IOff = FieldOffset("I2IOff", IDim, (IDim, I2IDim))

neighbor_data = np.array([
    [2, 1],
    [0, 2],
    [1, 0],
], dtype=np.int64)
neighbor_table = NeighborTableOffsetProvider(neighbor_data, IDim, I2IDim, 2)


@field_operator(backend=run_dace_iterator)
def add(a: Field[[IDim], float], b: Field[[IDim], float]) -> Field[[IDim], float]:
    return a + b + a - a


def test_pointwise():
    a = np_as_located_field(IDim)(np.array([1, 2, 3], dtype=float))
    b = np_as_located_field(IDim)(np.array([5, 7, 0], dtype=float))
    r = np_as_located_field(IDim)(np.array([0, 0, 0], dtype=float))
    add(a, b, out=r, offset_provider={})

    assert np.allclose(np.asarray(a) + np.asarray(b), np.asarray(r))


@field_operator
def shift_left(a: Field[[IDim], float]) -> Field[[IDim], float]:
    return a(IOff[1])


@program(backend=run_dace_iterator)
def shift_program(a: Field[[IDim], float], out: Field[[IDim], float]):
    shift_left(a, out=out, domain={IDim: (0, 2)})


def test_shift_left():
    a = np_as_located_field(IDim)(np.array([1, 2, 3], dtype=float))
    r = np_as_located_field(IDim)(np.array([0, 0], dtype=float))

    shift_program(a, out=r, offset_provider={"IOff": IDim})

    assert np.allclose(np.asarray(a)[1::], np.asarray(r))


@field_operator
def indirect_addressing(a: Field[[IDim], float]) -> Field[[IDim], float]:
    return a(I2IOff[1]) - a(I2IOff[0])


@program(backend=run_dace_iterator)
def indirect_addressing_program(z: Field[[IDim], float], out: Field[[IDim], float]):
    indirect_addressing(z, out=out)


def test_indirect_addressing():
    a = np_as_located_field(IDim)(np.array([1, 2, 3], dtype=float))
    r = np_as_located_field(IDim)(np.array([0, 0, 0], dtype=float))
    e = np_as_located_field(IDim)(np.array([-1, 2, -1], dtype=float))

    indirect_addressing_program(a, out=r, offset_provider={"I2IOff": neighbor_table})

    assert np.allclose(np.asarray(e), np.asarray(r))
