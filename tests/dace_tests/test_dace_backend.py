import numpy as np

from gt4py.next.common import Dimension, Field
from gt4py.next.ffront.decorator import field_operator
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.program_processors.runners.dace_iterator import run_dace_iterator


IDim = Dimension("IDim")


@field_operator(backend=run_dace_iterator)
def add(a: Field[[IDim], float], b: Field[[IDim], float]) -> Field[[IDim], float]:
    return a + b + a - a


def test_dace_backend():
    a = np_as_located_field(IDim)(np.array([1, 2, 3], dtype=float))
    b = np_as_located_field(IDim)(np.array([5, 7, 0], dtype=float))
    r = np_as_located_field(IDim)(np.array([0, 0, 0], dtype=float))
    add(a, b, out=r, offset_provider={})

    assert np.allclose(np.asarray(a) + np.asarray(b), np.asarray(r))
