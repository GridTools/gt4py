import numpy as np

from functional.common import Dimension, Field
from functional.ffront.decorator import field_operator, program, scan_operator
from functional.iterator.embedded import np_as_located_field
from functional.program_processors.runners.dace_fieldview import run_dace_fieldview


IDim = Dimension("IDim")


@field_operator(backend=run_dace_fieldview)
def add(a: Field[[IDim], float], b: Field[[IDim], float]) -> Field[[IDim], float]:
    return a + b


def test_dace_backend():
    a = np_as_located_field(IDim)(np.array([1, 2, 3], dtype=float))
    b = np_as_located_field(IDim)(np.array([5, 7, 0], dtype=float))
    r = np_as_located_field(IDim)(np.array([0, 0, 0], dtype=float))
    add(a, b, out=r, offset_provider={})

    assert np.allclose(np.asarray(a) + np.asarray(b), np.asarray(r))
