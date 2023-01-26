import dace

from functional.ffront.decorator import field_operator, program, scan_operator
from functional.common import Field, Dimension
from functional.iterator.embedded import np_as_located_field
from functional.program_processors.runners.gtfn_cpu import run_gtfn
import numpy as np


IDim = Dimension("IDim")


@field_operator
def field_whatever(...):
    return ...

@field_operator(backend=run_gtfn)
def add(a: Field[[IDim], float], b: Field[[IDim], float]) -> Field[[IDim], float]:
    tmp = field_whatever(a, b)
    return a + b + tmp


def test_dace_backend():
    a = np_as_located_field(IDim)(np.array([1, 2, 3], dtype=float))
    b = np_as_located_field(IDim)(np.array([5, 7, 0], dtype=float))
    r = np_as_located_field(IDim)(np.array([0, 0, 0], dtype=float))
    add(a, b, out=r, offset_provider={})

    assert np.allclose(np.asarray(a) + np.asarray(b), np.asarray(r))

@dace.program
def inner(...):
    return ...

@dace.program
def outer(...):
    return inner()+inner()