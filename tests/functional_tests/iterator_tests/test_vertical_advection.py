import numpy as np
import pytest

from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import *


@fundef
def tridiag_forward(state, a, b, c, d):
    return make_tuple(
        deref(c) / (deref(b) - deref(a) * tuple_get(0, state)),
        (deref(d) - deref(a) * tuple_get(1, state)) / (deref(b) - deref(a) * tuple_get(0, state)),
    )


@fundef
def tridiag_backward(x_kp1, cpdp):
    cpdpv = deref(cpdp)
    cp = tuple_get(0, cpdpv)
    dp = tuple_get(1, cpdpv)
    return dp - cp * x_kp1


@fundef
def solve_tridiag(a, b, c, d):
    cpdp = lift(scan(tridiag_forward, True, make_tuple(0.0, 0.0)))(a, b, c, d)
    return scan(tridiag_backward, False, 0.0)(cpdp)


@pytest.fixture
def tridiag_reference():
    shape = (3, 7, 5)
    rng = np.random.default_rng()
    a = rng.normal(size=shape)
    b = rng.normal(size=shape) * 2
    c = rng.normal(size=shape)
    d = rng.normal(size=shape)

    matrices = np.zeros(shape + shape[-1:])
    i = np.arange(shape[2])
    matrices[:, :, i[1:], i[:-1]] = a[:, :, 1:]
    matrices[:, :, i, i] = b
    matrices[:, :, i[:-1], i[1:]] = c[:, :, :-1]
    x = np.linalg.solve(matrices, d)
    return a, b, c, d, x


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")


@fendef
def fen_solve_tridiag(i_size, j_size, k_size, a, b, c, d, x):
    closure(
        domain(
            named_range(IDim, 0, i_size),
            named_range(JDim, 0, j_size),
            named_range(KDim, 0, k_size),
        ),
        solve_tridiag,
        x,
        [a, b, c, d],
    )


def test_tridiag(tridiag_reference, backend, use_tmps):
    if use_tmps:
        pytest.xfail("use_tmps currently not supported for scans")
    backend, validate = backend
    a, b, c, d, x = tridiag_reference
    shape = a.shape
    as_3d_field = np_as_located_field(IDim, JDim, KDim)
    a_s = as_3d_field(a)
    b_s = as_3d_field(b)
    c_s = as_3d_field(c)
    d_s = as_3d_field(d)
    x_s = as_3d_field(np.zeros_like(x))

    fen_solve_tridiag(
        shape[0],
        shape[1],
        shape[2],
        a_s,
        b_s,
        c_s,
        d_s,
        x_s,
        offset_provider={},
        column_axis=KDim,
        backend=backend,
        use_tmps=use_tmps,
    )

    if validate:
        assert np.allclose(x, np.asarray(x_s))
