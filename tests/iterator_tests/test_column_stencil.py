import numpy as np
import pytest

from iterator.builtins import *
from iterator.embedded import np_as_located_field
from iterator.runtime import *


I = offset("I")
K = offset("K")


@fundef
def multiply_stencil(inp):
    return deref(shift(K, 1, I, 1)(inp))


KDim = CartesianAxis("KDim")
IDim = CartesianAxis("IDim")


@fendef(column_axis=KDim)
def fencil(i_size, k_size, inp, out):
    closure(
        domain(named_range(IDim, 0, i_size), named_range(KDim, 0, k_size)),
        multiply_stencil,
        [out],
        [inp],
    )


def test_column_stencil(backend, use_tmps):
    backend, validate = backend
    shape = [5, 7]
    inp = np_as_located_field(IDim, KDim)(
        np.fromfunction(lambda i, k: i * 10 + k, [shape[0] + 1, shape[1] + 1])
    )
    out = np_as_located_field(IDim, KDim)(np.zeros(shape))

    ref = np.asarray(inp)[1:, 1:]

    fencil(
        shape[0],
        shape[1],
        inp,
        out,
        offset_provider={"I": IDim, "K": KDim},
        backend=backend,
        use_tmps=use_tmps,
    )

    if validate:
        assert np.allclose(ref, out)


def test_column_stencil_with_k_origin(backend, use_tmps):
    backend, validate = backend
    shape = [5, 7]
    raw_inp = np.fromfunction(lambda i, k: i * 10 + k, [shape[0] + 1, shape[1] + 2])
    inp = np_as_located_field(IDim, KDim, origin={IDim: 0, KDim: 1})(raw_inp)
    out = np_as_located_field(IDim, KDim)(np.zeros(shape))

    ref = np.asarray(inp)[1:, 2:]

    fencil(
        shape[0],
        shape[1],
        inp,
        out,
        offset_provider={"I": IDim, "K": KDim},
        backend=backend,
        use_tmps=use_tmps,
    )

    if validate:
        assert np.allclose(ref, out)


@fundef
def sum_scanpass(state, inp):
    return if_(is_none(state), deref(inp), state + deref(inp))


@fundef
def ksum(inp):
    return scan(sum_scanpass, True, None)(inp)


@fendef(column_axis=KDim)
def ksum_fencil(i_size, k_size, inp, out):
    closure(
        domain(named_range(IDim, 0, i_size), named_range(KDim, 0, k_size)),
        ksum,
        [out],
        [inp],
    )


def test_ksum_scan(backend, use_tmps):
    if use_tmps:
        pytest.xfail("use_tmps currently not supported for scans")
    backend, validate = backend
    shape = [1, 7]
    inp = np_as_located_field(IDim, KDim)(np.asarray([list(range(7))]))
    out = np_as_located_field(IDim, KDim)(np.zeros(shape))

    ref = np.asarray([[0, 1, 3, 6, 10, 15, 21]])

    ksum_fencil(
        shape[0],
        shape[1],
        inp,
        out,
        offset_provider={"I": IDim, "K": KDim},
        backend=backend,
        use_tmps=use_tmps,
    )

    if validate:
        assert np.allclose(ref, np.asarray(out))


@fundef
def ksum_back(inp):
    return scan(sum_scanpass, False, None)(inp)


@fendef(column_axis=KDim)
def ksum_back_fencil(i_size, k_size, inp, out):
    closure(
        domain(named_range(IDim, 0, i_size), named_range(KDim, 0, k_size)),
        ksum_back,
        [out],
        [inp],
    )


def test_ksum_back_scan(backend, use_tmps):
    if use_tmps:
        pytest.xfail("use_tmps currently not supported for scans")
    backend, validate = backend
    shape = [1, 7]
    inp = np_as_located_field(IDim, KDim)(np.asarray([list(range(7))]))
    out = np_as_located_field(IDim, KDim)(np.zeros(shape))

    ref = np.asarray([[21, 21, 20, 18, 15, 11, 6]])

    ksum_back_fencil(
        shape[0],
        shape[1],
        inp,
        out,
        offset_provider={"I": IDim, "K": KDim},
        backend=backend,
        use_tmps=use_tmps,
    )

    if validate:
        assert np.allclose(ref, np.asarray(out))
