import numpy as np
import pytest

from functional.common import Dimension, DimensionKind
from functional.fencil_processors.runners.gtfn_cpu import run_gtfn
from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import closure, fendef, fundef, offset

from .conftest import run_processor


I = offset("I")
J = offset("J")
K = offset("K")

IDim = Dimension("IDim")
JDim = Dimension("JDim")
KDim = Dimension("KDim", kind=DimensionKind.VERTICAL)


@fundef
def foo(foo_inp):
    return deref(foo_inp)


@fundef
def bar(bar_inp):
    return deref(lift(foo)(bar_inp))


@fundef
def baz(baz_inp):
    return deref(lift(bar)(baz_inp))


def test_trivial(fencil_processor, lift_mode):
    fencil_processor, validate = fencil_processor

    if fencil_processor == run_gtfn:
        pytest.xfail("origin not yet supported in gtfn")

    rng = np.random.default_rng()
    inp = rng.uniform(size=(5, 7, 9))
    out = np.copy(inp)
    shape = (out.shape[0], out.shape[1])

    inp_s = np_as_located_field(IDim, JDim, origin={IDim: 0, JDim: 0})(inp[:, :, 0])
    out_s = np_as_located_field(IDim, JDim)(np.zeros_like(inp[:, :, 0]))

    run_processor(
        baz[cartesian_domain(named_range(IDim, 0, shape[0]), named_range(JDim, 0, shape[1]))],
        fencil_processor,
        inp_s,
        out=out_s,
        lift_mode=lift_mode,
        offset_provider={"I": IDim, "J": JDim},
    )

    if validate:
        assert np.allclose(out[:, :, 0], out_s)


@fendef
def fen_direct_deref(i_size, j_size, out, inp):
    closure(
        cartesian_domain(
            named_range(IDim, 0, i_size),
            named_range(JDim, 0, j_size),
        ),
        deref,
        out,
        [inp],
    )


def test_direct_deref(fencil_processor, lift_mode):
    fencil_processor, validate = fencil_processor
    if fencil_processor == run_gtfn:
        pytest.xfail("extract_fundefs_from_closures() doesn't work for builtins in gtfn")

    rng = np.random.default_rng()
    inp = rng.uniform(size=(5, 7))
    out = np.copy(inp)

    inp_s = np_as_located_field(IDim, JDim)(inp)
    out_s = np_as_located_field(IDim, JDim)(np.zeros_like(inp))

    run_processor(
        fen_direct_deref,
        fencil_processor,
        *out.shape,
        out_s,
        inp_s,
        lift_mode=lift_mode,
        offset_provider=dict(),
    )

    if validate:
        assert np.allclose(out, out_s)


@fundef
def vertical_shift(inp):
    return deref(shift(K, 1)(inp))


def test_vertical_shift_unstructured(fencil_processor):
    fencil_processor, validate = fencil_processor

    k_size = 7

    rng = np.random.default_rng()
    inp = rng.uniform(size=(1, k_size))

    inp_s = np_as_located_field(IDim, KDim)(inp)
    out_s = np_as_located_field(IDim, KDim)(np.zeros_like(inp))

    run_processor(
        vertical_shift[
            unstructured_domain(named_range(IDim, 0, 1), named_range(KDim, 0, k_size - 1))
        ],
        fencil_processor,
        inp_s,
        out=out_s,
        offset_provider={"K": KDim},
    )

    if validate:
        assert np.allclose(inp_s[:, 1:], np.asarray(out_s)[:, :-1])
