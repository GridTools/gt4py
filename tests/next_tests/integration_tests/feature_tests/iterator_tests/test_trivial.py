# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.iterator import transforms
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import set_at, fendef, fundef, offset

from next_tests.integration_tests.cases import IDim, JDim, KDim
from next_tests.unit_tests.conftest import program_processor, run_processor


I = offset("I")
J = offset("J")
K = offset("K")


@fundef
def foo(foo_inp):
    return deref(foo_inp)


@fundef
def bar(bar_inp):
    return deref(lift(foo)(bar_inp))


@fundef
def baz(baz_inp):
    return deref(lift(bar)(baz_inp))


def test_trivial(program_processor):
    program_processor, validate = program_processor

    rng = np.random.default_rng()
    inp = rng.uniform(size=(5, 7, 9))
    out = np.copy(inp)
    shape = (out.shape[0], out.shape[1])

    inp_s = gtx.as_field([IDim, JDim], inp[:, :, 0], origin={IDim: 0, JDim: 0})
    out_s = gtx.as_field([IDim, JDim], np.zeros_like(inp[:, :, 0]))

    run_processor(
        baz[cartesian_domain(named_range(IDim, 0, shape[0]), named_range(JDim, 0, shape[1]))],
        program_processor,
        inp_s,
        out=out_s,
        offset_provider={"I": IDim, "J": JDim},
    )

    if validate:
        assert np.allclose(out[:, :, 0], out_s.asnumpy())


@fundef
def stencil_shifted_arg_to_lift(inp):
    return deref(lift(deref)(shift(I, -1)(inp)))


def test_shifted_arg_to_lift(program_processor):
    program_processor, validate = program_processor

    rng = np.random.default_rng()
    inp = rng.uniform(size=(5, 7))
    out = np.zeros_like(inp)
    out[1:, :] = inp[:-1, :]
    shape = (out.shape[0], out.shape[1])

    inp_s = gtx.as_field([IDim, JDim], inp[:, :], origin={IDim: 0, JDim: 0})
    out_s = gtx.as_field([IDim, JDim], np.zeros_like(inp[:, :]))

    run_processor(
        stencil_shifted_arg_to_lift[
            cartesian_domain(named_range(IDim, 1, shape[0]), named_range(JDim, 0, shape[1]))
        ],
        program_processor,
        inp_s,
        out=out_s,
        offset_provider={"I": IDim, "J": JDim},
    )

    if validate:
        assert np.allclose(out, out_s.asnumpy())


@fendef
def fen_direct_deref(i_size, j_size, out, inp):
    domain = cartesian_domain(named_range(IDim, 0, i_size), named_range(JDim, 0, j_size))
    set_at(as_fieldop(deref, domain)(inp), domain, out)


def test_direct_deref(program_processor):
    program_processor, validate = program_processor

    rng = np.random.default_rng()
    inp = rng.uniform(size=(5, 7))
    out = np.copy(inp)

    inp_s = gtx.as_field([IDim, JDim], inp)
    out_s = gtx.as_field([IDim, JDim], np.zeros_like(inp))

    run_processor(
        fen_direct_deref,
        program_processor,
        *out.shape,
        out_s,
        inp_s,
        offset_provider=dict(),
    )

    if validate:
        assert np.allclose(out, out_s.asnumpy())


@fundef
def vertical_shift(inp):
    return deref(shift(K, 1)(inp))


def test_vertical_shift_unstructured(program_processor):
    program_processor, validate = program_processor

    k_size = 7

    rng = np.random.default_rng()
    inp = rng.uniform(size=(1, k_size))

    inp_s = gtx.as_field([IDim, KDim], inp)
    out_s = gtx.as_field([IDim, KDim], np.zeros_like(inp))

    run_processor(
        vertical_shift[
            unstructured_domain(named_range(IDim, 0, 1), named_range(KDim, 0, k_size - 1))
        ],
        program_processor,
        inp_s,
        out=out_s,
        offset_provider={"K": KDim},
    )

    if validate:
        assert np.allclose(inp_s[:, 1:].asnumpy(), out_s[:, :-1].asnumpy())
