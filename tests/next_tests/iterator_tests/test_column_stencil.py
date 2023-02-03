# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from gt4py.next.common import Dimension
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.iterator.runtime import closure, fendef, fundef, offset
from gt4py.next.program_processors.formatters.gtfn import (
    format_sourcecode as gtfn_format_sourcecode,
)
from gt4py.next.program_processors.runners.gtfn_cpu import run_gtfn, run_gtfn_imperative

from .conftest import run_processor


I = offset("I")
K = offset("K")

KDim = Dimension("KDim")
IDim = Dimension("IDim")


@fundef
def add_scalar(inp):
    return deref(inp) + 1.0


@fundef
def if_scalar_cond(inp):
    return if_(True, deref(inp), 1.0)


@fundef
def if_scalar_return(inp):
    return if_(deref(inp) < 0.0, deref(inp), 1.0)


@fundef
def shift_stencil(inp):
    return deref(shift(K, 1, I, 1)(inp))


@pytest.fixture(
    params=[
        # (stencil, reference_function, inp_fun (None=default), (skip_backend_fun, msg))
        (add_scalar, lambda inp: np.asarray(inp) + 1.0, None, None),
        (if_scalar_cond, lambda inp: np.asarray(inp), None, None),
        (if_scalar_return, lambda inp: np.ones_like(inp), None, None),
        (
            shift_stencil,
            lambda inp: np.asarray(inp)[1:, 1:],
            lambda shape: np_as_located_field(IDim, KDim)(
                np.fromfunction(lambda i, k: i * 10 + k, [shape[0] + 1, shape[1] + 1])
            ),
            None,
        ),
        (
            shift_stencil,
            lambda inp: np.asarray(inp)[1:, 2:],
            lambda shape: np_as_located_field(IDim, KDim, origin={IDim: 0, KDim: 1})(
                np.fromfunction(lambda i, k: i * 10 + k, [shape[0] + 1, shape[1] + 2])
            ),
            (
                lambda backend: backend == run_gtfn or backend == run_gtfn_imperative,
                "origin not supported in gtfn",
            ),
        ),
    ],
    ids=lambda p: f"{p[0].__name__}",
)
def basic_stencils(request):
    return request.param


def test_basic_column_stencils(program_processor, lift_mode, basic_stencils):
    program_processor, validate = program_processor
    stencil, ref_fun, inp_fun, skip_backend = basic_stencils
    if skip_backend is not None:
        skip_backend_fun, msg = skip_backend
        if skip_backend_fun(program_processor):
            pytest.xfail(msg)

    shape = [5, 7]
    inp = (
        np_as_located_field(IDim, KDim)(np.fromfunction(lambda i, k: i * 10 + k, shape))
        if inp_fun is None
        else inp_fun(shape)
    )
    out = np_as_located_field(IDim, KDim)(np.zeros(shape))

    ref = ref_fun(inp)

    run_processor(
        stencil[{IDim: range(0, shape[0]), KDim: range(0, shape[1])}],
        program_processor,
        inp,
        out=out,
        offset_provider={"I": IDim, "K": KDim},
        column_axis=KDim,
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(ref, out)


@fundef
def sum_scanpass(state, inp):
    return state + deref(inp)


@fundef
def ksum(inp):
    return scan(sum_scanpass, True, 0.0)(inp)


@fendef(column_axis=KDim)
def ksum_fencil(i_size, k_start, k_end, inp, out):
    closure(
        cartesian_domain(named_range(IDim, 0, i_size), named_range(KDim, k_start, k_end)),
        ksum,
        out,
        [inp],
    )


@pytest.mark.parametrize(
    "kstart, reference",
    [
        (0, np.asarray([[0, 1, 3, 6, 10, 15, 21]])),
        (2, np.asarray([[0, 0, 2, 5, 9, 14, 20]])),
    ],
)
def test_ksum_scan(program_processor, lift_mode, kstart, reference):
    program_processor, validate = program_processor
    shape = [1, 7]
    inp = np_as_located_field(IDim, KDim)(np.asarray([list(range(7))]))
    out = np_as_located_field(IDim, KDim)(np.zeros(shape))

    run_processor(
        ksum_fencil,
        program_processor,
        shape[0],
        kstart,
        shape[1],
        inp,
        out,
        offset_provider={"I": IDim, "K": KDim},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(reference, np.asarray(out))


@fundef
def ksum_back(inp):
    return scan(sum_scanpass, False, 0.0)(inp)


@fendef(column_axis=KDim)
def ksum_back_fencil(i_size, k_size, inp, out):
    closure(
        cartesian_domain(named_range(IDim, 0, i_size), named_range(KDim, 0, k_size)),
        ksum_back,
        out,
        [inp],
    )


def test_ksum_back_scan(program_processor, lift_mode):
    program_processor, validate = program_processor
    shape = [1, 7]
    inp = np_as_located_field(IDim, KDim)(np.asarray([list(range(7))]))
    out = np_as_located_field(IDim, KDim)(np.zeros(shape))

    ref = np.asarray([[21, 21, 20, 18, 15, 11, 6]])

    run_processor(
        ksum_back_fencil,
        program_processor,
        shape[0],
        shape[1],
        inp,
        out,
        offset_provider={"I": IDim, "K": KDim},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(ref, np.asarray(out))


@fundef
def doublesum_scanpass(state, inp0, inp1):
    return make_tuple(tuple_get(0, state) + deref(inp0), tuple_get(1, state) + deref(inp1))


@fundef
def kdoublesum(inp0, inp1):
    return scan(doublesum_scanpass, True, make_tuple(0.0, 0))(inp0, inp1)


@fendef(column_axis=KDim)
def kdoublesum_fencil(i_size, k_start, k_end, inp0, inp1, out):
    closure(
        cartesian_domain(named_range(IDim, 0, i_size), named_range(KDim, k_start, k_end)),
        kdoublesum,
        out,
        [inp0, inp1],
    )


@pytest.mark.parametrize(
    "kstart, reference",
    [
        (
            0,
            np.asarray(
                [[(0, 0), (1, 1), (3, 3), (6, 6), (10, 10), (15, 15), (21, 21)]],
                dtype=np.dtype([("foo", np.float64), ("bar", np.int32)]),
            ),
        ),
        (
            2,
            np.asarray(
                [[(0, 0), (0, 0), (2, 2), (5, 5), (9, 9), (14, 14), (20, 20)]],
                dtype=np.dtype([("foo", np.float64), ("bar", np.int32)]),
            ),
        ),
    ],
)
def test_kdoublesum_scan(program_processor, lift_mode, kstart, reference):
    program_processor, validate = program_processor
    if (
        program_processor == run_gtfn
        or program_processor == run_gtfn_imperative
        or program_processor == gtfn_format_sourcecode
    ):
        pytest.xfail("structured dtype input/output currently unsupported")
    shape = [1, 7]
    inp0 = np_as_located_field(IDim, KDim)(np.asarray([list(range(7))], dtype=np.float64))
    inp1 = np_as_located_field(IDim, KDim)(np.asarray([list(range(7))], dtype=np.int32))
    out = np_as_located_field(IDim, KDim)(np.zeros(shape, dtype=reference.dtype))

    run_processor(
        kdoublesum_fencil,
        program_processor,
        shape[0],
        kstart,
        shape[1],
        inp0,
        inp1,
        out,
        offset_provider={"I": IDim, "K": KDim},
        lift_mode=lift_mode,
    )

    if validate:
        for n in reference.dtype.names:
            assert np.allclose(reference[n], np.asarray(out)[n])


@fundef
def sum_shifted(inp0, inp1):
    return deref(inp0) + deref(shift(K, 1)(inp1))


@fendef(column_axis=KDim)
def sum_shifted_fencil(out, inp0, inp1, k_size):
    closure(
        cartesian_domain(named_range(KDim, 1, k_size)),
        sum_shifted,
        out,
        [inp0, inp1],
    )


def test_different_vertical_sizes(program_processor):
    program_processor, validate = program_processor

    k_size = 10
    inp0 = np_as_located_field(KDim)(np.asarray(list(range(k_size))))
    inp1 = np_as_located_field(KDim)(np.asarray(list(range(k_size + 1))))
    out = np_as_located_field(KDim)(np.zeros(k_size))
    ref = inp0 + inp1[1:]

    run_processor(
        sum_shifted_fencil,
        program_processor,
        out,
        inp0,
        inp1,
        k_size,
        offset_provider={"K": KDim},
    )

    if validate:
        assert np.allclose(ref[1:], out[1:])


@fundef
def sum(inp0, inp1):
    return deref(inp0) + deref(shift(K, -1)(inp1))


@fendef(column_axis=KDim)
def sum_fencil(out, inp0, inp1, k_size):
    closure(
        cartesian_domain(named_range(KDim, 0, k_size)),
        sum,
        out,
        [inp0, inp1],
    )


def test_different_vertical_sizes_with_origin(program_processor):
    program_processor, validate = program_processor
    if program_processor in [run_gtfn, run_gtfn_imperative]:
        pytest.xfail("origin not supported in gtfn")

    k_size = 10
    inp0 = np_as_located_field(KDim)(np.asarray(list(range(k_size))))
    inp1 = np_as_located_field(KDim, origin={KDim: 1})(np.asarray(list(range(k_size + 1))))
    out = np_as_located_field(KDim)(np.zeros(k_size))
    ref = inp0 + np.asarray(inp1)[:-1]

    run_processor(
        sum_fencil,
        program_processor,
        out,
        inp0,
        inp1,
        k_size,
        offset_provider={"K": KDim},
    )

    if validate:
        assert np.allclose(ref, out)


# TODO(havogt) test tuple_get builtin on a Column
