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
from gt4py.next import field_utils
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import set_at, fendef, fundef, offset

from next_tests.integration_tests.cases import IDim, KDim
from next_tests.unit_tests.conftest import program_processor, run_processor


I = offset("I")
K = offset("K")


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
        # (stencil, reference_function, inp_fun (None=default)
        (add_scalar, lambda inp: np.asarray(inp) + 1.0, None),
        (if_scalar_cond, lambda inp: np.asarray(inp), None),
        (if_scalar_return, lambda inp: np.ones_like(inp), None),
        (
            shift_stencil,
            lambda inp: np.asarray(inp)[1:, 1:],
            lambda shape: gtx.as_field(
                [IDim, KDim], np.fromfunction(lambda i, k: i * 10 + k, [shape[0] + 1, shape[1] + 1])
            ),
        ),
        (
            shift_stencil,
            lambda inp: np.asarray(inp)[1:, 2:],
            lambda shape: gtx.as_field(
                [IDim, KDim],
                np.fromfunction(lambda i, k: i * 10 + k, [shape[0] + 1, shape[1] + 2]),
                origin={IDim: 0, KDim: 1},
            ),
        ),
    ],
    ids=lambda p: f"{p[0].__name__}",
)
def basic_stencils(request):
    return request.param


@pytest.mark.uses_origin
def test_basic_column_stencils(program_processor, basic_stencils):
    program_processor, validate = program_processor
    stencil, ref_fun, inp_fun = basic_stencils

    shape = [5, 7]
    inp = (
        gtx.as_field([IDim, KDim], np.fromfunction(lambda i, k: i * 10 + k, shape))
        if inp_fun is None
        else inp_fun(shape)
    )
    out = gtx.as_field([IDim, KDim], np.zeros(shape))

    ref = ref_fun(inp.asnumpy())

    run_processor(
        stencil[{IDim: range(0, shape[0]), KDim: range(0, shape[1])}],
        program_processor,
        inp,
        out=out,
        offset_provider={"I": IDim, "K": KDim},
        column_axis=KDim,
    )

    if validate:
        assert np.allclose(ref, out.asnumpy())


@fundef
def k_level_condition_lower(k_idx, k_level):
    return if_(deref(k_idx) > deref(k_level), deref(shift(K, -1)(k_idx)), 0)


@fundef
def k_level_condition_upper(k_idx, k_level):
    return if_(deref(k_idx) < deref(k_level), deref(shift(K, +1)(k_idx)), 0)


@fundef
def k_level_condition_upper_tuple(k_idx, k_level):
    shifted_val = deref(shift(K, +1)(k_idx))
    return if_(
        tuple_get(0, deref(k_idx)) < deref(k_level),
        tuple_get(0, shifted_val) + tuple_get(1, shifted_val),
        0,
    )


@pytest.mark.parametrize(
    "fun, k_level, inp_function, ref_function",
    [
        (
            k_level_condition_lower,
            lambda inp: 0,
            lambda k_size: gtx.as_field([KDim], np.arange(k_size, dtype=np.int32)),
            lambda inp: np.concatenate([[0], inp[:-1]]),
        ),
        (
            k_level_condition_upper,
            lambda inp: inp.shape[0] - 1,
            lambda k_size: gtx.as_field([KDim], np.arange(k_size, dtype=np.int32)),
            lambda inp: np.concatenate([inp[1:], [0]]),
        ),
        (
            k_level_condition_upper_tuple,
            lambda inp: inp[0].shape[0] - 1,
            lambda k_size: (
                gtx.as_field([KDim], np.arange(k_size, dtype=np.int32)),
                gtx.as_field([KDim], np.arange(k_size, dtype=np.int32)),
            ),
            lambda inp: np.concatenate([(inp[0][1:] + inp[1][1:]), [0]]),
        ),
    ],
)
@pytest.mark.uses_tuple_args
def test_k_level_condition(program_processor, fun, k_level, inp_function, ref_function):
    program_processor, validate = program_processor

    k_size = 5
    inp = inp_function(k_size)
    ref = ref_function(field_utils.asnumpy(inp))

    out = gtx.as_field([KDim], np.zeros((5,), dtype=np.int32))

    run_processor(
        fun[{KDim: range(0, k_size)}],
        program_processor,
        inp,
        k_level(inp),
        out=out,
        offset_provider={"K": KDim},
        column_axis=KDim,
    )

    if validate:
        np.allclose(ref, out.asnumpy())


@fundef
def ksum(state, inp):
    return state + deref(inp)


@fendef(column_axis=KDim)
def ksum_fencil(i_size, k_start, k_end, inp, out):
    domain = cartesian_domain(named_range(IDim, 0, i_size), named_range(KDim, k_start, k_end))
    set_at(as_fieldop(scan(ksum, True, 0.0), domain)(inp), domain, out)


@pytest.mark.parametrize(
    "kstart, reference",
    [(0, np.asarray([[0, 1, 3, 6, 10, 15, 21]])), (2, np.asarray([[0, 0, 2, 5, 9, 14, 20]]))],
)
def test_ksum_scan(program_processor, kstart, reference):
    program_processor, validate = program_processor
    shape = [1, 7]
    inp = gtx.as_field([IDim, KDim], np.array(np.broadcast_to(np.arange(0.0, 7.0), shape)))
    out = gtx.as_field([IDim, KDim], np.zeros(shape, dtype=inp.dtype))

    run_processor(
        ksum_fencil,
        program_processor,
        shape[0],
        kstart,
        shape[1],
        inp,
        out,
        offset_provider={"I": IDim, "K": KDim},
    )

    if validate:
        assert np.allclose(reference, out.asnumpy())


@fendef(column_axis=KDim)
def ksum_back_fencil(i_size, k_size, inp, out):
    domain = cartesian_domain(named_range(IDim, 0, i_size), named_range(KDim, 0, k_size))
    set_at(as_fieldop(scan(ksum, False, 0.0), domain)(inp), domain, out)


def test_ksum_back_scan(program_processor):
    program_processor, validate = program_processor
    shape = [1, 7]
    inp = gtx.as_field([IDim, KDim], np.array(np.broadcast_to(np.arange(0.0, 7.0), shape)))
    out = gtx.as_field([IDim, KDim], np.zeros(shape, dtype=inp.dtype))

    ref = np.asarray([[21, 21, 20, 18, 15, 11, 6]])

    run_processor(
        ksum_back_fencil,
        program_processor,
        shape[0],
        shape[1],
        inp,
        out,
        offset_provider={"I": IDim, "K": KDim},
    )

    if validate:
        assert np.allclose(ref, out.asnumpy())


@fundef
def kdoublesum(state, inp0, inp1):
    return make_tuple(tuple_get(0, state) + deref(inp0), tuple_get(1, state) + deref(inp1))


@fendef(column_axis=KDim)
def kdoublesum_fencil(i_size, k_start, k_end, inp0, inp1, out):
    domain = cartesian_domain(named_range(IDim, 0, i_size), named_range(KDim, k_start, k_end))
    set_at(as_fieldop(scan(kdoublesum, True, make_tuple(0.0, 0)), domain)(inp0, inp1), domain, out)


@pytest.mark.parametrize(
    "kstart, reference",
    [
        (
            0,
            (
                np.asarray([0, 1, 3, 6, 10, 15, 21], dtype=np.float64),
                np.asarray([0, 1, 3, 6, 10, 15, 21], dtype=np.int32),
            ),
        ),
        (
            2,
            (
                np.asarray([0, 0, 2, 5, 9, 14, 20], dtype=np.float64),
                np.asarray([0, 0, 2, 5, 9, 14, 20], dtype=np.int32),
            ),
        ),
    ],
)
def test_kdoublesum_scan(program_processor, kstart, reference):
    program_processor, validate = program_processor
    pytest.xfail("structured dtype input/output currently unsupported")
    shape = [1, 7]
    inp0 = gtx.as_field([IDim, KDim], np.asarray([list(range(7))], dtype=np.float64))
    inp1 = gtx.as_field([IDim, KDim], np.asarray([list(range(7))], dtype=np.int32))
    out = (
        gtx.as_field([IDim, KDim], np.zeros(shape, dtype=np.float64)),
        gtx.as_field([IDim, KDim], np.zeros(shape, dtype=np.float32)),
    )

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
    )

    if validate:
        for ref, o in zip(reference, out):
            assert np.allclose(ref, o)


@fundef
def sum_shifted(inp0, inp1):
    return deref(inp0) + deref(shift(K, 1)(inp1))


@fendef(column_axis=KDim)
def sum_shifted_fencil(out, inp0, inp1, k_size):
    domain = cartesian_domain(named_range(KDim, 1, k_size))
    set_at(as_fieldop(sum_shifted, domain)(inp0, inp1), domain, out)


def test_different_vertical_sizes(program_processor):
    program_processor, validate = program_processor

    k_size = 10
    inp0 = gtx.as_field([KDim], np.arange(0, k_size))
    inp1 = gtx.as_field([KDim], np.arange(0, k_size + 1))
    out = gtx.as_field([KDim], np.zeros(k_size, dtype=inp0.dtype))
    ref = inp0.ndarray + inp1.ndarray[1:]

    run_processor(
        sum_shifted_fencil, program_processor, out, inp0, inp1, k_size, offset_provider={"K": KDim}
    )

    if validate:
        assert np.allclose(ref[1:], out.asnumpy()[1:])


@fundef
def sum(inp0, inp1):
    return deref(inp0) + deref(shift(K, -1)(inp1))


@fendef(column_axis=KDim)
def sum_fencil(out, inp0, inp1, k_size):
    domain = cartesian_domain(named_range(KDim, 0, k_size))
    set_at(as_fieldop(sum, domain)(inp0, inp1), domain, out)


@pytest.mark.uses_origin
def test_different_vertical_sizes_with_origin(program_processor):
    program_processor, validate = program_processor

    k_size = 10
    inp0 = gtx.as_field([KDim], np.arange(0, k_size))
    inp1 = gtx.as_field([KDim], np.arange(0, k_size + 1), origin={KDim: 1})
    out = gtx.as_field([KDim], np.zeros(k_size, dtype=np.int64))
    ref = inp0.asnumpy() + inp1.asnumpy()[:-1]

    run_processor(
        sum_fencil, program_processor, out, inp0, inp1, k_size, offset_provider={"K": KDim}
    )

    if validate:
        assert np.allclose(ref, out.asnumpy())


# TODO(havogt) test tuple_get builtin on a Column
