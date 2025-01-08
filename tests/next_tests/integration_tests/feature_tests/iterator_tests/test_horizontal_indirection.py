# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# (defun calc (p_vn input_on_cell)
#   (do_some_math
#     (deref
#         ((if (less (deref p_vn) 0)
#             (shift e2c 0)
#             (shift e2c 1)
#          )
#          input_on_cell
#         )
#     )
#   )
# )
import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import fundef, offset

from next_tests.integration_tests.cases import IDim
from next_tests.unit_tests.conftest import program_processor, run_processor


I = offset("I")


@fundef
def compute_shift(cond):
    return if_(deref(cond) < 0.0, shift(I, -1), shift(I, 1))


@fundef
def conditional_indirection(inp, cond):
    return deref(compute_shift(cond)(inp))


@pytest.mark.uses_applied_shifts
def test_simple_indirection(program_processor):
    program_processor, validate = program_processor

    shape = [8]
    inp = gtx.as_field([IDim], np.arange(0, shape[0] + 2), origin={IDim: 1})
    rng = np.random.default_rng()
    cond = gtx.as_field([IDim], rng.normal(size=shape))
    out = gtx.as_field([IDim], np.zeros(shape, dtype=inp.dtype))

    ref = np.zeros(shape, dtype=inp.dtype)
    for i in range(shape[0]):
        ref[i] = inp.asnumpy()[i + 1 - 1] if cond.asnumpy()[i] < 0.0 else inp.asnumpy()[i + 1 + 1]

    run_processor(
        conditional_indirection[cartesian_domain(named_range(IDim, 0, shape[0]))],
        program_processor,
        inp,
        cond,
        out=out,
        offset_provider={"I": IDim},
    )

    if validate:
        assert np.allclose(ref, out.asnumpy())


@fundef
def direct_indirection(inp, cond):
    return deref(shift(I, deref(cond))(inp))


@pytest.mark.uses_dynamic_offsets
def test_direct_offset_for_indirection(program_processor):
    program_processor, validate = program_processor

    shape = [4]
    inp = gtx.as_field([IDim], np.asarray(range(shape[0]), dtype=np.float64))
    cond = gtx.as_field([IDim], np.asarray([2, 1, -1, -2], dtype=np.int32))
    out = gtx.as_field([IDim], np.zeros(shape, dtype=np.float64))

    ref = np.zeros(shape)
    for i in range(shape[0]):
        ref[i] = inp.asnumpy()[i + cond.asnumpy()[i]]

    run_processor(
        direct_indirection[cartesian_domain(named_range(IDim, 0, shape[0]))],
        program_processor,
        inp,
        cond,
        out=out,
        offset_provider={"I": IDim},
    )

    if validate:
        assert np.allclose(ref, out.asnumpy())
