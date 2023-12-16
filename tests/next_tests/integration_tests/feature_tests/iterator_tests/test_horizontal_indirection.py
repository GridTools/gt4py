# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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
from gt4py.next.program_processors.formatters import type_check
from gt4py.next.program_processors.formatters.gtfn import format_cpp as gtfn_format_sourcecode

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

    if program_processor in [
        type_check.check_type_inference,
        gtfn_format_sourcecode,
    ]:
        pytest.xfail(
            "We only support applied shifts in type_inference."
        )  # TODO fix test or generalize itir?

    shape = [8]
    inp = gtx.as_field([IDim], np.arange(0, shape[0] + 2), origin={IDim: 1})
    rng = np.random.default_rng()
    cond = gtx.as_field([IDim], rng.normal(size=shape))
    out = gtx.as_field([IDim], np.zeros(shape, dtype=inp.dtype))

    ref = np.zeros(shape, dtype=inp.dtype)
    for i in range(shape[0]):
        ref[i] = inp.ndarray[i + 1 - 1] if cond[i] < 0.0 else inp.ndarray[i + 1 + 1]

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
        ref[i] = inp[i + cond[i]]

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
