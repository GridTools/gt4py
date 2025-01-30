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
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import set_at, fendef, fundef

from next_tests.unit_tests.conftest import program_processor, run_processor


IDim = gtx.Dimension("IDim")


@fundef
def stencil_conditional(inp):
    tmp = if_(eq(deref(inp), 0), make_tuple(1.0, 2.0), make_tuple(3.0, 4.0))
    return tuple_get(0, tmp) + tuple_get(1, tmp)


def test_conditional_w_tuple(program_processor):
    program_processor, validate = program_processor

    shape = [5]

    inp = gtx.as_field([IDim], np.random.randint(0, 2, shape, dtype=np.int32))
    out = gtx.as_field([IDim], np.zeros(shape))

    dom = {IDim: range(0, shape[0])}
    run_processor(stencil_conditional[dom], program_processor, inp, out=out, offset_provider={})
    if validate:
        assert np.all(out.asnumpy()[inp.asnumpy() == 0] == 3.0)
        assert np.all(out.asnumpy()[inp.asnumpy() == 1] == 7.0)
