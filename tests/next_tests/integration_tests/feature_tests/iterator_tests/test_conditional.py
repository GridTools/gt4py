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

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import closure, fendef, fundef
from gt4py.next.program_processors.runners.dace_iterator import run_dace_iterator

from next_tests.unit_tests.conftest import program_processor, run_processor


IDim = gtx.Dimension("IDim")


@fundef
def test_conditional(inp):
    tmp = if_(eq(deref(inp), 0), make_tuple(1.0, 2.0), make_tuple(3.0, 4.0))
    return tuple_get(0, tmp) + tuple_get(1, tmp)


def test_conditional_w_tuple(program_processor):
    program_processor, validate = program_processor
    if program_processor == run_dace_iterator:
        pytest.xfail("Not supported in DaCe backend: tuple returns")

    shape = [5]

    inp = gtx.np_as_located_field(IDim)(np.random.randint(0, 2, shape, dtype=np.int32))
    out = gtx.np_as_located_field(IDim)(np.zeros(shape))

    dom = {
        IDim: range(0, shape[0]),
    }
    run_processor(
        test_conditional[dom],
        program_processor,
        inp,
        out=out,
        offset_provider={},
    )
    if validate:
        assert np.all(out[np.asarray(inp) == 0] == 3.0)
        assert np.all(out[np.asarray(inp) == 1] == 7.0)
