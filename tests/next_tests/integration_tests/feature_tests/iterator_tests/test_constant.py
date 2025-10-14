# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import math

import numpy as np

import gt4py.next as gtx
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import fundef
from gt4py.next.program_processors.runners import roundtrip
from next_tests.unit_tests.conftest import program_processor, run_processor

from next_tests.integration_tests.cases import IDim


def test_constant(program_processor):
    program_processor, validate = program_processor

    @fundef
    def add_constant(inp):
        def constant_stencil():  # this is traced as a lambda, TODO: directly feed iterator IR nodes
            return 1

        return deref(inp) + deref(lift(constant_stencil)())

    inp = gtx.as_field([IDim], np.asarray([0, 42], dtype=np.int32))
    res = gtx.as_field([IDim], np.zeros_like(inp.asnumpy()))

    run_processor(
        add_constant[{IDim: range(2)}],
        program_processor,
        inp,
        out=res,
        offset_provider={},
    )

    if validate:
        assert np.allclose(res.asnumpy(), np.asarray([1, 43]))
