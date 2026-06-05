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
from gt4py.next.iterator.builtins import as_fieldop, cartesian_domain, deref, named_range
from gt4py.next.iterator.runtime import fendef, fundef, if_stmt, offset, set_at

from next_tests.unit_tests.conftest import program_processor_no_transforms, run_processor


i = offset("i")


@fundef
def multiply(alpha, inp):
    return deref(alpha) * deref(inp)


IDim = gtx.Dimension("IDim")


@pytest.mark.uses_ir_if_stmts
@pytest.mark.parametrize("cond", [True, False])
def test_if_stmt(program_processor_no_transforms, cond):
    program_processor, validate = program_processor_no_transforms
    size = 10

    @fendef(offset_provider={"i": IDim})
    def fencil(cond1, inp, out):
        domain = cartesian_domain(named_range(IDim, 0, size))
        if_stmt(
            cond1,
            lambda: set_at(
                as_fieldop(multiply, domain)(1.0, inp),
                domain,
                out,
            ),
            lambda: set_at(
                as_fieldop(multiply, domain)(2.0, inp),
                domain,
                out,
            ),
        )

    rng = np.random.default_rng()
    inp = gtx.as_field([IDim], rng.normal(size=size))
    out = gtx.as_field([IDim], np.zeros(size))
    ref = inp if cond else 2.0 * inp

    run_processor(fencil, program_processor, cond, inp, out)

    if validate:
        assert np.allclose(out.asnumpy(), ref.asnumpy())
