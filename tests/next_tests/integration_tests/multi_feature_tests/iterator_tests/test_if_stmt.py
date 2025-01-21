# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
from gt4py.next.iterator.builtins import cartesian_domain, deref, as_fieldop, named_range
from gt4py.next.iterator.runtime import set_at, if_stmt, fendef, fundef, offset
from gt4py.next.program_processors.runners import gtfn

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
