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
from gt4py.next.iterator.builtins import as_fieldop, cartesian_domain, deref, named_range
from gt4py.next.iterator.runtime import fendef, fundef, set_at
from gt4py.next.program_processors.formatters import type_check
from gt4py.next.program_processors.runners import dace, gtfn

from next_tests.unit_tests.conftest import program_processor, run_processor


I = gtx.Dimension("I")


@fundef
def copy_stencil(inp):
    return deref(inp)


@fendef
def copy_program(inp, out, size):
    set_at(
        as_fieldop(copy_stencil, cartesian_domain(named_range(I, 0, size)))(inp),
        cartesian_domain(named_range(I, 0, size)),
        out,
    )


def test_prog(program_processor):
    program_processor, validate = program_processor

    if program_processor in [
        gtfn.run_gtfn.executor,
        gtfn.run_gtfn_imperative.executor,
        gtfn.run_gtfn_with_temporaries.executor,
        dace.run_dace_cpu.executor,
        type_check.check_type_inference,
    ]:
        # TODO(havogt): Remove skip during refactoring to GTIR
        pytest.skip("Executor requires to start from fencil.")

    isize = 10
    inp = gtx.as_field([I], np.arange(0, isize, dtype=np.float64))
    out = gtx.as_field([I], np.zeros(shape=(isize,)))

    run_processor(copy_program, program_processor, inp, out, isize, offset_provider={})
    if validate:
        assert np.allclose(inp.asnumpy(), out.asnumpy())
