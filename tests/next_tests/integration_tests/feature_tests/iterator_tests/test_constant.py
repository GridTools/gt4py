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

import gt4py.next as gtx
from gt4py.next.iterator.builtins import *
from gt4py.next.iterator.runtime import fundef
from gt4py.next.program_processors.runners import roundtrip

from next_tests.integration_tests.cases import IDim


def test_constant():
    @fundef
    def add_constant(inp):
        def constant_stencil():  # this is traced as a lambda, TODO directly feed iterator IR nodes
            return 1

        return deref(inp) + deref(lift(constant_stencil)())

    inp = gtx.as_field([IDim], np.asarray([0, 42], dtype=np.int32))
    res = gtx.as_field([IDim], np.zeros_like(inp.asnumpy()))

    add_constant[{IDim: range(2)}](inp, out=res, offset_provider={}, backend=roundtrip.executor)

    assert np.allclose(res.asnumpy(), np.asarray([1, 43]))
