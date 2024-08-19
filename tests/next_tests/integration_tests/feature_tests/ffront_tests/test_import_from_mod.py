# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from tests.next_tests.integration_tests.feature_tests.ffront_tests import dimensions_mod as dims
from tests.next_tests.integration_tests.cases import IDim, KDim
import gt4py.next as gtx
import numpy as np
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


def test_import_dims_module(cartesian_case):
    @gtx.field_operator
    def mod_op(f: cases.IKField) -> cases.IKField:
        return f

    @gtx.program
    def mod_prog(f: cases.IKField, out: cases.IKField):
        mod_op(f, out=out, domain={dims.IDim: (0, 8), dims.KDim: (0, 3)})

    f = cases.allocate(cartesian_case, mod_prog, "f")()
    out = cases.allocate(cartesian_case, mod_prog, "out")()
    expected = np.zeros_like(f.asnumpy())
    expected[0:8, 0:3] = f.asnumpy()[0:8, 0:3]

    cases.verify(cartesian_case, mod_prog, f, out=out, ref=expected)
