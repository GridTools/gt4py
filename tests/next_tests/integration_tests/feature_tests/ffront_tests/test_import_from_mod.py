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


def test_import_dims_module():
    @gtx.field_operator
    def mod_op(f: gtx.Field[[IDim, KDim], float]) -> gtx.Field[[IDim, KDim], float]:
        return f

    @gtx.program
    def mod_prog(f: gtx.Field[[IDim, KDim], float], out: gtx.Field[[IDim, KDim], float]):
        mod_op(f, out=out, domain={dims.IDim: (0, 8), dims.KDim: (0, 3)})

    f = gtx.as_field([IDim, KDim], np.ones([10, 10]))
    out = gtx.as_field([IDim, KDim], np.zeros([10, 10]))
    mod_prog(f, out, offset_provider={})
