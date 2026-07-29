# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
from next_tests.integration_tests import cases

dummy_int = 42

dummy_field = gtx.as_field([cases.IDim], np.ones((10,), dtype=gtx.int32))


@gtx.field_operator
def field_op_sample(a: cases.IKField) -> cases.IKField:
    return a
