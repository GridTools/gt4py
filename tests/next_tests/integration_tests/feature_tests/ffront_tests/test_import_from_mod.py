# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import gt4py.next as gtx
import numpy as np
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import Ioff
from next_tests.integration_tests.cases import cartesian_case, unstructured_case

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)
from gt4py.next.ffront import experimental as exp
from gt4py.next import common


def test_import_dims_module(cartesian_case):
    @gtx.field_operator
    def mod_op(f: cases.IKField) -> cases.IKField:
        return f(cases.Ioff[1])

    @gtx.program
    def mod_prog(f: cases.IKField, out: cases.IKField):
        mod_op(f, out=out, domain={cases.IDim: (0, 8), cases.KDim: (0, 3)})

    f = cases.allocate(cartesian_case, mod_prog, "f")()
    out = cases.allocate(cartesian_case, mod_prog, "out")()
    expected = np.zeros_like(f.asnumpy())
    expected[0:8, 0:3] = f.asnumpy()[1:9, 0:3]

    cases.verify(cartesian_case, mod_prog, f, out=out, ref=expected)


def test_import_as_offset_module(cartesian_case):
    @gtx.field_operator
    def mod_builtin(f: cases.IKField, offset_field: cases.IKField) -> cases.IKField:
        field_offset = f(exp.as_offset(Ioff, offset_field))
        return field_offset

    @gtx.program
    def mod_prog(f: cases.IKField, offset_field: cases.IKField, out: cases.IKField):
        mod_builtin(f, offset_field, out=out)

    f = cases.allocate(cartesian_case, mod_prog, "f")()
    offset_field = cases.allocate(cartesian_case, mod_prog, "offset_field").strategy(
        cases.ConstInitializer(2)
    )()
    out = cases.allocate(cartesian_case, mod_prog, "out")()
    expected = np.zeros_like(f.asnumpy())
    expected[0:8, :] = f.asnumpy()[2:10, :]

    cases.verify(cartesian_case, mod_prog, f, offset_field, out=out[0:8, :], ref=expected[0:8, :])


@pytest.mark.uses_unstructured_shift
def test_minover_execution(unstructured_case):
    @gtx.field_operator
    def minover(edge_f: cases.EField) -> cases.VField:
        out = gtx.min_over(edge_f(cases.V2E), axis=cases.V2EDim)
        return out

    v2e_table = unstructured_case.offset_provider["V2E"].table
    cases.verify_with_default_data(
        unstructured_case,
        minover,
        ref=lambda edge_f: np.min(
            edge_f[v2e_table],
            axis=1,
            initial=np.max(edge_f),
            where=v2e_table != common._DEFAULT_SKIP_VALUE,
        ),
    )
