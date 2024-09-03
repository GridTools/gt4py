# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np

import gt4py.next as gtx
from gt4py.next import broadcast

from next_tests import integration_tests
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)

from next_tests.dummy_package import dummy_module


def test_import_dims_module(cartesian_case):
    @gtx.field_operator
    def mod_op(f: cases.IField) -> cases.IKField:
        f_i_k = broadcast(f, (cases.IDim, cases.KDim))
        return f_i_k

    @gtx.program
    def mod_prog(f: cases.IField, out: cases.IKField):
        mod_op(
            f,
            out=out,
            domain={
                integration_tests.cases.IDim: (
                    0,
                    8,
                ),  # Nested import done on purpose, do not change
                cases.KDim: (0, 3),
            },
        )

    f = cases.allocate(cartesian_case, mod_prog, "f")()
    out = cases.allocate(cartesian_case, mod_prog, "out")()
    expected = np.zeros_like(out.asnumpy())
    expected[0:8, 0:3] = np.reshape(np.repeat(f.asnumpy(), out.shape[1], axis=0), out.shape)[
        0:8, 0:3
    ]

    cases.verify(cartesian_case, mod_prog, f, out=out, ref=expected)


# TODO: these set of features should be allowed as module imports in a later PR
def test_import_module_errors_future_allowed(cartesian_case):
    with pytest.raises(gtx.errors.DSLError):

        @gtx.field_operator
        def field_op(f: cases.IField):
            f_i_k = gtx.broadcast(f, (cases.IDim, cases.KDim))
            return f_i_k

    with pytest.raises(ValueError):

        @gtx.field_operator
        def field_op(f: cases.IField):
            type_ = gtx.int32
            return f

    with pytest.raises(gtx.errors.DSLError):

        @gtx.field_operator
        def field_op(f: cases.IField):
            f_new = dummy_module.field_op_sample(f)
            return f_new

    with pytest.raises(gtx.errors.DSLError):

        @gtx.field_operator
        def field_op(f: cases.IField):
            return f

        @gtx.program
        def field_op(f: cases.IField):
            dummy_module.field_op_sample(f, out=f, offset_provider={})
