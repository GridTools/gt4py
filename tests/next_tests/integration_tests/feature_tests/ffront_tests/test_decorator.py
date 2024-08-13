# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# TODO(dropd): Remove as soon as `gt4py.next.ffront.decorator` is type checked.
from gt4py import next as gtx
from gt4py.next.iterator import ir as itir

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


def test_program_itir_regression(cartesian_case):
    @gtx.field_operator(backend=None)
    def testee_op(a: cases.IField) -> cases.IField:
        return a

    @gtx.program(backend=None)
    def testee(a: cases.IField, out: cases.IField):
        testee_op(a, out=out)

    assert isinstance(testee.itir, itir.FencilDefinition)
    assert isinstance(testee.with_backend(cartesian_case.executor).itir, itir.FencilDefinition)
