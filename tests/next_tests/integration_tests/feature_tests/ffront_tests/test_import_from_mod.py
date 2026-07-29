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
from gt4py.next import broadcast, astype, common, int32, neighbor_sum

from next_tests import integration_tests
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    cartesian_case,
    unstructured_case,
    IDim,
    IHalfDim,
    KDim,
)

from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


def test_import_dims_module(cartesian_case):
    @gtx.field_operator
    def mod_op(f: cases.IField) -> cases.IKField:
        f_i_k = broadcast(f, (cases.IDim, cases.KDim))
        return f_i_k

    @gtx.program
    def mod_prog(f: cases.IField, isize: int32, ksize: int32, out: cases.IKField):
        mod_op(
            f,
            out=out,
            domain={
                integration_tests.cases.IDim: (
                    0,
                    isize,
                ),  # Nested import done on purpose, do not change
                cases.KDim: (0, ksize),
            },
        )

    f = cases.allocate(cartesian_case, mod_prog, "f")()
    out = cases.allocate(cartesian_case, mod_prog, "out")()
    expected = np.zeros_like(out.asnumpy())
    isize = cartesian_case.default_sizes[IDim] - 1
    ksize = cartesian_case.default_sizes[KDim] - 2
    expected[0:isize, 0:ksize] = np.reshape(
        np.repeat(f.asnumpy(), out.shape[1], axis=0), out.shape
    )[0:isize, 0:ksize]

    cases.verify(cartesian_case, mod_prog, f, isize, ksize, out=out, ref=expected)


@pytest.mark.uses_cartesian_shift
def test_import_dims_module_cartesian_shift(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IField) -> cases.IField:
        return a(cases.IDim + 1)

    size = cartesian_case.default_sizes[IDim]
    a = cases.allocate(cartesian_case, testee, "a", domain={IDim: (1, size + 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN, domain={IDim: (0, size)})()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[:], offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_import_dims_module_staggered_shift(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IField) -> cases.IHalfField:
        return a(cases.IHalfDim + 0.5)

    size = cartesian_case.default_sizes[IDim]
    a = cases.allocate(cartesian_case, testee, "a", sizes={IDim: size})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN, sizes={IHalfDim: size})()

    cases.verify(cartesian_case, testee, a, out=out, ref=a, offset_provider={})


@pytest.mark.uses_unstructured_shift
def test_import_offset_module_unstructured_shift(unstructured_case):
    @gtx.field_operator
    def testee(a: cases.EField) -> cases.VField:
        return neighbor_sum(a(cases.V2E), axis=cases.V2EDim)

    v2e_table = unstructured_case.offset_provider["V2E"].asnumpy()
    cases.verify_with_default_data(
        unstructured_case,
        testee,
        ref=lambda a: np.sum(a[v2e_table], axis=1, where=v2e_table != common._DEFAULT_SKIP_VALUE),
    )


# TODO: these set of features should be allowed as module imports in a later PR
def test_import_module_errors_future_allowed(cartesian_case):
    from ....artifacts.dummy_package import dummy_module

    with pytest.raises(gtx.errors.DSLError):

        @gtx.field_operator
        def field_op(f: cases.IField):
            f_i_k = gtx.broadcast(f, (cases.IDim, cases.KDim))
            return f_i_k

    with pytest.raises(gtx.errors.DSLError):

        @gtx.field_operator
        def field_op(f: cases.IField):
            type_ = gtx.int32
            return astype(f, type_)

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
