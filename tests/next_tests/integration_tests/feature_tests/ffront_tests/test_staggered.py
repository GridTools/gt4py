# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.ffront.experimental import as_offset

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    E2V,
    Edge,
    IDim,
    IHalfDim,
    JDim,
    KDim,
    KHalfDim,
    Vertex,
    cartesian_case,
    unstructured_case,
    unstructured_case_3d,
)
from next_tests.integration_tests.cases_utils import Koff, exec_alloc_descriptor, mesh_descriptor


@pytest.mark.uses_cartesian_shift
def test_copy_half_field(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IHalfField) -> cases.IHalfField:
        field_tuple = (a, a)
        field_0 = field_tuple[0]
        field_1 = field_tuple[1]
        return field_0

    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a: a, offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_shift_plus(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IField) -> cases.IField:
        return a(IDim + 1)  # always pass an I-index to an IField

    size = cartesian_case.default_sizes[IDim]
    a = cases.allocate(cartesian_case, testee, "a", domain={IDim: (1, size + 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN, domain={IDim: (0, size)})()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[:], offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_half_shift_plus(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IField) -> cases.IHalfField:
        return a(IHalfDim + 0.5)  # always pass an I-index to an IField

    size = cartesian_case.default_sizes[IDim]
    a = cases.allocate(cartesian_case, testee, "a", sizes={IDim: size})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN, sizes={IHalfDim: size})()

    cases.verify(cartesian_case, testee, a, out=out, ref=a, offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_half_shift_back(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IHalfField) -> cases.IHalfField:
        return a(IDim + 0.5)(IHalfDim - 0.5)  # always pass an I-index to an IField

    a = cases.allocate(cartesian_case, testee, "a")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a, offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_half_shift_plus1(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IHalfField) -> cases.IHalfField:
        return a(IHalfDim + 1)  # always pass an IHalf-index to an IHalfField

    size = cartesian_case.default_sizes[IDim]
    a = cases.allocate(cartesian_case, testee, "a", domain={IHalfDim: (1, size + 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN, domain={IHalfDim: (0, size)})()

    cases.verify(cartesian_case, testee, a, out=out[:], ref=a[:], offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_half_shift_minus(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IField) -> cases.IHalfField:
        return a(IHalfDim - 0.5)  # always pass an I-index to an IField

    size = cartesian_case.default_sizes[IDim]
    a = cases.allocate(cartesian_case, testee, "a", domain={IDim: (-1, size - 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN, domain={IHalfDim: (0, size)})()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[:], offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_half_shift_half2center(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IHalfField) -> cases.IField:
        return 2 * a(IDim + 0.5)  # always pass an IHalf-index to an IHalfField

    size = cartesian_case.default_sizes[IDim]
    a = cases.allocate(cartesian_case, testee, "a", domain={IHalfDim: (1, size + 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN, sizes={IDim: size})()

    cases.verify(cartesian_case, testee, a, out=out, ref=2 * a[:], offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_half_shift_vertical(cartesian_case):
    # vertical (K) staggering: identical mechanism, different dimension kind.
    @gtx.field_operator
    def testee(a: cases.KField) -> gtx.Field[[KHalfDim], np.int32]:
        return a(KHalfDim + 0.5)

    size = cartesian_case.default_sizes[KDim]
    a = cases.allocate(cartesian_case, testee, "a", sizes={KDim: size})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN, sizes={KHalfDim: size})()

    cases.verify(cartesian_case, testee, a, out=out, ref=a, offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_half_shift_multi_dim(cartesian_case):
    # staggering one axis of a multi-dimensional field leaves the other axis untouched.
    @gtx.field_operator
    def testee(a: cases.IJField) -> gtx.Field[[IHalfDim, JDim], np.int32]:
        return a(IHalfDim + 0.5)

    isize = cartesian_case.default_sizes[IDim]
    jsize = cartesian_case.default_sizes[JDim]
    a = cases.allocate(cartesian_case, testee, "a", sizes={IDim: isize, JDim: jsize})()
    out = cases.allocate(
        cartesian_case, testee, cases.RETURN, sizes={IHalfDim: isize, JDim: jsize}
    )()

    cases.verify(cartesian_case, testee, a, out=out, ref=a, offset_provider={})


@pytest.mark.uses_cartesian_shift
@pytest.mark.uses_dynamic_offsets
def test_cartesian_half_shift_as_offset(cartesian_case):
    @gtx.field_operator
    def testee(
        a: gtx.Field[[IDim, KHalfDim], np.int32], offset_field: cases.IKField
    ) -> cases.IKField:
        return a(KDim - 0.5)(as_offset(Koff, offset_field))

    ksize = cartesian_case.default_sizes[KDim]
    a = cases.allocate(cartesian_case, testee, "a", sizes={KHalfDim: ksize + 1})()
    offset_field = cases.allocate(
        cartesian_case, testee, "offset_field", strategy=cases.ConstInitializer(1)
    )()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(
        cartesian_case, testee, a, offset_field, out=out, ref=a.asnumpy()[:, 1:], offset_provider={}
    )


@pytest.mark.uses_cartesian_shift
@pytest.mark.uses_dynamic_offsets
def test_cartesian_half_shift_as_offset_of_chained_ops(cartesian_case):
    """The staggered shift and the dynamic offset are separated by a further operation."""

    @gtx.field_operator
    def testee(
        a: gtx.Field[[IDim, KHalfDim], np.int32], offset_field: cases.IKField
    ) -> cases.IKField:
        b = a + 1
        return b(KDim - 0.5)(as_offset(Koff, offset_field))

    ksize = cartesian_case.default_sizes[KDim]
    a = cases.allocate(cartesian_case, testee, "a", sizes={KHalfDim: ksize + 1})()
    offset_field = cases.allocate(
        cartesian_case, testee, "offset_field", strategy=cases.ConstInitializer(1)
    )()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(
        cartesian_case,
        testee,
        a,
        offset_field,
        out=out,
        ref=a.asnumpy()[:, 1:] + 1,
        offset_provider={},
    )


@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_dynamic_offsets
def test_unstructured_shift_half_shift_as_offset(unstructured_case_3d):
    @gtx.field_operator
    def testee(
        a: gtx.Field[[Vertex, KHalfDim], np.int32],
        offset_field: gtx.Field[[Edge, KDim], np.int32],
    ) -> gtx.Field[[Edge, KDim], np.int32]:
        return a(E2V[0])(KDim - 0.5)(as_offset(Koff, offset_field))

    nvertices = unstructured_case_3d.default_sizes[Vertex]
    ksize = unstructured_case_3d.default_sizes[KDim]
    a = cases.allocate(
        unstructured_case_3d,
        testee,
        "a",
        domain={Vertex: (0, nvertices), KHalfDim: (0, ksize + 1)},
    )()
    offset_field = cases.allocate(
        unstructured_case_3d, testee, "offset_field", strategy=cases.ConstInitializer(1)
    )()
    out = cases.allocate(unstructured_case_3d, testee, cases.RETURN)()

    e2v_table = unstructured_case_3d.offset_provider["E2V"].asnumpy()
    cases.verify(
        unstructured_case_3d,
        testee,
        a,
        offset_field,
        out=out,
        ref=a.asnumpy()[e2v_table[:, 0], 1:],
    )
