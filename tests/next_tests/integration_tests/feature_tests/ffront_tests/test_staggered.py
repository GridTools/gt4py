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

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    IDim,
    IHalfDim,
    JDim,
    KDim,
    KHalfDim,
    cartesian_case,
)
from next_tests.integration_tests.cases_utils import exec_alloc_descriptor


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
