# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import math
from functools import reduce
from typing import TypeAlias

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import (
    astype,
    broadcast,
    common,
    errors,
    float32,
    float64,
    int32,
    int64,
    minimum,
    neighbor_sum,
    utils as gt_utils,
)
from gt4py.next.ffront.experimental import as_offset

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    C2E,
    E2V,
    V2E,
    E2VDim,
    Edge,
    IDim,
    IHalfDim,
    Ioff,
    JDim,
    KDim,
    Koff,
    V2EDim,
    Vertex,
    cartesian_case,
    unstructured_case,
    unstructured_case_3d,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


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
    # TODO: center inlining probably doesn't work
    @gtx.field_operator
    def testee(a: cases.IField) -> cases.IField:
        return a(IDim + 1)  # always pass an I-index to an IField

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[1:], offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_half_shift_plus(cartesian_case):
    # TODO: center inlining probably doesn't work
    @gtx.field_operator
    def testee(a: cases.IField) -> cases.IHalfField:
        return a(IHalfDim + 0.5)  # always pass an I-index to an IField

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[1:], offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_half_shift_back(cartesian_case):
    # TODO: center inlining probably doesn't work
    @gtx.field_operator
    def testee(a: cases.IHalfField) -> cases.IHalfField:
        return a(IDim + 0.5)(IHalfDim - 0.5)  # always pass an I-index to an IField

    a = cases.allocate(cartesian_case, testee, "a")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[:], offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_half_shift_plus1(cartesian_case):
    # TODO: center inlining probably doesn't work
    @gtx.field_operator
    def testee(a: cases.IHalfField) -> cases.IHalfField:
        return a(IHalfDim + 1)  # always pass an IHalf-index to an IHalfField

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out[:-1], ref=a[1:], offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_half_shift_minus(cartesian_case):
    # TODO: center inlining probably doesn't work
    @gtx.field_operator
    def testee(a: cases.IField) -> cases.IHalfField:
        return a(IHalfDim - 0.5)  # always pass an I-index to an IField

    a = cases.allocate(cartesian_case, testee, "a")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[:], offset_provider={})


@pytest.mark.uses_cartesian_shift
def test_cartesian_half_shift_half2center(cartesian_case):
    # TODO: center inlining probably doesn't work
    @gtx.field_operator
    def testee(a: cases.IHalfField) -> cases.IField:
        return 2 * a(IDim + 0.5)  # always pass an IHalf-index to an IHalfField

    a = cases.allocate(cartesian_case, testee, "a")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=2 * a[:], offset_provider={})
