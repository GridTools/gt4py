# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import typing

import pytest

from gt4py import next as gtx
from gt4py.next.iterator.transforms import extractors

from next_tests.integration_tests.cases_utils import (
    IDim,
    JDim,
    KDim,
)


def test_input_names_extractor_cartesian():
    @gtx.field_operator
    def input_names_extractor_cartesian_op(
        a: gtx.Field[[IDim, JDim, KDim], gtx.int],
    ) -> gtx.Field[[IDim, JDim, KDim], gtx.int]:
        return a

    @gtx.program
    def input_names_extractor_cartesian_prog(
        a: gtx.Field[[IDim, JDim, KDim], gtx.int],
        b: gtx.Field[[IDim, JDim, KDim], gtx.int],
        c: gtx.Field[[IDim, JDim, KDim], gtx.int],
    ):
        input_names_extractor_cartesian_op(b, out=c)
        input_names_extractor_cartesian_op(a, out=b)

    input_field_names = extractors.InputNamesExtractor.only_fields(input_names_extractor_cartesian_prog.gtir)
    assert input_field_names == {"a", "b"}


def test_output_names_extractor():
    @gtx.field_operator
    def output_names_extractor_op(
        a: gtx.Field[[IDim, JDim, KDim], gtx.int],
    ) -> gtx.Field[[IDim, JDim, KDim], gtx.int]:
        return a

    @gtx.program
    def output_names_extractor_prog(
        a: gtx.Field[[IDim, JDim, KDim], gtx.int],
        b: gtx.Field[[IDim, JDim, KDim], gtx.int],
        c: gtx.Field[[IDim, JDim, KDim], gtx.int],
    ):
        output_names_extractor_op(a, out=b)
        output_names_extractor_op(a, out=c)

    output_field_names = extractors.OutputNamesExtractor.only_fields(output_names_extractor_prog.gtir)
    assert output_field_names == {"b", "c"}
