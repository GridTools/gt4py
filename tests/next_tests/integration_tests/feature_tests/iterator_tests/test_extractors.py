# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import typing

import pytest

from gt4py import next as gtx
from gt4py.next import common
from gt4py.next.iterator.transforms import extractors

from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    IDim,
    JDim,
    KDim,
)


if typing.TYPE_CHECKING:
    from types import ModuleType
    from typing import Optional

try:
    import dace

    from gt4py.next.program_processors.runners.dace import run_dace_cpu
except ImportError:
    from gt4py.next import backend as next_backend

    dace: Optional[ModuleType] = None
    run_dace_cpu: Optional[next_backend.Backend] = None


@pytest.fixture(params=[pytest.param(run_dace_cpu, marks=pytest.mark.requires_dace), gtx.gtfn_cpu])
def gtir_dace_backend(request):
    yield request.param


@pytest.fixture
def cartesian(request, gtir_dace_backend):
    if gtir_dace_backend is None:
        yield None

    yield cases.Case(
        backend=gtir_dace_backend,
        offset_provider={
            "Ioff": IDim,
            "Joff": JDim,
            "Koff": KDim,
        },
        default_sizes={IDim: 10, JDim: 10, KDim: 10},
        grid_type=common.GridType.CARTESIAN,
        allocator=gtir_dace_backend.allocator,
    )


@pytest.mark.skipif(dace is None, reason="DaCe not found")
def test_input_names_extractor_cartesian(cartesian):
    @gtx.field_operator(backend=cartesian.backend)
    def testee_op(
        a: gtx.Field[[IDim, JDim, KDim], gtx.int],
    ) -> gtx.Field[[IDim, JDim, KDim], gtx.int]:
        return a

    @gtx.program(backend=cartesian.backend)
    def testee(
        a: gtx.Field[[IDim, JDim, KDim], gtx.int],
        b: gtx.Field[[IDim, JDim, KDim], gtx.int],
        c: gtx.Field[[IDim, JDim, KDim], gtx.int],
    ):
        testee_op(b, out=c)
        testee_op(a, out=b)

    input_field_names = extractors.InputNamesExtractor.only_fields(testee.gtir)
    assert input_field_names == {"a", "b"}


@pytest.mark.skipif(dace is None, reason="DaCe not found")
def test_output_names_extractor(cartesian):
    @gtx.field_operator(backend=cartesian.backend)
    def testee_op(
        a: gtx.Field[[IDim, JDim, KDim], gtx.int],
    ) -> gtx.Field[[IDim, JDim, KDim], gtx.int]:
        return a

    @gtx.program(backend=cartesian.backend)
    def testee(
        a: gtx.Field[[IDim, JDim, KDim], gtx.int],
        b: gtx.Field[[IDim, JDim, KDim], gtx.int],
        c: gtx.Field[[IDim, JDim, KDim], gtx.int],
    ):
        testee_op(a, out=b)
        testee_op(a, out=c)

    output_field_names = extractors.OutputNamesExtractor.only_fields(testee.gtir)
    assert output_field_names == {"b", "c"}
