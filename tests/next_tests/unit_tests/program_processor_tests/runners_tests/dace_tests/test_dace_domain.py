# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test domain-related helper functions of the dace backend module."""

import pytest

dace = pytest.importorskip("dace")

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace import (
    gtir_domain,
    utils as gtx_dace_utils,
)
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners.dace import gtir_domain as gtx_dace_domain

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    Cell,
    KDim,
)


@pytest.mark.parametrize(
    "param",
    [
        (
            "-horizontal_start + Max(horizontal_start, Min(horizontal_end, start_nudge_line_idx_e))",
            "Max(0, -horizontal_start + Min(horizontal_end, start_nudge_line_idx_e))",
        ),
        ("Max(0, vertical_end - vertical_start)", "vertical_end - vertical_start"),
    ],
)
def test_simplify_domain_expr(param):
    domain_expr = im.domain(
        gtx_common.GridType.CARTESIAN,
        ranges={
            Cell: ("horizontal_start", "horizontal_end"),
            KDim: ("vertical_start", "vertical_end"),
        },
    )
    domain = gtx_dace_domain.extract_domain(domain_expr)

    expr = dace.symbolic.pystr_to_symbolic(param[0])
    expected_expr = dace.symbolic.pystr_to_symbolic(param[1])

    assert gtx_dace_domain.simplify_domain_expr(expr, domain) == expected_expr


def test_gtir_domain():
    Vertex = gtx_common.Dimension(value="Vertex", kind=gtx_common.DimensionKind.HORIZONTAL)
    KDim = gtx_common.Dimension(value="KDim", kind=gtx_common.DimensionKind.VERTICAL)

    ir = im.domain(
        gtx_common.GridType.UNSTRUCTURED,
        ranges={
            Vertex: (1, 10),
            KDim: (2, 20),
        },
    )

    assert gtir_domain.extract_domain(ir) == [
        gtir_domain.FieldopDomainRange(
            Vertex,
            1,
            10,
        ),
        gtir_domain.FieldopDomainRange(
            KDim,
            2,
            20,
        ),
    ]


def test_symbolic_domain():
    Vertex = gtx_common.Dimension(value="Vertex", kind=gtx_common.DimensionKind.HORIZONTAL)
    KDim = gtx_common.Dimension(value="KDim", kind=gtx_common.DimensionKind.VERTICAL)

    ir = domain_utils.SymbolicDomain(
        gtx_common.GridType.UNSTRUCTURED,
        ranges={
            Vertex: domain_utils.SymbolicRange(
                start=im.tuple_get(0, im.call("get_domain")("arg", im.axis_literal(Vertex))),
                stop=im.tuple_get(1, im.call("get_domain")("arg", im.axis_literal(Vertex))),
            ),
            KDim: domain_utils.SymbolicRange(
                start=im.tuple_get(0, im.call("get_domain")("arg", im.axis_literal(KDim))),
                stop=im.tuple_get(1, im.call("get_domain")("arg", im.axis_literal(KDim))),
            ),
        },
    )

    assert gtir_domain.extract_domain(ir) == [
        gtir_domain.FieldopDomainRange(
            Vertex,
            dace.symbolic.SymExpr(gtx_dace_utils.range_start_symbol("arg", Vertex)),
            dace.symbolic.SymExpr(gtx_dace_utils.range_stop_symbol("arg", Vertex)),
        ),
        gtir_domain.FieldopDomainRange(
            KDim,
            dace.symbolic.SymExpr(gtx_dace_utils.range_start_symbol("arg", KDim)),
            dace.symbolic.SymExpr(gtx_dace_utils.range_stop_symbol("arg", KDim)),
        ),
    ]
