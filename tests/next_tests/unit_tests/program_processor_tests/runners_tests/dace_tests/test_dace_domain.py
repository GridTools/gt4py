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
    gtir_domain as gtx_dace_domain,
    utils as gtx_dace_utils,
)
from gt4py.next.iterator.ir_utils import domain_utils, ir_makers as im

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    KDim,
    Vertex,
)


def test_symbolic_domain():
    domain = domain_utils.SymbolicDomain.from_expr(
        im.get_field_domain(gtx_common.GridType.UNSTRUCTURED, "arg", [Vertex, KDim])
    )

    assert gtx_dace_domain.get_field_domain(domain) == [
        gtx_dace_domain.FieldopDomainRange(
            Vertex,
            dace.symbolic.SymExpr(gtx_dace_utils.range_start_symbol("arg", Vertex)),
            dace.symbolic.SymExpr(gtx_dace_utils.range_stop_symbol("arg", Vertex)),
        ),
        gtx_dace_domain.FieldopDomainRange(
            KDim,
            dace.symbolic.SymExpr(gtx_dace_utils.range_start_symbol("arg", KDim)),
            dace.symbolic.SymExpr(gtx_dace_utils.range_stop_symbol("arg", KDim)),
        ),
    ]
