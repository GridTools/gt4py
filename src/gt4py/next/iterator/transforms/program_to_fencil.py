# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm


def program_to_fencil(node: itir.Program) -> itir.FencilDefinition:
    assert not node.declarations
    closures = []
    for stmt in node.body:
        assert isinstance(stmt, itir.SetAt)
        assert isinstance(stmt.expr, itir.FunCall) and cpm.is_call_to(stmt.expr.fun, "as_fieldop")
        stencil, domain = stmt.expr.fun.args
        inputs = stmt.expr.args
        assert all(isinstance(inp, itir.SymRef) for inp in inputs)
        closures.append(
            itir.StencilClosure(domain=domain, stencil=stencil, output=stmt.target, inputs=inputs)
        )

    return itir.FencilDefinition(
        id=node.id,
        function_definitions=node.function_definitions,
        params=node.params,
        closures=closures,
    )
