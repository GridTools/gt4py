# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir


class PruneClosureInputs(PreserveLocationVisitor, NodeTranslator):
    """Removes all unused input arguments from a stencil closure."""

    def visit_StencilClosure(self, node: ir.StencilClosure) -> ir.StencilClosure:
        if not isinstance(node.stencil, ir.Lambda):
            return node

        unused: set[str] = {p.id for p in node.stencil.params}
        expr = self.visit(node.stencil.expr, unused=unused, shadowed=set[str]())
        params = []
        inputs = []
        for param, inp in zip(node.stencil.params, node.inputs):
            if param.id not in unused:
                params.append(param)
                inputs.append(inp)

        return ir.StencilClosure(
            domain=node.domain,
            stencil=ir.Lambda(params=params, expr=expr),
            output=node.output,
            inputs=inputs,
        )

    def visit_SymRef(self, node: ir.SymRef, *, unused: set[str], shadowed: set[str]) -> ir.SymRef:
        if node.id not in shadowed:
            unused.discard(node.id)
        return node

    def visit_Lambda(self, node: ir.Lambda, *, unused: set[str], shadowed: set[str]) -> ir.Lambda:
        return self.generic_visit(
            node, unused=unused, shadowed=shadowed | {p.id for p in node.params}
        )
