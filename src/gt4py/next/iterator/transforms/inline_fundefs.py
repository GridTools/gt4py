# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Set

from gt4py.eve import NOTHING, NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir


class InlineFundefs(PreserveLocationVisitor, NodeTranslator):
    def visit_SymRef(self, node: ir.SymRef, *, symtable: Dict[str, Any]):
        if node.id in symtable and isinstance((symbol := symtable[node.id]), ir.FunctionDefinition):
            return ir.Lambda(
                params=self.generic_visit(symbol.params, symtable=symtable),
                expr=self.generic_visit(symbol.expr, symtable=symtable),
            )
        return self.generic_visit(node)

    def visit_Program(self, node: ir.Program):
        return self.generic_visit(node, symtable=node.annex.symtable)


class PruneUnreferencedFundefs(PreserveLocationVisitor, NodeTranslator):
    def visit_FunctionDefinition(
        self, node: ir.FunctionDefinition, *, referenced: Set[str], second_pass: bool
    ):
        if second_pass and node.id not in referenced:
            return NOTHING
        return self.generic_visit(node, referenced=referenced, second_pass=second_pass)

    def visit_SymRef(self, node: ir.SymRef, *, referenced: Set[str], second_pass: bool):
        referenced.add(node.id)
        return node

    def visit_Program(self, node: ir.Program):
        referenced: Set[str] = set()
        self.generic_visit(node, referenced=referenced, second_pass=False)
        return self.generic_visit(node, referenced=referenced, second_pass=True)
