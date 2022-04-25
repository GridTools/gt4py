from typing import Any, Dict, Set

from eve import NOTHING, NodeTranslator
from functional.iterator import ir


class InlineFundefs(NodeTranslator):
    def visit_SymRef(self, node: ir.SymRef, *, symtable: Dict[str, Any]):
        if node.id in symtable and isinstance((symbol := symtable[node.id]), ir.FunctionDefinition):
            return ir.Lambda(
                params=self.generic_visit(symbol.params, symtable=symtable),
                expr=self.generic_visit(symbol.expr, symtable=symtable),
            )
        return self.generic_visit(node)

    def visit_FencilDefinition(self, node: ir.FencilDefinition):
        return self.generic_visit(node, symtable=node.symtable_)


class PruneUnreferencedFundefs(NodeTranslator):
    def visit_FunctionDefinition(
        self, node: ir.FunctionDefinition, *, referenced: Set[str], second_pass: bool
    ):
        if second_pass and node.id not in referenced:
            return NOTHING
        return self.generic_visit(node, referenced=referenced, second_pass=second_pass)

    def visit_SymRef(self, node: ir.SymRef, *, referenced: Set[str], second_pass: bool):
        referenced.add(node.id)
        return node

    def visit_FencilDefinition(self, node: ir.FencilDefinition):
        referenced: Set[str] = set()
        self.generic_visit(node, referenced=referenced, second_pass=False)
        return self.generic_visit(node, referenced=referenced, second_pass=True)
