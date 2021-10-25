from typing import Any, Dict, Optional, Set

from eve import NodeTranslator
from iterator import ir


class RemapSymbolRefs(NodeTranslator):
    def visit_SymRef(self, node: ir.SymRef, *, symbol_map: Dict[str, ir.Node]):
        return symbol_map.get(node.id, node)

    def visit_Lambda(self, node: ir.Lambda, *, symbol_map: Dict[str, ir.Node]):
        params = {str(p.id) for p in node.params}
        new_symbol_map = {k: v for k, v in symbol_map.items() if k not in params}
        return ir.Lambda(
            params=node.params,
            expr=self.generic_visit(node.expr, symbol_map=new_symbol_map),
        )

    def generic_visit(self, node: ir.Node, **kwargs: Any):
        assert isinstance(node, ir.SymbolTableTrait) == isinstance(
            node, ir.Lambda
        ), "found unexpected new symbol scope"
        return super().generic_visit(node, **kwargs)


class RenameSymbols(NodeTranslator):
    def visit_Sym(
        self, node: ir.Sym, *, name_map: Dict[str, str], active: Optional[Set[str]] = None
    ):
        if active and node.id in active:
            return ir.Sym(id=name_map.get(node.id, node.id))
        return node

    def visit_SymRef(
        self, node: ir.SymRef, *, name_map: Dict[str, str], active: Optional[Set[str]] = None
    ):
        if active and node.id in active:
            return ir.SymRef(id=name_map.get(node.id, node.id))
        return node

    def generic_visit(
        self, node: ir.Node, *, name_map: Dict[str, str], active: Optional[Set[str]] = None
    ):
        if isinstance(node, ir.SymbolTableTrait):
            if active is None:
                active = set()
            active = active | set(node.symtable_)
        return super().generic_visit(node, name_map=name_map, active=active)
