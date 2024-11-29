# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Optional, Set

from gt4py.eve import NodeTranslator, PreserveLocationVisitor, SymbolTableTrait
from gt4py.next.iterator import ir
from gt4py.next.iterator.type_system import inference as type_inference


class RemapSymbolRefs(PreserveLocationVisitor, NodeTranslator):
    # This pass preserves, but doesn't use the `type`, `recorded_shifts`, `domain` annex.
    PRESERVED_ANNEX_ATTRS = ("type", "recorded_shifts", "domain")

    def visit_SymRef(self, node: ir.SymRef, *, symbol_map: Dict[str, ir.Node]):
        return symbol_map.get(str(node.id), node)

    def visit_Lambda(self, node: ir.Lambda, *, symbol_map: Dict[str, ir.Node]):
        params = {str(p.id) for p in node.params}
        new_symbol_map = {k: v for k, v in symbol_map.items() if k not in params}
        return ir.Lambda(params=node.params, expr=self.visit(node.expr, symbol_map=new_symbol_map))

    def generic_visit(self, node: ir.Node, **kwargs: Any):  # type: ignore[override]
        assert isinstance(node, SymbolTableTrait) == isinstance(
            node, ir.Lambda
        ), "found unexpected new symbol scope"
        return super().generic_visit(node, **kwargs)


class RenameSymbols(PreserveLocationVisitor, NodeTranslator):
    # This pass preserves, but doesn't use the `type`, `recorded_shifts`, `domain` annex.
    PRESERVED_ANNEX_ATTRS = ("type", "recorded_shifts", "domain")

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
            new_ref = ir.SymRef(id=name_map.get(node.id, node.id))
            type_inference.copy_type(from_=node, to=new_ref)
            return new_ref
        return node

    def generic_visit(  # type: ignore[override]
        self, node: ir.Node, *, name_map: Dict[str, str], active: Optional[Set[str]] = None
    ):
        if isinstance(node, SymbolTableTrait):
            if active is None:
                active = set()
            active = active | set(node.annex.symtable)
        return super().generic_visit(node, name_map=name_map, active=active)
