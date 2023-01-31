# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any, Dict, Optional, Set

from gt4py.eve import NodeTranslator, SymbolTableTrait
from gt4py.next.iterator import ir


class RemapSymbolRefs(NodeTranslator):
    def visit_SymRef(self, node: ir.SymRef, *, symbol_map: Dict[str, ir.Node]):
        return symbol_map.get(str(node.id), node)

    def visit_Lambda(self, node: ir.Lambda, *, symbol_map: Dict[str, ir.Node]):
        params = {str(p.id) for p in node.params}
        new_symbol_map = {k: v for k, v in symbol_map.items() if k not in params}
        return ir.Lambda(
            params=node.params,
            expr=self.visit(node.expr, symbol_map=new_symbol_map),
        )

    def generic_visit(self, node: ir.Node, **kwargs: Any):  # type: ignore[override]
        assert isinstance(node, SymbolTableTrait) == isinstance(
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

    def generic_visit(  # type: ignore[override]
        self, node: ir.Node, *, name_map: Dict[str, str], active: Optional[Set[str]] = None
    ):
        if isinstance(node, SymbolTableTrait):
            if active is None:
                active = set()
            active = active | set(node.annex.symtable)
        return super().generic_visit(node, name_map=name_map, active=active)
