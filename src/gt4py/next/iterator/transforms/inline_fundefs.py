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

from typing import Any, Dict, Set

from gt4py.eve import NOTHING, NodeTranslator
from gt4py.next.iterator import ir


class InlineFundefs(NodeTranslator):
    def visit_SymRef(self, node: ir.SymRef, *, symtable: Dict[str, Any]):
        if node.id in symtable and isinstance((symbol := symtable[node.id]), ir.FunctionDefinition):
            return ir.Lambda(
                params=self.generic_visit(symbol.params, symtable=symtable),
                expr=self.generic_visit(symbol.expr, symtable=symtable),
            )
        return self.generic_visit(node)

    def visit_FencilDefinition(self, node: ir.FencilDefinition):
        return self.generic_visit(node, symtable=node.annex.symtable)


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
