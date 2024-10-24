# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import gt4py.next.ffront.field_operator_ast as foast
from gt4py import eve


class DeadClosureVarElimination(eve.NodeTranslator, eve.traits.VisitorWithSymbolTableTrait):
    """Remove closure variable symbols that are not referenced in the AST."""

    _referenced_symbols: list[foast.Symbol]

    @classmethod
    def apply(cls, node: foast.FunctionDefinition) -> foast.FunctionDefinition:
        return cls().visit(node)

    def visit_Name(
        self, node: foast.Name, symtable: dict[str, foast.Symbol], **kwargs: Any
    ) -> foast.Name:
        if node.id in symtable:
            self._referenced_symbols.append(symtable[node.id])
        return node

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs: Any
    ) -> foast.FunctionDefinition:
        self._referenced_symbols = []
        self.visit(node.body, **kwargs)
        referenced_closure_vars = [
            closure_var
            for closure_var in node.closure_vars
            if closure_var in self._referenced_symbols
        ]
        return foast.FunctionDefinition(
            id=node.id,
            params=node.params,
            body=node.body,
            closure_vars=referenced_closure_vars,
            type=node.type,
            location=node.location,
        )
