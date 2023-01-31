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

from typing import Any

import gt4py.next.ffront.field_operator_ast as foast
from gt4py.eve import NodeTranslator, traits


class DeadClosureVarElimination(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """Remove closure variable symbols that are not referenced in the AST."""

    _referenced_symbols: list[foast.Symbol]

    @classmethod
    def apply(cls, node: foast.FunctionDefinition) -> foast.FunctionDefinition:
        return cls().visit(node)

    def visit_Name(self, node: foast.Name, symtable, **kwargs: Any) -> foast.Name:
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
