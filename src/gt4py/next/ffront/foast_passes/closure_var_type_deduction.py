# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from dataclasses import dataclass
from typing import Any

import gt4py.next.ffront.field_operator_ast as foast
from gt4py.eve import NodeTranslator, traits
from gt4py.next.type_system import type_translation


@dataclass(frozen=True)
class ClosureVarTypeDeduction(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Deduce the type of all closure variable declarations.

    After this pass all closure var declarations, i.e. the `Symbol`s contained in
    `foast.FunctionDefinition.closure_vars`, have the type as specified by their
    actual values (given by `closure_vars` to `apply`).
    """

    closure_vars: dict[str, Any]

    @classmethod
    def apply(
        cls, node: foast.FunctionDefinition, closure_vars: dict[str, Any]
    ) -> foast.FunctionDefinition:
        return cls(closure_vars=closure_vars).visit(node)

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs
    ) -> foast.FunctionDefinition:
        new_closure_vars: list[foast.Symbol] = []
        for sym in node.closure_vars:
            if not isinstance(self.closure_vars[sym.id], type):
                new_symbol: foast.Symbol = foast.Symbol(
                    id=sym.id,
                    location=sym.location,
                    type=type_translation.from_value(self.closure_vars[sym.id]),
                )
                new_closure_vars.append(new_symbol)
            else:
                new_closure_vars.append(sym)
        return foast.FunctionDefinition(
            id=node.id,
            params=node.params,
            body=node.body,
            closure_vars=new_closure_vars,
            type=node.type,
            location=node.location,
        )
