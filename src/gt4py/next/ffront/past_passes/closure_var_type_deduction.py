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

from dataclasses import dataclass
from typing import Any

import gt4py.next.ffront.program_ast as past
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
    def apply(cls, node: past.Program, closure_vars: dict[str, Any]) -> past.Program:
        return cls(closure_vars=closure_vars).visit(node)

    def visit_Program(self, node: past.Program, **kwargs) -> past.Program:
        new_closure_vars: list[past.Symbol] = []
        for sym in node.closure_vars:
            if not isinstance(self.closure_vars[sym.id], type):
                new_symbol: past.Symbol = past.Symbol(
                    id=sym.id,
                    location=sym.location,
                    type=type_translation.from_value(self.closure_vars[sym.id]),
                )
                new_closure_vars.append(new_symbol)
            else:
                new_closure_vars.append(sym)
        return past.Program(
            id=node.id,
            params=node.params,
            body=node.body,
            location=node.location,
            type=node.type,
            closure_vars=new_closure_vars,
        )
