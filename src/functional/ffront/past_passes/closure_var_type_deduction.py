# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either vervisit_Constantsion 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import functional.ffront.program_ast as past
from eve import NodeTranslator, traits
from functional.ffront.symbol_makers import make_symbol_type_from_value


class ClosureVarTypeDeduction(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    """
    Deduce the type of `Symbol` nodes that introduce closure variables.

    The types are deduced by looking at the values inside the list of
    closure variables. The types are inferred from the Python value.

    For the general type deduction pass, all Symbol nodes must already be typed,
    this pass must run before.
    """

    closure_vars: dict[str, Any]

    def __init__(self, closure_vars: dict[str, Any]):
        self.closure_vars = closure_vars

    @classmethod
    def apply(cls, node: past.Program, closure_vars: dict[str, Any]):
        return cls(closure_vars=closure_vars).visit(node)

    def visit_Program(self, node: past.Program, **kwargs):
        new_closure_vars: list[past.Symbol] = []
        for sym in node.closure_vars:
            if not isinstance(self.closure_vars[sym.id], type):
                new_symbol: past.Symbol = past.Symbol(
                    id=sym.id,
                    location=sym.location,
                    type=make_symbol_type_from_value(self.closure_vars[sym.id]),
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
