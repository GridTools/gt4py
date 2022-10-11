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

import functional.ffront.field_operator_ast as foast
from eve import NodeTranslator, traits
from functional.ffront.symbol_makers import make_symbol_type_from_value


class ClosureVarTypeDeduction(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    closure_vars: dict[str, Any]

    def __init__(self, closure_vars: dict[str, Any]):
        self.closure_vars = closure_vars

    @classmethod
    def apply(cls, node: foast.FieldOperator, closure_vars: dict[str, Any]):
        return cls(closure_vars=closure_vars).visit(node)

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs):
        new_closure_vars: list[foast.Symbol] = []
        for sym in node.closure_vars:
            if sym.id in self.closure_vars and not isinstance(self.closure_vars[sym.id], type):
                new_symbol = foast.Symbol(id=sym.id,
                                          location=sym.location,
                                          type=make_symbol_type_from_value(self.closure_vars[sym.id])
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
