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

import gt4py.next.ffront.field_operator_ast as foast
from gt4py.eve import NodeTranslator, traits
from gt4py.eve.utils import FrozenNamespace


@dataclass
class ClosureVarFolding(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Replace references to closure variables or their attributes with constants.

    `Name` nodes that refer to closure variables are replaced by `Constant`
     nodes. `Attribute` nodes that refer to attributes of closure variables
     are recursively replaced by `Constant` nodes.
    """

    closure_vars: dict[str, Any]

    @classmethod
    def apply(
        cls, node: foast.FunctionDefinition | foast.FieldOperator, closure_vars: dict[str, Any]
    ) -> foast.FunctionDefinition:
        return cls(closure_vars=closure_vars).visit(node)

    def visit_Name(
        self, node: foast.Name, current_closure_vars, symtable, **kwargs
    ) -> foast.Name | foast.Constant:
        if node.id in symtable:
            definition = symtable[node.id]
            if definition in current_closure_vars:
                value = self.closure_vars[node.id]
                if isinstance(value, FrozenNamespace):
                    return foast.Constant(value=value, location=node.location)
        return node

    def visit_Attribute(self, node: foast.Attribute, **kwargs) -> foast.Constant:
        # TODO: fix import form parent module by restructuring exception classis
        from gt4py.next.ffront.func_to_foast import FieldOperatorSyntaxError

        value = self.visit(node.value, **kwargs)
        if isinstance(value, foast.Constant):
            if hasattr(value.value, node.attr):
                return foast.Constant(value=getattr(value.value, node.attr), location=node.location)
            # TODO: use proper exception type (requires refactoring `FieldOperatorSyntaxError`)
            raise FieldOperatorSyntaxError.from_location(
                msg="Constant does not have the attribute specified by the AST.",
                location=node.location,
            )
        # TODO: use proper exception type (requires refactoring `FieldOperatorSyntaxError`)
        raise FieldOperatorSyntaxError.from_location(
            msg="Attribute can only be used on constants.", location=node.location
        )

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs
    ) -> foast.FunctionDefinition:
        return self.generic_visit(node, current_closure_vars=node.closure_vars, **kwargs)
