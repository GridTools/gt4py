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
from eve.utils import FrozenNamespace


class ClosureVarFolding(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Replace references to closure variables or their attributes with constants.

    `Name` nodes that refer to closure variables are replaced by `Constant`
     nodes. `Attribute` nodes that refer to attributes of closure variables
     are recursively replaced by `Constant` nodes.
    """

    closure_vars: dict[str, Any]
    _current_closure_vars: list[foast.Symbol]

    def __init__(self, closure_vars: dict[str, Any]):
        self.closure_vars = closure_vars

    @classmethod
    def apply(cls, node: foast.FieldOperator, closure_vars: dict[str, Any]):
        return cls(closure_vars=closure_vars).visit(node)

    def visit_Name(self, node: foast.Name, **kwargs):
        symtable = kwargs["symtable"]
        if node.id in symtable:
            definition = symtable[node.id]
            if definition in self._current_closure_vars:
                value = self.closure_vars[node.id]
                if isinstance(value, FrozenNamespace):
                    return foast.Constant(value=value, location=node.location)
        return node

    def visit_Attribute(self, node: foast.Attribute, **kwargs):
        # TODO: fix import form parent module by restructuring exception classis
        from functional.ffront.func_to_foast import FieldOperatorSyntaxError

        value = self.visit(node.value, **kwargs)
        if isinstance(value, foast.Constant):
            if hasattr(value.value, node.attr):
                return foast.Constant(value=getattr(value.value, node.attr), location=node.location)
            # TODO: use proper exception type (requires refactoring `FieldOperatorSyntaxError`)
            raise FieldOperatorSyntaxError(
                msg="Constant does not have the attribute specified by the AST.",
                filename=node.location.source,
                lineno=node.location.line,
                offset=node.location.column,
                end_lineno=node.location.end_line,
                end_offset=node.location.end_column,
            )
        # TODO: use proper exception type (requires refactoring `FieldOperatorSyntaxError`)
        raise FieldOperatorSyntaxError(
            msg="Attribute can only be used on constants.",
            filename=node.location.source,
            lineno=node.location.line,
            offset=node.location.column,
            end_lineno=node.location.end_line,
            end_offset=node.location.end_column,
        )

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs):
        self._current_closure_vars = node.closure_vars
        return self.generic_visit(node, **kwargs)
