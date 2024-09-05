# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

import gt4py.next.ffront.field_operator_ast as foast
from gt4py.eve import NodeTranslator, traits
from gt4py.eve.utils import FrozenNamespace
from gt4py.next import errors


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
        self,
        node: foast.Name,
        current_closure_vars: dict[str, Any],
        symtable: dict[str, foast.Symbol],
        **kwargs: Any,
    ) -> foast.Name | foast.Constant:
        if node.id in symtable:
            definition = symtable[node.id]
            if definition in current_closure_vars:
                value = self.closure_vars[node.id]
                if isinstance(value, FrozenNamespace):
                    return foast.Constant(value=value, location=node.location)
        return node

    def visit_Attribute(
        self, node: foast.Attribute, **kwargs: Any
    ) -> foast.Constant | foast.Attribute:
        value = self.visit(node.value, **kwargs)
        if isinstance(value, foast.Constant):
            if hasattr(value.value, node.attr):
                return foast.Constant(value=getattr(value.value, node.attr), location=node.location)
            raise errors.MissingAttributeError(node.location, node.attr)
        return node

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs: Any
    ) -> foast.FunctionDefinition:
        return self.generic_visit(node, current_closure_vars=node.closure_vars, **kwargs)
