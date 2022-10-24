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
from dataclasses import dataclass
from typing import Any

import functional.ffront.field_operator_ast as foast
from eve import NodeTranslator, traits
from eve.utils import ConstantNamespace


@dataclass
class ClosureVarFolding(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Replace `Name` nodes that refer to closure variable by `Constant` nodes.

    `Attribute` nodes are also replaced by a `Constant` nodes with the value of the
     closure variable's corresponding attribute. This is executed recursively for
     attributes of attributes.
    """

    closure_vars: dict[str, Any]

    def __init__(self, closure_vars: dict[str, Any]):
        self.closure_vars = closure_vars

    @classmethod
    def apply(cls, node: foast.FieldOperator, closure_vars: dict[str, Any]):
        return cls(closure_vars=closure_vars).visit(node)

    def visit_Name(self, node: foast.Name, **kwargs):
        if node.id in self.closure_vars:
            value = self.closure_vars[node.id]
            if isinstance(value, ConstantNamespace):
                return foast.Constant(value=value, location=node.location)
        return node

    def visit_Attribute(self, node: foast.Attribute, **kwargs):
        value = self.visit(node.value)
        if isinstance(value, foast.Constant):
            if hasattr(value.value, node.attr):
                return foast.Constant(value=getattr(value.value, node.attr), location=node.location)
            # TODO: throw a syntax error with all necessary information.
            raise ValueError("Constant does not have the attribute specified by the AST.")
        # TODO: throw a proper syntax error
        raise ValueError("Attribute can only be used on constants.")
