# GT4Py Project - GridTools Framework
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

import functional.ffront.field_operator_ast as foast
from eve import NodeTranslator, traits, SourceLocation
from functional.ffront.gtcallable import GTCallable
from functional.common import Dimension
from functional.ffront.fbuiltins import FieldOffset


class ClosureVarFolding(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    closure_vars: dict[str, Any]

    def __init__(self, closure_vars: dict[str, Any]):
        self.closure_vars = closure_vars

    def visit_SymbolRef(self, node: foast.SymbolRef, **kwargs):
        if node in self.closure_vars:
            value = self.closure_vars[node]
            if not isinstance(value, (GTCallable, Dimension, FieldOffset)):
                return foast.Constant(value=value, location=SourceLocation(1, 1, "<unknown>"))
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
