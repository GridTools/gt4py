# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
import ast
import dataclasses


@dataclasses.dataclass(kw_only=True)
class RemoveDocstrings(ast.NodeTransformer):
    """
    Docstrings within a field_operator or program appear as type ast.Expr with ast.Constant value of type string
    If such patterns is detected, this entry in the node.body list is removed.
    """

    _parent_node: ast.AST

    @classmethod
    def apply(cls, node: ast.AST) -> ast.AST:
        return cls(_parent_node=node).visit(node)

    def generic_visit(self, node: ast.AST) -> ast.AST:

        if hasattr(node, "body"):
            for obj in node.body:
                if isinstance(obj, ast.Expr):
                    if isinstance(obj.value, ast.Constant) and isinstance(obj.value.value, str):
                        node.body.remove(obj)

        return node
