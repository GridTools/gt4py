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
class FixMissingLocations(ast.NodeTransformer):
    """
    AST pass adding source information to every node.

    While most nodes of the Python AST have source information (lineno,
    col_offset, end_lineno, end_col_offset) after parsing, some nodes, e.g.
    :class:`ast.Pow`, do not. This pass adds this information, taking it from
    the parent node.

    Note that :func:`ast.fix_missing_locations` only adds source information to
    some ast nodes and is hence not a replacement for this pass.
    """

    _parent_node: ast.AST

    @classmethod
    def apply(cls, node: ast.AST) -> ast.AST:
        return cls(_parent_node=node).visit(node)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if getattr(node, "lineno", None):
            self._parent_node = node
        else:
            node.lineno = self._parent_node.lineno
            node.col_offset = self._parent_node.col_offset
            # the end positions are optional according to
            #  https://docs.python.org/3/library/ast.html#ast.AST.end_col_offset
            if hasattr(node, "end_lineno"):
                node.end_lineno = self._parent_node.end_lineno
            if hasattr(node, "end_col_offset"):
                node.end_col_offset = self._parent_node.end_col_offset

        return super().generic_visit(node)
