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
import copy
import dataclasses


@dataclasses.dataclass(kw_only=True)
class RewriteLineNumbers(ast.NodeTransformer):
    """
    AST pass transforming line numbers by adding a fixed offset.

    This pass is useful to make line numbers consistent with the source file
    an AST node originated from.
    """

    starting_line: int
    inherit_from_parent: bool

    _parent_node: ast.AST

    @classmethod
    def apply(
        cls,
        node: ast.AST,
        starting_line: int,
        *,
        inherit_from_parent: bool = True,
        inplace: bool = False,
    ):
        """
        Add fixed offset to all line numbers in an AST node.

        Arguments:
            node:
            starting_line: The offset added to each nodes linenumber.

        Keyword arguments:
            inherit_from_parent: If a node has no line number information use
                the information of a parent node.
            inplace: Inplace modifications to the original ``node``.
        """
        return cls(
            starting_line=starting_line, inherit_from_parent=inherit_from_parent, _parent_node=node
        ).visit(node if inplace else copy.deepcopy(node))

    def generic_visit(self, node: ast.AST):
        if hasattr(node, "lineno") and node.lineno:
            node.lineno = node.lineno + self.starting_line - 1
            self._parent_node = node
        elif self.inherit_from_parent:
            node.lineno = self._parent_node.lineno
            node.col_offset = self._parent_node.col_offset
            # the end positions are optional according to
            #  https://docs.python.org/3/library/ast.html#ast.AST.end_col_offset
            if hasattr(node, "end_lineno"):
                node.end_lineno = self._parent_node.end_lineno
            if hasattr(node, "end_col_offset"):
                node.end_col_offset = self._parent_node.end_col_offset

        return super().generic_visit(node)
