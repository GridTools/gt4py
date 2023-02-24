# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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


class UnchainComparesPass(ast.NodeTransformer):
    """
    Turn comparison chain into a tree of comparisons linked by ``&`` operations.

    ``a > b < c == d ...`` is turned into ``((a > b) & ((b < c) & (c == d))`` and so on.
    The resulting AST will have only ``Comparison`` nodes with to operands and one operation.
    That is to say with the same structure as FOAST ``Comparison`` nodes.

    Examples:
    ---------
    >>> import inspect
    >>> def example(a, b, c, d):
    ...     return a > b < c == d
    >>> print(ast.unparse(UnchainComparesPass.apply(ast.parse(inspect.getsource(example)))))
    def example(a, b, c, d):
        return (a > b) & ((b < c) & (c == d))
    """

    @classmethod
    def apply(cls, node: ast.AST) -> ast.AST:
        return cls().visit(node)

    def visit_Compare(self, node: ast.Compare) -> ast.Compare | ast.BinOp:
        # stopping case: single comparison a <op> b
        if len(node.comparators) == 1:
            return node

        # left leaf of the new tree: ``a < b``
        # example: ``a < b > c > d``
        left_leaf = ast.Compare(
            ops=node.ops[0:1],
            left=node.left,
            comparators=node.comparators[0:1],
        )
        ast.copy_location(left_leaf, node)

        # the remainder of the chain -> right branch of the new tree
        # example: ``b > c > d``
        remaining_chain = copy.copy(node)
        remaining_chain.left = remaining_chain.comparators.pop(0)
        remaining_chain.ops.pop(0)

        # create the tree root
        # example: ``(a < b) & ((b > c) & (c > d))``
        root = ast.BinOp(
            op=ast.BitAnd(),
            left=left_leaf,
            right=self.visit(remaining_chain),  # example: recursively visit ``b > c > d``
        )
        ast.copy_location(root, node)

        return root
