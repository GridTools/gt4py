# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
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

from typing import Any, Dict, Iterator, List

from eve import Node, NodeTranslator
from eve.concepts import TreeNode
from gtc import gtir
from gtc.common import DataType
from gtc.gtir import Expr


def _upcast_node(target_dtype: DataType, node: Expr) -> Expr:
    return node if node.dtype == target_dtype else gtir.Cast(dtype=target_dtype, expr=node)


def _upcast_nodes(*exprs: Expr) -> Iterator[Expr]:
    assert all(e.dtype for e in exprs)
    dtypes: List[DataType] = [e.dtype for e in exprs]  # type: ignore # guaranteed to be not None
    target_dtype = max(dtypes)
    return map(lambda e: _upcast_node(target_dtype, e), exprs)


def _update_node(node: Node, updated_children: Dict[str, TreeNode]) -> Expr:
    # create new node only if children changed
    old_children = node.dict(include={*updated_children.keys()})
    if any([old_children[k] != updated_children[k] for k in updated_children.keys()]):
        return node.copy(update=updated_children)
    else:
        return node


class _GTIRUpcasting(NodeTranslator):
    """
    Introduces Cast nodes (upcasting) for expr involving different datatypes.

    Precondition: all dtypes are resolved (no `None`, `Auto`, `Default`)
    Postcondition: all dtype transitions are explicit via a `Cast` node
    """

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs: Any) -> gtir.BinaryOp:
        left, right = _upcast_nodes(self.visit(node.left), self.visit(node.right))
        return _update_node(node, {"left": left, "right": right})

    def visit_TernaryOp(self, node: gtir.TernaryOp, **kwargs: Any) -> gtir.TernaryOp:
        true_expr, false_expr = _upcast_nodes(
            self.visit(node.true_expr), self.visit(node.false_expr)
        )
        return _update_node(
            node, {"true_expr": true_expr, "false_expr": false_expr, "cond": self.visit(node.cond)}
        )

    def visit_NativeFuncCall(self, node: gtir.NativeFuncCall, **kwargs: Any) -> gtir.NativeFuncCall:
        args = [*_upcast_nodes(*self.visit(node.args))]
        return _update_node(node, {"args": args})

    def visit_ParAssignStmt(self, node: gtir.ParAssignStmt, **kwargs: Any) -> gtir.ParAssignStmt:
        assert node.left.dtype
        right = _upcast_node(node.left.dtype, self.visit(node.right))
        return _update_node(node, {"right": right})


def upcast(node: gtir.Stencil) -> gtir.Stencil:
    return _GTIRUpcasting().visit(node)
