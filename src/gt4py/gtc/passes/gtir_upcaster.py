from typing import Dict, Iterable, List, Union
from eve import NodeTranslator, Node

from gt4py.gtc import gtir

import itertools

from devtools import debug
from gt4py.gtc.common import DataType
from gt4py.gtc.gtir import Expr


def _upcast_nodes(*exprs):
    target_dtype = max([e.dtype for e in exprs])
    return map(
        lambda e: e if e.dtype == target_dtype else gtir.Cast(dtype=target_dtype, expr=e), exprs
    )


def _update_node(node: Node, updated_children: Dict[str, Node]):
    old_children = node.dict(include={*updated_children.keys()})
    if any([old_children[k] != updated_children[k] for k in updated_children.keys()]):
        return node.copy(update=updated_children)
    else:
        return node


class _GTIRUpcasting(NodeTranslator):
    """
    Introduces Cast nodes (upcasting) for expr involving different datatypes
    Precondition: all dtypes are resolved (no `None`, `Auto`, `Default`)
    Postcondition: all dtype transitions are explicit via a `Cast` node
    """

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
        left, right = _upcast_nodes(self.visit(node.left), self.visit(node.right))
        return _update_node(node, {"left": left, "right": right})

    def visit_TernaryOp(self, node: gtir.TernaryOp, **kwargs):
        true_expr, false_expr = _upcast_nodes(
            self.visit(node.true_expr), self.visit(node.false_expr)
        )
        return _update_node(
            node, {"true_expr": true_expr, "false_expr": false_expr, "cond": self.visit(node.cond)}
        )

    def visit_NativeFuncCall(self, node: gtir.NativeFuncCall, **kwargs):
        args = _upcast_nodes(self.visit(node.args))

        return _update_node(node, {"args": args})


def upcast(node: gtir.Stencil):
    return _GTIRUpcasting().visit(node)
