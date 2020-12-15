from eve import NodeTranslator

from gt4py.gtc import gtir

from devtools import debug


def _upcast_nodes(*exprs):
    target_dtype = max([e.dtype for e in exprs])
    return map(
        lambda e: e if e.dtype == target_dtype else gtir.Cast(dtype=target_dtype, expr=e), exprs
    )


class _GTIRUpcasting(NodeTranslator):
    """
    Introduces Cast nodes (upcasting) for expr involving different datatypes
    Precondition: all dtypes are resolved (no `None`, `Auto`, `Default`)
    Postcondition: all dtype transitions are explicit via a `Cast` node
    """

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
        left = self.visit(node.left)
        right = self.visit(node.right)
        left, right = _upcast_nodes(left, right)

        if left != node.left or right != node.right:
            return gtir.BinaryOp(left=left, right=right, op=node.op)
        else:
            return node

    def visit_TernaryOp(self, node: gtir.TernaryOp, **kwargs):
        cond = self.visit(node.cond)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        true_expr, false_expr = _upcast_nodes(true_expr, false_expr)

        if true_expr != node.true_expr or false_expr != node.false_expr or cond != node.cond:
            return gtir.TernaryOp(true_expr=true_expr, false_expr=false_expr, cond=cond)
        else:
            return node

    def visit_NativeFuncCall(self, node: gtir.NativeFuncCall, **kwargs):
        args = self.visit(node.args)
        args = _upcast_nodes(args)

        if args != node.args:
            return gtir.NativeFuncCall(args=args)
        else:
            return node


def upcast(node: gtir.Stencil):
    return _GTIRUpcasting().visit(node)
