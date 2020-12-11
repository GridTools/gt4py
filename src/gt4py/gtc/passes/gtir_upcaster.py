from eve import NodeTranslator

from gt4py.gtc import gtir
from gt4py.gtc.common import LogicalOperator


class _GTIRUpcasting(NodeTranslator):
    """
    Introduces Cast nodes (upcasting) for expr involving different datatypes
    Precondition: all dtypes are resolved (no `None`, `Auto`, `Default`)
    Postcondition: all dtype transitions are explicit via a `Cast` node
    """

    def visit_Expr(self, node: gtir.Expr, **kwargs):
        # this default visit() relies on dtype propagation in the `Expr` node validators
        # needs to be specialized for `Expr`s where node.dtype != children.dtype
        # (e.g. BinaryOp with ComparisonOperator, where node.dtype=Bool,
        # but children have other dtype)
        if (
            "target_dtype" in kwargs
            and kwargs["target_dtype"]
            and kwargs["target_dtype"] != node.dtype
        ):
            return gtir.Cast(
                expr=self.generic_visit(node, target_dtype=node.dtype),
                dtype=kwargs["target_dtype"],
            )
        else:
            return self.generic_visit(node, target_dtype=node.dtype)

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
        target_dtype = max(node.left.dtype, node.right.dtype)
        if isinstance(node.op, LogicalOperator):
            target_dtype = None  # don't cast, if they are not both boolean it's an error
        return self.generic_visit(node, target_dtype=target_dtype)


def upcast(node: gtir.Stencil):
    return _GTIRUpcasting().visit(node)
