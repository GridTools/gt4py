from eve import NodeTranslator
from functional.iterator import ir


class EtaReduction(NodeTranslator):
    """Eta reduction: simplifies `λ(args...) → f(args...)` to `f`."""

    def visit_Lambda(self, node: ir.Lambda, **kwargs) -> ir.Node:
        """Do not apply eta "inside" reductions."""
        if "in_reduce" in kwargs and kwargs["in_reduce"]:
            return self.generic_visit(node)

        if (
            isinstance(node.expr, ir.FunCall)
            and len(node.params) == len(node.expr.args)
            and all(
                isinstance(a, ir.SymRef) and p.id == a.id
                for p, a in zip(node.params, node.expr.args)
            )
        ):
            return self.visit(node.expr.fun)

        return self.generic_visit(node)

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Node:
        if isinstance(node.fun, ir.SymRef) and node.fun.id == "reduce":
            fun = self.visit(node.fun, in_reduce=True)
            args = [self.visit(arg, in_reduce=True) for arg in node.args]
            return ir.FunCall(fun=fun, args=args)
        return self.generic_visit(node, in_reduce=False)
