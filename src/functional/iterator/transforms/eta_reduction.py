from eve import NodeTranslator
from functional.iterator import ir


class EtaReduction(NodeTranslator):
    """Eta reduction: simplifies `λ(args...) → f(args...)` to `f`."""

    def visit_Lambda(self, node: ir.Lambda) -> ir.Node:
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
