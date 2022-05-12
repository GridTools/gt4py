from eve import NodeTranslator
from functional.iterator import ir


class EtaReduction(NodeTranslator):
    def visit_Lambda(self, node):
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
