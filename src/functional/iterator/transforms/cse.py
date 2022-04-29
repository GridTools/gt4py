from collections import Counter

from eve import NodeTranslator, NodeVisitor
from eve.utils import UIDs
from functional.iterator import ir


class CollectSubexpressions(NodeVisitor):
    def visit_Lambda(self, node: ir.Lambda, *, subexprs: Counter):
        self.generic_visit(node, subexprs=subexprs)

        params = {p.id for p in node.params}

        for expr in list(subexprs.keys()):
            refs = expr.iter_tree().if_isinstance(ir.SymRef).getattr("id").to_set()
            if refs & params:
                del subexprs[expr]

        subexprs[node] += 1

    def visit_FunCall(self, node: ir.Lambda, *, subexprs: Counter):
        self.generic_visit(node, subexprs=subexprs)
        if node.fun == ir.SymRef(id="shift"):
            return
        subexprs[node] += 1


class CSE(NodeTranslator):
    def visit_FunCall(self, node):
        node = self.generic_visit(node)

        subexprs = Counter()
        CollectSubexpressions().visit(node, subexprs=subexprs)
        expr_map = {
            expr: UIDs.sequential_id(prefix="_cs") for expr, count in subexprs.items() if count > 1
        }
        if not expr_map:
            return node

        class Replace(NodeTranslator):
            def visit_Expr(self, node):
                if node in expr_map:
                    return ir.SymRef(id=expr_map[node])
                return self.generic_visit(node)

        return ir.FunCall(
            fun=ir.Lambda(
                params=[ir.Sym(id=p) for p in expr_map.values()], expr=Replace().visit(node)
            ),
            args=list(expr_map.keys()),
        )
