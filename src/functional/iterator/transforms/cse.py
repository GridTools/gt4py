from collections import ChainMap

from eve import NodeTranslator, NodeVisitor
from eve.utils import UIDs
from functional.iterator import ir


class CollectSubexpressions(NodeVisitor):
    def visit_SymRef(
        self, node: ir.SymRef, *, subexprs: dict[ir.Node, list[int]], refs: ChainMap[str, bool]
    ) -> None:
        if node.id in refs:
            refs[node.id] = True

    def visit_Lambda(
        self, node: ir.Lambda, *, subexprs: dict[ir.Node, list[int]], refs: ChainMap[str, bool]
    ) -> None:
        r = refs.new_child({p.id: False for p in node.params})
        self.generic_visit(node, subexprs=subexprs, refs=r)

        if not any(refs.maps[0].values()):
            subexprs.setdefault(node, []).append(id(node))

    def visit_FunCall(
        self, node: ir.Lambda, *, subexprs: dict[ir.Node, list[int]], refs: ChainMap[str, bool]
    ) -> None:
        self.generic_visit(node, subexprs=subexprs, refs=refs)
        # do not collect (and thus deduplicate in CSE) shift(offsets…) calls
        if node.fun == ir.SymRef(id="shift"):
            return
        if not any(refs.maps[0].values()):
            subexprs.setdefault(node, []).append(id(node))


class CSE(NodeTranslator):
    def visit_FunCall(self, node):
        node = self.generic_visit(node)

        # collect expressions
        subexprs = dict[ir.Node, list[int]]()
        refs = ChainMap()
        CollectSubexpressions().visit(node, subexprs=subexprs, refs=refs)

        # collect multiple occurrences and map them to fresh symbols
        expr_map = dict[int, ir.SymRef]()
        params = []
        args = []
        for expr, ids in subexprs.items():
            if len(ids) > 1:
                expr_id = UIDs.sequential_id(prefix="_cs")
                params.append(ir.Sym(id=expr_id))
                args.append(expr)
                expr_ref = ir.SymRef(id=expr_id)
                for i in ids:
                    expr_map[i] = expr_ref

        if not expr_map:
            return node

        # apply remapping
        class Replace(NodeTranslator):
            def visit_Expr(self, node):
                if id(node) in expr_map:
                    return expr_map[id(node)]
                return self.generic_visit(node)

        return ir.FunCall(
            fun=ir.Lambda(params=params, expr=Replace().visit(node)),
            args=args,
        )
