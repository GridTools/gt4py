from collections import ChainMap
from typing import Optional

from eve import NodeTranslator, NodeVisitor
from eve.utils import UIDs
from functional.iterator import ir


class CollectSubexpressions(NodeVisitor):
    @classmethod
    def apply(cls, node: ir.Node):
        subexprs = dict[ir.Node, tuple[list[int], Optional[ir.Node]]]()
        refs: ChainMap[str, bool] = ChainMap()

        cls().visit(node, subexprs=subexprs, refs=refs, parent=None)

        return subexprs

    def visit_SymRef(
        self,
        node: ir.SymRef,
        *,
        refs: ChainMap[str, bool],
        **kwargs,
    ) -> None:
        if node.id in refs:
            refs[node.id] = True

    def visit_Lambda(
        self,
        node: ir.Lambda,
        *,
        subexprs: dict[ir.Node, tuple[list[int], Optional[ir.Node]]],
        refs: ChainMap[str, bool],
        parent: Optional[ir.Node],
    ) -> None:
        r = refs.new_child({p.id: False for p in node.params})
        self.generic_visit(node, subexprs=subexprs, refs=r, parent=node)

        if not any(refs.maps[0].values()):
            subexprs.setdefault(node, ([], parent))[0].append(id(node))

    def visit_FunCall(
        self,
        node: ir.Lambda,
        *,
        subexprs: dict[ir.Node, tuple[list[int], Optional[ir.Node]]],
        refs: ChainMap[str, bool],
        parent: Optional[ir.Node],
    ) -> None:
        self.generic_visit(node, subexprs=subexprs, refs=refs, parent=node)
        # do not collect (and thus deduplicate in CSE) shift(offsets…) calls
        if node.fun == ir.SymRef(id="shift"):
            return
        if not any(refs.maps[0].values()):
            subexprs.setdefault(node, ([], parent))[0].append(id(node))


class CommonSubexpressionElimination(NodeTranslator):
    """
    Perform common subexpression elimination.

    Examples:
        >>> x = ir.SymRef(id="x")
        >>> plus_ = lambda a, b: ir.FunCall(fun=ir.SymRef(id=("plus")), args=[a, b])
        >>> expr = plus_(plus_(x, x), plus_(x, x))
        >>> print(CommonSubexpressionElimination().visit(expr))
        (λ(_cs_1) → _cs_1 + _cs_1)(x + x)
    """

    def visit_FunCall(self, node: ir.FunCall):
        if isinstance(node.fun, ir.SymRef) and node.fun.id in [
            "cartesian_domain",
            "unstructured_domain",
        ]:
            return node

        node = self.generic_visit(node)

        # collect expressions
        subexprs = CollectSubexpressions.apply(node)

        # collect multiple occurrences and map them to fresh symbols
        expr_map = dict[int, ir.SymRef]()
        params = []
        args = []
        for expr, (ids, parent) in subexprs.items():
            if len(ids) > 1:
                # ignore if parent will be eliminated anyway
                if parent and parent in subexprs and len(subexprs[parent][0]) > 1:
                    continue
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
