from eve import NodeTranslator
from functional.iterator import ir
from functional.iterator.transforms.remap_symbols import RemapSymbolRefs, RenameSymbols


class InlineLambdas(NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if isinstance(node.fun, ir.Lambda):
            assert len(node.fun.params) == len(node.args)
            refs = (
                set.union(
                    *(
                        arg.pre_walk_values().if_isinstance(ir.SymRef).getattr("id").to_set()
                        for arg in node.args
                    )
                )
                if len(node.args) > 0
                else set()
            )
            syms = node.fun.expr.pre_walk_values().if_isinstance(ir.Sym).getattr("id").to_set()
            clashes = refs & syms
            expr = node.fun.expr
            if clashes:

                def new_name(name):
                    while name in refs or name in syms:
                        name += "_"
                    return name

                name_map = {sym: new_name(sym) for sym in clashes}
                expr = RenameSymbols().visit(expr, name_map=name_map)

            symbol_map = {param.id: arg for param, arg in zip(node.fun.params, node.args)}
            return RemapSymbolRefs().visit(expr, symbol_map=symbol_map)
        return node
