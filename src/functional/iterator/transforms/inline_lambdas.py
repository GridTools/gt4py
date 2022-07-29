import dataclasses

from eve import NodeTranslator, NodeVisitor
from functional.iterator import ir
from functional.iterator.transforms.remap_symbols import RemapSymbolRefs, RenameSymbols


@dataclasses.dataclass
class CountSymRefs(NodeVisitor):
    sym_name: str
    count: int = 0

    @classmethod
    def apply(cls, node: ir.Node, sym_name: str):
        obj = cls(sym_name=sym_name, count=0)
        obj.visit(node)
        return obj.count

    def visit_SymRef(self, node: ir.SymRef):
        if node.id == self.sym_name:
            self.count += 1

    def visit_Lambda(self, node: ir.Lambda):
        if any(param.id == self.sym_name for param in node.params):
            self.generic_visit(node)


class InlineLambdas(NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if isinstance(node.fun, ir.Lambda):
            assert len(node.fun.params) == len(node.args)

            all_params_referenced_once = True
            for param in node.fun.params:
                # TODO(tehrengruber): slow
                c = CountSymRefs.apply(node.fun, param.id)
                if c != 1:
                    all_params_referenced_once = False

            primitive = all(isinstance(arg, ir.SymRef) for arg in node.args)

            # only inline when we don't increase the number of operations
            # TODO(tehrengruber): make configurable
            if not primitive and not all_params_referenced_once:
                return node

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
