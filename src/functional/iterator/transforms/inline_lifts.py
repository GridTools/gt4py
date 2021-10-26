from eve import NodeTranslator
from functional.iterator import ir


class InlineLifts(NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if isinstance(node.fun, ir.SymRef) and node.fun.id == "deref":
            assert len(node.args) == 1
            if (
                isinstance(node.args[0], ir.FunCall)
                and isinstance(node.args[0].fun, ir.FunCall)
                and isinstance(node.args[0].fun.fun, ir.SymRef)
                and node.args[0].fun.fun.id == "lift"
            ):
                # deref(lift(f)(args...)) -> f(args...)
                assert len(node.args[0].fun.args) == 1
                f = node.args[0].fun.args[0]
                args = node.args[0].args
                return ir.FunCall(fun=f, args=args)
            elif (
                isinstance(node.args[0], ir.FunCall)
                and isinstance(node.args[0].fun, ir.FunCall)
                and isinstance(node.args[0].fun.fun, ir.SymRef)
                and node.args[0].fun.fun.id == "shift"
                and isinstance(node.args[0].args[0], ir.FunCall)
                and isinstance(node.args[0].args[0].fun, ir.FunCall)
                and isinstance(node.args[0].args[0].fun.fun, ir.SymRef)
                and node.args[0].args[0].fun.fun.id == "lift"
            ):
                # deref(shift(...)(lift(f)(args...)) -> f(shift(...)(args)...)
                f = node.args[0].args[0].fun.args[0]
                shift = node.args[0].fun
                args = node.args[0].args[0].args
                res = ir.FunCall(fun=f, args=[ir.FunCall(fun=shift, args=[arg]) for arg in args])
                return res
        return node
