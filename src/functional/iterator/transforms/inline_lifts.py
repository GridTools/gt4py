from collections.abc import Callable
from typing import Optional

from eve import NodeTranslator
from functional.iterator import ir


class InlineLifts(NodeTranslator):
    def __init__(self, predicate: Optional[Callable[[int], bool]] = None) -> None:
        super().__init__()
        if predicate is None:
            self.predicate = lambda x: True
        else:
            self.predicate = predicate

    def visit_FunCall(self, node: ir.FunCall) -> ir.FunCall:
        if isinstance(node.fun, ir.SymRef) and node.fun.id == "deref":
            assert len(node.args) == 1
            if (
                isinstance(node.args[0], ir.FunCall)
                and isinstance(node.args[0].fun, ir.FunCall)
                and node.args[0].fun.fun == ir.SymRef(id="lift")
                and self.predicate(node.args[0].fun)
            ):
                # deref(lift(f)(args...)) -> f(args...)
                assert len(node.args[0].fun.args) == 1
                f = self.visit(node.args[0].fun.args[0])
                args = self.visit(node.args[0].args)
                return ir.FunCall(fun=f, args=args)
            elif (
                isinstance(node.args[0], ir.FunCall)
                and isinstance(node.args[0].fun, ir.FunCall)
                and node.args[0].fun.fun == ir.SymRef(id="shift")
                and isinstance(node.args[0].args[0], ir.FunCall)
                and isinstance(node.args[0].args[0].fun, ir.FunCall)
                and node.args[0].args[0].fun.fun == ir.SymRef(id="lift")
                and self.predicate(node.args[0].args[0].fun)
            ):
                # deref(shift(...)(lift(f)(args...)) -> f(shift(...)(args)...)
                f = self.visit(node.args[0].args[0].fun.args[0])
                shift = self.visit(node.args[0].fun)
                args = self.visit(node.args[0].args[0].args)
                return ir.FunCall(fun=f, args=[ir.FunCall(fun=shift, args=[arg]) for arg in args])
        return self.generic_visit(node)
