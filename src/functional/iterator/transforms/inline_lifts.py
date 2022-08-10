from collections.abc import Callable
from typing import Optional

from eve import NodeTranslator
from functional.iterator import ir


class InlineLifts(NodeTranslator):
    """Inline lifted function calls.

    Optionally a predicate function can be passed which can enable or disable inlining of specific function nodes.
    """

    def __init__(self, predicate: Optional[Callable[[ir.Expr], bool]] = None) -> None:
        super().__init__()
        if predicate is None:
            self.predicate = lambda x: True
        else:
            self.predicate = predicate

    @staticmethod
    def _is_lift(node: ir.Node):
        return (
            isinstance(node, ir.FunCall)
            and isinstance(node.fun, ir.FunCall)
            and node.fun.fun == ir.SymRef(id="lift")
        )

    @staticmethod
    def _is_shift_lift(node: ir.Expr):
        return (
            isinstance(node, ir.FunCall)
            and isinstance(node.fun, ir.FunCall)
            and node.fun.fun == ir.SymRef(id="shift")
            and isinstance(node.args[0], ir.FunCall)
            and isinstance(node.args[0].fun, ir.FunCall)
            and node.args[0].fun.fun == ir.SymRef(id="lift")
        )

    def visit_FunCall(self, node: ir.FunCall):
        if node.fun == ir.SymRef(id="deref"):
            assert len(node.args) == 1
            if self._is_lift(node.args[0]) and self.predicate(node.args[0].fun):  # type: ignore[attr-defined]
                # deref(lift(f)(args...)) -> f(args...)
                assert isinstance(node.args[0], ir.FunCall)
                assert isinstance(node.args[0].fun, ir.FunCall)
                assert len(node.args[0].fun.args) == 1
                f = self.visit(node.args[0].fun.args[0])
                args = self.visit(node.args[0].args)
                return ir.FunCall(fun=f, args=args)
            elif self._is_shift_lift(node.args[0]) and self.predicate(node.args[0].args[0].fun):  # type: ignore[attr-defined]
                # deref(shift(...)(lift(f)(args...)) -> f(shift(...)(args)...)
                assert isinstance(node.args[0], ir.FunCall)
                assert isinstance(node.args[0].args[0], ir.FunCall)
                assert isinstance(node.args[0].args[0].fun, ir.FunCall)
                f = self.visit(node.args[0].args[0].fun.args[0])
                shift = self.visit(node.args[0].fun)
                args = self.visit(node.args[0].args[0].args)
                return ir.FunCall(fun=f, args=[ir.FunCall(fun=shift, args=[arg]) for arg in args])
        if node.fun == ir.SymRef(id="can_deref"):
            # TODO(havogt): this `can_deref` transformation doesn't look into lifted functions, this need to be changed to be 100% compliant
            assert len(node.args) == 1
            if self._is_lift(node.args[0]) and self.predicate(node.args[0].fun):  # type: ignore[attr-defined]
                # can_deref(lift(f)(args...)) -> and(can_deref(arg[0]), and(can_deref(arg[1]), ...))
                assert isinstance(node.args[0], ir.FunCall)
                assert isinstance(node.args[0].fun, ir.FunCall)
                assert len(node.args[0].fun.args) == 1
                args = self.visit(node.args[0].args)
                res = ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[args[0]])
                for arg in args[1:]:
                    res = ir.FunCall(
                        fun=ir.SymRef(id="and_"),
                        args=[res, ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[arg])],
                    )
                return res
            elif self._is_shift_lift(node.args[0]) and self.predicate(node.args[0].args[0].fun):  # type: ignore[attr-defined]
                # can_deref(shift(...)(lift(f)(args...)) -> and(can_deref(shift(...)(arg[0])), and(can_deref(shift(...)(arg[1])), ...))
                assert isinstance(node.args[0], ir.FunCall)
                assert isinstance(node.args[0].args[0], ir.FunCall)
                shift = self.visit(node.args[0].fun)
                args = self.visit(node.args[0].args[0].args)
                res = ir.FunCall(
                    fun=ir.SymRef(id="can_deref"),
                    args=[ir.FunCall(fun=shift, args=[args[0]])],
                )
                for arg in args[1:]:
                    res = ir.FunCall(
                        fun=ir.SymRef(id="and_"),
                        args=[
                            res,
                            ir.FunCall(
                                fun=ir.SymRef(id="can_deref"),
                                args=[ir.FunCall(fun=shift, args=[arg])],
                            ),
                        ],
                    )
                return res

        return self.generic_visit(node)
