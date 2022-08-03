from dataclasses import dataclass

from eve import NodeTranslator
from functional.iterator import ir


def _is_lift(node: ir.Node) -> bool:
    return (
        isinstance(node, ir.FunCall)
        and isinstance(node.fun, ir.FunCall)
        and node.fun.fun == ir.SymRef(id="lift")
    )


def _is_shift_lift(node: ir.Expr) -> bool:
    return (
        isinstance(node, ir.FunCall)
        and isinstance(node.fun, ir.FunCall)
        and node.fun.fun == ir.SymRef(id="shift")
        and isinstance(node.args[0], ir.FunCall)
        and isinstance(node.args[0].fun, ir.FunCall)
        and node.args[0].fun.fun == ir.SymRef(id="lift")
    )


@dataclass
class InlineLifts(NodeTranslator):
    preserve_shift_count: bool

    @classmethod
    def apply(cls, node: ir.Node, *, preserve_shift_count=False):
        return cls(preserve_shift_count=preserve_shift_count).visit(node)

    def visit_FunCall(self, node: ir.FunCall) -> ir.FunCall:
        node = self.generic_visit(node)
        if node.fun == ir.SymRef(id="deref"):
            assert len(node.args) == 1
            if _is_lift(node.args[0]):
                # deref(lift(f)(args...)) -> f(args...)
                assert isinstance(node.args[0], ir.FunCall)
                assert isinstance(node.args[0].fun, ir.FunCall)
                assert len(node.args[0].fun.args) == 1
                f = node.args[0].fun.args[0]
                args = node.args[0].args
                return ir.FunCall(fun=f, args=args)
            elif _is_shift_lift(node.args[0]) and (not self.preserve_shift_count or len(node.args[0].args[0].args) == 1):
                # deref(shift(...)(lift(f)(args...))) -> f(shift(...)(args)...)
                assert isinstance(node.args[0], ir.FunCall)
                assert isinstance(node.args[0].args[0], ir.FunCall)
                assert isinstance(node.args[0].args[0].fun, ir.FunCall)
                f = node.args[0].args[0].fun.args[0]
                shift = node.args[0].fun
                args = node.args[0].args[0].args
                res = ir.FunCall(fun=f, args=[ir.FunCall(fun=shift, args=[arg]) for arg in args])
                return res
        if node.fun == ir.SymRef(id="can_deref"):
            # TODO(havogt): this `can_deref` transformation doesn't look into lifted functions, this need to be changed to be 100% compliant
            assert len(node.args) == 1
            if _is_lift(node.args[0]):
                # can_deref(lift(f)(args...)) -> and(can_deref(arg[0]), and(can_deref(arg[1]), ...))
                assert isinstance(node.args[0], ir.FunCall)
                assert isinstance(node.args[0].fun, ir.FunCall)
                assert len(node.args[0].fun.args) == 1
                args = node.args[0].args
                res = ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[args[0]])
                for arg in args[1:]:
                    res = ir.FunCall(
                        fun=ir.SymRef(id="and_"),
                        args=[res, ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[arg])],
                    )
                return res
            elif _is_shift_lift(node.args[0]) and (not self.preserve_shift_count or len(node.args[0].args[0].args) == 1):
                # can_deref(shift(...)(lift(f)(args...)) -> and(can_deref(shift(...)(arg[0])), and(can_deref(shift(...)(arg[1])), ...))
                assert isinstance(node.args[0], ir.FunCall)
                assert isinstance(node.args[0].args[0], ir.FunCall)
                shift = node.args[0].fun
                args = node.args[0].args[0].args
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

        return node
