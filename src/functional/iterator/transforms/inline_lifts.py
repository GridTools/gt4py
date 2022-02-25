from eve import NodeTranslator
from functional.iterator import ir


class InlineLifts(NodeTranslator):
    @staticmethod
    def _is_lift(node: ir.Node):
        return (
            isinstance(node, ir.FunCall)
            and isinstance(node.fun, ir.FunCall)
            and isinstance(node.fun.fun, ir.SymRef)
            and node.fun.fun.id == "lift"
        )

    @staticmethod
    def _is_shift_lift(node: ir.FunCall):
        return (
            isinstance(node, ir.FunCall)
            and isinstance(node.fun, ir.FunCall)
            and isinstance(node.fun.fun, ir.SymRef)
            and node.fun.fun.id == "shift"
            and isinstance(node.args[0], ir.FunCall)
            and isinstance(node.args[0].fun, ir.FunCall)
            and isinstance(node.args[0].fun.fun, ir.SymRef)
            and node.args[0].fun.fun.id == "lift"
        )

    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if isinstance(node.fun, ir.SymRef) and node.fun.id == "deref":
            assert len(node.args) == 1
            if self._is_lift(node.args[0]):
                # deref(lift(f)(args...)) -> f(args...)
                assert len(node.args[0].fun.args) == 1
                f = node.args[0].fun.args[0]
                args = node.args[0].args
                return ir.FunCall(fun=f, args=args)
            elif self._is_shift_lift(node.args[0]):
                # deref(shift(...)(lift(f)(args...)) -> f(shift(...)(args)...)
                f = node.args[0].args[0].fun.args[0]
                shift = node.args[0].fun
                args = node.args[0].args[0].args
                res = ir.FunCall(fun=f, args=[ir.FunCall(fun=shift, args=[arg]) for arg in args])
                return res
        if isinstance(node.fun, ir.SymRef) and node.fun.id == "can_deref":
            assert len(node.args) == 1
            if self._is_lift(node.args[0]):
                # can_deref(lift(f)(args...)) -> and(can_deref(arg[0]), and(can_deref(arg[1]), ...))
                assert len(node.args[0].fun.args) == 1
                args = node.args[0].args
                res = ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[args[0]])
                for arg in args[1:]:
                    res = ir.FunCall(
                        fun=ir.SymRef(id="and_"),
                        args=[res, ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[arg])],
                    )
                return res
            elif self._is_shift_lift(node.args[0]):
                # can_deref(shift(...)(lift(f)(args...)) -> and(can_deref(shift(...)(arg[0])), and(can_deref(shift(...)(arg[1])), ...))
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
