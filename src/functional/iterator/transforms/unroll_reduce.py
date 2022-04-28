from eve import NodeTranslator
from functional.iterator import ir


class UnrollReduce(NodeTranslator):
    @staticmethod
    def _get_last_offset(node: ir.FunCall):
        assert isinstance(node.fun, ir.FunCall) and node.fun.fun == ir.SymRef(id="shift")
        return node.fun.args[0]

    @staticmethod
    def _is_reduce(node: ir.FunCall):
        return isinstance(node.fun, ir.FunCall) and node.fun.fun == ir.SymRef(id="reduce")

    @staticmethod
    def _make_shift(offsets: list[ir.OffsetLiteral], iterator: ir.Expr):
        return ir.FunCall(fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=offsets), args=[iterator])

    @staticmethod
    def _make_deref(iterator: ir.Expr):
        return ir.FunCall(fun=ir.SymRef(id="deref"), args=[iterator])

    @staticmethod
    def _make_if(cond: ir.Expr, true_expr: ir.Expr, false_expr: ir.Expr):
        return ir.FunCall(
            fun=ir.SymRef(id="if_"),
            args=[cond, true_expr, false_expr],
        )

    @staticmethod
    def _wrap_can_deref(prev: ir.Expr, current: ir.Expr, i: int, arg: ir.Expr):
        can_deref = ir.FunCall(
            fun=ir.SymRef(id="can_deref"),
            args=[UnrollReduce._make_shift([ir.OffsetLiteral(value=i)], arg)],
        )
        return UnrollReduce._make_if(can_deref, current, prev)

    @staticmethod
    def _make_step(prev: ir.Expr, i: int, fun: ir.Expr, args: list[ir.Expr]):
        return ir.FunCall(
            fun=fun,
            args=[
                prev,
                *map(
                    lambda arg: UnrollReduce._make_deref(
                        UnrollReduce._make_shift([ir.OffsetLiteral(value=i)], arg)
                    ),
                    args,
                ),
            ],
        )

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        node = self.generic_visit(node, **kwargs)
        if not self._is_reduce(node):
            return node

        offset_provider = kwargs["offset_provider"]
        last_offset = self._get_last_offset(node.args[0])
        unroll_factor = offset_provider[last_offset.value].max_neighbors
        has_skip_values = offset_provider[last_offset.value].has_skip_values
        reduce_fun = node.fun.args[0]
        reduce_init = node.fun.args[1]

        expr = reduce_init
        for i in range(unroll_factor):
            prev = expr
            expr = self._make_step(expr, i, reduce_fun, node.args)
            if has_skip_values:
                expr = self._wrap_can_deref(prev, expr, i, node.args[0])
        return expr
