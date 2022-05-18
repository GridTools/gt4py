from eve import NodeTranslator
from eve.utils import UIDs
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
    def _make_shift(offsets: list[ir.Expr], iterator: ir.Expr):
        return ir.FunCall(fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=offsets), args=[iterator])

    @staticmethod
    def _make_deref(iterator: ir.Expr):
        return ir.FunCall(fun=ir.SymRef(id="deref"), args=[iterator])

    @staticmethod
    def _make_can_deref(iterator: ir.Expr):
        return ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[iterator])

    @staticmethod
    def _make_if(cond: ir.Expr, true_expr: ir.Expr, false_expr: ir.Expr):
        return ir.FunCall(
            fun=ir.SymRef(id="if_"),
            args=[cond, true_expr, false_expr],
        )

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        node = self.generic_visit(node, **kwargs)
        if not self._is_reduce(node):
            return node

        offset_provider = kwargs["offset_provider"]
        assert offset_provider is not None
        last_offset = self._get_last_offset(node.args[0])
        max_neighbors = offset_provider[last_offset.value].max_neighbors
        has_skip_values = offset_provider[last_offset.value].has_skip_values

        acc = ir.SymRef(id=UIDs.sequential_id(prefix="_acc"))
        offset = ir.SymRef(id=UIDs.sequential_id(prefix="_i"))
        step = ir.SymRef(id=UIDs.sequential_id(prefix="_step"))

        fun, init = node.fun.args

        derefed_shifted_args = [
            self._make_deref(self._make_shift([offset], arg)) for arg in node.args
        ]
        step_fun = ir.FunCall(fun=fun, args=[acc] + derefed_shifted_args)
        if has_skip_values:
            can_deref = self._make_can_deref(self._make_shift([offset], node.args[0]))
            step_fun = self._make_if(can_deref, step_fun, acc)
        step_fun = ir.Lambda(params=[ir.Sym(id=acc.id), ir.Sym(id=offset.id)], expr=step_fun)
        expr = init
        for i in range(max_neighbors):
            expr = ir.FunCall(fun=step, args=[expr, ir.OffsetLiteral(value=i)])
        expr = ir.FunCall(fun=ir.Lambda(params=[ir.Sym(id=step.id)], expr=expr), args=[step_fun])

        return expr
