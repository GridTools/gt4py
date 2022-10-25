import dataclasses
from collections.abc import Iterable

from eve import NodeTranslator
from eve.utils import UIDGenerator
from functional.iterator import ir


@dataclasses.dataclass(frozen=True)
class UnrollReduce(NodeTranslator):
    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    @classmethod
    def apply(cls, node: ir.Node, **kwargs):
        return cls().visit(node, **kwargs)

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
        connectivity = offset_provider[
            node.fun.args[2].value[:-3]
        ]  # TODO(tehrengruber): find a better way to remove Dim
        max_neighbors = connectivity.max_neighbors
        has_skip_values = connectivity.has_skip_values

        acc = ir.SymRef(id=self.uids.sequential_id(prefix="_acc"))
        offset = ir.SymRef(id=self.uids.sequential_id(prefix="_i"))
        step = ir.SymRef(id=self.uids.sequential_id(prefix="_step"))
        shifted_args = [
            ir.SymRef(id=self.uids.sequential_id(prefix="_shifted_arg")) for _ in node.args
        ]

        assert isinstance(node.fun, ir.FunCall)
        fun, init, axis = node.fun.args

        shifted_args_expr = [self._make_shift([axis, offset], arg) for arg in node.args]
        derefed_shifted_args = [self._make_deref(shifted_arg) for shifted_arg in shifted_args]
        step_fun: ir.Expr = ir.FunCall(fun=fun, args=[acc] + derefed_shifted_args)
        if has_skip_values:
            can_deref = self._make_can_deref(shifted_args[0])
            step_fun = self._make_if(can_deref, step_fun, acc)
        step_fun = ir.FunCall(
            fun=ir.Lambda(
                params=[ir.Sym(id=shifted_arg.id) for shifted_arg in shifted_args], expr=step_fun
            ),
            args=shifted_args_expr,
        )
        step_fun = ir.Lambda(params=[ir.Sym(id=acc.id), ir.Sym(id=offset.id)], expr=step_fun)
        expr = init
        for i in range(max_neighbors):
            expr = ir.FunCall(fun=step, args=[expr, ir.OffsetLiteral(value=i)])
        expr = ir.FunCall(fun=ir.Lambda(params=[ir.Sym(id=step.id)], expr=expr), args=[step_fun])

        return expr
