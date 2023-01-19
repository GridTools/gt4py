import dataclasses
from collections.abc import Iterable, Iterator
from typing import TypeGuard

from eve import NodeTranslator
from eve.utils import UIDGenerator
from functional import common
from functional.iterator import ir
from functional.iterator.transforms.deduce_conn_of_reductions import DeduceConnOfReductions


def _is_shifted(arg: ir.Expr) -> TypeGuard[ir.FunCall]:
    return (
        isinstance(arg, ir.FunCall)
        and isinstance(arg.fun, ir.FunCall)
        and arg.fun.fun == ir.SymRef(id="shift")
    )


def _is_applied_lift(arg: ir.Expr) -> TypeGuard[ir.FunCall]:
    return (
        isinstance(arg, ir.FunCall)
        and isinstance(arg.fun, ir.FunCall)
        and arg.fun.fun == ir.SymRef(id="lift")
    )


def _is_shifted_or_lifted_and_shifted(arg: ir.Expr) -> TypeGuard[ir.FunCall]:
    return _is_shifted(arg) or (
        _is_applied_lift(arg)
        and any(_is_shifted_or_lifted_and_shifted(nested_arg) for nested_arg in arg.args)
    )


def _get_shifted_args(reduce_args: Iterable[ir.Expr]) -> Iterator[ir.FunCall]:
    return filter(
        _is_shifted_or_lifted_and_shifted,
        reduce_args,
    )


def _is_reduce(node: ir.FunCall):
    return isinstance(node.fun, ir.FunCall) and node.fun.fun == ir.SymRef(id="reduce")


def _make_shift(offsets: list[ir.Expr], iterator: ir.Expr):
    return ir.FunCall(fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=offsets), args=[iterator])


def _make_deref(iterator: ir.Expr):
    return ir.FunCall(fun=ir.SymRef(id="deref"), args=[iterator])


def _make_can_deref(iterator: ir.Expr):
    return ir.FunCall(fun=ir.SymRef(id="can_deref"), args=[iterator])


def _make_if(cond: ir.Expr, true_expr: ir.Expr, false_expr: ir.Expr):
    return ir.FunCall(
        fun=ir.SymRef(id="if_"),
        args=[cond, true_expr, false_expr],
    )


def _get_connectivity(partial_offsets: Iterable[str], offset_provider) -> common.Connectivity:
    connectivities = []
    for o in partial_offsets:
        connectivities.append(offset_provider[o])

    if not connectivities:
        raise RuntimeError("Couldn't detect partial shift in any arguments of reduce.")

    if len({(c.max_neighbors, c.has_skip_values) for c in connectivities}) != 1:
        # The condition for this check is required but not sufficient: the actual neighbor tables could still be incompatible.
        raise RuntimeError("Arguments to reduce have incompatible partial shifts.")
    return connectivities[0]


@dataclasses.dataclass(frozen=True)
class UnrollReduce(NodeTranslator):
    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    @classmethod
    def apply(cls, node: ir.Node, **kwargs):
        return cls().visit(node, **kwargs)

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        # we lose annex here: node = self.generic_visit(node, **kwargs)
        if not _is_reduce(node):
            return self.generic_visit(node)

        offset_provider = kwargs["offset_provider"]
        assert offset_provider is not None
        connectivity = _get_connectivity(node.annex.reduction_offsets, offset_provider)
        max_neighbors = connectivity.max_neighbors
        has_skip_values = connectivity.has_skip_values

        node = self.generic_visit(node)  # here it's safe as annex is consumed

        acc = ir.SymRef(id=self.uids.sequential_id(prefix="_acc"))
        offset = ir.SymRef(id=self.uids.sequential_id(prefix="_i"))
        step = ir.SymRef(id=self.uids.sequential_id(prefix="_step"))

        assert isinstance(node.fun, ir.FunCall)
        fun, init = node.fun.args

        derefed_shifted_args = [_make_deref(_make_shift([offset], arg)) for arg in node.args]
        step_fun: ir.Expr = ir.FunCall(fun=fun, args=[acc] + derefed_shifted_args)
        if has_skip_values:
            check_arg = next(_get_shifted_args(node.args))
            can_deref = _make_can_deref(_make_shift([offset], check_arg))
            step_fun = _make_if(can_deref, step_fun, acc)
        step_fun = ir.Lambda(params=[ir.Sym(id=acc.id), ir.Sym(id=offset.id)], expr=step_fun)
        expr = init
        for i in range(max_neighbors):
            expr = ir.FunCall(fun=step, args=[expr, ir.OffsetLiteral(value=i)])
        expr = ir.FunCall(fun=ir.Lambda(params=[ir.Sym(id=step.id)], expr=expr), args=[step_fun])

        return expr


def apply_unroll_reduce(node: ir.Node, offset_provider):
    deduced_conns = DeduceConnOfReductions.apply(node)
    return UnrollReduce.apply(deduced_conns, offset_provider=offset_provider)
