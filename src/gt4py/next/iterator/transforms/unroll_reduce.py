# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from collections.abc import Iterable, Iterator
from typing import TypeGuard

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.eve.utils import UIDGenerator
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.ir_utils.common_pattern_matcher import is_applied_lift


def _is_neighbors(arg: itir.Expr) -> TypeGuard[itir.FunCall]:
    return isinstance(arg, itir.FunCall) and arg.fun == itir.SymRef(id="neighbors")


def _is_neighbors_or_lifted_and_neighbors(arg: itir.Expr) -> TypeGuard[itir.FunCall]:
    return _is_neighbors(arg) or (
        is_applied_lift(arg)
        and any(_is_neighbors_or_lifted_and_neighbors(nested_arg) for nested_arg in arg.args)
    )


def _get_neighbors_args(reduce_args: Iterable[itir.Expr]) -> Iterator[itir.FunCall]:
    flat_reduce_args: list[itir.Expr] = []
    for arg in reduce_args:
        if cpm.is_call_to(arg, "if_"):
            flat_reduce_args.extend(_get_neighbors_args(arg.args[1:3]))
        else:
            flat_reduce_args.append(arg)

    return filter(_is_neighbors_or_lifted_and_neighbors, flat_reduce_args)


def _is_list_of_funcalls(lst: list) -> TypeGuard[list[itir.FunCall]]:
    return all(isinstance(f, itir.FunCall) for f in lst)


def _get_partial_offset_tag(arg: itir.FunCall) -> str:
    if _is_neighbors(arg):
        offset = arg.args[0]
        assert isinstance(offset, itir.OffsetLiteral)
        assert isinstance(offset.value, str)
        return offset.value
    else:
        assert is_applied_lift(arg)
        assert _is_list_of_funcalls(arg.args)
        partial_offsets = [_get_partial_offset_tag(arg) for arg in arg.args]
        assert all(o == partial_offsets[0] for o in partial_offsets)
        return partial_offsets[0]


def _get_partial_offset_tags(reduce_args: Iterable[itir.Expr]) -> Iterable[str]:
    return [_get_partial_offset_tag(arg) for arg in _get_neighbors_args(reduce_args)]


def _get_connectivity(
    applied_reduce_node: itir.FunCall,
    offset_provider_type: common.OffsetProviderType,
) -> common.NeighborConnectivityType:
    """Return single connectivity that is compatible with the arguments of the reduce."""
    if not cpm.is_applied_reduce(applied_reduce_node):
        raise ValueError("Expected a call to a 'reduce' object, i.e. 'reduce(...)(...)'.")

    connectivities: list[common.NeighborConnectivityType] = []
    for o in _get_partial_offset_tags(applied_reduce_node.args):
        conn = offset_provider_type[o]
        assert isinstance(conn, common.NeighborConnectivityType)
        connectivities.append(conn)

    if not connectivities:
        raise RuntimeError("Couldn't detect partial shift in any arguments of 'reduce'.")

    if len({(c.max_neighbors, c.has_skip_values) for c in connectivities}) != 1:
        # The condition for this check is required but not sufficient: the actual neighbor tables could still be incompatible.
        raise RuntimeError("Arguments to 'reduce' have incompatible partial shifts.")
    return connectivities[0]


def _make_shift(offsets: list[itir.Expr], iterator: itir.Expr) -> itir.FunCall:
    return itir.FunCall(
        fun=itir.FunCall(fun=itir.SymRef(id="shift"), args=offsets),
        args=[iterator],
        location=iterator.location,
    )


def _make_deref(iterator: itir.Expr) -> itir.FunCall:
    return itir.FunCall(fun=itir.SymRef(id="deref"), args=[iterator], location=iterator.location)


def _make_can_deref(iterator: itir.Expr) -> itir.FunCall:
    return itir.FunCall(
        fun=itir.SymRef(id="can_deref"), args=[iterator], location=iterator.location
    )


def _make_if(cond: itir.Expr, true_expr: itir.Expr, false_expr: itir.Expr) -> itir.FunCall:
    return itir.FunCall(
        fun=itir.SymRef(id="if_"), args=[cond, true_expr, false_expr], location=cond.location
    )


def _make_list_get(offset: itir.Expr, expr: itir.Expr) -> itir.FunCall:
    return itir.FunCall(fun=itir.SymRef(id="list_get"), args=[offset, expr], location=expr.location)


@dataclasses.dataclass(frozen=True)
class UnrollReduce(PreserveLocationVisitor, NodeTranslator):
    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    @classmethod
    def apply(cls, node: itir.Node, offset_provider_type: common.OffsetProviderType) -> itir.Node:
        return cls().visit(node, offset_provider_type=offset_provider_type)

    def _visit_reduce(
        self, node: itir.FunCall, offset_provider_type: common.OffsetProviderType
    ) -> itir.Expr:
        connectivity_type = _get_connectivity(node, offset_provider_type)
        max_neighbors = connectivity_type.max_neighbors
        has_skip_values = connectivity_type.has_skip_values

        acc = itir.SymRef(id=self.uids.sequential_id(prefix="_acc"))
        offset = itir.SymRef(id=self.uids.sequential_id(prefix="_i"))
        step = itir.SymRef(id=self.uids.sequential_id(prefix="_step"))

        assert isinstance(node.fun, itir.FunCall)
        fun, init = node.fun.args

        elems = [_make_list_get(offset, arg) for arg in node.args]
        step_fun: itir.Expr = itir.FunCall(fun=fun, args=[acc, *elems])
        if has_skip_values:
            check_arg = next(_get_neighbors_args(node.args))
            offset_tag, it = check_arg.args
            can_deref = _make_can_deref(_make_shift([offset_tag, offset], it))
            step_fun = _make_if(can_deref, step_fun, acc)
        step_fun = itir.Lambda(params=[itir.Sym(id=acc.id), itir.Sym(id=offset.id)], expr=step_fun)
        expr = init
        for i in range(max_neighbors):
            expr = itir.FunCall(fun=step, args=[expr, itir.OffsetLiteral(value=i)])
        expr = itir.FunCall(
            fun=itir.Lambda(params=[itir.Sym(id=step.id)], expr=expr), args=[step_fun]
        )

        return expr

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> itir.Expr:
        node = self.generic_visit(node, **kwargs)
        if cpm.is_applied_reduce(node):
            return self._visit_reduce(node, **kwargs)
        return node
