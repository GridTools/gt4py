# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
from collections.abc import Iterable, Iterator
from typing import TypeGuard

from functools import reduce
from gt4py.eve import NodeTranslator
from gt4py.eve.utils import UIDGenerator
from gt4py.next.iterator import ir as itir


def _is_shifted(arg: itir.Expr) -> TypeGuard[itir.FunCall]:
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and arg.fun.fun == itir.SymRef(id="shift")
    )


def _is_applied_lift(arg: itir.Expr) -> TypeGuard[itir.FunCall]:
    return (
        isinstance(arg, itir.FunCall)
        and isinstance(arg.fun, itir.FunCall)
        and arg.fun.fun == itir.SymRef(id="lift")
    )


def _is_shifted_or_lifted_and_shifted(arg: itir.Expr) -> TypeGuard[itir.FunCall]:
    return _is_shifted(arg) or (
        _is_applied_lift(arg)
        and any(_is_shifted_or_lifted_and_shifted(nested_arg) for nested_arg in arg.args)
    )


def _get_shifted_args(reduce_args: Iterable[itir.Expr]) -> Iterator[itir.FunCall]:
    return filter(
        _is_shifted_or_lifted_and_shifted,
        reduce_args,
    )


def _is_reduce(node: itir.FunCall):
    return isinstance(node.fun, itir.FunCall) and node.fun.fun == itir.SymRef(id="reduce")


def _make_shift(offsets: list[itir.Expr], iterator: itir.Expr):
    return itir.FunCall(
        fun=itir.FunCall(fun=itir.SymRef(id="shift"), args=offsets), args=[iterator]
    )


def _make_deref(iterator: itir.Expr):
    return itir.FunCall(fun=itir.SymRef(id="deref"), args=[iterator])


def _make_can_deref(iterator: itir.Expr):
    return itir.FunCall(fun=itir.SymRef(id="can_deref"), args=[iterator])


def _make_if(cond: itir.Expr, true_expr: itir.Expr, false_expr: itir.Expr):
    return itir.FunCall(
        fun=itir.SymRef(id="if_"),
        args=[cond, true_expr, false_expr],
    )


@dataclasses.dataclass(frozen=True)
class UnrollReduce(NodeTranslator):
    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    @classmethod
    def apply(cls, node:itir.Node, **kwargs):
        return cls().visit(node, **kwargs)

    def visit_FunCall(self, node:itir.FunCall, **kwargs):
        node = self.generic_visit(node, **kwargs)
        if not _is_reduce(node):
            return node

        assert isinstance(node.fun,itir.FunCall)
        assert len(node.fun.args) == 3
        assert isinstance(node.fun.args[2],itir.OffsetLiteral) and isinstance(
            node.fun.args[2].value, str
        )

        offset_provider = kwargs["offset_provider"]
        assert offset_provider is not None
        connectivity = offset_provider[
            node.fun.args[2].value[:-3]
        ]  # TODO(tehrengruber): find a better way to remove Dim
        max_neighbors = connectivity.max_neighbors
        has_skip_values = connectivity.has_skip_values

        acc =itir.SymRef(id=self.uids.sequential_id(prefix="_acc"))
        offset =itir.SymRef(id=self.uids.sequential_id(prefix="_i"))
        step =itir.SymRef(id=self.uids.sequential_id(prefix="_step"))
        shifted_args = [
           itir.SymRef(id=self.uids.sequential_id(prefix="_shifted_arg")) for _ in node.args
        ]

        assert isinstance(node.fun,itir.FunCall)
        fun, init, axis = node.fun.args

        shifted_args_expr = [_make_shift([axis, offset], arg) for arg in node.args]
        derefed_shifted_args = [_make_deref(shifted_arg) for shifted_arg in shifted_args]
        step_fun: itir.Expr =itir.FunCall(fun=fun, args=[acc] + derefed_shifted_args)
        if has_skip_values:
            can_deref = reduce(
                lambda prev, el:itir.FunCall(fun=itir.SymRef("and"), args=[prev, _make_can_deref(el)]),
                shifted_args[1:],
                _make_can_deref(shifted_args[0]),
            )
            step_fun = _make_if(can_deref, step_fun, acc)
        step_fun =itir.FunCall(
            fun=itir.Lambda(
                params=[itir.Sym(id=shifted_arg.id) for shifted_arg in shifted_args], expr=step_fun
            ),
            args=shifted_args_expr,
        )
        step_fun =itir.Lambda(params=[itir.Sym(id=acc.id), itir.Sym(id=offset.id)], expr=step_fun)
        expr = init
        for i in range(max_neighbors):
            expr =itir.FunCall(fun=step, args=[expr,itir.OffsetLiteral(value=i)])
        expr =itir.FunCall(fun=itir.Lambda(params=[itir.Sym(id=step.id)], expr=expr), args=[step_fun])

        return expr
