# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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
import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Iterable, Literal

from gt4py.eve import NodeTranslator
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.collect_shifts import ALL_NEIGHBORS


class Sentinel(enum.Enum):
    VALUE = object()
    TYPE = object()


class IteratorTracer:
    pass


@dataclass(frozen=True)
class InputTracer(IteratorTracer):
    inp: str
    register_deref: Callable[[str, tuple[ir.OffsetLiteral, ...]], None]
    offsets: tuple[ir.OffsetLiteral, ...] = ()
    lift_level: int = 0

    def shift(self, offsets):
        return InputTracer(
            inp=self.inp,
            register_deref=self.register_deref,
            offsets=self.offsets + tuple(offsets),
            lift_level=self.lift_level,
        )

    def deref(self):
        self.register_deref(self.inp, self.offsets)
        return Sentinel.VALUE


@dataclass(frozen=True)
class CombinedTracer(IteratorTracer):
    its: tuple[IteratorTracer, ...]

    def shift(self, offsets):
        return CombinedTracer(tuple(_shift(*offsets)(it) for it in self.its))

    def deref(self):
        derefed_its = [it.deref() for it in self.its]
        if not all(it == Sentinel.VALUE for it in derefed_its[1:]):
            raise AssertionError("The result of a `deref` must be a `Sentinel.VALUE`.")
        return Sentinel.VALUE


def _combine(*values):
    # `OffsetLiteral`s may occur in `list_get` calls
    if not all(
        val in [Sentinel.VALUE, Sentinel.TYPE] or isinstance(val, ir.OffsetLiteral)
        for val in values
    ):
        raise AssertionError("All arguments must be values or types.")
    return Sentinel.VALUE


# implementations of builtins
def _deref(x):
    return x.deref()


def _can_deref(x):
    return Sentinel.VALUE


def _shift(*offsets):
    def apply(arg):
        assert isinstance(arg, IteratorTracer)
        return arg.shift(offsets)

    return apply


@dataclass(frozen=True)
class AppliedLift(IteratorTracer):
    stencil: Callable
    its: tuple[IteratorTracer, ...]

    def shift(self, offsets):
        return AppliedLift(self.stencil, tuple(_shift(it) for it in self.its))

    def deref(self):
        return self.stencil(*self.its)


def _lift(f):
    def apply(*its):
        if not all(isinstance(it, IteratorTracer) for it in its):
            raise AssertionError("All arguments must be iterators.")
        return AppliedLift(f, its)

    return apply


def _reduce(f, init):
    return _combine


def _map(f):
    return _combine


def _neighbors(o, x):
    return _deref(_shift(o, ALL_NEIGHBORS)(x))


def _scan(f, forward, init):
    def apply(*args):
        return f(init, *args)

    return apply


def _primitive_constituents(
    val: Literal[Sentinel.VALUE] | IteratorTracer | tuple,
) -> Iterable[Literal[Sentinel.VALUE] | IteratorTracer]:
    if val is Sentinel.VALUE or isinstance(val, IteratorTracer):
        yield val
    elif isinstance(val, tuple):
        for el in val:
            if isinstance(el, tuple):
                yield from _primitive_constituents(el)
            elif el is Sentinel.VALUE or isinstance(el, IteratorTracer):
                yield el
            else:
                raise AssertionError(
                    "Expected a `Sentinel.VALUE`, `IteratorTracer` or tuple thereof."
                )
    else:
        raise ValueError()


def _if(cond: Literal[Sentinel.VALUE], true_branch, false_branch):
    assert cond is Sentinel.VALUE
    if any(isinstance(branch, tuple) for branch in (false_branch, true_branch)):
        # Broadcast branches to tuple of same length. This is required for cases like:
        #  `if_(cond, deref(iterator_of_tuples), make_tuple(...))`.
        if not isinstance(true_branch, tuple):
            assert all(el == Sentinel.VALUE for el in false_branch)
            true_branch = (true_branch,) * len(false_branch)
        if not isinstance(false_branch, tuple):
            assert all(el == Sentinel.VALUE for el in true_branch)
            false_branch = (false_branch,) * len(true_branch)

        result = []
        for el_true_branch, el_false_branch in zip(true_branch, false_branch):
            # just reuse `if_` to recursively build up the result
            result.append(_if(Sentinel.VALUE, el_true_branch, el_false_branch))
        return tuple(result)

    is_iterator_arg = tuple(
        isinstance(arg, IteratorTracer) for arg in (cond, true_branch, false_branch)
    )
    if is_iterator_arg == (False, True, True):
        return CombinedTracer((true_branch, false_branch))
    assert is_iterator_arg == (False, False, False) and all(
        arg in [Sentinel.VALUE, Sentinel.TYPE] for arg in (cond, true_branch, false_branch)
    )
    return Sentinel.VALUE


def _make_tuple(*args):
    return args


def _tuple_get(index, tuple_val):
    if isinstance(tuple_val, tuple):
        return tuple_val[index]
    assert tuple_val is Sentinel.VALUE
    return Sentinel.VALUE


_START_CTX: Final = {
    "deref": _deref,
    "can_deref": _can_deref,
    "shift": _shift,
    "lift": _lift,
    "scan": _scan,
    "reduce": _reduce,
    "neighbors": _neighbors,
    "map_": _map,
    "if_": _if,
    "make_tuple": _make_tuple,
}


class TraceShifts(NodeTranslator):
    def visit_Literal(self, node: ir.SymRef, *, ctx: dict[str, Any]) -> Any:
        return Sentinel.VALUE

    def visit_SymRef(self, node: ir.SymRef, *, ctx: dict[str, Any]) -> Any:
        if node.id in ctx:
            return ctx[node.id]
        elif node.id in ir.TYPEBUILTINS:
            return Sentinel.TYPE
        return _combine

    def visit_FunCall(self, node: ir.FunCall, *, ctx: dict[str, Any]) -> Any:
        if node.fun == ir.SymRef(id="tuple_get"):
            assert isinstance(node.args[0], ir.Literal)
            index = int(node.args[0].value)
            return _tuple_get(index, self.visit(node.args[1], ctx=ctx))

        fun = self.visit(node.fun, ctx=ctx)
        args = self.visit(node.args, ctx=ctx)
        return fun(*args)

    def visit_Lambda(self, node: ir.Lambda, *, ctx: dict[str, Any]) -> Callable:
        def fun(*args):
            return self.visit(
                node.expr, ctx=ctx | {p.id: a for p, a in zip(node.params, args, strict=True)}
            )

        return fun

    def visit_StencilClosure(
        self, node: ir.StencilClosure, *, shifts: dict[str, list[tuple[ir.OffsetLiteral, ...]]]
    ):
        def register_deref(inp: str, offsets: tuple[ir.OffsetLiteral, ...]):
            shifts[inp].append(offsets)

        tracers = []
        for inp in node.inputs:
            shifts.setdefault(inp.id, [])
            tracers.append(InputTracer(inp=inp.id, register_deref=register_deref))

        result = self.visit(node.stencil, ctx=_START_CTX)(*tracers)
        assert all(el is Sentinel.VALUE for el in _primitive_constituents(result))

    @classmethod
    def apply(cls, node: ir.StencilClosure) -> dict[str, list[tuple[ir.OffsetLiteral, ...]]]:
        shifts = dict[str, list[tuple[ir.OffsetLiteral, ...]]]()
        cls().visit(node, shifts=shifts)
        return shifts
