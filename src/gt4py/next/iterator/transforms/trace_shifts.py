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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from gt4py.eve import NodeTranslator
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.collect_shifts import ALL_NEIGHBORS


@dataclass(frozen=True)
class InputTracer:
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
        if self.lift_level:
            return self
        self.register_deref(self.inp, self.offsets)

    def lift(self):
        return InputTracer(
            inp=self.inp,
            register_deref=self.register_deref,
            offsets=self.offsets,
            lift_level=self.lift_level + 1,
        )

    def unlift(self):
        assert self.lift_level > 0
        return InputTracer(
            inp=self.inp,
            register_deref=self.register_deref,
            offsets=self.offsets,
            lift_level=self.lift_level - 1,
        )


@dataclass(frozen=True)
class CombinedTracer:
    tracers: tuple[InputTracer, ...]

    def shift(self, offsets):
        return CombinedTracer(tracers=tuple(t.shift(offsets) for t in self.tracers))

    def deref(self):
        lift_levels = {t.lift_level for t in self.tracers}
        assert len(lift_levels) == 1
        if lift_levels.pop():
            return self
        for t in self.tracers:
            t.deref()

    def lift(self):
        return CombinedTracer(tracers=tuple(t.lift() for t in self.tracers))

    def unlift(self):
        return CombinedTracer(tracers=tuple(t.unlift() for t in self.tracers))


def _combine(*tracers):
    input_tracers = []

    def handle_tracer(tracer):
        if isinstance(tracer, InputTracer):
            input_tracers.append(tracer)
        elif isinstance(tracer, CombinedTracer):
            input_tracers.extend(tracer.tracers)
        elif isinstance(tracer, tuple):
            for t in tracer:
                handle_tracer(t)

    for tracer in tracers:
        handle_tracer(tracer)

    return CombinedTracer(tracers=tuple(input_tracers))


# implementations of builtins


def _deref(x):
    return x.deref()


def _can_deref(x):
    return


def _shift(*offsets):
    def apply(arg):
        return arg.shift(offsets)

    return apply


def _lift(f):
    def apply(*args):
        return f(*(arg.lift() for arg in args)).unlift()

    return apply


def _reduce(f, init):
    def apply(*args):
        return _combine(*args).shift((ALL_NEIGHBORS,)).deref()

    return apply


def _scan(f, forward, init):
    def apply(*args):
        return f(init, *args)

    return apply


def _make_tuple(*args):
    return args


def _tuple_get(idx, tup):
    if tup is not None:
        return tup[int(idx.value)]


_START_CTX: Final = {
    "deref": _deref,
    "can_deref": _can_deref,
    "shift": _shift,
    "lift": _lift,
    "reduce": _reduce,
    "scan": _scan,
    "make_tuple": _make_tuple,
    "tuple_get": _tuple_get,
}


class TraceShifts(NodeTranslator):
    def visit_SymRef(self, node: ir.SymRef, *, ctx: dict[str, Any]) -> Any:
        if node.id in ctx:
            return ctx[node.id]
        return _combine

    def visit_FunCall(self, node: ir.FunCall, *, ctx: dict[str, Any]) -> Any:
        fun = self.visit(node.fun, ctx=ctx)
        args = self.visit(node.args, ctx=ctx)
        return fun(*args)

    def visit_Lambda(self, node: ir.Lambda, *, ctx: dict[str, Any]) -> Callable:
        def fun(*args):
            return self.visit(node.expr, ctx=ctx | {p.id: a for p, a in zip(node.params, args)})

        return fun

    def visit_StencilClosure(
        self, node: ir.StencilClosure, *, shifts: dict[str, list[tuple[ir.OffsetLiteral, ...]]]
    ):
        def register_deref(inp: str, offsets: tuple[ir.OffsetLiteral, ...]):
            shifts.setdefault(inp, []).append(offsets)

        tracers = [InputTracer(inp=inp.id, register_deref=register_deref) for inp in node.inputs]
        self.visit(node.stencil, ctx=_START_CTX)(*tracers)

    @classmethod
    def apply(cls, node: ir.StencilClosure) -> dict[str, list[tuple[ir.OffsetLiteral, ...]]]:
        shifts = dict[str, list[tuple[ir.OffsetLiteral, ...]]]()
        cls().visit(node, shifts=shifts)
        return shifts
