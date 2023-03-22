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

import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, List, Union

from gt4py.eve import NodeTranslator
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.collect_shifts import ALL_NEIGHBORS


VALUE_TOKEN = types.new_class("VALUE_TOKEN")


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
        self.register_deref(self.inp, self.offsets)
        return VALUE_TOKEN


def _combine(*values):
    if not all(val is VALUE_TOKEN for val in values):
        raise AssertionError("All arguments must be values.")
    return VALUE_TOKEN


# implementations of builtins
def _deref(x):
    return x.deref()


def _can_deref(x):
    return VALUE_TOKEN


def _shift(*offsets):
    def apply(arg):
        return arg.shift(offsets)

    return apply


@dataclass(frozen=True)
class AppliedLift:
    stencil: Callable
    its: List[Union[InputTracer, "AppliedLift"]]

    def __post_init__(self):
        for it in self.its:
            assert isinstance(it, (InputTracer, AppliedLift))

    def shift(self, offsets):
        return AppliedLift(self.stencil, [_shift(it) for it in self.its])

    def deref(self):
        return self.stencil(*self.its)


def _lift(f):
    def apply(*args):
        return AppliedLift(f, list(args))

    return apply


def _reduce(f, init):
    def apply(*args):
        for arg in args:
            _deref(_shift(ALL_NEIGHBORS)(arg))
        return VALUE_TOKEN

    return apply


def _scan(f, forward, init):
    def apply(*args):
        return f(init, *args)

    return apply


def _cast(it, dtype):
    return it


_START_CTX: Final = {
    "deref": _deref,
    "can_deref": _can_deref,
    "shift": _shift,
    "lift": _lift,
    "reduce": _reduce,
    "scan": _scan,
    "cast_": _cast,
}


class TraceShifts(NodeTranslator):
    def visit_Literal(self, node: ir.SymRef, *, ctx: dict[str, Any]) -> Any:
        return VALUE_TOKEN

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
            shifts[inp].append(offsets)

        tracers = []
        for inp in node.inputs:
            shifts.setdefault(inp.id, [])
            tracers.append(InputTracer(inp=inp.id, register_deref=register_deref))

        result = self.visit(node.stencil, ctx=_START_CTX)(*tracers)
        assert result is VALUE_TOKEN

    @classmethod
    def apply(cls, node: ir.StencilClosure) -> dict[str, list[tuple[ir.OffsetLiteral, ...]]]:
        shifts = dict[str, list[tuple[ir.OffsetLiteral, ...]]]()
        cls().visit(node, shifts=shifts)
        return shifts
