# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import sys
from collections.abc import Callable
from typing import Any, Final, Iterable, Literal, Optional

from gt4py import eve
from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im


class ValidateRecordedShiftsAnnex(eve.NodeVisitor):
    """Ensure every applied lift and its arguments have the `recorded_shifts` annex populated."""

    def visit_FunCall(self, node: ir.FunCall):
        if cpm.is_applied_lift(node):
            assert hasattr(node.annex, "recorded_shifts")

            if len(node.annex.recorded_shifts) == 0:
                return

            if isinstance(node.fun.args[0], ir.Lambda):  # type: ignore[attr-defined]  # ensured by is_applied_lift
                stencil = node.fun.args[0]  # type: ignore[attr-defined]  # ensured by is_applied_lift
                for param in stencil.params:
                    assert hasattr(param.annex, "recorded_shifts")

        self.generic_visit(node)


def copy_recorded_shifts(from_: ir.Node, to: ir.Node) -> None:
    """
    Copy `recorded_shifts` annex attribute from one node to another.

    This function mainly exists for readability reasons.
    """
    to.annex.recorded_shifts = from_.annex.recorded_shifts


class Sentinel(eve.StrEnum):
    VALUE = "VALUE"
    TYPE = "TYPE"

    ALL_NEIGHBORS = "ALL_NEIGHBORS"


@dataclasses.dataclass(frozen=True)
class ShiftRecorder:
    recorded_shifts: dict[int, set[tuple[ir.OffsetLiteral, ...]]] = dataclasses.field(
        default_factory=dict
    )

    def register_node(self, inp: ir.Expr | ir.Sym) -> None:
        self.recorded_shifts.setdefault(id(inp), set())

    def __call__(self, inp: ir.Expr | ir.Sym, offsets: tuple[ir.OffsetLiteral, ...]) -> None:
        self.recorded_shifts[id(inp)].add(offsets)


@dataclasses.dataclass(frozen=True)
class ForwardingShiftRecorder:
    wrapped_tracer: Any
    shift_recorder: ShiftRecorder

    def __call__(self, inp: ir.Expr | ir.Sym, offsets: tuple[ir.OffsetLiteral, ...]):
        self.shift_recorder(inp, offsets)
        # Forward shift to wrapped tracer such it can record the shifts of the parent nodes
        self.wrapped_tracer.shift(offsets).deref()


# for performance reasons (`isinstance` is slow otherwise) we don't use abc here
class Tracer:
    def deref(self):
        raise NotImplementedError()

    def shift(self, offsets: tuple[ir.OffsetLiteral, ...]):
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class ArgTracer(Tracer):
    arg: ir.Expr | ir.Sym
    shift_recorder: ShiftRecorder | ForwardingShiftRecorder
    offsets: tuple[ir.OffsetLiteral, ...] = ()

    def shift(self, offsets: tuple[ir.OffsetLiteral, ...]):
        return ArgTracer(
            arg=self.arg, shift_recorder=self.shift_recorder, offsets=self.offsets + tuple(offsets)
        )

    def deref(self):
        self.shift_recorder(self.arg, self.offsets)
        return Sentinel.VALUE


# This class is only needed because we currently allow conditionals on iterators. Since this is
# not supported in the C++ backend it can likely be removed again in the future.
@dataclasses.dataclass(frozen=True)
class CombinedTracer(Tracer):
    its: tuple[Tracer, ...]

    def shift(self, offsets: tuple[ir.OffsetLiteral, ...]):
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
    assert all(
        isinstance(offset, ir.OffsetLiteral) or offset in [Sentinel.ALL_NEIGHBORS, Sentinel.VALUE]
        for offset in offsets
    )

    def apply(arg):
        assert isinstance(arg, Tracer)
        return arg.shift(offsets)

    return apply


@dataclasses.dataclass(frozen=True)
class AppliedLift(Tracer):
    stencil: Callable
    its: tuple[Tracer, ...]

    def shift(self, offsets):
        return AppliedLift(self.stencil, tuple(_shift(*offsets)(it) for it in self.its))

    def deref(self):
        return self.stencil(*self.its)


def _lift(f):
    def apply(*its):
        if not all(isinstance(it, Tracer) for it in its):
            raise AssertionError("All arguments must be iterators.")
        return AppliedLift(f, its)

    return apply


def _reduce(f, init):
    return _combine


def _map(f):
    return _combine


def _neighbors(o, x):
    return _deref(_shift(o, Sentinel.ALL_NEIGHBORS)(x))


def _scan(f, forward, init):
    def apply(*args):
        return f(init, *args)

    return apply


def _primitive_constituents(
    val: Literal[Sentinel.VALUE] | Tracer | tuple,
) -> Iterable[Literal[Sentinel.VALUE] | Tracer]:
    if val is Sentinel.VALUE or isinstance(val, Tracer):
        yield val
    elif isinstance(val, tuple):
        for el in val:
            if isinstance(el, tuple):
                yield from _primitive_constituents(el)
            elif el is Sentinel.VALUE or isinstance(el, Tracer):
                yield el
            else:
                raise AssertionError("Expected a `Sentinel.VALUE`, `Tracer` or tuple thereof.")
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

    is_iterator_arg = tuple(isinstance(arg, Tracer) for arg in (cond, true_branch, false_branch))
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


def _as_fieldop(stencil, domain=None):
    def applied_as_fieldop(*args):
        return stencil(*args)

    return applied_as_fieldop


_START_CTX: Final = {
    "as_fieldop": _as_fieldop,
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


# TODO(tehrengruber): This pass is unnecessarily very inefficient and easily exceeds the default
#  recursion limit.
@dataclasses.dataclass(frozen=True)
class TraceShifts(PreserveLocationVisitor, NodeTranslator):
    shift_recorder: ShiftRecorder = dataclasses.field(default_factory=ShiftRecorder)

    def visit_Literal(self, node: ir.SymRef, *, ctx: dict[str, Any]) -> Any:
        return Sentinel.VALUE

    def visit_SymRef(self, node: ir.SymRef, *, ctx: dict[str, Any]) -> Any:
        if node.id in ctx:
            return ctx[node.id]
        elif node.id in ir.TYPEBUILTINS:
            return Sentinel.TYPE
        elif node.id in (ir.ARITHMETIC_BUILTINS | {"list_get", "make_const_list", "cast_"}):
            return _combine
        raise ValueError(f"Undefined symbol {node.id}")

    def visit_FunCall(self, node: ir.FunCall, *, ctx: dict[str, Any]) -> Any:
        if node.fun == ir.SymRef(id="tuple_get"):
            assert isinstance(node.args[0], ir.Literal)
            index = int(node.args[0].value)
            return _tuple_get(index, self.visit(node.args[1], ctx=ctx))

        fun = self.visit(node.fun, ctx=ctx)
        args = self.visit(node.args, ctx=ctx)
        return fun(*args)

    def visit(self, node, **kwargs):
        result = super().visit(node, **kwargs)
        if isinstance(result, Tracer):
            assert isinstance(node, (ir.Sym, ir.Expr))

            self.shift_recorder.register_node(node)
            result = ArgTracer(
                arg=node, shift_recorder=ForwardingShiftRecorder(result, self.shift_recorder)
            )
        return result

    def visit_Lambda(self, node: ir.Lambda, *, ctx: dict[str, Any]) -> Callable:
        def fun(*args):
            new_args = []
            for param, arg in zip(node.params, args, strict=True):
                if isinstance(arg, Tracer):
                    self.shift_recorder.register_node(param)
                    new_args.append(
                        ArgTracer(
                            arg=param,
                            shift_recorder=ForwardingShiftRecorder(arg, self.shift_recorder),
                        )
                    )
                else:
                    new_args.append(arg)

            return self.visit(
                node.expr, ctx=ctx | {p.id: a for p, a in zip(node.params, new_args, strict=True)}
            )

        return fun

    @classmethod
    def trace_stencil(
        cls, stencil: ir.Expr, *, num_args: Optional[int] = None, save_to_annex: bool = False
    ):
        # If we get a lambda we can deduce the number of arguments.
        if isinstance(stencil, ir.Lambda):
            assert num_args is None or num_args == len(stencil.params)
            num_args = len(stencil.params)
        elif cpm.is_call_to(stencil, "scan"):
            assert isinstance(stencil.args[0], ir.Lambda)
            num_args = len(stencil.args[0].params) - 1
        if not isinstance(num_args, int):
            raise ValueError("Stencil must be an 'itir.Lambda', scan, or `num_args` is given.")
        assert isinstance(num_args, int)

        args = [im.ref(f"__arg{i}") for i in range(num_args)]

        old_recursionlimit = sys.getrecursionlimit()
        sys.setrecursionlimit(100000000)

        instance = cls()

        # initialize shift recorder & context with all built-ins and the iterator argument tracers
        ctx: dict[str, Any] = {**_START_CTX}
        for arg in args:
            instance.shift_recorder.register_node(arg)
            ctx[arg.id] = ArgTracer(arg=arg, shift_recorder=instance.shift_recorder)

        # actually trace stencil
        instance.visit(im.call(stencil)(*args), ctx=ctx)

        sys.setrecursionlimit(old_recursionlimit)

        recorded_shifts = instance.shift_recorder.recorded_shifts

        param_shifts = []
        for arg in args:
            param_shifts.append(recorded_shifts[id(arg)])

        if save_to_annex:
            _save_to_annex(stencil, recorded_shifts)

        return param_shifts


trace_stencil = TraceShifts.trace_stencil


def _save_to_annex(
    node: ir.Node, recorded_shifts: dict[int, set[tuple[ir.OffsetLiteral, ...]]]
) -> None:
    for child_node in node.pre_walk_values():
        if id(child_node) in recorded_shifts:
            child_node.annex.recorded_shifts = recorded_shifts[id(child_node)]

    if __debug__:
        ValidateRecordedShiftsAnnex().visit(node)
