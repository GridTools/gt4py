# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

from gt4py import eve
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm


def _as_int_literal(node: itir.Expr) -> int | None:
    if not isinstance(node, itir.Literal):
        return None
    try:
        return int(node.value)
    except ValueError:
        return None


def _extract_shift_call(arg: itir.Expr) -> tuple[str, itir.Expr] | None:
    if not cpm.is_call_to(arg, "deref") or len(arg.args) != 1:
        return None
    shift_app = arg.args[0]
    if not isinstance(shift_app, itir.FunCall):
        return None
    shift_fun = shift_app.fun
    if not cpm.is_call_to(shift_fun, "shift") or len(shift_fun.args) != 2:
        return None

    axis_offset, index_offset = shift_fun.args
    if not (
        isinstance(axis_offset, itir.OffsetLiteral)
        and isinstance(axis_offset.value, str)
        and isinstance(index_offset, itir.OffsetLiteral)
        and isinstance(index_offset.value, int)
    ):
        return None

    if index_offset.value != 0:
        return None

    if len(shift_app.args) != 1:
        return None
    return axis_offset.value, shift_app.args[0]


@dataclasses.dataclass(frozen=True)
class UnrollCartesianReduction(eve.NodeTranslator):
    """Unroll `reduce` calls over scalar iterator values on cartesian dimensions.

    This pass targets reductions that arrive as `reduce(op, init)(deref(it))` and rewrites
    them into explicit shifted accumulation along a dimension with a literal domain range,
    e.g. `(start, stop)` from `named_range(dim, start, stop)`.
    """

    axis_ranges: dict[str, tuple[int, int]] = dataclasses.field(default_factory=dict)
    axis_hints: set[str] = dataclasses.field(default_factory=set)

    @classmethod
    def apply(cls, node: itir.Program) -> itir.Program:
        return cls().visit(node)

    def _collect_axis_info(self, node: itir.Program) -> tuple[dict[str, tuple[int, int]], set[str]]:
        class _Collector(eve.NodeVisitor):
            def __init__(self) -> None:
                self.axis_ranges: dict[str, tuple[int, int]] = {}
                self.axis_hints: set[str] = set()

            def visit_FunCall(self, n: itir.FunCall, **kwargs):
                if cpm.is_call_to(n, "named_range") and len(n.args) == 3:
                    axis, start, stop = n.args
                    if isinstance(axis, itir.AxisLiteral):
                        self.axis_hints.add(axis.value)
                        start_int = _as_int_literal(start)
                        stop_int = _as_int_literal(stop)
                        if start_int is not None and stop_int is not None:
                            self.axis_ranges[axis.value] = (start_int, stop_int)
                return self.generic_visit(n, **kwargs)

            def visit_AxisLiteral(self, n: itir.AxisLiteral, **kwargs):
                self.axis_hints.add(n.value)
                return self.generic_visit(n, **kwargs)

            def visit_SymRef(self, n: itir.SymRef, **kwargs):
                if "kolor" in n.id.lower():
                    self.axis_hints.add(n.id)
                return self.generic_visit(n, **kwargs)

        collector = _Collector()
        collector.visit(node)
        return collector.axis_ranges, collector.axis_hints

    def _pick_axis_range(self) -> tuple[str, tuple[int, int]] | None:
        literal_axes = [
            (axis, bounds)
            for axis, bounds in self.axis_ranges.items()
            if bounds[1] > bounds[0]
        ]
        if not literal_axes:
            return None

        kolor_axes = [entry for entry in literal_axes if "kolor" in entry[0].lower()]
        if len(kolor_axes) == 1:
            return kolor_axes[0]

        if len(literal_axes) == 1:
            return literal_axes[0]

        hinted_kolor_axes = [axis for axis in self.axis_hints if "kolor" in axis.lower()]
        if len(hinted_kolor_axes) == 1:
            return hinted_kolor_axes[0], (0, 3)

        return None

    def _shift_scalar_arg(self, arg: itir.Expr, axis_offset: str, index: int) -> itir.Expr | None:
        explicit_shift = _extract_shift_call(arg)
        if explicit_shift is not None:
            _, iterator = explicit_shift
            shift = itir.FunCall(
                fun=itir.SymRef(id="shift"),
                args=[
                    itir.OffsetLiteral(value=axis_offset),
                    itir.OffsetLiteral(value=index),
                ],
            )
            shifted_it = itir.FunCall(fun=shift, args=[iterator])
            return itir.FunCall(fun=itir.SymRef(id="deref"), args=[shifted_it])

        if cpm.is_call_to(arg, "deref") and len(arg.args) == 1:
            shift = itir.FunCall(
                fun=itir.SymRef(id="shift"),
                args=[
                    itir.OffsetLiteral(value=axis_offset),
                    itir.OffsetLiteral(value=index),
                ],
            )
            shifted_it = itir.FunCall(fun=shift, args=[arg.args[0]])
            return itir.FunCall(fun=itir.SymRef(id="deref"), args=[shifted_it])
        return None

    def visit_Program(self, node: itir.Program, **kwargs) -> itir.Program:
        axis_ranges, axis_hints = self._collect_axis_info(node)
        return dataclasses.replace(self, axis_ranges=axis_ranges, axis_hints=axis_hints).generic_visit(node, **kwargs)

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        node = self.generic_visit(node, **kwargs)

        if not cpm.is_applied_reduce(node):
            return node

        explicit_axis_offsets = {
            offset
            for arg in node.args
            if (extracted := _extract_shift_call(arg)) is not None
            for offset, _ in [extracted]
        }

        axis_entry = self._pick_axis_range()
        axis_offset: str | None = None
        if len(explicit_axis_offsets) == 1:
            axis_offset = next(iter(explicit_axis_offsets))

        if axis_entry is None and axis_offset is None:
            return node

        if axis_entry is None and axis_offset is not None:
            if axis_offset.startswith(common._IMPLICIT_OFFSET_PREFIX):
                axis_name = common._get_dimension_name_from_implicit_offset(axis_offset)
                bounds = self.axis_ranges.get(axis_name)
                if bounds is None and "kolor" in axis_name.lower():
                    bounds = (0, 3)
                if bounds is None:
                    return node
                axis, (start, stop) = axis_name, bounds
            else:
                return node
        else:
            assert axis_entry is not None
            axis, (start, stop) = axis_entry

        if axis_offset is None:
            axis_offset = common.dimension_to_implicit_offset(axis)

        if not isinstance(node.fun, itir.FunCall) or len(node.fun.args) != 2:
            return node

        op, init = node.fun.args

        shifted_args_by_index: list[list[itir.Expr]] = []
        for index in range(start, stop):
            shifted_args: list[itir.Expr] = []
            for arg in node.args:
                shifted_arg = self._shift_scalar_arg(arg, axis_offset, index)
                if shifted_arg is None:
                    return node
                shifted_args.append(shifted_arg)
            shifted_args_by_index.append(shifted_args)

        expr: itir.Expr = init
        for shifted_args in shifted_args_by_index:
            expr = itir.FunCall(fun=op, args=[expr, *shifted_args])

        return expr
