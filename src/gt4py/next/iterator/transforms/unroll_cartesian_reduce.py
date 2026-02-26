# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

from gt4py import eve
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.type_system import type_specifications as ts
import gt4py.next as gtx


def _is_applied_cartesian_reduce(node: itir.FunCall) -> bool:
    direct_cartesian_reduce = (
        isinstance(node.fun, itir.FunCall)
        and cpm.is_call_to(node.fun, "cartesian_reduce")
        and len(node.fun.args) == 3
        and len(node.args) >= 1
    )
    as_fieldop_cartesian_reduce = (
        cpm.is_applied_as_fieldop(node)
        and isinstance(node.fun.args[0], itir.FunCall)
        and cpm.is_call_to(node.fun.args[0], "cartesian_reduce")
        and len(node.fun.args[0].args) == 3
        and len(node.args) >= 1
    )
    
    # print(f"[GT4PY_DEBUG_UNROLL_CARTESIAN] Checking node {node} for cartesian_reduce: ")
    # print(f"  node.fun.args: ", getattr(node.fun, "args", None))
    # print(f"  direct_cartesian_reduce: {direct_cartesian_reduce}")
    # print(f"  as_fieldop_cartesian_reduce: {as_fieldop_cartesian_reduce}")
    return direct_cartesian_reduce or as_fieldop_cartesian_reduce


def _maybe_int_bound(expr: itir.Expr) -> int | None:
    if isinstance(expr, itir.Literal):
        try:
            return int(expr.value)
        except ValueError:
            return None

    if isinstance(expr, itir.OffsetLiteral) and isinstance(expr.value, int):
        return expr.value

    return None

class UnrollCartesianReduce(eve.NodeTranslator):
    """Unroll cartesian axis reductions with literal extents into scalar accumulation."""

    def __init__(self):
        self.axis_ranges: dict[common.Dimension, tuple[int, int]] = {}

    @classmethod
    def apply(cls, node: itir.Program) -> itir.Program:
        return cls().visit(node)

    @staticmethod
    def _debug_enabled() -> bool:
        return bool(os.environ.get("GT4PY_DEBUG_UNROLL_CARTESIAN"))

    def _collect_axis_ranges(self, node: itir.Program) -> dict[common.Dimension, tuple[int, int]]:
        axis_ranges: dict[common.Dimension, tuple[int, int]] = {}
        finite_bounds: dict[common.Dimension, set[int]] = {}

        class _Collector(eve.NodeVisitor):
            def visit_FunCall(self, n: itir.FunCall, **kwargs):
                if cpm.is_call_to(n, "named_range") and len(n.args) == 3:
                    axis, start, stop = n.args
                    if isinstance(axis, itir.AxisLiteral):
                        try:
                            axis_dim = common.Dimension(value=axis.value, kind=axis.kind)
                            start_bound = _maybe_int_bound(start)
                            stop_bound = _maybe_int_bound(stop)
                            if start_bound is not None and stop_bound is not None:
                                axis_ranges[axis_dim] = (start_bound, stop_bound)
                            else:
                                bounds = finite_bounds.setdefault(axis_dim, set())
                                if start_bound is not None:
                                    bounds.add(start_bound)
                                if stop_bound is not None:
                                    bounds.add(stop_bound)
                        except ValueError:
                            pass
                return self.generic_visit(n, **kwargs)

        _Collector().visit(node)
        for axis, bounds in finite_bounds.items():
            if axis in axis_ranges or not bounds:
                continue
            min_bound = min(bounds)
            max_bound = max(bounds)
            if min_bound >= 0:
                axis_ranges[axis] = (0, max_bound + 1)
        return axis_ranges
    

    def visit_Program(self, node: itir.Program, **kwargs):
        # IDim = gtx.Dimension("IDim")
        # JDim = gtx.Dimension("JDim")
        # Kolor = gtx.Dimension("Kolor")
        self.axis_ranges = self._collect_axis_ranges(node)
        # looks like this: {Dimension(value='Kolor', kind=<DimensionKind.HORIZONTAL: 'horizontal'>): (0, 3), Dimension(value='IDim', kind=<DimensionKind.HORIZONTAL: 'horizontal'>): (0, 1), Dimension(value='JDim', kind=<DimensionKind.HORIZONTAL: 'horizontal'>): (0, 1)}
        # self.axis_ranges = {IDim: (0, 1), JDim: (0, 1), Kolor: (0, 3)}
        if self._debug_enabled():
            print(
                f"[GT4PY_DEBUG_UNROLL_CARTESIAN] axis_ranges={self.axis_ranges}",
                file=sys.stderr,
            )
        return self.generic_visit(node, **kwargs)

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        node = self.generic_visit(node, **kwargs)

        if not _is_applied_cartesian_reduce(node):
            return node

        reduce_def: itir.FunCall
        reduce_args: tuple[itir.Expr, ...]
        is_as_fieldop_reduce = False
        if isinstance(node.fun, itir.FunCall) and cpm.is_call_to(node.fun, "cartesian_reduce"):
            reduce_def = node.fun
            reduce_args = tuple(node.args)
        else:
            assert cpm.is_applied_as_fieldop(node)
            assert isinstance(node.fun.args[0], itir.FunCall)
            reduce_def = node.fun.args[0]
            reduce_args = tuple(node.args)
            is_as_fieldop_reduce = True

        op, init, axis_expr = reduce_def.args
        if not isinstance(axis_expr, itir.AxisLiteral):
            return node

        axis = common.Dimension(value=axis_expr.value, kind=axis_expr.kind)
        if axis not in self.axis_ranges:
            if self._debug_enabled():
                print(
                    "[GT4PY_DEBUG_UNROLL_CARTESIAN] missing axis "
                    f"{axis}; available={list(self.axis_ranges.keys())}",
                    file=sys.stderr,
                )
            return node

        start, stop = self.axis_ranges[axis]
        if stop <= start:
            return init

        axis_offset = common.dimension_to_implicit_offset(axis.value)
        if is_as_fieldop_reduce:
            param_names = tuple(f"__cartred_it{i}" for i, _ in enumerate(reduce_args))
            expr: itir.Expr = init
            for index in range(start, stop):
                shifted_args = tuple(
                    im.deref(im.shift(axis_offset, index)(param_name)) for param_name in param_names
                )
                expr = itir.FunCall(fun=op, args=[expr, *shifted_args])
            domain_expr = None
            if reduce_args and isinstance(reduce_args[0], itir.Expr):
                field_type = getattr(reduce_args[0], "type", None)
                if isinstance(field_type, ts.FieldType):
                    reduced_dims = [dim for dim in field_type.dims if dim != axis]
                    ranges = {}
                    for dim in reduced_dims:
                        dim_range = im.call("get_domain_range")(reduce_args[0], dim)
                        ranges[dim] = (im.tuple_get(0, dim_range), im.tuple_get(1, dim_range))
                    domain_expr = im.domain(common.GridType.CARTESIAN, ranges)
            return im.as_fieldop(im.lambda_(*param_names)(expr), domain=domain_expr)(*reduce_args)

        expr: itir.Expr = init
        for index in range(start, stop):
            shifted_args = tuple(im.deref(im.shift(axis_offset, index)(arg)) for arg in reduce_args)
            expr = itir.FunCall(fun=op, args=[expr, *shifted_args])

        return expr
