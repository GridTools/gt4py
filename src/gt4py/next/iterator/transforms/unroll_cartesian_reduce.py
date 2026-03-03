# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Mapping

from gt4py import eve
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.type_system import type_specifications as ts


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
    return direct_cartesian_reduce or as_fieldop_cartesian_reduce


class UnrollCartesianReduce(eve.NodeTranslator):
    """Unroll cartesian axis reductions with literal extents into scalar accumulation."""

    def __init__(self, axis_ranges: Mapping[common.Dimension, tuple[int, int]] | None = None):
        self.axis_ranges: dict[common.Dimension, tuple[int, int]] = dict(axis_ranges or {})

    @classmethod
    def apply(
        cls,
        node: itir.Program,
        *,
        axis_ranges: Mapping[common.Dimension, tuple[int, int]] | None = None,
    ) -> itir.Program:
        return cls(axis_ranges=axis_ranges).visit(node)

    @staticmethod
    def _debug_enabled() -> bool:
        return bool(os.environ.get("GT4PY_DEBUG_UNROLL_CARTESIAN"))

    def visit_Program(self, node: itir.Program, **kwargs):
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
            return node

        start, stop = self.axis_ranges[axis]
        if stop <= start:
            return init

        axis_offset = common.dimension_to_implicit_offset(axis.value)
        if is_as_fieldop_reduce:
            param_names = tuple(f"__cartred_it{i}" for i, _ in enumerate(reduce_args))
            bound_arg_names = tuple(f"__cartred_arg{i}" for i, _ in enumerate(reduce_args))
            reduced_expr: itir.Expr = init
            for index in range(start, stop):
                shifted_args = tuple(
                    im.deref(im.shift(axis_offset, index)(param_name)) for param_name in param_names
                )
                reduced_expr = itir.FunCall(fun=op, args=[reduced_expr, *shifted_args])

            domain_expr: itir.Expr | None = None
            if reduce_args and isinstance(reduce_args[0], itir.Expr):
                field_type = getattr(reduce_args[0], "type", None)
                if domain_expr is None and isinstance(field_type, ts.FieldType):
                    reduced_dims = [dim for dim in field_type.dims if dim != axis]
                    ranges = {
                        dim: (
                            im.tuple_get(0, im.call("get_domain_range")(bound_arg_names[0], dim)),
                            im.tuple_get(1, im.call("get_domain_range")(bound_arg_names[0], dim)),
                        )
                        for dim in reduced_dims
                    }
                    domain_expr = im.domain(common.GridType.CARTESIAN, ranges)

            if domain_expr is None:
                reduced_call = im.as_fieldop(im.lambda_(*param_names)(reduced_expr))(
                    *bound_arg_names
                )
            else:
                reduced_call = im.as_fieldop(
                    im.lambda_(*param_names)(reduced_expr), domain=domain_expr
                )(*bound_arg_names)

            return im.let(*tuple(zip(bound_arg_names, reduce_args, strict=True)))(reduced_call)

        reduced_expr = init
        for index in range(start, stop):
            shifted_args = tuple(im.deref(im.shift(axis_offset, index)(arg)) for arg in reduce_args)
            reduced_expr = itir.FunCall(fun=op, args=[reduced_expr, *shifted_args])

        return reduced_expr
