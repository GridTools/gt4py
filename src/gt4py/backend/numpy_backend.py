# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import textwrap
from typing import TYPE_CHECKING

import numpy as np

from gt4py import backend as gt_backend
from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir
from gt4py.utils import text as gt_text

from .python_generator import PythonSourceGenerator


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder


class NumPySourceGenerator(PythonSourceGenerator):
    NATIVE_FUNC_TO_PYTHON = {
        gt_ir.NativeFunction.ABS: "np.abs",
        gt_ir.NativeFunction.MIN: "np.minimum",
        gt_ir.NativeFunction.MAX: "np.maximum",
        gt_ir.NativeFunction.MOD: "np.mod",
        gt_ir.NativeFunction.SIN: "np.sin",
        gt_ir.NativeFunction.COS: "np.cos",
        gt_ir.NativeFunction.TAN: "np.tan",
        gt_ir.NativeFunction.ARCSIN: "np.arcsin",
        gt_ir.NativeFunction.ARCCOS: "np.arccos",
        gt_ir.NativeFunction.ARCTAN: "np.arctan",
        gt_ir.NativeFunction.SQRT: "np.sqrt",
        gt_ir.NativeFunction.EXP: "np.exp",
        gt_ir.NativeFunction.LOG: "np.log",
        gt_ir.NativeFunction.ISFINITE: "np.isfinite",
        gt_ir.NativeFunction.ISINF: "np.isinf",
        gt_ir.NativeFunction.ISNAN: "np.isnan",
        gt_ir.NativeFunction.FLOOR: "np.floor",
        gt_ir.NativeFunction.CEIL: "np.ceil",
        gt_ir.NativeFunction.TRUNC: "np.trunc",
    }

    def __init__(self, *args, interval_k_start_name, interval_k_end_name, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval_k_start_name = interval_k_start_name
        self.interval_k_end_name = interval_k_end_name
        self.conditions_depth = 0

    def _make_field_origin(self, name: str, origin=None):
        if origin is None:
            origin = "{origin_arg}['{name}']".format(origin_arg=self.origin_arg_name, name=name)

        source_lines = [
            "{name}{marker} = {origin}".format(name=name, marker=self.origin_marker, origin=origin)
        ]

        return source_lines

    def _make_regional_computation(self, iteration_order, interval_definition, body_sources):
        source_lines = []
        loop_bounds = [None, None]

        for r, bound in enumerate(interval_definition):
            loop_bounds[r] = "{}".format(self.k_splitters_value[bound[0]])
            if bound[1]:
                loop_bounds[r] += "{:+d}".format(bound[1])

        if iteration_order != gt_ir.IterationOrder.BACKWARD:
            range_args = loop_bounds
        else:
            range_args = [loop_bounds[1] + " -1", loop_bounds[0] + " -1", "-1"]

        if iteration_order != gt_ir.IterationOrder.PARALLEL:
            range_expr = "range({args})".format(args=", ".join(a for a in range_args))
            seq_axis = self.impl_node.domain.sequential_axis.name
            source_lines.append(
                "for {ax} in {range_expr}:".format(ax=seq_axis, range_expr=range_expr)
            )
            source_lines.extend(" " * self.indent_size + line for line in body_sources)
        else:
            source_lines.append(
                "{interval_k_start_name} = {lb}".format(
                    interval_k_start_name=self.interval_k_start_name, lb=loop_bounds[0]
                )
            )
            source_lines.append(
                "{interval_k_end_name} = {ub}".format(
                    interval_k_end_name=self.interval_k_end_name, ub=loop_bounds[1]
                )
            )
            source_lines.extend(body_sources)
            source_lines.extend("\n")
        return source_lines

    def make_temporary_field(
        self, name: str, dtype: gt_ir.DataType, extent: gt_definitions.Extent
    ):
        source_lines = super().make_temporary_field(name, dtype, extent)
        source_lines.extend(self._make_field_origin(name, extent.to_boundary().lower_indices))

        return source_lines

    def make_stage_source(self, iteration_order: gt_ir.IterationOrder, regions: list):
        source_lines = []

        lower_extent = self.block_info.extent.lower_indices
        upper_extent = self.block_info.extent.upper_indices

        # Computations body is split in different vertical regions
        regions = sorted(regions)
        if iteration_order == gt_ir.IterationOrder.BACKWARD:
            regions = reversed(regions)

        for bounds, parallel_interval, body in regions:
            region_lines = self._make_regional_computation(iteration_order, bounds, body)

            if parallel_interval:
                conditions = []
                for d, interval in enumerate(parallel_interval):
                    for axis_bound, extent, regular_bound, op in zip(
                        (interval.start, interval.end),
                        (lower_extent, upper_extent),
                        ("0", f"{self.domain_arg_name}[{d}]"),
                        (">=", "<="),
                    ):
                        if isinstance(axis_bound.level, gt_ir.VarRef):
                            conditions.append(
                                f"{axis_bound.level.name}{axis_bound.offset:+d} {op} {regular_bound}{extent[d]:+d}"
                            )
                        elif isinstance(axis_bound.level, int):
                            conditions.append(
                                f"{axis_bound.level}{axis_bound.offset:+d} {op} {regular_bound}{extent[d]:+d}"
                            )
                source_lines.append(f"if {' and '.join(conditions)}:")
                source_lines.extend([(" " * self.indent_size) + line for line in region_lines])
            else:
                source_lines.extend(region_lines)

        return source_lines

    # ---- Visitor handlers ----
    def visit_FieldRef(self, node: gt_ir.FieldRef):
        assert node.name in self.block_info.accessors

        is_parallel = self.block_info.iteration_order == gt_ir.IterationOrder.PARALLEL
        extent = self.block_info.extent

        lower_extent = list(extent.lower_indices)
        upper_extent = list(extent.upper_indices)

        for d, ax in enumerate(self.domain.axes_names):
            idx = node.offset.get(ax, 0)
            if idx:
                lower_extent[d] += idx
                upper_extent[d] += idx

        index = []
        for d, axis_name in enumerate([axis.name for axis in self.impl_node.domain.parallel_axes]):
            if self.block_info.parallel_interval:
                axis_interval = self.block_info.parallel_interval[d]
                bounds = []

                for axis_bound, extent, regular_bound in zip(
                    (axis_interval.start, axis_interval.end),
                    (lower_extent[d], upper_extent[d]),
                    ("0", f"{self.domain_arg_name}[{d}]"),
                ):
                    if isinstance(axis_bound.level, gt_ir.VarRef):
                        this_bound = f"{axis_bound.level.name}{axis_bound.offset:+d}"
                    elif isinstance(axis_bound.level, int):
                        this_bound = f"{axis_bound.level}"
                    else:
                        this_bound = regular_bound

                    bounds.append(
                        f"{node.name}{self.origin_marker}[{d}] + {this_bound}{node.offset[axis_name]:+d}"
                    )

                index.append(" : ".join(bounds))
            else:
                start_expr = " {:+d}".format(lower_extent[d]) if lower_extent[d] != 0 else ""
                size_expr = "{dom}[{d}]".format(dom=self.domain_arg_name, d=d)
                size_expr += " {:+d}".format(upper_extent[d]) if upper_extent[d] != 0 else ""

                lower_index = f"{node.name}{self.origin_marker}[{d}]{start_expr}"
                upper_index = f"{node.name}{self.origin_marker}[{d}] + {size_expr}"

                index.append(f"{lower_index}: {upper_index}")

        k_ax = self.domain.sequential_axis.name
        k_offset = node.offset.get(k_ax, 0)
        if is_parallel:
            start_expr = self.interval_k_start_name
            start_expr += " {:+d}".format(k_offset) if k_offset else ""
            end_expr = self.interval_k_end_name
            end_expr += " {:+d}".format(k_offset) if k_offset else ""
            index.append(
                "{name}{marker}[2] + {start}:{name}{marker}[2] + {stop}".format(
                    name=node.name, start=start_expr, marker=self.origin_marker, stop=end_expr
                )
            )
        else:
            idx = "{:+d}".format(k_offset) if k_offset else ""
            index.append(
                "{name}{marker}[{d}] + {ax}{idx}".format(
                    name=node.name,
                    marker=self.origin_marker,
                    d=len(self.domain.parallel_axes),
                    ax=k_ax,
                    idx=idx,
                )
            )

        source = "{name}[{index}]".format(name=node.name, index=", ".join(index))

        return source

    def visit_StencilImplementation(self, node: gt_ir.StencilImplementation):
        self.sources.empty_line()

        # Accessors for IO fields
        self.sources.append("# Sliced views of the stencil fields (domain + borders)")
        for info in node.api_signature:
            if info.name in node.fields and info.name not in node.unreferenced:
                self.sources.extend(self._make_field_origin(info.name))
                self.sources.extend(
                    "{name} = {name}.view({np}.ndarray)".format(
                        name=info.name, np=self.numpy_prefix
                    )
                )
        self.sources.empty_line()

        super().visit_StencilImplementation(node)

    def visit_UnaryOpExpr(self, node: gt_ir.UnaryOpExpr):

        if node.op is gt_ir.UnaryOperator.NOT:
            source = "np.logical_not({expr})".format(expr=self.visit(node.arg))
        else:
            fmt = "({})" if isinstance(node.arg, gt_ir.CompositeExpr) else "{}"
            source = "{op}{expr}".format(
                op=self.OP_TO_PYTHON[node.op], expr=fmt.format(self.visit(node.arg))
            )

        return source

    def visit_BinOpExpr(self, node: gt_ir.BinOpExpr):
        if node.op is gt_ir.BinaryOperator.AND:
            source = "np.logical_and({lhs}, {rhs})".format(
                lhs=self.visit(node.lhs), rhs=self.visit(node.rhs)
            )
        elif node.op is gt_ir.BinaryOperator.OR:
            source = "np.logical_or({lhs}, {rhs})".format(
                lhs=self.visit(node.lhs), rhs=self.visit(node.rhs)
            )
        else:
            lhs_fmt = "({})" if isinstance(node.lhs, gt_ir.CompositeExpr) else "{}"
            rhs_fmt = "({})" if isinstance(node.rhs, gt_ir.CompositeExpr) else "{}"
            source = "{lhs} {op} {rhs}".format(
                lhs=lhs_fmt.format(self.visit(node.lhs)),
                op=self.OP_TO_PYTHON[node.op],
                rhs=rhs_fmt.format(self.visit(node.rhs)),
            )

        return source

    def visit_TernaryOpExpr(self, node: gt_ir.TernaryOpExpr):
        then_fmt = "({})" if isinstance(node.then_expr, gt_ir.CompositeExpr) else "{}"
        else_fmt = "({})" if isinstance(node.else_expr, gt_ir.CompositeExpr) else "{}"

        source = "vectorized_ternary_op(condition={condition}, then_expr={then_expr}, else_expr={else_expr}, dtype={np}.{dtype})".format(
            condition=self.visit(node.condition),
            then_expr=then_fmt.format(self.visit(node.then_expr)),
            else_expr=else_fmt.format(self.visit(node.else_expr)),
            dtype=node.data_type.dtype.name,
            np=self.numpy_prefix,
        )

        return source

    def _visit_branch_stmt(self, stmt):
        sources = []
        if isinstance(stmt, gt_ir.Assign):
            condition = "__condition_1"
            for i in range(1, self.conditions_depth):
                condition = "{np}.logical_and({outer_condition}, {inner_condition})".format(
                    np=self.numpy_prefix,
                    outer_condition=condition,
                    inner_condition="__condition_{level}".format(level=i + 1),
                )

            target = self.visit(stmt.target)
            value = self.visit(stmt.value)

            data_type = (
                self.block_info.symbols[target].data_type
                if target in self.block_info.symbols
                else stmt.target.data_type
            )

            sources.append(
                "{target} = vectorized_ternary_op(condition={condition}, then_expr={then_expr}, else_expr={else_expr}, dtype={np}.{dtype})".format(
                    condition=condition,
                    target=target,
                    then_expr=value,
                    else_expr=target,
                    dtype=data_type.dtype.name,
                    np=self.numpy_prefix,
                )
            )
        else:
            stmt_sources = self.visit(stmt)
            if isinstance(stmt_sources, list):
                sources.extend(stmt_sources)
            else:
                sources.append(stmt_sources)
        return sources

    def visit_If(self, node: gt_ir.If):
        sources = []
        self.conditions_depth += 1
        sources.append(
            "__condition_{level} = {condition}".format(
                level=self.conditions_depth, condition=self.visit(node.condition)
            )
        )

        for stmt in node.main_body.stmts:
            sources.extend(self._visit_branch_stmt(stmt))
        if node.else_body is not None:
            sources.append(
                "__condition_{level} = np.logical_not(__condition_{level})".format(
                    level=self.conditions_depth, condition=self.visit(node.condition)
                )
            )
            for stmt in node.else_body.stmts:
                sources.extend(self._visit_branch_stmt(stmt))

        self.conditions_depth -= 1
        # return "\n".join(sources)
        return sources


class NumPyModuleGenerator(gt_backend.BaseModuleGenerator):
    def __init__(self):
        super().__init__()
        self.source_generator = NumPySourceGenerator(
            indent_size=self.TEMPLATE_INDENT_SIZE,
            origin_marker="__O",
            domain_arg_name=self.DOMAIN_ARG_NAME,
            origin_arg_name=self.ORIGIN_ARG_NAME,
            splitters_name=self.SPLITTERS_NAME,
            numpy_prefix="np",
            interval_k_start_name="interval_k_start",
            interval_k_end_name="interval_k_end",
        )

    def generate_module_members(self):
        source = """
def vectorized_ternary_op(*, condition, then_expr, else_expr, dtype):
    return np.choose(
        condition,
        [else_expr, then_expr],
        out=np.empty(
            np.max(
                (
                    np.asanyarray(condition).shape,
                    np.asanyarray(then_expr).shape,
                    np.asanyarray(else_expr).shape,
                ),
                axis=0,
            ),
            dtype=dtype,
        ),
    )
"""
        return source

    def generate_implementation(self):
        block = gt_text.TextBlock(indent_size=self.TEMPLATE_INDENT_SIZE)
        self.source_generator(self.builder.implementation_ir, block)
        if self.builder.options.backend_opts.get("ignore_np_errstate", True):
            source = "with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):\n"
            source += textwrap.indent(block.text, " " * self.TEMPLATE_INDENT_SIZE)
        else:
            source = block.text
        return source


def numpy_layout(mask):
    ctr = iter(range(sum(mask)))
    layout = [next(ctr) if m else None for m in mask]
    return tuple(layout)


def numpy_is_compatible_layout(field):
    return sum(field.shape) > 0


def numpy_is_compatible_type(field):
    return isinstance(field, np.ndarray)


@gt_backend.register
class NumPyBackend(gt_backend.BaseBackend, gt_backend.PurePythonBackendCLIMixin):
    """Pure Python backend using NumPy for faster computations than the debug backend.

    Other Parameters
    ----------------
    Backend options include:
    - ignore_np_errstate: `bool`
        If False, does not ignore NumPy floating-point errors. (`True` by default.)
    """

    name = "numpy"
    options = {"ignore_np_errstate": {"versioning": True}}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": numpy_layout,
        "is_compatible_layout": numpy_is_compatible_layout,
        "is_compatible_type": numpy_is_compatible_type,
    }

    languages = {"computation": "python", "bindings": []}

    MODULE_GENERATOR_CLASS = NumPyModuleGenerator
