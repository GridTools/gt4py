# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

from typing import TYPE_CHECKING

import numpy as np

from gt4py import backend as gt_backend
from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir
from gt4py.storage import StorageDefaults
from gt4py.utils import text as gt_text

from .python_generator import PythonSourceGenerator


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder


class DebugSourceGenerator(PythonSourceGenerator):
    def _make_field_accessor(self, name: str, origin=None):
        if origin is None:
            origin = "{origin_arg}['{name}']".format(origin_arg=self.origin_arg_name, name=name)
        source_lines = [
            "{name}{marker} = _Accessor(np.asarray({name}), {origin})".format(
                marker=self.origin_marker, name=name, origin=origin
            )
        ]

        return source_lines

    def _make_regional_computation(self, iteration_order, interval_definition):
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

        range_expr = "range({args})".format(args=", ".join(a for a in range_args))
        seq_axis = self.impl_node.domain.sequential_axis.name
        source_lines.append("for {ax} in {range_expr}:".format(ax=seq_axis, range_expr=range_expr))

        return source_lines

    def make_temporary_field(self, name: str, dtype: gt_ir.DataType, extent: gt_definitions.Extent):
        source_lines = super().make_temporary_field(name, dtype, extent)
        source_lines.extend(self._make_field_accessor(name, extent.to_boundary().lower_indices))

        return source_lines

    def make_stage_source(self, iteration_order: gt_ir.IterationOrder, regions: list):
        extent = self.block_info.extent
        lower_extent = extent.lower_indices
        upper_extent = extent.upper_indices
        seq_axis_name = self.impl_node.domain.sequential_axis.name
        axes_names = self.impl_node.domain.axes_names

        # Create IJ for-loops
        ij_loop_lines = []
        for d in range(extent.ndims):
            axis_name = axes_names[d]
            if axis_name != seq_axis_name:
                i = d + 1
                start_expr = "{:+d}".format(lower_extent[d]) if lower_extent[d] != 0 else ""
                size_expr = "{dom}[{d}]".format(dom=self.domain_arg_name, d=d)
                size_expr += " {:+d}".format(upper_extent[d]) if upper_extent[d] != 0 else ""
                range_expr = "range({args})".format(
                    args=", ".join(a for a in (start_expr, size_expr, "") if a)
                )
                ij_loop_lines.append(
                    " " * self.indent_size * i
                    + "for {ax} in {range_expr}:".format(ax=axis_name, range_expr=range_expr)
                )

        # Create K for-loop: computation body is split in different vertical regions
        source_lines = []
        assert sorted(regions, reverse=iteration_order == gt_ir.IterationOrder.BACKWARD) == regions

        for bounds, body_sources in regions:
            region_lines = self._make_regional_computation(iteration_order, bounds)
            source_lines.extend(region_lines)
            source_lines.extend(ij_loop_lines)
            source_lines.extend(
                " " * self.indent_size * extent.ndims + line for line in body_sources
            )

        return source_lines

    # ---- Visitor handlers ----
    def visit_FieldRef(self, node: gt_ir.FieldRef):
        assert node.name in self.block_info.accessors
        index = []
        for ax in self.domain.axes_names:
            offset = "{:+d}".format(node.offset[ax]) if ax in node.offset else ""
            index.append("{ax}{offset}".format(ax=ax, offset=offset))

        source = "{name}{marker}[{index}]".format(
            marker=self.origin_marker, name=node.name, index=", ".join(index)
        )

        return source

    def visit_StencilImplementation(self, node: gt_ir.StencilImplementation):
        self.sources.empty_line()

        # Accessors for IO fields
        self.sources.append("# Accessors for origin-based indexing")
        for info in node.api_signature:
            if info.name in node.fields and info.name not in node.unreferenced:
                self.sources.extend(self._make_field_accessor(info.name))
        self.sources.empty_line()

        super().visit_StencilImplementation(node)

    def visit_TernaryOpExpr(self, node: gt_ir.TernaryOpExpr):
        then_fmt = "({})" if isinstance(node.then_expr, gt_ir.CompositeExpr) else "{}"
        else_fmt = "({})" if isinstance(node.else_expr, gt_ir.CompositeExpr) else "{}"
        source = "{np}.{dtype}({then_expr} if {condition} else {else_expr})".format(
            condition=self.visit(node.condition),
            then_expr=then_fmt.format(self.visit(node.then_expr)),
            else_expr=else_fmt.format(self.visit(node.else_expr)),
            dtype=node.data_type.dtype.name,
            np=self.numpy_prefix,
        )

        return source

    def visit_If(self, node: gt_ir.If):
        body_sources = gt_text.TextBlock()
        body_sources.append("if {condition}:".format(condition=self.visit(node.condition)))
        body_sources.indent()
        for stmt in node.main_body.stmts:
            body_sources.extend(self.visit(stmt))
        body_sources.dedent()
        if node.else_body:
            body_sources.append("else:")
            body_sources.indent()

            for stmt in node.else_body.stmts:
                body_sources.extend(self.visit(stmt))
            body_sources.dedent()
        return ["".join([str(item) for item in line]) for line in body_sources.lines]


class DebugModuleGenerator(gt_backend.BaseModuleGenerator):
    def __init__(self):
        super().__init__()
        self.source_generator = DebugSourceGenerator(
            indent_size=self.TEMPLATE_INDENT_SIZE,
            origin_marker="_at",
            domain_arg_name=self.DOMAIN_ARG_NAME,
            origin_arg_name=self.ORIGIN_ARG_NAME,
            splitters_name=self.SPLITTERS_NAME,
            numpy_prefix="np",
        )

    def generate_module_members(self):
        source = """
class _Accessor:
    def __init__(self, array, origin):
        self.array = array
        self.origin = origin

    def _shift(self, index):
        return tuple(i + offset for i, offset in zip(index, self.origin))

    def __getitem__(self, index):
        return self.array[self._shift(index)]

    def __setitem__(self, index, value):
        self.array[self._shift(index)] = value
"""
        return source

    def generate_implementation(self):
        sources = gt_text.TextBlock(indent_size=self.TEMPLATE_INDENT_SIZE)
        self.source_generator(self.builder.implementation_ir, sources)

        return sources.text

    def generate_imports(self) -> str:
        source = (
            """
import math
"""
            + super().generate_imports()
        )
        return source


def debug_layout(mask):
    ctr = iter(range(sum(mask)))
    layout = [next(ctr) if m else None for m in mask]
    return tuple(layout)


def debug_is_compatible_layout(field):
    return sum(field.shape) > 0


@gt_backend.register
class DebugBackend(gt_backend.BaseBackend, gt_backend.PurePythonBackendCLIMixin):
    """Pure Python backend, unoptimized for debugging."""

    name = "debug"
    options = {}
    compute_device = "cpu"
    assert_specified_layout = False
    storage_defaults = StorageDefaults()

    languages = {"computation": "python", "bindings": []}

    MODULE_GENERATOR_CLASS = DebugModuleGenerator
