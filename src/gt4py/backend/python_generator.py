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

import types

from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir
from gt4py.utils import text as gt_text


class PythonSourceGenerator(gt_ir.IRNodeVisitor):

    OP_TO_PYTHON = {
        **gt_ir.UnaryOperator.IR_OP_TO_PYTHON_SYMBOL,
        **gt_ir.BinaryOperator.IR_OP_TO_PYTHON_SYMBOL,
    }

    NATIVE_FUNC_TO_PYTHON = {
        gt_ir.NativeFunction.ABS: "abs",
        gt_ir.NativeFunction.MIN: "min",
        gt_ir.NativeFunction.MAX: "max",
        gt_ir.NativeFunction.MOD: "math.fmod",
        gt_ir.NativeFunction.SIN: "math.sin",
        gt_ir.NativeFunction.COS: "math.cos",
        gt_ir.NativeFunction.TAN: "math.tan",
        gt_ir.NativeFunction.ARCSIN: "math.asin",
        gt_ir.NativeFunction.ARCCOS: "math.acos",
        gt_ir.NativeFunction.ARCTAN: "math.atan",
        gt_ir.NativeFunction.SQRT: "math.sqrt",
        gt_ir.NativeFunction.EXP: "math.exp",
        gt_ir.NativeFunction.LOG: "math.log",
        gt_ir.NativeFunction.ISFINITE: "math.isfinite",
        gt_ir.NativeFunction.ISINF: "math.isinf",
        gt_ir.NativeFunction.ISNAN: "math.isnan",
        gt_ir.NativeFunction.FLOOR: "math.floor",
        gt_ir.NativeFunction.CEIL: "math.ceil",
        gt_ir.NativeFunction.TRUNC: "math.trunc",
    }

    def __init__(
        self,
        *,
        indent_size,
        origin_marker,
        domain_arg_name,
        origin_arg_name,
        splitters_name,
        numpy_prefix,
    ):
        self.indent_size = indent_size
        self.origin_marker = origin_marker
        self.domain_arg_name = domain_arg_name
        self.origin_arg_name = origin_arg_name
        self.splitters_name = splitters_name
        self.numpy_prefix = numpy_prefix

        self.impl_node = None
        self.sources = None

        self.k_axis_splitters = None
        self.k_splitters_value = None
        self.io_field_names = None
        self.temp_field_names = None
        self.io_field_names = None
        self.param_names = None

        self.var_refs_defined = set()

    def __call__(self, impl_node: gt_ir.Node, sources: gt_text.TextBlock):
        assert isinstance(impl_node, gt_ir.StencilImplementation)
        assert impl_node.domain.sequential_axis.name == gt_definitions.CartesianSpace.Axis.K.name

        self.impl_node = impl_node
        self.domain = impl_node.domain
        self.block_info = types.SimpleNamespace()

        k_ax = self.domain.sequential_axis.name
        k_ax_idx = self.domain.axes_names.index(k_ax)
        # assert set(self.impl_node.axis_splitters.keys()) <= {k_ax}
        # self.k_axis_splitters = self.impl_node.axis_splitters.get(k_ax, [])
        self.k_axis_splitters = []
        self.k_splitters_value = ["0"]
        # for item in self.k_axis_splitters:
        #     if item.is_scalar:
        #         self.k_splitters_value.append(item.name)
        #     else:
        #         self.k_splitters_value.extend(
        #             ["{}[{}]".format(item.name, i) for i in range(item.length)]
        #         )
        self.k_splitters_value.append(
            "{dom}[{idx}]".format(dom=self.domain_arg_name, idx=k_ax_idx)
        )

        self.sources = sources
        self.visit(impl_node)

        return self.sources

    def make_temporary_field(
        self, name: str, data_type: gt_ir.DataType, extent: gt_definitions.Extent
    ):
        source_lines = []
        boundary = extent.to_boundary()
        shape = ", ".join(
            "{domain}[{d}]{size}".format(
                domain=self.domain_arg_name, d=d, size=" {:+d}".format(size) if size > 0 else ""
            )
            for d, size in enumerate(boundary.frame_size)
        )
        source_lines.append(
            "{name} = {np_prefix}.empty(({shape}), dtype={np_prefix}.{dtype})".format(
                name=name, np_prefix=self.numpy_prefix, shape=shape, dtype=data_type.dtype.name
            )
        )

        return source_lines

    def make_stage_source(self, iteration_order: gt_ir.IterationOrder, regions: list):
        raise NotImplementedError()

    # ---- Visitor handlers ----
    def generic_visit(self, node: gt_ir.Node, **kwargs):
        raise RuntimeError("Invalid IR node: {}".format(node))

    def visit_Cast(self, node: gt_ir.Cast):
        return self.visit(node.expr)

    def visit_Decl(self, node: gt_ir.Decl):
        raise NotImplementedError()

    def visit_Statement(self, node: gt_ir.Statement):
        raise NotImplementedError()

    def visit_ScalarLiteral(self, node: gt_ir.ScalarLiteral):
        return str(node.value)

    def visit_FieldRef(self, node: gt_ir.FieldRef):
        raise NotImplementedError()

    def visit_VarRef(self, node: gt_ir.VarRef):
        assert (
            node.name in self.block_info.symbols or node.name in self.param_names
        ), "Unknown variable '{}'".format(node.name)

        idx = ", ".join(str(i) for i in node.index) if node.index else ""
        source = "{name}".format(name=node.name)
        if idx:
            source += "[{idx}]".format(idx=idx)

        return source

    def visit_UnaryOpExpr(self, node: gt_ir.UnaryOpExpr):
        fmt = "({})" if isinstance(node.arg, gt_ir.CompositeExpr) else "{}"
        source = "{op} {expr}".format(
            op=self.OP_TO_PYTHON[node.op], expr=fmt.format(self.visit(node.arg))
        )

        return source

    def visit_BinOpExpr(self, node: gt_ir.BinOpExpr):
        lhs_fmt = "({})" if isinstance(node.lhs, gt_ir.CompositeExpr) else "{}"
        rhs_fmt = "({})" if isinstance(node.rhs, gt_ir.CompositeExpr) else "{}"
        source = "{lhs} {op} {rhs}".format(
            lhs=lhs_fmt.format(self.visit(node.lhs)),
            op=self.OP_TO_PYTHON[node.op],
            rhs=rhs_fmt.format(self.visit(node.rhs)),
        )

        return source

    def visit_NativeFuncCall(self, node: gt_ir.NativeFuncCall):
        call = self.NATIVE_FUNC_TO_PYTHON[node.func]
        args = ",".join(self.visit(arg) for arg in node.args)
        return f"{call}({args})"

    def visit_TernaryOpExpr(self, node: gt_ir.TernaryOpExpr):
        then_fmt = "({})" if isinstance(node.then_expr, gt_ir.CompositeExpr) else "{}"
        else_fmt = "({})" if isinstance(node.else_expr, gt_ir.CompositeExpr) else "{}"
        source = "({then_expr} if {condition} else {else_expr})".format(
            condition=self.visit(node.condition),
            then_expr=then_fmt.format(self.visit(node.then_expr)),
            else_expr=else_fmt.format(self.visit(node.else_expr)),
        )

        return source

    def visit_Assign(self, node: gt_ir.Assign):
        lhs = self.visit(node.target)
        rhs = self.visit(node.value)

        source = "{lhs} = {rhs}".format(lhs=lhs, rhs=rhs)

        # self.var_refs_defined is used in the numpy backend
        if isinstance(node.target, gt_ir.VarRef):
            self.var_refs_defined.add(node.target.name)

        return source
        # if node.target.name in self.state["variables"]:
        #     self.state["init_stmts"].append(source)
        # else:
        #     self.state["body_stmts"].append(source)

    # def visit_Return(self, node: gt_ir.Return):
    #     self.state["body_stmts"].append("return")

    def visit_BlockStmt(self, node: gt_ir.BlockStmt):
        body_sources = []
        for stmt in node.stmts:
            stmt_source = self.visit(stmt)
            if isinstance(stmt_source, list):
                body_sources.extend(stmt_source)
            else:
                body_sources.append(stmt_source)

        return body_sources

    def visit_AxisBound(self, node: gt_ir.AxisBound):
        if node.level == gt_ir.LevelMarker.START:
            level = 0
        elif node.level == gt_ir.LevelMarker.END:
            level = len(self.k_splitters_value) - 1
        else:
            assert False, "Not implemented yet"
            assert isinstance(node.level, gt_ir.VarRef)
            assert len(node.level.index) == 1
            level = self.k_splitters_value.index((node.name, node.index[0]))

        offset = node.offset

        return level, offset

    def visit_AxisInterval(self, node: gt_ir.AxisInterval):
        start_splitter, start_offset = self.visit(node.start)
        end_splitter, end_offset = self.visit(node.end)

        return (start_splitter, start_offset), (end_splitter, end_offset)

    def visit_ApplyBlock(self, node: gt_ir.ApplyBlock):
        interval_definition = self.visit(node.interval)
        self.block_info.interval = interval_definition
        self.block_info.symbols = node.local_symbols
        body_sources = self.visit(node.body)

        return interval_definition, body_sources

    def visit_Stage(self, node: gt_ir.Stage, *, iteration_order):
        # Initialize symbols for the generation of references in this stage
        self.block_info.accessors = {accessor.symbol for accessor in node.accessors}
        self.block_info.iteration_order = iteration_order
        self.block_info.extent = node.compute_extent

        # Create regions and computations
        regions = []
        for apply_block in node.apply_blocks:
            regions.append(self.visit(apply_block))

        # Final assembly of the stage source
        self.sources += "# {}:".format(node.name)
        self.sources.extend(self.make_stage_source(iteration_order, regions))

    def visit_StencilImplementation(self, node: gt_ir.StencilImplementation):
        # Splitters declarations
        self.sources.empty_line()
        self.sources.append("# K splitters")
        self.sources.append(
            "{} = [{}]".format(self.splitters_name, ", ".join(self.k_splitters_value))
        )

        # Temporary field declarations
        self.param_names = set(
            info.name for info in node.api_signature if info.name in node.parameters
        )
        self.io_field_names = set(info.name for info in node.api_signature) - self.param_names
        self.temp_field_names = set(node.fields.keys()) - self.io_field_names

        if len(self.temp_field_names) > 0:
            self.sources.empty_line()
            self.sources.append("# Allocation of temporary fields")
            for name in self.temp_field_names:
                field = node.fields[name]
                self.sources.extend(
                    self.make_temporary_field(
                        field.name, field.data_type, node.fields_extents[field.name]
                    )
                )
            self.sources.empty_line()

        # Constants
        # self.constants = {}
        # self.sources.append("# Constants")
        # if len(node.constants) > 0:
        #     for name, decl in node.constants.items():
        #         src_name = self.state["constants"][name] = gt_utils.slugify(
        #             name, valid_symbols="", invalid_marker="_"
        #         )
        #         self.sources.append("{} = {}".format(src_name, decl.init.value))
        #     self.sources.empty_line()

        # Stages
        self.sources.append("# Computations")
        for multi_stage in node.multi_stages:
            for group in multi_stage.groups:
                for stage in group.stages:
                    self.visit(stage, iteration_order=multi_stage.iteration_order)
                    self.sources.append("")
