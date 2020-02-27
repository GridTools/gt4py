# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
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

import abc
import functools
import numbers
import os
import types

import jinja2

from gt4py import analysis as gt_analysis
from gt4py import backend as gt_backend
from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils
from gt4py.utils import text as gt_text


class _MaxKOffsetExtractor(gt_ir.IRNodeVisitor):
    @classmethod
    def apply(cls, root_node):
        return cls()(root_node)

    def __init__(self):
        self.max_offset = 2

    def __call__(self, node):
        self.visit(node)
        return self.max_offset

    def visit_AxisBound(self, node: gt_ir.AxisBound):
        self.max_offset = max(self.max_offset, abs(node.offset) + 1)


_extract_max_k_offset = _MaxKOffsetExtractor.apply


class GTPyExtGenerator(gt_ir.IRNodeVisitor):

    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
    TEMPLATE_FILES = {
        "computation.hpp": "computation.hpp.in",
        "computation.src": "computation.src.in",
        "bindings.cpp": "bindings.cpp.in",
    }

    OP_TO_CPP = {
        gt_ir.UnaryOperator.POS: "+",
        gt_ir.UnaryOperator.NEG: "-",
        gt_ir.UnaryOperator.NOT: "!",
        gt_ir.BinaryOperator.ADD: "+",
        gt_ir.BinaryOperator.SUB: "-",
        gt_ir.BinaryOperator.MUL: "*",
        gt_ir.BinaryOperator.DIV: "/",
        gt_ir.BinaryOperator.POW: lambda lhs, rhs: "pow({lhs}, {rhs})".format(lhs=lhs, rhs=rhs),
        gt_ir.BinaryOperator.AND: "&&",
        gt_ir.BinaryOperator.OR: "||",
        gt_ir.BinaryOperator.LT: "<",
        gt_ir.BinaryOperator.LE: "<=",
        gt_ir.BinaryOperator.EQ: "==",
        gt_ir.BinaryOperator.GE: ">=",
        gt_ir.BinaryOperator.GT: ">",
        gt_ir.BinaryOperator.NE: "!=",
    }

    DATA_TYPE_TO_CPP = {
        gt_ir.DataType.INT8: "int8_t",
        gt_ir.DataType.INT16: "int16_t",
        gt_ir.DataType.INT32: "int32_t",
        gt_ir.DataType.INT64: "int64_t",
        gt_ir.DataType.FLOAT32: "float32_t",
        gt_ir.DataType.FLOAT64: "float64_t",
    }

    def __init__(self, class_name, module_name, backend, options):
        self.class_name = class_name
        self.module_name = module_name
        self.backend = backend
        self.options = options

        self.templates = {}
        for key, file_name in self.TEMPLATE_FILES.items():
            with open(os.path.join(self.TEMPLATE_DIR, file_name), "r") as f:
                self.templates[key] = jinja2.Template(f.read())
        self.impl_node = None
        self.stage_symbols = None
        self.apply_block_symbols = None
        self.declared_symbols = None

    def __call__(self, impl_node: gt_ir.StencilImplementation):
        assert isinstance(impl_node, gt_ir.StencilImplementation)
        assert impl_node.domain.sequential_axis.name == gt_definitions.CartesianSpace.Axis.K.name

        self.impl_node = impl_node

        self.domain = impl_node.domain
        self.k_splitters = []
        # k_ax = self.domain.sequential_axis.name
        # if k_ax in self.impl_node.axis_splitters:
        #     for item in self.impl_node.axis_splitters[k_ax]:
        #         if item.is_scalar:
        #             values = [(item.name, None)]
        #         else:
        #             values = [(item.name, i) for i in range(item.length)]
        #         self.k_splitters.extend(values)

        source = self.visit(impl_node)

        return source

    def _make_cpp_value(self, value):
        if isinstance(value, numbers.Number):
            if isinstance(value, bool):
                value = int(value)
            result = str(value)
        else:
            result = None

        return result

    def _make_cpp_type(self, data_type: gt_ir.DataType):
        result = self.DATA_TYPE_TO_CPP[data_type]

        return result

    def _make_cpp_variable(self, decl: gt_ir.VarDecl):
        result = "{t} {name}:".format(t=self.DATA_TYPE_TO_CPP[decl.data_type], name=decl.name)

        return result

    def visit_ScalarLiteral(self, node: gt_ir.ScalarLiteral):
        source = "{dtype}{{{value}}}".format(
            dtype=self.DATA_TYPE_TO_CPP[node.data_type], value=node.value
        )

        return source

    def visit_FieldRef(self, node: gt_ir.FieldRef, **kwargs):
        assert node.name in self.apply_block_symbols
        offset = [node.offset.get(name, 0) for name in self.domain.axes_names]
        if not all(i == 0 for i in offset):
            idx = ", ".join(str(i) for i in offset)
        else:
            idx = ""
        source = "eval({name}({idx}))".format(name=node.name, idx=idx)

        return source

    def visit_VarRef(self, node: gt_ir.VarRef, *, write_context=False):
        assert node.name in self.apply_block_symbols

        if write_context and node.name not in self.declared_symbols:
            self.declared_symbols.add(node.name)
            source = self._make_cpp_type(self.apply_block_symbols[node.name].data_type) + " "
        else:
            source = ""

        idx = ", ".join(str(i) for i in node.index) if node.index else ""

        if node.name in self.impl_node.parameters:
            source += "eval({name}({idx}))".format(name=node.name, idx=idx)
        else:
            source += "{name}".format(name=node.name)
            if idx:
                source += "[{idx}]".format(idx=idx)

        return source

    def visit_UnaryOpExpr(self, node: gt_ir.UnaryOpExpr):
        fmt = "({})" if isinstance(node.arg, gt_ir.CompositeExpr) else "{}"
        source = "{op}{expr}".format(
            op=self.OP_TO_CPP[node.op], expr=fmt.format(self.visit(node.arg))
        )

        return source

    def visit_BinOpExpr(self, node: gt_ir.BinOpExpr):
        lhs_fmt = "({})" if isinstance(node.lhs, gt_ir.CompositeExpr) else "{}"
        lhs_expr = lhs_fmt.format(self.visit(node.lhs))
        rhs_fmt = "({})" if isinstance(node.rhs, gt_ir.CompositeExpr) else "{}"
        rhs_expr = rhs_fmt.format(self.visit(node.rhs))

        cpp_op = self.OP_TO_CPP[node.op]
        if callable(cpp_op):
            source = cpp_op(lhs_expr, rhs_expr)
        else:
            source = "{lhs} {op} {rhs}".format(lhs=lhs_expr, op=cpp_op, rhs=rhs_expr)

        return source

    def visit_TernaryOpExpr(self, node: gt_ir.TernaryOpExpr):
        then_fmt = "({})" if isinstance(node.then_expr, gt_ir.CompositeExpr) else "{}"
        else_fmt = "({})" if isinstance(node.else_expr, gt_ir.CompositeExpr) else "{}"
        source = "({condition}) ? {then_expr} : {else_expr}".format(
            condition=self.visit(node.condition),
            then_expr=then_fmt.format(self.visit(node.then_expr)),
            else_expr=else_fmt.format(self.visit(node.else_expr)),
        )

        return source

    def visit_Assign(self, node: gt_ir.Assign):
        lhs = self.visit(node.target, write_context=True)
        rhs = self.visit(node.value)
        source = "{lhs} = {rhs};".format(lhs=lhs, rhs=rhs)

        return [source]

    def visit_BlockStmt(self, node: gt_ir.BlockStmt):
        body_sources = gt_text.TextBlock()
        for stmt in node.stmts:
            body_sources.extend(self.visit(stmt))

        return body_sources.text

    def visit_If(self, node: gt_ir.If):
        body_sources = gt_text.TextBlock()
        body_sources.append("if ({condition}) {{".format(condition=self.visit(node.condition)))
        for stmt in node.main_body.stmts:
            body_sources.extend(self.visit(stmt))
        if node.else_body:
            body_sources.append("} else {")

            for stmt in node.else_body.stmts:
                body_sources.extend(self.visit(stmt))

        body_sources.append("}")
        return body_sources

    def visit_AxisBound(self, node: gt_ir.AxisBound):
        if node.level == gt_ir.LevelMarker.START:
            level = 0
        elif node.level == gt_ir.LevelMarker.END:
            level = len(self.k_splitters) + 1
        else:
            assert isinstance(node.level, gt_ir.VarRef)
            assert len(node.level.index) == 1
            level = self.k_splitters.index((node.name, node.index[0]))

        # Shift offset to make it relative to the splitter (in-between levels)
        offset = node.offset + 1 if node.offset >= 0 else node.offset

        return level, offset

    def visit_AxisInterval(self, node: gt_ir.AxisInterval):
        start_splitter, start_offset = self.visit(node.start)
        end_splitter, end_offset = self.visit(node.end)

        # Transform range from excluded endpoint to including endpoint
        end_offset = -1 if end_offset == 1 else end_offset - 1

        return (start_splitter, start_offset), (end_splitter, end_offset)

    def visit_ApplyBlock(self, node: gt_ir.ApplyBlock):
        # if node.intervals:
        #     assert set(node.intervals.keys()) == {self.domain.sequential_axis.name}
        #     interval_definition = self.visit(node.intervals[self.domain.sequential_axis.name])
        # else:
        #     interval_definition = (None, None)
        interval_definition = self.visit(node.interval)

        self.declared_symbols = set()
        self.apply_block_symbols = {**self.stage_symbols, **node.local_symbols}
        body_sources = self.visit(node.body)

        return interval_definition, body_sources

    def visit_Stage(self, node: gt_ir.Stage):
        # Initialize symbols for the generation of references in this stage
        # self.stage_symbols = dict(node.local_symbols)
        self.stage_symbols = {}
        args = []
        for accessor in node.accessors:
            self.stage_symbols[accessor.symbol] = accessor
            arg = {"name": accessor.symbol, "access_type": "in", "extent": None}
            if isinstance(accessor, gt_ir.FieldAccessor):
                arg["access_type"] = (
                    "in" if accessor.intent == gt_ir.AccessIntent.READ_ONLY else "inout"
                )
                arg["extent"] = gt_utils.flatten(accessor.extent)
            args.append(arg)

        # Create regions and computations
        regions = []
        for apply_block in node.apply_blocks:
            interval_definition, body_sources = self.visit(apply_block)
            regions.append(
                {
                    "interval_start": interval_definition[0],
                    "interval_end": interval_definition[1],
                    "body": body_sources,
                }
            )
        functor_content = {"args": args, "regions": regions}

        return functor_content

    def visit_StencilImplementation(self, node: gt_ir.StencilImplementation):
        offset_limit = _extract_max_k_offset(node)
        k_axis = {"n_intervals": 1, "offset_limit": offset_limit}
        max_extent = functools.reduce(
            lambda a, b: a | b, node.fields_extents.values(), gt_definitions.Extent.zeros()
        )
        halo_sizes = tuple(max(lower, upper) for lower, upper in max_extent.to_boundary())
        constants = {}
        if node.externals:
            for name, value in node.externals.items():
                value = self._make_cpp_value(name)
                if value is not None:
                    constants[name] = value

        arg_fields = []
        tmp_fields = []
        storage_ids = []
        max_ndim = 0
        for name, field_decl in node.fields.items():
            if name not in node.unreferenced:
                max_ndim = max(max_ndim, len(field_decl.axes))
                field_attributes = {
                    "name": field_decl.name,
                    "dtype": self._make_cpp_type(field_decl.data_type),
                }
                if field_decl.is_api:
                    if field_decl.layout_id not in storage_ids:
                        storage_ids.append(field_decl.layout_id)
                    field_attributes["layout_id"] = storage_ids.index(field_decl.layout_id)
                    arg_fields.append(field_attributes)
                else:
                    tmp_fields.append(field_attributes)

        parameters = [
            {"name": parameter.name, "dtype": self._make_cpp_type(parameter.data_type)}
            for name, parameter in node.parameters.items()
            if name not in node.unreferenced
        ]

        stage_functors = {}
        for multi_stage in node.multi_stages:
            for group in multi_stage.groups:
                for stage in group.stages:
                    stage_functors[stage.name] = self.visit(stage)

        multi_stages = []
        for multi_stage in node.multi_stages:
            steps = [[stage.name for stage in group.stages] for group in multi_stage.groups]
            multi_stages.append({"exec": str(multi_stage.iteration_order).lower(), "steps": steps})

        template_args = dict(
            arg_fields=arg_fields,
            constants=constants,
            gt_backend=self.backend,
            halo_sizes=halo_sizes,
            k_axis=k_axis,
            module_name=self.module_name,
            multi_stages=multi_stages,
            parameters=parameters,
            stage_functors=stage_functors,
            stencil_unique_name=self.class_name,
            tmp_fields=tmp_fields,
        )

        sources = {}
        for key, template in self.templates.items():
            sources[key] = template.render(**template_args)

        return sources


class GTPyModuleGenerator(gt_backend.BaseModuleGenerator):
    def __init__(
        self, backend_class, options, **kwargs,
    ):
        super().__init__(backend_class, options, **kwargs)
        self.pyext_module_name = kwargs["pyext_module_name"]
        self.pyext_file_path = kwargs["pyext_file_path"]
        with open(self.TEMPLATE_PATH, "r") as f:
            self.template = jinja2.Template(f.read())

    def generate_imports(self) -> str:
        source = """
import functools

from gt4py import utils as gt_utils

pyext_module = gt_utils.make_module_from_file("{pyext_module_name}", "{pyext_file_path}", public_import=True)
        """.format(
            pyext_module_name=self.pyext_module_name, pyext_file_path=self.pyext_file_path,
        )

        return source

    def generate_implementation(self) -> str:
        sources = gt_text.TextBlock(
            indent_size=gt_backend.BaseModuleGenerator.TEMPLATE_INDENT_SIZE
        )

        args = []
        for arg in self.implementation_ir.api_signature:
            if arg.name not in self.implementation_ir.unreferenced:
                args.append(arg.name)
                if arg.name in self.implementation_ir.fields:
                    args.append("list(_origin_['{}'])".format(arg.name))

        source = """
# Load or generate a GTComputation object for the current domain size
pyext_module.run_computation(list(_domain_), {run_args}, exec_info)
""".format(
            run_args=", ".join(args)
        )
        if self.backend_name == "gtcuda":
            source = (
                source
                + """import cupy
cupy.cuda.Device(0).synchronize()
"""
            )
        source = source + (
            """if exec_info is not None:
    exec_info["run_end_time"] = time.perf_counter()
"""
        )
        sources.extend(source.splitlines())

        return sources.text


class BaseGTBackend(gt_backend.BasePyExtBackend):

    MODULE_GENERATOR_CLASS = GTPyModuleGenerator

    PYEXT_GENERATOR_CLASS = GTPyExtGenerator

    GT_BACKEND_OPTS = {
        "verbose": {"versioning": False},
        "clean": {"versioning": False},
        "debug_mode": {"versioning": True},
        "add_profile_info": {"versioning": True},
    }

    @classmethod
    def generate(
        cls,
        stencil_id: gt_definitions.StencilID,
        definition_ir: gt_ir.StencilDefinition,
        definition_func: types.FunctionType,
        options: gt_definitions.BuildOptions,
    ):
        # TODO: move this import to the top and find a better way to avoid circular imports
        from gt4py import gt_src_manager

        cls._check_options(options)
        implementation_ir = gt_analysis.transform(definition_ir, options)

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt_src_manager.has_gt_sources() and not gt_src_manager.install_gt_sources():
            raise RuntimeError("Missing GridTools sources.")
        pyext_module_name, pyext_file_path = cls.generate_extension(
            stencil_id, implementation_ir, options
        )

        # Generate and return the Python wrapper class
        return super(BaseGTBackend, cls)._generate_module(
            stencil_id,
            implementation_ir,
            definition_func,
            options,
            extra_cache_info={"pyext_file_path": pyext_file_path},
            implementation_ir=implementation_ir,
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
        )

    @classmethod
    @abc.abstractmethod
    def generate_extension(
        cls,
        stencil_id: gt_definitions.StencilID,
        implementation_ir: gt_ir.StencilImplementation,
        options: gt_definitions.BuildOptions,
    ):
        pass
