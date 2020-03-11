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
import inspect
import numbers
import os
import re
import types

import jinja2
import numpy as np

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils

from gt4py import backend as gt_backend
from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils

DOMAIN_AXES = gt_definitions.CartesianSpace.names


class FieldDeclCollector(gt_ir.IRNodeVisitor):
    @classmethod
    def apply(cls, definition_ir):
        return cls()(definition_ir)

    def __call__(self, definition_ir):
        self.fields = []
        self.visit(definition_ir)
        return self.fields

    def visit_FieldDecl(self, node: gt_ir.FieldDecl, **kwargs):
        # NOTE Add unstructured support here
        field_dimensions = sir_utils.make_field_dimensions_cartesian(
            [1 if ax in node.axes else 0 for ax in DOMAIN_AXES]
        )
        is_temporary = not node.is_api
        self.fields.append(
            sir_utils.make_field(
                name=node.name, dimensions=field_dimensions, is_temporary=is_temporary
            )
        )


class SIRConverter(gt_ir.IRNodeVisitor):
    @classmethod
    def apply(cls, definition_ir):
        return cls()(definition_ir)

    def __call__(self, definition_ir):
        return self.visit(definition_ir)

    def _make_global_variables(self, parameters: list, externals: dict):
        global_variables = SIR.GlobalVariableMap()

        for param in parameters:
            global_variables.map[param.name].is_constexpr = False
            if param.data_type in [gt_ir.DataType.BOOL]:
                global_variables.map[param.name].boolean_value = param.init or False
            elif param.data_type in [
                gt_ir.DataType.INT8,
                gt_ir.DataType.INT16,
                gt_ir.DataType.INT32,
                gt_ir.DataType.INT64,
            ]:
                global_variables.map[param.name].integer_value = param.init or 0
            elif param.data_type in [gt_ir.DataType.FLOAT32, gt_ir.DataType.FLOAT64]:
                global_variables.map[param.name].double_value = param.init or 0.0

        # for key, value in externals.items():
        #     if isinstance(value, numbers.Number):
        #         global_variables.map[key].is_constexpr = True
        #         if isinstance(value, bool):
        #             global_variables.map[key].boolean_value = value
        #         elif isinstance(value, int):
        #             global_variables.map[key].integer_value = value
        #         elif isinstance(value, float):
        #             global_variables.map[key].double_value = value

        return global_variables

    def visit_ScalarLiteral(self, node: gt_ir.ScalarLiteral, **kwargs):
        assert node.data_type != gt_ir.DataType.INVALID
        if node.data_type in (gt_ir.DataType.AUTO, gt_ir.DataType.DEFAULT):
            sir_type = SIR.BuiltinType.type_id = SIR.BuiltinType.Auto
        elif node.data_type in (
            gt_ir.DataType.INT8,
            gt_ir.DataType.INT16,
            gt_ir.DataType.INT32,
            gt_ir.DataType.INT64,
        ):
            sir_type = SIR.BuiltinType.Integer
        elif node.data_type in (gt_ir.DataType.FLOAT32, gt_ir.DataType.FLOAT64):
            sir_type = SIR.BuiltinType.Float
        else:
            assert False, "Unrecognized data type"
        return sir_utils.make_literal_access_expr(value=repr(node.value), type=sir_type)

    def visit_VarRef(self, node: gt_ir.VarRef, **kwargs):
        return sir_utils.make_var_access_expr(name=node.name, is_external=True)

    def visit_FieldRef(self, node: gt_ir.FieldRef, **kwargs):
        offset = [node.offset[ax] for ax in DOMAIN_AXES]
        return sir_utils.make_field_access_expr(name=node.name, offset=offset)

    def visit_UnaryOpExpr(self, node: gt_ir.UnaryOpExpr, **kwargs):
        op = node.op.python_symbol
        operand = self.visit(node.arg)
        return sir_utils.make_unary_operator(op, operand)

    def visit_BinOpExpr(self, node: gt_ir.BinOpExpr, **kwargs):
        left = self.visit(node.lhs)
        right = self.visit(node.rhs)
        op = node.op.python_symbol
        return sir_utils.make_binary_operator(left, op, right)

    def visit_TernaryOpExpr(self, node: gt_ir.TernaryOpExpr, **kwargs):
        cond = self.visit(node.condition)
        left = self.visit(node.then_expr)
        right = self.visit(node.else_expr)
        return sir_utils.make_ternary_operator(cond, left, right)

    def visit_BlockStmt(self, node: gt_ir.BlockStmt, *, make_block=True, **kwargs):
        stmts = [self.visit(stmt) for stmt in node.stmts if not isinstance(stmt, gt_ir.FieldDecl)]
        if make_block:
            stmts = sir_utils.make_block_stmt(stmts)
        return stmts

    def visit_Assign(self, node: gt_ir.Assign, **kwargs):
        left = self.visit(node.target)
        right = self.visit(node.value)
        stmt = sir_utils.make_assignment_stmt(left, right, "=")
        return stmt

    def visit_If(self, node: gt_ir.If, **kwargs):
        cond = sir_utils.make_expr_stmt(self.visit(node.condition))
        then_part = self.visit(node.main_body)
        else_part = self.visit(node.else_body)
        stmt = sir_utils.make_if_stmt(cond, then_part, else_part)
        return stmt

    def visit_AxisBound(self, node: gt_ir.AxisBound, **kwargs):
        assert isinstance(node.level, gt_ir.LevelMarker)
        level = SIR.Interval.Start if node.level == gt_ir.LevelMarker.START else SIR.Interval.End
        offset = node.offset
        return level, offset

    def visit_AxisInterval(self, node: gt_ir.AxisInterval, **kwargs):
        lower_level, lower_offset = self.visit(node.start)
        upper_level, upper_offset = self.visit(node.end)
        return sir_utils.make_interval(lower_level, upper_level, lower_offset, upper_offset)

    def visit_ComputationBlock(self, node: gt_ir.ComputationBlock, **kwargs):
        interval = self.visit(node.interval)

        body_ast = sir_utils.make_ast(self.visit(node.body, make_block=False))

        loop_order = (
            SIR.VerticalRegion.Backward
            if node.iteration_order == gt_ir.IterationOrder.BACKWARD
            else SIR.VerticalRegion.Forward
        )

        vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
            body_ast, interval, loop_order
        )

        return vertical_region_stmt

    def visit_StencilDefinition(self, node: gt_ir.StencilDefinition, **kwargs):
        stencils = []
        functions = []
        global_variables = self._make_global_variables(node.parameters, node.externals)

        fields = FieldDeclCollector.apply(node)  # [self.visit(field) for field in node.api_fields]
        stencil_ast = sir_utils.make_ast(
            [self.visit(computation) for computation in node.computations]
        )
        name = node.name.split(".")[-1]
        stencils.append(sir_utils.make_stencil(name=name, ast=stencil_ast, fields=fields))

        sir = sir_utils.make_sir(
            filename="<gt4py>",
            grid_type=SIR.GridType.Value("Cartesian"),
            stencils=stencils,
            functions=functions,
            global_variables=global_variables,
        )
        return sir


convert_to_SIR = SIRConverter.apply


class BaseDawnBackend(gt_backend.BasePyExtBackend):

    DAWN_BACKEND_NS = None
    DAWN_BACKEND_NAME = None
    DAWN_BACKEND_OPTS = {
        "add_profile_info": {"versioning": True},
        "clean": {"versioning": False},
        "debug_mode": {"versioning": True},
        "dump_sir": {"versioning": False},
        "verbose": {"versioning": False},
    }

    GT_BACKEND_T = None

    MODULE_GENERATOR_CLASS = gt_backend.PyExtModuleGenerator

    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
    TEMPLATE_FILES = {
        "computation.hpp": "computation.hpp.in",
        "computation.src": "dawn_computation.src.in",
        "bindings.cpp": "bindings.cpp.in",
    }

    _DATA_TYPE_TO_CPP = {
        gt_ir.DataType.INT8: "int",
        gt_ir.DataType.INT16: "int",
        gt_ir.DataType.INT32: "int",
        gt_ir.DataType.INT64: "int",
        gt_ir.DataType.FLOAT32: "double",
        gt_ir.DataType.FLOAT64: "double",
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

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt_src_manager.has_gt_sources() and not gt_src_manager.install_gt_sources():
            raise RuntimeError("Missing GridTools sources.")
        module_kwargs = {"implementation_ir": None}
        pyext_module_name, pyext_file_path = cls.generate_extension(
            stencil_id, definition_ir, options, module_kwargs=module_kwargs
        )

        # Generate and return the Python wrapper class
        return cls._generate_module(
            stencil_id,
            definition_ir,
            definition_func,
            options,
            extra_cache_info={"pyext_file_path": pyext_file_path},
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
            **module_kwargs,
        )

    @classmethod
    def generate_extension_sources(cls, stencil_id, definition_ir, options, gt_backend_t):
        sir = convert_to_SIR(definition_ir)
        stencil_short_name = stencil_id.qualified_name.split(".")[-1]

        backend_opts = dict(**options.backend_opts)
        backend_opts["backend"] = cls.DAWN_BACKEND_NAME
        dawn_backend = cls.DAWN_BACKEND_NS

        dump_sir_opt = backend_opts.get("dump_sir", False)
        if dump_sir_opt:
            if isinstance(dump_sir_opt, str):
                dump_sir_file = dump_sir_opt
            else:
                assert isinstance(dump_sir_opt, bool)
                dump_sir_file = f"{stencil_short_name}_gt4py.sir"

            with open(dump_sir_file, "w") as f:
                f.write(sir_utils.to_json(sir))

        dawn_opts = {
            key: value
            for key, value in backend_opts.items()
            if key in _DAWN_TOOLCHAIN_OPTIONS.keys()
        }
        source = dawn4py.compile(sir, **dawn_opts)
        stencil_unique_name = cls.get_pyext_class_name(stencil_id)
        module_name = cls.get_pyext_module_name(stencil_id)
        pyext_sources = {f"_dawn_{stencil_short_name}.hpp": source}

        arg_fields = [
            {"name": field.name, "dtype": cls._DATA_TYPE_TO_CPP[field.data_type], "layout_id": i}
            for i, field in enumerate(definition_ir.api_fields)
        ]
        header_file = "computation.hpp"
        parameters = []
        for parameter in definition_ir.parameters:
            if parameter.data_type in [gt_ir.DataType.BOOL]:
                dtype = "bool"
            elif parameter.data_type in [
                gt_ir.DataType.INT8,
                gt_ir.DataType.INT16,
                gt_ir.DataType.INT32,
                gt_ir.DataType.INT64,
            ]:
                dtype = "int"
            elif parameter.data_type in [gt_ir.DataType.FLOAT32, gt_ir.DataType.FLOAT64]:
                dtype = "double"
            else:
                assert False, "Wrong data_type for parameter"
            parameters.append({"name": parameter.name, "dtype": dtype})

        template_args = dict(
            arg_fields=arg_fields,
            dawn_backend=dawn_backend,
            gt_backend=gt_backend_t,
            header_file=header_file,
            module_name=module_name,
            parameters=parameters,
            stencil_short_name=stencil_short_name,
            stencil_unique_name=stencil_unique_name,
        )

        for key, file_name in cls.TEMPLATE_FILES.items():
            with open(os.path.join(cls.TEMPLATE_DIR, file_name), "r") as f:
                template = jinja2.Template(f.read())
                pyext_sources[key] = template.render(**template_args)

        return pyext_sources

    @classmethod
    def _generate_module(
        cls,
        stencil_id,
        definition_ir,
        definition_func,
        options,
        *,
        extra_cache_info=None,
        **kwargs,
    ):
        if options.dev_opts.get("code-generation", True):
            info = {}
            if definition_ir.sources is not None:
                info["sources"].update(
                    {
                        key: gt_utils.text.format_source(value, line_length=100)
                        for key, value in definition_ir.sources
                    }
                )
            else:
                info["sources"] = {}

            parallel_axes = definition_ir.domain.parallel_axes or []
            sequential_axis = definition_ir.domain.sequential_axis.name
            domain_info = gt_definitions.DomainInfo(
                parallel_axes=tuple(ax.name for ax in parallel_axes),
                sequential_axis=sequential_axis,
                ndims=len(parallel_axes) + (1 if sequential_axis else 0),
            )
            info["domain_info"] = repr(domain_info)

            info["field_info"] = field_info = {}
            info["parameter_info"] = parameter_info = {}

            fields = {item.name: item for item in definition_ir.api_fields}
            parameters = {item.name: item for item in definition_ir.parameters}

            halo_size = kwargs.pop("halo_size")
            boundary = gt_definitions.Boundary(
                ([(halo_size, halo_size)] * len(domain_info.parallel_axes)) + [(0, 0)]
            )

            for arg in definition_ir.api_signature:
                if arg.name in fields:
                    field_info[arg.name] = gt_definitions.FieldInfo(
                        access=gt_definitions.AccessKind.READ_WRITE,
                        dtype=fields[arg.name].data_type.dtype,
                        boundary=boundary,
                    )
                else:
                    parameter_info[arg.name] = gt_definitions.ParameterInfo(
                        dtype=parameters[arg.name].data_type.dtype
                    )

            if definition_ir.externals:
                info["gt_constants"] = {
                    name: repr(value)
                    for name, value in definition_ir.externals.items()
                    if isinstance(value, numbers.Number)
                }
            else:
                info["gt_constants"] = {}

            info["gt_options"] = {
                key: value for key, value in options.as_dict().items() if key not in ["build_info"]
            }

            info["unreferenced"] = {}

            generator = cls.MODULE_GENERATOR_CLASS(cls)
            module_source = generator(stencil_id, definition_ir, options, meta_info=info, **kwargs)

            file_name = cls.get_stencil_module_path(stencil_id)
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "w") as f:
                f.write(module_source)
            extra_cache_info = extra_cache_info or {}
            cls.update_cache(stencil_id, extra_cache_info)

        return cls._load(stencil_id, definition_func)

    @classmethod
    def _generic_generate_extension(
        cls, stencil_id, definition_ir, options, *, uses_cuda=False, **kwargs
    ):
        module_kwargs = kwargs["module_kwargs"]
        dawn_src_file = f"_dawn_{stencil_id.qualified_name.split('.')[-1]}.hpp"

        # Generate source
        if options.dev_opts.get("code-generation", True):
            gt_pyext_sources = cls.generate_extension_sources(
                stencil_id, definition_ir, options, cls.GT_BACKEND_T
            )
            module_kwargs["halo_size"] = int(
                re.search(
                    r"#define GRIDTOOLS_DAWN_HALO_EXTENT ([0-9]+)", gt_pyext_sources[dawn_src_file]
                )[1]
            )

        else:
            # Pass NOTHING to the builder means try to reuse the source code files
            gt_pyext_sources = {key: gt_utils.NOTHING for key in cls.TEMPLATE_FILES.keys()}
            gt_pyext_sources[dawn_src_file] = gt_utils.NOTHING

        final_ext = ".cu" if uses_cuda else ".cpp"
        keys = list(gt_pyext_sources.keys())
        for key in keys:
            if key.split(".")[-1] == "src":
                new_key = key.replace(".src", final_ext)
                gt_pyext_sources[new_key] = gt_pyext_sources.pop(key)

        # Build extension module
        pyext_opts = dict(
            verbose=options.backend_opts.get("verbose", False),
            clean=options.backend_opts.get("clean", False),
            debug_mode=options.backend_opts.get("debug_mode", False),
            add_profile_info=options.backend_opts.get("add_profile_info", False),
        )
        include_dirs = [
            "{install_dir}/_external_src".format(
                install_dir=os.path.dirname(inspect.getabsfile(dawn4py))
            )
        ]

        return cls.build_extension_module(
            stencil_id,
            gt_pyext_sources,
            pyext_opts,
            pyext_extra_include_dirs=include_dirs,
            uses_cuda=uses_cuda,
        )

    @classmethod
    @abc.abstractmethod
    def generate_extension(
        cls,
        stencil_id: gt_definitions.StencilID,
        definition_ir: gt_ir.StencilDefinition,
        options: gt_definitions.BuildOptions,
        **kwargs,
    ):
        pass


_DAWN_BASE_OPTIONS = {
    "add_profile_info": {"versioning": True},
    "clean": {"versioning": False},
    "debug_mode": {"versioning": True},
    "dump_sir": {"versioning": False},
    "verbose": {"versioning": False},
}


_DAWN_TOOLCHAIN_OPTIONS = {}
for name in dir(dawn4py.Options):
    if (
        name.startswith("print")
        or name.startswith("dump")
        or name.startswith("report")
        or name.startswith("serialize")
        or name.startswith("deserialize")
    ):
        _DAWN_TOOLCHAIN_OPTIONS[name] = {"versioning": False}
    elif not name.startswith("_") and name != "backend":
        _DAWN_TOOLCHAIN_OPTIONS[name] = {"versioning": True}


_DAWN_BACKEND_OPTIONS = {**_DAWN_BASE_OPTIONS, **_DAWN_TOOLCHAIN_OPTIONS}


@gt_backend.register
class DawnGTX86Backend(BaseDawnBackend):

    DAWN_BACKEND_NS = "gt"
    DAWN_BACKEND_NAME = "gridtools"
    GT_BACKEND_T = "x86"

    name = "dawn:gtx86"
    options = _DAWN_BACKEND_OPTIONS
    storage_info = gt_backend.GTX86Backend.storage_info

    @classmethod
    def generate_extension(cls, stencil_id, definition_ir, options, **kwargs):
        return cls._generic_generate_extension(
            stencil_id, definition_ir, options, uses_cuda=False, **kwargs
        )


@gt_backend.register
class DawnGTMCBackend(BaseDawnBackend):

    DAWN_BACKEND_NS = "gt"
    DAWN_BACKEND_NAME = "gridtools"
    GT_BACKEND_T = "mc"

    name = "dawn:gtmc"
    options = _DAWN_BACKEND_OPTIONS
    storage_info = gt_backend.GTMCBackend.storage_info

    @classmethod
    def generate_extension(cls, stencil_id, definition_ir, options, **kwargs):
        return cls._generic_generate_extension(
            stencil_id, definition_ir, options, uses_cuda=False, **kwargs
        )


@gt_backend.register
class DawnGTCUDABackend(BaseDawnBackend):

    DAWN_BACKEND_NS = "gt"
    DAWN_BACKEND_NAME = "gridtools"
    GT_BACKEND_T = "cuda"

    MODULE_GENERATOR_CLASS = gt_backend.CUDAPyExtModuleGenerator

    name = "dawn:gtcuda"
    options = _DAWN_BACKEND_OPTIONS
    storage_info = gt_backend.GTCUDABackend.storage_info

    @classmethod
    def generate_extension(cls, stencil_id, definition_ir, options, **kwargs):
        return cls._generic_generate_extension(
            stencil_id, definition_ir, options, uses_cuda=True, **kwargs
        )


# @gt_backend.register
# class DawnNaiveBackend(BaseDawnBackend):
#
#     DAWN_BACKEND_NS = "cxxnaive"
#     DAWN_BACKEND_NAME = "c++-naive"
#     GT_BACKEND_T = "x86"
#
#     name = "dawn:naive"
#     options = _DAWN_BACKEND_OPTIONS
#     storage_info = gt_backend.GTX86Backend.storage_info
#
#     @classmethod
#     def generate_extension(cls, stencil_id, definition_ir, options, **kwargs):
#         return cls._generic_generate_extension(
#             stencil_id, definition_ir, options, uses_cuda=False, **kwargs
#         )
