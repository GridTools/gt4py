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
import collections
import copy
import enum
import inspect
import os
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Tuple, Type, Union

import dawn4py
import jinja2
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils

from gt4py import backend as gt_backend
from gt4py import definitions as gt_definitions
from gt4py import gt_src_manager
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils
from gt4py.utils import text as gt_text

from . import pyext_builder


if TYPE_CHECKING:
    from gt4py import StencilObject


DOMAIN_AXES = gt_definitions.CartesianSpace.names


def _enum_dict(enum: enum.Enum) -> Dict[str, Any]:
    return {k: v for k, v in enum.__dict__.items() if not k.startswith("__") and not k == "name"}


DAWN_PASS_GROUPS = _enum_dict(dawn4py.PassGroup)
DAWN_CODEGEN_BACKENDS = _enum_dict(dawn4py.CodeGenBackend)


class FieldInfoCollector(gt_ir.IRNodeVisitor):
    @classmethod
    def apply(cls, definition_ir: gt_ir.StencilDefinition) -> Dict[str, Any]:
        return cls()(definition_ir)

    def __call__(self, definition_ir: gt_ir.StencilDefinition) -> Dict[str, Any]:
        self.field_info: Dict[str, Any] = collections.OrderedDict()
        self.visit(definition_ir, write_field="")
        return self.field_info

    def visit_FieldDecl(self, node: gt_ir.FieldDecl, **kwargs: Any) -> None:
        # NOTE Add unstructured support here
        field_dimensions = sir_utils.make_field_dimensions_cartesian(
            [1 if ax in node.axes else 0 for ax in DOMAIN_AXES]
        )
        self.field_info[node.name] = dict(
            field_decl=sir_utils.make_field(
                name=node.name, dimensions=field_dimensions, is_temporary=not node.is_api
            ),
            access=None,
            extent=gt_definitions.Extent.zeros(),
            inputs=set(),
        )

    def visit_FieldRef(self, node: gt_ir.FieldRef, **kwargs: Any) -> None:
        field_name = node.name
        if len(kwargs["write_field"]) < 1:
            self.field_info[field_name]["access"] = gt_definitions.AccessKind.READ_WRITE
        elif self.field_info[field_name]["access"] is None:
            self.field_info[field_name]["access"] = gt_definitions.AccessKind.READ_ONLY
            if kwargs["write_field"] in self.field_info:
                self.field_info[kwargs["write_field"]]["inputs"].add(field_name)

        offset = tuple(node.offset.values())
        self.field_info[field_name]["extent"] |= gt_definitions.Extent(
            [(offset[0], offset[0]), (offset[1], offset[1]), (0, 0)]
        )  # exclude sequential axis

    def visit_Assign(self, node: gt_ir.Assign, **kwargs: Any) -> None:
        kwargs["write_field"] = ""
        self.visit(node.target, **kwargs)
        kwargs["write_field"] = node.target.name
        self.visit(node.value, **kwargs)


class SIRConverter(gt_ir.IRNodeVisitor):
    # mypy is unable to determine the type for some reason
    OP_TO_CPP = gt_backend.GTPyExtGenerator.OP_TO_CPP  # type: ignore

    @classmethod
    def apply(cls, definition_ir: gt_ir.StencilDefinition) -> Tuple[SIR.SIR, Dict[str, Any]]:
        return cls()(definition_ir)

    def __call__(self, definition_ir: gt_ir.StencilDefinition) -> Tuple[SIR.SIR, Dict[str, Any]]:
        return self.visit(definition_ir)

    def _make_global_variables(self, parameters: list, externals: dict) -> SIR.GlobalVariableMap:
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

        return global_variables

    def _make_operator(self, op: gt_ir.UnaryOperator) -> str:
        # This should be limited to the Enum subclass representing an operator
        # where the :code:`python_symbol` is defined.
        return self.OP_TO_CPP.get(op, op.python_symbol)  # type: ignore

    def _update_field_extents(self, field_info: Dict[str, Any]) -> None:
        out_fields = [
            field
            for field in field_info.values()
            if field["access"] == gt_definitions.AccessKind.READ_WRITE
        ]

        compute_extent = gt_definitions.Extent.zeros()
        for i in range(len(out_fields) - 1, -1, -1):
            out_field = out_fields[i]
            compute_extent |= out_field["extent"]
            for input_f in out_field["inputs"]:
                in_field = field_info[input_f]
                accumulated_extent = compute_extent + in_field["extent"]
                in_field["extent"] |= accumulated_extent

    def visit_BuiltinLiteral(self, node: gt_ir.BuiltinLiteral) -> SIR.LiteralAccessExpr:
        if node.value == gt_ir.Builtin.TRUE:
            source = "true"
        elif node.value == gt_ir.Builtin.FALSE:
            source = "false"
        else:  # Builtin.NONE
            source = "nullptr"
        return sir_utils.make_literal_access_expr(source, type=SIR.BuiltinType.Boolean)

    def visit_ScalarLiteral(
        self, node: gt_ir.ScalarLiteral, **kwargs: Any
    ) -> SIR.LiteralAccessExpr:
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
            raise AssertionError("Unrecognized data type")
        return sir_utils.make_literal_access_expr(value=repr(node.value), type=sir_type)

    def visit_VarRef(self, node: gt_ir.VarRef, **kwargs: Any) -> SIR.VarAccessExpr:
        return sir_utils.make_var_access_expr(name=node.name, is_external=True)

    def visit_FieldRef(self, node: gt_ir.FieldRef, **kwargs: Any) -> SIR.FieldAccessExpr:
        offset = [node.offset[ax] if ax in node.offset else 0 for ax in DOMAIN_AXES]
        return sir_utils.make_field_access_expr(name=node.name, offset=offset)

    def visit_UnaryOpExpr(self, node: gt_ir.UnaryOpExpr, **kwargs: Any) -> SIR.UnaryOperator:
        op = self._make_operator(node.op)
        operand = self.visit(node.arg)
        return sir_utils.make_unary_operator(op, operand)

    def visit_BinOpExpr(self, node: gt_ir.BinOpExpr, **kwargs: Any) -> SIR.BinaryOperator:
        left = self.visit(node.lhs)
        right = self.visit(node.rhs)
        if node.op.python_symbol == "**":
            sir = sir_utils.make_fun_call_expr("gridtools::dawn::math::pow", [left, right])
        else:
            op = self._make_operator(node.op)
            sir = sir_utils.make_binary_operator(left, op, right)
        return sir

    def visit_TernaryOpExpr(self, node: gt_ir.TernaryOpExpr, **kwargs: Any) -> SIR.TernaryOperator:
        cond = self.visit(node.condition)
        left = self.visit(node.then_expr)
        right = self.visit(node.else_expr)
        return sir_utils.make_ternary_operator(cond, left, right)

    def visit_NativeFuncCall(self, node: gt_ir.NativeFuncCall):
        return sir_utils.make_fun_call_expr(
            "gridtools::dawn::math::" + gt_backend.GTPyExtGenerator.NATIVE_FUNC_TO_CPP[node.func],
            [self.visit(arg) for arg in node.args],
        )

    def visit_BlockStmt(
        self, node: gt_ir.BlockStmt, *, make_block: bool = True, **kwargs: Any
    ) -> SIR.BlockStmt:
        stmts = [self.visit(stmt) for stmt in node.stmts if not isinstance(stmt, gt_ir.FieldDecl)]
        if make_block:
            stmts = sir_utils.make_block_stmt(stmts)
        return stmts

    def visit_Assign(self, node: gt_ir.Assign, **kwargs: Any) -> SIR.AssignmentExpr:
        left = self.visit(node.target)
        right = self.visit(node.value)
        stmt = sir_utils.make_assignment_stmt(left, right, "=")
        return stmt

    def visit_If(self, node: gt_ir.If, **kwargs: Any) -> SIR.IfStmt:
        cond = sir_utils.make_expr_stmt(self.visit(node.condition))
        then_part = self.visit(node.main_body)
        else_part = self.visit(node.else_body)
        stmt = sir_utils.make_if_stmt(cond, then_part, else_part)
        return stmt

    def visit_AxisBound(self, node: gt_ir.AxisBound, **kwargs: Any) -> Tuple[int, int]:
        assert isinstance(node.level, gt_ir.LevelMarker)
        level = SIR.Interval.Start if node.level == gt_ir.LevelMarker.START else SIR.Interval.End
        offset = node.offset
        return level, offset

    def visit_AxisInterval(self, node: gt_ir.AxisInterval, **kwargs: Any) -> SIR.Interval:
        lower_level, lower_offset = self.visit(node.start)
        upper_level, upper_offset = self.visit(node.end)
        return sir_utils.make_interval(lower_level, upper_level, lower_offset, upper_offset)

    def visit_ComputationBlock(
        self, node: gt_ir.ComputationBlock, **kwargs: Any
    ) -> SIR.VerticalRegionDeclStmt:
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

    def visit_StencilDefinition(
        self, node: gt_ir.StencilDefinition, **kwargs: Any
    ) -> Tuple[SIR.SIR, Dict[str, Any]]:
        stencils = []
        functions: List = []
        global_variables = self._make_global_variables(node.parameters, node.externals)

        field_info = FieldInfoCollector.apply(node)
        self._update_field_extents(field_info)
        fields = [field_info[field_name]["field_decl"] for field_name in field_info]

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
        return sir, field_info


_DAWN_BASE_OPTIONS = {
    "add_profile_info": {"versioning": True, "type": bool},
    "clean": {"versioning": False, "type": bool},
    "debug_mode": {"versioning": True, "type": bool},
    "dump_sir": {"versioning": False, "type": bool},
    "verbose": {"versioning": False, "type": bool},
    "no_opt": {"versioning": False, "type": bool},
}


_DAWN_TOOLCHAIN_OPTIONS = {}
for name in dir(dawn4py.CodeGenOptions) + dir(dawn4py.OptimizerOptions):
    if (
        name.startswith("print")
        or name.startswith("dump")
        or name.startswith("report")
        or name.startswith("serialize")
        or name.startswith("deserialize")
    ):
        _DAWN_TOOLCHAIN_OPTIONS[name] = {"versioning": False, "type": bool}
    elif not name.startswith("_") and name != "backend":
        _DAWN_TOOLCHAIN_OPTIONS[name] = {"versioning": True, "type": bool}


_DAWN_BACKEND_OPTIONS = {**_DAWN_BASE_OPTIONS, **_DAWN_TOOLCHAIN_OPTIONS}


class DawnPyModuleGenerator(gt_backend.PyExtModuleGenerator):
    def generate_implementation(self) -> str:
        sources = gt_text.TextBlock(indent_size=gt_backend.BaseModuleGenerator.TEMPLATE_INDENT_SIZE)

        args = []
        empty_checks = []
        none_checks = []

        # Other backends allow None to be passed to unreferenced fields or scalars.
        # Dawn backends do not check for null values passed to parameters, so the
        # None values must be replaced with defaults for the unreferenced parameters,
        # (0,0,0) for the origins, np.empty for fields, and zero for scalars.

        field_info = self.args_data["field_info"]
        unreferenced = self.args_data["unreferenced"]

        for arg in self.builder.definition_ir.api_signature:
            args.append(arg.name)
            if arg.name in field_info:
                if arg.name in unreferenced:
                    ndims = len(field_info[arg.name].boundary)
                    zeros = ", ".join(["0"] * ndims)
                    args.append("[{}]".format(zeros))
                    empty_checks.append(
                        f"{arg.name} = np.empty(({zeros})) if {arg.name} is None else {arg.name}"
                    )
                else:
                    args.append("list(_origin_['{}'])".format(arg.name))
            elif arg.name in self.builder.definition_ir.parameters and arg.default is None:
                none_checks.append(f"{arg.name} = 0 if {arg.name} is None else {arg.name}")

        source = """
# Load or generate a Dawn Computation object for the current domain size
{empty_checks}
{none_checks}
pyext_module.run_computation(list(_domain_), {run_args}, exec_info)
""".format(
            empty_checks="\n".join(empty_checks),
            none_checks="\n".join(none_checks),
            run_args=", ".join(args),
        )
        sources.extend(source.splitlines())

        return sources.text

    def backend_pre_run(self) -> List[str]:
        return []

    def backend_imports(self) -> List[str]:
        return []

    def generate_pre_run(self) -> str:
        field_size_check = """
for name, other_name in itertools.combinations(field_args, 2):
    field = field_args[name]
    other_field = field_args[other_name]
    if field.mask == other_field.mask and not other_field.shape == field.shape:
        raise ValueError(
            f"The fields {name} and {other_name} have the same mask but different shapes."
        )"""

        return "\n".join(self.backend_pre_run() + [field_size_check])

    def generate_imports(self) -> str:
        return "\n".join(
            self.backend_imports() + ["import itertools"] + [super().generate_imports()]
        )


class DawnCUDAPyModuleGenerator(DawnPyModuleGenerator):
    def generate_implementation(self) -> str:
        source = (
            super().generate_implementation()
            + """
cupy.cuda.Device(0).synchronize()
        """
        )

        return source

    def backend_imports(self) -> List[str]:
        return ["import cupy"]

    def backend_pre_run(self) -> List[str]:
        field_names = [
            key for key, value in self.args_data["field_info"].items() if value is not None
        ]
        return [f"{name}.host_to_device()" for name in field_names]

    def generate_post_run(self) -> str:
        output_field_names = [
            name
            for name, info in self.args_data["field_info"].items()
            if info and info.access == gt_definitions.AccessKind.READ_WRITE
        ]
        return "\n".join([f + "._set_device_modified()" for f in output_field_names])


class BaseDawnBackend(gt_backend.BasePyExtBackend):

    DAWN_BACKEND_NS: ClassVar[str]
    DAWN_BACKEND_NAME: ClassVar[str]
    DAWN_BACKEND_OPTS: ClassVar[Dict[str, Any]] = copy.deepcopy(_DAWN_BASE_OPTIONS)

    GT_BACKEND_T: ClassVar[str]

    MODULE_GENERATOR_CLASS = DawnPyModuleGenerator

    PYEXT_GENERATOR_CLASS = gt_backend.GTPyExtGenerator

    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
    TEMPLATE_FILES = {
        "computation.hpp": "computation.hpp.in",
        "computation.src": "dawn_computation.src.in",
        "bindings.cpp": "bindings.cpp.in",
    }

    _DATA_TYPE_TO_CPP = {
        gt_ir.DataType.BOOL: "bool",
        gt_ir.DataType.INT8: "int",
        gt_ir.DataType.INT16: "int",
        gt_ir.DataType.INT32: "int",
        gt_ir.DataType.INT64: "int",
        gt_ir.DataType.FLOAT32: "double",
        gt_ir.DataType.FLOAT64: "double",
        gt_ir.DataType.DEFAULT: "double",
    }

    sir: Any
    sir_field_info: Dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        self.sir, self.sir_field_info = SIRConverter.apply(self.builder.definition_ir)

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt_src_manager.has_gt_sources() and not gt_src_manager.install_gt_sources():
            raise RuntimeError("Missing GridTools sources.")

        pyext_module_name, pyext_file_path = self.generate_extension(**kwargs)

        # Dawn backends do not use the internal analysis pipeline, so a custom
        # args_data object should be passed to the module generator
        args_data = self.make_args_data(self.builder.definition_ir, self.sir_field_info)

        # Generate and return the Python wrapper class
        return self.make_module(
            args_data=args_data,
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
        )

    def generate_computation(self) -> Dict[str, Union[str, Dict]]:
        dir_name = f"{self.builder.options.name}_src"
        src_files = self.make_extension_sources(self.GT_BACKEND_T)
        return {dir_name: src_files}

    @abc.abstractmethod
    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        """
        Generate and build a python extension for the stencil computation.

        Returns the name and file path (as string) of the compiled extension ".so" module.
        """
        pass

    def make_extension(self, *, uses_cuda: bool = False, **kwargs: Any) -> Tuple[str, str]:
        dawn_src_file = f"_dawn_{self.builder.stencil_id.qualified_name.split('.')[-1]}.hpp"

        # Generate source
        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            gt_pyext_sources = self.make_extension_sources(self.GT_BACKEND_T)
        else:
            # Pass NOTHING to the self.builder means try to reuse the source code files
            gt_pyext_sources = {key: gt_utils.NOTHING for key in self.TEMPLATE_FILES.keys()}
            gt_pyext_sources[dawn_src_file] = gt_utils.NOTHING

        final_ext = ".cu" if uses_cuda else ".cpp"
        keys = list(gt_pyext_sources.keys())
        for key in keys:
            if key.split(".")[-1] == "src":
                new_key = key.replace(".src", final_ext)
                gt_pyext_sources[new_key] = gt_pyext_sources.pop(key)

        # Build extension module
        pyext_opts = dict(
            verbose=self.builder.options.backend_opts.get("verbose", False),
            clean=self.builder.options.backend_opts.get("clean", False),
            **pyext_builder.get_gt_pyext_build_opts(
                debug_mode=self.builder.options.backend_opts.get("debug_mode", False),
                add_profile_info=self.builder.options.backend_opts.get("add_profile_info", False),
                uses_cuda=uses_cuda,
            ),
        )

        pyext_opts["include_dirs"].extend(
            [
                "{install_dir}/_external_src".format(
                    install_dir=os.path.dirname(inspect.getabsfile(dawn4py))
                )
            ]
        )

        result = self.build_extension_module(gt_pyext_sources, pyext_opts, uses_cuda=uses_cuda)
        return result

    def make_extension_sources(self, gt_backend_t: str) -> Dict[str, Any]:
        stencil_short_name = self.builder.stencil_id.qualified_name.split(".")[-1]
        backend_opts = dict(**self.builder.options.backend_opts)
        dawn_namespace = self.DAWN_BACKEND_NS

        dump_sir_opt = backend_opts.get("dump_sir", False)
        if dump_sir_opt:
            if isinstance(dump_sir_opt, str):
                dump_sir_file = dump_sir_opt
            else:
                assert isinstance(dump_sir_opt, bool)
                dump_sir_file = f"{stencil_short_name}_gt4py.sir"
            with open(dump_sir_file, "w") as f:
                f.write(sir_utils.to_json(self.sir))

        # Get list of pass groups
        if "no_opt" in backend_opts:
            pass_groups = []
        elif "opt_groups" in backend_opts:
            pass_groups = [DAWN_PASS_GROUPS[k] for k in backend_opts["opt_groups"]]
            if "default_opt" in backend_opts:
                raise ValueError(
                    "Do not add 'default_opt' when opt 'opt_groups'. "
                    + "Instead, append dawn4py.default_pass_groups()"
                )
        else:
            pass_groups = dawn4py.default_pass_groups()

        # If present, parse backend string
        dawn_backend = DAWN_CODEGEN_BACKENDS[self.DAWN_BACKEND_NAME or "GridTools"]

        dawn_opts = {
            key: value
            for key, value in backend_opts.items()
            if key in _DAWN_TOOLCHAIN_OPTIONS.keys()
        }
        source = dawn4py.compile(
            self.sir, groups=pass_groups, backend=dawn_backend, run_with_sync=False, **dawn_opts
        )
        stencil_unique_name = self.pyext_class_name
        module_name = self.pyext_module_name
        pyext_sources = {f"_dawn_{stencil_short_name}.hpp": source}

        arg_fields = [
            {"name": field.name, "dtype": self._DATA_TYPE_TO_CPP[field.data_type], "layout_id": i}
            for i, field in enumerate(self.builder.definition_ir.api_fields)
        ]
        header_file = "computation.hpp"
        parameters = []
        for parameter in self.builder.definition_ir.parameters:
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
                raise AssertionError("Wrong data_type for parameter")
            parameters.append({"name": parameter.name, "dtype": dtype})

        template_args = dict(
            arg_fields=arg_fields,
            dawn_namespace=dawn_namespace,
            gt_backend=gt_backend_t,
            header_file=header_file,
            module_name=module_name,
            parameters=parameters,
            stencil_short_name=stencil_short_name,
            stencil_unique_name=stencil_unique_name,
        )

        for key, file_name in self.TEMPLATE_FILES.items():
            with open(os.path.join(self.TEMPLATE_DIR, file_name), "r") as f:
                template = jinja2.Template(f.read())
                pyext_sources[key] = template.render(**template_args)

        return pyext_sources

    @staticmethod
    def make_args_data(
        definition_ir: gt_ir.StencilDefinition, sir_field_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {"field_info": {}, "parameter_info": {}, "unreferenced": set()}

        fields = {item.name: item for item in definition_ir.api_fields}
        parameters = {item.name: item for item in definition_ir.parameters}

        for arg in definition_ir.api_signature:
            if arg.name in fields:
                access = sir_field_info[arg.name]["access"]
                if access is None:
                    access = gt_definitions.AccessKind.READ_ONLY
                    data["unreferenced"].add(arg.name)
                extent = sir_field_info[arg.name]["extent"]
                boundary = gt_definitions.Boundary([(-pair[0], pair[1]) for pair in extent])
                data["field_info"][arg.name] = gt_definitions.FieldInfo(
                    access=access, dtype=fields[arg.name].data_type.dtype, boundary=boundary
                )
            else:
                data["parameter_info"][arg.name] = gt_definitions.ParameterInfo(
                    dtype=parameters[arg.name].data_type.dtype
                )

        return data


@gt_backend.register
class DawnGTX86Backend(BaseDawnBackend):

    DAWN_BACKEND_NS = "gt"
    DAWN_BACKEND_NAME = "GridTools"
    GT_BACKEND_T = "x86"

    name = "dawn:gtx86"
    options = _DAWN_BACKEND_OPTIONS
    # ignore types because mypy is bugging out on those
    storage_info = gt_backend.GTX86Backend.storage_info  # type: ignore
    languages = gt_backend.GTX86Backend.languages  # type: ignore

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=False, **kwargs)


@gt_backend.register
class DawnGTMCBackend(BaseDawnBackend):

    DAWN_BACKEND_NS = "gt"
    DAWN_BACKEND_NAME = "GridTools"
    GT_BACKEND_T = "mc"

    name = "dawn:gtmc"
    options = _DAWN_BACKEND_OPTIONS
    # ignore types because mypy is bugging out on those
    storage_info = gt_backend.GTMCBackend.storage_info  # type: ignore
    languages = gt_backend.GTMCBackend.languages  # type: ignore

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=False, **kwargs)


@gt_backend.register
class DawnGTCUDABackend(BaseDawnBackend):

    DAWN_BACKEND_NS = "gt"
    DAWN_BACKEND_NAME = "GridTools"
    MODULE_GENERATOR_CLASS = DawnCUDAPyModuleGenerator
    GT_BACKEND_T = "cuda"

    name = "dawn:gtcuda"
    options = _DAWN_BACKEND_OPTIONS
    # ignore types because mypy is bugging out on those
    storage_info = gt_backend.GTCUDABackend.storage_info  # type: ignore
    languages = gt_backend.GTCUDABackend.languages  # type: ignore

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=True, **kwargs)


@gt_backend.register
class DawnNaiveBackend(BaseDawnBackend):

    DAWN_BACKEND_NS = "cxxnaive"
    DAWN_BACKEND_NAME = "CXXNaive"
    GT_BACKEND_T = "x86"

    name = "dawn:naive"
    options = _DAWN_BACKEND_OPTIONS
    # ignore types because mypy is bugging out on those
    storage_info = gt_backend.GTX86Backend.storage_info  # type: ignore
    languages = gt_backend.GTX86Backend.languages  # type: ignore

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=False, **kwargs)


@gt_backend.register
class DawnOptBackend(BaseDawnBackend):

    DAWN_BACKEND_NS = "cxxopt"
    DAWN_BACKEND_NAME = "CXXOpt"
    GT_BACKEND_T = "x86"

    name = "dawn:cxxopt"
    options = _DAWN_BACKEND_OPTIONS
    # ignore types because mypy is bugging out on those
    storage_info = gt_backend.GTX86Backend.storage_info  # type: ignore
    languages = gt_backend.GTX86Backend.languages  # type: ignore

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=False, **kwargs)


@gt_backend.register
class DawnCUDABackend(BaseDawnBackend):

    DAWN_BACKEND_NS = "cuda"
    DAWN_BACKEND_NAME = "CUDA"
    MODULE_GENERATOR_CLASS = DawnCUDAPyModuleGenerator
    GT_BACKEND_T = "cuda"

    name = "dawn:cuda"
    options = _DAWN_BACKEND_OPTIONS
    # ignore types because mypy is bugging out on those
    storage_info = gt_backend.GTCUDABackend.storage_info  # type: ignore
    languages = gt_backend.GTCUDABackend.languages  # type: ignore

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=True, **kwargs)
