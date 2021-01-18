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

import abc
import functools
import numbers
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import jinja2
import numpy as np

from gt4py import backend as gt_backend
from gt4py import definitions as gt_definitions
from gt4py import gt_src_manager
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils
from gt4py.utils import text as gt_text

from . import pyext_builder


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject
    from gt4py.storage.storage import Storage


def make_x86_layout_map(mask: Tuple[int, ...]) -> Tuple[Optional[int], ...]:
    ctr = iter(range(sum(mask)))
    if len(mask) < 3:
        layout: List[Optional[int]] = [next(ctr) if m else None for m in mask]
    else:
        swapped_mask: List[Optional[int]] = [*mask[3:], *mask[:3]]
        layout = [next(ctr) if m else None for m in swapped_mask]

        layout = [*layout[-3:], *layout[:-3]]

    return tuple(layout)


def x86_is_compatible_layout(field: "Storage") -> bool:
    stride = 0
    layout_map = make_x86_layout_map(field.mask)
    if len(field.strides) < len(layout_map):
        return False
    for dim in reversed(np.argsort(layout_map)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


def gtcpu_is_compatible_type(field: "Storage") -> bool:
    return isinstance(field, np.ndarray)


def make_mc_layout_map(mask: Tuple[int, ...]) -> Tuple[Optional[int], ...]:
    ctr = reversed(range(sum(mask)))
    if len(mask) < 3:
        layout: List[Optional[int]] = [next(ctr) if m else None for m in mask]
    else:
        swapped_mask: List[Optional[int]] = list(mask)
        tmp = swapped_mask[1]
        swapped_mask[1] = swapped_mask[2]
        swapped_mask[2] = tmp

        layout = [next(ctr) if m else None for m in swapped_mask]

        tmp = layout[1]
        layout[1] = layout[2]
        layout[2] = tmp

    return tuple(layout)


def mc_is_compatible_layout(field: "Storage") -> bool:
    stride = 0
    layout_map = make_mc_layout_map(field.mask)
    if len(field.strides) < len(layout_map):
        return False
    for dim in reversed(np.argsort(layout_map)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


def cuda_layout(mask: Tuple[int, ...]) -> Tuple[Optional[int], ...]:
    ctr = reversed(range(sum(mask)))
    return tuple([next(ctr) if m else None for m in mask])


def cuda_is_compatible_layout(field: "Storage") -> bool:
    stride = 0
    layout_map = cuda_layout(field.mask)
    if len(field.strides) < len(layout_map):
        return False
    for dim in reversed(np.argsort(layout_map)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


def cuda_is_compatible_type(field: Any) -> bool:
    from gt4py.storage.storage import ExplicitlySyncedGPUStorage, GPUStorage

    return isinstance(field, (GPUStorage, ExplicitlySyncedGPUStorage))


class _MaxKOffsetExtractor(gt_ir.IRNodeVisitor):
    @classmethod
    def apply(cls, root_node: gt_ir.Node) -> int:
        return cls()(root_node)

    def __init__(self):
        self.max_offset = 2

    def __call__(self, node: gt_ir.Node) -> int:
        self.visit(node)
        return self.max_offset

    def visit_AxisBound(self, node: gt_ir.AxisBound) -> None:
        self.max_offset = max(self.max_offset, abs(node.offset) + 1)


_extract_max_k_offset = _MaxKOffsetExtractor.apply


class GTPyExtGenerator(gt_ir.IRNodeVisitor):

    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
    TEMPLATE_FILES = {
        "computation.hpp": "computation.hpp.in",
        "computation.src": "computation.src.in",
        "bindings.cpp": "bindings.cpp.in",
    }
    COMPUTATION_FILES = ["computation.hpp", "computation.src"]
    BINDINGS_FILES = ["bindings.cpp"]

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
        gt_ir.DataType.BOOL: "bool",
        gt_ir.DataType.INT8: "int8_t",
        gt_ir.DataType.INT16: "int16_t",
        gt_ir.DataType.INT32: "int32_t",
        gt_ir.DataType.INT64: "int64_t",
        gt_ir.DataType.FLOAT32: "float32_t",
        gt_ir.DataType.FLOAT64: "float64_t",
        gt_ir.DataType.DEFAULT: "float64_t",
    }

    NATIVE_FUNC_TO_CPP = {
        gt_ir.NativeFunction.ABS: "fabs",
        gt_ir.NativeFunction.MIN: "min",
        gt_ir.NativeFunction.MAX: "max",
        gt_ir.NativeFunction.MOD: "fmod",
        gt_ir.NativeFunction.SIN: "sin",
        gt_ir.NativeFunction.COS: "cos",
        gt_ir.NativeFunction.TAN: "tan",
        gt_ir.NativeFunction.ARCSIN: "asin",
        gt_ir.NativeFunction.ARCCOS: "acos",
        gt_ir.NativeFunction.ARCTAN: "atan",
        gt_ir.NativeFunction.SQRT: "sqrt",
        gt_ir.NativeFunction.EXP: "exp",
        gt_ir.NativeFunction.LOG: "log",
        gt_ir.NativeFunction.ISFINITE: "isfinite",
        gt_ir.NativeFunction.ISINF: "isinf",
        gt_ir.NativeFunction.ISNAN: "isnan",
        gt_ir.NativeFunction.FLOOR: "floor",
        gt_ir.NativeFunction.CEIL: "ceil",
        gt_ir.NativeFunction.TRUNC: "trunc",
    }

    BUILTIN_TO_CPP = {
        gt_ir.Builtin.NONE: "nullptr",  # really?
        gt_ir.Builtin.FALSE: "false",
        gt_ir.Builtin.TRUE: "true",
    }

    def __init__(self, class_name, module_name, gt_backend_t, options):
        self.class_name = class_name
        self.module_name = module_name
        self.gt_backend_t = gt_backend_t
        self.options = options

        self.templates = {}
        for key, file_name in self.TEMPLATE_FILES.items():
            with open(os.path.join(self.TEMPLATE_DIR, file_name), "r") as f:
                self.templates[key] = jinja2.Template(f.read())
        self.impl_node = None
        self.stage_symbols = None
        self.apply_block_symbols = None
        self.declared_symbols = None

    def __call__(self, impl_node: gt_ir.StencilImplementation) -> Dict[str, Dict[str, str]]:
        assert isinstance(impl_node, gt_ir.StencilImplementation)
        assert impl_node.domain.sequential_axis.name == gt_definitions.CartesianSpace.Axis.K.name

        self.impl_node = impl_node

        self.domain = impl_node.domain
        self.k_splitters: List[Tuple[str, int]] = []

        source = self.visit(impl_node)

        return source

    def _make_cpp_value(self, value: Any) -> Optional[str]:
        if isinstance(value, numbers.Number):
            if isinstance(value, bool):
                value = int(value)
                result: Optional[str] = str(value)
        else:
            result = None

        return result

    def _make_cpp_type(self, data_type: gt_ir.DataType) -> str:
        result = self.DATA_TYPE_TO_CPP[data_type]

        return result

    def _make_cpp_variable(self, decl: gt_ir.VarDecl) -> str:
        result = "{t} {name};".format(t=self._make_cpp_type(decl.data_type), name=decl.name)

        return result

    def visit_ScalarLiteral(self, node: gt_ir.ScalarLiteral) -> str:
        source = "{dtype}{{{value}}}".format(
            dtype=self.DATA_TYPE_TO_CPP[node.data_type], value=node.value
        )

        return source

    def visit_FieldRef(self, node: gt_ir.FieldRef, **kwargs: Any) -> str:
        assert node.name in self.apply_block_symbols
        offset = [node.offset.get(name, 0) for name in self.domain.axes_names]
        if not all(i == 0 for i in offset):
            idx = ", ".join(str(i) for i in offset)
        else:
            idx = ""
        source = "eval({name}({idx}))".format(name=node.name, idx=idx)

        return source

    def visit_VarRef(self, node: gt_ir.VarRef, *, write_context: bool = False) -> str:
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

    def visit_UnaryOpExpr(self, node: gt_ir.UnaryOpExpr) -> str:
        fmt = "({})" if isinstance(node.arg, gt_ir.CompositeExpr) else "{}"
        source = "{op}{expr}".format(
            op=self.OP_TO_CPP[node.op], expr=fmt.format(self.visit(node.arg))
        )

        return source

    def visit_BinOpExpr(self, node: gt_ir.BinOpExpr) -> str:
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

    def visit_Cast(self, node: gt_ir.Cast) -> str:
        expr = self.visit(node.expr)
        dtype = self.DATA_TYPE_TO_CPP[node.dtype]
        return f"static_cast<{dtype}>({expr})"

    def visit_BuiltinLiteral(self, node: gt_ir.BuiltinLiteral) -> str:
        return self.BUILTIN_TO_CPP[node.value]

    def visit_NativeFuncCall(self, node: gt_ir.NativeFuncCall) -> str:
        call = self.NATIVE_FUNC_TO_CPP[node.func]
        if self.gt_backend_t != "cuda":
            call = "std::" + call
        args = ",".join(self.visit(arg) for arg in node.args)
        return f"{call}({args})"

    def visit_TernaryOpExpr(self, node: gt_ir.TernaryOpExpr) -> str:
        then_fmt = "({})" if isinstance(node.then_expr, gt_ir.CompositeExpr) else "{}"
        else_fmt = "({})" if isinstance(node.else_expr, gt_ir.CompositeExpr) else "{}"
        source = "({condition}) ? {then_expr} : {else_expr}".format(
            condition=self.visit(node.condition),
            then_expr=then_fmt.format(self.visit(node.then_expr)),
            else_expr=else_fmt.format(self.visit(node.else_expr)),
        )

        return source

    def visit_Assign(self, node: gt_ir.Assign) -> List[str]:
        lhs = self.visit(node.target, write_context=True)
        rhs = self.visit(node.value)
        source = "{lhs} = {rhs};".format(lhs=lhs, rhs=rhs)

        return [source]

    def visit_BlockStmt(self, node: gt_ir.BlockStmt) -> str:
        body_sources = gt_text.TextBlock()
        for stmt in node.stmts:
            body_sources.extend(self.visit(stmt))

        return body_sources.text

    def visit_If(self, node: gt_ir.If) -> gt_text.TextBlock:
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

    def visit_AxisBound(self, node: gt_ir.AxisBound) -> Tuple[int, int]:
        if node.level == gt_ir.LevelMarker.START:
            level = 0
        elif node.level == gt_ir.LevelMarker.END:
            level = len(self.k_splitters) + 1
        else:
            raise NotImplementedError("VarRefs are not yet supported")

        # Shift offset to make it relative to the splitter (in-between levels)
        offset = node.offset + 1 if node.offset >= 0 else node.offset

        return level, offset

    def visit_AxisInterval(
        self, node: gt_ir.AxisInterval
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        start_splitter, start_offset = self.visit(node.start)
        end_splitter, end_offset = self.visit(node.end)

        # Transform range from excluded endpoint to including endpoint
        end_offset = -1 if end_offset == 1 else end_offset - 1

        return (start_splitter, start_offset), (end_splitter, end_offset)

    def visit_ApplyBlock(
        self, node: gt_ir.ApplyBlock
    ) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], str]:
        interval_definition = self.visit(node.interval)

        body_sources = gt_text.TextBlock()

        self.declared_symbols = set()
        for name, var_decl in node.local_symbols.items():
            assert isinstance(var_decl, gt_ir.VarDecl)
            body_sources.append(self._make_cpp_variable(var_decl))
            self.declared_symbols.add(name)

        self.apply_block_symbols = {**self.stage_symbols, **node.local_symbols}
        body_sources.extend(self.visit(node.body))

        return interval_definition, body_sources.text

    def visit_Stage(self, node: gt_ir.Stage) -> Dict[str, Any]:
        # Initialize symbols for the generation of references in this stage
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

    def visit_StencilImplementation(
        self, node: gt_ir.StencilImplementation
    ) -> Dict[str, Dict[str, str]]:
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
            gt_backend=self.gt_backend_t,
            halo_sizes=halo_sizes,
            k_axis=k_axis,
            module_name=self.module_name,
            multi_stages=multi_stages,
            parameters=parameters,
            stage_functors=stage_functors,
            stencil_unique_name=self.class_name,
            tmp_fields=tmp_fields,
        )

        sources: Dict[str, Dict[str, str]] = {"computation": {}, "bindings": {}}
        for key, template in self.templates.items():
            if key in self.COMPUTATION_FILES:
                sources["computation"][key] = template.render(**template_args)
            elif key in self.BINDINGS_FILES:
                sources["bindings"][key] = template.render(**template_args)

        return sources


class BaseGTBackend(gt_backend.BasePyExtBackend, gt_backend.CLIBackendMixin):

    GT_BACKEND_OPTS = {
        "add_profile_info": {"versioning": True, "type": bool},
        "clean": {"versioning": False, "type": bool},
        "debug_mode": {"versioning": True, "type": bool},
        "verbose": {"versioning": False, "type": bool},
    }

    GT_BACKEND_T: str

    MODULE_GENERATOR_CLASS = gt_backend.PyExtModuleGenerator

    PYEXT_GENERATOR_CLASS = GTPyExtGenerator

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        implementation_ir = self.builder.implementation_ir

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt_src_manager.has_gt_sources() and not gt_src_manager.install_gt_sources():
            raise RuntimeError("Missing GridTools sources.")

        pyext_module_name: Optional[str]
        pyext_file_path: Optional[str]
        if implementation_ir.has_effect:
            pyext_module_name, pyext_file_path = self.generate_extension()
        else:
            # if computation has no effect, there is no need to create an extension
            pyext_module_name, pyext_file_path = None, None

        # Generate and return the Python wrapper class
        return self.make_module(
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
        )

    def generate_computation(self, *, ir: Any = None) -> Dict[str, Union[str, Dict]]:
        if not ir:
            ir = self.builder.implementation_ir
        dir_name = f"{self.builder.options.name}_src"
        src_files = self.make_extension_sources(ir=ir)
        return {dir_name: src_files["computation"]}

    def generate_bindings(
        self, language_name: str, *, ir: Any = None
    ) -> Dict[str, Union[str, Dict]]:
        if not ir:
            ir = self.builder.implementation_ir
        if language_name != "python":
            return super().generate_bindings(language_name)
        dir_name = f"{self.builder.options.name}_src"
        src_files = self.make_extension_sources(ir=ir)
        return {dir_name: src_files["bindings"]}

    @abc.abstractmethod
    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        """
        Generate and build a python extension for the stencil computation.

        Returns the name and file path (as string) of the compiled extension ".so" module.
        """
        pass

    def make_extension(
        self, *, gt_version: int = 1, ir: Any = None, uses_cuda: bool = False
    ) -> Tuple[str, str]:
        if not ir:
            # in the GTC backend, `ir` is the definition_ir
            ir = self.builder.implementation_ir
        # Generate source
        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            gt_pyext_sources: Dict[str, Any] = self.make_extension_sources(ir=ir)
            gt_pyext_sources = {**gt_pyext_sources["computation"], **gt_pyext_sources["bindings"]}
        else:
            # Pass NOTHING to the self.builder means try to reuse the source code files
            gt_pyext_sources = {
                key: gt_utils.NOTHING for key in self.PYEXT_GENERATOR_CLASS.TEMPLATE_FILES.keys()
            }

        # Build extension module
        pyext_opts = dict(
            verbose=self.builder.options.backend_opts.get("verbose", False),
            clean=self.builder.options.backend_opts.get("clean", False),
            **pyext_builder.get_gt_pyext_build_opts(
                debug_mode=self.builder.options.backend_opts.get("debug_mode", False),
                add_profile_info=self.builder.options.backend_opts.get("add_profile_info", False),
                uses_cuda=uses_cuda,
                gt_version=gt_version,
            ),
        )

        result = self.build_extension_module(gt_pyext_sources, pyext_opts, uses_cuda=uses_cuda)
        return result

    def make_extension_sources(self, *, ir) -> Dict[str, Dict[str, str]]:
        """Generate the source for the stencil independently from use case."""
        if "computation_src" in self.builder.backend_data:
            return self.builder.backend_data["computation_src"]
        class_name = self.pyext_class_name if self.builder.stencil_id else self.builder.options.name
        module_name = (
            self.pyext_module_name
            if self.builder.stencil_id
            else f"{self.builder.options.name}_pyext"
        )
        gt_pyext_generator = self.PYEXT_GENERATOR_CLASS(
            class_name, module_name, self.GT_BACKEND_T, self.builder.options
        )
        gt_pyext_sources = gt_pyext_generator(ir)
        final_ext = ".cu" if self.languages and self.languages["computation"] == "cuda" else ".cpp"
        comp_src = gt_pyext_sources["computation"]
        for key in [k for k in comp_src.keys() if k.endswith(".src")]:
            comp_src[key.replace(".src", final_ext)] = comp_src.pop(key)
        self.builder.backend_data["computation_src"] = gt_pyext_sources
        return gt_pyext_sources


@gt_backend.register
class GTX86Backend(BaseGTBackend):

    GT_BACKEND_T = "x86"

    name = "gtx86"
    options = BaseGTBackend.GT_BACKEND_OPTS
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": make_x86_layout_map,
        "is_compatible_layout": x86_is_compatible_layout,
        "is_compatible_type": gtcpu_is_compatible_type,
    }

    languages = {"computation": "c++", "bindings": ["python"]}

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=False)


@gt_backend.register
class GTMCBackend(BaseGTBackend):

    GT_BACKEND_T = "mc"

    name = "gtmc"
    options = BaseGTBackend.GT_BACKEND_OPTS
    storage_info = {
        "alignment": 8,
        "device": "cpu",
        "layout_map": make_mc_layout_map,
        "is_compatible_layout": mc_is_compatible_layout,
        "is_compatible_type": gtcpu_is_compatible_type,
    }

    languages = {"computation": "c++", "bindings": ["python"]}

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=False)


class GTCUDAPyModuleGenerator(gt_backend.CUDAPyExtModuleGenerator):
    def generate_pre_run(self) -> str:
        field_names = [
            key
            for key in self.args_data["field_info"]
            if self.args_data["field_info"][key] is not None
        ]

        return "\n".join([f + ".host_to_device()" for f in field_names])

    def generate_post_run(self) -> str:
        output_field_names = [
            name
            for name, info in self.args_data["field_info"].items()
            if info is not None and info.access == gt_definitions.AccessKind.READ_WRITE
        ]

        return "\n".join([f + "._set_device_modified()" for f in output_field_names])


@gt_backend.register
class GTCUDABackend(BaseGTBackend):

    MODULE_GENERATOR_CLASS = GTCUDAPyModuleGenerator

    GT_BACKEND_T = "cuda"

    name = "gtcuda"
    options = BaseGTBackend.GT_BACKEND_OPTS
    storage_info = {
        "alignment": 32,
        "device": "gpu",
        "layout_map": cuda_layout,
        "is_compatible_layout": cuda_is_compatible_layout,
        "is_compatible_type": cuda_is_compatible_type,
    }

    languages = {"computation": "cuda", "bindings": ["python"]}

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=True)
