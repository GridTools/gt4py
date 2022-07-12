# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


from typing import Any, Sequence, Type, TypeVar

import jinja2

import eve.codegen
import functional.fencil_processors.cpp as cpp
import functional.fencil_processors.defs as defs
from eve import Node
from eve.codegen import JinjaTemplate, TemplatedGenerator


T = TypeVar("T")


class BindingFunction(Node):
    exported_name: str
    wrapper_name: str
    doc: str


class BindingModule(Node):
    name: str
    doc: str
    functions: Sequence[BindingFunction]


class FunctionParameter(Node):
    name: str
    ndim: int
    dtype: Type


class DimExpr(Node):
    name: str


class SidExpr(Node):
    buffer_name: str
    dimensions: Sequence[DimExpr]
    scalar_type: Type
    dim_config: int


class FunctionCall(Node):
    target: defs.Function
    args: Any


class ReturnStmt(Node):
    expr: Any


class WrapperFunction(Node):
    name: str
    parameters: Sequence[FunctionParameter]
    body: Any


class BindingFile(Node):
    callee_header_file: str
    header_files: list[str]
    wrapper: WrapperFunction
    binding_module: BindingModule


class BindingCodeGenerator(TemplatedGenerator):
    BindingFile = JinjaTemplate(
        """\
        #include "{{callee_header_file}}"

        {% for header_file in header_files: %}\
        #include <{{header_file}}>
        {% endfor %}

        {{wrapper}}

        {{binding_module}}\
        """
    )

    WrapperFunction = JinjaTemplate(
        """\
        decltype(auto) {{name}}(
            {{"\n,".join(parameters)}}
        )
        {
            {{body}}
        }\
        """
    )

    def visit_FunctionParameter(self, param: FunctionParameter):
        if param.ndim > 0:
            type_str = "pybind11::buffer"
        else:
            type_str = cpp.render_python_type(param.dtype)
        return type_str + " " + param.name

    ReturnStmt = JinjaTemplate("""return {{expr}};""")

    BindingModule = JinjaTemplate(
        """\
        PYBIND11_MODULE({{name}}, module) {
            module.doc() = "{{doc}}";
            {{"\n".join(functions)}}
        }\
        """
    )

    BindingFunction = JinjaTemplate(
        """module.def("{{exported_name}}", &{{wrapper_name}}, "{{doc}}");"""
    )

    def visit_FunctionCall(self, call: FunctionCall):
        args = [self.visit(arg) for arg in call.args]
        return cpp.render_function_call(call.target, args)

    def visit_SidExpr(self, sid: SidExpr):
        template = jinja2.Template(
            """\
            gridtools::sid::rename_numbered_dimensions<{{", ".join(dimensions)}}>(
                gridtools::as_sid<{{scalar_type}},\
                                  {{dimensions.__len__()}},\
                                  gridtools::integral_constant<int, {{dim_config}}>,\
                                  999'999'999>({{buffer_name}})
            )\
            """
        )
        return template.render(
            buffer_name=sid.buffer_name,
            dimensions=[self.visit(dim) for dim in sid.dimensions],
            scalar_type=cpp.render_python_type(sid.scalar_type),
            dim_config=sid.dim_config,
        )

    DimExpr = JinjaTemplate("""generated::{{name}}_t""")


def make_parameter(parameter: defs.ScalarParameter | defs.BufferParameter) -> FunctionParameter:
    name = parameter.name
    ndim = 0 if isinstance(parameter, defs.ScalarParameter) else len(parameter.dimensions)
    scalar_type = parameter.scalar_type
    return FunctionParameter(name=name, ndim=ndim, dtype=scalar_type)


def make_argument(index: int, param: defs.ScalarParameter | defs.BufferParameter) -> str | SidExpr:
    if isinstance(param, defs.ScalarParameter):
        return param.name
    else:
        return SidExpr(
            buffer_name=param.name,
            dimensions=[DimExpr(name=dim) for dim in param.dimensions],
            scalar_type=param.scalar_type,
            dim_config=index,
        )


def create_bindings(target: defs.Function, target_header: str) -> defs.BindingCodeModule:
    wrapper_name = target.name + "_wrapper"

    file_binding = BindingFile(
        callee_header_file=target_header,
        header_files=[
            "pybind11/pybind11.h",
            "pybind11/stl.h",
            "gridtools/storage/adapter/python_sid_adapter.hpp",
            "gridtools/sid/rename_dimensions.hpp",
            "gridtools/common/defs.hpp",
            "gridtools/fn/unstructured.hpp",
            "gridtools/fn/cartesian.hpp",
            "gridtools/fn/backend/naive.hpp",
        ],
        wrapper=WrapperFunction(
            name=wrapper_name,
            parameters=[make_parameter(param) for param in target.parameters],
            body=ReturnStmt(
                expr=FunctionCall(
                    target=target,
                    args=[
                        make_argument(index, param) for index, param in enumerate(target.parameters)
                    ],
                )
            ),
        ),
        binding_module=BindingModule(
            name=target.name,
            doc="",
            functions=[
                BindingFunction(exported_name=target.name, wrapper_name=wrapper_name, doc="")
            ],
        ),
    )

    src = eve.codegen.format_source("cpp", BindingCodeGenerator.apply(file_binding), style="LLVM")
    return defs.BindingCodeModule(src, [defs.LibraryDependency("pybind11", "2.9.2")])
