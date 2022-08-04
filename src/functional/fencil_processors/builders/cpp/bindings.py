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
"""Python bindings generator for C++ functions."""


from typing import Any, Sequence

import numpy

import eve.codegen
import functional.fencil_processors.source_modules.cpp_gen as cpp
from eve import Node
from eve.codegen import JinjaTemplate as as_jinja, TemplatedGenerator
from functional.fencil_processors.source_modules import source_modules


class Expr(Node):
    pass


class DimensionType(Expr):
    name: str


class SidConversion(Expr):
    buffer_name: str
    dimensions: Sequence[DimensionType]
    scalar_type: numpy.dtype
    dim_config: int


class FunctionCall(Expr):
    target: source_modules.Function
    args: Sequence[Any]


class ReturnStmt(Node):
    expr: Expr


class FunctionParameter(Node):
    name: str
    ndim: int
    dtype: numpy.dtype


class WrapperFunction(Node):
    name: str
    parameters: Sequence[FunctionParameter]
    body: ReturnStmt


class BindingFunction(Node):
    exported_name: str
    wrapper_name: str
    doc: str


class BindingModule(Node):
    name: str
    doc: str
    functions: Sequence[BindingFunction]


class BindingFile(Node):
    callee_header_file: str
    header_files: Sequence[str]
    wrapper: WrapperFunction
    binding_module: BindingModule


class BindingCodeGenerator(TemplatedGenerator):
    BindingFile = as_jinja(
        """\
        #include "{{callee_header_file}}"

        {% for header_file in header_files: %}\
        #include <{{header_file}}>
        {% endfor %}

        {{wrapper}}

        {{binding_module}}\
        """
    )

    WrapperFunction = as_jinja(
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
            type_str = cpp.render_python_type(param.dtype.type)
        return type_str + " " + param.name

    ReturnStmt = as_jinja("""return {{expr}};""")

    BindingModule = as_jinja(
        """\
        PYBIND11_MODULE({{name}}, module) {
            module.doc() = "{{doc}}";
            {{"\n".join(functions)}}
        }\
        """
    )

    BindingFunction = as_jinja("""module.def("{{exported_name}}", &{{wrapper_name}}, "{{doc}}");""")

    def visit_FunctionCall(self, call: FunctionCall):
        args = [self.visit(arg) for arg in call.args]
        return cpp.render_function_call(call.target, args)

    def visit_SidConversion(self, sid: SidConversion):
        return self.generic_visit(
            sid, rendered_scalar_type=cpp.render_python_type(sid.scalar_type.type)
        )

    SidConversion = as_jinja(
        """gridtools::sid::rename_numbered_dimensions<{{", ".join(dimensions)}}>(
                gridtools::as_sid<{{rendered_scalar_type}},\
                                  {{dimensions.__len__()}},\
                                  gridtools::integral_constant<int, {{dim_config}}>,\
                                  999'999'999>({{buffer_name}})
            )"""
    )

    DimensionType = as_jinja("""generated::{{name}}_t""")


def make_parameter(
    parameter: source_modules.ScalarParameter | source_modules.BufferParameter,
) -> FunctionParameter:
    name = parameter.name
    ndim = 0 if isinstance(parameter, source_modules.ScalarParameter) else len(parameter.dimensions)
    scalar_type = parameter.scalar_type
    return FunctionParameter(name=name, ndim=ndim, dtype=scalar_type)


def make_argument(
    index: int, param: source_modules.ScalarParameter | source_modules.BufferParameter
) -> str | SidConversion:
    if isinstance(param, source_modules.ScalarParameter):
        return param.name
    else:
        return SidConversion(
            buffer_name=param.name,
            dimensions=[DimensionType(name=dim) for dim in param.dimensions],
            scalar_type=param.scalar_type,
            dim_config=index,
        )


def create_bindings(
    target: source_modules.Function, target_header: str
) -> source_modules.BindingModule:
    """
    Generate Python bindings through which a C++ function can be called.

    Parameters
    ----------
    target
        The signature of the C++ function
    target_header
        The name of the C++ (header) file in which the function resides; used for includes
    """
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
    return source_modules.BindingModule(
        src, [source_modules.LibraryDependency("pybind11", "2.9.2")]
    )
