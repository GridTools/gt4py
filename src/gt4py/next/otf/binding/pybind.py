# GT4Py - GridTools Framework
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


from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import gt4py.eve as eve
from gt4py.eve.codegen import JinjaTemplate as as_jinja, TemplatedGenerator
from gt4py.next.otf import languages, stages, workflow
from gt4py.next.otf.binding import cpp_interface, interface


class Expr(eve.Node):
    pass


class DimensionType(Expr):
    name: str


class SidConversion(Expr):
    buffer_name: str
    dimensions: Sequence[DimensionType]
    scalar_type: np.dtype
    dim_config: int


class FunctionCall(Expr):
    target: interface.Function
    args: Sequence[Any]


class ReturnStmt(eve.Node):
    expr: Expr


class FunctionParameter(eve.Node):
    name: str
    ndim: int
    dtype: np.dtype


class WrapperFunction(eve.Node):
    name: str
    parameters: Sequence[FunctionParameter]
    body: ReturnStmt


class BindingFunction(eve.Node):
    exported_name: str
    wrapper_name: str
    doc: str


class BindingModule(eve.Node):
    name: str
    doc: str
    functions: Sequence[BindingFunction]


class BindingFile(eve.Node):
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
            type_str = cpp_interface.render_python_type(param.dtype.type)
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
        return cpp_interface.render_function_call(call.target, args)

    def visit_SidConversion(self, sid: SidConversion):
        return self.generic_visit(
            sid, rendered_scalar_type=cpp_interface.render_python_type(sid.scalar_type.type)
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
    parameter: interface.ScalarParameter
    | interface.BufferParameter
    | interface.ConnectivityParameter,
) -> FunctionParameter:
    if isinstance(parameter, interface.ConnectivityParameter):
        return FunctionParameter(name=parameter.name, ndim=2, dtype=parameter.index_type)
    name = parameter.name
    ndim = 0 if isinstance(parameter, interface.ScalarParameter) else len(parameter.dimensions)
    scalar_type = parameter.scalar_type
    return FunctionParameter(name=name, ndim=ndim, dtype=scalar_type)


def make_argument(
    index: int,
    param: interface.ScalarParameter | interface.BufferParameter | interface.ConnectivityParameter,
) -> str | SidConversion:
    if isinstance(param, interface.ScalarParameter):
        return param.name
    elif isinstance(param, interface.ConnectivityParameter):
        return SidConversion(
            buffer_name=param.name,
            dimensions=[
                DimensionType(name=param.origin_axis),
                DimensionType(name=param.offset_tag),
            ],
            scalar_type=param.index_type,
            dim_config=index,
        )
    else:
        return SidConversion(
            buffer_name=param.name,
            dimensions=[DimensionType(name=dim) for dim in param.dimensions],
            scalar_type=param.scalar_type,
            dim_config=index,
        )


def create_bindings(
    program_source: stages.ProgramSource[languages.Cpp, languages.LanguageWithHeaderFilesSettings],
) -> stages.BindingSource[languages.Cpp, languages.Python]:
    """
    Generate Python bindings through which a C++ function can be called.

    Parameters
    ----------
    program_source
        The program source for which the bindings are created
    """
    if program_source.language is not languages.Cpp:
        raise ValueError(
            f"Can only create bindings for C++ program sources, received {program_source.language}."
        )
    wrapper_name = program_source.entry_point.name + "_wrapper"

    file_binding = BindingFile(
        callee_header_file=f"{program_source.entry_point.name}.{program_source.language_settings.header_extension}",
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
            parameters=[make_parameter(param) for param in program_source.entry_point.parameters],
            body=ReturnStmt(
                expr=FunctionCall(
                    target=program_source.entry_point,
                    args=[
                        make_argument(index, param)
                        for index, param in enumerate(program_source.entry_point.parameters)
                    ],
                )
            ),
        ),
        binding_module=BindingModule(
            name=program_source.entry_point.name,
            doc="",
            functions=[
                BindingFunction(
                    exported_name=program_source.entry_point.name,
                    wrapper_name=wrapper_name,
                    doc="",
                )
            ],
        ),
    )

    src = interface.format_source(
        program_source.language_settings,
        BindingCodeGenerator.apply(file_binding),
    )

    return stages.BindingSource(
        src,
        (interface.LibraryDependency("pybind11", "2.9.2"),),
    )


@workflow.make_step
def bind_source(
    inp: stages.ProgramSource[languages.Cpp, languages.LanguageWithHeaderFilesSettings],
) -> stages.CompilableSource[
    languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python
]:
    return stages.CompilableSource(program_source=inp, binding_source=create_bindings(inp))
