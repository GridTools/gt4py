# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import gt4py.eve as eve
import gt4py.next.type_system.type_specifications as ts
from gt4py.eve.codegen import JinjaTemplate as as_jinja, TemplatedGenerator
from gt4py.next.otf import languages, stages, workflow
from gt4py.next.otf.binding import cpp_interface, interface


class Expr(eve.Node):
    pass


class DimensionType(Expr):
    name: str


class BufferSID(Expr):
    source_buffer: str
    dimensions: Sequence[DimensionType]
    scalar_type: ts.ScalarType
    # strides_kind: int # TODO(havogt): implement strides_kind once we have the "frozen stencil" mechanism


class FunctionCall(Expr):
    target: interface.Function
    args: Sequence[Any]


class ReturnStmt(eve.Node):
    expr: Expr


class FunctionParameter(eve.Node):
    name: str
    type_: ts.TypeSpec


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
        if isinstance(param.type_, ts.FieldType):
            type_str = "pybind11::buffer"
        elif isinstance(param.type_, ts.ScalarType):
            type_str = cpp_interface.render_scalar_type(param.type_)
        else:
            raise ValueError(f"Type '{param.type_}' is not supported in pybind11 interfaces.")
        return f"{type_str} {param.name}"

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

    def visit_BufferSID(self, sid: BufferSID):
        return self.generic_visit(
            sid, rendered_scalar_type=cpp_interface.render_scalar_type(sid.scalar_type)
        )

    BufferSID = as_jinja(
        """gridtools::sid::rename_numbered_dimensions<{{", ".join(dimensions)}}>(
                gridtools::as_sid<{{rendered_scalar_type}},\
                                  {{dimensions.__len__()}},\
                                  gridtools::sid::unknown_kind>({{source_buffer}})
            )"""
    )

    DimensionType = as_jinja("""generated::{{name}}_t""")


def make_argument(index: int, param: interface.Parameter) -> str | BufferSID:
    if isinstance(param.type_, ts.FieldType):
        return BufferSID(
            source_buffer=param.name,
            dimensions=[DimensionType(name=dim.value) for dim in param.type_.dims],
            scalar_type=param.type_.dtype,
        )
    else:
        return param.name


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
            "gridtools/sid/unknown_kind.hpp",
            "gridtools/sid/rename_dimensions.hpp",
            "gridtools/common/defs.hpp",
            "gridtools/fn/unstructured.hpp",
            "gridtools/fn/cartesian.hpp",
            "gridtools/fn/backend/naive.hpp",
        ],
        wrapper=WrapperFunction(
            name=wrapper_name,
            parameters=[
                FunctionParameter(name=param.name, type_=param.type_)
                for param in program_source.entry_point.parameters
            ],
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
