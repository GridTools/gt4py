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

from typing import Final, Sequence

from gt4py.next.otf import languages
from gt4py.next.otf.binding import interface
from gt4py.next.type_system import type_info as ti, type_specifications as ts


CPP_DEFAULT: Final = languages.LanguageWithHeaderFilesSettings(
    formatter_key="cpp",
    formatter_style="LLVM",
    file_extension="cpp",
    header_extension="cpp.inc",
)


def render_scalar_type(scalar_type: ts.ScalarType) -> str:
    match scalar_type.kind:
        case ts.ScalarKind.BOOL:
            return "bool"
        case ts.ScalarKind.INT32:
            return "std::int32_t"
        case ts.ScalarKind.INT64:
            return "std::int64_t"
        case ts.ScalarKind.FLOAT32:
            return "float"
        case ts.ScalarKind.FLOAT64:
            return "double"
        case ts.ScalarKind.STRING:
            return "std::string"
        case _:
            raise AssertionError(
                f"Scalar kind '{scalar_type}' is not implemented when it should be."
            )


def _render_function_param(param: interface.Parameter, index: int) -> str:
    if isinstance(param.type_, ts.ScalarType):
        return f"{render_scalar_type(param.type_)} {param.name}"
    elif ti.is_type_or_tuple_of_type(
        param.type_, ts.FieldType
    ):  # TODO(havogt): add support for scalar tuples
        return f"BufferT{index}&& {param.name}"
    else:
        raise ValueError(f"Type '{param.type_}' is not supported in C++ interfaces.")


def render_function_declaration(function: interface.Function, body: str) -> str:
    rendered_params = [
        _render_function_param(param, index) for index, param in enumerate(function.parameters)
    ]
    rendered_decl = f"""decltype(auto) {function.name}({", ".join(rendered_params)}) {{
        {body}
    }}"""
    template_params = [
        f"class BufferT{index}"
        for index, param in enumerate(function.parameters)
        if ti.is_type_or_tuple_of_type(
            param.type_, ts.FieldType
        )  # TODO(havogt): add support for scalar tuples
    ]
    if template_params:
        return f"""
        template <{', '.join(template_params)}>
        {rendered_decl}
        """.strip()
    return rendered_decl


def render_function_call(function: interface.Function, args: Sequence[str]) -> str:
    return f"{function.name}({', '.join(args)})"
