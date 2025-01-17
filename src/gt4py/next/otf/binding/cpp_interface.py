# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final, Sequence

from gt4py.next.otf import cpp_utils, languages
from gt4py.next.otf.binding import interface
from gt4py.next.type_system import type_info as ti, type_specifications as ts


CPP_DEFAULT: Final = languages.LanguageWithHeaderFilesSettings(
    formatter_key="cpp", formatter_style="LLVM", file_extension="cpp", header_extension="cpp.inc"
)


def render_function_declaration(function: interface.Function, body: str) -> str:
    template_params: list[str] = []
    rendered_params: list[str] = []
    for index, param in enumerate(function.parameters):
        if isinstance(param.type_, ts.ScalarType):
            rendered_params.append(f"{cpp_utils.pytype_to_cpptype(param.type_)} {param.name}")
        elif ti.is_type_or_tuple_of_type(param.type_, (ts.FieldType, ts.ScalarType)):
            template_param = f"ArgT{index}"
            template_params.append(f"class {template_param}")
            rendered_params.append(f"{template_param}&& {param.name}")
        else:
            raise ValueError(f"Type '{param.type_}' is not supported in C++ interfaces.")

    rendered_decl = f"""decltype(auto) {function.name}({", ".join(rendered_params)}) {{
        {body}
    }}"""
    if template_params:
        return f"""
        template <{', '.join(template_params)}>
        {rendered_decl}
        """.strip()
    return rendered_decl


def render_function_call(function: interface.Function, args: Sequence[str]) -> str:
    return f"{function.name}({', '.join(args)})"
