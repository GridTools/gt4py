# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final, Sequence

from gt4py.next.otf import languages
from gt4py.next.otf.binding import interface
from gt4py.next.type_system import type_info as ti, type_specifications as ts


CPP_DEFAULT: Final = languages.LanguageWithHeaderFilesSettings(
    formatter_key="cpp", formatter_style="LLVM", file_extension="cpp", header_extension="cpp.inc"
)


def render_scalar_type(scalar_type: ts.ScalarType) -> str:
    match scalar_type.kind:  # TODO: merge with dict in itir_tp_gtfn
        case ts.ScalarKind.BOOL:
            return "bool"
        case ts.ScalarKind.INT8:
            return "std::int8_t"
        case ts.ScalarKind.UINT8:
            return "std::uint8_t"
        case ts.ScalarKind.INT16:
            return "std::int16_t"
        case ts.ScalarKind.UINT16:
            return "std::uint16_t"
        case ts.ScalarKind.INT32:
            return "std::int32_t"
        case ts.ScalarKind.UINT32:
            return "std::uint32_t"
        case ts.ScalarKind.INT64:
            return "std::int64_t"
        case ts.ScalarKind.UINT64:
            return "std::uint64_t"
        case ts.ScalarKind.FLOAT16:
            return "std::float16_t"
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


def render_function_declaration(function: interface.Function, body: str) -> str:
    template_params: list[str] = []
    rendered_params: list[str] = []
    for index, param in enumerate(function.parameters):
        if isinstance(param.type_, ts.ScalarType):
            rendered_params.append(f"{render_scalar_type(param.type_)} {param.name}")
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
