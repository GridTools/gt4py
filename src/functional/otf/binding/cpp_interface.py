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


import ctypes
import types
from typing import Final, Sequence, Type

import numpy as np

import functional.otf.binding.type_specifications as ts_binding
import functional.type_system.type_specifications as ts
from functional.otf import languages
from functional.otf.binding import interface


CPP_DEFAULT: Final = languages.LanguageWithHeaderFilesSettings(
    formatter_key="cpp",
    formatter_style="LLVM",
    file_extension="cpp",
    header_extension="cpp.inc",
)

_TYPE_MAPPING: Final = types.MappingProxyType(
    {
        bool: "bool",
        int: "long",
        float: "double",
        complex: "std::complex<double>",
        np.bool_: "bool",
        np.byte: "signed char",
        np.ubyte: "unsigned char",
        np.short: "short",
        np.ushort: "unsigned short",
        np.intc: "int",
        np.uintc: "unsigned int",
        np.int_: "long",
        np.uint: "unsigned long",
        np.longlong: "long long",
        np.ulonglong: "unsigned long long",
        np.single: "float",
        np.double: "double",
        np.longdouble: "long double",
        np.csingle: "std::complex<float>",
        np.cdouble: "std::complex<double>",
        np.clongdouble: "std::complex<long double>",
        ctypes.c_bool: "bool",
        ctypes.c_char: "char",
        ctypes.c_wchar: "wchar_t",
        ctypes.c_byte: "char",
        ctypes.c_ubyte: "unsigned char",
        ctypes.c_short: "short",
        ctypes.c_ushort: "unsigned short",
        ctypes.c_int: "int",
        ctypes.c_uint: "unsigned int",
        ctypes.c_long: "long",
        ctypes.c_ulong: "unsigned long",
        ctypes.c_longlong: "long long",
        ctypes.c_ulonglong: "unsigned long long",
        ctypes.c_size_t: "std::size_t",
        ctypes.c_ssize_t: "std::ptrdiff_t",
        ctypes.c_float: "float",
        ctypes.c_double: "double",
        ctypes.c_longdouble: "long double",
    }
)


def render_python_type(python_type: Type) -> str:
    return _TYPE_MAPPING[python_type]


def render_scalar_type(scalar_type: ts.ScalarType) -> str:
    if scalar_type.kind == ts.ScalarKind.BOOL:
        return "bool"
    elif scalar_type.kind == ts.ScalarKind.INT32:
        return "int32_t"
    elif scalar_type.kind == ts.ScalarKind.INT64:
        return "int64_t"
    elif scalar_type.kind == ts.ScalarKind.FLOAT32:
        return "float"
    elif scalar_type.kind == ts.ScalarKind.FLOAT64:
        return "double"
    elif scalar_type.kind == ts.ScalarKind.STRING:
        return "std::string"
    elif scalar_type.kind == ts.ScalarKind.DIMENSION:
        raise AssertionError(f"Deprecated type '{scalar_type}' is not supported.")
    else:
        raise AssertionError(f"Scalar kind '{scalar_type}' is not implemented when it should be.")


def _render_function_param(param: interface.Parameter, index: int) -> str:
    if isinstance(param.type_, ts.ScalarType):
        return f"{render_scalar_type(param.type_)} {param.name}"
    elif isinstance(param.type_, (ts.FieldType, ts_binding.IndexFieldType)):
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
        if isinstance(param.type_, (ts.FieldType, ts_binding.IndexFieldType))
    ]
    if template_params:
        return f"""
        template <{', '.join(template_params)}>
        {rendered_decl}
        """.strip()
    return rendered_decl


def render_function_call(function: interface.Function, args: Sequence[str]) -> str:
    return f"{function.name}({', '.join(args)})"
