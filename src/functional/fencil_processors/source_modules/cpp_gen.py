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
from types import MappingProxyType
from typing import Final, Sequence, Type

import numpy

from functional.fencil_processors import source_modules


LANGUAGE_ID: Final = "cpp"

_TYPE_MAPPING: Final = MappingProxyType(
    {
        bool: "bool",
        int: "long",
        float: "double",
        complex: "std::complex<double>",
        numpy.bool_: "bool",
        numpy.byte: "signed char",
        numpy.ubyte: "unsigned char",
        numpy.short: "short",
        numpy.ushort: "unsigned short",
        numpy.intc: "int",
        numpy.uintc: "unsigned int",
        numpy.int_: "long",
        numpy.uint: "unsigned long",
        numpy.longlong: "long long",
        numpy.ulonglong: "unsigned long long",
        numpy.single: "float",
        numpy.double: "double",
        numpy.longdouble: "long double",
        numpy.csingle: "std::complex<float>",
        numpy.cdouble: "std::complex<double>",
        numpy.clongdouble: "std::complex<long double>",
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


def _render_function_param(
    param: source_modules.ScalarParameter | source_modules.BufferParameter, index: int
) -> str:
    if isinstance(param, source_modules.ScalarParameter):
        return f"{render_python_type(param.scalar_type.type)} {param.name}"
    else:
        return f"BufferT{index}&& {param.name}"


def render_function_declaration(function: source_modules.Function, body: str) -> str:
    rendered_params = [
        _render_function_param(param, index) for index, param in enumerate(function.parameters)
    ]
    rendered_decl = f"""decltype(auto) {function.name}({", ".join(rendered_params)}) {{
        {body}
    }}"""
    template_params = [
        f"class BufferT{index}"
        for index, param in enumerate(function.parameters)
        if isinstance(param, source_modules.BufferParameter)
    ]
    if template_params:
        return f"""
        template <{', '.join(template_params)}>
        {rendered_decl}
        """.strip()
    return rendered_decl


def render_function_call(function: source_modules.Function, args: Sequence[str]) -> str:
    return f"{function.name}({', '.join(args)})"
