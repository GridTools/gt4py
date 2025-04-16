# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from gt4py.next.type_system import type_specifications as ts


def pytype_to_cpptype(t: ts.ScalarType | str) -> str:
    if isinstance(t, ts.ScalarType):
        t = t.kind.name.lower()
    try:
        return {
            "float32": "float",
            "float64": "double",
            "int8": "std::int8_t",
            "uint8": "std::uint8_t",
            "int16": "std::int16_t",
            "uint16": "std::uint16_t",
            "int32": "std::int32_t",
            "uint32": "std::uint32_t",
            "int64": "std::int64_t",
            "uint64": "std::uint64_t",
            "bool": "bool",
            "string": "string",
        }[t]
    except KeyError:
        raise TypeError(f"Unsupported type '{t}'.") from None
