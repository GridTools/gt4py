# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import gt4py.next.type_system.type_specifications as ts
from gt4py.next import common
from gt4py.next.type_system.type_specifications import TupleType


class ProgramType(ts.TypeSpec, ts.CallableType):
    definition: ts.FunctionType


class FieldOperatorType(ts.TypeSpec, ts.CallableType):
    definition: ts.FunctionType


class ConstructorType(ts.TypeSpec, ts.CallableType):
    definition: ts.FunctionType


class ScanOperatorType(ts.TypeSpec, ts.CallableType):
    axis: common.Dimension
    definition: ts.FunctionType


class ConstructorType(ts.TypeSpec, ts.CallableType):
    definition: ts.FunctionType


class NamedTupleType(TupleType):
    keys: list[str]

    def __getattr__(self, name):
        keys = object.__getattribute__(self, "keys")
        if name in keys:
            return self.types[keys.index(name)]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __str__(self) -> str:
        return f"NamedTuple{{{', '.join(f'{k}: {v}' for k, v in zip(self.keys, self.types))}}}"
