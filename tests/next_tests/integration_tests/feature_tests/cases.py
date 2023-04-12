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

from __future__ import annotations

import dataclasses
from typing import Type, TypeAlias

import numpy as np
import pytest
from typing_extensions import Self

from gt4py.next.ffront import decorator, fbuiltins
from gt4py.next.iterator import embedded
from gt4py.next.program_processors import processor_interface as ppi
from gt4py.next.type_system import type_specifications as ts

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    IDim,
    Ioff,
    JDim,
    Joff,
    KDim,
    Koff,
    fieldview_backend,
)


IField: TypeAlias = fbuiltins.Field[[IDim], int]
IJKField: TypeAlias = fbuiltins.Field[[IDim, JDim, KDim], int]


def no_backend(*args, **kwargs) -> None:
    raise ValueError("No backend selected! Backend selection is mandatory in tests.")


@dataclasses.dataclass
class FieldBuilder:
    backend: ppi.ProgramProcessor
    ftype: ts.FieldType
    shape: tuple[int, int, int] = (10, 10, 10)
    _dtype: str = int

    def zeros(self) -> fbuiltins.Field:
        return embedded.np_as_located_field(*self.dims)(
            np.zeros(self.effective_shape, dtype=self._dtype)
        )

    def default(self) -> fbuiltins.Field:
        dims = self.dims
        match dims:
            case (_,):
                func = self.i_defaults
            case (_, _):
                func = self.ij_defaults
            case (_, _, _):
                func = self.ijk_defaults
        return embedded.np_as_located_field(*self.dims)(
            np.fromfunction(func, self.effective_shape, dtype=self._dtype)
        )

    def size(self, size: int) -> Self:
        return self.__class__(self.backend, self.ftype, (size, size, size), self.dtype)

    def dtype(self, dtype: str) -> Self:
        return self.__class__(self.backend, self.ftype, self.shape, dtype)

    @property
    def dims(self) -> tuple[fbuiltins.Dimension]:
        #  if self.name == "out":
        #      my_type = self.fieldop.foast_node.definition.body.stmts[-1].value.type
        #  else:
        #      params = {
        #          str(param.id): param.type for param in self.fieldop.foast_node.definition.params
        #      }
        #      my_type = params[self.name]
        #  if not isinstance(my_type, ts.FieldType):
        #      raise TypeError("Can not allocate non-field argument!")
        return self.ftype.dims

    @property
    def effective_shape(self):
        match self.dims:
            case (_,):
                shape = (self.shape[0],)
            case (_, _):
                shape = (self.shape[0], self.shape[1])
            case (_, _, _):
                shape = self.shape
        return shape

    def i_defaults(self, i: int) -> int:
        return i

    def ij_defaults(self, i: int, j: int) -> int:
        return self.i_defaults(i) + (self.shape[0] * j)

    def ijk_defaults(self, i: int, j: int, k: int) -> int:
        return self.ij_defaults(i, j) + (self.shape[0] * self.shape[1] * k)


@dataclasses.dataclass
class FieldTupleBuilder:
    field_builders: tuple[FieldBuilder, ...]

    def zeros(self) -> tuple[fbuiltins.Field, ...]:
        return tuple(fb.zeros() for fb in self.field_builders)

    def default(self) -> tuple[fbuiltins.Field, ...]:
        return tuple(fb.default() for fb in self.field_builders)

    def size(self, size: int) -> Self:
        return self.__class__((fb.size(size) for fb in self.field_builders))

    def dtype(self, dtype: str) -> Self:
        return self.__class__((fb.dtype(dtype) for fb in self.field_builders))


@dataclasses.dataclass
class CartesianCase:
    backend: ppi.ProgramProcessor

    def allocate(
        self, fieldop: decorator.FieldOperator, name: str
    ) -> FieldBuilder | FieldTupleBuilder:
        arg_type = (
            fieldop.foast_node.definition.body.stmts[-1].value.type
            if name == "out"
            else {str(param.id): param.type for param in fieldop.foast_node.definition.params}[name]
        )
        return self._allocate_for(arg_type)

    def _allocate_for(
        self, arg_type: ts.FieldType | ts.TupleType
    ) -> FieldBuilder | FieldTupleBuilder:
        match arg_type:
            case ts.FieldType():
                return FieldBuilder(self.backend, arg_type)
            case ts.TupleType():
                return FieldTupleBuilder((self._allocate_for(t) for t in arg_type.types))
            case _:
                raise TypeError(f"Can not allocate for type {arg_type}")

    def verify(
        self,
        fieldop: decorator.FieldOperator,
        *args: fbuiltins.Field,
        out: fbuiltins.Field,
        ref: fbuiltins.Field,
        offset_provider={},
    ) -> None:
        fieldop.with_backend(self.backend)(*args, out=out, offset_provider=offset_provider)

        assert np.allclose(ref, out)


@pytest.fixture
def cartesian_case(fieldview_backend):  # noqa: F811 # fixture
    backup_backend = decorator.DEFAULT_BACKEND
    decorator.DEFAULT_BACKEND = no_backend
    yield CartesianCase(fieldview_backend)
    decorator.DEFAULT_BACKEND = backup_backend
