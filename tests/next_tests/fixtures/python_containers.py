# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import dataclasses
import typing

from typing import NamedTuple, Final, Protocol

import pytest

from gt4py import next as gtx
from gt4py.eve.extended_typing import NestedTuple
from gt4py.next import Dimension, Field, float32, Dims, containers
from gt4py.next.type_system import type_specifications as ts


# Meaningless dimensions, used for tests.
TDim = Dimension("TDim")
SDim = Dimension("SDim")


@typing.runtime_checkable
class PythonContainerDefinition(Protocol):
    @classmethod
    @abc.abstractmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.common.NumericValue], containers.PyContainer]: ...

    @classmethod
    @abc.abstractmethod
    def reference_type_spec(cls) -> ts.NamedTupleType: ...

    # @classmethod
    # @abc.abstractmethod
    # def reference_constructor_type_spec(cls) -> ts.ConstructorType: ...


# -- Sample container types --
def _make_type_string(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


class NamedTupleContainer(NamedTuple):
    x: Field[Dims[TDim], float32]
    y: Field[Dims[TDim], float32]

    @classmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.common.NumericValue], NamedTupleContainer]:
        return (
            (
                x := gtx.constructors.full({TDim: 5}, 2.0),
                y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            NamedTupleContainer(x, y),
        )

    @classmethod
    def reference_type_spec(cls) -> ts.NamedTupleType:
        return ts.NamedTupleType(
            types=[
                ts.FieldType(dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
                ts.FieldType(dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
            ],
            keys=["x", "y"],
            original_python_type=_make_type_string(cls),
        )


@dataclasses.dataclass
class DataclassContainer:
    x: Field[Dims[TDim], float32]
    y: Field[Dims[TDim], float32]

    @classmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.common.NumericValue], DataclassContainer]:
        return (
            (
                x := gtx.constructors.full({TDim: 5}, 2.0),
                y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            DataclassContainer(x, y),
        )

    @classmethod
    def reference_type_spec(cls) -> ts.NamedTupleType:
        return ts.NamedTupleType(
            types=[
                ts.FieldType(dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
                ts.FieldType(dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
            ],
            keys=["x", "y"],
            original_python_type=_make_type_string(cls),
        )


@dataclasses.dataclass
class NestedDataclassContainer:
    a: DataclassContainer
    b: DataclassContainer
    c: DataclassContainer

    @classmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.common.NumericValue], NestedDataclassContainer]:
        return (
            (
                (
                    a_x := gtx.constructors.full({TDim: 5}, 2.0),
                    a_y := gtx.constructors.full({TDim: 5}, 3.0),
                ),
                (
                    b_x := gtx.constructors.full({TDim: 5}, 4.0),
                    b_y := gtx.constructors.full({TDim: 5}, 5.0),
                ),
                (
                    c_x := gtx.constructors.full({TDim: 5}, 6.0),
                    c_y := gtx.constructors.full({TDim: 5}, 7.0),
                ),
            ),
            NestedDataclassContainer(
                DataclassContainer(a_x, a_y),
                DataclassContainer(b_x, b_y),
                DataclassContainer(c_x, c_y),
            ),
        )

    @classmethod
    def reference_type_spec(cls) -> ts.NamedTupleType:
        return ts.NamedTupleType(
            types=[
                DataclassContainer.reference_type_spec(),
                DataclassContainer.reference_type_spec(),
                DataclassContainer.reference_type_spec(),
            ],
            keys=["a", "b", "c"],
            original_python_type=_make_type_string(cls),
        )


class NestedNamedTupleDataclassContainer(NamedTuple):
    a: DataclassContainer
    b: DataclassContainer
    c: DataclassContainer

    @classmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.common.NumericValue], NestedNamedTupleDataclassContainer]:
        return (
            (
                (
                    a_x := gtx.constructors.full({TDim: 5}, 2.0),
                    a_y := gtx.constructors.full({TDim: 5}, 3.0),
                ),
                (
                    b_x := gtx.constructors.full({TDim: 5}, 4.0),
                    b_y := gtx.constructors.full({TDim: 5}, 5.0),
                ),
                (
                    c_x := gtx.constructors.full({TDim: 5}, 6.0),
                    c_y := gtx.constructors.full({TDim: 5}, 7.0),
                ),
            ),
            NestedNamedTupleDataclassContainer(
                DataclassContainer(a_x, a_y),
                DataclassContainer(b_x, b_y),
                DataclassContainer(c_x, c_y),
            ),
        )

    @classmethod
    def reference_type_spec(cls) -> ts.NamedTupleType:
        return ts.NamedTupleType(
            types=[
                DataclassContainer.reference_type_spec(),
                DataclassContainer.reference_type_spec(),
                DataclassContainer.reference_type_spec(),
            ],
            keys=["a", "b", "c"],
            original_python_type=_make_type_string(cls),
        )


@dataclasses.dataclass
class NestedDataclassNamedTupleContainer:
    a: NamedTupleContainer
    b: NamedTupleContainer
    c: NamedTupleContainer

    @classmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.common.NumericValue], NestedDataclassNamedTupleContainer]:
        return (
            (
                (
                    a_x := gtx.constructors.full({TDim: 5}, 2.0),
                    a_y := gtx.constructors.full({TDim: 5}, 3.0),
                ),
                (
                    b_x := gtx.constructors.full({TDim: 5}, 4.0),
                    b_y := gtx.constructors.full({TDim: 5}, 5.0),
                ),
                (
                    c_x := gtx.constructors.full({TDim: 5}, 6.0),
                    c_y := gtx.constructors.full({TDim: 5}, 7.0),
                ),
            ),
            NestedDataclassNamedTupleContainer(
                NamedTupleContainer(a_x, a_y),
                NamedTupleContainer(b_x, b_y),
                NamedTupleContainer(c_x, c_y),
            ),
        )

    @classmethod
    def reference_type_spec(cls) -> ts.NamedTupleType:
        return ts.NamedTupleType(
            types=[
                NamedTupleContainer.reference_type_spec(),
                NamedTupleContainer.reference_type_spec(),
                NamedTupleContainer.reference_type_spec(),
            ],
            keys=["a", "b", "c"],
            original_python_type=_make_type_string(cls),
        )


@dataclasses.dataclass
class NestedMixedTupleContainer:
    a: NamedTupleContainer
    b: DataclassContainer
    c: NamedTupleContainer

    @classmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.common.NumericValue], NestedMixedTupleContainer]:
        return (
            (
                (
                    a_x := gtx.constructors.full({TDim: 5}, 2.0),
                    a_y := gtx.constructors.full({TDim: 5}, 3.0),
                ),
                (
                    b_x := gtx.constructors.full({TDim: 5}, 4.0),
                    b_y := gtx.constructors.full({TDim: 5}, 5.0),
                ),
                (
                    c_x := gtx.constructors.full({TDim: 5}, 6.0),
                    c_y := gtx.constructors.full({TDim: 5}, 7.0),
                ),
            ),
            NestedMixedTupleContainer(
                NamedTupleContainer(a_x, a_y),
                DataclassContainer(b_x, b_y),
                NamedTupleContainer(c_x, c_y),
            ),
        )

    @classmethod
    def reference_type_spec(cls) -> ts.NamedTupleType:
        return ts.NamedTupleType(
            types=[
                NamedTupleContainer.reference_type_spec(),
                DataclassContainer.reference_type_spec(),
                NamedTupleContainer.reference_type_spec(),
            ],
            keys=["a", "b", "c"],
            original_python_type=_make_type_string(cls),
        )


PYTHON_CONTAINER_DEFINITIONS: list[type] = [
    definition
    for definition in globals().values()
    if type(definition) is type and issubclass(definition, PythonContainerDefinition)
]


@pytest.fixture(params=PYTHON_CONTAINER_DEFINITIONS, ids=lambda cls: cls.__name__)
def python_container_definition(request) -> type[PythonContainerDefinition]:
    return request.param
