# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import typing

from typing import NamedTuple, Final, Protocol

import dataclasses
import pytest

from gt4py import next as gtx
from gt4py.eve.extended_typing import NestedTuple
from gt4py.next import Dimension, Field, float32, Dims, containers
from gt4py.next.type_system import type_specifications as ts


# Meaningless dimensions, used for tests.
TDim = Dimension("TDim")
SDim = Dimension("SDim")


# Sample container types
def _make_type_string(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


class NamedTupleContainer(NamedTuple):
    x: Field[Dims[TDim], float32]
    y: Field[Dims[TDim], float32]

    # @classmethod
    # def reference_type_spec(cls) -> ts.NamedTupleType:
    #     return ts.NamedTupleType(
    #         types=[
    #             ts.FieldType(dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
    #             ts.FieldType(dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
    #         ],
    #         keys=["x", "y"],
    #         original_python_type=_make_type_string(cls),
    #     )


@dataclasses.dataclass
class DataclassContainer:
    x: Field[Dims[TDim], float32]
    y: Field[Dims[TDim], float32]

    # @classmethod
    # def reference_type_spec(cls) -> ts.NamedTupleType:
    #     return ts.NamedTupleType(
    #         types=[
    #             ts.FieldType(dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
    #             ts.FieldType(dims=[TDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32)),
    #         ],
    #         keys=["x", "y"],
    #         original_python_type=_make_type_string(cls),
    #     )


@dataclasses.dataclass
class NestedDataclassContainer:
    a: DataclassContainer
    b: DataclassContainer
    c: DataclassContainer

    # @classmethod
    # def reference_type_spec(cls) -> ts.NamedTupleType:
    #     return ts.NamedTupleType(
    #         types=[
    #             DataclassContainer.reference_type_spec(),
    #             DataclassContainer.reference_type_spec(),
    #             DataclassContainer.reference_type_spec(),
    #         ],
    #         keys=["a", "b", "c"],
    #         original_python_type=_make_type_string(cls),
    #     )


class NestedNamedTupleDataclassContainer(NamedTuple):
    a: DataclassContainer
    b: DataclassContainer
    c: DataclassContainer

    # @classmethod
    # def reference_type_spec(cls) -> ts.NamedTupleType:
    #     return ts.NamedTupleType(
    #         types=[
    #             DataclassContainer.reference_type_spec(),
    #             DataclassContainer.reference_type_spec(),
    #             DataclassContainer.reference_type_spec(),
    #         ],
    #         keys=["a", "b", "c"],
    #         original_python_type=_make_type_string(cls),
    #     )


@dataclasses.dataclass
class NestedDataclassNamedTupleContainer:
    a: NamedTupleContainer
    b: NamedTupleContainer
    c: NamedTupleContainer

    # @classmethod
    # def reference_type_spec(cls) -> ts.NamedTupleType:
    #     return ts.NamedTupleType(
    #         types=[
    #             NamedTupleContainer.reference_type_spec(),
    #             NamedTupleContainer.reference_type_spec(),
    #             NamedTupleContainer.reference_type_spec(),
    #         ],
    #         keys=["a", "b", "c"],
    #         original_python_type=_make_type_string(cls),
    #     )


@dataclasses.dataclass
class NestedMixedTupleContainer:
    a: NamedTupleContainer
    b: DataclassContainer
    c: NamedTupleContainer

    # @classmethod
    # def reference_type_spec(cls) -> ts.NamedTupleType:
    #     return ts.NamedTupleType(
    #         types=[
    #             NamedTupleContainer.reference_type_spec(),
    #             DataclassContainer.reference_type_spec(),
    #             NamedTupleContainer.reference_type_spec(),
    #         ],
    #         keys=["a", "b", "c"],
    #         original_python_type=_make_type_string(cls),
    #     )
