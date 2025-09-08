# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import NamedTuple, Final

import pytest

from gt4py import next as gtx
from gt4py.eve.extended_typing import NestedTuple
from gt4py.next import Dimension, Field, float32, Dims, containers


# Meaningless dimensions, used for tests.
TDim = Dimension("TDim")
SDim = Dimension("SDim")


# Sample container types
class NamedTupleContainer(NamedTuple):
    x: Field[Dims[TDim], float32]
    y: Field[Dims[TDim], float32]

    @classmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.PythonContainerValue], NamedTupleContainer]:
        return (
            (
                x := gtx.constructors.full({TDim: 5}, 2.0),
                y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            NamedTupleContainer(x, y),
        )


@dataclasses.dataclass
class DataclassContainer:
    x: Field[Dims[TDim], float32]
    y: Field[Dims[TDim], float32]

    @classmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.PythonContainerValue], DataclassContainer]:
        return (
            (
                x := gtx.constructors.full({TDim: 5}, 2.0),
                y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            DataclassContainer(x, y),
        )


@dataclasses.dataclass
class NestedDataclassContainer:
    a: DataclassContainer
    b: DataclassContainer
    c: DataclassContainer

    @classmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.PythonContainerValue], NestedDataclassContainer]:
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


class NestedNamedTupleDataclassContainer(NamedTuple):
    a: DataclassContainer
    b: DataclassContainer
    c: DataclassContainer

    @classmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.PythonContainerValue], NestedNamedTupleDataclassContainer]:
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


@dataclasses.dataclass
class NestedDataclassNamedTupleContainer:
    a: NamedTupleContainer
    b: NamedTupleContainer
    c: NamedTupleContainer

    @classmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.PythonContainerValue], NestedDataclassNamedTupleContainer]:
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


@dataclasses.dataclass
class NestedMixedTupleContainer:
    a: NamedTupleContainer
    b: DataclassContainer
    c: NamedTupleContainer

    @classmethod
    def reference_sample(
        cls,
    ) -> tuple[NestedTuple[containers.PythonContainerValue], NestedMixedTupleContainer]:
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


CONTAINER_SAMPLES: Final[
    list[tuple[NestedTuple[containers.PythonContainerValue], containers.PythonContainer]]
] = [
    DataclassContainer.reference_sample(),
    NamedTupleContainer.reference_sample(),
    NestedDataclassContainer.reference_sample(),
    NestedNamedTupleDataclassContainer.reference_sample(),
    NestedDataclassNamedTupleContainer.reference_sample(),
    NestedMixedTupleContainer.reference_sample(),
]


@pytest.mark.parametrize(["expected_nested_tuple", "container"], CONTAINER_SAMPLES)
def test_make_container_extractor(expected_nested_tuple, container):
    container_type = type(container)
    extractor = gtx.containers.make_container_extractor(container_type)

    assert extractor(container) == expected_nested_tuple
    # assert extractor(container) is container or not isinstance(container, tuple)


@pytest.mark.parametrize(["nested_tuple", "expected_container"], CONTAINER_SAMPLES)
def test_make_container_constructor(nested_tuple, expected_container):
    container_type = type(expected_container)
    constructor = gtx.containers.make_container_constructor(container_type)
    constructed_container = constructor(nested_tuple)

    assert isinstance(constructed_container, container_type)
    assert constructed_container == expected_container
