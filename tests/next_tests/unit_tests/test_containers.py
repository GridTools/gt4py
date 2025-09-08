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


@dataclasses.dataclass
class DataclassContainer:
    x: Field[Dims[TDim], float32]
    y: Field[Dims[TDim], float32]


@dataclasses.dataclass
class NestedDataclassContainer:
    a: DataclassContainer
    b: DataclassContainer
    c: DataclassContainer


class NestedNamedTupleDataclassContainer(NamedTuple):
    a: DataclassContainer
    b: DataclassContainer
    c: DataclassContainer


@dataclasses.dataclass
class NestedDataclassNamedTupleContainer:
    a: NamedTupleContainer
    b: NamedTupleContainer
    c: NamedTupleContainer


@dataclasses.dataclass
class NestedMixedTupleContainer:
    a: NamedTupleContainer
    b: DataclassContainer
    c: NamedTupleContainer


CONTAINER_SAMPLES: Final[
    list[tuple[containers.PythonContainer, NestedTuple[containers.PythonContainerValue]]]
] = [
    (
        NamedTupleContainer(
            x := gtx.constructors.full({TDim: 5}, 2.0),
            y := gtx.constructors.full({TDim: 5}, 3.0),
        ),
        (x, y),
    ),
    (
        DataclassContainer(
            x := gtx.constructors.full({TDim: 5}, 2.0),
            y := gtx.constructors.full({TDim: 5}, 3.0),
        ),
        (x, y),
    ),
    (
        NestedDataclassContainer(
            a := DataclassContainer(
                a_x := gtx.constructors.full({TDim: 5}, 2.0),
                a_y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            b := DataclassContainer(
                b_x := gtx.constructors.full({TDim: 5}, 4.0),
                b_y := gtx.constructors.full({TDim: 5}, 5.0),
            ),
            c := DataclassContainer(
                c_x := gtx.constructors.full({TDim: 5}, 6.0),
                c_y := gtx.constructors.full({TDim: 5}, 7.0),
            ),
        ),
        ((a_x, a_y), (b_x, b_y), (c_x, c_y)),
    ),
    (
        NestedNamedTupleDataclassContainer(
            a := DataclassContainer(
                a_x := gtx.constructors.full({TDim: 5}, 2.0),
                a_y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            b := DataclassContainer(
                b_x := gtx.constructors.full({TDim: 5}, 4.0),
                b_y := gtx.constructors.full({TDim: 5}, 5.0),
            ),
            c := DataclassContainer(
                c_x := gtx.constructors.full({TDim: 5}, 6.0),
                c_y := gtx.constructors.full({TDim: 5}, 7.0),
            ),
        ),
        ((a_x, a_y), (b_x, b_y), (c_x, c_y)),
    ),
    (
        NestedDataclassNamedTupleContainer(
            a := NamedTupleContainer(
                a_x := gtx.constructors.full({TDim: 5}, 2.0),
                a_y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            b := NamedTupleContainer(
                b_x := gtx.constructors.full({TDim: 5}, 4.0),
                b_y := gtx.constructors.full({TDim: 5}, 5.0),
            ),
            c := NamedTupleContainer(
                c_x := gtx.constructors.full({TDim: 5}, 6.0),
                c_y := gtx.constructors.full({TDim: 5}, 7.0),
            ),
        ),
        ((a_x, a_y), (b_x, b_y), (c_x, c_y)),
    ),
    (
        NestedMixedTupleContainer(
            a := NamedTupleContainer(
                a_x := gtx.constructors.full({TDim: 5}, 2.0),
                a_y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            b := DataclassContainer(
                b_x := gtx.constructors.full({TDim: 5}, 4.0),
                b_y := gtx.constructors.full({TDim: 5}, 5.0),
            ),
            c := NamedTupleContainer(
                c_x := gtx.constructors.full({TDim: 5}, 6.0),
                c_y := gtx.constructors.full({TDim: 5}, 7.0),
            ),
        ),
        ((a_x, a_y), (b_x, b_y), (c_x, c_y)),
    ),
]


@pytest.mark.parametrize(["container", "expected_nested_tuple"], CONTAINER_SAMPLES)
def test_make_container_extractor(container, expected_nested_tuple):
    container_type = type(container)
    extractor = gtx.containers.make_container_extractor(container_type)

    assert extractor(container) == expected_nested_tuple
    # assert extractor(container) is container or not isinstance(container, tuple)


@pytest.mark.parametrize(["expected_container", "nested_tuple"], CONTAINER_SAMPLES)
def test_make_container_constructor(expected_container, nested_tuple):
    container_type = type(expected_container)
    constructor = gtx.containers.make_container_constructor(container_type)
    constructed_container = constructor(nested_tuple)

    assert isinstance(constructed_container, container_type)
    assert constructed_container == expected_container
