# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import pytest

from gt4py import next as gtx
from gt4py.next import Dimension, Field, float32, Dims


# Meaningless dimensions, used for tests.
TDim = Dimension("TDim")
SDim = Dimension("SDim")


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


@pytest.mark.parametrize(
    ["nested_tuple", "container_type", "expected_container"],
    [
        (
            (
                x := gtx.constructors.full({TDim: 5}, 2.0),
                y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            NamedTupleContainer,
            NamedTupleContainer(x, y),
        ),
        (
            (
                x := gtx.constructors.full({TDim: 5}, 2.0),
                y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            DataclassContainer,
            DataclassContainer(x, y),
        ),
        (
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
            NestedDataclassContainer,
            NestedDataclassContainer(
                DataclassContainer(a_x, a_y),
                DataclassContainer(b_x, b_y),
                DataclassContainer(c_x, c_y),
            ),
        ),
        (
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
            NestedNamedTupleDataclassContainer,
            NestedNamedTupleDataclassContainer(
                DataclassContainer(a_x, a_y),
                DataclassContainer(b_x, b_y),
                DataclassContainer(c_x, c_y),
            ),
        ),
        (
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
            NestedDataclassNamedTupleContainer,
            NestedDataclassNamedTupleContainer(
                NamedTupleContainer(a_x, a_y),
                NamedTupleContainer(b_x, b_y),
                NamedTupleContainer(c_x, c_y),
            ),
        ),
        (
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
            NestedMixedTupleContainer,
            NestedMixedTupleContainer(
                NamedTupleContainer(a_x, a_y),
                DataclassContainer(b_x, b_y),
                NamedTupleContainer(c_x, c_y),
            ),
        ),
    ],
)
def test_make_container_constructor(nested_tuple, container_type, expected_container):
    constructor = gtx.containers.make_container_constructor(container_type)
    constructed_container = constructor(nested_tuple)

    assert isinstance(constructed_container, container_type)
    assert constructed_container == expected_container
