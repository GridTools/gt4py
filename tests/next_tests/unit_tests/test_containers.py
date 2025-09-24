# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Final

import pytest

from gt4py import next as gtx
from gt4py.eve.extended_typing import NestedTuple
from gt4py.next import common, Field, containers

from ..artifacts import pycontainers as pc

# Meaningless dimensions
TDim = pc.TDim
SDim = pc.SDim

CONTAINERS_AS_VALUES: Final[
    list[tuple[NestedTuple[containers.common.NumericValue], containers.PyContainer]]
] = [
    (
        (
            x := gtx.constructors.full({TDim: 5}, 2.0),
            y := gtx.constructors.full({TDim: 5}, 3.0),
        ),
        pc.NamedTupleContainer(x, y),
    ),
    (
        (
            x := gtx.constructors.full({TDim: 5}, 2.0),
            y := gtx.constructors.full({TDim: 5}, 3.0),
        ),
        pc.DataclassContainer(x, y),
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
        pc.NestedDataclassContainer(
            pc.DataclassContainer(a_x, a_y),
            pc.DataclassContainer(b_x, b_y),
            pc.DataclassContainer(c_x, c_y),
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
        pc.NestedNamedTupleDataclassContainer(
            pc.DataclassContainer(a_x, a_y),
            pc.DataclassContainer(b_x, b_y),
            pc.DataclassContainer(c_x, c_y),
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
        pc.NestedDataclassNamedTupleContainer(
            pc.NamedTupleContainer(a_x, a_y),
            pc.NamedTupleContainer(b_x, b_y),
            pc.NamedTupleContainer(c_x, c_y),
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
        pc.NestedMixedTupleContainer(
            pc.NamedTupleContainer(a_x, a_y),
            pc.DataclassContainer(b_x, b_y),
            pc.NamedTupleContainer(c_x, c_y),
        ),
    ),
]


@dataclasses.dataclass
class DataclassWithDefaults:
    a: int = 10


@dataclasses.dataclass
class DataclassWithDefaultFactory:
    a: int = dataclasses.field(default_factory=lambda: 10)


@dataclasses.dataclass
class DataclassWithInitVar:
    b: dataclasses.InitVar[int]
    a: int


print(DataclassWithInitVar.__dataclass_fields__)


@pytest.mark.parametrize(
    "type_,is_container",
    [
        (pc.DataclassContainer, True),
        (DataclassWithDefaults, False),
        (DataclassWithDefaultFactory, False),
        (DataclassWithInitVar, False),
        (gtx.Field, False),
        (pc.NamedTupleContainer, False),
    ],
)
def test_is_dataclass_container_(type_: type, is_container: bool):
    assert issubclass(type_, containers.PyContainerDataclassABC) is is_container


@pytest.mark.parametrize(
    "expected_nested_tuple, container",
    CONTAINERS_AS_VALUES,
    ids=lambda val: val.__class__.__name__,
)
def test_make_container_extractor(
    expected_nested_tuple: NestedTuple[common.NumericValue],
    container: containers.PyContainer,
):
    container_type = type(container)
    extractor = containers.make_container_extractor(container_type)
    extracted_tuple = extractor(container)

    assert isinstance(extracted_tuple, tuple)
    assert extractor(container) == expected_nested_tuple

    # Test fast path for pure named tuples
    assert extractor(container) is container or not (
        isinstance(container, tuple) and all(isinstance(v, Field) for v in expected_nested_tuple)
    )


@pytest.mark.parametrize(
    "nested_tuple, expected_container",
    CONTAINERS_AS_VALUES,
    ids=lambda val: val.__class__.__name__,
)
def test_make_container_constructor(
    nested_tuple: NestedTuple[common.NumericValue], expected_container: containers.PyContainer
):
    container_type = type(expected_container)
    constructor = containers.make_container_constructor(container_type)
    constructed_container = constructor(nested_tuple)

    assert isinstance(constructed_container, container_type)
    assert constructed_container == expected_container
