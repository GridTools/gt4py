# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

import pytest

from gt4py import next as gtx
from gt4py.eve.extended_typing import NestedTuple
from gt4py.next import common, Field, containers

from ..artifacts import pycontainers as pc


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
    ["container", "expected_nested_tuple"],
    [(pc.from_nested_tuple(cls, value), value) for cls, value in pc.PYCONTAINERS_SAMPLES.items()],
    ids=lambda val: val.__class__.__name__,
)
def test_make_container_extractor(
    container: containers.PyContainer,
    expected_nested_tuple: NestedTuple[common.NumericValue],
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
    ["nested_tuple", "expected_container"],
    [
        (value, pc.from_nested_tuple(cls, value))
        for cls, value in pc.PYCONTAINERS_SAMPLES.items()
    ],
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
