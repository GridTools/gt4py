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
from gt4py.next import common, Field, named_collections

from next_tests.artifacts import custom_named_collections as cnc


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
        (cnc.DataclassNamedCollection, True),
        (DataclassWithDefaults, False),
        (DataclassWithDefaultFactory, False),
        (DataclassWithInitVar, False),
        (gtx.Field, False),
        (cnc.NamedTupleNamedCollection, False),
    ],
)
def test_is_dataclass_container_(type_: type, is_container: bool):
    assert issubclass(type_, named_collections.CustomDataclassNamedCollectionABC) is is_container


@pytest.mark.parametrize(
    ["container", "expected_nested_tuple"],
    [(cnc.from_nested_tuple(cls, value), value) for cls, value in cnc.PYCONTAINERS_SAMPLES.items()],
    ids=lambda val: val.__class__.__name__,
)
def test_make_container_extractor(
    container: named_collections.CustomNamedCollection,
    expected_nested_tuple: NestedTuple[common.NumericValue],
):
    container_type = type(container)
    extractor = named_collections.make_named_collection_extractor(container_type)
    extracted_tuple = extractor(container)

    assert isinstance(extracted_tuple, tuple)
    assert extractor(container) == expected_nested_tuple

    # Test fast path for pure named tuples
    assert extractor(container) is container or not (
        isinstance(container, tuple) and all(isinstance(v, Field) for v in expected_nested_tuple)
    )


@pytest.mark.parametrize(
    ["nested_tuple", "expected_container"],
    [(value, cnc.from_nested_tuple(cls, value)) for cls, value in cnc.PYCONTAINERS_SAMPLES.items()],
    ids=lambda val: val.__class__.__name__,
)
def test_make_container_constructor(
    nested_tuple: NestedTuple[common.NumericValue],
    expected_container: named_collections.CustomNamedCollection,
):
    container_type = type(expected_container)
    constructor = named_collections.make_named_collection_constructor(container_type)
    constructed_container = constructor(nested_tuple)

    assert isinstance(constructed_container, container_type)
    assert constructed_container == expected_container
