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

from next_tests.fixtures.python_containers import (
    python_container_definition,
    PythonContainerDefinition,
)


def test_make_container_extractor(python_container_definition: type[PythonContainerDefinition]):
    expected_nested_tuple, container = python_container_definition.reference_sample()
    container_type = type(container)
    extractor = gtx.containers.make_container_extractor(container_type)
    extracted_tuple = extractor(container)

    assert isinstance(extracted_tuple, tuple)
    assert extractor(container) == expected_nested_tuple

    # Test fast path for pure named tuples
    assert extractor(container) is container or not (
        isinstance(container, tuple) and all(isinstance(v, Field) for v in expected_nested_tuple)
    )


def test_make_container_constructor(python_container_definition: type[PythonContainerDefinition]):
    nested_tuple, expected_container = python_container_definition.reference_sample()
    container_type = type(expected_container)
    constructor = gtx.containers.make_container_constructor(container_type)
    constructed_container = constructor(nested_tuple)

    assert isinstance(constructed_container, container_type)
    assert constructed_container == expected_container
