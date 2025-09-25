# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import pytest

from gt4py.eve.extended_typing import NestedTuple
from gt4py.next import common
from gt4py.next import containers
from gt4py.next.otf import arguments

from ...artifacts import pycontainers as pc


def test_needs_value_extraction_with_non_container():
    """Test needs_value_extraction with a non-container argument."""
    assert not arguments.needs_value_extraction(42)
    assert not arguments.needs_value_extraction("string")
    assert not arguments.needs_value_extraction([1, 2, 3])
    assert not arguments.needs_value_extraction({"key": "value"})
    assert not arguments.needs_value_extraction(None)


def test_needs_value_extraction_with_container():
    """Test needs_value_extraction with a container argument."""
    assert arguments.needs_value_extraction(pc.DataclassContainer(x=1, y=2))
    assert arguments.needs_value_extraction(pc.NamedTupleContainer(x=1, y=2))
    assert arguments.needs_value_extraction(
        pc.NestedMixedTupleContainer(
            a=pc.DataclassContainer(x=1, y=2),
            b=pc.NamedTupleContainer(x=1, y=2),
            c=pc.DataclassContainer(x=1, y=2),
        )
    )


def test_extract_with_non_container():
    """Test extract with a non-container argument."""
    assert arguments.extract(42) == 42
    assert arguments.extract(42, pass_through_values=False) == 42

    assert arguments.extract("string") == "string"
    assert arguments.extract([1, 2, 3]) == [1, 2, 3]
    assert arguments.extract({"key": "value"}) == {"key": "value"}
    assert arguments.extract(None) is None

    with pytest.raises(TypeError):
        arguments.extract("string", pass_through_values=False)
    with pytest.raises(TypeError):
        arguments.extract([1, 2, 3], pass_through_values=False)


@pytest.mark.parametrize(
    "expected_nested_tuple, container",
    pc.CONTAINERS_AND_VALUES,
    ids=lambda val: val.__class__.__name__,
)
def test_extract_with_container(
    expected_nested_tuple: NestedTuple[common.NumericValue],
    container: containers.PyContainer,
):
    """Test extract with a container argument."""
    assert arguments.extract(container) == expected_nested_tuple
    assert arguments.extract(container, pass_through_values=False) == expected_nested_tuple
