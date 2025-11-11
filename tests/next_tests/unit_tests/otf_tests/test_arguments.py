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
from gt4py.next import named_collections
from gt4py.next.type_system import type_specifications as ts, type_translation as tt
from gt4py.next.otf import arguments

from next_tests.artifacts import custom_named_collections as cnc


def test_needs_value_extraction_with_non_container():
    """Test needs_value_extraction with a non-container argument."""
    assert not arguments.needs_value_extraction(42)
    assert not arguments.needs_value_extraction("string")
    assert not arguments.needs_value_extraction([1, 2, 3])
    assert not arguments.needs_value_extraction({"key": "value"})
    assert not arguments.needs_value_extraction(None)


def test_needs_value_extraction_with_container():
    """Test needs_value_extraction with a container argument."""
    assert arguments.needs_value_extraction(cnc.DataclassNamedCollection(x=1, y=2))
    assert arguments.needs_value_extraction(cnc.NamedTupleNamedCollection(x=1, y=2))
    assert arguments.needs_value_extraction(
        cnc.NestedMixedTupleNamedCollection(
            a=cnc.DataclassNamedCollection(x=1, y=2),
            b=cnc.NamedTupleNamedCollection(x=1, y=2),
            c=cnc.DataclassNamedCollection(x=1, y=2),
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
    ["container", "expected_nested_tuple"],
    [(cnc.from_nested_tuple(cls, value), value) for cls, value in cnc.PYCONTAINERS_SAMPLES.items()],
    ids=lambda val: val.__class__.__name__,
)
def test_extract_with_container(
    container: named_collections.CustomNamedCollection,
    expected_nested_tuple: NestedTuple[common.NumericValue],
):
    """Test extract with a container argument."""
    assert arguments.extract(container) == expected_nested_tuple
    assert arguments.extract(container, pass_through_values=False) == expected_nested_tuple


def test_make_primitive_value_args_extractor_no_extraction_needed():
    """Test make_primitive_value_args_extractor when no arguments need extraction."""

    function_type = ts.FunctionType(
        pos_only_args=[ts.ScalarType(kind=ts.ScalarKind.INT32)],
        pos_or_kw_args={"b": ts.ScalarType(kind=ts.ScalarKind.FLOAT64)},
        kw_only_args={"c": ts.ScalarType(kind=ts.ScalarKind.BOOL)},
        returns=ts.VoidType(),
    )

    extractor = arguments.make_primitive_value_args_extractor(function_type)
    assert extractor is None


@pytest.mark.parametrize(
    "pycontainer_type", [cnc.NamedTupleNamedCollection, cnc.DataclassNamedCollection]
)
def test_make_primitive_value_args_extractor_with_pos_args(
    pycontainer_type: type[named_collections.CustomNamedCollection],
):
    """Test make_primitive_value_args_extractor with positional arguments needing extraction."""

    container_type_spec = tt.from_type_hint(pycontainer_type)

    function_type_pos_only = ts.FunctionType(
        pos_only_args=[container_type_spec],
        pos_or_kw_args={"b": ts.ScalarType(kind=ts.ScalarKind.FLOAT64)},
        kw_only_args={},
        returns=ts.VoidType(),
    )

    function_type_pos_or_kw = ts.FunctionType(
        pos_only_args=[container_type_spec],
        pos_or_kw_args={"b": ts.ScalarType(kind=ts.ScalarKind.FLOAT64)},
        kw_only_args={},
        returns=ts.VoidType(),
    )

    for function_type in [function_type_pos_only, function_type_pos_or_kw]:
        extractor = arguments.make_primitive_value_args_extractor(function_type)
        assert extractor is not None

        # Test the generated extractor
        container = pycontainer_type(x=1.0, y=2.0)
        args, kwargs = extractor(container, 3.14)
        assert args == ((1.0, 2.0), 3.14)
        assert kwargs == {}


@pytest.mark.parametrize(
    "pycontainer_type", [cnc.NamedTupleNamedCollection, cnc.DataclassNamedCollection]
)
def test_make_primitive_value_args_extractor_with_kw_args(
    pycontainer_type: type[named_collections.CustomNamedCollection],
):
    """Test make_primitive_value_args_extractor with keyword arguments needing extraction."""

    container_type = tt.from_type_hint(pycontainer_type)
    function_type = ts.FunctionType(
        pos_only_args=[],
        pos_or_kw_args={"a": ts.ScalarType(kind=ts.ScalarKind.INT32)},
        kw_only_args={"container_arg": container_type},
        returns=ts.VoidType(),
    )

    extractor = arguments.make_primitive_value_args_extractor(function_type)
    assert extractor is not None

    # Test the generated extractor
    container = pycontainer_type(x=1.0, y=2.0)
    args, kwargs = extractor(42, container_arg=container)
    assert args == (42,)
    assert kwargs == {"container_arg": (1.0, 2.0)}


@pytest.mark.parametrize(
    "pycontainer_type", [cnc.NamedTupleNamedCollection, cnc.DataclassNamedCollection]
)
def test_make_primitive_value_args_extractor_with_tuple_args(
    pycontainer_type: type[named_collections.CustomNamedCollection],
):
    """Test make_primitive_value_args_extractor with tuple arguments containing containers."""

    container_type = tt.from_type_hint(pycontainer_type)
    tuple_type = ts.TupleType(types=[ts.ScalarType(kind=ts.ScalarKind.INT32), container_type])
    function_type = ts.FunctionType(
        pos_only_args=[tuple_type], pos_or_kw_args={}, kw_only_args={}, returns=ts.VoidType()
    )

    extractor = arguments.make_primitive_value_args_extractor(function_type)
    assert extractor is not None

    # Test the generated extractor
    container = pycontainer_type(x=1.0, y=2.0)
    args, kwargs = extractor((42, container))
    print(f"{args = }, {kwargs = }")
    assert args == ((42, (1.0, 2.0)),)
    assert kwargs == {}


@pytest.mark.parametrize(
    "pycontainer_type", [cnc.NamedTupleNamedCollection, cnc.DataclassNamedCollection]
)
def test_make_primitive_value_args_extractor_mixed_args(
    pycontainer_type: type[named_collections.CustomNamedCollection],
):
    """Test make_primitive_value_args_extractor with mixed positional and keyword arguments."""

    container_type = tt.from_type_hint(pycontainer_type)
    function_type = ts.FunctionType(
        pos_only_args=[container_type],
        pos_or_kw_args={"b": ts.ScalarType(kind=ts.ScalarKind.FLOAT64)},
        kw_only_args={"c": container_type},
        returns=ts.VoidType(),
    )

    extractor = arguments.make_primitive_value_args_extractor(function_type)
    assert extractor is not None

    # Test the generated extractor
    container1 = pycontainer_type(x=1.0, y=2.0)
    container2 = pycontainer_type(x=3.0, y=4.0)
    args, kwargs = extractor(container1, 3.14, c=container2)
    assert args == ((1.0, 2.0), 3.14)
    assert kwargs == {"c": (3.0, 4.0)}
