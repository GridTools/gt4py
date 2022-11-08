# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import dataclasses
import enum
import sys
import typing

import pytest

from eve import extended_typing as xtyping
from eve import type_definitions as type_def
from eve import type_validation as type_val
from eve.extended_typing import (
    Any,
    Dict,
    Final,
    ForwardRef,
    List,
    Optional,
    Sequence,
    Set,
    SourceTypeAnnotation,
    Tuple,
    Union,
)


VALIDATORS: Final = [type_val.simple_type_validator]
FACTORIES: Final = [type_val.simple_type_validator_factory]


class SampleEnum(enum.Enum):
    FOO = "foo"
    BLA = "bla"


class SampleEmptyClass:
    pass


@dataclasses.dataclass
class SampleDataClass:
    a: int


# Each item should be a tuple like:
#   ( annotation: Any, valid_values: Sequence, wrong_values: Sequence,
#     globalns: Optional[Dict[str, Any]], localns: Optional[Dict[str, Any]] )
SAMPLE_TYPE_DEFINITIONS: List[
    Tuple[Any, Sequence, Sequence, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]
] = [
    (bool, [True, False], [1, "True"], None, None),
    (int, [1, -1], [1.0, "1"], None, None),
    (float, [1.0], [1, "1.0"], None, None),
    (complex, [1.0j, 1 + 2j, 3j], [1, "1.0"], None, None),
    (str, ["", "one"], [1, ("one",)], None, None),
    (complex, [1j], [1, 1.0, "1j"], None, None),
    (bytes, [b"bytes", b""], ["string", ["a"]], None, None),
    (typing.Any, ["any"], tuple(), None, None),
    (
        typing.Literal[1, True],
        [1, True],
        [False],
        None,
        None,
    ),  # float literals are not supported by PEP 586
    (typing.Tuple[int, str], [(3, "three")], [(), (3, 3)], None, None),
    (typing.Tuple[int, ...], [(1, 2, 3), ()], [3, (3, "three")], None, None),
    (typing.List[int], ([1, 2, 3], []), (1, [1.0]), None, None),
    (typing.Set[int], ({1, 2, 3}, set()), (1, [1], (1,), {1: None}), None, None),
    (typing.Dict[int, str], ({}, {3: "three"}), ([(3, "three")], 3, "three", []), None, None),
    (
        type_def.frozendict[int, str],  # type: ignore[misc]
        (
            type_def.frozendict(),
            type_def.frozendict({3: "three"}),
            type_def.frozendict({3: "three", -1: ""}),
        ),
        ({}, {3: "three"}, [(3, "three")], 3, "three", []),
        None,
        None,
    ),
    (typing.Sequence[int], ([1, 2, 3], [], (1, 2, 3), tuple()), (1, [1.0], {1}), None, None),
    (typing.MutableSequence[int], ([1, 2, 3], []), ((1, 2, 3), tuple(), 1, [1.0], {1}), None, None),
    (typing.Set[int], ({1, 2, 3}, set()), (1, [1], (1,), {1: None}), None, None),
    (typing.Union[int, float, str], [1, 3.0, "one"], [[1], [], 1j], None, None),
    (typing.Optional[int], [1, None], [[1], [], 1j], None, None),
    (
        typing.Dict[Union[int, float, str], Union[Tuple[int, Optional[float]], Set[int]]],
        [{1: (2, 3.0)}, {1.0: (2, None)}, {"1": {1, 2}}],
        [{(1, 1.0, "1"): set()}, {1: [1]}, {"1": (1,)}],
        None,
        None,
    ),
    (SampleEnum, [SampleEnum.FOO, SampleEnum.BLA], [SampleEnum, "foo", "bla"], None, None),
    (
        SampleEmptyClass,
        [SampleEmptyClass(), SampleEmptyClass()],
        [object(), "", None, SampleDataClass(1), SampleEmptyClass],
        None,
        None,
    ),
    (
        SampleDataClass,
        [SampleDataClass(1), SampleDataClass(-42)],
        [object(), int(1), "1", SampleDataClass],
        None,
        None,
    ),
    (
        ForwardRef("SampleDataClass"),
        [SampleDataClass(1), SampleDataClass(-42)],
        [object(), int(1), "1", SampleDataClass],
        {"SampleDataClass": SampleDataClass},
        None,
    ),
    (
        ForwardRef("SampleDataClass"),
        [SampleDataClass(1), SampleDataClass(-42)],
        [object(), int(1), "1", SampleDataClass],
        None,
        {"SampleDataClass": SampleDataClass},
    ),
    (
        ForwardRef("typing.List[SampleEmptyClass]"),
        ([], [SampleEmptyClass()], [SampleEmptyClass()] * 5),
        (SampleEmptyClass(), [1], (SampleEmptyClass(),), {1: SampleEmptyClass()}),
        globals(),
        None,
    ),
]

if sys.version_info >= (3, 10):

    @dataclasses.dataclass(slots=True)
    class SampleSlottedDataClass:
        b: float

    SAMPLE_TYPE_DEFINITIONS.append(
        (
            SampleSlottedDataClass,
            [SampleSlottedDataClass(1.0), SampleSlottedDataClass(1)],
            [object(), float(1.2), int(1), "1.2", SampleSlottedDataClass],
            None,
            None,
        )
    )


@pytest.mark.parametrize("validator", VALIDATORS)
@pytest.mark.parametrize(
    ["type_hint", "valid_values", "wrong_values", "globalns", "localns"], SAMPLE_TYPE_DEFINITIONS
)
def test_validators(
    validator: type_val.TypeValidator,
    type_hint: SourceTypeAnnotation,
    valid_values: Sequence,
    wrong_values: Sequence,
    globalns: Optional[Dict[str, Any]],
    localns: Optional[Dict[str, Any]],
):
    for value in valid_values:
        validator(value, type_hint, "<value>", globalns=globalns, localns=localns)

    for value in wrong_values:
        with pytest.raises((TypeError), match="'<value>'"):
            validator(value, type_hint, "<value>", globalns=globalns, localns=localns)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize(
    ["type_hint", "valid_values", "wrong_values", "globalns", "localns"], SAMPLE_TYPE_DEFINITIONS
)
def test_validator_factories(
    factory: type_val.TypeValidatorFactory,
    type_hint: SourceTypeAnnotation,
    valid_values: Sequence,
    wrong_values: Sequence,
    globalns: Optional[Dict[str, Any]],
    localns: Optional[Dict[str, Any]],
):
    validator = factory(type_hint, name="<value>", globalns=globalns, localns=localns)
    for value in valid_values:
        validator(value)

    for value in wrong_values:
        with pytest.raises((TypeError), match="'<value>'"):
            validator(value)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("type_hint", [123, callable, True, "asdfasdf"])
def test_validator_factories_with_invalid_hints(
    factory: type_val.TypeValidatorFactory, type_hint: SourceTypeAnnotation
):
    with pytest.raises(ValueError, match="annotation is not supported"):
        factory(type_hint, name="<value>")


@pytest.mark.parametrize(
    "type_hint",
    [
        int,
        float,
        SampleEmptyClass,
        SampleDataClass,
        SampleEnum,
        List[int],
        Dict[Tuple[int, ...], List[Set[complex]]],
    ],
)
def test_simple_validation_cache(type_hint):
    validator = type_val.simple_type_validator_factory(type_hint, "value")
    assert type_val.simple_type_validator_factory(type_hint, "value") is validator

    assert type_val.simple_type_validator_factory(type_hint, "value_2") is not validator
    assert type_val.simple_type_validator_factory(Optional[float], "value") is not validator
    assert type_val.simple_type_validator_factory(List[float], "value") is not validator

    opt_validator = type_val.simple_type_validator_factory(type_hint, "value", required=False)
    assert opt_validator not in (validator, None)


def test_simple_validation_particularities():
    # strict int
    strict_validator = type_val.simple_type_validator_factory(int, "value", strict_int=True)
    lenient_validator = type_val.simple_type_validator_factory(int, "value", strict_int=False)
    strict_validator(3)
    lenient_validator(3)

    with pytest.raises(TypeError, match="'bool'>"):
        strict_validator(True)
    lenient_validator(True)

    # not supported annotations
    InvalidAnnotation = xtyping.TypeGuard[str]
    assert (
        type_val.simple_type_validator_factory(InvalidAnnotation, "value", required=False) is None
    )

    with pytest.raises(ValueError, match="annotation is not supported"):
        type_val.simple_type_validator_factory(InvalidAnnotation, "value", required=True)

    with pytest.raises(ValueError, match="annotation is not supported"):
        type_val.simple_type_validator_factory(InvalidAnnotation, "value")
