# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
import typing
from frozendict import frozendict

import pytest

from gt4py.eve import (
    extended_typing as xtyping,
    type_validation as type_val,
)
from gt4py.eve.extended_typing import (
    Any,
    Callable,
    Final,
    ForwardRef,
    Optional,
    Sequence,
    SourceTypeAnnotation,
    Union,
)


VALIDATORS: Final[list[Callable]] = [type_val.simple_type_validator]
FACTORIES: Final[list[Callable]] = [type_val.simple_type_validator_factory]


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
SAMPLE_TYPE_DEFINITIONS: list[
    tuple[Any, Sequence, Sequence, Optional[dict[str, Any]], Optional[dict[str, Any]]]
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
    (type[SampleEmptyClass], [SampleEmptyClass], [SampleEmptyClass(), int, 3], None, None),
    (type[SampleDataClass], [SampleDataClass], [SampleDataClass(a=1), str], None, None),
    (type[Any], [SampleEmptyClass, int, str], [3, "int", SampleEmptyClass()], None, None),
    (
        frozendict[int, str],
        (
            frozendict(),
            frozendict({3: "three"}),
            frozendict({3: "three", -1: ""}),
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
        typing.Dict[Union[int, float, str], Union[tuple[int, Optional[float]], set[int]]],
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


# -- PEP 695 type aliases --
type SampleIntAlias = int
type SampleChainedAlias = SampleIntAlias
type SampleListAlias = list[SampleIntAlias]
type SamplePairAlias[T] = tuple[T, T]
type SampleNestedTupleAlias[T] = tuple[T | SampleNestedTupleAlias[T], ...]


SAMPLE_TYPE_DEFINITIONS.extend(
    [
        (SampleIntAlias, [1, -1], [1.0, "1"], None, None),
        (SampleChainedAlias, [1, -1], [1.0, "1"], None, None),
        (SampleListAlias, ([1, 2, 3], []), (1, [1.0]), None, None),
        (list[SampleIntAlias], ([1, 2, 3], []), (1, [1.0]), None, None),
        (Optional[SampleIntAlias], [1, None], ["1"], None, None),
        (SamplePairAlias[int], [(1, 2)], [(1, "2"), (1,)], None, None),
        # Aliases recursing through a container
        (
            SampleNestedTupleAlias[int],
            ((), (1, 2), (1, (2, (3, 4)), ())),
            (1, [1], (1, "2"), (1, [2])),
            None,
            None,
        ),
        (
            SampleNestedTupleAlias[SampleDataClass],
            ((), (SampleDataClass(1),), (SampleDataClass(1), ((SampleDataClass(2),), ()))),
            (SampleDataClass(1), [SampleDataClass(1)], (1,), (SampleDataClass(1), (1,))),
            None,
            None,
        ),
        (
            xtyping.NestedTuple[int],
            ((), (1, 2), (1, (2, (3, 4)), ())),
            (1, [1], (1, "2"), (1, [2])),
            None,
            None,
        ),
        (
            xtyping.NestedList[int],
            ([], [1, 2], [1, [2, [3, 4]], []]),
            (1, (1,), [1, "2"], [1, (2,)]),
            None,
            None,
        ),
        (xtyping.MaybeNestedInTuple[int], (1, (), (1, (2, 3))), ("1", [1], (1, "2")), None, None),
    ]
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
    globalns: Optional[dict[str, Any]],
    localns: Optional[dict[str, Any]],
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
    globalns: Optional[dict[str, Any]],
    localns: Optional[dict[str, Any]],
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
        list[int],
        dict[tuple[int, ...], list[set[complex]]],
    ],
)
def test_simple_validation_cache(type_hint):
    validator = type_val.simple_type_validator_factory(type_hint, "value")
    assert type_val.simple_type_validator_factory(type_hint, "value") is validator

    assert type_val.simple_type_validator_factory(type_hint, "value_2") is not validator
    assert type_val.simple_type_validator_factory(Optional[float], "value") is not validator
    assert type_val.simple_type_validator_factory(list[float], "value") is not validator

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


def test_cyclic_type_alias_is_not_supported():
    # An alias standing for itself directly never reaches an actual annotation, unlike
    # one recursing through a container (see the samples above).
    type RecursiveAlias = RecursiveAlias

    # The reason the alias could not be resolved is kept instead of the generic
    # 'not supported' message, since it is the only thing pointing at the cause.
    with pytest.raises(ValueError, match="'RecursiveAlias' cannot be resolved"):
        type_val.simple_type_validator_factory(RecursiveAlias, "value")

    assert type_val.simple_type_validator_factory(RecursiveAlias, "value", required=False) is None


def test_type_alias_with_undefined_value_propagates_name_error():
    # Alias values are evaluated lazily, so the missing name only shows up here.
    # The error must not be swallowed: 'datamodels' relies on it to defer the
    # creation of the validator.
    type LazyAlias = _defined_later  # noqa: F821 [undefined-name]  # defined below

    with pytest.raises(NameError):
        type_val.simple_type_validator_factory(LazyAlias, "value")

    _defined_later = int

    validator = type_val.simple_type_validator_factory(LazyAlias, "value")
    validator(1)
    with pytest.raises(TypeError):
        validator("1")
