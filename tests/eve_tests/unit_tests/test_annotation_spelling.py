# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Conformance matrix: equivalent annotation spellings must behave identically.

Python offers several spellings for the same type -- ``Optional[int]`` and
``int | None``, ``List[int]`` and ``list[int]``, ``X: TypeAlias = ...`` and
``type X = ...``, a live object and the string naming it. ``eve`` builds field
validators and converters from live annotation objects and dispatches on their
*identity*, at several independent sites (the "funnels" below).

A funnel that has not been taught a spelling does not raise: it falls through to
its no-match branch and silently produces a wrong validator, a wrong converter,
or a wrong answer. Every test here therefore compares two spellings of the *same*
logical type against each other rather than against a hard-coded expectation --
the property being asserted is agreement, so a funnel that is uniformly wrong for
both spellings is caught elsewhere, and one that is wrong for only the new
spelling is caught here.

Adding a funnel to ``eve`` means adding an observer to ``FUNNELS``.

Note on annotation-evaluation regimes: the ``str`` entries in ``SPELLINGS`` cover
PEP 563 (``from __future__ import annotations``), which stores annotations as
strings. PEP 649 (the Python 3.14 default) evaluates them lazily but hands back
live objects, so it is covered by the non-string entries when the suite runs on
3.14.
"""

# NOTE: this module deliberately does *not* use 'from __future__ import annotations'.
# The observers below build datamodels whose field annotation is a local variable
# ('value: annotation'). Under PEP 563 that is stored as the literal string
# "annotation" instead of the type under test, and every assertion below silently
# compares nothing. The PEP 563 regime is covered explicitly by
# 'test_string_annotations_agree_with_live_objects' instead.

import collections.abc
import dataclasses
import sys
import types
import typing
from typing import (  # noqa: UP035 [deprecated-import] deliberately exercising the legacy spellings
    Annotated,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import pytest

from gt4py.eve import datamodels, exceptions, extra_typing, type_validation as type_val

# Imported directly on purpose: reaching these two through a datamodel is not enough.
# 'get_partial_type_hints' strips 'Annotated' before a field annotation gets this far,
# so a datamodel-mediated observer compares 'X' against 'X' and cannot see a missing
# 'Annotated' branch in either function.
from gt4py.eve.datamodels.core import _is_strictly_immutable_type, _make_type_converter


# -- The spellings under test --


@dataclasses.dataclass(frozen=True)
class Spelling:
    """Two spellings of one logical type, plus values to probe the funnels with."""

    id: str
    legacy: Any
    modern: Any
    probes: tuple[Any, ...]


#: Probe set reused by the container cases. It has to contain at least one value that
#: each container spelling *accepts*: a row whose every probe is rejected by both sides
#: cannot tell a correct validator from one that rejects everything -- which is exactly
#: the failure this file exists to catch.
_CONTAINER_PROBES: tuple[Any, ...] = (
    None,
    0,
    "x",
    [1],
    (1,),
    (1, "x"),
    ([1],),
    {1},
    frozenset({1}),
    {"a": 1},
)

SPELLINGS: list[Spelling] = [
    Spelling("optional", Optional[int], int | None, (None, 3, True, "3", 1.5, [])),
    Spelling("optional-reversed", Optional[int], Union[None, int], (None, 3, "3", 1.5)),
    Spelling("union", Union[int, str], int | str, (1, "a", None, 1.5, [])),
    Spelling("union-3", Union[int, str, float], int | str | float, (1, "a", 1.5, None)),
    Spelling("list", List[int], list[int], _CONTAINER_PROBES),
    Spelling("dict", Dict[str, int], dict[str, int], _CONTAINER_PROBES),
    Spelling("set", Set[int], set[int], _CONTAINER_PROBES),
    Spelling("frozenset", FrozenSet[int], frozenset[int], _CONTAINER_PROBES),
    Spelling("tuple-var", Tuple[int, ...], tuple[int, ...], _CONTAINER_PROBES),
    Spelling("tuple-fixed", Tuple[int, str], tuple[int, str], _CONTAINER_PROBES),
    Spelling("type", Type[int], type[int], (int, bool, str, "int", 3, None)),
    # Nesting: the new spelling has to survive being an argument of another generic.
    Spelling("optional-list", Optional[List[int]], list[int] | None, _CONTAINER_PROBES),
    Spelling("list-optional", List[Optional[int]], list[int | None], _CONTAINER_PROBES),
    Spelling("dict-union", Dict[str, Union[int, str]], dict[str, int | str], _CONTAINER_PROBES),
    Spelling("tuple-of-list", Tuple[List[int], ...], tuple[list[int], ...], _CONTAINER_PROBES),
    # 'typing.Callable' vs 'collections.abc.Callable' is a 'UP035' target of the sweep.
    Spelling(
        "callable",
        Callable[[int], int],
        collections.abc.Callable[[int], int],
        (None, 0, "x", len, lambda x: x),
    ),
    Spelling("literal", Literal[1, 2], Literal[1] | Literal[2], (1, 2, 3, None, "1")),
]

#: 'X | Y' is a runtime value, so it can be stored in a plain variable; the legacy
#: and modern spellings of an *alias* are compared the same way as the types above.
LegacyAlias: typing.TypeAlias = Optional[int]

type ModernAlias = Optional[int]
type ModernAliasPEP604 = int | None
type AliasedInt = int

ALIAS_SPELLINGS: list[Spelling] = [
    Spelling("alias-legacy-vs-695", LegacyAlias, ModernAlias, (None, 3, "3", 1.5)),
    Spelling("alias-695-604", ModernAlias, ModernAliasPEP604, (None, 3, "3", 1.5)),
]


#: `Annotated[X, ...]` carries metadata that is not part of the type: every funnel
#: must see straight through it to `X`. On 3.12 `typing.Annotated` is a class, so a
#: funnel dispatching on `isinstance(origin, type)` claims it by accident.
TRANSPARENT_WRAPPERS: list[Spelling] = [
    Spelling("annotated-int", int, Annotated[int, "meta"], (3, "3", None, 1.5)),
    Spelling(
        "annotated-optional", Optional[int], Annotated[Optional[int], "meta"], (None, 3, "3", 1.5)
    ),
    Spelling("annotated-list", list[int], Annotated[list[int], "meta"], _CONTAINER_PROBES),
    Spelling(
        "annotated-tuple", tuple[int, ...], Annotated[tuple[int, ...], "meta"], _CONTAINER_PROBES
    ),
    # Nested inside 'type[...]': the metadata is not stripped by the whole-annotation
    # pass, so the 'type[X]' branch has to strip it itself.
    Spelling(
        "annotated-inside-type",
        type[int],
        type[Annotated[int, "meta"]],
        (int, bool, str, 3),
    ),
    # Pins the order of the two rewrites inside the 'type[...]' branch: the metadata
    # has to come off before the union is normalized, or the union is never seen.
    Spelling(
        "annotated-inside-type-union",
        type[int | str],
        type[Annotated[int | str, "meta"]],
        (int, str, bool, float, 3),
    ),
    # 'Annotated' wrapping a PEP 695 alias: resolving the alias once, before the
    # metadata is stripped, leaves the alias unresolved underneath it.
    Spelling(
        "annotated-inside-type-alias",
        type[AliasedInt],
        type[Annotated[AliasedInt, "meta"]],
        (int, bool, str, 3),
    ),
]


# -- The funnels --
#
# Each observer maps an annotation to a value that is comparable with '=='. They
# must never raise: a funnel blowing up for one spelling and not the other is
# exactly the asymmetry under test, so exceptions are captured and compared too.


def _outcome(fn: typing.Callable[[], Any]) -> Any:
    """Run `fn`, returning its result or a marker naming the exception type.

    Used for *values*, which must compare with '=='.
    """
    try:
        result = fn()
    # A blind `except` is the point here: the exception type *is* the observation.
    except Exception as error:
        return ("error", type(error).__name__)
    # The type is part of the observation: 'True == 1' and 'True' is in the probe set,
    # while eve's 'strict_int' handling deliberately tells bool and int apart.
    return ("value", type(result), result)


def _build(fn: typing.Callable[[], Any]) -> tuple[Any, Any]:
    """Run a class-building `fn`, returning `(cls, None)` or `(None, observation)`.

    Kept separate from `_outcome` because a freshly built class has a unique
    identity: returning it as the observation would make every success/success
    comparison fail, and folding it into a marker would hide the class from the
    caller that still needs to instantiate it.
    """
    try:
        return fn(), None
    # A blind `except` is the point here: the exception type *is* the observation.
    except Exception as error:
        return None, ("error", type(error).__name__)


def _probe_all(model: type, probes: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(_outcome(lambda p=p: model(value=p).value) for p in probes)


def observe_validator(annotation: Any, probes: tuple[Any, ...]) -> Any:
    return (
        "validator",
        tuple(_outcome(lambda p=p: type_val.simple_type_validator(p, annotation)) for p in probes),
    )


def observe_converter(annotation: Any, probes: tuple[Any, ...]) -> Any:
    """Coercion requested through `field(converter="coerce")`."""

    def build() -> type:
        @datamodels.datamodel
        class Model:
            value: annotation = datamodels.field(converter="coerce")  # type: ignore[valid-type]  # a variable annotation is the point of the test

        return Model

    model, failure = _build(build)
    return ("converter", failure if model is None else _probe_all(model, probes))


def observe_coerced_annotation(annotation: Any, probes: tuple[Any, ...]) -> Any:
    """The other route to a converter: the `Coerced[...]` marker rather than `field()`."""

    def build() -> type:
        @datamodels.datamodel
        class Model:
            value: datamodels.Coerced[annotation]  # type: ignore[valid-type]  # idem

        return Model

    model, failure = _build(build)
    return ("coerced", failure if model is None else _probe_all(model, probes))


def observe_datamodel(annotation: Any, probes: tuple[Any, ...]) -> Any:
    """End-to-end: field type validation on a plain (non-coercing) datamodel."""

    def build() -> type:
        @datamodels.datamodel
        class Model:
            value: annotation  # type: ignore[valid-type]  # idem

        return Model

    model, failure = _build(build)
    return ("datamodel", failure if model is None else _probe_all(model, probes))


def observe_represented_types(annotation: Any, probes: tuple[Any, ...]) -> Any:
    # Compared as a set: the result is meant to be handed to 'isinstance()', and
    # 'typing' preserves the order the union members were written in, so
    # 'Optional[int]' and 'Union[None, int]' legitimately differ in order alone.
    outcome = _outcome(lambda: extra_typing.get_represented_types(annotation))
    if outcome[0] == "value":
        return ("represented", "value", frozenset(outcome[2]))
    return ("represented", *outcome)


def observe_strict_frozen(annotation: Any, probes: tuple[Any, ...]) -> Any:
    """Immutability analysis, reachable only through `frozen='strict'`."""

    def build() -> type:
        @datamodels.datamodel(frozen="strict")
        class Model:
            value: annotation  # type: ignore[valid-type]  # idem

        return Model

    model, failure = _build(build)
    return ("strict-frozen", ("accepted",) if model is not None else failure)


def observe_type_converter(annotation: Any, probes: tuple[Any, ...]) -> Any:
    """`_make_type_converter` called directly, bypassing the datamodel."""
    converter, failure = _build(lambda: _make_type_converter(annotation, "<field>"))
    if converter is None:
        return ("type-converter", failure)
    return ("type-converter", tuple(_outcome(lambda p=p: converter(p)) for p in probes))


def observe_immutability(annotation: Any, probes: tuple[Any, ...]) -> Any:
    """`_is_strictly_immutable_type` called directly, bypassing the datamodel."""
    return ("immutability", _outcome(lambda: _is_strictly_immutable_type(annotation)))


FUNNELS: list[typing.Callable[[Any, tuple[Any, ...]], Any]] = [
    observe_type_converter,
    observe_immutability,
    observe_validator,
    observe_converter,
    observe_coerced_annotation,
    observe_datamodel,
    observe_represented_types,
    observe_strict_frozen,
]


# -- The matrix --


@pytest.mark.parametrize("funnel", FUNNELS, ids=lambda f: f.__name__.removeprefix("observe_"))
@pytest.mark.parametrize("spelling", SPELLINGS, ids=lambda s: s.id)
def test_funnels_agree_across_spellings(spelling: Spelling, funnel: Any) -> None:
    """Legacy and modern spellings of one type must be indistinguishable to `eve`."""
    assert funnel(spelling.legacy, spelling.probes) == funnel(spelling.modern, spelling.probes)


@pytest.mark.parametrize("funnel", FUNNELS, ids=lambda f: f.__name__.removeprefix("observe_"))
@pytest.mark.parametrize("spelling", ALIAS_SPELLINGS, ids=lambda s: s.id)
def test_funnels_agree_across_alias_spellings(spelling: Spelling, funnel: Any) -> None:
    """`TypeAlias` and PEP 695 `type X = ...` must resolve to the same behaviour."""
    assert funnel(spelling.legacy, spelling.probes) == funnel(spelling.modern, spelling.probes)


#: Rows allowed to round-trip to the very same object on *both* sides, which makes
#: their cells compare a value with itself. 'typing' caches parametrizations, so
#: 'eval_forward_ref(str(x))' can hand back 'x'; 'literal' is the only row where that
#: happens on both sides, and only before 3.14.
#:
#: Asserted as a subset, not an equality: the cache is an LRU, so a parallel run can
#: evict an entry and stop a row being vacuous. Only "no *other* row is vacuous" is
#: stable.
_IDENTITY_ROUND_TRIP_ROWS = frozenset({"literal"})


def test_string_annotation_rows_are_not_silently_vacuous() -> None:
    """Keep the set of vacuous string round-trips from growing.

    Without this, adding a spelling whose `str()` re-evaluates to the same cached
    object would add eight green -- and entirely vacuous -- cells to
    `test_string_annotations_agree_with_live_objects`.
    """
    vacuous = {
        spelling.id
        for spelling in SPELLINGS
        if all(
            extra_typing.eval_forward_ref(_as_source(annotation), globalns=globals()) is annotation
            for annotation in (spelling.legacy, spelling.modern)
        )
    }
    assert vacuous <= _IDENTITY_ROUND_TRIP_ROWS, (
        f"Rows {sorted(vacuous - _IDENTITY_ROUND_TRIP_ROWS)} compare a value with itself "
        "on both sides, so their cells in "
        "'test_string_annotations_agree_with_live_objects' assert nothing."
    )


@pytest.mark.parametrize("funnel", FUNNELS, ids=lambda f: f.__name__.removeprefix("observe_"))
@pytest.mark.parametrize("spelling", SPELLINGS, ids=lambda s: s.id)
def test_string_annotations_agree_with_live_objects(spelling: Spelling, funnel: Any) -> None:
    """A PEP 563 string annotation must be indistinguishable from the object it names.

    This is the regime `from __future__ import annotations` puts a module in, and the
    one `get_partial_type_hints` has to reconstruct. Comparing only the represented
    types would not say much -- that collapses every container to its origin -- so the
    resolved annotation is pushed through the whole funnel list.

    A 'typing.'-spelled side may round-trip to the identical cached object, in which
    case that side asserts nothing; the modern side of the same row still exercises a
    real reconstruction. `test_string_annotation_rows_are_not_silently_vacuous` pins
    the rows where *both* sides are identities.
    """
    for annotation in (spelling.legacy, spelling.modern):
        # PEP 563 resolves a string annotation against the *defining module's*
        # globals, so that is what gets passed here -- this module imports 'typing'
        # and 'collections.abc', exactly as a module writing those annotations would.
        resolved = extra_typing.eval_forward_ref(_as_source(annotation), globalns=globals())
        assert funnel(resolved, spelling.probes) == funnel(annotation, spelling.probes)


def _as_source(annotation: Any) -> str:
    """Spell an annotation the way it would appear in source code.

    `str(list[int])` is already source. `str(typing.List[int])` keeps the module
    prefix, which is what exercises the deprecated-alias normalization; the same
    goes for `str(collections.abc.Callable[...])`, which is why the caller has to
    supply a namespace where those module names are bound.

    This is a faithful spelling, not the original source text: `typing` normalizes
    some forms on the way in, so `str(Union[None, int])` is `'typing.Optional[int]'`
    and the written argument order is not recoverable.
    """
    return str(annotation)


# -- Regression test for the specific defect the matrix was written to catch --


def test_pep604_union_builds_the_same_converter_as_optional() -> None:
    """'int | None' must coerce exactly like 'Optional[int]'.

    'get_origin(int | None)' is 'types.UnionType', not 'typing.Union', so a funnel
    comparing against 'typing.Union' alone skips the Optional branch and falls
    through to a wrong one instead of raising.
    """

    @datamodels.datamodel
    class Legacy:
        value: Optional[int] = datamodels.field(converter="coerce")

    @datamodels.datamodel
    class Modern:
        value: int | None = datamodels.field(converter="coerce")

    for model in (Legacy, Modern):
        assert model(value=None).value is None
        assert model(value=3).value == 3
        assert model(value="3").value == 3


@pytest.mark.parametrize("funnel", FUNNELS, ids=lambda f: f.__name__.removeprefix("observe_"))
@pytest.mark.parametrize("spelling", TRANSPARENT_WRAPPERS, ids=lambda s: s.id)
def test_funnels_see_through_annotated(spelling: Spelling, funnel: Any) -> None:
    """`Annotated[X, meta]` must be indistinguishable from `X` to every funnel."""
    assert funnel(spelling.legacy, spelling.probes) == funnel(spelling.modern, spelling.probes)


#: Funnels observed *through* a datamodel field, and those called directly.
#:
#: `get_partial_type_hints` strips `Annotated` recursively, so a datamodel-mediated
#: funnel never sees the metadata: for every `TRANSPARENT_WRAPPERS` row it compares
#: `X` with `X` and passes whatever the funnel does. Those rows are really tested by
#: the directly-called funnels, which is why those observers exist.
#:
#: Both sets are pinned so that adding a funnel is a deliberate act. Without this, a
#: new datamodel-mediated observer would quietly join the tautological set while
#: appearing to add four green cells per wrapper row.
_DATAMODEL_MEDIATED_FUNNELS = frozenset(
    {"converter", "coerced_annotation", "datamodel", "strict_frozen"}
)
_DIRECTLY_CALLED_FUNNELS = frozenset(
    {"validator", "represented_types", "type_converter", "immutability"}
)


def test_every_funnel_is_classified() -> None:
    """Adding a funnel must force a decision about whether it can see `Annotated`."""
    observed = {funnel.__name__.removeprefix("observe_") for funnel in FUNNELS}
    assert observed == _DATAMODEL_MEDIATED_FUNNELS | _DIRECTLY_CALLED_FUNNELS


def test_get_partial_type_hints_strips_annotated_recursively() -> None:
    """The reason the directly-called observers exist.

    This is what makes the `TRANSPARENT_WRAPPERS` rows vacuous for every funnel
    reached through a datamodel: the metadata is gone before the funnel runs, at any
    depth. If this ever stopped being true, the compensation would no longer be
    needed -- and, more to the point, the funnels would start seeing `Annotated`
    where they currently never do.
    """

    class _Holder:
        plain: Annotated[int, "meta"]
        nested: list[Annotated[int, "meta"]]
        in_value: dict[str, Annotated[int, "meta"]]

    hints = extra_typing.get_partial_type_hints(_Holder)

    assert hints["plain"] is int
    assert hints["nested"] == list[int]
    assert hints["in_value"] == dict[str, int]


def test_represented_types_of_literal_are_the_types_of_its_values() -> None:
    """`Literal` arguments are values, so their types cannot be found by recursion.

    Recursing into them matches no branch and yields an empty tuple, which callers
    turn into an `isinstance()` argument -- making every check against it `False`.
    """
    assert extra_typing.get_represented_types(Literal[1, 2]) == (int,)
    assert extra_typing.get_represented_types(Literal["a", 1]) == (str, int)
    assert extra_typing.get_represented_types(Union[Literal["a"], int]) == (str, int)


def test_represented_types_sees_through_annotated() -> None:
    """The metadata of an `Annotated` is not a represented type, and neither is the form."""
    assert extra_typing.get_represented_types(Annotated[int, "meta"]) == (int,)
    assert extra_typing.get_represented_types(Annotated[Optional[int], "meta"]) == (int, type(None))


@pytest.mark.parametrize(
    "spelling",
    [*SPELLINGS, *ALIAS_SPELLINGS, *TRANSPARENT_WRAPPERS],
    ids=lambda s: s.id,
)
def test_spelling_pairs_are_distinct(spelling: Spelling) -> None:
    """Guard against a row silently degenerating into `f(x) == f(x)`.

    Every assertion in this file compares two spellings of one type. If a row's two
    entries are the same object, the comparison is a tautology that passes whatever
    the funnels do -- and nothing else in the file would notice.

    Python 3.14 makes `typing.Union` *be* `types.UnionType`, so on that version the
    union rows legitimately collapse: the bug they guard against cannot exist there.
    That case is allowed through, but only there.
    """
    if spelling.legacy is spelling.modern:
        unified_by_interpreter = sys.version_info >= (3, 14) and typing.get_origin(
            spelling.legacy
        ) in (typing.Union, types.UnionType)
        assert unified_by_interpreter, (
            f"Spelling row '{spelling.id}' compares an object with itself, so it asserts nothing."
        )


def test_infer_type_passes_annotations_through_unchanged() -> None:
    """`infer_type` documents annotations as valid input, and must not type() them.

    Only the PEP 585 spelling used to be recognized; every other one fell through to
    the `type(value)` fallback and came back as the CPython-internal class that
    implements it (`typing._GenericAlias`, `types.UnionType`, ...).
    """
    for annotation in (
        list[int],
        List[int],
        Dict[str, int],
        Optional[int],
        int | None,
        Union[int, str],
        int | str,
        Literal[1, 2],
        Annotated[int, "meta"],
        Callable[[int], int],
        ModernAlias,
    ):
        assert extra_typing.infer_type(annotation) is annotation


def test_infer_type_still_infers_the_type_of_runtime_values() -> None:
    """The pass-through above must not swallow ordinary values.

    `get_origin` is part of the test for "is this an annotation", so anything it
    reports an origin for would be handed back as-is rather than described.
    """

    class Generic_(typing.Generic[typing.TypeVar("_T")]): ...

    assert extra_typing.infer_type(3) is int
    assert extra_typing.infer_type("x") is str
    assert extra_typing.infer_type(None) is type(None)
    assert extra_typing.infer_type([1, 2]) == list[int]
    assert extra_typing.infer_type({"a": 1}) == dict[str, int]
    assert extra_typing.infer_type(Generic_()) is Generic_


# -- Regressions from unwrapping transparent wrappers --

type _SelfWrappingAlias = Annotated[_SelfWrappingAlias, "meta"]
type _AliasOfInt = Annotated[int, "inner"]
type _AliasOfAliasOfInt = Annotated[_AliasOfInt, "outer"]


def test_self_referential_alias_is_reported_not_recursed() -> None:
    """An alias that resolves only to itself must fail loudly, while building.

    `Annotated` carries no structure of its own, so `type R = Annotated[R, ...]` hands
    the recursion nothing to descend through. The cycle-breaker for genuine recursive
    aliases makes this worse rather than better if left alone: the placeholder becomes
    its own validator, the annotation builds cleanly, and the stack runs out on the
    first value checked.
    """
    with pytest.raises(exceptions.EveValueError, match="resolves only to itself"):
        type_val.simple_type_validator(1, _SelfWrappingAlias)


type _MutuallyWrappingA = Annotated[_MutuallyWrappingB, "a"]
type _MutuallyWrappingB = Annotated[_MutuallyWrappingA, "b"]


@pytest.mark.parametrize("alias", [_SelfWrappingAlias, _MutuallyWrappingA], ids=["self", "mutual"])
@pytest.mark.parametrize(
    "funnel",
    [
        extra_typing.get_represented_types,
        _is_strictly_immutable_type,
        lambda annotation: _make_type_converter(annotation, "<field>"),
    ],
    ids=["represented_types", "immutability", "type_converter"],
)
def test_alias_annotated_cycles_terminate_in_every_funnel(funnel: Any, alias: Any) -> None:
    """A cycle of alias and `Annotated` layers must be reported, not recursed into.

    Neither wrapper carries structure the recursion can descend through, so a funnel
    resolving the alias in one branch and stripping the metadata in another walks the
    same two annotations forever. The stack then runs out at whichever call site is
    deepest, so the observed failure is not even stable.
    """
    try:
        result = funnel(alias)
    except TypeError as error:
        # Specifically the cycle being recognized. Walking the layers until a depth
        # bound trips, or until the interpreter stack runs out, also raises 'TypeError'
        # -- 'eval_type_alias' converts the 'RecursionError' into one -- so a looser
        # assertion here passes against the very behaviour this test exists to catch.
        assert "(recursive definition)" in str(error), str(error)
    else:
        # Reporting the annotation as unresolvable is also a valid answer, as long as
        # it is an answer: `_is_strictly_immutable_type` proves nothing about an
        # annotation it cannot resolve, and says so with `False`.
        assert result in (False, ())


def test_normalize_union_returns_identical_objects_for_normalized_input() -> None:
    """The ``is`` contract must not depend on the interpreter.

    On 3.14 `typing.Union` *is* `types.UnionType`, so an `isinstance` test alone
    matches annotations that are already normalized and rebuilds them into equal but
    distinct objects. A caller detecting the no-op with `is` -- which the docstring
    invites -- would then see a spelling change that did not happen.
    """
    for annotation in (Optional[int], Union[int, str], int, None, list[int]):
        assert extra_typing.normalize_union(annotation) is annotation

    # Genuine PEP 604 unions are rewritten before 3.14, and are already 'typing.Union'
    # from 3.14 on, so this is the one row whose identity legitimately varies.
    assert extra_typing.normalize_union(int | None) == Optional[int]


def test_genuine_recursive_aliases_still_build() -> None:
    """The guard above must not catch an alias with real structure between the two
    occurrences."""

    type Tree = list[Tree]

    type_val.simple_type_validator([[], [[]]], Tree)
    with pytest.raises(TypeError):
        type_val.simple_type_validator("not-a-list", Tree)


def test_type_arg_unwraps_alias_and_annotated_to_a_fixpoint() -> None:
    """`type[X]` must reach `X` however many alias/`Annotated` layers wrap it.

    Resolving once and stripping once covers `type[Annotated[Alias, ...]]` but not
    `Alias -> Annotated -> Alias -> Annotated -> int`, which then falls through to the
    loose "is any class at all" check and silently accepts everything.
    """
    for annotation in (type[int], type[_AliasOfInt], type[_AliasOfAliasOfInt]):
        type_val.simple_type_validator(int, annotation)
        with pytest.raises(TypeError):
            type_val.simple_type_validator(str, annotation)
