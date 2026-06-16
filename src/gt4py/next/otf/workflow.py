# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import dataclasses
import functools
import typing
from typing import Any, Callable, Generic, Protocol, TypeVar

from typing_extensions import Self

from gt4py.eve.extended_typing import OpaqueMutableMapping
from gt4py.next import config, fingerprinting


StartT = TypeVar("StartT")
StartT_contra = TypeVar("StartT_contra", contravariant=True)
EndT = TypeVar("EndT")
EndT_co = TypeVar("EndT_co", covariant=True)
NewEndT = TypeVar("NewEndT")
IntermediateT = TypeVar("IntermediateT")
HashT = TypeVar("HashT")
DataT = TypeVar("DataT")
ArgT = TypeVar("ArgT")


def make_step(function: Workflow[StartT, EndT]) -> ChainableWorkflowMixin[StartT, EndT]:
    """
    Wrap a function in the workflow step convenience wrapper.

    Examples:
    ---------
    >>> @make_step
    ... def times_two(x: int) -> int:
    ...     return x * 2

    >>> def stringify(x: int) -> str:
    ...     return str(x)

    >>> # create a workflow int -> int -> str
    >>> times_two.chain(stringify)(3)
    '6'
    """
    return StepSequence.start(function)


@typing.runtime_checkable
class Workflow(Protocol[StartT_contra, EndT_co]):
    """
    Workflow protocol.

    Anything that implements this interface can be a workflow of one or more steps.
    - callable
    - take a single input argument
    """

    def __call__(self, inp: StartT_contra) -> EndT_co: ...


class ReplaceEnabledWorkflowMixin(Workflow[StartT_contra, EndT_co], Protocol):
    """
    Subworkflow replacement mixin.

    Any subclass MUST be a dataclass for `.replace` to work
    """

    def replace(self, **kwargs: Any) -> Self:
        """
        Build a new instance with replaced substeps.

        Raises:
            TypeError: If `self` is not a dataclass.
        """
        if not dataclasses.is_dataclass(self):
            raise TypeError(f"'{self.__class__}' is not a dataclass.")
        assert not isinstance(self, type)
        return dataclasses.replace(self, **kwargs)


class ChainableWorkflowMixin(Workflow[StartT, EndT_co], Protocol[StartT, EndT_co]):
    def chain(
        self, next_step: Workflow[EndT_co, NewEndT]
    ) -> ChainableWorkflowMixin[StartT, NewEndT]:
        return make_step(self).chain(next_step)


@dataclasses.dataclass(frozen=True)
class NamedStepSequence(
    ChainableWorkflowMixin[StartT, EndT],
    ReplaceEnabledWorkflowMixin[StartT, EndT],
):
    """
    Workflow with linear succession of named steps.

    Examples:
    ---------
    >>> import dataclasses

    >>> def parse(x: str) -> int:
    ...     return int(x)

    >>> def plus_half(x: int) -> float:
    ...     return x + 0.5

    >>> def stringify(x: float) -> str:
    ...     return str(x)

    >>> @dataclasses.dataclass(frozen=True)
    ... class ParseOpPrint(NamedStepSequence[str, str]):
    ...     parse: Workflow[str, int]
    ...     op: Workflow[int, float]
    ...     print: Workflow[float, str]

    >>> pop = ParseOpPrint(parse=parse, op=plus_half, print=stringify)

    >>> pop.step_order
    ['parse', 'op', 'print']

    >>> pop(73)
    '73.5'

    >>> def plus_tenth(x: int) -> float:
    ...     return x + 0.1


    >>> pop.replace(op=plus_tenth)(73)
    '73.1'
    """

    def __call__(self, inp: StartT) -> EndT:
        """Compose the steps in the order defined in the `.step_order` class attribute."""
        step_result: Any = inp
        for step_name in self.step_order:
            step_result = getattr(self, step_name)(step_result)
        return step_result

    @functools.cached_property
    def step_order(self) -> list[str]:
        """
        Read step order from class definition by default.

        Only attributes who are type hinted to be of a type that
        conforms to the Workflow protocol are considered steps.
        """
        step_names: list[str] = []
        annotations = typing.get_type_hints(self.__class__)
        for field in dataclasses.fields(self):
            field_type = annotations[field.name]
            field_type = typing.get_origin(field_type) or field_type
            if issubclass(field_type, Workflow):
                step_names.append(field.name)
        return step_names


@dataclasses.dataclass(frozen=True)
class MultiWorkflow(
    ChainableWorkflowMixin[StartT, EndT],
    ReplaceEnabledWorkflowMixin[StartT, EndT],
):
    """A flexible workflow, where the sequence of steps depends on the input type."""

    def __call__(self, inp: StartT) -> EndT:
        step_result: Any = inp
        for step_name in self.step_order(inp):
            step_result = getattr(self, step_name)(step_result)
        return step_result

    @abc.abstractmethod
    def step_order(self, inp: StartT) -> list[str]:
        pass


@dataclasses.dataclass(frozen=True)
class StepSequence(
    ChainableWorkflowMixin[StartT, EndT],
):
    """
    Composable workflow of single input callables.

    Examples:
    ---------
    >>> def plus_one(x: int) -> int:
    ...     return x + 1

    >>> def plus_half(x: int) -> float:
    ...     return x + 0.5

    >>> def stringify(x: float) -> str:
    ...     return str(x)

    >>> StepSequence.start(plus_one).chain(plus_half).chain(stringify)(73)
    '74.5'

    """

    steps: tuple[Workflow[Any, Any], ...]

    def __call__(self, inp: StartT) -> EndT:
        step_result: Any = inp
        for step in self.steps:
            step_result = step(step_result)
        return step_result

    def chain(self, next_step: Workflow[EndT, NewEndT]) -> ChainableWorkflowMixin[StartT, NewEndT]:
        return typing.cast(
            ChainableWorkflowMixin[StartT, NewEndT],
            self.__class__((*self.steps, next_step)),
        )

    @classmethod
    def start(cls, first_step: Workflow[StartT, EndT]) -> ChainableWorkflowMixin[StartT, EndT]:
        return cls((first_step,))


@dataclasses.dataclass(frozen=True)
class CachedStep(
    ChainableWorkflowMixin[StartT, EndT],
    ReplaceEnabledWorkflowMixin[StartT, EndT],
    Generic[StartT, EndT, HashT],
):
    """
    Cached workflow of single input callables.

    The cache key combines a fingerprint of the workflow state (so that changes
    to the step invalidate the cache) with the value returned by
    ``input_fingerprinter`` for the given input. The caller selects both
    fingerprinters explicitly:

    - ``input_fingerprinter`` fingerprints each input value.
    - ``step_fingerprinter`` fingerprints the workflow state and combines the
      two halves into the final key. Its choice fixes the cache's durability.
      The default, :data:`~gt4py.next.fingerprinting.session_fingerprinter`, suits an
      in-memory cache (a plain ``dict``, the default): it lives within a single
      process and so tolerates non-importable callables in the step graph
      (lambdas, closures, dynamically-created steps) by hashing them
      structurally. A persistent cache (e.g. ``FileCache``) instead requires
      :data:`~gt4py.next.fingerprinting.stable_fingerprinter`, so that every referenced
      function is importable by qualified name and the on-disk keys stay
      reproducible across processes.

    Because the ``step_fingerprinter`` must match the cache's durability, prefer
    the :meth:`in_memory` and :meth:`persistent` constructors, which pair the
    cache with the matching fingerprinter for you. Use the raw constructor only
    when you need a non-default fingerprinter combination.

    Examples:
    ---------
    >>> cached_step = CachedStep.in_memory(step=tuple, input_fingerprinter=lambda x: x)

    The result of the first invocation is computed and stored in the cache:
    >>> result = cached_step([1, 2, 3])
    >>> result
    (1, 2, 3)

    The next invocation for the same argument returns the cached object without
    recomputing it:
    >>> cached_step([1, 2, 3]) is result
    True

    A different argument is computed and cached separately:
    >>> cached_step([4, 5]) is result
    False
    """

    step: Workflow[StartT, EndT]
    input_fingerprinter: Callable[[StartT], HashT] = dataclasses.field(
        metadata=fingerprinting.gt4py_metadata(fingerprint=False)
    )
    step_fingerprinter: fingerprinting.Fingerprinter = dataclasses.field(
        default=fingerprinting.session_fingerprinter,
        metadata=fingerprinting.gt4py_metadata(fingerprint=False),
    )
    cache: OpaqueMutableMapping[str, EndT] = dataclasses.field(
        repr=False, default_factory=dict, metadata=fingerprinting.gt4py_metadata(fingerprint=False)
    )

    def __call__(self, inp: StartT) -> EndT:
        """Run the step only if the input is not cached, else return from cache."""
        hash_ = self.cache_key(inp)
        try:
            result = self.cache[hash_]
        except KeyError:
            result = self.cache[hash_] = self.step(inp)
        return result

    @functools.cached_property
    def _step_fingerprint(self) -> str:
        # The step and the build-cache version are immutable, so their
        # (potentially expensive) fingerprint is computed once per instance
        # instead of on every cache lookup.
        return self.step_fingerprinter((config.BUILD_CACHE_VERSION_ID, self))

    def cache_key(self, inp: StartT) -> str:
        return self.step_fingerprinter((self._step_fingerprint, self.input_fingerprinter(inp)))

    @classmethod
    def in_memory(
        cls,
        step: Workflow[StartT, EndT],
        input_fingerprinter: Callable[[StartT], HashT],
        cache: OpaqueMutableMapping[str, EndT] | None = None,
    ) -> CachedStep[StartT, EndT, HashT]:
        """
        Build a single-process cache, pairing an in-memory store with the session fingerprinter.

        The session fingerprinter tolerates non-importable steps (lambdas,
        closures, dynamically-created steps) by hashing them structurally, which
        is valid because the cache lives and dies within a single process.
        Defaults to a fresh ``dict``.
        """
        return cls(
            step=step,
            input_fingerprinter=input_fingerprinter,
            step_fingerprinter=fingerprinting.session_fingerprinter,
            cache={} if cache is None else cache,
        )

    @classmethod
    def persistent(
        cls,
        step: Workflow[StartT, EndT],
        input_fingerprinter: Callable[[StartT], HashT],
        cache: OpaqueMutableMapping[str, EndT],
    ) -> CachedStep[StartT, EndT, HashT]:
        """
        Build a cross-process cache, pairing a persistent store with the stable fingerprinter.

        The stable fingerprinter requires every referenced function to be
        importable by qualified name, so the on-disk keys stay reproducible
        across processes: a non-importable step raises a ``TypeError`` instead
        of risking a non-reproducible key and stale-binary reuse.
        """
        return cls(
            step=step,
            input_fingerprinter=input_fingerprinter,
            step_fingerprinter=fingerprinting.stable_fingerprinter,
            cache=cache,
        )


@dataclasses.dataclass(frozen=True)
class SkippableStep(
    ChainableWorkflowMixin[StartT, EndT],
    ReplaceEnabledWorkflowMixin[StartT, EndT],
):
    step: Workflow[StartT, EndT]

    def __call__(self, inp: StartT) -> EndT:
        if not self.skip_condition(inp):
            return self.step(inp)
        return inp  # type: ignore[return-value]  # up to the implementer to make sure StartT == EndT

    def skip_condition(self, inp: StartT) -> bool:
        raise NotImplementedError()
