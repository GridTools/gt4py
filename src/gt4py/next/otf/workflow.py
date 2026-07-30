# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import pathlib
import typing
from typing import Any, Callable, Generic, Protocol, TypeAlias, TypeVar

from gt4py._core import definitions as core_defs, filecache
from gt4py.eve.extended_typing import OpaqueMutableMapping
from gt4py.next import config, fingerprinting, utils
from gt4py.next.instrumentation import hook_machinery


StartT = TypeVar("StartT")
EndT = TypeVar("EndT")
HashT = TypeVar("HashT")
DefT = TypeVar("DefT")
ArgsT = TypeVar("ArgsT")


#: A pipeline step: any callable taking a single input and returning the next
#: stage. This alias is the whole composition "framework": pipelines are plain
#: frozen dataclasses of steps with an explicit `__call__`, and customization
#: is composition-time via `dataclasses.replace`.
Step: TypeAlias = Callable[[StartT], EndT]


@dataclasses.dataclass
class ProgramWithArgs(Generic[DefT, ArgsT]):
    """Pair of a program definition in any stage with the arguments it is compiled for.

    This is the envelope threaded through the definition-transforming half of
    the toolchain. It deliberately lives in this DSL-neutral bottom module:
    `ffront.stages` parameterizes it at module-import time, while the
    DSL-aware stage definitions in `otf.stages` import `ffront.stages`.
    """

    definition: DefT
    args: ArgsT


@hook_machinery.event_hook
def stage_hook(name: str, artifact: Any) -> None:
    """
    Event hook emitted when a named pipeline step produces an artifact.

    It is emitted by the named pipelines (`gt4py.next.backend.Transforms` and
    `gt4py.next.backend.CompilePipeline`) after each executed step, and by
    `Toolchain.translate` for the translation step it runs directly. The step
    names of the standard pipelines (`func_to_past`, `past_to_itir`,
    `translation`, `bindings`, `compilation`, ...) are therefore observable.

    A subscriber must declare its parameters with exactly these names, `name`
    and `artifact`: the hook machinery validates the signature on registration
    and rejects a mismatch.

    Args:
        name: Name of the step that just ran (its field name in the pipeline
            dataclass, e.g. `past_to_itir` or `translation`).
        artifact: The artifact returned by the step. It is passed through
            opaquely and unformatted; subscribers decide how to inspect it.
    """


@dataclasses.dataclass(frozen=True)
class CachedStep(Generic[StartT, EndT, HashT]):
    """
    Cached workflow of single input callables.

    The cache key combines a fingerprint of the workflow state (so that changes
    to the step invalidate the cache) with the value returned by
    ``input_fingerprinter`` for the given input. The caller selects both
    fingerprinters explicitly:

    - ``input_fingerprinter`` fingerprints each input value.
    - ``step_fingerprinter`` fingerprints the workflow state and combines the
      two halves into the final key. Its choice fixes the cache's durability.
      The default, :data:`~gt4py.next.fingerprinting.lenient_fingerprinter`, suits an
      in-memory cache (a plain ``dict``, the default): it lives within a single
      process and so tolerates non-importable callables in the step graph
      (lambdas, closures, dynamically-created steps) by hashing them
      structurally. A persistent cache (e.g. ``FileCache``) instead requires
      :data:`~gt4py.next.fingerprinting.strict_fingerprinter`, so that every referenced
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

    step: Step[StartT, EndT]
    input_fingerprinter: Callable[[StartT], HashT] = dataclasses.field(
        metadata=utils.gt4py_metadata(fingerprint=False)
    )
    step_fingerprinter: fingerprinting.Fingerprinter = dataclasses.field(
        default=fingerprinting.lenient_fingerprinter,
        metadata=utils.gt4py_metadata(fingerprint=False),
    )
    cache: OpaqueMutableMapping[str, EndT] = dataclasses.field(
        repr=False, default_factory=dict, metadata=utils.gt4py_metadata(fingerprint=False)
    )

    @classmethod
    def in_memory(
        cls,
        step: Step[StartT, EndT],
        *,
        input_fingerprinter: Callable[[StartT], HashT],
        cache: OpaqueMutableMapping[str, EndT] | None = None,
    ) -> CachedStep[StartT, EndT, HashT]:
        """
        Build an in-memory store with the lenient fingerprinter.

        Defaults to a fresh ``dict``.
        """
        return cls(
            step=step,
            input_fingerprinter=input_fingerprinter,
            step_fingerprinter=fingerprinting.lenient_fingerprinter,
            cache={} if cache is None else cache,
        )

    @classmethod
    def persistent(
        cls,
        step: Step[StartT, EndT],
        *,
        input_fingerprinter: Callable[[StartT], HashT],
        cache: OpaqueMutableMapping[str, EndT] | str | pathlib.Path,
    ) -> CachedStep[StartT, EndT, HashT]:
        """Build a persistent folder-based store with the lenient fingerprinter."""

        if isinstance(cache, (str, pathlib.Path)):
            cache = filecache.FileCache(cache)

        return cls(
            step=step,
            input_fingerprinter=input_fingerprinter,
            step_fingerprinter=fingerprinting.lenient_fingerprinter,
            cache=cache,
        )

    @functools.cached_property
    def _step_fingerprint(self) -> str:
        # The step and the build-cache version are immutable, so their
        # (potentially expensive) fingerprint is computed once per instance
        # instead of on every cache lookup.
        return self.step_fingerprinter((config.BUILD_CACHE_VERSION_ID, self))

    def __call__(self, inp: StartT) -> EndT:
        """Run the step only if the input is not cached, else return from cache."""
        key = self.cache_key(inp)
        try:
            result = self.cache[key]
        except KeyError:
            result = self.cache[key] = self.step(inp)
        return result

    def cache_key(self, inp: StartT) -> str:
        return self.step_fingerprinter((self._step_fingerprint, self.input_fingerprinter(inp)))


@typing.runtime_checkable
class DeviceConfigurable(Protocol):
    """A step that records the device it was configured for."""

    device_type: core_defs.DeviceType


def check_device_agreement(step: Any, device_type: core_defs.DeviceType, what: str) -> None:
    """
    Raise if an injected step is configured for a different device.

    Builders configure the steps they create themselves from the requested
    device, but an injected step is used verbatim. Without this check a
    mismatch would silently produce a pipeline whose steps disagree about the
    target device, which surfaces much later as a confusing compilation or
    runtime failure.

    Args:
        step: The step to check. Steps that do not record a device are accepted.
        device_type: The device the surrounding pipeline is built for.
        what: Name of the step, used in the error message.

    Raises:
        ValueError: If `step` records a device other than `device_type`.
    """
    if isinstance(step, DeviceConfigurable) and step.device_type is not device_type:
        raise ValueError(
            f"The injected {what} is configured for device '{step.device_type.name}',"
            f" but the workflow is being built for '{device_type.name}'. Build the step"
            f" with 'device_type=DeviceType.{device_type.name}' or leave it out to get"
            " the default."
        )


if config.DUMP_STAGES is not None:
    # Register the `GT4PY_DUMP_STAGES` subscriber so plain program runs dump their
    # stage artifacts without any user code. Registering directly (instead of
    # calling `stage_dump.enable()`) avoids re-importing this module while it is
    # still being initialized.
    from gt4py.next.instrumentation import stage_dump as _stage_dump

    stage_hook.register(_stage_dump.dump_stage, name=_stage_dump.SUBSCRIBER_NAME)
