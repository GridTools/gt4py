# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import collections
import contextlib
import contextvars
import dataclasses
import itertools
import json
import numbers
import pathlib
import sys
import time
import types
import typing
from collections.abc import Callable, Mapping
from typing import ClassVar

import numpy as np

from gt4py.eve import extended_typing as xtyping, utils
from gt4py.eve.extended_typing import Any, Final
from gt4py.next import config
from gt4py.next.otf import arguments


_NO_KEY_SET_MARKER_: Final[str] = sys.intern("@_NO_KEY_SET_MARKER_@")


# Common metric names
COMPUTE_METRIC: Final[str] = sys.intern("compute")
TOTAL_METRIC: Final[str] = sys.intern("total")


# Metric collection levels
DISABLED: Final[int] = 0
MINIMAL: Final[int] = 1
GPU_TX_MARKERS: Final[int] = 2
PERFORMANCE: Final[int] = 10
INFO: Final[int] = 30
VERBOSE: Final[int] = 50
ALL: Final[int] = 100


def is_enabled() -> bool:
    """Check if a given metrics collection level is enabled."""
    return config.COLLECT_METRICS_LEVEL > DISABLED


def is_level_enabled(level: int) -> bool:
    """Check if a given metrics collection level is enabled."""
    return config.COLLECT_METRICS_LEVEL >= level


def get_current_level() -> int:
    """Retrieve the current metrics collection level from the configuration."""
    return config.COLLECT_METRICS_LEVEL


@dataclasses.dataclass(frozen=True)
class Metric:
    """
    A class to collect and analyze samples of a metric.

    Examples:
        >>> metric = Metric(name="execution_time")
        >>> metric.add_sample(0.1)
        >>> metric.add_sample(0.2)
        >>> print(f"{metric.mean:.2}")
        0.15
        >>> print(metric)
        1.50000e-01 +/- 7.07107e-02
    """

    name: str | None = None
    samples: list[float] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.name:
            object.__setattr__(self, "name", sys.intern(self.name))

    @property
    def mean(self) -> np.floating:
        if len(self.samples) == 0:
            raise ValueError("Cannot compute mean of empty sample list.")
        return np.mean(self.samples)

    @property
    def std(self) -> np.floating:
        if len(self.samples) == 0:
            raise ValueError("Cannot compute std of empty sample list.")
        return np.std(self.samples, ddof=1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, values={self.samples})"

    def __str__(self) -> str:
        return f"{self.mean:.5e} +/- {self.std:.5e}"

    def add_sample(self, sample: float) -> None:
        self.samples.append(sample)


class MetricsCollection(utils.CustomDefaultDictBase[str, Metric]):
    """
    A collection of metrics, organized as a mapping from metric names to `Metric` objects.

    Empty `Metric` instances are created automatically when accessing keys
    that do not exist.

    Example:
        >>> metrics = MetricsCollection()
        >>> metrics["execution_time"].add_sample(0.1)
        >>> metrics["execution_time"].add_sample(0.2)
        >>> metrics["execution_time"].samples
        [0.1, 0.2]
    """

    def value_factory(self, key: str) -> Metric:
        assert isinstance(key, str)
        return Metric(name=key)


@dataclasses.dataclass(slots=True)
class Source:
    """A source of metrics, typically associated with a program."""

    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    metrics: MetricsCollection = dataclasses.field(default_factory=MetricsCollection)
    assigned_key: str | None = dataclasses.field(default=None, init=False)


#: Global store for all measurements.
sources: collections.defaultdict[str, Source] = collections.defaultdict(Source)

# Context variables storing the active source keys.
_source_key_cvar: contextvars.ContextVar[str] = contextvars.ContextVar("source_key")


def is_current_source_key_set() -> bool:
    """Check if there is an on-going metrics collection."""
    return _source_key_cvar.get(_NO_KEY_SET_MARKER_) is not _NO_KEY_SET_MARKER_


def get_current_source_key() -> str:
    """Retrieve the current source key for metrics collection."""
    return _source_key_cvar.get()


def set_current_source_key(key: str) -> Source:
    """Set the current source key for metrics collection."""
    assert _source_key_cvar.get(_NO_KEY_SET_MARKER_) is _NO_KEY_SET_MARKER_, (
        "A source key is already set."
    )
    _source_key_cvar.set(key)
    return sources[key]


def get_current_source() -> Source:
    """Retrieve the active metrics collection source."""
    return sources[_source_key_cvar.get()]


def add_sample_to_current_source(metric_name: str, sample: float) -> None:
    """Add a sample to a metric in the current source."""
    return get_current_source().metrics[metric_name].add_sample(sample)


@dataclasses.dataclass(slots=True)
class SourceKeyContextManager(contextlib.AbstractContextManager):
    """
    A context manager to handle metrics collection sources.

    When entering this context manager, it sets up a new source key for collection
    of metrics in a module contextvar. Upon exiting the context, it resets the
    contextvar to its previous state.

    Note:
        This is implemented as a context manager class instead of a generator
        function with `@contextlib.contextmanager` to avoid the extra overhead
        of renewing the generator inside `contextlib.contextmanager`.
    """

    key: str | None = None
    previous_cvar_token: contextvars.Token | None = dataclasses.field(init=False)

    def __enter__(self) -> None:
        if is_enabled():
            self.previous_cvar_token = _source_key_cvar.set(self.key or _NO_KEY_SET_MARKER_)
        else:
            self.previous_cvar_token = None

    def __exit__(
        self,
        exc_type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if self.previous_cvar_token is not None:
            _source_key_cvar.reset(self.previous_cvar_token)


metrics_context = SourceKeyContextManager


@dataclasses.dataclass(slots=True)
class AbstractCollectorContextManager(contextlib.AbstractContextManager):
    """
    A context manager to handle metrics collection.

    This context manager sets up a new `SourceHandler` for collecting metrics
    in a module contextvar when entering the context. Upon exiting the context,
    it resets the contextvar to its previous state and checks that the collected
    metrics have a proper form.

    Note:
        This is implemented as a context manager class instead of a generator
        function with `@contextlib.contextmanager` to avoid the extra overhead
        of renewing the generator inside `contextlib.contextmanager`.
    """

    @property
    @abc.abstractmethod
    def level(self) -> int: ...

    @property
    @abc.abstractmethod
    def metric_name(self) -> str: ...

    enter_collection_callback: ClassVar[Callable[[], float]] = staticmethod(time.perf_counter)

    exit_collection_callback: ClassVar[Callable[[float], float]] = staticmethod(
        lambda enter_state: time.perf_counter() - enter_state
    )

    key: str | None = None
    previous_cvar_token: contextvars.Token = dataclasses.field(init=False)
    enter_state: float | None = dataclasses.field(init=False)

    def __enter__(self) -> None:
        if is_level_enabled(self.level):
            self.enter_state = self.enter_collection_callback()
            self.previous_cvar_token = _source_key_cvar.set(self.key or _NO_KEY_SET_MARKER_)
        else:
            self.enter_state = None

    def __exit__(
        self,
        exc_type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if self.enter_state is not None:
            assert is_current_source_key_set() is True
            sources[_source_key_cvar.get()].metrics[self.metric_name].add_sample(
                self.exit_collection_callback(self.enter_state)
            )
            _source_key_cvar.reset(self.previous_cvar_token)


def dumps(metric_sources: Mapping[str, Source] | None = None) -> str:
    """
    Format the metrics in the collection store as a string table.

    This function generates a formatted string representation of the metrics
    in the collection store. Each row represents a program, and each column
    represents a metric, showing both the mean value and standard deviation.

    If no explicit collection store is provided, uses the global `program_metrics`.
    """
    if metric_sources is None:
        metric_sources = typing.cast(Mapping[str, Source], sources)
    assert metric_sources is not None

    source_names: list[str] = [*sources.keys()]
    metric_names: list[str] = [
        *dict.fromkeys(
            itertools.chain.from_iterable(
                source.metrics.keys() for source in metric_sources.values()
            )
        ).keys()
    ]
    title_program = "program"
    title_cols = max(len(k) for k in (*metric_names, title_program)) + 1
    width = max([10, *(len(name) for name in (*metric_names, *source_names))])
    titles = (f"{name:<{width}}  {'+/-':<{width}}" for name in metric_names)
    header = f"{title_program:{title_cols}}  {'  '.join(titles)}"

    rows = []
    for name, source in metric_sources.items():
        cells = []
        for metric_name in metric_names:
            if metric_name in source.metrics:
                cells.append(
                    f"{source.metrics[metric_name].mean:<.4e}  {source.metrics[metric_name].std:<.4e}"
                )
            else:
                cells.append(f"{'N/A':<{width}} {'N/A':<{width}}")
        rows.append(f"{name:{title_cols}}  {'  '.join(cells)}")

    return str.join("\n", ["", header, *rows, ""])


def dump(filename: str | pathlib.Path, metric_sources: Mapping[str, Source] | None = None) -> None:
    pathlib.Path(filename).write_text(dumps(metric_sources))


def dumps_json(metric_sources: Mapping[str, Source] | None = None) -> str:
    """
    Export metric sources as a JSON string.

    If no explicit metric sources mapping is provided, the global `sources`
    will be used.

    Note:

        The exported JSON snippets have the following structure:

        ```json
        {
            source_key: {
                'metadata': {'key': value},
                'metrics': {'metric_name': [samples]}
            }
        }
        ```
    """
    if metric_sources is None:
        metric_sources = typing.cast(Mapping[str, Source], sources)
    assert metric_sources is not None

    def default_json_encoder(obj: object) -> object:
        match obj:
            case Source() as src:
                return {
                    "metadata": src.metadata,
                    "metrics": {name: metric.samples for name, metric in src.metrics.items()},
                }
            case arguments.StaticArg() as arg:
                return arg.value
            case xtyping.DataclassABC():
                return dataclasses.asdict(obj)
            case numbers.Integral() as i:
                return int(i)
            case numbers.Real() as r:
                return float(r)

        try:
            return str(obj)
        except Exception:
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            ) from None

    return json.dumps(metric_sources, default=default_json_encoder)


def dump_json(
    filename: str | pathlib.Path, metric_sources: Mapping[str, Source] | None = None
) -> None:
    pathlib.Path(filename).write_text(dumps_json(metric_sources))
