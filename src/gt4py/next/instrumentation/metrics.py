# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import atexit
import collections
import contextlib
import contextvars
import dataclasses
import functools
import itertools
import json
import numbers
import operator
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
PERFORMANCE: Final[int] = 10
INFO: Final[int] = 30
VERBOSE: Final[int] = 50
ALL: Final[int] = 100


def is_any_level_enabled() -> bool:
    """Check if any metrics collection level is enabled."""
    return config.collect_metrics_level > DISABLED


def is_level_enabled(level: int) -> bool:
    """Check if a given metrics collection level is enabled."""
    return config.collect_metrics_level >= level


def get_current_level() -> int:
    """Retrieve the current metrics collection level (from the configuration module)."""
    return config.collect_metrics_level


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

    __slots__ = ()

    def value_factory(self, key: str) -> Metric:
        assert isinstance(key, str)
        return Metric(name=key)


@dataclasses.dataclass(slots=True)
class Source:  # type: ignore[misc]  # Mypy bug fixed by: https://github.com/python/mypy/pull/20573
    """A source of metrics, typically associated with a program."""

    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    metrics: MetricsCollection = dataclasses.field(default_factory=MetricsCollection)


#: Global store for all measurements.
sources: collections.defaultdict[str, Source] = collections.defaultdict(Source)

# Context variable storing the active source key.
_source_key_cvar: contextvars.ContextVar[str] = contextvars.ContextVar("source_key")


def is_current_source_key_set() -> bool:
    """Check if there is a source key set for metrics collection."""
    return _source_key_cvar.get(_NO_KEY_SET_MARKER_) is not _NO_KEY_SET_MARKER_


def set_current_source_key(key: str) -> None:
    """
    Set the current source key for metrics collection.

    It must be called only when no source key is set (or the same key is already set).
    """
    assert _source_key_cvar.get(_NO_KEY_SET_MARKER_) in {key, _NO_KEY_SET_MARKER_}, (
        "A different source key has been already set."
    )
    _source_key_cvar.set(key)


def get_current_source_key() -> str:
    """Retrieve the current source key for metrics collection (it must be set)."""
    return _source_key_cvar.get()


def get_current_source() -> Source:
    """Retrieve the active metrics collection source (a valid source key must be set)."""
    return sources[_source_key_cvar.get()]


def add_sample_to_current_source(metric_name: str, sample: float) -> None:
    """Add a sample to a metric in the current source (a valid source key must be set)."""
    return sources[_source_key_cvar.get()].metrics[metric_name].add_sample(sample)


@dataclasses.dataclass(slots=True)
class SourceKeyContextManager(contextlib.AbstractContextManager):  # type: ignore[misc]  # Mypy bug fixed by: https://github.com/python/mypy/pull/20573
    """
    A context manager to handle metrics collection source keys.

    When entering this context manager it saves the current source key
    for metrics collection and sets the new source key if provided, or
    a default marker indicating no key is set. Upon exiting the context,
    it resets the source key to its previous state.
    """

    key: str | None = None
    previous_cvar_token: contextvars.Token | None = dataclasses.field(init=False)

    # This class is implemented as a context manager class instead of a generator
    # function with `@contextlib.contextmanager` to avoid the extra overhead
    # of renewing the generator inside `contextlib.contextmanager`.
    def __enter__(self) -> None:
        if is_any_level_enabled():
            self.previous_cvar_token = _source_key_cvar.set(self.key or _NO_KEY_SET_MARKER_)
        else:
            self.previous_cvar_token = None

    def __exit__(
        self,
        exc_type_: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if self.previous_cvar_token is not None:
            _source_key_cvar.reset(self.previous_cvar_token)


class SourceKeySetterAtEnter(SourceKeyContextManager):
    def __exit__(
        self,
        exc_type_: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        pass


metrics_context = SourceKeyContextManager
metrics_setter_at_enter = SourceKeySetterAtEnter


@dataclasses.dataclass(slots=True)
class BaseMetricsCollector(contextlib.AbstractContextManager):  # type: ignore[misc]  # Mypy bug fixed by: https://github.com/python/mypy/pull/20573
    """
    A context manager to handle metrics collection.

    This is a base class for creating metrics collectors that measure
    specific metrics during the execution of a code block. It provides
    a convenient interface for managing the lifecycle of metrics collection.

    Subclasses need to define the `level` and `metric_name` attributes, and,
    optionally override the methods for collecting counters and computing
    the metric. This class offers a simple way to customize this class variables
    accepting them as keyword arguments when creating the subclass.
    """

    def __init_subclass__(
        cls: type[BaseMetricsCollector],
        *,
        level: int,
        metric_name: str,
        collect_enter_counter: Callable[[], float] | None = None,
        collect_exit_counter: Callable[[], float] | None = None,
        compute_metric: Callable[[float, float], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super(BaseMetricsCollector, cls).__init_subclass__(**kwargs)
        cls.level = level
        cls.metric_name = sys.intern(metric_name)
        if collect_enter_counter is not None:
            cls.collect_enter_counter = staticmethod(collect_enter_counter)
        if collect_exit_counter is not None:
            cls.collect_exit_counter = staticmethod(collect_exit_counter)
        if compute_metric is not None:
            cls.compute_metric = staticmethod(compute_metric)

    # Subclass must define these class variables
    level: ClassVar[int]
    metric_name: ClassVar[str]

    # Default implementations for these methods can be overridden by subclasses
    collect_enter_counter: ClassVar[Callable[[], float]] = staticmethod(time.perf_counter)
    collect_exit_counter: ClassVar[Callable[[], float]] = staticmethod(time.perf_counter)
    compute_metric: ClassVar[Callable[[float, float], float]] = staticmethod(operator.sub)
    #: compute_metric(exit_counter, enter_counter) -> metric_value

    # Instance state
    key: str | None = None
    previous_cvar_token: contextvars.Token | None = dataclasses.field(init=False)
    enter_counter: float | None = dataclasses.field(init=False)

    def __enter__(self) -> None:
        if is_level_enabled(self.level):
            self.previous_cvar_token = _source_key_cvar.set(self.key or _NO_KEY_SET_MARKER_)
            self.enter_counter = self.collect_enter_counter()  # type: ignore[misc] # mypy doesn't understand that this is a staticmethod
        else:
            self.previous_cvar_token = None

    def __exit__(
        self,
        exc_type_: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if self.previous_cvar_token is not None:
            assert is_current_source_key_set() is True
            assert hasattr(self, "enter_counter") and self.enter_counter is not None
            sources[_source_key_cvar.get()].metrics[self.metric_name].add_sample(
                self.compute_metric(self.collect_exit_counter(), self.enter_counter)  # type: ignore[call-arg,misc] # mypy doesn't understand that this is a staticmethod
            )
            _source_key_cvar.reset(self.previous_cvar_token)


@functools.cache
def make_collector(
    level: int,
    metric_name: str,
    *,
    collect_enter_counter: Callable[[], float] | None = None,
    collect_exit_counter: Callable[[], float] | None = None,
    compute_metric: Callable[[float, float], float] | None = None,
) -> type[BaseMetricsCollector]:
    """
    Create a custom metrics collector class.

    This function generates a new subclass of `BaseMetricsCollector` with
    the specified configuration for metrics collection.

    Args:
        level: The metrics collection level.
        metric_name: The name of the metric to be collected.
        collect_enter_counter: Optional function to collect the enter counter.
        collect_exit_counter: Optional function to collect the exit counter.
        compute_metric: Optional function to compute the metric from the counters.

    Returns:
        A new subclass of `BaseMetricsCollector` configured with the provided parameters.
    """

    return types.new_class(
        f"AutoMetricsCollectorFor_{metric_name}",
        bases=(BaseMetricsCollector,),
        kwds=dict(
            level=level,
            metric_name=metric_name,
            collect_enter_counter=collect_enter_counter,
            collect_exit_counter=collect_exit_counter,
            compute_metric=compute_metric,
        ),
        exec_body=None,
    )


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


# Handler registration to automatically dump metrics at program exit if
# the corresponding configuration flag is set.
def _dump_metrics_at_exit() -> None:
    """Dump collected metrics to a file at program exit if required."""

    # It is assumed that 'gt4py.next.config' is still alive at this point
    if config.DUMP_METRICS_AT_EXIT and (is_any_level_enabled() or sources):
        try:
            pathlib.Path(config.DUMP_METRICS_AT_EXIT).write_text(dumps_json())
            print(
                f"--- atexit: GT4Py performance metrics saved at {config.DUMP_METRICS_AT_EXIT} ---",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"--- atexit: ERROR: Failed to automatically save GT4Py performance metrics: ---\n{e}",
                file=sys.stderr,
            )


atexit.register(_dump_metrics_at_exit)
