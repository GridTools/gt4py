# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections
import contextlib
import contextvars
import dataclasses
import functools
import itertools
import json
import numbers
import pathlib
import sys
import types
import typing
from collections.abc import Mapping

import numpy as np

from gt4py.eve import extended_typing as xtyping, utils
from gt4py.eve.extended_typing import Any, Final
from gt4py.next import config
from gt4py.next.otf import arguments


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


@dataclasses.dataclass
class Source:
    """A source of metrics, typically associated with a program."""

    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    metrics: MetricsCollection = dataclasses.field(default_factory=MetricsCollection)


#: Global store for all measurements.
sources: collections.defaultdict[str, Source] = collections.defaultdict(Source)


class SourceHandler:
    """
    A handler to manage addition of metrics sources to the global store.

    This object is used to collect metrics for a specific source (e.g., a program)
    before a final key is assigned to it. The key is typically set when the program
    is first executed or compiled, and it uniquely identifies the source in the
    global metrics store.
    """

    def __init__(self, source: Source | None = None) -> None:
        if source is not None:
            self.source = source

    @property
    def key(self) -> str | None:
        return self.__dict__.get("_key", None)

    @key.setter
    def key(self, value: str) -> None:
        # The key can only be set once, and if it matches an existing source
        # in the global store, it must be the same object.
        if self.key is not None and self.key != value:
            raise RuntimeError("Metrics source key is already set.")

        if value not in sources:
            sources[value] = self.source
        else:
            source_in_store = sources[value]
            if self.__dict__.setdefault("source", source_in_store) is not source_in_store:
                raise RuntimeError("Conflicting metrics source data found in the global store.")

        self._key = value

    # The following attributes are implemented as `cached_properties`
    # for efficiency and to be able to initialize them lazily when needed,
    # even if the key is not set.
    @functools.cached_property
    def source(self) -> Source:
        return Source()

    @functools.cached_property
    def metrics(self) -> MetricsCollection:
        return self.source.metrics

    @functools.cached_property
    def metadata(self) -> dict[str, Any]:
        return self.source.metadata


# Context variable storing the active collection context.
_source_cvar: contextvars.ContextVar[SourceHandler | None] = contextvars.ContextVar(
    "source", default=None
)


def in_collection_mode() -> bool:
    """Check if there is an on-going metrics collection."""
    return _source_cvar.get() is not None


def get_current_source() -> SourceHandler:
    """Retrieve the active metrics collection source."""
    source_handler = _source_cvar.get()
    assert source_handler is not None
    return source_handler


def get_source(key: str, *, assign_current: bool = True) -> Source:
    """
    Retrieve a metrics source by its key, optionally associating it to the current context.
    """
    if in_collection_mode() and assign_current:
        metrics_source_handler = get_current_source()
        # Set the key if not already set, which will also add the
        # source to the global store. Note that if the key is already set,
        # this will only succeed if the same object.
        metrics_source_handler.key = key
        metrics_source = metrics_source_handler.source
    else:
        metrics_source = sources[key]

    return metrics_source


class CollectorContextManager(contextlib.AbstractContextManager):
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

    __slots__ = ("previous_collector_token", "source_handler")

    source_handler: SourceHandler | None
    previous_collector_token: contextvars.Token | None

    def __enter__(self) -> SourceHandler | None:
        if config.COLLECT_METRICS_LEVEL > 0:
            assert _source_cvar.get() is None
            self.source_handler = SourceHandler()
            self.previous_collector_token = _source_cvar.set(self.source_handler)
            return self.source_handler
        else:
            self.source_handler = self.previous_collector_token = None
            return None

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        if self.previous_collector_token is not None:
            _source_cvar.reset(self.previous_collector_token)
            if type_ is None:
                if self.source_handler is not None and self.source_handler.key is None:
                    raise RuntimeError("Metrics source key was not set during collection.")


def collect() -> CollectorContextManager:
    return CollectorContextManager()


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
