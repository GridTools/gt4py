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
import sys
from collections.abc import Generator

import numpy as np

from gt4py.eve.extended_typing import Any, Final, Hashable
from gt4py.next import config


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


class MetricCollection(collections.defaultdict[str, Metric]):
    def __init__(self) -> None:
        super().__init__()

    def __missing__(self, key: str) -> Metric:
        assert isinstance(key, str)
        self[key] = metric = Metric(name=key)
        return metric


@dataclasses.dataclass(slots=True, init=False)
class Source:
    metadata: dict[str, Any]
    metrics: MetricCollection

    def __init__(self, **kwargs: Any) -> None:
        self.metadata = kwargs
        self.metrics = MetricCollection()


#: Global store for all measurements.
sources: collections.defaultdict[int, Source] = collections.defaultdict(Source)


@dataclasses.dataclass
class SourceHandle:
    key_parts: tuple[Hashable, ...] = ()

    def is_finalized(self) -> bool:
        return "key" in self.__dict__

    @functools.cached_property
    def key(self) -> int:
        if self.key_parts == ():
            raise RuntimeError("Metrics collector has no key set.")
        source_key = hash(self.key_parts)
        if source_key not in sources:
            sources[source_key] = Source()
        return source_key

    @functools.cached_property
    def metrics(self) -> MetricCollection:
        return sources[self.key].metrics

    @functools.cached_property
    def metadata(self) -> dict[str, Any]:
        return sources[self.key].metadata

    def append_to_key(self, *args: Hashable) -> None:
        if "key" in self.__dict__:
            raise RuntimeError("Metrics collector key is already finalized.")
        self.key_parts += args


# Context variable storing the active collection context.
_source_cvar: contextvars.ContextVar[SourceHandle | None] = contextvars.ContextVar(
    "source", default=None
)


def in_collection_mode() -> bool:
    """Check if we are currently in a metrics collection context."""
    return _source_cvar.get() is not None


def get_current_source() -> SourceHandle:
    """Retrieve the active measured entity state."""
    state = _source_cvar.get()
    assert state is not None
    return state


@contextlib.contextmanager
def collect(*args: Hashable) -> Generator[SourceHandle | None, None, None]:
    if not config.COLLECT_METRICS_LEVEL:
        yield None
        return

    assert _source_cvar.get() is None
    source_handler = SourceHandle(key_parts=args)
    previous_collector_token = _source_cvar.set(source_handler)

    try:
        yield source_handler
    finally:
        _source_cvar.reset(previous_collector_token)


# def dumps(metric_cs: MetricCollectionStore | None = None) -> str:
#     """
#     Format the metrics in the collection store as a string table.

#     This function generates a formatted string representation of the metrics
#     in the collection store. Each row represents a program, and each column
#     represents a metric, showing both the mean value and standard deviation.

#     If no explicit collection store is provided, uses the global `program_metrics`.
#     """
#     if metric_cs is None:
#         metric_cs = program_metrics

#     title_program = "program"
#     title_cols = max(len(k) for k in (*metric_cs.keys(), title_program)) + 1
#     metric_names = metric_cs.metric_names
#     width = max(len(name) for name in (*metric_names, ""))
#     titles = (f"{name:<{width}}  {'+/-':<{width}}" for name in metric_names)
#     header = f"{title_program:{title_cols}} {'  '.join(titles)}"

#     rows = []
#     for program in metric_cs.keys():
#         cells = []
#         for metric_name in metric_names:
#             if metric_name in metric_cs[program]:
#                 cells.append(
#                     f"{metric_cs[program][metric_name].mean:<{width}.5e}  "
#                     f"{metric_cs[program][metric_name].std:<{width}.5e}"
#                 )
#             else:
#                 cells.append(f"{'N/A':<{width}} {'N/A':<{width}}")
#         rows.append(f"{program:{title_cols}} {'  '.join(cells)}")

#     return "\n".join(["", header, *rows, ""])


# def dump(filename: str | pathlib.Path, metric_cs: MetricCollectionStore | None = None) -> None:
#     pathlib.Path(filename).write_text(dumps(metric_cs))


# def dumps_json(metric_cs: MetricCollectionStore | None = None) -> str:
#     """
#     Export metrics as a JSON string with structure: {program: {metric_name: [samples]}}

#     If no explicit collection store is provided, uses the global `program_metrics`.
#     """
#     if metric_cs is None:
#         metric_cs = program_metrics

#     return json.dumps(
#         {
#             program: {name: metric.samples for name, metric in collection.items()}
#             for program, collection in metric_cs.items()
#         },
#     )


# def dump_json(filename: str | pathlib.Path, metric_cs: MetricCollectionStore | None = None) -> None:
#     pathlib.Path(filename).write_text(dumps_json(metric_cs))
