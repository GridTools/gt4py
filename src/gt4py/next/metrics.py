# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import contextvars
import dataclasses
import json
import pathlib
import sys
from collections.abc import MutableSequence, Sequence

import numpy as np

from gt4py.eve import utils
from gt4py.eve.extended_typing import Final, TypeVar


# Common metric names
STENCIL_METRIC: Final = sys.intern("stencil")
TOTAL_METRIC: Final = sys.intern("total")


# Metric collection levels
DISABLED: Final = 0
MINIMAL: Final = 1
PERFORMANCE: Final = 10
INFO: Final = 30
VERBOSE: Final = 50
ALL: Final = 100


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
    samples: MutableSequence[float] = dataclasses.field(default_factory=list)

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


class MetricCollection(utils.CustomDefaultDictBase[str, Metric]):
    """
    A collection of metrics, organized as a mapping from metric names to `Metric` objects.

    Empty `Metric` instances are created automatically when accessing keys
    that do not exist.

    Example:
        >>> metrics = MetricCollection()
        >>> metrics["execution_time"].add_sample(0.1)
        >>> metrics["execution_time"].add_sample(0.2)
        >>> metrics["execution_time"].samples
        [0.1, 0.2]
    """

    def value_factory(self, key: str) -> Metric:
        return Metric(name=key)

    def add_sample(self, name: str, sample: float) -> None:
        self[name].samples.append(sample)


_KeyT = TypeVar("_KeyT", default=str)


class MetricCollectionStore(utils.CustomDefaultDictBase[_KeyT, MetricCollection]):
    """
    A dictionary-like store for metric collections.

    Empty `MetricCollection` instances are created automatically when accessing keys
    that don't exist yet.

    Example:
        >>> store = MetricCollectionStore()
        >>> store["program1"]["execution_time"].add_sample(0.1)
        >>> store["program1"]["execution_time"].samples
        [0.1]

        >>> store["program2"]["execution_time"].add_sample(0.2)
        >>> store["program2"]["foo_time"].add_sample(2.3)
        >>> store.metric_names
        ['execution_time', 'foo_time']
    """

    def value_factory(self, key: _KeyT) -> MetricCollection:
        return MetricCollection()

    @property
    def metric_names(self) -> Sequence[str]:
        """Returns a list of all metric names across all collections in the store."""

        return list(
            dict.fromkeys(
                (name for collection in self.values() for name in collection.keys())
            ).keys()
        )


#: Global metric collection store for the entire program.
program_metrics: MetricCollectionStore[str] = MetricCollectionStore()

#: Context variable to store the active metric collection of the current program.
active_metric_collection: contextvars.ContextVar[MetricCollection | None] = contextvars.ContextVar(
    "active_metric_collection", default=None
)


def dumps(metric_cs: MetricCollectionStore | None = None) -> str:
    """
    Format the metrics in the collection store as a string table.

    This function generates a formatted string representation of the metrics
    in the collection store. Each row represents a program, and each column
    represents a metric, showing both the mean value and standard deviation.

    If no explicit collection store is provided, uses the global `program_metrics`.
    """
    if metric_cs is None:
        metric_cs = program_metrics

    title_program = "program"
    title_cols = max(len(k) for k in (*metric_cs.keys(), title_program)) + 1
    metric_names = metric_cs.metric_names
    width = max(len(name) for name in (*metric_names, ""))
    titles = (f"{name:<{width}}  {'+/-':<{width}}" for name in metric_names)
    header = f"{title_program:{title_cols}} {'  '.join(titles)}"

    rows = []
    for program in metric_cs.keys():
        cells = []
        for metric_name in metric_names:
            if metric_name in metric_cs[program]:
                cells.append(
                    f"{metric_cs[program][metric_name].mean:<{width}.5e}  "
                    f"{metric_cs[program][metric_name].std:<{width}.5e}"
                )
            else:
                cells.append(f"{'N/A':<{width}} {'N/A':<{width}}")
        rows.append(f"{program:{title_cols}} {'  '.join(cells)}")

    return "\n".join(["", header, *rows, ""])


def dump(filename: str | pathlib.Path, metric_cs: MetricCollectionStore | None = None) -> None:
    pathlib.Path(filename).write_text(dumps(metric_cs))


def dumps_json(metric_cs: MetricCollectionStore | None = None) -> str:
    """
    Export metrics as a JSON string with structure: {program: {metric_name: [samples]}}

    If no explicit collection store is provided, uses the global `program_metrics`.
    """
    if metric_cs is None:
        metric_cs = program_metrics

    return json.dumps(
        {
            program: {name: metric.samples for name, metric in collection.items()}
            for program, collection in metric_cs.items()
        },
    )


def dump_json(filename: str | pathlib.Path, metric_cs: MetricCollectionStore | None = None) -> None:
    pathlib.Path(filename).write_text(dumps_json(metric_cs))
