# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import json
import pathlib
from collections import defaultdict
from collections.abc import Iterator

import numpy as np


@dataclasses.dataclass
class MetricAccumulator:
    values: list[float] = dataclasses.field(default_factory=list)

    def mean(self) -> np.floating:
        return np.mean(self.values)

    def std(self) -> np.floating:
        return np.std(self.values)

    def __str__(self) -> str:
        return f"{self.mean():.5e} +/- {self.std():.5e}"

    def append(self, value: float) -> None:
        self.values.append(value)


@dataclasses.dataclass
class RuntimeMetric:
    cpp: MetricAccumulator = dataclasses.field(default_factory=MetricAccumulator)
    total: MetricAccumulator = dataclasses.field(default_factory=MetricAccumulator)
    transforms: MetricAccumulator = dataclasses.field(default_factory=MetricAccumulator)

    @classmethod
    def metric_keys(cls) -> Iterator[str]:
        for f in dataclasses.fields(cls):
            yield f.name

    def metrics(self) -> Iterator[MetricAccumulator]:
        for f in dataclasses.fields(self):
            yield getattr(self, f.name)

    def metric_items(self) -> Iterator[tuple[str, MetricAccumulator]]:
        for f in dataclasses.fields(self):
            yield f.name, getattr(self, f.name)


global_metric_container: dict[str, RuntimeMetric] = defaultdict(RuntimeMetric)


def print_stats(metric_container: dict[str, RuntimeMetric] | None = None) -> None:
    if metric_container is None:
        metric_container = global_metric_container
    title_program = "program"
    title_cols = max(len(k) for k in (*metric_container.keys(), title_program))

    print()
    titles = (f"{name:<11} {'+/-':<11}" for name in RuntimeMetric.metric_keys())
    print(f"{title_program:{title_cols + 1}} {'  '.join(titles)}")
    for program in metric_container:
        elems = (
            f"{metric.mean():.5e} {metric.std():.5e}"
            for metric in metric_container[program].metrics()
        )
        print(f"{program:{title_cols + 1}} {'  '.join(elems)}")


def dump_json(
    filename: str | pathlib.Path, metric_container: dict[str, RuntimeMetric] | None = None
) -> None:
    if metric_container is None:
        metric_container = global_metric_container
    with open(filename, "w") as f:
        json.dump(
            {
                k: {metric_name: metric.values for metric_name, metric in v.metric_items()}
                for k, v in metric_container.items()
            },
            f,
        )
