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
from collections.abc import Sequence
from typing import Final, TypeAlias

import numpy as np


# definitions for commonly used metric names
CPP: Final = "cpp"
TOTAL: Final = "total"


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


# first level is program name, second level is type of the metric
MetricContainer: TypeAlias = dict[str, dict[str, MetricAccumulator]]
global_metric_container: MetricContainer = defaultdict(lambda: defaultdict(MetricAccumulator))


def _collect_metric_names(metric_container: MetricContainer) -> Sequence[str]:
    result: dict[str, None] = {}
    for metrics in metric_container.values():
        for metric_name in metrics.keys():
            result[metric_name] = None
    return list(result.keys())


def print_stats(metric_container: MetricContainer | None = None) -> None:
    if metric_container is None:
        metric_container = global_metric_container
    title_program = "program"
    title_cols = max(len(k) for k in (*metric_container.keys(), title_program))

    metric_names = _collect_metric_names(metric_container)
    print()
    titles = (f"{name:<11} {'+/-':<11}" for name in metric_names)
    print(f"{title_program:{title_cols + 1}} {'  '.join(titles)}")
    for program in metric_container:
        elems = []
        for metric_name in metric_names:
            if metric_name in metric_container[program]:
                elems.append(
                    f"{metric_container[program][metric_name].mean():.5e} "
                    f"{metric_container[program][metric_name].std():.5e}"
                )
            else:
                elems.append(f"{'N/A':<11} {'N/A':<11}")
        print(f"{program:{title_cols + 1}} {'  '.join(elems)}")


def dump_json(
    filename: str | pathlib.Path, metric_container: MetricContainer | None = None
) -> None:
    if metric_container is None:
        metric_container = global_metric_container
    with open(filename, "w") as f:
        json.dump(
            {
                k: {metric_name: metric.values for metric_name, metric in v.items()}
                for k, v in metric_container.items()
            },
            f,
        )
