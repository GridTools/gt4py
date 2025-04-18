# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from collections import defaultdict

import numpy as np


@dataclasses.dataclass
class MetricAccumulator:
    values: list[float] = dataclasses.field(default_factory=list)

    def mean(self) -> np.floating:
        return np.mean(self.values)

    def std(self) -> np.floating:
        return np.std(self.values)

    def __str__(self) -> str:
        return f"{self.mean:.3e} +/- {self.std:.3e}"

    def append(self, value: float) -> None:
        self.values.append(value)


@dataclasses.dataclass
class RuntimeMetric:
    cpp_time: MetricAccumulator = dataclasses.field(default_factory=MetricAccumulator)
    total_time: MetricAccumulator = dataclasses.field(default_factory=MetricAccumulator)


global_metric_container: dict[str, RuntimeMetric] = defaultdict(RuntimeMetric)


def summary() -> None:
    max_len = max(len(k) for k in global_metric_container.keys())
    print()
    print(f"{'program':{max_len + 1}} {'cpp':<9} {'std':<9} {'total':<9} {'std':<9}")
    for k, v in global_metric_container.items():
        print(
            f"{k:{max_len + 1}} {v.cpp_time.mean():.3e} {v.cpp_time.std():.3e} {v.total_time.mean():.3e} {v.total_time.std():.3e}"
        )
