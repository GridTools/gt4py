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

    @property
    def mean(self):
        return np.mean(self.values)

    @property
    def std(self):
        return np.std(self.values)

    @property
    def mean_skip_first(self):
        return np.mean(self.values[1:]) if len(self.values) > 1 else np.nan

    @property
    def std_skip_first(self):
        return np.std(self.values[1:]) if len(self.values) > 1 else np.nan

    def __str__(self):
        return f"{self.mean:.3e} +/- {self.std:.3e} ({self.mean_skip_first:.3e} +/- {self.std_skip_first:.3e})"

    def append(self, value):
        self.values.append(value)


@dataclasses.dataclass
class RuntimeMetric:
    cpp_time: MetricAccumulator = dataclasses.field(default_factory=MetricAccumulator)
    total_time: MetricAccumulator = dataclasses.field(default_factory=MetricAccumulator)


global_metric_container: dict[str, RuntimeMetric] = defaultdict(RuntimeMetric)


def summary():
    for k, v in global_metric_container.items():
        print(k)
        print(f"  cpp_time:   {v.cpp_time}")
        print(f"  total_time: {v.total_time}")
