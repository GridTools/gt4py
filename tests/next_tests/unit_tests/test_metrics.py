# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import json

import pytest

from gt4py.next import metrics


@pytest.fixture
def sample_metrics():
    return {
        "program1": metrics.RuntimeMetric(
            cpp=metrics.MetricAccumulator(values=[1.0, 2.0, 3.0]),
            total=metrics.MetricAccumulator(values=[4.0, 5.0, 6.0]),
            transforms=metrics.MetricAccumulator(values=[7.0, 8.0, 9.0]),
        ),
        "program2": metrics.RuntimeMetric(
            cpp=metrics.MetricAccumulator(values=[10.0, 20.0, 30.0]),
            total=metrics.MetricAccumulator(values=[40.0, 50.0, 60.0]),
            transforms=metrics.MetricAccumulator(values=[70.0, 80.0, 90.0]),
        ),
    }


@pytest.fixture
def empty_metrics():
    return {"program1": metrics.RuntimeMetric(), "program2": metrics.RuntimeMetric()}


def test_print_stats(sample_metrics, capsys):
    metrics.print_stats(sample_metrics)
    captured = capsys.readouterr()
    assert "program1" in captured.out
    assert "program2" in captured.out
    for metric in metrics.RuntimeMetric.metric_keys():
        assert metric in captured.out
    assert "5.0000" in captured.out  # Mean of [4.0, 5.0, 6.0]


def test_print_stats_empty(empty_metrics, capsys):
    metrics.print_stats(empty_metrics)
    captured = capsys.readouterr()
    assert "program1" in captured.out
    assert "program2" in captured.out
    for metric in metrics.RuntimeMetric.metric_keys():
        assert metric in captured.out


def test_dump_json(sample_metrics, tmp_path):
    output_file = tmp_path / "metrics.json"
    metrics.dump_json(output_file, sample_metrics)

    with open(output_file, "r") as f:
        data = json.load(f)

    assert "program1" in data
    for metric in metrics.RuntimeMetric.metric_keys():
        assert metric in data["program1"]
    assert data["program1"]["cpp"] == [1.0, 2.0, 3.0]
    assert data["program2"]["transforms"] == [70.0, 80.0, 90.0]


def test_dump_json_empty(empty_metrics, tmp_path):
    output_file = tmp_path / "metrics.json"
    metrics.dump_json(output_file, empty_metrics)

    with open(output_file, "r") as f:
        data = json.load(f)

    assert "program1" in data
    for metric in metrics.RuntimeMetric.metric_keys():
        assert metric in data["program1"]
    assert "program2" in data
