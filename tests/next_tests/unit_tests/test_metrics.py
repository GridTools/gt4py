# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import json
from collections.abc import Mapping

import pytest

from gt4py.next import metrics


@pytest.fixture
def sample_source_metrics() -> Mapping[str, metrics.Source]:
    return {
        "program1": metrics.Source(
            metadata={"description": "Test program 1"},
            metrics=metrics.MetricCollection(
                **{
                    metrics.COMPUTE_METRIC: metrics.Metric(samples=[1.0, 2.0, 3.0]),
                    metrics.TOTAL_METRIC: metrics.Metric(samples=[4.0, 5.0, 6.0]),
                }
            ),
        ),
        "program2": metrics.Source(
            metadata={"description": "Test program 2"},
            metrics=metrics.MetricCollection(
                **{
                    metrics.COMPUTE_METRIC: metrics.Metric(samples=[10.0, 20.0, 30.0]),
                    metrics.TOTAL_METRIC: metrics.Metric(samples=[40.0, 50.0, 60.0]),
                }
            ),
        ),
    }


@pytest.fixture
def empty_source_metrics() -> Mapping[str, metrics.Source]:
    return {
        "program1": metrics.Source(
            metadata={"description": "Test program 1"},
            metrics=metrics.MetricCollection(),
        ),
        "program2": metrics.Source(
            metadata={"description": "Test program 2"},
            metrics=metrics.MetricCollection(),
        ),
    }


# def test_dumps(sample_source_metrics):
#     """Test that dumps correctly formats the metrics as a string table."""
#     result = metrics.dumps(sample_source_metrics)
#     assert isinstance(result, str)

#     lines = result.splitlines()
#     i = 0
#     while i < len(lines) and lines[i] == "":
#         i += 1
#     assert (
#         lines[i].split()
#         == f"program {metrics.COMPUTE_METRIC} +/- {metrics.TOTAL_METRIC} +/-".split()
#     )
#     assert (
#         lines[i + 1].split() == "program1 2.00000e+00 1.00000e+00 5.00000e+00 1.00000e+00".split()
#     )
#     assert (
#         lines[i + 2].split() == "program2 2.00000e+01 1.00000e+01 5.00000e+01 1.00000e+01".split()
#     )


# def test_dumps_empty_metrics(empty_metrics):
#     """Test dumps with empty metrics collections."""
#     result = metrics.dumps(empty_metrics)
#     assert isinstance(result, str)

#     lines = result.splitlines()
#     i = 0
#     while i < len(lines) and lines[i] == "":
#         i += 1
#     assert lines[i].split() == "program".split()
#     assert lines[i + 1].split() == "program1".split()
#     assert lines[i + 2].split() == "program2".split()


# def test_dump(sample_source_metrics, tmp_path):
#     """Test that dump correctly writes metrics to a file."""
#     output_file = tmp_path / "metrics.txt"
#     metrics.dump(output_file, sample_source_metrics)

#     # Check that the file exists and has content
#     assert output_file.exists()
#     content = output_file.read_text()

#     # Verify the content contains expected information
#     lines = content.splitlines()
#     i = 0
#     while i < len(lines) and lines[i] == "":
#         i += 1
#     assert (
#         lines[i].split()
#         == f"program {metrics.COMPUTE_METRIC} +/- {metrics.TOTAL_METRIC} +/-".split()
#     )
#     assert (
#         lines[i + 1].split() == "program1 2.00000e+00 1.00000e+00 5.00000e+00 1.00000e+00".split()
#     )
#     assert (
#         lines[i + 2].split() == "program2 2.00000e+01 1.00000e+01 5.00000e+01 1.00000e+01".split()
#     )


def test_dumps_json(sample_source_metrics):
    """Test that dumps_json correctly formats metrics as JSON string."""
    result = metrics.dumps_json(sample_source_metrics)
    assert isinstance(result, str)

    # Parse the JSON string and verify its contents
    data = json.loads(result)

    # Check program keys
    assert data.keys() == sample_source_metrics.keys()

    # Check metadata keys and metric data
    for source_id in sample_source_metrics:
        assert (
            data[source_id]["metadata"].keys() == sample_source_metrics[source_id].metadata.keys()
        )
        assert data[source_id]["metrics"].keys() == sample_source_metrics[source_id].metrics.keys()
        assert list(data[source_id]["metrics"].values()) == [
            metric.samples for metric in sample_source_metrics[source_id].metrics.values()
        ]


def test_dump_json(sample_source_metrics, tmp_path):
    """Test that dump_json correctly writes metrics to a JSON file."""
    print(f"\n\nsample_source_metrics: {sample_source_metrics}")
    output_file = tmp_path / "metrics.json"

    metrics.dump_json(output_file, sample_source_metrics)

    # Check that the file exists
    assert output_file.exists()

    # Read the file and verify its contents
    data = json.loads(output_file.read_text())
    assert data.keys() == sample_source_metrics.keys()

    # Check program keys
    assert data.keys() == sample_source_metrics.keys()

    # Check metadata keys and metric data
    for source_id in sample_source_metrics:
        assert (
            data[source_id]["metadata"].keys() == sample_source_metrics[source_id].metadata.keys()
        )
        assert data[source_id]["metrics"].keys() == sample_source_metrics[source_id].metrics.keys()
        assert list(data[source_id]["metrics"].values()) == [
            metric.samples for metric in sample_source_metrics[source_id].metrics.values()
        ]

