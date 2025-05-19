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
    return metrics.MetricCollectionStore(
        {
            "program1": {
                metrics.CC_METRIC: metrics.Metric(samples=[1.0, 2.0, 3.0]),
                metrics.TOTAL_METRIC: metrics.Metric(samples=[4.0, 5.0, 6.0]),
            },
            "program2": {
                metrics.CC_METRIC: metrics.Metric(samples=[10.0, 20.0, 30.0]),
                metrics.TOTAL_METRIC: metrics.Metric(samples=[40.0, 50.0, 60.0]),
            },
        }
    )


@pytest.fixture
def empty_metrics():
    return metrics.MetricCollectionStore({"program1": {}, "program2": {}})


def test_dumps(sample_metrics):
    """Test that dumps correctly formats the metrics as a string table."""
    result = metrics.dumps(sample_metrics)
    assert isinstance(result, str)
    assert "program1" in result
    assert "program2" in result
    assert metrics.CC_METRIC in result
    assert metrics.TOTAL_METRIC in result
    # Check that the mean values are properly formatted
    assert "2.00000e+00" in result  # Mean of [1.0, 2.0, 3.0]
    assert "5.00000e+00" in result  # Mean of [4.0, 5.0, 6.0]
    assert "2.00000e+01" in result  # Mean of [10.0, 20.0, 30.0]
    assert "5.00000e+01" in result  # Mean of [40.0, 50.0, 60.0]
    # Check that std deviations are properly formatted
    assert "1.00000e+00" in result  # Std of [1.0, 2.0, 3.0]
    assert "1.00000e+00" in result  # Std of [4.0, 5.0, 6.0]
    assert "1.00000e+01" in result  # Std of [10.0, 20.0, 30.0]
    assert "1.00000e+01" in result  # Std of [40.0, 50.0, 60.0]


def test_dumps_empty_metrics(empty_metrics):
    """Test dumps with empty metrics collections."""
    result = metrics.dumps(empty_metrics)
    assert isinstance(result, str)
    assert "program1" in result
    assert "program2" in result
    # There should be no metrics columns
    assert result.count("N/A") == 0


def test_dump(sample_metrics, tmp_path):
    """Test that dump correctly writes metrics to a file."""
    output_file = tmp_path / "metrics.txt"
    metrics.dump(output_file, sample_metrics)

    # Check that the file exists and has content
    assert output_file.exists()
    content = output_file.read_text()

    # Verify the content contains expected information
    assert isinstance(content, str)
    assert "program1" in content
    assert "program2" in content
    assert metrics.CC_METRIC in content
    assert metrics.TOTAL_METRIC in content


def test_dumps_json(sample_metrics):
    """Test that dumps_json correctly formats metrics as JSON string."""
    result = metrics.dumps_json(sample_metrics)
    assert isinstance(result, str)

    # Parse the JSON string and verify its contents
    data = json.loads(result)

    # Check program keys
    assert "program1" in data
    assert "program2" in data

    # Check metric data for program1
    assert metrics.CC_METRIC in data["program1"]
    assert data["program1"][metrics.CC_METRIC] == [1.0, 2.0, 3.0]
    assert metrics.TOTAL_METRIC in data["program1"]
    assert data["program1"][metrics.TOTAL_METRIC] == [4.0, 5.0, 6.0]

    # Check metric data for program2
    assert metrics.CC_METRIC in data["program2"]
    assert data["program2"][metrics.CC_METRIC] == [10.0, 20.0, 30.0]
    assert metrics.TOTAL_METRIC in data["program2"]
    assert data["program2"][metrics.TOTAL_METRIC] == [40.0, 50.0, 60.0]


def test_dump_json(sample_metrics, tmp_path):
    """Test that dump_json correctly writes metrics to a JSON file."""
    output_file = tmp_path / "metrics.json"
    metrics.dump_json(output_file, sample_metrics)

    # Check that the file exists
    assert output_file.exists()

    # Read the file and verify its contents
    with open(output_file, "r") as f:
        _ = json.load(f)
