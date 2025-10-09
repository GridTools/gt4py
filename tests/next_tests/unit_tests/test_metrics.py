# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import json
import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from gt4py.next import metrics
from gt4py.next.otf import arguments


# TODO(egparedes): add tests for the logic around creating sources lazily
# (SourceHandler class and the context manager).


@pytest.fixture
def sample_source_metadata() -> dict[str, Any]:
    from gt4py.eve import utils

    return {
        f"""{
            utils.CaseStyleConverter.convert(
                arguments.StaticArg.__name__,
                utils.CaseStyleConverter.CASE_STYLE.PASCAL,
                utils.CaseStyleConverter.CASE_STYLE.SNAKE,
            )
        }s""": {
            "horizontal_start": arguments.StaticArg(value=7701),
            "horizontal_end": arguments.StaticArg(value=67096),
            "vertical_start": arguments.StaticArg(value=np.int32(0)),
            "vertical_end": arguments.StaticArg(value=np.int64(80)),
            "is_foo_active": arguments.StaticArg(value=False),
            "limited_area": arguments.StaticArg(value=True),
            "constant_bar": arguments.StaticArg(value=0.42),
            "constant_baz": arguments.StaticArg(value=np.float32(1.234)),
            "constant_complex": arguments.StaticArg(value=np.complex64(1.234)),
        }
    }


@pytest.fixture
def sample_source_metrics(sample_source_metadata: dict[str, Any]) -> Mapping[str, metrics.Source]:
    return {
        "program1": metrics.Source(
            metadata={"description": "Test program 1", **sample_source_metadata},
            metrics=metrics.MetricsCollection(
                **{
                    metrics.COMPUTE_METRIC: metrics.Metric(samples=[1.0, 2.0, 3.0]),
                    metrics.TOTAL_METRIC: metrics.Metric(samples=[4.0, 5.0, 6.0]),
                }
            ),
        ),
        "program2": metrics.Source(
            metadata={"description": "Test program 2", **sample_source_metadata},
            metrics=metrics.MetricsCollection(
                **{
                    metrics.COMPUTE_METRIC: metrics.Metric(samples=[10.0, 20.0, 30.0]),
                    metrics.TOTAL_METRIC: metrics.Metric(samples=[40.0, 50.0, 60.0]),
                }
            ),
        ),
    }


@pytest.fixture
def empty_source_metrics(sample_source_metadata: dict[str, Any]) -> Mapping[str, metrics.Source]:
    return {
        "program1": metrics.Source(
            metadata={"description": "Test program 1", **sample_source_metadata},
            metrics=metrics.MetricsCollection(),
        ),
        "program2": metrics.Source(
            metadata={"description": "Test program 2", **sample_source_metadata},
            metrics=metrics.MetricsCollection(),
        ),
    }


def test_dumps(sample_source_metrics: Mapping[str, metrics.Source]):
    """Test that dumps correctly formats the metrics as a string table."""
    result = metrics.dumps(sample_source_metrics)
    assert isinstance(result, str)
    lines = result.splitlines()
    i = 0
    while i < len(lines) and lines[i] == "":
        i += 1

    assert (
        lines[i].split()
        == f"program {metrics.COMPUTE_METRIC} +/- {metrics.TOTAL_METRIC} +/-".split()
    )
    assert (
        lines[i + 1].split() == "program1  2.0000e+00  1.0000e+00  5.0000e+00  1.0000e+00".split()
    )
    assert (
        lines[i + 2].split() == "program2  2.0000e+01  1.0000e+01  5.0000e+01  1.0000e+01".split()
    )


def test_dumps_empty_metrics(empty_source_metrics: Mapping[str, metrics.Source]):
    """Test dumps with empty metrics collections."""
    result = metrics.dumps(empty_source_metrics)
    assert isinstance(result, str)

    lines = result.splitlines()
    i = 0
    while i < len(lines) and lines[i] == "":
        i += 1
    assert lines[i].split() == "program".split()
    assert lines[i + 1].split() == "program1".split()
    assert lines[i + 2].split() == "program2".split()


def test_dump(sample_source_metrics: Mapping[str, metrics.Source], tmp_path: pathlib.Path):
    """Test that dump correctly writes metrics to a file."""
    output_file = tmp_path / "metrics.txt"
    metrics.dump(output_file, sample_source_metrics)

    # Check that the file exists and has content
    assert output_file.exists()
    content = output_file.read_text()

    # Verify the content contains expected information
    lines = content.splitlines()
    i = 0
    while i < len(lines) and lines[i] == "":
        i += 1
    assert (
        lines[i].split()
        == f"program {metrics.COMPUTE_METRIC} +/- {metrics.TOTAL_METRIC} +/-".split()
    )

    assert (
        lines[i + 1].split() == "program1  2.0000e+00  1.0000e+00  5.0000e+00  1.0000e+00".split()
    )
    assert (
        lines[i + 2].split() == "program2  2.0000e+01  1.0000e+01  5.0000e+01  1.0000e+01".split()
    )


def test_dumps_json(sample_source_metrics: Mapping[str, metrics.Source]):
    """Test that dumps_json correctly formats metrics as JSON string."""
    result = metrics.dumps_json(sample_source_metrics)
    assert isinstance(result, str)

    # Parse the JSON string and verify its contents
    data = json.loads(result)

    # Check program keys
    assert [*data.keys()] == [str(k) for k in sample_source_metrics.keys()]

    # Check metadata keys and metric data
    for source_id in sample_source_metrics:
        assert (
            data[source_id]["metadata"].keys() == sample_source_metrics[source_id].metadata.keys()
        )
        assert data[source_id]["metrics"].keys() == sample_source_metrics[source_id].metrics.keys()
        assert list(data[source_id]["metrics"].values()) == [
            metric.samples for metric in sample_source_metrics[source_id].metrics.values()
        ]


def test_dump_json(sample_source_metrics: Mapping[str, metrics.Source], tmp_path: pathlib.Path):
    """Test that dump_json correctly writes metrics to a JSON file."""
    output_file = tmp_path / "metrics.json"

    metrics.dump_json(output_file, sample_source_metrics)

    # Check that the file exists
    assert output_file.exists()

    # Read the file and verify its contents
    data = json.loads(output_file.read_text())

    # Check program keys
    assert [*data.keys()] == [str(k) for k in sample_source_metrics.keys()]

    # Check metadata keys and metric data
    for source_id in sample_source_metrics:
        assert (
            data[source_id]["metadata"].keys() == sample_source_metrics[source_id].metadata.keys()
        )
        assert data[source_id]["metrics"].keys() == sample_source_metrics[source_id].metrics.keys()
        assert list(data[source_id]["metrics"].values()) == [
            metric.samples for metric in sample_source_metrics[source_id].metrics.values()
        ]
