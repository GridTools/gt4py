# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import json
import pathlib
import unittest.mock
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from gt4py.next import config as gt_config
from gt4py.next.instrumentation import metrics
from gt4py.next.otf import arguments


class TestSetCurrentSourceKey:
    def test_set_current_source_key_basic(self):
        """Test setting a source key when none is currently set."""
        # Reset context variable before test
        metrics._source_key_cvar.set(metrics._NO_KEY_SET_MARKER_)

        key = "test_source"
        metrics.set_current_source_key(key)

        assert metrics.get_current_source_key() == key

    def test_set_current_source_key_same_key_twice(self):
        """Test setting the same source key twice."""
        # Reset context variable before test
        metrics._source_key_cvar.set(metrics._NO_KEY_SET_MARKER_)

        key = "test_source_same"
        source1 = metrics.set_current_source_key(key)
        source2 = metrics.set_current_source_key(key)

        assert source1 is source2
        assert metrics.get_current_source_key() == key

    def test_set_current_source_key_different_key_raises(self):
        """Test that setting a different source key raises AssertionError."""
        # Reset context variable before test
        metrics._source_key_cvar.set(metrics._NO_KEY_SET_MARKER_)

        key1 = "test_source_1"
        key2 = "test_source_2"

        metrics.set_current_source_key(key1)

        with pytest.raises(AssertionError, match="A different source key has been already set"):
            metrics.set_current_source_key(key2)


class TestSourceKeyContextManager:
    def test_context_manager_sets_and_resets_key(self):
        with gt_config.overrides(collect_metrics_level=metrics.MINIMAL):
            metrics._source_key_cvar.set(
                metrics._NO_KEY_SET_MARKER_
            )  # Reset context variable before test
            assert metrics.is_current_source_key_set() is False

            key = "context_test_key"
            with metrics.metrics_context(key):
                assert metrics.is_current_source_key_set() is True
                assert metrics._source_key_cvar.get() == key
                assert metrics.get_current_source_key() == key

            # After exit, should be reset to marker
            assert metrics.is_current_source_key_set() is False
            assert (
                metrics._source_key_cvar.get(metrics._NO_KEY_SET_MARKER_)
                == metrics._NO_KEY_SET_MARKER_
            )

    def test_context_manager_with_no_key(self):
        with gt_config.overrides(collect_metrics_level=metrics.MINIMAL):
            metrics._source_key_cvar.set("__BEFORE__MARKER__")  # Reset context variable before test

            with metrics.SourceKeyContextManager():
                # Should set to marker if no key is provided
                assert (
                    metrics._source_key_cvar.get(metrics._NO_KEY_SET_MARKER_)
                    == metrics._NO_KEY_SET_MARKER_
                )

            # After exit, should be the previous value
            assert metrics._source_key_cvar.get(metrics._NO_KEY_SET_MARKER_) == "__BEFORE__MARKER__"

    def test_context_manager_nested(self):
        with gt_config.overrides(collect_metrics_level=metrics.MINIMAL):
            metrics._source_key_cvar.set(metrics._NO_KEY_SET_MARKER_)
            key1 = "outer_key"
            key2 = "inner_key"

            with metrics.SourceKeyContextManager(key=key1):
                assert metrics.get_current_source_key() == key1
                with metrics.SourceKeyContextManager(key=key2):
                    assert metrics.get_current_source_key() == key2

                # After inner exit, should restore to outer key
                assert metrics.get_current_source_key() == key1

            # After outer exit, should be marker
            assert (
                metrics._source_key_cvar.get(metrics._NO_KEY_SET_MARKER_)
                == metrics._NO_KEY_SET_MARKER_
            )


class TestBaseMetricsCollector:
    def test_collector_basic_timers(self):
        """Test basic metrics collection with timing."""

        class TestCollector(
            metrics.BaseMetricsCollector, level=metrics.MINIMAL, metric_name="test_metric"
        ): ...

        metrics._source_key_cvar.set(metrics._NO_KEY_SET_MARKER_)
        with gt_config.overrides(collect_metrics_level=metrics.MINIMAL):
            outer_key = "outer_key"
            metrics.set_current_source_key("outer_key")
            assert metrics.get_current_source_key() == outer_key

            key = "test_collector"
            with TestCollector(key=key):
                assert metrics.get_current_source_key() == key

            assert metrics.get_current_source_key() == outer_key

            assert key in metrics.sources
            source = metrics.sources[key]
            assert "test_metric" in source.metrics
            assert len(source.metrics["test_metric"].samples) == 1
            assert source.metrics["test_metric"].samples[0] >= 0

        key = "test_disabled"
        metrics._source_key_cvar.set(metrics._NO_KEY_SET_MARKER_)
        with gt_config.overrides(collect_metrics_level=metrics.DISABLED):
            metrics.set_current_source_key(key)

            with TestCollector(key=key):
                pass

            assert key not in metrics.sources or "test_metric" not in metrics.sources[key].metrics

    def test_collector_with_custom_counters(self):
        """Test collector with custom counter functions."""

        class CustomCollector(
            metrics.BaseMetricsCollector,
            level=metrics.PERFORMANCE,
            metric_name="custom_metric",
            collect_enter_counter=(lambda: 10.0),
            collect_exit_counter=(lambda: 15.0),
        ): ...

        key = "test_custom"
        metrics._source_key_cvar.set(metrics._NO_KEY_SET_MARKER_)
        with gt_config.overrides(collect_metrics_level=metrics.PERFORMANCE):
            with CustomCollector(key=key):
                pass

            assert key in metrics.sources
            source = metrics.sources[key]
            assert "custom_metric" in source.metrics
            assert len(source.metrics["custom_metric"].samples) == 1
            assert source.metrics["custom_metric"].samples[0] == 5.0


class TestMakeCollector:
    def test_make_collector_creates_subclass(self):
        """Test that make_collector creates a proper subclass."""
        CollectorClass = metrics.make_collector(level=metrics.INFO, metric_name="test_metric")

        assert issubclass(CollectorClass, metrics.BaseMetricsCollector)
        assert CollectorClass.level == metrics.INFO
        assert CollectorClass.metric_name == "test_metric"

    def test_make_collector_with_custom_compute(self):
        """Test make_collector with custom compute function."""
        custom_compute = lambda exit, enter: exit - enter + 10
        CollectorClass = metrics.make_collector(
            level=metrics.MINIMAL, metric_name="custom_compute", compute_metric=custom_compute
        )

        assert CollectorClass.compute_metric(20, 5) == 25


class TestMetric:
    def test_metric_mean(self):
        """Test metric mean calculation."""
        metric = metrics.Metric(name="test")
        metric.add_sample(1.0)
        metric.add_sample(2.0)
        metric.add_sample(3.0)

        assert float(metric.mean) == 2.0

    def test_metric_std(self):
        """Test metric standard deviation calculation."""
        metric = metrics.Metric(name="test")
        metric.add_sample(1.0)
        metric.add_sample(2.0)
        metric.add_sample(3.0)

        assert np.isclose(float(metric.std), 1.0)

    def test_metric_mean_empty_raises(self):
        """Test that mean of empty metric raises ValueError."""
        metric = metrics.Metric(name="test")

        with pytest.raises(ValueError, match="Cannot compute mean"):
            _ = metric.mean

    def test_metric_std_empty_raises(self):
        """Test that std of empty metric raises ValueError."""
        metric = metrics.Metric(name="test")

        with pytest.raises(ValueError, match="Cannot compute std"):
            _ = metric.std

    def test_metric_str_representation(self):
        """Test metric string representation."""
        metric = metrics.Metric(name="test")
        metric.add_sample(1.0)
        metric.add_sample(3.0)

        str_repr = str(metric)
        assert "e+" in str_repr or "e-" in str_repr
        assert "+/-" in str_repr


class TestMetricsCollection:
    def test_metrics_collection_auto_creates_metric(self):
        """Test that MetricsCollection auto-creates Metric instances."""
        collection = metrics.MetricsCollection()

        metric = collection["new_metric"]

        assert isinstance(metric, metrics.Metric)
        assert metric.name == "new_metric"

    def test_metrics_collection_persists_values(self):
        """Test that MetricsCollection persists added values."""
        collection = metrics.MetricsCollection()

        collection["metric1"].add_sample(1.0)
        collection["metric1"].add_sample(2.0)

        assert collection["metric1"].samples == [1.0, 2.0]


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
            metrics=metrics.MetricsCollection(**{
                metrics.COMPUTE_METRIC: metrics.Metric(samples=[1.0, 2.0, 3.0]),
                metrics.TOTAL_METRIC: metrics.Metric(samples=[4.0, 5.0, 6.0]),
            }),
        ),
        "program2": metrics.Source(
            metadata={"description": "Test program 2", **sample_source_metadata},
            metrics=metrics.MetricsCollection(**{
                metrics.COMPUTE_METRIC: metrics.Metric(samples=[10.0, 20.0, 30.0]),
                metrics.TOTAL_METRIC: metrics.Metric(samples=[40.0, 50.0, 60.0]),
            }),
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


class TestDumpMetricsAtExit:
    @pytest.mark.parametrize("mode", ["explicit", "auto", False])
    def test_dump_metrics_at_exit_enabled(
        self,
        sample_source_metrics: Mapping[str, metrics.Source],
        tmp_path: pathlib.Path,
        mode: str | None,
    ):
        """Test _dump_metrics_at_exit writes to a file when enabled."""
        explicit_output_filename = tmp_path / "explicit_metrics.json"
        auto_output_filename = tmp_path / metrics._init_dump_metrics_filename()

        if mode == "explicit":
            output_filename = explicit_output_filename
        elif mode == "auto":
            output_filename = auto_output_filename
        else:
            output_filename = False

        with gt_config.overrides(dump_metrics_at_exit=output_filename):
            with unittest.mock.patch(
                "gt4py.next.instrumentation.metrics.sources", sample_source_metrics
            ):
                metrics._dump_metrics_at_exit()

        assert (output_filename is False) == (mode is False)
        if output_filename:
            assert output_filename.exists()
            data = json.loads(output_filename.read_text())
            assert "program1" in data
            assert "program2" in data
            output_filename.unlink()  # Clean up after test
        else:
            assert not explicit_output_filename.exists()
            assert not auto_output_filename.exists()
