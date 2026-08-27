# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DaCe configuration set up by the dace backend workflow.

Covers the detection of the number of GPU chiplets (XCDs), which no CI machine has the
hardware for, so `amdsmi` is faked throughout.
"""

import sys
import types
from typing import Any, Optional

import dace
import pytest

from gt4py._core import definitions as core_defs
from gt4py.next.program_processors.runners.dace.workflow import common as dace_wf_common


_CHIPLET_KEY = "compiler.cuda.chiplet_number"


@pytest.fixture(autouse=True)
def clear_chiplet_query_cache():
    """The chiplet query is cached per process, which would leak between tests."""
    dace_wf_common._query_gpu_chiplet_count.cache_clear()
    yield
    dace_wf_common._query_gpu_chiplet_count.cache_clear()


def make_fake_amdsmi(xcd_count: Optional[int] = None, handles: Any = ("handle",)):
    """Stand-in for the `amdsmi` module, recording whether it was queried."""
    calls: list[str] = []

    fake = types.ModuleType("amdsmi")
    fake.amdsmi_init = lambda *args: calls.append("init")
    fake.amdsmi_shut_down = lambda *args: calls.append("shut_down")
    fake.amdsmi_get_processor_handles = lambda: list(handles)

    def get_xcd_counter(handle):
        calls.append("xcd_counter")
        assert xcd_count is not None
        return xcd_count

    fake.amdsmi_get_gpu_xcd_counter = get_xcd_counter
    fake.calls = calls
    return fake


def chiplet_nondefault() -> Optional[int]:
    """The chiplet number as it reaches the GT4Py build-cache fingerprint, or None."""
    nondefaults = dace.Config._data.nondefaults()
    return nondefaults.get("compiler", {}).get("cuda", {}).get("chiplet_number")


def test_chiplet_number_config_key_exists():
    """Guards the DaCe source pin: setting a key DaCe does not know is a silent no-op."""
    try:
        metadata = dace.Config.get_metadata("compiler", "cuda", "chiplet_number")
    except KeyError as e:
        raise AssertionError(
            "The installed DaCe has no `compiler.cuda.chiplet_number`, so GT4Py would be "
            "setting a configuration entry that nothing reads. GT4Py pins DaCe to a branch "
            "that provides it, see `[tool.uv.sources]` in `pyproject.toml`."
        ) from e
    assert metadata["type"] == "int"


def test_chiplet_number_queried_from_amdsmi(monkeypatch):
    monkeypatch.delenv(dace_wf_common._CHIPLET_NUMBER_ENV_VAR, raising=False)
    fake = make_fake_amdsmi(xcd_count=6)
    monkeypatch.setitem(sys.modules, "amdsmi", fake)

    with dace_wf_common.dace_context(device_type=core_defs.DeviceType.ROCM):
        assert dace.Config.get(_CHIPLET_KEY) == 6
        assert chiplet_nondefault() == 6

    assert fake.calls == ["init", "xcd_counter", "shut_down"]


def test_chiplet_number_env_override_reaches_fingerprint(monkeypatch):
    """`Config.get` honours the env var on its own, but `nondefaults` would not see it."""
    monkeypatch.setenv(dace_wf_common._CHIPLET_NUMBER_ENV_VAR, "4")
    fake = make_fake_amdsmi(xcd_count=6)
    monkeypatch.setitem(sys.modules, "amdsmi", fake)

    with dace_wf_common.dace_context(device_type=core_defs.DeviceType.ROCM):
        assert chiplet_nondefault() == 4

    # The environment takes precedence, so the device is never queried.
    assert fake.calls == []


def test_chiplet_number_unset_without_amdsmi(monkeypatch):
    monkeypatch.delenv(dace_wf_common._CHIPLET_NUMBER_ENV_VAR, raising=False)
    # A `None` entry in `sys.modules` makes the import fail.
    monkeypatch.setitem(sys.modules, "amdsmi", None)

    with pytest.warns(UserWarning, match="chiplets"):
        with dace_wf_common.dace_context(device_type=core_defs.DeviceType.ROCM):
            # Left at the DaCe default, which disables the distribution.
            assert dace.Config.get(_CHIPLET_KEY) == 1
            assert chiplet_nondefault() is None


def test_chiplet_number_not_configured_on_cuda(monkeypatch):
    monkeypatch.delenv(dace_wf_common._CHIPLET_NUMBER_ENV_VAR, raising=False)
    fake = make_fake_amdsmi(xcd_count=6)
    monkeypatch.setitem(sys.modules, "amdsmi", fake)

    with dace_wf_common.dace_context(device_type=core_defs.DeviceType.CUDA):
        assert dace.Config.get(_CHIPLET_KEY) == 1
        assert chiplet_nondefault() is None

    assert fake.calls == []
