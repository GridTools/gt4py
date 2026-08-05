# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DaCe program decoration stage."""

import sys
import unittest.mock as mock
from typing import Any

import pytest


dace = pytest.importorskip("dace")

from gt4py._core import definitions as core_defs
from gt4py.next.program_processors.runners.dace.workflow import (
    common as dace_wf_common,
)
from gt4py.next.program_processors.runners.dace.workflow import (
    decoration as dace_wf_decoration,
)


class _FakeStream:
    """Stand-in for a ``cupy.cuda.Stream`` that does not require a GPU."""

    def __init__(self, ptr: int = 42, device_id: int = 0) -> None:
        self.ptr = ptr
        self.device_id = device_id


def _make_sdfg_with_stream_sync_symbols() -> dace.SDFG:
    """Return a minimal SDFG whose symbols contain the sync-stream symbol."""
    sdfg = dace.SDFG("stream_sync_program")
    state = sdfg.add_state("state", is_start_block=True)
    sdfg.add_symbol(dace_wf_common.SDFG_ARG_EXTERNAL_SYNC_STREAM, dace.uint64)
    # Add a no-op tasklet so the SDFG is not empty.
    state.add_tasklet(
        "noop", inputs=set(), outputs=set(), code="", language=dace.dtypes.Language.CPP
    )
    sdfg.validate()
    return sdfg


def _make_decorated_program(
    *, device_type: core_defs.DeviceType = core_defs.DeviceType.CUDA
) -> tuple[dace_wf_decoration.DaCeDecoratedProgram, mock.MagicMock]:
    compiled_sdfg = mock.MagicMock()
    compiled_sdfg.sdfg = _make_sdfg_with_stream_sync_symbols()
    compiled_sdfg.sdfg.build_folder = "/tmp"

    program = dace_wf_decoration.DaCeDecoratedProgram(
        compiled_sdfg,
        device_type=device_type,
    )
    return program, compiled_sdfg


def test_set_external_sync_stream_rejects_non_cupy_stream(monkeypatch):
    """A non-cupy stream object raises TypeError."""
    fake_cupy_module = mock.MagicMock()
    fake_cupy_module.cuda.Stream = _FakeStream
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy_module)

    program, _ = _make_decorated_program(device_type=core_defs.DeviceType.CUDA)

    with pytest.raises(TypeError, match="external_sync_stream must be a cupy.cuda.Stream"):
        program.set_external_sync_stream("not-a-stream")


def test_set_external_sync_stream_rejects_cpu_target(monkeypatch):
    """A CPU target does not support an external sync stream."""
    fake_cupy_module = mock.MagicMock()
    fake_cupy_module.cuda.Stream = _FakeStream
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy_module)

    program, _ = _make_decorated_program(device_type=core_defs.DeviceType.CPU)

    with pytest.raises(ValueError, match="Stream synchronization is not supported for CPU target"):
        program.set_external_sync_stream(_FakeStream())


def test_set_external_sync_stream_accepts_valid_stream(monkeypatch):
    """A valid cupy stream is stored on the underlying compiled program."""
    fake_cupy_module = mock.MagicMock()
    fake_cupy_module.cuda.Stream = _FakeStream
    fake_cupy_module.cuda.Device.return_value.id = 0
    fake_cupy_module.cuda.runtime.cudaSuccess = 0
    fake_cupy_module.cuda.runtime.cudaErrorNotReady = 600
    fake_cupy_module.cuda.runtime.cudaStreamQuery.return_value = 0
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy_module)

    stream = _FakeStream(ptr=7, device_id=0)
    program, compiled_sdfg = _make_decorated_program(device_type=core_defs.DeviceType.CUDA)

    program.set_external_sync_stream(stream)

    assert compiled_sdfg.external_sync_stream is stream
