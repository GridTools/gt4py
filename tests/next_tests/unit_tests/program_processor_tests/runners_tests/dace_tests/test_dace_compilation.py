# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal contract tests for :class:`compilation.DaCeCompilationArtifact`."""

import pathlib
import pickle

import pytest

pytest.importorskip("dace")

from gt4py._core import definitions as core_defs  # noqa: E402
from gt4py.next.program_processors.runners.dace.workflow import compilation  # noqa: E402


def test_dace_compilation_artifact_pickle_round_trip_drops_live_program(tmp_path: pathlib.Path):
    artifact = compilation.DaCeCompilationArtifact(
        build_folder=tmp_path,
        sdfg_json="{}",
        binding_source_code="def update_sdfg_args(*a, **k): ...",
        bind_func_name="update_sdfg_args",
        device_type=core_defs.DeviceType.CPU,
    )
    object.__setattr__(artifact, "_live_program", "<pretend live handle>")

    restored = pickle.loads(pickle.dumps(artifact))

    # The data fields round-trip, the live in-process handle does not.
    assert restored == artifact
    assert restored._live_program is None
