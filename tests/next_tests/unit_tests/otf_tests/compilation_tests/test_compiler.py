# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal contract tests for :class:`compiler.CPPCompilationArtifact`."""

import pathlib
import pickle

from gt4py._core import definitions as core_defs
from gt4py.next.otf.compilation import compiler


def test_cpp_build_artifact_pickle_round_trip():
    artifact = compiler.CPPCompilationArtifact(
        src_dir=pathlib.Path("/tmp/build"),
        module=pathlib.Path("entry.so"),
        entry_point_name="entry",
        device_type=core_defs.DeviceType.CPU,
    )
    restored = pickle.loads(pickle.dumps(artifact))
    assert restored == artifact
