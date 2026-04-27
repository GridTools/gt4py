# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf import workflow
from gt4py.next.otf.binding import nanobind
from gt4py.next.otf.compilation import compiler, importer
from gt4py.next.otf.compilation.build_systems import cmake, compiledb

from next_tests.unit_tests.otf_tests.compilation_tests.build_systems_tests.conftest import (
    program_source_with_name,
)


def _import_artifact_entry_point(artifact: compiler.CPPBuildArtifact):
    """Import the .so directly and return the raw entry point.

    Bypasses :meth:`CPPBuildArtifact.materialize` so the test can call the
    nanobind-bound function with raw arguments rather than gt4py-shaped ones —
    this is a build-system / binding integration test, not an end-to-end
    program test.
    """
    m = importer.import_from_path(
        artifact.src_dir / artifact.module,
        sys_modules_prefix="gt4py.__compiled_programs__.",
    )
    return getattr(m, artifact.entry_point_name)


def _identity(raw, _device):
    return raw


def test_gtfn_cpp_with_cmake(program_source_with_name):
    example_program_source = program_source_with_name("gtfn_cpp_with_cmake")
    build_the_program = workflow.make_step(nanobind.bind_source).chain(
        compiler.CPPCompiler(
            cache_lifetime=config.BuildCacheLifetime.SESSION,
            builder_factory=cmake.CMakeFactory(),
            device_type=core_defs.DeviceType.CPU,
            decorator=_identity,
        )
    )
    artifact = build_the_program(example_program_source)
    compiled_program = _import_artifact_entry_point(artifact)
    buf = (np.zeros(shape=(6, 5), dtype=np.float32), (0, 0))
    tup = [
        (np.zeros(shape=(6, 5), dtype=np.float32), (0, 0)),
        (np.zeros(shape=(6, 5), dtype=np.float32), (0, 0)),
    ]
    sc = np.float32(3.1415926)
    res = compiled_program(buf, tup, sc)
    assert math.isclose(res, 6 * 5 * 3.1415926, rel_tol=1e-4)


def test_gtfn_cpp_with_compiledb(program_source_with_name):
    example_program_source = program_source_with_name("gtfn_cpp_with_compiledb")
    build_the_program = workflow.make_step(nanobind.bind_source).chain(
        compiler.CPPCompiler(
            cache_lifetime=config.BuildCacheLifetime.SESSION,
            builder_factory=compiledb.CompiledbFactory(),
            device_type=core_defs.DeviceType.CPU,
            decorator=_identity,
        )
    )
    artifact = build_the_program(example_program_source)
    compiled_program = _import_artifact_entry_point(artifact)
    buf = (np.zeros(shape=(6, 5), dtype=np.float32), (0, 0))
    tup = [
        (np.zeros(shape=(6, 5), dtype=np.float32), (0, 0)),
        (np.zeros(shape=(6, 5), dtype=np.float32), (0, 0)),
    ]
    sc = np.float32(3.1415926)
    res = compiled_program(buf, tup, sc)
    assert math.isclose(res, 6 * 5 * 3.1415926, rel_tol=1e-4)
