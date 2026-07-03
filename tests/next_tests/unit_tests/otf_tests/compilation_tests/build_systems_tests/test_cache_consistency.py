# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Crash-consistency of the gtfn build-folder cache.

An interrupted compile (Ctrl-C / SIGKILL / OOM, scratch cleanup, full disk) can
leave the build folder in a state where the lookup reports a cache HIT but the
artifacts are missing/truncated/half-written. The next run must rebuild cleanly
instead of failing. These tests drive ``compiler.CPPCompiler.__call__`` (the
HIT/rebuild gate) through a real build system and corrupt the cached state to
mimic each interruption window.
"""

import pathlib
import shutil

import pytest

from gt4py._core import definitions as core_defs
from gt4py.next import config, fingerprinting
from gt4py.next.otf.compilation import build_data, cache, compiler
from gt4py.next.otf.compilation.build_systems import compiledb


def _compiler() -> compiler.CPPCompiler:
    return compiler.CPPCompiler(
        cache_lifetime=config.BuildCacheLifetime.SESSION,
        builder_factory=compiledb.CompiledbFactory(),
        device_type=core_defs.DeviceType.CPU,
    )


def _src_dir(comp: compiler.CPPCompiler, extension_source) -> pathlib.Path:
    return cache.get_cache_folder(
        extension_source,
        config.BuildCacheLifetime.SESSION,
        build_context_id=fingerprinting.strict_fingerprinter(comp.builder_factory),
    )


@pytest.fixture
def clean_session_cache(extension_source_example):
    cache_dir = _src_dir(_compiler(), extension_source_example)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    yield


def test_compiler_recovers_from_corrupt_build_data(extension_source_example, clean_session_cache):
    """F2: an interrupted write truncates ``gt4py.json``. The next compile must
    treat unreadable build metadata as 'not built' and rebuild, not crash in
    ``json.loads``."""
    comp = _compiler()
    comp(extension_source_example)
    src_dir = _src_dir(comp, extension_source_example)

    (src_dir / "gt4py.json").write_text("{ truncated")

    artifact = comp(extension_source_example)  # must rebuild, not raise

    data = build_data.read_data(src_dir)
    assert data is not None and data.status == build_data.BuildStatus.COMPILED
    assert (src_dir / data.module).exists()
    assert callable(artifact.load())


def test_compiler_recovers_from_missing_module(extension_source_example, clean_session_cache):
    """F3: ``gt4py.json`` says COMPILED but the module artifact is gone (scratch
    cleanup, partial ``rm``). The compiler must rebuild instead of raising
    ``CompilationError``."""
    comp = _compiler()
    comp(extension_source_example)
    src_dir = _src_dir(comp, extension_source_example)
    data = build_data.read_data(src_dir)

    (src_dir / data.module).unlink()  # remove the module, keep status == COMPILED

    artifact = comp(extension_source_example)  # must rebuild

    new_data = build_data.read_data(src_dir)
    assert (src_dir / new_data.module).exists()
    assert callable(artifact.load())
