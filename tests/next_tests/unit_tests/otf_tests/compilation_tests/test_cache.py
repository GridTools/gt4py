# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import json
import re

import pytest

from gt4py.next import config, fingerprinting
from gt4py.next.otf.compilation import cache

from next_tests.unit_tests.otf_tests.compilation_tests.build_systems_tests.conftest import (
    extension_source_example,
    program_source_example,
)


def test_cache_folder_includes_program_name_and_version_id(extension_source_example):
    cache_dir = cache.get_cache_folder(extension_source_example, config.BuildCacheLifetime.SESSION)

    assert extension_source_example.program_source.entry_point.name in cache_dir.name
    assert config.BUILD_CACHE_VERSION_ID in cache_dir.name


def test_version_id_busts_cache_folder(monkeypatch, extension_source_example):
    folder_a = cache.get_cache_folder(extension_source_example, config.BuildCacheLifetime.SESSION)

    monkeypatch.setattr(
        config, "BUILD_CACHE_VERSION_ID", f"{config.BUILD_CACHE_VERSION_ID}-changed"
    )
    folder_b = cache.get_cache_folder(extension_source_example, config.BuildCacheLifetime.SESSION)

    # A different version id must land the build in a different folder, even though the
    # extension source (and therefore its fingerprint) is unchanged.
    assert folder_a != folder_b


def test_pyext_marker_only_when_bindings_present(extension_source_example):
    with_bindings = cache.get_cache_folder(
        extension_source_example, config.BuildCacheLifetime.SESSION
    )
    assert cache.BINDINGS_NAME_SUFFIX in with_bindings.name

    without_bindings = cache.get_cache_folder(
        dataclasses.replace(extension_source_example, binding_source=None),
        config.BuildCacheLifetime.SESSION,
    )
    assert cache.BINDINGS_NAME_SUFFIX not in without_bindings.name


def test_translation_caches_share_one_parent(tmp_path):
    folders = [
        cache.get_translation_cache_folder(tmp_path, backend)
        for backend in cache.TRANSLATION_CACHE_BACKENDS
    ]

    assert len(folders) == len(set(folders))
    assert {folder.parent for folder in folders} == {tmp_path / cache.TRANSLATION_CACHE_DIR_NAME}
    # a translation cache is never mistaken for a build folder of a program
    assert not any(re.fullmatch(cache.CACHE_FOLDER_NAME_PATTERN, f.parent.name) for f in folders)


def test_build_context_id_busts_cache_folder(extension_source_example):
    folder_a = cache.get_cache_folder(extension_source_example, config.BuildCacheLifetime.SESSION)
    folder_b = cache.get_cache_folder(
        extension_source_example, config.BuildCacheLifetime.SESSION, build_context_id="ctx"
    )

    assert folder_a != folder_b
    assert folder_b.name.endswith("_ctx")


@pytest.mark.parametrize("with_bindings", [True, False])
@pytest.mark.parametrize("with_context_id", [True, False])
def test_folder_name_round_trips_through_documented_pattern(
    extension_source_example, with_bindings, with_context_id
):
    """Names built by `get_cache_folder` must parse back through `CACHE_FOLDER_NAME_PATTERN`.

    External tooling (e.g. `scripts/python/dace_determinism.py`) recognizes cached
    program folders and recovers the program name through this pattern, so any
    change to the folder naming scheme must update the pattern together with it.
    """
    ext_source = (
        extension_source_example
        if with_bindings
        else dataclasses.replace(extension_source_example, binding_source=None)
    )
    context_id = (
        fingerprinting.strict_fingerprinter("build configuration") if with_context_id else ""
    )
    folder = cache.get_cache_folder(
        ext_source, config.BuildCacheLifetime.SESSION, build_context_id=context_id
    )

    match = re.fullmatch(cache.CACHE_FOLDER_NAME_PATTERN, folder.name)
    assert match is not None
    expected_name = ext_source.program_source.entry_point.name + ("_pyext" if with_bindings else "")
    assert match["name"] == expected_name
    assert match["version_id"] == config.BUILD_CACHE_VERSION_ID
    assert match["build_context_id"] == (context_id if with_context_id else None)


TOOLCHAIN = {"machine": "x86_64", "python": "3.13.0", "dace": "2.0.0", "cxx": "g++ 14.3.0"}


@pytest.fixture
def toolchain_cache(monkeypatch, tmp_path):
    """A persistent cache base in `tmp_path`, with the toolchain faked as `TOOLCHAIN`."""
    monkeypatch.setattr(config, "BUILD_CACHE_DIR", tmp_path / ".gt4py_cache")
    monkeypatch.setattr(
        config, "BUILD_CACHE_TOOLCHAIN_CHECK", config.BuildCacheToolchainCheck.ERROR
    )
    monkeypatch.setattr(cache, "current_toolchain", lambda: dict(TOOLCHAIN))
    check = cache.check_build_cache_toolchain
    check.cache_clear()
    yield config.BUILD_CACHE_DIR
    check.cache_clear()


def _record(base_path):
    return json.loads((base_path / cache.TOOLCHAIN_FILE_NAME).read_text())


def _use_cache(extension_source_example):
    cache.check_build_cache_toolchain.cache_clear()
    return cache.get_cache_folder(extension_source_example, config.BuildCacheLifetime.PERSISTENT)


def test_toolchain_recorded_on_first_use(toolchain_cache, extension_source_example):
    _use_cache(extension_source_example)

    assert _record(toolchain_cache) == TOOLCHAIN


def test_same_toolchain_passes(toolchain_cache, extension_source_example):
    _use_cache(extension_source_example)
    _use_cache(extension_source_example)

    assert _record(toolchain_cache) == TOOLCHAIN


def test_changed_toolchain_raises(monkeypatch, toolchain_cache, extension_source_example):
    _use_cache(extension_source_example)
    monkeypatch.setattr(cache, "current_toolchain", lambda: {**TOOLCHAIN, "dace": "2.1.0"})

    with pytest.raises(
        cache.BuildCacheToolchainError, match=r"dace: '2\.0\.0' \(cache\) -> '2\.1\.0'"
    ):
        _use_cache(extension_source_example)
    assert _record(toolchain_cache) == TOOLCHAIN


def test_changed_toolchain_warns_if_downgraded(
    monkeypatch, toolchain_cache, extension_source_example
):
    _use_cache(extension_source_example)
    monkeypatch.setattr(cache, "current_toolchain", lambda: {**TOOLCHAIN, "cxx": "g++ 15.1.0"})
    monkeypatch.setattr(config, "BUILD_CACHE_TOOLCHAIN_CHECK", config.BuildCacheToolchainCheck.WARN)

    with pytest.warns(UserWarning, match="cxx"):
        _use_cache(extension_source_example)
    assert _record(toolchain_cache) == TOOLCHAIN


def test_new_toolchain_entry_is_added_not_compared(
    monkeypatch, toolchain_cache, extension_source_example
):
    _use_cache(extension_source_example)
    monkeypatch.setattr(
        cache, "current_toolchain", lambda: {**TOOLCHAIN, "cuda_cxx": "release 13.1"}
    )

    _use_cache(extension_source_example)
    assert _record(toolchain_cache) == {**TOOLCHAIN, "cuda_cxx": "release 13.1"}


def test_toolchain_check_off_records_nothing(
    monkeypatch, toolchain_cache, extension_source_example
):
    monkeypatch.setattr(config, "BUILD_CACHE_TOOLCHAIN_CHECK", config.BuildCacheToolchainCheck.OFF)

    _use_cache(extension_source_example)
    assert not (toolchain_cache / cache.TOOLCHAIN_FILE_NAME).exists()


def test_session_cache_is_not_checked(monkeypatch, toolchain_cache, extension_source_example):
    monkeypatch.setattr(cache, "check_build_cache_toolchain", lambda base_path: pytest.fail())

    cache.get_cache_folder(extension_source_example, config.BuildCacheLifetime.SESSION)


def test_current_toolchain_reports_machine_and_python():
    toolchain = cache.current_toolchain()

    assert toolchain["machine"] and toolchain["python"]
